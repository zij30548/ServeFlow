#include "transport.h"

#include <QDateTime>
#include <QHostAddress>
#include <QIODevice>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkCookie>
#include <QNetworkCookieJar>
#include <QNetworkProxy>
#include <QNetworkReply>
#include <QRegularExpression>
#include <cstring>
#include <limits>
#include <memory>

namespace {
constexpr qsizetype maxReplyBytes = 1024 * 1024;
const QString loginKey = QStringLiteral("/login");  // Outside the allowed operation ID alphabet.

bool validId(const QString &id)
{
    static const QRegularExpression pattern(QStringLiteral("\\A[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\\z"));
    return pattern.match(id).hasMatch();
}

bool validOrigin(const QUrl &origin)
{
    const bool localHttp = origin.scheme() == "http" && QHostAddress(origin.host()).isLoopback();
    return origin.isValid() && !origin.host().isEmpty() && !origin.authority().contains('@')
        && (origin.path().isEmpty() || origin.path() == "/") && !origin.hasQuery()
        && !origin.hasFragment() && (origin.scheme() == "https" || localHttp);
}

QString errorMessage(int http, const QJsonObject &body)
{
    const QString code = body.value("error_code").toString(body.value("detail").toString());
    if (code == "demo_model_quota_exhausted") return QStringLiteral("演示模型额度已用完，请联系管理员。");
    if (code == "demo_model_quota_unavailable") return QStringLiteral("演示额度暂时无法核验，请联系管理员。");
    if (code == "tool_outcome_unknown") return QStringLiteral("相关操作结果待核实，请勿重复操作。");
    if (code == "agent_busy") return QStringLiteral("处理通道繁忙，本次未完成。");
    if (code == "agent_timeout") return QStringLiteral("处理超时，已发送的请求结果待核实。");
    if (http == 401 || http == 403 || http == 428) return QStringLiteral("访问被拒绝，请核对演示身份和服务地址。");
    if (http == 429) return QStringLiteral("已达到请求频率限制，请稍后再试。");
    if (http == 409) return QStringLiteral("请求状态存在冲突，请核对当前会话。");
    return QStringLiteral("本次未取得有效回复，请核对会话或服务连接。");
}

// Qt may try to rewind an upload when a keep-alive connection disappears.
// A chat POST has one physical body and cannot be replayed at the HTTP layer.
class TurnUpload final : public QIODevice {
public:
    explicit TurnUpload(QByteArray bytes) : m_bytes(std::move(bytes)) { open(ReadOnly | Unbuffered); }
    bool isSequential() const override { return true; }
    bool reset() override { return false; }
    qint64 bytesAvailable() const override { return m_bytes.size() - m_offset + QIODevice::bytesAvailable(); }
protected:
    qint64 readData(char *data, qint64 limit) override
    {
        const qint64 count = qMin(limit, m_bytes.size() - m_offset);
        if (count == 0) return -1;
        std::memcpy(data, m_bytes.constData() + m_offset, static_cast<size_t>(count));
        m_offset += count;
        return count;
    }
    qint64 writeData(const char *, qint64) override { return -1; }
private:
    QByteArray m_bytes;
    qint64 m_offset = 0;
};
}

Transport::Transport(QObject *parent) : QObject(parent)
{
    m_expiration.setSingleShot(true);
    connect(&m_expiration, &QTimer::timeout, this, &Transport::expire);
}

void Transport::logout()
{
    m_expiration.stop();
    const auto replies = m_pending;
    m_pending.clear();
    for (auto *reply : replies) reply->abort();
    if (m_network) m_network->deleteLater();
    m_network = nullptr;
    m_origin.clear();
    m_userId.clear();
    m_basic.clear();
    m_csrf.clear();
}

void Transport::expire()
{
    logout();
    emit sessionExpired();
}

QNetworkRequest Transport::request(const QString &path) const
{
    QUrl url = m_origin;
    url.setPath(QStringLiteral("/api/python/") + path);
    QNetworkRequest req(url);
    req.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::ManualRedirectPolicy);
    req.setAttribute(QNetworkRequest::AuthenticationReuseAttribute, QNetworkRequest::Manual);
    req.setRawHeader("Origin", m_origin.toEncoded());
    req.setRawHeader("Accept", "application/json");
    req.setRawHeader("Authorization", m_basic);
    return req;
}

void Transport::watch(const QString &key, QNetworkReply *reply, int timeoutMs, Completion completion)
{
    m_pending.insert(key, reply);
    auto *timer = new QTimer(reply);
    timer->setSingleShot(true);
    connect(timer, &QTimer::timeout, reply, &QNetworkReply::abort);
    timer->start(timeoutMs);
    auto body = std::make_shared<QByteArray>();
    reply->setReadBufferSize(maxReplyBytes + 1);
    connect(reply, &QNetworkReply::readyRead, reply, [reply, body] {
        body->append(reply->read(maxReplyBytes + 1 - body->size()));
        if (body->size() > maxReplyBytes) reply->abort();
    });
    connect(reply, &QNetworkReply::finished, this, [this, key, timer, reply, body, completion = std::move(completion)] {
        timer->stop();
        reply->deleteLater();
        if (m_pending.value(key) != reply) return;
        m_pending.remove(key);
        const int http = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        if (reply->isOpen() && body->size() <= maxReplyBytes)
            body->append(reply->read(maxReplyBytes + 1 - body->size()));
        // Qt maps HTTP 4xx/5xx to network errors; those can carry structured failures.
        const bool complete = body->size() <= maxReplyBytes && (reply->error() == QNetworkReply::NoError
            || (http >= 400 && static_cast<int>(reply->error()) >= 200));
        completion(http, complete, *body);
    });
}

void Transport::login(const QUrl &origin, const QString &username, const QString &password)
{
    logout();
    if (!validOrigin(origin) || username.isEmpty() || username.contains(':') || password.isEmpty()) {
        emit loginFinished(false, QStringLiteral("请输入演示服务的 HTTPS 根地址、账号和密码。"));
        return;
    }
    m_origin = origin;
    m_origin.setPath({});
    if ((m_origin.scheme() == "https" && m_origin.port() == 443) ||
        (m_origin.scheme() == "http" && m_origin.port() == 80)) m_origin.setPort(-1);
    m_basic = "Basic " + (username + ':' + password).toUtf8().toBase64();
    m_network = new QNetworkAccessManager(this);
    if (QHostAddress(origin.host()).isLoopback()) m_network->setProxy(QNetworkProxy::NoProxy);
    const auto req = request("demo/session");
    watch(loginKey, m_network->get(req), 10000, [this](int http, bool complete, const QByteArray &body) {
        const auto session = QJsonDocument::fromJson(body).object();
        const QString workspace = session.value("workspace_user_id").toString();
        const QString csrf = session.value("csrf_token").toString();
        const QString label = session.value("customer_label").toString();
        const auto expiry = QDateTime::fromString(session.value("expires_at").toString(), Qt::ISODateWithMs);
        const auto remaining = QDateTime::currentDateTimeUtc().msecsTo(expiry);
        static const QRegularExpression workspacePattern(QStringLiteral("\\Asf_demo_[A-Za-z0-9_-]{16,128}\\z"));
        static const QRegularExpression csrfPattern(QStringLiteral("\\A[A-Za-z0-9_-]{1,256}\\z"));
        bool hasCookie = false;
        for (const auto &cookie : m_network->cookieJar()->cookiesForUrl(request("chat").url())) {
            if (cookie.name() == "serveflow_demo_session" && !cookie.value().isEmpty() && cookie.isHttpOnly()
                && (m_origin.scheme() == "http" || cookie.isSecure())) hasCookie = true;
        }
        if (!complete || http != 200 || !session.value("is_demo").toBool() || !session.value("enabled").toBool()
            || !hasCookie || !expiry.isValid() || remaining <= 0 || remaining > std::numeric_limits<int>::max()
            || !workspacePattern.match(workspace).hasMatch() || !csrfPattern.match(csrf).hasMatch()
            || label.trimmed().isEmpty() || label.size() > 40) {
            logout();
            emit loginFinished(false, http == 401 ? QStringLiteral("演示账号或密码未通过验证。")
                                                  : QStringLiteral("无法建立演示会话，请核对服务地址及访问权限。"));
            return;
        }
        m_userId = workspace;
        m_csrf = csrf.toLatin1();
        m_expiration.start(static_cast<int>(remaining));
        emit loginFinished(true, label);
    });
}

void Transport::send(const QString &operationId, const QString &conversationId, const QString &message)
{
    if (m_pending.contains(operationId)) return;
    if (m_userId.isEmpty() || !validId(operationId) || !validId(conversationId)
        || message.trimmed().isEmpty() || (message.size() > 12000 && message.toUcs4().size() > 12000)) {
        emit chatFinished(operationId, conversationId, "failed", QStringLiteral("身份或消息无效，请检查后再发送。"));
        return;
    }
    const QJsonObject payload{{"message", message}, {"user_id", m_userId}, {"conv_id", conversationId},
                              {"client_request_id", operationId}};
    const auto bytes = QJsonDocument(payload).toJson(QJsonDocument::Compact);
    auto req = request("chat");
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    req.setHeader(QNetworkRequest::ContentLengthHeader, bytes.size());
    req.setAttribute(QNetworkRequest::DoNotBufferUploadDataAttribute, true);
    req.setRawHeader("X-ServeFlow-Demo-CSRF", m_csrf);
    auto *upload = new TurnUpload(bytes);
    auto *reply = m_network->post(req, upload);
    upload->setParent(reply);
    watch(operationId, reply, 55000, [this, operationId, conversationId](int http, bool complete, const QByteArray &bytes) {
        const auto body = QJsonDocument::fromJson(bytes).object();
        const auto status = body.value("status").toString();
        const auto answer = body.value("response").toString();
        const auto code = body.value("error_code");
        const bool receipt = complete && body.value("conv_id").toString() == conversationId
            && body.value("retryable").isBool() && body.value("response").isString()
            && answer.size() <= 100000 && (code.isUndefined() || code.isNull() || (code.isString() && code.toString().size() <= 100));
        if (receipt && http == 200 && (status == "success" || status == "partial") && !answer.trimmed().isEmpty()) {
            emit chatFinished(operationId, conversationId, status, answer);
        } else if (complete && (http == 401 || http == 403 || http == 428)) {
            expire();
        } else {
            const bool failedReceipt = receipt && (http == 503 || http == 200) && status == "failed"
                && code.toString() != "tool_outcome_unknown" && code.toString() != "agent_timeout";
            const bool knownRejection = complete && (http == 400 || http == 413 || http == 422 || http == 429);
            emit chatFinished(operationId, conversationId, failedReceipt || knownRejection ? "failed" : "unknown",
                              errorMessage(http, body));
        }
    });
}

void Transport::cancel(const QString &operationId)
{
    if (auto *reply = m_pending.take(operationId)) reply->abort();
}
