#include "transport.h"

#include <QDateTime>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPointer>
#include <QSignalSpy>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTest>
#include <memory>

namespace {
const QString conversation = "conversation-A";
const QString op = "op-a";
const QString uid = "sf_demo_0123456789abcdef0123456789abcdef";

struct Received {
    QByteArray requestLine;
    QMap<QByteArray, QByteArray> headers;
    QByteArray body;
    QPointer<QTcpSocket> socket;
};

class LocalGateway : public QTcpServer {
public:
    QList<Received> calls;
    LocalGateway()
    {
        connect(this, &QTcpServer::newConnection, this, [this] {
            while (auto *socket = nextPendingConnection()) {
                auto buffer = std::make_shared<QByteArray>();
                connect(socket, &QTcpSocket::readyRead, this, [this, socket, buffer] {
                    *buffer += socket->readAll();
                    const auto head = buffer->indexOf("\r\n\r\n");
                    if (head < 0) return;
                    auto lines = buffer->left(head).split('\n');
                    Received req{lines.takeFirst().trimmed(), {}, {}, socket};
                    for (const auto &line : lines) {
                        const auto delimiter = line.indexOf(':');
                        if (delimiter > 0) req.headers.insert(line.first(delimiter).trimmed().toLower(),
                                                               line.sliced(delimiter + 1).trimmed());
                    }
                    const auto length = req.headers.value("content-length").toInt();
                    if (buffer->size() < head + 4 + length) return;
                    req.body = buffer->mid(head + 4, length);
                    buffer->remove(0, head + 4 + length);
                    calls.append(req);
                });
            }
        });
    }
    QUrl origin() const { return QUrl(QStringLiteral("http://127.0.0.1:%1").arg(serverPort())); }

    void respond(qsizetype index, int status, const QByteArray &body, const QByteArray &headers = {})
    {
        const auto socket = calls.at(index).socket;
        if (!socket || socket->state() != QAbstractSocket::ConnectedState) return;
        socket->write("HTTP/1.1 " + QByteArray::number(status) + " Result\r\nConnection: close\r\n"
                      + "Content-Type: application/json\r\nContent-Length: " + QByteArray::number(body.size())
                      + "\r\n" + headers + "\r\n" + body);
        socket->disconnectFromHost();
    }

    void session(qsizetype index, bool withCookie = true)
    {
        const auto expiry = QDateTime::currentDateTimeUtc().addSecs(120).toString(Qt::ISODateWithMs);
        QJsonObject body{{"is_demo", true}, {"enabled", true}, {"workspace_user_id", uid},
                         {"customer_label", "fixture"}, {"csrf_token", "fixture-csrf"}, {"expires_at", expiry}};
        respond(index, 200, QJsonDocument(body).toJson(QJsonDocument::Compact),
                withCookie ? "Set-Cookie: serveflow_demo_session=fake; Path=/; HttpOnly; SameSite=Strict\r\n" : "");
    }
};
QByteArray receipt(const QString &conv, const QString &status = "success")
{
    return QJsonDocument(QJsonObject{{"conv_id", conv}, {"status", status}, {"response", "仅测试正文"},
                                     {"retryable", false}, {"request_id", "service-trace-123"}}).toJson();
}
}

class TestTransport : public QObject {
    Q_OBJECT
private slots:
    void protocolAndSession()
    {
        LocalGateway server;
        QVERIFY(server.listen(QHostAddress::LocalHost, 0));
        Transport client;
        QSignalSpy login(&client, &Transport::loginFinished), result(&client, &Transport::chatFinished);
        client.login(server.origin(), "fixture", "local-only");
        QTRY_COMPARE(server.calls.size(), 1);
        const auto &get = server.calls[0];
        QCOMPARE(get.requestLine, "GET /api/python/demo/session HTTP/1.1");
        QCOMPARE(get.headers.value("authorization"), "Basic " + QByteArray("fixture:local-only").toBase64());
        QVERIFY(!get.headers.contains("x-serveflow-demo-csrf"));
        server.session(0);
        QTRY_COMPARE(login.count(), 1);
        QVERIFY(login.first()[0].toBool());
        client.send(op, conversation, "查演示订单");
        client.send(op, conversation, "查演示订单");
        QTRY_COMPARE(server.calls.size(), 2);
        const auto &post = server.calls[1];
        QCOMPARE(post.requestLine, "POST /api/python/chat HTTP/1.1");
        QCOMPARE(post.headers.value("cookie"), "serveflow_demo_session=fake");
        QCOMPARE(post.headers.value("x-serveflow-demo-csrf"), "fixture-csrf");
        QCOMPARE(post.headers.value("origin"), server.origin().toEncoded());
        const QJsonObject expected{{"conv_id", conversation}, {"client_request_id", op},
                                   {"user_id", uid}, {"message", "查演示订单"}};
        QCOMPARE(QJsonDocument::fromJson(post.body).object(), expected);
        server.respond(1, 200, receipt(conversation, "partial"));
        QTRY_COMPARE(result.count(), 1);
        QCOMPARE(result.first()[0].toString(), op);
        QCOMPARE(result.first()[1].toString(), conversation);
        QCOMPARE(result.first()[2].toString(), "partial");
        QCOMPARE(result.first()[3].toString(), "仅测试正文");
    }

    void replyStatuses_data()
    {
        QTest::addColumn<int>("http");
        QTest::addColumn<QByteArray>("body");
        QTest::addColumn<QString>("expected");
        QTest::newRow("misbound conversation") << 200 << receipt("conversation-B") << QString("unknown");
        QTest::newRow("malformed") << 200 << QByteArray("html") << QString("unknown");
        QTest::newRow("throttled") << 429 << QByteArray("{}") << QString("failed");
        QTest::newRow("backend failure receipt") << 503 << receipt(conversation, "failed") << QString("failed");
        QTest::newRow("unconfirmed upstream") << 502 << QByteArray("{}") << QString("unknown");
    }
    void replyStatuses()
    {
        QFETCH(int, http); QFETCH(QByteArray, body); QFETCH(QString, expected);
        LocalGateway server;
        QVERIFY(server.listen(QHostAddress::LocalHost, 0));
        Transport client;
        QSignalSpy login(&client, &Transport::loginFinished), result(&client, &Transport::chatFinished);
        client.login(server.origin(), "fixture", "local-only");
        QTRY_COMPARE(server.calls.size(), 1);
        server.session(0);
        QTRY_COMPARE(login.count(), 1);
        client.send(op, conversation, "hello");
        QTRY_COMPARE(server.calls.size(), 2);
        server.respond(1, http, body);
        QTRY_COMPARE(result.count(), 1);
        QCOMPARE(result.first()[2].toString(), expected);
        QVERIFY(result.first()[3].toString() != "仅测试正文");
        QCOMPARE(server.calls.size(), 2);
    }

    void lostReplyKeepsOutcomeUnknown()
    {
        LocalGateway server;
        QVERIFY(server.listen(QHostAddress::LocalHost, 0));
        Transport client;
        QSignalSpy login(&client, &Transport::loginFinished), result(&client, &Transport::chatFinished);
        client.login(server.origin(), "fixture", "local-only");
        QTRY_COMPARE(server.calls.size(), 1);
        server.session(0);
        QTRY_COMPARE(login.count(), 1);
        // 12,000 supplementary characters use 24,000 UTF-16 code units but fit the backend limit.
        const QString message = QStringLiteral("🌐").repeated(12000);
        client.send(op, conversation, message);
        QTRY_COMPARE(server.calls.size(), 2);
        QCOMPARE(QJsonDocument::fromJson(server.calls[1].body).object().value("message").toString(), message);
        server.calls[1].socket->abort();
        QTRY_COMPARE_WITH_TIMEOUT(result.count(), 1, 2500);
        QCOMPARE(result.first()[2].toString(), "unknown");
        QCOMPARE(server.calls.size(), 2);
    }

    void cancelAndExpiry()
    {
        LocalGateway server;
        QVERIFY(server.listen(QHostAddress::LocalHost, 0));
        Transport client;
        QSignalSpy login(&client, &Transport::loginFinished), result(&client, &Transport::chatFinished);
        QSignalSpy expired(&client, &Transport::sessionExpired);
        client.login(server.origin(), "a", "local-only");
        QTRY_COMPARE(server.calls.size(), 1);
        server.session(0);
        QTRY_COMPARE(login.count(), 1);
        client.send(op, conversation, "original");
        QTRY_COMPARE(server.calls.size(), 2);
        client.cancel(op);
        server.respond(1, 200, receipt(conversation));
        client.send("op-b", conversation, "another");
        QTRY_COMPARE(server.calls.size(), 3);
        server.respond(2, 403, "{}");
        QTRY_COMPARE(expired.count(), 1);
        QCOMPARE(result.count(), 0);
        client.login(server.origin(), "b", "local-only");
        QTRY_COMPARE(server.calls.size(), 4);
        QVERIFY(!server.calls[3].headers.contains("cookie"));
        server.session(3, false);
        QTRY_COMPARE(login.count(), 2);
        QVERIFY(!login.last()[0].toBool());
    }

    void redirectDoesNotForwardCredentials()
    {
        LocalGateway server, other;
        QVERIFY(server.listen(QHostAddress::LocalHost, 0));
        QVERIFY(other.listen(QHostAddress::LocalHost, 0));
        Transport client;
        QSignalSpy login(&client, &Transport::loginFinished);
        client.login(server.origin(), "a", "local-only");
        QTRY_COMPARE(server.calls.size(), 1);
        server.respond(0, 307, "{}", "Location: " + other.origin().toEncoded() + "/api/python/demo/session\r\n");
        QTRY_COMPARE(login.count(), 1);
        QVERIFY(!login.first()[0].toBool());
        QVERIFY(other.calls.isEmpty());
    }
    void invalidOrigins()
    {
        Transport client;
        QSignalSpy login(&client, &Transport::loginFinished);
        for (const auto &bad : {"http://example.invalid", "https://user@localhost", "https://localhost/path",
                               "https://localhost/?query", "https://localhost/#fragment"}) {
            client.login(QUrl(bad), "a", "local-only");
            QVERIFY(!login.isEmpty());
            QVERIFY(!login.takeFirst()[0].toBool());
        }
    }
};

QTEST_GUILESS_MAIN(TestTransport)
#include "tst_transport.moc"
