#include "chatworkspace.h"

#include <QUuid>

namespace {
QString newId() { return QUuid::createUuid().toString(QUuid::WithoutBraces); }
constexpr auto unknownText = "本次发送结果待核对，服务端可能仍在处理。";
}

ChatWorkspace::ChatWorkspace(QObject *parent) : QAbstractListModel(parent) {}

ChatWorkspace::Conversation *ChatWorkspace::active() {
    return m_active >= 0 ? &m_conversations[m_active] : nullptr;
}

const ChatWorkspace::Conversation *ChatWorkspace::active() const {
    return m_active >= 0 ? &m_conversations.at(m_active) : nullptr;
}

int ChatWorkspace::rowCount(const QModelIndex &parent) const {
    return !parent.isValid() && active() ? static_cast<int>(active()->messages.size()) : 0;
}

QVariant ChatWorkspace::data(const QModelIndex &index, int role) const {
    if (!index.isValid() || index.model() != this || index.column() != 0 || index.row() < 0 || index.row() >= rowCount()) return {};
    const auto &message = active()->messages.at(index.row());
    switch (role) {
    case SpeakerRole: return message.speaker;
    case BodyRole: return message.body;
    case StatusRole: return message.status;
    default: return {};
    }
}

QHash<int, QByteArray> ChatWorkspace::roleNames() const {
    return {{SpeakerRole,"speaker"}, {BodyRole,"body"}, {StatusRole,"status"}};
}

QVariantList ChatWorkspace::conversations() const {
    QVariantList result;
    for (const auto &conversation : m_conversations) {
        result.append(QVariantMap{{"id", conversation.id}, {"title", conversation.title},
                                  {"busy", !conversation.pendingId.isEmpty()}, {"unread", conversation.unread}});
    }
    return result;
}

QString ChatWorkspace::activeId() const { return active() ? active()->id : QString(); }
QString ChatWorkspace::draft() const { return active() ? active()->draft : QString(); }
bool ChatWorkspace::busy() const { return active() && !active()->pendingId.isEmpty(); }
bool ChatWorkspace::online() const { return m_online; }

QString ChatWorkspace::sendHint() const {
    if (!active()) return {};
    const QString message = draft().trimmed();
    if (message.size() > 12000 && message.toUcs4().size() > 12000)
        return QStringLiteral("单次消息最多 12000 个字符。");
    if (active()->unresolvedTexts.contains(message))
        return QStringLiteral("这条消息的处理结果仍待核对，请勿重复发送。");
    return {};
}

bool ChatWorkspace::canSend() const {
    return online() && active() && !busy() && !draft().trimmed().isEmpty() && sendHint().isEmpty();
}

void ChatWorkspace::setDraft(const QString &text) {
    if (!active() || active()->draft == text) return;
    active()->draft = text;
    emit draftChanged();
    emit availabilityChanged();
}

void ChatWorkspace::notifySelection() {
    emit activeChanged();
    emit draftChanged();
    emit availabilityChanged();
    emit conversationsChanged();
}

void ChatWorkspace::newConversation() {
    if (!online()) return;
    beginResetModel();
    Conversation next;
    next.id = newId();
    next.title = QStringLiteral("新对话");
    m_conversations.append(next);
    m_active = static_cast<int>(m_conversations.size() - 1);
    endResetModel();
    notifySelection();
}

void ChatWorkspace::selectConversation(const QString &id) {
    if (id == activeId()) return;
    for (int i = 0; i < m_conversations.size(); ++i) {
        if (m_conversations[i].id != id) continue;
        beginResetModel();
        m_active = i;
        active()->unread = false;
        endResetModel();
        notifySelection();
        return;
    }
}

void ChatWorkspace::openSession() {
    clearSession();
    m_online = true;
    emit onlineChanged();
    newConversation();
}

void ChatWorkspace::clearSession() {
    if (!m_online && m_conversations.isEmpty()) return;
    QStringList pending;
    for (const auto &conversation : m_conversations)
        if (!conversation.pendingId.isEmpty()) pending.append(conversation.pendingId);
    beginResetModel();
    m_online = false;
    m_active = -1;
    m_conversations.clear();
    endResetModel();
    emit onlineChanged();
    notifySelection();
    // Invalidate ownership first: aborts may synchronously complete on some backends.
    for (const auto &id : pending) emit cancelRequested(id);
}

void ChatWorkspace::append(Conversation &conversation, const Message &message) {
    const bool visible = conversation.id == activeId();
    const int row = static_cast<int>(conversation.messages.size());
    if (visible) beginInsertRows({}, row, row);
    conversation.messages.append(message);
    if (visible) endInsertRows();
}

void ChatWorkspace::send() {
    if (!canSend()) return;
    auto &conversation = *active();
    const QString id = conversation.id;
    const QString text = conversation.draft.trimmed();
    QString operation = conversation.failedIds.take(text);
    if (operation.isEmpty()) operation = newId();
    conversation.pendingId = operation;
    conversation.pendingText = text;
    conversation.draft.clear();
    if (conversation.messages.isEmpty()) conversation.title = text.simplified().left(48);
    append(conversation, {"user", text, "sent"});
    emit draftChanged();
    emit availabilityChanged();
    emit conversationsChanged();
    emit sendRequested(operation, id, text);
}

void ChatWorkspace::cancel() {
    if (!busy()) return;
    const QString id = activeId();
    const QString operation = active()->pendingId;
    finishTurn(operation, id, "unknown", QStringLiteral("已取消等待。服务端可能仍在处理本次发送。"));
    emit cancelRequested(operation);
}

void ChatWorkspace::finishTurn(const QString &operationId, const QString &conversationId,
                               const QString &status, const QString &text) {
    if (!online() || operationId.isEmpty()) return;
    for (auto &conversation : m_conversations) {
        if (conversation.id != conversationId || conversation.pendingId != operationId) continue;
        const bool answered = (status == "success" || status == "partial") && !text.trimmed().isEmpty();
        const bool failed = status == "failed";
        if (failed) conversation.failedIds.insert(conversation.pendingText, operationId);
        if (!answered && !failed) conversation.unresolvedTexts.insert(conversation.pendingText);
        conversation.pendingId.clear();
        conversation.pendingText.clear();
        conversation.unread = conversation.id != activeId();
        if (answered) {
            append(conversation, {"assistant", text, status});
        } else {
            const bool trustedText = (failed || status == "unknown") && !text.trimmed().isEmpty();
            append(conversation, {"notice", trustedText ? text : QString::fromUtf8(unknownText),
                                  failed ? "failed" : "unknown"});
        }
        emit conversationsChanged();
        if (conversation.id == activeId()) emit availabilityChanged();
        return;
    }
}
