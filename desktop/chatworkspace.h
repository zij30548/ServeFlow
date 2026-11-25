#pragma once

#include <QAbstractListModel>
#include <QHash>
#include <QSet>
#include <QVariantList>

class ChatWorkspace final : public QAbstractListModel {
    Q_OBJECT
    Q_PROPERTY(QVariantList conversations READ conversations NOTIFY conversationsChanged)
    Q_PROPERTY(QString activeId READ activeId NOTIFY activeChanged)
    Q_PROPERTY(QString draft READ draft WRITE setDraft NOTIFY draftChanged)
    Q_PROPERTY(bool busy READ busy NOTIFY availabilityChanged)
    Q_PROPERTY(bool canSend READ canSend NOTIFY availabilityChanged)
    Q_PROPERTY(bool online READ online NOTIFY onlineChanged)
    Q_PROPERTY(QString sendHint READ sendHint NOTIFY availabilityChanged)

public:
    enum Role { SpeakerRole = Qt::UserRole + 1, BodyRole, StatusRole };
    explicit ChatWorkspace(QObject *parent = nullptr);
    int rowCount(const QModelIndex &parent = {}) const override;
    QVariant data(const QModelIndex &index, int role) const override;
    QHash<int, QByteArray> roleNames() const override;
    QVariantList conversations() const;
    QString activeId() const;
    QString draft() const;
    bool busy() const;
    bool online() const;
    bool canSend() const;
    QString sendHint() const;
    void setDraft(const QString &text);

    Q_INVOKABLE void newConversation();
    Q_INVOKABLE void selectConversation(const QString &id);
    Q_INVOKABLE void send();
    Q_INVOKABLE void cancel();

public slots:
    void openSession();
    void clearSession();
    void finishTurn(const QString &operationId, const QString &conversationId,
                    const QString &status, const QString &text);

signals:
    void conversationsChanged();
    void activeChanged();
    void draftChanged();
    void availabilityChanged();
    void onlineChanged();
    void sendRequested(QString operationId, QString conversationId, QString message);
    void cancelRequested(QString operationId);

private:
    struct Message { QString speaker; QString body; QString status; };
    struct Conversation {
        QString id;
        QString title;
        QString draft;
        QList<Message> messages;
        QString pendingId;
        QString pendingText;
        // Preserve the wire identity if the user explicitly resubmits a known failed turn.
        QHash<QString, QString> failedIds;
        QSet<QString> unresolvedTexts;
        bool unread = false;
    };

    Conversation *active();
    const Conversation *active() const;
    void append(Conversation &conversation, const Message &message);
    void notifySelection();
    QList<Conversation> m_conversations;
    int m_active = -1;
    bool m_online = false;
};
