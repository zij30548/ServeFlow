#pragma once

#include <QByteArray>
#include <QHash>
#include <QObject>
#include <QTimer>
#include <QUrl>
#include <functional>

class QNetworkAccessManager;
class QNetworkReply;
class QNetworkRequest;

// One gateway identity per instance. Login creates a fresh in-memory cookie jar.
class Transport : public QObject {
    Q_OBJECT
public:
    explicit Transport(QObject *parent = nullptr);
    void login(const QUrl &origin, const QString &username, const QString &password);
    void logout();
    void send(const QString &operationId, const QString &conversationId, const QString &message);
    void cancel(const QString &operationId);

signals:
    void loginFinished(bool ok, QString detail);
    void sessionExpired();
    void chatFinished(QString operationId, QString conversationId, QString status, QString text);

private:
    QNetworkRequest request(const QString &path) const;
    using Completion = std::function<void(int, bool, const QByteArray &)>;
    void watch(const QString &key, QNetworkReply *reply, int timeoutMs, Completion completion);
    void expire();

    QNetworkAccessManager *m_network = nullptr;
    QHash<QString, QNetworkReply *> m_pending;
    QUrl m_origin;
    QByteArray m_basic;
    QString m_userId;
    QByteArray m_csrf;
    QTimer m_expiration;
};
