#pragma once

#include <QObject>
#include <QUrl>

class ChatWorkspace;
class Transport;

// Owns the sign-in lifecycle shared by the window and its conversations.
class Gateway : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString origin READ origin CONSTANT)
    Q_PROPERTY(bool busy READ busy NOTIFY changed)
    Q_PROPERTY(bool signedIn READ signedIn NOTIFY changed)
    Q_PROPERTY(QString error READ error NOTIFY changed)
public:
    Gateway(QUrl origin, Transport &transport, ChatWorkspace &workspace, QObject *parent = nullptr);
    QString origin() const { return m_origin.toString(); }
    bool busy() const { return m_busy; }
    bool signedIn() const { return m_signedIn; }
    QString error() const { return m_error; }
    Q_INVOKABLE void signIn(const QString &user, const QString &password);
    Q_INVOKABLE void signOut();

signals:
    void changed();

private:
    QUrl m_origin;
    Transport &m_transport;
    ChatWorkspace &m_workspace;
    bool m_busy = false;
    bool m_signedIn = false;
    QString m_error;
};
