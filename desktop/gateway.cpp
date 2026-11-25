#include "gateway.h"
#include "chatworkspace.h"
#include "transport.h"

#include <utility>

Gateway::Gateway(QUrl origin, Transport &transport, ChatWorkspace &workspace, QObject *parent)
    : QObject(parent), m_origin(std::move(origin)), m_transport(transport), m_workspace(workspace)
{
    connect(&workspace, &ChatWorkspace::sendRequested, &transport, &Transport::send);
    connect(&workspace, &ChatWorkspace::cancelRequested, &transport, &Transport::cancel);
    connect(&transport, &Transport::chatFinished, &workspace, &ChatWorkspace::finishTurn);
    connect(&transport, &Transport::loginFinished, this, [this](bool ok, const QString &detail) {
        if (!m_busy) return;
        m_busy = false;
        m_signedIn = ok;
        m_error = ok ? QString() : detail;
        if (ok) m_workspace.openSession();
        emit changed();
    });
    connect(&transport, &Transport::sessionExpired, this, [this] {
        signOut();
        m_error = QStringLiteral("访问身份已失效，请重新登录。");
        emit changed();
    });
}

void Gateway::signIn(const QString &user, const QString &password)
{
    if (m_busy || m_signedIn || user.trimmed().isEmpty() || password.isEmpty()) return;
    m_busy = true;
    m_error.clear();
    emit changed();
    m_transport.login(m_origin, user.trimmed(), password);
}

void Gateway::signOut()
{
    m_busy = false;
    m_signedIn = false;
    m_error.clear();
    m_workspace.clearSession();
    m_transport.logout();
    emit changed();
}
