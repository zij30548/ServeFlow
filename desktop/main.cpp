#include <QCommandLineParser>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>

#include "chatworkspace.h"
#include "gateway.h"
#include "transport.h"

int main(int argc, char **argv)
{
    QGuiApplication app(argc, argv);
    app.setApplicationName(QStringLiteral("ServeFlow"));
    app.setApplicationDisplayName(QStringLiteral("智应客服工作台"));
    QCommandLineParser parser;
    parser.addHelpOption();
    QCommandLineOption originOption(QStringLiteral("origin"), QStringLiteral("受保护网关的 HTTPS 来源"),
                                    QStringLiteral("url"), QStringLiteral("https://serveflow.zijinglab.com"));
    parser.addOption(originOption);
    parser.process(app);
    const QUrl origin(parser.value(originOption), QUrl::StrictMode);
    if (!origin.isValid() || origin.scheme() != QStringLiteral("https") || origin.host().isEmpty()
        || origin.authority().contains('@') || origin.hasQuery() || origin.hasFragment()
        || (!origin.path().isEmpty() && origin.path() != QStringLiteral("/"))) {
        qCritical("--origin requires an HTTPS origin with no user information, query, fragment or path.");
        return 2;
    }
    QQuickStyle::setStyle(QStringLiteral("Basic"));
    Transport transport;
    ChatWorkspace workspace;
    Gateway gateway(origin.adjusted(QUrl::StripTrailingSlash), transport, workspace);
    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty(QStringLiteral("workspace"), &workspace);
    engine.rootContext()->setContextProperty(QStringLiteral("gateway"), &gateway);
    engine.load(QUrl(QStringLiteral("qrc:/Main.qml")));
    if (engine.rootObjects().isEmpty()) return 1;
    QObject::connect(&app, &QCoreApplication::aboutToQuit, &gateway, &Gateway::signOut);
    return app.exec();
}
