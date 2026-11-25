#include "chatworkspace.h"
#include "gateway.h"
#include "transport.h"

#include <QFile>
#include <QInputMethodEvent>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickItem>
#include <QQuickStyle>
#include <QQuickWindow>
#include <QtTest>

namespace {
QQuickItem *control(QQuickItem *root, const QString &name)
{
    if (root->objectName() == name) return root;
    for (auto *item : root->childItems()) {
        if (auto *match = control(item, name)) return match;
    }
    return nullptr;
}
void click(QQuickWindow *window, QQuickItem *item)
{
    const auto center = item->mapToScene(item->boundingRect().center()).toPoint();
    QTest::mouseClick(window, Qt::LeftButton, Qt::NoModifier, center);
}
void enter(QQuickItem *item, const QString &text)
{
    item->forceActiveFocus();
    QInputMethodEvent input;
    input.setCommitString(text);
    QCoreApplication::sendEvent(item, &input);
}
}

class NativeFlow : public QObject {
    Q_OBJECT
private slots:
    void checkOrderInBackground();
};

void NativeFlow::checkOrderInBackground()
{
    const auto port = qEnvironmentVariableIntValue("SERVEFLOW_FIXTURE_PORT");
    QVERIFY(port > 0 && port <= 65535);
    Transport transport;
    ChatWorkspace workspace;
    Gateway gateway(QUrl(QStringLiteral("http://127.0.0.1:%1").arg(port)), transport, workspace);
    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty("workspace", &workspace);
    engine.rootContext()->setContextProperty("gateway", &gateway);
    QQuickStyle::setStyle("Basic");
    engine.load(QUrl(QStringLiteral("qrc:/Main.qml")));
    QCOMPARE(engine.rootObjects().size(), 1);
    auto *win = qobject_cast<QQuickWindow *>(engine.rootObjects().first());
    QVERIFY(win && QTest::qWaitForWindowExposed(win));
    auto field = [win](const QString &id) { return control(win->contentItem(), id); };
    QVERIFY(field("loginUsername") && field("loginPassword") && field("loginSubmit"));
    enter(field("loginUsername"), "seat");
    enter(field("loginPassword"), qEnvironmentVariable("SERVEFLOW_FIXTURE_PASSWORD"));
    click(win, field("loginSubmit"));
    QTRY_VERIFY_WITH_TIMEOUT(gateway.signedIn(), 5000);
    const QString first = workspace.activeId();
    QVERIFY(!first.isEmpty());
    QTRY_VERIFY(field("chatComposer") && field("newChat"));
    enter(field("chatComposer"), QStringLiteral("查询 DEMO-1001 的物流状态"));
    QTRY_VERIFY(workspace.canSend());
    QVERIFY(field("chatSend"));
    click(win, field("chatSend"));
    QTRY_VERIFY(workspace.busy());
    click(win, field("newChat"));
    QTRY_VERIFY(workspace.activeId() != first);
    const QString second = workspace.activeId();
    enter(field("chatComposer"), QStringLiteral("另一位客户的草稿"));
    QFile releaseOrder(qEnvironmentVariable("SERVEFLOW_FIXTURE_RELEASE_ORDER"));
    QVERIFY(releaseOrder.open(QIODevice::WriteOnly | QIODevice::NewOnly));
    releaseOrder.close();
    QTRY_VERIFY(workspace.conversations().first().toMap().value("unread").toBool());
    QCOMPARE(workspace.rowCount(), 0);
    QCOMPARE(workspace.draft(), QStringLiteral("另一位客户的草稿"));
    QTRY_VERIFY(field("conversation-" + first));
    click(win, field("conversation-" + first));
    QTRY_COMPARE(workspace.activeId(), first);
    QCOMPARE(workspace.rowCount(), 2);
    const auto answer = workspace.data(workspace.index(1), ChatWorkspace::BodyRole).toString();
    QVERIFY2(answer.contains("DEMO-1001") && answer.contains(QStringLiteral("演示家庭中控屏")), qPrintable(answer));
    const QString screenshot = qEnvironmentVariable("SERVEFLOW_TEST_SCREENSHOT");
    if (!screenshot.isEmpty()) {
        QTest::qWait(100);
        QVERIFY(win->grabWindow().save(screenshot));
    }
    enter(field("chatComposer"), QStringLiteral("请说明配送规则、送达时效和免邮门槛。"));
    click(win, field("chatSend"));
    QTRY_COMPARE(workspace.rowCount(), 4);
    const auto policy = workspace.data(workspace.index(3), ChatWorkspace::BodyRole).toString();
    QVERIFY2(policy.contains(QStringLiteral("配送说明")) && policy.contains(QStringLiteral("不设全国统一送达天数")), qPrintable(policy));
    QTRY_VERIFY(field("conversation-" + second));
    click(win, field("conversation-" + second));
    QTRY_COMPARE(workspace.activeId(), second);
    QTRY_COMPARE(field("chatComposer")->property("text").toString(), workspace.draft());
    QVERIFY(field("logoutButton"));
    click(win, field("logoutButton"));
    QTRY_VERIFY(!gateway.signedIn());
    QVERIFY(workspace.conversations().isEmpty());
}
QTEST_MAIN(NativeFlow)
#include "tst_window.moc"
