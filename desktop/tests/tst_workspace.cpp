#include "chatworkspace.h"

#include <QAbstractItemModelTester>
#include <QSignalSpy>
#include <QtTest>

class WorkspaceTest : public QObject {
    Q_OBJECT
private slots:
    void backgroundCompletionPreservesOtherDraft();
    void cancelBeforeAbortDeliveryAndLateCompletion();
    void logoutIsolatesOldIdentity();
    void duplicateClicksAndManualRejectedRetry();
    void unicodeBoundaryAndMalformedResult();
};

void WorkspaceTest::backgroundCompletionPreservesOtherDraft() {
    ChatWorkspace workspace;
    QAbstractItemModelTester modelCheck(&workspace, QAbstractItemModelTester::FailureReportingMode::QtTest);
    QSignalSpy sends(&workspace, &ChatWorkspace::sendRequested);
    workspace.openSession();
    const auto a = workspace.activeId();
    workspace.setDraft("查订单 A");
    workspace.send();
    const auto operationA = sends.last().at(0).toString();
    workspace.newConversation();
    const auto b = workspace.activeId();
    QVERIFY(a != b);
    workspace.setDraft("查订单 B");
    workspace.send();
    const auto operationB = sends.last().at(0).toString();
    workspace.setDraft("属于 B 的半条追问");
    workspace.finishTurn(operationA, b, "success", "错误绑定");
    QVERIFY(workspace.busy());
    QCOMPARE(workspace.rowCount(), 1);
    workspace.finishTurn(operationA, a, "partial", "订单 A 的回复");
    QCOMPARE(workspace.activeId(), b);
    QVERIFY(workspace.busy());
    QVERIFY(workspace.conversations().at(0).toMap().value("unread").toBool());
    QCOMPARE(workspace.rowCount(), 1);
    workspace.selectConversation(a);
    QCOMPARE(workspace.rowCount(), 2);
    QCOMPARE(workspace.data(workspace.index(1), ChatWorkspace::BodyRole).toString(), "订单 A 的回复");
    QCOMPARE(workspace.data(workspace.index(1), ChatWorkspace::StatusRole).toString(), "partial");
    QVERIFY(!workspace.conversations().at(0).toMap().value("unread").toBool());
    workspace.selectConversation(b);
    QCOMPARE(workspace.draft(), "属于 B 的半条追问");
    workspace.finishTurn(operationB, b, "success", "订单 B 的回复");
    QCOMPARE(workspace.data(workspace.index(1), ChatWorkspace::BodyRole).toString(), "订单 B 的回复");
    QVERIFY(workspace.canSend());
}

void WorkspaceTest::cancelBeforeAbortDeliveryAndLateCompletion() {
    ChatWorkspace workspace;
    QSignalSpy sends(&workspace, &ChatWorkspace::sendRequested);
    workspace.openSession();
    const auto a = workspace.activeId();
    workspace.setDraft("第一个问题");
    workspace.send();
    const auto first = sends.last().at(0).toString();
    connect(&workspace, &ChatWorkspace::cancelRequested, &workspace, [&](const QString &id){
        workspace.finishTurn(id, a, "failed", "同步送达的旧错误");
    });
    workspace.cancel();
    QCOMPARE(workspace.data(workspace.index(1), ChatWorkspace::StatusRole).toString(), "unknown");
    workspace.setDraft("第一个问题");
    QVERIFY(!workspace.canSend());
    QVERIFY(!workspace.sendHint().isEmpty());
    workspace.send();
    QCOMPARE(sends.size(), 1);
    workspace.setDraft("另一个问题");
    workspace.send();
    workspace.newConversation();
    workspace.selectConversation(a);
    workspace.finishTurn(first, a, "success", "取消后的旧成功");
    QVERIFY(workspace.busy());
    QCOMPARE(workspace.rowCount(), 3);
    workspace.finishTurn(sends.last().at(0).toString(), a, "success", "本轮回复");
    QCOMPARE(workspace.rowCount(), 4);
    QVERIFY(!workspace.busy());
}

void WorkspaceTest::logoutIsolatesOldIdentity() {
    ChatWorkspace workspace;
    QSignalSpy sends(&workspace, &ChatWorkspace::sendRequested);
    QSignalSpy cancels(&workspace, &ChatWorkspace::cancelRequested);
    workspace.newConversation();
    workspace.setDraft("未登录发送");
    workspace.send();
    QVERIFY(sends.isEmpty());
    QVERIFY(workspace.conversations().isEmpty());
    workspace.openSession();
    const auto a = workspace.activeId();
    workspace.setDraft("旧身份的问题");
    workspace.send();
    workspace.newConversation();
    workspace.setDraft("另一会话的问题");
    workspace.send();
    workspace.setDraft("旧身份草稿");
    workspace.clearSession();
    QCOMPARE(cancels.size(), 2);
    QVERIFY(!workspace.online());
    QVERIFY(workspace.draft().isEmpty());
    workspace.openSession();
    QVERIFY(workspace.activeId() != a);
    workspace.finishTurn(sends.first().at(0).toString(), a, "success", "旧身份资料");
    QCOMPARE(workspace.conversations().size(), 1);
    QCOMPARE(workspace.rowCount(), 0);
    QCOMPARE(workspace.draft(), QString());
}

void WorkspaceTest::duplicateClicksAndManualRejectedRetry() {
    ChatWorkspace workspace;
    QSignalSpy sends(&workspace, &ChatWorkspace::sendRequested);
    workspace.openSession();
    const auto a = workspace.activeId();
    workspace.setDraft("  测试  ");
    workspace.send();
    workspace.setDraft("测试");
    workspace.send();
    QCOMPARE(sends.size(), 1);
    const auto operation = sends.first().at(0).toString();
    QCOMPARE(sends.first().at(2).toString(), "测试");
    workspace.finishTurn(operation, a, "failed", "本次未完成");
    QCOMPARE(sends.size(), 1);
    QCOMPARE(workspace.data(workspace.index(1), ChatWorkspace::SpeakerRole).toString(), "notice");
    workspace.send();
    QCOMPARE(sends.size(), 2);
    QCOMPARE(sends.last().at(0).toString(), operation);
    workspace.finishTurn(operation, a, "success", "已确认");
    workspace.setDraft("测试");
    workspace.send();
    QVERIFY(sends.last().at(0).toString() != operation);
}

void WorkspaceTest::unicodeBoundaryAndMalformedResult() {
    ChatWorkspace workspace;
    QSignalSpy sends(&workspace, &ChatWorkspace::sendRequested);
    workspace.openSession();
    const QString emoji = QStringLiteral("🚲");
    workspace.setDraft(emoji.repeated(12001));
    QVERIFY(!workspace.canSend());
    workspace.setDraft(emoji.repeated(12000));
    QVERIFY(workspace.canSend());
    workspace.send();
    workspace.finishTurn(sends.first().at(0).toString(), workspace.activeId(), "bogus", "不可展示");
    QCOMPARE(workspace.rowCount(), 2);
    QVERIFY(workspace.data(workspace.index(1), ChatWorkspace::BodyRole).toString() != "不可展示");
    QCOMPARE(workspace.data(workspace.index(1), ChatWorkspace::StatusRole).toString(), "unknown");
}

QTEST_GUILESS_MAIN(WorkspaceTest)
#include "tst_workspace.moc"
