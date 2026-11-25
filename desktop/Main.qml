pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: app
    width: 980
    height: 680
    minimumWidth: 680
    minimumHeight: 520
    title: "智应客服工作台"
    visible: true
    color: canvas

    readonly property color canvas: "#f7f8f6"
    readonly property color ink: "#192923"
    readonly property color muted: "#52655d"
    readonly property color green: "#1e5947"
    readonly property color line: "#cfd8d1"
    readonly property color tint: "#deebe2"

    component Action: Button {
        id: button
        property bool primary: false
        padding: 10
        leftPadding: 16
        rightPadding: 16
        implicitHeight: 40
        font.pixelSize: 14
        contentItem: Text {
            text: button.text
            font: button.font
            textFormat: Text.PlainText
            color: !button.enabled ? app.muted : button.primary ? "#ffffff" : app.ink
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            elide: Text.ElideRight
        }
        background: Rectangle {
            radius: 7
            color: !button.enabled ? "#e1e6e2" : button.primary
                   ? (button.down ? "#123a2e" : button.hovered ? "#174b3b" : app.green)
                   : (button.down || button.hovered ? app.tint : "#ffffff")
            border.color: button.visualFocus ? app.green : button.primary ? "transparent" : app.line
            border.width: button.visualFocus ? 2 : 1
        }
    }

    component Field: TextField {
        implicitHeight: 44
        font.pixelSize: 15
        leftPadding: 12
        rightPadding: 12
        color: app.ink
        selectByMouse: true
        background: Rectangle {
            radius: 7
            color: "#ffffff"
            border.color: parent.activeFocus ? app.green : app.line
            border.width: parent.activeFocus ? 2 : 1
        }
    }

    Loader {
        anchors.fill: parent
        sourceComponent: gateway.signedIn ? chatPage : signInPage
    }

    Component {
        id: signInPage
        Item {
            id: signIn
            function submit() {
                if (!login.enabled) return
                gateway.signIn(user.text.trim(), password.text)
                password.clear()
            }
            ColumnLayout {
                anchors.centerIn: parent
                width: 336
                spacing: 10
                Label {
                    text: "智应客服工作台"
                    color: app.ink
                    font.pixelSize: 28
                    font.weight: Font.DemiBold
                }
                Label {
                    Layout.fillWidth: true
                    Layout.bottomMargin: 14
                    text: gateway.origin
                    elide: Text.ElideMiddle
                    textFormat: Text.PlainText
                    color: app.muted
                }
                Label { text: "坐席账号"; color: app.ink }
                Field {
                    id: user
                    objectName: "loginUsername"
                    Layout.fillWidth: true
                    Accessible.name: "坐席账号"
                    enabled: !gateway.busy
                    Component.onCompleted: forceActiveFocus()
                    onAccepted: password.forceActiveFocus()
                }
                Label { text: "访问密码"; color: app.ink; Layout.topMargin: 6 }
                Field {
                    id: password
                    objectName: "loginPassword"
                    Layout.fillWidth: true
                    Accessible.name: "访问密码"
                    echoMode: TextInput.Password
                    enabled: !gateway.busy
                    onAccepted: signIn.submit()
                }
                Label {
                    Layout.fillWidth: true
                    visible: text.length > 0
                    text: gateway.error
                    textFormat: Text.PlainText
                    wrapMode: Text.Wrap
                    color: "#98412e"
                }
                Action {
                    id: login
                    objectName: "loginSubmit"
                    Layout.fillWidth: true
                    Layout.topMargin: 12
                    primary: true
                    enabled: !gateway.busy && user.text.trim().length > 0 && password.text.length > 0
                    text: gateway.busy ? "正在登录…" : "登录工作台"
                    onClicked: signIn.submit()
                }
            }
        }
    }

    Component {
        id: chatPage
        RowLayout {
            id: page
            spacing: 0
            Shortcut {
                sequences: [StandardKey.New]
                onActivated: workspace.newConversation()
            }
            Rectangle {
                Layout.preferredWidth: app.width < 800 ? 188 : 234
                Layout.fillHeight: true
                color: "#eff3ef"
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 14
                    Label {
                        Layout.topMargin: 7
                        text: "智应客服工作台"
                        font.pixelSize: 19
                        font.weight: Font.DemiBold
                        color: app.ink
                    }
                    Action {
                        objectName: "newChat"
                        Layout.fillWidth: true
                        primary: true
                        text: "新建对话"
                        onClicked: workspace.newConversation()
                    }
                    Label { text: "会话"; color: app.muted; font.pixelSize: 13 }
                    ListView {
                        id: threadList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        spacing: 3
                        model: workspace.conversations
                        delegate: ItemDelegate {
                            id: row
                            required property var modelData
                            objectName: "conversation-" + modelData.id
                            width: threadList.width
                            height: 48
                            checkable: false
                            Accessible.name: modelData.title + (modelData.unread ? "，有新回复" : "")
                            onClicked: workspace.selectConversation(modelData.id)
                            background: Rectangle {
                                radius: 7
                                color: row.modelData.id === workspace.activeId ? app.tint : row.hovered ? "#e8ede8" : "transparent"
                                border.color: row.visualFocus ? app.green : "transparent"
                            }
                            contentItem: RowLayout {
                                spacing: 6
                                Label {
                                    text: row.modelData.title
                                    textFormat: Text.PlainText
                                    Layout.fillWidth: true
                                    elide: Text.ElideRight
                                    color: app.ink
                                }
                                Label {
                                    visible: row.modelData.busy || row.modelData.unread
                                    text: row.modelData.busy ? "处理中" : "新回复"
                                    font.pixelSize: 11
                                    color: app.green
                                }
                            }
                        }
                        ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }
                    }
                    Action {
                        objectName: "logoutButton"
                        text: "退出登录"
                        Layout.fillWidth: true
                        onClicked: gateway.signOut()
                    }
                }
            }
            Rectangle { Layout.preferredWidth: 1; Layout.fillHeight: true; color: app.line }
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 0
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 62
                    color: "white"
                    Label {
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.left: parent.left
                        anchors.leftMargin: 24
                        text: "客服对话"
                        font.pixelSize: 20
                        font.weight: Font.DemiBold
                        color: app.ink
                    }
                }
                Rectangle { Layout.preferredHeight: 1; Layout.fillWidth: true; color: app.line }
                ListView {
                    id: messages
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.leftMargin: 20
                    Layout.rightMargin: 20
                    topMargin: 14
                    bottomMargin: 14
                    spacing: 12
                    clip: true
                    model: workspace
                    ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }
                    function revealLatest() {
                        if (count === 0) return
                        forceLayout()
                        positionViewAtIndex(count - 1, ListView.Contain)
                    }
                    onCountChanged: Qt.callLater(revealLatest)
                    Connections {
                        target: workspace
                        function onActiveChanged() { Qt.callLater(messages.revealLatest) }
                    }
                    Label {
                        anchors.centerIn: parent
                        visible: messages.count === 0
                        text: "输入问题，开始对话"
                        color: app.muted
                        font.pixelSize: 17
                    }
                    delegate: Item {
                        id: entry
                        required property string speaker
                        required property string body
                        required property string status
                        readonly property bool own: speaker === "user"
                        width: messages.width
                        height: bubble.height
                        Rectangle {
                            id: bubble
                            anchors.right: entry.own ? parent.right : undefined
                            anchors.left: entry.own ? undefined : parent.left
                            width: Math.min(parent.width - 12, Math.max(140, messageText.implicitWidth + 28))
                            height: column.implicitHeight + 20
                            radius: 10
                            color: entry.own ? app.tint : "#ffffff"
                            border.color: entry.own ? "transparent" : app.line
                            Column {
                                id: column
                                x: 14
                                y: 10
                                width: parent.width - 28
                                spacing: 5
                                Label {
                                    text: entry.own ? "我" : entry.speaker === "notice" ? "系统提示"
                                          : entry.status === "partial" ? "客服助手 · 部分完成"
                                          : entry.status === "failed" ? "客服助手 · 处理失败" : "客服助手"
                                    color: app.muted
                                    font.pixelSize: 12
                                }
                                TextEdit {
                                    id: messageText
                                    width: parent.width
                                    readOnly: true
                                    selectByMouse: true
                                    textFormat: TextEdit.PlainText
                                    text: entry.body
                                    wrapMode: TextEdit.Wrap
                                    color: app.ink
                                    font.pixelSize: 15
                                    Accessible.name: entry.body
                                }
                            }
                        }
                    }
                }
                Rectangle {
                    Layout.fillWidth: true
                    Layout.leftMargin: 18
                    Layout.rightMargin: 18
                    Layout.bottomMargin: 16
                    implicitHeight: controls.implicitHeight + 20
                    radius: 10
                    color: "white"
                    border.color: composer.activeFocus ? app.green : app.line
                    ColumnLayout {
                        id: controls
                        anchors.fill: parent
                        anchors.margins: 10
                        spacing: 6
                        ScrollView {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 72
                            contentWidth: availableWidth
                            TextArea {
                                id: composer
                                objectName: "chatComposer"
                                Accessible.name: "消息"
                                placeholderText: "输入消息…"
                                textFormat: TextEdit.PlainText
                                text: workspace.draft
                                onTextChanged: { if (workspace.draft !== text) workspace.draft = text }
                                selectByMouse: true
                                wrapMode: TextEdit.Wrap
                                color: app.ink
                                font.pixelSize: 15
                                background: Item {}
                                Keys.onPressed: function(event) {
                                    if (!inputMethodComposing && (event.key === Qt.Key_Return || event.key === Qt.Key_Enter)
                                            && (event.modifiers & (Qt.ControlModifier | Qt.MetaModifier))) {
                                        if (workspace.canSend && !event.isAutoRepeat) workspace.send()
                                        event.accepted = true
                                    }
                                }
                            }
                        }
                        Label {
                            Layout.fillWidth: true
                            visible: text.length > 0
                            text: workspace.sendHint
                            color: "#98412e"
                            wrapMode: Text.Wrap
                            textFormat: Text.PlainText
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            Label {
                                Layout.fillWidth: true
                                text: workspace.busy ? "正在处理…" : Qt.platform.os === "osx" ? "⌘ Enter 发送" : "Ctrl + Enter 发送"
                                color: app.muted
                                font.pixelSize: 12
                            }
                            Action {
                                visible: workspace.busy
                                text: "取消等待"
                                onClicked: workspace.cancel()
                            }
                            Action {
                                objectName: "chatSend"
                                enabled: workspace.canSend
                                primary: true
                                text: "发送"
                                onClicked: workspace.send()
                            }
                        }
                    }
                }
            }
        }
    }
}
