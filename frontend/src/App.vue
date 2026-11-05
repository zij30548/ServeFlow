<template>
  <div class="app-shell" :class="`view-${activeView}`">
    <aside class="navigation-rail" aria-label="主导航">
      <a class="brand" href="#" @click.prevent="activeView = 'chat'" aria-label="智应客服协同系统 首页">
        <span class="brand-mark"><BrandMark /></span>
        <span class="brand-copy"><span class="brand-name">智应</span><span class="brand-caption">客服协同系统</span></span>
      </a>
      <button class="new-chat-button" @click="activeView = 'chat'; clearConversation()">
        <AppIcon name="plus" />
        <span>新对话</span>
      </button>
      <span class="nav-caption">工作空间</span>
      <nav class="view-nav">
        <button
          v-for="view in views"
          :key="view.id"
          :aria-label="view.label"
          :class="{ active: activeView === view.id }"
          :aria-current="activeView === view.id ? 'page' : undefined"
          @click="activeView = view.id"
        >
          <AppIcon :name="view.icon" />
          <span>{{ view.label }}</span>
          <AppIcon class="nav-chevron" name="chevron" />
        </button>
      </nav>
      <div class="rail-bottom">
        <button class="user-button" aria-label="当前用户与会话配置" @click="openConnection">
          <span class="avatar">{{ userInitial }}</span>
          <span>
            <strong>{{ demoSession?.customer_label || settings.userId || 'anonymous' }}</strong>
            <small>{{ demoSession ? '独立浏览器演示会话' : '个人工作空间' }}</small>
          </span>
          <AppIcon name="sliders" />
        </button>
      </div>
    </aside>
    <main class="workspace">
      <header class="topbar">
        <div class="page-breadcrumb">
          <span>工作空间</span>
          <AppIcon name="chevron" />
          <strong>{{ currentView.label }}</strong>
        </div>
        <a class="mobile-brand" href="#" @click.prevent="activeView = 'chat'" aria-label="智应客服协同系统 首页">
          <span class="brand-mark"><BrandMark /></span>
          <span>智应客服协同系统</span>
        </a>
        <div class="topbar-tools">
          <span class="connection-status" :title="statusText || healthLabel">
            <i :class="healthOk ? 'online' : 'offline'"></i>
            {{ healthOk ? '已连接' : healthLabel === '未检查' ? '连接中' : '未连接' }}
          </span>
          <button class="icon-button settings-button" aria-label="连接配置" title="连接配置" @click="openConnection">
            <AppIcon name="sliders" />
          </button>
        </div>
      </header>
      <div v-if="toast" class="toast" role="status">
        <AppIcon name="chat" />
        {{ toast }}
      </div>
      <section v-show="activeView === 'chat'" class="page-chat">
        <div class="chat-toolbar">
          <span class="conversation-label">
            {{ messages.length ? '当前对话' : '新对话' }}
            <span v-if="messages.length" class="conversation-count">{{ messages.length }} 条消息</span>
          </span>
          <div class="chat-toolbar-actions">
            <button class="icon-button" @click="clearConversation" aria-label="新对话" title="新对话">
              <AppIcon name="compose" />
            </button>
            <button class="trace-toggle" @click="inspectorDialog.showModal()">
              <AppIcon name="panel" />
              <span>运行详情</span>
            </button>
          </div>
        </div>
        <details v-if="demoSession || demoLoading || demoError" class="demo-panel" @toggle="handleDemoToggle">
          <summary>
            <span>演示数据</span>
            <small>{{ demoLoading ? '正在准备' : demoSession ? `${demoSession.customer_label} · 合成记录` : '暂不可用' }}</small>
          </summary>
          <div class="demo-panel-content">
            <p class="demo-note">以下为独立 MySQL 中的合成业务记录。查看记录不调用模型；填入对话后仍需你主动发送。</p>
            <p v-if="demoError" class="demo-error" role="alert">{{ demoError }}</p>
            <template v-if="demoSession">
              <div class="demo-samples" aria-label="可用演示订单">
                <button v-for="sample in demoSession.sample_orders" :key="sample.order_id" type="button" @click="chooseDemoOrder(sample.order_id)" :disabled="demoOrderBusy">
                  <strong>{{ sample.order_id }}</strong><span>{{ sample.scenario }}</span>
                </button>
              </div>
              <form class="demo-query" @submit.prevent="lookupDemoOrder">
                <label>
                  <span>演示订单号</span>
                  <input v-model="demoOrderId" aria-label="演示订单号" maxlength="9" autocomplete="off" placeholder="DEMO-1001" :disabled="demoOrderBusy" />
                </label>
                <button type="submit" :disabled="demoOrderBusy || !demoOrderId.trim()">{{ demoOrderBusy ? '读取中' : '查看记录' }}</button>
                <button v-if="demoOrderBusy" class="quiet-button" type="button" @click="cancelDemoOrder">取消读取</button>
              </form>
              <p v-if="demoOrderError" class="demo-error" role="alert">{{ demoOrderError }}</p>
              <section v-if="demoOrder" class="demo-record" aria-label="演示订单记录" aria-live="polite">
                <template v-if="demoOrder.found">
                  <div class="demo-record-heading"><h2>{{ demoOrder.orderId }}</h2><button class="quiet-button" type="button" @click="fillDemoQuestion">填入对话</button></div>
                  <dl><template v-for="(row, index) in demoOrder.rows" :key="`${row.name}-${index}`"><dt>{{ row.name }}</dt><dd>{{ row.value }}</dd></template></dl>
                  <p class="demo-note">记录更新时间：{{ demoOrder.updatedAt }}</p>
                </template>
                <p v-else>未找到可访问的演示订单，请核对订单号。</p>
                <p class="demo-note">数据快照：{{ demoOrder.snapshot }}；这是预设演示状态，不是实时物流或真实交易。</p>
              </section>
            </template>
          </div>
        </details>
        <div class="conversation-layout" :class="{ 'has-conversation': messages.length }">
          <section class="conversation" :class="{ 'is-empty': !messages.length }" aria-label="客服对话">
            <div class="messages" ref="messageList" role="log" aria-label="对话消息" aria-live="polite" :aria-busy="busy && pendingAction === 'chat'">
              <div v-if="!messages.length" class="welcome">
                <div class="welcome-mark" aria-hidden="true"><BrandMark /></div>
                <h1>今天，需要解决什么？</h1>
                <p>智应客服协同系统 · 面向复杂客服任务的多 Agent 协作系统</p>
              </div>
              <template v-else>
                <article v-for="item in messages" :key="item.id" :class="['message', item.role, { 'message-error': item.error }]">
                  <span v-if="item.role === 'assistant'" class="message-avatar"><BrandMark /></span>
                  <div class="message-content">
                    <div class="message-meta">
                      <strong>{{ item.role === 'user' ? '你' : '智应客服协同系统' }}</strong>
                      <span v-if="item.error" class="error-label">请求失败</span>
                      <span v-else-if="item.status === 'partial'" class="error-label">部分完成</span>
                      <small v-else-if="item.role === 'assistant'">Agent</small>
                    </div>
                    <p>{{ item.content }}</p>
                    <p v-if="item.status === 'partial'" class="muted">本次仅取得部分结果，尚未完整处理你的问题；不会自动重试。</p>
                    <span v-if="item.meta && !item.error" class="agent-tag">
                      <AppIcon name="route" />
                      {{ item.meta }}
                    </span>
                    <details v-if="item.trace?.toolCalls?.length || item.trace?.toolsUsed?.length" class="message-trace">
                      <summary>
                        <AppIcon name="chevron" />
                        <span>工具调用</span>
                        <span class="count-tag">{{ item.trace.toolCalls?.length || 0 }}</span>
                      </summary>
                      <div v-for="(call, index) in item.trace.toolCalls" :key="`${item.id}-${index}`" class="trace-call-item">
                        <div class="trace-call-meta">
                          <strong>{{ call.tool_name || 'unknown_tool' }}</strong>
                          <span :class="call.success ? 'success' : 'danger'">{{ call.success ? '成功' : '失败' }}</span>
                        </div>
                        <pre>{{ formatJson(call.input || {}) }}</pre>
                      </div>
                      <p v-if="!item.trace.toolCalls?.length" class="muted">已调用：{{ item.trace.toolsUsed.join(' · ') }}</p>
                    </details>
                    <button
                      v-if="item.role === 'assistant' && !item.error"
                      class="icon-button copy-button"
                      @click="copyMessage(item.content)"
                      aria-label="复制回复"
                      title="复制回复"
                    >
                      <AppIcon name="copy" />
                    </button>
                  </div>
                </article>
                <div v-if="busy && pendingAction === 'chat'" class="thinking" role="status">
                  <span class="message-avatar"><BrandMark /></span>
                  <span class="thinking-dots">
                    <i></i>
                    <i></i>
                    <i></i>
                  </span>
                  <span>正在处理问题</span>
                </div>
              </template>
            </div>
            <div class="composer-area">
              <form class="composer" @submit.prevent="performAction('chat', sendMessage)">
                <textarea
                  ref="composerInput"
                  v-model="draft"
                  rows="2"
                  aria-label="客户问题"
                  placeholder="输入你的问题…"
                  @keydown.meta.enter.prevent="performAction('chat', sendMessage)"
                  @keydown.ctrl.enter.prevent="performAction('chat', sendMessage)"
                >
                </textarea>
                <div class="composer-bottom">
                  <span class="routing-label">
                    <AppIcon name="route" />
                    自动路由
                  </span>
                  <div class="send-controls">
                    <span class="keyboard-hint">⌘ / Ctrl + Enter</span>
                    <button
                      class="send-button"
                      type="submit"
                      :disabled="busy || demoLoading || Boolean(demoError) || !draft.trim()"
                      :aria-label="busy && pendingAction === 'chat' ? '处理中' : '发送'"
                      :title="busy && pendingAction === 'chat' ? '处理中' : '发送'"
                    >
                      <AppIcon :name="busy && pendingAction === 'chat' ? 'refresh' : 'arrow'" :class="{ spinning: busy && pendingAction === 'chat' }" />
                    </button>
                  </div>
                </div>
              </form>
              <div v-if="!messages.length" class="prompt-grid">
                <button v-for="prompt in prompts" :key="prompt.label" @click="usePrompt(prompt.text)">
                  <AppIcon :name="prompt.icon" />
                  <span>
                    <strong>{{ prompt.label }}</strong>
                    <small>{{ prompt.description }}</small>
                  </span>
                  <AppIcon class="prompt-arrow" name="northeast" />
                </button>
              </div>
              <p class="composer-note">AI 生成的内容可能不准确，请结合业务规则核实。</p>
            </div>
          </section>
          <aside v-if="messages.length" class="inspector-desktop" aria-label="请求追踪与运行状态">
            <div class="inspector-heading">
              <AppIcon name="panel" />
              <h2>运行详情</h2>
            </div>
            <RequestInspector :last-response="lastResponse" :last-trace="lastTrace" :busy="busy && pendingAction === 'chat'" :monitor-data="monitorData" @refresh="loadMonitor" />
          </aside>
        </div>
      </section>
      <section v-show="activeView === 'monitor'" class="page page-monitor" aria-label="运行监控">
        <MonitorDashboard :snapshot="monitorReadAt ? monitorData : null" :loading="monitorLoading" :error="monitorError" :read-at="monitorReadAt" @refresh="loadMonitor" />
      </section>
    </main>
    <dialog ref="inspectorDialog" class="inspector-dialog" @click="closeOnBackdrop" aria-labelledby="inspector-title">
      <div class="drawer-heading">
        <div>
          <AppIcon name="panel" />
          <h2 id="inspector-title">运行详情</h2>
        </div>
        <button class="icon-button" aria-label="关闭运行详情" @click="inspectorDialog.close()">
          <AppIcon name="close" />
        </button>
      </div>
      <RequestInspector :last-response="lastResponse" :last-trace="lastTrace" :busy="busy && pendingAction === 'chat'" :monitor-data="monitorData" @refresh="loadMonitor" />
    </dialog>
    <dialog ref="connectionDialog" class="connection-dialog" @click="closeOnBackdrop" aria-labelledby="connection-title">
      <div class="dialog-heading">
        <div>
          <span class="kicker">CONNECTION</span>
          <h2 id="connection-title">连接配置</h2>
        </div>
        <button class="icon-button" aria-label="关闭连接配置" @click="connectionDialog.close()">
          <AppIcon name="close" />
        </button>
      </div>
      <p class="dialog-description">配置客服服务与当前会话。</p>
      <form @submit.prevent="saveConnection">
        <label>
          <span>服务地址</span>
          <input v-model="connectionDraft.endpoint" required placeholder="/api/python" />
          <small>使用代理路径或完整的 HTTP / HTTPS 地址。</small>
        </label>
        <label>
          <span>用户 ID</span>
          <input v-model="connectionDraft.userId" placeholder="u1001" :disabled="Boolean(demoSession)" />
          <small v-if="demoSession">演示身份由当前浏览器会话确定，不能通过修改用户 ID 切换客户。</small>
        </label>
        <label>
          <span>会话 ID</span>
          <input v-model="connectionDraft.conversationId" placeholder="留空以自动生成" />
        </label>
        <label>
          <span>API 访问 Token（启用鉴权时填写，仅本页保存）</span>
          <input v-model="connectionDraft.pythonToken" type="password" autocomplete="off" placeholder="不是模型 API Key" />
        </label>
        <div class="dialog-health">
          <i :class="healthOk ? 'online' : 'offline'"></i>
          <span>{{ healthOk ? '服务已连接' : healthLabel }}</span>
          <button type="button" class="text-button" @click="checkHealth">
            检查当前连接
            <AppIcon name="refresh" />
          </button>
        </div>
        <details v-if="statusText" class="connection-detail">
          <summary>查看连接详情</summary>
          <pre>{{ statusText }}</pre>
        </details>
        <div class="dialog-actions">
          <button type="button" class="quiet-button" @click="connectionDialog.close()">取消</button>
          <button type="submit">
            保存配置
            <AppIcon name="check" />
          </button>
        </div>
      </form>
    </dialog>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import {
  backendMeta,
  bindDemoAuthority,
  chatFailureMessage,
  clearDemoAuthority,
  createInitialSettings,
  requestChat,
  requestDemoOrder,
  requestDemoSession,
  requestHealth,
  requestMonitor,
  requestToolTrace,
  saveSettings
} from './lib/backends'

import AppIcon from './components/AppIcon.vue'
import BrandMark from './components/BrandMark.vue'
import RequestInspector from './components/RequestInspector.vue'
import MonitorDashboard from './components/MonitorDashboard.vue'

import { createRequestState } from './lib/requestState'
import { isDemoOrderId, normalizeDemoSession, projectDemoOrder } from './lib/demoOrders'

const settings = reactive(createInitialSettings())
const requestState = createRequestState(settings)
const activeView = ref('chat')
const messages = ref([])
const draft = ref('')
const busy = ref(false)
const healthOk = ref(false)
const healthLabel = ref('未检查')
const statusText = ref('')
const messageList = ref(null)
const composerInput = ref(null)
const connectionDialog = ref(null)
const inspectorDialog = ref(null)
const connectionDraft = reactive({ endpoint: '', userId: '', conversationId: '', pythonToken: '' })
const pendingAction = ref('')
const demoSession = ref(null)
const demoLoading = ref(false)
const demoError = ref('')
const demoOrderId = ref('')
const demoOrder = ref(null)
const demoOrderBusy = ref(false)
const demoOrderError = ref('')
const views = [
  { id: 'chat', label: '对话', icon: 'chat' },
  { id: 'monitor', label: '监控', icon: 'chart' }
]
const prompts = computed(() => buildConversationPrompts(demoSession.value))
const currentView = computed(() => views.find(view => view.id === activeView.value))
const monitorData = ref({ agent_stats: {}, tool_stats: {}, active_alerts: [], suggestions: [] })
const monitorLoading = ref(false)
const monitorError = ref('')
const monitorReadAt = ref(null)
let monitorController = null
let monitorSequence = 0
const lastResponse = ref(null)
const lastTrace = ref(null)
const toast = ref('')
let toastTimer
let messageSequence = 0
let demoSequence = 0
let demoOrderSequence = 0
let demoSessionController = null
let demoOrderController = null
let demoExpiryTimer
let boundDemoSettings = null
let chatController = null
let traceController = null

const currentBackend = computed(() => backendMeta('python', settings))
const userInitial = computed(() => (settings.userId || 'U').slice(0, 1).toUpperCase())

watch(() => settings.conversationId, persist)
watch(activeView, (view, previous) => {
  if (view === 'monitor') loadMonitor()
  else if (previous === 'monitor') cancelMonitor()
})
watch(demoOrderId, () => {
  cancelDemoOrder()
  demoOrder.value = null
  demoOrderError.value = ''
}, { flush: 'sync' })
onMounted(refreshConsole)
onBeforeUnmount(() => {
  requestState.invalidate()
  cancelChat()
  resetDemoState()
  cancelMonitor()
  clearTimeout(toastTimer)
})

function persist() { saveSettings(settings) }

function performAction(name, action, ...args) {
  if (busy.value || demoLoading.value || demoError.value) return
  pendingAction.value = name
  return action(...args)
}

function openConnection() {
  connectionDraft.pythonToken = settings.pythonToken
  connectionDraft.endpoint = currentBackend.value.baseUrl
  connectionDraft.userId = settings.userId
  connectionDraft.conversationId = settings.conversationId
  connectionDialog.value.showModal()
}

function closeOnBackdrop(event) {
  const dialog = event.currentTarget
  if (event.target !== dialog) return
  const rect = dialog.getBoundingClientRect()
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close()
}

function saveConnection() {
  const endpoint = connectionDraft.endpoint.trim().replace(/\/+$/, '')
  if (!endpoint || !(endpoint.startsWith('/') || /^https?:\/\//i.test(endpoint))) {
    showToast('请输入代理路径或 HTTP / HTTPS 服务地址')
    return
  }
  const changed = connectionDraft.pythonToken !== settings.pythonToken || endpoint !== currentBackend.value.baseUrl || connectionDraft.userId.trim() !== settings.userId || connectionDraft.conversationId.trim() !== settings.conversationId
  if (changed) {
    requestState.invalidate()
    cancelChat()
    busy.value = false
    messages.value = []
    lastResponse.value = null
    lastTrace.value = null
    resetMonitor()
    healthOk.value = false
    healthLabel.value = '未检查'
    statusText.value = ''
  }
  const identityChanged = connectionDraft.userId.trim() !== settings.userId || endpoint !== currentBackend.value.baseUrl
  settings.backend = 'python'
  settings.endpoints.python = endpoint
  settings.pythonToken = connectionDraft.pythonToken.trim()
  settings.userId = connectionDraft.userId.trim()
  settings.conversationId = identityChanged && connectionDraft.conversationId.trim() === settings.conversationId ? '' : connectionDraft.conversationId.trim()
  persist()
  connectionDialog.value.close()
  refreshConsole()
  showToast('连接配置已保存')
}

async function copyMessage(content) {
  try {
    await navigator.clipboard.writeText(content)
    showToast('回复已复制')
  } catch {
    showToast('复制失败，请手动选择回复文字')
  }
}

async function refreshConsole() {
  await loadDemoSession()
  await checkHealth()
  if (activeView.value === 'monitor') await loadMonitor()
}

function cancelDemoOrder() {
  demoOrderSequence += 1
  demoOrderController?.abort()
  demoOrderController = null
  demoOrderBusy.value = false
}

function resetDemoState() {
  demoSequence += 1
  demoSessionController?.abort()
  demoSessionController = null
  cancelDemoOrder()
  clearTimeout(demoExpiryTimer)
  if (boundDemoSettings) clearDemoAuthority(boundDemoSettings)
  boundDemoSettings = null
  demoSession.value = null
  demoLoading.value = false
  demoError.value = ''
  demoOrder.value = null
  demoOrderError.value = ''
  demoOrderId.value = ''
}

function expireDemoSession(message = '演示会话已过期，请刷新页面重新读取身份。') {
  resetDemoState()
  requestState.invalidate()
  cancelChat()
  messages.value = []
  draft.value = ''
  lastResponse.value = null
  lastTrace.value = null
  resetMonitor()
  settings.conversationId = ''
  busy.value = false
  pendingAction.value = ''
  demoError.value = message
}

async function loadDemoSession() {
  resetDemoState()
  if (settings.backend !== 'python') return
  const sequence = demoSequence
  const snapshot = { ...settings, endpoints: { ...settings.endpoints } }
  const endpoint = backendMeta('python', snapshot).baseUrl
  const current = () => sequence === demoSequence && settings.backend === 'python' && endpoint === backendMeta('python', settings).baseUrl
  demoSessionController = new AbortController()
  demoLoading.value = true
  try {
    const raw = await requestDemoSession(snapshot, demoSessionController.signal)
    if (!current()) return
    const session = normalizeDemoSession(raw)
    if (settings.userId !== session.workspace_user_id) {
      requestState.invalidate()
      cancelChat()
      messages.value = []
      draft.value = ''
      lastResponse.value = null
      lastTrace.value = null
      resetMonitor()
      busy.value = false
      pendingAction.value = ''
      settings.conversationId = ''
    }
    settings.userId = session.workspace_user_id
    demoSession.value = session
    boundDemoSettings = { ...snapshot, userId: session.workspace_user_id }
    bindDemoAuthority(boundDemoSettings, session)
    persist()
    demoExpiryTimer = setTimeout(() => {
      if (!current()) return
      expireDemoSession()
    }, Math.min(2147483647, Date.parse(session.expires_at) - Date.now()))
  } catch (error) {
    if (!current()) return
    if (error.status !== 404) demoError.value = '演示数据暂不可用，请检查当前连接后刷新页面。'
  } finally {
    if (current()) {
      demoLoading.value = false
      demoSessionController = null
    }
  }
}

function chooseDemoOrder(orderId) {
  if (!demoSession.value || demoOrderBusy.value || !isDemoOrderId(orderId)) return
  demoOrderId.value = orderId
  demoOrder.value = null
  demoOrderError.value = ''
}

function handleDemoToggle(event) {
  if (!event.currentTarget.open) cancelDemoOrder()
}

async function lookupDemoOrder() {
  if (!demoSession.value || demoOrderBusy.value || demoLoading.value || settings.backend !== 'python') return
  const orderId = demoOrderId.value.trim()
  demoOrder.value = null
  demoOrderError.value = ''
  if (!isDemoOrderId(orderId)) {
    demoOrderError.value = '请输入 DEMO- 加四位数字的演示订单号。'
    return
  }
  const request = requestState.begin('demoOrder')
  const identity = demoSession.value.workspace_user_id
  const sequence = ++demoOrderSequence
  const controller = new AbortController()
  demoOrderController = controller
  const current = () => request.isCurrent() && sequence === demoOrderSequence && demoSession.value?.workspace_user_id === identity
  demoOrderBusy.value = true
  try {
    const raw = await requestDemoOrder(request.settings, orderId, controller.signal)
    if (!current()) return
    demoOrder.value = projectDemoOrder(raw, orderId)
  } catch (error) {
    if (!current()) return
    if ([401, 403, 428].includes(error.status)) expireDemoSession()
    else demoOrderError.value = '演示订单读取失败，本次未自动重试。'
  } finally {
    if (current()) {
      demoOrderBusy.value = false
      demoOrderController = null
    }
  }
}

function fillDemoQuestion() {
  if (!demoSession.value || !demoOrder.value?.found || demoOrderBusy.value) return
  usePrompt(`请查询我的演示订单 ${demoOrder.value.orderId}，说明当前进度及还需要核实的事项。`)
}

async function checkHealth() {
  const request = requestState.begin('health')

  try {
    const data = await requestHealth(request.backend, request.settings)
    if (!request.isCurrent()) return
    healthOk.value = data.status === 'ok'
    healthLabel.value = data.status || 'ok'
    statusText.value = JSON.stringify(data, null, 2)
  } catch (error) {
    if (!request.isCurrent()) return
    // 健康检查只更新状态，保留用户选择的后端。
    healthOk.value = false
    healthLabel.value = '不可用'
    statusText.value = error.message
  }
}

function cancelMonitor() {
  monitorSequence += 1
  monitorController?.abort()
  monitorController = null
  monitorLoading.value = false
}

function resetMonitor() {
  cancelMonitor()
  monitorData.value = { agent_stats: {}, tool_stats: {}, active_alerts: [], suggestions: [] }
  monitorReadAt.value = null
  monitorError.value = ''
}

async function loadMonitor() {
  if (monitorLoading.value) return
  const request = requestState.begin('monitor')
  const sequence = ++monitorSequence
  const controller = new AbortController()
  const identity = settings.userId
  monitorController = controller
  const current = () => request.isCurrent() && sequence === monitorSequence && identity === settings.userId
  monitorLoading.value = true
  monitorError.value = ''
  try {
    const data = await requestMonitor('python', request.settings, { signal: controller.signal })
    if (!current()) return
    monitorData.value = { ...data, loaded: true }
    monitorReadAt.value = new Date().toISOString()
  } catch {
    if (current()) monitorError.value = '监控读取失败，请手动刷新。本次未自动重试。'
  } finally {
    if (current()) {
      monitorLoading.value = false
      monitorController = null
    }
  }
}

async function sendMessage() {
  const content = draft.value.trim()
  if (!content || busy.value || demoLoading.value || demoError.value) return
  messages.value.push({ id: createMessageId(), role: 'user', content })
  draft.value = ''
  lastResponse.value = null
  lastTrace.value = null
  nextTick(() => messageList.value?.scrollTo({ top: messageList.value.scrollHeight, behavior: 'smooth' }))
  const request = requestState.begin('action')
  cancelChat()
  const controller = new AbortController()
  chatController = controller
  const current = () => request.isCurrent() && settings.userId === request.settings.userId
  busy.value = true
  try {
    const response = await requestChat(request.backend, request.settings, content, { signal: controller.signal })
    if (!current()) return
    if (response.conversationId && !settings.conversationId) {
      settings.conversationId = response.conversationId
      persist()
    }
    lastResponse.value = response
    const meta = [response.intent, response.primaryAgent || response.agentType, response.knowledgeUsed ? 'RAG' : '', response.escalated ? '转人工' : ''].filter(Boolean).join(' · ')
    const message = { id: createMessageId(), role: 'assistant', content: response.response, meta,
      status: response.status, error: response.status === 'failed', trace: null }
    messages.value.push(message)
    // Optional tracing must never delay the reply or keep the composer busy.
    void loadResponseTrace(response.requestId, request, message.id)
  } catch (error) {
    if (!current()) return
    if (demoSession.value && [401, 403, 428].includes(error.status)) {
      expireDemoSession()
      return
    }
    messages.value.push({ id: createMessageId(), role: 'assistant', content: chatFailureMessage(error), meta: '请求失败', error: true })
  } finally {
    if (!current()) return
    if (chatController === controller) chatController = null
    busy.value = false
    await nextTick()
    messageList.value?.scrollTo({ top: messageList.value.scrollHeight, behavior: 'smooth' })
  }
}

function buildConversationPrompts(session) {
  const samples = session?.sample_orders || []
  const first = samples[0]
  const refund = samples.find(sample => sample.scenario.includes('退款'))
  return [
    first ? { label: '订单查询', text: `请查询演示订单 ${first.order_id} 的状态。`, description: '查看当前演示身份的订单记录', icon: 'document' }
      : { label: '订单咨询', text: '查询订单状态需要提供哪些信息？', description: '了解查单需要的信息', icon: 'document' },
    refund ? { label: '退款查询', text: `请查询演示订单 ${refund.order_id} 的退款状态。`, description: '查询演示退款，不发起退款操作', icon: 'route' }
      : { label: '退款咨询', text: '申请退款需要满足什么条件？', description: '了解售后规则，不发起退款操作', icon: 'route' },
    { label: '技术排查', text: '登录时提示错误，应该怎么排查？', description: '排查登录与使用问题', icon: 'sliders' }
  ]
}

function usePrompt(prompt) { draft.value = prompt; nextTick(() => composerInput.value?.focus()) }

function clearConversation() {
  cancelDemoOrder()
  cancelMonitor()
  requestState.invalidate()
  cancelChat()
  busy.value = false
  messages.value = []
  lastResponse.value = null
  lastTrace.value = null
  settings.conversationId = ''
  persist()
}

function cancelChat() {
  chatController?.abort()
  chatController = null
  traceController?.abort()
  traceController = null
}

async function loadResponseTrace(requestId, request, messageId) {
  if (!requestId) return
  const controller = new AbortController()
  traceController = controller
  try {
    const trace = await requestToolTrace(request.backend, request.settings, requestId, { signal: controller.signal })
    if (!request.isCurrent() || settings.userId !== request.settings.userId || controller.signal.aborted) return
    const message = messages.value.find(item => item.id === messageId)
    if (!message) return
    lastTrace.value = trace
    message.trace = trace?.trace || null
  } catch {
    // Trace availability is independent from the already rendered chat result.
  } finally {
    if (traceController === controller) traceController = null
  }
}

function formatJson(value) {
  try {
    return JSON.stringify(value ?? {}, null, 2)
  } catch {
    return String(value ?? '')
  }
}

function createMessageId() {
  messageSequence += 1
  return `message-${Date.now()}-${messageSequence}`
}

function showToast(message) {
  toast.value = message
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => { toast.value = '' }, 2600)
}
</script>
