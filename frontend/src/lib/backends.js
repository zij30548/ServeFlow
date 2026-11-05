import { pollEvaluation } from './evaluationJobs.js'
import { createDemoAuthority, isDemoOrderId } from './demoOrders.js'
const DEFAULT_BACKENDS = {
  python: {
    id: 'python',
    label: 'Python',
    baseUrl: runtimeConfig().pythonApiUrl || '/api/python',
    port: '8000'
  },
  java: {
    id: 'java',
    label: 'Java',
    baseUrl: runtimeConfig().javaApiUrl || '/api/java',
    port: '8080'
  }
}

export function createInitialSettings() {
  const saved = readSettings()
  return {
    pythonToken: '',
    backend: 'python',
    userId: saved.userId || 'u1001',
    conversationId: saved.conversationId || '',
    endpoints: {
      python: saved.endpoints?.python || DEFAULT_BACKENDS.python.baseUrl,
      java: saved.endpoints?.java || DEFAULT_BACKENDS.java.baseUrl
    }
  }
}

export function saveSettings(settings) {
  localStorage.setItem('echomind.frontend.settings', JSON.stringify({ ...settings, backend: 'python', pythonToken: undefined }))
}

export function backendMeta(type, settings) {
  const meta = DEFAULT_BACKENDS[type] || DEFAULT_BACKENDS.java
  return {
    ...meta,
    baseUrl: normalizeBaseUrl(settings.endpoints[type] || meta.baseUrl)
  }
}

export async function requestHealth(type, settings) {
  return requestFor(type, settings, '/health')
}

export async function requestMonitor(type, settings, options = {}) {
  const deadline = AbortSignal.timeout(20000)
  const signal = options.signal ? AbortSignal.any([options.signal, deadline]) : deadline
  return requestFor(type, settings, '/monitor', { ...options, signal })
}

export async function requestSkills(type, settings) {
  return requestFor(type, settings, '/skills')
}

export async function reloadSkills(type, settings) {
  return requestFor(type, settings, '/skills/reload', { method: 'POST' })
}

export async function requestKnowledgeStats(type, settings) {
  return requestFor(type, settings, '/knowledge/stats')
}

export async function requestDemoSession(settings, signal) {
  return requestFor('python', settings, '/demo/session', { signal: demoSignal(signal) })
}

export async function requestDemoOrder(settings, orderId, signal) {
  if (!isDemoOrderId(orderId)) throw new Error('demo_order_id_invalid')
  return requestFor('python', settings, `/demo/orders/${encodeURIComponent(orderId)}`, { signal: demoSignal(signal) })
}

function demoSignal(signal) {
  const timeout = AbortSignal.timeout(10000)
  return signal ? AbortSignal.any([signal, timeout]) : timeout
}

export function bindDemoAuthority(settings, session) {
  demoAuthority.set(backendMeta('python', settings).baseUrl, session)
}

export function clearDemoAuthority(settings) {
  demoAuthority.clear(backendMeta('python', settings).baseUrl)
}

export async function runEvaluation(type, settings, body = null, options = {}) {
  const taskKey = `echomind.eval:${type}:${backendMeta(type, settings).baseUrl}:${settings.userId}`
  const savedJob = type === 'python' ? sessionStorage.getItem(taskKey) : null
  const initial = savedJob ? {job_id: savedJob, status: 'queued', completed: 0, total: 0} : await requestFor(type, settings, '/eval/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined
  })
  if (!initial.job_id) return initial // Java 后端仍兼容同步报告。
  sessionStorage.setItem(taskKey, initial.job_id)
  try {
    const result = await pollEvaluation(initial, jobId => requestFor(type, settings,
      `/eval/jobs/${encodeURIComponent(jobId)}`), options)
    if (result) sessionStorage.removeItem(taskKey)
    return result
  } catch (error) {
    if (error.terminal || error.status === 404) sessionStorage.removeItem(taskKey)
    throw error
  }
}

export async function requestSearch(type, settings, query, topK = 5) {
  const params = new URLSearchParams({ query, top_k: String(topK) })
  return requestFor(type, settings, `/search?${params}`, { method: 'POST' })
}

export async function requestChat(type, settings, message, options = {}) {
  const meta = backendMeta(type, settings)
  const payload = buildChatPayload(type, settings, message)
  const attemptKey = JSON.stringify([type, meta.baseUrl, settings.userId, settings.conversationId, message])
  if (type === 'python') {
    if (!chatAttempts.has(attemptKey)) {
      if (chatAttempts.size >= 100) throw new Error('chat_attempt_capacity')
      chatAttempts.set(attemptKey, crypto.randomUUID())
    }
    payload.client_request_id = chatAttempts.get(attemptKey)
  }
  try {
    const raw = await requestFor(type, settings, '/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: boundedSignal(options.signal, 55000)
    })
    const response = normalizeChatResponse(type, raw)
    if (response.status !== 'failed') chatAttempts.delete(attemptKey)
    else retainFailedAttempt(response)
    return response
  } catch (error) {
    // A failed turn is an intentional ChatResponse, not an arbitrary error body.
    if (error.status === 503 && error.responseBody?.status === 'failed') {
      const response = normalizeChatResponse(type, error.responseBody)
      retainFailedAttempt(response)
      return response
    }
    const safe = new Error(chatFailureMessage(error))
    safe.status = error.status
    safe.name = error.name
    safe.responseBody = { error_code: demoFailureCode(error) }
    throw safe
  }

  function retainFailedAttempt(response) {
    // Binding the returned conversation must not silently create a new operation
    // if the user repeats the same failed turn after the initial unbound request.
    const boundKey = JSON.stringify([type, meta.baseUrl, settings.userId, response.conversationId, message])
    if (chatAttempts.has(boundKey) || chatAttempts.size < 100) chatAttempts.set(boundKey, payload.client_request_id)
  }
}

export async function requestToolTrace(type, settings, requestId, options = {}) {
  if (!requestId) return null
  const raw = await requestFor(type, settings, `/trace/tool/${encodeURIComponent(requestId)}`, {
    signal: boundedSignal(options.signal, 20000)
  })
  return normalizeToolTraceResponse(raw)
}

function boundedSignal(signal, timeout) {
  const deadline = AbortSignal.timeout(timeout)
  return signal ? AbortSignal.any([signal, deadline]) : deadline
}

const DEMO_FAILURES = Object.freeze({
  demo_model_quota_exhausted: '演示模型调用额度已用完，请联系管理员；等待不会自动恢复额度，不会自动重试。',
  demo_model_quota_unavailable: '演示计数服务暂不可用，已停止新的模型请求；请联系管理员核查，不会自动重试。',
  demo_model_request_rejected: '本次模型请求未通过演示环境校验，已停止处理；不会自动重试。',
  demo_model_provider_unavailable: '模型服务本次未取得有效结果；已发送请求可能仍计费，不会自动重试。'
})
function demoFailureCode(error) {
  const code = error?.responseBody?.error_code || error?.responseBody?.detail
  return typeof code === 'string' && Object.hasOwn(DEMO_FAILURES, code) ? code : null
}

export function chatFailureMessage(error) {
  const code = demoFailureCode(error)
  if (code) return DEMO_FAILURES[code]
  if ([401, 403, 428].includes(error?.status)) return '当前会话已失效，请重新读取演示身份。'
  if (error?.status === 429) return '请求过于频繁，请稍后再试；不会自动重试。'
  if (['AbortError', 'TimeoutError'].includes(error?.name) || error?.status === 504) {
    return '本次等待已结束，处理结果尚未确认；已发送的请求可能仍在处理，不会自动重试。'
  }
  return '本次请求未完成，请核对当前会话或检查服务连接；不会自动重试。'
}

const CHAT_FAILURES = Object.freeze({
  agent_busy: '当前处理通道忙，本次请求未完成；不会自动重试。',
  agent_failed: '本次处理未完成，暂时无法提供有效答复；不会自动重试。',
  agent_unavailable: '当前处理通道不可用，本次请求未完成；不会自动重试。',
  agent_timeout: '本次处理超时，尚未取得有效结果；已发送的请求可能仍在处理，不会自动重试。',
  agent_protocol_error: '本次响应格式无效，未取得有效结果；不会自动重试。',
  tool_outcome_unknown: '相关操作结果尚未确认，请先核对执行结果，不要重复操作。'
})

export async function addKnowledge(type, settings, documents) {
  return requestFor(type, settings, '/knowledge/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ documents })
  })
}

export async function uploadKnowledge(type, settings, file) {
  const form = new FormData()
  form.append('file', file)
  return requestFor(type, settings, '/knowledge/upload', {
    method: 'POST',
    body: form
  })
}

function buildChatPayload(type, settings, message) {
  return {
    message,
    user_id: settings.userId || 'anonymous',
    conv_id: settings.conversationId || undefined
  }
}

function normalizeChatResponse(type, raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw) ||
      !['success', 'partial', 'failed'].includes(raw.status) ||
      typeof raw.retryable !== 'boolean' || typeof raw.response !== 'string' || raw.response.length > 100000 ||
      (raw.status !== 'failed' && !raw.response.trim()) ||
      (raw.error_code != null && (typeof raw.error_code !== 'string' || raw.error_code.length > 100))) {
    throw new Error('chat_response_invalid')
  }
  const conversationId = raw.conversation_id || raw.conversationId || raw.conv_id || ''
  const requestId = raw.request_id || raw.requestId || ''
  if (!/^[A-Za-z0-9_.:-]{1,128}$/.test(conversationId) ||
      (requestId !== '' && !/^[A-Za-z0-9_.:-]{1,128}$/.test(requestId))) throw new Error('chat_response_invalid')
  const errorCode = Object.hasOwn(CHAT_FAILURES, raw.error_code) ? raw.error_code : raw.error_code == null ? null : 'unknown_error'
  return {
    backend: type,
    conversationId,
    requestId,
    status: raw.status,
    retryable: raw.retryable,
    errorCode,
    response: raw.status === 'failed' ? (CHAT_FAILURES[errorCode] || chatFailureMessage()) : raw.response,
    intent: raw.intent || 'other',
    intentGroup: raw.intent_group || raw.intentGroup || 'other',
    agentType: raw.agent_type || raw.agentType || '',
    agentTypes: raw.agent_types || raw.agentTypes || [],
    primaryAgent: raw.primary_agent || raw.primaryAgent || '',
    supportingAgents: raw.supporting_agents || raw.supportingAgents || [],
    routingReason: raw.routing_reason || raw.routingReason || '',
    routingConfidence: Number(raw.routing_confidence ?? raw.routingConfidence ?? 0),
    entities: raw.entities || {},
    intentConfidence: Number(raw.intent_confidence ?? raw.intentConfidence ?? 0),
    intentSourceScores: raw.intent_source_scores || raw.intentSourceScores || {},
    escalated: Boolean(raw.escalated),
    latencyMs: Number(raw.latency_ms ?? raw.latencyMs ?? 0),
    knowledgeUsed: Boolean(raw.knowledge_used ?? raw.knowledgeUsed),
    verified: raw.verified,
    grounded: raw.grounded
  }
}

function normalizeToolTraceResponse(raw) {
  const trace = raw?.trace || {}
  return {
    requestId: raw?.request_id || raw?.requestId || '',
    found: Boolean(raw?.found),
    trace: {
      ...trace,
      toolsUsed: trace.tools_used || trace.toolsUsed || [],
      toolCalls: trace.tool_calls || trace.toolCalls || []
    },
    raw
  }
}

const chatAttempts = new Map()
const demoAuthority = createDemoAuthority()
async function requestFor(type, settings, path, options = {}) {
  const headers = new Headers(options.headers || {})
  if (type === 'python' && settings.pythonToken) headers.set('Authorization', `Bearer ${settings.pythonToken}`)
  const baseUrl = backendMeta(type, settings).baseUrl
  if (type === 'python' && !['GET', 'HEAD'].includes((options.method || 'GET').toUpperCase())) {
    const csrf = demoAuthority.csrfFor(baseUrl, settings.userId)
    if (csrf) headers.set('X-ServeFlow-Demo-CSRF', csrf)
  }
  return requestJson(baseUrl, path, { ...options, credentials: 'same-origin', headers,
    signal: options.signal || AbortSignal.timeout(path === '/chat' ? 55000 : 20000) })
}

async function requestJson(baseUrl, path, options = {}) {
  const url = `${normalizeBaseUrl(baseUrl)}${path}`
  const response = await fetch(url, options)
  const text = await response.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = text
  }
  if (!response.ok) {
    const detail = typeof data === 'string' ? data : JSON.stringify(data)
    const error = new Error(`${response.status} ${response.statusText}: ${detail}`)
    error.status = response.status
    error.responseBody = data
    throw error
  }
  return data
}

function normalizeBaseUrl(value) {
  return String(value || '').replace(/\/+$/, '')
}

function readSettings() {
  try {
    return JSON.parse(localStorage.getItem('echomind.frontend.settings') || '{}')
  } catch {
    return {}
  }
}

function runtimeConfig() {
  if (typeof window === 'undefined') return {}
  return window.__ECHOMIND_CONFIG__ || {}
}
