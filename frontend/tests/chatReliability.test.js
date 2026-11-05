import { test } from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import vm from 'node:vm'
import { chatFailureMessage, requestChat, requestToolTrace } from '../src/lib/backends.js'
import { createRequestState } from '../src/lib/requestState.js'

const settings = () => ({ backend: 'python', endpoints: { python: '/controlled-python' }, userId: 'qa-workspace', conversationId: '' })
const reply = overrides => ({ conv_id: 'qa-conversation', request_id: 'qa-request', response: '受控回答',
  status: 'success', retryable: false, error_code: null, ...overrides })
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b }); return { promise, resolve, reject } }

test('demo hard quota and counter failures retain safe codes without suggesting a timed reset or exposing server text', async () => {
  const previous = globalThis.fetch
  try {
    for (const [code, status, expected] of [
      ['demo_model_quota_exhausted', 429, /额度.*管理员/],
      ['demo_model_quota_unavailable', 503, /计数.*管理员/],
      ['demo_model_provider_unavailable', 503, /已发送.*计费/]
    ]) {
      let calls = 0
      globalThis.fetch = async () => { calls++; return new Response(JSON.stringify({ error_code: code, detail: 'PRIVATE SERVER TEXT' }), { status }) }
      await assert.rejects(() => requestChat('python', settings(), `controlled ${code}`), error => {
        assert.match(chatFailureMessage(error), expected)
        assert.doesNotMatch(error.message, /PRIVATE|稍后再试/)
        assert.equal(error.responseBody.error_code, code)
        return true
      })
      assert.equal(calls, 1)
    }
    assert.match(chatFailureMessage({ status: 429 }), /稍后/)
    assert.doesNotMatch(chatFailureMessage({ status: 503, responseBody: { error_code: 'PRIVATE UNKNOWN' } }), /PRIVATE/)
  } finally { globalThis.fetch = previous }
})

test('structured failed chat keeps identity and status, replaces arbitrary prose and error codes with fixed messages', async () => {
  const previous = globalThis.fetch
  globalThis.fetch = async () => new Response(JSON.stringify(reply({ status: 'failed', retryable: true,
    error_code: 'agent_failed', response: 'PRIVATE RAW EXCEPTION BODY' })), { status: 503 })
  try {
    const result = await requestChat('python', settings(), 'controlled failed turn')
    assert.equal(result.status, 'failed')
    assert.equal(result.retryable, true)
    assert.equal(result.errorCode, 'agent_failed')
    assert.equal(result.conversationId, 'qa-conversation')
    assert.equal(result.requestId, 'qa-request')
    assert.match(result.response, /本次处理未完成/)
    assert.ok(!result.response.includes('PRIVATE'))
    globalThis.fetch = async () => new Response(JSON.stringify(reply({ status: 'failed',
      error_code: 'PRIVATE_UNKNOWN_ERROR', response: 'PRIVATE RAW EXCEPTION BODY' })), { status: 503 })
    const unknown = await requestChat('python', settings(), 'controlled unknown failure')
    assert.equal(unknown.errorCode, 'unknown_error')
    assert.equal(unknown.response, chatFailureMessage())
  } finally { globalThis.fetch = previous }
})

test('partial is retained; invalid structured response and arbitrary HTTP error cannot become successful answers', async () => {
  const previous = globalThis.fetch
  try {
    globalThis.fetch = async () => new Response(JSON.stringify(reply({ status: 'partial' })))
    const partial = await requestChat('python', settings(), 'controlled partial turn')
    assert.equal(partial.status, 'partial')
    assert.equal(partial.response, '受控回答')
    for (const body of [reply({ status: 'invented' }), reply({ retryable: 'false' }), reply({ conv_id: '../private' }),
      reply({ response: '' }), reply({ error_code: { private: 'error' } })]) {
      globalThis.fetch = async () => new Response(JSON.stringify(body))
      await assert.rejects(() => requestChat('python', settings(), JSON.stringify(body)), error => !error.message.includes('invented') && !error.message.includes('../private'))
    }
    globalThis.fetch = async () => new Response('PRIVATE RAW SERVER BODY', { status: 500 })
    await assert.rejects(() => requestChat('python', settings(), 'controlled raw failure'), error => error.message === chatFailureMessage())
  } finally { globalThis.fetch = previous }
})

test('timeout and invalid-protocol failures have explicit safe messages and never expose backend prose', async () => {
  const previous = globalThis.fetch
  try {
    for (const [errorCode, expected] of [
      ['agent_timeout', /本次处理超时.*已发送的请求可能仍在处理.*不会自动重试/],
      ['agent_protocol_error', /本次响应格式无效.*未取得有效结果.*不会自动重试/]
    ]) {
      globalThis.fetch = async () => new Response(JSON.stringify(reply({ status: 'failed', error_code: errorCode,
        response: 'PRIVATE PROTOCOL OR UPSTREAM ERROR' })), { status: 503 })
      const result = await requestChat('python', settings(), `controlled ${errorCode}`)
      assert.equal(result.status, 'failed')
      assert.equal(result.errorCode, errorCode)
      assert.match(result.response, expected)
      assert.ok(!result.response.includes('PRIVATE'))
    }
  } finally { globalThis.fetch = previous }
})

test('a failed turn keeps its client operation after conversation binding; transport unknowns do not get new IDs', async () => {
  const previous = globalThis.fetch, sent = [], configured = settings()
  globalThis.fetch = async (_, options) => {
    sent.push(JSON.parse(options.body))
    return new Response(JSON.stringify(reply({ status: 'failed', error_code: 'tool_outcome_unknown' })), { status: 503 })
  }
  try {
    const result = await requestChat('python', configured, 'controlled unknown tool turn')
    configured.conversationId = result.conversationId
    await requestChat('python', configured, 'controlled unknown tool turn')
    assert.equal(sent[0].client_request_id, sent[1].client_request_id)
    globalThis.fetch = async (_, options) => { sent.push(JSON.parse(options.body)); throw new Error('private transport detail') }
    await assert.rejects(() => requestChat('python', configured, 'controlled unknown transport'))
    await assert.rejects(() => requestChat('python', configured, 'controlled unknown transport'))
    assert.equal(sent[2].client_request_id, sent[3].client_request_id)
    assert.equal(sent.length, 4) // only explicit calls: no retry performed by the helper
  } finally { globalThis.fetch = previous }
})

for (const kind of ['chat', 'trace']) {
  test(`${kind} retains its deadline as well as caller cancellation without retry`, async () => {
    const previousFetch = globalThis.fetch, previousTimeout = AbortSignal.timeout
    const deadline = new AbortController(), caller = new AbortController()
    let signal, calls = 0
    AbortSignal.timeout = milliseconds => {
      assert.equal(milliseconds, kind === 'chat' ? 55000 : 20000)
      return deadline.signal
    }
    globalThis.fetch = async (_, options) => {
      calls += 1; signal = options.signal
      return new Response(JSON.stringify(kind === 'chat' ? reply() : { request_id: 'qa-request', found: false, trace: {} }))
    }
    try {
      if (kind === 'chat') await requestChat('python', settings(), 'controlled cancel helper', { signal: caller.signal })
      else await requestToolTrace('python', settings(), 'qa-request', { signal: caller.signal })
      assert.notEqual(signal, caller.signal)
      deadline.abort()
      assert.equal(signal.aborted, true)
      assert.equal(caller.signal.aborted, false)
      assert.equal(calls, 1)
    } finally { globalThis.fetch = previousFetch; AbortSignal.timeout = previousTimeout }
  })
}

const source = fs.readFileSync(new URL('../src/App.vue', import.meta.url), 'utf8')
function loadFunction(name, context) {
  const match = new RegExp(`^(?:async )?function ${name}\\(`, 'm').exec(source)
  assert.ok(match, name)
  const start = match.index, next = source.slice(start + 1).search(/\n(?:async )?function /)
  const end = next < 0 ? source.indexOf('</script>', start) : start + 1 + next
  vm.runInContext(source.slice(start, end), context)
}

function chatContext(overrides = {}) {
  const configured = settings(), ref = value => ({ value })
  let id = 0
  const context = vm.createContext({ settings: configured, requestState: createRequestState(configured), AbortController,
    draft: ref('受控客户问题'), busy: ref(false), demoLoading: ref(false), demoError: ref(''),
    messages: ref([]), lastResponse: ref(null), lastTrace: ref(null), messageList: ref(null), demoSession: ref(null),
    pendingAction: ref('chat'), chatController: null, traceController: null,
    createMessageId: () => `qa-message-${++id}`, nextTick: async callback => callback?.(), persist: () => {},
    cancelDemoOrder: () => {}, cancelMonitor: () => {}, chatFailureMessage,
    expireDemoSession: () => assert.fail('unexpected identity expiry'),
    requestChat: async () => ({ conversationId: 'qa-conversation', requestId: 'qa-request', status: 'success', response: '受控回答' }),
    requestToolTrace: async () => null, ...overrides })
  for (const name of ['cancelChat', 'sendMessage', 'clearConversation', 'loadResponseTrace']) loadFunction(name, context)
  return context
}

test('chat success is visible and composer released before optional trace completes', async () => {
  const trace = deferred(), context = chatContext({ requestToolTrace: () => trace.promise })
  await context.sendMessage()
  assert.equal(context.messages.value[1].content, '受控回答')
  assert.equal(context.busy.value, false)
  assert.equal(context.lastTrace.value, null)
  trace.resolve({ trace: { toolsUsed: ['controlled-tool'] } })
  await new Promise(resolve => setImmediate(resolve))
  assert.equal(context.messages.value[1].trace.toolsUsed[0], 'controlled-tool')
})

test('trace failure never hides a reply or creates an assistant error', async () => {
  const context = chatContext({ requestToolTrace: async () => { throw new Error('private trace failure') } })
  await context.sendMessage()
  assert.equal(context.messages.value.length, 2)
  assert.equal(context.messages.value[1].error, false)
  assert.equal(context.busy.value, false)
})

for (const fail of [false, true]) {
  test(`clear/new conversation aborts old chat; late ${fail ? 'error' : 'success'} and finally cannot corrupt the new request`, async () => {
    const pending = deferred()
    let signal, calls = 0
    const context = chatContext({ requestChat: (_, __, ___, options) => { calls += 1; signal = options.signal; return pending.promise } })
    const first = context.sendMessage()
    context.draft.value = 'duplicate while busy'
    await context.sendMessage()
    assert.equal(calls, 1)
    context.clearConversation()
    assert.equal(signal.aborted, true)
    const marker = { controller: 'new' }
    context.chatController = marker
    context.busy.value = true
    if (fail) pending.reject(new Error('old private failure'))
    else pending.resolve({ conversationId: 'old-conversation', requestId: 'old-request', response: 'old result' })
    await first
    assert.equal(context.messages.value.length, 0)
    assert.equal(context.settings.conversationId, '')
    assert.equal(context.busy.value, true)
    assert.equal(context.chatController, marker)
  })
}

for (const fail of [false, true]) {
  test(`old trace ${fail ? 'error' : 'success'} cannot restore state after conversation A→B→A`, async () => {
    const pending = deferred()
    let signal
    const context = chatContext({ requestToolTrace: (_, __, ___, options) => { signal = options.signal; return pending.promise } })
    await context.sendMessage()
    context.clearConversation()
    assert.equal(signal.aborted, true)
    context.settings.conversationId = 'qa-conversation'
    const fresh = { trace: { toolsUsed: ['new'] } }, controller = { fresh: true }
    context.lastTrace.value = fresh
    context.traceController = controller
    if (fail) pending.reject(new Error('old private error')); else pending.resolve({ trace: { toolsUsed: ['old'] } })
    await new Promise(resolve => setImmediate(resolve))
    assert.equal(context.messages.value.length, 0)
    assert.equal(context.lastTrace.value, fresh)
    assert.equal(context.traceController, controller)
  })
}

test('identity change rejects late reply even before explicit request generation invalidation', async () => {
  const pending = deferred(), context = chatContext({ requestChat: () => pending.promise })
  const task = context.sendMessage()
  context.settings.userId = 'different-workspace'
  context.busy.value = 'new workspace'
  pending.resolve({ conversationId: 'old', response: 'old reply' })
  await task
  assert.equal(context.messages.value.length, 1)
  assert.equal(context.settings.conversationId, '')
  assert.equal(context.busy.value, 'new workspace')
})

test('inspector distinguishes successful, partial and failed turns', () => {
  const inspector = fs.readFileSync(new URL('../src/components/RequestInspector.vue', import.meta.url), 'utf8')
  const stateLine = inspector.split('\n').find(line => line.startsWith('const responseState ='))
  for (const [status, expected] of [['success', '已完成'], ['partial', '部分完成'], ['failed', '未完成']]) {
    const context = vm.createContext({ props: { lastResponse: { status } }, computed: read => read() })
    vm.runInContext(`${stateLine};globalThis.result=responseState`, context)
    assert.equal(context.result, expected)
  }
})

test('suggested questions use only current demo order IDs, do not invent a real order or send automatically', async () => {
  const context = vm.createContext({ draft: { value: '' }, nextTick: async callback => callback?.(),
    composerInput: { value: { focus() {} } }, requestChat: () => assert.fail('a suggestion must not send') })
  for (const name of ['buildConversationPrompts', 'usePrompt']) loadFunction(name, context)
  const normal = context.buildConversationPrompts(null)
  assert.ok(normal.every(prompt => !/DEMO-|#\d|订单号是/.test(prompt.text)))
  const demo = context.buildConversationPrompts({ sample_orders: [
    { order_id: 'DEMO-2001', scenario: '拆单配送' },
    { order_id: 'DEMO-2003', scenario: '退款待审核' }
  ] })
  assert.match(demo[0].text, /DEMO-2001/)
  assert.match(demo[1].text, /DEMO-2003/)
  assert.ok(demo.every(prompt => !/DEMO-1001|DEMO-1004|#12345/.test(prompt.text)))
  context.usePrompt(demo[1].text)
  assert.equal(context.draft.value, demo[1].text)
})
