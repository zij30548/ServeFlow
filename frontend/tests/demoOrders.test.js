import { test } from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import vm from 'node:vm'
import {
  createDemoAuthority, formatDemoMoney, isDemoOrderId, normalizeDemoSession, projectDemoOrder
} from '../src/lib/demoOrders.js'
import {
  bindDemoAuthority, clearDemoAuthority, requestDemoOrder, requestDemoSession, requestSearch, saveSettings
} from '../src/lib/backends.js'

const workspace = 'sf_demo_0123456789abcdefghijklmnop'
const snapshot = '2026-10-25T00:00:00Z'
const session = () => ({
  enabled: true, is_demo: true, workspace_user_id: workspace, customer_label: '演示客户A',
  csrf_token: 'private-demo-csrf', expires_at: '2099-10-25T08:00:00Z',
  sample_orders: [{ order_id: 'DEMO-1001', scenario: '设备运输中，配件待发货' }]
})
const record = () => ({
  source: 'serveflow-mysql-demo-v1', is_demo: true, status: 'found', as_of: snapshot,
  order: {
    order_id: 'DEMO-1001', customer_id: 'demo-customer-a', currency: 'CNY', order_type: 'bundle',
    total_amount_minor: 129900, payment_status: 'paid', record_updated_at: snapshot,
    items: [{ order_id: 'DEMO-1001', item_id: 'item-device', name: '演示设备', item_type: 'device', quantity: 1, unit_amount_minor: 129900 }],
    payments: [{ order_id: 'DEMO-1001', payment_id: 'payment-1', status: 'paid', channel: 'demo-wallet', amount_minor: 129900 }],
    shipments: [{ order_id: 'DEMO-1001', shipment_id: 'shipment-1', item_id: 'item-device', status: 'in_transit', tracking_id: 'DEMO-TRACK-1', estimated_delivery: null }],
    entitlements: [], refunds: [{ order_id: 'DEMO-1001', refund_id: 'refund-1', status: 'pending_review', amount_minor: null }]
  }
})

test('demo session validates scope and returns detached samples without persisting CSRF', () => {
  const raw = session()
  const normalized = normalizeDemoSession(raw, 0)
  normalized.sample_orders[0].scenario = 'changed'
  assert.notEqual(raw.sample_orders[0].scenario, 'changed')
  for (const changed of [
    { is_demo: false }, { workspace_user_id: 'u1001' }, { csrf_token: 'bad token' },
    { expires_at: snapshot }, { sample_orders: [raw.sample_orders[0], raw.sample_orders[0]] }
  ]) assert.throws(() => normalizeDemoSession({ ...raw, ...changed }))
  const stored = []
  const previous = globalThis.localStorage
  globalThis.localStorage = { setItem: (...args) => stored.push(args) }
  try {
    saveSettings({ backend: 'python', userId: workspace, endpoints: { python: '/py' }, pythonToken: 'private-api-token' })
    assert.equal(stored.length, 1)
    assert.ok(!stored[0][1].includes('private-demo-csrf'))
    assert.ok(!stored[0][1].includes('private-api-token'))
  } finally { globalThis.localStorage = previous }
})

test('record projection preserves minor units, unknown refund and original query times', () => {
  const result = projectDemoOrder(record(), 'DEMO-1001')
  assert.equal(result.found, true)
  assert.equal(result.snapshot, '2026-10-25 00:00:00 UTC')
  assert.ok(result.rows.some(row => row.value === '¥1299.00'))
  assert.ok(result.rows.some(row => row.value.includes('金额待核实')))
  assert.ok(result.rows.some(row => row.value.includes('到货日期未知')))
  assert.equal(formatDemoMoney(0), '¥0.00')
  assert.equal(formatDemoMoney(1), '¥0.01')
  assert.equal(formatDemoMoney(null), '金额待核实')
  assert.throws(() => formatDemoMoney('0'))
  assert.throws(() => formatDemoMoney(Infinity))
  const offset = record()
  offset.as_of = '2026-10-25T00:00:00+00:00'
  assert.equal(projectDemoOrder(offset, 'DEMO-1001').snapshot, result.snapshot)
})

test('projection rejects cross-order records, unsupported provenance and wrong references', () => {
  for (const change of [
    raw => { raw.order.order_id = 'DEMO-1005' },
    raw => { raw.source = 'model-prose' },
    raw => { raw.order.items[0].order_id = 'DEMO-1005' },
    raw => { raw.order.shipments[0].item_id = 'unknown' },
    raw => { raw.order.payment_status = 'probably-paid' }
  ]) {
    const raw = record(); change(raw)
    assert.throws(() => projectDemoOrder(raw, 'DEMO-1001'))
  }
  assert.deepEqual(projectDemoOrder({ source: 'serveflow-mysql-demo-v1', is_demo: true, status: 'not_found', as_of: snapshot, order: null }, 'DEMO-1005'), {
    found: false, snapshot: '2026-10-25 00:00:00 UTC'
  })
})

test('demo authority is bound to endpoint and workspace, then removed explicitly', () => {
  const authority = createDemoAuthority()
  authority.set('/py', session())
  assert.equal(authority.csrfFor('/py', workspace), 'private-demo-csrf')
  assert.equal(authority.csrfFor('/other', workspace), null)
  assert.equal(authority.csrfFor('/py', 'different-workspace'), null)
  authority.clear('/py')
  assert.equal(authority.csrfFor('/py', workspace), null)
})

test('demo HTTP helpers only read fixed paths; non-GET receives scoped CSRF and cookie semantics', async () => {
  const settings = { backend: 'python', userId: workspace, endpoints: { python: '/controlled-python', java: '/java' } }
  const original = globalThis.fetch, sent = []
  globalThis.fetch = async (url, options) => {
    sent.push({ url, options })
    return new Response(JSON.stringify(url.endsWith('/demo/session') ? session() : { results: [] }), { status: 200 })
  }
  try {
    await requestDemoSession(settings)
    await requestDemoOrder(settings, 'DEMO-1001')
    await assert.rejects(() => requestDemoOrder(settings, '../chat'))
    bindDemoAuthority(settings, session())
    await requestSearch('python', settings, 'controlled question')
    await requestSearch('python', { ...settings, userId: 'different-workspace' }, 'controlled question')
    assert.equal(sent.length, 4)
    assert.deepEqual(sent.slice(0, 2).map(item => item.url), ['/controlled-python/demo/session', '/controlled-python/demo/orders/DEMO-1001'])
    assert.ok(sent.slice(0, 2).every(item => !item.options.method || item.options.method === 'GET'))
    assert.equal(sent[2].options.headers.get('X-ServeFlow-Demo-CSRF'), 'private-demo-csrf')
    assert.equal(sent[3].options.headers.get('X-ServeFlow-Demo-CSRF'), null)
    assert.ok(sent.every(item => item.options.credentials === 'same-origin'))
    assert.ok(!sent.some(item => item.url.includes('/chat') || item.url.includes('/eval')))
  } finally { clearDemoAuthority(settings); globalThis.fetch = original }
})

const source = fs.readFileSync(new URL('../src/App.vue', import.meta.url), 'utf8')
function loadFunction(name, context) {
  const found = source.match(new RegExp(`(?:async )?function ${name}\\(`))
  assert.ok(found, name)
  const start = found.index, next = source.slice(start + 1).search(/\n(?:async )?function /)
  const end = next < 0 ? source.indexOf('</script>', start) : start + 1 + next
  vm.runInContext(source.slice(start, end), context)
}
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b }); return { promise, resolve, reject } }

for (const fail of [false, true]) {
  test(`old demo session ${fail ? 'error' : 'success'}/finally cannot bind after A→B→A`, async () => {
    const pending = deferred()
    let context
    context = vm.createContext({
      settings: { backend: 'python', userId: 'old', endpoints: { python: '/py', java: '/java' } },
      demoSequence: 0, demoSessionController: null, demoSession: { value: null },
      demoLoading: { value: false }, demoError: { value: '' },
      resetDemoState: () => { context.demoSequence += 1 },
      backendMeta: (type, settings) => ({ baseUrl: settings.endpoints[type] }),
      AbortController, requestDemoSession: () => pending.promise,
      normalizeDemoSession: () => assert.fail('old session must not be parsed or bound')
    })
    loadFunction('loadDemoSession', context)
    const task = context.loadDemoSession()
    context.demoSequence += 2
    context.demoSession.value = 'new session'
    context.demoError.value = 'new error'
    context.demoLoading.value = true
    if (fail) pending.reject(new Error('old session failed')); else pending.resolve(session())
    await task
    assert.equal(context.demoSession.value, 'new session')
    assert.equal(context.demoError.value, 'new error')
    assert.equal(context.demoLoading.value, true)
  })
}

test('unconfigured demo 404 remains silent and Java never bootstraps a Python identity', async () => {
  let resets = 0, calls = 0
  const context = vm.createContext({
    settings: { backend: 'python', userId: 'original', endpoints: { python: '/py', java: '/java' } },
    demoSequence: 0, demoSessionController: null, demoSession: { value: null },
    demoLoading: { value: false }, demoError: { value: '' },
    resetDemoState: () => { resets += 1 },
    backendMeta: (type, settings) => ({ baseUrl: settings.endpoints[type] }),
    AbortController, requestDemoSession: async () => { calls += 1; throw Object.assign(new Error('not configured'), { status: 404 }) }
  })
  loadFunction('loadDemoSession', context)
  await context.loadDemoSession()
  assert.equal(context.demoError.value, '')
  assert.equal(context.demoLoading.value, false)
  assert.equal(context.settings.userId, 'original')
  context.settings.backend = 'java'
  await context.loadDemoSession()
  assert.equal(calls, 1)
  assert.equal(resets, 2)
})

for (const fail of [false, true]) {
  test(`old demo order ${fail ? 'error' : 'success'}/finally cannot overwrite a new A→B→A read`, async () => {
    const pending = deferred()
    const settings = { backend: 'python' }
    const context = vm.createContext({
      settings, demoSession: { value: { workspace_user_id: workspace } }, demoOrderBusy: { value: false },
      demoLoading: { value: false }, demoOrderId: { value: 'DEMO-1001' }, demoOrder: { value: null },
      demoOrderError: { value: '' }, demoOrderSequence: 0, demoOrderController: null,
      requestState: { begin: () => ({ settings, isCurrent: () => true }) },
      requestDemoOrder: () => pending.promise, isDemoOrderId, AbortController,
      projectDemoOrder: () => assert.fail('stale projection'), expireDemoSession: () => assert.fail('stale identity reset')
    })
    loadFunction('lookupDemoOrder', context)
    const task = context.lookupDemoOrder()
    context.demoOrderSequence += 2
    context.demoOrder.value = 'new record'
    context.demoOrderError.value = 'new error'
    context.demoOrderBusy.value = true
    if (fail) pending.reject(new Error('old failure')); else pending.resolve(record())
    await task
    assert.equal(context.demoOrder.value, 'new record')
    assert.equal(context.demoOrderError.value, 'new error')
    assert.equal(context.demoOrderBusy.value, true)
  })
}

test('duplicate read is suppressed and filling a question never sends a request', async () => {
  const pending = deferred(), calls = [], filled = []
  const settings = { backend: 'python' }
  const context = vm.createContext({
    settings, demoSession: { value: { workspace_user_id: workspace } }, demoOrderBusy: { value: false },
    demoLoading: { value: false }, demoOrderId: { value: 'DEMO-1001' }, demoOrder: { value: null },
    demoOrderError: { value: '' }, demoOrderSequence: 0, demoOrderController: null,
    requestState: { begin: () => ({ settings, isCurrent: () => true }) },
    requestDemoOrder: (...args) => { calls.push(args); return pending.promise }, isDemoOrderId, AbortController,
    projectDemoOrder, usePrompt: message => filled.push(message)
  })
  loadFunction('lookupDemoOrder', context); loadFunction('fillDemoQuestion', context)
  const task = context.lookupDemoOrder()
  await context.lookupDemoOrder()
  assert.equal(calls.length, 1)
  pending.resolve(record()); await task
  context.fillDemoQuestion()
  assert.equal(calls.length, 1)
  assert.equal(filled.length, 1)
  assert.ok(filled[0].includes('DEMO-1001'))
})

test('closing/cancelling invalidates pending read and aborts its fetch once', () => {
  const controller = new AbortController()
  const context = vm.createContext({ demoOrderSequence: 2, demoOrderController: controller, demoOrderBusy: { value: true } })
  loadFunction('cancelDemoOrder', context)
  context.cancelDemoOrder()
  assert.equal(controller.signal.aborted, true)
  assert.equal(context.demoOrderBusy.value, false)
  assert.equal(context.demoOrderSequence, 3)
  context.cancelDemoOrder()
  assert.equal(context.demoOrderBusy.value, false)
})
