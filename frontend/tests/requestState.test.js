import { test } from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import vm from 'node:vm'
import { createRequestState } from '../src/lib/requestState.js'

function state() { return { backend:'python', userId:'a', conversationId:'', endpoints:{python:'/py',java:'/java'} } }

test('switching away and back invalidates old responses', () => {
  const settings = state()
  const requests = createRequestState(settings)
  const old = requests.begin('stats')
  settings.backend = 'java'; requests.invalidate()
  settings.backend = 'python'; requests.invalidate()
  assert.equal(old.isCurrent(), false)
  assert.equal(requests.begin('stats').isCurrent(), true)
})

test('latest request wins within each slot and request settings are snapshots', () => {
  const settings = state()
  const requests = createRequestState(settings)
  const old = requests.begin('stats')
  const fresh = requests.begin('stats')
  requests.begin('monitor')
  assert.equal(old.isCurrent(), false)
  assert.equal(fresh.isCurrent(), true)
  settings.endpoints.python = '/changed'
  assert.equal(fresh.settings.endpoints.python, '/py')
  assert.equal(fresh.isCurrent(), false)
})

const source = fs.readFileSync(new URL('../src/App.vue', import.meta.url), 'utf8')
function loadFunction(name, context) {
  const match = new RegExp(`^(?:async )?function ${name}\\(`, 'm').exec(source)
  assert.ok(match, `Expected the current application function ${name}`)
  const start = match.index
  const next = source.slice(start + 1).search(/\n(?:async )?function /)
  const end = next < 0 ? source.indexOf('</script>',start) : start + 1 + next
  vm.runInContext(source.slice(start,end),context)
}

// Knowledge management and evaluation pages were explicitly removed. Keep their
// shared request-state coverage above; exercise the current health/monitor UI
// functions here instead of retaining calls to deleted page controllers.
function deferred() {
  let resolve, reject
  const promise = new Promise((a,b) => { resolve = a; reject = b })
  return { promise, resolve, reject }
}

for (const fail of [false,true]) {
  test(`health: old ${fail ? 'error' : 'success'} cannot overwrite connection A→B→A`, async () => {
    const settings = state(), requests = createRequestState(settings), wait = deferred()
    const context = vm.createContext({ settings, requestState: requests,
      healthOk: {value:'fresh'}, healthLabel: {value:'fresh'}, statusText: {value:'fresh'},
      requestHealth: () => wait.promise })
    loadFunction('checkHealth', context)
    const task = context.checkHealth()
    settings.endpoints.python = '/changed'; requests.invalidate()
    settings.endpoints.python = '/py'; requests.invalidate()
    if (fail) wait.reject(new Error('old private server error')); else wait.resolve({status:'ok'})
    await task
    assert.equal(context.healthOk.value, 'fresh')
    assert.equal(context.healthLabel.value, 'fresh')
    assert.equal(context.statusText.value, 'fresh')
  })
}

function monitorContext(requestMonitor) {
  const settings = state()
  const context = vm.createContext({ settings, requestState: createRequestState(settings), AbortController,
    monitorSequence: 0, monitorController: null, monitorLoading: {value:false},
    monitorError: {value:''}, monitorReadAt: {value:'previous read'},
    monitorData: {value:{loaded:true,agent_stats:{previous:true}}}, requestMonitor })
  for (const name of ['cancelMonitor','resetMonitor','loadMonitor']) loadFunction(name,context)
  return context
}

for (const fail of [false,true]) {
  test(`monitor: old ${fail ? 'error' : 'success'} and finally cannot change the new request after A→B→A`, async () => {
    const wait = deferred(), context = monitorContext(() => wait.promise)
    const task = context.loadMonitor()
    context.settings.endpoints.python = '/changed'; context.requestState.invalidate()
    context.cancelMonitor()
    context.settings.endpoints.python = '/py'; context.requestState.invalidate()
    const fresh = {agent_stats:{fresh:true}}, controller = {fresh:true}
    context.monitorData.value = fresh
    context.monitorReadAt.value = 'new read'
    context.monitorError.value = 'new error'
    context.monitorLoading.value = true
    context.monitorController = controller
    if (fail) wait.reject(new Error('old private server error')); else wait.resolve({agent_stats:{old:true}})
    await task
    assert.equal(context.monitorData.value, fresh)
    assert.equal(context.monitorReadAt.value, 'new read')
    assert.equal(context.monitorError.value, 'new error')
    assert.equal(context.monitorLoading.value, true)
    assert.equal(context.monitorController, controller)
  })
}

test('monitor read checks workspace identity even without a request-state invalidation', async () => {
  const wait = deferred(), context = monitorContext(() => wait.promise)
  const task = context.loadMonitor(), previous = context.monitorData.value
  context.settings.userId = 'another workspace'
  context.monitorReadAt.value = 'new workspace read'
  context.monitorError.value = 'new workspace error'
  wait.resolve({agent_stats:{old:true}})
  await task
  assert.equal(context.monitorData.value, previous)
  assert.equal(context.monitorReadAt.value, 'new workspace read')
  assert.equal(context.monitorError.value, 'new workspace error')
  assert.equal(context.monitorLoading.value, true)
})

test('leaving/cancelling monitor aborts once and preserves the last snapshot and read time', async () => {
  const wait = deferred(), context = monitorContext(() => wait.promise)
  const task = context.loadMonitor(), previous = context.monitorData.value
  let aborts = 0
  context.monitorController.signal.addEventListener('abort', () => { aborts += 1 })
  context.cancelMonitor()
  context.cancelMonitor()
  wait.reject(new Error('aborted'))
  await task
  assert.equal(aborts, 1)
  assert.equal(context.monitorLoading.value, false)
  assert.equal(context.monitorError.value, '')
  assert.equal(context.monitorReadAt.value, 'previous read')
  assert.equal(context.monitorData.value, previous)
})

test('failed current monitor refresh keeps old data/date and only shows a fixed error', async () => {
  let calls = 0
  const context = monitorContext(async () => { calls += 1; throw new Error('private raw server detail') })
  const previous = context.monitorData.value
  await context.loadMonitor()
  assert.equal(calls, 1)
  assert.equal(context.monitorData.value, previous)
  assert.equal(context.monitorReadAt.value, 'previous read')
  assert.equal(context.monitorError.value, '监控读取失败，请手动刷新。本次未自动重试。')
  assert.equal(context.monitorLoading.value, false)
})

test('successful monitor refresh is a single cancelable Python read and records its read time', async () => {
  let calls = 0
  const context = monitorContext(async (type, settings, options) => {
    calls += 1
    assert.equal(type, 'python')
    assert.equal(settings.userId, 'a')
    assert.ok(options.signal instanceof AbortSignal)
    return {agent_stats:{general:{total:1}}}
  })
  await context.loadMonitor()
  assert.equal(calls, 1)
  assert.equal(context.monitorData.value.agent_stats.general.total, 1)
  assert.equal(context.monitorData.value.loaded, true)
  assert.ok(Number.isFinite(Date.parse(context.monitorReadAt.value)))
  assert.equal(context.monitorError.value, '')
  assert.equal(context.monitorLoading.value, false)
})
