import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createInitialSettings, requestMonitor, saveSettings } from '../src/lib/backends.js'

test('old Java selection cannot change the Python-only product entry', () => {
  const previous = globalThis.localStorage
  globalThis.localStorage = { getItem: () => JSON.stringify({ backend: 'java', userId: 'old-user',
    endpoints: { python: '/api/python', java: '/old-java' } }) }
  try {
    const settings = createInitialSettings()
    assert.equal(settings.backend, 'python')
    assert.equal(settings.endpoints.python, '/api/python')
    assert.equal(settings.userId, 'old-user')
  } finally { globalThis.localStorage = previous }
})

test('saved product settings keep Python fixed without persisting credentials', () => {
  const previous = globalThis.localStorage
  let stored
  globalThis.localStorage = { setItem: (_, value) => { stored = JSON.parse(value) } }
  try {
    saveSettings({ backend: 'java', pythonToken: 'test-only-secret', endpoints: { python: '/api/python' } })
    assert.equal(stored.backend, 'python')
    assert.equal(stored.pythonToken, undefined)
    assert.equal(stored.endpoints.python, '/api/python')
  } finally { globalThis.localStorage = previous }
})

test('monitor remains a readonly request with caller cancellation and no retry', async () => {
  const previous = globalThis.fetch
  const calls = []
  const controller = new AbortController()
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options })
    return new Response(JSON.stringify({ agent_stats: {}, tool_stats: {} }), { status: 200 })
  }
  try {
    await requestMonitor('python', { endpoints: { python: '/api/python' }, userId: 'demo' }, { signal: controller.signal })
    assert.equal(calls.length, 1)
    assert.equal(calls[0].url, '/api/python/monitor')
    assert.equal(calls[0].options.method, undefined)
    assert.notEqual(calls[0].options.signal, controller.signal)
    assert.equal(calls[0].options.signal.aborted, false)
    controller.abort()
    assert.equal(calls[0].options.signal.aborted, true)
    assert.equal(calls[0].options.credentials, 'same-origin')
  } finally { globalThis.fetch = previous }
})

test('monitor keeps its timeout when a caller also supplies cancellation', async () => {
  const previousFetch = globalThis.fetch
  const previousTimeout = AbortSignal.timeout
  const deadline = new AbortController()
  const caller = new AbortController()
  let signal
  AbortSignal.timeout = milliseconds => {
    assert.equal(milliseconds, 20000)
    return deadline.signal
  }
  globalThis.fetch = async (_, options) => {
    signal = options.signal
    return new Response('{}', { status: 200 })
  }
  try {
    await requestMonitor('python', { endpoints: { python: '/api/python' }, userId: 'demo' }, { signal: caller.signal })
    assert.equal(signal.aborted, false)
    deadline.abort()
    assert.equal(signal.aborted, true)
    assert.equal(caller.signal.aborted, false)
  } finally {
    globalThis.fetch = previousFetch
    AbortSignal.timeout = previousTimeout
  }
})
