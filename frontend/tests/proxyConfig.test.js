import { test } from 'node:test'
import assert from 'node:assert/strict'
import http from 'node:http'
import { fileURLToPath } from 'node:url'
import { createServer, resolveConfig } from 'vite'
import { createFrontendConfig, resolveBackendOrigin } from '../vite.config.js'

test('backend target accepts an explicit trusted origin and rejects malformed configuration', () => {
  assert.equal(resolveBackendOrigin(undefined), 'http://127.0.0.1:8816')
  assert.equal(resolveBackendOrigin('http://127.0.0.1:19876/'), 'http://127.0.0.1:19876')
  assert.equal(resolveBackendOrigin('https://api.example.test'), 'https://api.example.test')
  for (const invalid of ['', null, 'localhost:8000', 'file:///tmp/app', ' http://127.0.0.1:8816',
    'http://127.0.0.1:8816/api', 'http://127.0.0.1:8816?x=1', 'http://127.0.0.1:8816#test',
    'http://user:password@127.0.0.1:8816', 'http://127.0.0.1:invalid', 'http:\\127.0.0.1']) {
    assert.throws(() => resolveBackendOrigin(invalid), /SERVEFLOW_BACKEND_ORIGIN/)
  }
})
test('the browser retains a same-origin default and server origin is not exposed as client env', async () => {
  const previous = process.env.SERVEFLOW_BACKEND_ORIGIN
  process.env.SERVEFLOW_BACKEND_ORIGIN = 'http://127.0.0.1:19876'
  try {
    const config = createFrontendConfig({ mode: 'test' })
    assert.equal(config.server.host, '127.0.0.1')
    assert.equal(config.preview.host, '127.0.0.1')
    assert.deepEqual(Object.keys(config.server.proxy), ['/api/python'])
    assert.equal(config.preview.proxy['/api/python'].target, 'http://127.0.0.1:19876')
    const resolved = await resolveConfig({ ...config, configFile: false }, 'serve', 'test')
    assert.ok(!Object.hasOwn(resolved.env, 'SERVEFLOW_BACKEND_ORIGIN'))
    assert.ok(!Object.values(resolved.env).includes('http://127.0.0.1:19876'))
    const { createInitialSettings } = await import('../src/lib/backends.js')
    assert.equal(createInitialSettings().endpoints.python, '/api/python')
  } finally {
    if (previous === undefined) delete process.env.SERVEFLOW_BACKEND_ORIGIN
    else process.env.SERVEFLOW_BACKEND_ORIGIN = previous
  }
})

test('local proxy preserves method, query, body, CSRF headers and upstream error status', { timeout: 15000 }, async () => {
  const received = []
  const upstream = http.createServer(async (request, response) => {
    let body = ''
    for await (const chunk of request) body += chunk
    const observation = { method: request.method, path: request.url, body,
      origin: request.headers.origin, csrf: request.headers['x-serveflow-demo-csrf'] }
    received.push(observation)
    response.writeHead(request.url.startsWith('/failure') ? 503 : 200, { 'Content-Type': 'application/json' })
    response.end(JSON.stringify(observation))
  })
  await new Promise(resolve => upstream.listen(0, '127.0.0.1', resolve))
  const origin = `http://127.0.0.1:${upstream.address().port}`
  const previous = process.env.SERVEFLOW_BACKEND_ORIGIN
  process.env.SERVEFLOW_BACKEND_ORIGIN = origin
  let vite, gateway
  try {
    const config = createFrontendConfig({ mode: 'test' })
    vite = await createServer({ ...config, configFile: false,
      root: fileURLToPath(new URL('..', import.meta.url)), logLevel: 'silent',
      server: { ...config.server, middlewareMode: true, hmr: false, watch: null } })
    gateway = http.createServer(vite.middlewares)
    await new Promise(resolve => gateway.listen(0, '127.0.0.1', resolve))
    const address = `http://127.0.0.1:${gateway.address().port}`
    const response = await fetch(`${address}/api/python/check?item=one&item=two`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', Origin: address,
        'X-ServeFlow-Demo-CSRF': 'controlled-only' }, body: JSON.stringify({ message: 'controlled request' })
    })
    assert.equal(response.status, 200)
    assert.deepEqual(await response.json(), { method: 'POST', path: '/check?item=one&item=two',
      body: '{"message":"controlled request"}', origin: address, csrf: 'controlled-only' })
    const failure = await fetch(`${address}/api/python/failure?reason=controlled`)
    assert.equal(failure.status, 503)
    assert.equal((await failure.json()).path, '/failure?reason=controlled')
    assert.equal(received.length, 2, 'no implicit proxy retry')
  } finally {
    if (previous === undefined) delete process.env.SERVEFLOW_BACKEND_ORIGIN
    else process.env.SERVEFLOW_BACKEND_ORIGIN = previous
    if (gateway) {
      gateway.closeAllConnections()
      await new Promise(resolve => gateway.close(resolve))
    }
    await vite?.close()
    upstream.closeAllConnections()
    await new Promise(resolve => upstream.close(resolve))
  }
})
