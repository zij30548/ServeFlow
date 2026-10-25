import test from 'node:test'
import assert from 'node:assert/strict'
import { projectMonitorSnapshot, formatMonitorCount, formatMonitorRate, formatMonitorDuration, formatMonitorReadTime } from '../src/lib/monitorStats.js'

const agent = (total, success_rate, avg_ms) => ({ total, success_rate, avg_ms, recovery_state: 'healthy' })
const tool = (total, success_rate, avg_latency_ms) => ({ total, success_rate, avg_latency_ms, circuit_state: 'closed', consecutive_fails: 0 })

test('zero backend samples never become a fabricated 100 percent success rate', () => {
  const view = projectMonitorSnapshot({ agent_stats: { general_0: agent(0, 1, 0) }, tool_stats: { knowledge_search: tool(0, 1, 0) } })
  assert.equal(view.available, true)
  assert.deepEqual(view.agentSummary, { total: 0, rate: null, averageMs: null })
  assert.equal(view.agents[0].sampled, false)
  assert.equal(view.agents[0].rate, null)
  assert.equal(view.tools[0].averageMs, null)
  assert.equal(view.metricScope, 'technical-execution')
})

test('summary counts role executions and weights reported rate and latency without inventing success counts', () => {
  const view = projectMonitorSnapshot({ agent_stats: { general_0: agent(2, 1, 100), billing_0: agent(6, .5, 300) }, tool_stats: { knowledge_search: tool(3, 1, 20) } })
  assert.deepEqual(view.agentSummary, { total: 8, rate: .625, averageMs: 250 })
  assert.equal(view.toolSummary.total, 3)
  assert.equal('successCount' in view.agentSummary, false)
  assert.equal(view.agents[1].label, '账单服务 · 1')
})

test('invalid numbers are unknown, never coerced or clamped into good metrics', () => {
  for (const invalid of [NaN, Infinity, -1, '4', null, true]) {
    const view = projectMonitorSnapshot({ agent_stats: { general_0: agent(invalid, .9, 25) }, tool_stats: {} })
    assert.equal(view.agentSummary.total, null)
    assert.equal(view.agents[0].rate, null)
    assert.equal(formatMonitorCount(invalid), '—')
  }
  const view = projectMonitorSnapshot({ agent_stats: { general_0: agent(3, 95, Infinity) }, tool_stats: {} })
  assert.equal(view.agentSummary.total, 3)
  assert.equal(view.agentSummary.rate, null)
  assert.equal(view.agentSummary.averageMs, null)
})

test('fractional success values have one scale and missing values do not turn into zero', () => {
  assert.equal(formatMonitorRate(.953), '95.3%')
  assert.equal(formatMonitorRate(1), '100.0%')
  assert.equal(formatMonitorRate(95), '—')
  assert.equal(formatMonitorRate(null), '—')
  assert.equal(formatMonitorDuration(0), '0 ms')
  assert.equal(formatMonitorDuration(1500), '1.50 s')
  assert.equal(formatMonitorDuration(undefined), '—')
})

test('projection never exposes role prompts, customer text, unknown identifiers or server advice bodies', () => {
  const view = projectMonitorSnapshot({
    agent_stats: { general_0: { ...agent(1, 1, 50), role: 'secret role', workflow: ['customer message'] }, 'customer-private-name': agent(1, 1, 1) },
    tool_stats: { 'secret-tool-id': tool(1, 1, 1), knowledge_search: tool(1, 1, 10) },
    suggestions: [{ title: 'private title', action: 'private action' }],
    active_alerts: [{ severity: 'error', metric: 'agent_success_rate:general_0', value: .5, threshold: .9, message: 'private payload' }]
  })
  const serialized = JSON.stringify(view)
  for (const forbidden of ['secret role', 'customer message', 'customer-private-name', 'secret-tool-id', 'private title', 'private action', 'private payload']) {
    assert.equal(serialized.includes(forbidden), false)
  }
  assert.equal(view.incomplete, true)
  assert.equal(view.agentSummary.total, null)
  assert.equal(view.alerts[0].label, '通用接待 · 1 · 执行成功率')
})

test('unavailable snapshots stay unavailable and unsafe counter totals are not added', () => {
  for (const input of [null, undefined, [], {}, { agent_stats: [], tool_stats: {} }]) {
    assert.equal(projectMonitorSnapshot(input).available, false)
  }
  const view = projectMonitorSnapshot({ agent_stats: { general_0: agent(Number.MAX_SAFE_INTEGER, 1, 1), technical_0: agent(1, 1, 1) }, tool_stats: {} })
  assert.equal(view.agentSummary.total, null)
})

test('alert projection uses only recognized metrics and deduplicates repeated threshold samples', () => {
  const alert = { severity: 'warning', metric: 'tool_avg_ms:knowledge_search', value: 6000, threshold: 5000 }
  const view = projectMonitorSnapshot({ agent_stats: {}, tool_stats: {}, active_alerts: [
    alert, alert, { ...alert, metric: 'secret:customer' }, { ...alert, value: Infinity }, { ...alert, resolved: true },
    { ...alert, metric: 'tool_success_rate:knowledge_search', value: 95, threshold: .95 }
  ] })
  assert.equal(view.alerts.length, 1)
  assert.equal(view.alerts[0].value, '6.00 s')
  assert.equal(view.alerts[0].threshold, '5.00 s')
})

test('known recovery and circuit states are fixed labels; arbitrary strings do not reach the page', () => {
  const view = projectMonitorSnapshot({ agent_stats: { general_0: { ...agent(3, 1, 0), recovery_state: 'cooldown' } }, tool_stats: { knowledge_search: { ...tool(2, .5, 20), circuit_state: 'open' } } })
  assert.equal(view.agents[0].state, '冷却中')
  assert.equal(view.tools[0].state, '已熔断')
  const invalid = projectMonitorSnapshot({ agent_stats: { general_0: { ...agent(1, 1, 1), recovery_state: 'private failure text' } }, tool_stats: {} })
  assert.equal(invalid.agents[0].state, '未知')
  assert.equal(formatMonitorReadTime('private time text'), '—')
})
