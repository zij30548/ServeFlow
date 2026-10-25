const AGENT_LABELS = Object.freeze({
  general: '通用接待', technical: '技术支持', billing: '账单服务', escalation: '人工交接'
})
const TOOL_LABELS = Object.freeze({
  knowledge_search: '知识检索', search_knowledge_base: '知识库查询',
  inspect_request_context: '请求核对', suggest_required_fields: '补充信息',
  lookup_error_code: '错误码查询', build_diagnostic_plan: '排障建议',
  check_billing_fields: '账单核对', compare_amounts: '金额比较',
  create_handoff_summary: '交接摘要', read_demo_order: '演示订单查询'
})
const AGENT_ORDER = Object.keys(AGENT_LABELS)
const RECOVERY = Object.freeze({ healthy: '正常', cooldown: '冷却中', probing: '恢复探测' })
const CIRCUIT = Object.freeze({ closed: '正常', open: '已熔断', half_open: '恢复探测' })
const ALERT_METRICS = Object.freeze({
  agent_success_rate: { label: '执行成功率', type: 'agent', unit: 'rate' },
  agent_avg_ms: { label: '平均耗时', type: 'agent', unit: 'ms' },
  tool_success_rate: { label: '执行成功率', type: 'tool', unit: 'rate' },
  tool_avg_ms: { label: '平均耗时', type: 'tool', unit: 'ms' }
})
const SEVERITY = Object.freeze({ info: '提示', warning: '提醒', error: '异常', critical: '严重异常' })

function record(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function number(value, maximum = Number.MAX_SAFE_INTEGER) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= maximum ? value : null
}

function count(value) {
  return Number.isSafeInteger(value) && value >= 0 ? value : null
}

function agentIdentity(value) {
  const match = /^(general|technical|billing|escalation)_([0-9]{1,2})$/.exec(value)
  return match ? { label: `${AGENT_LABELS[match[1]]} · ${Number(match[2]) + 1}`, group: match[1], instance: Number(match[2]) } : null
}

function rowsFor(input, kind) {
  const rows = []
  let rejected = 0
  if (!record(input)) return { rows, rejected: 0, available: false }
  for (const [id, raw] of Object.entries(input).slice(0, 100)) {
    const identity = kind === 'agent' ? agentIdentity(id) : Object.hasOwn(TOOL_LABELS, id) ? { label: TOOL_LABELS[id] } : null
    if (!identity || !record(raw)) { rejected++; continue }
    const total = count(raw.total)
    const sampled = total !== null && total > 0
    const state = kind === 'agent' ? raw.recovery_state : raw.circuit_state
    const states = kind === 'agent' ? RECOVERY : CIRCUIT
    rows.push({
      id, label: identity.label, group: identity.group, instance: identity.instance,
      total, sampled,
      rate: sampled ? number(raw.success_rate, 1) : null,
      averageMs: sampled ? number(kind === 'agent' ? raw.avg_ms : raw.avg_latency_ms) : null,
      state: Object.hasOwn(states, state) ? states[state] : '未知',
      stateTone: state === 'open' || state === 'cooldown' ? 'warning' : state === 'half_open' || state === 'probing' ? 'pending' : state === 'healthy' || state === 'closed' ? 'healthy' : 'unknown',
      consecutiveFails: kind === 'tool' ? count(raw.consecutive_fails) : null
    })
  }
  if (Object.keys(input).length > 100) rejected += Object.keys(input).length - 100
  rows.sort((a, b) => kind === 'agent'
    ? AGENT_ORDER.indexOf(a.group) - AGENT_ORDER.indexOf(b.group) || a.instance - b.instance
    : a.id.localeCompare(b.id))
  return { rows, rejected, available: true }
}

function aggregate(group) {
  const { rows } = group
  if (!group.available || group.rejected || rows.some(row => row.total === null)) return { total: null, rate: null, averageMs: null }
  const total = rows.reduce((sum, row) => sum + row.total, 0)
  if (!Number.isSafeInteger(total)) return { total: null, rate: null, averageMs: null }
  const sampled = rows.filter(row => row.total > 0)
  if (!total) return { total, rate: null, averageMs: null }
  const weighted = field => sampled.some(row => row[field] === null) ? null
    : sampled.reduce((sum, row) => sum + row.total / total * row[field], 0)
  return { total, rate: weighted('rate'), averageMs: weighted('averageMs') }
}

function alertsFor(value) {
  if (!Array.isArray(value)) return []
  const seen = new Set()
  const alerts = []
  for (const raw of value.slice(-10)) {
    if (!record(raw) || raw.resolved === true || !Object.hasOwn(SEVERITY, raw.severity) || typeof raw.metric !== 'string') continue
    const [metric, identifier, extra] = raw.metric.split(':')
    const definition = ALERT_METRICS[metric]
    if (!definition || extra !== undefined) continue
    const subject = definition.type === 'agent' ? agentIdentity(identifier)?.label
      : Object.hasOwn(TOOL_LABELS, identifier) ? TOOL_LABELS[identifier] : null
    const maximum = definition.unit === 'rate' ? 1 : Number.MAX_SAFE_INTEGER
    const measured = number(raw.value, maximum)
    const threshold = number(raw.threshold, maximum)
    if (!subject || measured === null || threshold === null) continue
    const key = `${raw.metric}:${raw.value}:${raw.threshold}:${raw.severity}`
    if (seen.has(key)) continue
    seen.add(key)
    alerts.push({ key, severity: SEVERITY[raw.severity], tone: ['error', 'critical'].includes(raw.severity) ? 'warning' : 'pending',
      label: `${subject} · ${definition.label}`, value: definition.unit === 'rate' ? formatMonitorRate(measured) : formatMonitorDuration(measured),
      threshold: definition.unit === 'rate' ? formatMonitorRate(threshold) : formatMonitorDuration(threshold) })
  }
  return alerts
}

export function projectMonitorSnapshot(snapshot) {
  const raw = record(snapshot) ? snapshot : {}
  const agents = rowsFor(raw.agent_stats, 'agent')
  const tools = rowsFor(raw.tool_stats, 'tool')
  return {
    available: record(snapshot) && agents.available && tools.available,
    agents: agents.rows, tools: tools.rows,
    agentSummary: aggregate(agents), toolSummary: aggregate(tools),
    alerts: alertsFor(raw.active_alerts),
    incomplete: agents.rejected > 0 || tools.rejected > 0,
    metricScope: 'technical-execution'
  }
}

export function formatMonitorCount(value) {
  return count(value) === null ? '—' : String(value)
}

export function formatMonitorRate(value) {
  return number(value, 1) === null ? '—' : `${(value * 100).toFixed(1)}%`
}

export function formatMonitorDuration(value) {
  if (number(value) === null) return '—'
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`
}

export function formatMonitorReadTime(value) {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}T/.test(value) || !Number.isFinite(Date.parse(value))) return '—'
  return new Date(value).toLocaleString('zh-CN', { hour12: false })
}
