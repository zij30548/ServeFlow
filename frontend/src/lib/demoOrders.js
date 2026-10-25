const ORDER_ID = /^DEMO-[0-9]{4}$/
const ISO_TIME = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)$/
const statuses = Object.freeze({
  physical: '实体商品', digital: '数字服务', bundle: '设备与服务组合',
  unpaid: '未支付', paid: '已支付', refunded: '已退款',
  in_transit: '运输中', awaiting_dispatch: '待发货', preorder: '预售待出库',
  delivered: '已签收', delivery_disputed: '签收存在争议',
  activation_failed: '权益开通失败', revoked_after_refund: '退款后权益已撤销', active: '权益有效',
  pending_review: '退款待审核', completed: '退款已完成',
  device: '设备', accessory: '配件', subscription: '订阅', digital_content: '数字内容',
  'demo-wallet': '演示钱包', 'demo-store': '演示商店'
})

function text(value, maximum = 160) {
  if (typeof value !== 'string' || !value.trim() || value.length > maximum) throw new Error('demo_response_invalid')
  return value
}

function time(value) {
  if (!ISO_TIME.test(value) || !Number.isFinite(Date.parse(value))) throw new Error('demo_response_invalid')
  return value.replace('T', ' ').replace(/(?:Z|\+00:00)$/, ' UTC')
}

function label(value) {
  if (!Object.hasOwn(statuses, value)) throw new Error('demo_response_invalid')
  return statuses[value]
}

function list(value, maximum = 20) {
  if (!Array.isArray(value) || value.length > maximum) throw new Error('demo_response_invalid')
  return value
}

export function isDemoOrderId(value) {
  return typeof value === 'string' && ORDER_ID.test(value)
}

export function formatDemoMoney(minor) {
  if (minor === null || minor === undefined) return '金额待核实'
  if (!Number.isSafeInteger(minor) || minor < 0) throw new Error('demo_response_invalid')
  const digits = String(minor).padStart(3, '0')
  return `¥${digits.slice(0, -2)}.${digits.slice(-2)}`
}

export function normalizeDemoSession(raw, now = Date.now()) {
  if (raw?.enabled !== true || raw?.is_demo !== true) throw new Error('demo_response_invalid')
  if (!/^sf_demo_[A-Za-z0-9_-]{16,128}$/.test(raw.workspace_user_id)) throw new Error('demo_response_invalid')
  const csrf = text(raw.csrf_token, 256)
  if (/\s/.test(csrf)) throw new Error('demo_response_invalid')
  time(raw.expires_at)
  if (Date.parse(raw.expires_at) <= now) throw new Error('demo_session_expired')
  const samples = list(raw.sample_orders, 12).map(sample => {
    if (!isDemoOrderId(sample.order_id)) throw new Error('demo_response_invalid')
    return { order_id: sample.order_id, scenario: text(sample.scenario, 100) }
  })
  if (new Set(samples.map(sample => sample.order_id)).size !== samples.length) throw new Error('demo_response_invalid')
  return {
    workspace_user_id: raw.workspace_user_id,
    customer_label: text(raw.customer_label, 40),
    csrf_token: csrf,
    expires_at: raw.expires_at,
    sample_orders: samples
  }
}

export function projectDemoOrder(raw, requestedId) {
  if (raw?.source !== 'serveflow-mysql-demo-v1' || raw?.is_demo !== true) throw new Error('demo_response_invalid')
  const snapshot = time(raw.as_of)
  if (raw.status === 'not_found' && raw.order === null) return { found: false, snapshot }
  const order = raw.order
  if (raw.status !== 'found' || !order || order.order_id !== requestedId || !isDemoOrderId(order.order_id) || order.currency !== 'CNY') {
    throw new Error('demo_response_invalid')
  }
  const rows = [
    { name: '订单类型', value: label(order.order_type) },
    { name: '订单金额', value: formatDemoMoney(order.total_amount_minor) },
    { name: '支付状态', value: label(order.payment_status) }
  ]
  const itemIds = new Set()
  const belongs = row => {
    if (row.order_id !== order.order_id) throw new Error('demo_response_invalid')
  }
  for (const item of list(order.items)) {
    belongs(item)
    itemIds.add(text(item.item_id, 80))
    if (!Number.isSafeInteger(item.quantity) || item.quantity < 1) throw new Error('demo_response_invalid')
    rows.push({ name: '商品', value: `${text(item.name)} × ${item.quantity}（${label(item.item_type)}，单价${formatDemoMoney(item.unit_amount_minor)}）` })
  }
  for (const payment of list(order.payments)) {
    belongs(payment)
    rows.push({ name: '支付记录', value: `${text(payment.payment_id, 80)} · ${label(payment.status)} · ${formatDemoMoney(payment.amount_minor)} · ${label(payment.channel)}` })
  }
  for (const shipment of list(order.shipments)) {
    belongs(shipment)
    if (!itemIds.has(shipment.item_id)) throw new Error('demo_response_invalid')
    if (shipment.estimated_delivery !== null && !/^\d{4}-\d{2}-\d{2}$/.test(shipment.estimated_delivery)) throw new Error('demo_response_invalid')
    const estimate = shipment.estimated_delivery === null ? '到货日期未知' : text(shipment.estimated_delivery, 10)
    rows.push({ name: '包裹', value: `${text(shipment.shipment_id, 80)} · ${label(shipment.status)} · ${shipment.tracking_id === null ? '无物流单号' : text(shipment.tracking_id, 80)} · ${estimate}` })
  }
  for (const entitlement of list(order.entitlements)) {
    belongs(entitlement)
    if (typeof entitlement.auto_renew !== 'boolean') throw new Error('demo_response_invalid')
    rows.push({ name: '数字权益', value: `${text(entitlement.package_name)} · ${label(entitlement.status)} · ${entitlement.auto_renew ? '自动续费已开启' : '自动续费已关闭'} · ${entitlement.expires_at === null ? '到期时间未知' : time(entitlement.expires_at)}` })
  }
  for (const refund of list(order.refunds)) {
    belongs(refund)
    rows.push({ name: '退款', value: `${text(refund.refund_id, 80)} · ${label(refund.status)} · ${formatDemoMoney(refund.amount_minor)}` })
  }
  return { found: true, orderId: order.order_id, snapshot, updatedAt: time(order.record_updated_at), rows }
}

export function createDemoAuthority() {
  const sessions = new Map()
  return {
    set(baseUrl, session) { sessions.set(baseUrl, { userId: session.workspace_user_id, csrf: session.csrf_token }) },
    clear(baseUrl) { sessions.delete(baseUrl) },
    csrfFor(baseUrl, userId) {
      const session = sessions.get(baseUrl)
      return session?.userId === userId ? session.csrf : null
    }
  }
}
