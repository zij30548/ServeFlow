<template>
  <section class="monitor-dashboard" aria-labelledby="monitor-heading" :aria-busy="loading">
    <header class="monitor-heading">
      <div>
        <span class="kicker">RUNTIME MONITOR</span>
        <h1 id="monitor-heading">运行监控</h1>
        <p>查看 Agent 与工具的执行状态。</p>
      </div>
      <button class="quiet-button" type="button" :disabled="loading" @click="$emit('refresh')">
        <AppIcon name="refresh" />{{ loading ? '读取中' : '刷新指标' }}
      </button>
    </header>

    <p v-if="error" class="monitor-error" role="alert">监控数据读取失败，请手动刷新。{{ view.available ? '下方保留上次读取结果。' : '' }}</p>
    <p v-if="view.incomplete" class="monitor-note" role="status">部分指标格式异常，已隐藏；未知值以 — 显示。</p>

    <template v-if="view.available">
      <div class="monitor-overview" aria-label="Agent 执行概览">
        <div><span>Agent 执行次数</span><strong>{{ formatMonitorCount(view.agentSummary.total) }}</strong></div>
        <div><span>执行成功率</span><strong>{{ summaryRate }}</strong></div>
        <div><span>平均执行耗时</span><strong>{{ summaryDuration }}</strong></div>
      </div>
      <p class="monitor-note overview-note">按 Agent 执行统计，并行角色分别计数；成功率反映执行是否完成，不代表回答准确率。汇总成功率和耗时为加权约数。</p>

      <section class="monitor-section" aria-labelledby="agent-monitor-heading">
        <div class="monitor-section-heading"><h2 id="agent-monitor-heading">Agent 状态</h2><span>{{ view.agents.length }} 个实例</span></div>
        <div class="monitor-table-wrap">
          <table class="monitor-table">
            <thead><tr><th scope="col">处理角色</th><th scope="col">执行次数</th><th scope="col">成功率</th><th scope="col">平均耗时</th><th scope="col">恢复状态</th></tr></thead>
            <tbody>
              <tr v-for="agent in view.agents" :key="agent.id">
                <th scope="row">{{ agent.label }}</th><td>{{ formatMonitorCount(agent.total) }}</td>
                <td>{{ agent.total === 0 ? '暂无样本' : formatMonitorRate(agent.rate) }}</td>
                <td>{{ agent.total === 0 ? '—' : formatMonitorDuration(agent.averageMs) }}</td>
                <td><span class="monitor-state" :class="agent.stateTone">{{ agent.state }}</span></td>
              </tr>
              <tr v-if="!view.agents.length"><td colspan="5" class="monitor-empty-cell">暂无 Agent 指标</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <section class="monitor-section" aria-labelledby="tool-monitor-heading">
        <div class="monitor-section-heading"><h2 id="tool-monitor-heading">已注册工具</h2><span>{{ view.tools.length }} 项</span></div>
        <div class="monitor-table-wrap">
          <table class="monitor-table">
            <thead><tr><th scope="col">工具</th><th scope="col">调用次数</th><th scope="col">成功率</th><th scope="col">平均耗时</th><th scope="col">熔断状态</th></tr></thead>
            <tbody>
              <tr v-for="tool in view.tools" :key="tool.id">
                <th scope="row">{{ tool.label }}</th><td>{{ formatMonitorCount(tool.total) }}</td>
                <td>{{ tool.total === 0 ? '暂无样本' : formatMonitorRate(tool.rate) }}</td>
                <td>{{ tool.total === 0 ? '—' : formatMonitorDuration(tool.averageMs) }}</td>
                <td><span class="monitor-state" :class="tool.stateTone">{{ tool.state }}</span></td>
              </tr>
              <tr v-if="!view.tools.length"><td colspan="5" class="monitor-empty-cell">暂无已注册工具指标</td></tr>
            </tbody>
          </table>
        </div>
        <p class="monitor-note">此表仅覆盖后端工具管理器，含缓存命中；不是全部工具、外部发送或费用计数。</p>
      </section>

      <section v-if="view.alerts.length" class="monitor-section" aria-labelledby="monitor-alerts-heading">
        <div class="monitor-section-heading"><h2 id="monitor-alerts-heading">阈值提醒</h2><span>{{ view.alerts.length }} 项</span></div>
        <ul class="monitor-alert-list">
          <li v-for="alert in view.alerts" :key="alert.key">
            <span class="monitor-state" :class="alert.tone">{{ alert.severity }}</span>
            <strong>{{ alert.label }}</strong><span>{{ alert.value }} · 阈值 {{ alert.threshold }}</span>
          </li>
        </ul>
      </section>
      <footer class="monitor-footer"><span>读取于 {{ formatMonitorReadTime(readAt) }}</span><span>本进程统计，服务重启可能清零。</span></footer>
    </template>
    <div v-else class="monitor-empty" role="status"><AppIcon name="chart" /><h2>{{ loading ? '正在读取运行数据' : '尚未取得监控数据' }}</h2><p>刷新后查看当前服务的指标。</p></div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import AppIcon from './AppIcon.vue'
import { projectMonitorSnapshot, formatMonitorCount, formatMonitorRate, formatMonitorDuration, formatMonitorReadTime } from '../lib/monitorStats.js'

const props = defineProps({
  snapshot: { type: Object, default: null },
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
  readAt: { type: String, default: null }
})
defineEmits(['refresh'])
const view = computed(() => projectMonitorSnapshot(props.snapshot))
const summaryRate = computed(() => view.value.agentSummary.total === 0 ? '暂无样本'
  : view.value.agentSummary.rate === null ? '—' : `≈ ${formatMonitorRate(view.value.agentSummary.rate)}`)
const summaryDuration = computed(() => view.value.agentSummary.averageMs === null ? '—' : `≈ ${formatMonitorDuration(view.value.agentSummary.averageMs)}`)
</script>

<style scoped>
.monitor-dashboard { flex: 1; min-height: 0; overflow-y: auto; padding: 38px 40px; }
.monitor-heading { display: flex; justify-content: space-between; align-items: center; gap: 20px; margin-bottom: 30px; }
.monitor-heading h1 { margin-top: 6px; }
.monitor-heading p { color: var(--muted); margin-top: 10px; font-size: 13px; line-height: 1.8; }
.monitor-heading button { flex-shrink: 0; }
.monitor-heading .icon { width: 15px; height: 15px; }
.monitor-overview { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); border: 1px solid var(--line); border-radius: 14px; background: white; }
.monitor-overview > div { display: flex; flex-direction: column; gap: 15px; min-width: 0; padding: 25px; }
.monitor-overview > div + div { border-left: 1px solid var(--line); }
.monitor-overview span { color: var(--muted); font-size: 12px; }
.monitor-overview strong { font-size: clamp(20px, 2.6vw, 30px); font-weight: 600; letter-spacing: -.04em; line-height: 1.3; overflow-wrap: anywhere; }
.monitor-note { color: var(--muted); font-size: 11px; line-height: 1.8; margin-top: 14px; }
.overview-note { margin: 14px 0 27px; }
.monitor-error { color: var(--danger); padding: 12px 15px; background: #fff7f5; border: 1px solid #f2d8d2; border-radius: 9px; font-size: 12px; line-height: 1.8; margin-bottom: 18px; }
.monitor-section { margin-top: 22px; padding: 24px; border: 1px solid var(--line); border-radius: 14px; background: white; }
.monitor-section-heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 17px; }
.monitor-section-heading span { color: var(--muted); font-size: 11px; }
.monitor-table-wrap { overflow-x: auto; }
.monitor-table { width: 100%; border-collapse: collapse; text-align: left; font-size: 12px; table-layout: fixed; }
.monitor-table th, .monitor-table td { padding: 15px 10px; line-height: 1.7; border-bottom: 1px solid var(--line); overflow-wrap: anywhere; }
.monitor-table thead th { color: var(--muted); font-size: 11px; font-weight: 500; padding-top: 5px; }
.monitor-table th:first-child { width: 28%; padding-left: 0; }
.monitor-table th:last-child, .monitor-table td:last-child { padding-right: 0; }
.monitor-table tbody th { font-weight: 500; }
.monitor-table tbody tr:last-child th, .monitor-table tbody tr:last-child td { border-bottom: none; }
.monitor-state { display: inline-flex; align-items: center; gap: 5px; color: var(--muted); font-size: 11px; white-space: nowrap; }
.monitor-state:before { content: ''; width: 5px; height: 5px; border-radius: 50%; background: currentColor; }
.monitor-state.healthy { color: #397557; }
.monitor-state.warning { color: #a34c38; }
.monitor-state.pending { color: #8b6a25; }
.monitor-empty-cell { color: var(--muted); text-align: center; }
.monitor-alert-list { list-style: none; margin: 0; padding: 0; }
.monitor-alert-list li { display: flex; align-items: center; flex-wrap: wrap; gap: 12px; padding: 13px 0; font-size: 12px; border-bottom: 1px solid var(--line); }
.monitor-alert-list li:last-child { border-bottom: none; }
.monitor-alert-list strong { font-weight: 500; }
.monitor-alert-list li > span:last-child { color: var(--muted); margin-left: auto; }
.monitor-footer { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px; color: var(--muted); font-size: 11px; line-height: 1.8; margin: 20px 0 8px; }
.monitor-empty { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 320px; gap: 16px; text-align: center; }
.monitor-empty > .icon { width: 30px; height: 30px; color: var(--faint); }
.monitor-empty h2 { font-size: 20px; font-weight: 500; }
.monitor-empty p { color: var(--muted); font-size: 12px; }
@media (max-width: 760px) {
  .monitor-dashboard { padding: 26px 22px; }
  .monitor-heading { align-items: flex-start; margin-bottom: 24px; }
  .monitor-heading h1 { font-size: 26px; }
  .monitor-overview > div { padding: 20px 14px; gap: 12px; }
  .monitor-overview span { font-size: 10px; }
  .monitor-overview strong { font-size: 20px; }
  .monitor-section { padding: 20px 16px; }
  .monitor-table { font-size: 11px; }
  .monitor-table th, .monitor-table td { padding: 13px 5px; }
  .monitor-table th:first-child { width: 26%; }
  .monitor-table th:nth-child(2) { width: 14%; }
  .monitor-table th:nth-child(3) { width: 21%; }
  .monitor-table th:nth-child(4) { width: 19%; }
  .monitor-table th:nth-child(5) { width: 20%; }
  .monitor-table thead th, .monitor-state { font-size: 10px; }
}
@media (max-width: 420px) {
  .monitor-dashboard { padding: 24px 17px; }
  .monitor-heading { gap: 12px; }
  .monitor-heading p { font-size: 12px; }
  .monitor-heading button { min-height: 34px; padding-inline: 10px; font-size: 11px; }
  .monitor-overview > div { padding: 18px 10px; }
  .monitor-overview strong { font-size: 18px; }
  .monitor-section { padding: 18px 12px; }
  .monitor-table th, .monitor-table td { padding-inline: 3px; }
  .monitor-state { gap: 3px; }
  .monitor-state:before { width: 4px; height: 4px; }
  .monitor-note { font-size: 10px; }
}
</style>
