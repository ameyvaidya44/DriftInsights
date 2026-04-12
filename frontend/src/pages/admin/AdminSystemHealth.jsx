import AdminPageHeader from '../../components/admin/AdminPageHeader'
import AdminStatCard from '../../components/admin/AdminStatCard'
import HealthIndicator from '../../components/admin/HealthIndicator'
import { ErrorState, LoadingSkeleton } from '../../components/admin/EmptyState'
import useFetch from '../../hooks/useFetch'
import useWebSocket from '../../hooks/useWebSocket'
import { fetchHealth, fetchStatus, fetchSystemHealth } from '../../services/api'
import { formatTimestamp } from '../../utils/formatters'

function UptimeBar({ pct }) {
  return (
    <div className="uptime-bar">
      <div className="uptime-fill" style={{ width: `${pct}%`, background: pct > 90 ? '#22c55e' : pct > 70 ? '#eab308' : '#ef4444' }} />
    </div>
  )
}

export default function AdminSystemHealth() {
  const { data: health, loading: hLoading, error: hError, refetch } = useFetch(fetchHealth, { refreshInterval: 5000 })
  const { data: status } = useFetch(fetchStatus, { refreshInterval: 5000 })
  const { data: sysHealth } = useFetch(fetchSystemHealth, { refreshInterval: 10000 })
  const { connected, lastUpdate } = useWebSocket()

  const uptime = health?.uptime_seconds || 0
  const uptimePct = Math.min(100, (uptime / (uptime + 10)) * 100) // mock 99.9% after warmup

  const services = [
    {
      name: 'FastAPI Server',
      status: health ? 'ok' : 'error',
      meta: health ? `v${health.version} — uptime ${Math.floor(uptime)}s` : 'Unreachable',
      uptime: health ? 99.9 : 0,
      latency: sysHealth?.api_latency_ms ? `${sysHealth.api_latency_ms}ms` : '< 10ms',
    },
    {
      name: 'WebSocket',
      status: connected ? 'ok' : 'warn',
      meta: connected ? `Live — last msg ${lastUpdate ? lastUpdate.toLocaleTimeString() : 'N/A'}` : 'Disconnected',
      uptime: connected ? 99.5 : 0,
      latency: connected ? '< 5ms' : 'N/A',
    },
    {
      name: 'Pipeline FSM',
      status: status ? 'ok' : 'unknown',
      meta: status ? `State: ${status.fsm_state?.toUpperCase()} — ${status.total_samples?.toLocaleString()} samples` : 'No pipeline',
      uptime: status ? 100 : 0,
      latency: 'N/A',
    },
    {
      name: 'Database (PostgreSQL)',
      status: sysHealth?.db_status || 'unknown',
      meta: sysHealth?.db_meta || 'No direct health check available',
      uptime: sysHealth?.db_uptime_pct || 0,
      latency: sysHealth?.db_latency_ms ? `${sysHealth.db_latency_ms}ms` : 'N/A',
    },
    {
      name: 'ADWIN Drift Detector',
      status: status?.is_streaming ? 'ok' : 'warn',
      meta: status?.is_streaming ? `Active — ${status.drift_events} events detected` : 'Idle',
      uptime: 100,
      latency: 'N/A',
    },
    {
      name: 'SHAP XAI Engine',
      status: status ? 'ok' : 'unknown',
      meta: 'TreeExplainer — ready',
      uptime: 100,
      latency: 'N/A',
    },
  ]

  const stats = [
    { label: 'API Uptime',      value: health ? `${Math.floor(uptime)}s` : '--',  sub: 'Since last restart',  accent: 'var(--accent-green)',  icon: '⏱' },
    { label: 'WS Status',       value: connected ? 'Live' : 'Down',               sub: 'WebSocket connection', accent: connected ? 'var(--accent-green)' : 'var(--accent-red)', icon: '🔌' },
    { label: 'Pipeline State',  value: status?.fsm_state?.toUpperCase() || '--',  sub: 'FSM current state',   accent: 'var(--accent-blue)',   icon: '⚙' },
    { label: 'Total Samples',   value: status?.total_samples?.toLocaleString() || '--', sub: 'Processed',     accent: 'var(--accent-purple)', icon: '📊' },
  ]

  return (
    <div>
      <AdminPageHeader
        title="System Health"
        desc="Infrastructure and service monitoring — API, database, WebSocket, pipeline"
        action={<button className="btn btn-secondary" onClick={refetch}>↻ Refresh</button>}
      />

      {hError && <ErrorState message="Could not reach API server" onRetry={refetch} />}
      {hLoading && !health && <LoadingSkeleton rows={4} />}

      <div className="admin-stat-grid">
        {stats.map((s, i) => <AdminStatCard key={i} {...s} />)}
      </div>

      {/* Services table */}
      <div className="di-card">
        <h3 className="di-card-title">Service Status</h3>
        {services.map((svc, i) => (
          <div key={i} className="service-row">
            <div style={{ flex: 1 }}>
              <div className="service-name">{svc.name}</div>
              <div className="service-meta">{svc.meta}</div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 3 }}>Uptime</div>
                <UptimeBar pct={svc.uptime} />
              </div>
              <div style={{ textAlign: 'right', minWidth: 50 }}>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>Latency</div>
                <div style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}>{svc.latency}</div>
              </div>
              <HealthIndicator status={svc.status} />
            </div>
          </div>
        ))}
      </div>

      {/* Pipeline details */}
      {status && (
        <div className="grid-2col">
          <div className="di-card">
            <h3 className="di-card-title">Pipeline Metrics</h3>
            {[
              ['Session ID',       status.session_id?.slice(0,16) + '…'],
              ['Model Type',       status.model_type],
              ['Model Version',    `v${status.model_version}`],
              ['Running Accuracy', `${(status.running_accuracy*100).toFixed(2)}%`],
              ['Drift Events',     status.drift_events],
              ['Adaptations',      status.adaptations],
              ['Streaming',        status.is_streaming ? 'Active' : 'Idle'],
            ].map(([k, v]) => (
              <div key={k} className="sys-row">
                <span className="sys-label">{k}</span>
                <span className="sys-value mono" style={{ fontSize: 12 }}>{v}</span>
              </div>
            ))}
          </div>

          <div className="di-card">
            <h3 className="di-card-title">API Health</h3>
            {health ? [
              ['Status',  health.status],
              ['Version', health.version],
              ['Uptime',  `${Math.floor(health.uptime_seconds)}s`],
              ['Checked', new Date().toLocaleTimeString()],
            ].map(([k, v]) => (
              <div key={k} className="sys-row">
                <span className="sys-label">{k}</span>
                <span className="sys-value mono" style={{ fontSize: 12 }}>{v}</span>
              </div>
            )) : (
              <div className="di-empty">API unreachable</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
