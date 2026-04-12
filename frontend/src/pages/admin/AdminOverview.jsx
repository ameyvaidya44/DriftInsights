import { useState, useEffect } from 'react'
import AdminPageHeader from '../../components/admin/AdminPageHeader'
import AdminStatCard from '../../components/admin/AdminStatCard'
import HealthIndicator from '../../components/admin/HealthIndicator'
import { LoadingSkeleton, ErrorState } from '../../components/admin/EmptyState'
import useFetch from '../../hooks/useFetch'
import useWebSocket from '../../hooks/useWebSocket'
import { fetchStatus, fetchHealth } from '../../services/api'
import { formatTimestamp } from '../../utils/formatters'

// Seed a recent activity feed from WS events + static mock
function useActivityFeed() {
  const [feed, setFeed] = useState([
    { id: 1, msg: 'Pipeline initialized and model trained', type: 'system', time: new Date(Date.now() - 120000).toISOString() },
    { id: 2, msg: 'Admin logged in from 127.0.0.1',         type: 'auth',   time: new Date(Date.now() - 60000).toISOString()  },
  ])
  const { subscribe } = useWebSocket()

  useEffect(() => {
    const types = ['drift_detected', 'adaptation_complete', 'system_event', 'xai_complete']
    const unsubs = types.map(type =>
      subscribe(type, (data) => {
        const msgs = {
          drift_detected:     `Drift detected at sample ${data.sample_index?.toLocaleString()} — ${data.severity}`,
          adaptation_complete:`Model ${data.model_version} ${data.result} via ${data.strategy?.replace(/_/g,' ')}`,
          system_event:       data.message || data.event,
          xai_complete:       `XAI: ΔE=${data.delta_e_total?.toFixed(3)} — top: ${data.top_feature}`,
        }
        setFeed(prev => [{
          id: Date.now(), msg: msgs[type] || type, type, time: new Date().toISOString(),
        }, ...prev].slice(0, 50))
      })
    )
    return () => unsubs.forEach(u => u())
  }, [subscribe])

  return feed
}

const typeColors = {
  drift_detected:     '#ef4444',
  adaptation_complete:'#f97316',
  system_event:       '#555570',
  xai_complete:       '#6366f1',
  auth:               '#22c55e',
  system:             '#06b6d4',
}

export default function AdminOverview() {
  const { data: status, loading: sLoading, error: sError, refetch } = useFetch(fetchStatus, { refreshInterval: 5000 })
  const { data: health } = useFetch(fetchHealth, { refreshInterval: 10000 })
  const { connected } = useWebSocket()
  const feed = useActivityFeed()

  const stats = [
    { label: 'Total Samples',   value: status?.total_samples?.toLocaleString() ?? '--', sub: 'Processed',          accent: 'var(--accent-blue)',   icon: '📊' },
    { label: 'Drift Events',    value: status?.drift_events ?? '--',                    sub: 'Detected by ADWIN',  accent: 'var(--accent-red)',    icon: '⚡' },
    { label: 'Adaptations',     value: status?.adaptations ?? '--',                     sub: 'Model updates',      accent: 'var(--accent-orange)', icon: '🔄' },
    { label: 'Model Version',   value: status ? `v${status.model_version}` : '--',      sub: status?.model_type,   accent: 'var(--accent-purple)', icon: '🤖' },
  ]

  return (
    <div>
      <AdminPageHeader
        title="Admin Overview"
        desc="System health, quick statistics, and recent activity"
        action={<button className="btn btn-secondary" onClick={refetch}>↻ Refresh</button>}
      />

      {sError && !status && <ErrorState message="Failed to load pipeline status" onRetry={refetch} />}

      {sLoading && !status ? <LoadingSkeleton rows={4} /> : (
        <div className="admin-stat-grid">
          {stats.map((s, i) => <AdminStatCard key={i} {...s} />)}
        </div>
      )}

      <div className="grid-2col">
        {/* System Health */}
        <div className="di-card">
          <h3 className="di-card-title">System Health</h3>
          <div>
            {[
              { name: 'API Server',       status: health ? 'ok' : 'error',   meta: health ? `Uptime ${Math.floor(health.uptime_seconds)}s` : 'Unreachable' },
              { name: 'WebSocket',        status: connected ? 'ok' : 'warn', meta: connected ? 'Live connection active' : 'Disconnected' },
              { name: 'Pipeline FSM',     status: status ? 'ok' : 'unknown', meta: status?.fsm_state?.toUpperCase() || 'Unknown' },
              { name: 'Streaming',        status: status?.is_streaming ? 'ok' : 'warn', meta: status?.is_streaming ? 'Running' : 'Idle' },
              { name: 'Database',         status: 'unknown', meta: 'No direct check available' },
            ].map((svc, i) => (
              <div key={i} className="service-row">
                <div>
                  <div className="service-name">{svc.name}</div>
                  <div className="service-meta">{svc.meta}</div>
                </div>
                <HealthIndicator status={svc.status} />
              </div>
            ))}
          </div>
        </div>

        {/* Quick Stats */}
        <div className="di-card">
          <h3 className="di-card-title">Pipeline Details</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
            {[
              ['Session ID',      status?.session_id?.slice(0, 16) + '…' || '--'],
              ['Status',          status?.status || '--'],
              ['FSM State',       status?.fsm_state?.toUpperCase() || '--'],
              ['Running Accuracy',status ? `${(status.running_accuracy * 100).toFixed(2)}%` : '--'],
              ['API Version',     health?.version || '--'],
              ['Uptime',          health ? `${Math.floor(health.uptime_seconds)}s` : '--'],
            ].map(([k, v], i) => (
              <div key={i} className="sys-row">
                <span className="sys-label">{k}</span>
                <span className="sys-value mono" style={{ fontSize: 12 }}>{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent Activity Feed */}
      <div className="di-card">
        <h3 className="di-card-title">Recent Activity <span className="di-badge-sm">{feed.length} events</span></h3>
        <div className="activity-feed">
          {feed.length === 0 ? (
            <div className="di-empty">No activity yet — start the pipeline to see events.</div>
          ) : feed.map(item => (
            <div key={item.id} className="activity-item">
              <span className="activity-dot" style={{ background: typeColors[item.type] || '#555570' }} />
              <span className="activity-msg">{item.msg}</span>
              <span className="activity-time">{formatTimestamp(item.time)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
