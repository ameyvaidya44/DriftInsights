import { useState } from 'react'
import { fetchStatus, fetchHealth, fetchModelHistory, fetchFsmStatus, startPipeline } from '../services/api'
import useFetch from '../hooks/useFetch'

const CONFIG = {
  data: { dataset: 'SEA', n_samples: 20000, drift_position: 17500, split: '60/20/20', noise: 0.1 },
  model: { type: 'xgboost', n_estimators: 100, max_depth: 6, learning_rate: 0.1 },
  drift: { method: 'ADWIN', adwin_delta: 0.002, warmup_samples: 100, min_gap: 200 },
  xai: { max_samples: 200, top_k_features: 5, window_size: 200 },
  decision: { minor: 0.05, moderate: 0.15, severe: 0.30, cooldown_samples: 500 },
  adaptation: { decay_factor: 0.95, incremental_window: 1000, full_retrain_min: 2000, accuracy_threshold: 0.75, tolerance: 0.05 },
}

export default function Pipeline() {
  const [starting, setStarting] = useState(false)

  const { data: status, refetch: refetchStatus } = useFetch(fetchStatus, { refreshInterval: 5000 })
  const { data: health } = useFetch(fetchHealth, { refreshInterval: 10000 })
  const { data: modelHistData } = useFetch(fetchModelHistory, { refreshInterval: 5000 })
  const { data: fsm } = useFetch(fetchFsmStatus, { refreshInterval: 5000 })

  const histMeta = { total: modelHistData?.total || 0, successful: modelHistData?.successful || 0 }
  const fsmState = fsm?.state || status?.fsm_state || 'idle'
  const fsmColors = { idle: '#22c55e', adapting: '#f97316', cooldown: '#eab308' }

  const handleStart = async () => {
    setStarting(true)
    try { await startPipeline() } catch {}
    setStarting(false)
    refetchStatus()
  }

  const configSections = [
    { title: 'Data', items: Object.entries(CONFIG.data).map(([k, v]) => [k.replace(/_/g, ' '), String(v)]) },
    { title: 'Model', items: Object.entries(CONFIG.model).map(([k, v]) => [k.replace(/_/g, ' '), String(v)]) },
    { title: 'Drift Detection', items: Object.entries(CONFIG.drift).map(([k, v]) => [k.replace(/_/g, ' '), String(v)]) },
    { title: 'XAI / SHAP', items: Object.entries(CONFIG.xai).map(([k, v]) => [k.replace(/_/g, ' '), String(v)]) },
    { title: 'Decision Thresholds', items: Object.entries(CONFIG.decision).map(([k, v]) => [k.replace(/_/g, ' '), String(v)]) },
    { title: 'Adaptation', items: Object.entries(CONFIG.adaptation).map(([k, v]) => [k.replace(/_/g, ' '), String(v)]) },
  ]

  return (
    <div className="page-pipeline">
      <div className="page-header">
        <h2 className="page-title">Pipeline</h2>
        <p className="page-desc">FSM state, system configuration, and pipeline controls</p>
      </div>

      <div className="grid-2col">
        <div className="di-card">
          <h3 className="di-card-title">FSM State</h3>
          <div className="fsm-panel">
            <div className="fsm-current">
              <span className="fsm-current-badge" style={{ color: fsmColors[fsmState], background: `${fsmColors[fsmState]}15`, borderColor: `${fsmColors[fsmState]}30` }}>
                {fsmState.toUpperCase()}
              </span>
            </div>
            <div className="fsm-diagram">
              {['idle', 'adapting', 'cooldown'].map(s => (
                <div key={s} className={`fsm-node ${fsmState === s ? 'fsm-node-active' : ''}`} style={{ borderColor: fsmColors[s] }}>
                  <span className="fsm-node-dot" style={{ background: fsmState === s ? fsmColors[s] : 'var(--text-muted)' }} />
                  <span>{s.toUpperCase()}</span>
                </div>
              ))}
            </div>
            <div className="fsm-arrows"><span>IDLE → ADAPTING → COOLDOWN → IDLE</span></div>
            {fsm && (
              <div className="fsm-details">
                <div className="sys-row"><span className="sys-label">Adaptations</span><span className="sys-value mono">{fsm.adaptation_count}</span></div>
                <div className="sys-row"><span className="sys-label">Cooldown Samples</span><span className="sys-value mono">{fsm.cooldown_samples}</span></div>
              </div>
            )}
          </div>
        </div>

        <div className="di-card">
          <h3 className="di-card-title">System Status</h3>
          <div className="sys-status-grid">
            <div className="sys-row"><span className="sys-label">Session ID</span><span className="sys-value mono">{status?.session_id || '--'}</span></div>
            <div className="sys-row"><span className="sys-label">Pipeline</span><span className={`sys-value ${status?.is_streaming ? 'text-green' : ''}`}>{status?.is_streaming ? 'Running' : 'Stopped'}</span></div>
            <div className="sys-row"><span className="sys-label">Model Version</span><span className="sys-value accent">v{status?.model_version || 1}</span></div>
            <div className="sys-row"><span className="sys-label">Total Drift Events</span><span className="sys-value">{status?.drift_events || 0}</span></div>
            <div className="sys-row"><span className="sys-label">Adaptations</span><span className="sys-value"><span className="text-green">{histMeta.successful}</span> / <span className="text-red">{histMeta.total - histMeta.successful}</span></span></div>
            <div className="sys-row"><span className="sys-label">Total Samples</span><span className="sys-value mono">{status?.total_samples?.toLocaleString() || 0}</span></div>
            {health && (
              <>
                <div className="sys-row"><span className="sys-label">API Status</span><span className="sys-value text-green">{health.status}</span></div>
                <div className="sys-row"><span className="sys-label">Uptime</span><span className="sys-value mono">{Math.floor(health.uptime_seconds)}s</span></div>
              </>
            )}
          </div>
          <div className="sys-controls">
            <button className="btn btn-primary" onClick={handleStart} disabled={starting}>
              {starting ? <><span className="spinner" /> Starting...</> : 'Start Pipeline'}
            </button>
            <button className="btn btn-secondary" onClick={refetchStatus}>Refresh All Data</button>
          </div>
        </div>
      </div>

      <div className="di-card">
        <h3 className="di-card-title">Configuration <span className="di-badge-sm">Read-only from default.yaml</span></h3>
        <div className="config-sections">
          {configSections.map((sec, i) => (
            <div key={i} className="config-section">
              <h4 className="config-section-title">{sec.title}</h4>
              <div className="config-grid">
                {sec.items.map(([key, val], j) => (
                  <div key={j} className="config-item">
                    <span className="config-key">{key}</span>
                    <span className="config-val mono">{val}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
