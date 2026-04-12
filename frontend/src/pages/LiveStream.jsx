import { useState, useEffect, useCallback, useRef } from 'react'
import { fetchMetricsSnapshot, fetchMetricsHistory, fetchDriftEvents } from '../services/api'
import useFetch from '../hooks/useFetch'
import useWebSocket from '../hooks/useWebSocket'
import { formatSampleIndex, formatNumber, formatTime } from '../utils/formatters'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

const typeConfig = {
  metrics_update:     { color: '#22c55e', label: 'METRICS' },
  drift_detected:     { color: '#ef4444', label: 'DRIFT' },
  adaptation_complete:{ color: '#f97316', label: 'ADAPT' },
  xai_complete:       { color: '#6366f1', label: 'XAI' },
  system_event:       { color: '#555570', label: 'SYSTEM' },
}

const buildMessage = (type, data) => {
  switch (type) {
    case 'metrics_update':
      return `Accuracy: ${formatNumber(data.running_accuracy * 100)}% | Error: ${formatNumber(data.windowed_error_rate * 100)}% | Samples: ${formatSampleIndex(data.sample_index)}`
    case 'drift_detected':
      return `Drift at sample ${formatSampleIndex(data.sample_index)} — ${data.severity} severity (error: ${formatNumber(data.pre_error_rate, 2)} → ${formatNumber(data.post_error_rate, 2)})`
    case 'adaptation_complete': {
      const delta = ((data.delta_accuracy || 0) * 100).toFixed(2)
      const sign = data.delta_accuracy >= 0 ? '+' : ''
      return `Model ${data.model_version} ${data.result} via ${data.strategy?.replace(/_/g, ' ')} (${sign}${delta}% accuracy)`
    }
    case 'xai_complete':
      return `XAI: ΔE = ${formatNumber(data.delta_e_total, 3)} — top feature: ${data.top_feature} (+${formatNumber(data.top_delta_e, 3)})`
    case 'system_event':
      return data.message || data.event || 'System event'
    default:
      return JSON.stringify(data).slice(0, 100)
  }
}

const Skeleton = () => (
  <div className="animate-pulse space-y-3">
    <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
    <div className="h-32 bg-gray-200 dark:bg-gray-700 rounded"></div>
  </div>
)

export default function LiveStream() {
  const { data: snapshot, loading: snapLoading } = useFetch(fetchMetricsSnapshot, { refreshInterval: 3000 })
  const { data: histData } = useFetch(fetchMetricsHistory, { refreshInterval: 3000 })
  const { data: driftData } = useFetch(fetchDriftEvents, { refreshInterval: 3000 })
  const [eventLog, setEventLog] = useState([])
  const [sps, setSps] = useState(0)
  const prevSamplesRef = useRef(0)
  const logRef = useRef(null)
  const { subscribe, connected } = useWebSocket()

  // Compute samples/sec from snapshot changes
  useEffect(() => {
    if (snapshot) {
      const delta = snapshot.total_samples - prevSamplesRef.current
      setSps(Math.max(0, Math.round(delta / 3)))
      prevSamplesRef.current = snapshot.total_samples
    }
  }, [snapshot])

  // Subscribe to all 5 event types
  useEffect(() => {
    const types = ['metrics_update', 'drift_detected', 'adaptation_complete', 'xai_complete', 'system_event']
    const unsubs = types.map(type =>
      subscribe(type, (data) => {
        setEventLog(prev => [{
          type,
          data,
          time: new Date().toISOString(),
        }, ...prev].slice(0, 100))
      })
    )
    return () => unsubs.forEach(u => u())
  }, [subscribe])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = 0
  }, [eventLog])

  const driftEvents = driftData?.events || []
  const history = histData

  const chartData = history?.timestamps?.slice(-200).map((t, i) => {
    const offset = Math.max(0, history.timestamps.length - 200)
    return {
      sample: t,
      accuracy: +(history.accuracy[offset + i] * 100).toFixed(2),
      error: +(history.error[offset + i] * 100).toFixed(2),
    }
  }) || []

  const driftSamples = driftEvents.map(e => e.sample_idx)
  const accPct = snapshot ? (snapshot.running_accuracy * 100).toFixed(1) : 0
  const errPct = snapshot ? (snapshot.windowed_error * 100).toFixed(1) : 0
  const fillPct = snapshot?.window_fill_pct || 0

  return (
    <div className="page-stream">
      <div className="page-header">
        <h2 className="page-title">Live Stream</h2>
        <p className="page-desc">Real-time sample-by-sample monitoring</p>
      </div>

      {snapLoading && !snapshot ? <Skeleton /> : (
        <div className="stream-gauges">
          <div className="gauge-card">
            <span className="gauge-label">Running Accuracy</span>
            <div className="gauge-bar-wrap">
              <div className="gauge-bar" style={{ width: `${accPct}%`, background: 'var(--accent-green)' }} />
            </div>
            <span className="gauge-value" style={{ color: 'var(--accent-green)' }}>{accPct}%</span>
          </div>
          <div className="gauge-card">
            <span className="gauge-label">Windowed Error</span>
            <div className="gauge-bar-wrap">
              <div className="gauge-bar" style={{ width: `${errPct}%`, background: 'var(--accent-orange)' }} />
            </div>
            <span className="gauge-value" style={{ color: 'var(--accent-orange)' }}>{errPct}%</span>
          </div>
          <div className="gauge-card">
            <span className="gauge-label">Window Fill</span>
            <div className="gauge-bar-wrap">
              <div className="gauge-bar" style={{ width: `${fillPct}%`, background: 'var(--accent-blue)' }} />
            </div>
            <span className="gauge-value" style={{ color: 'var(--accent-blue)' }}>{snapshot?.window_fill || 0}/{snapshot?.window_size || 0}</span>
          </div>
          <div className="gauge-card">
            <span className="gauge-label">Samples/sec</span>
            <span className="gauge-value-lg">{sps}</span>
            <span className="gauge-sub">Total: {formatSampleIndex(snapshot?.total_samples)}</span>
          </div>
        </div>
      )}

      <div className="di-card">
        <h3 className="di-card-title">Real-time Accuracy <span className="di-badge-sm">Last 200 points</span></h3>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="sample" stroke="rgba(255,255,255,0.3)" fontSize={11} tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v} />
              <YAxis stroke="rgba(255,255,255,0.3)" fontSize={11} domain={[0, 100]} tickFormatter={v => `${v}%`} />
              <Tooltip contentStyle={{ background: '#1e1e2a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 11, color: '#f0f0f5' }} />
              <Line type="monotone" dataKey="accuracy" stroke="#22c55e" strokeWidth={2} dot={false} name="Accuracy" />
              <Line type="monotone" dataKey="error" stroke="#f97316" strokeWidth={2} dot={false} name="Error" />
              {driftSamples.map((s, i) => <ReferenceLine key={i} x={s} stroke="#ef4444" strokeDasharray="4 4" strokeWidth={1.5} />)}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex flex-col items-center justify-center py-12 text-gray-400 dark:text-gray-600">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none"><path d="M4 24 L10 14 L16 18 L22 10 L28 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
            <p className="mt-2 text-sm">Waiting for streaming data...</p>
          </div>
        )}
      </div>

      {driftEvents.length > 0 && (
        <div className="di-card">
          <h3 className="di-card-title">ADWIN Detector State</h3>
          <div className="adwin-info">
            <div className="adwin-stat">
              <span className="adwin-label">Last ADWIN Width</span>
              <span className="adwin-value mono">{driftEvents[driftEvents.length - 1]?.adwin_width || '--'}</span>
            </div>
            <div className="adwin-stat">
              <span className="adwin-label">Total Detections</span>
              <span className="adwin-value mono">{driftEvents.length}</span>
            </div>
            <div className="adwin-stat">
              <span className="adwin-label">Last Detection Sample</span>
              <span className="adwin-value mono">{formatSampleIndex(driftEvents[driftEvents.length - 1]?.sample_idx)}</span>
            </div>
          </div>
        </div>
      )}

      <div className="di-card">
        <h3 className="di-card-title">Live Event Log <span className="di-badge-sm">{eventLog.length} events</span></h3>
        <div className="event-log" ref={logRef}>
          {eventLog.length === 0 ? (
            <div className="di-empty">
              {connected ? 'Waiting for WebSocket events...' : 'WebSocket disconnected — events will appear when connected.'}
            </div>
          ) : (
            eventLog.map((entry, i) => {
              const cfg = typeConfig[entry.type] || typeConfig.system_event
              return (
                <div key={i} className="event-log-item">
                  <span className="event-log-time mono">{formatTime(entry.time)}</span>
                  <span className="event-log-badge" style={{ color: cfg.color, background: `${cfg.color}15`, borderColor: `${cfg.color}30` }}>
                    {cfg.label}
                  </span>
                  <span className="event-log-msg">{buildMessage(entry.type, entry.data)}</span>
                </div>
              )
            })
          )}
        </div>
      </div>
    </div>
  )
}
