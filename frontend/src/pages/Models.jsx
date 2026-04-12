import { useState } from 'react'
import { fetchModelInfo, fetchModelHistory, fetchMetricsHistory, fetchDriftEvents } from '../services/api'
import useFetch from '../hooks/useFetch'
import { formatTimestamp, formatDuration, formatSampleIndex } from '../utils/formatters'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

const Skeleton = () => (
  <div className="animate-pulse space-y-3 p-4">
    <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
    <div className="h-32 bg-gray-200 dark:bg-gray-700 rounded"></div>
  </div>
)

const EmptyChart = () => (
  <div className="flex flex-col items-center justify-center py-12 text-gray-400 dark:text-gray-600">
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none"><path d="M4 20 L10 12 L16 16 L22 8 L28 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
    <p className="mt-2 text-sm">Waiting for data...</p>
  </div>
)

export default function Models() {
  const [page, setPage] = useState(0)
  const pageSize = 20

  const { data: modelInfo, loading: miLoading } = useFetch(fetchModelInfo, { refreshInterval: 10000 })
  const { data: modelHistData, loading: mhLoading, error: mhError, refetch: refetchHistory } = useFetch(fetchModelHistory, { refreshInterval: 10000 })
  const { data: metricsHistory } = useFetch(fetchMetricsHistory, { refreshInterval: 10000 })
  const { data: driftData } = useFetch(fetchDriftEvents, { refreshInterval: 10000 })

  const history = modelHistData?.history || []
  const histMeta = { total: modelHistData?.total || 0, successful: modelHistData?.successful || 0 }
  const driftEvents = driftData?.events || []

  if ((miLoading || mhLoading) && !modelInfo && !modelHistData) return (
    <div className="page-models">
      <div className="page-header"><h2 className="page-title">Models</h2></div>
      <Skeleton />
    </div>
  )

  const chartData = metricsHistory?.timestamps?.map((t, i) => ({
    sample: t,
    accuracy: +(metricsHistory.accuracy[i] * 100).toFixed(2),
    error: +(metricsHistory.error[i] * 100).toFixed(2),
  })) || []

  const driftSamples = driftEvents.map(e => e.sample_idx)
  const tooltipStyle = { background: '#1e1e2a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 11, color: '#f0f0f5' }

  const totalPages = Math.ceil(history.length / pageSize)
  const paginatedHistory = history.slice(page * pageSize, (page + 1) * pageSize)
  const failed = histMeta.total - histMeta.successful
  const successRate = histMeta.total > 0 ? ((histMeta.successful / histMeta.total) * 100).toFixed(0) : 0
  const successfulVersions = history.filter(h => h.success)

  return (
    <div className="page-models">
      <div className="page-header">
        <h2 className="page-title">Models</h2>
        <p className="page-desc">Model performance, version history, and adaptation records</p>
      </div>

      <div className="drift-stats-row">
        <div className="drift-stat-tile"><span className="drift-stat-label">Total Attempts</span><span className="drift-stat-value">{histMeta.total}</span></div>
        <div className="drift-stat-tile"><span className="drift-stat-label">Successful</span><span className="drift-stat-value text-green">{histMeta.successful}</span></div>
        <div className="drift-stat-tile"><span className="drift-stat-label">Failed</span><span className="drift-stat-value text-red">{failed}</span></div>
        <div className="drift-stat-tile"><span className="drift-stat-label">Success Rate</span><span className="drift-stat-value">{successRate}%</span></div>
      </div>

      <div className="di-card">
        <h3 className="di-card-title">
          Current Model
          <span className="di-badge" style={{ background: 'rgba(34,197,94,0.1)', color: '#22c55e' }}>{modelInfo?.status || 'active'}</span>
        </h3>
        <div className="model-info-grid">
          <div className="model-stat"><span className="model-stat-label">Type</span><span className="model-stat-value">{modelInfo?.model_type || '--'}</span></div>
          <div className="model-stat"><span className="model-stat-label">Version</span><span className="model-stat-value accent">v{modelInfo?.version || 0}</span></div>
          <div className="model-stat"><span className="model-stat-label">Features</span><span className="model-stat-value">{modelInfo?.n_features || 0}</span></div>
          <div className="model-stat">
            <span className="model-stat-label">Feature Names</span>
            <div className="feature-tags">
              {(modelInfo?.feature_names || []).map((f, i) => <span key={i} className="feature-tag">{f}</span>)}
            </div>
          </div>
        </div>
      </div>

      <div className="grid-2col">
        <div className="di-card">
          <h3 className="di-card-title">Accuracy Over Time <span className="di-badge-sm">{chartData.length} points</span></h3>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="sample" stroke="rgba(255,255,255,0.3)" fontSize={11} tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v} />
                <YAxis stroke="rgba(255,255,255,0.3)" fontSize={11} domain={[0, 100]} tickFormatter={v => `${v}%`} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="accuracy" stroke="#22c55e" strokeWidth={2} dot={false} name="Accuracy" />
                {driftSamples.map((s, i) => <ReferenceLine key={i} x={s} stroke="#ef4444" strokeDasharray="4 4" strokeWidth={1.5} />)}
              </LineChart>
            </ResponsiveContainer>
          ) : <EmptyChart />}
        </div>
        <div className="di-card">
          <h3 className="di-card-title">Error Rate Over Time</h3>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="sample" stroke="rgba(255,255,255,0.3)" fontSize={11} tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v} />
                <YAxis stroke="rgba(255,255,255,0.3)" fontSize={11} domain={[0, 100]} tickFormatter={v => `${v}%`} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="error" stroke="#f97316" strokeWidth={2} dot={false} name="Error Rate" />
                {driftSamples.map((s, i) => <ReferenceLine key={i} x={s} stroke="#ef4444" strokeDasharray="4 4" strokeWidth={1.5} />)}
              </LineChart>
            </ResponsiveContainer>
          ) : <EmptyChart />}
        </div>
      </div>

      {successfulVersions.length > 0 && (
        <div className="di-card">
          <h3 className="di-card-title">Model Version Timeline</h3>
          <div className="version-timeline">
            <div className="version-node version-node-initial"><div className="version-dot" /><span className="version-label">v1</span><span className="version-meta">Initial</span></div>
            {history.map((h, i) => (
              <div key={i} className="version-segment">
                <div className="version-line" />
                <div className={`version-node ${h.success ? 'version-node-success' : 'version-node-fail'}`}>
                  <div className="version-dot" />
                  <span className="version-label">v{h.model_version || i + 2}</span>
                  <span className={`version-strategy ${h.success ? 'success' : 'fail'}`}>{h.strategy?.replace(/_/g, ' ')}</span>
                  <span className={`version-badge ${h.success ? 'success' : 'fail'}`}>{h.success ? 'OK' : 'FAIL'}</span>
                  <span className="version-meta mono">{(h.new_accuracy * 100).toFixed(1)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="di-card">
        <h3 className="di-card-title">Adaptation History <span className="di-badge-sm">{history.length} total</span></h3>
        {mhError && !history.length ? (
          <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg text-red-700 dark:text-red-400 text-sm">
            <span>Failed to load data</span>
            <button onClick={refetchHistory} className="ml-auto underline">Retry</button>
          </div>
        ) : history.length > 0 ? (
          <>
            <div className="di-table-wrap">
              <table className="di-table">
                <thead>
                  <tr><th>Version</th><th>Strategy</th><th>Result</th><th>Acc Before</th><th>Acc After</th><th>Δ Acc</th><th>Samples</th><th>Duration</th><th>Threshold</th><th>Rollback</th><th>Reason</th><th>Timestamp</th></tr>
                </thead>
                <tbody>
                  {paginatedHistory.map((h, i) => {
                    const delta = ((h.new_accuracy - h.old_accuracy) * 100).toFixed(2)
                    const isRollback = h.rollback === true
                    return (
                      <tr key={i} className={h.success ? 'row-success' : 'row-fail'}>
                        <td className="mono">v{h.model_version || '--'}</td>
                        <td>{h.strategy?.replace(/_/g, ' ')}</td>
                        <td>
                          <span className={`table-badge ${h.success ? 'success' : 'fail'}`}>
                            {h.success ? 'Deployed' : 'Rejected'}
                          </span>
                        </td>
                        <td className="mono">{(h.old_accuracy * 100).toFixed(2)}%</td>
                        <td className="mono">{(h.new_accuracy * 100).toFixed(2)}%</td>
                        <td className={`mono ${parseFloat(delta) >= 0 ? 'text-green' : 'text-red'}`}>{delta > 0 ? '+' : ''}{delta}%</td>
                        <td className="mono">{formatSampleIndex(h.n_samples_used)}</td>
                        <td className="mono">{formatDuration(h.duration_ms) || `${h.duration_seconds?.toFixed(2)}s`}</td>
                        <td className="mono">{h.threshold}</td>
                        <td>
                          {isRollback && (
                            <span className="table-badge fail">ROLLBACK</span>
                          )}
                        </td>
                        <td className="text-muted">{h.reason || '--'}</td>
                        <td className="text-muted">{formatTimestamp(h.timestamp)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            {totalPages > 1 && (
              <div className="pagination">
                <button className="btn-sm" disabled={page === 0} onClick={() => setPage(p => p - 1)}>← Prev</button>
                <span className="pagination-info">Page {page + 1} of {totalPages}</span>
                <button className="btn-sm" disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)}>Next →</button>
              </div>
            )}
          </>
        ) : (
          <div className="flex flex-col items-center justify-center py-12 text-gray-400 dark:text-gray-600">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none"><rect x="4" y="4" width="20" height="20" rx="2" stroke="currentColor" strokeWidth="1.5"/><path d="M9 14h10M9 10h10M9 18h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
            <p className="mt-2 text-sm">No adaptations yet</p>
          </div>
        )}
      </div>
    </div>
  )
}
