import { useState, useEffect, useRef } from 'react'
import { useSearchParams } from 'react-router-dom'
import { fetchXaiLatest, fetchXaiHistory, fetchXaiSummary, fetchModelHistory, classifySeverity, severityColor } from '../services/api'
import useFetch from '../hooks/useFetch'
import useWebSocket from '../hooks/useWebSocket'
import { formatNumber } from '../utils/formatters'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const Skeleton = () => (
  <div className="animate-pulse space-y-3 p-4">
    <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
    <div className="h-32 bg-gray-200 dark:bg-gray-700 rounded"></div>
  </div>
)

const EmptyChart = ({ label }) => (
  <div className="flex flex-col items-center justify-center py-12 text-gray-400 dark:text-gray-600">
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none"><circle cx="14" cy="14" r="10" stroke="currentColor" strokeWidth="1.5"/><path d="M10 14h8M14 10v8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
    <p className="mt-2 text-sm">{label}</p>
  </div>
)

export default function Explainability() {
  const [expandedEvent, setExpandedEvent] = useState(null)
  const [searchParams] = useSearchParams()
  const eventRefs = useRef({})
  const { subscribe } = useWebSocket()

  const { data: latest, loading: latestLoading, error: latestError, refetch: refetchLatest } = useFetch(fetchXaiLatest, { refreshInterval: 10000 })
  const { data: histData, refetch: refetchHist } = useFetch(fetchXaiHistory, { refreshInterval: 10000 })
  const { data: summary } = useFetch(fetchXaiSummary, { refreshInterval: 10000 })
  const { data: modelHistData } = useFetch(fetchModelHistory, { refreshInterval: 10000 })

  // Refresh on xai_complete WS event
  useEffect(() => {
    return subscribe('xai_complete', () => { refetchLatest(); refetchHist() })
  }, [subscribe, refetchLatest, refetchHist])

  const xaiHistory = histData?.explanations || []
  const modelHistory = modelHistData?.history || []

  useEffect(() => {
    const eventParam = searchParams.get('event')
    if (eventParam !== null) {
      const idx = parseInt(eventParam)
      setExpandedEvent(idx)
      setTimeout(() => {
        eventRefs.current[idx]?.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }, 500)
    }
  }, [searchParams, xaiHistory])

  const tooltipStyle = { background: '#1e1e2a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 11, color: '#f0f0f5' }
  const dirColor = (dir) => dir === 'increase' ? '#ef4444' : '#06b6d4'

  const hasData = latest && latest.top_features && latest.top_features.length > 0
  const shapData = hasData ? latest.top_features.map(f => ({
    feature: f.feature, pre: f.shap_before, post: f.shap_after, delta: f.delta, direction: f.direction,
    mean_before: f.mean_before, mean_after: f.mean_after, std_before: f.std_before, std_after: f.std_after,
  })) : []
  const ranked = [...shapData].sort((a, b) => b.delta - a.delta)
  const cumulativeData = summary?.top_drifting_features?.map(f => ({ feature: f.feature, cumulative_delta: f.cumulative_delta })) || []
  const latestAdaptation = modelHistory.length > 0 ? modelHistory[modelHistory.length - 1] : null

  if (latestLoading && !latest) return (
    <div className="page-xai">
      <div className="page-header"><h2 className="page-title">Explainability</h2></div>
      <Skeleton />
    </div>
  )

  if (latestError && !latest) return (
    <div className="page-xai">
      <div className="page-header"><h2 className="page-title">Explainability</h2></div>
      <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg text-red-700 dark:text-red-400 text-sm m-4">
        <span>Failed to load XAI data</span>
        <button onClick={refetchLatest} className="ml-auto underline">Retry</button>
      </div>
    </div>
  )

  return (
    <div className="page-xai">
      <div className="page-header">
        <h2 className="page-title">Explainability</h2>
        <p className="page-desc">SHAP-based drift diagnosis — real XAI analysis from the pipeline</p>
      </div>

      {!hasData && (
        <div className="di-card di-card-info">
          <div className="di-info-banner">
            <span className="di-info-icon">◉</span>
            <div>
              <div className="di-info-title">No Drift Events Detected Yet</div>
              <div className="di-info-desc">XAI analysis will appear here after the first drift event is detected and analyzed by SHAP. Start the pipeline to begin streaming.</div>
            </div>
          </div>
        </div>
      )}

      {hasData && (
        <div className="di-card">
          <h3 className="di-card-title">Latest Explanation</h3>
          <div className="xai-summary-row">
            <div className="xai-summary-item">
              <span className="xai-summary-label">ΔE Total</span>
              <span className="xai-summary-value mono" style={{color:'#f97316'}}>{formatNumber(latest.delta_e_total, 4)}</span>
            </div>
            <div className="xai-summary-item">
              <span className="xai-summary-label">Direction</span>
              <span className="xai-summary-value" style={{color: latest.direction === 'increase' ? '#ef4444' : latest.direction === 'decrease' ? '#06b6d4' : '#eab308'}}>
                {latest.direction?.toUpperCase()}
              </span>
            </div>
            <div className="xai-summary-item">
              <span className="xai-summary-label">Top Feature</span>
              <span className="xai-summary-value mono">{latest.top_features[0]?.feature}</span>
            </div>
            {latestAdaptation && (
              <div className="xai-summary-item">
                <span className="xai-summary-label">Strategy</span>
                <span className="xai-summary-value" style={{color:'#06b6d4'}}>{latestAdaptation.strategy?.replace(/_/g, ' ')}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {hasData && (
        <div className="grid-2col">
          <div className="di-card">
            <h3 className="di-card-title">SHAP Comparison <span className="di-badge-sm">Pre vs Post Drift</span></h3>
            {shapData.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={shapData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="feature" stroke="rgba(255,255,255,0.3)" fontSize={11} />
                  <YAxis stroke="rgba(255,255,255,0.3)" fontSize={11} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar dataKey="pre" name="Pre-Drift |SHAP|" fill="#6366f1" radius={[4, 4, 0, 0]} opacity={0.8} />
                  <Bar dataKey="post" name="Post-Drift |SHAP|" fill="#06b6d4" radius={[4, 4, 0, 0]} opacity={0.8} />
                </BarChart>
              </ResponsiveContainer>
            ) : <EmptyChart label="No SHAP data yet" />}
          </div>

          <div className="di-card">
            <h3 className="di-card-title">ΔE Feature Ranking</h3>
            {ranked.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={ranked} layout="vertical" margin={{ top: 5, right: 20, left: 60, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis type="number" stroke="rgba(255,255,255,0.3)" fontSize={11} />
                  <YAxis type="category" dataKey="feature" stroke="rgba(255,255,255,0.3)" fontSize={11} width={60} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar dataKey="delta" name="ΔE" radius={[0, 4, 4, 0]}>
                    {ranked.map((d, i) => <Cell key={i} fill={dirColor(d.direction)} opacity={0.8} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : <EmptyChart label="No ranking data yet" />}
            <div className="xai-legend">
              <span className="xai-legend-item"><span className="xai-dot" style={{ background: '#ef4444' }} /> Increase</span>
              <span className="xai-legend-item"><span className="xai-dot" style={{ background: '#06b6d4' }} /> Decrease</span>
            </div>
          </div>
        </div>
      )}

      {hasData && (
        <div className="di-card">
          <h3 className="di-card-title">Feature Behavior Analysis</h3>
          <div className="di-table-wrap">
            <table className="di-table">
              <thead>
                <tr><th>Feature</th><th>SHAP Before</th><th>SHAP After</th><th>ΔE</th><th>Direction</th><th>Mean Before</th><th>Mean After</th><th>Std Before</th><th>Std After</th></tr>
              </thead>
              <tbody>
                {ranked.map((d, i) => (
                  <tr key={i}>
                    <td className="mono">{d.feature}</td>
                    <td className="mono">{formatNumber(d.pre, 4)}</td>
                    <td className="mono">{formatNumber(d.post, 4)}</td>
                    <td className="mono" style={{ color: '#f97316' }}>{formatNumber(d.delta, 4)}</td>
                    <td>
                      <span className="table-dir-badge" style={{ color: dirColor(d.direction), background: `${dirColor(d.direction)}15` }}>
                        {d.direction === 'increase' ? '↑' : '↓'} {d.direction}
                      </span>
                    </td>
                    <td className="mono">{formatNumber(d.mean_before, 3)}</td>
                    <td className="mono">{formatNumber(d.mean_after, 3)}</td>
                    <td className="mono">{formatNumber(d.std_before, 3)}</td>
                    <td className="mono">{formatNumber(d.std_after, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {hasData && (
        <div className="di-card">
          <h3 className="di-card-title">Feature Distribution Shift</h3>
          <div className="dist-shift-grid">
            {ranked.map((d, i) => (
              <div key={i} className="dist-shift-item">
                <span className="dist-shift-feature mono">{d.feature}</span>
                <div className="dist-shift-bars">
                  <div className="dist-shift-bar-group">
                    <span className="dist-shift-label">Pre</span>
                    <div className="dist-shift-bar" style={{ width: `${Math.min(100, (d.mean_before || 0) * 10)}%`, background: '#6366f1' }} />
                    <span className="dist-shift-val mono">{formatNumber(d.mean_before, 2)} ±{formatNumber(d.std_before, 2)}</span>
                  </div>
                  <div className="dist-shift-bar-group">
                    <span className="dist-shift-label">Post</span>
                    <div className="dist-shift-bar" style={{ width: `${Math.min(100, (d.mean_after || 0) * 10)}%`, background: '#06b6d4' }} />
                    <span className="dist-shift-val mono">{formatNumber(d.mean_after, 2)} ±{formatNumber(d.std_after, 2)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {cumulativeData.length > 0 && (
        <div className="di-card">
          <h3 className="di-card-title">Most Drifting Features Across All Events <span className="di-badge-sm">{summary.events_analyzed} analyzed</span></h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={cumulativeData} layout="vertical" margin={{ top: 5, right: 20, left: 60, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis type="number" stroke="rgba(255,255,255,0.3)" fontSize={11} />
              <YAxis type="category" dataKey="feature" stroke="rgba(255,255,255,0.3)" fontSize={11} width={60} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="cumulative_delta" fill="#8b5cf6" radius={[0, 4, 4, 0]} name="Cumulative ΔE" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="di-card">
        <h3 className="di-card-title">Explanation History <span className="di-badge-sm">{xaiHistory.length} events</span></h3>
        {xaiHistory.length > 0 ? (
          <div className="xai-history-timeline">
            {xaiHistory.map((exp, i) => {
              const isExpanded = expandedEvent === i
              const sev = classifySeverity(exp.delta_e_total || exp.delta_total || 0)
              const topF = exp.top_features?.[0]
              return (
                <div key={i} ref={el => eventRefs.current[i] = el}
                  className={`xai-history-item ${isExpanded ? 'xai-history-expanded' : ''}`}
                  onClick={() => setExpandedEvent(isExpanded ? null : i)}>
                  <div className="xai-history-header">
                    <span className="xai-history-num">Event #{i + 1}</span>
                    <span className="mono" style={{color:'#f97316'}}>ΔE={(exp.delta_e_total || exp.delta_total || 0).toFixed(4)}</span>
                    <span style={{color: severityColor(sev)}}>{sev.toUpperCase()}</span>
                    {topF && <span className="text-muted">Top: {topF.feature} (Δ={topF.delta?.toFixed(3)})</span>}
                    <span style={{color: exp.direction === 'increase' ? '#ef4444' : exp.direction === 'decrease' ? '#06b6d4' : '#eab308'}}>{exp.direction}</span>
                    <span className="xai-history-expand">{isExpanded ? '▼' : '▶'}</span>
                  </div>
                  {isExpanded && exp.top_features && (
                    <div className="xai-history-detail">
                      <table className="di-table">
                        <thead><tr><th>Feature</th><th>SHAP Before</th><th>SHAP After</th><th>ΔE</th><th>Direction</th></tr></thead>
                        <tbody>
                          {exp.top_features.map((f, j) => (
                            <tr key={j}>
                              <td className="mono">{f.feature}</td>
                              <td className="mono">{formatNumber(f.shap_before, 4)}</td>
                              <td className="mono">{formatNumber(f.shap_after, 4)}</td>
                              <td className="mono" style={{color:'#f97316'}}>{formatNumber(f.delta, 4)}</td>
                              <td><span style={{color: dirColor(f.direction)}}>{f.direction === 'increase' ? '↑' : '↓'} {f.direction}</span></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-12 text-gray-400 dark:text-gray-600">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none"><circle cx="14" cy="14" r="10" stroke="currentColor" strokeWidth="1.5"/></svg>
            <p className="mt-2 text-sm">No explanations yet</p>
          </div>
        )}
      </div>
    </div>
  )
}
