import { useNavigate } from 'react-router-dom'
import { fetchDriftEvents, classifySeverity, severityColor } from '../services/api'
import useFetch from '../hooks/useFetch'
import { formatSampleIndex, formatTimestamp } from '../utils/formatters'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, PieChart, Pie } from 'recharts'

const Skeleton = () => (
  <div className="animate-pulse space-y-3 p-4">
    <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
    <div className="h-32 bg-gray-200 dark:bg-gray-700 rounded"></div>
  </div>
)

export default function DriftAnalytics() {
  const navigate = useNavigate()
  const { data, loading, error, refetch } = useFetch(fetchDriftEvents, { refreshInterval: 5000 })

  const events = data?.events || []

  if (loading && events.length === 0) return (
    <div className="page-drift">
      <div className="page-header"><h2 className="page-title">Drift Analytics</h2></div>
      <Skeleton />
    </div>
  )

  if (error && events.length === 0) return (
    <div className="page-drift">
      <div className="page-header"><h2 className="page-title">Drift Analytics</h2></div>
      <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg text-red-700 dark:text-red-400 text-sm m-4">
        <span>Failed to load drift events</span>
        <button onClick={refetch} className="ml-auto underline">Retry</button>
      </div>
    </div>
  )

  const classified = events.map(e => ({ ...e, severity: classifySeverity(e.severity_estimate) }))
  const sevCounts = { minor: 0, moderate: 0, severe: 0 }
  classified.forEach(e => { if (sevCounts[e.severity] !== undefined) sevCounts[e.severity]++ })
  const pieData = [
    { name: 'Minor', value: sevCounts.minor, fill: '#eab308' },
    { name: 'Moderate', value: sevCounts.moderate, fill: '#f97316' },
    { name: 'Severe', value: sevCounts.severe, fill: '#ef4444' },
  ].filter(d => d.value > 0)

  const freqMap = {}
  events.forEach(e => {
    const bucket = Math.floor(e.sample_idx / 1000) * 1000
    const key = `${(bucket/1000).toFixed(0)}k-${((bucket+1000)/1000).toFixed(0)}k`
    freqMap[key] = (freqMap[key] || 0) + 1
  })
  const freqData = Object.entries(freqMap).map(([range, count]) => ({ range, count }))
  const shiftData = events.map((e, i) => ({
    idx: i + 1,
    pre: +(e.pre_error_rate * 100).toFixed(2),
    post: +(e.post_error_rate * 100).toFixed(2),
  }))

  const avgSeverity = events.length > 0 ? (events.reduce((s, e) => s + e.severity_estimate, 0) / events.length) : 0
  const maxJump = events.length > 0 ? Math.max(...events.map(e => Math.abs(e.post_error_rate - e.pre_error_rate))) : 0
  const avgAdwinWidth = events.length > 0 ? Math.round(events.reduce((s, e) => s + (e.adwin_width || 0), 0) / events.length) : 0
  const tooltipStyle = { background: '#1e1e2a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 11, color: '#f0f0f5' }

  const EmptyChart = ({ label }) => (
    <div className="flex flex-col items-center justify-center py-12 text-gray-400 dark:text-gray-600">
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none"><circle cx="14" cy="14" r="10" stroke="currentColor" strokeWidth="1.5"/><path d="M10 14h8M14 10v8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
      <p className="mt-2 text-sm">{label}</p>
    </div>
  )

  return (
    <div className="page-drift">
      <div className="page-header">
        <h2 className="page-title">Drift Analytics</h2>
        <p className="page-desc">Analysis of concept drift events detected by ADWIN</p>
      </div>

      <div className="drift-stats-row">
        <div className="drift-stat-tile"><span className="drift-stat-label">Total Drifts</span><span className="drift-stat-value">{events.length}</span></div>
        <div className="drift-stat-tile"><span className="drift-stat-label">Avg Severity</span><span className="drift-stat-value mono">{avgSeverity.toFixed(3)}</span></div>
        <div className="drift-stat-tile"><span className="drift-stat-label">Max Error Jump</span><span className="drift-stat-value mono">{(maxJump * 100).toFixed(1)}%</span></div>
        <div className="drift-stat-tile"><span className="drift-stat-label">Avg ADWIN Width</span><span className="drift-stat-value mono">{avgAdwinWidth}</span></div>
      </div>

      <div className="grid-3col">
        <div className="di-card">
          <h3 className="di-card-title">Severity Distribution</h3>
          {pieData.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" innerRadius={40} outerRadius={70} dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {pieData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} />
              </PieChart>
            </ResponsiveContainer>
          ) : <EmptyChart label="No drift events yet" />}
        </div>

        <div className="di-card">
          <h3 className="di-card-title">Drift Frequency</h3>
          {freqData.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={freqData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="range" stroke="rgba(255,255,255,0.3)" fontSize={10} />
                <YAxis stroke="rgba(255,255,255,0.3)" fontSize={11} allowDecimals={false} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} name="Events" />
              </BarChart>
            </ResponsiveContainer>
          ) : <EmptyChart label="No drift events yet" />}
        </div>

        <div className="di-card">
          <h3 className="di-card-title">Error Rate Shift</h3>
          {shiftData.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={shiftData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="idx" stroke="rgba(255,255,255,0.3)" fontSize={11} />
                <YAxis stroke="rgba(255,255,255,0.3)" fontSize={11} tickFormatter={v => `${v}%`} />
                <Tooltip contentStyle={tooltipStyle} formatter={v => `${v}%`} />
                <Bar dataKey="pre" name="Pre-Drift Error" fill="#6366f1" radius={[4, 4, 0, 0]} />
                <Bar dataKey="post" name="Post-Drift Error" fill="#ef4444" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <EmptyChart label="No drift events yet" />}
        </div>
      </div>

      <div className="di-card">
        <h3 className="di-card-title">Drift Events <span className="di-badge-sm">{events.length} total</span></h3>
        {classified.length > 0 ? (
          <div className="di-table-wrap">
            <table className="di-table">
              <thead>
                <tr><th>#</th><th>Sample Index</th><th>Severity</th><th>Pre Error</th><th>Post Error</th><th>Error Δ</th><th>ADWIN Width</th><th>Detection #</th><th>Timestamp</th><th></th></tr>
              </thead>
              <tbody>
                {classified.map((e, i) => {
                  const delta = ((e.post_error_rate - e.pre_error_rate) * 100).toFixed(2)
                  return (
                    <tr key={i}>
                      <td className="mono">{i + 1}</td>
                      <td className="mono">{formatSampleIndex(e.sample_idx)}</td>
                      <td>
                        <span className="table-severity-badge" style={{ color: severityColor(e.severity), background: `${severityColor(e.severity)}15`, borderColor: `${severityColor(e.severity)}30` }}>
                          {e.severity.toUpperCase()}
                        </span>
                      </td>
                      <td className="mono">{(e.pre_error_rate * 100).toFixed(2)}%</td>
                      <td className="mono">{(e.post_error_rate * 100).toFixed(2)}%</td>
                      <td className="mono text-red">+{delta}%</td>
                      <td className="mono">{e.adwin_width || '--'}</td>
                      <td className="mono">{e.detection_count}</td>
                      <td className="text-muted">{formatTimestamp(e.timestamp)}</td>
                      <td><button className="btn-sm" onClick={() => navigate(`/explainability?event=${i}`)}>View XAI</button></td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-12 text-gray-400 dark:text-gray-600">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none"><path d="M16 4 L19 13 L28 13 L21 19 L24 28 L16 22 L8 28 L11 19 L4 13 L13 13 Z" stroke="currentColor" strokeWidth="1.5" fill="none"/></svg>
            <p className="mt-2 text-sm">No drift events detected yet</p>
          </div>
        )}
      </div>
    </div>
  )
}
