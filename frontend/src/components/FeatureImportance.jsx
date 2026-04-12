import { useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import useFetch from '../hooks/useFetch'
import useWebSocket from '../hooks/useWebSocket'
import { fetchXaiLatest } from '../services/api'

export default function FeatureImportance() {
  const { data: xaiData, loading, error, refetch } = useFetch(fetchXaiLatest, { refreshInterval: 5000 })
  const { subscribe } = useWebSocket()

  // Refresh immediately when xai_complete arrives
  useEffect(() => {
    return subscribe('xai_complete', () => refetch())
  }, [subscribe, refetch])

  if (loading && !xaiData) {
    return (
      <div className="chart-card">
        <h3 className="chart-title">Feature Importance Shift</h3>
        <div className="animate-pulse space-y-3 p-4">
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
          <div className="h-32 bg-gray-200 dark:bg-gray-700 rounded"></div>
        </div>
      </div>
    )
  }

  if (error && !xaiData) {
    return (
      <div className="chart-card">
        <h3 className="chart-title">Feature Importance Shift</h3>
        <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg text-red-700 dark:text-red-400 text-sm">
          <span>Failed to load data</span>
          <button onClick={refetch} className="ml-auto underline">Retry</button>
        </div>
      </div>
    )
  }

  if (!xaiData || !xaiData.top_features || xaiData.top_features.length === 0) {
    return (
      <div className="chart-card">
        <h3 className="chart-title">
          Feature Importance Shift
          <span className="chart-badge delta-badge">SHAP-based</span>
        </h3>
        <div className="flex flex-col items-center justify-center py-12 text-gray-400 dark:text-gray-600">
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
            <rect x="4" y="20" width="4" height="8" rx="1" fill="currentColor" opacity="0.4"/>
            <rect x="12" y="14" width="4" height="14" rx="1" fill="currentColor" opacity="0.6"/>
            <rect x="20" y="8" width="4" height="20" rx="1" fill="currentColor" opacity="0.8"/>
            <rect x="28" y="4" width="4" height="24" rx="1" fill="currentColor"/>
          </svg>
          <p className="mt-2 text-sm">No drift events yet</p>
        </div>
      </div>
    )
  }

  const featureData = xaiData.top_features.map(f => ({
    name: f.feature,
    pre: f.shap_before,
    post: f.shap_after,
    delta: f.delta,
    direction: f.direction,
  }))

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null
    const d = payload[0]?.payload
    return (
      <div className="chart-tooltip">
        <p className="tooltip-label">{d.name}</p>
        <p style={{ color: '#6366f1' }}>Pre-drift: {d.pre?.toFixed(4)}</p>
        <p style={{ color: '#06b6d4' }}>Post-drift: {d.post?.toFixed(4)}</p>
        <p style={{ color: '#f97316' }}>ΔE: {d.delta?.toFixed(4)}</p>
        <p style={{ color: d.direction === 'increase' ? '#ef4444' : '#06b6d4' }}>
          Direction: {d.direction}
        </p>
      </div>
    )
  }

  return (
    <div className="chart-card">
      <h3 className="chart-title">
        Feature Importance Shift
        <span className="chart-badge delta-badge">SHAP-based</span>
        <span className="chart-badge" style={{ marginLeft: 4 }}>ΔE={xaiData.delta_e_total?.toFixed(3)}</span>
      </h3>
      <div className="chart-container">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={featureData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="name" stroke="rgba(255,255,255,0.3)" fontSize={11} />
            <YAxis stroke="rgba(255,255,255,0.3)" fontSize={11} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="pre" name="Pre-drift |SHAP|" radius={[4, 4, 0, 0]} opacity={0.8}>
              {featureData.map((_, i) => <Cell key={i} fill="#6366f1" />)}
            </Bar>
            <Bar dataKey="post" name="Post-drift |SHAP|" radius={[4, 4, 0, 0]} opacity={0.8}>
              {featureData.map((_, i) => <Cell key={i} fill="#06b6d4" />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
