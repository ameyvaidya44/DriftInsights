import { useMemo } from 'react'
import useWebSocket from '../../hooks/useWebSocket'
import useFetch from '../../hooks/useFetch'
import { classifySeverity, severityColor, fetchModelHistory, fetchXaiHistory } from '../../services/api'
import { formatNumber } from '../../utils/formatters'

function mapHistoryToFeed(history, explanations) {
  const xai = explanations || []
  return (history || []).slice().reverse().slice(0, 20).map((h, i) => {
    const matched = xai[xai.length - 1 - i] || {}
    const deltaE = matched.delta_e_total ?? 0
    return {
      id: `${h.timestamp || i}-${h.model_version || i}`,
      deltaE,
      severity: classifySeverity(deltaE),
      strategy: h.strategy,
      result: h.success ? 'deployed' : 'rejected',
      success: Boolean(h.success),
      topFeature: matched.top_features?.[0]?.feature || null,
      timestamp: h.timestamp || new Date().toISOString(),
    }
  })
}

export default function LiveDecisionFeed() {
  const { connected } = useWebSocket()
  const { data: histData } = useFetch(fetchModelHistory, { refreshInterval: 2000 })
  const { data: xaiData } = useFetch(fetchXaiHistory, { refreshInterval: 2000 })

  const feed = useMemo(
    () => mapHistoryToFeed(histData?.history, xaiData?.explanations),
    [histData, xaiData],
  )

  if (feed.length === 0) {
    return (
      <div className="di-card">
        <h3 className="di-card-title">Live Decision Feed</h3>
        <div className="di-empty">
          No adaptation decisions yet. Start the pipeline; decisions appear after ADWIN detects drift.
          {!connected && <div className="text-muted" style={{ marginTop: 8 }}>WebSocket disconnected — feed still updates from the API.</div>}
        </div>
      </div>
    )
  }

  return (
    <div className="di-card">
      <h3 className="di-card-title">Live Decision Feed <span className="di-badge-sm">{feed.length} total</span></h3>
      <div className="decision-feed">
        {feed.map((item) => (
          <div key={item.id} className="decision-feed-item">
            <div className="decision-feed-flow">
              <span className="mono" style={{ color: '#f97316' }}>ΔE={formatNumber(item.deltaE, 3)}</span>
              <span className="decision-feed-arrow">→</span>
              <span style={{ color: severityColor(item.severity) }}>{item.severity}</span>
              <span className="decision-feed-arrow">→</span>
              <span style={{ color: '#06b6d4' }}>{item.strategy?.replace(/_/g, ' ')}</span>
              <span className="decision-feed-arrow">→</span>
              <span className={item.success ? 'text-green' : 'text-red'}>
                {item.success ? 'DEPLOYED' : 'ROLLBACK'}
              </span>
            </div>
            <span className="decision-feed-time">{new Date(item.timestamp).toLocaleTimeString()}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
