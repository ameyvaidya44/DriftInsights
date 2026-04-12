import { useState, useEffect, useRef } from 'react'
import useWebSocket from '../../hooks/useWebSocket'
import { classifySeverity, severityColor } from '../../services/api'
import { formatNumber } from '../../utils/formatters'

const MATCH_WINDOW_MS = 2000

export default function LiveDecisionFeed() {
  const [feed, setFeed] = useState([])
  const pendingXaiRef = useRef(null)
  const { subscribe } = useWebSocket()

  useEffect(() => {
    const unsubs = [
      subscribe('xai_complete', (data) => {
        pendingXaiRef.current = { data, arrivedAt: Date.now() }
      }),
      subscribe('adaptation_complete', (adaptData) => {
        const pending = pendingXaiRef.current
        const xaiData = pending && (Date.now() - pending.arrivedAt) < MATCH_WINDOW_MS
          ? pending.data
          : null
        if (xaiData) pendingXaiRef.current = null

        const deltaE = xaiData?.delta_e_total ?? 0
        const severity = classifySeverity(deltaE)

        setFeed(prev => [{
          id: Date.now(),
          deltaE,
          severity,
          strategy: adaptData.strategy,
          result: adaptData.result,
          success: adaptData.result === 'deployed',
          topFeature: xaiData?.top_feature || null,
          timestamp: adaptData.timestamp || new Date().toISOString(),
        }, ...prev].slice(0, 20))
      }),
    ]
    return () => unsubs.forEach(u => u())
  }, [subscribe])

  if (feed.length === 0) {
    return (
      <div className="di-card">
        <h3 className="di-card-title">Live Decision Feed</h3>
        <div className="di-empty">No adaptation decisions yet. Decisions will appear after drift is detected.</div>
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
