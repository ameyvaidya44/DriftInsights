export default function DriftTimeline({ events }) {
  const severityColor = (estimate) => {
    if (estimate >= 0.3) return 'var(--accent-red)'
    if (estimate >= 0.1) return 'var(--accent-orange)'
    return 'var(--accent-yellow)'
  }

  const severityLabel = (estimate) => {
    if (estimate >= 0.3) return 'Severe'
    if (estimate >= 0.1) return 'Moderate'
    return 'Minor'
  }

  return (
    <div className="card drift-timeline">
      <h3 className="card-title">
        Drift Event Timeline
        <span className="card-count">{events.length}</span>
      </h3>
      <div className="timeline-container">
        {events.length === 0 ? (
          <div className="timeline-empty">
            <div className="timeline-empty-icon">
              <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
                <circle cx="20" cy="20" r="16" stroke="rgba(255,255,255,0.1)" strokeWidth="2"/>
                <path d="M14 20h12M20 14v12" stroke="rgba(255,255,255,0.2)" strokeWidth="2" strokeLinecap="round"/>
              </svg>
            </div>
            <p>No drift events detected</p>
            <span>Events will appear here in real-time</span>
          </div>
        ) : (
          <div className="timeline-list">
            {events.map((event, i) => (
              <div key={i} className="timeline-item">
                <div className="timeline-dot" style={{ background: severityColor(event.severity_estimate) }}></div>
                <div className="timeline-content">
                  <div className="timeline-header">
                    <span className="timeline-label" style={{ color: severityColor(event.severity_estimate) }}>
                      {severityLabel(event.severity_estimate)} Drift
                    </span>
                    <span className="timeline-sample">Sample #{event.sample_idx}</span>
                  </div>
                  <div className="timeline-details">
                    <span>Error: {(event.pre_error_rate * 100).toFixed(1)}% &rarr; {(event.post_error_rate * 100).toFixed(1)}%</span>
                    <span className="timeline-badge">Detection #{event.detection_count}</span>
                  </div>
                  <div className="timeline-time">{new Date(event.timestamp).toLocaleString()}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
