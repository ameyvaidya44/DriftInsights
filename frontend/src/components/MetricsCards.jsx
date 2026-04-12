export default function MetricsCards({ metrics, status }) {
  const cards = [
    {
      title: 'Running Accuracy',
      value: metrics ? `${(metrics.running_accuracy * 100).toFixed(1)}%` : '--',
      subtitle: 'Cumulative',
      color: 'var(--accent-green)',
      icon: (
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
          <path d="M2 18 L7 10 L11 13 L18 3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      ),
    },
    {
      title: 'Windowed Error',
      value: metrics ? `${(metrics.windowed_error * 100).toFixed(1)}%` : '--',
      subtitle: `Window: ${metrics?.window_size || 0}`,
      color: 'var(--accent-orange)',
      icon: (
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
          <circle cx="10" cy="10" r="7" stroke="currentColor" strokeWidth="2"/>
          <path d="M10 6v5l3 2" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
        </svg>
      ),
    },
    {
      title: 'Samples Processed',
      value: metrics ? metrics.total_samples.toLocaleString() : '--',
      subtitle: 'Total stream',
      color: 'var(--accent-blue)',
      icon: (
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
          <rect x="3" y="8" width="3" height="9" rx="1" fill="currentColor" opacity="0.5"/>
          <rect x="8.5" y="5" width="3" height="12" rx="1" fill="currentColor" opacity="0.7"/>
          <rect x="14" y="2" width="3" height="15" rx="1" fill="currentColor"/>
        </svg>
      ),
    },
    {
      title: 'Drift Events',
      value: status ? status.drift_events : '--',
      subtitle: `Adaptations: ${status?.adaptations || 0}`,
      color: 'var(--accent-purple)',
      icon: (
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
          <path d="M10 1 L12 8 L19 8 L13 12 L15 19 L10 15 L5 19 L7 12 L1 8 L8 8 Z" stroke="currentColor" strokeWidth="1.5" fill="none"/>
        </svg>
      ),
    },
  ]

  return (
    <div className="metrics-cards">
      {cards.map((card, i) => (
        <div key={i} className="metric-card" style={{ '--card-accent': card.color }}>
          <div className="metric-card-header">
            <span className="metric-icon" style={{ color: card.color }}>
              {card.icon}
            </span>
            <span className="metric-title">{card.title}</span>
          </div>
          <div className="metric-value">{card.value}</div>
          <div className="metric-subtitle">{card.subtitle}</div>
        </div>
      ))}
    </div>
  )
}
