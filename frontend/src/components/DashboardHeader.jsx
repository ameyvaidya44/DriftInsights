export default function DashboardHeader({ connected, lastUpdate }) {
  return (
    <header className="header">
      <div className="header-left">
        <div className="logo">
          <div className="logo-icon">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <circle cx="14" cy="14" r="12" stroke="url(#grad)" strokeWidth="2.5" />
              <path d="M8 18 L12 10 L16 15 L20 8" stroke="url(#grad)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              <defs>
                <linearGradient id="grad" x1="0" y1="0" x2="28" y2="28">
                  <stop offset="0%" stopColor="#6366f1" />
                  <stop offset="100%" stopColor="#06b6d4" />
                </linearGradient>
              </defs>
            </svg>
          </div>
          <h1>DriftInsights</h1>
        </div>
        <span className="subtitle">Explainable & Adaptive ML System</span>
      </div>
      <div className="header-right">
        <div className={`status-badge ${connected ? 'connected' : 'disconnected'}`}>
          <span className="status-dot"></span>
          {connected ? 'Connected' : 'Disconnected'}
        </div>
        {lastUpdate && (
          <span className="last-update">
            Updated {lastUpdate.toLocaleTimeString()}
          </span>
        )}
      </div>
    </header>
  )
}
