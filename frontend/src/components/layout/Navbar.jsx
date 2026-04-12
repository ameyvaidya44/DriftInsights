import { useAuth } from '../../context/AuthContext'

export default function Navbar({ connected, lastUpdate, fsmState, totalSamples }) {
  const { user, logout, isAdmin } = useAuth()

  const stateColors = {
    idle:     { bg: 'rgba(34,197,94,0.1)',  color: '#22c55e', border: 'rgba(34,197,94,0.2)'  },
    adapting: { bg: 'rgba(249,115,22,0.1)', color: '#f97316', border: 'rgba(249,115,22,0.2)' },
    cooldown: { bg: 'rgba(234,179,8,0.1)',  color: '#eab308', border: 'rgba(234,179,8,0.2)'  },
  }
  const sc = stateColors[fsmState] || stateColors.idle

  return (
    <header className="navbar">
      <div className="navbar-title">
        <h1>DriftInsights</h1>
        <span className="navbar-subtitle">ML Monitoring Platform</span>
      </div>

      <div className="navbar-right">
        {totalSamples > 0 && (
          <span className="navbar-samples mono">{totalSamples.toLocaleString()} samples</span>
        )}
        {fsmState && (
          <span className="navbar-fsm-badge" style={{ background: sc.bg, color: sc.color, borderColor: sc.border }}>
            {(fsmState || 'idle').toUpperCase()}
          </span>
        )}
        <span className={`status-badge ${connected ? 'connected' : 'disconnected'}`}>
          <span className="status-dot" />
          {connected ? 'Connected' : 'Disconnected'}
        </span>
        {lastUpdate && (
          <span className="last-update">Updated {lastUpdate.toLocaleTimeString()}</span>
        )}

        {/* User info + logout */}
        {user && (
          <div className="navbar-user">
            <div className="navbar-user-info">
              <span className="navbar-user-name">{user.name}</span>
              <span className={`navbar-role-badge ${isAdmin ? 'role-admin' : 'role-user'}`}>
                {isAdmin ? 'Admin' : 'User'}
              </span>
            </div>
            <button className="navbar-logout-btn" onClick={logout} title="Sign out">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
                <polyline points="16 17 21 12 16 7"/>
                <line x1="21" y1="12" x2="9" y2="12"/>
              </svg>
            </button>
          </div>
        )}
      </div>
    </header>
  )
}
