import { NavLink } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'

const userNav = [
  { path: '/',              label: 'Dashboard',      icon: '⊞' },
  { path: '/stream',        label: 'Live Stream',    icon: '▸' },
  { path: '/drift',         label: 'Drift Analytics',icon: '⚡' },
  { path: '/explainability',label: 'Explainability', icon: '◉' },
  { path: '/models',        label: 'Models',         icon: '◈' },
  { path: '/pipeline',      label: 'Pipeline',       icon: '⚙' },
  { path: '/input',         label: 'Input Mode',     icon: '⬆' },
]

const adminNav = [
  { path: '/admin',               label: 'Admin Overview',  icon: '⬡' },
  { path: '/admin/users',         label: 'Users',           icon: '👤' },
  { path: '/admin/config',        label: 'Config',          icon: '⚒' },
  { path: '/admin/datasets',      label: 'Datasets',        icon: '🗄' },
  { path: '/admin/models',        label: 'Model Lifecycle', icon: '🔬' },
  { path: '/admin/audit',         label: 'Audit Log',       icon: '📋' },
  { path: '/admin/system-health', label: 'System Health',   icon: '💓' },
]

export default function Sidebar() {
  const { isAdmin } = useAuth()

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">
          <svg width="22" height="22" viewBox="0 0 28 28" fill="none">
            <circle cx="14" cy="14" r="12" stroke="#6366f1" strokeWidth="2.5"/>
            <path d="M8 18 L12 10 L16 15 L20 8" stroke="#06b6d4" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </div>
        <span className="sidebar-logo-text">DriftInsights</span>
      </div>

      <nav className="sidebar-nav">
        {/* User section */}
        <div className="sidebar-section-label">Monitoring</div>
        {userNav.map(item => (
          <NavLink key={item.path} to={item.path} end={item.path === '/'}
            className={({ isActive }) => `sidebar-link ${isActive ? 'sidebar-link-active' : ''}`}>
            <span className="sidebar-link-icon">{item.icon}</span>
            <span className="sidebar-link-label">{item.label}</span>
          </NavLink>
        ))}

        {/* Admin section — only visible to admins */}
        {isAdmin && (
          <>
            <div className="sidebar-divider" />
            <div className="sidebar-section-label sidebar-section-admin">Administration</div>
            {adminNav.map(item => (
              <NavLink key={item.path} to={item.path} end={item.path === '/admin'}
                className={({ isActive }) => `sidebar-link sidebar-link-admin ${isActive ? 'sidebar-link-active-admin' : ''}`}>
                <span className="sidebar-link-icon">{item.icon}</span>
                <span className="sidebar-link-label">{item.label}</span>
              </NavLink>
            ))}
          </>
        )}
      </nav>

      <div className="sidebar-footer">
        <div className="sidebar-version">v1.0.0</div>
      </div>
    </aside>
  )
}
