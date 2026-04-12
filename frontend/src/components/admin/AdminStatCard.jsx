/**
 * Reusable KPI stat card for admin overview pages.
 */
export default function AdminStatCard({ label, value, sub, accent = 'var(--accent-purple)', icon }) {
  return (
    <div className="admin-stat-card" style={{ '--card-accent': accent }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        <div>
          <div className="admin-stat-label">{label}</div>
          <div className="admin-stat-value">{value ?? '--'}</div>
          {sub && <div className="admin-stat-sub">{sub}</div>}
        </div>
        {icon && (
          <div style={{ fontSize: 22, opacity: 0.5 }}>{icon}</div>
        )}
      </div>
    </div>
  )
}
