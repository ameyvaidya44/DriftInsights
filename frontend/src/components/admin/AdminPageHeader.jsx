/**
 * Consistent page header for all admin pages.
 */
export default function AdminPageHeader({ title, desc, action }) {
  return (
    <div className="page-header" style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
          <span style={{ fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 4, background: 'rgba(249,115,22,0.1)', color: 'var(--accent-orange)', border: '1px solid rgba(249,115,22,0.2)', textTransform: 'uppercase', letterSpacing: '.5px' }}>
            Admin
          </span>
          <h2 className="page-title">{title}</h2>
        </div>
        {desc && <p className="page-desc">{desc}</p>}
      </div>
      {action && <div>{action}</div>}
    </div>
  )
}
