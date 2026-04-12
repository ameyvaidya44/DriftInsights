/** Consistent empty / loading / error states for admin sections. */

export function LoadingSkeleton({ rows = 3 }) {
  return (
    <div className="animate-pulse space-y-3 p-2">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="h-8 rounded" style={{ background: 'rgba(255,255,255,0.05)', width: i % 2 === 0 ? '100%' : '75%' }} />
      ))}
    </div>
  )
}

export function EmptyState({ icon, message, sub }) {
  return (
    <div className="flex flex-col items-center justify-center py-12" style={{ color: 'var(--text-muted)' }}>
      {icon && <div style={{ fontSize: 28, marginBottom: 8, opacity: 0.5 }}>{icon}</div>}
      <p style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{message}</p>
      {sub && <p style={{ fontSize: 11, marginTop: 4 }}>{sub}</p>}
    </div>
  )
}

export function ErrorState({ message, onRetry }) {
  return (
    <div className="flex items-center gap-2 p-3 rounded-lg" style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', color: '#ef4444', fontSize: 12 }}>
      <span>⚠ {message || 'Failed to load data'}</span>
      {onRetry && (
        <button onClick={onRetry} style={{ marginLeft: 'auto', textDecoration: 'underline', background: 'none', border: 'none', cursor: 'pointer', color: 'inherit', fontSize: 12 }}>
          Retry
        </button>
      )}
    </div>
  )
}
