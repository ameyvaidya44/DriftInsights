/**
 * Coloured dot + label for service health status.
 */
export default function HealthIndicator({ status, label }) {
  const map = {
    ok:      { cls: 'ok',      text: label || 'Healthy'     },
    warn:    { cls: 'warn',    text: label || 'Degraded'    },
    error:   { cls: 'error',   text: label || 'Down'        },
    unknown: { cls: 'unknown', text: label || 'Unknown'     },
  }
  const cfg = map[status] || map.unknown
  return (
    <div className="health-indicator">
      <span className={`health-dot ${cfg.cls}`} />
      <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{cfg.text}</span>
    </div>
  )
}
