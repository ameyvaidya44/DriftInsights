import { useEffect } from 'react'

/**
 * Generic modal overlay with keyboard close support.
 */
export default function Modal({ open, onClose, title, children, actions }) {
  useEffect(() => {
    if (!open) return
    const handler = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, onClose])

  if (!open) return null

  return (
    <div className="modal-overlay" onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="modal-box">
        <div className="modal-title">
          <span>{title}</span>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>
        <div>{children}</div>
        {actions && <div className="modal-actions">{actions}</div>}
      </div>
    </div>
  )
}
