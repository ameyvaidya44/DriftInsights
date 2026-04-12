import { useState } from 'react'
import { useAuth } from '../../context/AuthContext'

export default function LoginPage() {
  const { login } = useAuth()
  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [error, setError]       = useState('')
  const [loading, setLoading]   = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      login(email, password)
    } catch (err) {
      setError(err.message)
    }
    setLoading(false)
  }

  return (
    <div className="min-h-screen flex items-center justify-center" style={{ background: 'var(--bg-primary)' }}>
      <div style={{ width: 380 }}>
        {/* Logo */}
        <div className="flex items-center gap-3 justify-center mb-8">
          <div style={{ width: 40, height: 40, background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.2)', borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="22" height="22" viewBox="0 0 28 28" fill="none">
              <circle cx="14" cy="14" r="12" stroke="#6366f1" strokeWidth="2.5"/>
              <path d="M8 18 L12 10 L16 15 L20 8" stroke="#06b6d4" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </div>
          <span style={{ fontSize: 22, fontWeight: 700, background: 'linear-gradient(135deg,#6366f1,#06b6d4)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            DriftInsights
          </span>
        </div>

        <div className="di-card" style={{ padding: '28px 32px' }}>
          <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 4 }}>Sign in</h2>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 24 }}>ML Monitoring Platform</p>

          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div>
              <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '.4px' }}>Email</label>
              <input
                type="email" value={email} onChange={e => setEmail(e.target.value)} required
                placeholder="admin@driftinsights.io"
                style={{ width: '100%', padding: '9px 12px', background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text-primary)', fontSize: 13, outline: 'none' }}
              />
            </div>
            <div>
              <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '.4px' }}>Password</label>
              <input
                type="password" value={password} onChange={e => setPassword(e.target.value)} required
                placeholder="••••••••"
                style={{ width: '100%', padding: '9px 12px', background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text-primary)', fontSize: 13, outline: 'none' }}
              />
            </div>

            {error && (
              <div style={{ padding: '8px 12px', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 8, fontSize: 12, color: '#ef4444' }}>
                {error}
              </div>
            )}

            <button type="submit" disabled={loading} className="btn btn-primary" style={{ width: '100%', justifyContent: 'center', marginTop: 4 }}>
              {loading ? <><span className="spinner" /> Signing in...</> : 'Sign in'}
            </button>
          </form>

          <div style={{ marginTop: 20, padding: '12px', background: 'rgba(255,255,255,0.02)', borderRadius: 8, border: '1px solid var(--border)' }}>
            <p style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '.4px' }}>Demo credentials</p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <p style={{ fontSize: 11, color: 'var(--text-secondary)' }}>Admin: <span className="mono" style={{ color: 'var(--accent-purple)' }}>admin@driftinsights.io / admin</span></p>
              <p style={{ fontSize: 11, color: 'var(--text-secondary)' }}>User: <span className="mono" style={{ color: 'var(--accent-blue)' }}>user@driftinsights.io / user</span></p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
