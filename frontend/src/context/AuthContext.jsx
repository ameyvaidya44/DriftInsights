import { createContext, useContext, useState, useCallback } from 'react'

/**
 * AuthContext — lightweight client-side RBAC.
 * In production, replace the mock login with a real JWT/session API call.
 */

const AuthContext = createContext(null)

// Mock user store — swap for real API auth
const MOCK_USERS = [
  { id: 1, name: 'Admin User',  email: 'admin@driftinsights.io',  role: 'admin',  password: 'admin' },
  { id: 2, name: 'ML Engineer', email: 'user@driftinsights.io',   role: 'user',   password: 'user'  },
]

const SESSION_KEY = 'di_auth_user'

function loadSession() {
  try {
    const raw = sessionStorage.getItem(SESSION_KEY)
    return raw ? JSON.parse(raw) : null
  } catch { return null }
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(loadSession)

  const login = useCallback((email, password) => {
    const found = MOCK_USERS.find(u => u.email === email && u.password === password)
    if (!found) throw new Error('Invalid credentials')
    const { password: _, ...safe } = found
    sessionStorage.setItem(SESSION_KEY, JSON.stringify(safe))
    setUser(safe)
    return safe
  }, [])

  const logout = useCallback(() => {
    sessionStorage.removeItem(SESSION_KEY)
    setUser(null)
  }, [])

  const isAdmin = user?.role === 'admin'
  const isAuthenticated = !!user

  return (
    <AuthContext.Provider value={{ user, login, logout, isAdmin, isAuthenticated }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider')
  return ctx
}
