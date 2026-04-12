import { Navigate } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'

/** Redirects to /login if not authenticated. */
export function RequireAuth({ children }) {
  const { isAuthenticated } = useAuth()
  return isAuthenticated ? children : <Navigate to="/login" replace />
}

/** Redirects to / if authenticated but not admin. */
export function RequireAdmin({ children }) {
  const { isAuthenticated, isAdmin } = useAuth()
  if (!isAuthenticated) return <Navigate to="/login" replace />
  if (!isAdmin) return <Navigate to="/" replace />
  return children
}
