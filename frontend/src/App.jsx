import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import './App.css'
import { AuthProvider } from './context/AuthContext'
import { RequireAuth, RequireAdmin } from './components/auth/ProtectedRoute'
import LoginPage from './components/auth/LoginPage'
import Layout from './components/layout/Layout'
import useWebSocket from './hooks/useWebSocket'
import useFetch from './hooks/useFetch'
import { fetchStatus } from './services/api'

// User pages
import Dashboard      from './pages/Dashboard'
import LiveStream     from './pages/LiveStream'
import DriftAnalytics from './pages/DriftAnalytics'
import Explainability from './pages/Explainability'
import Models         from './pages/Models'
import Pipeline       from './pages/Pipeline'

// Admin pages
import AdminOverview     from './pages/admin/AdminOverview'
import AdminUsers        from './pages/admin/AdminUsers'
import AdminConfig       from './pages/admin/AdminConfig'
import AdminDatasets     from './pages/admin/AdminDatasets'
import AdminModels       from './pages/admin/AdminModels'
import AdminAudit        from './pages/admin/AdminAudit'
import AdminSystemHealth from './pages/admin/AdminSystemHealth'

function AppShell() {
  const { connected, lastUpdate, connectionFailed } = useWebSocket()
  const { data: status } = useFetch(fetchStatus, { refreshInterval: 3000 })

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />

      {/* All authenticated routes share the Layout */}
      <Route element={
        <RequireAuth>
          <Layout
            connected={connected}
            lastUpdate={lastUpdate}
            fsmState={status?.fsm_state || null}
            totalSamples={status?.total_samples || 0}
            connectionFailed={connectionFailed}
          />
        </RequireAuth>
      }>
        {/* User routes */}
        <Route index element={<Dashboard />} />
        <Route path="/stream"        element={<LiveStream />} />
        <Route path="/drift"         element={<DriftAnalytics />} />
        <Route path="/explainability"element={<Explainability />} />
        <Route path="/models"        element={<Models />} />
        <Route path="/pipeline"      element={<Pipeline />} />

        {/* Admin routes — require admin role */}
        <Route path="/admin" element={<RequireAdmin><AdminOverview /></RequireAdmin>} />
        <Route path="/admin/users"         element={<RequireAdmin><AdminUsers /></RequireAdmin>} />
        <Route path="/admin/config"        element={<RequireAdmin><AdminConfig /></RequireAdmin>} />
        <Route path="/admin/datasets"      element={<RequireAdmin><AdminDatasets /></RequireAdmin>} />
        <Route path="/admin/models"        element={<RequireAdmin><AdminModels /></RequireAdmin>} />
        <Route path="/admin/audit"         element={<RequireAdmin><AdminAudit /></RequireAdmin>} />
        <Route path="/admin/system-health" element={<RequireAdmin><AdminSystemHealth /></RequireAdmin>} />
      </Route>

      {/* Catch-all */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppShell />
      </AuthProvider>
    </BrowserRouter>
  )
}
