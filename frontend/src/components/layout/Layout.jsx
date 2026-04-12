import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'
import Navbar from './Navbar'

export default function Layout({ connected, lastUpdate, fsmState, totalSamples, connectionFailed }) {
  return (
    <div className="app-layout">
      <Sidebar />
      <div className="app-main">
        {connectionFailed && (
          <div className="ws-failure-banner">
            WebSocket connection lost — live updates unavailable. Restart the backend and refresh.
          </div>
        )}
        <Navbar connected={connected} lastUpdate={lastUpdate} fsmState={fsmState} totalSamples={totalSamples || 0} />
        <main className="app-content">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
