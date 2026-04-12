import { useState } from 'react'
import { startPipeline } from '../services/api'

export default function PipelineControls({ onRefresh, status }) {
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')

  const handleStart = async () => {
    setLoading(true)
    setMessage('')
    try {
      const result = await startPipeline({ dataset: 'sea', model_type: 'xgboost' })
      setMessage(`Pipeline started: ${result.task_id}`)
    } catch (e) {
      setMessage(`Error: ${e.response?.data?.detail || e.message}`)
    }
    setLoading(false)
  }

  return (
    <div className="card controls-card">
      <h3 className="card-title">Pipeline Controls</h3>
      <div className="controls-row">
        <button className="btn btn-primary" onClick={handleStart} disabled={loading}>
          {loading ? (
            <span className="btn-loading">
              <span className="spinner"></span>
              Starting...
            </span>
          ) : (
            'Start Pipeline'
          )}
        </button>
        <button className="btn btn-secondary" onClick={onRefresh}>
          Refresh Data
        </button>
        <div className="controls-info">
          <span className="info-item">
            <span className="info-label">Session:</span>
            <span className="info-value">{status?.session_id ? status.session_id.slice(0, 24) + '...' : 'N/A'}</span>
          </span>
          <span className="info-item">
            <span className="info-label">Model Version:</span>
            <span className="info-value">v{status?.model_version || 0}</span>
          </span>
        </div>
      </div>
      {message && <div className="controls-message">{message}</div>}
    </div>
  )
}
