import axios from 'axios'

const API_BASE = 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
})

// ── Core API calls ─────────────────────────────────────────────────────────

export const fetchStatus = () => api.get('/status').then(r => r.data)
export const fetchHealth = () => api.get('/health').then(r => r.data)
export const fetchMetrics = () => api.get('/metrics').then(r => r.data)
export const fetchMetricsHistory = () => api.get('/metrics/history').then(r => r.data)
export const fetchMetricsSnapshot = () => api.get('/metrics/snapshot').then(r => r.data)
export const fetchDriftEvents = () => api.get('/drift/events').then(r => r.data)
export const fetchLatestDrift = () => api.get('/drift/latest').then(r => r.data)
export const fetchModelInfo = () => api.get('/models').then(r => r.data)
export const fetchModelHistory = () => api.get('/models/history').then(r => r.data)
export const startPipeline = (config = {}) => api.post('/pipeline/start', null, { params: config }).then(r => r.data)

// ── New XAI endpoints ──────────────────────────────────────────────────────

export const fetchXaiLatest = () => api.get('/xai/latest').then(r => r.data)
export const fetchXaiHistory = () => api.get('/xai/history').then(r => r.data)
export const fetchXaiSummary = () => api.get('/xai/summary').then(r => r.data)
export const fetchEventExplanation = (id) => api.get(`/drift/events/${id}/explanation`).then(r => r.data)

// ── Pipeline/FSM endpoints ─────────────────────────────────────────────────

export const fetchFsmStatus = () => api.get('/pipeline/status/fsm').then(r => r.data)

// ── Aggregate fetcher for dashboard ────────────────────────────────────────

export const fetchAllDashboardData = async () => {
  const [status, metrics, metricsHistory, driftData, modelInfo, modelHistory] = await Promise.all([
    fetchStatus(), fetchMetrics(), fetchMetricsHistory(),
    fetchDriftEvents(), fetchModelInfo(), fetchModelHistory(),
  ])
  return {
    status, metrics, metricsHistory,
    driftEvents: driftData.events || [],
    driftCount: driftData.total_count || 0,
    modelInfo,
    modelHistory: modelHistory.history || [],
    modelHistoryMeta: { total: modelHistory.total, successful: modelHistory.successful },
  }
}

// ── Severity helpers (derived from backend thresholds) ─────────────────────

export const classifySeverity = (estimate) => {
  if (estimate >= 0.30) return 'severe'
  if (estimate >= 0.10) return 'moderate'
  if (estimate >= 0.05) return 'minor'
  return 'none'
}

export const severityColor = (severity) => ({
  severe: '#ef4444',
  moderate: '#f97316',
  minor: '#eab308',
  none: '#555570',
}[severity] || '#555570')

// ── Input Mode endpoints ───────────────────────────────────────────────────

export const fetchInputDatasets   = () => api.get('/input/datasets').then(r => r.data)
export const startRepositoryMode  = (params) => api.post('/input/repository', null, { params }).then(r => r.data)
export const startUploadMode      = (formData) => api.post('/input/upload', formData, {
  headers: { 'Content-Type': 'multipart/form-data' },
  timeout: 60000,
}).then(r => r.data)
export const fetchUploadStatus    = () => api.get('/input/upload/status').then(r => r.data)

// Users
export const fetchUsers        = () => api.get('/admin/users').then(r => r.data).catch(() => ({ users: [] }))
export const createUser        = (data) => api.post('/admin/users', data).then(r => r.data)
export const updateUser        = (id, data) => api.put(`/admin/users/${id}`, data).then(r => r.data)
export const deactivateUser    = (id) => api.patch(`/admin/users/${id}/deactivate`).then(r => r.data)

// Config
export const fetchAdminConfig  = () => api.get('/admin/config').then(r => r.data).catch(() => null)
export const updateAdminConfig = (data) => api.put('/admin/config', data).then(r => r.data)

// Datasets
export const fetchDatasets     = () => api.get('/admin/datasets').then(r => r.data).catch(() => ({ datasets: [] }))
export const activateDataset   = (name) => api.post(`/admin/datasets/${name}/activate`).then(r => r.data)
export const uploadDataset     = (formData) => api.post('/admin/datasets/upload', formData, { headers: { 'Content-Type': 'multipart/form-data' } }).then(r => r.data)

// Model lifecycle
export const fetchAllModels    = () => api.get('/admin/models').then(r => r.data).catch(() => ({ models: [] }))
export const activateModel     = (version) => api.post(`/admin/models/${version}/activate`).then(r => r.data)
export const triggerRetrain    = (params) => api.post('/admin/models/retrain', params).then(r => r.data)

// Audit log
export const fetchAuditLog     = (params) => api.get('/admin/audit', { params }).then(r => r.data).catch(() => ({ entries: [], total: 0 }))
export const exportAuditLog    = (params) => api.get('/admin/audit/export', { params, responseType: 'blob' }).then(r => r.data)

// System health
export const fetchSystemHealth = () => api.get('/admin/system-health').then(r => r.data).catch(() => null)
