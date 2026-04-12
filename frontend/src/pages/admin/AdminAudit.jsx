import { useState, useCallback } from 'react'
import AdminPageHeader from '../../components/admin/AdminPageHeader'
import { EmptyState, ErrorState, LoadingSkeleton } from '../../components/admin/EmptyState'
import useFetch from '../../hooks/useFetch'
import { fetchAuditLog } from '../../services/api'
import { formatTimestamp } from '../../utils/formatters'

// Mock audit entries for when backend isn't connected
const MOCK_ENTRIES = [
  { id: 1,  actor: 'admin@driftinsights.io', action: 'LOGIN',           resource: 'auth',          detail: 'Successful login from 127.0.0.1',                  level: 'info',    timestamp: new Date(Date.now()-3600000).toISOString() },
  { id: 2,  actor: 'system',                 action: 'PIPELINE_START',  resource: 'pipeline',      detail: 'Pipeline started with model v1 on dataset sea',    level: 'info',    timestamp: new Date(Date.now()-3500000).toISOString() },
  { id: 3,  actor: 'system',                 action: 'DRIFT_DETECTED',  resource: 'drift_detector',detail: 'ADWIN drift at sample 17,500 — Moderate severity', level: 'warning', timestamp: new Date(Date.now()-3000000).toISOString() },
  { id: 4,  actor: 'system',                 action: 'MODEL_ADAPTED',   resource: 'adaptation',    detail: 'incremental_retrain deployed — acc +4.2%',          level: 'info',    timestamp: new Date(Date.now()-2900000).toISOString() },
  { id: 5,  actor: 'admin@driftinsights.io', action: 'CONFIG_UPDATE',   resource: 'config',        detail: 'adwin_delta changed 0.002 → 0.001',                level: 'info',    timestamp: new Date(Date.now()-1800000).toISOString() },
  { id: 6,  actor: 'system',                 action: 'MODEL_ROLLBACK',  resource: 'adaptation',    detail: 'full_retrain rejected — accuracy below threshold',  level: 'warning', timestamp: new Date(Date.now()-1200000).toISOString() },
  { id: 7,  actor: 'user@driftinsights.io',  action: 'LOGIN',           resource: 'auth',          detail: 'Successful login from 192.168.1.5',                level: 'info',    timestamp: new Date(Date.now()-600000).toISOString()  },
  { id: 8,  actor: 'admin@driftinsights.io', action: 'USER_CREATED',    resource: 'users',         detail: 'Created user analyst@example.com (role: user)',    level: 'info',    timestamp: new Date(Date.now()-300000).toISOString()  },
  { id: 9,  actor: 'system',                 action: 'PIPELINE_COMPLETE',resource: 'pipeline',     detail: 'Pipeline complete — 5000 samples, 1 drift',        level: 'info',    timestamp: new Date(Date.now()-60000).toISOString()   },
  { id: 10, actor: 'admin@driftinsights.io', action: 'DATASET_ACTIVATE',resource: 'datasets',      detail: 'Activated dataset: elec2',                         level: 'info',    timestamp: new Date(Date.now()-30000).toISOString()   },
]

const LEVEL_COLORS = {
  info:    { color: '#06b6d4', bg: 'rgba(6,182,212,0.1)',   border: 'rgba(6,182,212,0.2)'   },
  warning: { color: '#eab308', bg: 'rgba(234,179,8,0.1)',   border: 'rgba(234,179,8,0.2)'   },
  error:   { color: '#ef4444', bg: 'rgba(239,68,68,0.1)',   border: 'rgba(239,68,68,0.2)'   },
  debug:   { color: '#555570', bg: 'rgba(85,85,112,0.1)',   border: 'rgba(85,85,112,0.2)'   },
}

const ALL_ACTIONS = [...new Set(MOCK_ENTRIES.map(e => e.action))]

export default function AdminAudit() {
  const [search, setSearch]       = useState('')
  const [levelFilter, setLevel]   = useState('all')
  const [actionFilter, setAction] = useState('all')
  const [page, setPage]           = useState(0)
  const PAGE_SIZE = 15

  const fetcher = useCallback(() => fetchAuditLog({ search, level: levelFilter !== 'all' ? levelFilter : undefined, action: actionFilter !== 'all' ? actionFilter : undefined }), [search, levelFilter, actionFilter])
  const { data, loading, error, refetch } = useFetch(fetcher, {})

  const entries = data?.entries?.length ? data.entries : MOCK_ENTRIES

  // Client-side filter on mock data
  const filtered = entries.filter(e => {
    const matchSearch = !search || e.actor.includes(search) || e.action.includes(search.toUpperCase()) || e.detail.toLowerCase().includes(search.toLowerCase())
    const matchLevel  = levelFilter === 'all' || e.level === levelFilter
    const matchAction = actionFilter === 'all' || e.action === actionFilter
    return matchSearch && matchLevel && matchAction
  })

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE)
  const paginated  = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  const handleExport = () => {
    const csv = ['ID,Actor,Action,Resource,Detail,Level,Timestamp',
      ...filtered.map(e => `${e.id},"${e.actor}","${e.action}","${e.resource}","${e.detail}","${e.level}","${e.timestamp}"`)
    ].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a'); a.href = url; a.download = `audit_log_${Date.now()}.csv`; a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div>
      <AdminPageHeader
        title="Audit Log"
        desc="Full audit trail of all system and user actions"
        action={
          <button className="btn btn-secondary" onClick={handleExport}>↓ Export CSV</button>
        }
      />

      {error && <ErrorState message="Could not load audit log from backend — showing mock data" />}

      {/* Filters */}
      <div className="di-card" style={{ padding: '12px 16px', marginBottom: 12 }}>
        <div className="audit-filters">
          <input className="audit-filter-input" placeholder="Search actor, action, detail…"
            value={search} onChange={e => { setSearch(e.target.value); setPage(0) }} style={{ flex: 1, minWidth: 200 }} />
          <select className="audit-filter-input admin-select" value={levelFilter} onChange={e => { setLevel(e.target.value); setPage(0) }}>
            <option value="all">All Levels</option>
            <option value="info">Info</option>
            <option value="warning">Warning</option>
            <option value="error">Error</option>
          </select>
          <select className="audit-filter-input admin-select" value={actionFilter} onChange={e => { setAction(e.target.value); setPage(0) }}>
            <option value="all">All Actions</option>
            {ALL_ACTIONS.map(a => <option key={a} value={a}>{a}</option>)}
          </select>
          <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 'auto' }}>{filtered.length} entries</span>
        </div>
      </div>

      <div className="di-card">
        {loading && !entries.length ? <LoadingSkeleton /> : paginated.length === 0 ? (
          <EmptyState icon="📋" message="No audit entries match your filters" />
        ) : (
          <>
            <div className="di-table-wrap">
              <table className="di-table">
                <thead>
                  <tr><th>Time</th><th>Actor</th><th>Action</th><th>Resource</th><th>Detail</th><th>Level</th></tr>
                </thead>
                <tbody>
                  {paginated.map(e => {
                    const lc = LEVEL_COLORS[e.level] || LEVEL_COLORS.info
                    return (
                      <tr key={e.id}>
                        <td className="mono" style={{ fontSize: 10, whiteSpace: 'nowrap' }}>{formatTimestamp(e.timestamp)}</td>
                        <td style={{ fontSize: 11 }}>{e.actor}</td>
                        <td>
                          <span className="audit-action-badge" style={{ color: lc.color, background: lc.bg, border: `1px solid ${lc.border}` }}>
                            {e.action}
                          </span>
                        </td>
                        <td style={{ fontSize: 11, color: 'var(--text-muted)' }}>{e.resource}</td>
                        <td style={{ fontSize: 11, maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis' }}>{e.detail}</td>
                        <td>
                          <span style={{ fontSize: 10, color: lc.color }}>{e.level}</span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            {totalPages > 1 && (
              <div className="pagination">
                <button className="btn-sm" disabled={page === 0} onClick={() => setPage(p => p-1)}>← Prev</button>
                <span className="pagination-info">Page {page+1} of {totalPages}</span>
                <button className="btn-sm" disabled={page >= totalPages-1} onClick={() => setPage(p => p+1)}>Next →</button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
