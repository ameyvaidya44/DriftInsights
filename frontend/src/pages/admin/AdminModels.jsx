import { useState } from 'react'
import AdminPageHeader from '../../components/admin/AdminPageHeader'
import Modal from '../../components/admin/Modal'
import { EmptyState, ErrorState, LoadingSkeleton } from '../../components/admin/EmptyState'
import useFetch from '../../hooks/useFetch'
import { fetchModelHistory, fetchModelInfo, activateModel, triggerRetrain } from '../../services/api'
import { formatTimestamp, formatDuration } from '../../utils/formatters'

export default function AdminModels() {
  const { data: histData, loading, error, refetch } = useFetch(fetchModelHistory, { refreshInterval: 10000 })
  const { data: current } = useFetch(fetchModelInfo, { refreshInterval: 10000 })

  const history = histData?.history || []

  const [retrainModal, setRetrainModal] = useState(false)
  const [retrainForm, setRetrainForm]   = useState({ strategy: 'incremental_retrain', max_samples: 1000 })
  const [retraining, setRetraining]     = useState(false)
  const [retrainErr, setRetrainErr]     = useState('')
  const [compare, setCompare]           = useState([])

  const handleActivate = async (version) => {
    if (!confirm(`Activate model v${version}?`)) return
    try { await activateModel(version); refetch() }
    catch { alert('Activate failed — backend endpoint not yet implemented') }
  }

  const handleRetrain = async () => {
    setRetraining(true); setRetrainErr('')
    try { await triggerRetrain(retrainForm); refetch(); setRetrainModal(false) }
    catch (e) { setRetrainErr(e?.response?.data?.detail || 'Retrain failed — backend endpoint not yet implemented') }
    setRetraining(false)
  }

  const toggleCompare = (idx) => {
    setCompare(prev => prev.includes(idx) ? prev.filter(i => i !== idx) : prev.length < 2 ? [...prev, idx] : prev)
  }

  const compareItems = compare.map(i => history[i]).filter(Boolean)

  return (
    <div>
      <AdminPageHeader
        title="Model Lifecycle"
        desc="View all model versions, activate, retrain, and compare performance"
        action={
          <button className="btn btn-primary" onClick={() => { setRetrainErr(''); setRetrainModal(true) }}>
            ⟳ Trigger Retrain
          </button>
        }
      />

      {error && <ErrorState message="Could not load model history" onRetry={refetch} />}

      {/* Current model */}
      {current && (
        <div className="di-card" style={{ marginBottom: 14 }}>
          <h3 className="di-card-title">
            Active Model
            <span style={{ fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 4, background: 'rgba(34,197,94,0.1)', color: '#22c55e', border: '1px solid rgba(34,197,94,0.2)' }}>LIVE</span>
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
            {[['Type', current.model_type], ['Version', `v${current.version}`], ['Features', current.n_features], ['Status', current.status]].map(([k, v]) => (
              <div key={k}>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.3px', marginBottom: 3 }}>{k}</div>
                <div style={{ fontSize: 14, fontWeight: 600 }}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Compare panel */}
      {compareItems.length === 2 && (
        <div className="di-card" style={{ marginBottom: 14, borderColor: 'rgba(99,102,241,0.3)' }}>
          <h3 className="di-card-title">Model Comparison</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {compareItems.map((m, i) => (
              <div key={i} style={{ padding: '12px', background: 'rgba(255,255,255,0.02)', borderRadius: 8, border: '1px solid var(--border)' }}>
                <div style={{ fontWeight: 600, marginBottom: 8 }}>v{m.model_version} — {m.strategy?.replace(/_/g,' ')}</div>
                {[
                  ['Accuracy Before', `${(m.old_accuracy*100).toFixed(2)}%`],
                  ['Accuracy After',  `${(m.new_accuracy*100).toFixed(2)}%`],
                  ['Δ Accuracy',      `${((m.new_accuracy-m.old_accuracy)*100).toFixed(2)}%`],
                  ['Samples Used',    m.n_samples_used?.toLocaleString()],
                  ['Duration',        formatDuration(m.duration_ms) || `${m.duration_seconds?.toFixed(2)}s`],
                  ['Result',          m.success ? 'Deployed' : 'Rejected'],
                ].map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '3px 0', borderBottom: '1px solid var(--border)' }}>
                    <span style={{ color: 'var(--text-muted)' }}>{k}</span>
                    <span style={{ fontWeight: 500 }}>{v}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
          <button className="btn-sm" style={{ marginTop: 10 }} onClick={() => setCompare([])}>Clear Comparison</button>
        </div>
      )}

      {compare.length > 0 && compare.length < 2 && (
        <div style={{ marginBottom: 12, fontSize: 12, color: 'var(--text-muted)', padding: '8px 12px', background: 'rgba(255,255,255,0.02)', borderRadius: 6 }}>
          Select one more model to compare ({compare.length}/2 selected)
        </div>
      )}

      {/* History table */}
      <div className="di-card">
        <h3 className="di-card-title">
          Version History <span className="di-badge-sm">{history.length} records</span>
          {compare.length > 0 && <span style={{ fontSize: 10, color: 'var(--accent-purple)', marginLeft: 8 }}>{compare.length} selected for compare</span>}
        </h3>
        {loading && !history.length ? <LoadingSkeleton /> : history.length === 0 ? (
          <EmptyState icon="🤖" message="No adaptation history yet" sub="Run the pipeline to generate model versions" />
        ) : (
          <div className="di-table-wrap">
            <table className="di-table">
              <thead>
                <tr><th>Compare</th><th>Version</th><th>Strategy</th><th>Result</th><th>Acc Before</th><th>Acc After</th><th>Δ Acc</th><th>Samples</th><th>Duration</th><th>Rollback</th><th>Timestamp</th><th>Actions</th></tr>
              </thead>
              <tbody>
                {history.map((h, i) => {
                  const delta = ((h.new_accuracy - h.old_accuracy) * 100).toFixed(2)
                  const isSelected = compare.includes(i)
                  return (
                    <tr key={i} className={h.success ? 'row-success' : 'row-fail'} style={isSelected ? { background: 'rgba(99,102,241,0.06)' } : {}}>
                      <td>
                        <input type="checkbox" checked={isSelected} onChange={() => toggleCompare(i)}
                          disabled={!isSelected && compare.length >= 2}
                          style={{ cursor: 'pointer', accentColor: 'var(--accent-purple)' }}
                        />
                      </td>
                      <td className="mono">v{h.model_version || '--'}</td>
                      <td>{h.strategy?.replace(/_/g,' ')}</td>
                      <td><span className={`table-badge ${h.success ? 'success' : 'fail'}`}>{h.success ? 'Deployed' : 'Rejected'}</span></td>
                      <td className="mono">{(h.old_accuracy*100).toFixed(2)}%</td>
                      <td className="mono">{(h.new_accuracy*100).toFixed(2)}%</td>
                      <td className={`mono ${parseFloat(delta)>=0?'text-green':'text-red'}`}>{delta>0?'+':''}{delta}%</td>
                      <td className="mono">{h.n_samples_used?.toLocaleString()}</td>
                      <td className="mono">{formatDuration(h.duration_ms) || `${h.duration_seconds?.toFixed(2)}s`}</td>
                      <td>{h.rollback && <span className="table-badge fail">ROLLBACK</span>}</td>
                      <td className="text-muted" style={{ fontSize: 11 }}>{formatTimestamp(h.timestamp)}</td>
                      <td>
                        {h.success && (
                          <button className="btn-sm" onClick={() => handleActivate(h.model_version)}>Activate</button>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Retrain Modal */}
      <Modal
        open={retrainModal}
        onClose={() => setRetrainModal(false)}
        title="Trigger Manual Retrain"
        actions={
          <>
            <button className="btn btn-secondary" onClick={() => setRetrainModal(false)}>Cancel</button>
            <button className="btn btn-primary" onClick={handleRetrain} disabled={retraining}>
              {retraining ? <><span className="spinner" /> Retraining…</> : 'Start Retrain'}
            </button>
          </>
        }
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {retrainErr && <ErrorState message={retrainErr} />}
          <div className="admin-form-group">
            <label className="admin-label">Strategy</label>
            <select className="admin-input admin-select" value={retrainForm.strategy} onChange={e => setRetrainForm(p => ({ ...p, strategy: e.target.value }))}>
              <option value="reweight">Reweight</option>
              <option value="incremental_retrain">Incremental Retrain</option>
              <option value="full_retrain">Full Retrain</option>
            </select>
          </div>
          <div className="admin-form-group">
            <label className="admin-label">Max Samples</label>
            <input type="number" className="admin-input" min={100} max={10000} step={100}
              value={retrainForm.max_samples} onChange={e => setRetrainForm(p => ({ ...p, max_samples: parseInt(e.target.value) }))} />
          </div>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
            This will trigger an immediate adaptation cycle using the selected strategy on the most recent streaming data.
          </p>
        </div>
      </Modal>
    </div>
  )
}
