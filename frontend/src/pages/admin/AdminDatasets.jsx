import { useState, useRef } from 'react'
import AdminPageHeader from '../../components/admin/AdminPageHeader'
import { EmptyState, ErrorState, LoadingSkeleton } from '../../components/admin/EmptyState'
import useFetch from '../../hooks/useFetch'
import { fetchDatasets, activateDataset, uploadDataset } from '../../services/api'

// Mock datasets for when backend isn't connected
const MOCK_DATASETS = [
  {
    name: 'sea', label: 'SEA (Synthetic)', active: true, type: 'synthetic',
    rows: 20000, features: 3, drift_position: 17500, noise: 0.1,
    description: 'Synthetic SEA concept drift dataset with abrupt drift at sample 17,500.',
    stats: { class_balance: '52% / 48%', missing: '0%', created: '2025-04-09' },
  },
  {
    name: 'elec2', label: 'ELEC2 (Electricity)', active: false, type: 'real-world',
    rows: 45312, features: 8, drift_position: null, noise: null,
    description: 'Real-world electricity demand dataset with gradual concept drift.',
    stats: { class_balance: '57% / 43%', missing: '0%', created: '2025-04-09' },
  },
]

export default function AdminDatasets() {
  const { data, loading, error, refetch } = useFetch(fetchDatasets, {})
  const datasets = data?.datasets?.length ? data.datasets : MOCK_DATASETS

  const [activating, setActivating] = useState(null)
  const [uploading, setUploading]   = useState(false)
  const [uploadErr, setUploadErr]   = useState('')
  const [uploadOk, setUploadOk]     = useState('')
  const [selected, setSelected]     = useState(null)
  const fileRef = useRef(null)

  const handleActivate = async (name) => {
    setActivating(name)
    try { await activateDataset(name); refetch() }
    catch { alert('Activate failed — backend endpoint not yet implemented') }
    setActivating(null)
  }

  const handleUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true); setUploadErr(''); setUploadOk('')
    const fd = new FormData()
    fd.append('file', file)
    fd.append('name', file.name.replace(/\.[^.]+$/, ''))
    try {
      await uploadDataset(fd)
      setUploadOk(`"${file.name}" uploaded successfully`)
      refetch()
    } catch (err) {
      setUploadErr(err?.response?.data?.detail || 'Upload failed — backend endpoint not yet implemented')
    }
    setUploading(false)
    e.target.value = ''
  }

  const active = datasets.find(d => d.active)
  const sel = selected ? datasets.find(d => d.name === selected) : null

  return (
    <div>
      <AdminPageHeader
        title="Dataset Management"
        desc="List, upload, preview, and activate datasets for the pipeline"
        action={
          <>
            <input ref={fileRef} type="file" accept=".csv,.parquet" style={{ display: 'none' }} onChange={handleUpload} />
            <button className="btn btn-primary" onClick={() => fileRef.current?.click()} disabled={uploading}>
              {uploading ? <><span className="spinner" /> Uploading…</> : '↑ Upload Dataset'}
            </button>
          </>
        }
      />

      {error && <ErrorState message="Could not load datasets from backend — showing defaults" />}
      {uploadErr && <div style={{ marginBottom: 12 }}><ErrorState message={uploadErr} /></div>}
      {uploadOk && (
        <div style={{ marginBottom: 12, padding: '8px 14px', background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.2)', borderRadius: 8, fontSize: 12, color: 'var(--accent-green)' }}>
          ✓ {uploadOk}
        </div>
      )}

      {active && (
        <div style={{ marginBottom: 14, padding: '10px 14px', background: 'rgba(99,102,241,0.06)', border: '1px solid rgba(99,102,241,0.2)', borderRadius: 8, fontSize: 12, color: 'var(--accent-purple)' }}>
          Active dataset: <strong>{active.label}</strong> — {active.rows?.toLocaleString()} rows, {active.features} features
        </div>
      )}

      {loading && !datasets.length ? <LoadingSkeleton rows={3} /> : (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 16 }}>
          {datasets.map(ds => (
            <div
              key={ds.name}
              className={`dataset-card ${ds.active ? 'dataset-card-active' : ''} ${selected === ds.name ? 'dataset-card-active' : ''}`}
              onClick={() => setSelected(selected === ds.name ? null : ds.name)}
            >
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 6 }}>
                <div>
                  <div className="dataset-card-name">{ds.label}</div>
                  <div className="dataset-card-meta">{ds.type} · {ds.rows?.toLocaleString()} rows · {ds.features} features</div>
                </div>
                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  {ds.active && <span style={{ fontSize: 9, fontWeight: 700, padding: '2px 7px', borderRadius: 4, background: 'rgba(34,197,94,0.1)', color: '#22c55e', border: '1px solid rgba(34,197,94,0.2)' }}>ACTIVE</span>}
                  {!ds.active && (
                    <button className="btn-sm" onClick={e => { e.stopPropagation(); handleActivate(ds.name) }} disabled={activating === ds.name}>
                      {activating === ds.name ? 'Activating…' : 'Activate'}
                    </button>
                  )}
                </div>
              </div>
              <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>{ds.description}</p>
              <div className="dataset-stat-row">
                {ds.drift_position && <div className="dataset-stat"><span className="dataset-stat-label">Drift At</span><span className="dataset-stat-val">{ds.drift_position?.toLocaleString()}</span></div>}
                {ds.noise !== null && ds.noise !== undefined && <div className="dataset-stat"><span className="dataset-stat-label">Noise</span><span className="dataset-stat-val">{ds.noise}</span></div>}
                <div className="dataset-stat"><span className="dataset-stat-label">Balance</span><span className="dataset-stat-val">{ds.stats?.class_balance || '--'}</span></div>
                <div className="dataset-stat"><span className="dataset-stat-label">Missing</span><span className="dataset-stat-val">{ds.stats?.missing || '0%'}</span></div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Preview panel */}
      {sel && (
        <div className="di-card">
          <h3 className="di-card-title">Preview — {sel.label}</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(180px,1fr))', gap: 10 }}>
            {Object.entries({
              Name: sel.name, Type: sel.type, Rows: sel.rows?.toLocaleString(),
              Features: sel.features, 'Drift Position': sel.drift_position?.toLocaleString() || 'N/A',
              Noise: sel.noise ?? 'N/A', 'Class Balance': sel.stats?.class_balance,
              'Missing Values': sel.stats?.missing, Created: sel.stats?.created,
            }).map(([k, v]) => (
              <div key={k} style={{ padding: '8px 10px', background: 'rgba(255,255,255,0.02)', borderRadius: 6, border: '1px solid var(--border)' }}>
                <div style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.3px', marginBottom: 3 }}>{k}</div>
                <div style={{ fontSize: 13, fontWeight: 500 }}>{v ?? '--'}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
