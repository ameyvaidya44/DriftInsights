/**
 * DriftInsights — Input Mode Page
 *
 * Two tabs:
 *   1. Repository  — pick a built-in dataset + model type → POST /input/repository
 *   2. Upload      — upload your own model (.joblib) + two CSVs → POST /input/upload
 *
 * Wires directly to:
 *   GET  /input/datasets
 *   POST /input/repository
 *   POST /input/upload
 *   GET  /input/upload/status
 */

import { useState, useRef, useEffect } from 'react'
import axios from 'axios'

const API = 'http://localhost:8000'

// ── tiny helpers ────────────────────────────────────────────────────────────

const api = axios.create({ baseURL: API, timeout: 30000 })

const fetchDatasets   = () => api.get('/input/datasets').then(r => r.data.datasets || [])
const startRepository = (params) => api.post('/input/repository', null, { params })
const startUpload     = (formData) =>
  api.post('/input/upload', formData, { headers: { 'Content-Type': 'multipart/form-data' } })

// ── small UI atoms ──────────────────────────────────────────────────────────

function TabBar({ active, onChange }) {
  return (
    <div style={{ display: 'flex', gap: 0, marginBottom: 24, borderBottom: '1px solid var(--border)' }}>
      {[
        { id: 'repository', label: '🗄  Repository Dataset' },
        { id: 'upload',     label: '↑  Upload Your Own'    },
      ].map(t => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          style={{
            padding: '9px 22px',
            fontSize: 13,
            fontWeight: active === t.id ? 600 : 400,
            background: 'none',
            border: 'none',
            borderBottom: active === t.id ? '2px solid var(--accent-purple)' : '2px solid transparent',
            color: active === t.id ? 'var(--accent-purple)' : 'var(--text-muted)',
            cursor: 'pointer',
            transition: 'all .15s',
            marginBottom: -1,
          }}
        >
          {t.label}
        </button>
      ))}
    </div>
  )
}

function StatusBadge({ ok, text }) {
  const color = ok ? '#22c55e' : '#ef4444'
  return (
    <span style={{
      fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4,
      background: `${color}15`, color, border: `1px solid ${color}30`,
    }}>
      {ok ? '✓' : '✗'} {text}
    </span>
  )
}

function Alert({ type = 'info', children }) {
  const colors = {
    success: { bg: 'rgba(34,197,94,.08)', border: 'rgba(34,197,94,.25)', text: '#22c55e' },
    error:   { bg: 'rgba(239,68,68,.08)', border: 'rgba(239,68,68,.25)', text: '#ef4444' },
    warning: { bg: 'rgba(234,179,8,.08)', border: 'rgba(234,179,8,.25)', text: '#eab308' },
    info:    { bg: 'rgba(99,102,241,.08)', border: 'rgba(99,102,241,.25)', text: 'var(--accent-purple)' },
  }
  const c = colors[type]
  return (
    <div style={{
      padding: '10px 14px', borderRadius: 8, fontSize: 12, lineHeight: 1.6,
      background: c.bg, border: `1px solid ${c.border}`, color: c.text,
      marginBottom: 14,
    }}>
      {children}
    </div>
  )
}

function FileDropZone({ label, accept, hint, file, onChange }) {
  const ref = useRef(null)
  const [dragging, setDragging] = useState(false)

  const handleDrop = (e) => {
    e.preventDefault(); setDragging(false)
    const f = e.dataTransfer.files?.[0]
    if (f) onChange(f)
  }

  return (
    <div>
      <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.3px', display: 'block', marginBottom: 6 }}>
        {label}
      </label>
      <div
        onClick={() => ref.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        style={{
          border: `1.5px dashed ${dragging ? 'var(--accent-purple)' : file ? '#22c55e' : 'var(--border)'}`,
          borderRadius: 8, padding: '14px 18px', cursor: 'pointer', textAlign: 'center',
          background: dragging ? 'rgba(99,102,241,.05)' : file ? 'rgba(34,197,94,.04)' : 'rgba(255,255,255,.01)',
          transition: 'all .15s',
        }}
      >
        <input ref={ref} type="file" accept={accept} style={{ display: 'none' }} onChange={e => onChange(e.target.files?.[0])} />
        {file ? (
          <div>
            <span style={{ fontSize: 13, fontWeight: 600, color: '#22c55e' }}>✓ {file.name}</span>
            <span style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', marginTop: 2 }}>
              {(file.size / 1024).toFixed(1)} KB
            </span>
          </div>
        ) : (
          <div>
            <span style={{ fontSize: 22, opacity: .4 }}>⬆</span>
            <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '4px 0 0' }}>{hint}</p>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Tab 1: Repository ────────────────────────────────────────────────────────

function RepositoryTab() {
  const [datasets, setDatasets]     = useState([])
  const [loading, setLoading]       = useState(true)
  const [selected, setSelected]     = useState(null)
  const [modelType, setModelType]   = useState('xgboost')
  const [maxSamples, setMaxSamples] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult]         = useState(null)
  const [error, setError]           = useState('')

  useEffect(() => {
    fetchDatasets()
      .then(ds => { setDatasets(ds); if (ds.length) setSelected(ds[0].id) })
      .catch(() => setDatasets([
        { id: 'elec2', name: 'ELEC2 — Electricity Pricing', records: 45312, features: 8, drift_type: 'Gradual / Seasonal', available: true, description: 'Real-world electricity demand dataset from the Australian NSW market.' },
        { id: 'sea',   name: 'SEA Concepts (Synthetic)',   records: 20000, features: 3, drift_type: 'Sudden (synthetic)',  available: true, description: 'Synthetic benchmark with configurable concept drift injection.' },
        { id: 'ieee_cis', name: 'IEEE-CIS Fraud Detection', records: 590540, features: 394, drift_type: 'Gradual (temporal)', available: false, description: 'E-commerce transactions — requires data/ieee_cis_fraud.csv.' },
      ]))
      .finally(() => setLoading(false))
  }, [])

  const handleStart = async () => {
    if (!selected) return
    setSubmitting(true); setError(''); setResult(null)
    try {
      const params = { dataset: selected, model_type: modelType }
      if (maxSamples) params.max_samples = parseInt(maxSamples)
      const r = await startRepository(params)
      setResult(r)
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || 'Failed to start pipeline')
    }
    setSubmitting(false)
  }

  const sel = datasets.find(d => d.id === selected)

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, alignItems: 'start' }}>

      {/* Left — dataset picker */}
      <div>
        <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, color: 'var(--text-secondary)' }}>
          Select Dataset
        </h3>
        {loading ? (
          <div style={{ fontSize: 12, color: 'var(--text-muted)', padding: 12 }}>Loading datasets…</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {datasets.map(ds => (
              <div
                key={ds.id}
                onClick={() => ds.available && setSelected(ds.id)}
                style={{
                  padding: '12px 14px', borderRadius: 8, cursor: ds.available ? 'pointer' : 'not-allowed',
                  border: `1.5px solid ${selected === ds.id ? 'var(--accent-purple)' : 'var(--border)'}`,
                  background: selected === ds.id ? 'rgba(99,102,241,.06)' : 'rgba(255,255,255,.01)',
                  opacity: ds.available ? 1 : 0.5,
                  transition: 'all .15s',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <span style={{ fontSize: 13, fontWeight: 600 }}>{ds.name}</span>
                  {!ds.available && <StatusBadge ok={false} text="File missing" />}
                  {ds.available && selected === ds.id && <StatusBadge ok={true} text="Selected" />}
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>{ds.description}</div>
                <div style={{ display: 'flex', gap: 12, marginTop: 8 }}>
                  {[
                    ['Records', ds.records?.toLocaleString()],
                    ['Features', ds.features],
                    ['Drift', ds.drift_type],
                  ].map(([k, v]) => (
                    <div key={k}>
                      <span style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.3px' }}>{k} </span>
                      <span style={{ fontSize: 11, fontWeight: 600 }}>{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Right — config + launch */}
      <div>
        <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, color: 'var(--text-secondary)' }}>
          Configuration
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

          <div>
            <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.3px', display: 'block', marginBottom: 6 }}>
              Model Type
            </label>
            <select
              value={modelType} onChange={e => setModelType(e.target.value)}
              className="admin-input admin-select"
            >
              <option value="xgboost">XGBoost</option>
              <option value="lightgbm">LightGBM</option>
              <option value="random_forest">Random Forest</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.3px', display: 'block', marginBottom: 6 }}>
              Max Samples <span style={{ fontWeight: 400 }}>(optional — leave blank for all)</span>
            </label>
            <input
              type="number" min={100} step={100} placeholder="e.g. 5000"
              value={maxSamples} onChange={e => setMaxSamples(e.target.value)}
              className="admin-input"
            />
          </div>

          {sel && (
            <div style={{ padding: '10px 12px', background: 'rgba(255,255,255,.02)', border: '1px solid var(--border)', borderRadius: 8 }}>
              <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 6, color: 'var(--text-secondary)' }}>Ready to start</div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.7 }}>
                <div>Dataset: <strong style={{ color: 'var(--text-primary)' }}>{sel.name}</strong></div>
                <div>Model: <strong style={{ color: 'var(--text-primary)' }}>{modelType}</strong></div>
                {maxSamples && <div>Max samples: <strong style={{ color: 'var(--text-primary)' }}>{parseInt(maxSamples).toLocaleString()}</strong></div>}
              </div>
            </div>
          )}

          {error && <Alert type="error">{error}</Alert>}

          {result && (
            <Alert type="success">
              Pipeline started ✓ — task <code style={{ fontSize: 11 }}>{result.task_id}</code><br />
              Dataset: <strong>{result.dataset}</strong> · Model: <strong>{result.model_type}</strong><br />
              Live updates will appear in the <strong>Dashboard</strong> and <strong>Live Stream</strong>.
            </Alert>
          )}

          <button
            className="btn btn-primary"
            onClick={handleStart}
            disabled={submitting || !selected || !datasets.find(d => d.id === selected)?.available}
            style={{ width: '100%', justifyContent: 'center' }}
          >
            {submitting ? <><span className="spinner" /> Starting…</> : '▶  Start Pipeline'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Tab 2: Upload ────────────────────────────────────────────────────────────

const STEP_LABELS = [
  'Model file loaded',
  'Historical dataset parsed',
  'Current dataset parsed',
  'Feature compatibility verified',
  'Prediction smoke-test passed',
  'All validations passed',
]

function ValidationSteps({ results }) {
  if (!results?.length) return null
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {results.map((r, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 10, fontSize: 12 }}>
          <span style={{ color: r.ok ? '#22c55e' : '#ef4444', fontSize: 14, lineHeight: 1, marginTop: 1, flexShrink: 0 }}>
            {r.ok ? '✓' : '✗'}
          </span>
          <div>
            <span style={{ color: r.ok ? 'var(--text-primary)' : '#ef4444', fontWeight: r.ok ? 400 : 600 }}>
              {r.message}
            </span>
            {r.details?.feature_names && (
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                Features: {r.details.feature_names.slice(0, 6).join(', ')}{r.details.feature_names.length > 6 ? ` +${r.details.feature_names.length - 6} more` : ''}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

function UploadTab() {
  const [modelFile,   setModelFile]   = useState(null)
  const [histFile,    setHistFile]    = useState(null)
  const [currFile,    setCurrFile]    = useState(null)
  const [labelCol,    setLabelCol]    = useState('target')
  const [modelType,   setModelType]   = useState('xgboost')
  const [maxSamples,  setMaxSamples]  = useState('')
  const [submitting,  setSubmitting]  = useState(false)
  const [validation,  setValidation]  = useState(null)   // { results, warnings, bundle_ready }
  const [launched,    setLaunched]    = useState(null)
  const [error,       setError]       = useState('')

  const canSubmit = modelFile && histFile && currFile && !submitting

  const handleSubmit = async () => {
    if (!canSubmit) return
    setSubmitting(true); setError(''); setValidation(null); setLaunched(null)

    const fd = new FormData()
    fd.append('model_file',       modelFile)
    fd.append('historical_file',  histFile)
    fd.append('current_file',     currFile)
    fd.append('label_column',     labelCol)
    fd.append('model_type',       modelType)
    if (maxSamples) fd.append('max_samples', maxSamples)

    try {
      const r = await startUpload(fd)
      setValidation({ results: r.validation_results, warnings: r.warnings, bundle_ready: true })
      setLaunched(r)
    } catch (e) {
      const detail = e?.response?.data?.detail
      if (detail && typeof detail === 'object') {
        // Backend returned structured validation error
        setValidation({ results: detail.all_results || [], warnings: [], bundle_ready: false })
        setError(detail.failed_step || 'Validation failed')
      } else {
        setError(typeof detail === 'string' ? detail : (e.message || 'Upload failed'))
      }
    }
    setSubmitting(false)
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, alignItems: 'start' }}>

      {/* Left — file inputs */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: 0, color: 'var(--text-secondary)' }}>
          Upload Files
        </h3>

        <FileDropZone
          label="Trained Model"
          accept=".joblib,.pkl,.pickle"
          hint="Drag & drop or click — .joblib / .pkl / .pickle"
          file={modelFile}
          onChange={setModelFile}
        />
        <FileDropZone
          label="Historical / Reference Dataset"
          accept=".csv"
          hint="Drag & drop or click — CSV with feature columns + label"
          file={histFile}
          onChange={setHistFile}
        />
        <FileDropZone
          label="Current / New Dataset"
          accept=".csv"
          hint="Drag & drop or click — CSV with same schema as reference"
          file={currFile}
          onChange={setCurrFile}
        />

        <Alert type="info" >
          The <strong>reference dataset</strong> is used to train SHAP W_before baselines.<br />
          The <strong>current dataset</strong> is streamed through the drift detection pipeline.
        </Alert>
      </div>

      {/* Right — options + results */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: 0, color: 'var(--text-secondary)' }}>
          Options
        </h3>

        <div>
          <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.3px', display: 'block', marginBottom: 6 }}>
            Label Column Name
          </label>
          <input
            type="text" placeholder="target"
            value={labelCol} onChange={e => setLabelCol(e.target.value)}
            className="admin-input"
          />
        </div>

        <div>
          <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.3px', display: 'block', marginBottom: 6 }}>
            Model Type Hint <span style={{ fontWeight: 400 }}>(for retraining)</span>
          </label>
          <select value={modelType} onChange={e => setModelType(e.target.value)} className="admin-input admin-select">
            <option value="xgboost">XGBoost</option>
            <option value="lightgbm">LightGBM</option>
            <option value="random_forest">Random Forest</option>
          </select>
        </div>

        <div>
          <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.3px', display: 'block', marginBottom: 6 }}>
            Max Streaming Samples <span style={{ fontWeight: 400 }}>(optional)</span>
          </label>
          <input
            type="number" min={100} step={100} placeholder="e.g. 5000"
            value={maxSamples} onChange={e => setMaxSamples(e.target.value)}
            className="admin-input"
          />
        </div>

        {/* Validation progress */}
        {validation && (
          <div style={{ padding: '12px 14px', background: 'rgba(255,255,255,.02)', border: '1px solid var(--border)', borderRadius: 8 }}>
            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 10, color: 'var(--text-secondary)' }}>
              Validation Results
            </div>
            <ValidationSteps results={validation.results} />

            {validation.warnings?.length > 0 && (
              <div style={{ marginTop: 10 }}>
                {validation.warnings.map((w, i) => (
                  <Alert key={i} type="warning">{w}</Alert>
                ))}
              </div>
            )}
          </div>
        )}

        {error && <Alert type="error">{error}</Alert>}

        {launched && (
          <Alert type="success">
            Pipeline started ✓ — task <code style={{ fontSize: 11 }}>{launched.task_id}</code><br />
            Model: <strong>{launched.model_type}</strong> · Features: <strong>{launched.features?.length}</strong><br />
            Labels present: <strong>{launched.has_labels ? 'Yes — full drift monitoring active' : 'No — SHAP-only mode'}</strong><br />
            Live updates will appear in the <strong>Dashboard</strong> and <strong>Live Stream</strong>.
          </Alert>
        )}

        <button
          className="btn btn-primary"
          onClick={handleSubmit}
          disabled={!canSubmit}
          style={{ width: '100%', justifyContent: 'center' }}
        >
          {submitting
            ? <><span className="spinner" /> Validating & Starting…</>
            : '↑  Upload & Start Pipeline'}
        </button>

        <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6 }}>
          The adapter will validate your files (5 checks), hot-swap your model into the live pipeline,
          recompute SHAP W_before on your reference data, then begin streaming your current dataset.
        </div>
      </div>
    </div>
  )
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function InputMode() {
  const [tab, setTab] = useState('repository')

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">Input Mode</h2>
        <p className="page-desc">
          Start the pipeline with a built-in dataset or upload your own model and data
        </p>
      </div>

      <div className="di-card">
        <TabBar active={tab} onChange={setTab} />
        {tab === 'repository' ? <RepositoryTab /> : <UploadTab />}
      </div>
    </div>
  )
}
