import { useState, useEffect } from 'react'
import AdminPageHeader from '../../components/admin/AdminPageHeader'
import { ErrorState, LoadingSkeleton } from '../../components/admin/EmptyState'
import useFetch from '../../hooks/useFetch'
import { fetchAdminConfig, updateAdminConfig } from '../../services/api'

// Default values mirrored from config/default.yaml
const DEFAULTS = {
  adwin_delta:          0.002,
  min_gap:              200,
  warmup_samples:       100,
  severity_minor:       0.05,
  severity_moderate:    0.15,
  severity_severe:      0.30,
  cooldown_samples:     500,
  accuracy_threshold:   0.75,
  tolerance:            0.05,
  window_size:          500,
  shap_top_k:           5,
  shap_max_samples:     200,
  incremental_window:   1000,
  full_retrain_min:     2000,
  decay_factor:         0.95,
}

const FIELDS = [
  {
    section: 'Drift Detection (ADWIN)',
    fields: [
      { key: 'adwin_delta',      label: 'ADWIN Delta',       desc: 'Sensitivity — lower = more detections',  min: 0.0001, max: 0.1,  step: 0.0001 },
      { key: 'min_gap',          label: 'Min Gap (samples)', desc: 'Minimum samples between detections',      min: 50,     max: 2000, step: 10     },
      { key: 'warmup_samples',   label: 'Warmup Samples',    desc: 'Skip initial N samples',                  min: 0,      max: 500,  step: 10     },
    ],
  },
  {
    section: 'Severity Thresholds (ΔE)',
    fields: [
      { key: 'severity_minor',    label: 'Minor Threshold',    desc: 'ΔE ≥ this → Minor drift',    min: 0.01, max: 0.5, step: 0.01 },
      { key: 'severity_moderate', label: 'Moderate Threshold', desc: 'ΔE ≥ this → Moderate drift', min: 0.01, max: 0.5, step: 0.01 },
      { key: 'severity_severe',   label: 'Severe Threshold',   desc: 'ΔE ≥ this → Severe drift',   min: 0.01, max: 1.0, step: 0.01 },
    ],
  },
  {
    section: 'Adaptation & Validation',
    fields: [
      { key: 'cooldown_samples',   label: 'Cooldown Samples',       desc: 'Min samples between adaptations',    min: 100,  max: 5000, step: 50   },
      { key: 'accuracy_threshold', label: 'Accuracy Threshold',     desc: 'Min accuracy to deploy new model',   min: 0.5,  max: 1.0,  step: 0.01 },
      { key: 'tolerance',          label: 'Regression Tolerance',   desc: 'Allowed accuracy drop on adaptation',min: 0.0,  max: 0.2,  step: 0.01 },
      { key: 'decay_factor',       label: 'Reweight Decay Factor',  desc: 'Exponential decay for sample weights',min: 0.5, max: 1.0,  step: 0.01 },
      { key: 'incremental_window', label: 'Incremental Window',     desc: 'Samples used for incremental retrain',min: 100, max: 5000, step: 100  },
      { key: 'full_retrain_min',   label: 'Full Retrain Min Samples',desc: 'Min samples for full retrain',       min: 500, max: 10000,step: 100  },
    ],
  },
  {
    section: 'Metrics & XAI',
    fields: [
      { key: 'window_size',     label: 'Metrics Window Size', desc: 'Rolling window for error rate',    min: 100,  max: 2000, step: 50  },
      { key: 'shap_top_k',     label: 'SHAP Top-K Features', desc: 'Features reported per explanation', min: 1,    max: 20,   step: 1   },
      { key: 'shap_max_samples',label: 'SHAP Max Samples',   desc: 'Samples per SHAP window',           min: 50,   max: 500,  step: 10  },
    ],
  },
]

export default function AdminConfig() {
  const { data: remote, loading, error, refetch } = useFetch(fetchAdminConfig, {})
  const [values, setValues] = useState(DEFAULTS)
  const [dirty, setDirty]   = useState(false)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved]   = useState(false)
  const [saveErr, setSaveErr] = useState('')

  useEffect(() => {
    if (remote) { setValues({ ...DEFAULTS, ...remote }); setDirty(false) }
  }, [remote])

  const set = (key, val) => {
    setValues(p => ({ ...p, [key]: parseFloat(val) }))
    setDirty(true); setSaved(false)
  }

  const handleSave = async () => {
    setSaving(true); setSaveErr('')
    try {
      await updateAdminConfig(values)
      setSaved(true); setDirty(false); refetch()
    } catch (e) {
      setSaveErr(e?.response?.data?.detail || 'Save failed — backend endpoint not yet implemented')
    }
    setSaving(false)
  }

  const handleReset = () => { setValues(DEFAULTS); setDirty(true); setSaved(false) }

  return (
    <div>
      <AdminPageHeader
        title="Configuration"
        desc="Edit pipeline parameters — changes take effect on next pipeline run"
        action={
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-secondary" onClick={handleReset}>Reset Defaults</button>
            <button className="btn btn-primary" onClick={handleSave} disabled={saving || !dirty}>
              {saving ? <><span className="spinner" /> Saving…</> : saved ? '✓ Saved' : 'Save Changes'}
            </button>
          </div>
        }
      />

      {error && <ErrorState message="Could not load config from backend — showing defaults" />}
      {saveErr && <div style={{ marginBottom: 12 }}><ErrorState message={saveErr} /></div>}
      {loading && !remote && <LoadingSkeleton rows={6} />}

      {dirty && !saved && (
        <div style={{ marginBottom: 12, padding: '8px 14px', background: 'rgba(234,179,8,0.08)', border: '1px solid rgba(234,179,8,0.2)', borderRadius: 8, fontSize: 12, color: 'var(--accent-yellow)' }}>
          ⚠ You have unsaved changes
        </div>
      )}

      {FIELDS.map(section => (
        <div key={section.section} className="di-card">
          <h3 className="di-card-title">{section.section}</h3>
          <div className="config-editor-grid">
            {section.fields.map(f => (
              <div key={f.key} className="config-field">
                <label className="config-field-label">{f.label}</label>
                <input
                  type="number" className="admin-input"
                  value={values[f.key] ?? ''} min={f.min} max={f.max} step={f.step}
                  onChange={e => set(f.key, e.target.value)}
                />
                <span className="config-field-desc">{f.desc}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}
