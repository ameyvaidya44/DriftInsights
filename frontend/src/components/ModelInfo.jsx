export default function ModelInfo({ info, history }) {
  return (
    <div className="card model-info">
      <h3 className="card-title">
        Model Status
        <span className={`model-status-badge ${info?.status === 'active' ? 'active' : 'inactive'}`}>
          {info?.status || 'Unknown'}
        </span>
      </h3>

      <div className="model-details">
        <div className="model-detail-row">
          <span className="detail-label">Model Type</span>
          <span className="detail-value">{info?.model_type || '--'}</span>
        </div>
        <div className="model-detail-row">
          <span className="detail-label">Version</span>
          <span className="detail-value version-badge">v{info?.version || 0}</span>
        </div>
        <div className="model-detail-row">
          <span className="detail-label">Features</span>
          <span className="detail-value">{info?.n_features || 0}</span>
        </div>
        {info?.feature_names && info.feature_names.length > 0 && (
          <div className="model-detail-row feature-list-row">
            <span className="detail-label">Feature Names</span>
            <div className="feature-tags">
              {info.feature_names.map((f, i) => (
                <span key={i} className="feature-tag">{f}</span>
              ))}
            </div>
          </div>
        )}
      </div>

      {history && history.length > 0 && (
        <div className="adaptation-history">
          <h4>Adaptation History</h4>
          <div className="history-list">
            {history.slice(-5).reverse().map((h, i) => (
              <div key={i} className={`history-item ${h.success ? 'success' : 'failed'}`}>
                <div className="history-header">
                  <span className="history-strategy">{h.strategy.replace('_', ' ')}</span>
                  <span className={`history-badge ${h.success ? 'success' : 'failed'}`}>
                    {h.success ? 'Deployed' : 'Rejected'}
                  </span>
                </div>
                <div className="history-metrics">
                  <span>Acc: {(h.old_accuracy * 100).toFixed(1)}% → {(h.new_accuracy * 100).toFixed(1)}%</span>
                  <span>{h.duration_seconds?.toFixed(2)}s</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
