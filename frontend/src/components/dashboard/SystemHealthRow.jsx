export default function SystemHealthRow({ status, modelInfo }) {
  const isStreaming = status?.is_streaming
  const fsm = status?.fsm_state || 'idle'
  const fsmColors = { idle: '#22c55e', adapting: '#f97316', cooldown: '#eab308' }

  const tiles = [
    {
      label: 'Pipeline',
      value: isStreaming ? 'Running' : 'Stopped',
      color: isStreaming ? '#22c55e' : '#555570',
    },
    {
      label: 'Model',
      value: modelInfo ? `v${modelInfo.version} / ${modelInfo.model_type}` : '--',
      color: '#06b6d4',
    },
    {
      label: 'FSM State',
      value: fsm.toUpperCase(),
      color: fsmColors[fsm] || '#555570',
    },
    {
      label: 'Session',
      value: status?.session_id ? status.session_id.slice(0, 8) : '--',
      color: '#6366f1',
    },
  ]

  return (
    <div className="health-row">
      {tiles.map((t, i) => (
        <div key={i} className="health-tile">
          <span className="health-tile-label">{t.label}</span>
          <span className="health-tile-value" style={{ color: t.color }}>{t.value}</span>
        </div>
      ))}
    </div>
  )
}
