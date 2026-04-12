import { useState, useEffect, useRef } from 'react'
import useWebSocket from '../../hooks/useWebSocket'

const stages = [
  { key: 'data', label: 'Data', icon: '⊟' },
  { key: 'predict', label: 'Predict', icon: '▸' },
  { key: 'drift', label: 'Drift', icon: '⚡' },
  { key: 'explain', label: 'Explain', icon: '◉' },
  { key: 'decide', label: 'Decide', icon: '◈' },
  { key: 'adapt', label: 'Adapt', icon: '⟳' },
]

export default function PipelineFlowViz({ status }) {
  const [highlighted, setHighlighted] = useState(new Set())
  const timeoutsRef = useRef([])
  const { subscribe } = useWebSocket()

  const clearTimeouts = () => {
    timeoutsRef.current.forEach(clearTimeout)
    timeoutsRef.current = []
  }

  const highlight = (keys, durationMs) => {
    setHighlighted(new Set(keys))
    const t = setTimeout(() => setHighlighted(new Set()), durationMs)
    timeoutsRef.current.push(t)
  }

  useEffect(() => {
    const unsubs = [
      subscribe('drift_detected', () => {
        clearTimeouts()
        highlight(['drift'], 5000)
      }),
      subscribe('xai_complete', () => {
        clearTimeouts()
        highlight(['explain'], 3000)
        const t = setTimeout(() => highlight(['decide'], 3000), 3000)
        timeoutsRef.current.push(t)
      }),
      subscribe('adaptation_complete', () => {
        clearTimeouts()
        highlight(['adapt'], 4000)
      }),
    ]
    return () => {
      clearTimeouts()
      unsubs.forEach(u => u())
    }
  }, [subscribe])

  // When FSM returns to idle, dim all; when not streaming, grey out
  const fsm = status?.fsm_state || 'idle'
  const isStreaming = status?.is_streaming
  const notStreaming = !isStreaming && !status?.total_samples

  // Base active stage from FSM (fallback when no WS events)
  const getBaseActive = () => {
    if (fsm === 'adapting') return 'adapt'
    if (fsm === 'cooldown') return 'adapt'
    if (isStreaming || (status?.total_samples > 0)) return 'predict'
    return null
  }
  const baseActive = getBaseActive()

  return (
    <div className="di-card">
      <h3 className="di-card-title">Pipeline Flow</h3>
      <div className="pipeline-flow">
        {stages.map((stage, i) => {
          const isWsHighlighted = highlighted.has(stage.key)
          const isBaseActive = !highlighted.size && stage.key === baseActive
          const isActive = isWsHighlighted || isBaseActive
          const isPast = !highlighted.size && baseActive && stages.findIndex(s => s.key === baseActive) > i
          const isCooldown = fsm === 'cooldown' && stage.key === 'adapt'
          const isAdapting = fsm === 'adapting' && (stage.key === 'decide' || stage.key === 'adapt')
          const isDimmed = notStreaming

          return (
            <div key={stage.key} className="pipeline-flow-item">
              <div className={[
                'pipeline-node',
                isActive ? 'pipeline-node-active' : '',
                isPast ? 'pipeline-node-done' : '',
                isCooldown ? 'pipeline-node-cooldown' : '',
                isAdapting ? 'pipeline-node-adapting' : '',
                isDimmed ? 'pipeline-node-dimmed' : '',
              ].filter(Boolean).join(' ')}>
                <span className="pipeline-node-icon">{stage.icon}</span>
                <span className="pipeline-node-label">{stage.label}</span>
              </div>
              {i < stages.length - 1 && (
                <div className={`pipeline-connector ${isPast ? 'pipeline-connector-done' : ''}`} />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
