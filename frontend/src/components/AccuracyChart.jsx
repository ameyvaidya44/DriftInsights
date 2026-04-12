import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

export default function AccuracyChart({ history, driftEvents }) {
  if (!history || !history.timestamps || history.timestamps.length === 0) {
    return (
      <div className="chart-card">
        <h3 className="chart-title">Accuracy Over Time</h3>
        <div className="chart-empty">Waiting for streaming data...</div>
      </div>
    )
  }

  const data = history.timestamps.map((t, i) => ({
    sample: t,
    accuracy: +(history.accuracy[i] * 100).toFixed(2),
    error: +(history.error[i] * 100).toFixed(2),
  }))

  const driftSamples = (driftEvents || []).map(e => e.sample_idx)

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    return (
      <div className="chart-tooltip">
        <p className="tooltip-label">Sample: {label}</p>
        {payload.map((p, i) => (
          <p key={i} style={{ color: p.color }}>
            {p.name}: {p.value}%
          </p>
        ))}
      </div>
    )
  }

  return (
    <div className="chart-card">
      <h3 className="chart-title">
        Accuracy & Error Over Time
        <span className="chart-badge">{data.length} points</span>
      </h3>
      <div className="chart-container">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis
              dataKey="sample"
              stroke="rgba(255,255,255,0.3)"
              fontSize={11}
              tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v}
            />
            <YAxis
              stroke="rgba(255,255,255,0.3)"
              fontSize={11}
              domain={[0, 100]}
              tickFormatter={v => `${v}%`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="accuracy"
              stroke="#22c55e"
              strokeWidth={2}
              dot={false}
              name="Accuracy"
            />
            <Line
              type="monotone"
              dataKey="error"
              stroke="#f97316"
              strokeWidth={2}
              dot={false}
              name="Error Rate"
            />
            {driftSamples.map((s, i) => (
              <ReferenceLine
                key={i}
                x={s}
                stroke="#ef4444"
                strokeDasharray="4 4"
                strokeWidth={2}
                label={{ value: 'DRIFT', position: 'top', fill: '#ef4444', fontSize: 10 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
