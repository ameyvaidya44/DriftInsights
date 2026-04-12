/**
 * DriftInsights — Shared formatting utilities
 * Use these everywhere instead of inline .toFixed() / date formatting.
 */

/** "2025-04-10 14:32:07" */
export const formatTimestamp = (ts) => {
  if (!ts) return '--'
  try {
    const d = new Date(ts)
    if (isNaN(d)) return ts
    const pad = (n) => String(n).padStart(2, '0')
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  } catch {
    return ts
  }
}

/** "HH:MM:SS" */
export const formatTime = (ts) => {
  if (!ts) return '--'
  try {
    const d = new Date(ts)
    if (isNaN(d)) return ts
    const pad = (n) => String(n).padStart(2, '0')
    return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  } catch {
    return ts
  }
}

/** "87.31" — rounds to given decimals */
export const formatNumber = (val, decimals = 2) => {
  if (val === null || val === undefined || isNaN(val)) return '--'
  return Number(val).toFixed(decimals)
}

/** "4,200" — locale-formatted integer */
export const formatSampleIndex = (n) => {
  if (n === null || n === undefined) return '--'
  return Number(n).toLocaleString()
}

/** "2.3s" or "341ms" */
export const formatDuration = (ms) => {
  if (ms === null || ms === undefined || isNaN(ms)) return '--'
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`
  return `${Math.round(ms)}ms`
}

/**
 * "+8.28%" with sign — returns { text, className }
 * className is 'text-green-600' for positive, 'text-red-600' for negative
 */
export const formatDelta = (val) => {
  if (val === null || val === undefined || isNaN(val)) return { text: '--', className: '' }
  const pct = (val * 100).toFixed(2)
  const sign = val >= 0 ? '+' : ''
  const className = val >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
  return { text: `${sign}${pct}%`, className }
}
