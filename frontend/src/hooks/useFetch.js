import { useState, useEffect, useCallback, useRef } from 'react'

/**
 * Shared data-fetching hook with optional polling and stale-while-error.
 *
 * @param {string|Function} url - URL string or async function returning data
 * @param {object} options
 * @param {number|null} options.refreshInterval - Poll interval in ms (null = no polling)
 * @param {boolean} options.enabled - If false, skips initial fetch
 * @returns {{ data, loading, error, refetch }}
 */
export default function useFetch(url, { refreshInterval = null, enabled = true } = {}) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const intervalRef = useRef(null)
  const mountedRef = useRef(true)

  const fetchData = useCallback(async () => {
    if (!enabled) return
    try {
      let result
      if (typeof url === 'function') {
        result = await url()
      } else {
        const res = await fetch(url)
        if (!res.ok) {
          throw Object.assign(new Error(res.statusText), { status: res.status })
        }
        result = await res.json()
      }
      if (mountedRef.current) {
        setData(result)
        setError(null)
      }
    } catch (e) {
      if (mountedRef.current) {
        setError({ status: e.status || 0, message: e.message || 'Unknown error' })
        // data stays at last successful value (stale-while-error)
      }
    } finally {
      if (mountedRef.current) setLoading(false)
    }
  }, [url, enabled])

  useEffect(() => {
    mountedRef.current = true
    setLoading(true)
    fetchData()

    if (refreshInterval) {
      intervalRef.current = setInterval(fetchData, refreshInterval)
    }

    return () => {
      mountedRef.current = false
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [fetchData, refreshInterval])

  return { data, loading, error, refetch: fetchData }
}
