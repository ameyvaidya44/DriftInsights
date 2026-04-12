import { useState, useEffect, useRef, useCallback } from 'react'

const MAX_RETRIES = 5
const BACKOFF_DELAYS = [1000, 2000, 4000, 8000, 16000]

/**
 * WebSocket hook with exponential backoff reconnection.
 * After MAX_RETRIES failed attempts, sets connectionFailed = true.
 */
export default function useWebSocket(url = 'ws://localhost:8000/ws') {
  const [connected, setConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState(null)
  const [lastUpdate, setLastUpdate] = useState(null)
  const [connectionFailed, setConnectionFailed] = useState(false)
  const wsRef = useRef(null)
  const reconnectRef = useRef(null)
  const retryCountRef = useRef(0)
  const listenersRef = useRef({})
  const mountedRef = useRef(true)

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    if (retryCountRef.current >= MAX_RETRIES) {
      setConnectionFailed(true)
      return
    }

    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = url.startsWith('ws') ? url : `${protocol}//localhost:8000/ws`
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        if (!mountedRef.current) return
        setConnected(true)
        setConnectionFailed(false)
        retryCountRef.current = 0
        if (reconnectRef.current) {
          clearTimeout(reconnectRef.current)
          reconnectRef.current = null
        }
      }

      ws.onmessage = (event) => {
        if (!mountedRef.current) return
        try {
          const msg = JSON.parse(event.data)
          setLastMessage(msg)
          setLastUpdate(new Date())
          const type = msg.type
          if (type && listenersRef.current[type]) {
            listenersRef.current[type].forEach(cb => cb(msg.data))
          }
        } catch {
          // non-JSON, ignore
        }
      }

      ws.onclose = () => {
        if (!mountedRef.current) return
        setConnected(false)
        const delay = BACKOFF_DELAYS[Math.min(retryCountRef.current, BACKOFF_DELAYS.length - 1)]
        retryCountRef.current += 1
        reconnectRef.current = setTimeout(connect, delay)
      }

      ws.onerror = () => ws.close()
      wsRef.current = ws
    } catch {
      setConnected(false)
    }
  }, [url])

  useEffect(() => {
    mountedRef.current = true
    connect()
    return () => {
      mountedRef.current = false
      if (wsRef.current) wsRef.current.close()
      if (reconnectRef.current) clearTimeout(reconnectRef.current)
    }
  }, [connect])

  const subscribe = useCallback((type, callback) => {
    if (!listenersRef.current[type]) listenersRef.current[type] = []
    listenersRef.current[type].push(callback)
    return () => {
      listenersRef.current[type] = listenersRef.current[type].filter(cb => cb !== callback)
    }
  }, [])

  return { connected, lastMessage, lastUpdate, subscribe, connectionFailed }
}
