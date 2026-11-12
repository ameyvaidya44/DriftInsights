"""
DriftInsights — WebSocket Manager

Manages WebSocket connections and broadcasts real-time events
(drift detections, metrics updates) to connected clients.
"""

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger("driftinsights.websocket")


class ConnectionManager:
    """
    Manages WebSocket connections for real-time event broadcasting.

    Clients connect to /ws and receive JSON messages for:
      - drift_event: When concept drift is detected
      - metrics_update: Periodic metrics snapshots
      - adaptation_event: When model adaptation occurs
      - system_event: Pipeline lifecycle events
    """

    def __init__(self):
        self._active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Store the FastAPI event loop so worker threads can broadcast safely."""
        self._loop = loop

    def broadcast_from_thread(self, event_type: str, data: dict[str, Any]) -> None:
        """Schedule a broadcast from a background pipeline thread."""
        if self._loop is None or not self._loop.is_running():
            logger.debug("Skipping WS broadcast — event loop not attached")
            return
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(event_type, data), self._loop)
        except Exception as exc:
            logger.debug(f"Threadsafe broadcast failed: {exc}")

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self._active_connections.append(websocket)
        logger.info(f"WebSocket connected — total: {len(self._active_connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            if websocket in self._active_connections:
                self._active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected — total: {len(self._active_connections)}")

    async def broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        """
        Broadcast an event to all connected clients.

        Args:
            event_type: Type of event (drift_event, metrics_update, etc.)
            data: Event data dictionary
        """
        if not self._active_connections:
            return

        message = json.dumps({
            "type": event_type,
            "data": data,
        }, default=str)

        # Send to all connections, removing dead ones
        dead_connections = []
        async with self._lock:
            for connection in self._active_connections:
                try:
                    await connection.send_text(message)
                except Exception:
                    dead_connections.append(connection)

            for conn in dead_connections:
                self._active_connections.remove(conn)

    async def send_personal(self, websocket: WebSocket, event_type: str, data: dict) -> None:
        """Send a message to a specific client."""
        message = json.dumps({"type": event_type, "data": data}, default=str)
        await websocket.send_text(message)

    @property
    def connection_count(self) -> int:
        return len(self._active_connections)


# Global manager instance
manager = ConnectionManager()
