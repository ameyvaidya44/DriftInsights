"""
DriftInsights — FastAPI Application

Main API entrypoint with REST endpoints, WebSocket support,
and background pipeline management.
"""

import asyncio
import logging
import sys
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from api.schemas import PipelineStatusResponse, HealthResponse, FSMStatusResponse
from api.websocket import manager
from api.tasks import task_manager
from api.routes import predict, drift, models, metrics
from api.routes import xai

logger = logging.getLogger("driftinsights.api")

# ── Global Pipeline State ───────────────────────────────────────────────────

_pipeline = None
_pipeline_lock = threading.Lock()
_start_time = time.time()


def get_pipeline():
    """Get the global pipeline instance."""
    return _pipeline


def set_pipeline(pipeline):
    """Set the global pipeline instance."""
    global _pipeline
    with _pipeline_lock:
        _pipeline = pipeline


# ── App Lifespan ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    logger.info("DriftInsights API starting up...")

    # Try to initialize pipeline with default config
    try:
        from src.pipeline import Pipeline
        pipeline = Pipeline()
        pipeline.setup()
        pipeline.train()
        pipeline.set_ws_manager(manager)
        set_pipeline(pipeline)
        logger.info("Pipeline initialized and trained successfully")
    except Exception as e:
        logger.warning(f"Pipeline auto-init failed (will need manual setup): {e}")

    yield

    # Shutdown
    logger.info("DriftInsights API shutting down...")
    if _pipeline:
        _pipeline.shutdown()
    task_manager.shutdown(wait=False)


# ── Create App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="DriftInsights API",
    description=(
        "Explainable and Adaptive ML System for Concept Drift Detection, "
        "Diagnosis, and Adaptation"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include Routers ─────────────────────────────────────────────────────────

app.include_router(predict.router)
app.include_router(drift.router)
app.include_router(models.router)
app.include_router(metrics.router)
app.include_router(xai.router)


# ── Core Endpoints ──────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    """API root — basic info."""
    return {
        "name": "DriftInsights API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/status", response_model=PipelineStatusResponse, tags=["System"])
async def get_status():
    """Get pipeline status including FSM state and streaming status."""
    pipeline = get_pipeline()
    if pipeline is None:
        return PipelineStatusResponse(
            session_id="",
            status="not_initialized",
            model_type="none",
            model_version=0,
            total_samples=0,
            drift_events=0,
            adaptations=0,
            running_accuracy=0.0,
            is_streaming=False,
            fsm_state="idle",
            current_strategy=None,
        )

    metrics_snapshot = pipeline.metrics.get_snapshot() if pipeline.metrics else {}

    # Get FSM state from decision engine
    fsm_state = "idle"
    current_strategy = None
    if pipeline.decision_engine:
        fsm_state = pipeline.decision_engine.state.value

    # Check if streaming is active
    is_streaming = task_manager.is_running("pipeline_stream")

    # Get successful/failed adaptation counts
    adaptations = 0
    if pipeline.adaptation_module:
        adaptations = pipeline.adaptation_module.total_adaptations

    return PipelineStatusResponse(
        session_id=pipeline.session_id,
        status="active",
        model_type=pipeline.model.model_type if pipeline.model else "none",
        model_version=pipeline.model.version if pipeline.model else 0,
        total_samples=metrics_snapshot.get("total_samples", 0),
        drift_events=len(pipeline.drift_events),
        adaptations=adaptations,
        running_accuracy=metrics_snapshot.get("running_accuracy", 0.0),
        is_streaming=is_streaming,
        fsm_state=fsm_state,
        current_strategy=current_strategy,
    )


@app.get("/pipeline/status/fsm", response_model=FSMStatusResponse, tags=["Pipeline"])
async def get_fsm_status():
    """Get detailed FSM state."""
    pipeline = get_pipeline()
    if pipeline is None or pipeline.decision_engine is None:
        return FSMStatusResponse(state="idle", adaptation_count=0, cooldown_samples=0)

    de = pipeline.decision_engine
    return FSMStatusResponse(
        state=de.state.value,
        current_strategy=None,
        adaptation_count=de.adaptation_count,
        cooldown_samples=de._cooldown_samples,
    )


@app.post("/pipeline/start", tags=["Pipeline"])
async def start_pipeline(dataset: str = "sea", model_type: str = "xgboost", max_samples: int | None = None):
    """Start the streaming pipeline in a background thread."""
    pipeline = get_pipeline()
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Pipeline not initialized. Please restart the server."},
        )

    def run_stream():
        try:
            summary = pipeline.run_stream(max_samples=max_samples)
            # Broadcast completion
            asyncio.run(manager.broadcast("pipeline_complete", summary))
        except Exception as e:
            logger.error(f"Pipeline streaming error: {e}")

    task_id = task_manager.submit("pipeline_stream", run_stream)
    return {"status": "started", "task_id": task_id}


@app.get("/tasks", tags=["System"])
async def get_tasks():
    """Get status of all background tasks."""
    return {"tasks": task_manager.get_all_tasks()}


# ── WebSocket ───────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time event streaming."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, handle client messages
            data = await websocket.receive_text()
            # Echo back as acknowledgment
            await manager.send_personal(websocket, "ack", {"message": data})
    except WebSocketDisconnect:
        await manager.disconnect(websocket)


# ── Run with uvicorn ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
