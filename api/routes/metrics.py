"""
DriftInsights — Metrics Routes

GET /metrics          — Current metrics snapshot
GET /metrics/history  — Historical metrics for plotting
GET /metrics/snapshot — Extended snapshot with fill percentage
"""

from fastapi import APIRouter

from api.schemas import MetricsResponse, MetricsHistoryResponse, MetricsSnapshotResponse

router = APIRouter(tags=["Metrics"])


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get current metrics snapshot."""
    from api.main import get_pipeline

    pipeline = get_pipeline()
    if pipeline is None or pipeline.metrics is None:
        return MetricsResponse(
            total_samples=0, running_accuracy=0.0, running_error=0.0,
            windowed_accuracy=0.0, windowed_error=0.0, window_size=0, window_fill=0,
        )

    snapshot = pipeline.metrics.get_snapshot()
    return MetricsResponse(**snapshot)


@router.get("/metrics/history", response_model=MetricsHistoryResponse)
async def get_metrics_history():
    """Get metrics history for plotting."""
    from api.main import get_pipeline

    pipeline = get_pipeline()
    if pipeline is None or pipeline.metrics is None:
        return MetricsHistoryResponse(timestamps=[], accuracy=[], error=[])

    history = pipeline.metrics.get_history()
    return MetricsHistoryResponse(**history)


@router.get("/metrics/snapshot", response_model=MetricsSnapshotResponse)
async def get_metrics_snapshot():
    """Get extended metrics snapshot with window fill percentage."""
    from api.main import get_pipeline

    pipeline = get_pipeline()
    if pipeline is None or pipeline.metrics is None:
        return MetricsSnapshotResponse(
            running_accuracy=0.0, running_error=0.0,
            windowed_accuracy=0.0, windowed_error=0.0,
            window_size=0, window_fill=0, window_fill_pct=0.0, total_samples=0,
        )

    snapshot = pipeline.metrics.get_snapshot()
    ws = snapshot.get("window_size", 1) or 1
    wf = snapshot.get("window_fill", 0)
    return MetricsSnapshotResponse(
        running_accuracy=snapshot.get("running_accuracy", 0.0),
        running_error=snapshot.get("running_error", 0.0),
        windowed_accuracy=snapshot.get("windowed_accuracy", 0.0),
        windowed_error=snapshot.get("windowed_error", 0.0),
        window_size=snapshot.get("window_size", 0),
        window_fill=wf,
        window_fill_pct=round((wf / ws) * 100, 1),
        total_samples=snapshot.get("total_samples", 0),
    )
