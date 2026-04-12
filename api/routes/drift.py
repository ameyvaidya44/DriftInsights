"""
DriftInsights — Drift Events Route

GET /drift/events                    — Retrieve drift detection history
GET /drift/latest                    — Most recent drift event
GET /drift/events/{event_id}/explanation — XAI explanation for specific event
"""

from fastapi import APIRouter, Query, Path
from fastapi.responses import JSONResponse

from api.schemas import DriftEventResponse, DriftEventsListResponse

router = APIRouter(prefix="/drift", tags=["Drift Detection"])


@router.get("/events", response_model=DriftEventsListResponse)
async def get_drift_events(
    limit: int = Query(50, ge=1, le=500, description="Max events to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """Get drift detection event history."""
    from api.main import get_pipeline

    pipeline = get_pipeline()
    if pipeline is None:
        return DriftEventsListResponse(events=[], total_count=0)

    all_events = pipeline.drift_events
    total = len(all_events)
    paginated = all_events[offset: offset + limit]

    events = []
    for e in paginated:
        events.append(DriftEventResponse(
            timestamp=e.get("timestamp", ""),
            sample_idx=e.get("sample_idx", 0),
            pre_error_rate=e.get("pre_error_rate", 0.0),
            post_error_rate=e.get("post_error_rate", 0.0),
            severity_estimate=e.get("severity_estimate", 0.0),
            detection_count=e.get("detection_count", 0),
            adwin_width=e.get("adwin_width", 0),
        ))

    return DriftEventsListResponse(events=events, total_count=total)


@router.get("/latest")
async def get_latest_drift():
    """Get the most recent drift event."""
    from api.main import get_pipeline

    pipeline = get_pipeline()
    if pipeline is None or not pipeline.drift_events:
        return JSONResponse(content=None)

    e = pipeline.drift_events[-1]
    return DriftEventResponse(
        timestamp=e.get("timestamp", ""),
        sample_idx=e.get("sample_idx", 0),
        pre_error_rate=e.get("pre_error_rate", 0.0),
        post_error_rate=e.get("post_error_rate", 0.0),
        severity_estimate=e.get("severity_estimate", 0.0),
        detection_count=e.get("detection_count", 0),
        adwin_width=e.get("adwin_width", 0),
    )


@router.get("/events/{event_id}/explanation")
async def get_event_explanation(
    event_id: int = Path(..., ge=0, description="0-based drift event index"),
):
    """Get the XAI explanation associated with a specific drift event."""
    from api.main import get_pipeline

    pipeline = get_pipeline()
    if pipeline is None or pipeline.xai_engine is None:
        return JSONResponse(status_code=404, content={"detail": "No pipeline or XAI engine"})

    history = pipeline.xai_engine.explanation_history
    if event_id >= len(history):
        return JSONResponse(status_code=404, content={"detail": f"No explanation for event {event_id}"})

    return history[event_id]
