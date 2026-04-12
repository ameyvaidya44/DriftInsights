"""
DriftInsights — XAI Routes

GET /xai/latest   — Most recent DriftExplanation
GET /xai/history  — All past explanations
GET /xai/summary  — Cumulative feature drift summary
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.schemas import XAIExplanationResponse, XAIHistoryResponse, XAISummaryResponse

router = APIRouter(prefix="/xai", tags=["XAI / Explainability"])


@router.get("/latest")
async def get_xai_latest():
    """Get the most recent drift explanation from SHAP analysis."""
    from api.main import get_pipeline

    pipeline = get_pipeline()
    if pipeline is None or pipeline.xai_engine is None:
        return JSONResponse(content=None)

    history = pipeline.xai_engine.explanation_history
    if not history:
        return JSONResponse(content=None)

    latest = history[-1]
    return latest


@router.get("/history", response_model=XAIHistoryResponse)
async def get_xai_history():
    """Get all past drift explanations."""
    from api.main import get_pipeline

    pipeline = get_pipeline()
    if pipeline is None or pipeline.xai_engine is None:
        return XAIHistoryResponse(explanations=[], total=0)

    history = pipeline.xai_engine.explanation_history
    return XAIHistoryResponse(explanations=history, total=len(history))


@router.get("/summary")
async def get_xai_summary():
    """Get cumulative feature drift summary across all events."""
    from api.main import get_pipeline

    pipeline = get_pipeline()
    if pipeline is None or pipeline.xai_engine is None:
        return {"events_analyzed": 0, "top_drifting_features": []}

    return pipeline.xai_engine.get_feature_drift_summary()
