"""
DriftInsights — Models Route

GET /models — Retrieve model version information.
"""

from fastapi import APIRouter

from api.schemas import ModelInfoResponse

router = APIRouter(tags=["Models"])


@router.get("/models", response_model=ModelInfoResponse)
async def get_model_info():
    """Get current model information."""
    from api.main import get_pipeline

    pipeline = get_pipeline()

    if pipeline is None or pipeline.model is None:
        return ModelInfoResponse(
            model_type="none",
            version=0,
            feature_names=[],
            n_features=0,
            status="not_initialized",
        )

    model = pipeline.model
    return ModelInfoResponse(
        model_type=model.model_type,
        version=model.version,
        feature_names=model.feature_names,
        n_features=len(model.feature_names),
        status="active",
    )


@router.get("/models/history")
async def get_model_history():
    """Get adaptation history for the model."""
    from api.main import get_pipeline

    pipeline = get_pipeline()
    if pipeline is None or pipeline.adaptation_module is None:
        return {"history": [], "total": 0}

    history = [r.to_dict() for r in pipeline.adaptation_module.history]
    # Ensure rollback field is always present (Part 3 fix)
    for record in history:
        record["rollback"] = record.get("rollback", False)
    return {
        "history": history,
        "total": len(history),
        "successful": pipeline.adaptation_module.successful_adaptations,
    }
