"""
DriftInsights — Predict Route

POST /predict — Make predictions using the current model.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import PredictRequest, PredictResponse
from src.utils.helpers import timestamp_now

router = APIRouter(tags=["Predictions"])


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Make a prediction using the current active model."""
    from api.main import get_pipeline

    pipeline = get_pipeline()
    if pipeline is None or pipeline.model is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    import numpy as np

    try:
        X = np.array(request.features).reshape(1, -1)
        y_pred = pipeline.model.predict(X)
        y_proba = pipeline.model.predict_proba(X)
        confidence = float(np.max(y_proba))

        return PredictResponse(
            prediction=int(y_pred[0]),
            confidence=round(confidence, 4),
            model_version=pipeline.model.version,
            timestamp=timestamp_now(),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
