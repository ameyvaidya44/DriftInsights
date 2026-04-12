"""
DriftInsights — Pydantic Schemas for API request/response validation.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Request Schemas ─────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Request body for /predict endpoint."""
    features: list[float] = Field(..., description="Feature values for prediction")
    model_version: Optional[int] = Field(None, description="Specific model version to use")


class StreamConfigRequest(BaseModel):
    """Request body for configuring streaming parameters."""
    dataset: str = Field("sea", description="Dataset name: 'sea' or 'elec2'")
    model_type: str = Field("xgboost", description="Model type")
    max_samples: Optional[int] = Field(None, description="Max samples to stream")


# ── Response Schemas ────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    """Response for /predict endpoint."""
    prediction: int
    confidence: float
    model_version: int
    timestamp: str


class DriftEventResponse(BaseModel):
    """Single drift event in API responses."""
    timestamp: str
    sample_idx: int
    pre_error_rate: float
    post_error_rate: float
    severity_estimate: float
    detection_count: int
    adwin_width: int = 0


class DriftEventsListResponse(BaseModel):
    """Response for /drift/events endpoint."""
    events: list[DriftEventResponse]
    total_count: int


class ModelInfoResponse(BaseModel):
    """Response for /models endpoint."""
    model_type: str
    version: int
    feature_names: list[str]
    n_features: int
    status: str = "active"


class MetricsResponse(BaseModel):
    """Response for /metrics endpoint."""
    total_samples: int
    running_accuracy: float
    running_error: float
    windowed_accuracy: float
    windowed_error: float
    window_size: int
    window_fill: int


class MetricsHistoryResponse(BaseModel):
    """Response for /metrics/history endpoint."""
    timestamps: list[int]
    accuracy: list[float]
    error: list[float]


class MetricsSnapshotResponse(BaseModel):
    """Response for /metrics/snapshot — includes fill percentage."""
    running_accuracy: float
    running_error: float
    windowed_accuracy: float
    windowed_error: float
    window_size: int
    window_fill: int
    window_fill_pct: float
    total_samples: int


class PipelineStatusResponse(BaseModel):
    """Response for /status endpoint."""
    session_id: str
    status: str
    model_type: str
    model_version: int
    total_samples: int
    drift_events: int
    adaptations: int
    running_accuracy: float
    is_streaming: bool = False
    fsm_state: str = "idle"
    current_strategy: Optional[str] = None


class AdaptationHistoryResponse(BaseModel):
    """Single adaptation record."""
    strategy: str
    success: bool
    old_accuracy: float
    new_accuracy: float
    threshold: float
    n_samples_used: int
    duration_seconds: float
    timestamp: str
    model_version: int
    reason: str = ""
    rollback: bool = False


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    uptime_seconds: float


# ── XAI Schemas ─────────────────────────────────────────────────────────────

class FeatureDriftDetailResponse(BaseModel):
    """Per-feature drift detail from SHAP analysis."""
    feature: str
    delta: float
    direction: str
    mean_before: float
    mean_after: float
    std_before: float
    std_after: float
    shap_before: float
    shap_after: float


class XAIExplanationResponse(BaseModel):
    """Response for /xai/latest — full drift explanation."""
    delta_e_total: float
    direction: str
    top_features: list[FeatureDriftDetailResponse]
    pre_importances: dict[str, float]
    post_importances: dict[str, float]


class XAIHistoryResponse(BaseModel):
    """Response for /xai/history."""
    explanations: list[dict]
    total: int


class XAISummaryResponse(BaseModel):
    """Response for /xai/summary."""
    events_analyzed: int
    top_drifting_features: list[dict]


class FSMStatusResponse(BaseModel):
    """Response for /pipeline/status/fsm."""
    state: str
    current_strategy: Optional[str] = None
    adaptation_count: int = 0
    cooldown_samples: int = 0
