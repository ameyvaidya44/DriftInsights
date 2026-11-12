"""
DriftInsights — Input Mode Routes

POST /input/repository   — Start pipeline using a repository dataset
POST /input/upload       — Validate & start pipeline from uploaded files
GET  /input/datasets     — List available repository datasets with metadata
GET  /input/upload/status — Get last upload validation result
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("driftinsights.routes.input_modes")

router = APIRouter(prefix="/input", tags=["Input Modes"])

# ── Repository dataset catalogue ─────────────────────────────────────────────

REPO_DATASETS = [
    {
        "id": "elec2",
        "name": "ELEC2 — Electricity Pricing",
        "description": (
            "Real-world electricity demand dataset from the Australian NSW market "
            "(45,312 half-hourly observations). Exhibits well-documented seasonal and "
            "behavioural concept drift — 8 drift episodes detected by ADWIN."
        ),
        "records": 45312,
        "features": 8,
        "label": "class (UP / DOWN)",
        "drift_type": "Gradual / Seasonal",
        "available": True,
    },
    {
        "id": "sea",
        "name": "SEA Concepts (Synthetic)",
        "description": (
            "Synthetic benchmark with 3 continuous features and a configurable "
            "classification boundary. Supports sudden concept drift injection at a "
            "known position — ideal for controlled drift evaluation."
        ),
        "records": 20000,
        "features": 3,
        "label": "target (0 / 1)",
        "drift_type": "Sudden (synthetic)",
        "available": True,
    },
    {
        "id": "ieee_cis",
        "name": "IEEE-CIS Fraud Detection",
        "description": (
            "Real-world e-commerce transaction dataset with 590,540 rows and 394 "
            "features. Exhibits temporal evolution of fraud patterns. "
            "Note: dataset file must be present in data/ieee_cis_fraud.csv."
        ),
        "records": 590540,
        "features": 394,
        "label": "isFraud (0 / 1)",
        "drift_type": "Gradual (temporal)",
        "available": False,   # set True once the file is downloaded
    },
]


# ── Shared upload validation state (in-memory, per server instance) ──────────

_last_upload_result: dict = {}
_upload_lock = threading.Lock()


# ── GET /input/datasets ───────────────────────────────────────────────────────

@router.get("/datasets")
async def list_datasets():
    """Return the catalogue of available repository datasets."""
    return {"datasets": REPO_DATASETS}


# ── POST /input/repository ────────────────────────────────────────────────────

@router.post("/repository")
async def start_repository_pipeline(
    dataset: str = "elec2",
    model_type: str = "xgboost",
    max_samples: Optional[int] = None,
):
    """
    Start the pipeline using a repository dataset.

    Internally calls the same /pipeline/start logic that the existing
    Pipeline controls use — no duplication of core components.
    """
    from api.main import get_pipeline, set_pipeline
    from api.websocket import manager
    from api.tasks import task_manager

    valid_ids = {d["id"] for d in REPO_DATASETS}
    if dataset not in valid_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown dataset '{dataset}'. Available: {sorted(valid_ids)}",
        )

    # Reinitialise pipeline with the requested dataset
    try:
        from src.config import Config
        from src.pipeline import Pipeline

        config = Config()
        # Override dataset and model type
        config._data["data"]["dataset"] = dataset if dataset != "ieee_cis" else "elec2"
        config._data["model"]["type"] = model_type

        pipeline = Pipeline(config=config)
        pipeline.setup()
        pipeline.train()
        pipeline.set_ws_manager(manager)
        set_pipeline(pipeline)
        logger.info("Repository pipeline initialised: dataset=%s model=%s", dataset, model_type)
    except Exception as exc:
        logger.error("Pipeline init failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline initialisation failed: {exc}")

    # Start streaming in background — capture pipeline reference now,
    # not inside the closure, to avoid a race with set_pipeline().
    captured_pipeline = pipeline

    def run_stream():
        try:
            summary = captured_pipeline.run_stream(max_samples=max_samples)
            manager.broadcast_from_thread("pipeline_complete", summary)
        except Exception as exc:
            logger.error("Streaming error: %s", exc)

    task_id = task_manager.submit("pipeline_stream", run_stream)

    return {
        "status": "started",
        "mode": "repository",
        "dataset": dataset,
        "model_type": model_type,
        "task_id": task_id,
    }


# ── POST /input/upload ────────────────────────────────────────────────────────

@router.post("/upload")
async def start_upload_pipeline(
    model_file: UploadFile = File(..., description="Trained ML model (.joblib / .pkl)"),
    historical_file: UploadFile = File(..., description="Historical / reference dataset (.csv)"),
    current_file: UploadFile = File(..., description="New / current dataset (.csv)"),
    label_column: str = Form("target", description="Name of the label column"),
    model_type: str = Form("xgboost", description="Model type hint for retraining"),
    max_samples: Optional[int] = Form(None),
):
    """
    Validate uploaded model + datasets, then start the existing pipeline.

    Validation checks:
      1. Model file can be loaded (joblib/pickle)
      2. Historical CSV can be parsed
      3. Current CSV can be parsed
      4. Features are compatible between model and datasets
      5. Model can produce predictions on the historical data

    If labels are missing, ADWIN monitoring and retraining are skipped;
    SHAP-based explanation shift analysis still runs.
    """
    from api.main import get_pipeline, set_pipeline
    from api.websocket import manager
    from api.tasks import task_manager
    from src.upload_adapter import validate_and_prepare

    # Read file bytes
    model_bytes = await model_file.read()
    hist_bytes   = await historical_file.read()
    curr_bytes   = await current_file.read()

    # Run validation
    bundle, validation_results = validate_and_prepare(
        model_bytes=model_bytes,
        model_filename=model_file.filename,
        historical_bytes=hist_bytes,
        historical_filename=historical_file.filename,
        current_bytes=curr_bytes,
        current_filename=current_file.filename,
        label_column=label_column,
    )

    serialised_results = [
        {"ok": r.ok, "message": r.message, "details": r.details}
        for r in validation_results
    ]

    with _upload_lock:
        _last_upload_result.update({
            "validation_results": serialised_results,
            "bundle_ready": bundle is not None,
            "warnings": bundle.warnings if bundle else [],
        })

    if bundle is None:
        # Find the first failing step
        failed = next((r for r in serialised_results if not r["ok"]), None)
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Validation failed",
                "failed_step": failed["message"] if failed else "Unknown",
                "all_results": serialised_results,
            },
        )

    # Inject the uploaded model + data into the existing Pipeline
    try:
        pipeline = _build_upload_pipeline(bundle, model_type)
        pipeline.set_ws_manager(manager)
        set_pipeline(pipeline)
        logger.info(
            "Upload pipeline ready — model=%s features=%d has_labels=%s",
            bundle.model_type, len(bundle.feature_names), bundle.has_labels,
        )
    except Exception as exc:
        logger.error("Upload pipeline build failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline build failed: {exc}")

    captured_pipeline = pipeline

    def run_stream():
        try:
            summary = captured_pipeline.run_stream(max_samples=max_samples)
            manager.broadcast_from_thread("pipeline_complete", summary)
        except Exception as exc:
            logger.error("Upload pipeline streaming error: %s", exc)

    task_id = task_manager.submit("pipeline_stream", run_stream)

    return {
        "status": "started",
        "mode": "upload",
        "model_type": bundle.model_type,
        "features": bundle.feature_names,
        "has_labels": bundle.has_labels,
        "warnings": bundle.warnings,
        "validation_results": serialised_results,
        "task_id": task_id,
    }


# ── GET /input/upload/status ─────────────────────────────────────────────────

@router.get("/upload/status")
async def get_upload_status():
    """Return the result of the most recent upload validation."""
    with _upload_lock:
        return dict(_last_upload_result) if _last_upload_result else {"bundle_ready": False}


# ── Internal: build Pipeline from uploaded bundle ─────────────────────────────

def _build_upload_pipeline(bundle, model_type_hint: str):
    """
    Wire the validated upload bundle into the existing Pipeline.

    Strategy:
      - Build a minimal Config with no auto-loading dataset (we provide data directly).
      - Call Pipeline.setup() which initialises all 6 components.
      - Replace the pipeline's model with the uploaded estimator via hot_swap().
      - Replace the data splits with the bundle's reference / current arrays.
      - Re-compute W_before SHAP on the reference data.
    """
    import numpy as np
    from src.config import Config
    from src.pipeline import Pipeline

    # Minimal config — point dataset to SEA so setup() can train a placeholder
    # model, which we immediately hot-swap out.
    config = Config()
    config._data["data"]["dataset"] = "sea"
    config._data["model"]["type"] = (
        model_type_hint
        if model_type_hint in {"xgboost", "lightgbm", "random_forest"}
        else "xgboost"
    )

    pipeline = Pipeline(config=config)

    # setup() loads SEA data and trains a placeholder model
    pipeline.setup()
    # train() trains the placeholder on SEA — needed to fully init all components
    pipeline.train()

    # ── Swap in the uploaded model ──────────────────────────────────────────
    pipeline.model.hot_swap(bundle.model_estimator)
    pipeline.model.feature_names = bundle.feature_names

    # ── Override data splits with uploaded arrays ───────────────────────────
    n_ref = len(bundle.X_ref)
    split_val = max(1, int(n_ref * 0.15))
    split_train = n_ref - split_val

    pipeline._X_train = bundle.X_ref[:split_train]
    pipeline._y_train = bundle.y_ref[:split_train] if bundle.y_ref is not None else np.zeros(split_train, dtype=int)
    pipeline._X_val   = bundle.X_ref[split_train:]
    pipeline._y_val   = bundle.y_ref[split_train:] if bundle.y_ref is not None else np.zeros(split_val, dtype=int)

    # Use the current / new dataset as the streaming test set
    pipeline._X_test  = bundle.X_new
    pipeline._y_test  = bundle.y_new if bundle.y_new is not None else np.zeros(len(bundle.X_new), dtype=int)

    pipeline._feature_names = bundle.feature_names

    # ── Recompute W_before SHAP on uploaded reference data ─────────────────
    try:
        pipeline.xai_engine.set_model(bundle.model_estimator)
        # Use a subsample for SHAP efficiency
        max_shap = min(200, len(bundle.X_ref))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(bundle.X_ref), max_shap, replace=False)
        import shap as _shap
        explainer = _shap.TreeExplainer(bundle.model_estimator)
        shap_vals = explainer.shap_values(bundle.X_ref[idx])
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]
        import numpy as _np
        pipeline._w_before_shap = _np.mean(_np.abs(shap_vals), axis=0)
        logger.info("W_before SHAP computed on uploaded reference data")
    except Exception as exc:
        logger.warning("Could not pre-compute W_before SHAP: %s", exc)

    return pipeline
