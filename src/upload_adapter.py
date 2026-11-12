"""
DriftInsights — Upload Adapter

Validates and adapts user-uploaded models + datasets so they can be passed
directly into the existing Pipeline (src/pipeline.py) without any changes
to the six core components.

Supported model formats: joblib (.joblib, .pkl, .pickle)
Supported dataset formats: CSV (.csv)

The adapter acts as an orchestration shim:
  UploadedModel + HistoricalCSV + CurrentCSV
        ↓
  upload_adapter.validate_and_prepare()
        ↓
  Returns PipelineInputBundle — a dict that Pipeline.setup_from_bundle() consumes
"""

from __future__ import annotations

import io
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("driftinsights.upload_adapter")

# ── Supported formats ────────────────────────────────────────────────────────

SUPPORTED_MODEL_SUFFIXES = {".joblib", ".pkl", ".pickle"}
SUPPORTED_DATA_SUFFIXES  = {".csv"}


@dataclass
class ValidationResult:
    """Returned by each validation step."""
    ok: bool
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class PipelineInputBundle:
    """
    Normalised inputs ready for Pipeline.setup_from_bundle().

    mode: "upload" | "repository"
    model_estimator: a fitted sklearn-compatible estimator (upload mode)
    model_type: "xgboost" | "lightgbm" | "random_forest" | "uploaded"
    X_ref: historical / reference feature matrix  (upload mode)
    y_ref: historical / reference labels          (upload mode, may be None)
    X_new: current / new feature matrix           (upload mode)
    y_new: current / new labels                   (upload mode, may be None)
    feature_names: list of feature column names
    has_labels: True if ground-truth labels are present in both datasets
    dataset_name: "sea" | "elec2" | "ieee_cis" (repo mode) OR "uploaded" (upload mode)
    warnings: list of non-fatal issues the UI should surface
    """
    mode: str
    model_estimator: Any = None
    model_type: str = "uploaded"
    X_ref: Optional[np.ndarray] = None
    y_ref: Optional[np.ndarray] = None
    X_new: Optional[np.ndarray] = None
    y_new: Optional[np.ndarray] = None
    feature_names: list[str] = field(default_factory=list)
    has_labels: bool = False
    dataset_name: str = "uploaded"
    warnings: list[str] = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def validate_and_prepare(
    model_bytes: bytes,
    model_filename: str,
    historical_bytes: bytes,
    historical_filename: str,
    current_bytes: bytes,
    current_filename: str,
    label_column: str = "target",
) -> tuple[Optional[PipelineInputBundle], list[ValidationResult]]:
    """
    Run all validations and return a ready-to-use bundle.

    Returns:
        (bundle, results) — bundle is None if any critical validation fails.
    """
    results: list[ValidationResult] = []

    # ── Step 1: Model file ──────────────────────────────────────────────────
    model_result, estimator = _validate_model(model_bytes, model_filename)
    results.append(model_result)
    if not model_result.ok:
        return None, results

    # ── Step 2: Historical dataset ──────────────────────────────────────────
    hist_result, df_hist = _validate_csv(historical_bytes, historical_filename, "historical")
    results.append(hist_result)
    if not hist_result.ok:
        return None, results

    # ── Step 3: Current dataset ─────────────────────────────────────────────
    curr_result, df_curr = _validate_csv(current_bytes, current_filename, "current")
    results.append(curr_result)
    if not curr_result.ok:
        return None, results

    # ── Step 4: Feature compatibility ───────────────────────────────────────
    compat_result, feature_names, warnings = _validate_feature_compatibility(
        estimator, df_hist, df_curr, label_column
    )
    results.append(compat_result)
    if not compat_result.ok:
        return None, results

    # ── Step 5: Prediction smoke-test ───────────────────────────────────────
    pred_result = _validate_prediction(estimator, df_hist, feature_names)
    results.append(pred_result)
    if not pred_result.ok:
        return None, results

    # ── Build bundle ─────────────────────────────────────────────────────────
    X_ref, y_ref = _split_xy(df_hist, feature_names, label_column)
    X_new, y_new = _split_xy(df_curr, feature_names, label_column)
    has_labels = (y_ref is not None) and (y_new is not None)

    if not has_labels:
        warnings.append(
            "Ground-truth labels not found in one or both datasets. "
            "ADWIN drift monitoring requires prediction errors — supervised monitoring "
            "and retraining will be skipped. Only SHAP-based distribution comparison "
            "will run."
        )

    model_type = _infer_model_type(estimator)

    bundle = PipelineInputBundle(
        mode="upload",
        model_estimator=estimator,
        model_type=model_type,
        X_ref=X_ref,
        y_ref=y_ref,
        X_new=X_new,
        y_new=y_new,
        feature_names=feature_names,
        has_labels=has_labels,
        dataset_name="uploaded",
        warnings=warnings,
    )

    results.append(ValidationResult(ok=True, message="All validations passed. Ready to start pipeline."))
    return bundle, results


# ── Validation helpers ────────────────────────────────────────────────────────

def _validate_model(
    model_bytes: bytes, filename: str
) -> tuple[ValidationResult, Any]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_MODEL_SUFFIXES:
        return ValidationResult(
            ok=False,
            message=f"Unsupported model format '{suffix}'. Use .joblib, .pkl, or .pickle.",
        ), None

    try:
        import joblib
        buf = io.BytesIO(model_bytes)
        payload = joblib.load(buf)
    except Exception as e:
        return ValidationResult(ok=False, message=f"Could not load model file: {e}"), None

    # Unwrap DriftInsights-style payload dict
    if isinstance(payload, dict) and "model" in payload:
        estimator = payload["model"]
    else:
        estimator = payload

    # Must support predict
    if not hasattr(estimator, "predict"):
        return ValidationResult(
            ok=False,
            message="Loaded object does not have a predict() method. "
                    "Please upload a fitted sklearn-compatible estimator.",
        ), None

    model_type = _infer_model_type(estimator)
    return ValidationResult(
        ok=True,
        message=f"Model loaded successfully ({model_type}).",
        details={"model_class": type(estimator).__name__, "model_type": model_type},
    ), estimator


def _validate_csv(
    data_bytes: bytes, filename: str, role: str
) -> tuple[ValidationResult, Optional[pd.DataFrame]]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_DATA_SUFFIXES:
        return ValidationResult(
            ok=False,
            message=f"Unsupported {role} dataset format '{suffix}'. Use .csv.",
        ), None

    try:
        df = pd.read_csv(io.BytesIO(data_bytes))
    except Exception as e:
        return ValidationResult(ok=False, message=f"Could not parse {role} CSV: {e}"), None

    if len(df) < 10:
        return ValidationResult(
            ok=False,
            message=f"{role.capitalize()} dataset is too small ({len(df)} rows). Need at least 10.",
        ), None

    return ValidationResult(
        ok=True,
        message=f"{role.capitalize()} dataset loaded ({len(df):,} rows, {len(df.columns)} columns).",
        details={"rows": len(df), "columns": list(df.columns)},
    ), df


def _validate_feature_compatibility(
    estimator: Any,
    df_hist: pd.DataFrame,
    df_curr: pd.DataFrame,
    label_column: str,
) -> tuple[ValidationResult, list[str], list[str]]:
    warnings: list[str] = []

    # Determine feature columns (everything except label if present)
    hist_cols = [c for c in df_hist.columns if c != label_column]
    curr_cols = [c for c in df_curr.columns if c != label_column]

    # Try to get expected feature names from the estimator
    expected_features: Optional[list[str]] = None
    for attr in ("feature_names_in_", "feature_name_", "feature_names"):
        val = getattr(estimator, attr, None)
        if val is not None:
            try:
                expected_features = list(val)
            except Exception:
                pass
            break

    if expected_features is not None:
        missing_hist = set(expected_features) - set(hist_cols)
        missing_curr = set(expected_features) - set(curr_cols)
        if missing_hist:
            return ValidationResult(
                ok=False,
                message=f"Historical dataset is missing model features: {sorted(missing_hist)}",
            ), [], warnings
        if missing_curr:
            return ValidationResult(
                ok=False,
                message=f"Current dataset is missing model features: {sorted(missing_curr)}",
            ), [], warnings
        feature_names = expected_features
        extra_hist = set(hist_cols) - set(expected_features)
        extra_curr = set(curr_cols) - set(expected_features)
        if extra_hist:
            warnings.append(f"Historical dataset has extra columns (will be ignored): {sorted(extra_hist)}")
        if extra_curr:
            warnings.append(f"Current dataset has extra columns (will be ignored): {sorted(extra_curr)}")
    else:
        # No feature names from model — use intersection of both datasets
        common = [c for c in hist_cols if c in set(curr_cols)]
        if not common:
            return ValidationResult(
                ok=False,
                message="Historical and current datasets share no common feature columns.",
            ), [], warnings
        if set(hist_cols) != set(curr_cols):
            warnings.append(
                f"Column mismatch between datasets. Using intersection: {common}"
            )
        feature_names = common

    # Check numeric dtype compatibility
    for name in feature_names[:5]:  # spot-check first 5
        if name in df_hist.columns and not pd.api.types.is_numeric_dtype(df_hist[name]):
            warnings.append(f"Feature '{name}' is non-numeric in historical dataset — will be label-encoded.")

    return ValidationResult(
        ok=True,
        message=f"Feature compatibility verified ({len(feature_names)} features).",
        details={"feature_names": feature_names},
    ), feature_names, warnings


def _validate_prediction(
    estimator: Any, df: pd.DataFrame, feature_names: list[str]
) -> ValidationResult:
    try:
        X_sample = df[feature_names].head(5).values.astype(float)
        preds = estimator.predict(X_sample)
        return ValidationResult(
            ok=True,
            message=f"Prediction smoke-test passed (5 samples → {list(preds)[:3]}...).",
        )
    except Exception as e:
        return ValidationResult(
            ok=False,
            message=f"Model prediction failed: {e}. "
                    "Ensure the model is compatible with the provided feature schema.",
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_xy(
    df: pd.DataFrame, feature_names: list[str], label_column: str
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    X = df[feature_names].values.astype(float)
    y = df[label_column].values if label_column in df.columns else None
    return X, y


def _infer_model_type(estimator: Any) -> str:
    cls = type(estimator).__name__.lower()
    if "xgb" in cls:
        return "xgboost"
    if "lgbm" in cls or "lightgbm" in cls:
        return "lightgbm"
    if "randomforest" in cls:
        return "random_forest"
    return "uploaded"
