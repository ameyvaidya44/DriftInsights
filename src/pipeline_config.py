"""
DriftInsights — Configuration & Initialization Setup Module

PipelineManager ingests baseline data, trains or loads M_0, and computes
the baseline SHAP reference distribution (W_before).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import Config
from src.utils.helpers import timestamp_now

logger = logging.getLogger("driftinsights.pipeline_config")


def generate_synthetic_drift_dataset(
    n_baseline: int = 2000,
    n_stream: int = 3000,
    drift_in_stream: int = 800,
    noise: float = 0.08,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a SEA-style synthetic pair: stable baseline + streaming data with
    an abrupt concept change at `drift_in_stream`.
    """
    rng = np.random.RandomState(random_state)
    n_total = n_baseline + n_stream
    X = rng.uniform(0, 10, size=(n_total, 4))
    y = np.zeros(n_total, dtype=int)

    y[:n_baseline] = (X[:n_baseline, 0] + X[:n_baseline, 1] <= 8).astype(int)
    stream_start = n_baseline
    drift_at = n_baseline + drift_in_stream
    y[stream_start:drift_at] = (
        X[stream_start:drift_at, 0] + X[stream_start:drift_at, 1] <= 8
    ).astype(int)
    y[drift_at:] = (X[drift_at:, 0] + X[drift_at:, 2] <= 9).astype(int)

    noise_mask = rng.random(n_total) < noise
    y[noise_mask] = 1 - y[noise_mask]

    columns = ["feature_0", "feature_1", "feature_2", "feature_3"]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y
    return df.iloc[:n_baseline].reset_index(drop=True), df.iloc[n_baseline:].reset_index(drop=True)


def _cfg_get(config: Optional[Config], dotted: str, default: Any) -> Any:
    """Read a dotted config path, returning default when missing."""
    if config is None:
        return default
    node: Any = config
    for part in dotted.split("."):
        try:
            if isinstance(node, dict):
                node = node[part]
            else:
                node = getattr(node, part)
        except Exception:
            return default
    return node.to_dict() if hasattr(node, "to_dict") else node


class PipelineManager:
    """
    Project setup for the explainable drift-aware pipeline.

    Methods:
      load_baseline_data(df_base, target_column)
      set_base_model(model_or_path)
      compute_baseline_shap()
    """

    def __init__(
        self,
        config: Config | None = None,
        model_type: str | None = None,
    ) -> None:
        self.config = config
        self.model_type = model_type or str(_cfg_get(config, "model.type", "xgboost"))
        self.target_column: str = "target"
        self.feature_names: list[str] = []

        self.scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self._fitted = False

        self.X_base: Optional[np.ndarray] = None
        self.y_base: Optional[np.ndarray] = None
        self.X_holdout: Optional[np.ndarray] = None
        self.y_holdout: Optional[np.ndarray] = None

        self.model: Any = None  # M_t (initially M_0)
        self.model_version: int = 0

        # W_before: mean |SHAP| per feature on historical baseline data
        self.shap_before: Optional[np.ndarray] = None
        self.shap_rankings: list[tuple[str, float]] = []
        self.baseline_shap_computed_at: Optional[str] = None

        self._rng = np.random.RandomState(42)

    # ── Phase 1.1: Baseline ingestion ─────────────────────────────────────

    def load_baseline_data(
        self,
        df_base: pd.DataFrame,
        target_column: str,
        holdout_frac: float = 0.15,
    ) -> "PipelineManager":
        """
        Ingest baseline dataset, preprocess features, and identify feature names.

        A chronological holdout slice is reserved for candidate-model validation.
        """
        if target_column not in df_base.columns:
            raise ValueError(
                f"target_column '{target_column}' not in columns: {list(df_base.columns)}"
            )

        self.target_column = target_column
        df = self._preprocess(df_base, target_column, fit=True)

        y = df[target_column].to_numpy()
        X = df.drop(columns=[target_column]).to_numpy(dtype=float)
        self.feature_names = [c for c in df.columns if c != target_column]

        n = len(X)
        if n < 10:
            raise ValueError("Baseline dataset is too small (need at least 10 rows).")

        holdout_n = max(1, int(n * holdout_frac))
        split = n - holdout_n
        self.X_base, self.X_holdout = X[:split], X[split:]
        self.y_base, self.y_holdout = y[:split], y[split:]

        logger.info(
            "Baseline loaded — n=%d features=%s holdout=%d",
            len(self.X_base),
            self.feature_names,
            len(self.X_holdout),
        )
        return self

    def transform_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        """Apply fitted preprocessing to a streaming frame. Returns (X, y_or_None)."""
        if not self._fitted:
            raise RuntimeError("Call load_baseline_data() before transforming stream data.")

        has_target = self.target_column in df.columns
        df_work = df.copy()
        if not has_target:
            df_work[self.target_column] = 0

        processed = self._preprocess(df_work, self.target_column, fit=False)
        y = processed[self.target_column].to_numpy() if has_target else None
        X = processed.drop(columns=[self.target_column]).to_numpy(dtype=float)
        return X, y

    # ── Phase 1.2: Base model M_0 ─────────────────────────────────────────

    def set_base_model(self, model_or_path: Any | str | Path | None = None) -> "PipelineManager":
        """
        Accept a pre-trained sklearn/XGBoost/LightGBM model (or joblib path),
        or train a default classifier on the baseline dataset (M_0).
        """
        if self.X_base is None or self.y_base is None:
            raise RuntimeError("Call load_baseline_data() before set_base_model().")

        if model_or_path is None:
            self.model = self._create_estimator()
            self.model.fit(self.X_base, self.y_base)
            self.model_version = 1
            logger.info("Trained default %s classifier as M_0", self.model_type)
            return self

        if isinstance(model_or_path, (str, Path)):
            path = Path(model_or_path)
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")
            payload = joblib.load(path)
            self.model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
            self.model_type = (
                payload.get("model_type", self.model_type)
                if isinstance(payload, dict)
                else self.model_type
            )
            self.model_version = int(payload.get("version", 1)) if isinstance(payload, dict) else 1
            logger.info("Loaded pre-trained model from %s", path)
            return self

        self.model = model_or_path
        self.model_version = 1
        logger.info("Attached external estimator as M_0 (%s)", type(model_or_path).__name__)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("No active model. Call set_base_model() first.")
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("No active model. Call set_base_model() first.")
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Active model does not support predict_proba.")
        return np.asarray(self.model.predict_proba(X))

    # ── Phase 1.3: Baseline SHAP W_before ─────────────────────────────────

    def compute_baseline_shap(self, max_samples: int | None = None) -> dict[str, float]:
        """
        Compute baseline SHAP values (W_before) on historical baseline data
        and store feature importance rankings.
        """
        import shap

        if self.model is None or self.X_base is None:
            raise RuntimeError("Need baseline data and M_0 before computing SHAP.")

        max_samples = max_samples or int(_cfg_get(self.config, "xai.shap.max_samples", 200))
        X_ref = self._subsample(self.X_base, max_samples)

        explainer = shap.TreeExplainer(self.model)
        shap_values = self._normalize_shap(explainer.shap_values(X_ref))
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        n_features = min(len(self.feature_names), len(mean_abs))
        self.shap_before = np.asarray(mean_abs[:n_features], dtype=float)

        self.shap_rankings = sorted(
            [
                (self.feature_names[i], float(self.shap_before[i]))
                for i in range(n_features)
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        self.baseline_shap_computed_at = timestamp_now()

        ranking_preview = ", ".join(f"{n}={v:.4f}" for n, v in self.shap_rankings[:5])
        logger.info("Baseline SHAP (W_before) computed — top: %s", ranking_preview)
        return {name: score for name, score in self.shap_rankings}

    def update_baseline_shap_from_window(self, X_window: np.ndarray, max_samples: int | None = None) -> None:
        """Replace W_before with SHAP of the current active model on a new reference window."""
        import shap

        if self.model is None:
            raise RuntimeError("No active model for SHAP update.")

        max_samples = max_samples or int(_cfg_get(self.config, "xai.shap.max_samples", 200))
        X_ref = self._subsample(X_window, max_samples)
        explainer = shap.TreeExplainer(self.model)
        shap_values = self._normalize_shap(explainer.shap_values(X_ref))
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        n_features = min(len(self.feature_names), len(mean_abs))
        self.shap_before = np.asarray(mean_abs[:n_features], dtype=float)
        self.shap_rankings = sorted(
            [
                (self.feature_names[i], float(self.shap_before[i]))
                for i in range(n_features)
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        self.baseline_shap_computed_at = timestamp_now()
        logger.info("Updated baseline SHAP reference after hot-swap")

    # ── Estimators ────────────────────────────────────────────────────────

    def create_fresh_estimator(self) -> Any:
        return self._create_estimator()

    def _create_estimator(self) -> Any:
        if self.model_type == "lightgbm":
            from lightgbm import LGBMClassifier

            params = _cfg_get(self.config, "model.params.lightgbm", {}) or {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
                "verbose": -1,
            }
            return LGBMClassifier(**params)

        if self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            params = _cfg_get(self.config, "model.params.random_forest", {}) or {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
            }
            return RandomForestClassifier(**params)

        from xgboost import XGBClassifier

        params = dict(
            _cfg_get(self.config, "model.params.xgboost", {})
            or {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "eval_metric": "logloss",
                "random_state": 42,
            }
        )
        params.pop("use_label_encoder", None)
        return XGBClassifier(**params)

    # ── Preprocessing ─────────────────────────────────────────────────────

    def _preprocess(self, df: pd.DataFrame, target_column: str, fit: bool) -> pd.DataFrame:
        df = df.copy()
        target = df[target_column].copy()
        features = df.drop(columns=[target_column])

        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = features.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in numeric_cols:
            if features[col].isnull().any():
                fill_val = features[col].median()
                features[col] = features[col].fillna(fill_val)

        for col in categorical_cols:
            if features[col].isnull().any():
                mode = features[col].mode()
                fill_val = mode[0] if len(mode) else "MISSING"
                features[col] = features[col].fillna(fill_val)

        for col in categorical_cols:
            if fit:
                encoder = LabelEncoder()
                features[col] = encoder.fit_transform(features[col].astype(str))
                self.label_encoders[col] = encoder
            else:
                encoder = self.label_encoders.get(col)
                if encoder is None:
                    encoder = LabelEncoder()
                    features[col] = encoder.fit_transform(features[col].astype(str))
                else:
                    features[col] = features[col].astype(str).map(
                        lambda x, _le=encoder: (
                            _le.transform([x])[0] if x in _le.classes_ else -1
                        )
                    )

        if not np.issubdtype(target.dtype, np.number):
            if fit:
                encoder = LabelEncoder()
                target = pd.Series(encoder.fit_transform(target.astype(str)), name=target_column)
                self.label_encoders["__target__"] = encoder
            else:
                encoder = self.label_encoders.get("__target__")
                if encoder is not None:
                    target = pd.Series(
                        encoder.transform(target.astype(str)),
                        name=target_column,
                    )

        columns = features.columns.tolist()
        if fit:
            features = pd.DataFrame(
                self.scaler.fit_transform(features),
                columns=columns,
                index=features.index,
            )
            self._fitted = True
        else:
            features = pd.DataFrame(
                self.scaler.transform(features),
                columns=columns,
                index=features.index,
            )

        features[target_column] = target.to_numpy()
        return features

    def _subsample(self, X: np.ndarray, max_samples: int) -> np.ndarray:
        if len(X) <= max_samples:
            return X
        idx = self._rng.choice(len(X), max_samples, replace=False)
        return X[idx]

    @staticmethod
    def _normalize_shap(shap_values: Any) -> np.ndarray:
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            arr = arr[:, :, 1] if arr.shape[-1] >= 2 else arr[:, :, 0]
        return arr
