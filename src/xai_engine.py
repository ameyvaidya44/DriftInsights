"""
DriftInsights — XAI Engine (Enhanced)

SHAP-based explainability for concept drift diagnosis.
Computes feature importance shifts (ΔE) between pre-drift and post-drift windows,
including per-feature direction of change and distribution statistics.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("driftinsights.xai_engine")


@dataclass
class FeatureDriftDetail:
    """Per-feature drift analysis with direction and distribution statistics."""
    feature: str
    delta: float                  # |SHAP_before - SHAP_after|
    direction: str                # "increase" if SHAP_after > SHAP_before, else "decrease"
    mean_before: float            # Mean feature value in pre-drift window
    mean_after: float             # Mean feature value in post-drift window
    std_before: float             # Std dev in pre-drift window
    std_after: float              # Std dev in post-drift window
    shap_before: float = 0.0     # Mean |SHAP| before drift
    shap_after: float = 0.0      # Mean |SHAP| after drift

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftExplanation:
    """
    Explanation of why drift occurred, based on SHAP value changes.

    Enhanced output format:
    {
      "delta_e_total": float,
      "direction": "increase/decrease/mixed",
      "top_features": [FeatureDriftDetail, ...],
      "pre_importances": {...},
      "post_importances": {...},
    }
    """
    top_features: list[FeatureDriftDetail]    # Ranked by |delta|
    delta_total: float                         # L2 norm of all deltas
    direction: str                             # "increase" | "decrease" | "mixed"
    pre_importances: dict[str, float] = field(default_factory=dict)
    post_importances: dict[str, float] = field(default_factory=dict)
    all_deltas: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "delta_e_total": self.delta_total,
            "direction": self.direction,
            "top_features": [f.to_dict() for f in self.top_features],
            "pre_importances": self.pre_importances,
            "post_importances": self.post_importances,
        }


class XAIEngine:
    """
    SHAP-based explanation engine for concept drift.

    Uses TreeExplainer to compute SHAP values for pre-drift and post-drift
    data windows, then calculates the feature importance shift (ΔE) to
    identify which features drove the drift.

    Enhanced Process (v2):
      1. Compute mean |SHAP| per feature for pre-drift window
      2. Compute mean |SHAP| per feature for post-drift window
      3. ΔE_i = |mean_pre_i - mean_post_i| for each feature
      4. ΔE_total = L2 norm of the ΔE vector
      5. Per-feature: direction, distribution stats (mean, std)
      6. Rank features by ΔE_i to identify drift drivers

    Usage:
        xai = XAIEngine(config)
        xai.set_model(model.estimator)
        explanation = xai.explain_drift(X_pre, X_post, feature_names)
    """

    def __init__(self, config):
        self.config = config
        self._explainer = None
        self._model = None

        # Config
        xai_cfg = config.xai
        self._max_samples = xai_cfg.shap.max_samples
        self._top_k = xai_cfg.shap.top_k_features
        self._window_size = xai_cfg.window_size

        # Track explanations for feature drift summary (Bonus: Improvement 7)
        self._explanation_history: list[dict] = []

        logger.info(
            f"XAIEngine initialized — max_samples={self._max_samples}, "
            f"top_k={self._top_k}, window_size={self._window_size}"
        )

    def set_model(self, model: Any) -> None:
        """
        Set or update the model for SHAP explanations.

        Creates a TreeExplainer for tree-based models (XGBoost, LightGBM, RF).

        Args:
            model: A fitted tree-based estimator
        """
        import shap

        self._model = model
        try:
            self._explainer = shap.TreeExplainer(model)
            logger.info("SHAP TreeExplainer created successfully")
        except Exception as e:
            logger.warning(f"TreeExplainer failed, falling back to KernelExplainer: {e}")
            self._explainer = None

    def explain_drift(
        self,
        X_pre: np.ndarray,
        X_post: np.ndarray,
        feature_names: list[str],
    ) -> DriftExplanation:
        """
        Explain drift by comparing SHAP values between pre and post windows.

        Enhanced with per-feature direction and distribution statistics.

        Args:
            X_pre: Pre-drift feature matrix
            X_post: Post-drift feature matrix
            feature_names: List of feature names

        Returns:
            DriftExplanation with ranked feature deltas, directions, and stats
        """
        import shap

        if self._explainer is None and self._model is not None:
            self.set_model(self._model)

        if self._explainer is None:
            raise RuntimeError("No SHAP explainer available. Call set_model() first.")

        # Subsample if windows are too large (efficiency optimization)
        X_pre_sample = self._subsample(X_pre)
        X_post_sample = self._subsample(X_post)

        logger.info(
            f"Computing SHAP explanations — "
            f"pre: {X_pre_sample.shape[0]} samples, "
            f"post: {X_post_sample.shape[0]} samples"
        )

        # Compute SHAP values for both windows
        shap_pre = self._compute_shap_values(X_pre_sample)
        shap_post = self._compute_shap_values(X_post_sample)

        # Mean absolute SHAP per feature (importance measure)
        mean_abs_pre = np.mean(np.abs(shap_pre), axis=0)
        mean_abs_post = np.mean(np.abs(shap_post), axis=0)

        # Ensure feature_names matches dimensionality
        n_features = min(len(feature_names), len(mean_abs_pre), len(mean_abs_post))
        feature_names = feature_names[:n_features]
        mean_abs_pre = mean_abs_pre[:n_features]
        mean_abs_post = mean_abs_post[:n_features]

        # Compute importance deltas
        deltas = np.abs(mean_abs_pre - mean_abs_post)

        # ΔE_total = L2 norm of the delta vector
        delta_total = float(np.linalg.norm(deltas))

        # Build per-feature detailed analysis
        # Direction: "increase" if SHAP importance grew after drift, else "decrease"
        signed_changes = mean_abs_post - mean_abs_pre

        # Distribution stats: compute mean/std of raw feature values
        feature_details = []
        for i, fname in enumerate(feature_names):
            direction = "increase" if mean_abs_post[i] > mean_abs_pre[i] else "decrease"
            detail = FeatureDriftDetail(
                feature=fname,
                delta=round(float(deltas[i]), 6),
                direction=direction,
                # Distribution statistics of raw feature values (not SHAP)
                mean_before=round(float(np.mean(X_pre_sample[:, i])), 6),
                mean_after=round(float(np.mean(X_post_sample[:, i])), 6),
                std_before=round(float(np.std(X_pre_sample[:, i])), 6),
                std_after=round(float(np.std(X_post_sample[:, i])), 6),
                # SHAP importances
                shap_before=round(float(mean_abs_pre[i]), 6),
                shap_after=round(float(mean_abs_post[i]), 6),
            )
            feature_details.append(detail)

        # Sort by delta descending and take top-k
        feature_details.sort(key=lambda f: f.delta, reverse=True)
        top_features = feature_details[:self._top_k]

        # Build importance dicts for backward compatibility
        pre_importances = {f: round(float(v), 6) for f, v in zip(feature_names, mean_abs_pre)}
        post_importances = {f: round(float(v), 6) for f, v in zip(feature_names, mean_abs_post)}
        all_deltas = {f: round(float(d), 6) for f, d in zip(feature_names, deltas)}

        # Determine overall direction
        n_increased = np.sum(signed_changes > 0)
        n_decreased = np.sum(signed_changes < 0)
        if n_increased > n_decreased * 2:
            direction = "increase"
        elif n_decreased > n_increased * 2:
            direction = "decrease"
        else:
            direction = "mixed"

        explanation = DriftExplanation(
            top_features=top_features,
            delta_total=round(delta_total, 6),
            direction=direction,
            pre_importances=pre_importances,
            post_importances=post_importances,
            all_deltas=all_deltas,
        )

        # Store in history for feature drift summary (Bonus)
        self._explanation_history.append(explanation.to_dict())

        logger.info(
            f"Drift explanation: delta_e_total={delta_total:.4f}, "
            f"direction={direction}, "
            f"top_feature={top_features[0].feature} "
            f"(delta={top_features[0].delta:.4f}, dir={top_features[0].direction})"
        )

        return explanation

    def get_feature_drift_summary(self, last_n: int = 5) -> dict:
        """
        Bonus: Summarize top drifting features across the last N drift events.

        Returns:
            dict with feature names mapped to cumulative drift delta
        """
        if not self._explanation_history:
            return {"summary": "No drift events recorded yet"}

        recent = self._explanation_history[-last_n:]
        feature_cumulative: dict[str, float] = {}

        for explanation in recent:
            for feat in explanation.get("top_features", []):
                name = feat["feature"]
                feature_cumulative[name] = feature_cumulative.get(name, 0.0) + feat["delta"]

        # Sort by cumulative delta
        ranked = sorted(feature_cumulative.items(), key=lambda x: x[1], reverse=True)

        return {
            "events_analyzed": len(recent),
            "top_drifting_features": [
                {"feature": name, "cumulative_delta": round(d, 6)} for name, d in ranked
            ],
        }

    def _compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values, handling multi-output models."""
        shap_values = self._explainer.shap_values(X)

        # Handle multi-output (e.g., binary classification returns list of 2)
        if isinstance(shap_values, list):
            # Use class 1 SHAP values for binary classification
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

        return np.array(shap_values)

    def _subsample(self, X: np.ndarray) -> np.ndarray:
        """Subsample data to max_samples for efficiency."""
        if len(X) > self._max_samples:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(X), self._max_samples, replace=False)
            return X[indices]
        return X

    @property
    def window_size(self) -> int:
        """Window size for pre/post drift data."""
        return self._window_size

    @property
    def explanation_history(self) -> list[dict]:
        """History of all drift explanations."""
        return self._explanation_history.copy()
