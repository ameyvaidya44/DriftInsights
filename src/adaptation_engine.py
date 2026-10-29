"""
DriftInsights — Decision Engine & Adaptation Module

Rule-based mapping of explanation-shift magnitude to adaptation strategy:

  MINOR    → exponential age-decay reweight of baseline samples
  MODERATE → warm-start incremental update on W_after
  SEVERE   → full retrain on baseline + stream buffer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from src.pipeline_config import PipelineManager, _cfg_get

logger = logging.getLogger("driftinsights.adaptation_engine")


class DriftType(str, Enum):
    MINOR = "MINOR"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"


@dataclass
class AdaptationCandidate:
    estimator: Any
    drift_type: DriftType
    strategy: str
    n_samples_used: int
    sample_weight_used: bool = False


class AdaptationEngine:
    """Rule-based adaptation producing a candidate model M_{t+1}."""

    def __init__(self, manager: PipelineManager) -> None:
        self.manager = manager
        config = manager.config

        self._minor = float(_cfg_get(config, "decision.thresholds.minor", 0.05))
        self._moderate = float(_cfg_get(config, "decision.thresholds.moderate", 0.15))
        self._severe = float(_cfg_get(config, "decision.thresholds.severe", 0.30))
        self._decay = float(_cfg_get(config, "adaptation.reweight.decay_factor", 0.95))
        self._inc_window = int(_cfg_get(config, "adaptation.incremental.window_size", 1000))
        self._warm_start = bool(_cfg_get(config, "adaptation.incremental.warm_start", True))
        self._full_min = int(_cfg_get(config, "adaptation.full_retrain.min_samples", 200))

        logger.info(
            "AdaptationEngine thresholds: minor=%.3f moderate=%.3f severe=%.3f",
            self._minor,
            self._moderate,
            self._severe,
        )

    def classify(self, delta_e_total: float) -> DriftType:
        if delta_e_total >= self._severe:
            return DriftType.SEVERE
        if delta_e_total >= self._moderate:
            return DriftType.MODERATE
        return DriftType.MINOR

    def execute(
        self,
        delta_e_total: float,
        X_after: np.ndarray,
        y_after: np.ndarray,
    ) -> AdaptationCandidate:
        drift_type = self.classify(delta_e_total)
        logger.info("Decision: %s drift (Delta_E_total=%.4f)", drift_type.value, delta_e_total)

        if drift_type == DriftType.MINOR:
            return self._sample_reweight()
        if drift_type == DriftType.MODERATE:
            return self._incremental_retrain(X_after, y_after)
        return self._full_retrain(X_after, y_after)

    def _sample_reweight(self) -> AdaptationCandidate:
        """MINOR: exponential age-decay weights on baseline samples."""
        X, y = self._require_baseline()
        n = len(X)
        weights = np.array([self._decay ** (n - 1 - i) for i in range(n)], dtype=float)
        weights *= n / weights.sum()

        estimator = self.manager.create_fresh_estimator()
        estimator.fit(X, y, sample_weight=weights)
        logger.info("MINOR adaptation: reweighted %d baseline samples (decay=%.3f)", n, self._decay)
        return AdaptationCandidate(
            estimator=estimator,
            drift_type=DriftType.MINOR,
            strategy="sample_reweight",
            n_samples_used=n,
            sample_weight_used=True,
        )

    def _incremental_retrain(self, X_after: np.ndarray, y_after: np.ndarray) -> AdaptationCandidate:
        """MODERATE: warm-start update using the post-drift window W_after."""
        window = min(self._inc_window, len(X_after))
        if window == 0:
            raise ValueError("W_after is empty — cannot run incremental retrain.")

        X_w, y_w = X_after[-window:], y_after[-window:]
        current = self.manager.model
        estimator = self.manager.create_fresh_estimator()

        fitted = False
        if self._warm_start and current is not None:
            fitted = self._try_warm_start(estimator, current, X_w, y_w)

        if not fitted:
            if self._warm_start and hasattr(estimator, "warm_start"):
                estimator.warm_start = True
            estimator.fit(X_w, y_w)

        logger.info("MODERATE adaptation: incremental retrain on %d W_after samples", window)
        return AdaptationCandidate(
            estimator=estimator,
            drift_type=DriftType.MODERATE,
            strategy="incremental_retrain",
            n_samples_used=window,
        )

    def _full_retrain(self, X_after: np.ndarray, y_after: np.ndarray) -> AdaptationCandidate:
        """SEVERE: train a new model from scratch on baseline + stream buffer."""
        X_base, y_base = self._require_baseline()
        if len(X_after) == 0:
            X_all, y_all = X_base, y_base
        else:
            X_all = np.vstack([X_base, X_after])
            y_all = np.concatenate([y_base, y_after])

        if len(X_all) < self._full_min:
            logger.warning(
                "SEVERE full retrain on %d samples (configured min=%d); proceeding anyway",
                len(X_all),
                self._full_min,
            )

        estimator = self.manager.create_fresh_estimator()
        estimator.fit(X_all, y_all)
        logger.info("SEVERE adaptation: full retrain on %d combined samples", len(X_all))
        return AdaptationCandidate(
            estimator=estimator,
            drift_type=DriftType.SEVERE,
            strategy="full_retrain",
            n_samples_used=len(X_all),
        )

    def _try_warm_start(self, estimator: Any, current: Any, X: np.ndarray, y: np.ndarray) -> bool:
        try:
            if self.manager.model_type == "xgboost" and hasattr(current, "get_booster"):
                estimator.fit(X, y, xgb_model=current.get_booster())
                return True
            if self.manager.model_type == "lightgbm" and hasattr(current, "booster_"):
                estimator.fit(X, y, init_model=current.booster_)
                return True
        except Exception as exc:
            logger.debug("Warm-start continuation failed (%s); training W_after from scratch", exc)
        return False

    def _require_baseline(self) -> tuple[np.ndarray, np.ndarray]:
        if self.manager.X_base is None or self.manager.y_base is None:
            raise RuntimeError("Baseline data is required for adaptation.")
        return self.manager.X_base, self.manager.y_base
