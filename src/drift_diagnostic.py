"""
DriftInsights — Monitoring & Diagnostic Engine

Monitors stream error rate with River ADWIN. On a drift trigger, computes
post-drift SHAP (W_after) and feature-level explanation-shift scores:

    Delta_E_i = | SHAP_after(i) - SHAP_before(i) |
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
from river.drift import ADWIN

from src.pipeline_config import PipelineManager, _cfg_get
from src.utils.helpers import timestamp_now

logger = logging.getLogger("driftinsights.drift_diagnostic")


@dataclass
class FeatureShift:
    feature: str
    shap_before: float
    shap_after: float
    delta_e: float
    direction: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DiagnosisResult:
    timestamp: str
    sample_idx: int
    pre_error_rate: float
    post_error_rate: float
    adwin_width: int
    delta_e_total: float
    top_features: list[FeatureShift] = field(default_factory=list)
    all_deltas: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "sample_idx": self.sample_idx,
            "pre_error_rate": self.pre_error_rate,
            "post_error_rate": self.post_error_rate,
            "adwin_width": self.adwin_width,
            "delta_e_total": self.delta_e_total,
            "top_features": [f.to_dict() for f in self.top_features],
            "all_deltas": self.all_deltas,
        }


class DriftDiagnostic:
    """
    ADWIN error monitor plus SHAP explanation-shift diagnostics.
    """

    def __init__(self, manager: PipelineManager) -> None:
        self.manager = manager
        config = manager.config

        self._delta = float(_cfg_get(config, "drift.adwin.delta", 0.002))
        self._min_gap = int(_cfg_get(config, "drift.min_gap", 200))
        self._warmup = int(_cfg_get(config, "drift.warmup_samples", 100))
        self._top_k = int(_cfg_get(config, "xai.shap.top_k_features", 5))
        self._max_samples = int(_cfg_get(config, "xai.shap.max_samples", 200))
        self._window_size = int(_cfg_get(config, "xai.window_size", 200))

        self._adwin = ADWIN(delta=self._delta)
        self._last_drift_idx = -self._min_gap
        self._detection_count = 0
        self._total_updates = 0
        self._rng = np.random.RandomState(42)

        logger.info(
            "DriftDiagnostic initialized — ADWIN delta=%s min_gap=%s warmup=%s top_k=%s",
            self._delta,
            self._min_gap,
            self._warmup,
            self._top_k,
        )

    def update(self, error: int, sample_idx: int) -> Optional[dict]:
        """
        Feed a binary error (0=correct, 1=incorrect).

        Returns a lightweight ADWIN event dict when drift is detected, else None.
        """
        self._total_updates += 1
        if self._total_updates < self._warmup:
            self._adwin.update(error)
            return None

        pre_estimation = self._adwin.estimation if self._adwin.width > 0 else 0.0
        self._adwin.update(error)

        if not self._adwin.drift_detected:
            return None

        if sample_idx - self._last_drift_idx < self._min_gap:
            logger.debug("Drift suppressed at idx=%s (min_gap)", sample_idx)
            return None

        self._detection_count += 1
        self._last_drift_idx = sample_idx
        post_estimation = self._adwin.estimation if self._adwin.width > 0 else 0.0

        event = {
            "timestamp": timestamp_now(),
            "sample_idx": sample_idx,
            "pre_error_rate": round(float(pre_estimation), 4),
            "post_error_rate": round(float(post_estimation), 4),
            "adwin_width": int(self._adwin.width),
            "detection_count": self._detection_count,
        }
        logger.warning(
            "[DRIFT] ADWIN trigger at sample %s | error %.4f -> %.4f",
            sample_idx,
            event["pre_error_rate"],
            event["post_error_rate"],
        )
        return event

    def diagnose(
        self,
        X_after: np.ndarray,
        adwin_event: dict,
        model: Any | None = None,
    ) -> DiagnosisResult:
        """
        Run TreeExplainer on W_after with active model M_t and compute Delta_E.
        """
        import shap

        model = model if model is not None else self.manager.model
        shap_before = self.manager.shap_before
        if model is None:
            raise RuntimeError("No active model M_t for SHAP diagnosis.")
        if shap_before is None:
            raise RuntimeError("Baseline SHAP (W_before) has not been computed.")
        if len(X_after) == 0:
            raise ValueError("W_after is empty — cannot diagnose explanation shift.")

        X_sample = self._subsample(X_after)
        explainer = shap.TreeExplainer(model)
        shap_after_raw = PipelineManager._normalize_shap(explainer.shap_values(X_sample))
        shap_after = np.mean(np.abs(shap_after_raw), axis=0)

        n_features = min(
            len(self.manager.feature_names),
            len(shap_before),
            len(shap_after),
        )
        names = self.manager.feature_names[:n_features]
        before = np.asarray(shap_before[:n_features], dtype=float)
        after = np.asarray(shap_after[:n_features], dtype=float)
        deltas = np.abs(after - before)

        shifts = []
        all_deltas: dict[str, float] = {}
        for i, name in enumerate(names):
            delta_i = float(deltas[i])
            all_deltas[name] = round(delta_i, 6)
            shifts.append(
                FeatureShift(
                    feature=name,
                    shap_before=round(float(before[i]), 6),
                    shap_after=round(float(after[i]), 6),
                    delta_e=round(delta_i, 6),
                    direction="increase" if after[i] >= before[i] else "decrease",
                )
            )

        shifts.sort(key=lambda s: s.delta_e, reverse=True)
        top = shifts[: self._top_k]
        delta_e_total = float(np.sum([s.delta_e for s in top]))

        result = DiagnosisResult(
            timestamp=adwin_event.get("timestamp", timestamp_now()),
            sample_idx=int(adwin_event.get("sample_idx", -1)),
            pre_error_rate=float(adwin_event.get("pre_error_rate", 0.0)),
            post_error_rate=float(adwin_event.get("post_error_rate", 0.0)),
            adwin_width=int(adwin_event.get("adwin_width", self._adwin.width)),
            delta_e_total=round(delta_e_total, 6),
            top_features=top,
            all_deltas=all_deltas,
        )
        logger.info(
            "Diagnosis Delta_E_total=%.4f (top-%d) | top=%s",
            result.delta_e_total,
            self._top_k,
            top[0].feature if top else "n/a",
        )
        return result

    def reset(self) -> None:
        """Reset ADWIN monitoring state after a successful model hot-swap."""
        self._adwin = ADWIN(delta=self._delta)
        logger.info("ADWIN monitoring state reset")

    def _subsample(self, X: np.ndarray) -> np.ndarray:
        if len(X) <= self._max_samples:
            return X
        idx = self._rng.choice(len(X), self._max_samples, replace=False)
        return X[idx]

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def detection_count(self) -> int:
        return self._detection_count
