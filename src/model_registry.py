"""
DriftInsights — Model Validation & Hot-Swap Registry

Validates candidate M_{t+1}, performs an atomic pointer swap, resets ADWIN,
refreshes the baseline SHAP reference, and appends audit JSON metadata.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

from src.adaptation_engine import AdaptationCandidate
from src.drift_diagnostic import DiagnosisResult, DriftDiagnostic
from src.pipeline_config import PipelineManager, _cfg_get
from src.utils.helpers import ensure_directory, timestamp_now

logger = logging.getLogger("driftinsights.model_registry")


class ModelRegistry:
    """Production pointer for the active model M_t plus audit trail."""

    def __init__(
        self,
        manager: PipelineManager,
        diagnostic: DriftDiagnostic,
        audit_path: str | Path | None = None,
    ) -> None:
        self.manager = manager
        self.diagnostic = diagnostic
        self._lock = threading.Lock()

        self._accuracy_threshold = float(
            _cfg_get(manager.config, "adaptation.validation.accuracy_threshold", 0.75)
        )
        self._tolerance = float(
            _cfg_get(manager.config, "adaptation.validation.tolerance", 0.05)
        )
        save_dir = Path(_cfg_get(manager.config, "model.save_dir", "models"))
        log_dir = Path(_cfg_get(manager.config, "logging.dir", "logs"))
        ensure_directory(save_dir)
        ensure_directory(log_dir)

        self.save_dir = save_dir
        self.audit_path = Path(audit_path) if audit_path else log_dir / "audit.json"
        self.hot_swap_count = 0
        self.last_validation: Optional[dict] = None

        if not self.audit_path.exists():
            self._write_audit([])

    def validate(
        self,
        candidate: AdaptationCandidate,
        X_holdout: np.ndarray,
        y_holdout: np.ndarray,
    ) -> dict:
        """Score candidate M_{t+1} against a recent held-out sample batch."""
        if len(X_holdout) == 0:
            raise ValueError("Held-out validation batch is empty.")

        y_new = np.asarray(candidate.estimator.predict(X_holdout))
        new_acc = float(accuracy_score(y_holdout, y_new))

        old_acc = new_acc
        if self.manager.model is not None:
            y_old = np.asarray(self.manager.model.predict(X_holdout))
            old_acc = float(accuracy_score(y_holdout, y_old))

        passes_floor = new_acc >= self._accuracy_threshold
        passes_tolerance = new_acc >= (old_acc - self._tolerance)
        accepted = passes_floor and passes_tolerance
        recovery = new_acc - old_acc

        result = {
            "accepted": accepted,
            "old_accuracy": round(old_acc, 6),
            "new_accuracy": round(new_acc, 6),
            "accuracy_recovery": round(recovery, 6),
            "threshold": self._accuracy_threshold,
            "tolerance": self._tolerance,
            "n_holdout": int(len(y_holdout)),
        }
        self.last_validation = result
        logger.info(
            "Validation %s — old=%.4f new=%.4f recovery=%+.4f",
            "PASS" if accepted else "FAIL",
            old_acc,
            new_acc,
            recovery,
        )
        return result

    def hot_swap(self, candidate: AdaptationCandidate) -> int:
        """Atomic reference pointer swap M_t -> M_{t+1}."""
        with self._lock:
            self.manager.model = candidate.estimator
            self.manager.model_version += 1
            version = self.manager.model_version
            self.hot_swap_count += 1

        payload = {
            "model": candidate.estimator,
            "model_type": self.manager.model_type,
            "feature_names": self.manager.feature_names,
            "version": version,
            "strategy": candidate.strategy,
        }
        path = self.save_dir / f"model_{self.manager.model_type}_v{version}.joblib"
        joblib.dump(payload, path)
        logger.info("Hot-swapped M_t -> M_{t+1} (v%s) saved to %s", version, path)
        return version

    def commit(
        self,
        candidate: AdaptationCandidate,
        diagnosis: DiagnosisResult,
        X_holdout: np.ndarray,
        y_holdout: np.ndarray,
        X_after: np.ndarray | None = None,
    ) -> dict:
        """
        Validate, optionally hot-swap, reset ADWIN, refresh W_before, and audit.
        """
        validation = self.validate(candidate, X_holdout, y_holdout)
        record = {
            "timestamp": timestamp_now(),
            "drift_type": candidate.drift_type.value,
            "strategy": candidate.strategy,
            "sample_idx": diagnosis.sample_idx,
            "delta_e_total": diagnosis.delta_e_total,
            "top_delta_e_features": [f.to_dict() for f in diagnosis.top_features],
            "accuracy_recovery": validation["accuracy_recovery"],
            "old_accuracy": validation["old_accuracy"],
            "new_accuracy": validation["new_accuracy"],
            "accepted": validation["accepted"],
            "model_version": self.manager.model_version,
        }

        if not validation["accepted"]:
            record["action"] = "rejected"
            self.append_audit(record)
            return record

        version = self.hot_swap(candidate)
        self.diagnostic.reset()
        if X_after is not None and len(X_after) > 0:
            self.manager.update_baseline_shap_from_window(X_after)
        else:
            self.manager.compute_baseline_shap()

        record["action"] = "hot_swap"
        record["model_version"] = version
        self.append_audit(record)
        return record

    def append_audit(self, record: dict) -> None:
        """Append execution metadata to the audit JSON log."""
        entries = self._read_audit()
        entries.append(record)
        self._write_audit(entries)
        logger.info("Audit log appended (%s events) -> %s", len(entries), self.audit_path)

    def _read_audit(self) -> list[dict]:
        if not self.audit_path.exists():
            return []
        try:
            with open(self.audit_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    def _write_audit(self, entries: list[dict]) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.audit_path, "w", encoding="utf-8") as handle:
            json.dump(entries, handle, indent=2, default=str)
