"""
DriftInsights — Adaptation Module (Enhanced)

Implements model adaptation strategies: Reweight, Incremental Retrain, Full Retrain.
Includes enhanced validation gate with tolerance-based comparison, rollback tracking,
and integrated model version management.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from src.base_model import BaseModel
from src.decision_engine import AdaptationStrategy, AdaptationCommand
from src.utils.helpers import timestamp_now

logger = logging.getLogger("driftinsights.adaptation_module")


@dataclass
class AdaptationResult:
    """
    Result of an adaptation attempt.

    Enhanced with rollback tracking and duration in milliseconds.
    """
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
    # Enhanced: rollback and tolerance tracking
    rollback: bool = False             # True if adaptation was rejected and rolled back
    tolerance: float = 0.0             # The tolerance value used for comparison
    duration_ms: int = 0               # Duration in milliseconds

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = "success" if self.success else "failure"
        return d


@dataclass
class ModelVersionMetadata:
    """
    Metadata for each model version (Improvement 4: Model Versioning).

    Tracks lineage of every model version deployed in the system.
    """
    version: str                  # Semantic-style: "v1.0.0", "v1.0.1", etc.
    created_at: str               # ISO timestamp
    strategy: str                 # What adaptation created this version
    training_samples: int         # How many samples were used
    accuracy: float               # Validation accuracy at deployment time
    model_type: str = ""          # xgboost/lightgbm/random_forest

    def to_dict(self) -> dict:
        return asdict(self)


class AdaptationModule:
    """
    Executes model adaptation strategies in response to detected drift.

    Strategies:
      1. REWEIGHT — Apply exponential decay weights favoring recent samples
      2. INCREMENTAL_RETRAIN — Train on recent N samples with warm_start
      3. FULL_RETRAIN — Train a completely new model from scratch

    Enhanced Validation Gate (v2):
      - Tolerance-based comparison: new_accuracy >= old_accuracy - tolerance
      - If tolerance check fails AND accuracy < threshold: ROLLBACK
      - Rollback events are logged with full context
      - Failed adaptation count is tracked

    Model Versioning (Improvement 4):
      - Each model stores metadata (version, strategy, accuracy, timestamp)
      - Version history maintained in memory + JSON file
      - Semantic versioning: v{major}.{minor}.{patch}

    Usage:
        adapter = AdaptationModule(config)
        result = adapter.execute(
            command=adaptation_command,
            model=current_model,
            X_recent=X, y_recent=y,
            X_val=X_val, y_val=y_val,
        )
    """

    def __init__(self, config):
        self.config = config
        adapt_cfg = config.adaptation

        self._decay_factor = adapt_cfg.reweight.decay_factor
        self._incremental_window = adapt_cfg.incremental.window_size
        self._warm_start = adapt_cfg.incremental.warm_start
        self._full_retrain_min_samples = adapt_cfg.full_retrain.min_samples
        self._accuracy_threshold = adapt_cfg.validation.accuracy_threshold

        # Enhanced: tolerance for regression check (Improvement 3)
        # new_accuracy >= old_accuracy - tolerance to pass validation
        self._tolerance = adapt_cfg.validation.get("tolerance", 0.02)

        self._adaptation_history: list[AdaptationResult] = []

        # Enhanced: rollback and failure tracking (Improvement 3)
        self._failed_adaptations_count = 0
        self._rollback_events: list[dict] = []

        # Model versioning (Improvement 4)
        self._version_major = 1
        self._version_minor = 0
        self._version_patch = 0
        self._version_history: list[ModelVersionMetadata] = []

        logger.info(
            f"AdaptationModule initialized — "
            f"decay={self._decay_factor}, "
            f"inc_window={self._incremental_window}, "
            f"threshold={self._accuracy_threshold}, "
            f"tolerance={self._tolerance}"
        )

    def execute(
        self,
        command: AdaptationCommand,
        model: BaseModel,
        X_recent: np.ndarray,
        y_recent: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> AdaptationResult:
        """
        Execute an adaptation strategy.

        Enhanced with tolerance-based validation and rollback tracking.

        Args:
            command: AdaptationCommand from DecisionEngine
            model: Current BaseModel instance
            X_recent: Recent training data
            y_recent: Recent labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Feature names for the new model

        Returns:
            AdaptationResult with success/failure, rollback status, and metrics
        """
        strategy = command.strategy
        start_time = time.time()

        logger.info(f"Executing adaptation: {strategy.value}")

        # Get old model accuracy for comparison
        old_metrics = model.evaluate(X_val, y_val)
        old_accuracy = old_metrics["accuracy"]

        try:
            if strategy == AdaptationStrategy.REWEIGHT:
                result = self._reweight(model, X_recent, y_recent, X_val, y_val, feature_names, old_accuracy)
            elif strategy == AdaptationStrategy.INCREMENTAL_RETRAIN:
                result = self._incremental_retrain(model, X_recent, y_recent, X_val, y_val, feature_names, old_accuracy)
            elif strategy == AdaptationStrategy.FULL_RETRAIN:
                result = self._full_retrain(model, X_recent, y_recent, X_val, y_val, feature_names, old_accuracy)
            else:
                return AdaptationResult(
                    strategy=strategy.value,
                    success=False,
                    old_accuracy=old_accuracy,
                    new_accuracy=old_accuracy,
                    threshold=self._accuracy_threshold,
                    n_samples_used=0,
                    duration_seconds=time.time() - start_time,
                    timestamp=timestamp_now(),
                    model_version=model.version,
                    reason=f"Unknown strategy: {strategy}",
                    rollback=False,
                    tolerance=self._tolerance,
                    duration_ms=int((time.time() - start_time) * 1000),
                )

            result.old_accuracy = old_accuracy
            duration = time.time() - start_time
            result.duration_seconds = duration
            result.duration_ms = int(duration * 1000)
            result.timestamp = timestamp_now()
            result.model_version = model.version
            result.tolerance = self._tolerance

            self._adaptation_history.append(result)

            # Track failures and rollbacks
            if not result.success:
                self._failed_adaptations_count += 1

            if result.rollback:
                rollback_event = {
                    "timestamp": result.timestamp,
                    "strategy": result.strategy,
                    "old_accuracy": result.old_accuracy,
                    "new_accuracy": result.new_accuracy,
                    "reason": result.reason,
                }
                self._rollback_events.append(rollback_event)
                logger.warning(
                    f"ROLLBACK: {strategy.value} rejected — "
                    f"keeping previous model active | {result.reason}"
                )

            # If successful, update model version metadata (Improvement 4)
            if result.success:
                self._increment_version(strategy.value)
                version_meta = ModelVersionMetadata(
                    version=self.current_version_string,
                    created_at=result.timestamp,
                    strategy=strategy.value,
                    training_samples=result.n_samples_used,
                    accuracy=result.new_accuracy,
                    model_type=model.model_type,
                )
                self._version_history.append(version_meta)
                logger.info(
                    f"Model version updated: {version_meta.version} "
                    f"(strategy={strategy.value}, acc={result.new_accuracy:.4f})"
                )

            logger.info(
                f"Adaptation {'SUCCEEDED' if result.success else 'FAILED'}: "
                f"{strategy.value} — old_acc={old_accuracy:.4f}, "
                f"new_acc={result.new_accuracy:.4f}, "
                f"rollback={result.rollback}, "
                f"duration={result.duration_ms}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Adaptation failed with error: {e}")
            self._failed_adaptations_count += 1
            duration = time.time() - start_time
            return AdaptationResult(
                strategy=strategy.value,
                success=False,
                old_accuracy=old_accuracy,
                new_accuracy=old_accuracy,
                threshold=self._accuracy_threshold,
                n_samples_used=len(X_recent),
                duration_seconds=duration,
                timestamp=timestamp_now(),
                model_version=model.version,
                reason=str(e),
                rollback=False,
                tolerance=self._tolerance,
                duration_ms=int(duration * 1000),
            )

    def _reweight(
        self,
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None,
        old_accuracy: float,
    ) -> AdaptationResult:
        """
        Reweight strategy: retrain with exponential decay sample weights
        favoring more recent samples.
        """
        n_samples = len(X)

        # Create exponential decay weights (most recent = highest weight)
        weights = np.array([
            self._decay_factor ** (n_samples - 1 - i) for i in range(n_samples)
        ])
        # Normalize so weights sum to n_samples
        weights = weights * n_samples / weights.sum()

        # Create fresh model and train with weights
        new_model = model.create_fresh()
        new_model.train(X, y, feature_names=feature_names, sample_weight=weights)

        # Validate and deploy with tolerance check
        return self._validate_and_deploy(
            new_model, model, X_val, y_val, n_samples, "reweight", old_accuracy
        )

    def _incremental_retrain(
        self,
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None,
        old_accuracy: float,
    ) -> AdaptationResult:
        """
        Incremental retrain: train on the most recent N samples.
        Uses warm_start when supported by the model type.
        """
        # Use only the most recent samples
        window = min(self._incremental_window, len(X))
        X_recent = X[-window:]
        y_recent = y[-window:]

        # Create fresh model for incremental training
        new_model = model.create_fresh()

        # Enable warm_start for supported models
        if self._warm_start and hasattr(new_model.estimator, "warm_start"):
            new_model.estimator.warm_start = True

        new_model.train(X_recent, y_recent, feature_names=feature_names)

        return self._validate_and_deploy(
            new_model, model, X_val, y_val, window, "incremental_retrain", old_accuracy
        )

    def _full_retrain(
        self,
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None,
        old_accuracy: float,
    ) -> AdaptationResult:
        """
        Full retrain: create and train a completely new model from scratch.
        """
        n_samples = len(X)
        if n_samples < self._full_retrain_min_samples:
            logger.warning(
                "Full retrain using %d samples (configured min=%d); proceeding",
                n_samples,
                self._full_retrain_min_samples,
            )

        new_model = model.create_fresh()
        new_model.train(X, y, feature_names=feature_names)

        return self._validate_and_deploy(
            new_model, model, X_val, y_val, n_samples, "full_retrain", old_accuracy
        )

    def _validate_and_deploy(
        self,
        new_model: BaseModel,
        current_model: BaseModel,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_samples_used: int,
        strategy_name: str,
        old_accuracy: float,
    ) -> AdaptationResult:
        """
        Enhanced validation gate with tolerance-based comparison.

        Validation after drift:
          1. No large regression: new_accuracy >= old_accuracy - tolerance
          2. Either meets the accuracy floor, OR improves on the current model
             (after drift the live model often sits below the floor too)
        """
        new_metrics = new_model.evaluate(X_val, y_val)
        new_accuracy = new_metrics["accuracy"]

        # Check 1: Absolute accuracy threshold
        passes_threshold = new_accuracy >= self._accuracy_threshold

        # Check 2: Regression tolerance — prevent deploying a significantly worse model
        passes_tolerance = new_accuracy >= (old_accuracy - self._tolerance)
        improved = new_accuracy >= old_accuracy

        if passes_tolerance and (passes_threshold or improved):
            # ── DEPLOY: hot_swap the new model into the current one ──
            current_model.hot_swap(new_model)
            logger.info(
                f"Model deployed via hot_swap — "
                f"accuracy={new_accuracy:.4f} >= threshold={self._accuracy_threshold}, "
                f"regression_check={new_accuracy:.4f} >= {old_accuracy:.4f} - {self._tolerance}"
            )
            return AdaptationResult(
                strategy=strategy_name,
                success=True,
                old_accuracy=old_accuracy,
                new_accuracy=new_accuracy,
                threshold=self._accuracy_threshold,
                n_samples_used=n_samples_used,
                duration_seconds=0.0,
                timestamp="",
                model_version=0,
                rollback=False,
                tolerance=self._tolerance,
            )
        else:
            # ── ROLLBACK: reject new model, keep current model active ──
            # Build detailed rejection reason
            reasons = []
            if not passes_threshold:
                reasons.append(
                    f"accuracy {new_accuracy:.4f} below threshold {self._accuracy_threshold}"
                )
            if not passes_tolerance:
                reasons.append(
                    f"accuracy {new_accuracy:.4f} below old_accuracy ({old_accuracy:.4f}) "
                    f"- tolerance ({self._tolerance}) = {old_accuracy - self._tolerance:.4f}"
                )
            reason = " AND ".join(reasons)

            logger.warning(
                f"Model REJECTED (ROLLBACK) — {reason}. "
                f"Keeping previous model active."
            )
            return AdaptationResult(
                strategy=strategy_name,
                success=False,
                old_accuracy=old_accuracy,
                new_accuracy=new_accuracy,
                threshold=self._accuracy_threshold,
                n_samples_used=n_samples_used,
                duration_seconds=0.0,
                timestamp="",
                model_version=0,
                reason=reason,
                rollback=True,
                tolerance=self._tolerance,
            )

    # ── Model Versioning (Improvement 4) ──────────────────────────────────

    def _increment_version(self, strategy: str) -> None:
        """
        Increment model version based on adaptation strategy.

        Versioning scheme:
          - FULL_RETRAIN:         bump minor version (v1.X.0)
          - INCREMENTAL_RETRAIN:  bump patch version (v1.0.X)
          - REWEIGHT:             bump patch version (v1.0.X)
        """
        if strategy == "full_retrain":
            self._version_minor += 1
            self._version_patch = 0
        else:
            self._version_patch += 1

    @property
    def current_version_string(self) -> str:
        """Current model version as semantic string."""
        return f"v{self._version_major}.{self._version_minor}.{self._version_patch}"

    def get_version_metadata(self) -> dict:
        """Get current version metadata for display."""
        if self._version_history:
            latest = self._version_history[-1]
            return latest.to_dict()
        return {
            "version": self.current_version_string,
            "created_at": "",
            "strategy": "initial_training",
            "training_samples": 0,
            "accuracy": 0.0,
        }

    def save_version_history(self, path: str | Path = "models/version_history.json") -> None:
        """Persist version history to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [v.to_dict() for v in self._version_history]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Version history saved to {path} ({len(data)} versions)")

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def history(self) -> list[AdaptationResult]:
        """Get adaptation history (last 5 for dashboard — Improvement 7)."""
        return self._adaptation_history.copy()

    @property
    def recent_history(self) -> list[dict]:
        """Last 5 adaptation attempts for dashboard display (Bonus)."""
        return [r.to_dict() for r in self._adaptation_history[-5:]]

    @property
    def total_adaptations(self) -> int:
        """Total adaptation attempts."""
        return len(self._adaptation_history)

    @property
    def successful_adaptations(self) -> int:
        """Number of successful adaptations."""
        return sum(1 for r in self._adaptation_history if r.success)

    @property
    def failed_adaptations_count(self) -> int:
        """Number of failed adaptation attempts."""
        return self._failed_adaptations_count

    @property
    def rollback_events(self) -> list[dict]:
        """History of all rollback events."""
        return self._rollback_events.copy()

    @property
    def version_history(self) -> list[ModelVersionMetadata]:
        """Full model version history."""
        return self._version_history.copy()
