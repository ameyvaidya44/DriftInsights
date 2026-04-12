"""
DriftInsights — Pipeline Orchestrator (Enhanced)

Coordinates the full ML pipeline: setup -> train -> stream with
drift detection, XAI explanation, decision engine, and adaptation.

Enhanced with:
  - Decision reasoning logging
  - Rollback event logging
  - Model version tracking
  - FSM transition logging
  - Feature drift summary (Bonus)
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import Config
from src.data_handler import DataHandler
from src.base_model import BaseModel
from src.metrics_tracker import MetricsTracker
from src.event_logger import EventLogger
from src.drift_detector import DriftDetector
from src.xai_engine import XAIEngine
from src.decision_engine import DecisionEngine, AdaptationStrategy
from src.adaptation_module import AdaptationModule
from src.utils.helpers import generate_session_id, timestamp_now

logger = logging.getLogger("driftinsights.pipeline")

# DB persistence (non-critical — failures are logged, never crash the pipeline)
try:
    from api.database import (
        db_write, DriftEventRecord, MetricsSnapshotRecord,
        AdaptationRecord as AdaptationRecordDB, XAIRecord,
    )
    _DB_AVAILABLE = True
except Exception:
    _DB_AVAILABLE = False


class Pipeline:
    """
    Main orchestrator for the DriftInsights streaming ML pipeline.

    Lifecycle:
      1. setup()   — Initialize all components from config
      2. train()   — Train base model on training data
      3. run_stream() — Stream test data through the full pipeline:
           predict -> update metrics -> detect drift ->
           explain drift -> decide action -> adapt model

    Usage:
        pipeline = Pipeline("config/default.yaml")
        pipeline.setup()
        pipeline.train()
        pipeline.run_stream()
    """

    def __init__(self, config_path: str | None = None, config: Config | None = None):
        self._config_path = config_path
        self._config = config

        # Components (initialized in setup())
        self.config: Optional[Config] = config
        self.data_handler: Optional[DataHandler] = None
        self.model: Optional[BaseModel] = None
        self.metrics: Optional[MetricsTracker] = None
        self.event_logger: Optional[EventLogger] = None
        self.drift_detector: Optional[DriftDetector] = None
        self.xai_engine: Optional[XAIEngine] = None
        self.decision_engine: Optional[DecisionEngine] = None
        self.adaptation_module: Optional[AdaptationModule] = None

        # Data splits
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._X_val: Optional[np.ndarray] = None
        self._y_val: Optional[np.ndarray] = None
        self._X_test: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None
        self._feature_names: list[str] = []

        # Streaming state
        self._stream_buffer_X: list[np.ndarray] = []
        self._stream_buffer_y: list[int] = []
        self._drift_events: list[dict] = []
        self._session_id: str = ""

        # WebSocket manager (injected after construction)
        self._ws_manager = None

    def set_ws_manager(self, manager) -> None:
        """Inject the WebSocket manager for broadcasting."""
        self._ws_manager = manager

    def _broadcast(self, event_type: str, data: dict) -> None:
        """Fire-and-forget broadcast to all WebSocket clients."""
        if self._ws_manager is None:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._ws_manager.broadcast(event_type, data), loop
                )
            else:
                loop.run_until_complete(self._ws_manager.broadcast(event_type, data))
        except Exception as e:
            logger.debug(f"Broadcast failed (non-critical): {e}")

    def setup(self) -> "Pipeline":
        """
        Initialize all pipeline components from configuration.

        Returns:
            self (for chaining)
        """
        # Load config
        if self.config is None:
            self.config = Config(self._config_path)

        self._session_id = (
            self.config.project.get("session_id") or generate_session_id()
        )

        # Initialize components
        self.event_logger = EventLogger(self.config, session_id=self._session_id)
        self.event_logger.set_phase("setup")

        self.data_handler = DataHandler(self.config)
        self.model = BaseModel(self.config)
        self.metrics = MetricsTracker(
            window_size=self.config.data.streaming.buffer_size
        )
        self.drift_detector = DriftDetector(self.config)
        self.xai_engine = XAIEngine(self.config)
        self.decision_engine = DecisionEngine(self.config)
        self.adaptation_module = AdaptationModule(self.config)

        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        df = self.data_handler.load_dataset()
        df = self.data_handler.preprocess(df, fit=True)

        # Temporal split
        train_df, val_df, test_df = self.data_handler.temporal_split(df)

        self._X_train, self._y_train = self.data_handler.split_xy(train_df)
        self._X_val, self._y_val = self.data_handler.split_xy(val_df)
        self._X_test, self._y_test = self.data_handler.split_xy(test_df)
        self._feature_names = self.data_handler.feature_names

        self.event_logger.log("system", {
            "action": "setup_complete",
            "dataset": self.config.data.dataset,
            "train_size": len(self._X_train),
            "val_size": len(self._X_val),
            "test_size": len(self._X_test),
            "n_features": len(self._feature_names),
            "model_type": self.config.model.type,
        })

        logger.info(
            f"Setup complete — {self.config.data.dataset} dataset, "
            f"model={self.config.model.type}, "
            f"features={len(self._feature_names)}"
        )
        return self

    def train(self) -> dict:
        """
        Train the base model on training data and evaluate on validation set.

        Returns:
            Validation metrics dict
        """
        self.event_logger.set_phase("training")
        logger.info("Training base model...")

        self.model.train(
            self._X_train,
            self._y_train,
            feature_names=self._feature_names,
        )

        # Evaluate on validation
        val_metrics = self.model.evaluate(self._X_val, self._y_val)

        # Save model
        model_path = self.model.save()

        # Set up XAI engine with trained model
        self.xai_engine.set_model(self.model.estimator)

        # Log training event
        self.event_logger.log_training(
            model_type=self.model.model_type,
            n_samples=len(self._X_train),
            metrics=val_metrics,
            version=self.model.version,
        )

        # Log initial model version (Improvement 4)
        self.event_logger.log_version_update({
            "version": self.adaptation_module.current_version_string,
            "strategy": "initial_training",
            "accuracy": val_metrics["accuracy"],
            "training_samples": len(self._X_train),
            "model_type": self.model.model_type,
        })

        logger.info(
            f"Training complete — val_accuracy={val_metrics['accuracy']:.4f}, "
            f"val_f1={val_metrics['f1']:.4f}, saved to {model_path}"
        )
        return val_metrics

    def run_stream(self, max_samples: int | None = None) -> dict:
        """
        Run the streaming prediction loop with full drift pipeline.

        For each sample:
          1. Predict
          2. Update metrics
          3. Check for drift (ADWIN)
          4. If drift -> explain (SHAP) -> decide -> adapt

        Args:
            max_samples: Limit number of samples to process (None = all)

        Returns:
            Summary dict with final metrics and drift events
        """
        self.event_logger.set_phase("streaming")
        logger.info(
            f"Starting streaming loop — {len(self._X_test)} test samples"
        )

        start_time = time.time()
        processed = 0
        n_samples = len(self._X_test)
        if max_samples:
            n_samples = min(n_samples, max_samples)

        # Broadcast pipeline_started
        dataset_name = self.config.data.dataset if self.config else "unknown"
        model_ver = self.adaptation_module.current_version_string if self.adaptation_module else "v1"
        self._broadcast("system_event", {
            "event": "pipeline_started",
            "message": f"Pipeline started with model {model_ver} on dataset {dataset_name}",
            "timestamp": timestamp_now(),
        })

        # Determine broadcast interval from config
        try:
            broadcast_every = self.config.logging.get("log_metrics_every_n_samples", 100)
        except Exception:
            broadcast_every = 100

        # Streaming loop
        # NOTE: In production, this loop would consume from Kafka/Redis Streams
        # instead of iterating over in-memory arrays. The pipeline logic remains identical.
        for X_batch, y_batch, idx in self.data_handler.stream(
            self._X_test[:n_samples],
            self._y_test[:n_samples],
        ):
            # ── 1. Predict ──
            y_pred = self.model.predict(X_batch)

            for i in range(len(X_batch)):
                sample_idx = idx + i
                pred = int(y_pred[i])

                # ── 2. Update metrics ──
                if y_batch is not None:
                    true_label = int(y_batch[i])
                    self.metrics.update(true_label, pred)
                    error = 0 if true_label == pred else 1

                    # Maintain streaming buffer for adaptation
                    self._stream_buffer_X.append(X_batch[i])
                    self._stream_buffer_y.append(true_label)

                    # ── 3. Drift Detection ──
                    drift_event = self.drift_detector.update(error, sample_idx)

                    if drift_event:
                        self._on_drift(drift_event, sample_idx)

                    # Log prediction (every 100 samples for efficiency)
                    if sample_idx % 100 == 0:
                        confidence = None
                        try:
                            proba = self.model.predict_proba(X_batch[i:i+1])
                            confidence = float(np.max(proba))
                        except Exception:
                            pass

                        self.event_logger.log_prediction(
                            sample_idx=sample_idx,
                            y_true=true_label,
                            y_pred=pred,
                            confidence=confidence,
                        )

                processed += 1

            # Periodic metrics logging + broadcast
            if processed % 500 == 0:
                snapshot = self.metrics.get_snapshot()
                self.event_logger.log_metrics(snapshot)
                logger.info(
                    f"Progress: {processed}/{n_samples} | "
                    f"acc={snapshot['running_accuracy']:.4f} | "
                    f"w_err={snapshot['windowed_error']:.4f}"
                )

            if processed % broadcast_every == 0 and processed > 0:
                snapshot = self.metrics.get_snapshot()
                elapsed_so_far = time.time() - start_time
                sps = round(processed / elapsed_so_far, 1) if elapsed_so_far > 0 else 0.0
                self._broadcast("metrics_update", {
                    "sample_index": snapshot["total_samples"],
                    "running_accuracy": round(snapshot["running_accuracy"], 4),
                    "windowed_error_rate": round(snapshot["windowed_error"], 4),
                    "window_fill": snapshot["window_fill"],
                    "window_size": snapshot["window_size"],
                    "samples_per_second": sps,
                    "timestamp": timestamp_now(),
                })
                # Persist metrics snapshot
                if _DB_AVAILABLE:
                    db_write(MetricsSnapshotRecord(
                        session_id=self._session_id,
                        sample_idx=snapshot["total_samples"],
                        running_accuracy=snapshot["running_accuracy"],
                        running_error=snapshot["running_error"],
                        windowed_accuracy=snapshot["windowed_accuracy"],
                        windowed_error=snapshot["windowed_error"],
                        window_size=snapshot["window_size"],
                    ))

        # Final summary
        elapsed = time.time() - start_time
        final_snapshot = self.metrics.get_snapshot()

        summary = {
            "total_samples": processed,
            "elapsed_seconds": round(elapsed, 2),
            "final_metrics": final_snapshot,
            "drift_events": len(self._drift_events),
            "adaptations": self.adaptation_module.total_adaptations,
            "successful_adaptations": self.adaptation_module.successful_adaptations,
            "failed_adaptations": self.adaptation_module.failed_adaptations_count,
            "rollbacks": len(self.adaptation_module.rollback_events),
            "model_version": self.model.version,
            "model_version_string": self.adaptation_module.current_version_string,
        }

        self.event_logger.log("system", {
            "action": "streaming_complete",
            **summary,
        })

        # Broadcast pipeline_complete
        self._broadcast("system_event", {
            "event": "pipeline_complete",
            "message": f"Pipeline complete — {processed} samples, {len(self._drift_events)} drifts, model {self.adaptation_module.current_version_string}",
            "timestamp": timestamp_now(),
        })

        # Save version history to JSON (Improvement 4)
        self.adaptation_module.save_version_history()

        logger.info(
            f"Streaming complete — {processed} samples in {elapsed:.1f}s | "
            f"accuracy={final_snapshot['running_accuracy']:.4f} | "
            f"drifts={len(self._drift_events)} | "
            f"adaptations={self.adaptation_module.total_adaptations} | "
            f"rollbacks={len(self.adaptation_module.rollback_events)} | "
            f"model={self.adaptation_module.current_version_string}"
        )

        return summary

    def _on_drift(self, drift_event, sample_idx: int) -> None:
        """
        Handle a drift detection event.

        Triggers: XAI explanation -> Decision -> Adaptation
        """
        self.event_logger.log_drift(drift_event.to_dict())
        self._drift_events.append(drift_event.to_dict())

        # Broadcast drift_detected
        d = drift_event.to_dict()
        error_delta = round(abs(d.get("post_error_rate", 0) - d.get("pre_error_rate", 0)), 4)
        sev_est = d.get("severity_estimate", 0)
        if sev_est >= 0.30:
            severity_label = "Severe"
        elif sev_est >= 0.10:
            severity_label = "Moderate"
        else:
            severity_label = "Minor"
        self._broadcast("drift_detected", {
            "sample_index": d.get("sample_idx", sample_idx),
            "severity": severity_label,
            "pre_error_rate": d.get("pre_error_rate", 0),
            "post_error_rate": d.get("post_error_rate", 0),
            "error_delta": error_delta,
            "adwin_width": d.get("adwin_width", 0),
            "detection_number": d.get("detection_count", 0),
            "timestamp": timestamp_now(),
        })

        # Persist drift event to DB
        if _DB_AVAILABLE:
            db_write(DriftEventRecord(
                session_id=self._session_id,
                sample_idx=d.get("sample_idx", sample_idx),
                pre_error_rate=d.get("pre_error_rate", 0),
                post_error_rate=d.get("post_error_rate", 0),
                severity_estimate=d.get("severity_estimate", 0),
                adwin_width=d.get("adwin_width", 0),
                detection_count=d.get("detection_count", 0),
            ))

        # ── 4. XAI Explanation ──
        explanation = self._on_xai(sample_idx)

        if explanation is None:
            logger.warning("XAI explanation failed — skipping decision/adaptation")
            return

        # ── 5. Decision Engine ──
        command = self.decision_engine.evaluate(
            delta_total=explanation.delta_total,
            sample_idx=sample_idx,
            top_features=[
                (f.feature, f.delta) for f in explanation.top_features
            ],
        )

        if command is None:
            logger.info("Decision engine suppressed adaptation (cooldown/adapting)")
            return

        # Log the decision with full reasoning (Improvement 2 & 6)
        self.event_logger.log_decision(command.to_dict())

        # ── 6. Adaptation ──
        self._on_adapt(command, sample_idx)

    def _on_xai(self, sample_idx: int):
        """
        Compute SHAP explanation for detected drift.

        Uses rolling buffer to extract pre-drift and post-drift windows.
        """
        window_size = self.xai_engine.window_size

        # Get pre-drift and post-drift windows from buffer
        buffer_X = np.array(self._stream_buffer_X)
        n = len(buffer_X)

        if n < window_size * 2:
            logger.warning(
                f"Insufficient data for XAI: {n} < {window_size * 2} samples"
            )
            return None

        try:
            X_pre = buffer_X[-(window_size * 2):-window_size]
            X_post = buffer_X[-window_size:]

            explanation = self.xai_engine.explain_drift(
                X_pre, X_post, self._feature_names
            )

            # Log XAI explanation
            self.event_logger.log("xai", explanation.to_dict())

            # Broadcast xai_complete
            top_f = explanation.top_features[0] if explanation.top_features else None
            # Determine strategy that will be triggered (peek at decision engine)
            sev_est = explanation.delta_total
            if sev_est >= 0.30:
                sev_label = "Severe"
            elif sev_est >= 0.10:
                sev_label = "Moderate"
            else:
                sev_label = "Minor"
            self._broadcast("xai_complete", {
                "delta_e_total": round(explanation.delta_total, 4),
                "top_feature": top_f.feature if top_f else "",
                "top_delta_e": round(top_f.delta, 4) if top_f else 0,
                "direction": top_f.direction if top_f else "unknown",
                "strategy_triggered": "pending",
                "severity": sev_label,
                "timestamp": timestamp_now(),
            })

            # Persist XAI record
            if _DB_AVAILABLE:
                db_write(XAIRecord(
                    session_id=self._session_id,
                    delta_e_total=round(explanation.delta_total, 4),
                    direction=explanation.direction,
                    top_feature=top_f.feature if top_f else None,
                    top_delta_e=round(top_f.delta, 4) if top_f else None,
                    explanation_json=explanation.to_dict(),
                ))

            return explanation

        except Exception as e:
            logger.error(f"XAI explanation error: {e}")
            return None

    def _on_adapt(self, command, sample_idx: int) -> None:
        """Execute adaptation based on decision engine command."""
        self.event_logger.set_phase("adapting")

        # Log FSM transition (Improvement 2 & 6)
        self.event_logger.log_fsm_transition(
            state_before=self.decision_engine.state.value,
            state_after="adapting",
            sample_idx=sample_idx,
            trigger=f"Drift adaptation: {command.strategy.value}",
        )

        self.decision_engine.begin_adaptation()

        # Prepare data for adaptation
        buffer_X = np.array(self._stream_buffer_X)
        buffer_y = np.array(self._stream_buffer_y)

        result = self.adaptation_module.execute(
            command=command,
            model=self.model,
            X_recent=buffer_X,
            y_recent=buffer_y,
            X_val=self._X_val,
            y_val=self._y_val,
            feature_names=self._feature_names,
        )

        # Log adaptation result (Improvement 6)
        self.event_logger.log_adaptation(result.to_dict())

        # Broadcast adaptation_complete
        ver_str = self.adaptation_module.current_version_string
        self._broadcast("adaptation_complete", {
            "strategy": result.strategy,
            "accuracy_before": round(result.old_accuracy, 4),
            "accuracy_after": round(result.new_accuracy, 4),
            "delta_accuracy": round(result.new_accuracy - result.old_accuracy, 4),
            "model_version": ver_str,
            "result": "deployed" if result.success else "rejected",
            "duration_ms": result.duration_ms,
            "timestamp": timestamp_now(),
        })

        # Persist adaptation record
        if _DB_AVAILABLE:
            db_write(AdaptationRecordDB(
                session_id=self._session_id,
                strategy=result.strategy,
                success=result.success,
                old_accuracy=result.old_accuracy,
                new_accuracy=result.new_accuracy,
                threshold=result.threshold,
                n_samples_used=result.n_samples_used,
                duration_seconds=result.duration_seconds,
                model_version=result.model_version,
                reason=result.reason,
                rollback=result.rollback,
            ))

        # Log rollback if adaptation failed (Improvement 3 & 6)
        if result.rollback:
            self.event_logger.log_rollback({
                "strategy": result.strategy,
                "old_accuracy": result.old_accuracy,
                "new_accuracy": result.new_accuracy,
                "reason": result.reason,
                "model_version": result.model_version,
            })

        # If model was swapped, update XAI engine and log version
        if result.success:
            self.xai_engine.set_model(self.model.estimator)
            self.drift_detector.reset()

            # Broadcast model hot-swap system event
            self._broadcast("system_event", {
                "event": "model_hot_swap",
                "message": f"Model hot-swapped to {self.adaptation_module.current_version_string} via {result.strategy}",
                "timestamp": timestamp_now(),
            })

            # Log model version update (Improvement 4 & 6)
            self.event_logger.log_version_update({
                "version": self.adaptation_module.current_version_string,
                "strategy": result.strategy,
                "accuracy": result.new_accuracy,
                "training_samples": result.n_samples_used,
            })

            # Save updated model
            self.model.save()

        # Transition FSM to cooldown
        self.event_logger.log_fsm_transition(
            state_before="adapting",
            state_after="cooldown",
            sample_idx=sample_idx,
            trigger="Adaptation complete",
        )
        self.decision_engine.complete_adaptation(sample_idx)
        self.event_logger.set_phase("streaming")

    def shutdown(self) -> None:
        """Clean up resources."""
        if self.event_logger:
            self.event_logger.close()
        logger.info("Pipeline shutdown complete")

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def drift_events(self) -> list[dict]:
        return self._drift_events.copy()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def feature_drift_summary(self) -> dict:
        """Bonus: feature drift summary from XAI engine."""
        if self.xai_engine:
            return self.xai_engine.get_feature_drift_summary()
        return {}

    @property
    def adaptation_history(self) -> list[dict]:
        """Bonus: last 5 adaptation attempts for dashboard."""
        if self.adaptation_module:
            return self.adaptation_module.recent_history
        return []
