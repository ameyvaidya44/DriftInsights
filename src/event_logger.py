"""
DriftInsights — Event Logger (Enhanced)

JSONL-based event logging with structured records for all pipeline events.
Enhanced with decision reasoning, rollback events, and model version logging.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.utils.helpers import ensure_directory, generate_session_id, timestamp_now

logger = logging.getLogger("driftinsights.event_logger")


class EventLogger:
    """
    Structured JSONL event logger.

    Writes one JSON object per line to a log file. Each event includes a
    timestamp, session ID, phase, and event-specific data.

    Event types:
      - prediction: model predictions with confidence
      - training: model training/retraining events
      - metrics: periodic metrics snapshots
      - drift: drift detection events
      - xai: SHAP explanation events
      - decision: decision engine reasoning (Improvement 6)
      - adaptation: model adaptation events
      - rollback: rollback events when adaptation is rejected (Improvement 6)
      - version_update: model version changes (Improvement 6)
      - system: system lifecycle events

    All logs include: timestamp, phase, session_id (Improvement 6).

    Usage:
        logger = EventLogger(config)
        logger.log("prediction", {"y_true": 1, "y_pred": 0, "confidence": 0.85})
        logger.log("drift", {"delta_total": 0.23, "severity": "moderate"})
        logger.log_decision(command)
        logger.log_rollback(rollback_event)
    """

    def __init__(
        self,
        config,
        session_id: str | None = None,
        log_dir: str | None = None,
    ):
        self.config = config
        self.session_id = session_id or generate_session_id()
        self._phase = "init"
        self._event_count = 0

        # Setup log directory and file
        log_dir = log_dir or config.logging.dir
        self._log_dir = ensure_directory(log_dir)
        self._log_file = self._log_dir / f"{self.session_id}.jsonl"
        self._file_handle = open(self._log_file, "a", encoding="utf-8")

        # Setup Python logging
        self._setup_python_logger(config)

        # Log session start
        self.log("system", {
            "action": "session_start",
            "project": config.project.name,
            "version": config.project.version,
        })

        logger.info(f"EventLogger initialized — session: {self.session_id}")

    def _setup_python_logger(self, config) -> None:
        """Configure Python's logging module."""
        root_logger = logging.getLogger("driftinsights")

        if root_logger.handlers:
            return  # Already configured

        level = getattr(logging, config.logging.level.upper(), logging.INFO)
        root_logger.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if config.logging.console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        if config.logging.file:
            file_path = self._log_dir / f"{self.session_id}.log"
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    def set_phase(self, phase: str) -> None:
        """Set the current pipeline phase (e.g., 'training', 'streaming', 'adapting')."""
        self._phase = phase
        self.log("system", {"action": "phase_change", "new_phase": phase})

    def log(self, event_type: str, data: dict[str, Any] | None = None) -> dict:
        """
        Log a structured event.

        Every record includes timestamp, session_id, and phase for
        full traceability (Improvement 6).

        Args:
            event_type: Type of event (prediction, training, metrics, drift,
                        decision, adaptation, rollback, version_update, system)
            data: Event-specific data dictionary

        Returns:
            The complete event record
        """
        record = {
            "timestamp": timestamp_now(),
            "session_id": self.session_id,
            "phase": self._phase,
            "event_type": event_type,
            "event_id": self._event_count,
            **(data or {}),
        }

        line = json.dumps(record, default=str)
        self._file_handle.write(line + "\n")
        self._file_handle.flush()
        self._event_count += 1

        return record

    def log_prediction(
        self,
        sample_idx: int,
        y_true: int | None,
        y_pred: int,
        confidence: float | None = None,
    ) -> dict:
        """Log a prediction event."""
        return self.log("prediction", {
            "sample_idx": sample_idx,
            "y_true": int(y_true) if y_true is not None else None,
            "y_pred": int(y_pred),
            "correct": (int(y_true) == int(y_pred)) if y_true is not None else None,
            "confidence": confidence,
        })

    def log_training(
        self,
        model_type: str,
        n_samples: int,
        metrics: dict,
        version: int,
    ) -> dict:
        """Log a training event."""
        return self.log("training", {
            "model_type": model_type,
            "n_samples": n_samples,
            "metrics": metrics,
            "model_version": version,
        })

    def log_metrics(self, metrics_snapshot: dict) -> dict:
        """Log a metrics snapshot."""
        return self.log("metrics", {"snapshot": metrics_snapshot})

    def log_drift(self, drift_event: dict) -> dict:
        """Log a drift detection event."""
        return self.log("drift", drift_event)

    def log_adaptation(self, adaptation_record: dict) -> dict:
        """Log an adaptation event (success or failure)."""
        return self.log("adaptation", adaptation_record)

    # ── Enhanced Logging Methods (Improvement 6) ──────────────────────────

    def log_decision(self, decision_data: dict) -> dict:
        """
        Log a decision engine event with full reasoning.

        Captures: severity, strategy, delta_e_total, reason, fsm_state
        """
        return self.log("decision", decision_data)

    def log_rollback(self, rollback_data: dict) -> dict:
        """
        Log a rollback event when an adaptation is rejected.

        Captures: strategy, old_accuracy, new_accuracy, reason
        """
        return self.log("rollback", rollback_data)

    def log_version_update(self, version_data: dict) -> dict:
        """
        Log a model version update event.

        Captures: version string, strategy, accuracy, training_samples
        """
        return self.log("version_update", version_data)

    def log_fsm_transition(
        self,
        state_before: str,
        state_after: str,
        sample_idx: int,
        trigger: str,
    ) -> dict:
        """
        Log an FSM state transition.

        Captures: state_before, state_after, trigger reason
        """
        return self.log("fsm_transition", {
            "state_before": state_before,
            "state_after": state_after,
            "sample_idx": sample_idx,
            "trigger": trigger,
        })

    # ── File Management ──────────────────────────────────────────────────

    def get_log_path(self) -> Path:
        """Return the path to the current log file."""
        return self._log_file

    def close(self) -> None:
        """Close the log file and log session end."""
        self.log("system", {
            "action": "session_end",
            "total_events": self._event_count,
        })
        self._file_handle.close()
        logger.info(f"EventLogger closed — {self._event_count} events logged")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if hasattr(self, "_file_handle") and not self._file_handle.closed:
            self.close()
