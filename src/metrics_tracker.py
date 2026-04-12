"""
DriftInsights — Metrics Tracker

Maintains running and windowed performance metrics for the streaming pipeline.
"""

import logging
from collections import deque
from typing import Optional

import numpy as np

from src.utils.helpers import safe_divide

logger = logging.getLogger("driftinsights.metrics_tracker")


class MetricsTracker:
    """
    Tracks prediction performance in real-time.

    Maintains:
      - Running (cumulative) accuracy and error rate
      - Windowed error rate over the most recent N predictions
      - Per-update counts for monitoring

    Usage:
        tracker = MetricsTracker(window_size=200)
        tracker.update(y_true=1, y_pred=1)
        acc = tracker.get_running_accuracy()
        err = tracker.get_windowed_error()
        snapshot = tracker.get_snapshot()
    """

    def __init__(self, window_size: int = 200):
        self.window_size = window_size

        # Running totals
        self._total_correct = 0
        self._total_samples = 0

        # Windowed buffer: stores (correct: bool) for last N predictions
        self._window: deque[bool] = deque(maxlen=window_size)

        # Detailed history for plotting
        self._accuracy_history: list[float] = []
        self._error_history: list[float] = []
        self._timestamps: list[int] = []  # sample index

    def update(self, y_true: int | np.integer, y_pred: int | np.integer) -> None:
        """
        Record a single prediction result.

        Args:
            y_true: Actual label
            y_pred: Predicted label
        """
        correct = int(y_true) == int(y_pred)

        self._total_samples += 1
        if correct:
            self._total_correct += 1

        self._window.append(correct)

        # Record history every 10 samples for efficiency
        if self._total_samples % 10 == 0:
            self._accuracy_history.append(self.get_running_accuracy())
            self._error_history.append(self.get_windowed_error())
            self._timestamps.append(self._total_samples)

    def update_batch(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Record a batch of predictions."""
        for yt, yp in zip(y_true, y_pred):
            self.update(yt, yp)

    def get_running_accuracy(self) -> float:
        """Get cumulative accuracy over all samples seen."""
        return safe_divide(self._total_correct, self._total_samples)

    def get_running_error(self) -> float:
        """Get cumulative error rate over all samples seen."""
        return 1.0 - self.get_running_accuracy()

    def get_windowed_error(self, window: int | None = None) -> float:
        """
        Get error rate over the most recent N samples.

        Args:
            window: Custom window size. Uses default if None.

        Returns:
            Error rate in [0.0, 1.0]
        """
        if not self._window:
            return 0.0

        if window is not None and window < len(self._window):
            recent = list(self._window)[-window:]
        else:
            recent = self._window

        n_correct = sum(recent)
        return safe_divide(len(recent) - n_correct, len(recent))

    def get_windowed_accuracy(self, window: int | None = None) -> float:
        """Get accuracy over the most recent N samples."""
        return 1.0 - self.get_windowed_error(window)

    def get_snapshot(self) -> dict:
        """
        Get a complete snapshot of all current metrics.

        Returns:
            Dict with running/windowed accuracy and error rates,
            total sample count, and recent history.
        """
        return {
            "total_samples": self._total_samples,
            "running_accuracy": self.get_running_accuracy(),
            "running_error": self.get_running_error(),
            "windowed_accuracy": self.get_windowed_accuracy(),
            "windowed_error": self.get_windowed_error(),
            "window_size": self.window_size,
            "window_fill": len(self._window),
        }

    def get_history(self) -> dict:
        """Get the recorded history for plotting."""
        return {
            "timestamps": self._timestamps.copy(),
            "accuracy": self._accuracy_history.copy(),
            "error": self._error_history.copy(),
        }

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self._total_correct = 0
        self._total_samples = 0
        self._window.clear()
        self._accuracy_history.clear()
        self._error_history.clear()
        self._timestamps.clear()
        logger.info("MetricsTracker reset")

    @property
    def total_samples(self) -> int:
        return self._total_samples
