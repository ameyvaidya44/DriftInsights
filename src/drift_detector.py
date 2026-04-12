"""
DriftInsights — Drift Detector

Real-time concept drift detection using ADWIN from the river library.
Produces DriftEvent records when statistical change is detected.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

from src.utils.helpers import timestamp_now

logger = logging.getLogger("driftinsights.drift_detector")


@dataclass
class DriftEvent:
    """Represents a detected concept drift event."""
    timestamp: str
    sample_idx: int
    pre_error_rate: float
    post_error_rate: float
    adwin_width: int
    detection_count: int
    severity_estimate: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class DriftDetector:
    """
    ADWIN-based concept drift detector for streaming data.

    Uses the river library's ADWIN algorithm to monitor the error stream
    and detect statistically significant distribution changes.

    Features:
      - Configurable ADWIN sensitivity (delta)
      - Minimum gap between detections to avoid rapid-fire alerts
      - Warmup period to skip initial unstable predictions
      - Pre/post error rate tracking for drift events

    Usage:
        detector = DriftDetector(config)
        for error in error_stream:
            event = detector.update(error, sample_idx)
            if event:
                handle_drift(event)
    """

    def __init__(self, config):
        from river.drift import ADWIN

        self.config = config
        drift_cfg = config.drift

        # ADWIN detector
        self._delta = drift_cfg.adwin.delta
        self._adwin = ADWIN(delta=self._delta)

        # Control parameters
        self._min_gap = drift_cfg.min_gap
        self._warmup = drift_cfg.warmup_samples

        # State tracking
        self._last_drift_idx = -self._min_gap  # allow first detection
        self._detection_count = 0
        self._total_updates = 0

        # Error tracking for pre/post rates
        self._error_buffer: list[int] = []
        self._pre_drift_errors: list[int] = []

        logger.info(
            f"DriftDetector initialized — ADWIN delta={self._delta}, "
            f"min_gap={self._min_gap}, warmup={self._warmup}"
        )

    def update(self, error: int, sample_idx: int) -> Optional[DriftEvent]:
        """
        Feed an error indicator (0=correct, 1=incorrect) and check for drift.

        Args:
            error: Binary error indicator (0 or 1)
            sample_idx: Current sample index in the stream

        Returns:
            DriftEvent if drift detected, None otherwise
        """
        self._total_updates += 1
        self._error_buffer.append(error)

        # Skip warmup period
        if self._total_updates < self._warmup:
            return None

        # Store pre-drift snapshot before update
        pre_width = self._adwin.width
        pre_estimation = self._adwin.estimation if self._adwin.width > 0 else 0.0

        # Update ADWIN
        self._adwin.update(error)

        # Check for drift
        if self._adwin.drift_detected:
            # Enforce minimum gap
            if sample_idx - self._last_drift_idx < self._min_gap:
                logger.debug(
                    f"Drift suppressed at idx={sample_idx} "
                    f"(gap={sample_idx - self._last_drift_idx} < {self._min_gap})"
                )
                return None

            self._detection_count += 1

            # Calculate pre/post error rates
            post_estimation = self._adwin.estimation if self._adwin.width > 0 else 0.0
            pre_error_rate = pre_estimation
            post_error_rate = post_estimation

            # Severity estimate: magnitude of error rate shift
            severity_estimate = abs(post_error_rate - pre_error_rate)

            event = DriftEvent(
                timestamp=timestamp_now(),
                sample_idx=sample_idx,
                pre_error_rate=round(pre_error_rate, 4),
                post_error_rate=round(post_error_rate, 4),
                adwin_width=self._adwin.width,
                detection_count=self._detection_count,
                severity_estimate=round(severity_estimate, 4),
            )

            # Save pre-drift buffer for XAI analysis
            buffer_len = min(len(self._error_buffer), 500)
            self._pre_drift_errors = self._error_buffer[-buffer_len:]

            self._last_drift_idx = sample_idx

            logger.warning(
                f"[DRIFT] DETECTED at sample {sample_idx} | "
                f"pre_error={pre_error_rate:.4f} -> post_error={post_error_rate:.4f} | "
                f"detection #{self._detection_count}"
            )

            return event

        return None

    def reset(self) -> None:
        """Reset the ADWIN detector (typically called after adaptation)."""
        from river.drift import ADWIN

        self._adwin = ADWIN(delta=self._delta)
        self._error_buffer.clear()
        logger.info("DriftDetector ADWIN window reset")

    @property
    def estimation(self) -> float:
        """Current ADWIN error estimation."""
        return self._adwin.estimation if self._adwin.width > 0 else 0.0

    @property
    def width(self) -> int:
        """Current ADWIN window width."""
        return self._adwin.width

    @property
    def detection_count(self) -> int:
        """Total number of drift detections."""
        return self._detection_count

    @property
    def total_updates(self) -> int:
        """Total number of samples processed."""
        return self._total_updates
