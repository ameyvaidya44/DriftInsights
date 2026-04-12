"""
DriftInsights — Decision Engine (Enhanced)

Finite State Machine (FSM) that maps drift severity to adaptation strategies.
Prevents excessive retraining through cooldown management.
Enhanced with decision reasoning and FSM transition logging.
"""

import logging
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional

from src.utils.helpers import timestamp_now

logger = logging.getLogger("driftinsights.decision_engine")


class AdaptationStrategy(Enum):
    """Possible adaptation strategies based on drift severity."""
    NONE = "none"
    REWEIGHT = "reweight"
    INCREMENTAL_RETRAIN = "incremental_retrain"
    FULL_RETRAIN = "full_retrain"


class DriftSeverity(Enum):
    """Classified drift severity levels."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class FSMState(Enum):
    """Finite State Machine states for the decision engine."""
    IDLE = "idle"
    ADAPTING = "adapting"
    COOLDOWN = "cooldown"


@dataclass
class AdaptationCommand:
    """Command issued by the DecisionEngine to the AdaptationModule."""
    strategy: AdaptationStrategy
    severity: DriftSeverity
    delta_total: float
    timestamp: str
    drift_sample_idx: int
    top_features: list[tuple[str, float]]
    # Enhanced: human-readable reasoning for the decision
    reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["strategy"] = self.strategy.value
        d["severity"] = self.severity.value
        return d


class DecisionEngine:
    """
    Finite State Machine that decides how to respond to detected drift.

    States:
      - IDLE: Normal operation, monitoring for drift
      - ADAPTING: Currently executing an adaptation strategy
      - COOLDOWN: Post-adaptation cooldown to prevent thrashing

    Severity -> Strategy Mapping:
      - Minor (delta_e < moderate_threshold):    REWEIGHT
      - Moderate (delta_e < severe_threshold):   INCREMENTAL_RETRAIN
      - Severe (delta_e >= severe_threshold):    FULL_RETRAIN

    Enhanced (v2):
      - Decision reasoning: every decision includes a human-readable 'reason' string
      - FSM transition logs: every state change logs state_before -> state_after
      - All decisions logged via structured logger

    Usage:
        engine = DecisionEngine(config)
        command = engine.evaluate(delta_total=0.23, sample_idx=5000, top_features=[...])
        if command:
            execute_adaptation(command)
            engine.begin_adaptation()
            # ... after adaptation ...
            engine.complete_adaptation(sample_idx=5200)
    """

    def __init__(self, config):
        self.config = config
        dec_cfg = config.decision

        # Thresholds for severity classification
        self._minor_threshold = dec_cfg.thresholds.minor
        self._moderate_threshold = dec_cfg.thresholds.moderate
        self._severe_threshold = dec_cfg.thresholds.severe

        # Cooldown configuration
        self._cooldown_samples = dec_cfg.cooldown_samples

        # FSM state
        self._state = FSMState.IDLE
        self._cooldown_start_idx = 0
        self._adaptation_count = 0
        self._decision_history: list[dict] = []

        logger.info(
            f"DecisionEngine initialized — thresholds: "
            f"minor={self._minor_threshold}, moderate={self._moderate_threshold}, "
            f"severe={self._severe_threshold}, cooldown={self._cooldown_samples}"
        )

    def evaluate(
        self,
        delta_total: float,
        sample_idx: int,
        top_features: list[tuple[str, float]] | None = None,
    ) -> Optional[AdaptationCommand]:
        """
        Evaluate drift severity and decide on an adaptation strategy.

        Enhanced: includes decision reasoning in every command and logs
        the logic path taken.

        Args:
            delta_total: delta_e_total from XAI engine
            sample_idx: Current sample index
            top_features: Top features driving drift

        Returns:
            AdaptationCommand if action needed, None if suppressed (cooldown/adapting)
        """
        # ── FSM state checks ──
        if self._state == FSMState.ADAPTING:
            reason = "Decision suppressed: FSM is in ADAPTING state"
            logger.debug(reason)
            return None

        if self._state == FSMState.COOLDOWN:
            elapsed = sample_idx - self._cooldown_start_idx
            if elapsed < self._cooldown_samples:
                reason = (
                    f"Decision suppressed: COOLDOWN "
                    f"({elapsed}/{self._cooldown_samples} samples elapsed)"
                )
                logger.debug(reason)
                return None
            else:
                # Cooldown expired — transition back to IDLE
                self._log_fsm_transition(FSMState.COOLDOWN, FSMState.IDLE, sample_idx,
                                         "Cooldown expired")
                self._state = FSMState.IDLE

        # ── Classify severity ──
        severity = self._classify_severity(delta_total)

        if severity == DriftSeverity.NONE:
            reason = (
                f"delta_e_total ({delta_total:.4f}) < minor_threshold "
                f"({self._minor_threshold}) — no action required"
            )
            logger.info(reason)
            return None

        # ── Map severity to strategy ──
        strategy = self._map_strategy(severity)

        # ── Build human-readable reasoning ──
        reason = self._build_reason(delta_total, severity, strategy)

        command = AdaptationCommand(
            strategy=strategy,
            severity=severity,
            delta_total=delta_total,
            timestamp=timestamp_now(),
            drift_sample_idx=sample_idx,
            top_features=top_features or [],
            reason=reason,
        )

        # Record decision with full reasoning
        decision_record = command.to_dict()
        decision_record["sample_idx"] = sample_idx
        self._decision_history.append(decision_record)

        logger.info(
            f"Decision: {severity.value} drift (delta_e={delta_total:.4f}) -> "
            f"strategy={strategy.value} | Reason: {reason}"
        )

        return command

    def _build_reason(
        self,
        delta_total: float,
        severity: DriftSeverity,
        strategy: AdaptationStrategy,
    ) -> str:
        """
        Build a human-readable explanation for why this decision was made.

        This makes decisions auditable — critical for production ML systems.
        """
        if severity == DriftSeverity.SEVERE:
            return (
                f"delta_e_total ({delta_total:.4f}) >= severe_threshold "
                f"({self._severe_threshold}) -> {strategy.value.upper()}"
            )
        elif severity == DriftSeverity.MODERATE:
            return (
                f"delta_e_total ({delta_total:.4f}) >= moderate_threshold "
                f"({self._moderate_threshold}) and < severe_threshold "
                f"({self._severe_threshold}) -> {strategy.value.upper()}"
            )
        elif severity == DriftSeverity.MINOR:
            return (
                f"delta_e_total ({delta_total:.4f}) >= minor_threshold "
                f"({self._minor_threshold}) and < moderate_threshold "
                f"({self._moderate_threshold}) -> {strategy.value.upper()}"
            )
        return "No action needed"

    def _classify_severity(self, delta_total: float) -> DriftSeverity:
        """Classify drift severity based on delta_e_total thresholds."""
        if delta_total >= self._severe_threshold:
            return DriftSeverity.SEVERE
        elif delta_total >= self._moderate_threshold:
            return DriftSeverity.MODERATE
        elif delta_total >= self._minor_threshold:
            return DriftSeverity.MINOR
        else:
            return DriftSeverity.NONE

    def _map_strategy(self, severity: DriftSeverity) -> AdaptationStrategy:
        """Map drift severity to adaptation strategy."""
        mapping = {
            DriftSeverity.MINOR: AdaptationStrategy.REWEIGHT,
            DriftSeverity.MODERATE: AdaptationStrategy.INCREMENTAL_RETRAIN,
            DriftSeverity.SEVERE: AdaptationStrategy.FULL_RETRAIN,
        }
        return mapping.get(severity, AdaptationStrategy.NONE)

    def _log_fsm_transition(
        self,
        state_before: FSMState,
        state_after: FSMState,
        sample_idx: int,
        trigger: str,
    ) -> None:
        """
        Log every FSM state transition for auditability.

        Records state_before, state_after, trigger reason, and timestamp.
        """
        logger.info(
            f"FSM transition: {state_before.value} -> {state_after.value} "
            f"at sample {sample_idx} | trigger: {trigger}"
        )

    def begin_adaptation(self) -> None:
        """Transition FSM to ADAPTING state."""
        state_before = self._state
        self._state = FSMState.ADAPTING
        self._adaptation_count += 1

        # Log FSM transition with state_before/state_after
        self._log_fsm_transition(
            state_before, FSMState.ADAPTING, -1,
            f"Adaptation #{self._adaptation_count} initiated"
        )

    def complete_adaptation(self, sample_idx: int) -> None:
        """Transition FSM to COOLDOWN state after adaptation completes."""
        state_before = self._state
        self._state = FSMState.COOLDOWN
        self._cooldown_start_idx = sample_idx

        # Log FSM transition with state_before/state_after
        self._log_fsm_transition(
            state_before, FSMState.COOLDOWN, sample_idx,
            f"Adaptation complete, cooldown for {self._cooldown_samples} samples"
        )

    def force_idle(self) -> None:
        """Force FSM back to IDLE (for testing/recovery)."""
        state_before = self._state
        self._state = FSMState.IDLE
        self._log_fsm_transition(state_before, FSMState.IDLE, -1, "Forced reset")

    @property
    def state(self) -> FSMState:
        """Current FSM state."""
        return self._state

    @property
    def adaptation_count(self) -> int:
        """Total number of adaptations triggered."""
        return self._adaptation_count

    @property
    def decision_history(self) -> list[dict]:
        """History of all adaptation decisions with reasoning."""
        return self._decision_history.copy()
