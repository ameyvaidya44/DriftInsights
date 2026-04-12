"""Tests for Drift Detector, XAI Engine, Decision Engine, and Adaptation Module.

Updated for enhanced interfaces (v2):
  - XAI: FeatureDriftDetail with direction + stats
  - Decision: AdaptationCommand with reason
  - Adaptation: tolerance-based validation, rollback tracking, model versioning
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_config import sample_config_data, config_file, config
from src.config import Config
from src.drift_detector import DriftDetector, DriftEvent
from src.xai_engine import XAIEngine, DriftExplanation, FeatureDriftDetail
from src.decision_engine import (
    DecisionEngine, AdaptationStrategy, DriftSeverity, FSMState, AdaptationCommand,
)
from src.adaptation_module import AdaptationModule, AdaptationResult, ModelVersionMetadata
from src.base_model import BaseModel
from src.metrics_tracker import MetricsTracker


# ── Drift Detector Tests ───────────────────────────────────────────────────

class TestDriftDetector:
    def test_no_drift_stable_stream(self, config):
        detector = DriftDetector(config)
        # Feed a stable stream of low error
        for i in range(300):
            event = detector.update(0, i)
        assert event is None
        assert detector.detection_count == 0

    def test_detects_drift(self, config):
        detector = DriftDetector(config)
        events = []
        # Stable period (low error)
        for i in range(200):
            e = detector.update(0 if np.random.random() > 0.1 else 1, i)
            if e:
                events.append(e)
        # Drift: sudden high error
        for i in range(200, 500):
            e = detector.update(1 if np.random.random() > 0.2 else 0, i)
            if e:
                events.append(e)
        assert len(events) > 0
        assert isinstance(events[0], DriftEvent)

    def test_min_gap_enforced(self, config):
        detector = DriftDetector(config)
        # Should not fire two detections within min_gap
        events = []
        for i in range(1000):
            error = 0 if i < 300 else (1 if i < 400 else (0 if i < 450 else 1))
            e = detector.update(error, i)
            if e:
                events.append(e)
        if len(events) >= 2:
            gap = events[1].sample_idx - events[0].sample_idx
            assert gap >= config.drift.min_gap

    def test_reset(self, config):
        detector = DriftDetector(config)
        for i in range(100):
            detector.update(0, i)
        detector.reset()
        assert detector.width == 0

    def test_drift_event_to_dict(self):
        event = DriftEvent(
            timestamp="2024-01-01",
            sample_idx=100,
            pre_error_rate=0.1,
            post_error_rate=0.4,
            adwin_width=50,
            detection_count=1,
        )
        d = event.to_dict()
        assert d["sample_idx"] == 100
        assert d["pre_error_rate"] == 0.1


# ── XAI Engine Tests (Enhanced) ────────────────────────────────────────────

class TestXAIEngine:
    @pytest.fixture
    def trained_model(self, config):
        rng = np.random.RandomState(42)
        X = rng.randn(300, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = BaseModel(config, model_type="xgboost")
        model.train(X, y, feature_names=["f0", "f1", "f2"])
        return model

    def test_explain_drift_enhanced(self, config, trained_model):
        """Test that explain_drift returns FeatureDriftDetail with direction and stats."""
        xai = XAIEngine(config)
        xai.set_model(trained_model.estimator)

        rng = np.random.RandomState(42)
        X_pre = rng.randn(50, 3)
        X_post = rng.randn(50, 3) + 2.0  # Shifted distribution

        explanation = xai.explain_drift(X_pre, X_post, ["f0", "f1", "f2"])

        assert isinstance(explanation, DriftExplanation)
        assert explanation.delta_total >= 0
        assert len(explanation.top_features) > 0
        assert explanation.direction in ("increase", "decrease", "mixed")

        # Enhanced: check FeatureDriftDetail structure
        top = explanation.top_features[0]
        assert isinstance(top, FeatureDriftDetail)
        assert top.direction in ("increase", "decrease")
        assert isinstance(top.mean_before, float)
        assert isinstance(top.mean_after, float)
        assert isinstance(top.std_before, float)
        assert isinstance(top.std_after, float)
        assert isinstance(top.shap_before, float)
        assert isinstance(top.shap_after, float)

    def test_explanation_to_dict_format(self, config, trained_model):
        """Test enhanced output format matches specification."""
        xai = XAIEngine(config)
        xai.set_model(trained_model.estimator)

        rng = np.random.RandomState(42)
        X_pre = rng.randn(50, 3)
        X_post = rng.randn(50, 3) + 1.0

        explanation = xai.explain_drift(X_pre, X_post, ["f0", "f1", "f2"])
        d = explanation.to_dict()

        # Check enhanced output format
        assert "delta_e_total" in d
        assert "direction" in d
        assert "top_features" in d
        assert len(d["top_features"]) > 0

        feature = d["top_features"][0]
        assert "feature" in feature
        assert "delta" in feature
        assert "direction" in feature
        assert "mean_before" in feature
        assert "mean_after" in feature
        assert "std_before" in feature
        assert "std_after" in feature

    def test_feature_drift_summary(self, config, trained_model):
        """Test feature drift summary across multiple events (Bonus)."""
        xai = XAIEngine(config)
        xai.set_model(trained_model.estimator)

        rng = np.random.RandomState(42)
        # Generate two drift explanations
        for _ in range(2):
            X_pre = rng.randn(50, 3)
            X_post = rng.randn(50, 3) + 1.5
            xai.explain_drift(X_pre, X_post, ["f0", "f1", "f2"])

        summary = xai.get_feature_drift_summary(last_n=5)
        assert "events_analyzed" in summary
        assert summary["events_analyzed"] == 2
        assert "top_drifting_features" in summary
        assert len(summary["top_drifting_features"]) > 0

    def test_explanation_history(self, config, trained_model):
        """Test that explanation history is stored."""
        xai = XAIEngine(config)
        xai.set_model(trained_model.estimator)

        rng = np.random.RandomState(42)
        X_pre = rng.randn(50, 3)
        X_post = rng.randn(50, 3) + 1.0
        xai.explain_drift(X_pre, X_post, ["f0", "f1", "f2"])

        assert len(xai.explanation_history) == 1


# ── Decision Engine Tests (Enhanced) ───────────────────────────────────────

class TestDecisionEngine:
    def test_initial_state(self, config):
        engine = DecisionEngine(config)
        assert engine.state == FSMState.IDLE

    def test_minor_drift_with_reason(self, config):
        """Test that minor drift includes reasoning."""
        engine = DecisionEngine(config)
        cmd = engine.evaluate(delta_total=0.08, sample_idx=100)
        assert cmd is not None
        assert cmd.strategy == AdaptationStrategy.REWEIGHT
        assert cmd.severity == DriftSeverity.MINOR
        # Enhanced: check reasoning
        assert "minor_threshold" in cmd.reason
        assert "REWEIGHT" in cmd.reason

    def test_moderate_drift_with_reason(self, config):
        engine = DecisionEngine(config)
        cmd = engine.evaluate(delta_total=0.20, sample_idx=100)
        assert cmd.strategy == AdaptationStrategy.INCREMENTAL_RETRAIN
        assert "moderate_threshold" in cmd.reason
        assert "INCREMENTAL_RETRAIN" in cmd.reason

    def test_severe_drift_with_reason(self, config):
        engine = DecisionEngine(config)
        cmd = engine.evaluate(delta_total=0.50, sample_idx=100)
        assert cmd.strategy == AdaptationStrategy.FULL_RETRAIN
        assert "severe_threshold" in cmd.reason
        assert "FULL_RETRAIN" in cmd.reason

    def test_below_threshold_no_action(self, config):
        engine = DecisionEngine(config)
        cmd = engine.evaluate(delta_total=0.01, sample_idx=100)
        assert cmd is None

    def test_fsm_cooldown(self, config):
        engine = DecisionEngine(config)
        cmd = engine.evaluate(delta_total=0.20, sample_idx=100)
        assert cmd is not None

        engine.begin_adaptation()
        assert engine.state == FSMState.ADAPTING

        engine.complete_adaptation(sample_idx=200)
        assert engine.state == FSMState.COOLDOWN

        cmd2 = engine.evaluate(delta_total=0.50, sample_idx=250)
        assert cmd2 is None

    def test_cooldown_expires(self, config):
        engine = DecisionEngine(config)
        engine.evaluate(0.20, 100)
        engine.begin_adaptation()
        engine.complete_adaptation(100)

        cmd = engine.evaluate(0.20, 100 + config.decision.cooldown_samples + 1)
        assert cmd is not None

    def test_adapting_blocks(self, config):
        engine = DecisionEngine(config)
        engine.begin_adaptation()
        cmd = engine.evaluate(0.50, 100)
        assert cmd is None

    def test_command_to_dict_has_reason(self, config):
        engine = DecisionEngine(config)
        cmd = engine.evaluate(0.20, 100, top_features=[("f0", 0.15)])
        d = cmd.to_dict()
        assert d["strategy"] == "incremental_retrain"
        assert d["severity"] == "moderate"
        assert "reason" in d
        assert len(d["reason"]) > 0

    def test_decision_history_with_reasoning(self, config):
        engine = DecisionEngine(config)
        engine.evaluate(0.20, 100)
        assert len(engine.decision_history) == 1
        assert "reason" in engine.decision_history[0]


# ── Metrics Tracker Tests ──────────────────────────────────────────────────

class TestMetricsTracker:
    def test_update_and_accuracy(self):
        tracker = MetricsTracker(window_size=50)
        for _ in range(80):
            tracker.update(1, 1)
        for _ in range(20):
            tracker.update(1, 0)
        assert tracker.get_running_accuracy() == 0.8
        assert tracker.total_samples == 100

    def test_windowed_error(self):
        tracker = MetricsTracker(window_size=20)
        for _ in range(100):
            tracker.update(1, 1)
        for _ in range(20):
            tracker.update(1, 0)
        assert tracker.get_windowed_error() == 1.0

    def test_reset(self):
        tracker = MetricsTracker()
        tracker.update(1, 1)
        tracker.reset()
        assert tracker.total_samples == 0

    def test_snapshot(self):
        tracker = MetricsTracker()
        tracker.update(1, 1)
        snap = tracker.get_snapshot()
        assert "running_accuracy" in snap
        assert "windowed_error" in snap


# ── Adaptation Module Tests (Enhanced) ─────────────────────────────────────

class TestAdaptationModule:
    @pytest.fixture
    def setup(self, config):
        rng = np.random.RandomState(42)
        X = rng.randn(300, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = BaseModel(config, model_type="random_forest")
        model.train(X, y, feature_names=["f0", "f1", "f2"])
        X_val = rng.randn(100, 3)
        y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(int)
        return model, X, y, X_val, y_val

    def test_reweight_with_enhanced_result(self, config, setup):
        model, X, y, X_val, y_val = setup
        adapter = AdaptationModule(config)
        cmd = AdaptationCommand(
            strategy=AdaptationStrategy.REWEIGHT,
            severity=DriftSeverity.MINOR,
            delta_total=0.08,
            timestamp="now",
            drift_sample_idx=100,
            top_features=[],
            reason="test",
        )
        result = adapter.execute(cmd, model, X, y, X_val, y_val, ["f0", "f1", "f2"])
        assert isinstance(result, AdaptationResult)
        assert result.strategy == "reweight"
        # Enhanced: check new fields
        assert isinstance(result.rollback, bool)
        assert isinstance(result.tolerance, float)
        assert isinstance(result.duration_ms, int)
        # Check output format
        d = result.to_dict()
        assert "status" in d
        assert d["status"] in ("success", "failure")

    def test_validation_gate_with_tolerance(self, config, setup):
        """Test that validation now uses tolerance-based comparison."""
        model, X, y, X_val, y_val = setup
        adapter = AdaptationModule(config)
        # Set impossibly high threshold
        adapter._accuracy_threshold = 1.0
        cmd = AdaptationCommand(
            strategy=AdaptationStrategy.REWEIGHT,
            severity=DriftSeverity.MINOR,
            delta_total=0.08,
            timestamp="now",
            drift_sample_idx=100,
            top_features=[],
            reason="test",
        )
        result = adapter.execute(cmd, model, X, y, X_val, y_val, ["f0", "f1", "f2"])
        assert result.success is False
        assert result.rollback is True
        assert "below threshold" in result.reason or "below old_accuracy" in result.reason

    def test_rollback_tracking(self, config, setup):
        """Test that failed adaptations track rollback events."""
        model, X, y, X_val, y_val = setup
        adapter = AdaptationModule(config)
        adapter._accuracy_threshold = 1.0  # Force failure
        cmd = AdaptationCommand(
            strategy=AdaptationStrategy.REWEIGHT,
            severity=DriftSeverity.MINOR,
            delta_total=0.08,
            timestamp="now",
            drift_sample_idx=100,
            top_features=[],
            reason="test",
        )
        adapter.execute(cmd, model, X, y, X_val, y_val, ["f0", "f1", "f2"])
        assert adapter.failed_adaptations_count == 1
        assert len(adapter.rollback_events) == 1

    def test_model_versioning(self, config, setup):
        """Test model version tracking on successful adaptation."""
        model, X, y, X_val, y_val = setup
        adapter = AdaptationModule(config)
        assert adapter.current_version_string == "v1.0.0"

        cmd = AdaptationCommand(
            strategy=AdaptationStrategy.INCREMENTAL_RETRAIN,
            severity=DriftSeverity.MODERATE,
            delta_total=0.20,
            timestamp="now",
            drift_sample_idx=200,
            top_features=[],
            reason="test",
        )
        result = adapter.execute(cmd, model, X, y, X_val, y_val, ["f0", "f1", "f2"])
        if result.success:
            assert adapter.current_version_string == "v1.0.1"
            assert len(adapter.version_history) == 1

    def test_full_retrain_bumps_minor(self, config, setup):
        """Test that full retrain bumps minor version."""
        model, X, y, X_val, y_val = setup
        adapter = AdaptationModule(config)
        cmd = AdaptationCommand(
            strategy=AdaptationStrategy.FULL_RETRAIN,
            severity=DriftSeverity.SEVERE,
            delta_total=0.50,
            timestamp="now",
            drift_sample_idx=300,
            top_features=[],
            reason="test",
        )
        result = adapter.execute(cmd, model, X, y, X_val, y_val, ["f0", "f1", "f2"])
        if result.success:
            assert adapter.current_version_string == "v1.1.0"

    def test_recent_history_for_dashboard(self, config, setup):
        """Test last 5 adaptation attempts for dashboard (Bonus)."""
        model, X, y, X_val, y_val = setup
        adapter = AdaptationModule(config)
        cmd = AdaptationCommand(
            strategy=AdaptationStrategy.REWEIGHT,
            severity=DriftSeverity.MINOR,
            delta_total=0.08,
            timestamp="now",
            drift_sample_idx=100,
            top_features=[],
            reason="test",
        )
        adapter.execute(cmd, model, X, y, X_val, y_val, ["f0", "f1", "f2"])
        recent = adapter.recent_history
        assert len(recent) <= 5
        assert "status" in recent[0]

    def test_version_history_persistence(self, config, setup, tmp_path):
        """Test saving version history to JSON file."""
        model, X, y, X_val, y_val = setup
        adapter = AdaptationModule(config)
        cmd = AdaptationCommand(
            strategy=AdaptationStrategy.REWEIGHT,
            severity=DriftSeverity.MINOR,
            delta_total=0.08,
            timestamp="now",
            drift_sample_idx=100,
            top_features=[],
            reason="test",
        )
        result = adapter.execute(cmd, model, X, y, X_val, y_val, ["f0", "f1", "f2"])
        if result.success:
            path = tmp_path / "version_history.json"
            adapter.save_version_history(path)
            assert path.exists()
