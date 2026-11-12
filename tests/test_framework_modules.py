"""Tests for the Phase 1-4 DriftInsights framework modules."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_config import sample_config_data
from src.adaptation_engine import AdaptationEngine, DriftType
from src.config import Config
from src.data_streamer import StreamSimulator
from src.drift_diagnostic import DriftDiagnostic
from src.model_registry import ModelRegistry
from src.pipeline_config import PipelineManager, generate_synthetic_drift_dataset


@pytest.fixture
def tiny_config(sample_config_data, tmp_path):
    cfg = sample_config_data.copy()
    cfg["model"]["params"]["xgboost"]["n_estimators"] = 8
    cfg["model"]["params"]["xgboost"]["max_depth"] = 3
    cfg["xai"]["shap"]["max_samples"] = 40
    cfg["xai"]["shap"]["top_k_features"] = 3
    cfg["xai"]["window_size"] = 40
    cfg["drift"]["warmup_samples"] = 15
    cfg["drift"]["min_gap"] = 10
    cfg["adaptation"]["validation"]["accuracy_threshold"] = 0.4
    cfg["adaptation"]["full_retrain"]["min_samples"] = 50
    cfg["logging"]["dir"] = str(tmp_path / "logs")
    cfg["model"]["save_dir"] = str(tmp_path / "models")
    return Config.from_dict(cfg)


def test_pipeline_manager_baseline_and_shap(tiny_config):
    df_base, _ = generate_synthetic_drift_dataset(n_baseline=250, n_stream=80, drift_in_stream=20)
    manager = PipelineManager(config=tiny_config, model_type="xgboost")
    manager.load_baseline_data(df_base, "target")
    assert manager.feature_names == ["feature_0", "feature_1", "feature_2", "feature_3"]
    manager.set_base_model()
    pred = manager.predict(manager.X_holdout)
    assert len(pred) == len(manager.y_holdout)
    rankings = manager.compute_baseline_shap()
    assert len(rankings) == 4
    assert manager.shap_before is not None
    assert manager.shap_before.shape[0] == 4


def test_stream_simulator_label_delay(tiny_config):
    _, df_stream = generate_synthetic_drift_dataset(n_baseline=40, n_stream=30, drift_in_stream=10)
    manager = PipelineManager(config=tiny_config)
    manager.load_baseline_data(
        generate_synthetic_drift_dataset(n_baseline=80, n_stream=10)[0],
        "target",
    )
    manager.set_base_model()

    streamer = StreamSimulator(
        df_stream,
        target_column="target",
        batch_size=1,
        label_delay=3,
        buffer_size=50,
        transform_fn=manager.transform_features,
    )
    released_total = 0
    for batch in streamer:
        preds = manager.predict(batch.X)
        released = streamer.record_predictions(batch, preds)
        released_total += len(released)

    assert streamer.records_emitted == 30
    assert released_total == 27  # 30 - label_delay
    X_w, y_w = streamer.get_recent_window()
    assert len(X_w) == 27


def test_adaptation_classify_and_reweight(tiny_config):
    df_base, df_stream = generate_synthetic_drift_dataset(
        n_baseline=180, n_stream=60, drift_in_stream=20
    )
    manager = PipelineManager(config=tiny_config)
    manager.load_baseline_data(df_base, "target")
    manager.set_base_model()
    engine = AdaptationEngine(manager)
    assert engine.classify(0.06) == DriftType.MINOR
    assert engine.classify(0.20) == DriftType.MODERATE
    assert engine.classify(0.40) == DriftType.SEVERE

    candidate = engine.execute(0.06, np.empty((0, 4)), np.empty((0,), dtype=int))
    assert candidate.strategy == "sample_reweight"
    assert candidate.sample_weight_used is True


def test_diagnostic_and_hot_swap(tiny_config, tmp_path):
    df_base, df_stream = generate_synthetic_drift_dataset(
        n_baseline=220, n_stream=120, drift_in_stream=40
    )
    manager = PipelineManager(config=tiny_config)
    manager.load_baseline_data(df_base, "target")
    manager.set_base_model()
    manager.compute_baseline_shap()

    diagnostic = DriftDiagnostic(manager)
    X_after = manager.X_holdout
    diagnosis = diagnostic.diagnose(
        X_after,
        {"sample_idx": 10, "pre_error_rate": 0.1, "post_error_rate": 0.3, "adwin_width": 20},
    )
    assert diagnosis.delta_e_total >= 0
    assert len(diagnosis.top_features) <= 3

    engine = AdaptationEngine(manager)
    candidate = engine.execute(0.40, X_after, manager.y_holdout)
    registry = ModelRegistry(manager, diagnostic, audit_path=tmp_path / "audit.json")
    record = registry.commit(candidate, diagnosis, manager.X_holdout, manager.y_holdout, X_after=X_after)
    assert record["action"] in {"hot_swap", "rejected"}
    assert (tmp_path / "audit.json").exists()
