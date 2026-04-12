"""End-to-end pipeline integration test."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_config import sample_config_data
from src.config import Config
from src.pipeline import Pipeline


@pytest.fixture
def pipeline_config(sample_config_data, tmp_path):
    """Create a config suitable for fast integration testing."""
    cfg = sample_config_data.copy()
    cfg["data"]["sea"]["n_samples"] = 500
    cfg["data"]["sea"]["drift_position"] = 250
    cfg["data"]["streaming"]["buffer_size"] = 100
    cfg["model"]["params"]["xgboost"]["n_estimators"] = 5
    cfg["model"]["params"]["lightgbm"]["n_estimators"] = 5
    cfg["model"]["params"]["random_forest"]["n_estimators"] = 5
    cfg["drift"]["min_gap"] = 30
    cfg["drift"]["warmup_samples"] = 10
    cfg["xai"]["window_size"] = 30
    cfg["xai"]["shap"]["max_samples"] = 30
    cfg["decision"]["cooldown_samples"] = 50
    cfg["adaptation"]["full_retrain"]["min_samples"] = 50
    cfg["adaptation"]["validation"]["accuracy_threshold"] = 0.5
    cfg["logging"]["dir"] = str(tmp_path / "logs")
    cfg["model"]["save_dir"] = str(tmp_path / "models")
    return Config.from_dict(cfg)


class TestPipelineE2E:
    def test_setup(self, pipeline_config):
        pipeline = Pipeline(config=pipeline_config)
        pipeline.setup()
        assert pipeline.model is not None
        assert pipeline.data_handler is not None
        assert pipeline.metrics is not None

    def test_train(self, pipeline_config):
        pipeline = Pipeline(config=pipeline_config)
        pipeline.setup()
        metrics = pipeline.train()
        assert metrics["accuracy"] > 0.5
        assert metrics["f1"] > 0.0
        assert pipeline.model.version >= 1

    def test_full_pipeline(self, pipeline_config):
        pipeline = Pipeline(config=pipeline_config)
        pipeline.setup()
        pipeline.train()
        summary = pipeline.run_stream()

        assert summary["total_samples"] > 0
        assert summary["elapsed_seconds"] > 0
        assert "final_metrics" in summary
        assert summary["final_metrics"]["running_accuracy"] > 0

        pipeline.shutdown()

    def test_pipeline_with_lightgbm(self, pipeline_config):
        pipeline_config._data["model"]["type"] = "lightgbm"
        pipeline = Pipeline(config=pipeline_config)
        pipeline.setup()
        metrics = pipeline.train()
        assert metrics["accuracy"] > 0.5
        pipeline.shutdown()

    def test_pipeline_with_random_forest(self, pipeline_config):
        pipeline_config._data["model"]["type"] = "random_forest"
        pipeline = Pipeline(config=pipeline_config)
        pipeline.setup()
        metrics = pipeline.train()
        assert metrics["accuracy"] > 0.5
        pipeline.shutdown()
