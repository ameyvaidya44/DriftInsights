"""Tests for the Config system."""

import os
import pytest
import tempfile
from pathlib import Path

import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, ConfigNode, ConfigError


@pytest.fixture
def sample_config_data():
    return {
        "project": {"name": "TestProject", "version": "0.1.0", "session_id": None},
        "data": {
            "dataset": "sea",
            "sea": {"n_samples": 1000, "noise": 0.1, "drift_position": 500, "n_features": 3},
            "elec2": {"path": "data/elec2.csv", "url": "https://example.com/elec2.csv"},
            "split": {"train": 0.6, "val": 0.15, "test": 0.25},
            "streaming": {"batch_size": 1, "label_delay": 0, "buffer_size": 200},
        },
        "model": {
            "type": "xgboost",
            "params": {
                "xgboost": {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1, "eval_metric": "logloss", "random_state": 42},
                "lightgbm": {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1, "random_state": 42, "verbose": -1},
                "random_forest": {"n_estimators": 10, "max_depth": 5, "random_state": 42},
            },
            "save_dir": "models",
        },
        "logging": {"dir": "logs", "level": "WARNING", "console": False, "file": False},
        "drift": {"method": "adwin", "adwin": {"delta": 0.002}, "min_gap": 50, "warmup_samples": 20},
        "xai": {"shap": {"max_samples": 50, "top_k_features": 3}, "window_size": 50},
        "decision": {"thresholds": {"minor": 0.05, "moderate": 0.15, "severe": 0.30}, "cooldown_samples": 100},
        "adaptation": {
            "reweight": {"decay_factor": 0.95},
            "incremental": {"window_size": 200, "warm_start": True},
            "full_retrain": {"min_samples": 100},
            "validation": {"accuracy_threshold": 0.6},
        },
    }


@pytest.fixture
def config_file(sample_config_data, tmp_path):
    path = tmp_path / "test_config.yaml"
    with open(path, "w") as f:
        yaml.dump(sample_config_data, f)
    return path


@pytest.fixture
def config(config_file):
    return Config(config_file)


class TestConfigNode:
    def test_dot_notation_access(self):
        node = ConfigNode({"a": 1, "b": {"c": 2}})
        assert node.a == 1
        assert node.b.c == 2

    def test_missing_key_raises(self):
        node = ConfigNode({"a": 1})
        with pytest.raises(ConfigError, match="not found"):
            _ = node.b

    def test_get_with_default(self):
        node = ConfigNode({"a": 1})
        assert node.get("a") == 1
        assert node.get("missing", 42) == 42

    def test_contains(self):
        node = ConfigNode({"a": 1})
        assert "a" in node
        assert "b" not in node

    def test_to_dict(self):
        node = ConfigNode({"a": 1, "b": 2})
        assert node.to_dict() == {"a": 1, "b": 2}


class TestConfig:
    def test_load_from_file(self, config):
        assert config.project.name == "TestProject"
        assert config.model.type == "xgboost"

    def test_nested_access(self, config):
        assert config.data.sea.n_samples == 1000
        assert config.model.params.xgboost.n_estimators == 10

    def test_from_dict(self, sample_config_data):
        cfg = Config.from_dict(sample_config_data)
        assert cfg.project.name == "TestProject"

    def test_missing_file_raises(self):
        with pytest.raises(ConfigError, match="not found"):
            Config("/nonexistent/config.yaml")

    def test_validation_missing_keys(self, tmp_path):
        path = tmp_path / "bad.yaml"
        with open(path, "w") as f:
            yaml.dump({"some_key": "value"}, f)
        with pytest.raises(ConfigError, match="Missing required"):
            Config(path)
