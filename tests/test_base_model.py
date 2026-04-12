"""Tests for the Base Model wrapper."""

import pytest
import numpy as np
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_config import sample_config_data, config_file, config
from src.config import Config
from src.base_model import BaseModel


@pytest.fixture
def training_data():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def model(config):
    return BaseModel(config)


class TestModelCreation:
    def test_create_xgboost(self, config):
        model = BaseModel(config, model_type="xgboost")
        assert model.model_type == "xgboost"

    def test_create_lightgbm(self, config):
        model = BaseModel(config, model_type="lightgbm")
        assert model.model_type == "lightgbm"

    def test_create_random_forest(self, config):
        model = BaseModel(config, model_type="random_forest")
        assert model.model_type == "random_forest"

    def test_unsupported_type_raises(self, config):
        with pytest.raises(ValueError, match="Unsupported"):
            BaseModel(config, model_type="nonexistent")


class TestTraining:
    def test_train_predict(self, model, training_data):
        X, y = training_data
        model.train(X, y, feature_names=["f0", "f1", "f2"])
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

    def test_predict_proba(self, model, training_data):
        X, y = training_data
        model.train(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_evaluate(self, model, training_data):
        X, y = training_data
        model.train(X, y)
        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert metrics["accuracy"] > 0.7  # Should learn the simple pattern

    def test_feature_names_stored(self, model, training_data):
        X, y = training_data
        model.train(X, y, feature_names=["a", "b", "c"])
        assert model.feature_names == ["a", "b", "c"]

    def test_version_increments(self, model, training_data):
        X, y = training_data
        v0 = model.version
        model.train(X, y)
        assert model.version == v0 + 1


class TestSaveLoad:
    def test_save_load(self, model, training_data, tmp_path):
        X, y = training_data
        model.train(X, y, feature_names=["f0", "f1", "f2"])
        path = tmp_path / "test_model.joblib"
        model.save(path)
        assert path.exists()

        loaded = BaseModel(model.config)
        loaded.load(path)
        assert loaded.model_type == model.model_type
        assert loaded.feature_names == model.feature_names

        # Predictions should match
        preds_orig = model.predict(X)
        preds_loaded = loaded.predict(X)
        np.testing.assert_array_equal(preds_orig, preds_loaded)

    def test_load_nonexistent_raises(self, model):
        with pytest.raises(FileNotFoundError):
            model.load("/nonexistent/model.joblib")


class TestHotSwap:
    def test_hot_swap(self, config, training_data):
        X, y = training_data
        model1 = BaseModel(config)
        model1.train(X, y)
        preds1 = model1.predict(X)

        model2 = BaseModel(config)
        # Train with flipped labels to get different predictions
        model2.train(X, 1 - y)

        v_before = model1.version
        model1.hot_swap(model2)
        assert model1.version == v_before + 1

    def test_create_fresh(self, model, training_data):
        X, y = training_data
        model.train(X, y, feature_names=["a", "b", "c"])
        fresh = model.create_fresh()
        assert fresh.model_type == model.model_type
        assert fresh.feature_names == model.feature_names
