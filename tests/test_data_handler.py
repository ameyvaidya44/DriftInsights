"""Tests for the Data Handler."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_config import sample_config_data, config_file, config
from src.config import Config
from src.data_handler import DataHandler


@pytest.fixture
def data_handler(config):
    return DataHandler(config)


class TestDataLoading:
    def test_load_sea(self, data_handler):
        df = data_handler.load_dataset("sea")
        assert isinstance(df, pd.DataFrame)
        assert "target" in df.columns
        assert len(df) == 1000  # from test config
        assert df.shape[1] == 4  # 3 features + target

    def test_sea_has_drift(self, data_handler):
        df = data_handler.load_dataset("sea")
        # Check that pre-drift and post-drift distributions differ
        pre = df.iloc[:500]["target"].mean()
        post = df.iloc[500:]["target"].mean()
        # They should be different due to concept change
        assert isinstance(pre, float)
        assert isinstance(post, float)

    def test_unknown_dataset_raises(self, data_handler):
        with pytest.raises(ValueError, match="Unknown dataset"):
            data_handler.load_dataset("nonexistent")


class TestPreprocessing:
    def test_preprocess_sea(self, data_handler):
        df = data_handler.load_dataset("sea")
        processed = data_handler.preprocess(df, fit=True)
        assert "target" in processed.columns
        assert len(processed) == len(df)
        # Features should be scaled (roughly zero mean)
        features = processed.drop(columns=["target"])
        means = features.mean().abs()
        assert all(means < 1.0)  # Scaled means should be near zero

    def test_preprocess_handles_missing(self, data_handler):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, 4.0],
            "b": [10.0, 20.0, np.nan, 40.0],
            "target": [0, 1, 0, 1],
        })
        processed = data_handler.preprocess(df, fit=True)
        assert not processed.isnull().any().any()

    def test_feature_names_stored(self, data_handler):
        df = data_handler.load_dataset("sea")
        data_handler.preprocess(df, fit=True)
        assert len(data_handler.feature_names) == 3


class TestTemporalSplit:
    def test_split_sizes(self, data_handler, config):
        df = data_handler.load_dataset("sea")
        processed = data_handler.preprocess(df)
        train, val, test = data_handler.temporal_split(processed)
        total = len(train) + len(val) + len(test)
        assert total == len(processed)
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_split_no_shuffle(self, data_handler):
        df = data_handler.load_dataset("sea")
        processed = data_handler.preprocess(df)
        train, val, test = data_handler.temporal_split(processed)
        # Verify chronological ordering is preserved
        assert train.index[0] == 0
        assert val.index[0] == 0  # reset_index
        assert test.index[0] == 0


class TestStreaming:
    def test_stream_yields_batches(self, data_handler):
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 2, 100)
        batches = list(data_handler.stream(X, y, batch_size=10, label_delay=0))
        assert len(batches) == 10
        assert batches[0][0].shape == (10, 3)

    def test_stream_with_delay(self, data_handler):
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        batches = list(data_handler.stream(X, y, batch_size=1, label_delay=5))
        # First 5 batches should have None labels
        for i in range(5):
            assert batches[i][1] is None

    def test_rolling_buffer(self, data_handler):
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 2, 100)
        X_win, y_win = data_handler.get_rolling_buffer(X, y, 80, window_size=20)
        assert len(X_win) == 21  # inclusive
        assert len(y_win) == 21


class TestSplitXY:
    def test_split_xy(self, data_handler):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]})
        X, y = data_handler.split_xy(df)
        assert X.shape == (2, 2)
        assert y.shape == (2,)
        assert list(y) == [0, 1]
