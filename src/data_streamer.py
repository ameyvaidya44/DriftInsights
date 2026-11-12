"""
DriftInsights — Data Streaming Engine

StreamSimulator feeds production-like records row-by-row or batch-by-batch
and maintains sliding buffers for predictions and delayed ground-truth labels.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Generator, Iterator, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("driftinsights.data_streamer")


@dataclass
class StreamBatch:
    """One yielded chunk from the simulated production stream."""

    index: int
    X: np.ndarray
    y: Optional[np.ndarray]
    raw: pd.DataFrame


@dataclass
class LabeledPair:
    """A feature row paired with prediction and (possibly delayed) label."""

    index: int
    x: np.ndarray
    y_true: int
    y_pred: int


class StreamSimulator:
    """
    Iterator/generator that replays a streaming DataFrame.

    Delay handling:
      Features (and predictions) arrive immediately. Ground-truth labels
      become available after `label_delay` records, matching delayed
      supervision in production scoring pipelines.
    """

    def __init__(
        self,
        df_stream: pd.DataFrame,
        target_column: str,
        batch_size: int = 1,
        label_delay: int = 0,
        buffer_size: int = 500,
        mode: str = "batch",
        inter_sample_delay: float = 0.0,
        transform_fn=None,
    ) -> None:
        if target_column not in df_stream.columns:
            raise ValueError(f"target_column '{target_column}' missing from stream frame")

        self.df_stream = df_stream.reset_index(drop=True)
        self.target_column = target_column
        self.batch_size = max(1, int(batch_size))
        self.label_delay = max(0, int(label_delay))
        self.buffer_size = max(1, int(buffer_size))
        self.mode = mode
        self.inter_sample_delay = float(inter_sample_delay)
        self.transform_fn = transform_fn

        # Sliding windows
        self._pred_buffer: deque[tuple[int, np.ndarray, int]] = deque(maxlen=self.buffer_size)
        self._delay_queue: deque[tuple[int, np.ndarray, int, np.ndarray]] = deque()
        self._labeled_pairs: deque[LabeledPair] = deque(maxlen=self.buffer_size)
        self._X_window: deque[np.ndarray] = deque(maxlen=self.buffer_size)
        self._y_window: deque[int] = deque(maxlen=self.buffer_size)
        self._pred_window: deque[int] = deque(maxlen=self.buffer_size)

        self.records_emitted = 0
        self.labels_released = 0

    def __iter__(self) -> Iterator[StreamBatch]:
        return self.stream()

    def stream(self) -> Generator[StreamBatch, None, None]:
        n = len(self.df_stream)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            raw = self.df_stream.iloc[start:end]
            if self.transform_fn is not None:
                X, y = self.transform_fn(raw)
            else:
                y = raw[self.target_column].to_numpy()
                X = raw.drop(columns=[self.target_column]).to_numpy(dtype=float)

            yield StreamBatch(index=start, X=X, y=y, raw=raw)
            self.records_emitted += len(X)

            if self.mode == "realtime" and self.inter_sample_delay > 0:
                time.sleep(self.inter_sample_delay)

    def record_predictions(self, batch: StreamBatch, y_pred: np.ndarray) -> list[LabeledPair]:
        """
        Store predictions and release any pairs whose label delay has elapsed.

        Returns newly available (x, y_true, y_pred) tuples for ADWIN updates.
        """
        y_pred = np.asarray(y_pred).reshape(-1)
        released: list[LabeledPair] = []

        for i in range(len(batch.X)):
            global_idx = batch.index + i
            x_i = np.asarray(batch.X[i])
            pred_i = int(y_pred[i])
            true_i = int(batch.y[i]) if batch.y is not None else None

            self._pred_buffer.append((global_idx, x_i, pred_i))

            if true_i is None:
                continue

            if self.label_delay <= 0:
                released.extend(self._commit_pair(global_idx, x_i, true_i, pred_i))
            else:
                self._delay_queue.append((global_idx, x_i, pred_i, np.array([true_i])))

        if self.label_delay > 0:
            while len(self._delay_queue) > self.label_delay:
                delayed_idx, delayed_x, delayed_pred, delayed_y = self._delay_queue.popleft()
                released.extend(
                    self._commit_pair(delayed_idx, delayed_x, int(delayed_y[0]), delayed_pred)
                )

        return released

    def get_recent_window(self, size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return the current post-drift data window (W_after) of labeled samples."""
        if not self._X_window:
            return np.empty((0, 0)), np.empty((0,), dtype=int)
        k = size or len(self._X_window)
        X = np.vstack(list(self._X_window)[-k:])
        y = np.asarray(list(self._y_window)[-k:], dtype=int)
        return X, y

    def get_holdout_batch(self, size: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Most recent labeled samples, used as a validation batch for M_{t+1}."""
        return self.get_recent_window(size=size)

    def labeled_history(self) -> list[LabeledPair]:
        return list(self._labeled_pairs)

    def _commit_pair(self, index: int, x: np.ndarray, y_true: int, y_pred: int) -> list[LabeledPair]:
        pair = LabeledPair(index=index, x=x, y_true=y_true, y_pred=y_pred)
        self._labeled_pairs.append(pair)
        self._X_window.append(x)
        self._y_window.append(y_true)
        self._pred_window.append(y_pred)
        self.labels_released += 1
        return [pair]
