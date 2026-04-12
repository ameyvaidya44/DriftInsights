"""
DriftInsights — Base Model Wrapper

Unified interface for XGBoost, LightGBM, and RandomForest models
with train, predict, evaluate, save/load, and hot_swap capabilities.
"""

import logging
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logger = logging.getLogger("driftinsights.base_model")


class BaseModel:
    """
    Unified model wrapper supporting XGBoost, LightGBM, and RandomForest.

    Provides a consistent interface for training, prediction, evaluation,
    serialization, and atomic model swapping (hot_swap).

    Usage:
        model = BaseModel(config)
        model.train(X_train, y_train)
        preds = model.predict(X_test)
        metrics = model.evaluate(X_test, y_test)
        model.save("models/model_v1.joblib")
        model.hot_swap(new_estimator)
    """

    SUPPORTED_TYPES = {"xgboost", "lightgbm", "random_forest"}

    def __init__(self, config, model_type: str | None = None):
        self.config = config
        self.model_type = model_type or config.model.type
        self.feature_names: list[str] = []
        self._lock = threading.Lock()
        self._model = None
        self._version = 0

        if self.model_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported model type: '{self.model_type}'. "
                f"Choose from: {self.SUPPORTED_TYPES}"
            )

        self._model = self._create_estimator()
        logger.info(f"Initialized {self.model_type} model")

    def _create_estimator(self, params_override: dict | None = None) -> Any:
        """Create a new estimator instance based on model type."""
        model_params_section = self.config.model.params

        if self.model_type == "xgboost":
            from xgboost import XGBClassifier
            params = model_params_section.xgboost.to_dict()
            if params_override:
                params.update(params_override)
            params.pop("use_label_encoder", None)
            return XGBClassifier(**params)

        elif self.model_type == "lightgbm":
            from lightgbm import LGBMClassifier
            params = model_params_section.lightgbm.to_dict()
            if params_override:
                params.update(params_override)
            return LGBMClassifier(**params)

        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            params = model_params_section.random_forest.to_dict()
            if params_override:
                params.update(params_override)
            return RandomForestClassifier(**params)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "BaseModel":
        """
        Train the model on provided data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: Optional feature names
            sample_weight: Optional per-sample weights

        Returns:
            self (for chaining)
        """
        if feature_names is not None:
            self.feature_names = feature_names

        logger.info(
            f"Training {self.model_type} on {X.shape[0]} samples, "
            f"{X.shape[1]} features"
        )

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        self._model.fit(X, y, **fit_kwargs)
        self._version += 1

        logger.info(f"Training complete (version {self._version})")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        with self._lock:
            return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        with self._lock:
            return self._model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate model performance.

        Returns:
            Dict with keys: accuracy, f1, roc_auc
        """
        y_pred = self.predict(X)
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        }

        # ROC-AUC requires probability estimates
        try:
            y_proba = self.predict_proba(X)
            if y_proba.shape[1] == 2:
                metrics["roc_auc"] = roc_auc_score(y, y_proba[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y, y_proba, multi_class="ovr", average="weighted"
                )
        except Exception:
            metrics["roc_auc"] = None

        logger.info(
            f"Evaluation — accuracy: {metrics['accuracy']:.4f}, "
            f"f1: {metrics['f1']:.4f}, roc_auc: {metrics.get('roc_auc', 'N/A')}"
        )
        return metrics

    def hot_swap(self, new_model: Any) -> None:
        """
        Atomically replace the internal estimator.

        This is thread-safe and used during model adaptation to swap in
        a newly trained model without stopping the prediction pipeline.

        Args:
            new_model: A fitted estimator (same type) or another BaseModel instance
        """
        with self._lock:
            if isinstance(new_model, BaseModel):
                self._model = new_model._model
                self.model_type = new_model.model_type
                self.feature_names = new_model.feature_names
            else:
                self._model = new_model
            self._version += 1

        logger.info(f"Hot-swapped model to version {self._version}")

    def save(self, path: str | Path | None = None) -> Path:
        """
        Save model to disk using joblib.

        Args:
            path: Save path. If None, uses config save_dir.

        Returns:
            Path where model was saved.
        """
        if path is None:
            save_dir = Path(self.config.model.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            path = save_dir / f"model_{self.model_type}_v{self._version}.joblib"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self._model,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "version": self._version,
        }
        joblib.dump(payload, path)
        logger.info(f"Model saved to {path}")
        return path

    def load(self, path: str | Path) -> "BaseModel":
        """
        Load model from disk.

        Args:
            path: Path to saved model file.

        Returns:
            self (for chaining)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        payload = joblib.load(path)
        self._model = payload["model"]
        self.model_type = payload["model_type"]
        self.feature_names = payload["feature_names"]
        self._version = payload["version"]

        logger.info(f"Model loaded from {path} (version {self._version})")
        return self

    @property
    def estimator(self) -> Any:
        """Access the underlying sklearn-compatible estimator."""
        return self._model

    @property
    def version(self) -> int:
        """Current model version number."""
        return self._version

    def create_fresh(self, params_override: dict | None = None) -> "BaseModel":
        """Create a new untrained instance with the same config."""
        fresh = BaseModel(self.config, model_type=self.model_type)
        if params_override:
            fresh._model = self._create_estimator(params_override)
        fresh.feature_names = self.feature_names.copy()
        return fresh
