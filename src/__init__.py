"""DriftInsights — Explainable and Adaptive ML System for Concept Drift."""

from src.adaptation_engine import AdaptationEngine, DriftType
from src.data_streamer import StreamSimulator
from src.drift_diagnostic import DriftDiagnostic
from src.model_registry import ModelRegistry
from src.pipeline_config import PipelineManager, generate_synthetic_drift_dataset

__version__ = "1.0.0"

__all__ = [
    "PipelineManager",
    "StreamSimulator",
    "DriftDiagnostic",
    "AdaptationEngine",
    "ModelRegistry",
    "DriftType",
    "generate_synthetic_drift_dataset",
]
