"""
DriftInsights — Database Layer (MongoDB Atlas via pymongo)

Replaces SQLAlchemy/PostgreSQL with MongoDB Atlas.
Collections:
  - drift_events
  - model_versions
  - metrics_snapshots
  - adaptation_events
  - xai_explanations
"""

import logging
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

_logger = logging.getLogger("driftinsights.database")

# ── Connection ───────────────────────────────────────────────────────────────

_client = None
_db = None

MONGO_URI = (
    os.environ.get("DRIFT_DATABASE_URL")
    or "mongodb+srv://ameymvaidya_db_user:lMQ8bcGY0AQBXzf5@cluster0.devqstn.mongodb.net/?appName=Cluster0"
)
DB_NAME = "driftinsights"


def get_db():
    """Return the MongoDB database handle, creating the client if needed."""
    global _client, _db
    if _db is None:
        from pymongo import MongoClient
        from pymongo.server_api import ServerApi
        _client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
        _db = _client[DB_NAME]
        _logger.info(f"Connected to MongoDB Atlas — database: {DB_NAME}")
    return _db


def ping():
    """Ping the deployment to confirm connection."""
    db = get_db()
    db.command("ping")


# ── Document dataclasses (replace SQLAlchemy ORM models) ────────────────────

@dataclass
class DriftEventRecord:
    session_id: str
    sample_idx: int
    pre_error_rate: float
    post_error_rate: float
    severity_estimate: float = 0.0
    adwin_width: int = 0
    detection_count: int = 0
    explanation: Optional[Any] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_doc(self) -> dict:
        return asdict(self)


@dataclass
class ModelVersionRecord:
    session_id: str
    version: int
    model_type: str
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    n_training_samples: Optional[int] = None
    file_path: Optional[str] = None
    is_active: bool = True
    adaptation_strategy: Optional[str] = None
    metadata_: Optional[Any] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_doc(self) -> dict:
        d = asdict(self)
        # rename metadata_ -> metadata for Mongo
        d["metadata"] = d.pop("metadata_", None)
        return d


@dataclass
class MetricsSnapshotRecord:
    session_id: str
    sample_idx: int
    running_accuracy: float
    running_error: float
    windowed_accuracy: Optional[float] = None
    windowed_error: Optional[float] = None
    window_size: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_doc(self) -> dict:
        return asdict(self)


@dataclass
class AdaptationRecord:
    session_id: str
    strategy: str
    success: bool
    old_accuracy: Optional[float] = None
    new_accuracy: Optional[float] = None
    threshold: Optional[float] = None
    n_samples_used: Optional[int] = None
    duration_seconds: Optional[float] = None
    model_version: Optional[int] = None
    reason: Optional[str] = None
    rollback: bool = False
    severity: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_doc(self) -> dict:
        return asdict(self)


@dataclass
class XAIRecord:
    session_id: str
    drift_event_id: Optional[int] = None
    delta_e_total: Optional[float] = None
    direction: Optional[str] = None
    top_feature: Optional[str] = None
    top_delta_e: Optional[float] = None
    explanation_json: Optional[Any] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_doc(self) -> dict:
        return asdict(self)


# ── Collection name map ──────────────────────────────────────────────────────

_COLLECTION_MAP = {
    DriftEventRecord:      "drift_events",
    ModelVersionRecord:    "model_versions",
    MetricsSnapshotRecord: "metrics_snapshots",
    AdaptationRecord:      "adaptation_events",
    XAIRecord:             "xai_explanations",
}


# ── Safe Write Helper ────────────────────────────────────────────────────────

def db_write(record) -> None:
    """
    Safely insert a single record into the appropriate MongoDB collection.
    Logs a warning on failure but never raises — pipeline speed is the priority.
    """
    try:
        collection_name = _COLLECTION_MAP.get(type(record))
        if collection_name is None:
            _logger.warning(f"Unknown record type: {type(record)}")
            return
        db = get_db()
        doc = record.to_doc()
        db[collection_name].insert_one(doc)
    except Exception as e:
        _logger.warning(f"MongoDB write failed (non-critical): {e}")


# ── Compatibility stubs (keep pipeline.py import working) ───────────────────
# pipeline.py imports: db_write, DriftEventRecord, MetricsSnapshotRecord,
#                      AdaptationRecord as AdaptationRecordDB, XAIRecord
# All are defined above — nothing extra needed.


def init_db(*args, **kwargs) -> None:
    """No-op: MongoDB creates collections automatically on first insert."""
    try:
        ping()
        _logger.info("MongoDB Atlas connection verified.")
    except Exception as e:
        _logger.warning(f"MongoDB ping failed: {e}")


def reset_engine():
    """Close and reset the MongoDB client."""
    global _client, _db
    if _client:
        _client.close()
    _client = None
    _db = None
