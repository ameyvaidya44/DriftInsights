"""
DriftInsights — Database Models (SQLAlchemy + PostgreSQL)

Defines ORM models for drift_events, model_versions, and metrics_snapshots.
"""

import os
from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, JSON, Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

Base = declarative_base()


# ── ORM Models ──────────────────────────────────────────────────────────────

class DriftEventRecord(Base):
    """Persisted drift detection event."""
    __tablename__ = "drift_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    sample_idx = Column(Integer, nullable=False)
    pre_error_rate = Column(Float, nullable=False)
    post_error_rate = Column(Float, nullable=False)
    severity_estimate = Column(Float, default=0.0)
    adwin_width = Column(Integer, default=0)
    detection_count = Column(Integer, default=0)
    explanation = Column(JSON, nullable=True)  # SHAP explanation data


class ModelVersionRecord(Base):
    """Persisted model version metadata."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), index=True, nullable=False)
    version = Column(Integer, nullable=False)
    model_type = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    accuracy = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    n_training_samples = Column(Integer, nullable=True)
    file_path = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True)
    adaptation_strategy = Column(String(50), nullable=True)
    metadata_ = Column("metadata", JSON, nullable=True)


class MetricsSnapshotRecord(Base):
    """Persisted metrics snapshot."""
    __tablename__ = "metrics_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    sample_idx = Column(Integer, nullable=False)
    running_accuracy = Column(Float, nullable=False)
    running_error = Column(Float, nullable=False)
    windowed_accuracy = Column(Float, nullable=True)
    windowed_error = Column(Float, nullable=True)
    window_size = Column(Integer, nullable=True)


class AdaptationRecord(Base):
    """Persisted adaptation event."""
    __tablename__ = "adaptation_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    strategy = Column(String(50), nullable=False)
    severity = Column(String(50), nullable=True)
    success = Column(Boolean, nullable=False)
    old_accuracy = Column(Float, nullable=True)
    new_accuracy = Column(Float, nullable=True)
    threshold = Column(Float, nullable=True)
    n_samples_used = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    model_version = Column(Integer, nullable=True)
    reason = Column(Text, nullable=True)
    rollback = Column(Boolean, default=False)


class XAIRecord(Base):
    """Persisted XAI explanation linked to a drift event."""
    __tablename__ = "xai_explanations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), index=True, nullable=False)
    drift_event_id = Column(Integer, nullable=True)  # FK to drift_events.id
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    delta_e_total = Column(Float, nullable=True)
    direction = Column(String(20), nullable=True)
    top_feature = Column(String(100), nullable=True)
    top_delta_e = Column(Float, nullable=True)
    explanation_json = Column(JSON, nullable=True)


# ── Database Engine / Session ───────────────────────────────────────────────

_engine = None
_SessionLocal = None


def get_engine(database_url: str | None = None, echo: bool = False):
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        url = database_url or os.environ.get(
            "DRIFT_DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/driftinsights"
        )
        _engine = create_engine(url, echo=echo, pool_pre_ping=True)
    return _engine


def get_session_factory(database_url: str | None = None) -> sessionmaker:
    """Get a session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine(database_url)
        _SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return _SessionLocal


def get_db_session(database_url: str | None = None) -> Session:
    """Get a new database session."""
    factory = get_session_factory(database_url)
    return factory()


def init_db(database_url: str | None = None, echo: bool = False) -> None:
    """Create all database tables."""
    engine = get_engine(database_url, echo=echo)
    Base.metadata.create_all(bind=engine)


def reset_engine():
    """Reset the global engine (for testing)."""
    global _engine, _SessionLocal
    if _engine:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


# ── Safe Write Helper ───────────────────────────────────────────────────────

import logging as _logging
_db_logger = _logging.getLogger("driftinsights.database")


def db_write(record) -> None:
    """
    Safely write a single ORM record to the database.
    Logs a warning on failure but never raises — pipeline speed is the priority.
    """
    try:
        session = get_db_session()
        try:
            session.add(record)
            session.commit()
        except Exception as e:
            session.rollback()
            _db_logger.warning(f"DB write failed (non-critical): {e}")
        finally:
            session.close()
    except Exception as e:
        _db_logger.warning(f"DB session creation failed (non-critical): {e}")
