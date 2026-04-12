"""Shared utility functions for DriftInsights."""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


def generate_session_id() -> str:
    """Generate a unique session identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"session_{timestamp}_{short_uuid}"


def ensure_directory(path: str | Path) -> Path:
    """Create directory if it doesn't exist. Returns Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_project_root() -> Path:
    """Get the project root directory (where config/ lives)."""
    current = Path(__file__).resolve()
    # Walk up from src/utils/helpers.py → src/utils → src → project_root
    for parent in [current.parent, current.parent.parent, current.parent.parent.parent]:
        if (parent / "config").is_dir():
            return parent
    # Fallback to cwd
    return Path.cwd()


def timestamp_now() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator
