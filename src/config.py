"""
DriftInsights — Configuration System

Loads YAML config with validation, dot-notation access, and clear error messages.
"""

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when configuration is invalid or missing required keys."""
    pass


class ConfigNode:
    """Recursive dot-notation access wrapper for nested dictionaries."""

    def __init__(self, data: dict, path: str = "config"):
        self._data = data
        self._path = path

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            return super().__getattribute__(key)
        if key not in self._data:
            raise ConfigError(
                f"Configuration key '{self._path}.{key}' not found. "
                f"Available keys: {list(self._data.keys())}"
            )
        value = self._data[key]
        if isinstance(value, dict):
            return ConfigNode(value, path=f"{self._path}.{key}")
        return value

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"ConfigNode({self._path}, keys={list(self._data.keys())})"

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value with an optional default."""
        try:
            return self.__getattr__(key)
        except ConfigError:
            return default

    def to_dict(self) -> dict:
        """Return the underlying dictionary."""
        return self._data.copy()


# ── Required top-level keys per phase ───────────────────────────────────────
REQUIRED_KEYS = {
    "core": ["project", "data", "model", "logging"],
    "drift": ["drift"],
    "xai": ["xai"],
    "decision": ["decision"],
    "adaptation": ["adaptation"],
    "api": ["api", "database"],
}


class Config(ConfigNode):
    """
    Main configuration class.

    Usage:
        cfg = Config()                          # loads default.yaml
        cfg = Config("config/custom.yaml")      # loads custom file
        cfg = Config.from_dict({...})            # from dictionary

        model_type = cfg.model.type             # dot-notation access
        lr = cfg.model.params.xgboost.learning_rate
    """

    def __init__(self, config_path: str | Path | None = None, data: dict | None = None):
        if data is not None:
            super().__init__(data)
            return

        if config_path is None:
            config_path = self._find_default_config()

        config_path = Path(config_path)
        if not config_path.exists():
            raise ConfigError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ConfigError(f"Config file must contain a YAML mapping, got {type(raw).__name__}")

        # Apply environment variable overrides
        raw = self._apply_env_overrides(raw)

        # Validate required keys
        self._validate(raw)

        super().__init__(raw)
        self._config_path = config_path

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create Config from a dictionary."""
        instance = cls.__new__(cls)
        ConfigNode.__init__(instance, data)
        instance._config_path = None
        return instance

    @staticmethod
    def _find_default_config() -> Path:
        """Search for default.yaml in standard locations."""
        candidates = [
            Path("config/default.yaml"),
            Path("../config/default.yaml"),
            Path(__file__).parent.parent / "config" / "default.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise ConfigError(
            "No config file found. Searched:\n"
            + "\n".join(f"  - {c.resolve()}" for c in candidates)
        )

    @staticmethod
    def _apply_env_overrides(raw: dict) -> dict:
        """Override config values from environment variables.

        Convention: DRIFT_SECTION_KEY=value
        Example: DRIFT_DATABASE_URL=postgresql://...
        """
        env_prefix = "DRIFT_"
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                parts = key[len(env_prefix):].lower().split("_", 1)
                if len(parts) == 2:
                    section, param = parts
                    if section in raw and isinstance(raw[section], dict):
                        raw[section][param] = value
        return raw

    @staticmethod
    def _validate(raw: dict) -> None:
        """Validate that core required keys are present."""
        missing = []
        for key in REQUIRED_KEYS["core"]:
            if key not in raw:
                missing.append(key)
        if missing:
            raise ConfigError(
                f"Missing required config sections: {missing}. "
                f"Ensure your YAML has: {REQUIRED_KEYS['core']}"
            )
