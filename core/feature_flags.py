#!/usr/bin/env python3

"""
Runtime Feature Toggle Framework.

Provides a centralized feature flag system for:
- A/B testing and gradual rollout
- Runtime feature toggling without code changes
- Environment-based feature configuration
- User/session-based feature targeting
"""

# === CORE INFRASTRUCTURE ===

import logging

logger = logging.getLogger(__name__)

# === STANDARD LIBRARY IMPORTS ===
import hashlib
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class FeatureState(Enum):
    """Feature flag states."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    ROLLOUT = "rollout"  # Percentage-based rollout


@dataclass
class FeatureFlag:
    """Definition of a feature flag."""

    name: str
    description: str
    default_enabled: bool = False
    rollout_percentage: float = 0.0  # 0-100, used when state is ROLLOUT
    state: FeatureState = FeatureState.DISABLED
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "default_enabled": self.default_enabled,
            "rollout_percentage": self.rollout_percentage,
            "state": self.state.value,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureFlag":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            default_enabled=data.get("default_enabled", False),
            rollout_percentage=data.get("rollout_percentage", 0.0),
            state=FeatureState(data.get("state", "disabled")),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


class FeatureFlags:
    """
    Centralized feature flag management.

    Supports:
    - Environment variable overrides (FEATURE_FLAG_<NAME>=true/false)
    - Percentage-based rollout with consistent user bucketing
    - Runtime flag updates
    - JSON configuration file loading
    """

    _instance: Optional["FeatureFlags"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "FeatureFlags":
        """Singleton pattern for global feature flag access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize feature flags registry."""
        if getattr(self, "_initialized", False):
            return

        self._flags: dict[str, FeatureFlag] = {}
        self._overrides: dict[str, bool] = {}
        self._config_path: Optional[Path] = None
        self._initialized = True
        logger.debug("FeatureFlags singleton initialized")

    def register(
        self,
        name: str,
        description: str = "",
        default_enabled: bool = False,
        rollout_percentage: float = 0.0,
    ) -> FeatureFlag:
        """
        Register a new feature flag.

        Args:
            name: Unique feature flag name (use SCREAMING_SNAKE_CASE)
            description: Human-readable description
            default_enabled: Default state when not in rollout
            rollout_percentage: 0-100 for gradual rollout

        Returns:
            The registered FeatureFlag
        """
        state = (
            FeatureState.ROLLOUT
            if rollout_percentage > 0
            else (FeatureState.ENABLED if default_enabled else FeatureState.DISABLED)
        )

        flag = FeatureFlag(
            name=name,
            description=description,
            default_enabled=default_enabled,
            rollout_percentage=rollout_percentage,
            state=state,
        )
        self._flags[name] = flag
        logger.debug("Registered feature flag: %s (state=%s)", name, state.value)
        return flag

    def is_enabled(
        self,
        name: str,
        user_id: Optional[str] = None,
        default: bool = False,
    ) -> bool:
        """
        Check if a feature flag is enabled.

        Args:
            name: Feature flag name
            user_id: Optional user ID for consistent rollout bucketing
            default: Default value if flag not registered

        Returns:
            True if feature is enabled for this context
        """
        # Check environment variable override first
        env_key = f"FEATURE_FLAG_{name.upper()}"
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return env_value.lower() in {"true", "1", "yes", "on"}

        # Check runtime overrides
        if name in self._overrides:
            return self._overrides[name]

        # Check registered flag
        flag = self._flags.get(name)
        if flag is None:
            logger.debug("Feature flag '%s' not registered, using default=%s", name, default)
            return default

        # Handle different states using mapping
        state_handlers: dict[FeatureState, bool | None] = {
            FeatureState.ENABLED: True,
            FeatureState.DISABLED: False,
            FeatureState.ROLLOUT: None,  # Special handling below
        }

        result = state_handlers.get(flag.state)
        if result is not None:
            return result
        if flag.state == FeatureState.ROLLOUT:
            return self._is_in_rollout(name, user_id, flag.rollout_percentage)
        return default

    @staticmethod
    def _is_in_rollout(
        flag_name: str,
        user_id: Optional[str],
        percentage: float,
    ) -> bool:
        """Determine if user falls within rollout percentage using consistent hashing."""
        if percentage >= 100:
            return True
        if percentage <= 0:
            return False

        # Create consistent hash from flag name + user_id
        hash_input = f"{flag_name}:{user_id or 'anonymous'}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = hash_value % 100

        return bucket < percentage

    def set_override(self, name: str, enabled: bool) -> None:
        """Set a runtime override for a feature flag."""
        self._overrides[name] = enabled
        logger.info("Feature flag '%s' overridden to %s", name, enabled)

    def clear_override(self, name: str) -> None:
        """Clear a runtime override."""
        if name in self._overrides:
            del self._overrides[name]
            logger.info("Feature flag '%s' override cleared", name)

    def clear_all_overrides(self) -> None:
        """Clear all runtime overrides."""
        self._overrides.clear()
        logger.info("All feature flag overrides cleared")

    def get_flag(self, name: str) -> Optional[FeatureFlag]:
        """Get a feature flag by name."""
        return self._flags.get(name)

    def list_flags(self) -> list[FeatureFlag]:
        """List all registered feature flags."""
        return list(self._flags.values())

    def get_all_states(self, user_id: Optional[str] = None) -> dict[str, bool]:
        """Get current state of all flags for a user."""
        return {name: self.is_enabled(name, user_id) for name in self._flags}

    def load_from_file(self, path: Path) -> int:
        """
        Load feature flags from JSON configuration file.

        Args:
            path: Path to JSON config file

        Returns:
            Number of flags loaded
        """
        if not path.exists():
            logger.warning("Feature flags config not found: %s", path)
            return 0

        try:
            with path.open(encoding="utf-8") as f:
                config = json.load(f)

            flags_data = config.get("feature_flags", [])
            for flag_data in flags_data:
                flag = FeatureFlag.from_dict(flag_data)
                self._flags[flag.name] = flag

            self._config_path = path
            logger.info("Loaded %d feature flags from %s", len(flags_data), path)
            return len(flags_data)

        except Exception as e:
            logger.error("Failed to load feature flags from %s: %s", path, e)
            return 0

    def save_to_file(self, path: Optional[Path] = None) -> bool:
        """
        Save current feature flags to JSON file.

        Args:
            path: Optional path (uses loaded path if not specified)

        Returns:
            True if save succeeded
        """
        save_path = path or self._config_path
        if save_path is None:
            logger.error("No path specified for saving feature flags")
            return False

        try:
            config = {
                "feature_flags": [flag.to_dict() for flag in self._flags.values()],
                "updated_at": datetime.now().isoformat(),
            }

            with save_path.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

            logger.info("Saved %d feature flags to %s", len(self._flags), save_path)
            return True

        except Exception as e:
            logger.error("Failed to save feature flags to %s: %s", save_path, e)
            return False

    def reset(self) -> None:
        """Reset all flags and overrides (useful for testing)."""
        self._flags.clear()
        self._overrides.clear()
        self._config_path = None
        logger.debug("FeatureFlags reset")

    def get_all_flags(self) -> dict[str, FeatureFlag]:
        """Return a snapshot of all registered flags."""
        return self._flags.copy()


# Convenience function for quick access
def is_feature_enabled(
    name: str,
    user_id: Optional[str] = None,
    default: bool = False,
) -> bool:
    """Check if a feature flag is enabled (convenience wrapper)."""
    return FeatureFlags().is_enabled(name, user_id, default)


def bootstrap_feature_flags(config: Any | None = None, default_path: Optional[Path] = None) -> FeatureFlags:
    """Load feature flags from config/env/default path and return the singleton.

    Order of precedence for loading:
    1) `FEATURE_FLAGS_FILE` environment variable
    2) `feature_flags_path` or `feature_flags_file` attribute on provided config
    3) explicit ``default_path`` argument
    4) fallback to ``config/feature_flags.json``

    Missing files are ignored; the function is intentionally tolerant.

    Default flags are registered for core functionality:
    - ACTION11_SEND_ENABLED: Master kill-switch for Action 11 sending (default: True)
    - AUTO_APPROVAL_ENABLED: Controls auto-approval in ApprovalQueue (default: False)
    """

    flags = FeatureFlags()

    # Register default feature flags for core functionality
    flags.register(
        "ACTION11_SEND_ENABLED",
        description="Master kill-switch for Action 11 send loop. Set to false to disable all sending.",
        default_enabled=True,
    )
    flags.register(
        "AUTO_APPROVAL_ENABLED",
        description="Enable auto-approval of high-confidence drafts in ApprovalQueue.",
        default_enabled=False,
    )
    flags.register(
        "CIRCUIT_BREAKER_ENABLED",
        description="Enable circuit breaker pattern for API failure protection.",
        default_enabled=True,
    )

    candidate_paths: list[Path] = []

    env_path = os.getenv("FEATURE_FLAGS_FILE")
    if env_path:
        candidate_paths.append(Path(env_path))

    if config is not None:
        for attr in ("feature_flags_path", "feature_flags_file"):
            cfg_value = getattr(config, attr, None)
            if cfg_value:
                candidate_paths.append(Path(cfg_value))

    if default_path is not None:
        candidate_paths.append(default_path)

    candidate_paths.append(Path("config/feature_flags.json"))

    for path in candidate_paths:
        try_path = Path(path)
        if try_path.exists():
            flags.load_from_file(try_path)
            break

    return flags


# =============================================================================
# Module Tests
# =============================================================================


def _test_singleton_pattern() -> bool:
    """Test that FeatureFlags uses singleton pattern."""
    ff1 = FeatureFlags()
    ff2 = FeatureFlags()
    assert ff1 is ff2, "FeatureFlags should be a singleton"
    return True


def _test_register_and_check_flag() -> bool:
    """Test basic flag registration and checking."""
    ff = FeatureFlags()
    ff.reset()

    ff.register("TEST_FEATURE", "Test feature", default_enabled=True)
    assert ff.is_enabled("TEST_FEATURE") is True, "Enabled flag should return True"

    ff.register("DISABLED_FEATURE", "Disabled feature", default_enabled=False)
    assert ff.is_enabled("DISABLED_FEATURE") is False, "Disabled flag should return False"

    ff.reset()
    return True


def _test_unregistered_flag_default() -> bool:
    """Test that unregistered flags return default value."""
    ff = FeatureFlags()
    ff.reset()

    assert ff.is_enabled("NONEXISTENT", default=False) is False
    assert ff.is_enabled("NONEXISTENT", default=True) is True

    ff.reset()
    return True


def _test_environment_override() -> bool:
    """Test environment variable overrides."""
    ff = FeatureFlags()
    ff.reset()

    ff.register("ENV_TEST", "Env test", default_enabled=False)

    # Set env var
    os.environ["FEATURE_FLAG_ENV_TEST"] = "true"
    assert ff.is_enabled("ENV_TEST") is True, "Env override should enable flag"

    os.environ["FEATURE_FLAG_ENV_TEST"] = "false"
    assert ff.is_enabled("ENV_TEST") is False, "Env override should disable flag"

    # Clean up
    del os.environ["FEATURE_FLAG_ENV_TEST"]
    ff.reset()
    return True


def _test_runtime_override() -> bool:
    """Test runtime overrides."""
    ff = FeatureFlags()
    ff.reset()

    ff.register("OVERRIDE_TEST", "Override test", default_enabled=False)
    assert ff.is_enabled("OVERRIDE_TEST") is False

    ff.set_override("OVERRIDE_TEST", True)
    assert ff.is_enabled("OVERRIDE_TEST") is True, "Override should enable flag"

    ff.clear_override("OVERRIDE_TEST")
    assert ff.is_enabled("OVERRIDE_TEST") is False, "After clear, should use default"

    ff.reset()
    return True


def _test_rollout_percentage() -> bool:
    """Test percentage-based rollout with consistent bucketing."""
    ff = FeatureFlags()
    ff.reset()

    ff.register("ROLLOUT_TEST", "Rollout test", rollout_percentage=50)

    # Same user should always get same result (consistent hashing)
    result1 = ff.is_enabled("ROLLOUT_TEST", user_id="user123")
    result2 = ff.is_enabled("ROLLOUT_TEST", user_id="user123")
    assert result1 == result2, "Same user should get consistent result"

    # 100% rollout should always enable
    ff.register("FULL_ROLLOUT", "Full rollout", rollout_percentage=100)
    assert ff.is_enabled("FULL_ROLLOUT", user_id="anyone") is True

    # 0% rollout should always disable
    ff.register("NO_ROLLOUT", "No rollout", rollout_percentage=0)
    assert ff.is_enabled("NO_ROLLOUT", user_id="anyone") is False

    ff.reset()
    return True


def _test_get_all_states() -> bool:
    """Test getting all flag states."""
    ff = FeatureFlags()
    ff.reset()

    ff.register("FLAG_A", "Flag A", default_enabled=True)
    ff.register("FLAG_B", "Flag B", default_enabled=False)

    states = ff.get_all_states()
    assert states["FLAG_A"] is True
    assert states["FLAG_B"] is False
    assert len(states) == 2

    ff.reset()
    return True


def _test_flag_serialization() -> bool:
    """Test flag to_dict and from_dict."""
    original = FeatureFlag(
        name="SERIALIZE_TEST",
        description="Test serialization",
        default_enabled=True,
        rollout_percentage=25.0,
        state=FeatureState.ROLLOUT,
        metadata={"team": "platform"},
    )

    data = original.to_dict()
    restored = FeatureFlag.from_dict(data)

    assert restored.name == original.name
    assert restored.description == original.description
    assert restored.default_enabled == original.default_enabled
    assert restored.rollout_percentage == original.rollout_percentage
    assert restored.state == original.state
    assert restored.metadata == original.metadata

    return True


def _test_convenience_function() -> bool:
    """Test is_feature_enabled convenience function."""
    ff = FeatureFlags()
    ff.reset()

    ff.register("CONVENIENCE_TEST", "Convenience test", default_enabled=True)
    assert is_feature_enabled("CONVENIENCE_TEST") is True
    assert is_feature_enabled("NONEXISTENT", default=False) is False

    ff.reset()
    return True


def module_tests() -> bool:
    """Run feature flags module tests."""
    from testing.test_framework import TestSuite

    suite = TestSuite("Feature Flags", "core/feature_flags.py")
    suite.start_suite()

    suite.run_test(
        "Singleton pattern",
        _test_singleton_pattern,
        "Verify FeatureFlags uses singleton pattern",
    )

    suite.run_test(
        "Register and check flag",
        _test_register_and_check_flag,
        "Verify basic flag registration and checking",
    )

    suite.run_test(
        "Unregistered flag default",
        _test_unregistered_flag_default,
        "Verify unregistered flags return default value",
    )

    suite.run_test(
        "Environment override",
        _test_environment_override,
        "Verify environment variable overrides work",
    )

    suite.run_test(
        "Runtime override",
        _test_runtime_override,
        "Verify runtime overrides work correctly",
    )

    suite.run_test(
        "Rollout percentage",
        _test_rollout_percentage,
        "Verify percentage-based rollout with consistent bucketing",
    )

    suite.run_test(
        "Get all states",
        _test_get_all_states,
        "Verify getting all flag states works",
    )

    suite.run_test(
        "Flag serialization",
        _test_flag_serialization,
        "Verify flag to_dict and from_dict work correctly",
    )

    suite.run_test(
        "Convenience function",
        _test_convenience_function,
        "Verify is_feature_enabled convenience wrapper",
    )

    return suite.finish_suite()


# Standard test runner integration
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(module_tests)

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
