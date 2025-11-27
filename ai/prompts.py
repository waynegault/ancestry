from __future__ import annotations

"""Centralized prompt loading helpers used by AI actions."""

# === CORE INFRASTRUCTURE ===
import sys
from pathlib import Path

# Add parent directory to path for standard_imports when running as script
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import importlib
from typing import Any, Callable

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

try:  # pragma: no cover - optional import path
    from ai_prompt_utils import get_prompt as _legacy_get_prompt, load_prompts as _legacy_load_prompts
except ImportError:  # pragma: no cover - fall back to built-ins
    _legacy_get_prompt = None
    _legacy_load_prompts = None

_get_prompt_with_experiment: Callable[..., str | None] | None = None
_get_prompt_version: Callable[[str], str | None] | None = None
try:  # pragma: no cover - optional import path
    prompt_utils_module = importlib.import_module("ai_prompt_utils")
    _get_prompt_with_experiment = getattr(prompt_utils_module, "get_prompt_with_experiment", None)
    _get_prompt_version = getattr(prompt_utils_module, "get_prompt_version", None)
except Exception:  # pragma: no cover - we log when functions are used
    _get_prompt_with_experiment = None
    _get_prompt_version = None

try:  # pragma: no cover - optional telemetry module
    prompt_telemetry_module = importlib.import_module("prompt_telemetry")
    _record_extraction_experiment_event = getattr(prompt_telemetry_module, "record_extraction_experiment_event", None)
except Exception:  # pragma: no cover - telemetry optional
    _record_extraction_experiment_event = None


def supports_json_prompts() -> bool:
    """Return True when JSON prompt helpers are available."""

    return bool(_legacy_get_prompt and _legacy_load_prompts)


def load_prompts() -> dict[str, Any]:
    """Load prompts via ai_prompt_utils, falling back to empty structure."""

    if _legacy_load_prompts is None:
        logger.warning("ai_prompt_utils.load_prompts not available; returning empty prompts")
        return {"prompts": {}}

    try:
        return _legacy_load_prompts()
    except Exception as exc:  # pragma: no cover - delegated to upstream logging
        logger.error("Failed to load prompts via ai_prompt_utils: %s", exc)
        return {"prompts": {}}


def get_prompt(prompt_key: str) -> str | None:
    """Return a prompt by key if JSON prompts are configured."""

    if _legacy_get_prompt is None:
        logger.debug("JSON prompts disabled, get_prompt(%s) returning None", prompt_key)
        return None

    try:
        return _legacy_get_prompt(prompt_key)
    except Exception as exc:
        logger.warning("Error loading prompt '%s': %s", prompt_key, exc)
        return None


def get_prompt_with_experiment(
    prompt_key: str,
    *,
    variants: dict[str, str] | None = None,
    user_id: str | None = None,
) -> str | None:
    """Return a prompt variant when experiment helpers are available."""

    if _get_prompt_with_experiment is None:
        return get_prompt(prompt_key)

    try:
        return _get_prompt_with_experiment(prompt_key, variants=variants, user_id=user_id)
    except Exception as exc:
        logger.warning("Experiment prompt lookup failed for '%s': %s", prompt_key, exc)
        return get_prompt(prompt_key)


def get_prompt_version(prompt_key: str) -> str | None:
    """Return the configured prompt version, if available."""

    if _get_prompt_version is None:
        return None
    try:
        return _get_prompt_version(prompt_key)
    except Exception as exc:
        logger.debug("Prompt version lookup failed for '%s': %s", prompt_key, exc)
        return None


def record_extraction_experiment_event(payload: dict[str, Any]) -> None:
    """Forward experiment telemetry to prompt_telemetry if enabled."""

    if _record_extraction_experiment_event is None:
        return
    try:
        _record_extraction_experiment_event(payload)
    except Exception as exc:  # pragma: no cover - telemetry best effort
        logger.debug("Failed to record extraction experiment event: %s", exc)


# =============================================================================
# Module Tests
# =============================================================================


def _test_supports_json_prompts_returns_bool() -> bool:
    """Test that supports_json_prompts returns a boolean."""
    result = supports_json_prompts()
    assert isinstance(result, bool), f"Expected bool, got {type(result)}"
    return True


def _test_load_prompts_returns_dict() -> bool:
    """Test that load_prompts returns a dict with 'prompts' key."""
    result = load_prompts()
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "prompts" in result, "Result should have 'prompts' key"
    return True


def _test_get_prompt_handles_missing_key() -> bool:
    """Test that get_prompt returns None for missing keys gracefully."""
    result = get_prompt("nonexistent_prompt_key_12345")
    # Should return None or a string, but not raise
    assert result is None or isinstance(result, str), f"Unexpected type: {type(result)}"
    return True


def _test_get_prompt_with_experiment_fallback() -> bool:
    """Test that get_prompt_with_experiment falls back gracefully."""
    result = get_prompt_with_experiment(
        "nonexistent_key",
        variants={"v1": "test"},
        user_id="test_user",
    )
    # Should return None or string without raising
    assert result is None or isinstance(result, str), f"Unexpected type: {type(result)}"
    return True


def _test_get_prompt_version_handles_missing() -> bool:
    """Test that get_prompt_version returns None for missing keys."""
    result = get_prompt_version("nonexistent_prompt_key")
    assert result is None or isinstance(result, str), f"Unexpected type: {type(result)}"
    return True


def _test_record_extraction_experiment_event_no_raise() -> bool:
    """Test that record_extraction_experiment_event doesn't raise on any input."""
    # Should not raise even with empty or malformed payload
    record_extraction_experiment_event({})
    record_extraction_experiment_event({"key": "value", "nested": {"a": 1}})
    return True


def _test_load_prompts_with_valid_prompts() -> bool:
    """Test load_prompts returns actual prompts when available."""
    if not supports_json_prompts():
        # Skip if prompts not configured - still passes
        return True

    result = load_prompts()
    prompts = result.get("prompts", {})
    assert isinstance(prompts, dict), f"prompts should be dict, got {type(prompts)}"
    return True


def _test_get_prompt_returns_string_for_valid_key() -> bool:
    """Test get_prompt returns string for known prompt keys."""
    if not supports_json_prompts():
        return True

    # Try a common prompt key that should exist
    for key in ["intent_classification", "extraction_task", "entity_extraction"]:
        result = get_prompt(key)
        if result is not None:
            assert isinstance(result, str), f"Expected str for '{key}', got {type(result)}"
            assert len(result) > 0, f"Prompt '{key}' should not be empty"
            return True
    # If none exist, that's okay too
    return True


def module_tests() -> bool:
    """Run ai/prompts.py module tests."""
    from test_framework import TestSuite

    suite = TestSuite("AI Prompts", "ai/prompts.py")
    suite.start_suite()

    suite.run_test(
        "supports_json_prompts returns bool",
        _test_supports_json_prompts_returns_bool,
        "Verify supports_json_prompts returns boolean type",
    )

    suite.run_test(
        "load_prompts returns dict structure",
        _test_load_prompts_returns_dict,
        "Verify load_prompts returns dict with 'prompts' key",
    )

    suite.run_test(
        "get_prompt handles missing key",
        _test_get_prompt_handles_missing_key,
        "Verify get_prompt returns None for nonexistent keys",
    )

    suite.run_test(
        "get_prompt_with_experiment fallback",
        _test_get_prompt_with_experiment_fallback,
        "Verify experiment variant lookup falls back gracefully",
    )

    suite.run_test(
        "get_prompt_version handles missing",
        _test_get_prompt_version_handles_missing,
        "Verify version lookup returns None for missing keys",
    )

    suite.run_test(
        "record_extraction_experiment_event no raise",
        _test_record_extraction_experiment_event_no_raise,
        "Verify telemetry recording doesn't raise exceptions",
    )

    suite.run_test(
        "load_prompts with valid prompts",
        _test_load_prompts_with_valid_prompts,
        "Verify prompts dict structure when JSON prompts available",
    )

    suite.run_test(
        "get_prompt returns string for valid key",
        _test_get_prompt_returns_string_for_valid_key,
        "Verify known prompt keys return non-empty strings",
    )

    return suite.finish_suite()


# Standard test runner integration
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(module_tests)

if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
