from __future__ import annotations

"""Centralized prompt loading helpers used by AI actions."""

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
