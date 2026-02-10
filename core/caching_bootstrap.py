#!/usr/bin/env python3
"""Centralized caching initialization helpers."""


import logging
from importlib import import_module

logger = logging.getLogger(__name__)


class _CachingState:
    """Tracks whether aggressive caching has been initialized."""

    initialized = False


def _get_caching_state() -> _CachingState:
    state = getattr(_get_caching_state, "_state", None)
    if state is None:
        state = _CachingState()
        _get_caching_state._state = state
    return state


def initialize_aggressive_caching() -> bool:
    """Initialize aggressive caching systems."""

    try:
        cache_module = import_module("core.system_cache")
    except ImportError:
        logger.debug("System cache module not available (non-critical)")
        return False
    except Exception as exc:  # pragma: no cover - defensive logging only
        logger.error("Failed to initialize aggressive caching: %s", exc)
        return False

    warm_system_caches = getattr(cache_module, "warm_system_caches", None)
    if not callable(warm_system_caches):
        logger.debug("warm_system_caches() not available in system cache module")
        return False

    try:
        return bool(warm_system_caches())
    except Exception as exc:  # pragma: no cover - cache init rarely fails
        logger.error("warm_system_caches() failed: %s", exc)
        return False


def ensure_caching_initialized() -> bool:
    """Initialize aggressive caching systems if not already done."""

    state = _get_caching_state()
    if not state.initialized:
        logger.debug("Initializing caching systems on-demand...")
        cache_init_success = initialize_aggressive_caching()
        if cache_init_success:
            logger.debug("Caching systems initialized successfully")
            state.initialized = True
        else:
            logger.debug("Some caching systems failed to initialize, continuing with reduced performance")
        return cache_init_success

    logger.debug("Caching systems already initialized")
    return True


# ============================================================================
# Module Tests
# ============================================================================


def _reset_state() -> None:
    if hasattr(_get_caching_state, "_state"):
        delattr(_get_caching_state, "_state")


def _test_initialize_handles_missing_module() -> bool:
    from unittest.mock import patch

    _reset_state()

    patch_target = f"{__name__}.import_module"
    with patch(patch_target, side_effect=ImportError):
        assert initialize_aggressive_caching() is False
    return True


def _test_ensure_caching_initialized_sets_state() -> bool:
    from unittest.mock import patch

    _reset_state()

    patch_target = f"{__name__}.initialize_aggressive_caching"
    with patch(patch_target, return_value=True) as mock_init:
        assert ensure_caching_initialized() is True
        assert ensure_caching_initialized() is True
        mock_init.assert_called_once()
    return True


def caching_bootstrap_module_tests() -> bool:
    from testing.test_framework import TestSuite, suppress_logging

    suite = TestSuite("core/caching_bootstrap.py - Caching Bootstrap", "core/caching_bootstrap.py")

    with suppress_logging():
        suite.run_test(
            "Missing module handling",
            _test_initialize_handles_missing_module,
            "initialize_aggressive_caching gracefully handles missing module",
            "initialize_aggressive_caching",
            "Ensures ImportError results in False without raising",
        )

        suite.run_test(
            "Stateful initialization",
            _test_ensure_caching_initialized_sets_state,
            "ensure_caching_initialized initializes once and caches state",
            "ensure_caching_initialized",
            "Validates the caching state flag only flips once",
        )

    return suite.finish_suite()


from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(caching_bootstrap_module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    raise SystemExit(0 if success else 1)
