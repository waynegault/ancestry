"""
Rate limiter persistence helpers for Action 6.

Provides periodic persistence of rate limiter state during long-running
gather operations to prevent loss of learned rates when runs are cancelled.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.session_manager import SessionManager

logger = logging.getLogger(__name__)

# Track last persisted page to avoid excessive writes
_last_persisted_page: int = 0


def persist_rates_periodically(
    session_manager: "SessionManager",
    current_page: int,
    interval: int = 10,
) -> bool:
    """Persist rate limiter state every N pages.

    This ensures learned rates are saved periodically during long runs,
    so if the run is cancelled early, the next run can start with
    the optimized rates.

    Args:
        session_manager: The session manager with rate limiter
        current_page: Current page number being processed
        interval: How often to persist (default: every 10 pages)

    Returns:
        True if state was persisted, False otherwise
    """
    global _last_persisted_page

    # Only persist at intervals
    if current_page > 0 and current_page % interval != 0:
        return False

    # Avoid duplicate persistence for same page
    if current_page == _last_persisted_page:
        return False

    limiter = getattr(session_manager, "rate_limiter", None)
    if not limiter:
        return False

    try:
        from core.rate_limiter import persist_rate_limiter_state

        metrics = limiter.get_metrics()
        persist_rate_limiter_state(limiter, metrics)
        _last_persisted_page = current_page

        # Log endpoint rates summary for visibility at this checkpoint
        logger.debug(f"📊 Rate state persisted at page {current_page}")
        return True

    except Exception as exc:
        logger.debug(f"Failed to persist rate state at page {current_page}: {exc}")
        return False


def reset_persistence_state() -> None:
    """Reset the last persisted page tracker (for testing)."""
    global _last_persisted_page
    _last_persisted_page = 0


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------


def _test_persist_at_interval() -> bool:
    """Test that persistence only happens at interval boundaries."""
    from unittest import mock

    reset_persistence_state()

    mock_limiter = mock.MagicMock()
    mock_limiter.get_metrics.return_value = mock.MagicMock()

    mock_sm = mock.MagicMock()
    mock_sm.rate_limiter = mock_limiter

    # Patch the actual function in the rate_limiter module
    with mock.patch("core.rate_limiter.persist_rate_limiter_state") as mock_persist:
        # Page 5 should not persist (interval=10)
        result = persist_rates_periodically(mock_sm, 5, interval=10)
        assert result is False, "Should not persist at non-interval page"
        assert mock_persist.call_count == 0

        # Page 10 should persist
        result = persist_rates_periodically(mock_sm, 10, interval=10)
        assert result is True, "Should persist at interval boundary"
        assert mock_persist.call_count == 1

        # Page 10 again should not persist (duplicate)
        result = persist_rates_periodically(mock_sm, 10, interval=10)
        assert result is False, "Should not persist duplicate page"
        assert mock_persist.call_count == 1

        # Page 20 should persist
        result = persist_rates_periodically(mock_sm, 20, interval=10)
        assert result is True, "Should persist at next interval"
        assert mock_persist.call_count == 2

    reset_persistence_state()
    return True


def _test_no_limiter_graceful() -> bool:
    """Test graceful handling when no rate limiter exists."""
    from unittest import mock

    reset_persistence_state()

    mock_sm = mock.MagicMock()
    mock_sm.rate_limiter = None

    result = persist_rates_periodically(mock_sm, 10, interval=10)
    assert result is False, "Should return False when no limiter"

    reset_persistence_state()
    return True


def rate_persistence_module_tests() -> bool:
    """Run tests for rate persistence helpers."""
    from testing.test_framework import TestSuite

    suite = TestSuite("Rate Persistence", "rate_persistence.py")
    suite.start_suite()

    suite.run_test(
        test_name="Persist at interval boundaries",
        test_func=_test_persist_at_interval,
        test_summary="Verify persistence only happens at interval pages",
    )

    suite.run_test(
        test_name="Graceful handling without limiter",
        test_func=_test_no_limiter_graceful,
        test_summary="Verify no error when rate limiter is missing",
    )

    return suite.finish_suite()


from testing.test_framework import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(rate_persistence_module_tests)

if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
