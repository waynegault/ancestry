#!/usr/bin/env python3
"""
core.cancellation - Cooperative cancellation signaling for long-running actions.

Provides a process-wide threading.Event-based flag that can be set by error
handlers (e.g., timeouts) to request graceful shutdown of in-progress work
running in other threads. Actions should periodically check is_cancel_requested()
inside their main loops and exit cleanly when True.
"""


import sys
import threading
from pathlib import Path


class _CancellationState:
    """Manages cancellation state for cooperative shutdown."""

    event = threading.Event()
    scope: str | None = None


def request_cancel(scope: str | None = None) -> None:
    """Signal that current long-running operation should cancel ASAP."""
    _CancellationState.scope = scope
    _CancellationState.event.set()


def clear_cancel() -> None:
    """Clear any prior cancellation request before starting a new operation."""
    _CancellationState.scope = None
    _CancellationState.event.clear()


def is_cancel_requested() -> bool:
    """Return True if a cancellation has been requested."""
    return _CancellationState.event.is_set()


def cancel_scope() -> str | None:
    """Optional string describing who requested cancel (for diagnostics)."""
    return _CancellationState.scope


# ==============================================
# Comprehensive Test Suite
# ==============================================


def _test_cancellation_state_initialization() -> bool:
    """Test that cancellation state initializes correctly."""
    clear_cancel()  # Reset state
    assert not is_cancel_requested(), "Should start with no cancellation"
    assert cancel_scope() is None, "Should start with no scope"
    return True


def _test_request_cancel_without_scope() -> bool:
    """Test requesting cancellation without scope."""
    clear_cancel()
    request_cancel()
    assert is_cancel_requested(), "Should be cancelled after request_cancel()"
    assert cancel_scope() is None, "Scope should be None when not specified"
    return True


def _test_request_cancel_with_scope() -> bool:
    """Test requesting cancellation with scope."""
    clear_cancel()
    request_cancel(scope="test_operation")
    assert is_cancel_requested(), "Should be cancelled"
    assert cancel_scope() == "test_operation", "Should store scope"
    return True


def _test_clear_cancel() -> bool:
    """Test clearing cancellation request."""
    request_cancel(scope="test")
    assert is_cancel_requested(), "Should be cancelled before clear"
    clear_cancel()
    assert not is_cancel_requested(), "Should not be cancelled after clear"
    assert cancel_scope() is None, "Scope should be None after clear"
    return True


def _test_multiple_cancel_requests() -> bool:
    """Test multiple cancellation requests."""
    clear_cancel()
    request_cancel(scope="first")
    assert cancel_scope() == "first", "Should have first scope"
    request_cancel(scope="second")
    assert cancel_scope() == "second", "Should update to second scope"
    assert is_cancel_requested(), "Should still be cancelled"
    return True


def _test_cancel_state_thread_safety() -> bool:
    """Test that cancellation state is thread-safe."""
    import threading

    clear_cancel()
    results: list[bool] = []

    def set_cancel() -> None:
        request_cancel(scope="thread_test")
        results.append(is_cancel_requested())

    def check_cancel() -> None:
        import time

        time.sleep(0.01)  # Small delay to ensure set_cancel runs first
        results.append(is_cancel_requested())

    t1 = threading.Thread(target=set_cancel)
    t2 = threading.Thread(target=check_cancel)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Both threads should see the cancellation
    assert all(results), "All threads should see cancellation"
    return True


def cancellation_module_tests() -> bool:
    """
    Comprehensive test suite for cancellation.py.
    Tests cooperative cancellation signaling and state management.
    """
    import sys

    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from testing.test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite("Cooperative Cancellation & Shutdown Signaling", "core/cancellation.py")
        suite.start_suite()

        suite.run_test(
            "Cancellation State Initialization",
            _test_cancellation_state_initialization,
            "Cancellation state initializes with no active cancellation",
            "Test initial state of cancellation system",
            "Test cancellation system startup",
        )

        suite.run_test(
            "Request Cancellation Without Scope",
            _test_request_cancel_without_scope,
            "Cancellation can be requested without specifying scope",
            "Test basic cancellation request",
            "Test simple cancellation signaling",
        )

        suite.run_test(
            "Request Cancellation With Scope",
            _test_request_cancel_with_scope,
            "Cancellation scope is correctly stored and retrieved",
            "Test cancellation with diagnostic scope",
            "Test scoped cancellation tracking",
        )

        suite.run_test(
            "Clear Cancellation",
            _test_clear_cancel,
            "Cancellation state can be cleared for new operations",
            "Test clearing cancellation request",
            "Test cancellation state reset",
        )

        suite.run_test(
            "Multiple Cancellation Requests",
            _test_multiple_cancel_requests,
            "Multiple cancellation requests update scope correctly",
            "Test updating cancellation scope",
            "Test cancellation scope updates",
        )

        suite.run_test(
            "Thread-Safe Cancellation",
            _test_cancel_state_thread_safety,
            "Cancellation state is thread-safe across multiple threads",
            "Test cancellation visibility across threads",
            "Test thread-safe cancellation signaling",
        )

        return suite.finish_suite()


# Use centralized test runner utility from test_utilities
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(cancellation_module_tests)


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
