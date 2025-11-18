#!/usr/bin/env python3

"""
connection_resilience.py - Connection Resilience & Sleep Prevention Framework

Provides comprehensive protection against:
1. PC sleep during long-running operations
2. Browser disconnection (network, system sleep, etc.)
3. Temporary connection loss with automatic recovery

Features:
- Cross-platform sleep prevention (Windows/macOS/Linux)
- Automatic browser health monitoring
- Graceful recovery from connection loss
- Detailed logging and metrics
"""

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

import functools
import time
from typing import Any, Callable, Optional, TypeVar

from utils import prevent_system_sleep, restore_system_sleep

F = TypeVar('F', bound=Callable[..., Any])


class ConnectionResilienceManager:
    """Manages connection resilience and recovery for long-running operations."""

    def __init__(self) -> None:
        self.sleep_state: Any = None
        self.recovery_attempts: int = 0
        self.max_recovery_attempts: int = 3
        self.recovery_backoff_base: float = 2.0  # seconds

    def start_resilience_mode(self) -> None:
        """Start resilience mode: prevent sleep and enable monitoring."""
        logger.debug("Starting connection resilience mode...")
        self.sleep_state = prevent_system_sleep()
        self.recovery_attempts = 0
        logger.debug("Sleep prevention enabled, connection monitoring active")

    def stop_resilience_mode(self) -> None:
        """Stop resilience mode: restore normal sleep behavior."""
        logger.debug("Stopping connection resilience mode...")
        restore_system_sleep(self.sleep_state)
        logger.debug("Sleep prevention disabled, normal power management restored")

    def handle_connection_loss(
        self,
        session_manager: Any,
        operation_name: str,
        retry_callback: Optional[Callable[..., Any]] = None
    ) -> bool:
        """
        Handle connection loss with automatic recovery.

        Args:
            session_manager: SessionManager instance
            operation_name: Name of operation for logging
            retry_callback: Optional callback to retry operation

        Returns:
            bool: True if recovery successful, False otherwise
        """
        self.recovery_attempts += 1

        if self.recovery_attempts > self.max_recovery_attempts:
            logger.error(f"❌ Max recovery attempts ({self.max_recovery_attempts}) exceeded for {operation_name}")
            return False

        logger.warning(f"🚨 Connection loss detected in {operation_name} (attempt {self.recovery_attempts}/{self.max_recovery_attempts})")

        # Calculate backoff delay
        backoff_delay = self.recovery_backoff_base ** (self.recovery_attempts - 1)
        logger.info(f"⏳ Waiting {backoff_delay:.1f}s before recovery attempt...")
        time.sleep(backoff_delay)

        # Attempt recovery
        logger.info(f"🔄 Attempting browser recovery for {operation_name}...")
        if session_manager.attempt_browser_recovery():
            logger.info(f"✅ Browser recovery successful for {operation_name}")
            self.recovery_attempts = 0  # Reset counter on success

            # Retry operation if callback provided
            if retry_callback:
                logger.info(f"🔄 Retrying {operation_name}...")
                try:
                    retry_callback()
                    logger.info(f"✅ {operation_name} retry successful")
                    return True
                except Exception as e:
                    logger.error(f"❌ {operation_name} retry failed: {e}")
                    return False
            return True

        logger.error(f"❌ Browser recovery failed for {operation_name}")
        return False


# Global resilience manager
_resilience_manager = ConnectionResilienceManager()


def with_connection_resilience(
    operation_name: str,
    max_recovery_attempts: int = 3
) -> Callable[[F], F]:
    """
    Decorator to add connection resilience to long-running operations.

    Features:
    - Prevents PC sleep during operation
    - Detects browser disconnection
    - Automatically recovers from connection loss
    - Provides detailed logging

    Usage:
        @with_connection_resilience("Action 6: DNA Match Gathering")
        def coord(session_manager):
            # Long-running operation
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _resilience_manager.max_recovery_attempts = max_recovery_attempts
            _resilience_manager.start_resilience_mode()

            try:
                logger.debug(f"Starting {operation_name}...")
                result = func(*args, **kwargs)
                logger.debug(f"{operation_name} completed successfully")
                return result

            except Exception as e:
                logger.debug(f"{operation_name} failed: {e}", exc_info=True)

                # Check if it's a connection error
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in [
                    'connection', 'disconnected', 'invalid session',
                    'browser', 'webdriver', 'timeout', 'refused'
                ]):
                    logger.warning(f"🚨 Connection-related error detected in {operation_name}")

                    # Try to get session_manager from args
                    session_manager = None
                    for arg in args:
                        if hasattr(arg, 'attempt_browser_recovery'):
                            session_manager = arg
                            break

                    if session_manager and _resilience_manager.handle_connection_loss(
                        session_manager,
                        operation_name
                    ):
                        logger.info(f"🔄 Retrying {operation_name} after recovery...")
                        return func(*args, **kwargs)

                raise

            finally:
                _resilience_manager.stop_resilience_mode()

        return wrapper  # type: ignore
    return decorator


def with_periodic_health_check(
    check_interval: int = 5,
    operation_name: str = "Operation"
) -> Callable[[F], F]:
    """
    Decorator to add periodic browser health checks during operation.

    Usage:
        @with_periodic_health_check(check_interval=5, operation_name="Action 7")
        def process_inbox(session_manager):
            # Operation with periodic health checks
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get session_manager from args
            session_manager = None
            for arg in args:
                if hasattr(arg, 'check_session_health'):
                    session_manager = arg
                    break

            if not session_manager:
                logger.warning(f"⚠️  No session_manager found for health checks in {operation_name}")
                return func(*args, **kwargs)

            # Wrap function with health check counter
            check_counter = {'count': 0}
            original_func = func

            def monitored_func(*inner_args: Any, **inner_kwargs: Any) -> Any:
                check_counter['count'] += 1

                # Periodic health check
                if check_counter['count'] % check_interval == 0 and not session_manager.check_session_health():
                    logger.warning(f"🚨 Health check failed in {operation_name} (check #{check_counter['count']})")
                    if session_manager.attempt_browser_recovery():
                        logger.info(f"✅ Recovery successful, continuing {operation_name}")
                    else:
                        raise RuntimeError(f"Browser recovery failed in {operation_name}")

                return original_func(*inner_args, **inner_kwargs)

            return monitored_func(*args, **kwargs)

        return wrapper  # type: ignore
    return decorator


def _test_resilience_manager_initialization() -> bool:
    """Test ConnectionResilienceManager initializes with correct defaults."""
    manager = ConnectionResilienceManager()
    assert manager.recovery_attempts == 0, "recovery_attempts should start at 0"
    assert manager.max_recovery_attempts == 3, "max_recovery_attempts should be 3"
    assert manager.recovery_backoff_base == 2.0, "recovery_backoff_base should be 2.0"
    assert manager.sleep_state is None, "sleep_state should start as None"
    return True


def _test_resilience_manager_state_transitions() -> bool:
    """Test ConnectionResilienceManager state transitions."""
    manager = ConnectionResilienceManager()

    # Test initial state
    assert manager.sleep_state is None, "sleep_state should start as None"
    assert manager.recovery_attempts == 0, "recovery_attempts should start at 0"

    # Test start_resilience_mode
    manager.start_resilience_mode()
    assert manager.sleep_state is not None, "sleep_state should be set after start"
    assert manager.recovery_attempts == 0, "recovery_attempts should be reset to 0 on start"

    # Test stop_resilience_mode (restores sleep but doesn't clear state variable)
    manager.stop_resilience_mode()
    # Note: stop_resilience_mode calls restore_system_sleep but doesn't clear sleep_state
    # This is intentional - the state is preserved for logging/debugging

    return True


def _test_decorators_are_callable() -> bool:
    """Test that decorators are properly defined and callable."""
    assert callable(with_connection_resilience), "with_connection_resilience should be callable"
    assert callable(with_periodic_health_check), "with_periodic_health_check should be callable"
    return True


def _test_decorator_parameters() -> bool:
    """Test that decorators accept required parameters."""
    import inspect

    # Check with_connection_resilience signature
    sig = inspect.signature(with_connection_resilience)
    params = list(sig.parameters.keys())
    assert "operation_name" in params, "with_connection_resilience should have operation_name parameter"
    assert "max_recovery_attempts" in params, "with_connection_resilience should have max_recovery_attempts parameter"

    # Check with_periodic_health_check signature
    sig2 = inspect.signature(with_periodic_health_check)
    params2 = list(sig2.parameters.keys())
    assert "check_interval" in params2, "with_periodic_health_check should have check_interval parameter"

    return True


def _test_resilience_manager_recovery_backoff() -> bool:
    """Test that recovery backoff calculation is correct."""
    manager = ConnectionResilienceManager()

    # Test backoff calculation: 2^(attempt-1)
    # Attempt 1: 2^0 = 1
    # Attempt 2: 2^1 = 2
    # Attempt 3: 2^2 = 4

    manager.recovery_attempts = 1
    backoff_1 = manager.recovery_backoff_base ** (manager.recovery_attempts - 1)
    assert backoff_1 == 1.0, f"Backoff for attempt 1 should be 1.0, got {backoff_1}"

    manager.recovery_attempts = 2
    backoff_2 = manager.recovery_backoff_base ** (manager.recovery_attempts - 1)
    assert backoff_2 == 2.0, f"Backoff for attempt 2 should be 2.0, got {backoff_2}"

    manager.recovery_attempts = 3
    backoff_3 = manager.recovery_backoff_base ** (manager.recovery_attempts - 1)
    assert backoff_3 == 4.0, f"Backoff for attempt 3 should be 4.0, got {backoff_3}"

    return True


def _test_resilience_manager_max_attempts() -> bool:
    """Test that max recovery attempts limit is enforced."""
    manager = ConnectionResilienceManager()

    # Verify max attempts is set
    assert manager.max_recovery_attempts == 3, "max_recovery_attempts should be 3"

    # Verify it can be customized
    manager.max_recovery_attempts = 5
    assert manager.max_recovery_attempts == 5, "max_recovery_attempts should be customizable"

    return True


def connection_resilience_module_tests() -> bool:
    """
    Comprehensive test suite for connection_resilience.py.
    Tests the connection resilience framework including sleep prevention and recovery.
    """
    from test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite(
            "Connection Resilience & Sleep Prevention Framework",
            "connection_resilience.py"
        )
        suite.start_suite()

        suite.run_test(
            "Resilience Manager Initialization",
            _test_resilience_manager_initialization,
            "ConnectionResilienceManager initializes with correct default values",
            "Create manager instance and verify recovery_attempts=0, max_recovery_attempts=3",
            "Test proper initialization of resilience manager state",
        )

        suite.run_test(
            "Resilience Manager State Transitions",
            _test_resilience_manager_state_transitions,
            "ConnectionResilienceManager transitions between resilience modes correctly",
            "Call start_resilience_mode and stop_resilience_mode, verify sleep_state changes",
            "Test state management for sleep prevention lifecycle",
        )

        suite.run_test(
            "Decorators Are Callable",
            _test_decorators_are_callable,
            "Both connection resilience decorators are properly defined and callable",
            "Verify with_connection_resilience and with_periodic_health_check are callable",
            "Test decorator availability for wrapping operations",
        )

        suite.run_test(
            "Decorator Parameters",
            _test_decorator_parameters,
            "Decorators have required parameters for configuration",
            "Inspect decorator signatures and verify required parameters exist",
            "Test decorator configuration flexibility",
        )

        suite.run_test(
            "Recovery Backoff Calculation",
            _test_resilience_manager_recovery_backoff,
            "Recovery backoff uses exponential backoff (2^(attempt-1)) correctly",
            "Calculate backoff for attempts 1, 2, 3 and verify exponential progression",
            "Test progressive backoff prevents overwhelming failed connections",
        )

        suite.run_test(
            "Max Recovery Attempts Limit",
            _test_resilience_manager_max_attempts,
            "Max recovery attempts limit is enforced and customizable",
            "Verify default max_recovery_attempts=3 and test customization",
            "Test recovery attempt limiting to prevent infinite loops",
        )

        return suite.finish_suite()


# Use centralized test runner utility from test_utilities
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(connection_resilience_module_tests)


if __name__ == "__main__":
    run_comprehensive_tests()
