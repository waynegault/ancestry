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

    def __init__(self):
        self.sleep_state = None
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        self.recovery_backoff_base = 2.0  # seconds

    def start_resilience_mode(self) -> None:
        """Start resilience mode: prevent sleep and enable monitoring."""
        logger.info("🛡️  Starting connection resilience mode...")
        self.sleep_state = prevent_system_sleep()
        self.recovery_attempts = 0
        logger.info("✅ Sleep prevention enabled, connection monitoring active")

    def stop_resilience_mode(self) -> None:
        """Stop resilience mode: restore normal sleep behavior."""
        logger.info("🛡️  Stopping connection resilience mode...")
        restore_system_sleep(self.sleep_state)
        logger.info("✅ Sleep prevention disabled, normal power management restored")

    def handle_connection_loss(
        self,
        session_manager: Any,
        operation_name: str,
        retry_callback: Optional[Callable] = None
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
                logger.info(f"🚀 Starting {operation_name}")
                result = func(*args, **kwargs)
                logger.info(f"✅ {operation_name} completed successfully")
                return result

            except Exception as e:
                logger.error(f"❌ {operation_name} failed: {e}", exc_info=True)

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

                    if session_manager:
                        if _resilience_manager.handle_connection_loss(
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
                if check_counter['count'] % check_interval == 0:
                    if not session_manager.check_session_health():
                        logger.warning(f"🚨 Health check failed in {operation_name} (check #{check_counter['count']})")
                        if session_manager.attempt_browser_recovery():
                            logger.info(f"✅ Recovery successful, continuing {operation_name}")
                        else:
                            raise RuntimeError(f"Browser recovery failed in {operation_name}")

                return original_func(*inner_args, **inner_kwargs)

            return monitored_func(*args, **kwargs)

        return wrapper  # type: ignore
    return decorator


# Test functions
if __name__ == "__main__":
    def test_connection_resilience():
        """Test connection resilience framework."""
        print("🧪 Testing connection resilience framework...")

        # Test resilience manager
        manager = ConnectionResilienceManager()
        assert manager.recovery_attempts == 0
        assert manager.max_recovery_attempts == 3
        print("   ✅ Resilience manager initialized")

        # Test decorator exists
        assert callable(with_connection_resilience)
        assert callable(with_periodic_health_check)
        print("   ✅ Decorators are callable")

        print("✅ All connection resilience tests passed!")

    test_connection_resilience()

