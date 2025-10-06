#!/usr/bin/env python3

"""
Enhanced Error Recovery Module

Provides advanced error recovery mechanisms with auto-retry, exponential backoff,
partial success handling, and clear user guidance. Designed to improve resilience
for long-running genealogical research operations.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"           # Warnings, non-critical issues
    MEDIUM = "medium"     # Recoverable errors
    HIGH = "high"         # Serious errors requiring attention
    CRITICAL = "critical" # System-threatening errors

class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    RETRY = "retry"                    # Simple retry
    EXPONENTIAL_BACKOFF = "exp_backoff"  # Exponential backoff retry
    CIRCUIT_BREAKER = "circuit_breaker"  # Circuit breaker pattern
    PARTIAL_SUCCESS = "partial_success" # Continue with partial results
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Reduce functionality

@dataclass
class ErrorContext:
    """Context information for error recovery"""
    operation_name: str
    attempt_number: int = 1
    max_attempts: int = 3
    last_error: Optional[Exception] = None
    error_history: list[Exception] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    partial_results: list[Any] = field(default_factory=list)
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.EXPONENTIAL_BACKOFF

    def add_error(self, error: Exception) -> None:
        """Add an error to the history"""
        self.last_error = error
        self.error_history.append(error)

    def should_retry(self) -> bool:
        """Determine if operation should be retried"""
        return self.attempt_number < self.max_attempts

    def get_backoff_delay(self, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        """Calculate exponential backoff delay"""
        if self.recovery_strategy != RecoveryStrategy.EXPONENTIAL_BACKOFF:
            return base_delay

        # Exponential backoff with jitter
        delay = min(base_delay * (2 ** (self.attempt_number - 1)), max_delay)
        jitter = random.uniform(0.1, 0.3) * delay
        return delay + jitter

class EnhancedErrorRecovery:
    """
    Enhanced error recovery system with multiple strategies.

    Features:
    - Exponential backoff with jitter
    - Circuit breaker pattern
    - Partial success handling
    - Clear error messages with suggested actions
    - Recovery statistics and monitoring
    """

    def __init__(self) -> None:
        self.recovery_stats: dict[str, dict[str, int]] = {}
        self.circuit_breakers: dict[str, dict[str, Any]] = {}

    def get_recovery_stats(self, operation: str) -> dict[str, int]:
        """Get recovery statistics for an operation"""
        return self.recovery_stats.get(operation, {
            'total_attempts': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'partial_successes': 0
        })

    def update_stats(self, operation: str, success: bool, partial: bool = False) -> None:
        """Update recovery statistics"""
        if operation not in self.recovery_stats:
            self.recovery_stats[operation] = {
                'total_attempts': 0,
                'successful_recoveries': 0,
                'failed_recoveries': 0,
                'partial_successes': 0
            }

        stats = self.recovery_stats[operation]
        stats['total_attempts'] += 1

        if success:
            stats['successful_recoveries'] += 1
        elif partial:
            stats['partial_successes'] += 1
        else:
            stats['failed_recoveries'] += 1

    def is_circuit_open(self, operation: str, failure_threshold: int = 5) -> bool:
        """Check if circuit breaker is open for an operation"""
        if operation not in self.circuit_breakers:
            return False

        breaker = self.circuit_breakers[operation]

        # Check if circuit should be reset
        if breaker.get('open_until', datetime.min) < datetime.now():
            self.circuit_breakers[operation] = {'failures': 0, 'open_until': datetime.min}
            return False

        return breaker.get('failures', 0) >= failure_threshold

    def record_failure(self, operation: str, recovery_timeout: int = 300):
        """Record a failure for circuit breaker"""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = {'failures': 0, 'open_until': datetime.min}

        breaker = self.circuit_breakers[operation]
        breaker['failures'] += 1

        # Open circuit if threshold exceeded
        if breaker['failures'] >= 5:  # Default threshold
            breaker['open_until'] = datetime.now() + timedelta(seconds=recovery_timeout)
            logger.warning(f"Circuit breaker opened for {operation} - cooling down for {recovery_timeout}s")

    def record_success(self, operation: str):
        """Record a success for circuit breaker"""
        if operation in self.circuit_breakers:
            self.circuit_breakers[operation] = {'failures': 0, 'open_until': datetime.min}

# Global instance
error_recovery = EnhancedErrorRecovery()

def _handle_successful_attempt(operation_name: str, attempt: int) -> None:
    """Handle successful attempt logging and recording."""
    error_recovery.record_success(operation_name)
    error_recovery.update_stats(operation_name, success=True)

    if attempt > 1:
        logger.info(f"✅ {operation_name} succeeded after {attempt} attempts")


def _handle_non_retryable_error(operation_name: str, e: Exception) -> None:
    """Handle non-retryable errors."""
    logger.error(f"❌ Non-retryable error in {operation_name}: {e}")
    error_recovery.record_failure(operation_name)
    error_recovery.update_stats(operation_name, success=False)


def _handle_partial_success(
    operation_name: str,
    partial_success_handler: Optional[Callable],
    context: 'ErrorContext',
    last_exception: Exception
) -> Any:
    """Handle partial success scenarios."""
    if partial_success_handler and context.partial_results:
        try:
            partial_result = partial_success_handler(context.partial_results, last_exception)
            error_recovery.update_stats(operation_name, success=False, partial=True)
            logger.warning(f"⚠️ {operation_name} completed with partial success")
            return partial_result
        except Exception as partial_error:
            logger.error(f"Partial success handler failed: {partial_error}")
    return None


def _handle_retry_failure(
    operation_name: str,
    max_attempts: int,
    partial_success_handler: Optional[Callable],
    context: 'ErrorContext',
    last_exception: Exception
) -> Any:
    """Handle final retry failure."""
    logger.error(f"❌ {operation_name} failed after {max_attempts} attempts")
    error_recovery.record_failure(operation_name)
    error_recovery.update_stats(operation_name, success=False)

    # Try partial success handler
    partial_result = _handle_partial_success(operation_name, partial_success_handler, context, last_exception)
    if partial_result is not None:
        return partial_result

    raise last_exception


def with_enhanced_recovery(  # noqa: PLR0913
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.EXPONENTIAL_BACKOFF,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    partial_success_handler: Optional[Callable] = None,
    user_guidance: Optional[dict[type[Exception], str]] = None
):
    """
    Decorator for enhanced error recovery with multiple strategies.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff
        max_delay: Maximum delay between retries
        recovery_strategy: Strategy to use for recovery
        retryable_exceptions: Tuple of exceptions that should trigger retry
        partial_success_handler: Function to handle partial successes
        user_guidance: Dictionary mapping exceptions to user guidance messages
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            operation_name = f"{func.__module__}.{func.__name__}"
            context = ErrorContext(
                operation_name=operation_name,
                max_attempts=max_attempts,
                recovery_strategy=recovery_strategy
            )

            # Check circuit breaker
            if error_recovery.is_circuit_open(operation_name):
                raise RuntimeError(f"Circuit breaker is open for {operation_name}")

            last_exception = None

            for attempt in range(1, max_attempts + 1):
                context.attempt_number = attempt

                try:
                    logger.debug(f"Attempting {operation_name} (attempt {attempt}/{max_attempts})")
                    result = func(*args, **kwargs)
                    _handle_successful_attempt(operation_name, attempt)
                    return result

                except Exception as e:
                    last_exception = e
                    context.add_error(e)

                    # Check if this exception is retryable
                    if not isinstance(e, retryable_exceptions):
                        _handle_non_retryable_error(operation_name, e)
                        raise

                    # Log the error with context
                    logger.warning(f"⚠️ {operation_name} failed (attempt {attempt}/{max_attempts}): {e}")

                    # Provide user guidance if available
                    if user_guidance and type(e) in user_guidance:
                        logger.info(f"💡 Suggestion: {user_guidance[type(e)]}")

                    # Check if we should retry
                    if not context.should_retry():
                        return _handle_retry_failure(operation_name, max_attempts, partial_success_handler, context, last_exception)

                    # Calculate delay for next attempt
                    delay = context.get_backoff_delay(base_delay, max_delay)
                    logger.debug(f"Retrying {operation_name} in {delay:.1f} seconds...")
                    time.sleep(delay)

            # This should never be reached, but just in case
            raise last_exception or RuntimeError(f"Unknown error in {operation_name}")

        return wrapper
    return decorator

def create_user_guidance() -> dict[type[Exception], str]:
    """Create default user guidance messages for common exceptions"""
    return {
        ConnectionError: "Check your internet connection and try again",
        TimeoutError: "The operation timed out - try reducing batch size or increasing timeout",
        PermissionError: "Check file permissions and ensure no other programs are using the files",
        FileNotFoundError: "Ensure all required files exist and paths are correct",
        ValueError: "Check input parameters and data format",
        KeyError: "Required configuration or data field is missing",
        ImportError: "Install missing dependencies or check Python environment",
    }

def handle_partial_success(partial_results: list[Any], error: Exception) -> Any:
    """Default partial success handler"""
    if not partial_results:
        raise error

    logger.warning(f"Returning {len(partial_results)} partial results due to: {error}")
    return partial_results

# Convenience decorators for common patterns
def with_api_recovery(max_attempts: int = 5, base_delay: float = 2.0):
    """Decorator optimized for API calls"""
    return with_enhanced_recovery(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=120.0,
        retryable_exceptions=(ConnectionError, TimeoutError, OSError),
        user_guidance=create_user_guidance()
    )

def with_database_recovery(max_attempts: int = 3, base_delay: float = 1.0):
    """Decorator optimized for database operations"""
    return with_enhanced_recovery(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=30.0,
        retryable_exceptions=(ConnectionError, TimeoutError),
        user_guidance=create_user_guidance()
    )

def with_file_recovery(max_attempts: int = 3, base_delay: float = 0.5):
    """Decorator optimized for file operations"""
    return with_enhanced_recovery(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=10.0,
        retryable_exceptions=(PermissionError, FileNotFoundError, OSError),
        user_guidance=create_user_guidance()
    )
