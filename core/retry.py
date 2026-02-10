#!/usr/bin/env python3

"""
Retry decorators, recovery strategies, and backoff logic.

This module provides retry decorators with telemetry-derived policies,
enhanced recovery with circuit breaker awareness, and utility functions
for resilient operation execution.
"""

import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

import requests

from config import config_schema
from config.config_schema import RetryPoliciesConfig
from core.exceptions import (
    AncestryError,
    APIRateLimitError,
    AuthenticationExpiredError,
    BrowserSessionError,
    ConfigurationError,
    DatabaseConnectionError,
    DataValidationError,
    FatalError,
    NetworkTimeoutError,
    RetryableError,
    _safe_update_error_context,
)

logger = logging.getLogger(__name__)

# Type variables for decorators
P = ParamSpec('P')
R = TypeVar('R')


class RetryStrategy(Enum):
    """Retry strategy options."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = True
    retry_on: list[type[Exception]] = field(default_factory=lambda: [Exception])
    stop_on: list[type[Exception]] = field(default_factory=list)


@dataclass(frozen=True)
class RetryPolicyProfile:
    """Resolved retry policy sourced from telemetry-tuned configuration."""

    name: str
    max_attempts: int
    initial_delay_seconds: float
    backoff_factor: float
    max_delay_seconds: float
    jitter_seconds: float
    retry_on: tuple[type[Exception], ...]
    stop_on: tuple[type[Exception], ...]


_DEFAULT_RETRY_BASELINE = RetryPoliciesConfig()
_RETRY_POLICY_CACHE: dict[str, RetryPolicyProfile] = {}


@dataclass(frozen=True)
class RetryDecoratorSettings:
    """Resolved retry decorator configuration."""

    policy_name: str | None
    max_attempts: int
    backoff_factor: float
    base_delay: float
    max_delay: float
    jitter_seconds: float
    retry_on: tuple[type[Exception], ...]
    stop_on: tuple[type[Exception], ...]


def _policy_exception_sets() -> dict[str, dict[str, tuple[type[Exception], ...]]]:
    """Return default retry/stop exception sets per policy channel."""

    return {
        "api": {
            "retry_on": (
                RetryableError,
                NetworkTimeoutError,
                DatabaseConnectionError,
                AuthenticationExpiredError,
                APIRateLimitError,
                requests.exceptions.RequestException,
                ConnectionError,
                TimeoutError,
            ),
            "stop_on": (
                FatalError,
                DataValidationError,
                ConfigurationError,
            ),
        },
        "selenium": {
            "retry_on": (
                RetryableError,
                NetworkTimeoutError,
                BrowserSessionError,
                AuthenticationExpiredError,
            ),
            "stop_on": (
                FatalError,
                DataValidationError,
                ConfigurationError,
            ),
        },
    }


def _get_channel_config(name: str) -> Any:
    cfg = getattr(config_schema, "retry_policies", None)
    if cfg and hasattr(cfg, name):
        return getattr(cfg, name)
    return getattr(_DEFAULT_RETRY_BASELINE, name, None)


def _build_retry_policy(name: str) -> RetryPolicyProfile:
    channel_cfg = _get_channel_config(name)
    if channel_cfg is None:
        raise ValueError(f"Unknown retry policy channel: {name}")

    exception_sets = _policy_exception_sets().get(name)
    if exception_sets is None:
        raise ValueError(f"No exception mapping defined for retry policy '{name}'")

    return RetryPolicyProfile(
        name=name,
        max_attempts=int(getattr(channel_cfg, "max_attempts", 3)),
        initial_delay_seconds=float(getattr(channel_cfg, "initial_delay_seconds", 1.0)),
        backoff_factor=float(getattr(channel_cfg, "backoff_factor", 2.0)),
        max_delay_seconds=float(getattr(channel_cfg, "max_delay_seconds", 20.0)),
        jitter_seconds=float(getattr(channel_cfg, "jitter_seconds", 0.3)),
        retry_on=exception_sets["retry_on"],
        stop_on=exception_sets["stop_on"],
    )


def resolve_retry_policy(
    policy: str | RetryPolicyProfile | None,
    default: str = "selenium",
) -> RetryPolicyProfile | None:
    """Return resolved RetryPolicyProfile for retry decorators."""

    if isinstance(policy, RetryPolicyProfile):
        return policy

    policy_name = (policy or default or "").strip().lower()
    if not policy_name:
        return None

    if policy_name not in _RETRY_POLICY_CACHE:
        _RETRY_POLICY_CACHE[policy_name] = _build_retry_policy(policy_name)

    return _RETRY_POLICY_CACHE[policy_name]


class RecoveryStrategy(Enum):
    """Recovery strategy types for enhanced retry decorators."""

    RETRY = "retry"
    EXPONENTIAL_BACKOFF = "exp_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"
    PARTIAL_SUCCESS = "partial_success"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class RecoveryContext:
    """Context container shared across enhanced recovery attempts."""

    operation_name: str
    attempt_number: int = 1
    max_attempts: int = 3
    last_error: Exception | None = None
    error_history: list[Exception] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    partial_results: list[Any] = field(default_factory=list)
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.EXPONENTIAL_BACKOFF

    def add_error(self, error: Exception) -> None:
        self.last_error = error
        self.error_history.append(error)

    def should_retry(self) -> bool:
        return self.attempt_number < self.max_attempts

    def get_backoff_delay(self, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        if self.recovery_strategy != RecoveryStrategy.EXPONENTIAL_BACKOFF:
            return base_delay

        delay = min(base_delay * (2 ** max(self.attempt_number - 1, 0)), max_delay)
        jitter = random.uniform(0.1, 0.3) * delay
        return delay + jitter


class EnhancedErrorRecovery:
    """Centralized error recovery telemetry with circuit breaker awareness."""

    def __init__(self) -> None:
        self.recovery_stats: dict[str, dict[str, int]] = {}
        self.circuit_breakers: dict[str, dict[str, Any]] = {}

    def get_recovery_stats(self, operation: str) -> dict[str, int]:
        return self.recovery_stats.get(
            operation,
            {
                "total_attempts": 0,
                "successful_recoveries": 0,
                "failed_recoveries": 0,
                "partial_successes": 0,
            },
        )

    def update_stats(self, operation: str, success: bool, partial: bool = False) -> None:
        stats = self.recovery_stats.setdefault(
            operation,
            {
                "total_attempts": 0,
                "successful_recoveries": 0,
                "failed_recoveries": 0,
                "partial_successes": 0,
            },
        )

        stats["total_attempts"] += 1
        if success:
            stats["successful_recoveries"] += 1
        elif partial:
            stats["partial_successes"] += 1
        else:
            stats["failed_recoveries"] += 1

    def is_circuit_open(self, operation: str, failure_threshold: int = 5) -> bool:
        breaker = self.circuit_breakers.get(operation)
        if breaker is None:
            return False

        open_until = breaker.get("open_until", datetime.min)
        if open_until < datetime.now():
            self.circuit_breakers[operation] = {"failures": 0, "open_until": datetime.min}
            return False

        return breaker.get("failures", 0) >= failure_threshold

    def record_failure(self, operation: str, recovery_timeout: int = 300) -> None:
        breaker = self.circuit_breakers.setdefault(operation, {"failures": 0, "open_until": datetime.min})
        breaker["failures"] += 1
        if breaker["failures"] >= 5:
            breaker["open_until"] = datetime.now() + timedelta(seconds=recovery_timeout)
            logger.warning(
                "Circuit breaker opened for %s - cooling down for %ss",
                operation,
                recovery_timeout,
            )

    def record_success(self, operation: str) -> None:
        if operation in self.circuit_breakers:
            self.circuit_breakers[operation] = {"failures": 0, "open_until": datetime.min}


error_recovery = EnhancedErrorRecovery()


def _handle_successful_attempt(operation_name: str, attempt: int) -> None:
    error_recovery.record_success(operation_name)
    error_recovery.update_stats(operation_name, success=True)

    if attempt > 1:
        logger.info("✅ %s succeeded after %d attempts", operation_name, attempt)


def _handle_non_retryable_error(operation_name: str, exc: Exception) -> None:
    logger.error("❌ Non-retryable error in %s: %s", operation_name, exc)
    error_recovery.record_failure(operation_name)
    error_recovery.update_stats(operation_name, success=False)


def _handle_partial_success(
    operation_name: str,
    partial_success_handler: Callable[[list[Any], Exception], Any] | None,
    context: RecoveryContext,
    last_exception: Exception,
) -> Any:
    if partial_success_handler and context.partial_results:
        try:
            partial_result = partial_success_handler(context.partial_results, last_exception)
            error_recovery.update_stats(operation_name, success=False, partial=True)
            logger.warning("⚠️ %s completed with partial success", operation_name)
            return partial_result
        except Exception as partial_error:
            logger.error("Partial success handler failed: %s", partial_error)
    return None


def _handle_retry_failure(
    operation_name: str,
    max_attempts: int,
    partial_success_handler: Callable[[list[Any], Exception], Any] | None,
    context: RecoveryContext,
    last_exception: Exception,
) -> Any:
    logger.error("❌ %s failed after %d attempts", operation_name, max_attempts)
    error_recovery.record_failure(operation_name)
    error_recovery.update_stats(operation_name, success=False)

    partial_result = _handle_partial_success(operation_name, partial_success_handler, context, last_exception)
    if partial_result is not None:
        return partial_result
    raise last_exception


def create_user_guidance() -> dict[type[Exception], str]:
    """Default user guidance for common retryable exceptions."""

    return {
        ConnectionError: "Check your internet connection and try again",
        TimeoutError: "The operation timed out - try reducing batch size or increasing timeout",
        PermissionError: "Verify file permissions and ensure no other process is locking it",
        FileNotFoundError: "Ensure all required files exist and paths are correct",
        ValueError: "Check input parameters and data format",
        KeyError: "Required configuration or data field is missing",
        ImportError: "Install missing dependencies or check the Python environment",
    }


def handle_partial_success(partial_results: list[Any], error: Exception) -> Any:
    """Return best-effort results when retries exhaust."""

    if not partial_results:
        raise error

    logger.warning("Returning %d partial results due to: %s", len(partial_results), error)
    return partial_results


def with_enhanced_recovery(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.EXPONENTIAL_BACKOFF,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    partial_success_handler: Callable[[list[Any], Exception], Any] | None = None,
    user_guidance: dict[type[Exception], str] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that standardizes retries, jittered backoff, and guidance logging."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            operation_name = f"{func.__module__}.{func.__name__}"
            context = RecoveryContext(
                operation_name=operation_name,
                max_attempts=max_attempts,
                recovery_strategy=recovery_strategy,
            )

            if error_recovery.is_circuit_open(operation_name):
                raise RuntimeError(f"Circuit breaker is open for {operation_name}")

            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                context.attempt_number = attempt
                try:
                    logger.debug("Attempting %s (%d/%d)", operation_name, attempt, max_attempts)
                    result = func(*args, **kwargs)
                    _handle_successful_attempt(operation_name, attempt)
                    return result
                except Exception as exc:
                    last_exception = exc
                    context.add_error(exc)

                    if not isinstance(exc, retryable_exceptions):
                        _handle_non_retryable_error(operation_name, exc)
                        raise

                    logger.warning("⚠️ %s failed (%d/%d): %s", operation_name, attempt, max_attempts, exc)
                    if user_guidance and type(exc) in user_guidance:
                        logger.info("💡 Suggestion: %s", user_guidance[type(exc)])

                    if not context.should_retry():
                        return cast(
                            R,
                            _handle_retry_failure(
                                operation_name,
                                max_attempts,
                                partial_success_handler,
                                context,
                                last_exception,
                            ),
                        )

                    delay = context.get_backoff_delay(base_delay, max_delay)
                    logger.debug("Retrying %s in %.1fs", operation_name, delay)
                    time.sleep(delay)

            if last_exception is not None:
                raise last_exception
            raise RuntimeError(f"Unknown error in {operation_name}")

        return wrapper

    return decorator


def with_api_recovery(max_attempts: int = 5, base_delay: float = 2.0) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator optimized for API calls using unified recovery infrastructure."""

    return with_enhanced_recovery(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=120.0,
        retryable_exceptions=(ConnectionError, TimeoutError, OSError),
        user_guidance=create_user_guidance(),
    )


def with_database_recovery(
    max_attempts: int = 3, base_delay: float = 1.0
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator optimized for database operations using unified recovery infrastructure."""

    return with_enhanced_recovery(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=30.0,
        retryable_exceptions=(ConnectionError, TimeoutError),
        user_guidance=create_user_guidance(),
    )


def with_file_recovery(max_attempts: int = 3, base_delay: float = 0.5) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator optimized for filesystem operations using unified recovery infrastructure."""

    return with_enhanced_recovery(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=10.0,
        retryable_exceptions=(PermissionError, FileNotFoundError, OSError),
        user_guidance=create_user_guidance(),
    )


# === RETRY DECORATOR INTERNALS ===


def _should_stop_retry(exception: Exception, stop_on: list[type[Exception]]) -> bool:
    """Check if exception should stop retry attempts."""
    return any(isinstance(exception, exc_type) for exc_type in stop_on)


def _should_retry_exception(exception: Exception, retry_on: list[type[Exception]]) -> bool:
    """Check if exception should trigger a retry."""
    return any(isinstance(exception, exc_type) for exc_type in retry_on)


def _calculate_retry_delay(
    attempt: int,
    base_delay: float,
    backoff_factor: float,
    jitter_seconds: float,
    max_delay: float,
) -> float:
    """Calculate delay before next retry attempt."""
    delay = min(base_delay * (backoff_factor**attempt), max_delay)
    if jitter_seconds > 0:
        delay = min(delay + random.uniform(0, jitter_seconds), max_delay)
    return max(0.05, delay)


def _handle_retry_exception(
    exception: Exception,
    func_name: str,
    attempt: int,
    max_attempts: int,
    stop_on: list[type[Exception]],
    retry_on: list[type[Exception]],
    backoff_factor: float,
    base_delay: float,
    max_delay: float,
    jitter_seconds: float,
) -> bool:
    """Handle exception during retry attempt.

    Returns True if the caller should stop retrying and re-raise the exception.
    """
    if _should_stop_retry(exception, stop_on):
        logger.error(f"{func_name} failed with non-retryable error: {exception}")
        return True

    if not _should_retry_exception(exception, retry_on):
        logger.error(f"{func_name} failed with unsupported error type: {exception}")
        return True

    if attempt < max_attempts - 1:
        delay = _calculate_retry_delay(
            attempt,
            base_delay,
            backoff_factor,
            jitter_seconds,
            max_delay,
        )
        logger.warning(
            "%s failed on attempt %d/%d, retrying in %.2fs: %s",
            func_name,
            attempt + 1,
            max_attempts,
            delay,
            exception,
        )
        time.sleep(delay)

    return False


_DEFAULT_RETRY_EXCEPTIONS = (
    RetryableError,
    NetworkTimeoutError,
    DatabaseConnectionError,
)
_DEFAULT_STOP_EXCEPTIONS = (
    FatalError,
    DataValidationError,
)


def _resolve_exception_tuples(
    retry_on: list[type[Exception]] | None,
    stop_on: list[type[Exception]] | None,
    resolved_policy: RetryPolicyProfile | None,
) -> tuple[tuple[type[Exception], ...], tuple[type[Exception], ...]]:
    """Resolve retry and stop exception tuples from arguments or policy."""
    if retry_on is not None:
        retry_source = tuple(retry_on)
    elif resolved_policy:
        retry_source = tuple(resolved_policy.retry_on)
    else:
        retry_source = _DEFAULT_RETRY_EXCEPTIONS

    if stop_on is not None:
        stop_source = tuple(stop_on)
    elif resolved_policy:
        stop_source = tuple(resolved_policy.stop_on)
    else:
        stop_source = _DEFAULT_STOP_EXCEPTIONS

    return retry_source, stop_source


def _resolve_retry_settings(
    max_attempts: int | None,
    backoff_factor: float | None,
    retry_on: list[type[Exception]] | None,
    stop_on: list[type[Exception]] | None,
    jitter: bool | None,
    base_delay: float | None,
    max_delay: float | None,
    policy: str | RetryPolicyProfile | None,
) -> RetryDecoratorSettings:
    resolved_policy = resolve_retry_policy(policy, default="selenium")

    def _int_value(value: int | None, attr: str, fallback: int) -> int:
        if value is not None:
            return value
        if resolved_policy is not None:
            return int(getattr(resolved_policy, attr))
        return fallback

    def _float_value(value: float | None, attr: str, fallback: float) -> float:
        if value is not None:
            return value
        if resolved_policy is not None:
            return float(getattr(resolved_policy, attr))
        return fallback

    retry_source, stop_source = _resolve_exception_tuples(retry_on, stop_on, resolved_policy)

    policy_jitter_value = _float_value(None, "jitter_seconds", 0.5)
    jitter_enabled = jitter if jitter is not None else (bool(policy_jitter_value) if resolved_policy else True)
    jitter_seconds = policy_jitter_value if jitter_enabled else 0.0

    return RetryDecoratorSettings(
        policy_name=resolved_policy.name if resolved_policy else None,
        max_attempts=_int_value(max_attempts, "max_attempts", 3),
        backoff_factor=_float_value(backoff_factor, "backoff_factor", 2.0),
        base_delay=_float_value(base_delay, "initial_delay_seconds", 1.0),
        max_delay=_float_value(max_delay, "max_delay_seconds", 60.0),
        jitter_seconds=jitter_seconds,
        retry_on=retry_source,
        stop_on=stop_source,
    )


def _wrap_with_retry[**P, R](func: Callable[P, R], settings: RetryDecoratorSettings) -> Callable[P, R]:
    stop_on = list(settings.stop_on)
    retry_on = list(settings.retry_on)

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        context_payload = {
            "operation": "retry_decorated_call",
            "module": func.__module__,
            "function": func.__name__,
            "args_preview": str(args)[:200],
            "kwargs_keys": list(kwargs.keys()),
        }
        start_time = time.time()
        last_exception: Exception | None = None

        for attempt in range(settings.max_attempts):
            try:
                attempt_start = time.time()
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(
                        "%s succeeded on attempt %d/%d after %.2fs",
                        func.__name__,
                        attempt + 1,
                        settings.max_attempts,
                        time.time() - attempt_start,
                    )
                return result
            except Exception as exc:
                last_exception = exc
                should_raise = _handle_retry_exception(
                    exc,
                    func.__name__,
                    attempt,
                    settings.max_attempts,
                    stop_on,
                    retry_on,
                    settings.backoff_factor,
                    settings.base_delay,
                    settings.max_delay,
                    settings.jitter_seconds,
                )
                if should_raise:
                    raise

        total_time = time.time() - start_time
        if last_exception is None:
            last_exception = Exception(f"{func.__name__} failed after {settings.max_attempts} attempts")

        logger.error(
            "%s failed after %d attempts in %.2fs: %s",
            func.__name__,
            settings.max_attempts,
            total_time,
            last_exception,
        )

        if isinstance(last_exception, AncestryError):
            _safe_update_error_context(last_exception, context_payload)

        raise last_exception

    setattr(wrapper, "__retry_policy__", settings.policy_name)
    setattr(
        wrapper,
        "__retry_settings__",
        {
            "max_attempts": settings.max_attempts,
            "backoff_factor": settings.backoff_factor,
            "base_delay": settings.base_delay,
            "max_delay": settings.max_delay,
        },
    )
    return wrapper


def retry_on_failure(
    max_attempts: int | None = None,
    backoff_factor: float | None = None,
    retry_on: list[type[Exception]] | None = None,
    stop_on: list[type[Exception]] | None = None,
    jitter: bool | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    policy: str | RetryPolicyProfile | None = "selenium",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for automatic retry with telemetry-derived policies."""

    settings = _resolve_retry_settings(
        max_attempts,
        backoff_factor,
        retry_on,
        stop_on,
        jitter,
        base_delay,
        max_delay,
        policy,
    )

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        return _wrap_with_retry(func, settings)

    return decorator


def api_retry(**overrides: Any) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Telemetry-derived retry helper for API operations."""

    base_decorator = retry_on_failure(policy="api", **overrides)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        wrapped = base_decorator(func)
        wrapped.__retry_helper__ = "api_retry"
        return wrapped

    return decorator


def selenium_retry(**overrides: Any) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Telemetry-derived retry helper for Selenium/browser operations."""

    base_decorator = retry_on_failure(policy="selenium", **overrides)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        wrapped = base_decorator(func)
        wrapped.__retry_helper__ = "selenium_retry"
        return wrapped

    return decorator


def timeout_protection(timeout: int = 30) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for timeout protection (cross-platform)."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            import platform

            # Use different timeout mechanisms based on platform
            if platform.system() == "Windows":
                # Windows doesn't support SIGALRM, use threading approach
                result_container: list[R] = []  # Use empty list instead of [None]
                exception: list[Exception | None] = [None]

                def target() -> None:
                    try:
                        result_container.append(func(*args, **kwargs))
                    except Exception as e:
                        exception[0] = e

                thread = threading.Thread(target=target)
                thread.daemon = True
                thread.start()
                thread.join(timeout)

                if thread.is_alive():
                    # Thread is still running, timeout occurred
                    raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds")

                if exception[0]:
                    raise exception[0] from None

                return result_container[0]
            # Unix-like systems can use signal
            import signal

            def timeout_handler(_signum: int, _frame: Any) -> None:
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds")

            sigalrm = getattr(signal, "SIGALRM", None)
            alarm_fn: Callable[[int], Any] | None = getattr(signal, "alarm", None)
            if sigalrm is None or alarm_fn is None:
                # Platform does not expose SIGALRM, fall back to direct execution
                return func(*args, **kwargs)

            old_handler = signal.signal(sigalrm, timeout_handler)
            alarm_fn(timeout)

            try:
                result = func(*args, **kwargs)
                alarm_fn(0)  # Disable the alarm once we have a result
                return result
            finally:
                signal.signal(sigalrm, old_handler)

        return wrapper

    return decorator


def graceful_degradation(
    fallback_value: Any = None, fallback_func: Callable[..., Any] | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for graceful degradation when service fails."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Function {func.__name__} failed: {e}, using fallback")
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                return fallback_value

        return wrapper

    return decorator


def with_recovery(recovery_strategy: Callable[..., Any]) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to add recovery strategy to functions."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Function {func.__name__} failed: {e}, attempting recovery")
                return recovery_strategy(*args, **kwargs)

        return wrapper

    return decorator


# =============================================
# TEST IMPLEMENTATION
# =============================================


def test_recovery_decorator_behavior() -> None:
    """Ensure with_recovery delegates to fallback logic after failures."""

    calls: dict[str, Any] = {"recovery": []}

    def _recovery_strategy(*args: Any, **kwargs: Any) -> str:
        calls["recovery"].append((args, kwargs))
        return "recovered"

    @with_recovery(_recovery_strategy)
    def flaky(value: str) -> str:
        raise RuntimeError(f"boom:{value}")

    assert flaky("one") == "recovered", "with_recovery should return fallback value on failure"
    assert calls["recovery"], "Recovery strategy should be invoked when wrapped func fails"
    assert calls["recovery"][0][0][0] == "one", "Arguments should be forwarded to recovery strategy"

    @with_recovery(_recovery_strategy)
    def stable(value: str) -> str:
        return value.upper()

    assert stable("ok") == "OK", "Successful calls should bypass recovery strategy"
    assert len(calls["recovery"]) == 1, "Recovery should only run for failures"


def module_tests() -> bool:
    """Retry decorators module test suite."""
    from testing.test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite("Retry Decorators & Recovery", "core/retry.py")

    print("🛡️ Running Retry Decorators & Recovery test suite...")

    with suppress_logging():
        suite.run_test(
            "Recovery decorator behavior",
            test_recovery_decorator_behavior,
            "Verify with_recovery wraps failures and returns fallback data",
            "Recovery decorator ensures critical paths can provide degraded-but-safe results",
            "with_recovery retries once, calls recovery strategy, and preserves successful return values",
        )

    return suite.finish_suite()


# Use centralized test runner utility
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    import sys
    import traceback

    print("\U0001faea Running Retry Decorators & Recovery test suite...")
    try:
        success = module_tests()
    except Exception:
        print(
            "\n[ERROR] Unhandled exception during retry tests:",
            file=sys.stderr,
        )
        traceback.print_exc()
        success = False
    if not success:
        print(
            "\n[FAIL] One or more retry tests failed. See above for details.",
            file=sys.stderr,
        )
    sys.exit(0 if success else 1)
