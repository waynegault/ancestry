#!/usr/bin/env python3

"""
rate_limiter.py - Unified Adaptive Rate Limiting System

This module provides a simplified, unified rate limiting system that replaces
the previous three competing mechanisms (token bucket + adaptive delay +
exponential backoff) with a single adaptive token bucket.

Key Design Principles:
1. Single source of truth: fill_rate controls request rate
2. Adaptive learning: Adjusts fill_rate based on API feedback
3. Balanced speedup: Requires 50 successes before increasing
4. Aggressive slowdown: Decreases 20% on 429 errors
5. No oscillation: Long stabilization period prevents fighting

Author: GitHub Copilot (Design Assistant)
Date: November 7, 2025
Status: Phase 1 - Foundation
"""

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from standard_imports import setup_module

logger = setup_module(globals(), __name__)


def _get_state_path() -> Path:
    """Return the on-disk path used for persisting rate limiter state."""
    project_root = Path(__file__).resolve().parent
    return project_root / "Cache" / "rate_limiter_state.json"


_persisted_state_cache: Optional[dict[str, Any]] = None
_rate_limiter_state_source: str = "default"


@dataclass
class RateLimiterMetrics:
    """Metrics for monitoring rate limiter performance."""

    total_requests: int = 0
    total_wait_time: float = 0.0
    rate_decreases: int = 0
    rate_increases: int = 0
    error_429_count: int = 0
    current_fill_rate: float = 0.0
    success_count: int = 0
    tokens_available: float = 0.0
    avg_wait_time: float = 0.0


@dataclass
class _LimiterConfig:
    """Resolved configuration for initializing the adaptive rate limiter."""

    rate: float
    success_threshold: int
    min_rate: float
    max_rate: float
    capacity: float
    source: str


def _safe_int(value: Any) -> Optional[int]:
    """Return an int if the value is numeric, otherwise None."""
    if isinstance(value, bool):  # Guard against bool subclassing int
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _safe_float(value: Any) -> Optional[float]:
    """Return a float if the value is numeric, otherwise None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _resolve_rate_from_persisted(
    rate: Optional[float],
    persisted: Optional[dict[str, Any]],
    source: str,
) -> tuple[Optional[float], str]:
    """Prefer persisted rate when no explicit rate is provided."""
    if not persisted:
        return rate, source

    persisted_rate = _safe_float(persisted.get("fill_rate"))
    if rate is None and persisted_rate is not None:
        return persisted_rate, "previous_run"
    return rate, source


def _finalize_rate(rate: Optional[float], explicit_rate: Optional[float]) -> float:
    """Determine the starting fill rate."""
    if explicit_rate is not None:
        return float(explicit_rate)
    if rate is not None:
        return float(rate)
    return 0.5


def _finalize_threshold(threshold_value: Optional[int]) -> int:
    """Ensure a valid success threshold is selected."""
    return max(threshold_value or 50, 1)


def _finalize_min_rate(min_rate: Optional[float]) -> float:
    """Ensure the minimum rate is positive and reasonable."""
    value = min_rate if min_rate is not None else 0.1
    return max(0.01, float(value))


def _finalize_max_rate(max_rate: Optional[float], min_rate: float) -> float:
    """Ensure the maximum rate is not below the minimum."""
    value = max_rate if max_rate is not None else 3.0
    return max(min_rate, float(value))


def _finalize_capacity(capacity: Optional[float]) -> float:
    """Ensure capacity is at least one token."""
    value = capacity if capacity is not None else 10.0
    return max(1.0, float(value))


def _sanitize_positive_int(value: Optional[int]) -> Optional[int]:
    """Return value when positive integer, otherwise None."""
    if value is None:
        return None
    return value if value > 0 else None


def _sanitize_positive_float(value: Optional[float]) -> Optional[float]:
    """Return value when positive float, otherwise None."""
    if value is None:
        return None
    return value if value > 0 else None


def _adjust_bounds_for_explicit_rate(
    min_rate: float,
    max_rate: float,
    explicit_rate: Optional[float],
    persisted_min_used: bool,
    persisted_max_used: bool,
) -> tuple[float, float]:
    """Broaden persisted bounds when explicit config requires it."""
    if explicit_rate is None:
        return min_rate, max_rate

    explicit_value = float(explicit_rate)
    if persisted_min_used and explicit_value < min_rate:
        min_rate = max(0.01, explicit_value)
    if persisted_max_used and explicit_value > max_rate:
        max_rate = max(min_rate, explicit_value)
    return min_rate, max_rate


def _clamp_rate(rate: float, min_rate: float, max_rate: float) -> float:
    """Clamp the rate within configured bounds."""
    return min(max(rate, min_rate), max_rate)


def _apply_persisted_config(
    persisted: Optional[dict[str, Any]],
    threshold_value: Optional[int],
    current_min_rate: Optional[float],
    current_max_rate: Optional[float],
    bucket_capacity: Optional[float],
) -> tuple[
    Optional[int],
    Optional[float],
    Optional[float],
    Optional[float],
    bool,
    bool,
]:
    """Merge persisted configuration defaults into requested overrides."""
    if not persisted:
        return (
            threshold_value,
            current_min_rate,
            current_max_rate,
            bucket_capacity,
            False,
            False,
        )

    persisted_min_used = False
    persisted_max_used = False

    if threshold_value is None:
        threshold_value = _safe_int(persisted.get("success_threshold"))
    if current_min_rate is None:
        current_min_rate = _safe_float(persisted.get("min_fill_rate"))
        persisted_min_used = current_min_rate is not None
    if current_max_rate is None:
        current_max_rate = _safe_float(persisted.get("max_fill_rate"))
        persisted_max_used = current_max_rate is not None
    if bucket_capacity is None:
        bucket_capacity = _safe_float(persisted.get("capacity"))

    return (
        threshold_value,
        current_min_rate,
        current_max_rate,
        bucket_capacity,
        persisted_min_used,
        persisted_max_used,
    )


def _finalize_config_values(
    rate: Optional[float],
    threshold_value: Optional[int],
    min_rate: Optional[float],
    max_rate: Optional[float],
    capacity: Optional[float],
    initial_fill_rate: Optional[float],
    persisted_min_used: bool,
    persisted_max_used: bool,
) -> tuple[float, int, float, float, float]:
    """Return finalized configuration values with clamped rate."""
    rate_value = _finalize_rate(rate, initial_fill_rate)
    threshold = _finalize_threshold(threshold_value)
    min_value = _finalize_min_rate(min_rate)
    max_value = _finalize_max_rate(max_rate, min_value)
    capacity_value = _finalize_capacity(capacity)
    min_value, max_value = _adjust_bounds_for_explicit_rate(
        min_value,
        max_value,
        initial_fill_rate,
        persisted_min_used,
        persisted_max_used,
    )
    clamped_rate = _clamp_rate(rate_value, min_value, max_value)
    return clamped_rate, threshold, min_value, max_value, capacity_value


def _build_limiter_config(
    initial_fill_rate: Optional[float],
    success_threshold: Optional[int],
    min_fill_rate: Optional[float],
    max_fill_rate: Optional[float],
    capacity: Optional[float],
) -> _LimiterConfig:
    """Resolve limiter configuration using overrides and persisted state."""
    persisted = _load_persisted_state()
    source = "config" if initial_fill_rate is not None else "default"

    rate = initial_fill_rate
    threshold_value = _sanitize_positive_int(success_threshold)
    current_min_rate = _sanitize_positive_float(min_fill_rate)
    current_max_rate = _sanitize_positive_float(max_fill_rate)
    bucket_capacity = _sanitize_positive_float(capacity)

    rate, source = _resolve_rate_from_persisted(rate, persisted, source)
    (
        threshold_value,
        current_min_rate,
        current_max_rate,
        bucket_capacity,
        persisted_min_used,
        persisted_max_used,
    ) = _apply_persisted_config(
        persisted,
        threshold_value,
        current_min_rate,
        current_max_rate,
        bucket_capacity,
    )

    (
        clamped_rate,
        threshold_value,
        min_rate_value,
        max_rate_value,
        capacity_value,
    ) = _finalize_config_values(
        rate,
        threshold_value,
        current_min_rate,
        current_max_rate,
        bucket_capacity,
        initial_fill_rate,
        persisted_min_used,
        persisted_max_used,
    )

    return _LimiterConfig(
        rate=clamped_rate,
        success_threshold=threshold_value,
        min_rate=min_rate_value,
        max_rate=max_rate_value,
        capacity=capacity_value,
        source=source,
    )


def _update_success_threshold(limiter: "AdaptiveRateLimiter", success_threshold: Optional[int]) -> None:
    """Apply a new success threshold when provided."""
    if success_threshold is None or success_threshold <= 0:
        return

    new_threshold = int(success_threshold)
    if new_threshold != limiter.success_threshold:
        limiter.success_threshold = new_threshold
        logger.debug(
            "Updated AdaptiveRateLimiter success threshold to %d",
            limiter.success_threshold,
        )


def _update_rate_bounds(
    limiter: "AdaptiveRateLimiter",
    min_fill_rate: Optional[float],
    max_fill_rate: Optional[float],
) -> None:
    """Update limiter bounds and clamp the current rate."""
    bounds_changed = False

    if min_fill_rate is not None and min_fill_rate > 0:
        new_min = max(0.01, float(min_fill_rate))
        if abs(new_min - limiter.min_fill_rate) > 1e-6:
            limiter.min_fill_rate = new_min
            bounds_changed = True

    if max_fill_rate is not None and max_fill_rate > 0:
        new_max = max(limiter.min_fill_rate, float(max_fill_rate))
        if abs(new_max - limiter.max_fill_rate) > 1e-6:
            limiter.max_fill_rate = new_max
            bounds_changed = True

    if bounds_changed:
        limiter.fill_rate = _clamp_rate(
            limiter.fill_rate,
            limiter.min_fill_rate,
            limiter.max_fill_rate,
        )
        logger.debug(
            "Adjusted AdaptiveRateLimiter bounds to %.3f-%.3f; current rate=%.3f",
            limiter.min_fill_rate,
            limiter.max_fill_rate,
            limiter.fill_rate,
        )


def _update_capacity(limiter: "AdaptiveRateLimiter", capacity: Optional[float]) -> None:
    """Update token bucket capacity when requested."""
    if capacity is None or capacity <= 0:
        return

    new_capacity = float(capacity)
    if abs(new_capacity - limiter.capacity) > 1e-6:
        limiter.capacity = new_capacity
        limiter.tokens = min(limiter.tokens, limiter.capacity)


def _update_existing_limiter(
    limiter: "AdaptiveRateLimiter",
    success_threshold: Optional[int],
    min_fill_rate: Optional[float],
    max_fill_rate: Optional[float],
    capacity: Optional[float],
) -> None:
    """Apply runtime updates to the existing singleton instance."""
    _update_success_threshold(limiter, success_threshold)
    _update_rate_bounds(limiter, min_fill_rate, max_fill_rate)
    _update_capacity(limiter, capacity)


class AdaptiveRateLimiter:
    """
    Unified adaptive rate limiter using token bucket algorithm.

    Core Concept:
    - fill_rate (tokens/second) is the ONLY rate control
    - Token bucket handles bursts naturally
    - Adaptive adjustment modifies fill_rate based on API feedback
    - No extra delays, no competing mechanisms

    Adaptive Logic:
    - 429 error → decrease fill_rate by 20% (slow down significantly)
    - Success → increase fill_rate by 2% after 50 consecutive successes
    - Separate immediate retry backoff from system-wide rate

    Thread Safety:
    - All mutations protected by threading.Lock()
    - Safe for concurrent API calls from multiple threads

    Example:
        >>> limiter = AdaptiveRateLimiter(initial_fill_rate=0.5)
        >>> limiter.wait()  # Wait for token
        >>> # Make API call
        >>> if response.status_code == 429:
        ...     limiter.on_429_error()
        >>> else:
        ...     limiter.on_success()
    """

    def __init__(
    self,
    initial_fill_rate: float = 1.5,
    capacity: float = 10.0,
    min_fill_rate: float = 0.1,
    max_fill_rate: float = 3.0,
    success_threshold: int = 50,
    ):
        """
        Initialize adaptive rate limiter.

        Args:
            initial_fill_rate: Starting rate in requests per second (default: 1.5)
            capacity: Maximum burst capacity in tokens (default: 10.0)
            min_fill_rate: Minimum allowed rate (default: 0.1 req/s = 10s between)
            max_fill_rate: Maximum allowed rate (default: 3.0 req/s)
            success_threshold: Successes required before speedup (default: 50)
        """
        # Validate parameters
        if initial_fill_rate <= 0:
            raise ValueError(f"initial_fill_rate must be > 0, got {initial_fill_rate}")
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if min_fill_rate <= 0:
            raise ValueError(f"min_fill_rate must be > 0, got {min_fill_rate}")
        if max_fill_rate < min_fill_rate:
            raise ValueError(
                f"max_fill_rate ({max_fill_rate}) must be >= "
                f"min_fill_rate ({min_fill_rate})"
            )
        if success_threshold < 1:
            raise ValueError(f"success_threshold must be >= 1, got {success_threshold}")

        # Token bucket state
        self.capacity = capacity
        self.fill_rate = initial_fill_rate
        self.tokens = capacity  # Start with full bucket
        self.last_refill_time = time.monotonic()

        # Rate bounds
        self.min_fill_rate = min_fill_rate
        self.max_fill_rate = max_fill_rate

        # Adaptive learning state
        self.success_count = 0
        self.success_threshold = success_threshold

        # Thread safety
        self._lock = threading.Lock()

        # Metrics
        self._metrics: dict[str, float | int] = {
            "total_requests": 0,
            "total_wait_time": 0.0,
            "rate_decreases": 0,
            "rate_increases": 0,
            "error_429_count": 0,
        }

        self.initial_source = "default"

        logger.debug(
            f"AdaptiveRateLimiter initialized: fill_rate={initial_fill_rate:.3f} req/s, "
            f"capacity={capacity:.1f}, range=[{min_fill_rate:.3f}, {max_fill_rate:.3f}], "
            f"success_threshold={success_threshold}, speedup=+2%"
        )

    def wait(self) -> float:
        """
        Wait according to token bucket algorithm.

        This is the core rate limiting method. It ensures requests are
        spaced according to fill_rate while allowing bursts up to capacity.

        Returns:
            float: Time spent waiting in seconds

        Example:
            >>> limiter = AdaptiveRateLimiter(initial_fill_rate=2.0)
            >>> wait_time = limiter.wait()  # May wait 0-0.5 seconds
            >>> # Now safe to make request
        """
        with self._lock:
            self._refill_tokens()

            if self.tokens >= 1.0:
                # Token available, consume it immediately
                self.tokens -= 1.0
                wait_time = 0.0
                logger.debug(
                    f"Token consumed: {self.tokens:.2f}/{self.capacity:.1f} remaining "
                    f"(rate: {self.fill_rate:.3f} req/s)"
                )
            else:
                # Wait for token to generate
                wait_time = (1.0 - self.tokens) / self.fill_rate
                logger.debug(
                    f"Token bucket empty ({self.tokens:.2f}), waiting {wait_time:.3f}s "
                    f"for refill (rate: {self.fill_rate:.3f} req/s)"
                )
                time.sleep(wait_time)
                self._refill_tokens()
                self.tokens -= 1.0  # Consume after refill

            # Update metrics
            self._metrics["total_requests"] += 1
            self._metrics["total_wait_time"] += wait_time

            return wait_time

    def on_429_error(self) -> None:
        """
        Handle 429 rate limit error by decreasing fill_rate.

        Decreases rate by 20% to quickly back off from rate limit.
        Resets success counter to prevent premature speedup.

        This aggressive slowdown helps find the safe rate quickly
        without oscillation.

        Example:
            >>> limiter = AdaptiveRateLimiter(initial_fill_rate=1.0)
            >>> limiter.on_429_error()  # rate → 0.8 req/s
            >>> limiter.on_429_error()  # rate → 0.64 req/s
        """
        with self._lock:
            old_rate = self.fill_rate
            self.fill_rate = max(
                self.fill_rate * 0.80,  # 20% decrease
                self.min_fill_rate,
            )
            self.success_count = 0  # Reset success streak

            # Update metrics
            self._metrics["error_429_count"] += 1
            self._metrics["rate_decreases"] += 1

            logger.warning(
                f"⚠️ 429 Rate Limit: Decreased rate from {old_rate:.3f} to "
                f"{self.fill_rate:.3f} req/s (-20%)"
            )

    def on_success(self) -> None:
        """
    Handle successful API call.

    Increases fill_rate by 2% only after success_threshold consecutive
    successes (default: 50). This prevents oscillation and ensures
    stable operation while converging faster.

        Example:
            >>> limiter = AdaptiveRateLimiter(initial_fill_rate=1.0)
            >>> for _ in range(50):
            ...     limiter.on_success()
            >>> # After 50th success, rate increases to 1.02 req/s
        """
        with self._lock:
            self.success_count += 1

            if self.success_count >= self.success_threshold:
                old_rate = self.fill_rate
                self.fill_rate = min(
                    self.fill_rate * 1.02,  # 2% increase
                    self.max_fill_rate,
                )
                self.success_count = 0  # Reset counter

                # Update metrics
                self._metrics["rate_increases"] += 1

                # Only log if rate actually changed
                if abs(old_rate - self.fill_rate) > 0.001:
                    logger.info(
                        f"✅ After {self.success_threshold} successes: "
                        f"Increased rate to {self.fill_rate:.3f} req/s "
                        f"(+2% from {old_rate:.3f})"
                    )

    def _refill_tokens(self) -> None:
        """
        Refill tokens based on elapsed time and fill_rate.

        Called internally before each wait() operation.
        Tokens refill continuously at fill_rate per second.
        """
        now = time.monotonic()
        elapsed = max(0.0, now - self.last_refill_time)
        tokens_to_add = elapsed * self.fill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill_time = now

    def get_metrics(self) -> RateLimiterMetrics:
        """
        Get current metrics for monitoring.

        Returns:
            RateLimiterMetrics: Current state and statistics

        Example:
            >>> limiter = AdaptiveRateLimiter()
            >>> metrics = limiter.get_metrics()
            >>> print(f"Total requests: {metrics.total_requests}")
            >>> print(f"Current rate: {metrics.current_fill_rate:.3f} req/s")
        """
        with self._lock:
            avg_wait = 0.0
            if self._metrics["total_requests"] > 0:
                avg_wait = self._metrics["total_wait_time"] / self._metrics["total_requests"]

            return RateLimiterMetrics(
                total_requests=int(self._metrics["total_requests"]),
                total_wait_time=float(self._metrics["total_wait_time"]),
                rate_decreases=int(self._metrics["rate_decreases"]),
                rate_increases=int(self._metrics["rate_increases"]),
                error_429_count=int(self._metrics["error_429_count"]),
                current_fill_rate=self.fill_rate,
                success_count=self.success_count,
                tokens_available=self.tokens,
                avg_wait_time=avg_wait,
            )

    def reset(self) -> None:
        """
        Reset rate limiter to initial state.

        Useful for testing or starting fresh after configuration change.

        Warning:
            This resets the learned rate. Use with caution in production.
        """
        with self._lock:
            self.tokens = self.capacity
            self.success_count = 0
            self.last_refill_time = time.monotonic()
            # Note: fill_rate is NOT reset - it represents learned optimal rate
            logger.info("Rate limiter state reset (fill_rate preserved)")


def _load_persisted_state() -> Optional[dict[str, Any]]:
    """Load persisted rate limiter state from disk."""
    global _persisted_state_cache  # noqa: PLW0603

    if _persisted_state_cache is not None:
        return _persisted_state_cache

    state_path = _get_state_path()
    if not state_path.exists():
        return None

    try:
        raw = state_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict) and "fill_rate" in data:
            _persisted_state_cache = data
            return data
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug(f"Failed to load persisted rate limiter state: {exc}")

    _persisted_state_cache = None
    return None


def _persist_state(payload: dict[str, Any]) -> None:
    """Persist rate limiter state to disk."""
    global _persisted_state_cache  # noqa: PLW0603

    state_path = _get_state_path()
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _persisted_state_cache = payload
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug(f"Failed to persist rate limiter state: {exc}")


# Singleton instance for global access
_global_rate_limiter: Optional[AdaptiveRateLimiter] = None
_global_rate_limiter_lock = threading.Lock()


def get_adaptive_rate_limiter(
    initial_fill_rate: Optional[float] = None,
    success_threshold: Optional[int] = None,
    min_fill_rate: Optional[float] = None,
    max_fill_rate: Optional[float] = None,
    capacity: Optional[float] = None,
) -> AdaptiveRateLimiter:
    """
    Get or create the global AdaptiveRateLimiter instance.

    This ensures all API calls use the same rate limiter for
    coordinated rate limiting across the application.

    Args:
        initial_fill_rate: Initial rate if creating new instance.
                          If None, uses persisted state or default (0.5 req/s)
        success_threshold: Number of successes before increasing rate.
        min_fill_rate: Minimum allowed rate (overrides default/persisted value).
        max_fill_rate: Maximum allowed rate (overrides default/persisted value).
        capacity: Token bucket capacity (overrides default/persisted value).

    Returns:
        AdaptiveRateLimiter: The global rate limiter instance

    Example:
        >>> limiter = get_adaptive_rate_limiter()
        >>> limiter.wait()
    """
    global _global_rate_limiter  # noqa: PLW0603
    global _rate_limiter_state_source  # noqa: PLW0603

    with _global_rate_limiter_lock:
        if _global_rate_limiter is None:
            config = _build_limiter_config(
                initial_fill_rate=initial_fill_rate,
                success_threshold=success_threshold,
                min_fill_rate=min_fill_rate,
                max_fill_rate=max_fill_rate,
                capacity=capacity,
            )

            _global_rate_limiter = AdaptiveRateLimiter(
                initial_fill_rate=config.rate,
                success_threshold=config.success_threshold,
                min_fill_rate=config.min_rate,
                max_fill_rate=config.max_rate,
                capacity=config.capacity,
            )
            _global_rate_limiter.initial_source = config.source
            _rate_limiter_state_source = config.source
            logger.debug(
                "Created global AdaptiveRateLimiter with rate=%.3f req/s (source=%s, threshold=%d, bounds=%.3f-%.3f, capacity=%.1f)",
                _global_rate_limiter.fill_rate,
                config.source,
                _global_rate_limiter.success_threshold,
                _global_rate_limiter.min_fill_rate,
                _global_rate_limiter.max_fill_rate,
                _global_rate_limiter.capacity,
            )
        else:
            _update_existing_limiter(
                _global_rate_limiter,
                success_threshold=success_threshold,
                min_fill_rate=min_fill_rate,
                max_fill_rate=max_fill_rate,
                capacity=capacity,
            )

        return _global_rate_limiter


def get_rate_limiter_state_source() -> str:
    """Return the origin of the current rate limiter initialization."""
    return _rate_limiter_state_source


def get_persisted_rate_state() -> Optional[dict[str, Any]]:
    """Expose the persisted rate limiter state (if available)."""
    return _load_persisted_state()


def persist_rate_limiter_state(
    limiter: Optional[AdaptiveRateLimiter],
    metrics: Optional[RateLimiterMetrics] = None,
) -> None:
    """Persist the latest limiter state and optional metrics for next run reuse."""

    if limiter is None:
        return

    metrics = metrics or limiter.get_metrics()

    payload: dict[str, Any] = {
        "fill_rate": float(limiter.fill_rate),
        "success_threshold": int(limiter.success_threshold),
        "capacity": float(limiter.capacity),
        "min_fill_rate": float(limiter.min_fill_rate),
        "max_fill_rate": float(limiter.max_fill_rate),
        "timestamp": time.time(),
        "total_requests": int(metrics.total_requests),
        "avg_wait_time": float(metrics.avg_wait_time),
        "rate_increases": int(metrics.rate_increases),
        "rate_decreases": int(metrics.rate_decreases),
        "error_429_count": int(metrics.error_429_count),
    }

    _persist_state(payload)


def reset_global_rate_limiter() -> None:
    """
    Reset the global rate limiter instance.

    Primarily for testing. In production, you typically want to
    preserve the learned rate across operations.

    Example:
        >>> reset_global_rate_limiter()
        >>> limiter = get_adaptive_rate_limiter(initial_fill_rate=1.0)
    """
    global _global_rate_limiter  # noqa: PLW0603
    global _rate_limiter_state_source  # noqa: PLW0603

    with _global_rate_limiter_lock:
        _global_rate_limiter = None
        _rate_limiter_state_source = "default"
        logger.info("Global rate limiter reset")


# =============================================================================
# TESTS
# =============================================================================


def run_comprehensive_tests() -> bool:
    """Run comprehensive tests for AdaptiveRateLimiter."""
    import sys

    from test_framework import TestSuite

    suite = TestSuite("AdaptiveRateLimiter", "rate_limiter.py")
    suite.start_suite()

    # Test 1: Basic initialization
    suite.run_test(
        test_name="Basic initialization",
        test_func=_test_initialization,
        test_summary="Verify AdaptiveRateLimiter initializes with valid parameters",
        functions_tested="AdaptiveRateLimiter.__init__",
        method_description="Create limiter with specific rate, capacity, and bounds",
        expected_outcome="Limiter starts with full token bucket and zero success count",
    )

    # Test 2: Token bucket enforces rate
    suite.run_test(
        test_name="Token bucket enforcement",
        test_func=_test_token_bucket_enforcement,
        test_summary="Verify token bucket enforces fill_rate correctly",
        functions_tested="AdaptiveRateLimiter.wait",
        method_description="Make burst requests followed by rate-limited requests",
        expected_outcome="Burst completes fast, then rate limiting enforces delay",
    )

    # Test 3: 429 decreases rate by 20%
    suite.run_test(
        test_name="429 error handling",
        test_func=_test_429_decreases_rate,
        test_summary="Verify 429 error decreases fill_rate by 20%",
        functions_tested="AdaptiveRateLimiter.on_429_error",
        method_description="Trigger consecutive 429 errors",
        expected_outcome="Rate decreases by 20% each time: 1.0 → 0.8 → 0.64",
    )

    # Test 4: Success requires threshold before increase
    suite.run_test(
        test_name="Success threshold",
        test_func=_test_success_threshold,
            test_summary="Verify success requires 50 calls before rate increase",
        functions_tested="AdaptiveRateLimiter.on_success",
        method_description="Call on_success 100 times",
            expected_outcome="No increase until 50th success, then +2%",
    )

    # Test 5: 429 resets success counter
    suite.run_test(
        test_name="Success counter reset",
        test_func=_test_429_resets_success_count,
        test_summary="Verify 429 error resets success counter",
        functions_tested="AdaptiveRateLimiter.on_429_error",
        method_description="Accumulate 50 successes then trigger 429",
        expected_outcome="Success counter resets to 0 on 429 error",
    )

    # Test 6: Rate stays within bounds
    suite.run_test(
        test_name="Rate bounds enforcement",
        test_func=_test_rate_bounds,
        test_summary="Verify rate stays within min/max bounds",
        functions_tested="AdaptiveRateLimiter.on_429_error, on_success",
        method_description="Try to push rate below min and above max",
        expected_outcome="Rate clamped to [min_fill_rate, max_fill_rate]",
    )

    # Test 7: Metrics tracking
    suite.run_test(
        test_name="Metrics collection",
        test_func=_test_metrics,
        test_summary="Verify metrics track requests and errors correctly",
        functions_tested="AdaptiveRateLimiter.get_metrics",
        method_description="Make requests and errors, then check metrics",
        expected_outcome="Metrics accurately reflect operations performed",
    )

    # Test 8: Thread safety
    suite.run_test(
        test_name="Thread safety",
        test_func=_test_thread_safety,
        test_summary="Verify operations are thread-safe",
        functions_tested="AdaptiveRateLimiter.wait, on_success",
        method_description="Run 5 threads making 10 requests each",
        expected_outcome="No errors, all 50 requests complete successfully",
    )

    # Test 9: Global singleton
    suite.run_test(
        test_name="Global singleton",
        test_func=_test_global_singleton,
        test_summary="Verify global singleton pattern works correctly",
        functions_tested="get_adaptive_rate_limiter, reset_global_rate_limiter",
        method_description="Get limiter twice, verify same instance",
        expected_outcome="Returns same instance with first initialization rate",
    )

    # Test 10: Parameter validation
    suite.run_test(
        test_name="Parameter validation",
        test_func=_test_parameter_validation,
        test_summary="Verify invalid parameters raise ValueError",
        functions_tested="AdaptiveRateLimiter.__init__",
        method_description="Try invalid values: zero rate, negative capacity, etc",
        expected_outcome="ValueError raised for all invalid inputs",
    )

    return suite.finish_suite()


def _test_initialization() -> None:
    """Test basic initialization."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=1.0, capacity=5.0)
    assert limiter.fill_rate == 1.0, "fill_rate should match initial"
    assert limiter.capacity == 5.0, "capacity should match"
    assert limiter.tokens == 5.0, "should start with full bucket"
    assert limiter.success_count == 0, "should start with no successes"


def _test_token_bucket_enforcement() -> None:
    """Test that token bucket enforces rate."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=5.0, capacity=10.0)

    start = time.time()
    # Make 10 requests (bucket has 10 tokens, so should be instant)
    for _ in range(10):
        limiter.wait()
    burst_time = time.time() - start

    # Burst should be very fast (<1 second)
    assert burst_time < 1.0, f"Burst of 10 requests took {burst_time:.2f}s, should be <1s"

    # Next 10 requests should enforce rate (5 req/s = 2 seconds for 10)
    start = time.time()
    for _ in range(10):
        limiter.wait()
    rate_limited_time = time.time() - start

    # Should take ~2 seconds (allow 50% margin for timing variance)
    assert 1.5 <= rate_limited_time <= 3.0, (
        f"10 requests at 5 req/s should take ~2s, got {rate_limited_time:.2f}s"
    )


def _test_429_decreases_rate() -> None:
    """Test 429 error decreases rate by 20%."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=1.0)

    assert limiter.fill_rate == 1.0
    limiter.on_429_error()
    assert abs(limiter.fill_rate - 0.8) < 0.001, f"Expected 0.8, got {limiter.fill_rate}"

    limiter.on_429_error()
    assert abs(limiter.fill_rate - 0.64) < 0.001, f"Expected 0.64, got {limiter.fill_rate}"


def _test_success_threshold() -> None:
    """Test success requires threshold before increasing rate."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=1.0, success_threshold=50)

    # First 49 successes: no change
    for _ in range(49):
        limiter.on_success()
    assert limiter.fill_rate == 1.0, "Rate should not change before threshold"
    assert limiter.success_count == 49

    # 50th success: increases by 2%
    limiter.on_success()
    assert abs(limiter.fill_rate - 1.02) < 0.001, f"Expected 1.02, got {limiter.fill_rate}"
    assert limiter.success_count == 0, "Counter should reset after increase"


def _test_429_resets_success_count() -> None:
    """Test 429 error resets success counter."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=1.0, success_threshold=60)

    # 50 successes
    for _ in range(50):
        limiter.on_success()
    assert limiter.success_count == 50

    # 429 error resets
    limiter.on_429_error()
    assert limiter.success_count == 0, "429 should reset success counter"


def _test_rate_bounds() -> None:
    """Test rate stays within min/max bounds."""
    limiter = AdaptiveRateLimiter(
        initial_fill_rate=0.2, min_fill_rate=0.1, max_fill_rate=0.5
    )

    # Try to decrease below min
    for _ in range(10):
        limiter.on_429_error()
    assert limiter.fill_rate >= 0.1, f"Rate should not go below min: {limiter.fill_rate}"

    # Reset and try to increase above max
    limiter.fill_rate = 0.49
    limiter.success_count = 0
    for _ in range(200):  # 2 increases of 1% each
        limiter.on_success()
    assert limiter.fill_rate <= 0.5, f"Rate should not go above max: {limiter.fill_rate}"


def _test_metrics() -> None:
    """Test metrics tracking."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=10.0)  # Fast for testing

    # Make some requests
    for _ in range(5):
        limiter.wait()
        limiter.on_success()

    # Trigger 429
    limiter.on_429_error()

    metrics = limiter.get_metrics()
    assert metrics.total_requests == 5, f"Expected 5 requests, got {metrics.total_requests}"
    assert metrics.error_429_count == 1, f"Expected 1 429 error, got {metrics.error_429_count}"
    assert metrics.rate_decreases == 1, f"Expected 1 decrease, got {metrics.rate_decreases}"


def _test_thread_safety() -> None:
    """Test thread safety of operations."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=10.0, capacity=20.0)
    errors = []

    def make_requests() -> None:
        try:
            for _ in range(10):
                limiter.wait()
                limiter.on_success()
        except Exception as e:
            errors.append(e)

    # Run 5 threads concurrently
    threads = [threading.Thread(target=make_requests) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(errors) == 0, f"Thread safety errors: {errors}"

    metrics = limiter.get_metrics()
    assert metrics.total_requests == 50, f"Expected 50 requests, got {metrics.total_requests}"


def _test_global_singleton() -> None:
    """Test global singleton pattern."""
    reset_global_rate_limiter()

    limiter1 = get_adaptive_rate_limiter(initial_fill_rate=0.7)
    limiter2 = get_adaptive_rate_limiter(initial_fill_rate=0.9)  # Should be ignored

    assert limiter1 is limiter2, "Should return same instance"
    assert limiter1.fill_rate == 0.7, "Should use first initialization rate"


def _test_parameter_validation() -> None:
    """Test parameter validation raises ValueError."""
    try:
        AdaptiveRateLimiter(initial_fill_rate=0)
        raise AssertionError("Should raise ValueError for fill_rate=0")
    except ValueError:
        pass  # Expected

    try:
        AdaptiveRateLimiter(capacity=-1)
        raise AssertionError("Should raise ValueError for negative capacity")
    except ValueError:
        pass  # Expected

    try:
        AdaptiveRateLimiter(min_fill_rate=2.0, max_fill_rate=1.0)
        raise AssertionError("Should raise ValueError when min > max")
    except ValueError:
        pass  # Expected


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
