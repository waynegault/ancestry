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
4. Gentle slowdown: Decreases 12% on 429 errors
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
from typing import Any, Optional, TypedDict, cast

from standard_imports import setup_module

logger = setup_module(globals(), __name__)


@dataclass
class _LimiterState:
    """Mutable container for module-level rate limiter state."""

    persisted_state_cache: Optional[dict[str, Any]] = None
    rate_limiter_state_source: str = "default"
    global_rate_limiter: Optional["AdaptiveRateLimiter"] = None


_LIMITER_STATE = _LimiterState()


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
    endpoint_rate_cap: Optional[float] = None


@dataclass
class _EndpointProfile:
    """Runtime throttle configuration for a specific endpoint."""

    min_interval: float = 0.0
    delay_multiplier: float = 1.0
    cooldown_after_429: float = 0.0
    max_rate: Optional[float] = None

    def __post_init__(self) -> None:
        self.min_interval = max(0.0, float(self.min_interval))
        self.delay_multiplier = max(1.0, float(self.delay_multiplier))
        self.cooldown_after_429 = max(0.0, float(self.cooldown_after_429))
        if self.max_rate is not None:
            max_rate = max(0.0, float(self.max_rate))
            self.max_rate = max_rate or None


@dataclass
class _LimiterConfig:
    """Resolved configuration for initializing the adaptive rate limiter."""

    rate: float
    success_threshold: int
    min_rate: float
    max_rate: float
    capacity: float
    source: str
    rate_limiter_429_backoff: float = 0.85
    rate_limiter_success_factor: float = 1.02


class LimiterStateDict(TypedDict, total=False):
    fill_rate: float
    success_threshold: int
    min_fill_rate: float
    max_fill_rate: float
    capacity: float
    timestamp: float
    total_requests: int
    avg_wait_time: float
    rate_increases: int
    rate_decreases: int
    error_429_count: int


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
    """Prefer persisted rate if it's safer (lower) than requested rate."""
    if not persisted:
        return rate, source

    persisted_rate = _safe_float(persisted.get("fill_rate"))
    if persisted_rate is None:
        return rate, source

    if rate is None:
        return persisted_rate, "previous_run"

    # If both exist, use the lower (safer) rate to prevent 429 loops
    if persisted_rate < rate:
        return persisted_rate, "previous_run_safer"

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

    p = cast(LimiterStateDict, persisted)

    persisted_min_used = False
    persisted_max_used = False

    if threshold_value is None:
        threshold_value = _safe_int(p.get("success_threshold"))
    if current_min_rate is None:
        current_min_rate = _safe_float(p.get("min_fill_rate"))
        persisted_min_used = current_min_rate is not None
    if current_max_rate is None:
        current_max_rate = _safe_float(p.get("max_fill_rate"))
        persisted_max_used = current_max_rate is not None
    if bucket_capacity is None:
        bucket_capacity = _safe_float(p.get("capacity"))

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
    rate_limiter_429_backoff: Optional[float] = None,
    rate_limiter_success_factor: Optional[float] = None,
) -> _LimiterConfig:
    """Resolve limiter configuration using overrides and persisted state."""
    persisted = _load_persisted_state()
    source = "config" if initial_fill_rate is not None else "default"

    rate = initial_fill_rate
    threshold_value = _sanitize_positive_int(success_threshold)
    current_min_rate = _sanitize_positive_float(min_fill_rate)
    current_max_rate = _sanitize_positive_float(max_fill_rate)
    bucket_capacity = _sanitize_positive_float(capacity)

    # Default values for new parameters if not provided
    backoff = rate_limiter_429_backoff if rate_limiter_429_backoff is not None else 0.85
    success_factor = rate_limiter_success_factor if rate_limiter_success_factor is not None else 1.02

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
        rate_limiter_429_backoff=backoff,
        rate_limiter_success_factor=success_factor,
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
    endpoint_profiles: Optional[dict[str, Any]],
) -> None:
    """Apply runtime updates to the existing singleton instance."""
    _update_success_threshold(limiter, success_threshold)
    _update_rate_bounds(limiter, min_fill_rate, max_fill_rate)
    _update_capacity(limiter, capacity)
    limiter.configure_endpoint_profiles(endpoint_profiles)


class AdaptiveRateLimiter:
    """
    Unified adaptive rate limiter using token bucket algorithm.

    Core Concept:
    - fill_rate (tokens/second) is the ONLY rate control
    - Token bucket handles bursts naturally
    - Adaptive adjustment modifies fill_rate based on API feedback
    - No extra delays, no competing mechanisms

    Adaptive Logic:
    - 429 error → decrease fill_rate by 12% (gentler slowdown)
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
        success_threshold: int = 20,
        rate_limiter_429_backoff: float = 0.85,
        rate_limiter_success_factor: float = 1.02,
    ):
        """
        Initialize adaptive rate limiter.

        Args:
            initial_fill_rate: Starting rate in requests per second (default: 1.5)
            capacity: Maximum burst capacity in tokens (default: 10.0)
            min_fill_rate: Minimum allowed rate (default: 0.1 req/s = 10s between)
            max_fill_rate: Maximum allowed rate (default: 3.0 req/s)
            success_threshold: Successes required before speedup (default: 20)
            rate_limiter_429_backoff: Multiplier for rate reduction on 429 (default: 0.85)
            rate_limiter_success_factor: Multiplier for rate increase on success (default: 1.02)
        """
        # Validate parameters
        if initial_fill_rate <= 0:
            raise ValueError(f"initial_fill_rate must be > 0, got {initial_fill_rate}")
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if min_fill_rate <= 0:
            raise ValueError(f"min_fill_rate must be > 0, got {min_fill_rate}")
        if max_fill_rate < min_fill_rate:
            raise ValueError(f"max_fill_rate ({max_fill_rate}) must be >= min_fill_rate ({min_fill_rate})")
        if success_threshold < 1:
            raise ValueError(f"success_threshold must be >= 1, got {success_threshold}")
        if not (0.0 < rate_limiter_429_backoff < 1.0):
            raise ValueError(f"rate_limiter_429_backoff must be between 0 and 1, got {rate_limiter_429_backoff}")
        if rate_limiter_success_factor <= 1.0:
            raise ValueError(f"rate_limiter_success_factor must be > 1.0, got {rate_limiter_success_factor}")

        # Token bucket state
        self.capacity = capacity
        self.fill_rate = initial_fill_rate
        self._initial_fill_rate = initial_fill_rate
        self.tokens = capacity  # Start with full bucket
        self.last_refill_time = time.monotonic()

        # Rate bounds
        self.min_fill_rate = min_fill_rate
        self.max_fill_rate = max_fill_rate

        # Adaptive learning state
        self.success_count = 0
        self.success_threshold = success_threshold
        self.rate_limiter_429_backoff = rate_limiter_429_backoff
        self.rate_limiter_success_factor = rate_limiter_success_factor

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

        self._endpoint_profiles: dict[str, _EndpointProfile] = {}
        self._endpoint_rate_cap: Optional[float] = None
        self._endpoint_last_call: dict[str, float] = {}
        self._endpoint_penalty_until: dict[str, float] = {}
        self._endpoint_summary: Optional[str] = None

        self.initial_source = "default"

        logger.debug(
            f"AdaptiveRateLimiter initialized: fill_rate={initial_fill_rate:.3f} req/s, "
            f"capacity={capacity:.1f}, range=[{min_fill_rate:.3f}, {max_fill_rate:.3f}], "
            f"success_threshold={success_threshold}, speedup=+2%"
        )

    def wait(self, endpoint: Optional[str] = None) -> float:
        """Wait according to token bucket algorithm.

        Args:
            endpoint: Optional identifier used to apply endpoint-specific throttling.

        Returns:
            Total time spent waiting in seconds (base token wait plus endpoint adjustments).
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

            wait_time = self._apply_endpoint_throttle(endpoint, wait_time)

            # Update metrics
            self._metrics["total_requests"] += 1
            self._metrics["total_wait_time"] += wait_time

            return wait_time

    def _apply_endpoint_throttle(self, endpoint: Optional[str], base_wait: float) -> float:
        """Apply endpoint-specific throttling rules and return updated wait time."""

        if not endpoint:
            return base_wait

        profile = self._endpoint_profiles.get(endpoint)
        if not profile:
            return base_wait

        now = time.monotonic()
        extra_delay = 0.0

        extra_delay += self._calculate_penalty_delay(endpoint, now)
        extra_delay += self._calculate_min_interval_delay(endpoint, profile, now)
        extra_delay += self._calculate_delay_multiplier(profile, base_wait)

        if extra_delay > 0.0:
            base_wait, now = self._perform_endpoint_delay(endpoint, base_wait, extra_delay)

        self._endpoint_last_call[endpoint] = now
        return base_wait

    def configure_endpoint_profiles(self, profiles: Optional[dict[str, Any]]) -> None:
        """Configure endpoint-specific throttling behavior."""

        with self._lock:
            if profiles is None:
                return

            self._reset_endpoint_profiles()
            if not profiles:
                return

            rate_caps = self._build_endpoint_profiles(profiles)

            if self._endpoint_profiles:
                self._log_endpoint_summary()

            if rate_caps:
                self._apply_endpoint_rate_cap(min(rate_caps))

    def _calculate_penalty_delay(self, endpoint: str, now: float) -> float:
        """Return cooldown delay remaining for an endpoint."""

        penalty_until = self._endpoint_penalty_until.get(endpoint)
        if penalty_until is None:
            return 0.0

        penalty_gap = penalty_until - now
        if penalty_gap > 0.0:
            return penalty_gap

        self._endpoint_penalty_until.pop(endpoint, None)
        return 0.0

    def _calculate_min_interval_delay(
        self,
        endpoint: str,
        profile: _EndpointProfile,
        now: float,
    ) -> float:
        """Compute additional delay required to satisfy min_interval."""

        if profile.min_interval <= 0.0:
            return 0.0

        last_call = self._endpoint_last_call.get(endpoint)
        if last_call is None:
            return 0.0

        gap = profile.min_interval - (now - last_call)
        return gap if gap > 0.0 else 0.0

    @staticmethod
    def _calculate_delay_multiplier(profile: _EndpointProfile, base_wait: float) -> float:
        """Return additional delay driven by the configured multiplier."""

        if profile.delay_multiplier <= 1.0 or base_wait <= 0.0:
            return 0.0

        return base_wait * (profile.delay_multiplier - 1.0)

    def _perform_endpoint_delay(
        self,
        endpoint: str,
        base_wait: float,
        extra_delay: float,
    ) -> tuple[float, float]:
        """Sleep for the calculated delay and refresh timing state."""

        logger.debug(
            "Endpoint '%s' throttle added %.3fs (base %.3fs)",
            endpoint,
            extra_delay,
            base_wait,
        )
        time.sleep(extra_delay)
        self._refill_tokens()
        updated_wait = base_wait + extra_delay
        self._endpoint_penalty_until.pop(endpoint, None)
        return updated_wait, time.monotonic()

    def _reset_endpoint_profiles(self) -> None:
        """Clear existing endpoint throttle state."""

        self._endpoint_profiles.clear()
        self._endpoint_last_call.clear()
        self._endpoint_penalty_until.clear()
        self._endpoint_rate_cap = None
        self._endpoint_summary = None

    def _build_endpoint_profiles(self, profiles: dict[str, Any]) -> list[float]:
        """Normalize provided profiles and collect any derived rate caps."""

        rate_caps: list[float] = []
        for endpoint, raw_profile in profiles.items():
            normalized = self._normalize_endpoint_profile(endpoint, raw_profile)
            if not normalized:
                continue

            profile, cap = normalized
            self._endpoint_profiles[endpoint] = profile
            if cap is not None:
                rate_caps.append(cap)

        return rate_caps

    def _normalize_endpoint_profile(
        self,
        endpoint: str,
        raw_profile: Any,
    ) -> Optional[tuple[_EndpointProfile, Optional[float]]]:
        """Return sanitized endpoint profile and any effective rate cap."""

        if not isinstance(raw_profile, dict):
            logger.warning("Invalid endpoint throttle entry for %s; expected dict", endpoint)
            return None

        profile_dict = cast(dict[str, Any], raw_profile)

        min_interval = _safe_float(profile_dict.get("min_interval")) or 0.0
        max_rate = self._sanitize_max_rate(profile_dict.get("max_rate"))
        delay_multiplier = _safe_float(profile_dict.get("delay_multiplier")) or 1.0
        cooldown_after_429 = _safe_float(profile_dict.get("cooldown_after_429")) or 0.0

        min_interval, rate_cap = self._apply_rate_cap_adjustments(min_interval, max_rate)

        delay_multiplier = max(1.0, delay_multiplier)
        cooldown_after_429 = max(0.0, cooldown_after_429)

        if self._is_profile_inactive(min_interval, delay_multiplier, cooldown_after_429):
            return None

        profile = _EndpointProfile(
            min_interval=min_interval,
            delay_multiplier=delay_multiplier,
            cooldown_after_429=cooldown_after_429,
            max_rate=max_rate,
        )

        return profile, rate_cap

    @staticmethod
    def _sanitize_max_rate(value: Any) -> Optional[float]:
        """Return a positive max rate when specified."""

        max_rate = _safe_float(value)
        if max_rate is None or max_rate <= 0.0:
            return None
        return max_rate

    @staticmethod
    def _is_profile_inactive(
        min_interval: float,
        delay_multiplier: float,
        cooldown_after_429: float,
    ) -> bool:
        """Return True when profile contains no throttling behaviour."""

        return min_interval <= 0.0 and delay_multiplier <= 1.0 and cooldown_after_429 <= 0.0

    @staticmethod
    def _apply_rate_cap_adjustments(
        min_interval: float,
        max_rate: Optional[float],
    ) -> tuple[float, Optional[float]]:
        """Harmonize min interval and max rate, returning any cap discovered."""

        if max_rate is not None:
            derived_interval = 1.0 / max_rate
            min_interval = max(derived_interval, min_interval)
            return max(min_interval, 0.0), max_rate

        if min_interval > 0.0:
            return max(min_interval, 0.0), 1.0 / min_interval

        return max(min_interval, 0.0), None

    def _log_endpoint_summary(self) -> None:
        """Emit a concise summary of active endpoint throttles."""

        summary_entries: list[str] = []
        for name, profile in self._endpoint_profiles.items():
            parts = [f"min={profile.min_interval:.2f}s", f"mult={profile.delay_multiplier:.2f}"]
            if profile.cooldown_after_429 > 0.0:
                parts.append(f"cooldown={profile.cooldown_after_429:.1f}s")
            if profile.max_rate:
                parts.append(f"max={profile.max_rate:.2f}/s")
            elif profile.min_interval > 0.0:
                parts.append(f"max≈{(1.0 / profile.min_interval):.2f}/s")
            summary_entries.append(f"{name}({', '.join(parts)})")

        if summary_entries:
            summary = ", ".join(summary_entries)
            self._endpoint_summary = f"Endpoint throttles configured: {summary}"
            logger.debug(self._endpoint_summary)
        else:
            self._endpoint_summary = None

    def _apply_endpoint_rate_cap(self, new_cap: float) -> None:
        """Clamp limiter bounds when endpoint caps require it."""

        self._endpoint_rate_cap = new_cap
        if new_cap >= self.max_fill_rate - 1e-6:
            return

        self.max_fill_rate = max(new_cap, self.min_fill_rate)
        if self.fill_rate > self.max_fill_rate:
            previous_rate = self.fill_rate
            self.fill_rate = self.max_fill_rate
            logger.info(
                "✅ Endpoint cap enforced: clamped fill rate from %.3f → %.3f req/s (endpoint limit %.3f req/s)",
                previous_rate,
                self.fill_rate,
                self.max_fill_rate,
            )

        if self.min_fill_rate > self.max_fill_rate:
            self.min_fill_rate = max(0.01, self.max_fill_rate * 0.5)

    def get_endpoint_summary(self) -> Optional[str]:
        """Return the most recent endpoint throttle summary, if available."""

        return self._endpoint_summary

    def on_429_error(self, endpoint: Optional[str] = None) -> None:
        """
        Handle 429 rate limit error by decreasing fill_rate.

        Decreases rate by 12% to gently back off from rate limit.
        Resets success counter to prevent premature speedup.

        This aggressive slowdown helps find the safe rate quickly
        without oscillation.

        Example:
            >>> limiter = AdaptiveRateLimiter(initial_fill_rate=1.0)
            >>> limiter.on_429_error()  # rate → 0.88 req/s
            >>> limiter.on_429_error()  # rate → 0.77 req/s
        """
        with self._lock:
            old_rate = self.fill_rate
            self.fill_rate = max(
                self.fill_rate * self.rate_limiter_429_backoff,
                self.min_fill_rate,
            )
            slowdown_pct = 0.0
            if old_rate > 0:
                slowdown_pct = max(0.0, (1 - (self.fill_rate / old_rate)) * 100)
            self.success_count = 0  # Reset success streak

            # Update metrics
            self._metrics["error_429_count"] += 1
            self._metrics["rate_decreases"] += 1

            # Calculate effective delay between requests
            effective_delay = 1.0 / self.fill_rate if self.fill_rate > 0 else float('inf')
            old_delay = 1.0 / old_rate if old_rate > 0 else float('inf')

            logger.warning(
                f"⚠️ 429 Rate Limit: Decreased rate from {old_rate:.3f} to "
                f"{self.fill_rate:.3f} req/s (-{slowdown_pct:.1f}%) | "
                f"Effective delay: {old_delay:.2f}s → {effective_delay:.2f}s | "
                f"Total 429s: {self._metrics['error_429_count']}"
            )

            if endpoint:
                profile = self._endpoint_profiles.get(endpoint)
                if profile and profile.cooldown_after_429 > 0.0:
                    cooldown_until = time.monotonic() + profile.cooldown_after_429
                    self._endpoint_penalty_until[endpoint] = cooldown_until
                    logger.warning(
                        "Endpoint '%s' entering cooldown for %.1fs after 429",
                        endpoint,
                        profile.cooldown_after_429,
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
                    self.fill_rate * self.rate_limiter_success_factor,
                    self.max_fill_rate,
                )
                self.success_count = 0  # Reset counter

                # Update metrics
                self._metrics["rate_increases"] += 1

                # Calculate effective delay
                old_delay = 1.0 / old_rate if old_rate > 0 else float('inf')
                new_delay = 1.0 / self.fill_rate if self.fill_rate > 0 else float('inf')

                # Only log if rate actually changed
                if abs(old_rate - self.fill_rate) > 0.001:
                    logger.info(
                        f"✅ After {self.success_threshold} successes: "
                        f"Increased rate to {self.fill_rate:.3f} req/s "
                        f"(+2% from {old_rate:.3f}) | "
                        f"Effective delay: {old_delay:.2f}s → {new_delay:.2f}s | "
                        f"Total increases: {self._metrics['rate_increases']}"
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
                endpoint_rate_cap=self._endpoint_rate_cap,
            )

    def print_metrics_summary(self) -> None:
        """Log a one-line summary of current limiter performance."""
        metrics = self.get_metrics()
        cap_fragment = f"\n - endpoint-cap={metrics.endpoint_rate_cap:.2f}/s" if metrics.endpoint_rate_cap else ""
        logger.info(
            "RateLimiter\n - rate=%.2f req/s\n - tokens=%.2f/%.2f\n - total=%d\n - avg-wait=%.3f\n - 429=%d\n - +%d/-%d%s",
            metrics.current_fill_rate,
            metrics.tokens_available,
            self.capacity,
            metrics.total_requests,
            metrics.avg_wait_time,
            metrics.error_429_count,
            metrics.rate_increases,
            metrics.rate_decreases,
            cap_fragment,
        )

    def get_status_message(self) -> str:
        """
        Get a human-readable status message for the rate limiter.

        Returns:
            A formatted string describing current rate limiter state.

        Example:
            >>> limiter = AdaptiveRateLimiter()
            >>> print(limiter.get_status_message())
            "⚡ Rate: 0.50 req/s | Tokens: 10.0/10.0 | Avg wait: 0.00s"
        """
        metrics = self.get_metrics()
        parts = [
            f"⚡ Rate: {metrics.current_fill_rate:.2f} req/s",
            f"Tokens: {metrics.tokens_available:.1f}/{self.capacity:.1f}",
        ]
        if metrics.avg_wait_time > 0:
            parts.append(f"Avg wait: {metrics.avg_wait_time:.2f}s")
        if metrics.error_429_count > 0:
            parts.append(f"429 errors: {metrics.error_429_count}")
        return " | ".join(parts)

    def estimate_time_for_requests(self, num_requests: int) -> float:
        """
        Estimate time needed to complete N requests at current rate.

        Args:
            num_requests: Number of requests to estimate.

        Returns:
            Estimated time in seconds.

        Example:
            >>> limiter = AdaptiveRateLimiter(rate=0.5)
            >>> limiter.estimate_time_for_requests(100)
            200.0  # 100 requests at 0.5/s = 200 seconds
        """
        if self.fill_rate <= 0:
            return float("inf")
        return num_requests / self.fill_rate

    def get_rate_budget(self, time_window_seconds: float = 60.0) -> int:
        """
        Calculate how many requests can be made in a time window.

        Args:
            time_window_seconds: Time window in seconds (default: 60s).

        Returns:
            Number of requests that can be made without throttling.

        Example:
            >>> limiter = AdaptiveRateLimiter(rate=0.5)
            >>> limiter.get_rate_budget(60)
            30  # 0.5 requests/sec * 60 sec = 30 requests
        """
        return int(self.fill_rate * time_window_seconds)

    def format_eta(self, remaining_requests: int) -> str:
        """
        Format an ETA string for remaining requests.

        Args:
            remaining_requests: Number of requests remaining.

        Returns:
            Human-readable ETA string (e.g., "~3m 20s").

        Example:
            >>> limiter = AdaptiveRateLimiter(rate=0.5)
            >>> limiter.format_eta(100)
            "~3m 20s"
        """
        if self.fill_rate <= 0 or remaining_requests <= 0:
            return "unknown"

        seconds = remaining_requests / self.fill_rate

        if seconds < 60:
            return f"~{seconds:.0f}s"
        if seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"~{minutes}m {secs}s"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"~{hours}h {minutes}m"

    def log_throttle_warning(self, wait_time: float, reason: str = "") -> None:
        """
        Log a user-visible warning when throttling occurs.

        Args:
            wait_time: How long we're waiting.
            reason: Optional reason for the throttle.
        """
        if wait_time < 0.1:
            return  # Don't log very short waits

        reason_str = f" ({reason})" if reason else ""
        logger.info(f"⏳ Rate limited{reason_str}, waiting {wait_time:.1f}s...")

    @property
    def current_delay(self) -> float:
        """Approximate current per-request delay derived from fill rate."""
        if self.fill_rate <= 0:
            return float("inf")
        return 1.0 / self.fill_rate

    @property
    def initial_delay(self) -> float:
        """Initial per-request delay before adaptive tuning."""
        if self._initial_fill_rate <= 0:
            return float("inf")
        return 1.0 / self._initial_fill_rate

    @property
    def initial_fill_rate(self) -> float:
        """Expose starting fill rate for diagnostics and logging."""
        return self._initial_fill_rate

    def calculate_budget(self, time_period_seconds: float = 60.0) -> dict[str, Any]:
        """
        Calculate the rate limit budget for a given time period.

        Args:
            time_period_seconds: Time period to calculate budget for (default: 60s).

        Returns:
            Dictionary with budget information:
            - estimated_requests: Number of requests possible in time period
            - time_period_seconds: The time period used
            - current_fill_rate: Current rate in requests per second
            - available_tokens: Currently available tokens in bucket

        Example:
            >>> limiter = AdaptiveRateLimiter(initial_fill_rate=2.0)
            >>> budget = limiter.calculate_budget(60.0)
            >>> print(f"Can make ~{budget['estimated_requests']} requests in 60s")
        """
        with self._lock:
            requests_from_rate = self.fill_rate * time_period_seconds
            requests_from_tokens = self.tokens
            return {
                "estimated_requests": int(requests_from_rate + requests_from_tokens),
                "time_period_seconds": time_period_seconds,
                "current_fill_rate": self.fill_rate,
                "available_tokens": self.tokens,
            }

    def get_health_status(self) -> str:
        """
        Get the current health status of the rate limiter.

        Returns:
            One of: 'optimal', 'degraded', 'throttled', 'critical'
            - optimal: No issues, rate at or above initial
            - degraded: Slightly below optimal (429 encountered)
            - throttled: Significantly below optimal rate
            - critical: Near minimum rate, severe throttling

        Example:
            >>> limiter = AdaptiveRateLimiter()
            >>> status = limiter.get_health_status()
            >>> if status == 'critical':
            ...     logger.warning("Rate limiter in critical state!")
        """
        metrics = self.get_metrics()
        rate_ratio = self.fill_rate / self._initial_fill_rate if self._initial_fill_rate > 0 else 1.0

        # No 429s and rate not degraded = optimal
        if metrics.error_429_count == 0 and rate_ratio >= 0.95:
            return "optimal"

        # Near minimum rate = critical
        min_ratio = self.min_fill_rate / self._initial_fill_rate if self._initial_fill_rate > 0 else 0.1
        if rate_ratio <= min_ratio * 1.1:  # Within 10% of minimum
            return "critical"

        # Significantly degraded (below 50% of initial)
        if rate_ratio < 0.5:
            return "throttled"

        # Some degradation
        return "degraded"

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
            self._endpoint_last_call.clear()
            self._endpoint_penalty_until.clear()
            # Note: fill_rate is NOT reset - it represents learned optimal rate
            logger.info("Rate limiter state reset (fill_rate preserved)")


def _get_state_path() -> Path:
    """Return the on-disk path used for persisting rate limiter state."""
    project_root = Path(__file__).resolve().parent
    return project_root / "Cache" / "rate_limiter_state.json"


def _load_persisted_state() -> Optional[dict[str, Any]]:
    """Load persisted rate limiter state from disk."""
    cached_state = _LIMITER_STATE.persisted_state_cache
    if cached_state is not None:
        return cached_state

    state_path = _get_state_path()
    if not state_path.exists():
        return None

    try:
        raw = state_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict) and "fill_rate" in data:
            _LIMITER_STATE.persisted_state_cache = data
            return data
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug(f"Failed to load persisted rate limiter state: {exc}")

    _LIMITER_STATE.persisted_state_cache = None
    return None


def _persist_state(payload: dict[str, Any]) -> None:
    """Persist rate limiter state to disk."""
    state_path = _get_state_path()
    try:
        # Use centralized atomic_write_file helper from test_utilities
        from test_utilities import atomic_write_file

        with atomic_write_file(state_path) as f:
            json.dump(payload, f, indent=2)

        _LIMITER_STATE.persisted_state_cache = payload
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug(f"Failed to persist rate limiter state: {exc}")


# Singleton instance for global access
_global_rate_limiter_lock = threading.Lock()


def get_adaptive_rate_limiter(
    initial_fill_rate: Optional[float] = None,
    success_threshold: Optional[int] = None,
    min_fill_rate: Optional[float] = None,
    max_fill_rate: Optional[float] = None,
    capacity: Optional[float] = None,
    endpoint_profiles: Optional[dict[str, Any]] = None,
    rate_limiter_429_backoff: Optional[float] = None,
    rate_limiter_success_factor: Optional[float] = None,
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
        rate_limiter_429_backoff: Multiplier for rate reduction on 429.
        rate_limiter_success_factor: Multiplier for rate increase on success.

    Returns:
        AdaptiveRateLimiter: The global rate limiter instance

    Example:
        >>> limiter = get_adaptive_rate_limiter()
        >>> limiter.wait()
    """
    with _global_rate_limiter_lock:
        limiter_state = _LIMITER_STATE
        if limiter_state.global_rate_limiter is None:
            config = _build_limiter_config(
                initial_fill_rate=initial_fill_rate,
                success_threshold=success_threshold,
                min_fill_rate=min_fill_rate,
                max_fill_rate=max_fill_rate,
                capacity=capacity,
                rate_limiter_429_backoff=rate_limiter_429_backoff,
                rate_limiter_success_factor=rate_limiter_success_factor,
            )

            limiter_state.global_rate_limiter = AdaptiveRateLimiter(
                initial_fill_rate=config.rate,
                success_threshold=config.success_threshold,
                min_fill_rate=config.min_rate,
                max_fill_rate=config.max_rate,
                capacity=config.capacity,
                rate_limiter_429_backoff=config.rate_limiter_429_backoff,
                rate_limiter_success_factor=config.rate_limiter_success_factor,
            )
            limiter_state.global_rate_limiter.configure_endpoint_profiles(endpoint_profiles)
            limiter_state.global_rate_limiter.initial_source = config.source
            limiter_state.rate_limiter_state_source = config.source
            logger.debug(
                "Created global AdaptiveRateLimiter with rate=%.3f req/s (source=%s, threshold=%d, bounds=%.3f-%.3f, capacity=%.1f, backoff=%.2f, success_factor=%.2f)",
                limiter_state.global_rate_limiter.fill_rate,
                config.source,
                limiter_state.global_rate_limiter.success_threshold,
                limiter_state.global_rate_limiter.min_fill_rate,
                limiter_state.global_rate_limiter.max_fill_rate,
                limiter_state.global_rate_limiter.capacity,
                limiter_state.global_rate_limiter.rate_limiter_429_backoff,
                limiter_state.global_rate_limiter.rate_limiter_success_factor,
            )
        else:
            _update_existing_limiter(
                limiter_state.global_rate_limiter,
                success_threshold=success_threshold,
                min_fill_rate=min_fill_rate,
                max_fill_rate=max_fill_rate,
                capacity=capacity,
                endpoint_profiles=endpoint_profiles,
            )

        return limiter_state.global_rate_limiter


def get_rate_limiter_state_source() -> str:
    """Return the origin of the current rate limiter initialization."""
    return _LIMITER_STATE.rate_limiter_state_source


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
    with _global_rate_limiter_lock:
        _LIMITER_STATE.global_rate_limiter = None
        _LIMITER_STATE.rate_limiter_state_source = "default"
        logger.info("Global rate limiter reset")


# =============================================================================
# TESTS
# =============================================================================


def rate_limiter_module_tests() -> bool:
    """Run comprehensive tests for AdaptiveRateLimiter."""

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

    # Test 3: 429 decreases rate by 12%
    suite.run_test(
        test_name="429 error handling",
        test_func=_test_429_decreases_rate,
        test_summary="Verify 429 error decreases fill_rate by 12%",
        functions_tested="AdaptiveRateLimiter.on_429_error",
        method_description="Trigger consecutive 429 errors",
        expected_outcome="Rate decreases by ~12% each time: 1.0 → 0.88 → 0.77",
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
        test_summary="Verify limiter is thread-safe for concurrent use",
        functions_tested="AdaptiveRateLimiter.wait, on_success",
        method_description="Run multiple threads making concurrent requests",
        expected_outcome="No race conditions or errors from concurrent access",
    )

    # Test 9: Capacity limits burst
    suite.run_test(
        test_name="Capacity limits burst",
        test_func=_test_capacity_limits_burst,
        test_summary="Verify capacity limits burst request count",
        functions_tested="AdaptiveRateLimiter.wait",
        method_description="Make burst exceeding capacity",
        expected_outcome="Only capacity tokens available for burst",
    )

    # Test 10: Cooldown on 429
    suite.run_test(
        test_name="Cooldown on 429 error",
        test_func=_test_cooldown_on_429,
        test_summary="Verify 429 error enforces cooldown period",
        functions_tested="AdaptiveRateLimiter.on_429_error, wait",
        method_description="Configure cooldown, trigger 429, ensure next call waits",
        expected_outcome="Subsequent call delayed by configured cooldown interval",
    )

    # Test 11: Global singleton reuse
    suite.run_test(
        test_name="Global singleton reuse",
        test_func=_test_global_singleton,
        test_summary="Verify global adaptive limiter is reused after initialization",
        functions_tested="get_adaptive_rate_limiter, reset_global_rate_limiter",
        method_description="Initialize limiter twice and confirm same instance is returned",
        expected_outcome="Second call returns first instance without reinitializing",
    )

    # Test 12: Parameter validation
    suite.run_test(
        test_name="Parameter validation guards",
        test_func=_test_parameter_validation,
        test_summary="Ensure invalid constructor arguments raise ValueError",
        functions_tested="AdaptiveRateLimiter.__init__",
        method_description="Attempt to create limiter with invalid fill rate and capacity",
        expected_outcome="ValueError raised for zero fill rate, negative capacity, or min>max",
    )

    # Test 13: Endpoint min interval enforcement
    suite.run_test(
        test_name="Endpoint min interval",
        test_func=_test_endpoint_min_interval,
        test_summary="Verify per-endpoint min intervals slow down consecutive calls",
        functions_tested="AdaptiveRateLimiter.configure_endpoint_profiles, wait",
        method_description="Set endpoint min_interval and measure delay between calls",
        expected_outcome="Second call delayed to respect configured min_interval",
    )

    # Test 14: Endpoint delay multiplier application
    suite.run_test(
        test_name="Endpoint delay multiplier",
        test_func=_test_endpoint_delay_multiplier,
        test_summary="Ensure endpoint delay multiplier increases wait duration",
        functions_tested="AdaptiveRateLimiter.configure_endpoint_profiles, wait",
        method_description="Apply delay multiplier and verify wait time is increased",
        expected_outcome="wait() respects multiplier and slows down requests",
    )

    # Test 15: Endpoint cooldown after 429
    suite.run_test(
        test_name="Endpoint cooldown enforcement",
        test_func=_test_endpoint_429_cooldown,
        test_summary="Ensure cooldown_after_429 prevents immediate reuse",
        functions_tested="AdaptiveRateLimiter.on_429_error, wait",
        method_description="Trigger 429 and ensure subsequent wait obeys cooldown",
        expected_outcome="Next request delayed by configured cooldown_after_429",
    )

    # Test 16: Status message output
    suite.run_test(
        test_name="Status message output",
        test_func=_test_status_message,
        test_summary="Ensure get_status_message returns appropriate status strings",
        functions_tested="AdaptiveRateLimiter.get_status_message",
        method_description="Test status message reflects limiter state",
        expected_outcome="Message indicates optimal/throttled status correctly",
    )

    # Test 17: Budget calculation
    suite.run_test(
        test_name="Budget calculation",
        test_func=_test_budget_calculation,
        test_summary="Ensure calculate_budget returns correct dictionary",
        functions_tested="AdaptiveRateLimiter.calculate_budget",
        method_description="Calculate request budget for time period",
        expected_outcome="Returns dict with estimated_requests, time_period, fill_rate",
    )

    # Test 18: Health status determination
    suite.run_test(
        test_name="Health status determination",
        test_func=_test_health_status,
        test_summary="Ensure get_health_status returns correct status",
        functions_tested="AdaptiveRateLimiter.get_health_status",
        method_description="Test health status based on 429 history and fill rate",
        expected_outcome="Returns optimal/degraded/throttled/critical as appropriate",
    )

    return suite.finish_suite()


# Use centralized test runner utility from test_utilities
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(rate_limiter_module_tests)


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
    assert 1.5 <= rate_limited_time <= 3.0, f"10 requests at 5 req/s should take ~2s, got {rate_limited_time:.2f}s"


def _test_429_decreases_rate() -> None:
    """Test 429 error decreases rate by approximately 15%."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=1.0)

    assert limiter.fill_rate == 1.0
    limiter.on_429_error()
    # Default backoff is 0.85
    assert abs(limiter.fill_rate - 0.85) < 0.001, f"Expected 0.85, got {limiter.fill_rate}"

    limiter.on_429_error()
    # 0.85 * 0.85 = 0.7225
    assert abs(limiter.fill_rate - 0.7225) < 0.001, f"Expected 0.7225, got {limiter.fill_rate}"


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
    limiter = AdaptiveRateLimiter(initial_fill_rate=0.2, min_fill_rate=0.1, max_fill_rate=0.5)

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
    errors: list[Exception] = []

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


def _test_capacity_limits_burst() -> None:
    """Ensure bursty usage can only consume up to the configured capacity."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=1.0, capacity=3.0)

    start = time.monotonic()
    for _ in range(3):
        limiter.wait()
    burst_time = time.monotonic() - start

    assert burst_time < 0.3, f"Burst of 3 tokens should be instant, took {burst_time:.3f}s"

    start = time.monotonic()
    limiter.wait()  # Fourth request should require refill
    refill_time = time.monotonic() - start

    assert refill_time >= 0.9, f"Expected ~1s refill wait, got {refill_time:.3f}s"


def _test_cooldown_on_429() -> None:
    """Ensure endpoint cooldown after 429 enforces additional delay."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=5.0, capacity=5.0)
    limiter.configure_endpoint_profiles(
        {
            "cooldown-endpoint": {
                "cooldown_after_429": 0.5,
            }
        }
    )

    limiter.wait("cooldown-endpoint")
    limiter.on_429_error("cooldown-endpoint")

    start = time.monotonic()
    limiter.wait("cooldown-endpoint")
    elapsed = time.monotonic() - start

    assert elapsed >= 0.45, f"Cooldown should enforce ~0.5s wait, got {elapsed:.3f}s"


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


def _test_endpoint_min_interval() -> None:
    """Ensure endpoint-specific min interval delays consecutive calls."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=10.0, capacity=10.0)
    limiter.configure_endpoint_profiles({"test-endpoint": {"min_interval": 0.3}})

    limiter.wait("test-endpoint")
    start = time.monotonic()
    limiter.wait("test-endpoint")
    elapsed = time.monotonic() - start

    assert elapsed >= 0.28, f"Expected at least 0.28s delay, got {elapsed:.3f}s"


def _test_endpoint_delay_multiplier() -> None:
    """Ensure endpoint-specific delay multiplier increases wait duration."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=4.0, capacity=1.0)
    limiter.configure_endpoint_profiles({"slow-endpoint": {"delay_multiplier": 2.0}})

    limiter.tokens = 0.0
    limiter.last_refill_time = time.monotonic()

    start = time.monotonic()
    limiter.wait("slow-endpoint")
    elapsed = time.monotonic() - start

    assert 0.45 <= elapsed <= 0.75, f"Expected ~0.5s wait, got {elapsed:.3f}s"


def _test_endpoint_429_cooldown() -> None:
    """Ensure endpoint cooldown prevents immediate reuse after a 429."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=5.0, capacity=5.0)
    limiter.configure_endpoint_profiles({"penalized": {"cooldown_after_429": 0.6}})

    limiter.wait("penalized")
    limiter.on_429_error("penalized")

    start = time.monotonic()
    limiter.wait("penalized")
    elapsed = time.monotonic() - start

    assert elapsed >= 0.55, f"Expected cooldown-driven wait >=0.55s, got {elapsed:.3f}s"


def _test_status_message() -> None:
    """Test get_status_message returns appropriate status strings."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=1.0, capacity=5.0)

    # Check basic format with rate and tokens
    msg = limiter.get_status_message()
    assert "Rate:" in msg, f"Expected 'Rate:' in status, got: {msg}"
    assert "Tokens:" in msg, f"Expected 'Tokens:' in status, got: {msg}"

    # After 429: should include error count
    limiter.on_429_error()
    msg = limiter.get_status_message()
    assert "429 errors:" in msg, f"Expected '429 errors:' after 429, got: {msg}"


def _test_budget_calculation() -> None:
    """Test calculate_budget returns expected dictionary structure."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=2.0, capacity=10.0)

    budget = limiter.calculate_budget(time_period_seconds=60.0)

    # Check required keys exist
    assert "estimated_requests" in budget, "Missing 'estimated_requests' key"
    assert "time_period_seconds" in budget, "Missing 'time_period_seconds' key"
    assert "current_fill_rate" in budget, "Missing 'current_fill_rate' key"
    assert "available_tokens" in budget, "Missing 'available_tokens' key"

    # Check values make sense
    assert budget["time_period_seconds"] == 60.0, "time_period should match input"
    assert budget["current_fill_rate"] == 2.0, "fill_rate should match limiter"
    # At 2 req/s for 60s = 120 + available tokens
    assert budget["estimated_requests"] >= 120, f"Expected at least 120 requests, got {budget['estimated_requests']}"


def _test_health_status() -> None:
    """Test get_health_status returns correct status based on metrics."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=1.0, capacity=5.0)

    # Fresh limiter should be optimal
    status = limiter.get_health_status()
    assert status == "optimal", f"Fresh limiter should be optimal, got: {status}"

    # Single 429: should be degraded
    limiter.on_429_error()
    status = limiter.get_health_status()
    assert status in ("degraded", "throttled"), f"After 429, expected degraded/throttled, got: {status}"

    # Multiple 429s: should be throttled or critical
    for _ in range(5):
        limiter.on_429_error()
    status = limiter.get_health_status()
    assert status in ("throttled", "critical"), f"After many 429s, expected throttled/critical, got: {status}"


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
