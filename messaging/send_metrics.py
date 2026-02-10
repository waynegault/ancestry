#!/usr/bin/env python3
"""Unified metrics for message send operations.

Phase 6.1: Provides Prometheus-compatible metrics for tracking:
- Sends by trigger type (AUTOMATED_SEQUENCE, REPLY_RECEIVED, OPT_OUT, HUMAN_APPROVED)
- Safety check block rates by check type
- Content generation time by source
- API success/failure rates

Usage:
    from messaging.send_metrics import record_send_attempt, record_safety_block, record_generation_time

    # Record successful send
    record_send_attempt(trigger_type="AUTOMATED_SEQUENCE", success=True)

    # Record safety block
    record_safety_block(check_type="duplicate_prevention")

    # Record content generation time
    record_generation_time(source="template", seconds=0.25)
"""

from __future__ import annotations

import functools
import logging
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from testing.test_framework import TestSuite

if TYPE_CHECKING:
    from observability.metrics_registry import MetricsBundle

logger = logging.getLogger(__name__)

# Module-level reference to MetricsBundle (set by configure_send_metrics)
_metrics_bundle: MetricsBundle | None = None
_metrics_enabled: bool = False


def configure_send_metrics(metrics_bundle: MetricsBundle | None) -> None:
    """Configure send metrics with the global MetricsBundle.

    Args:
        metrics_bundle: The MetricsBundle from MetricsRegistry, or None to disable.
    """
    global _metrics_bundle, _metrics_enabled
    _metrics_bundle = metrics_bundle
    _metrics_enabled = metrics_bundle is not None
    if _metrics_enabled:
        logger.debug("Send metrics configured with MetricsBundle")
    else:
        logger.debug("Send metrics disabled (no MetricsBundle)")


def _requires_metrics(func: Callable[..., None]) -> Callable[..., None]:
    """Decorator that skips the function when metrics are disabled."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        if not _metrics_enabled or _metrics_bundle is None:
            return None
        return func(*args, **kwargs)

    return wrapper


# --------------------------------------------------------------------------
# Metric Recording Functions
# --------------------------------------------------------------------------


@_requires_metrics
def record_send_attempt(
    trigger_type: str,
    success: bool,
    error_type: str | None = None,
) -> None:
    """Record a message send attempt.

    Args:
        trigger_type: The trigger type (AUTOMATED_SEQUENCE, REPLY_RECEIVED, OPT_OUT, HUMAN_APPROVED)
        success: Whether the send succeeded
        error_type: Optional error category if failed (e.g., "api_error", "timeout")
    """
    # Normalize trigger type to standard values
    trigger = _normalize_trigger_type(trigger_type)
    result = "success" if success else (error_type or "error")

    try:
        _metrics_bundle.send_attempts.inc(trigger=trigger, result=result)
    except Exception as e:
        logger.debug("Failed to record send_attempts metric: %s", e)


@_requires_metrics
def record_safety_block(check_type: str) -> None:
    """Record a send blocked by a safety check.

    Args:
        check_type: The safety check that blocked (opt_out, duplicate, policy, hard_stop)
    """
    # Normalize check type
    check = _normalize_check_type(check_type)

    try:
        _metrics_bundle.safety_blocks.inc(check_type=check)
    except Exception as e:
        logger.debug("Failed to record safety_blocks metric: %s", e)


@_requires_metrics
def record_generation_time(source: str, seconds: float) -> None:
    """Record content generation time.

    Args:
        source: Content source (template, ai_generated, approved_draft)
        seconds: Time taken in seconds
    """
    # Normalize source type
    src = _normalize_source_type(source)

    try:
        _metrics_bundle.content_generation_time.observe(source=src, seconds=max(seconds, 0.0))
    except Exception as e:
        logger.debug("Failed to record content_generation_time metric: %s", e)


@_requires_metrics
def record_api_result(endpoint: str, success: bool, status_code: int | None = None) -> None:
    """Record API call result.

    Args:
        endpoint: API endpoint name (e.g., "send_message")
        success: Whether the API call succeeded
        status_code: Optional HTTP status code
    """
    result = "success" if success else "failure"
    family = _status_code_family(status_code) if status_code else "unknown"

    try:
        _metrics_bundle.send_api_results.inc(endpoint=endpoint, result=result, status_family=family)
    except Exception as e:
        logger.debug("Failed to record send_api_results metric: %s", e)


@_requires_metrics
def record_decision_path(decision_type: str) -> None:
    """Record the decision path taken by the orchestrator.

    Args:
        decision_type: The decision made (send, block, skip, desist)
    """
    decision = decision_type.lower()
    if decision not in {"send", "block", "skip", "desist"}:
        decision = "other"

    try:
        _metrics_bundle.decision_paths.inc(decision=decision)
    except Exception as e:
        logger.debug("Failed to record decision_paths metric: %s", e)


# --------------------------------------------------------------------------
# Context Managers for Timing
# --------------------------------------------------------------------------


@contextmanager
def timed_generation(source: str) -> Iterator[None]:
    """Context manager to time and record content generation.

    Usage:
        with timed_generation("template"):
            content = template.render(context)

    Args:
        source: Content source for metric label
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        record_generation_time(source, elapsed)


# --------------------------------------------------------------------------
# Metric Snapshot for Testing
# --------------------------------------------------------------------------


@dataclass
class SendMetricsSnapshot:
    """Snapshot of send metrics for testing and analysis."""

    timestamp: datetime = field(default_factory=datetime.now)
    total_attempts: int = 0
    successful_sends: int = 0
    failed_sends: int = 0
    blocked_by_safety: int = 0
    triggers: dict[str, int] = field(default_factory=dict)
    safety_checks: dict[str, int] = field(default_factory=dict)
    sources: dict[str, float] = field(default_factory=dict)  # avg generation time

    def __str__(self) -> str:
        """Human-readable summary."""
        success_rate = f"{100 * self.successful_sends / self.total_attempts:.1f}%" if self.total_attempts > 0 else "N/A"
        return (
            f"SendMetrics @ {self.timestamp.isoformat()}: "
            f"{self.total_attempts} attempts, {success_rate} success, "
            f"{self.blocked_by_safety} blocked"
        )


def get_metrics_snapshot() -> SendMetricsSnapshot:
    """Get current metrics snapshot (for testing/analysis).

    Note: This returns a minimal snapshot when metrics are disabled.
    """
    return SendMetricsSnapshot(timestamp=datetime.now())


# --------------------------------------------------------------------------
# Normalization Helpers
# --------------------------------------------------------------------------


def _normalize_trigger_type(trigger: str) -> str:
    """Normalize trigger type to standard labels."""
    trigger_upper = trigger.upper()
    if trigger_upper in {"AUTOMATED_SEQUENCE", "AUTOMATED"}:
        return "automated"
    if trigger_upper in {"REPLY_RECEIVED", "REPLY"}:
        return "reply"
    if trigger_upper in {"OPT_OUT", "OPTOUT", "DESIST"}:
        return "opt_out"
    if trigger_upper in {"HUMAN_APPROVED", "APPROVED"}:
        return "approved"
    return "other"


def _normalize_check_type(check: str) -> str:
    """Normalize safety check type to standard labels."""
    check_lower = check.lower().replace(" ", "_").replace("-", "_")
    if "opt" in check_lower and "out" in check_lower:
        return "opt_out"
    if "duplicate" in check_lower:
        return "duplicate"
    if "policy" in check_lower:
        return "policy"
    if "hard" in check_lower and "stop" in check_lower:
        return "hard_stop"
    if "rate" in check_lower and "limit" in check_lower:
        return "rate_limit"
    return "other"


def _normalize_source_type(source: str) -> str:
    """Normalize content source to standard labels."""
    source_lower = source.lower().replace(" ", "_").replace("-", "_")
    if "template" in source_lower:
        return "template"
    if "ai" in source_lower or "generated" in source_lower:
        return "ai_generated"
    if "approved" in source_lower or "draft" in source_lower:
        return "approved_draft"
    if "desist" in source_lower:
        return "desist"
    return "other"


def _status_code_family(code: int) -> str:
    """Convert HTTP status code to family label."""
    if 200 <= code < 300:
        return "2xx"
    if 300 <= code < 400:
        return "3xx"
    if 400 <= code < 500:
        return "4xx"
    if 500 <= code < 600:
        return "5xx"
    return "unknown"


# --------------------------------------------------------------------------
# Module Tests
# --------------------------------------------------------------------------


def _run_module_tests() -> bool:
    """Run send_metrics module tests."""
    suite = TestSuite("Send Metrics", "messaging/send_metrics.py")
    suite.start_suite()

    # Test 1: Configure with None (disabled)
    def test_configure_disabled() -> None:
        configure_send_metrics(None)
        # Should not raise
        record_send_attempt("AUTOMATED_SEQUENCE", success=True)
        record_safety_block("opt_out")
        record_generation_time("template", 0.5)

    suite.run_test("configure_send_metrics with None", test_configure_disabled)

    # Test 2: Normalize trigger types
    def test_normalize_triggers() -> None:
        assert _normalize_trigger_type("AUTOMATED_SEQUENCE") == "automated"
        assert _normalize_trigger_type("automated") == "automated"
        assert _normalize_trigger_type("REPLY_RECEIVED") == "reply"
        assert _normalize_trigger_type("reply") == "reply"
        assert _normalize_trigger_type("OPT_OUT") == "opt_out"
        assert _normalize_trigger_type("DESIST") == "opt_out"
        assert _normalize_trigger_type("HUMAN_APPROVED") == "approved"
        assert _normalize_trigger_type("unknown") == "other"

    suite.run_test("normalize trigger types", test_normalize_triggers)

    # Test 3: Normalize check types
    def test_normalize_checks() -> None:
        assert _normalize_check_type("opt_out") == "opt_out"
        assert _normalize_check_type("OPT-OUT") == "opt_out"
        assert _normalize_check_type("duplicate_prevention") == "duplicate"
        assert _normalize_check_type("policy") == "policy"
        assert _normalize_check_type("hard_stop") == "hard_stop"
        assert _normalize_check_type("rate_limit") == "rate_limit"
        assert _normalize_check_type("unknown") == "other"

    suite.run_test("normalize check types", test_normalize_checks)

    # Test 4: Normalize source types
    def test_normalize_sources() -> None:
        assert _normalize_source_type("template") == "template"
        assert _normalize_source_type("TEMPLATE") == "template"
        assert _normalize_source_type("ai_generated") == "ai_generated"
        assert _normalize_source_type("AI") == "ai_generated"
        assert _normalize_source_type("approved_draft") == "approved_draft"
        assert _normalize_source_type("draft") == "approved_draft"
        assert _normalize_source_type("desist") == "desist"
        assert _normalize_source_type("unknown") == "other"

    suite.run_test("normalize source types", test_normalize_sources)

    # Test 5: Status code families
    def test_status_families() -> None:
        assert _status_code_family(200) == "2xx"
        assert _status_code_family(201) == "2xx"
        assert _status_code_family(301) == "3xx"
        assert _status_code_family(400) == "4xx"
        assert _status_code_family(404) == "4xx"
        assert _status_code_family(500) == "5xx"
        assert _status_code_family(100) == "unknown"

    suite.run_test("status code families", test_status_families)

    # Test 6: Metrics snapshot
    def test_metrics_snapshot() -> None:
        snapshot = get_metrics_snapshot()
        assert snapshot.total_attempts == 0
        assert snapshot.timestamp is not None
        str_repr = str(snapshot)
        assert "SendMetrics" in str_repr

    suite.run_test("metrics snapshot", test_metrics_snapshot)

    # Test 7: Timed generation context manager
    def test_timed_generation() -> None:
        configure_send_metrics(None)  # Disabled, should not raise
        with timed_generation("template"):
            time.sleep(0.01)  # Small delay
        # No assertion needed - just verify no error

    suite.run_test("timed_generation context manager", test_timed_generation)

    # Test 8: Record functions with disabled metrics
    def test_record_functions_disabled() -> None:
        configure_send_metrics(None)
        # All should execute without error
        record_send_attempt("AUTOMATED", success=True)
        record_send_attempt("REPLY", success=False, error_type="timeout")
        record_safety_block("duplicate")
        record_generation_time("ai_generated", 1.5)
        record_api_result("send_message", success=True, status_code=200)
        record_api_result("send_message", success=False, status_code=500)
        record_decision_path("send")
        record_decision_path("block")
        record_decision_path("invalid_type")

    suite.run_test("record functions with disabled metrics", test_record_functions_disabled)

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run all send_metrics tests."""
    return _run_module_tests()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
