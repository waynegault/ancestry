#!/usr/bin/env python3

"""
Correlation ID System for Request Tracking

Provides correlation IDs to track operations across modules and log entries.
Enables tracing a single user action through all system components.

Usage:
    from core.correlation import correlation_context, get_correlation_id

    # Start a new correlation context for an action
    with correlation_context("action6_gather"):
        # All logs within this context will include the correlation ID
        logger.info("Starting DNA match gathering")
        process_matches()

    # Or manually get/set correlation ID
    corr_id = get_correlation_id()
    logger.info(f"Processing with correlation_id={corr_id}")
"""

# === CORE INFRASTRUCTURE ===
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent.resolve())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

# === CORRELATION ID STORAGE ===

# Thread-local storage for correlation context
_correlation_context: threading.local = threading.local()


@dataclass
class CorrelationContext:
    """Holds correlation tracking data for a request/operation."""

    correlation_id: str
    operation_name: str
    start_time: float = field(default_factory=time.time)
    parent_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_seconds(self) -> float:
        """Return elapsed time since context creation."""
        return time.time() - self.start_time

    @property
    def elapsed_formatted(self) -> str:
        """Return elapsed time as formatted string."""
        elapsed = self.elapsed_seconds
        if elapsed < 1:
            return f"{elapsed * 1000:.0f}ms"
        if elapsed < 60:
            return f"{elapsed:.2f}s"
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        return f"{minutes}m {seconds:.1f}s"


def generate_correlation_id() -> str:
    """Generate a new unique correlation ID."""
    # Use UUID4 for uniqueness, take first 8 chars for readability
    return uuid.uuid4().hex[:8]


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID, if any."""
    ctx = getattr(_correlation_context, "current", None)
    return ctx.correlation_id if ctx else None


def get_correlation_context() -> Optional[CorrelationContext]:
    """Get the full correlation context, if any."""
    return getattr(_correlation_context, "current", None)


def set_correlation_context(ctx: Optional[CorrelationContext]) -> None:
    """Set the current correlation context."""
    _correlation_context.current = ctx


@contextmanager
def correlation_context(
    operation_name: str,
    correlation_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
):
    """
    Context manager for correlation tracking.

    Args:
        operation_name: Name of the operation (e.g., "action6_gather", "api_request")
        correlation_id: Optional existing correlation ID to use (for nested contexts)
        metadata: Optional metadata to attach to the context

    Yields:
        CorrelationContext: The active correlation context

    Example:
        with correlation_context("action6_gather", metadata={"page": 1}) as ctx:
            logger.info(f"Started operation {ctx.correlation_id}")
            # ... do work ...
        # Context automatically logs completion with elapsed time
    """
    # Save parent context for nesting
    parent_ctx = get_correlation_context()
    parent_id = parent_ctx.correlation_id if parent_ctx else None

    # Create new context
    ctx = CorrelationContext(
        correlation_id=correlation_id or generate_correlation_id(),
        operation_name=operation_name,
        parent_id=parent_id,
        metadata=metadata or {},
    )

    # Set as current context
    set_correlation_context(ctx)

    try:
        logger.debug(
            f"[{ctx.correlation_id}] Started: {operation_name}" + (f" (parent={parent_id})" if parent_id else "")
        )
        yield ctx
    finally:
        elapsed = ctx.elapsed_formatted
        logger.debug(f"[{ctx.correlation_id}] Completed: {operation_name} in {elapsed}")
        # Restore parent context
        set_correlation_context(parent_ctx)


# === LOGGING FILTER FOR CORRELATION IDS ===


class CorrelationFilter(logging.Filter):
    """
    Logging filter that adds correlation_id to log records.

    Install this filter on handlers to automatically include correlation IDs
    in all log messages when a correlation context is active.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: PLR6301
        """Add correlation_id to the log record."""
        ctx = get_correlation_context()
        record.correlation_id = ctx.correlation_id if ctx else "-"  # type: ignore[attr-defined]
        record.operation_name = ctx.operation_name if ctx else "-"  # type: ignore[attr-defined]
        return True


def install_correlation_filter(logger_name: Optional[str] = None) -> None:
    """
    Install the correlation filter on a logger.

    Args:
        logger_name: Name of logger to install filter on. None for root logger.
    """
    target_logger = logging.getLogger(logger_name)
    correlation_filter = CorrelationFilter()

    # Add filter to all handlers
    for handler in target_logger.handlers:
        handler.addFilter(correlation_filter)

    # Also add to the logger itself for propagated messages
    target_logger.addFilter(correlation_filter)


# === STRUCTURED LOG HELPERS ===


def log_with_context(
    level: int,
    message: str,
    logger_instance: Optional[logging.Logger] = None,
    **extra_fields: Any,
) -> None:
    """
    Log a message with correlation context and extra structured fields.

    Args:
        level: Logging level (e.g., logging.INFO)
        message: Log message
        logger_instance: Logger to use (defaults to module logger)
        **extra_fields: Additional fields to include in the log
    """
    log = logger_instance or logger
    ctx = get_correlation_context()

    # Build structured message
    parts = [message]
    if ctx:
        parts.insert(0, f"[{ctx.correlation_id}]")
    if extra_fields:
        field_str = " ".join(f"{k}={v}" for k, v in extra_fields.items())
        parts.append(f"| {field_str}")

    log.log(level, " ".join(parts))


def log_operation_start(
    operation: str,
    logger_instance: Optional[logging.Logger] = None,
    **context: Any,
) -> None:
    """Log the start of an operation with context."""
    log_with_context(
        logging.INFO,
        f"▶ Starting: {operation}",
        logger_instance,
        **context,
    )


def log_operation_end(
    operation: str,
    success: bool = True,
    logger_instance: Optional[logging.Logger] = None,
    **context: Any,
) -> None:
    """Log the end of an operation with context."""
    ctx = get_correlation_context()
    elapsed = ctx.elapsed_formatted if ctx else "?"

    status = "✓" if success else "✗"
    level = logging.INFO if success else logging.ERROR

    log_with_context(
        level,
        f"{status} Completed: {operation} in {elapsed}",
        logger_instance,
        success=success,
        **context,
    )


# === MODULE TESTS ===

from test_framework import TestSuite, create_standard_test_runner


def _test_correlation_id_generation() -> bool:
    """Test correlation ID generation."""
    id1 = generate_correlation_id()
    id2 = generate_correlation_id()

    # IDs should be 8 characters
    assert len(id1) == 8, f"Expected 8 chars, got {len(id1)}"
    assert len(id2) == 8, f"Expected 8 chars, got {len(id2)}"

    # IDs should be unique
    assert id1 != id2, "Generated IDs should be unique"

    # IDs should be hex strings
    int(id1, 16)  # Should not raise
    int(id2, 16)  # Should not raise

    return True


def _test_correlation_context_basic() -> bool:
    """Test basic correlation context usage."""
    # No context initially
    assert get_correlation_id() is None

    with correlation_context("test_operation") as ctx:
        # Context should be active
        assert get_correlation_id() == ctx.correlation_id
        assert ctx.operation_name == "test_operation"
        assert ctx.parent_id is None

    # Context should be cleared after exit
    assert get_correlation_id() is None

    return True


def _test_correlation_context_nested() -> bool:
    """Test nested correlation contexts."""
    with correlation_context("outer") as outer_ctx:
        outer_id = outer_ctx.correlation_id

        with correlation_context("inner") as inner_ctx:
            # Inner context should have parent reference
            assert inner_ctx.parent_id == outer_id
            assert inner_ctx.correlation_id != outer_id
            assert get_correlation_id() == inner_ctx.correlation_id

        # After inner exits, should be back to outer
        assert get_correlation_id() == outer_id

    # After both exit, should be None
    assert get_correlation_id() is None

    return True


def _test_correlation_context_with_metadata() -> bool:
    """Test correlation context with metadata."""
    metadata = {"page": 1, "action": "gather"}

    with correlation_context("test_with_meta", metadata=metadata) as ctx:
        assert ctx.metadata == metadata
        assert ctx.metadata["page"] == 1
        assert ctx.metadata["action"] == "gather"

    return True


def _test_correlation_context_elapsed_time() -> bool:
    """Test elapsed time tracking."""
    with correlation_context("timed_operation") as ctx:
        time.sleep(0.1)  # Sleep for 100ms
        elapsed = ctx.elapsed_seconds
        assert elapsed >= 0.1, f"Expected >= 0.1s, got {elapsed}"
        assert elapsed < 0.5, f"Expected < 0.5s, got {elapsed}"

        # Test formatted output
        formatted = ctx.elapsed_formatted
        assert "ms" in formatted or "s" in formatted

    return True


def _test_correlation_filter() -> bool:
    """Test correlation filter adds IDs to log records."""
    filter_instance = CorrelationFilter()

    # Create a mock log record
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    # Without context, should have "-"
    filter_instance.filter(record)
    assert getattr(record, "correlation_id", None) == "-"

    # With context, should have the ID
    with correlation_context("filter_test") as ctx:
        filter_instance.filter(record)
        assert getattr(record, "correlation_id", None) == ctx.correlation_id

    return True


def _test_explicit_correlation_id() -> bool:
    """Test using explicit correlation ID."""
    explicit_id = "test1234"

    with correlation_context("explicit_test", correlation_id=explicit_id) as ctx:
        assert ctx.correlation_id == explicit_id
        assert get_correlation_id() == explicit_id

    return True


def module_tests() -> bool:
    """Run all module tests."""
    suite = TestSuite("Correlation ID System", "core/correlation.py")
    suite.start_suite()

    suite.run_test("Correlation ID generation", _test_correlation_id_generation)
    suite.run_test("Basic correlation context", _test_correlation_context_basic)
    suite.run_test("Nested correlation contexts", _test_correlation_context_nested)
    suite.run_test("Correlation context with metadata", _test_correlation_context_with_metadata)
    suite.run_test("Elapsed time tracking", _test_correlation_context_elapsed_time)
    suite.run_test("Correlation filter", _test_correlation_filter)
    suite.run_test("Explicit correlation ID", _test_explicit_correlation_id)

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
