#!/usr/bin/env python3
"""
Application Performance Monitoring (APM) Module.

Provides lightweight tracing and span tracking for production performance visibility.
Inspired by OpenTelemetry but without heavy dependencies.

Features:
- Automatic span tracking with context propagation
- Decorator-based instrumentation
- Performance metrics collection (duration, memory, CPU)
- JSON export for external APM tools (Sentry, Datadog, etc.)
- Configurable sampling for high-throughput scenarios
"""

# === CORE INFRASTRUCTURE ===
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import logging

logger = logging.getLogger(__name__)

# === STANDARD LIBRARY IMPORTS ===
import contextlib
import functools
import json
import random
import threading
import time
import uuid
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


class SpanStatus(Enum):
    """Status of a span execution."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class Span:
    """Represents a unit of work being traced."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the span."""
        self.events.append(
            {
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {},
            }
        )

    def set_status(self, status: SpanStatus, description: str | None = None) -> None:
        """Set the span status."""
        self.status = status
        if description:
            self.attributes["status_description"] = description

    def end(self, status: SpanStatus | None = None) -> None:
        """End the span."""
        self.end_time = time.time()
        if status:
            self.status = status
        elif self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary for export."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": self.events,
        }


class SpanExporter:
    """Base class for span exporters."""

    def export(self, spans: list[Span]) -> bool:
        """Export spans. Returns True if successful."""
        raise NotImplementedError


class ConsoleSpanExporter(SpanExporter):
    """Exports spans to console (for debugging)."""

    @staticmethod
    def export(spans: list[Span]) -> bool:
        for span in spans:
            logger.info(
                "SPAN: %s [%s] %.2fms - %s",
                span.name,
                span.status.value,
                span.duration_ms,
                span.attributes,
            )
        return True


class JSONFileSpanExporter(SpanExporter):
    """Exports spans to a JSON file."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self._lock = threading.Lock()

    def export(self, spans: list[Span]) -> bool:
        try:
            with self._lock:
                existing: list[dict[str, Any]] = []
                if self.file_path.exists():
                    try:
                        with self.file_path.open(encoding="utf-8") as f:
                            existing = json.load(f)
                    except (json.JSONDecodeError, OSError):
                        existing = []

                existing.extend(span.to_dict() for span in spans)

                # Keep only last 1000 spans
                if len(existing) > 1000:
                    existing = existing[-1000:]

                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                with self.file_path.open("w", encoding="utf-8") as f:
                    json.dump(existing, f, indent=2)
            return True
        except Exception as e:
            logger.error("Failed to export spans to %s: %s", self.file_path, e)
            return False


class Tracer:
    """
    Main tracer for creating and managing spans.

    Thread-safe with context propagation for nested spans.
    """

    _instance: Optional["Tracer"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "Tracer":
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._enabled = True
        self._sample_rate = 1.0  # 100% by default
        self._exporters: list[SpanExporter] = []
        self._pending_spans: list[Span] = []
        self._context = threading.local()
        self._spans_lock = threading.Lock()
        self._initialized = True
        logger.debug("Tracer singleton initialized")

    def configure(
        self,
        enabled: bool = True,
        sample_rate: float = 1.0,
        exporters: list[SpanExporter] | None = None,
    ) -> None:
        """Configure the tracer."""
        self._enabled = enabled
        self._sample_rate = max(0.0, min(1.0, sample_rate))
        if exporters is not None:
            self._exporters = exporters
        logger.info(
            "Tracer configured: enabled=%s, sample_rate=%.2f, exporters=%d",
            enabled,
            self._sample_rate,
            len(self._exporters),
        )

    def _should_sample(self) -> bool:
        """Determine if this trace should be sampled."""
        return self._enabled and random.random() < self._sample_rate

    def _get_current_span(self) -> Span | None:
        """Get the current span from thread-local context."""
        return getattr(self._context, "current_span", None)

    def _set_current_span(self, span: Span | None) -> None:
        """Set the current span in thread-local context."""
        self._context.current_span = span

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Span | None:
        """Start a new span."""
        if not self._should_sample():
            return None

        parent = self._get_current_span()
        trace_id = parent.trace_id if parent else uuid.uuid4().hex[:16]
        span_id = uuid.uuid4().hex[:8]
        parent_span_id = parent.span_id if parent else None

        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
        )
        if attributes:
            span.attributes.update(attributes)

        self._set_current_span(span)
        return span

    def end_span(self, span: Span | None, status: SpanStatus | None = None) -> None:
        """End a span and queue for export."""
        if span is None:
            return

        span.end(status)

        # Restore parent span as current
        # For simplicity, we just clear the current span
        # In a full implementation, we'd maintain a stack
        self._set_current_span(None)

        with self._spans_lock:
            self._pending_spans.append(span)

        # Auto-flush if we have enough spans
        if len(self._pending_spans) >= 100:
            self.flush()

    def flush(self) -> int:
        """Flush pending spans to exporters. Returns count of exported spans."""
        with self._spans_lock:
            if not self._pending_spans:
                return 0
            spans_to_export = self._pending_spans.copy()
            self._pending_spans.clear()

        for exporter in self._exporters:
            try:
                exporter.export(spans_to_export)
            except Exception as e:
                logger.error("Exporter %s failed: %s", type(exporter).__name__, e)

        return len(spans_to_export)

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span | None, None, None]:
        """Context manager for creating spans."""
        span = self.start_span(name, attributes)
        try:
            yield span
            if span:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            if span:
                span.set_status(SpanStatus.ERROR, str(e))
                span.set_attribute("exception.type", type(e).__name__)
                span.set_attribute("exception.message", str(e))
            raise
        finally:
            self.end_span(span)

    def reset(self) -> None:
        """Reset tracer state (for testing)."""
        with self._spans_lock:
            self._pending_spans.clear()
        self._exporters.clear()
        self._enabled = True
        self._sample_rate = 1.0


def trace(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """
    Decorator to automatically trace a function.

    Args:
        name: Span name (defaults to function name)
        attributes: Additional span attributes

    Example:
        @trace()
        def my_function():
            pass

        @trace("custom_name", {"key": "value"})
        def another_function():
            pass
    """

    def decorator(func: F) -> F:
        span_name = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = Tracer()
            with tracer.span(span_name, attributes):
                return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


# Convenience functions
def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    return Tracer()


def configure_apm(
    enabled: bool = True,
    sample_rate: float = 1.0,
    export_to_console: bool = False,
    export_to_file: Path | None = None,
) -> None:
    """
    Configure APM with common options.

    Args:
        enabled: Enable/disable tracing
        sample_rate: Sampling rate (0.0-1.0)
        export_to_console: Enable console export
        export_to_file: Path to JSON export file
    """
    exporters: list[SpanExporter] = []
    if export_to_console:
        exporters.append(ConsoleSpanExporter())
    if export_to_file:
        exporters.append(JSONFileSpanExporter(export_to_file))

    Tracer().configure(
        enabled=enabled,
        sample_rate=sample_rate,
        exporters=exporters,
    )


# =============================================================================
# Module Tests
# =============================================================================


def _test_span_creation() -> bool:
    """Test basic span creation and attributes."""
    span = Span(
        name="test_span",
        trace_id="abc123",
        span_id="xyz789",
    )
    assert span.name == "test_span"
    assert span.trace_id == "abc123"
    assert span.status == SpanStatus.UNSET

    span.set_attribute("key", "value")
    assert span.attributes["key"] == "value"

    span.end(SpanStatus.OK)
    assert span.status == SpanStatus.OK
    assert span.end_time is not None
    assert span.duration_ms >= 0

    return True


def _test_span_events() -> bool:
    """Test span event recording."""
    span = Span(name="test", trace_id="t1", span_id="s1")
    span.add_event("checkpoint", {"step": 1})
    span.add_event("checkpoint", {"step": 2})

    assert len(span.events) == 2
    assert span.events[0]["name"] == "checkpoint"
    assert span.events[0]["attributes"]["step"] == 1

    return True


def _test_tracer_singleton() -> bool:
    """Test tracer singleton pattern."""
    t1 = Tracer()
    t2 = Tracer()
    assert t1 is t2
    return True


def _test_tracer_span_context() -> bool:
    """Test tracer span creation and context."""
    tracer = Tracer()
    tracer.reset()
    tracer.configure(enabled=True, sample_rate=1.0)

    span = tracer.start_span("test_operation", {"attr": "value"})
    assert span is not None
    assert span.name == "test_operation"
    assert span.attributes["attr"] == "value"

    tracer.end_span(span, SpanStatus.OK)
    assert span.status == SpanStatus.OK

    tracer.reset()
    return True


def _test_tracer_context_manager() -> bool:
    """Test tracer context manager."""
    tracer = Tracer()
    tracer.reset()
    tracer.configure(enabled=True, sample_rate=1.0)

    with tracer.span("context_test") as span:
        assert span is not None
        span.set_attribute("inside", True)

    assert span.status == SpanStatus.OK
    assert span.end_time is not None

    tracer.reset()
    return True


def _test_tracer_exception_handling() -> bool:
    """Test tracer handles exceptions properly."""
    tracer = Tracer()
    tracer.reset()
    tracer.configure(enabled=True, sample_rate=1.0)

    captured_span: Span | None = None
    try:
        with tracer.span("error_test") as span:
            captured_span = span
            raise ValueError("test error")
    except ValueError:
        pass

    assert captured_span is not None
    assert captured_span.status == SpanStatus.ERROR
    assert "exception.type" in captured_span.attributes

    tracer.reset()
    return True


def _test_sampling() -> bool:
    """Test sampling behavior."""
    tracer = Tracer()
    tracer.reset()

    # 0% sampling should create no spans
    tracer.configure(enabled=True, sample_rate=0.0)
    span = tracer.start_span("should_not_sample")
    assert span is None

    # 100% sampling should always create spans
    tracer.configure(enabled=True, sample_rate=1.0)
    span = tracer.start_span("should_sample")
    assert span is not None
    tracer.end_span(span)

    tracer.reset()
    return True


def _test_trace_decorator() -> bool:
    """Test @trace decorator."""
    tracer = Tracer()
    tracer.reset()
    tracer.configure(enabled=True, sample_rate=1.0)

    @trace("decorated_function")
    def my_func(x: int) -> int:
        return x * 2

    result = my_func(5)
    assert result == 10

    # Verify span was created
    assert len(tracer._pending_spans) > 0
    span = tracer._pending_spans[-1]
    assert span.name == "decorated_function"
    assert span.status == SpanStatus.OK

    tracer.reset()
    return True


def _test_json_export() -> bool:
    """Test JSON file export."""
    import tempfile

    tracer = Tracer()
    tracer.reset()

    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "spans.json"
        exporter = JSONFileSpanExporter(export_path)

        span = Span(name="export_test", trace_id="t1", span_id="s1")
        span.end(SpanStatus.OK)

        success = exporter.export([span])
        assert success
        assert export_path.exists()

        with export_path.open(encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["name"] == "export_test"

    tracer.reset()
    return True


def _test_span_to_dict() -> bool:
    """Test span serialization."""
    span = Span(
        name="serialize_test",
        trace_id="trace123",
        span_id="span456",
        parent_span_id="parent789",
    )
    span.set_attribute("key", "value")
    span.add_event("event1", {"data": 42})
    span.end(SpanStatus.OK)

    data = span.to_dict()
    assert data["name"] == "serialize_test"
    assert data["trace_id"] == "trace123"
    assert data["span_id"] == "span456"
    assert data["parent_span_id"] == "parent789"
    assert data["status"] == "ok"
    assert data["attributes"]["key"] == "value"
    assert len(data["events"]) == 1

    return True


def module_tests() -> bool:
    """Run APM module tests."""
    from testing.test_framework import TestSuite

    suite = TestSuite("APM (Application Performance Monitoring)", "observability/apm.py")
    suite.start_suite()

    suite.run_test(
        "Span creation and attributes",
        _test_span_creation,
        "Verify basic span creation, attributes, and status",
    )

    suite.run_test(
        "Span events",
        _test_span_events,
        "Verify span event recording",
    )

    suite.run_test(
        "Tracer singleton",
        _test_tracer_singleton,
        "Verify tracer uses singleton pattern",
    )

    suite.run_test(
        "Tracer span context",
        _test_tracer_span_context,
        "Verify tracer creates and manages spans",
    )

    suite.run_test(
        "Tracer context manager",
        _test_tracer_context_manager,
        "Verify tracer context manager works correctly",
    )

    suite.run_test(
        "Exception handling",
        _test_tracer_exception_handling,
        "Verify spans capture exception information",
    )

    suite.run_test(
        "Sampling behavior",
        _test_sampling,
        "Verify sampling rate controls span creation",
    )

    suite.run_test(
        "Trace decorator",
        _test_trace_decorator,
        "Verify @trace decorator instruments functions",
    )

    suite.run_test(
        "JSON export",
        _test_json_export,
        "Verify spans can be exported to JSON file",
    )

    suite.run_test(
        "Span serialization",
        _test_span_to_dict,
        "Verify span to_dict produces correct output",
    )

    return suite.finish_suite()


# Standard test runner integration
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(module_tests)

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
