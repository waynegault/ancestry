from __future__ import annotations

import logging
import logging.handlers
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

# Handle standalone execution for testing
if __package__ in {None, ""}:
    parent_dir = str(Path(__file__).resolve().parent.parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from performance.health_monitor import get_health_monitor

if TYPE_CHECKING:
    from core.session_manager import SessionManager

logger = logging.getLogger(__name__)

_api_performance_callbacks: list[Callable[[str, float, str], None]] = []
_HEALTH_MONITOR_EXCLUDED_APIS: set[str] = {"batch_processing"}


def register_api_metrics_callback(callback: Callable[[str, float, str], None]) -> None:
    """Register a callback invoked whenever API performance metrics are logged."""
    _api_performance_callbacks.append(callback)


def _record_health_monitor_metrics(duration: float, api_name: str, response_status: str) -> None:
    """Send timing data to the health monitor while filtering synthetic batch metrics."""
    if api_name in _HEALTH_MONITOR_EXCLUDED_APIS:
        return

    try:
        monitor = get_health_monitor()
        monitor.record_api_response_time(duration)
        if response_status.lower().startswith("error"):
            monitor.record_error(f"{api_name}_{response_status}")
    except Exception as monitor_exc:  # pragma: no cover - diagnostics only
        logger.debug(f"Health monitor recording failed for {api_name}: {monitor_exc}")


def _notify_api_callbacks(api_name: str, duration: float, response_status: str) -> None:
    """Invoke registered API metrics callbacks."""
    for callback in list(_api_performance_callbacks):
        try:
            callback(api_name, duration, response_status)
        except Exception as callback_exc:  # pragma: no cover - diagnostics only
            logger.debug(f"API metrics callback error for {api_name}: {callback_exc}")


def _log_api_duration_message(api_name: str, duration: float) -> None:
    """Emit context-aware log messages for slow calls."""
    if api_name == "batch_processing":
        if duration >= 180.0:
            logger.warning("⚠️  Batch processing window exceeded 180s (%.3fs)", duration)
        elif duration >= 90.0:
            logger.info("⚠️  Batch processing window took %.3fs", duration)
        else:
            logger.debug("Batch processing window took %.3fs", duration)
        return

    if duration > 10.0:
        logger.info(f"⚠️  Extended API call {api_name} took {duration:.3f}s (monitoring)")
    elif duration > 5.0:
        logger.info(f"Slow API call: {api_name} took {duration:.3f}s")
    elif duration > 2.0:
        logger.debug(f"Moderate API call: {api_name} took {duration:.3f}s")


def _track_api_metrics(api_name: str, duration: float, response_status: str) -> None:
    """Forward metrics to optional performance monitor."""
    try:
        from performance.performance_monitor import track_api_performance

        track_api_performance(api_name, duration, response_status)
    except ImportError:
        pass  # Graceful degradation if performance monitor not available


def log_api_performance(
    api_name: str,
    start_time: float,
    response_status: str = "unknown",
    session_manager: SessionManager | None = None,
) -> None:
    """Log API performance metrics for monitoring and optimization."""
    duration = time.time() - start_time
    logger.debug(f"API Performance: {api_name} took {duration:.3f}s (status: {response_status})")

    response_status = str(response_status or "unknown")
    _record_health_monitor_metrics(duration, api_name, response_status)
    _notify_api_callbacks(api_name, duration, response_status)

    if session_manager:
        _update_session_performance_tracking(session_manager, duration, response_status)

    _log_api_duration_message(api_name, duration)
    _track_api_metrics(api_name, duration, response_status)


def _update_session_performance_tracking(
    session_manager: SessionManager | None,
    duration: float,
    _response_status: str,
) -> None:
    """Update session manager with performance tracking data."""

    if session_manager is None:
        return

    try:
        # Use public API for performance tracking
        if hasattr(session_manager, "update_response_time_tracking"):
            session_manager._update_response_time_tracking(duration, slow_threshold=5.0)
        elif hasattr(session_manager, "reset_response_time_tracking"):
            # Fallback: initialize if needed then update
            session_manager._reset_response_time_tracking()
            session_manager._update_response_time_tracking(duration, slow_threshold=5.0)

    except Exception as exc:  # pragma: no cover - defensive telemetry
        logger.debug(f"Failed to update session performance tracking: {exc}")


# === Module Tests ===
def _test_log_api_performance() -> None:
    """Test the log_api_performance function logs with correct format."""
    import logging as _logging

    class _ListHandler(_logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[_logging.LogRecord] = []

        def emit(self, record: _logging.LogRecord) -> None:
            self.records.append(record)

    handler = _ListHandler()
    handler.setLevel(_logging.DEBUG)
    logger.addHandler(handler)
    original_level = logger.level
    logger.setLevel(_logging.DEBUG)
    try:
        start_time = time.time() - 1.5  # Simulate 1.5 second duration
        log_api_performance("test_api", start_time, "success")
        # Verify the log contains the API name and duration info
        log_messages = [r.getMessage() for r in handler.records]
        api_logged = any("test_api" in msg and "1.5" in msg for msg in log_messages)
        assert api_logged, f"Expected log with 'test_api' and duration ~1.5s, got: {log_messages}"
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


def _test_register_callback() -> None:
    """Test callback registration."""
    callback_called: list[tuple[str, float, str]] = []

    def test_callback(api_name: str, duration: float, status: str) -> None:
        callback_called.append((api_name, duration, status))

    register_api_metrics_callback(test_callback)
    start_time = time.time()
    log_api_performance("test_callback_api", start_time, "ok")
    assert len(callback_called) > 0, "Callback should have been invoked"


def _test_duration_messages() -> None:
    """Test duration message logging at various thresholds."""
    import logging as _logging

    class _ListHandler(_logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[_logging.LogRecord] = []

        def emit(self, record: _logging.LogRecord) -> None:
            self.records.append(record)

    handler = _ListHandler()
    handler.setLevel(_logging.DEBUG)
    logger.addHandler(handler)
    original_level = logger.level
    logger.setLevel(_logging.DEBUG)
    try:
        _log_api_duration_message("fast_api", 0.5)  # Below 2s threshold - no message
        _log_api_duration_message("moderate_api", 3.0)  # Moderate (>2s) - debug
        _log_api_duration_message("slow_api", 6.0)  # Slow (>5s) - info
        _log_api_duration_message("very_slow_api", 12.0)  # Extended (>10s) - info

        messages = [record.getMessage() for record in handler.records]
        # Moderate call should produce "Moderate API call" message
        assert any("Moderate" in m and "moderate_api" in m for m in messages), (
            f"Expected 'Moderate API call' for 3.0s duration, got: {messages}"
        )
        # Slow call should produce "Slow API call" message
        assert any("Slow" in m and "slow_api" in m for m in messages), (
            f"Expected 'Slow API call' for 6.0s duration, got: {messages}"
        )
        # Very slow call should produce "Extended API call" message
        assert any("Extended" in m and "very_slow_api" in m for m in messages), (
            f"Expected 'Extended API call' for 12.0s duration, got: {messages}"
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


def _test_batch_processing_message() -> None:
    """Test batch processing duration messages at escalating levels."""
    import logging as _logging

    class _ListHandler(_logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[_logging.LogRecord] = []

        def emit(self, record: _logging.LogRecord) -> None:
            self.records.append(record)

    handler = _ListHandler()
    handler.setLevel(_logging.DEBUG)
    logger.addHandler(handler)
    original_level = logger.level
    logger.setLevel(_logging.DEBUG)
    try:
        _log_api_duration_message("batch_processing", 30.0)  # Normal (<90s) - debug
        _log_api_duration_message("batch_processing", 100.0)  # Elevated (>=90s) - info
        _log_api_duration_message("batch_processing", 200.0)  # High (>=180s) - warning

        messages = [(r.levelno, r.getMessage()) for r in handler.records]
        # Normal batch should be debug level
        normal_msgs = [(lvl, m) for lvl, m in messages if "30.000" in m]
        assert len(normal_msgs) > 0, f"Expected log for 30s batch, got: {messages}"
        assert normal_msgs[0][0] == _logging.DEBUG, f"30s batch should be DEBUG, got {normal_msgs[0][0]}"
        # Elevated batch (>=90s) should be info level
        elevated_msgs = [(lvl, m) for lvl, m in messages if "100.000" in m]
        assert len(elevated_msgs) > 0, f"Expected log for 100s batch, got: {messages}"
        assert elevated_msgs[0][0] == _logging.INFO, f"100s batch should be INFO, got {elevated_msgs[0][0]}"
        # High batch (>=180s) should be warning level
        high_msgs = [(lvl, m) for lvl, m in messages if "200.000" in m]
        assert len(high_msgs) > 0, f"Expected log for 200s batch, got: {messages}"
        assert high_msgs[0][0] == _logging.WARNING, f"200s batch should be WARNING, got {high_msgs[0][0]}"
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


def logging_module_tests() -> bool:
    """Run all gather logging module tests."""
    from testing.test_framework import TestSuite

    suite = TestSuite("Gather Logging", "actions/gather/logging.py")
    suite.start_suite()

    suite.run_test(
        test_name="Log API performance",
        test_func=_test_log_api_performance,
        test_summary="Test API performance logging function",
        functions_tested="log_api_performance()",
        expected_outcome="Metrics logged without errors",
    )

    suite.run_test(
        test_name="Callback registration",
        test_func=_test_register_callback,
        test_summary="Test API metrics callback registration",
        functions_tested="register_api_metrics_callback()",
        expected_outcome="Callbacks are invoked when metrics are logged",
    )

    suite.run_test(
        test_name="Duration messages",
        test_func=_test_duration_messages,
        test_summary="Test duration-based log message levels",
        functions_tested="_log_api_duration_message()",
        expected_outcome="Appropriate log levels for different durations",
    )

    suite.run_test(
        test_name="Batch processing messages",
        test_func=_test_batch_processing_message,
        test_summary="Test batch processing specific messages",
        functions_tested="_log_api_duration_message() with batch_processing",
        expected_outcome="Batch processing has special duration handling",
    )

    return suite.finish_suite()


run_comprehensive_tests = logging_module_tests

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
