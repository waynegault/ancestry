from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable, Optional

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
    session_manager: Optional[SessionManager] = None,
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
    session_manager: Optional[SessionManager],
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
