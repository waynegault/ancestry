"""Prometheus metrics exporter lifecycle management."""

from __future__ import annotations

import socket
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import requests

from config.config_schema import ObservabilityConfig
from standard_imports import setup_module
from test_framework import TestSuite, suppress_logging

try:  # pragma: no cover - import guard
    import prometheus_client as _prometheus_client  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - handled gracefully
    _prometheus_client = None
    start_http_server = cast(Optional[type], None)
    PROMETHEUS_SERVER_AVAILABLE = False
    _IMPORT_ERROR = exc
else:
    start_http_server = _prometheus_client.start_http_server
    PROMETHEUS_SERVER_AVAILABLE = True
    _IMPORT_ERROR = None

if TYPE_CHECKING:  # pragma: no cover - typing hints only
    from wsgiref.simple_server import WSGIServer
else:  # pragma: no cover - runtime fallback
    WSGIServer = object

from observability.metrics_registry import (
    PROMETHEUS_AVAILABLE,
    configure_metrics,
    get_metrics_registry,
    is_metrics_enabled,
    reset_metrics,
)

logger = setup_module(globals(), __name__)

_EXPORTER_LOCK = threading.RLock()


class _ExporterState:
    """Mutable container for exporter server state."""

    __slots__ = ("address", "server")

    def __init__(self) -> None:
        self.server: Optional[WSGIServer] = None
        self.address: Optional[tuple[str, int]] = None


_EXPORTER_STATE = _ExporterState()


def start_metrics_exporter(host: str, port: int) -> bool:
    """Start the Prometheus metrics exporter if not already running."""
    if not PROMETHEUS_AVAILABLE or not PROMETHEUS_SERVER_AVAILABLE:
        logger.debug("Prometheus client unavailable; exporter not started")
        return False

    if not is_metrics_enabled():
        logger.debug("Metrics are disabled; exporter start skipped")
        return False

    registry = get_metrics_registry()
    if registry is None:
        logger.debug("No CollectorRegistry available; exporter start skipped")
        return False

    with _EXPORTER_LOCK:
        if _EXPORTER_STATE.server is not None:
            return True

        try:
            server_factory = start_http_server
            assert server_factory is not None  # For type checkers
            server = server_factory(port, addr=host, registry=registry)
        except OSError as exc:
            logger.error(
                "Failed to start Prometheus exporter on %s:%s (%s)", host, port, exc
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Unexpected error starting Prometheus exporter: %s", exc, exc_info=True)
        else:
            bound_port = getattr(server, "server_port", port)
            _EXPORTER_STATE.server = server
            _EXPORTER_STATE.address = (host, bound_port)
            logger.info("Prometheus metrics exporter listening on %s:%s", host, bound_port)
            return True

    return False


def stop_metrics_exporter() -> None:
    """Stop the Prometheus metrics exporter if running."""
    with _EXPORTER_LOCK:
        server = _EXPORTER_STATE.server
        if server is None:
            return

        try:
            shutdown = getattr(server, "shutdown", None)
            if callable(shutdown):
                shutdown()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Error shutting down Prometheus exporter: %s", exc, exc_info=True)
        finally:
            _EXPORTER_STATE.server = None
            _EXPORTER_STATE.address = None
            logger.info("Prometheus metrics exporter stopped")


def is_metrics_exporter_running() -> bool:
    """Return True when the exporter server is active."""
    with _EXPORTER_LOCK:
        return _EXPORTER_STATE.server is not None


def get_metrics_exporter_address() -> Optional[tuple[str, int]]:
    """Return the exporter bind address when running."""
    with _EXPORTER_LOCK:
        return _EXPORTER_STATE.address


def _find_free_port(host: str) -> tuple[str, int]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        address = sock.getsockname()
    return address[0], address[1]


def _configure_observability(enabled: bool, host: str, port: int) -> None:
    reset_metrics()
    configure_metrics(
        ObservabilityConfig(
            enable_prometheus_metrics=enabled,
            metrics_export_host=host,
            metrics_export_port=port,
            metrics_namespace="test_exporter",
        )
    )


def test_exporter_noop_when_metrics_disabled() -> None:
    host, port = _find_free_port("127.0.0.1")
    _configure_observability(False, host, port)

    assert start_metrics_exporter(host, port) is False
    assert not is_metrics_exporter_running()
    assert get_metrics_exporter_address() is None

    stop_metrics_exporter()


def test_exporter_lifecycle() -> None:
    if not (PROMETHEUS_AVAILABLE and PROMETHEUS_SERVER_AVAILABLE):
        return

    host, port = _find_free_port("127.0.0.1")
    _configure_observability(True, host, port)

    with suppress_logging():
        started = start_metrics_exporter(host, port)

    try:
        assert started is True
        assert is_metrics_exporter_running()

        address = get_metrics_exporter_address()
        assert address is not None
        bound_host, bound_port = address

        response = requests.get(f"http://{bound_host}:{bound_port}/metrics", timeout=2)
        assert response.status_code == 200
        assert "test_exporter_session_uptime_seconds" in response.text

        assert start_metrics_exporter(host, port) is True
    finally:
        stop_metrics_exporter()
        assert not is_metrics_exporter_running()
        assert get_metrics_exporter_address() is None


def run_comprehensive_tests() -> bool:
    suite = TestSuite("Metrics Exporter Tests", "observability/metrics_exporter.py")
    suite.start_suite()
    suite.run_test(
        "Exporter disabled is no-op",
        test_exporter_noop_when_metrics_disabled,
        "Exporter should not start when metrics are disabled",
    )
    suite.run_test(
        "Exporter lifecycle",
        test_exporter_lifecycle,
        "Exporter should start, expose metrics, and stop cleanly",
    )
    return suite.finish_suite()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)


__all__ = [
    "get_metrics_exporter_address",
    "is_metrics_exporter_running",
    "start_metrics_exporter",
    "stop_metrics_exporter",
]
