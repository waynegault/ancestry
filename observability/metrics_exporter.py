"""Prometheus metrics exporter lifecycle management."""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import requests

try:  # pragma: no cover - optional dependency
    from config.config_schema import ObservabilityConfig
except Exception:  # pragma: no cover - fallback for environments missing config deps
    if TYPE_CHECKING:
        from config.config_schema import ObservabilityConfig as ObservabilityConfig  # type: ignore
    else:

        @dataclass
        class ObservabilityConfig:  # type: ignore[override]
            """Fallback configuration used when full schema cannot be imported."""

            enable_prometheus_metrics: bool = False
            metrics_export_host: str = "127.0.0.1"
            metrics_export_port: int = 9000
            metrics_namespace: str = "ancestry"

            def __post_init__(self) -> None:
                if not self.metrics_export_host:
                    raise ValueError("metrics_export_host must be non-empty")
                if self.metrics_export_port <= 0 or self.metrics_export_port > 65535:
                    raise ValueError("metrics_export_port must be between 1 and 65535")
                if not self.metrics_namespace:
                    raise ValueError("metrics_namespace must be non-empty")
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

def _serve_metrics_endpoint(host: str, port: int, namespace: str) -> int:
    if not (PROMETHEUS_AVAILABLE and PROMETHEUS_SERVER_AVAILABLE):
        logger.error(
            "Prometheus client library is unavailable; install 'prometheus-client' to serve metrics",
        )
        return 1

    reset_metrics()
    configure_metrics(
        ObservabilityConfig(
            enable_prometheus_metrics=True,
            metrics_export_host=host,
            metrics_export_port=port,
            metrics_namespace=namespace,
        )
    )

    if not start_metrics_exporter(host, port):
        logger.error("Unable to start Prometheus exporter on %s:%s", host, port)
        return 1

    bound_address = get_metrics_exporter_address()
    display_host, display_port = bound_address if bound_address else (host, port)
    logger.info(
        "Serving Prometheus metrics at http://%s:%s/metrics – press Ctrl+C to stop",
        display_host,
        display_port,
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Metrics exporter interrupted; shutting down")
    finally:
        stop_metrics_exporter()

    return 0


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
    parser = argparse.ArgumentParser(description="Metrics exporter utility")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start a standalone Prometheus exporter instead of running tests",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the Prometheus exporter (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Port for the Prometheus exporter (default: 9000)",
    )
    parser.add_argument(
        "--namespace",
        default="ancestry",
        help="Metrics namespace to register when serving (default: ancestry)",
    )
    args = parser.parse_args()

    if args.serve:
        exit_code = _serve_metrics_endpoint(args.host, args.port, args.namespace)
        sys.exit(exit_code)

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)


__all__ = [
    "get_metrics_exporter_address",
    "is_metrics_exporter_running",
    "start_metrics_exporter",
    "stop_metrics_exporter",
]
