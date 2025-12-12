"""Prometheus metrics exporter lifecycle management."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

# Allow running this module directly (python observability/metrics_exporter.py)
# by ensuring the repository root is on sys.path.
if __package__ in {None, ""}:
    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import requests

if TYPE_CHECKING:
    from config.config_schema import ObservabilityConfig

logger = logging.getLogger(__name__)

from testing.test_framework import TestSuite, suppress_logging


@dataclass
class _PrometheusClientState:
    server_available: bool = False
    import_error: Optional[Exception] = None
    start_http_server: Callable[..., Any] | None = None


_PROM_CLIENT_STATE = _PrometheusClientState()

try:  # pragma: no cover - import guard
    import prometheus_client as _prometheus_client
except Exception as exc:  # pragma: no cover - handled gracefully
    _PROM_CLIENT_STATE.import_error = exc
else:
    _PROM_CLIENT_STATE.server_available = True
    _PROM_CLIENT_STATE.start_http_server = cast(Any, _prometheus_client).start_http_server

PROMETHEUS_SERVER_AVAILABLE = _PROM_CLIENT_STATE.server_available
start_http_server: Callable[..., Any] | None = _PROM_CLIENT_STATE.start_http_server
_IMPORT_ERROR = _PROM_CLIENT_STATE.import_error

if TYPE_CHECKING:  # pragma: no cover - typing hints only
    from wsgiref.simple_server import WSGIServer
else:  # pragma: no cover - runtime fallback
    WSGIServer = object

import observability.metrics_registry as _metrics_registry

_prometheus_available = _metrics_registry.PROMETHEUS_AVAILABLE
configure_metrics = _metrics_registry.configure_metrics
get_metrics_registry = _metrics_registry.get_metrics_registry
is_metrics_enabled = _metrics_registry.is_metrics_enabled
reset_metrics = _metrics_registry.reset_metrics


PROMETHEUS_AVAILABLE = _prometheus_available

# logger = setup_module(globals(), __name__)

_EXPORTER_LOCK = threading.RLock()


class ObservabilityState:
    _runtime_observability: Optional[ObservabilityConfig] = None


_DEFAULT_PROMETHEUS_BINARY = Path("C:/Programs/Prometheus/prometheus.exe")


class _ExporterState:
    """Mutable container for exporter server state."""

    __slots__ = ("address", "server")

    def __init__(self) -> None:
        self.server: WSGIServer | None = None
        self.address: tuple[str, int] | None = None


_EXPORTER_STATE = _ExporterState()

# Prometheus server process management


@dataclass
class _PrometheusProcessState:
    process: subprocess.Popen[bytes] | None = None


_PROMETHEUS_PROCESS_STATE = _PrometheusProcessState()
_PROMETHEUS_LOCK = threading.RLock()


def _start_prometheus_server() -> bool:
    """Start Prometheus server if available and not running.

    Returns:
        True if Prometheus started or already running, False otherwise
    """
    with _PROMETHEUS_LOCK:
        state = _PROMETHEUS_PROCESS_STATE
        # Check if already running
        if state.process is not None:
            if state.process.poll() is None:
                return True  # Already running
            state.process = None

        settings = ObservabilityState._runtime_observability
        auto_start_enabled = True if settings is None else bool(settings.auto_start_prometheus)
        if not auto_start_enabled:
            logger.debug("Prometheus auto-start disabled via configuration")
            return False

        binary_override = getattr(settings, "prometheus_binary_path", None) if settings else None
        prometheus_path = Path(binary_override) if binary_override else _DEFAULT_PROMETHEUS_BINARY

        # Check if Prometheus is available
        if not prometheus_path.exists():
            logger.debug("Prometheus not found at %s, skipping auto-start", prometheus_path)
            return False

        config_path = prometheus_path.parent / "prometheus.yml"
        if not config_path.exists():
            logger.debug("Prometheus config not found at %s", config_path)
            return False

        try:
            # Start Prometheus in background
            process = subprocess.Popen(
                [str(prometheus_path), f"--config.file={config_path}"],
                cwd=str(prometheus_path.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
            logger.info("✅ Prometheus server started on http://localhost:9090")
            # Give it a moment to start
            time.sleep(2)
            state.process = process
            return True
        except Exception as e:
            logger.warning("Failed to start Prometheus server: %s", e)
            return False


def _stop_prometheus_server() -> None:
    """Stop Prometheus server if running."""
    with _PROMETHEUS_LOCK:
        state = _PROMETHEUS_PROCESS_STATE
        if state.process is not None:
            try:
                state.process.terminate()
                state.process.wait(timeout=5)
                logger.info("Prometheus server stopped")
            except Exception as e:
                logger.debug("Error stopping Prometheus: %s", e)
            finally:
                state.process = None


def get_exporter_status() -> dict[str, Any] | None:
    """Get current exporter status.

    Returns:
        Dict with host, port if running, None otherwise
    """
    with _EXPORTER_LOCK:
        if _EXPORTER_STATE.server and _EXPORTER_STATE.address:
            return {'host': _EXPORTER_STATE.address[0], 'port': _EXPORTER_STATE.address[1], 'running': True}
        return None


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
            server = cast(WSGIServer, server_factory(port, addr=host, registry=registry))
        except OSError as exc:
            logger.error("Failed to start Prometheus exporter on %s:%s (%s)", host, port, exc)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Unexpected error starting Prometheus exporter: %s", exc, exc_info=True)
        else:
            bound_port = getattr(server, "server_port", port)
            _EXPORTER_STATE.server = server
            _EXPORTER_STATE.address = (host, bound_port)
            logger.info("✅ Prometheus metrics exporter listening on %s:%s", host, bound_port)

            # Start Prometheus server to scrape our metrics
            _start_prometheus_server()

            return True

    return False


def stop_metrics_exporter() -> None:
    """Stop the Prometheus metrics exporter if running."""
    # Stop Prometheus server first to prevent scraping during shutdown
    _stop_prometheus_server()

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


def get_metrics_exporter_address() -> tuple[str, int] | None:
    """Return the exporter bind address when running."""
    with _EXPORTER_LOCK:
        return _EXPORTER_STATE.address


def _find_free_port(host: str) -> tuple[str, int]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        address = sock.getsockname()
    return address[0], address[1]


def _configure_observability(enabled: bool, host: str, port: int) -> None:
    from config.config_schema import ObservabilityConfig

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

    from config.config_schema import ObservabilityConfig

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
        "Serving Prometheus metrics at http://%s:%s/metrics - press Ctrl+C to stop",
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


def observability_metrics_exporter_module_tests() -> bool:
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


# Use centralized test runner utility from test_utilities
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(observability_metrics_exporter_module_tests)


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
        default=9001,
        help="Port for the Prometheus exporter (default: 9001)",
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
    "apply_observability_settings",
    "get_metrics_exporter_address",
    "is_metrics_exporter_running",
    "start_metrics_exporter",
    "stop_metrics_exporter",
]


def apply_observability_settings(settings: Optional[ObservabilityConfig]) -> None:
    """Store the latest Observability configuration for exporter helpers."""

    ObservabilityState._runtime_observability = settings
