#!/usr/bin/env python3

"""Prometheus metrics registry helpers with safe fallbacks when disabled."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from config.config_schema import ObservabilityConfig
from standard_imports import setup_module
from test_framework import TestSuite, suppress_logging

logger = setup_module(globals(), __name__)

try:  # pragma: no cover - import-time guard
    import prometheus_client as _prometheus_client  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - handled gracefully
    _prometheus_client = None
    PROMETHEUS_AVAILABLE = False
    _IMPORT_ERROR: Optional[Exception] = exc
else:
    PROMETHEUS_AVAILABLE = True
    _IMPORT_ERROR = None

if _prometheus_client is not None:  # pragma: no cover - runtime wiring
    CollectorRegistry = _prometheus_client.CollectorRegistry
    Counter = _prometheus_client.Counter
    Gauge = _prometheus_client.Gauge
    Histogram = _prometheus_client.Histogram
else:  # pragma: no cover - typed fallback when dependency missing
    CollectorRegistry = cast(Any, None)
    Counter = cast(Any, None)
    Gauge = cast(Any, None)
    Histogram = cast(Any, None)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prometheus_client import (  # type: ignore[import-not-found]
        CollectorRegistry as PrometheusCollectorRegistry,
        Counter as PrometheusCounter,
        Gauge as PrometheusGauge,
        Histogram as PrometheusHistogram,
    )
else:  # pragma: no cover - runtime alias
    PrometheusCollectorRegistry = Any
    PrometheusCounter = Any
    PrometheusGauge = Any
    PrometheusHistogram = Any


class _ApiLatencyProxy:
    """Wrapper for API latency histogram."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusHistogram] = None

    def set_metric(self, metric: Optional[PrometheusHistogram]) -> None:
        self._metric = metric

    def observe(self, endpoint: str, status_family: str, seconds: float) -> None:
        metric = self._metric
        if metric is None:
            return
        seconds = max(seconds, 0.0)
        metric.labels(endpoint=endpoint, status_family=status_family).observe(seconds)


class _ApiRequestCounterProxy:
    """Wrapper for API request counter."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusCounter] = None

    def set_metric(self, metric: Optional[PrometheusCounter]) -> None:
        self._metric = metric

    def inc(self, endpoint: str, method: str, result: str, amount: float = 1.0) -> None:
        metric = self._metric
        if metric is None:
            return
        metric.labels(endpoint=endpoint, method=method, result=result).inc(amount)


class _CacheHitRatioGaugeProxy:
    """Wrapper for cache hit ratio gauge."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusGauge] = None

    def set_metric(self, metric: Optional[PrometheusGauge]) -> None:
        self._metric = metric

    def set(self, service: str, endpoint: str, ratio: float) -> None:
        metric = self._metric
        if metric is None:
            return
        clamped = min(max(ratio, 0.0), 1.0)
        metric.labels(service=service, endpoint=endpoint).set(clamped)


class _CacheOperationsCounterProxy:
    """Wrapper for cache operations counter."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusCounter] = None

    def set_metric(self, metric: Optional[PrometheusCounter]) -> None:
        self._metric = metric

    def inc(self, service: str, endpoint: str, operation: str, amount: float = 1.0) -> None:
        metric = self._metric
        if metric is None:
            return
        metric.labels(service=service, endpoint=endpoint, operation=operation).inc(amount)


class _SessionUptimeGaugeProxy:
    """Wrapper for session uptime gauge."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusGauge] = None

    def set_metric(self, metric: Optional[PrometheusGauge]) -> None:
        self._metric = metric

    def set(self, seconds: float) -> None:
        metric = self._metric
        if metric is None:
            return
        metric.set(max(seconds, 0.0))


class _SessionRefreshCounterProxy:
    """Wrapper for session refresh counter."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusCounter] = None

    def set_metric(self, metric: Optional[PrometheusCounter]) -> None:
        self._metric = metric

    def inc(self, reason: str, amount: float = 1.0) -> None:
        metric = self._metric
        if metric is None:
            return
        metric.labels(reason=reason).inc(amount)


class _ActionProcessedCounterProxy:
    """Wrapper for action throughput counter."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusCounter] = None

    def set_metric(self, metric: Optional[PrometheusCounter]) -> None:
        self._metric = metric

    def inc(self, action: str, result: str, amount: float = 1.0) -> None:
        metric = self._metric
        if metric is None:
            return
        metric.labels(action=action, result=result).inc(amount)


class _CircuitBreakerStateGaugeProxy:
    """Wrapper for circuit breaker state gauge."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusGauge] = None

    def set_metric(self, metric: Optional[PrometheusGauge]) -> None:
        self._metric = metric

    def set(self, breaker: str, state: float) -> None:
        metric = self._metric
        if metric is None:
            return
        metric.labels(breaker=breaker).set(state)


class _CircuitBreakerTripCounterProxy:
    """Wrapper for circuit breaker trip counter."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusCounter] = None

    def set_metric(self, metric: Optional[PrometheusCounter]) -> None:
        self._metric = metric

    def inc(self, breaker: str, amount: float = 1.0) -> None:
        metric = self._metric
        if metric is None:
            return
        metric.labels(breaker=breaker).inc(amount)


class _RateLimiterDelayHistogramProxy:
    """Wrapper for rate limiter delay histogram."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusHistogram] = None

    def set_metric(self, metric: Optional[PrometheusHistogram]) -> None:
        self._metric = metric

    def observe(self, seconds: float) -> None:
        metric = self._metric
        if metric is None:
            return
        metric.observe(max(seconds, 0.0))


class _WorkerThreadGaugeProxy:
    """Wrapper for worker thread count gauge."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusGauge] = None

    def set_metric(self, metric: Optional[PrometheusGauge]) -> None:
        self._metric = metric

    def set(self, count: float) -> None:
        metric = self._metric
        if metric is None:
            return
        metric.set(max(count, 0.0))


class MetricsBundle:
    """Container exposing all metric proxies."""

    def __init__(self) -> None:
        self.api_latency = _ApiLatencyProxy()
        self.api_requests = _ApiRequestCounterProxy()
        self.cache_hit_ratio = _CacheHitRatioGaugeProxy()
        self.cache_operations = _CacheOperationsCounterProxy()
        self.session_uptime = _SessionUptimeGaugeProxy()
        self.session_refresh = _SessionRefreshCounterProxy()
        self.action_processed = _ActionProcessedCounterProxy()
        self.circuit_breaker_state = _CircuitBreakerStateGaugeProxy()
        self.circuit_breaker_trips = _CircuitBreakerTripCounterProxy()
        self.rate_limiter_delay = _RateLimiterDelayHistogramProxy()
        self.worker_thread_count = _WorkerThreadGaugeProxy()

    def assign(self, metrics_map: dict[str, Any]) -> None:
        """Bind proxies to real metrics."""
        self.api_latency.set_metric(metrics_map.get("api_latency"))
        self.api_requests.set_metric(metrics_map.get("api_requests"))
        self.cache_hit_ratio.set_metric(metrics_map.get("cache_hit_ratio"))
        self.cache_operations.set_metric(metrics_map.get("cache_operations"))
        self.session_uptime.set_metric(metrics_map.get("session_uptime"))
        self.session_refresh.set_metric(metrics_map.get("session_refresh"))
        self.action_processed.set_metric(metrics_map.get("action_processed"))
        self.circuit_breaker_state.set_metric(metrics_map.get("circuit_breaker_state"))
        self.circuit_breaker_trips.set_metric(metrics_map.get("circuit_breaker_trips"))
        self.rate_limiter_delay.set_metric(metrics_map.get("rate_limiter_delay"))
        self.worker_thread_count.set_metric(metrics_map.get("worker_thread_count"))

    def reset(self) -> None:
        """Clear metric bindings (no-op proxies)."""
        self.assign({})


class MetricsRegistry:
    """Central Prometheus registry manager with enable/disable support."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._enabled = False
        self._namespace = "ancestry"
        self._registry: Optional[PrometheusCollectorRegistry] = None
        self._metrics = MetricsBundle()
        self._import_logged = False

    def configure(self, settings: Optional[ObservabilityConfig]) -> None:
        """Configure metrics using Observability settings."""
        with self._lock:
            if settings is None or not settings.enable_prometheus_metrics:
                if self._enabled:
                    logger.info("Prometheus metrics disabled by configuration")
                self._disable_locked()
                return

            if not PROMETHEUS_AVAILABLE:
                if not self._import_logged:
                    logger.warning(
                        "Prometheus client unavailable; metrics disabled (%s)",
                        _IMPORT_ERROR,
                    )
                    self._import_logged = True
                self._disable_locked()
                return

            namespace = settings.metrics_namespace or "ancestry"
            if self._enabled and self._registry is not None and self._namespace == namespace:
                return

            registry = CollectorRegistry(auto_describe=True)
            metrics_map = self._create_metrics(namespace, registry)
            self._registry = registry
            self._namespace = namespace
            self._metrics.assign(metrics_map)
            self._enabled = True
            logger.info("Prometheus metrics enabled (namespace=%s)", namespace)

    def reset(self) -> None:
        """Disable metrics and clear existing registry."""
        with self._lock:
            self._disable_locked()

    def _disable_locked(self) -> None:
        self._enabled = False
        self._registry = None
        self._metrics.reset()

    def _create_metrics(
        self,
        namespace: str,
        registry: PrometheusCollectorRegistry,
    ) -> dict[str, Any]:
        """Create Prometheus metrics in the provided registry."""
        metrics_map: dict[str, Any] = {}

        metrics_map["api_latency"] = Histogram(
            "api_latency_seconds",
            "API latency per endpoint",
            labelnames=("endpoint", "status_family"),
            namespace=namespace,
            buckets=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
            registry=registry,
        )

        metrics_map["api_requests"] = Counter(
            "api_requests_total",
            "API request counter",
            labelnames=("endpoint", "method", "result"),
            namespace=namespace,
            registry=registry,
        )

        metrics_map["cache_hit_ratio"] = Gauge(
            "cache_hit_ratio",
            "Cache hit ratio by service and endpoint",
            labelnames=("service", "endpoint"),
            namespace=namespace,
            registry=registry,
        )

        metrics_map["cache_operations"] = Counter(
            "cache_operations_total",
            "Cache operations counter",
            labelnames=("service", "endpoint", "operation"),
            namespace=namespace,
            registry=registry,
        )

        metrics_map["session_uptime"] = Gauge(
            "session_uptime_seconds",
            "Active session uptime in seconds",
            namespace=namespace,
            registry=registry,
        )

        metrics_map["session_refresh"] = Counter(
            "session_refresh_total",
            "Session refresh counter",
            labelnames=("reason",),
            namespace=namespace,
            registry=registry,
        )

        metrics_map["action_processed"] = Counter(
            "action_processed_total",
            "Action throughput counter",
            labelnames=("action", "result"),
            namespace=namespace,
            registry=registry,
        )

        metrics_map["circuit_breaker_state"] = Gauge(
            "circuit_breaker_state",
            "Circuit breaker state gauge",
            labelnames=("breaker",),
            namespace=namespace,
            registry=registry,
        )

        metrics_map["circuit_breaker_trips"] = Counter(
            "circuit_breaker_trip_total",
            "Circuit breaker trip counter",
            labelnames=("breaker",),
            namespace=namespace,
            registry=registry,
        )

        metrics_map["rate_limiter_delay"] = Histogram(
            "rate_limiter_delay_seconds",
            "Rate limiter wait durations",
            namespace=namespace,
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
            registry=registry,
        )

        metrics_map["worker_thread_count"] = Gauge(
            "worker_thread_count",
            "Active worker thread count",
            namespace=namespace,
            registry=registry,
        )

        return metrics_map

    def get_registry(self) -> Optional[PrometheusCollectorRegistry]:
        """Return the active Prometheus registry (if enabled)."""
        return self._registry

    @property
    def metrics(self) -> MetricsBundle:
        return self._metrics

    def is_enabled(self) -> bool:
        return self._enabled


_METRICS_REGISTRY = MetricsRegistry()


def configure_metrics(settings: Optional[ObservabilityConfig]) -> None:
    """Configure global metrics using provided settings."""
    _METRICS_REGISTRY.configure(settings)


def disable_metrics() -> None:
    """Disable metrics emission."""
    _METRICS_REGISTRY.reset()


def reset_metrics() -> None:
    """Alias for disable_metrics to keep test naming explicit."""
    _METRICS_REGISTRY.reset()


def metrics() -> MetricsBundle:
    """Return the shared MetricsBundle (proxies are safe when disabled)."""
    return _METRICS_REGISTRY.metrics


def get_metrics_registry() -> Optional[PrometheusCollectorRegistry]:
    """Return the active Prometheus registry for exporter wiring."""
    return _METRICS_REGISTRY.get_registry()


def is_metrics_enabled() -> bool:
    """Return True when Prometheus metrics are currently enabled."""
    return _METRICS_REGISTRY.is_enabled()


def _make_enabled_settings(namespace: str = "test_observability") -> ObservabilityConfig:
    return ObservabilityConfig(
        enable_prometheus_metrics=True,
        metrics_namespace=namespace,
        metrics_export_port=9100,
    )


def test_metrics_disabled_is_noop() -> None:
    """Disabling metrics should make proxies safe no-ops."""
    reset_metrics()
    configure_metrics(ObservabilityConfig(enable_prometheus_metrics=False))

    assert not is_metrics_enabled(), "Metrics should be disabled"
    assert get_metrics_registry() is None, "No registry expected when disabled"

    bundle = metrics()
    bundle.api_requests.inc("combined_details", "GET", "success")
    bundle.api_latency.observe("combined_details", "2xx", 1.0)
    bundle.cache_hit_ratio.set("svc", "endpoint", 0.5)


def test_metrics_enabled_records_samples() -> None:
    """Enabling metrics should register samples in the CollectorRegistry."""
    if not PROMETHEUS_AVAILABLE:
        return

    reset_metrics()
    with suppress_logging():
        configure_metrics(_make_enabled_settings("test_metrics"))

    assert is_metrics_enabled(), "Metrics should be enabled"

    bundle = metrics()
    bundle.api_requests.inc("combined_details", "GET", "success")
    bundle.api_requests.inc("combined_details", "GET", "success")
    bundle.api_latency.observe("combined_details", "2xx", 1.25)
    bundle.cache_hit_ratio.set("ancestry_api", "combined_details", 0.75)
    bundle.cache_operations.inc("ancestry_api", "combined_details", "get")
    bundle.session_uptime.set(1234)
    bundle.session_refresh.inc("proactive")
    bundle.action_processed.inc("action6", "success", amount=2)
    bundle.circuit_breaker_state.set("session", 1.0)
    bundle.circuit_breaker_trips.inc("session")
    bundle.rate_limiter_delay.observe(0.4)
    bundle.worker_thread_count.set(7)

    registry = get_metrics_registry()
    assert registry is not None, "CollectorRegistry should be available"

    count_value = registry.get_sample_value(
        "test_metrics_api_requests_total",
        labels={"endpoint": "combined_details", "method": "GET", "result": "success"},
    )
    assert count_value == 2.0

    latency_sum = registry.get_sample_value(
        "test_metrics_api_latency_seconds_sum",
        labels={"endpoint": "combined_details", "status_family": "2xx"},
    )
    assert latency_sum == 1.25

    cache_ratio = registry.get_sample_value(
        "test_metrics_cache_hit_ratio",
        labels={"service": "ancestry_api", "endpoint": "combined_details"},
    )
    assert cache_ratio == 0.75

    session_uptime = registry.get_sample_value(
        "test_metrics_session_uptime_seconds",
        labels={},
    )
    assert session_uptime == 1234.0

    refresh_total = registry.get_sample_value(
        "test_metrics_session_refresh_total",
        labels={"reason": "proactive"},
    )
    assert refresh_total == 1.0

    breaker_state = registry.get_sample_value(
        "test_metrics_circuit_breaker_state",
        labels={"breaker": "session"},
    )
    assert breaker_state == 1.0

    rate_delay_sum = registry.get_sample_value(
        "test_metrics_rate_limiter_delay_seconds_sum",
        labels={},
    )
    assert rate_delay_sum == 0.4

    disable_metrics()
    assert not is_metrics_enabled()


def test_disable_clears_registry() -> None:
    """Disabling metrics should clear the registry reference."""
    if not PROMETHEUS_AVAILABLE:
        return

    reset_metrics()
    with suppress_logging():
        configure_metrics(_make_enabled_settings("test_disable"))

    assert get_metrics_registry() is not None
    disable_metrics()
    assert get_metrics_registry() is None
    assert not is_metrics_enabled()


def observability_metrics_registry_module_tests() -> bool:
    suite = TestSuite("Metrics Registry Tests", "observability/metrics_registry.py")
    suite.start_suite()
    suite.run_test(
        "Metrics disabled noop",
        test_metrics_disabled_is_noop,
        "Disabled metrics should act as no-ops",
    )
    suite.run_test(
        "Metrics enabled records samples",
        test_metrics_enabled_records_samples,
        "Enabled metrics should push values into CollectorRegistry",
    )
    suite.run_test(
        "Disable resets registry",
        test_disable_clears_registry,
        "Disabling metrics should clear state",
    )
    return suite.finish_suite()


# Use centralized test runner utility from test_utilities
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(observability_metrics_registry_module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "MetricsBundle",
    "MetricsRegistry",
    "configure_metrics",
    "disable_metrics",
    "get_metrics_registry",
    "is_metrics_enabled",
    "metrics",
    "reset_metrics",
]
