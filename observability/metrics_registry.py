#!/usr/bin/env python3

"""Prometheus metrics registry helpers with safe fallbacks when disabled."""

from __future__ import annotations

import logging
import sys
import threading
from typing import TYPE_CHECKING, Any, Optional, Protocol, cast

from testing.test_framework import TestSuite, suppress_logging

if TYPE_CHECKING:
    from config.config_schema import ObservabilityConfig

logger = logging.getLogger(__name__)

_prometheus_import_error: Optional[Exception] = None

try:  # pragma: no cover - import-time guard
    import prometheus_client as _prometheus_client
except Exception as exc:  # pragma: no cover - handled gracefully
    _prometheus_client = None
    _prometheus_import_error = exc

PROMETHEUS_AVAILABLE = _prometheus_client is not None
_IMPORT_ERROR: Optional[Exception] = _prometheus_import_error

if _prometheus_client is not None:  # pragma: no cover - runtime wiring
    _client_any = cast(Any, _prometheus_client)
    CollectorRegistry = _client_any.CollectorRegistry
    Counter = _client_any.Counter
    Gauge = _client_any.Gauge
    Histogram = _client_any.Histogram
else:  # pragma: no cover - typed fallback when dependency missing
    CollectorRegistry = cast(Any, None)
    Counter = cast(Any, None)
    Gauge = cast(Any, None)
    Histogram = cast(Any, None)


class _CounterMetric(Protocol):
    def labels(self, **labels: str) -> _CounterMetric:  # pragma: no cover - protocol only
        ...

    def inc(self, amount: float = 1.0) -> None:  # pragma: no cover - protocol only
        ...


class _GaugeMetric(Protocol):
    def labels(self, **labels: str) -> _GaugeMetric:  # pragma: no cover - protocol only
        ...

    def set(self, value: float) -> None:  # pragma: no cover - protocol only
        ...


class _HistogramMetric(Protocol):
    def labels(self, **labels: str) -> _HistogramMetric:  # pragma: no cover - protocol only
        ...

    def observe(self, value: float) -> None:  # pragma: no cover - protocol only
        ...


PrometheusCounter = _CounterMetric
PrometheusGauge = _GaugeMetric
PrometheusHistogram = _HistogramMetric


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


class _DatabaseQueryHistogramProxy:
    """Wrapper for database query duration histogram."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusHistogram] = None

    def set_metric(self, metric: Optional[PrometheusHistogram]) -> None:
        self._metric = metric

    def observe(self, operation: str, seconds: float) -> None:
        metric = self._metric
        if metric is None:
            return
        safe_operation = operation or "unknown"
        metric.labels(operation=safe_operation).observe(max(seconds, 0.0))


class _DatabaseRowsCounterProxy:
    """Wrapper for rows-affected counter."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusCounter] = None

    def set_metric(self, metric: Optional[PrometheusCounter]) -> None:
        self._metric = metric

    def inc(self, operation: str, amount: float) -> None:
        metric = self._metric
        if metric is None:
            return
        safe_operation = operation or "unknown"
        metric.labels(operation=safe_operation).inc(max(amount, 0.0))


class _ActionDurationHistogramProxy:
    """Wrapper for action duration histogram."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusHistogram] = None

    def set_metric(self, metric: Optional[PrometheusHistogram]) -> None:
        self._metric = metric

    def observe(self, action: str, seconds: float) -> None:
        metric = self._metric
        if metric is None:
            return
        metric.labels(action=action).observe(max(seconds, 0.0))


class _InternalMetricGaugeProxy:
    """Wrapper for internal collector metric gauge."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusGauge] = None

    def set_metric(self, metric: Optional[PrometheusGauge]) -> None:
        self._metric = metric

    def set(self, service: str, metric_name: str, stat: str, value: float) -> None:
        metric = self._metric
        if metric is None:
            return
        metric.labels(service=service, metric=metric_name, stat=stat).set(value)


class _AIQualityHistogramProxy:
    """Wrapper for AI extraction quality histogram."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusHistogram] = None

    def set_metric(self, metric: Optional[PrometheusHistogram]) -> None:
        self._metric = metric

    def observe(self, provider: str, prompt_key: str, variant: str, score: float) -> None:
        metric = self._metric
        if metric is None:
            return
        metric.labels(
            provider=provider or "unknown", prompt_key=prompt_key or "unknown", variant=variant or "default"
        ).observe(max(score, 0.0))


class _AIParseResultCounterProxy:
    """Wrapper for AI parse success/failure counter."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusCounter] = None

    def set_metric(self, metric: Optional[PrometheusCounter]) -> None:
        self._metric = metric

    def inc(self, provider: str, prompt_key: str, result: str) -> None:
        metric = self._metric
        if metric is None:
            return
        normalized = result or "unknown"
        metric.labels(provider=provider or "unknown", prompt_key=prompt_key or "unknown", result=normalized).inc()


# === Phase 9.1: Draft and Review Queue Metrics ===


class _DraftsQueuedCounterProxy:
    """Wrapper for drafts_queued_total counter (Phase 9.1)."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusCounter] = None

    def set_metric(self, metric: Optional[PrometheusCounter]) -> None:
        self._metric = metric

    def inc(self, priority: str, confidence_bucket: str, amount: float = 1.0) -> None:
        metric = self._metric
        if metric is None:
            return
        metric.labels(priority=priority or "normal", confidence_bucket=confidence_bucket or "unknown").inc(amount)


class _DraftsSentCounterProxy:
    """Wrapper for drafts_sent_total counter (Phase 9.1)."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusCounter] = None

    def set_metric(self, metric: Optional[PrometheusCounter]) -> None:
        self._metric = metric

    def inc(self, outcome: str, amount: float = 1.0) -> None:
        metric = self._metric
        if metric is None:
            return
        metric.labels(outcome=outcome or "unknown").inc(amount)


class _ReviewQueueDepthGaugeProxy:
    """Wrapper for review_queue_depth gauge (Phase 9.1)."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusGauge] = None

    def set_metric(self, metric: Optional[PrometheusGauge]) -> None:
        self._metric = metric

    def set(self, status: str, count: float) -> None:
        metric = self._metric
        if metric is None:
            return
        metric.labels(status=status or "unknown").set(max(count, 0.0))


class _ResponseTimeHistogramProxy:
    """Wrapper for response_time histogram (Phase 9.1)."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusHistogram] = None

    def set_metric(self, metric: Optional[PrometheusHistogram]) -> None:
        self._metric = metric

    def observe(self, hours: float) -> None:
        metric = self._metric
        if metric is None:
            return
        metric.observe(max(hours, 0.0))


class _ResponseFunnelGaugeProxy:
    """Wrapper for response_funnel gauge (Phase 4.3)."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusGauge] = None

    def set_metric(self, metric: Optional[PrometheusGauge]) -> None:
        self._metric = metric

    def set(self, stage: str, count: float) -> None:
        """Set count for a funnel stage: sent, replied, productive, fact_extracted."""
        metric = self._metric
        if metric is None:
            return
        metric.labels(stage=stage or "unknown").set(max(count, 0.0))


class _QualityDistributionGaugeProxy:
    """Wrapper for draft_quality_distribution gauge (Phase 4.3)."""

    def __init__(self) -> None:
        self._metric: Optional[PrometheusGauge] = None

    def set_metric(self, metric: Optional[PrometheusGauge]) -> None:
        self._metric = metric

    def set(self, tier: str, count: float) -> None:
        """Set count for a quality tier: excellent, good, acceptable, poor."""
        metric = self._metric
        if metric is None:
            return
        metric.labels(tier=tier or "unknown").set(max(count, 0.0))


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
        self.database_query_latency = _DatabaseQueryHistogramProxy()
        self.database_rows = _DatabaseRowsCounterProxy()
        self.action_duration = _ActionDurationHistogramProxy()
        self.internal_metrics = _InternalMetricGaugeProxy()
        self.ai_quality = _AIQualityHistogramProxy()
        self.ai_parse_results = _AIParseResultCounterProxy()
        # Phase 9.1: Draft and review queue metrics
        self.drafts_queued = _DraftsQueuedCounterProxy()
        self.drafts_sent = _DraftsSentCounterProxy()
        self.review_queue_depth = _ReviewQueueDepthGaugeProxy()
        self.response_time = _ResponseTimeHistogramProxy()
        # Phase 4.3: Dashboard integration metrics
        self.response_funnel = _ResponseFunnelGaugeProxy()
        self.quality_distribution = _QualityDistributionGaugeProxy()

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
        self.database_query_latency.set_metric(metrics_map.get("database_query_latency"))
        self.database_rows.set_metric(metrics_map.get("database_rows"))
        self.action_duration.set_metric(metrics_map.get("action_duration"))
        self.internal_metrics.set_metric(metrics_map.get("internal_metrics"))
        self.ai_quality.set_metric(metrics_map.get("ai_quality"))
        self.ai_parse_results.set_metric(metrics_map.get("ai_parse_results"))
        # Phase 9.1: Draft and review queue metrics
        self.drafts_queued.set_metric(metrics_map.get("drafts_queued"))
        self.drafts_sent.set_metric(metrics_map.get("drafts_sent"))
        self.review_queue_depth.set_metric(metrics_map.get("review_queue_depth"))
        self.response_time.set_metric(metrics_map.get("response_time"))
        # Phase 4.3: Dashboard integration metrics
        self.response_funnel.set_metric(metrics_map.get("response_funnel"))
        self.quality_distribution.set_metric(metrics_map.get("quality_distribution"))

    def reset(self) -> None:
        """Clear metric bindings (no-op proxies)."""
        self.assign({})


class MetricsRegistry:
    """Central Prometheus registry manager with enable/disable support."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._enabled = False
        self._namespace = "ancestry"
        self._registry: Optional[Any] = None
        self._metrics = MetricsBundle()
        self._import_logged = False
        self._config_enabled = False

    def configure(self, settings: Optional[ObservabilityConfig]) -> None:
        """Configure metrics using Observability settings."""
        with self._lock:
            self._config_enabled = bool(settings and settings.enable_prometheus_metrics)

            if settings is None or not settings.enable_prometheus_metrics:
                if self._enabled:
                    logger.info("Prometheus metrics disabled by configuration")
                self._disable_locked()
                return

            if not PROMETHEUS_AVAILABLE:
                if not self._import_logged:
                    logger.debug(
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

    def status(self) -> dict[str, Any]:
        """Return a debug snapshot of the metrics subsystem state."""
        with self._lock:
            return {
                "config_enabled": self._config_enabled,
                "enabled": self._enabled,
                "namespace": self._namespace,
                "prometheus_available": PROMETHEUS_AVAILABLE,
                "import_error": repr(_IMPORT_ERROR) if _IMPORT_ERROR else None,
            }

    def reset(self) -> None:
        """Disable metrics and clear existing registry."""
        with self._lock:
            self._disable_locked()

    def _disable_locked(self) -> None:
        self._enabled = False
        self._registry = None
        self._metrics.reset()

    @staticmethod
    def _create_metrics(
        namespace: str,
        registry: Any,
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

        metrics_map["database_query_latency"] = Histogram(
            "database_query_duration_seconds",
            "Database query duration",
            labelnames=("operation",),
            namespace=namespace,
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
            registry=registry,
        )

        metrics_map["database_rows"] = Counter(
            "database_rows_total",
            "Rows affected per database operation",
            labelnames=("operation",),
            namespace=namespace,
            registry=registry,
        )

        metrics_map["action_duration"] = Histogram(
            "action_duration_seconds",
            "Action execution duration",
            labelnames=("action",),
            namespace=namespace,
            buckets=(30.0, 60.0, 120.0, 300.0, 600.0, 1200.0, 2400.0),
            registry=registry,
        )

        metrics_map["internal_metrics"] = Gauge(
            "internal_metric_value",
            "Internal aggregated metric snapshot",
            labelnames=("service", "metric", "stat"),
            namespace=namespace,
            registry=registry,
        )

        metrics_map["ai_quality"] = Histogram(
            "ai_quality_score",
            "AI extraction quality score",
            labelnames=("provider", "prompt_key", "variant"),
            namespace=namespace,
            buckets=(40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 100.0),
            registry=registry,
        )

        metrics_map["ai_parse_results"] = Counter(
            "ai_parse_results_total",
            "AI parse success/failure totals",
            labelnames=("provider", "prompt_key", "result"),
            namespace=namespace,
            registry=registry,
        )

        # Phase 9.1: Draft and review queue metrics
        metrics_map["drafts_queued"] = Counter(
            "drafts_queued_total",
            "Total drafts queued for review",
            labelnames=("priority", "confidence_bucket"),
            namespace=namespace,
            registry=registry,
        )

        metrics_map["drafts_sent"] = Counter(
            "drafts_sent_total",
            "Total drafts sent (by outcome: sent/skipped/error)",
            labelnames=("outcome",),
            namespace=namespace,
            registry=registry,
        )

        metrics_map["review_queue_depth"] = Gauge(
            "review_queue_depth",
            "Current depth of draft review queue by status",
            labelnames=("status",),
            namespace=namespace,
            registry=registry,
        )

        # Phase 4.3: Dashboard integration metrics
        metrics_map["response_funnel"] = Gauge(
            "response_funnel",
            "Response funnel counts by stage: sent, replied, productive, fact_extracted",
            labelnames=("stage",),
            namespace=namespace,
            registry=registry,
        )

        metrics_map["quality_distribution"] = Gauge(
            "quality_distribution",
            "Draft quality distribution by tier: excellent, good, acceptable, poor",
            labelnames=("tier",),
            namespace=namespace,
            registry=registry,
        )

        metrics_map["response_time"] = Histogram(
            "response_time_hours",
            "Time from message sent to reply received (hours)",
            namespace=namespace,
            buckets=(1.0, 6.0, 12.0, 24.0, 48.0, 72.0, 168.0, 336.0),
            registry=registry,
        )

        # Phase 11.4: Personalization effectiveness metrics
        metrics_map["personalization_usage"] = Counter(
            "personalization_function_usage_total",
            "Usage count of personalization functions",
            labelnames=("function_name", "template_key"),
            namespace=namespace,
            registry=registry,
        )

        metrics_map["personalization_effectiveness"] = Histogram(
            "personalization_effectiveness_score",
            "Effectiveness score per personalization function (0-5)",
            labelnames=("function_name",),
            namespace=namespace,
            buckets=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
            registry=registry,
        )

        metrics_map["personalization_ab_assignment"] = Counter(
            "personalization_ab_assignment_total",
            "A/B test variant assignments for personalization strategies",
            labelnames=("experiment_id", "variant_name"),
            namespace=namespace,
            registry=registry,
        )

        metrics_map["personalization_ab_outcome"] = Counter(
            "personalization_ab_outcome_total",
            "A/B test outcomes for personalization strategies",
            labelnames=("experiment_id", "variant_name", "response_intent"),
            namespace=namespace,
            registry=registry,
        )

        return metrics_map

    def get_registry(self) -> Optional[Any]:
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


def get_metrics_registry() -> Optional[Any]:
    """Return the active Prometheus registry for exporter wiring."""
    return _METRICS_REGISTRY.get_registry()


def is_metrics_enabled() -> bool:
    """Return True when Prometheus metrics are currently enabled."""
    return _METRICS_REGISTRY.is_enabled()


def get_metrics_status() -> dict[str, Any]:
    """Return debug information about metrics configuration and runtime state."""
    return _METRICS_REGISTRY.status()


def record_internal_metric_stat(service: str, metric_name: str, stat: str, value: float) -> None:
    """Public helper to forward internal metrics to Prometheus."""

    safe_service = service or "unknown"
    safe_metric = metric_name or "metric"
    safe_stat = stat or "stat"
    try:
        metrics().internal_metrics.set(safe_service, safe_metric, safe_stat, float(value))
    except Exception:
        logger.debug("Failed to record internal metric stat", exc_info=True)


def _make_enabled_settings(namespace: str = "test_observability") -> ObservabilityConfig:
    from config.config_schema import ObservabilityConfig

    return ObservabilityConfig(
        enable_prometheus_metrics=True,
        metrics_namespace=namespace,
        metrics_export_port=9100,
    )


def test_metrics_disabled_is_noop() -> None:
    """Disabling metrics should make proxies safe no-ops."""
    from config.config_schema import ObservabilityConfig

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
    bundle.database_query_latency.observe("select", 0.12)
    bundle.database_rows.inc("select", 5)
    bundle.action_duration.observe("action6", 42.0)
    bundle.internal_metrics.set("TestService", "latency", "p95", 1.23)
    bundle.ai_quality.observe("gemini", "intent", "v1", 88.0)
    bundle.ai_parse_results.inc("gemini", "intent", "success")

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

    db_latency_sum = registry.get_sample_value(
        "test_metrics_database_query_duration_seconds_sum",
        labels={"operation": "select"},
    )
    assert db_latency_sum == 0.12

    db_rows = registry.get_sample_value(
        "test_metrics_database_rows_total",
        labels={"operation": "select"},
    )
    assert db_rows == 5.0

    action_duration_sum = registry.get_sample_value(
        "test_metrics_action_duration_seconds_sum",
        labels={"action": "action6"},
    )
    assert action_duration_sum == 42.0

    internal_metric_val = registry.get_sample_value(
        "test_metrics_internal_metric_value",
        labels={"service": "TestService", "metric": "latency", "stat": "p95"},
    )
    assert internal_metric_val == 1.23

    ai_quality_sum = registry.get_sample_value(
        "test_metrics_ai_quality_score_sum",
        labels={"provider": "gemini", "prompt_key": "intent", "variant": "v1"},
    )
    assert ai_quality_sum == 88.0

    ai_parse_total = registry.get_sample_value(
        "test_metrics_ai_parse_results_total",
        labels={"provider": "gemini", "prompt_key": "intent", "result": "success"},
    )
    assert ai_parse_total == 1.0

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
from testing.test_utilities import create_standard_test_runner

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
    "record_internal_metric_stat",
    "reset_metrics",
]
