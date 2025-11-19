"""
Metrics Collection and Aggregation System

Centralized metrics collection for monitoring system performance, API usage,
cache effectiveness, and session health. Provides real-time aggregation,
time-series analysis, and alerting for performance degradation.

Architecture:
- MetricRegistry: Central collection point (singleton pattern)
- MetricType: Standardized metric categories (histogram, counter, gauge, timer)
- ServiceMetrics: Per-service aggregation with windowing
- MetricsSnapshot: Point-in-time aggregation for reporting
- PerformanceAlert: Deviation detection and alerting

Key Features:
- Thread-safe collection with minimal locking overhead
- Automatic time-window aggregation (1min, 5min, 15min, 1hour windows)
- Service-aware metrics (separate for APIManager, BrowserManager, DatabaseManager, etc.)
- Percentile-based analysis (p50, p95, p99 for response times)
- Memory-efficient ring buffers for large datasets
- JSON export for external analysis tools
- Configurable retention (default: 7 days)

Thread Safety:
- All public methods use RLock for concurrent access
- Internal collections (deques) are thread-safe for append operations
- Statistics calculations are lock-free where possible

Usage:
    from core.metrics_collector import get_metrics_registry

    registry = get_metrics_registry()

    # Record API call timing
    registry.record_metric("APIManager", "response_time_ms", duration_ms)

    # Record cache hit
    registry.record_metric("CacheManager", "hit_rate", 1 if hit else 0)

    # Get aggregated stats
    snapshot = registry.get_snapshot()
    print(f"API p95 latency: {snapshot.get_percentile('APIManager', 'response_time_ms', 95)}ms")

    # Export for monitoring
    json_data = registry.to_json()
"""

import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from statistics import median, quantiles
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from observability.metrics_registry import is_metrics_enabled, record_internal_metric_stat

# Support standalone execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))


class MetricType(Enum):
    """Standardized metric types."""
    HISTOGRAM = "histogram"  # Distribution (response times, sizes)
    COUNTER = "counter"      # Monotonic increase (API calls, errors)
    GAUGE = "gauge"          # Point-in-time value (memory, active sessions)
    TIMER = "timer"          # Duration measurement
    RATE = "rate"            # Per-second or per-minute rate


@dataclass
class MetricPoint:
    """Single data point for a metric."""
    timestamp: float
    value: float
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "value": self.value,
            "labels": self.labels
        }


@dataclass
class WindowedStats:
    """Statistics for a time window."""
    window_name: str  # "1min", "5min", "15min", "1hour"
    count: int
    sum_value: float
    min_value: float
    max_value: float
    avg_value: float
    p50: float  # Median
    p95: float
    p99: float


def _metric_series_map_factory() -> dict[str, deque[MetricPoint]]:
    """Return an empty dict for per-metric deque tracking."""
    return {}


@dataclass
class ServiceMetrics:
    """Aggregated metrics for a service."""
    service_name: str
    metrics: dict[str, deque[MetricPoint]] = field(default_factory=_metric_series_map_factory)
    windows: dict[str, dict[str, WindowedStats]] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def record_value(self, metric_name: str, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """Record a metric value."""
        with self._lock:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = deque(maxlen=10000)  # Keep last 10K points
            self.metrics[metric_name].append(MetricPoint(time.time(), value, labels or {}))

    def get_stats_for_window(self, metric_name: str, window_seconds: int) -> Optional[WindowedStats]:
        """Calculate statistics for metric over time window."""
        with self._lock:
            if metric_name not in self.metrics or not self.metrics[metric_name]:
                return None

            now = time.time()
            cutoff = now - window_seconds
            values = [
                point.value for point in self.metrics[metric_name]
                if point.timestamp >= cutoff
            ]

            if not values:
                return None

            sorted_values = sorted(values)
            count = len(values)
            sum_val = sum(values)
            min_val = min(values)
            max_val = max(values)
            avg_val = sum_val / count

            # Calculate percentiles
            p50 = median(values)
            try:
                # quantiles() requires at least 2 elements
                if count >= 2:
                    percentiles = quantiles(sorted_values, n=100)
                    p95 = percentiles[94]  # 95th percentile (index 94 in 0-99 range)
                    p99 = percentiles[98]  # 99th percentile
                else:
                    p95 = p99 = avg_val
            except (ValueError, IndexError):
                p95 = p99 = avg_val

            window_name = self._get_window_name(window_seconds)
            return WindowedStats(
                window_name=window_name,
                count=count,
                sum_value=sum_val,
                min_value=min_val,
                max_value=max_val,
                avg_value=avg_val,
                p50=p50,
                p95=p95,
                p99=p99
            )

    def get_percentile(self, metric_name: str, percentile: int) -> Optional[float]:
        """Get percentile value from all recorded values."""
        with self._lock:
            if metric_name not in self.metrics or not self.metrics[metric_name]:
                return None

            values = sorted([point.value for point in self.metrics[metric_name]])
            if not values:
                return None

            index = int(len(values) * percentile / 100)
            return values[min(index, len(values) - 1)]

    @staticmethod
    def _get_window_name(seconds: int) -> str:
        """Convert seconds to window name."""
        if seconds < 120:
            return "1min"
        if seconds < 600:
            return "5min"
        if seconds < 3600:
            return "15min"
        return "1hour"

    def clear(self) -> None:
        """Clear all recorded metrics."""
        with self._lock:
            self.metrics.clear()
            self.windows.clear()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        with self._lock:
            return {
                "service_name": self.service_name,
                "metric_count": len(self.metrics),
                "metrics": {
                    name: [point.to_dict() for point in list(points)]
                    for name, points in self.metrics.items()
                }
            }


@dataclass
class MetricsSnapshot:
    """Point-in-time snapshot of all metrics."""
    timestamp: datetime
    services: dict[str, dict[str, Any]]

    def get_percentile(self, service_name: str, metric_name: str, percentile: int) -> Optional[float]:
        """Get percentile from a specific service metric."""
        if service_name not in self.services:
            return None
        service_metrics = self.services[service_name]
        if metric_name not in service_metrics:
            return None
        return service_metrics[metric_name].get(f"p{percentile}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "services": self.services
        }


@dataclass
class PerformanceAlert:
    """Alert for performance degradation."""
    service_name: str
    metric_name: str
    current_value: float
    baseline_value: float
    deviation_percent: float
    severity: str  # "warning", "critical"
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "service": self.service_name,
            "metric": self.metric_name,
            "current": self.current_value,
            "baseline": self.baseline_value,
            "deviation_percent": self.deviation_percent,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat()
        }


class MetricRegistry:
    """
    Central registry for all system metrics.

    Thread-safe collection point supporting:
    - Multi-service aggregation
    - Time-window aggregation (1min, 5min, 15min, 1hour)
    - Percentile analysis
    - Performance alerting
    - JSON export

    Singleton pattern ensures single instance across application.
    """

    def __init__(self) -> None:
        self.services: dict[str, ServiceMetrics] = {}
        self._lock = threading.RLock()
        self._baseline: dict[str, dict[str, float]] = {}
        self._alerts: list[PerformanceAlert] = []
        self._retention_seconds = 7 * 24 * 3600  # 7 days default
        self._last_prom_sync = 0.0
        self._prom_sync_interval = 30.0

    def record_metric(
        self,
        service_name: str,
        metric_name: str,
        value: float,
        labels: Optional[dict[str, str]] = None
    ) -> None:
        """Record a metric value for a service."""
        with self._lock:
            if service_name not in self.services:
                self.services[service_name] = ServiceMetrics(service_name)
            self.services[service_name].record_value(metric_name, value, labels)

        self._maybe_sync_prometheus()

    def record_timer(
        self,
        service_name: str,
        metric_name: str,
        duration_seconds: float
    ) -> None:
        """Record a timer measurement (converted to milliseconds)."""
        self.record_metric(service_name, metric_name, duration_seconds * 1000)

    def get_service_metrics(self, service_name: str) -> Optional[ServiceMetrics]:
        """Get metrics for a specific service."""
        with self._lock:
            return self.services.get(service_name)

    def get_snapshot(self) -> MetricsSnapshot:
        """Get current snapshot of all metrics."""
        with self._lock:
            services_data = {}
            for service_name, service_metrics in self.services.items():
                service_data = {}

                # Get stats for each metric over multiple windows
                for metric_name in service_metrics.metrics:
                    metric_stats = {}

                    # Calculate stats for different time windows
                    for window_seconds in [60, 300, 900, 3600]:  # 1, 5, 15 min, 1 hour
                        window_stats = service_metrics.get_stats_for_window(metric_name, window_seconds)
                        if window_stats:
                            window_key = f"window_{window_stats.window_name}"
                            metric_stats[window_key] = {
                                "count": window_stats.count,
                                "min": window_stats.min_value,
                                "max": window_stats.max_value,
                                "avg": window_stats.avg_value,
                                "p50": window_stats.p50,
                                "p95": window_stats.p95,
                                "p99": window_stats.p99,
                            }

                    # Overall percentiles
                    metric_stats["p50"] = service_metrics.get_percentile(metric_name, 50)
                    metric_stats["p95"] = service_metrics.get_percentile(metric_name, 95)
                    metric_stats["p99"] = service_metrics.get_percentile(metric_name, 99)

                    service_data[metric_name] = metric_stats

                services_data[service_name] = service_data

            return MetricsSnapshot(datetime.now(), services_data)

    def set_baseline(self, service_name: str, metric_name: str, baseline_value: float) -> None:
        """Set baseline value for alert detection."""
        with self._lock:
            if service_name not in self._baseline:
                self._baseline[service_name] = {}
            self._baseline[service_name][metric_name] = baseline_value

    def check_for_alerts(self, deviation_threshold_percent: float = 10.0) -> list[PerformanceAlert]:
        """Check for performance degradation vs. baseline."""
        alerts = []

        with self._lock:
            for service_name, service_metrics in self.services.items():
                if service_name not in self._baseline:
                    continue

                for metric_name in service_metrics.metrics:
                    if metric_name not in self._baseline[service_name]:
                        continue

                    baseline = self._baseline[service_name][metric_name]
                    current = service_metrics.get_percentile(metric_name, 95)  # Use p95

                    if current is None:
                        continue

                    deviation = ((current - baseline) / baseline * 100) if baseline != 0 else 0

                    if abs(deviation) > deviation_threshold_percent:
                        severity = "critical" if abs(deviation) > 25 else "warning"
                        alert = PerformanceAlert(
                            service_name=service_name,
                            metric_name=metric_name,
                            current_value=current,
                            baseline_value=baseline,
                            deviation_percent=deviation,
                            severity=severity,
                            timestamp=datetime.now()
                        )
                        alerts.append(alert)
                        self._alerts.append(alert)

        return alerts

    def get_alerts(self, limit: Optional[int] = None) -> list[PerformanceAlert]:
        """Get recent alerts."""
        with self._lock:
            alerts = list(reversed(self._alerts))  # Most recent first
            return alerts[:limit] if limit else alerts

    def clear_service(self, service_name: str) -> None:
        """Clear metrics for a specific service."""
        with self._lock:
            if service_name in self.services:
                self.services[service_name].clear()

    def clear_all(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self.services.clear()
            self._alerts.clear()

    def to_json(self, pretty: bool = True) -> str:
        """Export metrics to JSON."""
        snapshot = self.get_snapshot()
        data = {
            "timestamp": snapshot.timestamp.isoformat(),
            "services": snapshot.services,
            "recent_alerts": [alert.to_dict() for alert in self.get_alerts(limit=10)]
        }
        return json.dumps(data, indent=2 if pretty else None, default=str)

    def save_to_file(self, file_path: Optional[Path] = None) -> Path:
        """Save metrics to JSON file."""
        if file_path is None:
            file_path = Path("Logs") / "metrics_snapshot.json"

        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with file_path.open("w", encoding="utf-8") as f:
                f.write(self.to_json(pretty=True))
            return file_path
        except Exception as e:
            print(f"⚠️ Failed to save metrics: {e}")
            return file_path

    def _maybe_sync_prometheus(self) -> None:
        """Periodically export collector stats to Prometheus."""

        if not is_metrics_enabled():
            return

        now = time.time()
        if now - self._last_prom_sync < self._prom_sync_interval:
            return

        self._last_prom_sync = now
        snapshot = self.get_snapshot()
        for service_name, metric_map in snapshot.services.items():
            for metric_name, stats in metric_map.items():
                self._record_internal_metric(service_name, metric_name, stats)

    @staticmethod
    def _record_internal_metric(service: str, metric_name: str, stats: dict[str, Any]) -> None:
        """Emit a curated subset of collector stats into Prometheus."""

        try:
            for stat_key in ("p50", "p95", "p99"):
                value = stats.get(stat_key)
                if isinstance(value, (int, float)):
                    record_internal_metric_stat(service, metric_name, stat_key, float(value))

            for window_key in ("window_1min", "window_5min"):
                window_stats = stats.get(window_key)
                if isinstance(window_stats, dict):
                    avg_value = window_stats.get("avg")
                    if isinstance(avg_value, (int, float)):
                        record_internal_metric_stat(service, metric_name, f"{window_key}_avg", float(avg_value))
        except Exception:
            # Silent failure - observability must stay best-effort
            pass


# Global singleton instance
_metrics_registry: Optional[MetricRegistry] = None
_registry_lock = threading.Lock()


def get_metrics_registry() -> MetricRegistry:
    """Get or create the global metrics registry (singleton pattern)."""
    global _metrics_registry  # noqa: PLW0603 - Singleton pattern requires global update

    if _metrics_registry is None:
        with _registry_lock:
            if _metrics_registry is None:
                _metrics_registry = MetricRegistry()

    return _metrics_registry


# Test suite
def core_metrics_collector_module_tests() -> bool:
    """Run comprehensive tests for metrics collection system."""
    from test_framework import TestSuite

    suite = TestSuite("MetricsCollector", "core/metrics_collector.py")
    suite.start_suite()

    # Test 1: Basic metric recording
    def test_basic_recording() -> None:
        registry = MetricRegistry()
        registry.record_metric("TestService", "response_time_ms", 100.0)
        registry.record_metric("TestService", "response_time_ms", 150.0)
        registry.record_metric("TestService", "response_time_ms", 200.0)

        snapshot = registry.get_snapshot()
        assert "TestService" in snapshot.services
        assert "response_time_ms" in snapshot.services["TestService"]

    suite.run_test("Basic metric recording", test_basic_recording)

    # Test 2: Percentile calculations
    def test_percentiles() -> None:
        registry = MetricRegistry()
        for i in range(1, 101):  # Record 1-100
            registry.record_metric("TestService", "latency_ms", float(i))

        service_metrics = registry.get_service_metrics("TestService")
        assert service_metrics is not None
        p95 = service_metrics.get_percentile("latency_ms", 95)
        assert p95 is not None
        assert 90 <= p95 <= 100  # Should be close to 95th percentile

    suite.run_test("Percentile calculations", test_percentiles)

    # Test 3: Multi-service isolation
    def test_multi_service() -> None:
        registry = MetricRegistry()
        registry.record_metric("Service1", "metric1", 100.0)
        registry.record_metric("Service2", "metric1", 200.0)

        snapshot = registry.get_snapshot()
        assert "Service1" in snapshot.services
        assert "Service2" in snapshot.services
        assert snapshot.services["Service1"] != snapshot.services["Service2"]

    suite.run_test("Multi-service isolation", test_multi_service)

    # Test 4: Baseline and alerts
    def test_alerts() -> None:
        registry = MetricRegistry()
        registry.set_baseline("TestService", "response_time_ms", 100.0)

        # Record values 50% above baseline
        for _ in range(10):
            registry.record_metric("TestService", "response_time_ms", 150.0)

        alerts = registry.check_for_alerts(deviation_threshold_percent=10.0)
        assert len(alerts) > 0
        assert alerts[0].service_name == "TestService"

    suite.run_test("Baseline and alerts", test_alerts)

    # Test 5: Thread safety
    def test_thread_safety() -> None:
        registry = MetricRegistry()
        results = []

        def record_metrics(service_id: int) -> None:
            for i in range(100):
                registry.record_metric(f"Service{service_id}", "metric", float(i))
            results.append(True)

        threads = [threading.Thread(target=record_metrics, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        snapshot = registry.get_snapshot()
        assert len(snapshot.services) == 5

    suite.run_test("Thread safety", test_thread_safety)

    # Test 6: Timer recording
    def test_timer() -> None:
        registry = MetricRegistry()
        registry.record_timer("TestService", "duration_ms", 0.5)  # 500ms
        registry.record_timer("TestService", "duration_ms", 0.1)  # 100ms

        service_metrics = registry.get_service_metrics("TestService")
        assert service_metrics is not None
        values = [p.value for p in service_metrics.metrics["duration_ms"]]
        assert len(values) == 2
        assert 400 <= values[0] <= 600  # 500ms in ms

    suite.run_test("Timer recording", test_timer)

    # Test 7: JSON export
    def test_json_export() -> None:
        registry = MetricRegistry()
        registry.record_metric("TestService", "metric1", 100.0)
        registry.set_baseline("TestService", "metric1", 100.0)

        json_str = registry.to_json()
        data = json.loads(json_str)

        assert "timestamp" in data
        assert "services" in data
        assert "TestService" in data["services"]

    suite.run_test("JSON export", test_json_export)

    # Test 8: Singleton pattern
    def test_singleton() -> None:
        registry1 = get_metrics_registry()
        registry2 = get_metrics_registry()
        assert registry1 is registry2

    suite.run_test("Singleton pattern", test_singleton)

    return suite.finish_suite()


# Use centralized test runner utility from test_utilities
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(core_metrics_collector_module_tests)


if __name__ == "__main__":
    import sys
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
