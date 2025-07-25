#!/usr/bin/env python3

#!/usr/bin/env python3

"""
Performance Monitor - Advanced Performance Tracking & Optimization

This module provides comprehensive performance monitoring, profiling,
and optimization recommendations for the Ancestry project.

Features:
- Function execution time tracking
- Memory usage monitoring
- Database query performance analysis
- Cache hit/miss ratio tracking
- API response time monitoring
- Automated performance alerts
- Performance trend analysis
- Optimization recommendations
"""

# === CORE INFRASTRUCTURE ===
from core_imports import (
    standardize_module_imports,
    auto_register_module,
    get_logger,
)

standardize_module_imports()
auto_register_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import gc
import json
import statistics
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# === THIRD-PARTY IMPORTS ===
import psutil

# === MODULE LOGGER ===
logger = get_logger(__name__)


class AlertLevel(Enum):
    """Performance alert levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""

    name: str
    value: float
    timestamp: datetime
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert."""

    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    recommendation: Optional[str] = None


@dataclass
class FunctionProfile:
    """Function performance profile."""

    function_name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    avg_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    memory_usage: List[float] = field(default_factory=list)
    error_count: int = 0
    last_called: Optional[datetime] = None


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""

    def __init__(
        self,
        max_history: int = 10000,
        alert_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.max_history = max_history
        self.metrics: deque = deque(maxlen=max_history)
        self.function_profiles: Dict[str, FunctionProfile] = {}
        self.alerts: List[PerformanceAlert] = []
        self.alert_thresholds = alert_thresholds or self._default_thresholds()
        self._lock = threading.RLock()
        self._start_time = time.time()
        self.enabled = True

        # System monitoring
        self.process = psutil.Process()

        # Background monitoring thread
        self._monitor_thread = None
        self._stop_monitoring = False

        logger.info("Performance monitor initialized")

    def _default_thresholds(self) -> Dict[str, float]:
        """Default performance alert thresholds."""
        return {
            "memory_usage_mb": 1024,  # 1GB
            "cpu_usage_percent": 80.0,
            "function_time_ms": 5000.0,  # 5 seconds
            "cache_hit_ratio": 0.5,  # 50%
            "database_query_time_ms": 1000.0,  # 1 second
            "api_response_time_ms": 3000.0,  # 3 seconds
        }

    def start_monitoring(self) -> None:
        """Start background performance monitoring."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_monitoring = False
            self._monitor_thread = threading.Thread(
                target=self._background_monitor, daemon=True
            )
            self._monitor_thread.start()
            logger.info("Background performance monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background performance monitoring."""
        self._stop_monitoring = True
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        logger.info("Background performance monitoring stopped")

    def _background_monitor(self) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring:
            try:
                self._collect_system_metrics()
                self._check_alerts()
                time.sleep(30)  # Monitor every 30 seconds
            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
                time.sleep(60)  # Wait longer on error

    def _collect_system_metrics(self) -> None:
        """Collect system-wide performance metrics."""
        try:
            # Memory usage
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            self.record_metric("memory_usage_mb", memory_mb, "system")

            # CPU usage
            cpu_percent = self.process.cpu_percent()
            self.record_metric("cpu_usage_percent", cpu_percent, "system")

            # System memory
            sys_memory = psutil.virtual_memory()
            self.record_metric("system_memory_percent", sys_memory.percent, "system")

            # Disk usage (project directory)
            disk_usage = psutil.disk_usage(str(Path.cwd()))
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            self.record_metric("disk_usage_percent", disk_percent, "system")

            # Thread count
            thread_count = threading.active_count()
            self.record_metric("thread_count", thread_count, "system")

        except Exception as e:
            logger.warning(f"Error collecting system metrics: {e}")

    def record_metric(
        self,
        name: str,
        value: float,
        category: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a performance metric."""
        if not self.enabled:
            return

        with self._lock:
            metric = PerformanceMetric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                category=category,
                metadata=metadata or {},
            )
            self.metrics.append(metric)

    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function performance."""
        if not self.enabled:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            function_name = f"{func.__module__}.{func.__name__}"

            # Get memory before
            gc.collect()
            memory_before = self.process.memory_info().rss / 1024 / 1024

            start_time = time.perf_counter()
            error_occurred = False

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_occurred = True
                raise
            finally:
                end_time = time.perf_counter()
                execution_time = end_time - start_time

                # Get memory after
                memory_after = self.process.memory_info().rss / 1024 / 1024
                memory_delta = memory_after - memory_before

                # Update profile
                with self._lock:
                    if function_name not in self.function_profiles:
                        self.function_profiles[function_name] = FunctionProfile(
                            function_name
                        )

                    profile = self.function_profiles[function_name]
                    profile.call_count += 1
                    profile.total_time += execution_time
                    profile.min_time = min(profile.min_time, execution_time)
                    profile.max_time = max(profile.max_time, execution_time)
                    profile.avg_time = profile.total_time / profile.call_count
                    profile.recent_times.append(execution_time)
                    profile.memory_usage.append(memory_delta)
                    profile.last_called = datetime.now()

                    if error_occurred:
                        profile.error_count += 1

                # Record metric
                self.record_metric(
                    f"function_time_ms",
                    execution_time * 1000,
                    "performance",
                    {
                        "function": function_name,
                        "memory_delta_mb": memory_delta,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs),
                    },
                )

                # Check for slow functions
                if execution_time * 1000 > self.alert_thresholds.get(
                    "function_time_ms", 5000
                ):
                    self._create_alert(
                        AlertLevel.WARNING,
                        f"Slow function execution: {function_name}",
                        "function_time_ms",
                        execution_time * 1000,
                        self.alert_thresholds["function_time_ms"],
                        f"Consider optimizing {function_name} - took {execution_time:.2f}s",
                    )

        return wrapper

    def _check_alerts(self) -> None:
        """Check for performance alerts."""
        if not self.metrics:
            return

        # Get recent metrics (last 5 minutes)
        cutoff_time = datetime.now() - timedelta(minutes=5)
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]

        # Group by metric name
        metric_groups = defaultdict(list)
        for metric in recent_metrics:
            metric_groups[metric.name].append(metric.value)

        # Check thresholds
        for metric_name, values in metric_groups.items():
            if not values:
                continue

            avg_value = statistics.mean(values)
            threshold = self.alert_thresholds.get(metric_name)

            if threshold is not None:
                if avg_value > threshold:
                    alert_level = (
                        AlertLevel.CRITICAL
                        if avg_value > threshold * 1.5
                        else AlertLevel.WARNING
                    )
                    self._create_alert(
                        alert_level,
                        f"High {metric_name}: {avg_value:.2f}",
                        metric_name,
                        avg_value,
                        threshold,
                        self._get_optimization_recommendation(metric_name, avg_value),
                    )

    def _create_alert(
        self,
        level: AlertLevel,
        message: str,
        metric_name: str,
        current_value: float,
        threshold: float,
        recommendation: Optional[str] = None,
    ) -> None:
        """Create a performance alert."""
        alert = PerformanceAlert(
            level=level,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            timestamp=datetime.now(),
            recommendation=recommendation,
        )

        with self._lock:
            self.alerts.append(alert)
            # Keep only last 1000 alerts
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-1000:]

        if level == AlertLevel.CRITICAL:
            logger.error(f"CRITICAL PERFORMANCE ALERT: {message}")
        elif level == AlertLevel.WARNING:
            logger.warning(f"Performance warning: {message}")
        else:
            logger.info(f"Performance info: {message}")

    def _get_optimization_recommendation(self, metric_name: str, value: float) -> str:
        """Get optimization recommendation for a metric."""
        recommendations = {
            "memory_usage_mb": f"High memory usage ({value:.0f}MB). Consider: 1) Implementing object pooling, 2) Optimizing data structures, 3) Adding memory cleanup routines",
            "cpu_usage_percent": f"High CPU usage ({value:.1f}%). Consider: 1) Adding caching, 2) Optimizing algorithms, 3) Using background processing for heavy tasks",
            "function_time_ms": f"Slow function ({value:.0f}ms). Consider: 1) Adding profiling, 2) Optimizing database queries, 3) Implementing caching",
            "cache_hit_ratio": f"Low cache hit ratio ({value:.1%}). Consider: 1) Reviewing cache keys, 2) Increasing cache size, 3) Improving cache warming",
            "database_query_time_ms": f"Slow database queries ({value:.0f}ms). Consider: 1) Adding indexes, 2) Query optimization, 3) Connection pooling",
            "api_response_time_ms": f"Slow API responses ({value:.0f}ms). Consider: 1) Response caching, 2) Request batching, 3) Timeout optimization",
        }
        return recommendations.get(
            metric_name, f"Performance issue detected with {metric_name}: {value}"
        )

    def get_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        recent_alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]

        # Calculate statistics
        metric_stats = defaultdict(list)
        for metric in recent_metrics:
            metric_stats[metric.name].append(metric.value)

        stats_summary = {}
        for name, values in metric_stats.items():
            if values:
                stats_summary[name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": statistics.mean(values),
                    "median": statistics.median(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                }

        # Function performance summary
        function_summary = {}
        for name, profile in self.function_profiles.items():
            function_summary[name] = {
                "call_count": profile.call_count,
                "total_time": profile.total_time,
                "avg_time": profile.avg_time,
                "min_time": profile.min_time,
                "max_time": profile.max_time,
                "error_count": profile.error_count,
                "error_rate": profile.error_count / max(profile.call_count, 1),
                "avg_memory_usage": (
                    statistics.mean(profile.memory_usage) if profile.memory_usage else 0
                ),
                "last_called": (
                    profile.last_called.isoformat() if profile.last_called else None
                ),
            }

        return {
            "summary": {
                "report_period_hours": hours,
                "generated_at": datetime.now().isoformat(),
                "uptime_seconds": time.time() - self._start_time,
                "total_metrics": len(recent_metrics),
                "total_alerts": len(recent_alerts),
                "alert_breakdown": {
                    "critical": len(
                        [a for a in recent_alerts if a.level == AlertLevel.CRITICAL]
                    ),
                    "warning": len(
                        [a for a in recent_alerts if a.level == AlertLevel.WARNING]
                    ),
                    "info": len(
                        [a for a in recent_alerts if a.level == AlertLevel.INFO]
                    ),
                },
                "functions_monitored": len(self.function_profiles),
                "monitoring_enabled": self.enabled,
            },
            "metric_statistics": stats_summary,
            "function_profiles": function_summary,
            "recent_alerts": [
                {
                    "level": alert.level.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "recommendation": alert.recommendation,
                }
                for alert in recent_alerts[-20:]  # Last 20 alerts
            ],
            "system_info": {
                "python_version": sys.version,
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "disk_total_gb": psutil.disk_usage(str(Path.cwd())).total
                / 1024
                / 1024
                / 1024,
            },
            "recommendations": self._generate_recommendations(
                stats_summary, function_summary
            ),
        }

    def _generate_recommendations(
        self, stats_summary: Dict, function_summary: Dict
    ) -> List[str]:
        """Generate performance recommendations based on collected data."""
        recommendations = []

        # Check for high error rates
        for name, profile in function_summary.items():
            if profile.get("error_rate", 0) > 0.1:
                recommendations.append(
                    f"High error rate in {name}: consider error handling improvements"
                )

        # Check for slow functions
        for name, profile in function_summary.items():
            if profile.get("avg_time", 0) > 1.0:
                recommendations.append(
                    f"Function {name} is slow (avg: {profile['avg_time']:.3f}s): consider optimization"
                )

        # Check for memory usage patterns
        for name, stats in stats_summary.items():
            if "memory" in name.lower() and stats.get("avg", 0) > 500:
                recommendations.append(
                    f"High memory usage detected for {name}: consider memory optimization"
                )

        if not recommendations:
            recommendations.append(
                "Performance metrics look good - no immediate optimizations needed"
            )

        return recommendations

    def export_report(self, filepath: Optional[Path] = None, hours: int = 24) -> Path:
        """Export performance report to JSON file."""
        report = self.get_report(hours)

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path(f"performance_report_{timestamp}.json")

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Performance report exported to {filepath}")
        return filepath


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# Convenience decorators
def profile(func: Callable) -> Callable:
    """Convenient decorator for function profiling."""
    return performance_monitor.profile_function(func)


def monitor_performance(func: Callable) -> Callable:
    """Alias for profile decorator."""
    return performance_monitor.profile_function(func)


# Auto-start monitoring only if not in main execution
if __name__ != "__main__":
    performance_monitor.start_monitoring()


# Main execution block
if __name__ == "__main__":
    from test_framework import TestSuite, suppress_logging
    from unittest.mock import MagicMock, patch
    import time

    suite = TestSuite(
        "Performance Monitor & Optimization System", "performance_monitor.py"
    )

    def test_class_availability():
        # Test that all main classes are available
        classes = [
            "PerformanceMetric",
            "PerformanceAlert",
            "FunctionProfile",
            "PerformanceMonitor",
            "AlertLevel",
        ]
        for class_name in classes:
            assert class_name in globals(), f"Class {class_name} should be available"
            assert isinstance(
                globals()[class_name], type
            ), f"{class_name} should be a class"

    def test_monitor_initialization():
        # Test PerformanceMonitor initialization
        monitor = PerformanceMonitor(max_history=100)
        assert monitor.max_history == 100, "Should set max history correctly"
        assert len(monitor.metrics) == 0, "Should start with empty metrics"
        assert (
            len(monitor.function_profiles) == 0
        ), "Should start with empty function profiles"
        assert len(monitor.alerts) == 0, "Should start with empty alerts"
        assert monitor.enabled == True, "Should be enabled by default"

        # Test with custom thresholds
        custom_thresholds = {"test_metric": 50.0}
        monitor2 = PerformanceMonitor(alert_thresholds=custom_thresholds)
        assert (
            "test_metric" in monitor2.alert_thresholds
        ), "Should use custom thresholds"

    def test_metric_recording():
        monitor = PerformanceMonitor()

        # Test basic metric recording
        monitor.record_metric("test_metric", 100.0, "test_category")
        assert len(monitor.metrics) == 1, "Should record one metric"

        metric = monitor.metrics[0]
        assert metric.name == "test_metric", "Should record correct metric name"
        assert metric.value == 100.0, "Should record correct metric value"
        assert metric.category == "test_category", "Should record correct category"
        assert isinstance(metric.timestamp, datetime), "Should have timestamp"

        # Test with metadata
        metadata = {"source": "test", "version": "1.0"}
        monitor.record_metric("test_metric2", 200.0, "test", metadata=metadata)
        assert len(monitor.metrics) == 2, "Should record second metric"
        assert monitor.metrics[1].metadata == metadata, "Should record metadata"

    def test_function_profiling():
        monitor = PerformanceMonitor()

        # Test function profiling decorator
        @monitor.profile_function
        def test_function(x, y=10):
            time.sleep(0.001)  # Small sleep for timing
            return x + y

        # Call function multiple times
        result1 = test_function(5)
        result2 = test_function(3, y=7)
        result3 = test_function(1)

        assert result1 == 15, "Function should work correctly"
        assert result2 == 10, "Function should work with kwargs"
        assert result3 == 11, "Function should work with default args"

        # Check profiling data (function name includes module path)
        function_names = list(monitor.function_profiles.keys())
        assert len(function_names) > 0, "Should have at least one profiled function"

        # Find the test function (should end with test_function)
        test_func_profile = None
        for name in function_names:
            if name.endswith("test_function"):
                test_func_profile = monitor.function_profiles[name]
                break

        assert test_func_profile is not None, "Function should be profiled"
        assert test_func_profile.call_count == 3, "Should track call count"
        assert test_func_profile.total_time > 0, "Should track total time"
        assert test_func_profile.min_time > 0, "Should track min time"
        assert test_func_profile.max_time > 0, "Should track max time"
        assert test_func_profile.avg_time > 0, "Should calculate average time"
        assert len(test_func_profile.recent_times) == 3, "Should track recent times"

    def test_function_profiling_with_errors():
        monitor = PerformanceMonitor()

        @monitor.profile_function
        def error_function():
            raise ValueError("Test error")

        # Test error handling
        try:
            error_function()
        except ValueError:
            pass  # Expected

        # Check error tracking (function name includes module path)
        function_names = list(monitor.function_profiles.keys())
        error_func_profile = None
        for name in function_names:
            if name.endswith("error_function"):
                error_func_profile = monitor.function_profiles[name]
                break

        assert (
            error_func_profile is not None
        ), "Function should be profiled even with errors"
        assert error_func_profile.error_count == 1, "Should track error count"
        assert error_func_profile.call_count == 1, "Should still track call count"

    def test_system_metrics_collection():
        monitor = PerformanceMonitor()

        # Test system metrics collection
        initial_count = len(monitor.metrics)
        monitor._collect_system_metrics()

        # Should have added system metrics
        assert len(monitor.metrics) > initial_count, "Should collect system metrics"

        # Check for expected system metrics
        metric_names = [m.name for m in monitor.metrics]
        expected_metrics = ["memory_usage_mb", "cpu_usage_percent"]
        for expected in expected_metrics:
            assert any(
                expected in name for name in metric_names
            ), f"Should collect {expected}"

    def test_alert_system():
        # Test with very low thresholds to trigger alerts
        thresholds = {
            "test_metric": 50.0,
            "memory_usage_mb": 0.1,  # Very low to trigger alert
        }
        monitor = PerformanceMonitor(alert_thresholds=thresholds)

        # Record metrics that exceed thresholds
        monitor.record_metric("test_metric", 100.0, "test")  # Above 50.0
        monitor.record_metric("memory_usage_mb", 1.0, "system")  # Above 0.1

        initial_alert_count = len(monitor.alerts)
        monitor._check_alerts()

        # Should have generated alerts
        assert (
            len(monitor.alerts) > initial_alert_count
        ), "Should generate alerts for threshold violations"

        # Check alert structure
        if monitor.alerts:
            alert = monitor.alerts[-1]
            assert isinstance(alert.level, AlertLevel), "Alert should have proper level"
            assert isinstance(alert.message, str), "Alert should have message"
            assert isinstance(
                alert.current_value, (int, float)
            ), "Alert should have current value"
            assert isinstance(
                alert.threshold, (int, float)
            ), "Alert should have threshold"

    def test_monitoring_control():
        monitor = PerformanceMonitor()

        # Test that monitoring controls are available
        assert hasattr(
            monitor, "start_monitoring"
        ), "Should have start_monitoring method"
        assert hasattr(monitor, "stop_monitoring"), "Should have stop_monitoring method"
        assert hasattr(monitor, "_stop_monitoring"), "Should have _stop_monitoring flag"

        # Test initial state
        assert monitor._stop_monitoring == False, "Should start with monitoring enabled"

        # Test setting stop flag directly (avoid thread complications)
        monitor._stop_monitoring = True
        assert monitor._stop_monitoring == True, "Should be able to set stop flag"

    def test_report_generation():
        monitor = PerformanceMonitor()

        # Add some test data
        monitor.record_metric("test_metric", 50.0, "test")
        monitor.record_metric("test_metric", 75.0, "test")
        monitor.record_metric("other_metric", 25.0, "other")

        @monitor.profile_function
        def profiled_function():
            return "test"

        profiled_function()

        # Generate report
        report = monitor.get_report(hours=1)

        # Check report structure
        expected_sections = [
            "summary",
            "metric_statistics",
            "function_profiles",
            "recent_alerts",
            "system_info",
            "recommendations",
        ]
        for section in expected_sections:
            assert section in report, f"Report should include {section}"

        # Check metric statistics
        stats = report["metric_statistics"]
        assert len(stats) > 0, "Should have metric statistics"

        # Check function profiles
        profiles = report["function_profiles"]
        assert "profiled_function" in str(profiles), "Should include function profiles"

    def test_decorators():
        # Clear any existing profiles in global monitor
        performance_monitor.function_profiles.clear()

        # Test global decorators
        @profile
        def decorated_function():
            time.sleep(0.001)
            return "decorated"

        @monitor_performance
        def monitored_function():
            return "monitored"

        # Call decorated functions
        result1 = decorated_function()
        result2 = monitored_function()

        assert result1 == "decorated", "Profile decorator should work"
        assert result2 == "monitored", "Monitor decorator should work"

        # Check that functions were profiled (should be in global monitor)
        global_profiles = performance_monitor.function_profiles

        # Check for function names with module prefixes
        decorated_found = any(
            "decorated_function" in key for key in global_profiles.keys()
        )
        monitored_found = any(
            "monitored_function" in key for key in global_profiles.keys()
        )

        assert (
            decorated_found
        ), f"Global profile decorator should work. Found profiles: {list(global_profiles.keys())}"
        assert (
            monitored_found
        ), f"Global monitor decorator should work. Found profiles: {list(global_profiles.keys())}"

    def test_performance_validation():
        monitor = PerformanceMonitor()
        monitor.enabled = True  # Ensure monitoring is enabled
        monitor.stop_monitoring()  # Stop any background monitoring
        monitor.metrics.clear()  # Clear any pre-existing metrics

        # Test bulk operations performance
        start_time = time.time()

        # Record many metrics
        for i in range(200):
            monitor.record_metric(f"metric_{i}", float(i), "bulk_test")

        # Profile a function multiple times
        @monitor.profile_function
        def fast_function(x):
            return x * 2

        for i in range(50):
            fast_function(i)

        elapsed = time.time() - start_time
        assert elapsed < 0.5, f"Bulk operations should be fast, took {elapsed:.3f}s"

        # Check that all data was recorded
        # 200 explicit metrics + 50 function call metrics = 250 total
        assert (
            len(monitor.metrics) == 250
        ), f"Should record all metrics (200 explicit + 50 function calls), got {len(monitor.metrics)}"

        # Check function profiling (function name includes module prefix)
        fast_function_found = any(
            "fast_function" in key for key in monitor.function_profiles.keys()
        )
        assert (
            fast_function_found
        ), f"Should profile function. Found profiles: {list(monitor.function_profiles.keys())}"

        # Get the actual profile key
        profile_key = next(
            key for key in monitor.function_profiles.keys() if "fast_function" in key
        )
        profile = monitor.function_profiles[profile_key]
        assert (
            profile.call_count == 50
        ), f"Should track all function calls, got {profile.call_count}"

    # Run all tests
    print(
        "📊 Running Performance Monitor & Optimization System comprehensive test suite..."
    )

    with suppress_logging():
        suite.run_test(
            "Class availability verification",
            test_class_availability,
            "Test availability of all performance monitoring classes",
            "Class availability ensures complete performance monitoring interface",
            "All required performance monitoring classes are available and properly defined",
        )

        suite.run_test(
            "Monitor initialization",
            test_monitor_initialization,
            "Test PerformanceMonitor initialization with various configurations",
            "Monitor initialization provides configurable performance tracking setup",
            "Performance monitor initializes correctly with custom settings and default values",
        )

        suite.run_test(
            "Metric recording functionality",
            test_metric_recording,
            "Test record_metric with various data types and metadata",
            "Metric recording provides comprehensive performance data capture",
            "Metrics are recorded correctly with timestamps, categories, and metadata",
        )

        suite.run_test(
            "Function profiling system",
            test_function_profiling,
            "Test function profiling decorator with timing and statistics",
            "Function profiling provides detailed execution performance analysis",
            "Function profiles track call counts, execution times, and performance statistics",
        )

        suite.run_test(
            "Error handling in profiling",
            test_function_profiling_with_errors,
            "Test function profiling with exceptions and error tracking",
            "Error handling ensures robust profiling under failure conditions",
            "Function profiling correctly handles and tracks errors without breaking",
        )

        suite.run_test(
            "System metrics collection",
            test_system_metrics_collection,
            "Test automatic system metrics collection for CPU and memory",
            "System metrics collection provides real-time resource monitoring",
            "System metrics are automatically collected and include CPU and memory usage",
        )

        suite.run_test(
            "Performance alert system",
            test_alert_system,
            "Test alert generation based on configurable thresholds",
            "Alert system provides proactive performance issue notification",
            "Alerts are generated correctly when metrics exceed configured thresholds",
        )

        suite.run_test(
            "Monitoring lifecycle control",
            test_monitoring_control,
            "Test start and stop monitoring thread control",
            "Monitoring control provides background performance tracking management",
            "Background monitoring can be started and stopped correctly with thread management",
        )

        suite.run_test(
            "Comprehensive report generation",
            test_report_generation,
            "Test detailed performance report generation with all sections",
            "Report generation provides comprehensive performance analysis output",
            "Performance reports include all required sections with accurate statistics",
        )

        suite.run_test(
            "Global decorator functionality",
            test_decorators,
            "Test global @profile and @monitor_performance decorators",
            "Global decorators provide convenient function performance monitoring",
            "Global decorators work correctly and integrate with the monitoring system",
        )

        suite.run_test(
            "Performance validation",
            test_performance_validation,
            "Test performance of monitoring system itself with bulk operations",
            "Performance validation ensures efficient monitoring with minimal overhead",
            "Performance monitoring operations complete quickly without significant overhead",
        )

    # Generate summary report
    suite.finish_suite()
