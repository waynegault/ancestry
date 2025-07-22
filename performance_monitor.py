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

from core_imports import standardize_module_imports, auto_register_module

standardize_module_imports()
auto_register_module(globals(), __name__)

import time
import threading
import psutil
import gc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
import json
import statistics
from enum import Enum
from collections import defaultdict, deque

from logging_config import logger


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
                "info": len([a for a in recent_alerts if a.level == AlertLevel.INFO]),
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
                "python_version": psutil.sys.version,
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "disk_total_gb": psutil.disk_usage(str(Path.cwd())).total
                / 1024
                / 1024
                / 1024,
            },
        }

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


# Auto-start monitoring
performance_monitor.start_monitoring()


# Test suite integration
def run_comprehensive_tests() -> bool:
    """Run comprehensive tests for the performance monitor."""
    from test_framework import suppress_logging, create_mock_data, assert_valid_function

    with suppress_logging():
        print("Running Performance Monitor Tests...")
        tests_passed = 0
        total_tests = 0

        # Test 1: Basic functionality
        try:
            monitor = PerformanceMonitor()
            monitor.record_metric("test_metric", 100.0, "test")
            assert len(monitor.metrics) == 1
            tests_passed += 1
            print("✅ Basic functionality - Metric recording works correctly")
        except Exception as e:
            print(f"❌ Basic functionality - Error: {e}")
        total_tests += 1

        # Test 2: Function profiling
        try:

            @monitor.profile_function
            def test_function():
                time.sleep(0.01)
                return "test"

            result = test_function()
            assert result == "test"
            assert "test_function" in str(monitor.function_profiles)
            tests_passed += 1
            print("✅ Function profiling - Decorator works correctly")
        except Exception as e:
            print(f"❌ Function profiling - Error: {e}")
        total_tests += 1

        # Test 3: Report generation
        try:
            report = monitor.get_report(1)
            assert "metric_statistics" in report
            assert "function_profiles" in report
            assert "system_info" in report
            tests_passed += 1
            print("✅ Report generation - Comprehensive report generated")
        except Exception as e:
            print(f"❌ Report generation - Error: {e}")
        total_tests += 1

        print(f"\nPerformance Monitor Tests: {tests_passed}/{total_tests} passed")
        return tests_passed == total_tests


if __name__ == "__main__":
    import sys

    print("📊 Running Performance Monitor comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
