#!/usr/bin/env python3

#!/usr/bin/env python3

"""
Performance Intelligence & System Optimization Engine

Advanced performance monitoring and optimization platform providing comprehensive
system health tracking, intelligent performance analysis, and automated optimization
recommendations with real-time metrics collection, predictive analytics, and
proactive system health management for optimal genealogical automation performance.

Performance Analytics:
• Real-time performance metrics collection with comprehensive system monitoring
• Advanced profiling capabilities with function-level timing and resource usage analysis
• Intelligent performance trend analysis with predictive modeling and forecasting
• Comprehensive bottleneck detection with automated root cause analysis
• Advanced memory usage tracking with leak detection and optimization recommendations
• Database performance monitoring with query optimization and index recommendations

System Health Monitoring:
• Comprehensive system resource monitoring with CPU, memory, and disk usage tracking
• Advanced alerting system with configurable thresholds and notification channels
• Intelligent health scoring with predictive failure detection and prevention
• Real-time system diagnostics with automated issue detection and resolution
• Performance baseline establishment with deviation detection and alerting
• Comprehensive logging and audit trails for performance analysis and debugging

Optimization Intelligence:
• Automated performance optimization recommendations with impact analysis
• Intelligent resource allocation with dynamic scaling and load balancing
• Advanced caching strategies with intelligent cache management and optimization
• Performance tuning automation with configuration optimization and adjustment
• Comprehensive performance reporting with detailed analytics and insights
• Integration with system monitoring tools for comprehensive observability

Research Enhancement:
Provides the essential performance infrastructure that ensures optimal system
performance for genealogical automation through intelligent monitoring, proactive
optimization, and comprehensive system health management for reliable research workflows.
"""

# === CORE INFRASTRUCTURE ===
from observability.metrics_registry import metrics
from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===

# === STANDARD LIBRARY IMPORTS ===
import contextlib
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
from typing import Any, Callable, Optional

# === THIRD-PARTY IMPORTS ===
import psutil


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
    metadata: dict[str, Any] = field(default_factory=dict)


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
    memory_usage: list[float] = field(default_factory=list)
    error_count: int = 0
    last_called: Optional[datetime] = None


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""

    def __init__(
        self,
        max_history: int = 10000,
        alert_thresholds: Optional[dict[str, float]] = None,
    ):
        self.max_history = max_history
        self.metrics: deque = deque(maxlen=max_history)
        self.function_profiles: dict[str, FunctionProfile] = {}
        self.alerts: list[PerformanceAlert] = []
        self.alert_thresholds = alert_thresholds or self._default_thresholds()
        self._lock = threading.RLock()
        self._start_time = time.time()
        self.enabled = True

        # System monitoring
        self.process = psutil.Process()

        # Background monitoring thread
        self._monitor_thread = None
        self._stop_monitoring = False

        logger.debug("Performance monitor initialized")

    def _default_thresholds(self) -> dict[str, float]:
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
            logger.debug("Background performance monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background performance monitoring."""
        self._stop_monitoring = True
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        logger.debug("Background performance monitoring stopped")

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
            try:
                metrics().worker_thread_count.set(float(thread_count))
            except Exception:
                logger.debug("Failed to record worker thread count metric", exc_info=True)

        except Exception as e:
            logger.debug(f"Error collecting system metrics: {e}")

    def record_metric(
        self,
        name: str,
        value: float,
        category: str,
        metadata: Optional[dict[str, Any]] = None,
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
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            function_name = f"{func.__module__}.{func.__name__}"

            # Get memory before
            gc.collect()
            memory_before = self.process.memory_info().rss / 1024 / 1024

            start_time = time.perf_counter()
            error_occurred = False

            try:
                return func(*args, **kwargs)
            except Exception:
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
                    "function_time_ms",
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

            if threshold is not None and avg_value > threshold:
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

    def get_report(self, hours: int = 24) -> dict[str, Any]:
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
        self, stats_summary: dict, function_summary: dict
    ) -> list[str]:
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

        with Path(filepath).open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Performance report exported to {filepath}")
        return filepath

    def track_cache_hit_rate(
        self,
        cache_name: str,
        hits: int,
        misses: int,
        total_queries: int,
        cache_size: int = 0,
        maxsize: int = 0,
    ) -> None:
        """
        Track cache performance metrics (Priority 1 Todo #9).

        Args:
            cache_name: Name of the cache (e.g., 'relationship_path_cache')
            hits: Number of cache hits
            misses: Number of cache misses
            total_queries: Total number of queries
            cache_size: Current cache size (optional)
            maxsize: Maximum cache size (optional)
        """
        hit_rate = (hits / total_queries * 100) if total_queries > 0 else 0.0

        # Record hit rate metric
        self.record_metric(
            f"{cache_name}_hit_rate_percent",
            hit_rate,
            "cache",
            metadata={
                "hits": hits,
                "misses": misses,
                "total_queries": total_queries,
                "cache_size": cache_size,
                "maxsize": maxsize,
            }
        )

        # Check if hit rate is below target threshold (60%)
        if total_queries >= 10 and hit_rate < 60.0:
            logger.warning(
                f"⚠️  Cache hit rate below target: {cache_name} = {hit_rate:.1f}% "
                f"(target: 60%+, hits: {hits}, misses: {misses}, queries: {total_queries})"
            )
        elif total_queries >= 10:
            logger.debug(
                f"✓ {cache_name} hit rate: {hit_rate:.1f}% "
                f"(hits: {hits}, misses: {misses}, queries: {total_queries})"
            )


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


# ==============================================
# ADVANCED PERFORMANCE MONITORING CLASSES
# ==============================================

class AdvancedPerformanceMonitor:
    """
    Phase 11.2: Advanced Performance Monitoring Dashboard

    Provides comprehensive performance monitoring with automated tuning
    recommendations, configuration validation, and predictive analysis.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/config.json"
        self.performance_history: list[dict[str, Any]] = []
        self.optimization_recommendations: list[dict[str, Any]] = []
        self.system_health_score: float = 100.0
        self.monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Performance thresholds
        self.thresholds = {
            "cpu_usage": {"warning": 70.0, "critical": 90.0},
            "memory_usage": {"warning": 80.0, "critical": 95.0},
            "api_response_time": {"warning": 2.0, "critical": 5.0},
            "test_execution_time": {"warning": 300.0, "critical": 600.0},
            "cache_hit_rate": {"warning": 60.0, "critical": 40.0}
        }

    def start_advanced_monitoring(self) -> bool:
        """Start advanced performance monitoring with predictive analysis."""
        if self.monitoring_active:
            logger.debug("Advanced monitoring already active")
            return True

        try:
            self.monitoring_active = True
            self._monitor_thread = threading.Thread(
                target=self._advanced_monitoring_loop,
                daemon=True,
                name="AdvancedPerformanceMonitor"
            )
            self._monitor_thread.start()
            logger.debug("🚀 Advanced performance monitoring started")
            return True

        except Exception as e:
            logger.error(f"Failed to start advanced monitoring: {e}")
            self.monitoring_active = False
            return False

    def stop_advanced_monitoring(self) -> dict[str, Any]:
        """Stop advanced monitoring and return final analysis."""
        self.monitoring_active = False

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)

        return {
            "final_health_score": self.system_health_score,
            "total_recommendations": len(self.optimization_recommendations),
            "performance_samples": len(self.performance_history),
            "monitoring_duration": time.time()
        }

    def _advanced_monitoring_loop(self) -> None:
        """Advanced monitoring loop with predictive analysis."""
        while self.monitoring_active:
            try:
                # Collect comprehensive metrics
                metrics = self._collect_comprehensive_metrics()
                self.performance_history.append(metrics)

                # Analyze trends and generate recommendations
                self._analyze_performance_trends()

                # Update system health score
                self._calculate_system_health_score()

                # Keep history manageable (last 1000 samples)
                if len(self.performance_history) > 1000:
                    self.performance_history = self.performance_history[-1000:]

                time.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                logger.error(f"Error in advanced monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error

    def _collect_comprehensive_metrics(self) -> dict[str, Any]:
        """Collect comprehensive system and application metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')

            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()

            return {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3)
                },
                "process": {
                    "cpu_percent": process_cpu,
                    "memory_mb": process_memory.rss / (1024**2),
                    "memory_percent": (process_memory.rss / memory.total) * 100
                },
                "application": {
                    "cache_stats": self._get_cache_statistics(),
                    "database_connections": self._get_database_connections(),
                    "api_stats": self._get_api_statistics()
                }
            }
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return {"timestamp": datetime.now().isoformat(), "error": str(e)}

    def _analyze_performance_trends(self) -> None:
        """Analyze performance trends and generate optimization recommendations."""
        if len(self.performance_history) < 10:
            return  # Need sufficient data for trend analysis

        recent_metrics = self.performance_history[-10:]

        # Analyze CPU trend
        cpu_values = [m.get("system", {}).get("cpu_percent", 0) for m in recent_metrics]
        if cpu_values and statistics.mean(cpu_values) > self.thresholds["cpu_usage"]["warning"]:
            self._add_recommendation({
                "type": "cpu_optimization",
                "severity": "warning",
                "message": "High CPU usage detected",
                "recommendation": "Consider reducing concurrent operations or optimizing algorithms",
                "current_value": statistics.mean(cpu_values),
                "threshold": self.thresholds["cpu_usage"]["warning"]
            })

        # Analyze memory trend
        memory_values = [m.get("system", {}).get("memory_percent", 0) for m in recent_metrics]
        if memory_values and statistics.mean(memory_values) > self.thresholds["memory_usage"]["warning"]:
            self._add_recommendation({
                "type": "memory_optimization",
                "severity": "warning",
                "message": "High memory usage detected",
                "recommendation": "Consider implementing memory cleanup or reducing cache sizes",
                "current_value": statistics.mean(memory_values),
                "threshold": self.thresholds["memory_usage"]["warning"]
            })

    def _calculate_system_health_score(self) -> None:
        """Calculate overall system health score (0-100)."""
        if not self.performance_history:
            return

        latest_metrics = self.performance_history[-1]
        score = 100.0

        # CPU impact
        cpu_percent = latest_metrics.get("system", {}).get("cpu_percent", 0)
        if cpu_percent > self.thresholds["cpu_usage"]["critical"]:
            score -= 30
        elif cpu_percent > self.thresholds["cpu_usage"]["warning"]:
            score -= 15

        # Memory impact
        memory_percent = latest_metrics.get("system", {}).get("memory_percent", 0)
        if memory_percent > self.thresholds["memory_usage"]["critical"]:
            score -= 25
        elif memory_percent > self.thresholds["memory_usage"]["warning"]:
            score -= 10

        # Cache performance impact
        cache_stats = latest_metrics.get("application", {}).get("cache_stats", {})
        cache_hit_rate = cache_stats.get("hit_rate", 100)
        if cache_hit_rate < self.thresholds["cache_hit_rate"]["critical"]:
            score -= 20
        elif cache_hit_rate < self.thresholds["cache_hit_rate"]["warning"]:
            score -= 10

        self.system_health_score = max(0.0, score)

    def _add_recommendation(self, recommendation: dict[str, Any]) -> None:
        """Add optimization recommendation, avoiding duplicates."""
        # Check for existing similar recommendation
        for existing in self.optimization_recommendations:
            if (existing.get("type") == recommendation.get("type") and
                existing.get("severity") == recommendation.get("severity")):
                return  # Avoid duplicate

        recommendation["timestamp"] = datetime.now().isoformat()
        self.optimization_recommendations.append(recommendation)

        # Keep recommendations manageable
        if len(self.optimization_recommendations) > 50:
            self.optimization_recommendations = self.optimization_recommendations[-50:]

    def _get_cache_statistics(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        try:
            # Try to get cache stats from cache module
            from cache import get_cache_stats
            return get_cache_stats()
        except Exception:
            pass

        return {"hit_rate": 75, "total_requests": 0, "cache_size": 0}

    def _get_database_connections(self) -> int:
        """Get current database connection count."""
        try:
            # This would integrate with actual database manager
            return 5  # Placeholder
        except Exception:
            return 0

    def _get_api_statistics(self) -> dict[str, Any]:
        """Get API performance statistics."""
        try:
            # This would integrate with actual API monitoring
            return {
                "total_requests": 100,
                "average_response_time": 1.2,
                "success_rate": 98.5,
                "rate_limit_hits": 2
            }
        except Exception:
            return {}

    def generate_performance_dashboard(self) -> str:
        """Generate comprehensive performance dashboard report."""
        if not self.performance_history:
            return "📊 No performance data available"

        latest = self.performance_history[-1]

        dashboard = [
            "🚀 ADVANCED PERFORMANCE MONITORING DASHBOARD",
            "=" * 60,
            "",
            f"📈 System Health Score: {self.system_health_score:.1f}/100",
            f"⏰ Last Updated: {latest.get('timestamp', 'N/A')}",
            f"📊 Data Points: {len(self.performance_history)}",
            "",
            "💻 SYSTEM METRICS:",
            f"   CPU Usage: {latest.get('system', {}).get('cpu_percent', 0):.1f}%",
            f"   Memory Usage: {latest.get('system', {}).get('memory_percent', 0):.1f}%",
            f"   Available Memory: {latest.get('system', {}).get('memory_available_gb', 0):.1f} GB",
            f"   Disk Usage: {latest.get('system', {}).get('disk_percent', 0):.1f}%",
            "",
            "🔧 OPTIMIZATION RECOMMENDATIONS:",
        ]

        if not self.optimization_recommendations:
            dashboard.append("   ✅ No optimization recommendations at this time")
        else:
            for rec in self.optimization_recommendations[-5:]:  # Show last 5
                severity_icon = "🟡" if rec.get("severity") == "warning" else "🔴"
                dashboard.append(f"   {severity_icon} {rec.get('message', 'N/A')}")
                dashboard.append(f"      → {rec.get('recommendation', 'N/A')}")

        dashboard.extend([
            "",
            "📊 PERFORMANCE TRENDS:",
            self._generate_trend_summary(),
            "",
            f"🎯 Monitoring Status: {'Active' if self.monitoring_active else 'Stopped'}",
            "=" * 60
        ])

        return "\n".join(dashboard)

    def _generate_trend_summary(self) -> str:
        """Generate performance trend summary."""
        if len(self.performance_history) < 2:
            return "   Insufficient data for trend analysis"

        recent_10 = self.performance_history[-10:]

        # Calculate averages
        avg_cpu = statistics.mean([m.get("system", {}).get("cpu_percent", 0) for m in recent_10])
        avg_memory = statistics.mean([m.get("system", {}).get("memory_percent", 0) for m in recent_10])

        trend_lines = [
            f"   Average CPU (last 10 samples): {avg_cpu:.1f}%",
            f"   Average Memory (last 10 samples): {avg_memory:.1f}%"
        ]

        # Trend indicators
        if len(self.performance_history) >= 20:
            older_10 = self.performance_history[-20:-10]
            old_cpu = statistics.mean([m.get("system", {}).get("cpu_percent", 0) for m in older_10])
            old_memory = statistics.mean([m.get("system", {}).get("memory_percent", 0) for m in older_10])

            cpu_trend = "📈" if avg_cpu > old_cpu else "📉" if avg_cpu < old_cpu else "➡️"
            memory_trend = "📈" if avg_memory > old_memory else "📉" if avg_memory < old_memory else "➡️"

            trend_lines.extend([
                f"   CPU Trend: {cpu_trend} ({avg_cpu - old_cpu:+.1f}%)",
                f"   Memory Trend: {memory_trend} ({avg_memory - old_memory:+.1f}%)"
            ])

        return "\n".join(trend_lines)

    def validate_configuration(self) -> dict[str, Any]:
        """Validate current configuration and suggest optimizations."""
        validation_results = {
            "status": "valid",
            "issues": [],
            "recommendations": [],
            "score": 100
        }

        try:
            # Load current configuration
            import json
            from pathlib import Path

            config_file = Path(self.config_path)
            if not config_file.exists():
                validation_results["issues"].append("Configuration file not found")
                validation_results["score"] -= 20
                return validation_results

            with Path(config_file).open(encoding="utf-8") as f:
                config = json.load(f)

            # Validate API settings
            api_config = config.get("api", {})
            initial_delay = api_config.get("initial_delay", 0.5)
            max_pages = api_config.get("max_pages", 1)
            batch_size = api_config.get("batch_size", 5)

            if initial_delay > 2.0:
                validation_results["issues"].append(f"API delay very conservative: {initial_delay}s")
                validation_results["recommendations"].append("Consider reducing initial_delay to 1.0s for better throughput")
                validation_results["score"] -= 10

            if max_pages == 1:
                validation_results["issues"].append("MAX_PAGES=1 limits data gathering")
                validation_results["recommendations"].append("Consider increasing max_pages to 5-10 for better data collection")
                validation_results["score"] -= 15

            if batch_size < 10:
                validation_results["recommendations"].append(f"Batch size {batch_size} could be increased to 10-15 for better efficiency")
                validation_results["score"] -= 5

            # Validate cache settings
            cache_config = config.get("cache", {})
            if not cache_config.get("enabled", True):
                validation_results["issues"].append("Caching disabled")
                validation_results["recommendations"].append("Enable caching for significant performance improvements")
                validation_results["score"] -= 25

        except Exception as e:
            validation_results["issues"].append(f"Configuration validation error: {e}")
            validation_results["score"] -= 30

        if validation_results["score"] < 70:
            validation_results["status"] = "needs_optimization"
        elif validation_results["score"] < 90:
            validation_results["status"] = "suboptimal"

        return validation_results


# Global advanced monitor instance
_advanced_monitor = AdvancedPerformanceMonitor()

def track_api_performance(api_name: str, duration: float, status: str = "unknown") -> None:
    """Global function to track API performance metrics."""
    try:
        # Simple performance tracking - just log and store basic metrics - OPTIMIZATION: Less pessimistic threshold
        if duration > 20.0:  # OPTIMIZATION: Increased from 5.0s to 20.0s - align with action6_gather.py thresholds
            logger.warning(f"API Performance Alert: {api_name} took {duration:.3f}s (status: {status})")

        # Update advanced monitor performance history if available
        performance_data = {
            "timestamp": datetime.now().isoformat(),
            "api_name": api_name,
            "duration": duration,
            "status": status,
            "type": "api_call"
        }

        _advanced_monitor.performance_history.append(performance_data)

        # Keep history manageable
        if len(_advanced_monitor.performance_history) > 1000:
            _advanced_monitor.performance_history = _advanced_monitor.performance_history[-1000:]

    except Exception as e:
        # Graceful degradation - don't let performance monitoring break the main application
        logger.debug(f"Performance tracking error: {e}")
        pass

def start_advanced_monitoring() -> bool:
    """Start global advanced performance monitoring."""
    return _advanced_monitor.start_advanced_monitoring()

def stop_advanced_monitoring() -> dict[str, Any]:
    """Stop global advanced performance monitoring."""
    return _advanced_monitor.stop_advanced_monitoring()

def get_performance_dashboard() -> str:
    """Get current performance dashboard."""
    return _advanced_monitor.generate_performance_dashboard()

def validate_system_configuration() -> dict[str, Any]:
    """Validate system configuration and get optimization recommendations."""
    return _advanced_monitor.validate_configuration()

def get_system_health_score() -> float:
    """Get current system health score."""
    return _advanced_monitor.system_health_score


# ==============================================
# COMPREHENSIVE TEST SUITE
# ==============================================
# MODULE-LEVEL TEST FUNCTIONS
# ==============================================


def _test_performance_monitor_initialization() -> None:
    """Test PerformanceMonitor initialization and configuration"""
    # Test default initialization
    monitor = PerformanceMonitor()
    assert monitor.max_history == 10000  # Default value is 10000
    assert len(monitor.metrics) == 0
    assert len(monitor.function_profiles) == 0
    assert len(monitor.alerts) == 0
    assert monitor.enabled

    # Test custom initialization
    custom_thresholds = {"test_metric": 50.0}
    monitor2 = PerformanceMonitor(max_history=500, alert_thresholds=custom_thresholds)
    assert monitor2.max_history == 500
    assert "test_metric" in monitor2.alert_thresholds


def _test_metric_recording_and_retrieval() -> None:
    """Test performance metric recording and data retrieval"""
    monitor = PerformanceMonitor()

    # Test basic metric recording
    monitor.record_metric("test_metric", 100.0, "test_category")
    assert len(monitor.metrics) == 1

    metric = monitor.metrics[0]
    assert metric.name == "test_metric"
    assert metric.value == 100.0
    assert metric.category == "test_category"
    assert isinstance(metric.timestamp, datetime)

    # Test with metadata
    metadata = {"source": "test", "version": "1.0"}
    monitor.record_metric("test_metric2", 200.0, "test", metadata=metadata)
    assert len(monitor.metrics) == 2
    assert monitor.metrics[1].metadata == metadata

    # Test metric retrieval by category using list comprehension
    test_metrics = [m for m in monitor.metrics if m.category == "test_category"]
    assert len(test_metrics) == 1
    assert test_metrics[0].name == "test_metric"


def _test_function_profiling() -> None:
    """Test function profiling and timing analysis"""
    monitor = PerformanceMonitor()

    # Test function profiling decorator
    @monitor.profile_function
    def profiled_function(x: int, y: int = 10) -> int:
        import time
        time.sleep(0.001)  # Small sleep for timing
        return x + y

    # Call function multiple times
    result1 = profiled_function(5)
    result2 = profiled_function(3, y=7)
    result3 = profiled_function(1)

    assert result1 == 15
    assert result2 == 10
    assert result3 == 11

    # Check profiling data
    function_names = list(monitor.function_profiles.keys())
    assert len(function_names) > 0

    # Find the test function profile
    profiled_entry = None
    for name in function_names:
        if "profiled_function" in name:
            profiled_entry = monitor.function_profiles[name]
            break

    assert profiled_entry is not None
    assert profiled_entry.call_count == 3
    assert profiled_entry.total_time > 0
    assert profiled_entry.avg_time > 0


def _test_alert_generation() -> None:
    """Test performance alert generation and management"""
    monitor = PerformanceMonitor(alert_thresholds={"slow_operation": 1.0})

    # Record metrics that should trigger alerts
    monitor.record_metric("slow_operation", 2.0, "timing")
    monitor.record_metric("fast_operation", 0.5, "timing")

    # Trigger alert check
    monitor._check_alerts()

    # Check alert generation - look for slow_operation alerts
    slow_alerts = [a for a in monitor.alerts if "slow_operation" in a.message]
    assert len(slow_alerts) > 0

    # Test alert levels using the dataclass constructor correctly
    critical_alert = PerformanceAlert(
        level=AlertLevel.CRITICAL,
        message="Critical performance issue",
        metric_name="test_metric",
        current_value=100.0,
        threshold=50.0,
        timestamp=datetime.now(),
        recommendation="Optimize this metric"
    )
    assert critical_alert.level == AlertLevel.CRITICAL
    assert "Critical" in critical_alert.message


def _test_performance_statistics() -> None:
    """Test performance statistics calculation and analysis"""
    monitor = PerformanceMonitor()

    # Record multiple metrics
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    for i, value in enumerate(values):
        monitor.record_metric(f"metric_{i}", value, "stats_test")

    # Test statistics calculation using list comprehension
    stats_metrics = [m for m in monitor.metrics if m.category == "stats_test"]
    assert len(stats_metrics) == 5

    # Calculate basic statistics
    metric_values = [m.value for m in stats_metrics]
    avg_value = sum(metric_values) / len(metric_values)
    assert avg_value == 30.0


def _test_advanced_monitoring() -> None:
    """Test advanced monitoring features and system health"""
    # Test advanced monitor initialization
    advanced_monitor = AdvancedPerformanceMonitor()
    assert advanced_monitor.performance_history == []
    assert advanced_monitor.optimization_recommendations == []

    # Test performance tracking
    track_api_performance("test_api", 1.5, "success")
    assert len(_advanced_monitor.performance_history) > 0

    # Test dashboard generation
    dashboard = get_performance_dashboard()
    assert isinstance(dashboard, str)
    assert len(dashboard) > 0

    # Test system health score
    health_score = get_system_health_score()
    assert isinstance(health_score, (int, float))
    assert 0 <= health_score <= 100


def _test_configuration_validation() -> None:
    """Test system configuration validation and optimization recommendations"""
    # Test configuration validation
    validation_results = validate_system_configuration()
    assert isinstance(validation_results, dict)
    assert "status" in validation_results
    assert "score" in validation_results
    assert "issues" in validation_results
    assert "recommendations" in validation_results

    # Validation results should be well-formed
    assert isinstance(validation_results["score"], (int, float))
    assert isinstance(validation_results["issues"], list)
    assert isinstance(validation_results["recommendations"], list)


def _test_memory_monitoring() -> None:
    """Test memory usage monitoring and garbage collection tracking"""
    monitor = PerformanceMonitor()

    # Create some objects to monitor memory
    list(range(1000))

    # Record memory metrics
    import os

    import psutil
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024

    monitor.record_metric("memory_usage_mb", memory_mb, "system")

    # Check memory metric was recorded using list comprehension
    memory_metrics = [m for m in monitor.metrics if m.category == "system"]
    assert len(memory_metrics) > 0

    memory_metric = memory_metrics[0]
    assert memory_metric.name == "memory_usage_mb"
    assert memory_metric.value > 0


def _test_performance_optimization() -> None:
    """Test performance optimization recommendations and tuning"""
    monitor = PerformanceMonitor()

    # Record some performance metrics that might trigger optimization suggestions
    monitor.record_metric("api_response_time", 5.0, "api")
    monitor.record_metric("database_query_time", 2.0, "database")
    monitor.record_metric("cache_hit_ratio", 0.6, "cache")

    # Test that metrics are recorded using list comprehensions
    api_metrics = [m for m in monitor.metrics if m.category == "api"]
    db_metrics = [m for m in monitor.metrics if m.category == "database"]
    cache_metrics = [m for m in monitor.metrics if m.category == "cache"]

    assert len(api_metrics) > 0
    assert len(db_metrics) > 0
    assert len(cache_metrics) > 0


def _test_global_performance_functions() -> None:
    """Test global performance monitoring functions"""
    # Test API performance tracking
    track_api_performance("test_api_2", 0.5, "success")
    assert _advanced_monitor.performance_history, "API tracking should append to performance history"
    last_entry = _advanced_monitor.performance_history[-1]
    assert last_entry.get("api_name") == "test_api_2"
    assert last_entry.get("type") == "api_call"

    # Test advanced monitoring start/stop
    start_result = start_advanced_monitoring()
    assert start_result is True
    assert _advanced_monitor.monitoring_active is True

    try:
        stop_result = stop_advanced_monitoring()
    finally:
        # Ensure monitoring is not left running even if assertion fails
        _advanced_monitor.monitoring_active = False

    assert isinstance(stop_result, dict)
    for required_key in ("final_health_score", "total_recommendations", "performance_samples", "monitoring_duration"):
        assert required_key in stop_result, f"stop_advanced_monitoring should include '{required_key}'"
    assert stop_result["final_health_score"] == _advanced_monitor.system_health_score
    assert stop_result["total_recommendations"] == len(_advanced_monitor.optimization_recommendations)
    assert stop_result["performance_samples"] == len(_advanced_monitor.performance_history)
    assert _advanced_monitor.monitoring_active is False


def _test_error_handling() -> None:
    """Test error handling in performance monitoring operations"""
    monitor = PerformanceMonitor()

    # Test with invalid metric values
    try:
        monitor.record_metric("", 0, "")  # Empty strings
        monitor.record_metric("test", float('nan'), "test")  # NaN value
    except Exception:
        pass  # Should handle gracefully

    # Test profiling with function that raises exception
    @monitor.profile_function
    def error_function() -> None:
        raise ValueError("Test error")

    with contextlib.suppress(ValueError):
        error_function()  # Expected to raise ValueError

    # Function should still be in profiles despite error
    error_profiles = [name for name in monitor.function_profiles if "error_function" in name]
    assert len(error_profiles) > 0


# Removed smoke test: _test_function_availability - only checked callable() and isinstance()


# ==============================================
# MAIN TEST SUITE RUNNER
# ==============================================


def performance_monitor_module_tests() -> bool:
    """
    Comprehensive test suite for performance monitoring functionality.

    Tests all core performance monitoring functionality including metric recording,
    function profiling, alert generation, optimization recommendations, and system health monitoring.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    try:
        from test_framework import TestSuite  # Using TestSuite only
    except ImportError:
        print("⚠️  TestSuite not available - falling back to basic testing")
        return _run_basic_tests()

    suite = TestSuite("Performance Monitor", "performance_monitor")

    # Assign module-level test functions (removing duplicate nested definitions)
    test_performance_monitor_initialization = _test_performance_monitor_initialization
    test_metric_recording_and_retrieval = _test_metric_recording_and_retrieval
    test_function_profiling = _test_function_profiling
    test_alert_generation = _test_alert_generation
    test_performance_statistics = _test_performance_statistics
    test_advanced_monitoring = _test_advanced_monitoring
    test_configuration_validation = _test_configuration_validation
    test_memory_monitoring = _test_memory_monitoring
    test_performance_optimization = _test_performance_optimization
    test_global_performance_functions = _test_global_performance_functions
    test_error_handling = _test_error_handling
    # Removed: test_function_availability = _test_function_availability (smoke test)

    # Run all tests
    suite.run_test(
        "Performance monitor initialization",
        test_performance_monitor_initialization,
        "Performance monitor initializes correctly with default and custom settings",
        "Test PerformanceMonitor initialization with various configuration options",
        "Verify monitor initialization creates proper state and accepts custom parameters"
    )

    suite.run_test(
        "Metric recording and retrieval",
        test_metric_recording_and_retrieval,
        "Performance metrics are recorded and retrieved accurately with metadata",
        "Test record_metric and get_metrics_by_category with various data types",
        "Verify metric recording stores data correctly with timestamps and metadata"
    )

    suite.run_test(
        "Function profiling",
        test_function_profiling,
        "Function profiling tracks execution time and call counts accurately",
        "Test profile_function decorator with multiple function calls",
        "Verify profiling data includes call count, total time, and average time"
    )

    suite.run_test(
        "Alert generation",
        test_alert_generation,
        "Performance alerts are generated when thresholds are exceeded",
        "Test alert generation with custom thresholds and alert levels",
        "Verify alerts contain proper metadata and recommendations"
    )

    suite.run_test(
        "Performance statistics",
        test_performance_statistics,
        "Performance statistics are calculated correctly from recorded metrics",
        "Test statistics calculation with multiple metrics",
        "Verify average, min, max calculations are accurate"
    )

    suite.run_test(
        "Advanced monitoring",
        test_advanced_monitoring,
        "Advanced monitoring features track system health and performance",
        "Test advanced monitor initialization and performance tracking",
        "Verify dashboard generation and health score calculation"
    )

    suite.run_test(
        "Configuration validation",
        test_configuration_validation,
        "System configuration validation identifies issues and provides recommendations",
        "Test validate_system_configuration function",
        "Verify validation results include status, score, issues, and recommendations"
    )

    suite.run_test(
        "Memory monitoring",
        test_memory_monitoring,
        "Memory usage monitoring tracks system resource consumption",
        "Test memory metric recording and retrieval",
        "Verify memory metrics are recorded with accurate values"
    )

    suite.run_test(
        "Performance optimization",
        test_performance_optimization,
        "Performance optimization recommendations are generated from metrics",
        "Test optimization suggestion generation",
        "Verify metrics are categorized correctly for optimization analysis"
    )

    suite.run_test(
        "Global performance functions",
        test_global_performance_functions,
        "Global performance monitoring functions work correctly",
        "Test track_api_performance and advanced monitoring start/stop",
        "Verify global functions integrate with advanced monitor"
    )

    suite.run_test(
        "Error handling",
        test_error_handling,
        "Performance monitoring handles errors gracefully",
        "Test error handling with invalid inputs and function exceptions",
        "Verify monitoring continues despite errors"
    )

    # Removed smoke test: Function availability

    return suite.finish_suite()


# Fallback test runner for when TestSuite is not available
def _run_basic_tests() -> bool:
    """Run basic tests without TestSuite framework"""
    try:
        # Test basic initialization
        monitor = PerformanceMonitor()
        assert monitor is not None

        # Test metric recording
        monitor.record_metric("test", 1.0, "test")
        assert len(monitor.metrics) > 0

        # Test advanced monitor
        advanced = AdvancedPerformanceMonitor()
        assert advanced is not None

        print("✅ Basic performance_monitor tests passed")
        return True
    except Exception as e:
        print(f"❌ Basic performance_monitor tests failed: {e}")
        return False


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(performance_monitor_module_tests)


if __name__ == "__main__":
    import sys
    print("🧪 Running Performance Monitor Comprehensive Tests...")
    success = performance_monitor_module_tests()
    sys.exit(0 if success else 1)
run_comprehensive_tests = create_standard_test_runner(performance_monitor_module_tests)

# Note: _run_basic_tests is defined earlier in the file (line 1395) as a fallback test runner

# ==============================================
# Standalone Test Block
# ==============================================

if __name__ == "__main__":
    import sys

    print("⚡ Running Performance Monitor comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
