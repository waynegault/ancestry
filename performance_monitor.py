#!/usr/bin/env python3
"""
Performance Monitoring and Metrics Collection for Ancestry.com Automation System
Provides comprehensive performance tracking, metrics collection, and health monitoring.
"""

import time
import threading
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from pathlib import Path
from logging_config import logger


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""

    timestamp: datetime
    duration: float
    success: bool
    error_type: Optional[str] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None


@dataclass
class ServiceMetrics:
    """Aggregated metrics for a service."""

    name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration: float = 0.0
    min_duration: float = float("inf")
    max_duration: float = 0.0
    avg_duration: float = 0.0
    last_call_time: Optional[datetime] = None
    error_counts: Dict[str, int] = field(default_factory=dict)
    recent_metrics: deque = field(default_factory=lambda: deque(maxlen=100))


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for tracking application metrics.
    """

    def __init__(self, metrics_file: str = "performance_metrics.json"):
        self.metrics_file = Path(metrics_file)
        self.services: Dict[str, ServiceMetrics] = {}
        self.system_metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._start_time = datetime.now()
        self._collection_enabled = True

        # Start background system monitoring
        self._start_system_monitoring()

    def _start_system_monitoring(self):
        """Start background thread for system metrics collection."""

        def monitor_system():
            while self._collection_enabled:
                try:
                    self._collect_system_metrics()
                    time.sleep(10)  # Collect every 10 seconds
                except Exception as e:
                    logger.error(f"Error collecting system metrics: {e}")
                    time.sleep(30)  # Wait longer on error

        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()

    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            process = psutil.Process()

            with self._lock:
                self.system_metrics.update(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "cpu_percent": psutil.cpu_percent(interval=1),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_usage": psutil.disk_usage("/").percent,
                        "process_memory_mb": process.memory_info().rss / 1024 / 1024,
                        "process_cpu_percent": process.cpu_percent(),
                        "thread_count": threading.active_count(),
                        "open_files": len(process.open_files()),
                        "uptime_seconds": (
                            datetime.now() - self._start_time
                        ).total_seconds(),
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")

    def record_operation(
        self,
        service_name: str,
        duration: float,
        success: bool,
        error_type: Optional[str] = None,
    ):
        """Record an operation's performance metrics."""
        with self._lock:
            if service_name not in self.services:
                self.services[service_name] = ServiceMetrics(name=service_name)

            service = self.services[service_name]

            # Create metric data point
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                duration=duration,
                success=success,
                error_type=error_type,
            )

            # Update aggregated metrics
            service.total_calls += 1
            service.last_call_time = metric.timestamp

            if success:
                service.successful_calls += 1
            else:
                service.failed_calls += 1
                if error_type:
                    service.error_counts[error_type] = (
                        service.error_counts.get(error_type, 0) + 1
                    )

            # Update duration statistics
            service.total_duration += duration
            service.min_duration = min(service.min_duration, duration)
            service.max_duration = max(service.max_duration, duration)
            service.avg_duration = service.total_duration / service.total_calls

            # Store recent metric
            service.recent_metrics.append(metric)

    def get_service_metrics(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific service."""
        with self._lock:
            if service_name not in self.services:
                return None

            service = self.services[service_name]
            success_rate = (
                (service.successful_calls / service.total_calls)
                if service.total_calls > 0
                else 0
            )

            return {
                "name": service.name,
                "total_calls": service.total_calls,
                "successful_calls": service.successful_calls,
                "failed_calls": service.failed_calls,
                "success_rate": success_rate,
                "avg_duration": service.avg_duration,
                "min_duration": (
                    service.min_duration if service.min_duration != float("inf") else 0
                ),
                "max_duration": service.max_duration,
                "total_duration": service.total_duration,
                "last_call_time": (
                    service.last_call_time.isoformat()
                    if service.last_call_time
                    else None
                ),
                "error_counts": dict(service.error_counts),
                "recent_call_count": len(service.recent_metrics),
            }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all performance metrics."""
        with self._lock:
            service_metrics = {}
            for service_name in self.services:
                service_metrics[service_name] = self.get_service_metrics(service_name)

            return {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
                "system_metrics": dict(self.system_metrics),
                "service_metrics": service_metrics,
                "total_services": len(self.services),
                "memory_objects": len(gc.get_objects()),
            }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of key performance indicators."""
        metrics = self.get_all_metrics()

        # Calculate overall statistics
        total_calls = sum(s["total_calls"] for s in metrics["service_metrics"].values())
        total_failures = sum(
            s["failed_calls"] for s in metrics["service_metrics"].values()
        )
        overall_success_rate = (
            ((total_calls - total_failures) / total_calls) if total_calls > 0 else 0
        )

        # Find slowest and fastest services
        slowest_service = None
        fastest_service = None
        slowest_avg = 0
        fastest_avg = float("inf")

        for service_data in metrics["service_metrics"].values():
            if service_data["total_calls"] > 0:
                avg = service_data["avg_duration"]
                if avg > slowest_avg:
                    slowest_avg = avg
                    slowest_service = service_data["name"]
                if avg < fastest_avg:
                    fastest_avg = avg
                    fastest_service = service_data["name"]

        return {
            "timestamp": metrics["timestamp"],
            "uptime_hours": round(metrics["uptime_seconds"] / 3600, 2),
            "total_operations": total_calls,
            "overall_success_rate": round(overall_success_rate * 100, 2),
            "total_services_monitored": metrics["total_services"],
            "system_memory_percent": metrics["system_metrics"].get("memory_percent", 0),
            "system_cpu_percent": metrics["system_metrics"].get("cpu_percent", 0),
            "process_memory_mb": round(
                metrics["system_metrics"].get("process_memory_mb", 0), 2
            ),
            "slowest_service": slowest_service,
            "slowest_avg_duration": round(slowest_avg, 3) if slowest_service else None,
            "fastest_service": fastest_service,
            "fastest_avg_duration": round(fastest_avg, 3) if fastest_service else None,
        }

    def save_metrics(self):
        """Save metrics to file."""
        try:
            metrics = self.get_all_metrics()
            with open(self.metrics_file, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            logger.debug(f"Metrics saved to {self.metrics_file}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def load_metrics(self) -> Optional[Dict[str, Any]]:
        """Load metrics from file."""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
        return None

    def reset_metrics(self):
        """Reset all metrics."""
        with self._lock:
            self.services.clear()
            self.system_metrics.clear()
            self._start_time = datetime.now()
        logger.info("All metrics reset")

    def stop_monitoring(self):
        """Stop background monitoring."""
        self._collection_enabled = False


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def monitor_performance(service_name: str):
    """Decorator to monitor function performance."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            error_type = None

            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error_type = type(e).__name__
                raise
            finally:
                duration = time.time() - start_time
                performance_monitor.record_operation(
                    service_name, duration, success, error_type
                )

        return wrapper

    return decorator


class HealthChecker:
    """
    Health check system for monitoring system components and dependencies.
    """

    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.last_results: Dict[str, Dict[str, Any]] = {}

    def register_health_check(self, name: str, check_function: Callable):
        """Register a health check function."""
        self.health_checks[name] = check_function
        logger.info(f"Registered health check: {name}")

    def run_health_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if name not in self.health_checks:
            return {
                "status": "error",
                "message": f"Health check '{name}' not found",
                "timestamp": datetime.now().isoformat(),
            }

        start_time = time.time()
        try:
            check_function = self.health_checks[name]
            result = check_function()

            # Ensure result is in expected format
            if not isinstance(result, dict):
                result = {"status": "healthy" if result else "unhealthy"}

            result.update(
                {
                    "check_duration": time.time() - start_time,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            self.last_results[name] = result
            return result

        except Exception as e:
            result = {
                "status": "error",
                "message": str(e),
                "check_duration": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
            }
            self.last_results[name] = result
            return result

    def run_all_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered health checks."""
        results = {}
        for name in self.health_checks:
            results[name] = self.run_health_check(name)
        return results

    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        results = self.run_all_health_checks()

        healthy_count = sum(1 for r in results.values() if r.get("status") == "healthy")
        total_count = len(results)

        overall_status = "healthy" if healthy_count == total_count else "degraded"
        if healthy_count == 0:
            overall_status = "unhealthy"

        return {
            "overall_status": overall_status,
            "healthy_checks": healthy_count,
            "total_checks": total_count,
            "health_percentage": (
                round((healthy_count / total_count) * 100, 2) if total_count > 0 else 0
            ),
            "timestamp": datetime.now().isoformat(),
            "individual_results": results,
        }


# Global health checker instance
health_checker = HealthChecker()


# Standard health check functions
def database_health_check() -> Dict[str, Any]:
    """Check database connectivity and performance."""
    try:
        from database import get_db_connection

        start_time = time.time()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()

        if result and result[0] == 1:
            return {
                "status": "healthy",
                "response_time": time.time() - start_time,
                "message": "Database connection successful",
            }
        else:
            return {
                "status": "unhealthy",
                "message": "Database query returned unexpected result",
            }
    except Exception as e:
        return {"status": "unhealthy", "message": f"Database connection failed: {e}"}


def memory_health_check() -> Dict[str, Any]:
    """Check system memory usage."""
    try:
        memory = psutil.virtual_memory()
        process = psutil.Process()

        memory_threshold = 85  # Alert if memory usage > 85%
        process_memory_mb = process.memory_info().rss / 1024 / 1024

        status = "healthy"
        message = (
            f"Memory usage: {memory.percent:.1f}%, Process: {process_memory_mb:.1f}MB"
        )

        if memory.percent > memory_threshold:
            status = "degraded"
            message += f" (HIGH - above {memory_threshold}% threshold)"

        return {
            "status": status,
            "message": message,
            "system_memory_percent": memory.percent,
            "process_memory_mb": process_memory_mb,
            "available_memory_gb": memory.available / 1024 / 1024 / 1024,
        }
    except Exception as e:
        return {"status": "error", "message": f"Memory check failed: {e}"}


def disk_health_check() -> Dict[str, Any]:
    """Check disk space usage."""
    try:
        disk = psutil.disk_usage("/")

        disk_threshold = 90  # Alert if disk usage > 90%
        status = "healthy"
        message = f"Disk usage: {disk.percent:.1f}%"

        if disk.percent > disk_threshold:
            status = "degraded"
            message += f" (HIGH - above {disk_threshold}% threshold)"

        return {
            "status": status,
            "message": message,
            "disk_percent": disk.percent,
            "free_space_gb": disk.free / 1024 / 1024 / 1024,
        }
    except Exception as e:
        return {"status": "error", "message": f"Disk check failed: {e}"}


# Register standard health checks
health_checker.register_health_check("database", database_health_check)
health_checker.register_health_check("memory", memory_health_check)
health_checker.register_health_check("disk", disk_health_check)


def self_test() -> bool:
    """Test the performance monitoring functionality."""
    logger.info("Starting performance monitoring self-test...")

    try:
        # Test performance monitoring
        @monitor_performance("test_service")
        def test_function(should_fail=False):
            time.sleep(0.1)  # Simulate work
            if should_fail:
                raise ValueError("Test error")
            return "success"

        # Test successful operations
        for i in range(5):
            test_function()

        # Test failed operations
        for i in range(2):
            try:
                test_function(should_fail=True)
            except ValueError:
                pass

        # Check metrics
        metrics = performance_monitor.get_service_metrics("test_service")
        if not metrics:
            logger.error("No metrics found for test service")
            return False

        if metrics["total_calls"] != 7:
            logger.error(f"Expected 7 calls, got {metrics['total_calls']}")
            return False

        if metrics["successful_calls"] != 5:
            logger.error(f"Expected 5 successes, got {metrics['successful_calls']}")
            return False

        # Test health checks
        health_results = health_checker.run_all_health_checks()
        if not health_results:
            logger.error("No health check results")
            return False

        overall_health = health_checker.get_overall_health()
        if "overall_status" not in overall_health:
            logger.error("Missing overall health status")
            return False

        # Test performance summary
        summary = performance_monitor.get_performance_summary()
        if "total_operations" not in summary:
            logger.error("Missing performance summary data")
            return False

        logger.info("Performance monitoring self-test passed successfully")
        return True

    except Exception as e:
        logger.error(f"Performance monitoring self-test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = self_test()

    # Display sample metrics
    if success:
        print("\n=== Performance Summary ===")
        summary = performance_monitor.get_performance_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")

        print("\n=== Health Status ===")
        health = health_checker.get_overall_health()
        print(f"Overall Status: {health['overall_status']}")
        print(f"Healthy Checks: {health['healthy_checks']}/{health['total_checks']}")

    exit(0 if success else 1)
