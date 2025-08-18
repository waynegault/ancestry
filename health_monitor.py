#!/usr/bin/env python3

"""
health_monitor.py - Comprehensive Health Monitoring System

Provides real-time health monitoring, session health scoring, early warning detection,
and predictive analytics for the Ancestry automation system.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import (
    setup_module,
    register_function,
    get_function,
    is_function_available,
)

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import sys
import time
import threading
import psutil
import gc
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

# === LOCAL IMPORTS ===
from test_framework import TestSuite, suppress_logging, create_mock_data, assert_valid_function


class HealthStatus(Enum):
    """Health status levels for different system components."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"
    ERROR = "error"


class AlertLevel(Enum):
    """Alert levels for health monitoring."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class HealthMetric:
    """Individual health metric data structure."""
    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    weight: float = 1.0
    timestamp: float = field(default_factory=time.time)
    
    @property
    def status(self) -> HealthStatus:
        """Get health status based on thresholds."""
        if self.value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.value >= self.threshold_warning:
            return HealthStatus.POOR
        elif self.value >= self.threshold_warning * 0.8:
            return HealthStatus.FAIR
        elif self.value >= self.threshold_warning * 0.6:
            return HealthStatus.GOOD
        else:
            return HealthStatus.EXCELLENT


@dataclass
class HealthAlert:
    """Health alert data structure."""
    level: AlertLevel
    component: str
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False


class SessionHealthMonitor:
    """
    Comprehensive session health monitoring with predictive analytics.
    """
    
    def __init__(self):
        self.metrics_history: Dict[str, deque] = {}
        self.current_metrics: Dict[str, HealthMetric] = {}
        self.alerts: List[HealthAlert] = []
        self.health_score_history: deque = deque(maxlen=100)
        self.session_start_time = time.time()
        self.last_health_check = time.time()
        self.monitoring_active = False
        self.lock = threading.Lock()
        
        # Performance tracking
        self.api_response_times: deque = deque(maxlen=50)
        self.error_counts: Dict[str, int] = {}
        self.page_processing_times: deque = deque(maxlen=20)
        self.memory_usage_history: deque = deque(maxlen=30)
        
        # Predictive analytics
        self.failure_patterns: List[Dict[str, Any]] = []
        self.success_patterns: List[Dict[str, Any]] = []
        
        # Initialize metrics
        self._initialize_metrics()
        
        logger.info("Session Health Monitor initialized")
    
    def _initialize_metrics(self):
        """Initialize health metrics with default thresholds."""
        metrics_config = {
            "api_response_time": {"warning": 5.0, "critical": 10.0, "weight": 2.0},
            "memory_usage_mb": {"warning": 200.0, "critical": 400.0, "weight": 1.5},
            "error_rate": {"warning": 0.05, "critical": 0.15, "weight": 3.0},
            "session_age_minutes": {"warning": 45.0, "critical": 60.0, "weight": 1.0},
            "browser_age_minutes": {"warning": 25.0, "critical": 35.0, "weight": 2.5},
            "pages_since_refresh": {"warning": 25.0, "critical": 35.0, "weight": 2.0},
            "cpu_usage_percent": {"warning": 70.0, "critical": 90.0, "weight": 1.0},
            "disk_usage_percent": {"warning": 85.0, "critical": 95.0, "weight": 0.5},
        }
        
        for name, config in metrics_config.items():
            self.current_metrics[name] = HealthMetric(
                name=name,
                value=0.0,
                threshold_warning=config["warning"],
                threshold_critical=config["critical"],
                weight=config["weight"]
            )
            self.metrics_history[name] = deque(maxlen=100)
    
    def update_metric(self, name: str, value: float):
        """Update a specific health metric."""
        with self.lock:
            if name in self.current_metrics:
                self.current_metrics[name].value = value
                self.current_metrics[name].timestamp = time.time()
                self.metrics_history[name].append((time.time(), value))
                
                # Check for alerts
                self._check_metric_alerts(name)
            else:
                logger.warning(f"Unknown health metric: {name}")
    
    def _check_metric_alerts(self, metric_name: str):
        """Check if a metric triggers any alerts."""
        metric = self.current_metrics[metric_name]
        
        if metric.value >= metric.threshold_critical:
            self._create_alert(
                AlertLevel.CRITICAL,
                "session_health",
                f"{metric_name} is critical: {metric.value:.2f} >= {metric.threshold_critical}",
                metric_name,
                metric.value,
                metric.threshold_critical
            )
        elif metric.value >= metric.threshold_warning:
            self._create_alert(
                AlertLevel.WARNING,
                "session_health", 
                f"{metric_name} is elevated: {metric.value:.2f} >= {metric.threshold_warning}",
                metric_name,
                metric.value,
                metric.threshold_warning
            )
    
    def _create_alert(self, level: AlertLevel, component: str, message: str, 
                     metric_name: str, metric_value: float, threshold: float):
        """Create a new health alert."""
        alert = HealthAlert(
            level=level,
            component=component,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold
        )
        
        self.alerts.append(alert)
        
        # Log alert based on level
        if level == AlertLevel.CRITICAL:
            logger.critical(f"🚨 CRITICAL ALERT: {message}")
        elif level == AlertLevel.WARNING:
            logger.warning(f"⚠️ WARNING: {message}")
        else:
            logger.info(f"ℹ️ INFO: {message}")
    
    def calculate_health_score(self) -> float:
        """
        Calculate overall health score (0-100) based on all metrics.
        
        Returns:
            Health score from 0 (critical) to 100 (excellent)
        """
        with self.lock:
            total_score = 100.0
            total_weight = 0.0
            
            for metric in self.current_metrics.values():
                # Calculate metric score (0-100)
                if metric.value <= metric.threshold_warning * 0.5:
                    metric_score = 100.0  # Excellent
                elif metric.value <= metric.threshold_warning * 0.8:
                    metric_score = 80.0   # Good
                elif metric.value <= metric.threshold_warning:
                    metric_score = 60.0   # Fair
                elif metric.value <= metric.threshold_critical:
                    metric_score = 30.0   # Poor
                else:
                    metric_score = 0.0    # Critical
                
                # Apply weighted average
                total_score += metric_score * metric.weight
                total_weight += metric.weight
            
            # Calculate final weighted score
            if total_weight > 0:
                final_score = total_score / total_weight
            else:
                final_score = 0.0
            
            # Store in history
            self.health_score_history.append((time.time(), final_score))
            
            return max(0.0, min(100.0, final_score))
    
    def get_health_status(self) -> HealthStatus:
        """Get overall health status based on current score."""
        score = self.calculate_health_score()
        
        if score >= 80:
            return HealthStatus.EXCELLENT
        elif score >= 60:
            return HealthStatus.GOOD
        elif score >= 40:
            return HealthStatus.FAIR
        elif score >= 20:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL
    
    def predict_session_death_risk(self) -> float:
        """
        Enhanced prediction of session death likelihood in next 10 pages (0.0-1.0).

        Returns:
            Risk score from 0.0 (very safe) to 1.0 (imminent failure)
        """
        risk_score = 0.0

        # Base risk from current health score (more aggressive)
        health_score = self.calculate_health_score()
        health_risk = (100 - health_score) / 100 * 0.5  # Increased from 0.4
        risk_score += health_risk

        # Risk from API response time trend (more sensitive)
        if len(self.api_response_times) >= 3:  # Reduced from 5
            recent_avg = sum(list(self.api_response_times)[-3:]) / 3
            if recent_avg > 10.0:
                risk_score += 0.4  # Increased from 0.3
            elif recent_avg > 8.0:
                risk_score += 0.3  # New threshold
            elif recent_avg > 5.0:
                risk_score += 0.2  # Increased from 0.15

        # Risk from error rate (more aggressive)
        total_errors = sum(self.error_counts.values())
        if total_errors > 15:
            risk_score += 0.4  # Increased from 0.2
        elif total_errors > 10:
            risk_score += 0.3  # Increased from 0.2
        elif total_errors > 5:
            risk_score += 0.2  # Increased from 0.1

        # Risk from memory usage trend
        if len(self.memory_usage_history) >= 3:
            memory_trend = list(self.memory_usage_history)[-1] - list(self.memory_usage_history)[-3]
            if memory_trend > 100:  # Memory increasing very rapidly
                risk_score += 0.2  # Increased from 0.1
            elif memory_trend > 50:  # Memory increasing rapidly
                risk_score += 0.1

        # Additional risk factors for critical metrics
        for metric_name, metric in self.current_metrics.items():
            if metric.value >= metric.threshold_critical:
                risk_score += 0.15  # Each critical metric adds significant risk
            elif metric.value >= metric.threshold_warning:
                risk_score += 0.05  # Each warning metric adds some risk

        return min(1.0, risk_score)
    
    def get_recommended_actions(self) -> List[str]:
        """Get recommended actions based on current health status."""
        actions = []
        health_score = self.calculate_health_score()
        risk_score = self.predict_session_death_risk()
        
        if risk_score > 0.8:
            actions.append("🚨 EMERGENCY: Trigger immediate session refresh")
            actions.append("🔄 Restart browser immediately")
            actions.append("⚡ Switch to emergency settings (1 worker, batch 1)")
        elif risk_score > 0.6:
            actions.append("⚠️ CRITICAL: Reduce concurrency to 1 worker")
            actions.append("📉 Reduce batch size to 3")
            actions.append("🔄 Schedule session refresh within 5 pages")
        elif risk_score > 0.4:
            actions.append("⚠️ WARNING: Reduce batch size to 5")
            actions.append("📊 Monitor closely for next 10 pages")
            actions.append("🔄 Consider session refresh within 15 pages")
        elif health_score < 60:
            actions.append("📊 Monitor performance metrics")
            actions.append("🧹 Consider garbage collection")
        else:
            actions.append("✅ System healthy - continue current operations")
        
        return actions
    
    def record_api_response_time(self, response_time: float):
        """Record API response time for monitoring."""
        self.api_response_times.append(response_time)
        
        # Update metric
        if len(self.api_response_times) >= 5:
            avg_response_time = sum(list(self.api_response_times)[-5:]) / 5
            self.update_metric("api_response_time", avg_response_time)
    
    def record_error(self, error_type: str):
        """Record an error for monitoring."""
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        # Update error rate metric
        total_errors = sum(self.error_counts.values())
        session_duration_hours = (time.time() - self.session_start_time) / 3600
        error_rate = total_errors / max(session_duration_hours, 0.1)  # Errors per hour
        self.update_metric("error_rate", error_rate)
    
    def record_page_processing_time(self, processing_time: float):
        """Record page processing time."""
        self.page_processing_times.append(processing_time)
    
    def update_system_metrics(self):
        """Update system-level metrics (CPU, memory, etc.)."""
        try:
            # Memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_usage_history.append(memory_mb)
            self.update_metric("memory_usage_mb", memory_mb)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.update_metric("cpu_usage_percent", cpu_percent)
            
            # Disk usage
            disk_usage = psutil.disk_usage('/').percent
            self.update_metric("disk_usage_percent", disk_usage)
            
        except Exception as e:
            logger.warning(f"Error updating system metrics: {e}")
    
    def update_session_metrics(self, session_manager=None):
        """Update session-specific metrics with enhanced error handling."""
        try:
            # Session age
            session_age_minutes = (time.time() - self.session_start_time) / 60
            self.update_metric("session_age_minutes", session_age_minutes)

            if session_manager:
                # Handle browser health monitor
                if hasattr(session_manager, 'browser_health_monitor'):
                    try:
                        monitor = session_manager.browser_health_monitor

                        # Check if monitor is a dict-like object
                        if hasattr(monitor, 'get') or isinstance(monitor, dict):
                            # Browser age
                            browser_start_time = monitor.get('browser_start_time') if hasattr(monitor, 'get') else monitor.get('browser_start_time', time.time())
                            if browser_start_time:
                                browser_age_minutes = (time.time() - browser_start_time) / 60
                                self.update_metric("browser_age_minutes", browser_age_minutes)

                            # Pages since refresh
                            pages_since_refresh = monitor.get('pages_since_refresh') if hasattr(monitor, 'get') else monitor.get('pages_since_refresh', 0)
                            if pages_since_refresh is not None:
                                self.update_metric("pages_since_refresh", pages_since_refresh)
                    except Exception as browser_exc:
                        logger.debug(f"Browser health monitor update failed: {browser_exc}")

                # Handle session health monitor
                if hasattr(session_manager, 'session_health_monitor'):
                    try:
                        session_monitor = session_manager.session_health_monitor
                        if hasattr(session_monitor, 'get') or isinstance(session_monitor, dict):
                            session_start = session_monitor.get('session_start_time') if hasattr(session_monitor, 'get') else session_monitor.get('session_start_time', time.time())
                            if session_start:
                                session_age_minutes = (time.time() - session_start) / 60
                                self.update_metric("session_age_minutes", session_age_minutes)
                    except Exception as session_exc:
                        logger.debug(f"Session health monitor update failed: {session_exc}")

        except Exception as e:
            logger.warning(f"Error updating session metrics: {e}")
    
    def get_health_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive health dashboard data."""
        health_score = self.calculate_health_score()
        health_status = self.get_health_status()
        risk_score = self.predict_session_death_risk()
        
        return {
            "timestamp": time.time(),
            "health_score": health_score,
            "health_status": health_status.value,
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score),
            "metrics": {name: {
                "value": metric.value,
                "status": metric.status.value,
                "threshold_warning": metric.threshold_warning,
                "threshold_critical": metric.threshold_critical
            } for name, metric in self.current_metrics.items()},
            "recent_alerts": [
                {
                    "level": alert.level.value,
                    "component": alert.component,
                    "message": alert.message,
                    "timestamp": alert.timestamp
                } for alert in self.alerts[-5:]  # Last 5 alerts
            ],
            "recommended_actions": self.get_recommended_actions(),
            "performance_summary": {
                "avg_api_response_time": sum(self.api_response_times) / len(self.api_response_times) if self.api_response_times else 0,
                "total_errors": sum(self.error_counts.values()),
                "avg_page_processing_time": sum(self.page_processing_times) / len(self.page_processing_times) if self.page_processing_times else 0,
                "current_memory_mb": self.memory_usage_history[-1] if self.memory_usage_history else 0
            }
        }
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to human-readable level."""
        if risk_score > 0.8:
            return "EMERGENCY"
        elif risk_score > 0.6:
            return "CRITICAL"
        elif risk_score > 0.4:
            return "WARNING"
        elif risk_score > 0.2:
            return "CAUTION"
        else:
            return "SAFE"
    
    def log_health_summary(self):
        """Log a comprehensive health summary."""
        dashboard = self.get_health_dashboard()
        
        logger.info(f"📊 HEALTH SUMMARY:")
        logger.info(f"   Score: {dashboard['health_score']:.1f}/100 ({dashboard['health_status'].upper()})")
        logger.info(f"   Risk: {dashboard['risk_score']:.2f} ({dashboard['risk_level']})")
        logger.info(f"   API: {dashboard['performance_summary']['avg_api_response_time']:.1f}s avg")
        logger.info(f"   Memory: {dashboard['performance_summary']['current_memory_mb']:.1f}MB")
        logger.info(f"   Errors: {dashboard['performance_summary']['total_errors']}")
        
        if dashboard['recommended_actions']:
            logger.info(f"   Actions: {dashboard['recommended_actions'][0]}")


# Global health monitor instance
_health_monitor = None


def get_health_monitor() -> SessionHealthMonitor:
    """Get the global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = SessionHealthMonitor()
    return _health_monitor


def initialize_health_monitoring():
    """Initialize health monitoring system."""
    monitor = get_health_monitor()
    logger.info("Health monitoring system initialized")
    return monitor


# === INTEGRATION HELPERS ===

def integrate_with_session_manager(session_manager):
    """Integrate health monitoring with session manager."""
    monitor = get_health_monitor()

    # Add health monitoring hooks to session manager
    if hasattr(session_manager, 'browser_health_monitor'):
        # Update session metrics
        monitor.update_session_metrics(session_manager)

        # Log health summary every 25 pages
        pages = session_manager.browser_health_monitor.get('pages_since_refresh', 0)
        if pages > 0 and pages % 25 == 0:
            monitor.log_health_summary()

    return monitor


def integrate_with_action6(action6_module):
    """Integrate health monitoring with Action 6."""
    monitor = get_health_monitor()

    # Hook into API response time tracking
    def track_api_call(original_func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = original_func(*args, **kwargs)
                response_time = time.time() - start_time
                monitor.record_api_response_time(response_time)
                return result
            except Exception as e:
                monitor.record_error(type(e).__name__)
                raise
        return wrapper

    return monitor


def get_performance_recommendations(health_score: float, risk_score: float) -> Dict[str, Any]:
    """Get specific performance setting recommendations based on health."""
    recommendations = {
        "max_concurrency": 3,
        "thread_pool_workers": 3,
        "batch_size": 8,
        "token_bucket_fill_rate": 2.5,
        "action_required": "continue"
    }

    if risk_score > 0.8:
        # Emergency settings
        recommendations.update({
            "max_concurrency": 1,
            "thread_pool_workers": 1,
            "batch_size": 1,
            "token_bucket_fill_rate": 1.0,
            "action_required": "emergency_refresh"
        })
    elif risk_score > 0.6:
        # Critical settings
        recommendations.update({
            "max_concurrency": 1,
            "thread_pool_workers": 1,
            "batch_size": 3,
            "token_bucket_fill_rate": 1.5,
            "action_required": "immediate_refresh"
        })
    elif risk_score > 0.4:
        # Warning settings
        recommendations.update({
            "max_concurrency": 2,
            "thread_pool_workers": 2,
            "batch_size": 5,
            "token_bucket_fill_rate": 2.0,
            "action_required": "schedule_refresh"
        })
    elif health_score > 80:
        # Excellent health - can optimize
        recommendations.update({
            "max_concurrency": 4,
            "thread_pool_workers": 4,
            "batch_size": 10,
            "token_bucket_fill_rate": 3.0,
            "action_required": "optimize"
        })

    return recommendations


# === TEST FRAMEWORK ===

def health_monitor_tests() -> bool:
    """
    Run all health monitor tests and return True if successful.

    Returns:
        bool: True if all tests pass, False otherwise
    """

    def test_health_monitor_initialization():
        """Test health monitor initialization."""
        monitor = SessionHealthMonitor()
        assert monitor is not None
        assert len(monitor.current_metrics) > 0
        assert monitor.session_start_time > 0
        return True

    def test_metric_updates():
        """Test metric update functionality."""
        monitor = SessionHealthMonitor()

        # Test updating a metric
        monitor.update_metric("api_response_time", 3.5)
        assert monitor.current_metrics["api_response_time"].value == 3.5

        # Test metric history
        assert len(monitor.metrics_history["api_response_time"]) == 1
        return True

    def test_health_score_calculation():
        """Test health score calculation."""
        monitor = SessionHealthMonitor()

        # Set all metrics to excellent values
        for metric_name in monitor.current_metrics:
            monitor.update_metric(metric_name, 0.0)

        score = monitor.calculate_health_score()
        assert score >= 80, f"Expected high score for excellent metrics, got {score}"

        # Set all metrics to critical values
        for metric_name, metric in monitor.current_metrics.items():
            monitor.update_metric(metric_name, metric.threshold_critical + 1)

        score = monitor.calculate_health_score()
        assert score <= 20, f"Expected low score for critical metrics, got {score}"
        return True

    def test_risk_prediction():
        """Test session death risk prediction."""
        monitor = SessionHealthMonitor()

        # Test with good conditions
        risk = monitor.predict_session_death_risk()
        assert 0.0 <= risk <= 1.0, f"Risk score should be 0-1, got {risk}"

        # Test with bad conditions
        monitor.api_response_times.extend([10.0, 12.0, 15.0, 20.0, 25.0])
        monitor.error_counts["test_error"] = 15
        monitor.memory_usage_history.extend([100, 200, 300])  # Increasing memory

        risk = monitor.predict_session_death_risk()
        assert risk > 0.3, f"Expected elevated risk with bad conditions, got {risk}"
        return True

    def test_alert_system():
        """Test health alert system."""
        monitor = SessionHealthMonitor()

        # Trigger a warning alert
        monitor.update_metric("api_response_time", 6.0)  # Above warning threshold

        # Check that alert was created
        warning_alerts = [a for a in monitor.alerts if a.level == AlertLevel.WARNING]
        assert len(warning_alerts) > 0, "Warning alert should have been created"

        # Trigger a critical alert
        monitor.update_metric("api_response_time", 12.0)  # Above critical threshold

        critical_alerts = [a for a in monitor.alerts if a.level == AlertLevel.CRITICAL]
        assert len(critical_alerts) > 0, "Critical alert should have been created"
        return True

    def test_performance_tracking():
        """Test performance tracking functionality."""
        monitor = SessionHealthMonitor()

        # Test API response time tracking
        monitor.record_api_response_time(2.5)
        assert len(monitor.api_response_times) == 1

        # Test error tracking
        monitor.record_error("ConnectionError")
        assert monitor.error_counts["ConnectionError"] == 1

        # Test page processing time tracking
        monitor.record_page_processing_time(45.0)
        assert len(monitor.page_processing_times) == 1
        return True

    def test_dashboard_generation():
        """Test health dashboard generation."""
        monitor = SessionHealthMonitor()

        dashboard = monitor.get_health_dashboard()

        # Check required fields
        required_fields = ["health_score", "health_status", "risk_score", "metrics", "recommended_actions"]
        for field in required_fields:
            assert field in dashboard, f"Dashboard missing required field: {field}"

        # Check data types
        assert isinstance(dashboard["health_score"], (int, float))
        assert isinstance(dashboard["risk_score"], (float))
        assert isinstance(dashboard["metrics"], dict)
        assert isinstance(dashboard["recommended_actions"], list)
        return True

    def test_integration_helpers():
        """Test integration helper functions."""
        # Test performance recommendations
        recommendations = get_performance_recommendations(90.0, 0.1)
        assert "max_concurrency" in recommendations
        assert "action_required" in recommendations

        # Test emergency recommendations
        emergency_recs = get_performance_recommendations(20.0, 0.9)
        assert emergency_recs["max_concurrency"] == 1
        assert emergency_recs["action_required"] == "emergency_refresh"
        return True

    def test_global_instance():
        """Test global health monitor instance."""
        monitor1 = get_health_monitor()
        monitor2 = get_health_monitor()

        # Should return the same instance
        assert monitor1 is monitor2, "get_health_monitor should return singleton instance"
        return True

    # Run all tests
    test_suite = TestSuite("Health Monitoring System", "health_monitor.py")

    test_suite.run_test("Health Monitor Initialization", test_health_monitor_initialization)
    test_suite.run_test("Metric Updates", test_metric_updates)
    test_suite.run_test("Health Score Calculation", test_health_score_calculation)
    test_suite.run_test("Risk Prediction", test_risk_prediction)
    test_suite.run_test("Alert System", test_alert_system)
    test_suite.run_test("Performance Tracking", test_performance_tracking)
    test_suite.run_test("Dashboard Generation", test_dashboard_generation)
    test_suite.run_test("Integration Helpers", test_integration_helpers)
    test_suite.run_test("Global Instance", test_global_instance)

    return test_suite.tests_failed == 0


# Register functions for external access
register_function("get_health_monitor", get_health_monitor)
register_function("initialize_health_monitoring", initialize_health_monitoring)
register_function("integrate_with_session_manager", integrate_with_session_manager)
register_function("integrate_with_action6", integrate_with_action6)
register_function("get_performance_recommendations", get_performance_recommendations)
register_function("health_monitor_tests", health_monitor_tests)


def run_comprehensive_tests():
    """
    Standardized test function for the test framework.
    This function is required for integration with run_all_tests.py.
    """
    return health_monitor_tests()


if __name__ == "__main__":
    # Run tests when executed directly
    success = health_monitor_tests()
    if success:
        print("🎉 All health monitor tests passed!")
    else:
        print("❌ Some health monitor tests failed!")
        sys.exit(1)
