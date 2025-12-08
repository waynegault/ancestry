"""
Health monitoring system for the Ancestry application.
Tracks performance metrics, error rates, and resource usage to prevent session death.
"""

import json
import logging
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import psutil

from core.registry_utils import auto_register_module
from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class HealthMetric:
    """Single health metric tracking."""

    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    timestamp: float = field(default_factory=time.time)
    status: HealthStatus = HealthStatus.HEALTHY


@dataclass
class HealthAlert:
    """Health alert details."""

    level: AlertLevel
    component: str
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)


# ==============================================
# MIXINS
# ==============================================


class MetricsManagementMixin:
    """Mixin for managing health metrics."""

    # Type hints for attributes expected from SessionHealthMonitor
    current_metrics: dict[str, HealthMetric]
    metrics_history: dict[str, deque[float]]
    api_response_times: deque[float]
    error_timestamps: deque[float]
    error_counts: dict[str, int]
    page_processing_times: deque[float]
    memory_usage_history: deque[float]
    error_rate_warnings_sent: dict[str, float]
    session_start_time: float

    # Type hints for methods expected from other Mixins
    if TYPE_CHECKING:

        def _create_alert(
            self,
            level: AlertLevel,
            component: str,
            message: str,
            metric_name: str,
            metric_value: float,
            threshold: float,
        ) -> None: ...
        def _trigger_emergency_intervention(self, reason: str, value: float, threshold: str) -> None: ...
        def _trigger_immediate_intervention(self, reason: str, value: float, threshold: str) -> None: ...
        def _trigger_enhanced_monitoring(self, reason: str, value: float, threshold: str) -> None: ...

    def _initialize_metrics(self) -> None:
        """Initialize default metrics."""

        # Try to get config
        try:
            from config.config_manager import get_config_manager

            config = get_config_manager().get_config()
            health_config = getattr(config, "health", None)
        except Exception:
            health_config = None

        # Default values
        api_warn = 5.0
        api_crit = 15.0
        error_warn = 10.0
        error_crit = 25.0
        mem_warn = 800.0
        mem_crit = 1200.0

        if health_config:
            api_warn = getattr(health_config, "api_response_time_warning", api_warn)
            api_crit = getattr(health_config, "api_response_time_critical", api_crit)
            error_warn = getattr(health_config, "error_rate_warning", error_warn)
            error_crit = getattr(health_config, "error_rate_critical", error_crit)
            mem_warn = getattr(health_config, "memory_usage_warning", mem_warn)
            mem_crit = getattr(health_config, "memory_usage_critical", mem_crit)

        defaults = {
            "api_response_time": (api_warn, api_crit),
            "error_rate": (error_warn, error_crit),
            "memory_usage_mb": (mem_warn, mem_crit),
            "cpu_usage_percent": (70.0, 90.0),  # Warning > 70%, Critical > 90%
            "disk_usage_percent": (85.0, 95.0),  # Warning > 85%, Critical > 95%
            "browser_age_minutes": (60.0, 120.0),  # Warning > 60m, Critical > 120m
            "session_age_minutes": (240.0, 480.0),  # Warning > 4h, Critical > 8h
            "pages_since_refresh": (50.0, 100.0),  # Warning > 50, Critical > 100
        }

        for name, (warn, crit) in defaults.items():
            self.current_metrics[name] = HealthMetric(
                name=name, value=0.0, threshold_warning=warn, threshold_critical=crit
            )
            self.metrics_history[name] = deque(maxlen=100)

    def update_metric(self, name: str, value: float) -> None:
        """Update a specific metric and check for alerts."""
        try:
            if name not in self.current_metrics:
                # Initialize with default thresholds if new
                self.current_metrics[name] = HealthMetric(
                    name=name, value=value, threshold_warning=float('inf'), threshold_critical=float('inf')
                )
                self.metrics_history[name] = deque(maxlen=100)

            metric = self.current_metrics[name]
            metric.value = value
            metric.timestamp = time.time()
            self.metrics_history[name].append(value)

            # Update status
            if value >= metric.threshold_critical:
                metric.status = HealthStatus.CRITICAL
            elif value >= metric.threshold_warning:
                metric.status = HealthStatus.DEGRADED
            else:
                metric.status = HealthStatus.HEALTHY

            self._check_metric_alerts(metric)

        except Exception as e:
            logger.debug(f"Error updating metric {name}: {e}")

    def _check_metric_alerts(self, metric: HealthMetric) -> None:
        """Check if metric should trigger an alert."""
        if metric.status == HealthStatus.CRITICAL:
            self._create_alert(
                AlertLevel.CRITICAL,
                "metrics",
                f"{metric.name} critical: {metric.value:.1f} (limit: {metric.threshold_critical:.1f})",
                metric.name,
                metric.value,
                metric.threshold_critical,
            )
        elif metric.status == HealthStatus.DEGRADED:
            # Only alert on degraded if it's a significant change or periodic reminder
            self._create_alert(
                AlertLevel.WARNING,
                "metrics",
                f"{metric.name} degraded: {metric.value:.1f} (limit: {metric.threshold_warning:.1f})",
                metric.name,
                metric.value,
                metric.threshold_warning,
            )

    def record_api_response_time(self, duration: float) -> None:
        """Record API response time."""
        self.api_response_times.append(duration)
        avg_time = sum(self.api_response_times) / len(self.api_response_times)
        self.update_metric("api_response_time", avg_time)

    def record_error(self, error_type: str) -> None:
        """Record an error occurrence."""
        current_time = time.time()
        self.error_timestamps.append(current_time)
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Calculate error rate (errors per minute)
        # Filter timestamps from last minute
        recent_errors = [t for t in self.error_timestamps if current_time - t <= 60]
        error_rate = len(recent_errors)
        self.update_metric("error_rate", float(error_rate))

        # Check for rapid error spikes (e.g. 10 errors in 10 seconds)
        very_recent = [t for t in self.error_timestamps if current_time - t <= 10]
        if len(very_recent) >= 10:
            self._trigger_emergency_intervention("RAPID_ERROR_SPIKE", len(very_recent), "10 errors in 10 seconds")

        # Check for sustained error rate
        self._process_error_window_threshold(current_time)

    def _process_error_window_threshold(self, current_time: float) -> None:
        """Check for sustained error rates over a 5-minute window."""
        # Filter timestamps from last 5 minutes
        five_min_errors = [t for t in self.error_timestamps if current_time - t <= 300]
        count = len(five_min_errors)

        # Thresholds for 5-minute window
        if count >= 50:  # ~10 errors/min sustained
            self._trigger_immediate_intervention("SUSTAINED_ERROR_RATE", count, "50+ errors in 5 minutes")
        elif count >= 20:  # ~4 errors/min sustained
            self._trigger_enhanced_monitoring("ELEVATED_ERROR_RATE", count, "20+ errors in 5 minutes")

        # Check for early warning signs
        self._check_error_rate_early_warning(count, current_time)

    def _check_error_rate_early_warning(self, five_min_count: int, current_time: float) -> None:
        """Check for early warning signs of increasing error rates."""
        # Only check if we haven't warned recently (every 5 mins)
        last_warning = self.error_rate_warnings_sent.get("early_warning", 0)
        if current_time - last_warning < 300:
            return

        if five_min_count >= 10:  # ~2 errors/min
            logger.warning(f"⚠️ Early warning: {five_min_count} errors in last 5 minutes")
            self.error_rate_warnings_sent["early_warning"] = current_time

    def get_error_rate_statistics(self) -> dict[str, Any]:
        """Get detailed error rate statistics."""
        current_time = time.time()
        one_min = len([t for t in self.error_timestamps if current_time - t <= 60])
        five_min = len([t for t in self.error_timestamps if current_time - t <= 300])
        fifteen_min = len([t for t in self.error_timestamps if current_time - t <= 900])
        hour = len([t for t in self.error_timestamps if current_time - t <= 3600])

        return {
            "1_min": one_min,
            "5_min": five_min,
            "15_min": fifteen_min,
            "60_min": hour,
            "total": len(self.error_timestamps),
            "top_errors": sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:5],
        }

    def record_page_processing_time(self, processing_time: float) -> None:
        """Record page processing time."""
        self.page_processing_times.append(processing_time)

    def update_system_metrics(self) -> None:
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
            logger.debug(f"Error updating system metrics: {e}")

    def _update_browser_health_metrics(self, monitor: Any) -> None:
        """Update browser health metrics from monitor."""
        if not (hasattr(monitor, 'get') or isinstance(monitor, dict)):
            return

        monitor_dict = cast(dict[str, Any], monitor)

        # Browser age
        browser_start_time = monitor_dict.get('browser_start_time', time.time())
        if browser_start_time:
            browser_age_minutes = (time.time() - browser_start_time) / 60
            self.update_metric("browser_age_minutes", browser_age_minutes)

        # Pages since refresh
        pages_since_refresh = monitor_dict.get('pages_since_refresh', 0)
        if pages_since_refresh is not None:
            self.update_metric("pages_since_refresh", pages_since_refresh)

    def _update_session_health_metrics(self, session_monitor: Any) -> None:
        """Update session health metrics from monitor."""
        if not (hasattr(session_monitor, 'get') or isinstance(session_monitor, dict)):
            return

        session_monitor_dict = cast(dict[str, Any], session_monitor)
        session_start = session_monitor_dict.get('session_start_time', time.time())
        if session_start:
            session_age_minutes = (time.time() - session_start) / 60
            self.update_metric("session_age_minutes", session_age_minutes)

    def update_session_metrics(self, session_manager: Any = None) -> None:
        """Update session-specific metrics with enhanced error handling."""
        try:
            # Session age
            session_age_minutes = (time.time() - self.session_start_time) / 60
            self.update_metric("session_age_minutes", session_age_minutes)

            if session_manager:
                # Handle browser health monitor
                if hasattr(session_manager, 'browser_health_monitor'):
                    try:
                        self._update_browser_health_metrics(session_manager.browser_health_monitor)
                    except Exception as browser_exc:
                        logger.debug(f"Browser health monitor update failed: {browser_exc}")

                # Handle session health monitor
                if hasattr(session_manager, 'session_health_monitor'):
                    try:
                        self._update_session_health_metrics(session_manager.session_health_monitor)
                    except Exception as session_exc:
                        logger.debug(f"Session health monitor update failed: {session_exc}")

        except Exception as e:
            logger.debug(f"Error updating session metrics: {e}")


class AlertingMixin:
    """Mixin for managing health alerts."""

    # Type hints for attributes expected from SessionHealthMonitor
    alerts: list[HealthAlert]
    last_alert_times: dict[str, float]
    _is_safety_testing: bool

    def begin_safety_test(self) -> None:
        """Mark the beginning of a safety test to prefix alerts."""
        self._is_safety_testing = True

    def end_safety_test(self) -> None:
        """Mark the end of a safety test."""
        self._is_safety_testing = False

    @property
    def _safety_prefix(self) -> str:
        """Return prefix for alerts if in safety testing mode."""
        return "[SAFETY_TEST] " if getattr(self, "_is_safety_testing", False) else ""

    def _create_alert(
        self,
        level: AlertLevel,
        component: str,
        message: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
    ) -> None:
        """Create a new health alert."""
        # Check for duplicate alerts to prevent spam
        current_time = time.time()

        # Use a composite key for deduplication
        alert_key = f"{component}:{metric_name}:{level.value}"

        # Don't repeat same alert within 5 minutes
        last_time = self.last_alert_times.get(alert_key, 0)
        if current_time - last_time < 300:
            return

        prefixed_message = f"{self._safety_prefix}{message}"
        alert = HealthAlert(
            level=level,
            component=component,
            message=prefixed_message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
        )
        self.alerts.append(alert)
        self.last_alert_times[alert_key] = current_time

        # Log based on severity
        if level == AlertLevel.EMERGENCY:
            logger.critical(f"🚨 EMERGENCY ALERT: {prefixed_message}")
        elif level == AlertLevel.CRITICAL:
            logger.error(f"🔴 CRITICAL ALERT: {prefixed_message}")
        elif level == AlertLevel.WARNING:
            logger.warning(f"⚠️ HEALTH WARNING: {prefixed_message}")


class HealthAssessmentMixin:
    """Mixin for calculating health scores and risks."""

    # Type hints for attributes expected from SessionHealthMonitor
    current_metrics: dict[str, HealthMetric]
    alerts: list[HealthAlert]
    health_score_history: deque[float]
    api_response_times: deque[float]
    memory_usage_history: deque[float]
    error_counts: dict[str, int]
    page_processing_times: deque[float]

    def calculate_health_score(self) -> float:
        """Calculate overall health score (0-100)."""
        score = 100.0
        deductions = 0.0

        for metric in self.current_metrics.values():
            if metric.status == HealthStatus.CRITICAL:
                deductions += 20.0
            elif metric.status == HealthStatus.DEGRADED:
                deductions += 10.0

        # Additional deductions for recent errors
        recent_errors = self.current_metrics.get("error_rate", HealthMetric("error_rate", 0, 0, 0)).value
        deductions += min(recent_errors * 2, 40.0)  # Cap error deductions at 40

        # Additional deductions for alerts
        recent_alerts = len([a for a in self.alerts if time.time() - a.timestamp < 3600])
        deductions += min(recent_alerts * 5, 30.0)

        final_score = max(0.0, score - deductions)
        self.health_score_history.append(final_score)
        return final_score

    def get_health_status(self) -> HealthStatus:
        """Get overall system health status."""
        score = self.calculate_health_score()
        if score >= 80:
            return HealthStatus.HEALTHY
        if score >= 60:
            return HealthStatus.DEGRADED
        if score >= 40:
            return HealthStatus.CRITICAL
        return HealthStatus.EMERGENCY

    def _calculate_api_response_risk(self) -> float:
        """Calculate risk from API response times."""
        if not self.api_response_times:
            return 0.0

        avg_time = sum(self.api_response_times) / len(self.api_response_times)
        if avg_time > 5.0:
            return 0.8
        if avg_time > 2.0:
            return 0.4
        return 0.1

    def _calculate_error_rate_risk(self) -> float:
        """Calculate risk from error rates."""
        error_rate = self.current_metrics.get("error_rate", HealthMetric("error_rate", 0, 0, 0)).value
        if error_rate > 25.0:
            return 0.9
        if error_rate > 10.0:
            return 0.6
        if error_rate > 5.0:
            return 0.3
        return 0.0

    def _calculate_memory_trend_risk(self) -> float:
        """Calculate risk from memory usage trend."""
        if len(self.memory_usage_history) < 10:
            return 0.0

        # Simple trend analysis
        recent = list(self.memory_usage_history)[-10:]
        start = recent[0]
        end = recent[-1]

        if end > 1200:  # Absolute high memory
            return 0.8

        if end > start * 1.2:  # 20% increase in last 10 readings
            return 0.5

        return 0.1

    def _calculate_critical_metrics_risk(self) -> float:
        """Calculate risk from critical metrics status."""
        critical_count = sum(1 for m in self.current_metrics.values() if m.status == HealthStatus.CRITICAL)
        if critical_count >= 3:
            return 1.0
        if critical_count >= 1:
            return 0.7
        return 0.0

    def predict_session_death_risk(self) -> float:
        """Predict probability of session death (0.0 - 1.0)."""
        factors = {
            "api_response": self._calculate_api_response_risk(),
            "error_rate": self._calculate_error_rate_risk(),
            "memory_trend": self._calculate_memory_trend_risk(),
            "critical_metrics": self._calculate_critical_metrics_risk(),
        }

        # Weighted average
        weights = {
            "api_response": 0.2,
            "error_rate": 0.4,  # Errors are strongest predictor
            "memory_trend": 0.2,
            "critical_metrics": 0.2,
        }

        risk_score = sum(factors[k] * weights[k] for k in factors)
        return min(1.0, risk_score)

    def get_recommended_actions(self) -> list[str]:
        """Get recommended actions based on health state."""
        actions: list[str] = []
        status = self.get_health_status()

        if status == HealthStatus.EMERGENCY:
            actions.append("IMMEDIATE_RESTART_REQUIRED")
            actions.append("STOP_ALL_TRAFFIC")
        elif status == HealthStatus.CRITICAL:
            actions.append("SCHEDULE_RESTART")
            actions.append("REDUCE_CONCURRENCY")
        elif status == HealthStatus.DEGRADED:
            actions.append("MONITOR_CLOSELY")
            actions.append("PAUSE_NON_ESSENTIAL")

        # Specific metric actions
        if self.current_metrics.get("memory_usage_mb", HealthMetric("", 0, 0, 0)).status == HealthStatus.CRITICAL:
            actions.append("CLEAR_CACHE")

        return actions

    def get_health_dashboard(self) -> dict[str, Any]:
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
            "metrics": {
                name: {
                    "value": metric.value,
                    "status": metric.status.value,
                    "threshold_warning": metric.threshold_warning,
                    "threshold_critical": metric.threshold_critical,
                }
                for name, metric in self.current_metrics.items()
            },
            "recent_alerts": [
                {
                    "level": alert.level.value,
                    "component": alert.component,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                }
                for alert in list(self.alerts)[-5:]  # Last 5 alerts
            ],
            "recommended_actions": self.get_recommended_actions(),
            "performance_summary": {
                "avg_api_response_time": sum(self.api_response_times) / len(self.api_response_times)
                if self.api_response_times
                else 0,
                "total_errors": sum(self.error_counts.values()),
                "avg_page_processing_time": sum(self.page_processing_times) / len(self.page_processing_times)
                if self.page_processing_times
                else 0,
                "current_memory_mb": self.memory_usage_history[-1] if self.memory_usage_history else 0,
            },
        }

    @staticmethod
    def _get_risk_level(risk_score: float) -> str:
        """Convert risk score to human-readable level."""
        if risk_score > 0.8:
            return "EMERGENCY"
        if risk_score > 0.6:
            return "CRITICAL"
        if risk_score > 0.4:
            return "WARNING"
        if risk_score > 0.2:
            return "CAUTION"
        return "SAFE"

    def log_health_summary(self) -> None:
        """Log a comprehensive health summary."""
        dashboard = self.get_health_dashboard()

        logger.info("📊 HEALTH SUMMARY:")
        logger.info(f"   Score: {dashboard['health_score']:.1f}/100 ({dashboard['health_status'].upper()})")
        logger.info(f"   Risk: {dashboard['risk_score']:.2f} ({dashboard['risk_level']})")
        logger.info(f"   API: {dashboard['performance_summary']['avg_api_response_time']:.1f}s avg")
        logger.info(f"   Memory: {dashboard['performance_summary']['current_memory_mb']:.1f}MB")
        logger.info(f"   Errors: {dashboard['performance_summary']['total_errors']}")

        if dashboard['recommended_actions']:
            logger.info(f"   Actions: {dashboard['recommended_actions'][0]}")


class InterventionMixin:
    """Mixin for handling system interventions."""

    # Type hints for attributes expected from SessionHealthMonitor
    _emergency_halt_requested: bool
    _emergency_halt_reason: Optional[str]
    _emergency_halt_timestamp: float
    _immediate_intervention_requested: bool
    _immediate_intervention_reason: Optional[str]
    _immediate_intervention_timestamp: float
    _enhanced_monitoring_active: bool
    _enhanced_monitoring_reason: Optional[str]
    _enhanced_monitoring_timestamp: float
    _monitoring_interval: float
    error_timestamps: deque[float]

    # Type hints for methods expected from other Mixins
    def _create_alert(
        self,
        level: AlertLevel,
        component: str,
        message: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
    ) -> None: ...

    def _trigger_emergency_intervention(self, reason: str, value: float, threshold: str) -> None:
        """Trigger emergency intervention (halt)."""
        if self._emergency_halt_requested:
            return

        logger.critical(f"🚨 EMERGENCY INTERVENTION TRIGGERED: {reason} (Value: {value}, Threshold: {threshold})")
        self._emergency_halt_requested = True
        self._emergency_halt_reason = f"{reason}: {value} (Threshold: {threshold})"
        self._emergency_halt_timestamp = time.time()

        # Also create an alert
        self._create_alert(
            AlertLevel.EMERGENCY, "intervention", f"Emergency halt: {reason}", "intervention", value, 0.0
        )

    def _trigger_immediate_intervention(self, reason: str, value: float, threshold: str) -> None:
        """Trigger immediate intervention (refresh/restart)."""
        if self._immediate_intervention_requested:
            return

        logger.error(f"🔴 IMMEDIATE INTERVENTION REQUESTED: {reason} (Value: {value}, Threshold: {threshold})")
        self._immediate_intervention_requested = True
        self._immediate_intervention_reason = f"{reason}: {value} (Threshold: {threshold})"
        self._immediate_intervention_timestamp = time.time()

    def _trigger_enhanced_monitoring(self, reason: str, value: float, threshold: str) -> None:
        """Trigger enhanced monitoring mode."""
        if self._enhanced_monitoring_active:
            return

        logger.warning(f"⚠️ ENHANCED MONITORING ACTIVATED: {reason} (Value: {value}, Threshold: {threshold})")
        self._enhanced_monitoring_active = True
        self._enhanced_monitoring_reason = f"{reason}: {value} (Threshold: {threshold})"
        self._enhanced_monitoring_timestamp = time.time()

    def should_emergency_halt(self) -> bool:
        """Check if emergency halt is requested."""
        return self._emergency_halt_requested

    def should_immediate_intervention(self) -> bool:
        """Check if immediate intervention is requested."""
        return self._immediate_intervention_requested

    def is_enhanced_monitoring_active(self) -> bool:
        """Check if enhanced monitoring is active."""
        return self._enhanced_monitoring_active

    def get_intervention_status(self) -> dict[str, Any]:
        """Get current intervention status."""
        return {
            "emergency_halt": {
                "requested": self._emergency_halt_requested,
                "reason": self._emergency_halt_reason,
                "timestamp": self._emergency_halt_timestamp,
            },
            "immediate_intervention": {
                "requested": self._immediate_intervention_requested,
                "reason": self._immediate_intervention_reason,
                "timestamp": self._immediate_intervention_timestamp,
            },
            "enhanced_monitoring": {
                "active": self._enhanced_monitoring_active,
                "reason": self._enhanced_monitoring_reason,
                "timestamp": self._enhanced_monitoring_timestamp,
            },
        }

    def reset_intervention_flags(self) -> None:
        """Reset intervention flags (use with caution)."""
        logger.debug("🔄 Resetting intervention flags")
        self._emergency_halt_requested = False
        self._emergency_halt_reason = ""
        self._emergency_halt_timestamp = 0.0
        self._immediate_intervention_requested = False
        self._immediate_intervention_reason = ""
        self._immediate_intervention_timestamp = 0.0
        self._enhanced_monitoring_active = False
        self._enhanced_monitoring_reason = ""
        self._enhanced_monitoring_timestamp = 0.0

    def _get_adaptive_monitoring_interval(self, current_time: float) -> float:
        """Get adaptive monitoring interval based on current system state."""
        try:
            # Base interval
            interval = self._monitoring_interval

            # If enhanced monitoring is active, check more frequently
            if self._enhanced_monitoring_active:
                interval = 5.0  # Check every 5 seconds during enhanced monitoring
            else:
                # Adaptive interval based on recent error rate
                recent_errors = sum(1 for ts in self.error_timestamps if current_time - ts < 300)  # Last 5 minutes

                if recent_errors >= 50:
                    interval = 10.0  # High error rate - check every 10 seconds
                elif recent_errors >= 20:
                    interval = 20.0  # Moderate error rate - check every 20 seconds
                elif recent_errors >= 5:
                    interval = 30.0  # Low error rate - normal interval
                else:
                    interval = 60.0  # Very low error rate - check every minute

            return interval

        except Exception as e:
            logger.debug(f"Error calculating adaptive interval: {e}")
            return self._monitoring_interval


class ResourceManagementMixin:
    """Mixin for managing resources and cleanup."""

    # Type hints for attributes expected from SessionHealthMonitor
    current_metrics: dict[str, HealthMetric]
    metrics_history: dict[str, deque[float]]
    error_timestamps: deque[float]
    error_rate_warnings_sent: dict[str, float]
    alerts: list[HealthAlert]
    _last_cleanup_time: float
    _cleanup_interval: float
    session_start_time: float
    _adaptive_interval: bool

    # Type hints for methods expected from other Mixins
    # Note: We don't define them here to avoid overriding actual implementations in MRO
    # def update_metric(self, name: str, value: float) -> None: ...

    def _perform_efficient_cleanup(self, current_time: float) -> None:
        """Perform efficient cleanup of old data to reduce memory usage."""
        try:
            # Only perform cleanup every 5 minutes to reduce overhead
            if current_time - self._last_cleanup_time < self._cleanup_interval:
                return

            self._last_cleanup_time = current_time

            # Clean old error timestamps (older than 2 hours for long sessions)
            cutoff_time = current_time - 7200  # 2 hours

            # PERFORMANCE: Batch cleanup instead of one-by-one
            cleanup_count = 0
            while self.error_timestamps and self.error_timestamps[0] < cutoff_time:
                self.error_timestamps.popleft()
                cleanup_count += 1

                # Prevent excessive cleanup in one operation
                if cleanup_count > 1000:
                    break

            # Clean old warning timestamps
            warning_cutoff = current_time - 3600  # 1 hour
            old_warnings = [
                key for key, timestamp in self.error_rate_warnings_sent.items() if timestamp < warning_cutoff
            ]
            for key in old_warnings:
                del self.error_rate_warnings_sent[key]

            # Clean old alerts (keep only last 4 hours)
            alert_cutoff = current_time - 14400  # 4 hours
            self.alerts = [alert for alert in self.alerts if alert.timestamp >= alert_cutoff]

            if cleanup_count > 0:
                logger.debug(f"Cleaned up {cleanup_count} old error timestamps and {len(old_warnings)} old warnings")

        except Exception as e:
            logger.debug(f"Error during efficient cleanup: {e}")

    def optimize_for_long_session(self) -> None:
        """Optimize monitoring settings for long-running sessions (20+ hours)."""
        try:
            logger.info("🔧 Optimizing health monitoring for long session")

            # Increase monitoring intervals to reduce overhead
            self._monitoring_interval = 60.0  # Check every minute instead of 30 seconds
            self._cleanup_interval = 600.0  # Clean up every 10 minutes instead of 5

            # Increase deque sizes for longer data retention
            if self.error_timestamps.maxlen is None or self.error_timestamps.maxlen < 3000:
                # Create new deque with larger size and copy existing data
                old_timestamps = list(self.error_timestamps)
                self.error_timestamps = deque(old_timestamps, maxlen=3000)

            # Optimize metrics history for longer sessions
            for name in self.metrics_history:
                current_maxlen = self.metrics_history[name].maxlen
                if current_maxlen is None or current_maxlen < 200:
                    old_history = list(self.metrics_history[name])
                    self.metrics_history[name] = deque(old_history, maxlen=200)

            logger.info("✅ Health monitoring optimized for long session")

        except Exception as e:
            logger.debug(f"Failed to optimize for long session: {e}")

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for monitoring overhead analysis."""
        try:
            current_time = time.time()
            session_duration = (current_time - self.session_start_time) / 3600  # Hours

            return {
                "session_duration_hours": session_duration,
                "error_timestamps_count": len(self.error_timestamps),
                "error_timestamps_max": self.error_timestamps.maxlen,
                "alerts_count": len(self.alerts),
                "warnings_sent_count": len(self.error_rate_warnings_sent),
                "current_monitoring_interval": self._monitoring_interval,
                "adaptive_monitoring_enabled": self._adaptive_interval,
                "last_cleanup_age_minutes": (current_time - self._last_cleanup_time) / 60,
                "memory_efficiency": {
                    "error_timestamps_usage": f"{len(self.error_timestamps)}/{self.error_timestamps.maxlen}",
                    "metrics_history_total": sum(len(hist) for hist in self.metrics_history.values()),
                    "alerts_retention_hours": 4,
                    "warnings_retention_hours": 1,
                },
            }

        except Exception as e:
            logger.debug(f"Error getting performance stats: {e}")
            return {"error": str(e)}


class PersistenceMixin:
    """Mixin for session state persistence."""

    # Type hints for attributes expected from SessionHealthMonitor
    current_metrics: dict[str, HealthMetric]
    metrics_history: dict[str, deque[float]]
    alerts: list[HealthAlert]
    health_score_history: deque[float]
    session_start_time: float
    api_response_times: deque[float]
    error_timestamps: deque[float]
    error_counts: dict[str, int]
    page_processing_times: deque[float]
    memory_usage_history: deque[float]
    _emergency_halt_requested: bool
    _immediate_intervention_requested: bool
    _enhanced_monitoring_active: bool
    _monitoring_interval: float
    _last_cleanup_time: float
    _last_checkpoint_time: float

    # Type hints for methods expected from other Mixins
    if TYPE_CHECKING:

        def get_health_dashboard(self) -> dict[str, Any]: ...
        def calculate_health_score(self) -> float: ...
        def get_performance_stats(self) -> dict[str, Any]: ...

    def create_session_checkpoint(self, checkpoint_name: Optional[str] = None) -> str:
        """Create a checkpoint of the current session state for recovery."""
        try:
            if checkpoint_name is None:
                checkpoint_name = f"checkpoint_{int(time.time())}"

            # Create checkpoint directory
            checkpoint_dir = Path("Cache/session_checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Prepare session state data
            session_state = {
                "timestamp": time.time(),
                "checkpoint_name": checkpoint_name,
                "session_start_time": self.session_start_time,
                "health_score_history": list(self.health_score_history),
                "current_metrics": {
                    name: {
                        "value": metric.value,
                        "status": metric.status.value,
                        "threshold_warning": metric.threshold_warning,
                        "threshold_critical": metric.threshold_critical,
                        "timestamp": metric.timestamp,
                    }
                    for name, metric in self.current_metrics.items()
                },
                "alerts": [
                    {
                        "level": alert.level.value,
                        "component": alert.component,
                        "message": alert.message,
                        "metric_name": alert.metric_name,
                        "metric_value": alert.metric_value,
                        "threshold": alert.threshold,
                        "timestamp": alert.timestamp,
                    }
                    for alert in self.alerts
                ],
                "error_timestamps": list(self.error_timestamps),
                "error_counts": dict(self.error_counts),
                "api_response_times": list(self.api_response_times),
                "page_processing_times": list(self.page_processing_times),
                "memory_usage_history": list(self.memory_usage_history),
                "intervention_state": {
                    "emergency_halt_requested": self._emergency_halt_requested,
                    "immediate_intervention_requested": self._immediate_intervention_requested,
                    "enhanced_monitoring_active": self._enhanced_monitoring_active,
                    "monitoring_interval": self._monitoring_interval,
                    "last_cleanup_time": self._last_cleanup_time,
                },
                "performance_stats": self.get_performance_stats(),
            }

            # Save checkpoint to file
            checkpoint_file = checkpoint_dir / f"{checkpoint_name}.json"
            with checkpoint_file.open('w', encoding='utf-8') as f:
                json.dump(session_state, f, indent=2, default=str)

            logger.info(f"📁 Session checkpoint created: {checkpoint_name}")
            return str(checkpoint_file)

        except Exception as e:
            logger.error(f"Failed to create session checkpoint: {e}")
            return ""

    def _restore_metrics_from_state(self, session_state: dict[str, Any]) -> None:
        """Restore current metrics from session state."""
        if "current_metrics" not in session_state:
            return

        for name, metric_data in session_state["current_metrics"].items():
            metric = HealthMetric(
                name=name,
                value=metric_data["value"],
                threshold_warning=metric_data["threshold_warning"],
                threshold_critical=metric_data["threshold_critical"],
                timestamp=metric_data.get("timestamp", time.time()),
            )
            self.current_metrics[name] = metric

    def _restore_alerts_from_state(self, session_state: dict[str, Any]) -> None:
        """Restore alerts from session state."""
        if "alerts" not in session_state:
            return

        self.alerts = []
        for alert_data in session_state["alerts"]:
            level = AlertLevel(alert_data["level"])
            alert = HealthAlert(
                level=level,
                component=alert_data["component"],
                message=alert_data["message"],
                metric_name=alert_data["metric_name"],
                metric_value=alert_data["metric_value"],
                threshold=alert_data["threshold"],
                timestamp=alert_data["timestamp"],
            )
            self.alerts.append(alert)

    def _restore_error_data_from_state(self, session_state: dict[str, Any]) -> None:
        """Restore error tracking data from session state."""
        if "error_timestamps" in session_state:
            self.error_timestamps = deque(session_state["error_timestamps"], maxlen=self.error_timestamps.maxlen)

        if "error_counts" in session_state:
            self.error_counts.update(session_state["error_counts"])

    def _restore_performance_data_from_state(self, session_state: dict[str, Any]) -> None:
        """Restore performance tracking data from session state."""
        if "api_response_times" in session_state:
            self.api_response_times = deque(session_state["api_response_times"], maxlen=self.api_response_times.maxlen)

        if "page_processing_times" in session_state:
            self.page_processing_times = deque(
                session_state["page_processing_times"], maxlen=self.page_processing_times.maxlen
            )

        if "memory_usage_history" in session_state:
            self.memory_usage_history = deque(
                session_state["memory_usage_history"], maxlen=self.memory_usage_history.maxlen
            )

    def _restore_intervention_state_from_state(self, session_state: dict[str, Any]) -> None:
        """Restore intervention state from session state."""
        if "intervention_state" not in session_state:
            return

        intervention = session_state["intervention_state"]
        self._emergency_halt_requested = intervention.get("emergency_halt_requested", False)
        self._immediate_intervention_requested = intervention.get("immediate_intervention_requested", False)
        self._enhanced_monitoring_active = intervention.get("enhanced_monitoring_active", False)
        self._monitoring_interval = intervention.get("monitoring_interval", 30.0)
        self._last_cleanup_time = intervention.get("last_cleanup_time", time.time())

    def restore_from_checkpoint(self, checkpoint_path: str) -> bool:
        """Restore session state from a checkpoint file."""
        try:
            checkpoint_file = Path(checkpoint_path)
            if not checkpoint_file.exists():
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return False

            # Load checkpoint data
            with checkpoint_file.open(encoding='utf-8') as f:
                session_state = json.load(f)

            # Restore session state
            self.session_start_time = session_state.get("session_start_time", time.time())

            # Restore health score history
            if "health_score_history" in session_state:
                self.health_score_history = deque(session_state["health_score_history"], maxlen=100)

            # Restore all state components
            self._restore_metrics_from_state(session_state)
            self._restore_alerts_from_state(session_state)
            self._restore_error_data_from_state(session_state)
            self._restore_performance_data_from_state(session_state)
            self._restore_intervention_state_from_state(session_state)

            logger.info(f"🔄 Session state restored from checkpoint: {checkpoint_file.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore from checkpoint: {e}")
            return False

    def auto_checkpoint(self, interval_minutes: int = 30) -> None:
        """Automatically create checkpoints at regular intervals."""
        try:
            current_time = time.time()
            if not hasattr(self, '_last_checkpoint_time'):
                self._last_checkpoint_time = current_time

            # Check if it's time for a checkpoint
            if current_time - self._last_checkpoint_time >= (interval_minutes * 60):
                checkpoint_name = f"auto_checkpoint_{int(current_time)}"
                self.create_session_checkpoint(checkpoint_name)
                self._last_checkpoint_time = current_time

                # Clean up old auto checkpoints (keep only last 5)
                self._cleanup_old_checkpoints()

        except Exception as e:
            logger.debug(f"Auto checkpoint failed: {e}")

    @staticmethod
    def _cleanup_old_checkpoints(keep_count: int = 5) -> None:
        """Clean up old checkpoint files, keeping only the most recent ones."""
        try:
            checkpoint_dir = Path("Cache/session_checkpoints")
            if not checkpoint_dir.exists():
                return

            # Get all auto checkpoint files
            auto_checkpoints = list(checkpoint_dir.glob("auto_checkpoint_*.json"))

            # Sort by modification time (newest first)
            auto_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Remove old checkpoints
            for checkpoint in auto_checkpoints[keep_count:]:
                try:
                    checkpoint.unlink()
                    logger.debug(f"Removed old checkpoint: {checkpoint.name}")
                except Exception as e:
                    logger.debug(f"Failed to remove checkpoint {checkpoint.name}: {e}")

        except Exception as e:
            logger.debug(f"Checkpoint cleanup failed: {e}")
        else:
            try:
                from caching.cache_retention import auto_enforce_retention

                auto_enforce_retention("session_checkpoints")
            except Exception as retention_error:
                logger.debug("Retention sweep for session checkpoints skipped: %s", retention_error)

    @staticmethod
    def list_available_checkpoints() -> list[dict[str, Any]]:
        """List all available checkpoint files with metadata."""
        try:
            checkpoint_dir = Path("Cache/session_checkpoints")
            if not checkpoint_dir.exists():
                return []

            checkpoints: list[dict[str, Any]] = []
            for checkpoint_file in checkpoint_dir.glob("*.json"):
                try:
                    # Get file metadata
                    stat = checkpoint_file.stat()

                    # Try to read checkpoint metadata
                    with checkpoint_file.open(encoding='utf-8') as f:
                        data = json.load(f)

                    checkpoints.append(
                        {
                            "name": checkpoint_file.stem,
                            "file_path": str(checkpoint_file),
                            "created_time": data.get("timestamp", stat.st_mtime),
                            "file_size_kb": stat.st_size / 1024,
                            "session_start_time": data.get("session_start_time"),
                            "health_score": data.get("performance_stats", {}).get("health_score", "N/A"),
                            "total_errors": data.get("performance_stats", {}).get("total_errors", 0),
                        }
                    )

                except Exception as e:
                    logger.debug(f"Failed to read checkpoint metadata for {checkpoint_file.name}: {e}")

            # Sort by creation time (newest first)
            checkpoints.sort(key=lambda x: x["created_time"], reverse=True)
            return checkpoints

        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []

    def persist_session_state_to_disk(self, session_data: Optional[dict[str, Any]] = None) -> str:
        """Persist current session state to disk for crash recovery."""
        try:
            # Create persistent state directory
            state_dir = Path("Cache/session_state")
            state_dir.mkdir(parents=True, exist_ok=True)

            # Prepare session state data
            if session_data is None:
                session_data = {}

            # Add health monitoring state
            session_data.update(
                {
                    "health_monitor": {
                        "session_start_time": self.session_start_time,
                        "current_metrics": {name: metric.value for name, metric in self.current_metrics.items()},
                        "error_count": len(self.error_timestamps),
                        "alert_count": len(self.alerts),
                        "health_score": self.calculate_health_score(),
                        "intervention_state": {
                            "emergency_halt": self._emergency_halt_requested,
                            "immediate_intervention": self._immediate_intervention_requested,
                            "enhanced_monitoring": self._enhanced_monitoring_active,
                        },
                    },
                    "timestamp": time.time(),
                }
            )

            # Save to persistent state file
            state_file = state_dir / "current_session.json"
            with state_file.open('w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, default=str)

            logger.debug(f"Session state persisted to disk: {state_file}")

            try:
                from caching.cache_retention import auto_enforce_retention

                auto_enforce_retention("session_state")
            except Exception as retention_error:
                logger.debug("Retention sweep for session state skipped: %s", retention_error)

            return str(state_file)

        except Exception as e:
            logger.error(f"Failed to persist session state: {e}")
            return ""

    def recover_session_state_from_disk(self) -> Optional[dict[str, Any]]:
        """Recover session state from disk after a crash."""
        try:
            state_file = Path("Cache/session_state/current_session.json")
            if not state_file.exists():
                logger.info("No previous session state found")
                return None

            # Load session state
            with state_file.open(encoding='utf-8') as f:
                session_data = json.load(f)

            # Check if state is recent (within last 24 hours)
            state_age = time.time() - session_data.get("timestamp", 0)
            if state_age > 86400:  # 24 hours
                logger.info("Previous session state is too old, ignoring")
                return None

            # Restore health monitoring state if available
            if "health_monitor" in session_data:
                health_data = session_data["health_monitor"]

                # Restore basic state
                if "session_start_time" in health_data:
                    self.session_start_time = health_data["session_start_time"]

                # Restore intervention state
                if "intervention_state" in health_data:
                    intervention = health_data["intervention_state"]
                    self._emergency_halt_requested = intervention.get("emergency_halt", False)
                    self._immediate_intervention_requested = intervention.get("immediate_intervention", False)
                    self._enhanced_monitoring_active = intervention.get("enhanced_monitoring", False)

            logger.info(f"🔄 Session state recovered from disk (age: {state_age / 60:.1f} minutes)")
            return session_data

        except Exception as e:
            logger.error(f"Failed to recover session state: {e}")
            return None


class SessionHealthMonitor(
    MetricsManagementMixin,
    AlertingMixin,
    HealthAssessmentMixin,
    InterventionMixin,
    ResourceManagementMixin,
    PersistenceMixin,
):
    """
    Monitor session health and performance metrics.
    Tracks error rates, API response times, and resource usage to prevent session death.
    """

    def __init__(self) -> None:
        """Initialize health monitor."""
        self.session_start_time = time.time()
        self.current_metrics: dict[str, HealthMetric] = {}
        self.metrics_history: dict[str, deque[float]] = {}
        self.alerts: list[HealthAlert] = []
        self.last_alert_times: dict[str, float] = {}

        # Performance tracking
        self.api_response_times: deque[float] = deque(maxlen=1000)
        self.error_timestamps: deque[float] = deque(maxlen=1000)
        self.error_counts: dict[str, int] = {}
        self.page_processing_times: deque[float] = deque(maxlen=100)
        self.memory_usage_history: deque[float] = deque(maxlen=100)
        self.health_score_history: deque[float] = deque(maxlen=100)

        # Intervention flags
        self._emergency_halt_requested = False
        self._emergency_halt_reason = ""
        self._emergency_halt_timestamp = 0.0
        self._immediate_intervention_requested = False
        self._immediate_intervention_reason = ""
        self._immediate_intervention_timestamp = 0.0
        self._enhanced_monitoring_active = False
        self._enhanced_monitoring_reason = ""
        self._enhanced_monitoring_timestamp = 0.0

        # Dynamic flags
        self._action6_callback_registered: bool = False

        # Configuration
        self._monitoring_interval = 30.0  # Seconds
        self._adaptive_interval = True
        self._last_cleanup_time = time.time()
        self._cleanup_interval = 300.0  # 5 minutes

        # Warning tracking
        self.error_rate_warnings_sent: dict[str, float] = {}

        # Initialize default metrics
        self._initialize_metrics()


# Global health monitor instance
class _HealthMonitorSingleton:
    """Singleton container for health monitor instance."""

    instance: Optional[SessionHealthMonitor] = None


def get_health_monitor() -> SessionHealthMonitor:
    """Get the global health monitor instance."""
    if _HealthMonitorSingleton.instance is None:
        _HealthMonitorSingleton.instance = SessionHealthMonitor()
    return _HealthMonitorSingleton.instance


def initialize_health_monitoring() -> Any:
    """Initialize health monitoring system."""
    monitor = get_health_monitor()
    logger.info("Health monitoring system initialized")
    return monitor


# === INTEGRATION HELPERS ===


def integrate_with_session_manager(session_manager: Any) -> Any:
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


def integrate_with_action6(action6_module: Any) -> Any:
    """Integrate health monitoring with Action 6.

    Action 6 exposes a lightweight callback registration hook that is invoked
    whenever API performance is logged. We attach a listener that updates the
    health monitor with per-endpoint metrics while relying on the module's
    native logging to handle baseline response-time tracking.
    """

    monitor = get_health_monitor()

    if hasattr(action6_module, "register_api_metrics_callback"):
        already_registered = getattr(monitor, "_action6_callback_registered", False)
        if not already_registered:

            def _record_api_metrics(api_name: str, duration: float, status: str) -> None:
                metric_key = f"api_{api_name}_last_duration"
                monitor.update_metric(metric_key, duration)
                if status.lower().startswith("error"):
                    monitor.record_error(f"{api_name}_{status}")

            try:
                action6_module.register_api_metrics_callback(_record_api_metrics)
                monitor._action6_callback_registered = True
                logger.debug("Registered Action 6 API performance callback with health monitor")
            except Exception as integration_error:
                logger.debug(f"Failed to register Action 6 health callback: {integration_error}")

    return monitor


def get_performance_recommendations(health_score: float, risk_score: float) -> dict[str, Any]:
    """Get specific performance setting recommendations based on health."""
    recommendations: dict[str, Any] = {
        "max_concurrency": 1,
        "batch_size": 8,
        "requests_per_second": 0.3,
        "action_required": "continue",
    }

    if risk_score > 0.8:
        # Emergency settings
        recommendations.update(
            {
                "max_concurrency": 1,
                "batch_size": 1,
                "requests_per_second": 0.2,
                "action_required": "emergency_refresh",
            }
        )
    elif risk_score > 0.6:
        # Critical settings
        recommendations.update(
            {
                "max_concurrency": 1,
                "batch_size": 3,
                "requests_per_second": 0.22,
                "action_required": "immediate_refresh",
            }
        )
    elif risk_score > 0.4:
        # Warning settings
        recommendations.update(
            {
                "max_concurrency": 1,
                "batch_size": 5,
                "requests_per_second": 0.26,
                "action_required": "schedule_refresh",
            }
        )
    elif health_score > 80:
        # Excellent health - can optimize
        recommendations.update(
            {
                "max_concurrency": 1,
                "batch_size": 10,
                "requests_per_second": 0.35,
                "action_required": "optimize",
            }
        )

    return recommendations


# === SESSION STATE PERSISTENCE INTEGRATION ===


def enable_session_state_persistence(
    session_manager: Optional[Any] = None,
    auto_checkpoint_interval: int = 30,
) -> SessionHealthMonitor:
    """Enable session state persistence with automatic checkpointing."""
    monitor = get_health_monitor()

    try:
        # Try to recover previous session state
        recovered_state = monitor.recover_session_state_from_disk()
        if recovered_state:
            logger.info("🔄 Previous session state recovered")

            # If session manager provided, restore its state too
            if session_manager and "session_manager_state" in recovered_state:
                try:
                    sm_state = recovered_state["session_manager_state"]
                    # Restore session manager specific state if available
                    if hasattr(session_manager, 'restore_state'):
                        session_manager.restore_state(sm_state)
                except Exception as e:
                    logger.debug(f"Failed to restore session manager state: {e}")

        # Enable auto checkpointing
        monitor.auto_checkpoint(auto_checkpoint_interval)

        logger.info(f"📁 Session state persistence enabled (auto-checkpoint every {auto_checkpoint_interval} minutes)")
        return monitor

    except Exception as e:
        logger.error(f"Failed to enable session state persistence: {e}")
        return monitor


def create_recovery_checkpoint(
    session_manager: Optional[Any] = None,
    checkpoint_name: str = "recovery_checkpoint",
) -> str:
    """Create a recovery checkpoint with session manager state."""
    monitor = get_health_monitor()

    try:
        # Gather session manager state if available
        session_data: dict[str, Any] = {}
        if session_manager:
            try:
                # Capture session manager state
                session_data["session_manager_state"] = {
                    "session_ready": getattr(session_manager, 'session_ready', False),
                    "browser_active": hasattr(session_manager, 'browser_manager')
                    and getattr(session_manager.browser_manager, 'driver', None) is not None,
                    "database_ready": hasattr(session_manager, 'db_manager'),
                    "api_ready": hasattr(session_manager, 'api_manager'),
                    "current_action": getattr(session_manager, 'current_action', None),
                    "pages_processed": getattr(session_manager, 'pages_processed', 0),
                    "matches_processed": getattr(session_manager, 'matches_processed', 0),
                }

                # Add browser health state if available
                if hasattr(session_manager, 'browser_health_monitor'):
                    session_data["browser_health"] = session_manager.browser_health_monitor

                # Add session health state if available
                if hasattr(session_manager, 'session_health_monitor'):
                    session_data["session_health"] = session_manager.session_health_monitor

            except Exception as e:
                logger.debug(f"Failed to capture session manager state: {e}")

        # Create checkpoint with session data
        checkpoint_path = monitor.create_session_checkpoint(checkpoint_name)

        # Also persist to disk for crash recovery
        monitor.persist_session_state_to_disk(session_data)

        logger.info(f"🛡️ Recovery checkpoint created: {checkpoint_name}")
        return checkpoint_path

    except Exception as e:
        logger.error(f"Failed to create recovery checkpoint: {e}")
        return ""


def get_session_recovery_status() -> dict[str, Any]:
    """Get session recovery status and available checkpoints."""
    monitor = get_health_monitor()

    try:
        checkpoints = monitor.list_available_checkpoints()

        # Check for crash recovery state
        crash_recovery_available = False
        try:
            state_file = Path("Cache/session_state/current_session.json")
            if state_file.exists():
                state_age = time.time() - state_file.stat().st_mtime
                crash_recovery_available = state_age < 86400  # Less than 24 hours old
        except Exception as exc:
            logger.debug("Could not check crash recovery state: %s", exc)

        return {
            "checkpoints_available": len(checkpoints),
            "latest_checkpoint": checkpoints[0] if checkpoints else None,
            "crash_recovery_available": crash_recovery_available,
            "checkpoints": checkpoints[:5],  # Last 5 checkpoints
            "recovery_recommendations": _get_recovery_recommendations(checkpoints, crash_recovery_available),
        }

    except Exception as e:
        logger.error(f"Failed to get recovery status: {e}")
        return {"error": str(e)}


def _get_recovery_recommendations(
    checkpoints: list[dict[str, Any]],
    crash_recovery: bool,
) -> list[str]:
    """Get recovery recommendations based on available data."""
    recommendations: list[str] = []

    if crash_recovery:
        recommendations.append("Crash recovery state available - consider recovering from last session")

    if checkpoints:
        latest = checkpoints[0]
        age_hours = (time.time() - latest["created_time"]) / 3600
        if age_hours < 1:
            recommendations.append(f"Recent checkpoint available ({age_hours:.1f}h old)")
        elif age_hours < 24:
            recommendations.append(f"Checkpoint available from {age_hours:.1f} hours ago")
    else:
        recommendations.append("No checkpoints available - consider enabling auto-checkpointing")

    return recommendations


# === TEST FRAMEWORK ===

# ==============================================
# MODULE-LEVEL TEST FUNCTIONS
# ==============================================


def _test_health_monitor_initialization() -> bool:
    """Test health monitor initialization."""
    monitor = SessionHealthMonitor()
    assert monitor is not None
    assert len(monitor.current_metrics) > 0
    assert monitor.session_start_time > 0
    return True


def _test_metric_updates() -> bool:
    """Test metric update functionality."""
    monitor = SessionHealthMonitor()
    monitor.update_metric("api_response_time", 3.5)
    assert monitor.current_metrics["api_response_time"].value == 3.5
    assert len(monitor.metrics_history["api_response_time"]) == 1
    return True


def _test_health_score_calculation() -> bool:
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


def _test_risk_prediction() -> bool:
    """Test session death risk prediction."""
    monitor = SessionHealthMonitor()
    risk = monitor.predict_session_death_risk()
    assert 0.0 <= risk <= 1.0, f"Risk score should be 0-1, got {risk}"
    # Test with bad conditions
    monitor.api_response_times.extend([10.0, 12.0, 15.0, 20.0, 25.0])
    monitor.update_metric("error_rate", 15.0)
    monitor.memory_usage_history.extend([100, 200, 300] * 4)
    risk = monitor.predict_session_death_risk()
    assert risk > 0.3, f"Expected elevated risk with bad conditions, got {risk}"
    return True


def _test_alert_system() -> bool:
    """Test health alert system."""
    monitor = SessionHealthMonitor()
    monitor.update_metric("api_response_time", 3.0)
    warning_alerts = [a for a in monitor.alerts if a.level == AlertLevel.WARNING]
    assert len(warning_alerts) > 0, "Warning alert should have been created"
    monitor.update_metric("api_response_time", 26.0)
    critical_alerts = [a for a in monitor.alerts if a.level == AlertLevel.CRITICAL]
    assert len(critical_alerts) > 0, "Critical alert should have been created"
    return True


def _test_performance_tracking() -> bool:
    """Test performance tracking functionality."""
    monitor = SessionHealthMonitor()
    monitor.record_api_response_time(2.5)
    assert len(monitor.api_response_times) == 1
    monitor.record_error("ConnectionError")
    assert monitor.error_counts["ConnectionError"] == 1
    monitor.record_page_processing_time(45.0)
    assert len(monitor.page_processing_times) == 1
    return True


def _test_dashboard_generation() -> bool:
    """Test health dashboard generation."""
    monitor = SessionHealthMonitor()
    dashboard = monitor.get_health_dashboard()
    required_fields = ["health_score", "health_status", "risk_score", "metrics", "recommended_actions"]
    for req_field in required_fields:
        assert req_field in dashboard, f"Dashboard missing required field: {req_field}"
    assert isinstance(dashboard["health_score"], (int, float))
    assert isinstance(dashboard["risk_score"], (float))
    assert isinstance(dashboard["metrics"], dict)
    assert isinstance(dashboard["recommended_actions"], list)
    return True


def _test_integration_helpers() -> bool:
    """Test integration helper functions."""
    recommendations = get_performance_recommendations(90.0, 0.1)
    assert "max_concurrency" in recommendations
    assert "action_required" in recommendations
    emergency_recs = get_performance_recommendations(20.0, 0.9)
    assert emergency_recs["max_concurrency"] == 1
    assert emergency_recs["action_required"] == "emergency_refresh"
    return True


def _test_global_instance() -> bool:
    """Test global health monitor instance."""
    monitor1 = get_health_monitor()
    monitor2 = get_health_monitor()
    assert monitor1 is monitor2, "get_health_monitor should return singleton instance"
    return True


def _test_memory_pressure_monitoring() -> bool:
    """Test health monitoring under memory pressure conditions."""
    import time

    monitor = SessionHealthMonitor()
    monitor.optimize_for_long_session()
    assert monitor._monitoring_interval == 60.0, "Should optimize monitoring interval"
    assert monitor.error_timestamps.maxlen == 3000, "Should increase error timestamp capacity"
    stats = monitor.get_performance_stats()
    assert "memory_efficiency" in stats, "Should include memory efficiency stats"
    assert "error_timestamps_usage" in stats["memory_efficiency"], "Should track memory usage"
    current_time = time.time()
    for i in range(100):
        monitor.error_timestamps.append(current_time - i)
    assert len(monitor.error_timestamps) == 100, "Should have 100 error timestamps"
    assert monitor.error_timestamps.maxlen == 3000, "Should have increased capacity for long sessions"
    return True


def _test_resource_constraint_handling() -> bool:
    """Test system behavior under resource constraints."""
    import time

    monitor = SessionHealthMonitor()
    current_time = time.time()
    for i in range(100):
        monitor.error_timestamps.append(current_time - i)
    interval = monitor._get_adaptive_monitoring_interval(current_time)
    assert interval <= 10.0, f"Should use short interval under high error rate, got {interval}"
    monitor._trigger_enhanced_monitoring("RESOURCE_CONSTRAINT", 100, "5-minute")
    assert monitor.is_enhanced_monitoring_active(), "Should activate enhanced monitoring"
    interval = monitor._get_adaptive_monitoring_interval(current_time)
    assert interval == 5.0, f"Enhanced monitoring should use 5s interval, got {interval}"
    return True


def _test_long_session_resource_management() -> bool:
    """Test resource management for 20+ hour sessions."""
    import time

    monitor = SessionHealthMonitor()
    session_start = time.time() - (20 * 3600)
    monitor.session_start_time = session_start
    monitor.optimize_for_long_session()
    current_time = time.time()
    for i in range(200):
        error_time = session_start + (i * (20 * 3600) / 200)
        monitor.error_timestamps.append(error_time)
    for i in range(100):
        alert_time = session_start + (i * (20 * 3600) / 100)
        alert = HealthAlert(
            level=AlertLevel.WARNING,
            component="test",
            message=f"Test alert {i}",
            metric_name="test_metric",
            metric_value=i,
            threshold=50,
            timestamp=alert_time,
        )
        monitor.alerts.append(alert)
    stats = monitor.get_performance_stats()
    assert stats["session_duration_hours"] >= 19.9, "Should report ~20 hour session"
    assert stats["error_timestamps_count"] == 200, "Should track all errors"
    assert stats["alerts_count"] == 100, "Should track all alerts"
    initial_alerts = len(monitor.alerts)
    initial_errors = len(monitor.error_timestamps)
    monitor._last_cleanup_time = current_time - 601
    monitor._perform_efficient_cleanup(current_time)
    final_alerts = len(monitor.alerts)
    final_errors = len(monitor.error_timestamps)
    assert final_alerts <= initial_alerts, "Should clean up old alerts"
    assert final_errors <= initial_errors, "Should clean up old errors"
    assert final_alerts > 0, "Should keep some recent alerts"
    assert final_errors > 0, "Should keep some recent errors"
    return True


def _test_session_checkpoint_creation() -> bool:
    """Test session checkpoint creation and restoration."""
    from pathlib import Path

    monitor = SessionHealthMonitor()
    monitor.update_metric("error_rate", 50.0)
    monitor.record_error("test_error")
    checkpoint_path = monitor.create_session_checkpoint("test_checkpoint")
    assert checkpoint_path, "Should create checkpoint successfully"
    assert Path(checkpoint_path).exists(), "Checkpoint file should exist"
    new_monitor = SessionHealthMonitor()
    success = new_monitor.restore_from_checkpoint(checkpoint_path)
    assert success, "Should restore checkpoint successfully"
    assert "error_rate" in new_monitor.current_metrics, "Should restore metrics"
    assert new_monitor.current_metrics["error_rate"].value >= 0, "Should restore metric with valid value"
    assert len(new_monitor.error_timestamps) > 0, "Should restore error timestamps"
    return True


def _test_session_state_persistence() -> bool:
    """Test session state persistence to disk."""
    monitor = SessionHealthMonitor()
    monitor.update_metric("memory_usage_mb", 75.0)
    monitor.record_error("persistence_error")
    test_data = {"test_key": "test_value", "page_count": 100}
    state_file = monitor.persist_session_state_to_disk(test_data)
    assert state_file, "Should persist session state successfully"
    recovered_data = monitor.recover_session_state_from_disk()
    assert recovered_data is not None, "Should recover session state"
    assert "test_key" in recovered_data, "Should recover custom data"
    assert "health_monitor" in recovered_data, "Should include health monitor state"
    return True


def _test_checkpoint_management() -> bool:
    """Test checkpoint listing and cleanup functionality."""
    monitor = SessionHealthMonitor()
    monitor.create_session_checkpoint("test_checkpoint_1")
    monitor.create_session_checkpoint("test_checkpoint_2")
    checkpoints = monitor.list_available_checkpoints()
    assert len(checkpoints) >= 2, "Should list created checkpoints"
    for checkpoint in checkpoints:
        assert "name" in checkpoint, "Should include checkpoint name"
        assert "created_time" in checkpoint, "Should include creation time"
        assert "file_size_kb" in checkpoint, "Should include file size"
    return True


def _test_auto_checkpoint_functionality() -> bool:
    """Test automatic checkpoint creation."""
    import time

    monitor = SessionHealthMonitor()
    monitor.auto_checkpoint(interval_minutes=1)
    monitor._last_checkpoint_time = time.time() - 120
    monitor.auto_checkpoint(interval_minutes=1)
    checkpoints = monitor.list_available_checkpoints()
    auto_checkpoints = [cp for cp in checkpoints if cp["name"].startswith("auto_checkpoint")]
    assert len(auto_checkpoints) >= 1, "Should create auto checkpoints"
    return True


# ==============================================
# MAIN TEST SUITE RUNNER
# ==============================================


def health_monitor_tests() -> bool:
    """Run all health monitor tests and return True if successful."""
    from testing.test_framework import suppress_logging

    test_suite = TestSuite("Health Monitoring System", "health_monitor.py")

    # Define all tests in a data structure to reduce complexity
    tests = [
        (
            "Health Monitor Initialization",
            _test_health_monitor_initialization,
            "Monitor initializes correctly",
            "direct",
            "Test initialization",
        ),
        ("Metric Updates", _test_metric_updates, "Metrics update correctly", "direct", "Test metric updates"),
        (
            "Health Score Calculation",
            _test_health_score_calculation,
            "Score reflects metrics",
            "direct",
            "Test health scoring",
        ),
        ("Risk Prediction", _test_risk_prediction, "Risk score calculation works", "direct", "Test risk prediction"),
        ("Alert System", _test_alert_system, "Alerts triggered correctly", "direct", "Test alert system"),
        (
            "Performance Tracking",
            _test_performance_tracking,
            "Performance tracked",
            "direct",
            "Test performance tracking",
        ),
        (
            "Dashboard Generation",
            _test_dashboard_generation,
            "Dashboard generated",
            "direct",
            "Test dashboard generation",
        ),
        ("Integration Helpers", _test_integration_helpers, "Helpers work", "direct", "Test integration helpers"),
        ("Global Instance", _test_global_instance, "Test global instance", "direct", "Test global instance"),
        (
            "Memory Pressure Monitoring",
            _test_memory_pressure_monitoring,
            "Test memory monitoring",
            "direct",
            "Test memory monitoring",
        ),
        (
            "Resource Constraint Handling",
            _test_resource_constraint_handling,
            "Test resource constraints",
            "direct",
            "Test resource constraints",
        ),
        (
            "Long Session Resource Management",
            _test_long_session_resource_management,
            "Test long session management",
            "direct",
            "Test long session management",
        ),
        (
            "Session Checkpoint Creation",
            _test_session_checkpoint_creation,
            "Test checkpoint creation",
            "direct",
            "Test checkpoint creation",
        ),
        (
            "Session State Persistence",
            _test_session_state_persistence,
            "Test state persistence",
            "direct",
            "Test state persistence",
        ),
        (
            "Checkpoint Management",
            _test_checkpoint_management,
            "Test checkpoint management",
            "direct",
            "Test checkpoint management",
        ),
        (
            "Auto Checkpoint Functionality",
            _test_auto_checkpoint_functionality,
            "Test auto checkpoint",
            "direct",
            "Test auto checkpoint",
        ),
    ]

    # Run all tests from the list
    with suppress_logging():
        for test_name, test_func, expected_behavior, test_description, method_description in tests:
            test_suite.run_test(test_name, test_func, expected_behavior, test_description, method_description)

    return test_suite.finish_suite()


# Register functions for external access

auto_register_module(globals(), __name__)


# Use centralized test runner utility

run_comprehensive_tests = create_standard_test_runner(health_monitor_tests)


if __name__ == "__main__":
    # Run tests when executed directly
    success = health_monitor_tests()
    if success:
        print("🎉 All health monitor tests passed!")
    else:
        print("❌ Some health monitor tests failed!")
        sys.exit(1)
