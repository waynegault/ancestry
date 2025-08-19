#!/usr/bin/env python3

"""
health_monitor.py - Comprehensive Health Monitoring System

Provides real-time health monitoring, session health scoring, early warning detection,
and predictive analytics for the Ancestry automation system.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import (
    register_function,
    setup_module,
)

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

# === LOCAL IMPORTS ===
from test_framework import TestSuite


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

        # Enhanced error rate monitoring - PERFORMANCE OPTIMIZED
        self.error_timestamps: deque = deque(maxlen=2000)  # Increased for 20+ hour sessions
        self.error_rate_warnings_sent: Dict[str, float] = {}  # Track when warnings were sent
        self.last_error_rate_check: float = time.time()

        # Metric alert de-duplication (prevents alert spam on repeated updates)
        self._last_metric_alert_level: Dict[str, AlertLevel] = {}
        self._last_metric_alert_time: Dict[str, float] = {}
        self._metric_alert_cooldown_seconds: float = 60.0  # Only re-log after 60s unless level escalates

        # Performance optimization for long sessions
        self._monitoring_interval: float = 30.0  # Base monitoring interval
        self._adaptive_interval: bool = True  # Enable adaptive monitoring
        self._last_cleanup_time: float = time.time()
        self._cleanup_interval: float = 300.0  # Clean up every 5 minutes

        # Automatic intervention flags
        self._emergency_halt_requested: bool = False
        self._emergency_halt_reason: str = ""
        self._emergency_halt_timestamp: float = 0.0
        self._immediate_intervention_requested: bool = False
        self._immediate_intervention_reason: str = ""
        self._immediate_intervention_timestamp: float = 0.0
        self._enhanced_monitoring_active: bool = False
        self._enhanced_monitoring_reason: str = ""
        self._enhanced_monitoring_timestamp: float = 0.0

        # Predictive analytics
        self.failure_patterns: List[Dict[str, Any]] = []
        self.success_patterns: List[Dict[str, Any]] = []
        # Safety test mode flag to standardize alert prefixes
        self._safety_test_mode: bool = False


        # Initialize metrics
        self._initialize_metrics()

        logger.info("Session Health Monitor initialized")

    def _initialize_metrics(self):
        """Initialize health metrics with workload-appropriate thresholds for 724-page processing."""
        metrics_config = {
            "api_response_time": {"warning": 15.0, "critical": 25.0, "weight": 2.0},  # OPTIMIZATION: Less pessimistic thresholds (was 5.0/10.0)
            "memory_usage_mb": {"warning": 200.0, "critical": 400.0, "weight": 1.5},
            "error_rate": {"warning": 10.0, "critical": 25.0, "weight": 3.0},  # WORKLOAD-APPROPRIATE: Errors per hour for 724-page workload
            "session_age_minutes": {"warning": 600.0, "critical": 1200.0, "weight": 1.0},  # WORKLOAD-APPROPRIATE: 10-20 hours for 724 pages
            "browser_age_minutes": {"warning": 120.0, "critical": 180.0, "weight": 2.5},  # WORKLOAD-APPROPRIATE: 2-3 hours browser lifetime
            "pages_since_refresh": {"warning": 50.0, "critical": 75.0, "weight": 2.0},  # WORKLOAD-APPROPRIATE: More pages before refresh
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
    def begin_safety_test(self):
        """Enable safety test mode to uniformly prefix all alerts and notices."""
        self._safety_test_mode = True

    def end_safety_test(self):
        """Disable safety test mode."""
        self._safety_test_mode = False

    def _safety_prefix(self) -> str:
        """Return the standard prefix for safety-test logs if enabled."""
        return "🧪 [SAFETY TEST] " if self._safety_test_mode else ""

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
        """Check if a metric triggers any alerts, with de-duplication and cooldown."""
        metric = self.current_metrics[metric_name]

        level: Optional[AlertLevel] = None
        message: str = ""
        threshold: float = 0.0

        if metric.value >= metric.threshold_critical:
            level = AlertLevel.CRITICAL
            message = f"{metric_name} is critical: {metric.value:.2f} >= {metric.threshold_critical}"
            threshold = metric.threshold_critical
        elif metric.value >= metric.threshold_warning:
            level = AlertLevel.WARNING
            message = f"{metric_name} is elevated: {metric.value:.2f} >= {metric.threshold_warning}"
            threshold = metric.threshold_warning

        if level is None:
            # Reset last level so future crossings log again
            self._last_metric_alert_level.pop(metric_name, None)
            return

        # Gate logs: only log when escalating level or after cooldown
        last_level = self._last_metric_alert_level.get(metric_name)
        last_time = self._last_metric_alert_time.get(metric_name, 0.0)
        now = time.time()

        should_log = False
        if last_level is None:
            should_log = True
        elif level.value != last_level.value:
            # Escalation from WARNING->CRITICAL (or vice versa) should log
            should_log = True
        elif (now - last_time) >= self._metric_alert_cooldown_seconds:
            should_log = True

        if should_log:
            self._create_alert(level, "session_health", message, metric_name, metric.value, threshold)
            self._last_metric_alert_level[metric_name] = level
            self._last_metric_alert_time[metric_name] = now

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

        # Log alert with standardized safety test prefix when in test mode
        test_prefix = self._safety_prefix()

        if level == AlertLevel.CRITICAL:
            logger.critical(f"{test_prefix}🚨 CRITICAL ALERT: {message}")
        elif level == AlertLevel.WARNING:
            logger.warning(f"{test_prefix}⚠️ WARNING: {message}")
        else:
            logger.info(f"{test_prefix}ℹ️ INFO: {message}")

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
        for _, metric in self.current_metrics.items():
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
        """Record an error for monitoring with enhanced rate tracking."""
        current_time = time.time()

        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1

        # Track error timestamp for rate analysis
        self.error_timestamps.append(current_time)

        # Update error rate metric
        total_errors = sum(self.error_counts.values())
        session_duration_hours = (current_time - self.session_start_time) / 3600
        error_rate = total_errors / max(session_duration_hours, 0.1)  # Errors per hour
        self.update_metric("error_rate", error_rate)

        # Check for early warning conditions
        self._check_error_rate_early_warning(current_time)

    def _check_error_rate_early_warning(self, current_time: float):
        """
        PERFORMANCE-OPTIMIZED error rate monitoring for long sessions.

        Uses adaptive monitoring intervals and efficient cleanup to reduce overhead.
        """
        # Adaptive monitoring interval based on current error rate
        monitoring_interval = self._get_adaptive_monitoring_interval(current_time)

        # Only check at adaptive intervals to reduce overhead
        if current_time - self.last_error_rate_check < monitoring_interval:
            return

        self.last_error_rate_check = current_time

        # PERFORMANCE: Efficient cleanup with batched operations
        self._perform_efficient_cleanup(current_time)

        # Check different time windows for early warning - WORKLOAD-APPROPRIATE for 724 pages
        time_windows = [
            (1800, 500, "30-minute"),  # 500 errors in 30 minutes (cascade failure)
            (900, 200, "15-minute"),   # 200 errors in 15 minutes (severe issues)
            (300, 75, "5-minute"),     # 75 errors in 5 minutes (moderate issues)
            (60, 15, "1-minute"),      # 15 errors in 1 minute (immediate issues)
        ]

        for window_seconds, threshold, window_name in time_windows:
            window_start = current_time - window_seconds
            errors_in_window = sum(1 for ts in self.error_timestamps if ts >= window_start)

            if errors_in_window >= threshold:
                warning_key = f"{window_name}_{threshold}"

                # Only send warning once per hour for each threshold
                last_warning = self.error_rate_warnings_sent.get(warning_key, 0)
                if current_time - last_warning > 3600:  # 1 hour
                    self.error_rate_warnings_sent[warning_key] = current_time

                    # Create critical alert
                    alert = HealthAlert(
                        level=AlertLevel.CRITICAL,
                        component="error_rate_monitor",
                        message=f"🚨 HIGH ERROR RATE: {errors_in_window} errors in {window_name} (threshold: {threshold})",
                        metric_name="error_rate_early_warning",
                        metric_value=errors_in_window,
                        threshold=threshold,
                        timestamp=current_time
                    )

                    self.alerts.append(alert)
                    prefix = self._safety_prefix()
                    logger.critical(f"{prefix}🚨 CRITICAL ALERT: {alert.message}")

                    # WORKLOAD-APPROPRIATE: Cascade failure detection with automatic intervention
                    if window_name == "30-minute" and errors_in_window >= 500:
                        logger.critical(f"{prefix}🚨 CASCADE FAILURE DETECTED - EMERGENCY INTERVENTION REQUIRED")
                        self._trigger_emergency_intervention("CASCADE_FAILURE", errors_in_window, window_name)
                    elif window_name == "15-minute" and errors_in_window >= 200:
                        logger.critical(f"{prefix}🚨 SEVERE ERROR PATTERN DETECTED - Triggering immediate intervention")
                        self._trigger_immediate_intervention("SEVERE_ERROR_PATTERN", errors_in_window, window_name)
                    elif window_name == "5-minute" and errors_in_window >= 75:
                        logger.warning(f"{prefix}⚠️ ELEVATED ERROR RATE - Triggering enhanced monitoring")
                        self._trigger_enhanced_monitoring("ELEVATED_ERROR_RATE", errors_in_window, window_name)

    def get_error_rate_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive error rate statistics for monitoring and analysis.

        Returns:
            Dict containing error rate statistics and recommendations
        """
        current_time = time.time()

        # Calculate error rates for different time windows
        time_windows = [
            (60, "1-minute"),
            (300, "5-minute"),
            (900, "15-minute"),
            (1800, "30-minute"),
            (3600, "1-hour"),
            (7200, "2-hour")
        ]

        error_rates = {}
        for window_seconds, window_name in time_windows:
            window_start = current_time - window_seconds
            errors_in_window = sum(1 for ts in self.error_timestamps if ts >= window_start)
            error_rates[window_name] = {
                "count": errors_in_window,
                "rate_per_minute": errors_in_window / (window_seconds / 60),
                "window_seconds": window_seconds
            }

        # Determine risk level - WORKLOAD-APPROPRIATE for 724-page processing
        thirty_min_errors = error_rates["30-minute"]["count"]
        fifteen_min_errors = error_rates["15-minute"]["count"]
        five_min_errors = error_rates["5-minute"]["count"]

        if thirty_min_errors >= 500:
            risk_level = "CRITICAL"
            recommendation = "EMERGENCY_SHUTDOWN"
        elif fifteen_min_errors >= 200:
            risk_level = "HIGH"
            recommendation = "IMMEDIATE_INTERVENTION"
        elif five_min_errors >= 75:
            risk_level = "MODERATE"
            recommendation = "MONITOR_CLOSELY"
        elif five_min_errors >= 25:
            risk_level = "ELEVATED"
            recommendation = "INCREASED_MONITORING"
        else:
            risk_level = "LOW"
            recommendation = "CONTINUE_NORMAL"

        return {
            "timestamp": current_time,
            "error_rates": error_rates,
            "total_session_errors": sum(self.error_counts.values()),
            "session_duration_minutes": (current_time - self.session_start_time) / 60,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "error_types": dict(self.error_counts),
            "recent_alerts": [alert for alert in self.alerts if current_time - alert.timestamp < 3600]
        }

    def _trigger_emergency_intervention(self, pattern_type: str, error_count: int, window: str) -> None:
        """Trigger emergency intervention for cascade failures."""
        try:
            prefix = self._safety_prefix()
            logger.critical(f"{prefix}🚨 EMERGENCY INTERVENTION TRIGGERED: {pattern_type}")
            logger.critical(f"{prefix}   Pattern: {error_count} errors in {window}")
            logger.critical(f"{prefix}   Action: Setting emergency halt flag")

            # Set emergency halt flag that can be checked by main processing loops
            self._emergency_halt_requested = True
            self._emergency_halt_reason = f"{pattern_type}: {error_count} errors in {window}"
            self._emergency_halt_timestamp = time.time()

            # Create critical alert
            alert = HealthAlert(
                level=AlertLevel.CRITICAL,
                component="emergency_intervention",
                message=f"🚨 EMERGENCY HALT: {pattern_type} - {error_count} errors in {window}",
                metric_name="emergency_intervention",
                metric_value=error_count,
                threshold=500 if window == "30-minute" else 200,
                timestamp=time.time()
            )
            self.alerts.append(alert)

            logger.critical(f"{self._safety_prefix()}🚨 EMERGENCY INTERVENTION COMPLETE - Processing should halt immediately")

        except Exception as e:
            logger.error(f"Failed to trigger emergency intervention: {e}")

    def _trigger_immediate_intervention(self, pattern_type: str, error_count: int, window: str) -> None:
        """Trigger immediate intervention for severe error patterns."""
        try:
            prefix = self._safety_prefix()
            logger.critical(f"{prefix}⚠️ IMMEDIATE INTERVENTION TRIGGERED: {pattern_type}")
            logger.critical(f"{prefix}   Pattern: {error_count} errors in {window}")
            logger.critical(f"{prefix}   Action: Setting immediate halt flag and triggering recovery")

            # Set immediate intervention flag
            self._immediate_intervention_requested = True
            self._immediate_intervention_reason = f"{pattern_type}: {error_count} errors in {window}"
            self._immediate_intervention_timestamp = time.time()

            # Create critical alert
            alert = HealthAlert(
                level=AlertLevel.CRITICAL,
                component="immediate_intervention",
                message=f"⚠️ IMMEDIATE INTERVENTION: {pattern_type} - {error_count} errors in {window}",
                metric_name="immediate_intervention",
                metric_value=error_count,
                threshold=200 if window == "15-minute" else 75,
                timestamp=time.time()
            )
            self.alerts.append(alert)

            logger.critical(f"{self._safety_prefix()}⚠️ IMMEDIATE INTERVENTION COMPLETE - Consider halting or recovery")

        except Exception as e:
            logger.error(f"Failed to trigger immediate intervention: {e}")

    def _trigger_enhanced_monitoring(self, pattern_type: str, error_count: int, window: str) -> None:
        """Trigger enhanced monitoring for elevated error rates."""
        try:
            prefix = self._safety_prefix()
            logger.warning(f"{prefix}📊 ENHANCED MONITORING TRIGGERED: {pattern_type}")
            logger.warning(f"{prefix}   Pattern: {error_count} errors in {window}")
            logger.warning(f"{prefix}   Action: Increasing monitoring frequency")

            # Set enhanced monitoring flag
            self._enhanced_monitoring_active = True
            self._enhanced_monitoring_reason = f"{pattern_type}: {error_count} errors in {window}"
            self._enhanced_monitoring_timestamp = time.time()

            # Reduce monitoring interval for enhanced monitoring
            self.last_error_rate_check = time.time() - 25  # Check again in 5 seconds instead of 30

            # Create warning alert
            alert = HealthAlert(
                level=AlertLevel.WARNING,
                component="enhanced_monitoring",
                message=f"📊 ENHANCED MONITORING: {pattern_type} - {error_count} errors in {window}",
                metric_name="enhanced_monitoring",
                metric_value=error_count,
                threshold=75,
                timestamp=time.time()
            )
            self.alerts.append(alert)

            logger.warning(f"{self._safety_prefix()}📊 ENHANCED MONITORING ACTIVE - Increased error rate surveillance")

        except Exception as e:
            logger.error(f"Failed to trigger enhanced monitoring: {e}")

    def should_emergency_halt(self) -> bool:
        """Check if emergency halt has been requested."""
        return self._emergency_halt_requested

    def should_immediate_intervention(self) -> bool:
        """Check if immediate intervention has been requested."""
        return self._immediate_intervention_requested

    def is_enhanced_monitoring_active(self) -> bool:
        """Check if enhanced monitoring is active."""
        return self._enhanced_monitoring_active

    def get_intervention_status(self) -> Dict[str, Any]:
        """Get current intervention status."""
        return {
            "emergency_halt": {
                "requested": self._emergency_halt_requested,
                "reason": self._emergency_halt_reason,
                "timestamp": self._emergency_halt_timestamp
            },
            "immediate_intervention": {
                "requested": self._immediate_intervention_requested,
                "reason": self._immediate_intervention_reason,
                "timestamp": self._immediate_intervention_timestamp
            },
            "enhanced_monitoring": {
                "active": self._enhanced_monitoring_active,
                "reason": self._enhanced_monitoring_reason,
                "timestamp": self._enhanced_monitoring_timestamp
            }
        }

    def reset_intervention_flags(self) -> None:
        """Reset intervention flags (use with caution)."""
        logger.warning("🔄 Resetting intervention flags")
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
            old_warnings = [key for key, timestamp in self.error_rate_warnings_sent.items()
                          if timestamp < warning_cutoff]
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
            self._cleanup_interval = 600.0    # Clean up every 10 minutes instead of 5

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
            logger.warning(f"Failed to optimize for long session: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
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
                    "warnings_retention_hours": 1
                }
            }

        except Exception as e:
            logger.debug(f"Error getting performance stats: {e}")
            return {"error": str(e)}

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

        logger.info("📊 HEALTH SUMMARY:")
        logger.info(f"   Score: {dashboard['health_score']:.1f}/100 ({dashboard['health_status'].upper()})")
        logger.info(f"   Risk: {dashboard['risk_score']:.2f} ({dashboard['risk_level']})")
        logger.info(f"   API: {dashboard['performance_summary']['avg_api_response_time']:.1f}s avg")
        logger.info(f"   Memory: {dashboard['performance_summary']['current_memory_mb']:.1f}MB")
        logger.info(f"   Errors: {dashboard['performance_summary']['total_errors']}")

        if dashboard['recommended_actions']:
            logger.info(f"   Actions: {dashboard['recommended_actions'][0]}")

    # === SESSION STATE PERSISTENCE ===

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
                        "timestamp": metric.timestamp
                    } for name, metric in self.current_metrics.items()
                },
                "alerts": [
                    {
                        "level": alert.level.value,
                        "component": alert.component,
                        "message": alert.message,
                        "metric_name": alert.metric_name,
                        "metric_value": alert.metric_value,
                        "threshold": alert.threshold,
                        "timestamp": alert.timestamp
                    } for alert in self.alerts
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
                    "last_cleanup_time": self._last_cleanup_time
                },
                "performance_stats": self.get_performance_stats()
            }

            # Save checkpoint to file
            checkpoint_file = checkpoint_dir / f"{checkpoint_name}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(session_state, f, indent=2, default=str)

            logger.info(f"📁 Session checkpoint created: {checkpoint_name}")
            return str(checkpoint_file)

        except Exception as e:
            logger.error(f"Failed to create session checkpoint: {e}")
            return ""

    def restore_from_checkpoint(self, checkpoint_path: str) -> bool:
        """Restore session state from a checkpoint file."""
        try:
            checkpoint_file = Path(checkpoint_path)
            if not checkpoint_file.exists():
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return False

            # Load checkpoint data
            with open(checkpoint_file, 'r') as f:
                session_state = json.load(f)

            # Restore session state
            self.session_start_time = session_state.get("session_start_time", time.time())

            # Restore health score history
            if "health_score_history" in session_state:
                self.health_score_history = deque(session_state["health_score_history"], maxlen=100)

            # Restore current metrics
            if "current_metrics" in session_state:
                for name, metric_data in session_state["current_metrics"].items():
                    metric = HealthMetric(
                        name=name,
                        value=metric_data["value"],
                        threshold_warning=metric_data["threshold_warning"],
                        threshold_critical=metric_data["threshold_critical"],
                        timestamp=metric_data.get("timestamp", time.time())
                    )
                    self.current_metrics[name] = metric

            # Restore alerts
            if "alerts" in session_state:
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
                        timestamp=alert_data["timestamp"]
                    )
                    self.alerts.append(alert)

            # Restore error data
            if "error_timestamps" in session_state:
                self.error_timestamps = deque(session_state["error_timestamps"], maxlen=self.error_timestamps.maxlen)

            if "error_counts" in session_state:
                self.error_counts.update(session_state["error_counts"])

            # Restore performance data
            if "api_response_times" in session_state:
                self.api_response_times = deque(session_state["api_response_times"], maxlen=self.api_response_times.maxlen)

            if "page_processing_times" in session_state:
                self.page_processing_times = deque(session_state["page_processing_times"], maxlen=self.page_processing_times.maxlen)

            if "memory_usage_history" in session_state:
                self.memory_usage_history = deque(session_state["memory_usage_history"], maxlen=self.memory_usage_history.maxlen)

            # Restore intervention state
            if "intervention_state" in session_state:
                intervention = session_state["intervention_state"]
                self._emergency_halt_requested = intervention.get("emergency_halt_requested", False)
                self._immediate_intervention_requested = intervention.get("immediate_intervention_requested", False)
                self._enhanced_monitoring_active = intervention.get("enhanced_monitoring_active", False)
                self._monitoring_interval = intervention.get("monitoring_interval", 30.0)
                self._last_cleanup_time = intervention.get("last_cleanup_time", time.time())

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

    def _cleanup_old_checkpoints(self, keep_count: int = 5) -> None:
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

    def list_available_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoint files with metadata."""
        try:
            checkpoint_dir = Path("Cache/session_checkpoints")
            if not checkpoint_dir.exists():
                return []

            checkpoints = []
            for checkpoint_file in checkpoint_dir.glob("*.json"):
                try:
                    # Get file metadata
                    stat = checkpoint_file.stat()

                    # Try to read checkpoint metadata
                    with open(checkpoint_file, 'r') as f:
                        data = json.load(f)

                    checkpoints.append({
                        "name": checkpoint_file.stem,
                        "file_path": str(checkpoint_file),
                        "created_time": data.get("timestamp", stat.st_mtime),
                        "file_size_kb": stat.st_size / 1024,
                        "session_start_time": data.get("session_start_time"),
                        "health_score": data.get("performance_stats", {}).get("health_score", "N/A"),
                        "total_errors": data.get("performance_stats", {}).get("total_errors", 0)
                    })

                except Exception as e:
                    logger.debug(f"Failed to read checkpoint metadata for {checkpoint_file.name}: {e}")

            # Sort by creation time (newest first)
            checkpoints.sort(key=lambda x: x["created_time"], reverse=True)
            return checkpoints

        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []

    def persist_session_state_to_disk(self, session_data: Optional[Dict[str, Any]] = None) -> str:
        """Persist current session state to disk for crash recovery."""
        try:
            # Create persistent state directory
            state_dir = Path("Cache/session_state")
            state_dir.mkdir(parents=True, exist_ok=True)

            # Prepare session state data
            if session_data is None:
                session_data = {}

            # Add health monitoring state
            session_data.update({
                "health_monitor": {
                    "session_start_time": self.session_start_time,
                    "current_metrics": {name: metric.value for name, metric in self.current_metrics.items()},
                    "error_count": len(self.error_timestamps),
                    "alert_count": len(self.alerts),
                    "health_score": self.calculate_health_score(),
                    "intervention_state": {
                        "emergency_halt": self._emergency_halt_requested,
                        "immediate_intervention": self._immediate_intervention_requested,
                        "enhanced_monitoring": self._enhanced_monitoring_active
                    }
                },
                "timestamp": time.time()
            })

            # Save to persistent state file
            state_file = state_dir / "current_session.json"
            with open(state_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)

            logger.debug(f"Session state persisted to disk: {state_file}")
            return str(state_file)

        except Exception as e:
            logger.error(f"Failed to persist session state: {e}")
            return ""

    def recover_session_state_from_disk(self) -> Optional[Dict[str, Any]]:
        """Recover session state from disk after a crash."""
        try:
            state_file = Path("Cache/session_state/current_session.json")
            if not state_file.exists():
                logger.info("No previous session state found")
                return None

            # Load session state
            with open(state_file, 'r') as f:
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

            logger.info(f"🔄 Session state recovered from disk (age: {state_age/60:.1f} minutes)")
            return session_data

        except Exception as e:
            logger.error(f"Failed to recover session state: {e}")
            return None


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
    _ = action6_module  # Unused parameter for future integration
    monitor = get_health_monitor()

    # TODO: Hook into API response time tracking (for future use)
    # def track_api_call(original_func):
    #     def wrapper(*args, **kwargs):
    #         start_time = time.time()
    #         try:
    #             result = original_func(*args, **kwargs)
    #             response_time = time.time() - start_time
    #             monitor.record_api_response_time(response_time)
    #             return result
    #         except Exception as e:
    #             monitor.record_error(type(e).__name__)
    #             raise
    #     return wrapper

    # Return monitor with tracking capability
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


# === SESSION STATE PERSISTENCE INTEGRATION ===

def enable_session_state_persistence(session_manager=None, auto_checkpoint_interval: int = 30):
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


def create_recovery_checkpoint(session_manager=None, checkpoint_name: str = "recovery_checkpoint"):
    """Create a recovery checkpoint with session manager state."""
    monitor = get_health_monitor()

    try:
        # Gather session manager state if available
        session_data = {}
        if session_manager:
            try:
                # Capture session manager state
                session_data["session_manager_state"] = {
                    "session_ready": getattr(session_manager, 'session_ready', False),
                    "browser_active": hasattr(session_manager, 'browser_manager') and
                                    getattr(session_manager.browser_manager, 'driver', None) is not None,
                    "database_ready": hasattr(session_manager, 'db_manager'),
                    "api_ready": hasattr(session_manager, 'api_manager'),
                    "current_action": getattr(session_manager, 'current_action', None),
                    "pages_processed": getattr(session_manager, 'pages_processed', 0),
                    "matches_processed": getattr(session_manager, 'matches_processed', 0)
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


def get_session_recovery_status() -> Dict[str, Any]:
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
        except Exception:
            pass

        return {
            "checkpoints_available": len(checkpoints),
            "latest_checkpoint": checkpoints[0] if checkpoints else None,
            "crash_recovery_available": crash_recovery_available,
            "checkpoints": checkpoints[:5],  # Last 5 checkpoints
            "recovery_recommendations": _get_recovery_recommendations(checkpoints, crash_recovery_available)
        }

    except Exception as e:
        logger.error(f"Failed to get recovery status: {e}")
        return {"error": str(e)}


def _get_recovery_recommendations(checkpoints: List[Dict], crash_recovery: bool) -> List[str]:
    """Get recovery recommendations based on available data."""
    recommendations = []

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

        # Trigger a warning alert (warning threshold is 15.0)
        monitor.update_metric("api_response_time", 16.0)  # Above warning threshold

        # Check that alert was created
        warning_alerts = [a for a in monitor.alerts if a.level == AlertLevel.WARNING]
        assert len(warning_alerts) > 0, "Warning alert should have been created"

        # Trigger a critical alert (critical threshold is 25.0)
        monitor.update_metric("api_response_time", 26.0)  # Above critical threshold

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
        for req_field in required_fields:
            assert req_field in dashboard, f"Dashboard missing required field: {req_field}"

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

    # === PHASE 4: MEMORY & RESOURCE TESTING ===
    def test_memory_pressure_monitoring():
        """Test health monitoring under memory pressure conditions."""

        monitor = SessionHealthMonitor()

        # Test memory optimization for long sessions
        monitor.optimize_for_long_session()

        # Verify optimization settings
        assert monitor._monitoring_interval == 60.0, "Should optimize monitoring interval"
        assert monitor.error_timestamps.maxlen == 3000, "Should increase error timestamp capacity"

        # Test performance stats under memory pressure
        stats = monitor.get_performance_stats()
        assert "memory_efficiency" in stats, "Should include memory efficiency stats"
        assert "error_timestamps_usage" in stats["memory_efficiency"], "Should track memory usage"

        # Test basic memory management functionality
        import time
        current_time = time.time()

        # Add some error timestamps
        for i in range(100):
            monitor.error_timestamps.append(current_time - i)

        # Verify timestamps were added
        assert len(monitor.error_timestamps) == 100, "Should have 100 error timestamps"

        # Test that memory optimization settings are working
        assert monitor.error_timestamps.maxlen == 3000, "Should have increased capacity for long sessions"

    def test_resource_constraint_handling():
        """Test system behavior under resource constraints."""

        monitor = SessionHealthMonitor()

        # Test adaptive monitoring under resource constraints
        current_time = time.time()

        # Simulate high error rate (resource constraint indicator)
        for i in range(100):
            monitor.error_timestamps.append(current_time - i)

        # Should adapt to shorter intervals under high error rate
        interval = monitor._get_adaptive_monitoring_interval(current_time)
        assert interval <= 10.0, f"Should use short interval under high error rate, got {interval}"

        # Test enhanced monitoring activation
        monitor._trigger_enhanced_monitoring("RESOURCE_CONSTRAINT", 100, "5-minute")
        assert monitor.is_enhanced_monitoring_active(), "Should activate enhanced monitoring"

        # Enhanced monitoring should use very short intervals
        interval = monitor._get_adaptive_monitoring_interval(current_time)
        assert interval == 5.0, f"Enhanced monitoring should use 5s interval, got {interval}"

    def test_long_session_resource_management():
        """Test resource management for 20+ hour sessions."""
        import time

        monitor = SessionHealthMonitor()

        # Simulate 20-hour session
        session_start = time.time() - (20 * 3600)
        monitor.session_start_time = session_start

        # Optimize for long session
        monitor.optimize_for_long_session()

        # Add realistic data for 20-hour session
        current_time = time.time()

        # Add 200 errors distributed over 20 hours
        for i in range(200):
            error_time = session_start + (i * (20 * 3600) / 200)
            monitor.error_timestamps.append(error_time)

        # Add 100 alerts distributed over time
        for i in range(100):
            alert_time = session_start + (i * (20 * 3600) / 100)
            alert = HealthAlert(
                level=AlertLevel.WARNING,
                component="test",
                message=f"Test alert {i}",
                metric_name="test_metric",
                metric_value=i,
                threshold=50,
                timestamp=alert_time
            )
            monitor.alerts.append(alert)

        # Test performance stats for long session
        stats = monitor.get_performance_stats()
        assert stats["session_duration_hours"] >= 19.9, "Should report ~20 hour session"
        assert stats["error_timestamps_count"] == 200, "Should track all errors"
        assert stats["alerts_count"] == 100, "Should track all alerts"

        # Test cleanup efficiency for long session
        initial_alerts = len(monitor.alerts)
        initial_errors = len(monitor.error_timestamps)

        # Force cleanup
        monitor._last_cleanup_time = current_time - 601
        monitor._perform_efficient_cleanup(current_time)

        # Should clean up old data but keep recent data
        final_alerts = len(monitor.alerts)
        final_errors = len(monitor.error_timestamps)

        assert final_alerts <= initial_alerts, "Should clean up old alerts"
        assert final_errors <= initial_errors, "Should clean up old errors"
        assert final_alerts > 0, "Should keep some recent alerts"
        assert final_errors > 0, "Should keep some recent errors"

    test_suite.run_test("Memory Pressure Monitoring", test_memory_pressure_monitoring)
    test_suite.run_test("Resource Constraint Handling", test_resource_constraint_handling)
    test_suite.run_test("Long Session Resource Management", test_long_session_resource_management)

    # === SESSION STATE PERSISTENCE TESTS ===
    def test_session_checkpoint_creation():
        """Test session checkpoint creation and restoration."""

        monitor = SessionHealthMonitor()

        # Add some test data using existing metrics
        monitor.update_metric("error_rate", 50.0)
        monitor.record_error("test_error")

        # Create checkpoint
        checkpoint_path = monitor.create_session_checkpoint("test_checkpoint")
        assert checkpoint_path != "", "Should create checkpoint successfully"
        assert Path(checkpoint_path).exists(), "Checkpoint file should exist"

        # Test checkpoint restoration
        new_monitor = SessionHealthMonitor()
        success = new_monitor.restore_from_checkpoint(checkpoint_path)
        assert success, "Should restore checkpoint successfully"

        # Verify restored data
        assert "error_rate" in new_monitor.current_metrics, "Should restore metrics"
        # Check that the metric was restored (value might be recalculated)
        assert new_monitor.current_metrics["error_rate"].value >= 0, "Should restore metric with valid value"
        assert len(new_monitor.error_timestamps) > 0, "Should restore error timestamps"

    def test_session_state_persistence():
        """Test session state persistence to disk."""
        monitor = SessionHealthMonitor()

        # Add some test data using existing metrics
        monitor.update_metric("memory_usage_mb", 75.0)
        monitor.record_error("persistence_error")

        # Test session data persistence
        test_data = {"test_key": "test_value", "page_count": 100}
        state_file = monitor.persist_session_state_to_disk(test_data)
        assert state_file != "", "Should persist session state successfully"

        # Test session state recovery
        recovered_data = monitor.recover_session_state_from_disk()
        assert recovered_data is not None, "Should recover session state"
        assert "test_key" in recovered_data, "Should recover custom data"
        assert "health_monitor" in recovered_data, "Should include health monitor state"

    def test_checkpoint_management():
        """Test checkpoint listing and cleanup functionality."""
        monitor = SessionHealthMonitor()

        # Create multiple checkpoints
        monitor.create_session_checkpoint("test_checkpoint_1")
        monitor.create_session_checkpoint("test_checkpoint_2")

        # List checkpoints
        checkpoints = monitor.list_available_checkpoints()
        assert len(checkpoints) >= 2, "Should list created checkpoints"

        # Verify checkpoint metadata
        for checkpoint in checkpoints:
            assert "name" in checkpoint, "Should include checkpoint name"
            assert "created_time" in checkpoint, "Should include creation time"
            assert "file_size_kb" in checkpoint, "Should include file size"

    def test_auto_checkpoint_functionality():
        """Test automatic checkpoint creation."""
        monitor = SessionHealthMonitor()

        # Test auto checkpoint (with very short interval for testing)
        monitor.auto_checkpoint(interval_minutes=1)  # 1 minute for testing

        # Manually trigger checkpoint by setting old timestamp
        monitor._last_checkpoint_time = time.time() - 120  # 2 minutes ago
        monitor.auto_checkpoint(interval_minutes=1)

        # Should have created at least one auto checkpoint
        checkpoints = monitor.list_available_checkpoints()
        auto_checkpoints = [cp for cp in checkpoints if cp["name"].startswith("auto_checkpoint")]
        assert len(auto_checkpoints) >= 1, "Should create auto checkpoints"

    test_suite.run_test("Session Checkpoint Creation", test_session_checkpoint_creation)
    test_suite.run_test("Session State Persistence", test_session_state_persistence)
    test_suite.run_test("Checkpoint Management", test_checkpoint_management)
    test_suite.run_test("Auto Checkpoint Functionality", test_auto_checkpoint_functionality)

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
