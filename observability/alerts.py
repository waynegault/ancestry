#!/usr/bin/env python3

"""
Alerting Rules Module (Phase 9.3)

Provides configurable alerting for critical system conditions:
- Opt-out rate threshold exceeded
- Review queue depth too high
- Circuit breaker trips
- Emergency stop activation

Alerts are logged and can be forwarded to external systems.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Severity levels for alerts."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """Types of system alerts."""

    OPT_OUT_RATE_HIGH = "OPT_OUT_RATE_HIGH"
    QUEUE_DEPTH_HIGH = "QUEUE_DEPTH_HIGH"
    CIRCUIT_BREAKER_TRIPPED = "CIRCUIT_BREAKER_TRIPPED"
    EMERGENCY_STOP_ENABLED = "EMERGENCY_STOP_ENABLED"


@dataclass
class Alert:
    """Represents a system alert."""

    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


@dataclass
class AlertThresholds:
    """Configurable thresholds for alert conditions."""

    opt_out_rate_threshold: float = 5.0  # Percentage
    opt_out_window_hours: int = 24  # Time window for rate calculation
    queue_depth_threshold: int = 50  # Maximum pending drafts
    queue_age_threshold_hours: int = 24  # Maximum age for queue items


class AlertChecker:
    """
    Checks system conditions and generates alerts.

    This class runs periodic checks against configurable thresholds
    and logs/emits alerts when conditions are exceeded.
    """

    def __init__(self, thresholds: Optional[AlertThresholds] = None) -> None:
        """Initialize the alert checker with thresholds."""
        self._active_alerts: list[Alert] = []
        if thresholds is not None:
            # Use provided thresholds as-is
            self.thresholds = thresholds
        else:
            # Load from config or use defaults
            self.thresholds = AlertThresholds()
            self._load_thresholds_from_config()

    def _load_thresholds_from_config(self) -> None:
        """Load thresholds from config_schema if available."""
        try:
            from config import config_schema

            self.thresholds.opt_out_rate_threshold = getattr(
                config_schema, "opt_out_rate_threshold", self.thresholds.opt_out_rate_threshold
            )
        except Exception:
            pass  # Use defaults

    def check_all(self, session: Session) -> list[Alert]:
        """
        Run all alert checks and return any triggered alerts.

        Args:
            session: Database session

        Returns:
            List of triggered alerts
        """
        alerts: list[Alert] = []

        # Check opt-out rate
        opt_out_alert = self._check_opt_out_rate(session)
        if opt_out_alert:
            alerts.append(opt_out_alert)

        # Check queue depth
        queue_alert = self._check_queue_depth(session)
        if queue_alert:
            alerts.append(queue_alert)

        # Check circuit breaker
        circuit_alert = self._check_circuit_breaker()
        if circuit_alert:
            alerts.append(circuit_alert)

        # Check emergency stop
        stop_alert = self._check_emergency_stop()
        if stop_alert:
            alerts.append(stop_alert)

        # Log and store alerts
        for alert in alerts:
            self._emit_alert(alert)
            self._active_alerts.append(alert)

        return alerts

    def _check_opt_out_rate(self, session: Session) -> Optional[Alert]:
        """Check if opt-out rate exceeds threshold in the time window."""
        try:
            from core.database import ConversationLog, MessageDirectionEnum

            window_start = datetime.now(timezone.utc) - timedelta(hours=self.thresholds.opt_out_window_hours)

            # Count outbound messages in window
            total_outbound = (
                session.query(ConversationLog)
                .filter(
                    ConversationLog.direction == MessageDirectionEnum.OUT,
                    ConversationLog.sent_at >= window_start,
                )
                .count()
            )

            if total_outbound == 0:
                return None

            # Count opt-outs (messages with opt_out classification)
            from core.database import ConversationState, ConversationStatusEnum

            opt_out_count = (
                session.query(ConversationState)
                .filter(
                    ConversationState.status == ConversationStatusEnum.OPT_OUT,
                    ConversationState.updated_at >= window_start,
                )
                .count()
            )

            opt_out_rate = (opt_out_count / total_outbound) * 100

            if opt_out_rate > self.thresholds.opt_out_rate_threshold:
                return Alert(
                    alert_type=AlertType.OPT_OUT_RATE_HIGH,
                    severity=AlertSeverity.CRITICAL,
                    message=f"Opt-out rate {opt_out_rate:.1f}% exceeds threshold "
                    f"{self.thresholds.opt_out_rate_threshold}%",
                    details={
                        "opt_out_rate": opt_out_rate,
                        "threshold": self.thresholds.opt_out_rate_threshold,
                        "window_hours": self.thresholds.opt_out_window_hours,
                        "total_outbound": total_outbound,
                        "opt_out_count": opt_out_count,
                    },
                )
        except Exception as e:
            logger.warning(f"Could not check opt-out rate: {e}")

        return None

    def _check_queue_depth(self, session: Session) -> Optional[Alert]:
        """Check if review queue depth exceeds threshold."""
        try:
            from core.database import DraftReply

            # Count pending drafts
            pending_count = session.query(DraftReply).filter(DraftReply.status == "PENDING").count()

            if pending_count > self.thresholds.queue_depth_threshold:
                # Check age of oldest pending draft
                oldest = (
                    session.query(DraftReply)
                    .filter(DraftReply.status == "PENDING")
                    .order_by(DraftReply.created_at.asc())
                    .first()
                )

                oldest_age_hours = 0.0
                if oldest and oldest.created_at:
                    age = datetime.now(timezone.utc) - oldest.created_at.replace(tzinfo=timezone.utc)
                    oldest_age_hours = age.total_seconds() / 3600

                if oldest_age_hours > self.thresholds.queue_age_threshold_hours:
                    return Alert(
                        alert_type=AlertType.QUEUE_DEPTH_HIGH,
                        severity=AlertSeverity.WARNING,
                        message=f"Review queue has {pending_count} pending drafts, "
                        f"oldest is {oldest_age_hours:.1f} hours old",
                        details={
                            "pending_count": pending_count,
                            "threshold": self.thresholds.queue_depth_threshold,
                            "oldest_age_hours": oldest_age_hours,
                            "age_threshold_hours": self.thresholds.queue_age_threshold_hours,
                        },
                    )
        except Exception as e:
            logger.warning(f"Could not check queue depth: {e}")

        return None

    @staticmethod
    def _check_circuit_breaker() -> Optional[Alert]:
        """Check if any circuit breakers are tripped."""
        try:
            from core.circuit_breaker import circuit_breaker_registry

            if circuit_breaker_registry:
                for name, breaker in circuit_breaker_registry.items():
                    if hasattr(breaker, "is_open") and breaker.is_open():
                        return Alert(
                            alert_type=AlertType.CIRCUIT_BREAKER_TRIPPED,
                            severity=AlertSeverity.WARNING,
                            message=f"Circuit breaker '{name}' is OPEN",
                            details={"breaker_name": name, "state": "OPEN"},
                        )
        except Exception:
            pass  # Circuit breaker registry may not exist

        return None

    @staticmethod
    def _check_emergency_stop() -> Optional[Alert]:
        """Check if emergency stop is enabled."""
        try:
            from config import config_schema

            if getattr(config_schema, "emergency_stop_enabled", False):
                return Alert(
                    alert_type=AlertType.EMERGENCY_STOP_ENABLED,
                    severity=AlertSeverity.CRITICAL,
                    message="Emergency stop is ENABLED - all messaging is halted",
                    details={"emergency_stop_enabled": True},
                )
        except Exception:
            pass

        return None

    @staticmethod
    def _emit_alert(alert: Alert) -> None:
        """Log and emit an alert."""
        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical(f"🚨 ALERT [{alert.alert_type.value}]: {alert.message}")
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(f"⚠️ ALERT [{alert.alert_type.value}]: {alert.message}")
        else:
            logger.info(f"INFO ALERT [{alert.alert_type.value}]: {alert.message}")

        # Emit to metrics if available
        try:
            from observability.metrics_registry import metrics

            metrics().alerts_total.increment(alert_type=alert.alert_type.value, severity=alert.severity.value)
        except Exception:
            pass  # Metrics not critical

        # Forward CRITICAL alerts to notification system (Phase 6.2)
        if alert.severity == AlertSeverity.CRITICAL:
            try:
                from observability.notifications import get_notification_manager

                manager = get_notification_manager()
                manager.notify_critical_alert(
                    alert_type=alert.alert_type.value,
                    message=alert.message,
                    details=alert.details,
                )
            except Exception as e:
                logger.warning(f"Could not forward critical alert to notification system: {e}")

    def get_active_alerts(self) -> list[Alert]:
        """Get list of currently active (unresolved) alerts."""
        return [a for a in self._active_alerts if not a.resolved]

    def clear_resolved(self) -> int:
        """Clear resolved alerts from memory. Returns count cleared."""
        original_count = len(self._active_alerts)
        self._active_alerts = [a for a in self._active_alerts if not a.resolved]
        return original_count - len(self._active_alerts)


# === MODULE-LEVEL FUNCTIONS ===


def run_alert_checks(session: Session) -> list[Alert]:
    """
    Convenience function to run all alert checks.

    Args:
        session: Database session

    Returns:
        List of triggered alerts
    """
    checker = AlertChecker()
    return checker.check_all(session)


def get_alert_summary(session: Session) -> dict[str, Any]:
    """
    Get a summary of current alert status.

    Args:
        session: Database session

    Returns:
        Dictionary with alert summary
    """
    checker = AlertChecker()
    alerts = checker.check_all(session)

    return {
        "total_alerts": len(alerts),
        "critical_count": sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL),
        "warning_count": sum(1 for a in alerts if a.severity == AlertSeverity.WARNING),
        "alerts": [
            {
                "type": a.alert_type.value,
                "severity": a.severity.value,
                "message": a.message,
                "timestamp": a.timestamp.isoformat(),
            }
            for a in alerts
        ],
    }


# === TESTS ===


def module_tests() -> bool:
    """Run module tests."""
    from testing.test_framework import TestSuite

    suite = TestSuite("Alerting Rules", "observability/alerts.py")

    # Test: AlertSeverity enum values
    def test_alert_severity_enum() -> None:
        assert AlertSeverity.INFO.value == "INFO"
        assert AlertSeverity.WARNING.value == "WARNING"
        assert AlertSeverity.CRITICAL.value == "CRITICAL"

    suite.run_test(
        "AlertSeverity enum values",
        test_alert_severity_enum,
        test_summary="Verify AlertSeverity enum has expected values",
        functions_tested="AlertSeverity enum",
        method_description="Check INFO, WARNING, CRITICAL values",
    )

    # Test: AlertType enum values
    def test_alert_type_enum() -> None:
        assert AlertType.OPT_OUT_RATE_HIGH.value == "OPT_OUT_RATE_HIGH"
        assert AlertType.QUEUE_DEPTH_HIGH.value == "QUEUE_DEPTH_HIGH"
        assert AlertType.CIRCUIT_BREAKER_TRIPPED.value == "CIRCUIT_BREAKER_TRIPPED"
        assert AlertType.EMERGENCY_STOP_ENABLED.value == "EMERGENCY_STOP_ENABLED"

    suite.run_test(
        "AlertType enum values",
        test_alert_type_enum,
        test_summary="Verify AlertType enum has expected values",
        functions_tested="AlertType enum",
        method_description="Check all alert type values",
    )

    # Test: Alert dataclass creation
    def test_alert_dataclass() -> None:
        alert = Alert(
            alert_type=AlertType.OPT_OUT_RATE_HIGH,
            severity=AlertSeverity.CRITICAL,
            message="Test alert",
            details={"rate": 7.5},
        )
        assert alert.alert_type == AlertType.OPT_OUT_RATE_HIGH
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.message == "Test alert"
        assert alert.details["rate"] == 7.5
        assert alert.resolved is False
        assert alert.timestamp is not None

    suite.run_test(
        "Alert dataclass creation",
        test_alert_dataclass,
        test_summary="Verify Alert dataclass stores values correctly",
        functions_tested="Alert dataclass",
        method_description="Create alert and verify fields",
    )

    # Test: AlertThresholds defaults
    def test_alert_thresholds_defaults() -> None:
        thresholds = AlertThresholds()
        assert thresholds.opt_out_rate_threshold == 5.0
        assert thresholds.opt_out_window_hours == 24
        assert thresholds.queue_depth_threshold == 50
        assert thresholds.queue_age_threshold_hours == 24

    suite.run_test(
        "AlertThresholds defaults",
        test_alert_thresholds_defaults,
        test_summary="Verify AlertThresholds has correct defaults",
        functions_tested="AlertThresholds dataclass",
        method_description="Check default threshold values",
    )

    # Test: AlertChecker initialization
    def test_alert_checker_init() -> None:
        checker = AlertChecker()
        assert checker.thresholds is not None
        assert checker.thresholds.opt_out_rate_threshold == 5.0
        assert checker._active_alerts == []

    suite.run_test(
        "AlertChecker initialization",
        test_alert_checker_init,
        test_summary="Verify AlertChecker initializes correctly",
        functions_tested="AlertChecker.__init__",
        method_description="Check initialization with defaults",
    )

    # Test: AlertChecker with custom thresholds
    def test_alert_checker_custom_thresholds() -> None:
        custom = AlertThresholds(opt_out_rate_threshold=10.0, queue_depth_threshold=100)
        checker = AlertChecker(thresholds=custom)
        assert checker.thresholds.opt_out_rate_threshold == 10.0
        assert checker.thresholds.queue_depth_threshold == 100

    suite.run_test(
        "AlertChecker with custom thresholds",
        test_alert_checker_custom_thresholds,
        test_summary="Verify AlertChecker accepts custom thresholds",
        functions_tested="AlertChecker.__init__",
        method_description="Check initialization with custom values",
    )

    # Test: _check_emergency_stop when disabled
    def test_check_emergency_stop_disabled() -> None:
        checker = AlertChecker()
        # Emergency stop should be disabled by default
        alert = checker._check_emergency_stop()
        # May or may not return alert depending on config
        if alert is not None:
            assert alert.alert_type == AlertType.EMERGENCY_STOP_ENABLED

    suite.run_test(
        "_check_emergency_stop behavior",
        test_check_emergency_stop_disabled,
        test_summary="Verify _check_emergency_stop returns correct alert",
        functions_tested="AlertChecker._check_emergency_stop",
        method_description="Check emergency stop detection",
    )

    # Test: get_active_alerts
    def test_get_active_alerts() -> None:
        checker = AlertChecker()
        checker._active_alerts = [
            Alert(AlertType.OPT_OUT_RATE_HIGH, AlertSeverity.CRITICAL, "Test 1", resolved=False),
            Alert(AlertType.QUEUE_DEPTH_HIGH, AlertSeverity.WARNING, "Test 2", resolved=True),
        ]
        active = checker.get_active_alerts()
        assert len(active) == 1
        assert active[0].message == "Test 1"

    suite.run_test(
        "get_active_alerts filters resolved",
        test_get_active_alerts,
        test_summary="Verify get_active_alerts only returns unresolved alerts",
        functions_tested="AlertChecker.get_active_alerts",
        method_description="Check filtering of resolved alerts",
    )

    # Test: clear_resolved
    def test_clear_resolved() -> None:
        checker = AlertChecker()
        checker._active_alerts = [
            Alert(AlertType.OPT_OUT_RATE_HIGH, AlertSeverity.CRITICAL, "Test 1", resolved=False),
            Alert(AlertType.QUEUE_DEPTH_HIGH, AlertSeverity.WARNING, "Test 2", resolved=True),
            Alert(AlertType.CIRCUIT_BREAKER_TRIPPED, AlertSeverity.WARNING, "Test 3", resolved=True),
        ]
        cleared = checker.clear_resolved()
        assert cleared == 2
        assert len(checker._active_alerts) == 1

    suite.run_test(
        "clear_resolved removes resolved alerts",
        test_clear_resolved,
        test_summary="Verify clear_resolved removes only resolved alerts",
        functions_tested="AlertChecker.clear_resolved",
        method_description="Check removal count and remaining alerts",
    )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run all tests with proper framework setup."""
    from testing.test_framework import create_standard_test_runner

    runner = create_standard_test_runner(module_tests)
    return runner()


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
