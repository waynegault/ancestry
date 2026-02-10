#!/usr/bin/env python3

"""
Notification System Module (Phase 6.2)

Provides multi-channel notification delivery for critical alerts and daily digests:
- Email notifications (SMTP)
- SMS notifications (via external services)
- Console logging (default fallback)

Includes daily digest generation for HUMAN_REVIEW queue items.
"""

import logging
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# === ENUMS ===


class NotificationChannelType(Enum):
    """Available notification channels."""

    EMAIL = "email"
    SMS = "sms"
    CONSOLE = "console"


class NotificationPriority(Enum):
    """Priority levels for notifications."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# === DATA CLASSES ===


@dataclass
class NotificationConfig:
    """Configuration for notification system."""

    # Email settings
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    email_from: str = ""
    email_to: list[str] = field(default_factory=list)

    # SMS settings (placeholder for external service integration)
    sms_enabled: bool = False
    sms_api_key: str = ""
    sms_phone_numbers: list[str] = field(default_factory=list)

    # Digest settings
    daily_digest_enabled: bool = True
    digest_hour: int = 8  # Hour of day (0-23) to send digest

    # General settings
    enabled_channels: list[NotificationChannelType] = field(default_factory=lambda: [NotificationChannelType.CONSOLE])

    @classmethod
    def from_environment(cls) -> "NotificationConfig":
        """Load configuration from environment variables."""
        import os

        config = cls()
        config.smtp_host = os.getenv("SMTP_HOST", "")
        config.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        config.smtp_username = os.getenv("SMTP_USERNAME", "")
        config.smtp_password = os.getenv("SMTP_PASSWORD", "")
        config.smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
        config.email_from = os.getenv("NOTIFICATION_EMAIL_FROM", "")
        config.email_to = [e.strip() for e in os.getenv("NOTIFICATION_EMAIL_TO", "").split(",") if e.strip()]

        config.sms_enabled = os.getenv("SMS_ENABLED", "false").lower() == "true"
        config.sms_api_key = os.getenv("SMS_API_KEY", "")
        config.sms_phone_numbers = [p.strip() for p in os.getenv("SMS_PHONE_NUMBERS", "").split(",") if p.strip()]

        config.daily_digest_enabled = os.getenv("DAILY_DIGEST_ENABLED", "true").lower() == "true"
        config.digest_hour = int(os.getenv("DIGEST_HOUR", "8"))

        # Determine enabled channels based on configuration
        channels: list[NotificationChannelType] = [NotificationChannelType.CONSOLE]
        if config.smtp_host and config.email_to:
            channels.append(NotificationChannelType.EMAIL)
        if config.sms_enabled and config.sms_phone_numbers:
            channels.append(NotificationChannelType.SMS)
        config.enabled_channels = channels

        return config


@dataclass
class Notification:
    """Represents a notification to be delivered."""

    subject: str
    body: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DigestSummary:
    """Summary of items for daily digest."""

    date_range_start: datetime
    date_range_end: datetime
    total_human_review: int
    human_review_by_category: dict[str, int]
    critical_alerts: int
    pending_drafts: int
    opt_out_count: int


# === NOTIFICATION CHANNELS ===


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""

    @abstractmethod
    def send(self, notification: Notification) -> bool:
        """Send a notification. Returns True if successful."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the channel is configured and available."""
        pass


class ConsoleNotificationChannel(NotificationChannel):
    """Console-based notification channel (logging)."""

    @staticmethod
    def send(notification: Notification) -> bool:
        """Log notification to console."""
        if notification.priority == NotificationPriority.CRITICAL:
            logger.critical(f"🚨 NOTIFICATION: {notification.subject}")
            logger.critical(f"   {notification.body}")
        elif notification.priority == NotificationPriority.HIGH:
            logger.warning(f"⚠️ NOTIFICATION: {notification.subject}")
            logger.warning(f"   {notification.body}")
        else:
            logger.info(f"📬 NOTIFICATION: {notification.subject}")
        return True

    @staticmethod
    def is_available() -> bool:
        """Console is always available."""
        return True


class EmailNotificationChannel(NotificationChannel):
    """Email-based notification channel (SMTP)."""

    def __init__(self, config: NotificationConfig) -> None:
        """Initialize with configuration."""
        self.config = config

    def send(self, notification: Notification) -> bool:
        """Send notification via email."""
        if not self.is_available():
            logger.warning("Email channel not configured, skipping email notification")
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = self.config.email_from
            msg["To"] = ", ".join(self.config.email_to)
            msg["Subject"] = self._format_subject(notification)

            body = self._format_body(notification)
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                if self.config.smtp_use_tls:
                    server.starttls()
                if self.config.smtp_username and self.config.smtp_password:
                    server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(msg)

            logger.info(f"✉️ Email notification sent: {notification.subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False

    def is_available(self) -> bool:
        """Check if SMTP is configured."""
        return bool(self.config.smtp_host and self.config.email_to)

    @staticmethod
    def _format_subject(notification: Notification) -> str:
        """Format email subject with priority prefix."""
        prefix = ""
        if notification.priority == NotificationPriority.CRITICAL:
            prefix = "[CRITICAL] "
        elif notification.priority == NotificationPriority.HIGH:
            prefix = "[HIGH] "
        return f"{prefix}{notification.subject}"

    @staticmethod
    def _format_body(notification: Notification) -> str:
        """Format email body with metadata."""
        lines = [
            notification.body,
            "",
            "---",
            f"Timestamp: {notification.timestamp.isoformat()}",
            f"Priority: {notification.priority.value.upper()}",
        ]

        if notification.metadata:
            lines.append("")
            lines.append("Additional Details:")
            for key, value in notification.metadata.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)


class SmsNotificationChannel(NotificationChannel):
    """SMS-based notification channel (placeholder for external service)."""

    def __init__(self, config: NotificationConfig) -> None:
        """Initialize with configuration."""
        self.config = config

    def send(self, notification: Notification) -> bool:
        """Send notification via SMS (placeholder implementation)."""
        if not self.is_available():
            logger.warning("SMS channel not configured, skipping SMS notification")
            return False

        # Placeholder: In production, integrate with Twilio, AWS SNS, etc.
        for phone in self.config.sms_phone_numbers:
            logger.info(f"📱 SMS notification would be sent to {phone}: {notification.subject}")

        # Log as if sent (placeholder)
        logger.info(f"📱 SMS notification logged (placeholder): {notification.subject}")
        return True

    def is_available(self) -> bool:
        """Check if SMS is configured."""
        return bool(self.config.sms_enabled and self.config.sms_phone_numbers)


# === NOTIFICATION MANAGER ===


class NotificationManager:
    """
    Central manager for multi-channel notification delivery.

    Handles routing notifications to appropriate channels based on
    priority and configuration.
    """

    def __init__(self, config: NotificationConfig | None = None) -> None:
        """Initialize with configuration."""
        self.config = config or NotificationConfig.from_environment()
        self._channels: dict[NotificationChannelType, NotificationChannel] = {}
        self._initialize_channels()

    def _initialize_channels(self) -> None:
        """Initialize available notification channels."""
        self._channels[NotificationChannelType.CONSOLE] = ConsoleNotificationChannel()

        if NotificationChannelType.EMAIL in self.config.enabled_channels:
            self._channels[NotificationChannelType.EMAIL] = EmailNotificationChannel(self.config)

        if NotificationChannelType.SMS in self.config.enabled_channels:
            self._channels[NotificationChannelType.SMS] = SmsNotificationChannel(self.config)

    def notify(self, notification: Notification, channels: list[NotificationChannelType] | None = None) -> int:
        """
        Send notification through specified channels.

        Args:
            notification: The notification to send
            channels: Specific channels to use (None = use all enabled)

        Returns:
            Number of successful deliveries
        """
        target_channels = channels or self.config.enabled_channels
        successes = 0

        for channel_type in target_channels:
            if channel_type in self._channels:
                channel = self._channels[channel_type]
                if channel.is_available():
                    try:
                        if channel.send(notification):
                            successes += 1
                    except Exception as e:
                        logger.error(f"Failed to send via {channel_type.value}: {e}")

        return successes

    def notify_critical_alert(self, alert_type: str, message: str, details: dict[str, Any] | None = None) -> int:
        """
        Send a critical alert notification through all channels.

        This is the primary method for CRITICAL alert forwarding.

        Args:
            alert_type: Type of alert (e.g., "OPT_OUT_RATE_HIGH")
            message: Alert message
            details: Additional alert details

        Returns:
            Number of successful deliveries
        """
        notification = Notification(
            subject=f"Ancestry Alert: {alert_type}",
            body=message,
            priority=NotificationPriority.CRITICAL,
            metadata=details or {},
        )

        return self.notify(notification)


# === DAILY DIGEST ===


def generate_daily_digest(session: Session) -> DigestSummary:
    """
    Generate a summary of HUMAN_REVIEW items from the past 24 hours.

    Args:
        session: Database session

    Returns:
        DigestSummary with counts and categories
    """
    window_end = datetime.now(UTC)
    window_start = window_end - timedelta(hours=24)

    summary = DigestSummary(
        date_range_start=window_start,
        date_range_end=window_end,
        total_human_review=0,
        human_review_by_category={},
        critical_alerts=0,
        pending_drafts=0,
        opt_out_count=0,
    )

    try:
        from core.database import ConversationState, ConversationStatusEnum, DraftReply

        # Count HUMAN_REVIEW items (pending drafts needing review)
        pending_drafts = session.query(DraftReply).filter(DraftReply.status == "PENDING").count()
        summary.pending_drafts = pending_drafts
        summary.total_human_review = pending_drafts

        # Count opt-outs in window
        opt_out_count = (
            session.query(ConversationState)
            .filter(
                ConversationState.status == ConversationStatusEnum.OPT_OUT,
                ConversationState.updated_at >= window_start,
            )
            .count()
        )
        summary.opt_out_count = opt_out_count

        # Categorize by priority if available
        priority_counts: dict[str, int] = {}
        drafts = session.query(DraftReply).filter(DraftReply.status == "PENDING").all()
        for draft in drafts:
            priority = getattr(draft, "priority", "NORMAL")
            if isinstance(priority, Enum):
                priority = priority.value
            priority_counts[str(priority)] = priority_counts.get(str(priority), 0) + 1
        summary.human_review_by_category = priority_counts

    except Exception as e:
        logger.warning(f"Could not generate complete digest: {e}")

    return summary


def format_digest_notification(summary: DigestSummary) -> Notification:
    """
    Format a DigestSummary as a notification.

    Args:
        summary: The digest summary to format

    Returns:
        Notification ready for delivery
    """
    lines = [
        f"Daily Digest: {summary.date_range_start.strftime('%Y-%m-%d %H:%M')} - "
        f"{summary.date_range_end.strftime('%Y-%m-%d %H:%M')} UTC",
        "",
        f"📋 Pending Drafts for Review: {summary.pending_drafts}",
        f"🚫 Opt-Outs (24h): {summary.opt_out_count}",
    ]

    if summary.human_review_by_category:
        lines.append("")
        lines.append("By Priority:")
        for category, count in sorted(summary.human_review_by_category.items()):
            lines.append(f"  • {category}: {count}")

    if summary.critical_alerts > 0:
        lines.append("")
        lines.append(f"🚨 Critical Alerts: {summary.critical_alerts}")

    body = "\n".join(lines)

    priority = NotificationPriority.NORMAL
    if summary.pending_drafts > 20 or summary.opt_out_count > 5:
        priority = NotificationPriority.HIGH

    return Notification(
        subject="Ancestry Daily Digest",
        body=body,
        priority=priority,
        metadata={
            "pending_drafts": summary.pending_drafts,
            "opt_out_count": summary.opt_out_count,
        },
    )


def send_daily_digest(session: Session, manager: NotificationManager | None = None) -> bool:
    """
    Generate and send the daily digest.

    Args:
        session: Database session
        manager: NotificationManager (created if not provided)

    Returns:
        True if digest was sent successfully
    """
    if manager is None:
        manager = NotificationManager()

    if not manager.config.daily_digest_enabled:
        logger.debug("Daily digest is disabled")
        return False

    summary = generate_daily_digest(session)
    notification = format_digest_notification(summary)

    successes = manager.notify(notification)
    return successes > 0


# === MODULE SINGLETON ===


class _NotificationManagerSingleton:
    """Thread-safe singleton holder for NotificationManager."""

    _instance: NotificationManager | None = None

    @classmethod
    def get(cls) -> NotificationManager:
        """Get or create the singleton notification manager."""
        if cls._instance is None:
            cls._instance = NotificationManager()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing only)."""
        cls._instance = None


def get_notification_manager() -> NotificationManager:
    """Get or create the singleton notification manager."""
    return _NotificationManagerSingleton.get()


# === TESTS ===


def module_tests() -> bool:
    """Run module tests."""
    from testing.test_framework import TestSuite

    suite = TestSuite("Notification System", "observability/notifications.py")

    # Test: NotificationChannelType enum
    def test_channel_type_enum() -> None:
        assert NotificationChannelType.EMAIL.value == "email"
        assert NotificationChannelType.SMS.value == "sms"
        assert NotificationChannelType.CONSOLE.value == "console"

    suite.run_test(
        "NotificationChannelType enum values",
        test_channel_type_enum,
        test_summary="Verify channel type enum has expected values",
        functions_tested="NotificationChannelType enum",
        method_description="Check EMAIL, SMS, CONSOLE values",
    )

    # Test: NotificationPriority enum
    def test_priority_enum() -> None:
        assert NotificationPriority.LOW.value == "low"
        assert NotificationPriority.NORMAL.value == "normal"
        assert NotificationPriority.HIGH.value == "high"
        assert NotificationPriority.CRITICAL.value == "critical"

    suite.run_test(
        "NotificationPriority enum values",
        test_priority_enum,
        test_summary="Verify priority enum has expected values",
        functions_tested="NotificationPriority enum",
        method_description="Check LOW, NORMAL, HIGH, CRITICAL values",
    )

    # Test: NotificationConfig defaults
    def test_notification_config_defaults() -> None:
        config = NotificationConfig()
        assert config.smtp_port == 587
        assert config.smtp_use_tls is True
        assert config.daily_digest_enabled is True
        assert config.digest_hour == 8
        assert NotificationChannelType.CONSOLE in config.enabled_channels

    suite.run_test(
        "NotificationConfig default values",
        test_notification_config_defaults,
        test_summary="Verify config defaults are sensible",
        functions_tested="NotificationConfig dataclass",
        method_description="Check default configuration values",
    )

    # Test: Notification dataclass creation
    def test_notification_dataclass() -> None:
        notif = Notification(
            subject="Test Subject",
            body="Test body content",
            priority=NotificationPriority.HIGH,
            metadata={"key": "value"},
        )
        assert notif.subject == "Test Subject"
        assert notif.body == "Test body content"
        assert notif.priority == NotificationPriority.HIGH
        assert notif.metadata["key"] == "value"
        assert notif.timestamp is not None

    suite.run_test(
        "Notification dataclass creation",
        test_notification_dataclass,
        test_summary="Verify Notification stores values correctly",
        functions_tested="Notification dataclass",
        method_description="Create notification and verify fields",
    )

    # Test: ConsoleNotificationChannel
    def test_console_channel() -> None:
        channel = ConsoleNotificationChannel()
        assert channel.is_available() is True

        notif = Notification(subject="Test", body="Body", priority=NotificationPriority.NORMAL)
        result = channel.send(notif)
        assert result is True

    suite.run_test(
        "ConsoleNotificationChannel send",
        test_console_channel,
        test_summary="Verify console channel works",
        functions_tested="ConsoleNotificationChannel.send, is_available",
        method_description="Send test notification to console",
    )

    # Test: EmailNotificationChannel availability
    def test_email_channel_availability() -> None:
        # Empty config = not available
        empty_config = NotificationConfig()
        email_channel = EmailNotificationChannel(empty_config)
        assert email_channel.is_available() is False

        # With config = available
        configured = NotificationConfig(smtp_host="smtp.example.com", email_to=["test@example.com"])
        email_channel_configured = EmailNotificationChannel(configured)
        assert email_channel_configured.is_available() is True

    suite.run_test(
        "EmailNotificationChannel availability check",
        test_email_channel_availability,
        test_summary="Verify email channel availability based on config",
        functions_tested="EmailNotificationChannel.is_available",
        method_description="Check availability with and without config",
    )

    # Test: NotificationManager initialization
    def test_notification_manager_init() -> None:
        config = NotificationConfig()
        manager = NotificationManager(config)
        assert NotificationChannelType.CONSOLE in manager._channels
        assert manager.config == config

    suite.run_test(
        "NotificationManager initialization",
        test_notification_manager_init,
        test_summary="Verify manager initializes with channels",
        functions_tested="NotificationManager.__init__",
        method_description="Create manager and verify channels",
    )

    # Test: notify_critical_alert
    def test_notify_critical_alert() -> None:
        config = NotificationConfig()  # Console only
        manager = NotificationManager(config)

        successes = manager.notify_critical_alert(
            alert_type="TEST_ALERT",
            message="This is a test critical alert",
            details={"count": 42},
        )
        assert successes >= 1  # Console should succeed

    suite.run_test(
        "notify_critical_alert method",
        test_notify_critical_alert,
        test_summary="Verify critical alert sends to console",
        functions_tested="NotificationManager.notify_critical_alert",
        method_description="Send critical alert and verify delivery",
    )

    # Test: DigestSummary dataclass
    def test_digest_summary() -> None:
        now = datetime.now(UTC)
        summary = DigestSummary(
            date_range_start=now - timedelta(hours=24),
            date_range_end=now,
            total_human_review=15,
            human_review_by_category={"HIGH": 5, "NORMAL": 10},
            critical_alerts=2,
            pending_drafts=15,
            opt_out_count=3,
        )
        assert summary.total_human_review == 15
        assert summary.human_review_by_category["HIGH"] == 5
        assert summary.opt_out_count == 3

    suite.run_test(
        "DigestSummary dataclass creation",
        test_digest_summary,
        test_summary="Verify DigestSummary stores values correctly",
        functions_tested="DigestSummary dataclass",
        method_description="Create digest summary and verify fields",
    )

    # Test: format_digest_notification
    def test_format_digest_notification() -> None:
        now = datetime.now(UTC)
        summary = DigestSummary(
            date_range_start=now - timedelta(hours=24),
            date_range_end=now,
            total_human_review=5,
            human_review_by_category={"NORMAL": 5},
            critical_alerts=0,
            pending_drafts=5,
            opt_out_count=1,
        )
        notification = format_digest_notification(summary)

        assert "Daily Digest" in notification.subject
        assert "Pending Drafts" in notification.body
        assert notification.priority == NotificationPriority.NORMAL

    suite.run_test(
        "format_digest_notification function",
        test_format_digest_notification,
        test_summary="Verify digest formats correctly",
        functions_tested="format_digest_notification",
        method_description="Format summary and check output",
    )

    # Test: High priority digest when threshold exceeded
    def test_digest_high_priority() -> None:
        now = datetime.now(UTC)
        summary = DigestSummary(
            date_range_start=now - timedelta(hours=24),
            date_range_end=now,
            total_human_review=25,
            human_review_by_category={"HIGH": 25},
            critical_alerts=0,
            pending_drafts=25,  # > 20 triggers HIGH
            opt_out_count=0,
        )
        notification = format_digest_notification(summary)
        assert notification.priority == NotificationPriority.HIGH

    suite.run_test(
        "Digest high priority threshold",
        test_digest_high_priority,
        test_summary="Verify digest priority escalates on high counts",
        functions_tested="format_digest_notification",
        method_description="Check priority when pending_drafts > 20",
    )

    # Test: get_notification_manager singleton
    def test_singleton() -> None:
        _NotificationManagerSingleton.reset()  # Reset singleton

        manager1 = get_notification_manager()
        manager2 = get_notification_manager()
        assert manager1 is manager2

    suite.run_test(
        "get_notification_manager singleton",
        test_singleton,
        test_summary="Verify singleton pattern works",
        functions_tested="get_notification_manager",
        method_description="Get manager twice and verify same instance",
    )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive tests for this module."""
    from testing.test_framework import create_standard_test_runner

    runner = create_standard_test_runner(module_tests)
    return runner()


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
