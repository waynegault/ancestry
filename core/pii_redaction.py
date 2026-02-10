#!/usr/bin/env python3
"""
PII Redaction Utilities for Log Security

Provides filters and utilities to redact personally identifiable information (PII)
from log messages before they are written to files or displayed on console.

Features:
- Email address redaction
- Profile ID partial masking
- Display name redaction
- UUID partial masking
- Phone number redaction
- Configurable via environment variable (PII_REDACTION_ENABLED)

Usage:
    Add PIIRedactionFilter to logging handlers to automatically redact
    sensitive information from all log output.
"""


import logging
import os
import re
import sys
from re import Pattern

# Module logger - uses parent's configuration
logger = logging.getLogger(__name__)


# === PII Patterns ===
# Email pattern: matches standard email formats
EMAIL_PATTERN: Pattern[str] = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Profile ID pattern: matches Ancestry profile IDs (8+ alphanumeric chars)
# Example: "abc12345" -> "ab****45"
PROFILE_ID_PATTERN: Pattern[str] = re.compile(r"\b(profile[_\s]?id[:\s=]+)([A-Za-z0-9-]{8,})\b", re.IGNORECASE)

# UUID pattern: matches standard UUIDs
# Example: "12345678-1234-1234-1234-123456789012" -> "1234****-****-****-****-********9012"
UUID_PATTERN: Pattern[str] = re.compile(
    r"\b([0-9a-f]{8})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{12})\b",
    re.IGNORECASE,
)

# Phone number patterns (various formats)
# Matches: 555-123-4567, (555) 123-4567, +1-555-123-4567, etc.
PHONE_PATTERN: Pattern[str] = re.compile(r"(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}")

# Display name in quotes pattern: "John Doe" -> "[REDACTED_NAME]"
QUOTED_NAME_PATTERN: Pattern[str] = re.compile(r'(?:display[_\s]?name[:\s=]+)["\']([^"\']+)["\']', re.IGNORECASE)

# Name after common prefixes: matches "name: John Doe" or "name=John Doe"
# Captures the name after the prefix to allow replacement
NAME_PREFIX_PATTERN: Pattern[str] = re.compile(
    r"((?:first|last|display|user)[\s_]?name[:\s=]+)([A-Za-z][A-Za-z\s'-]*)",
    re.IGNORECASE,
)


def _mask_middle(value: str, visible_start: int = 2, visible_end: int = 2, mask_char: str = "*") -> str:
    """Mask the middle portion of a string, keeping start and end visible.

    Args:
        value: The string to mask.
        visible_start: Number of characters to keep visible at start.
        visible_end: Number of characters to keep visible at end.
        mask_char: Character to use for masking.

    Returns:
        Masked string with only start and end characters visible.
    """
    if len(value) <= visible_start + visible_end:
        return mask_char * len(value)
    return value[:visible_start] + mask_char * (len(value) - visible_start - visible_end) + value[-visible_end:]


def redact_email(text: str) -> str:
    """Redact email addresses from text.

    Args:
        text: Input text potentially containing email addresses.

    Returns:
        Text with email addresses replaced by [REDACTED_EMAIL].
    """
    return EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)


def redact_profile_id(text: str) -> str:
    """Partially mask profile IDs in text.

    Args:
        text: Input text potentially containing profile IDs.

    Returns:
        Text with profile IDs partially masked.
    """

    def mask_profile_id(match: re.Match[str]) -> str:
        prefix = match.group(1)
        profile_id = match.group(2)
        return prefix + _mask_middle(profile_id)

    return PROFILE_ID_PATTERN.sub(mask_profile_id, text)


def redact_uuid(text: str) -> str:
    """Partially mask UUIDs in text.

    Args:
        text: Input text potentially containing UUIDs.

    Returns:
        Text with UUIDs partially masked.
    """

    def mask_uuid(match: re.Match[str]) -> str:
        # Keep first 4 and last 4 characters, mask the rest
        parts = match.groups()
        # Full match is parts[0]-parts[1]-parts[2]-parts[3]-parts[4]
        return f"{parts[0][:4]}****-****-****-****-********{parts[4][-4:]}"

    return UUID_PATTERN.sub(mask_uuid, text)


def redact_phone(text: str) -> str:
    """Redact phone numbers from text.

    Args:
        text: Input text potentially containing phone numbers.

    Returns:
        Text with phone numbers replaced by [REDACTED_PHONE].
    """
    return PHONE_PATTERN.sub("[REDACTED_PHONE]", text)


def redact_display_name(text: str) -> str:
    """Redact display names from text.

    Args:
        text: Input text potentially containing display names.

    Returns:
        Text with display names redacted.
    """
    # Redact quoted names after display_name keyword
    text = QUOTED_NAME_PATTERN.sub(r'display_name="[REDACTED_NAME]"', text)

    # Redact names after common prefixes (group 1 is prefix, group 2 is name)
    def mask_name_after_prefix(match: re.Match[str]) -> str:
        prefix = match.group(1)
        return prefix + "[REDACTED_NAME]"

    return NAME_PREFIX_PATTERN.sub(mask_name_after_prefix, text)


def redact_pii(text: str) -> str:
    """Apply all PII redaction rules to text.

    Args:
        text: Input text potentially containing PII.

    Returns:
        Text with all PII redacted.
    """
    text = redact_email(text)
    text = redact_profile_id(text)
    text = redact_uuid(text)
    text = redact_phone(text)
    return redact_display_name(text)


class PIIRedactionFilter(logging.Filter):
    """
    Logging filter that redacts PII from log messages.

    This filter can be added to any logging handler to automatically
    redact personally identifiable information before it is written.

    Activation:
        - Set environment variable PII_REDACTION_ENABLED=true
        - Or pass enabled=True to constructor

    Example:
        handler = logging.StreamHandler()
        handler.addFilter(PIIRedactionFilter())
        logger.addHandler(handler)
    """

    def __init__(
        self,
        name: str = "",
        *,
        enabled: bool | None = None,
        redact_emails: bool = True,
        redact_profile_ids: bool = True,
        redact_uuids: bool = True,
        redact_phones: bool = True,
        redact_names: bool = True,
    ) -> None:
        """Initialize the PII redaction filter.

        Args:
            name: Filter name (passed to parent).
            enabled: Whether redaction is enabled. If None, reads from
                     PII_REDACTION_ENABLED environment variable.
            redact_emails: Whether to redact email addresses.
            redact_profile_ids: Whether to mask profile IDs.
            redact_uuids: Whether to mask UUIDs.
            redact_phones: Whether to redact phone numbers.
            redact_names: Whether to redact display names.
        """
        super().__init__(name)

        # Determine if redaction is enabled
        if enabled is None:
            env_value = os.getenv("PII_REDACTION_ENABLED", "false").lower()
            self.enabled = env_value in {"true", "1", "yes", "on"}
        else:
            self.enabled = enabled

        # Store configuration
        self.redact_emails = redact_emails
        self.redact_profile_ids = redact_profile_ids
        self.redact_uuids = redact_uuids
        self.redact_phones = redact_phones
        self.redact_names = redact_names

    @staticmethod
    def _format_message(record: logging.LogRecord) -> str:
        """Format the log message from record, handling args."""
        if isinstance(record.msg, str):
            message = record.msg
            if record.args:
                try:
                    message = record.msg % record.args
                    record.args = ()
                except (TypeError, ValueError):
                    pass
        else:
            message = str(record.msg)
        return message

    def _apply_redactions(self, message: str) -> str:
        """Apply configured redactions to message."""
        if self.redact_emails:
            message = redact_email(message)
        if self.redact_profile_ids:
            message = redact_profile_id(message)
        if self.redact_uuids:
            message = redact_uuid(message)
        if self.redact_phones:
            message = redact_phone(message)
        if self.redact_names:
            message = redact_display_name(message)
        return message

    def filter(self, record: logging.LogRecord) -> bool:
        """Apply PII redaction to log record if enabled."""
        if not self.enabled:
            return True
        try:
            message = self._format_message(record)
            record.msg = self._apply_redactions(message)
        except Exception:
            pass  # Never let redaction failure break logging
        return True


# === Module Tests ===
def _test_email_redaction() -> bool:
    """Test email address redaction."""
    test_cases = [
        ("User email: user@example.com logged in", "User email: [REDACTED_EMAIL] logged in"),
        ("Contact: john.doe+tag@company.org", "Contact: [REDACTED_EMAIL]"),
        ("No email here", "No email here"),
        ("Multiple: a@b.com and c@d.org", "Multiple: [REDACTED_EMAIL] and [REDACTED_EMAIL]"),
    ]

    for input_text, expected in test_cases:
        result = redact_email(input_text)
        if result != expected:
            logger.error(f"Email redaction failed: {input_text!r} -> {result!r} (expected {expected!r})")
            return False

    return True


def _test_profile_id_redaction() -> bool:
    """Test profile ID masking."""
    test_cases = [
        ("profile_id=abc12345xyz", "profile_id=ab*******yz"),  # 11 chars -> 2+7+2
        ("Profile ID: 1234567890", "Profile ID: 12******90"),  # 10 chars -> 2+6+2
        ("No profile here", "No profile here"),
    ]

    for input_text, expected in test_cases:
        result = redact_profile_id(input_text)
        if result != expected:
            logger.error(f"Profile ID redaction failed: {input_text!r} -> {result!r} (expected {expected!r})")
            return False

    return True


def _test_uuid_redaction() -> bool:
    """Test UUID masking."""
    test_cases = [
        (
            "UUID: 12345678-1234-1234-1234-123456789012",
            "UUID: 1234****-****-****-****-********9012",
        ),
        ("No UUID here", "No UUID here"),
    ]

    for input_text, expected in test_cases:
        result = redact_uuid(input_text)
        if result != expected:
            logger.error(f"UUID redaction failed: {input_text!r} -> {result!r} (expected {expected!r})")
            return False

    return True


def _test_phone_redaction() -> bool:
    """Test phone number redaction."""
    test_cases = [
        ("Call me: 555-123-4567", "Call me: [REDACTED_PHONE]"),
        ("Phone: (555) 123-4567", "Phone: [REDACTED_PHONE]"),
        ("No phone here", "No phone here"),
    ]

    for input_text, expected in test_cases:
        result = redact_phone(input_text)
        if result != expected:
            logger.error(f"Phone redaction failed: {input_text!r} -> {result!r} (expected {expected!r})")
            return False

    return True


def _test_display_name_redaction() -> bool:
    """Test display name redaction."""
    test_cases = [
        ('display_name="John Doe"', 'display_name="[REDACTED_NAME]"'),
        ("first_name: John", "first_name: [REDACTED_NAME]"),
        ("No names here", "No names here"),
    ]

    for input_text, expected in test_cases:
        result = redact_display_name(input_text)
        if result != expected:
            logger.error(f"Name redaction failed: {input_text!r} -> {result!r} (expected {expected!r})")
            return False

    return True


def _test_pii_filter() -> bool:
    """Test the logging filter."""
    # Create a test record
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="User user@example.com with profile_id=abc12345xyz",
        args=(),
        exc_info=None,
    )

    # Apply filter with redaction enabled
    pii_filter = PIIRedactionFilter(enabled=True)
    result = pii_filter.filter(record)

    if not result:
        logger.error("Filter should always return True")
        return False

    if "[REDACTED_EMAIL]" not in record.msg:
        logger.error(f"Email should be redacted: {record.msg}")
        return False

    if "ab*******yz" not in record.msg:
        logger.error(f"Profile ID should be masked: {record.msg}")
        return False

    return True


def _test_filter_disabled() -> bool:
    """Test that filter does nothing when disabled."""
    original = "User user@example.com logged in"
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg=original,
        args=(),
        exc_info=None,
    )

    pii_filter = PIIRedactionFilter(enabled=False)
    pii_filter.filter(record)

    if record.msg != original:
        logger.error(f"Message should be unchanged when disabled: {record.msg}")
        return False

    return True


def _test_mask_middle() -> bool:
    """Test the _mask_middle helper function."""
    test_cases = [
        (("12345678", 2, 2, "*"), "12****78"),
        (("ab", 2, 2, "*"), "**"),
        (("abcd", 1, 1, "#"), "a##d"),
    ]

    for (value, start, end, char), expected in test_cases:
        result = _mask_middle(value, start, end, char)
        if result != expected:
            logger.error(f"mask_middle failed: {value!r} -> {result!r} (expected {expected!r})")
            return False

    return True


def pii_redaction_module_tests() -> bool:
    """Run all PII redaction module tests."""
    from testing.test_framework import TestSuite

    suite = TestSuite("PII Redaction", "pii_redaction.py")
    suite.start_suite()

    suite.run_test(
        test_name="Email address redaction",
        test_func=lambda: _test_email_redaction() or None,
        test_summary="Verify email addresses are redacted from text",
        functions_tested="redact_email()",
        expected_outcome="Emails replaced with [REDACTED_EMAIL]",
    )

    suite.run_test(
        test_name="Profile ID masking",
        test_func=lambda: _test_profile_id_redaction() or None,
        test_summary="Verify profile IDs are partially masked",
        functions_tested="redact_profile_id()",
        expected_outcome="Profile IDs have middle characters masked",
    )

    suite.run_test(
        test_name="UUID masking",
        test_func=lambda: _test_uuid_redaction() or None,
        test_summary="Verify UUIDs are partially masked",
        functions_tested="redact_uuid()",
        expected_outcome="UUIDs have middle sections masked",
    )

    suite.run_test(
        test_name="Phone number redaction",
        test_func=lambda: _test_phone_redaction() or None,
        test_summary="Verify phone numbers are redacted",
        functions_tested="redact_phone()",
        expected_outcome="Phone numbers replaced with [REDACTED_PHONE]",
    )

    suite.run_test(
        test_name="Display name redaction",
        test_func=lambda: _test_display_name_redaction() or None,
        test_summary="Verify display names are redacted",
        functions_tested="redact_display_name()",
        expected_outcome="Display names replaced with [REDACTED_NAME]",
    )

    suite.run_test(
        test_name="PII logging filter (enabled)",
        test_func=lambda: _test_pii_filter() or None,
        test_summary="Verify PII filter redacts when enabled",
        functions_tested="PIIRedactionFilter.filter()",
        expected_outcome="Log messages have PII redacted",
    )

    suite.run_test(
        test_name="PII logging filter (disabled)",
        test_func=lambda: _test_filter_disabled() or None,
        test_summary="Verify PII filter does nothing when disabled",
        functions_tested="PIIRedactionFilter.filter()",
        expected_outcome="Log messages unchanged when filter disabled",
    )

    suite.run_test(
        test_name="Mask middle helper function",
        test_func=lambda: _test_mask_middle() or None,
        test_summary="Verify mask_middle utility function",
        functions_tested="_mask_middle()",
        expected_outcome="Middle characters correctly masked",
    )

    return suite.finish_suite()


run_comprehensive_tests = pii_redaction_module_tests

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
