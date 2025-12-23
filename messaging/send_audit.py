#!/usr/bin/env python3
"""Send operation audit trail for message orchestrator.

Phase 6.2: Provides comprehensive audit logging for all send operations,
including decision path, safety check results, and content source.

This module creates a JSON-based audit log that can be queried by:
- person_id: Find all sends for a specific person
- trigger_type: Filter by AUTOMATED/REPLY/OPT_OUT/APPROVED
- date_range: Filter by timestamp

Usage:
    from messaging.send_audit import log_send_decision, query_audit_log

    # Log a send decision
    log_send_decision(
        person_id=123,
        trigger_type="AUTOMATED_SEQUENCE",
        decision="send",
        safety_checks={"opt_out": "pass", "duplicate": "pass"},
        content_source="template",
        result={"success": True, "api_status": 200}
    )

    # Query the audit log
    entries = query_audit_log(person_id=123, trigger_type="AUTOMATED_SEQUENCE")
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from testing.test_framework import TestSuite

logger = logging.getLogger(__name__)

# Default audit log location
DEFAULT_AUDIT_LOG_PATH = Path("Logs/send_audit.jsonl")

# Module-level lock for thread-safe writes
_audit_lock = threading.Lock()

# Configurable log path (set via configure_audit_log)
_audit_log_path: Path = DEFAULT_AUDIT_LOG_PATH


def configure_audit_log(log_path: Optional[Path] = None) -> None:
    """Configure the audit log file path.

    Args:
        log_path: Path to the audit log file. Defaults to Logs/send_audit.jsonl
    """
    global _audit_log_path  # noqa: PLW0603
    _audit_log_path = log_path or DEFAULT_AUDIT_LOG_PATH

    # Ensure directory exists
    _audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Audit log configured: %s", _audit_log_path)


# --------------------------------------------------------------------------
# Data Classes
# --------------------------------------------------------------------------


@dataclass
class SafetyCheckResult:
    """Result of a single safety check."""

    check_name: str
    passed: bool
    reason: Optional[str] = None
    details: Optional[dict[str, Any]] = None


@dataclass
class SendAuditEntry:
    """Complete audit record for a send operation."""

    # Identifiers
    timestamp: str  # ISO 8601 format
    person_id: Optional[int] = None
    profile_id: Optional[str] = None
    conversation_id: Optional[int] = None

    # Trigger and decision
    trigger_type: str = ""  # AUTOMATED_SEQUENCE, REPLY_RECEIVED, OPT_OUT, HUMAN_APPROVED
    decision: str = ""  # send, block, skip, desist
    decision_reason: Optional[str] = None

    # Safety checks
    safety_checks: list[dict[str, Any]] = field(default_factory=list)
    all_checks_passed: bool = False

    # Content generation
    content_source: str = ""  # template, ai_generated, approved_draft, desist
    content_template_key: Optional[str] = None
    content_generation_time_ms: Optional[float] = None

    # API result
    api_called: bool = False
    api_success: Optional[bool] = None
    api_status_code: Optional[int] = None
    api_error: Optional[str] = None

    # Database updates
    database_updated: bool = False
    database_fields_updated: list[str] = field(default_factory=list)

    # Context
    action_source: str = ""  # action8, action9, action11, orchestrator
    feature_flags: dict[str, bool] = field(default_factory=dict)
    additional_context: dict[str, Any] = field(default_factory=dict)


def create_audit_entry(
    person_id: Optional[int] = None,
    profile_id: Optional[str] = None,
    conversation_id: Optional[int] = None,
    trigger_type: str = "",
    action_source: str = "",
) -> SendAuditEntry:
    """Create a new audit entry with timestamp.

    Args:
        person_id: Database person ID
        profile_id: Ancestry profile ID
        conversation_id: Conversation ID if applicable
        trigger_type: The trigger type for this send
        action_source: Which action initiated this (action8, action9, action11)

    Returns:
        A new SendAuditEntry with timestamp set
    """
    return SendAuditEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        person_id=person_id,
        profile_id=profile_id,
        conversation_id=conversation_id,
        trigger_type=trigger_type,
        action_source=action_source,
    )


def add_safety_check(
    entry: SendAuditEntry,
    check_name: str,
    passed: bool,
    reason: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
) -> None:
    """Add a safety check result to the audit entry.

    Args:
        entry: The audit entry to update
        check_name: Name of the safety check (opt_out, duplicate, policy, etc.)
        passed: Whether the check passed
        reason: Optional reason for the result
        details: Optional additional details
    """
    check_result = SafetyCheckResult(
        check_name=check_name,
        passed=passed,
        reason=reason,
        details=details,
    )
    entry.safety_checks.append(asdict(check_result))


def finalize_safety_checks(entry: SendAuditEntry) -> None:
    """Mark whether all safety checks passed.

    Args:
        entry: The audit entry to finalize
    """
    entry.all_checks_passed = all(check.get("passed", False) for check in entry.safety_checks)


def set_decision(
    entry: SendAuditEntry,
    decision: str,
    reason: Optional[str] = None,
) -> None:
    """Set the decision for this send operation.

    Args:
        entry: The audit entry to update
        decision: The decision made (send, block, skip, desist)
        reason: Optional reason for the decision
    """
    entry.decision = decision
    entry.decision_reason = reason


def set_content_info(
    entry: SendAuditEntry,
    source: str,
    template_key: Optional[str] = None,
    generation_time_ms: Optional[float] = None,
) -> None:
    """Set content generation information.

    Args:
        entry: The audit entry to update
        source: Content source (template, ai_generated, approved_draft, desist)
        template_key: Template key if using template
        generation_time_ms: Time taken to generate content in milliseconds
    """
    entry.content_source = source
    entry.content_template_key = template_key
    entry.content_generation_time_ms = generation_time_ms


def set_api_result(
    entry: SendAuditEntry,
    called: bool,
    success: Optional[bool] = None,
    status_code: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    """Set API call result information.

    Args:
        entry: The audit entry to update
        called: Whether the API was called
        success: Whether the API call succeeded
        status_code: HTTP status code
        error: Error message if failed
    """
    entry.api_called = called
    entry.api_success = success
    entry.api_status_code = status_code
    entry.api_error = error


def set_database_update(
    entry: SendAuditEntry,
    updated: bool,
    fields: Optional[list[str]] = None,
) -> None:
    """Set database update information.

    Args:
        entry: The audit entry to update
        updated: Whether the database was updated
        fields: List of fields that were updated
    """
    entry.database_updated = updated
    entry.database_fields_updated = fields or []


def set_feature_flags(entry: SendAuditEntry, flags: dict[str, bool]) -> None:
    """Set feature flag state at time of operation.

    Args:
        entry: The audit entry to update
        flags: Dictionary of feature flags and their values
    """
    entry.feature_flags = flags


def add_context(entry: SendAuditEntry, key: str, value: Any) -> None:
    """Add additional context to the audit entry.

    Args:
        entry: The audit entry to update
        key: Context key
        value: Context value (must be JSON serializable)
    """
    entry.additional_context[key] = value


# --------------------------------------------------------------------------
# Audit Log Operations
# --------------------------------------------------------------------------


def write_audit_entry(entry: SendAuditEntry) -> bool:
    """Write an audit entry to the log file.

    Args:
        entry: The complete audit entry to write

    Returns:
        True if successfully written, False otherwise
    """
    try:
        # Ensure directory exists
        _audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        entry_dict = asdict(entry)
        line = json.dumps(entry_dict, default=str) + "\n"

        with _audit_lock, _audit_log_path.open("a", encoding="utf-8") as f:
            f.write(line)

        logger.debug("Audit entry written: person_id=%s, decision=%s", entry.person_id, entry.decision)
        return True

    except Exception as e:
        logger.warning("Failed to write audit entry: %s", e)
        return False


def log_send_decision(
    person_id: Optional[int] = None,
    trigger_type: str = "",
    decision: str = "",
    safety_checks: Optional[dict[str, str]] = None,
    content_source: str = "",
    result: Optional[dict[str, Any]] = None,
    action_source: str = "",
) -> bool:
    """Convenience function to log a complete send decision.

    Args:
        person_id: Database person ID
        trigger_type: Trigger type (AUTOMATED_SEQUENCE, REPLY_RECEIVED, etc.)
        decision: Decision made (send, block, skip, desist)
        safety_checks: Dict of check_name -> "pass" or "fail"
        content_source: Content source (template, ai_generated, etc.)
        result: Dict with api_success, api_status, error, etc.
        action_source: Which action initiated this

    Returns:
        True if successfully logged
    """
    entry = create_audit_entry(
        person_id=person_id,
        trigger_type=trigger_type,
        action_source=action_source,
    )

    set_decision(entry, decision)
    set_content_info(entry, content_source)

    # Add safety checks
    if safety_checks:
        for check_name, check_result in safety_checks.items():
            add_safety_check(entry, check_name, passed=(check_result.lower() == "pass"))
        finalize_safety_checks(entry)

    # Add API result
    if result:
        set_api_result(
            entry,
            called=result.get("api_called", True),
            success=result.get("success") or result.get("api_success"),
            status_code=result.get("api_status") or result.get("status_code"),
            error=result.get("error"),
        )
        if result.get("database_updated"):
            set_database_update(entry, updated=True, fields=result.get("fields_updated"))

    return write_audit_entry(entry)


# --------------------------------------------------------------------------
# Query Functions
# --------------------------------------------------------------------------


def query_audit_log(
    person_id: Optional[int] = None,
    trigger_type: Optional[str] = None,
    decision: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
) -> list[SendAuditEntry]:
    """Query the audit log with filters.

    Args:
        person_id: Filter by person ID
        trigger_type: Filter by trigger type
        decision: Filter by decision
        start_date: Filter entries after this date
        end_date: Filter entries before this date
        limit: Maximum number of entries to return

    Returns:
        List of matching SendAuditEntry objects
    """
    if not _audit_log_path.exists():
        return []

    results: list[SendAuditEntry] = []

    try:
        with _audit_log_path.open("r", encoding="utf-8") as f:
            for line in f:
                entry = _parse_and_filter_line(line, person_id, trigger_type, decision, start_date, end_date)
                if entry is not None:
                    results.append(entry)
                    if len(results) >= limit:
                        break

    except Exception as e:
        logger.warning("Failed to query audit log: %s", e)

    return results


def _parse_and_filter_line(
    line: str,
    person_id: Optional[int],
    trigger_type: Optional[str],
    decision: Optional[str],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
) -> Optional[SendAuditEntry]:
    """Parse a log line and apply filters.

    Returns SendAuditEntry if line matches filters, None otherwise.
    """
    if not line.strip():
        return None

    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None

    # Apply all filters - return None if any filter fails
    if not _matches_filters(data, person_id, trigger_type, decision, start_date, end_date):
        return None

    # Create entry from dict
    return SendAuditEntry(
        timestamp=data.get("timestamp", ""),
        person_id=data.get("person_id"),
        profile_id=data.get("profile_id"),
        conversation_id=data.get("conversation_id"),
        trigger_type=data.get("trigger_type", ""),
        decision=data.get("decision", ""),
        decision_reason=data.get("decision_reason"),
        safety_checks=data.get("safety_checks", []),
        all_checks_passed=data.get("all_checks_passed", False),
        content_source=data.get("content_source", ""),
        content_template_key=data.get("content_template_key"),
        content_generation_time_ms=data.get("content_generation_time_ms"),
        api_called=data.get("api_called", False),
        api_success=data.get("api_success"),
        api_status_code=data.get("api_status_code"),
        api_error=data.get("api_error"),
        database_updated=data.get("database_updated", False),
        database_fields_updated=data.get("database_fields_updated", []),
        action_source=data.get("action_source", ""),
        feature_flags=data.get("feature_flags", {}),
        additional_context=data.get("additional_context", {}),
    )


def _matches_filters(
    data: dict[str, Any],
    person_id: Optional[int],
    trigger_type: Optional[str],
    decision: Optional[str],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
) -> bool:
    """Check if data matches all filters."""
    if person_id is not None and data.get("person_id") != person_id:
        return False
    if trigger_type is not None and data.get("trigger_type") != trigger_type:
        return False
    if decision is not None and data.get("decision") != decision:
        return False
    return _passes_date_filter(data.get("timestamp", ""), start_date, end_date)


def _passes_date_filter(
    timestamp_str: str,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
) -> bool:
    """Check if timestamp passes date filters."""
    if not timestamp_str:
        return True
    if start_date is None and end_date is None:
        return True

    try:
        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        if start_date and ts < start_date:
            return False
        if end_date and ts > end_date:
            return False
    except ValueError:
        pass  # Invalid timestamp, allow through

    return True


def get_audit_stats(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> dict[str, Any]:
    """Get aggregate statistics from the audit log.

    Args:
        start_date: Filter entries after this date
        end_date: Filter entries before this date

    Returns:
        Dict with counts by trigger type, decision, etc.
    """
    entries = query_audit_log(start_date=start_date, end_date=end_date, limit=10000)

    stats: dict[str, Any] = {
        "total_entries": len(entries),
        "by_trigger_type": {},
        "by_decision": {},
        "by_content_source": {},
        "api_success_rate": 0.0,
        "safety_check_block_rate": 0.0,
    }

    if not entries:
        return stats

    return _compute_stats(entries, stats)


def _compute_stats(entries: list[SendAuditEntry], stats: dict[str, Any]) -> dict[str, Any]:
    """Compute statistics from entries."""
    api_calls = 0
    api_successes = 0
    blocked_count = 0

    for entry in entries:
        _update_category_counts(entry, stats)
        api_calls, api_successes = _update_api_counts(entry, api_calls, api_successes)
        blocked_count = _update_block_count(entry, blocked_count)

    if api_calls > 0:
        stats["api_success_rate"] = round(100.0 * api_successes / api_calls, 1)

    if len(entries) > 0:
        stats["safety_check_block_rate"] = round(100.0 * blocked_count / len(entries), 1)

    return stats


def _update_category_counts(entry: SendAuditEntry, stats: dict[str, Any]) -> None:
    """Update trigger type, decision, and content source counts."""
    tt = entry.trigger_type or "unknown"
    stats["by_trigger_type"][tt] = stats["by_trigger_type"].get(tt, 0) + 1

    dec = entry.decision or "unknown"
    stats["by_decision"][dec] = stats["by_decision"].get(dec, 0) + 1

    cs = entry.content_source or "unknown"
    stats["by_content_source"][cs] = stats["by_content_source"].get(cs, 0) + 1


def _update_api_counts(entry: SendAuditEntry, api_calls: int, api_successes: int) -> tuple[int, int]:
    """Update API call and success counts."""
    if entry.api_called:
        api_calls += 1
        if entry.api_success:
            api_successes += 1
    return api_calls, api_successes


def _update_block_count(entry: SendAuditEntry, blocked_count: int) -> int:
    """Update safety check block count."""
    if entry.decision == "block" and not entry.all_checks_passed:
        blocked_count += 1
    return blocked_count


def clear_audit_log() -> bool:
    """Clear the audit log (for testing purposes).

    Returns:
        True if cleared successfully
    """
    try:
        if _audit_log_path.exists():
            _audit_log_path.unlink()
        return True
    except Exception as e:
        logger.warning("Failed to clear audit log: %s", e)
        return False


# --------------------------------------------------------------------------
# Module Tests
# --------------------------------------------------------------------------


def _run_module_tests() -> bool:
    """Run send_audit module tests."""
    suite = TestSuite("Send Audit Trail", "messaging/send_audit.py")
    suite.start_suite()

    # Use temp path for tests
    test_log_path = Path("Logs/test_send_audit.jsonl")

    def setup_test_log() -> None:
        configure_audit_log(test_log_path)
        clear_audit_log()

    # Test 1: Create audit entry
    def test_create_entry() -> None:
        setup_test_log()
        entry = create_audit_entry(
            person_id=123,
            profile_id="ABC123",
            trigger_type="AUTOMATED_SEQUENCE",
            action_source="action8",
        )
        assert entry.person_id == 123
        assert entry.profile_id == "ABC123"
        assert entry.trigger_type == "AUTOMATED_SEQUENCE"
        assert entry.timestamp  # Should have timestamp

    suite.run_test("create_audit_entry", test_create_entry)

    # Test 2: Add safety checks
    def test_add_safety_checks() -> None:
        setup_test_log()
        entry = create_audit_entry(person_id=1)
        add_safety_check(entry, "opt_out", passed=True)
        add_safety_check(entry, "duplicate", passed=False, reason="Message sent within 7 days")
        finalize_safety_checks(entry)

        assert len(entry.safety_checks) == 2
        assert not entry.all_checks_passed  # One check failed

    suite.run_test("add_safety_checks", test_add_safety_checks)

    # Test 3: Write and read entry
    def test_write_and_read() -> None:
        setup_test_log()
        entry = create_audit_entry(person_id=456, trigger_type="REPLY_RECEIVED")
        set_decision(entry, "send", "All checks passed")
        set_content_info(entry, "ai_generated", generation_time_ms=150.5)
        set_api_result(entry, called=True, success=True, status_code=200)

        assert write_audit_entry(entry)

        # Query it back
        results = query_audit_log(person_id=456)
        assert len(results) >= 1
        found = results[0]
        assert found.person_id == 456
        assert found.decision == "send"
        assert found.content_source == "ai_generated"

    suite.run_test("write_and_read_entry", test_write_and_read)

    # Test 4: Convenience function
    def test_log_send_decision() -> None:
        setup_test_log()
        clear_audit_log()

        result = log_send_decision(
            person_id=789,
            trigger_type="HUMAN_APPROVED",
            decision="send",
            safety_checks={"opt_out": "pass", "duplicate": "pass"},
            content_source="approved_draft",
            result={"success": True, "api_status": 200},
            action_source="action11",
        )
        assert result

        entries = query_audit_log(person_id=789)
        assert len(entries) >= 1
        assert entries[0].trigger_type == "HUMAN_APPROVED"

    suite.run_test("log_send_decision convenience function", test_log_send_decision)

    # Test 5: Query filters
    def test_query_filters() -> None:
        setup_test_log()
        clear_audit_log()

        # Write multiple entries
        log_send_decision(person_id=100, trigger_type="AUTOMATED_SEQUENCE", decision="send")
        log_send_decision(person_id=101, trigger_type="REPLY_RECEIVED", decision="block")
        log_send_decision(person_id=100, trigger_type="OPT_OUT", decision="send")

        # Filter by person_id
        results = query_audit_log(person_id=100)
        assert len(results) == 2

        # Filter by trigger_type
        results = query_audit_log(trigger_type="REPLY_RECEIVED")
        assert len(results) == 1

        # Filter by decision
        results = query_audit_log(decision="block")
        assert len(results) == 1

    suite.run_test("query_filters", test_query_filters)

    # Test 6: Get stats
    def test_get_stats() -> None:
        setup_test_log()
        clear_audit_log()

        log_send_decision(person_id=1, trigger_type="AUTOMATED_SEQUENCE", decision="send")
        log_send_decision(person_id=2, trigger_type="AUTOMATED_SEQUENCE", decision="send")
        log_send_decision(person_id=3, trigger_type="REPLY_RECEIVED", decision="block")

        stats = get_audit_stats()
        assert stats["total_entries"] == 3
        assert stats["by_trigger_type"].get("AUTOMATED_SEQUENCE") == 2
        assert stats["by_decision"].get("send") == 2

    suite.run_test("get_audit_stats", test_get_stats)

    # Test 7: Feature flags and context
    def test_feature_flags_and_context() -> None:
        setup_test_log()
        entry = create_audit_entry(person_id=999)
        set_feature_flags(entry, {"ENABLE_UNIFIED_SEND_ORCHESTRATOR": True, "shadow_mode": False})
        add_context(entry, "batch_id", "batch-123")
        add_context(entry, "retry_count", 2)

        assert entry.feature_flags["ENABLE_UNIFIED_SEND_ORCHESTRATOR"] is True
        assert entry.additional_context["batch_id"] == "batch-123"

    suite.run_test("feature_flags_and_context", test_feature_flags_and_context)

    # Test 8: Database update tracking
    def test_database_update() -> None:
        setup_test_log()
        entry = create_audit_entry(person_id=888)
        set_database_update(entry, updated=True, fields=["last_contacted_at", "contact_count"])

        assert entry.database_updated is True
        assert "last_contacted_at" in entry.database_fields_updated
        assert len(entry.database_fields_updated) == 2

    suite.run_test("database_update_tracking", test_database_update)

    # Cleanup
    clear_audit_log()
    if test_log_path.exists():
        test_log_path.unlink()

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run all send_audit tests."""
    return _run_module_tests()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
