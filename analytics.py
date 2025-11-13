#!/usr/bin/env python3

"""
Lightweight Monitoring & Analytics Intelligence Engine

Records per-action metrics to Logs/analytics.jsonl, allows actions to attach transient
extras (e.g., merged 10/11 branch: gedcom/api_fallback), and provides simple weekly
summary generation and reporting.

Design goals:
- Zero external dependencies
- No DB schema changes (safe to add now)
- Robust to errors (analytics never breaks the main flow)
"""

from __future__ import annotations

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional


# Module-level transient extras storage (cleared after use)
class _State:
    extras: dict[str, Any] | None = None


def set_transient_extras(extras: dict[str, Any]) -> None:
    """Attach transient extras for the current action; consumed by exec_actn()."""
    try:
        _State.extras = dict(extras) if extras is not None else None
    except Exception as e:
        logger.debug(f"analytics.set_transient_extras failed: {e}")
        _State.extras = None


def pop_transient_extras() -> dict[str, Any] | None:
    """Return and clear previously attached extras."""
    extras = _State.extras
    _State.extras = None
    return extras


def _get_analytics_path() -> Path:
    """Resolve analytics output path and ensure directory exists."""
    logs_dir = Path("Logs")
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Fallback to current directory if Logs cannot be created
        logs_dir = Path()
    return logs_dir / "analytics.jsonl"


def log_event(
    action_name: str,
    choice: str,
    success: bool,
    duration_sec: float,
    mem_used_mb: float | None = None,
    extras: dict[str, Any] | None = None,
) -> None:
    """Append a single analytics event as JSON to the analytics file.

    Errors are swallowed (logged at DEBUG) to avoid affecting main flow.
    """
    try:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action_name": action_name,
            "choice": choice,
            "success": bool(success),
            "duration_sec": float(duration_sec),
            "mem_used_mb": float(mem_used_mb) if mem_used_mb is not None else None,
            "extras": extras or {},
            "pid": os.getpid(),
        }
        out_path = _get_analytics_path()
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug(f"analytics.log_event skipped due to error: {e}")


def _parse_analytics_line(line: str) -> dict[str, Any] | None:
    """Parse a single analytics log line.

    Args:
        line: JSON line from analytics log

    Returns:
        Parsed object or None if invalid
    """
    line_s = line.strip()
    if not line_s:
        return None
    try:
        return json.loads(line_s)
    except Exception:
        return None


def _parse_timestamp(obj: dict[str, Any]) -> datetime | None:
    """Parse timestamp from analytics object.

    Args:
        obj: Analytics object with 'ts' field

    Returns:
        Parsed datetime or None if invalid
    """
    ts_str = obj.get("ts")
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None


def _initialize_action_entry() -> dict[str, Any]:
    """Initialize a new action summary entry.

    Returns:
        Dictionary with counters and totals initialized to zero
    """
    return {
        "runs": 0,
        "success": 0,
        "duration_total": 0.0,
        "mem_total": 0.0,
        "mem_samples": 0,
    }


def _update_action_entry(ent: dict[str, Any], obj: dict[str, Any]) -> None:
    """Update action entry with data from analytics object.

    Args:
        ent: Action entry dictionary to update
        obj: Analytics object with run data
    """
    ent["runs"] += 1
    if obj.get("success"):
        ent["success"] += 1
    dur = obj.get("duration_sec")
    if isinstance(dur, (int, float)):
        ent["duration_total"] += float(dur)
    mem = obj.get("mem_used_mb")
    if isinstance(mem, (int, float)):
        ent["mem_total"] += float(mem)
        ent["mem_samples"] += 1


def _finalize_action_entry(ent: dict[str, Any]) -> None:
    """Finalize action entry by computing averages and cleaning up.

    Args:
        ent: Action entry dictionary to finalize
    """
    runs = max(1, ent["runs"])  # avoid div by zero
    ent["success_rate"] = ent["success"] / runs
    ent["avg_duration_sec"] = ent["duration_total"] / runs
    if ent["mem_samples"] > 0:
        ent["avg_mem_mb"] = ent["mem_total"] / ent["mem_samples"]
    else:
        ent["avg_mem_mb"] = None
    # drop raw totals to keep clean
    for k in ("duration_total", "mem_total", "mem_samples"):
        ent.pop(k, None)


def generate_weekly_summary(days: int = 7) -> dict[str, Any]:
    """Compute a simple rollup over the last N days.

    Returns a dict keyed by action_name with counts, success_rate, average duration, memory.
    """
    summary: dict[str, Any] = {}
    try:
        path = _get_analytics_path()
        if not path.exists():
            return summary

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = _parse_analytics_line(line)
                if obj is None:
                    continue

                ts = _parse_timestamp(obj)
                if ts is None or ts < cutoff:
                    continue

                action = obj.get("action_name", "unknown")
                if action not in summary:
                    summary[action] = _initialize_action_entry()

                _update_action_entry(summary[action], obj)

        # finalize averages
        for _action, ent in summary.items():
            _finalize_action_entry(ent)

    except Exception as e:
        logger.debug(f"analytics.generate_weekly_summary failed: {e}")

    return summary


def print_weekly_summary(days: int = 7) -> None:
    """Print a human-friendly weekly summary to stdout."""
    try:
        summary = generate_weekly_summary(days)
        print(f"\n=== Weekly Action Summary (last {days} days) ===")
        if not summary:
            print("No analytics data yet.")
            return
        for action, ent in sorted(summary.items()):
            sr = ent.get("success_rate", 0.0)
            avg_dur = ent.get("avg_duration_sec") or 0.0
            avg_mem = ent.get("avg_mem_mb")
            print(
                f"- {action}: runs={ent.get('runs',0)}, success={sr*100:.1f}%, "
                f"avg_dur={avg_dur:.2f}s, avg_mem={(f'{avg_mem:.1f} MB' if isinstance(avg_mem,(int,float)) else 'n/a')}"
            )
    except Exception as e:
        logger.debug(f"analytics.print_weekly_summary failed: {e}")


# === MODULE-LEVEL TEST FUNCTIONS ===
# These test functions are extracted from the main test suite for better
# modularity, maintainability, and reduced complexity. Each function tests
# a specific aspect of the analytics functionality.


def _test_analytics_initialization() -> bool:
    """Test analytics module initialization and state management."""
    # Test transient extras state
    assert _State.extras is None, "Initial extras should be None"

    test_extras = {"test": "value", "number": 42}
    set_transient_extras(test_extras)
    assert _State.extras == test_extras, "Extras should be set correctly"

    retrieved = pop_transient_extras()
    assert retrieved == test_extras, "Retrieved extras should match"
    assert _State.extras is None, "Extras should be cleared after pop"

    return True


def _test_analytics_path_resolution() -> bool:
    """Test analytics path resolution and directory creation."""
    path = _get_analytics_path()
    assert isinstance(path, Path), "Should return Path object"
    assert path.name == "analytics.jsonl", "Should have correct filename"

    # Test that directory exists or is created
    logs_dir = path.parent
    assert logs_dir.exists() or logs_dir == Path(), "Logs directory should exist or be current dir"

    return True


def _test_event_logging() -> bool:
    """Test event logging functionality."""
    import os
    import tempfile

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Temporarily override the logs directory
        if hasattr(sys.modules[__name__], '_get_analytics_path'):
            # Mock the function to return our temp path
            def mock_get_path() -> Path:
                return Path(temp_dir) / "analytics.jsonl"

            # Store original function
            original_func = _get_analytics_path
            # Replace with mock
            globals()['_get_analytics_path'] = mock_get_path

            try:
                # Test logging an event
                log_event(
                    action_name="test_action",
                    choice="test_choice",
                    success=True,
                    duration_sec=1.5,
                    mem_used_mb=100.5,
                    extras={"test": "extra"}
                )

                # Verify file was created and contains data
                analytics_path = mock_get_path()
                assert analytics_path.exists(), "Analytics file should be created"

                # Read and parse the logged event
                with analytics_path.open('r', encoding='utf-8') as f:
                    line = f.readline().strip()
                    event = json.loads(line)

                # Verify event structure
                assert event["action_name"] == "test_action"
                assert event["choice"] == "test_choice"
                assert event["success"] is True
                assert event["duration_sec"] == 1.5
                assert event["mem_used_mb"] == 100.5
                assert event["extras"]["test"] == "extra"
                assert "ts" in event
                assert "pid" in event

                return True

            finally:
                # Restore original function
                globals()['_get_analytics_path'] = original_func

    return False


def _test_analytics_parsing() -> bool:
    """Test analytics line parsing and timestamp extraction."""
    # Test valid JSON parsing
    valid_line = '{"ts": "2023-01-01T12:00:00+00:00", "action_name": "test", "success": true}'
    parsed = _parse_analytics_line(valid_line)
    assert parsed is not None, "Should parse valid JSON"
    assert parsed["action_name"] == "test", "Should extract action name"

    # Test invalid JSON handling
    invalid_line = "invalid json {"
    parsed_invalid = _parse_analytics_line(invalid_line)
    assert parsed_invalid is None, "Should return None for invalid JSON"

    # Test empty line handling
    empty_parsed = _parse_analytics_line("")
    assert empty_parsed is None, "Should return None for empty line"

    # Test timestamp parsing
    test_obj = {"ts": "2023-01-01T12:00:00+00:00"}
    timestamp = _parse_timestamp(test_obj)
    assert timestamp is not None, "Should parse valid timestamp"
    assert timestamp.year == 2023, "Should extract correct year"

    # Test invalid timestamp handling
    invalid_ts_obj = {"ts": "invalid-timestamp"}
    invalid_timestamp = _parse_timestamp(invalid_ts_obj)
    assert invalid_timestamp is None, "Should return None for invalid timestamp"

    return True


def _test_weekly_summary_generation() -> bool:
    """Test weekly summary generation with sample data."""
    import os
    import tempfile

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock the analytics path
        def mock_get_path() -> Path:
            return Path(temp_dir) / "analytics.jsonl"

        original_func = _get_analytics_path
        globals()['_get_analytics_path'] = mock_get_path

        try:
            # Create sample analytics data
            sample_events = [
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "action_name": "action1",
                    "choice": "choice1",
                    "success": True,
                    "duration_sec": 1.0,
                    "mem_used_mb": 50.0,
                    "extras": {},
                    "pid": 1234
                },
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "action_name": "action1",
                    "choice": "choice2",
                    "success": False,
                    "duration_sec": 2.0,
                    "mem_used_mb": 75.0,
                    "extras": {},
                    "pid": 1234
                },
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "action_name": "action2",
                    "choice": "choice1",
                    "success": True,
                    "duration_sec": 3.0,
                    "mem_used_mb": 100.0,
                    "extras": {},
                    "pid": 1234
                }
            ]

            # Write sample data to file
            analytics_path = mock_get_path()
            with analytics_path.open('w', encoding='utf-8') as f:
                for event in sample_events:
                    f.write(json.dumps(event) + "\n")

            # Generate weekly summary
            summary = generate_weekly_summary(days=7)

            # Verify summary structure
            assert "action1" in summary, "Should include action1"
            assert "action2" in summary, "Should include action2"

            action1_stats = summary["action1"]
            assert action1_stats["runs"] == 2, "Action1 should have 2 runs"
            assert action1_stats["success"] == 1, "Action1 should have 1 success"
            assert action1_stats["success_rate"] == 0.5, "Action1 should have 50% success rate"
            assert action1_stats["avg_duration_sec"] == 1.5, "Action1 should have 1.5s average duration"
            assert action1_stats["avg_mem_mb"] == 62.5, "Action1 should have 62.5MB average memory"

            action2_stats = summary["action2"]
            assert action2_stats["runs"] == 1, "Action2 should have 1 run"
            assert action2_stats["success"] == 1, "Action2 should have 1 success"
            assert action2_stats["success_rate"] == 1.0, "Action2 should have 100% success rate"

            return True

        finally:
            # Restore original function
            globals()['_get_analytics_path'] = original_func

    return False


def analytics_module_tests() -> bool:
    """Comprehensive test suite for analytics.py"""
    from test_framework import TestSuite

    suite = TestSuite("Analytics Module", "analytics.py")
    suite.start_suite()

    # Test analytics initialization
    suite.run_test(
        "Analytics Initialization",
        _test_analytics_initialization,
        "Test analytics module initialization and state management",
        "Test transient extras state management",
        "Verify extras are set, retrieved, and cleared correctly"
    )

    # Test path resolution
    suite.run_test(
        "Analytics Path Resolution",
        _test_analytics_path_resolution,
        "Test analytics path resolution and directory creation",
        "Test _get_analytics_path function",
        "Verify correct path format and directory existence"
    )

    # Test event logging
    suite.run_test(
        "Event Logging",
        _test_event_logging,
        "Test event logging functionality with temporary file",
        "Test log_event function with sample data",
        "Verify events are logged with correct structure and data"
    )

    # Test parsing functions
    suite.run_test(
        "Analytics Parsing",
        _test_analytics_parsing,
        "Test analytics line parsing and timestamp extraction",
        "Test _parse_analytics_line and _parse_timestamp functions",
        "Verify correct parsing of valid/invalid JSON and timestamps"
    )

    # Test weekly summary generation
    suite.run_test(
        "Weekly Summary Generation",
        _test_weekly_summary_generation,
        "Test weekly summary generation with sample data",
        "Test generate_weekly_summary function",
        "Verify summary statistics are calculated correctly"
    )

    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(analytics_module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    import sys
    sys.exit(0 if success else 1)
