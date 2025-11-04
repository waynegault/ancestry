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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional


# Module-level transient extras storage (cleared after use)
class _State:
    extras: Optional[dict[str, Any]] = None


def set_transient_extras(extras: dict[str, Any]) -> None:
    """Attach transient extras for the current action; consumed by exec_actn()."""
    try:
        _State.extras = dict(extras) if extras is not None else None
    except Exception as e:
        logger.debug(f"analytics.set_transient_extras failed: {e}")
        _State.extras = None


def pop_transient_extras() -> Optional[dict[str, Any]]:
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
    mem_used_mb: Optional[float] = None,
    extras: Optional[dict[str, Any]] = None,
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


def _parse_analytics_line(line: str) -> Optional[dict[str, Any]]:
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


def _parse_timestamp(obj: dict[str, Any]) -> Optional[datetime]:
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

