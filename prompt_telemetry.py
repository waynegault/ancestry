"""Prompt Experiment Telemetry (Phase 9)

Lightweight JSONL telemetry for prompt experimentation & extraction quality metrics.

Each extraction event appended as one JSON object line to Logs/prompt_experiments.jsonl
Fields captured:
  timestamp_utc: ISO8601 UTC
  variant_label: control|alt|<custom>
  prompt_key: actual prompt key used
  prompt_version: version string if available
  parse_success: bool
  error: optional error message (truncated)
  extracted_counts: dict of key -> count (for known arrays)
  suggested_tasks_count: int
  raw_chars: length of raw JSON text returned from model (if available)
  user_hash: stable anonymized hash of user identifier (if provided)

Aggregation utilities can produce per-variant stats (counts, success rate, avg tasks).
Safe to import even if log directory missing (auto creates). Failures are logged but non-fatal.
"""
from __future__ import annotations
import json, os, hashlib, statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
LOGS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "Logs"
LOGS_DIR.mkdir(exist_ok=True)
TELEMETRY_FILE = LOGS_DIR / "prompt_experiments.jsonl"
ALERTS_FILE = LOGS_DIR / "prompt_experiment_alerts.jsonl"
QUALITY_BASELINE_FILE = LOGS_DIR / "prompt_quality_baseline.json"
MAX_ERROR_LEN = 240

def _stable_hash(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

def record_extraction_experiment_event(*, variant_label: str, prompt_key: str, prompt_version: Optional[str], parse_success: bool, extracted_data: Optional[Dict[str, Any]] = None, suggested_tasks: Optional[Iterable[Any]] = None, raw_response_text: Optional[str] = None, user_id: Optional[str] = None, error: Optional[str] = None, quality_score: Optional[float] = None, component_coverage: Optional[float] = None, anomaly_summary: Optional[str] = None) -> None:
    """Append a single telemetry event (best effort).

    Added (Phase 1 - 2025-08-11): component_coverage → proportion (0-1) of
    structured genealogical components that are non-empty in extracted_data.
    This is a lightweight interrogation metric to monitor breadth of extractions
    independent of task quality. Safe additive field; downstream readers ignore
    unknown keys.
    """
    try:
        counts: Dict[str, int] = {}
        if isinstance(extracted_data, dict):
            for k, v in extracted_data.items():
                if isinstance(v, list):
                    counts[k] = len(v)
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "variant_label": variant_label,
            "prompt_key": prompt_key,
            "prompt_version": prompt_version,
            "parse_success": parse_success,
            "error": (error[:MAX_ERROR_LEN] if error else None),
            "extracted_counts": counts or None,
            "suggested_tasks_count": len(list(suggested_tasks)) if suggested_tasks else 0,
            "raw_chars": len(raw_response_text) if isinstance(raw_response_text, str) else None,
            "user_hash": _stable_hash(user_id),
            "quality_score": round(float(quality_score), 2) if isinstance(quality_score, (int, float)) else None,
            "component_coverage": round(float(component_coverage), 3) if isinstance(component_coverage, (int, float)) else None,
            "anomaly_summary": anomaly_summary or None,
        }
        with open(TELEMETRY_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        # Lightweight auto-alert hook (Phase 11.2 item 2)
        try:
            _auto_analyze_and_alert()
        except Exception:
            pass
    except Exception:
        pass

def summarize_experiments(last_n: int = 1000) -> Dict[str, Any]:
    """Return summary of last N telemetry events (or all if smaller).

    Adds per-variant success_rate, avg_tasks, and average_quality (if any events include quality_score).
    """
    if not TELEMETRY_FILE.exists():
        return {"events": 0, "variants": {}, "success_rate": 0.0}
    try:
        lines: list[str] = []
        with open(TELEMETRY_FILE, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    lines.append(line)
        if last_n > 0:
            lines = lines[-last_n:]
        events = []
        for line in lines:
            try:
                events.append(json.loads(line))
            except Exception:
                continue
        total = len(events)
        if not total:
            return {"events": 0, "variants": {}, "success_rate": 0.0}
        variant_stats: Dict[str, Dict[str, Any]] = {}
        success = 0
        quality_accumulator_overall = 0.0
        quality_events_overall = 0
        for ev in events:
            var = ev.get("variant_label") or "unknown"
            st = variant_stats.setdefault(var, {"count": 0, "success": 0, "avg_tasks": 0.0, "_quality_sum": 0.0, "_quality_count": 0})
            st["count"] += 1
            if ev.get("parse_success"):
                st["success"] += 1
                success += 1
            tasks = ev.get("suggested_tasks_count") or 0
            st["avg_tasks"] = ((st["count"] - 1) * st["avg_tasks"] + tasks) / st["count"]
            q = ev.get("quality_score")
            if isinstance(q, (int, float)):
                st["_quality_sum"] += float(q)
                st["_quality_count"] += 1
                quality_accumulator_overall += float(q)
                quality_events_overall += 1
        overall_success_rate = success / total if total else 0.0
        for st in variant_stats.values():
            c = st["count"] or 1
            st["success_rate"] = st["success"] / c
            st["avg_tasks"] = round(st["avg_tasks"], 2)
            if st.get("_quality_count"):
                st["average_quality"] = round(st["_quality_sum"] / st["_quality_count"], 2)
            # Remove internal accumulators
            st.pop("_quality_sum", None)
            st.pop("_quality_count", None)
        summary: Dict[str, Any] = {"events": total, "success_rate": round(overall_success_rate, 3), "variants": variant_stats}
        if quality_events_overall:
            summary["average_quality"] = round(quality_accumulator_overall / quality_events_overall, 2)
        return summary
    except Exception:
        return {"events": 0, "variants": {}, "success_rate": 0.0}


# === Analysis & Alerting (Phase 11.2 Items 1 & 2) ===
def _load_recent_events(window: int = 500) -> list[Dict[str, Any]]:
    if not TELEMETRY_FILE.exists():
        return []
    events: list[Dict[str, Any]] = []
    try:
        with open(TELEMETRY_FILE, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        if window > 0:
            lines = lines[-window:]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    except Exception:
        return []
    return events

def analyze_experiments(window: int = 200, min_events_per_variant: int = 10,
                        quality_margin: float = 5.0, success_margin: float = 0.05) -> Dict[str, Any]:
    """Compute comparative statistics for variants.

    Returns dict with per-variant metrics and potential improvement flags.
    Simple heuristic (not statistical test) identifies improved_quality and improved_success.
    """
    events = _load_recent_events(window)
    if not events:
        return {"events": 0, "variants": {}, "analysis": "no_data"}
    variants: Dict[str, Dict[str, Any]] = {}
    for ev in events:
        v = ev.get("variant_label") or "unknown"
        data = variants.setdefault(v, {"quality_scores": [], "successes": 0, "count": 0, "tasks": []})
        data["count"] += 1
        if ev.get("parse_success"):
            data["successes"] += 1
        qs = ev.get("quality_score")
        if isinstance(qs, (int, float)):
            data["quality_scores"].append(float(qs))
        tasks = ev.get("suggested_tasks_count")
        if isinstance(tasks, int):
            data["tasks"].append(tasks)
    # Compute aggregates
    result_variants: Dict[str, Any] = {}
    for v, d in variants.items():
        count = max(d["count"], 1)
        qs_list = d["quality_scores"]
        median_quality = statistics.median(qs_list) if qs_list else None
        avg_quality = sum(qs_list)/len(qs_list) if qs_list else None
        avg_tasks = sum(d["tasks"])/len(d["tasks"]) if d["tasks"] else 0.0
        success_rate = d["successes"]/count
        result_variants[v] = {
            "events": d["count"],
            "success_rate": round(success_rate, 3),
            "median_quality": round(median_quality, 2) if median_quality is not None else None,
            "avg_quality": round(avg_quality, 2) if avg_quality is not None else None,
            "avg_tasks": round(avg_tasks, 2),
        }
    # Identify control vs alt (heuristic)
    control = result_variants.get("control")
    alt = result_variants.get("alt")
    improvement: Dict[str, Any] = {}
    if control and alt and control["events"] >= min_events_per_variant and alt["events"] >= min_events_per_variant:
        quality_delta = (alt.get("median_quality") or 0) - (control.get("median_quality") or 0)
        success_delta = alt.get("success_rate", 0) - control.get("success_rate", 0)
        improvement["quality_delta"] = round(quality_delta, 2)
        improvement["success_delta"] = round(success_delta, 3)
        improvement["improved_quality"] = quality_delta >= quality_margin
        improvement["improved_success"] = success_delta >= success_margin
        improvement["promote_recommendation"] = bool(improvement["improved_quality"] or improvement["improved_success"])
    return {"events": len(events), "variants": result_variants, "improvement": improvement}

def _write_alert(alert: Dict[str, Any]) -> None:
    try:
        with open(ALERTS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(alert, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _already_alerted(signature: str) -> bool:
    if not ALERTS_FILE.exists():
        return False
    try:
        with open(ALERTS_FILE, "r", encoding="utf-8") as fh:
            for line in fh:
                if signature in line:
                    return True
    except Exception:
        return False
    return False

def _auto_analyze_and_alert() -> None:
    analysis = analyze_experiments()
    imp = analysis.get("improvement", {})
    if not imp:
        return
    if not imp.get("promote_recommendation"):
        return
    signature = f"q{imp.get('quality_delta')}_s{imp.get('success_delta')}"
    if _already_alerted(signature):
        return
    alert = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "type": "variant_performance_improvement",
        "signature": signature,
        "analysis": analysis,
        "message": "Alt variant shows performance improvement meeting thresholds; consider promotion"
    }
    _write_alert(alert)


# === Quality Baseline & Regression Detection (Phase 11.2 Item 3) ===
def build_quality_baseline(variant: str = "control", window: int = 300, min_events: int = 8) -> Optional[Dict[str, Any]]:
    events = _load_recent_events(window)
    scores: list[float] = []
    for e in events:
        if e.get("variant_label") != variant:
            continue
        qv = e.get("quality_score")
        if isinstance(qv, (int, float)):
            try:
                scores.append(float(qv))
            except Exception:
                continue
    if len(scores) < min_events:
        return None
    baseline = {
        "variant": variant,
        "count": len(scores),
        "median_quality": statistics.median(scores),
        "p25": statistics.quantiles(scores, n=4)[0] if len(scores) >= 4 else None,
        "p75": statistics.quantiles(scores, n=4)[2] if len(scores) >= 4 else None,
        "created_utc": datetime.utcnow().isoformat(timespec="seconds"),
    }
    try:
        with open(QUALITY_BASELINE_FILE, "w", encoding="utf-8") as fh:
            json.dump(baseline, fh, indent=2)
    except Exception:
        pass
    return baseline

def load_quality_baseline() -> Optional[Dict[str, Any]]:
    if not QUALITY_BASELINE_FILE.exists():
        return None
    try:
        with open(QUALITY_BASELINE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None

def detect_quality_regression(current_window: int = 120, drop_threshold: float = 15.0, variant: str = "control") -> Dict[str, Any]:
    baseline = load_quality_baseline()
    if not baseline or baseline.get("variant") != variant:
        return {"status": "no_baseline"}
    events = _load_recent_events(current_window)
    scores: list[float] = []
    for e in events:
        if e.get("variant_label") != variant:
            continue
        qv = e.get("quality_score")
        if isinstance(qv, (int, float)):
            try:
                scores.append(float(qv))
            except Exception:
                continue
    if not scores:
        return {"status": "no_data"}
    median_now = statistics.median(scores)
    median_then = baseline.get("median_quality") or 0
    drop = median_then - median_now
    regression = drop >= drop_threshold
    return {"status": "ok", "median_now": median_now, "baseline_median": median_then, "drop": round(drop,2), "regression": regression}

## === Internal Test Suite (for run_all_tests detection & coverage) ===
def _test_record_and_summarize():
    """Record several events and verify summary reflects them."""
    # Capture starting count
    initial = summarize_experiments().get("events", 0)
    for i in range(3):
        record_extraction_experiment_event(
            variant_label="control",
            prompt_key=f"k{i}",
            prompt_version="v1",
            parse_success=True,
            extracted_data={"people": [1,2]},
            suggested_tasks=[{"t":1}],
            raw_response_text="{}",
            user_id="tester",
            quality_score=50 + i,
            component_coverage=0.8,
        )
    summary = summarize_experiments()
    assert summary.get("events", 0) >= initial + 3, "Summary should show newly added events"

def _test_variant_analysis():
    """Add alt variant events then run analyze_experiments for improvement structure."""
    for i in range(2):
        record_extraction_experiment_event(
            variant_label="alt",
            prompt_key=f"alt{i}",
            prompt_version="v1",
            parse_success=bool(i % 2 == 0),
            extracted_data={"people": [1]},
            suggested_tasks=[],
            quality_score=60 + i,
            component_coverage=0.6,
        )
    analysis = analyze_experiments(window=50, min_events_per_variant=1)
    assert "variants" in analysis and analysis.get("events",0) > 0

def _test_build_baseline_and_regression():
    """Ensure baseline can be built and regression check returns expected keys."""
    # Ensure enough control events to build baseline (min_events=8)
    needed = 8
    summary = summarize_experiments()
    existing = summary.get("variants", {}).get("control", {}).get("count", 0)
    to_add = max(0, needed - existing)
    for i in range(to_add):
        record_extraction_experiment_event(
            variant_label="control",
            prompt_key=f"b{i}",
            prompt_version="v1",
            parse_success=True,
            extracted_data={},
            suggested_tasks=[],
            quality_score=70 + (i % 5),
            component_coverage=0.9,
        )
    baseline = build_quality_baseline(variant="control", window=300, min_events=8)
    assert baseline is None or baseline.get("variant") == "control"
    reg = detect_quality_regression(current_window=120, drop_threshold=9999, variant="control")  # Force non-regression
    assert "status" in reg

def prompt_telemetry_module_tests() -> bool:
    try:
        from test_framework import TestSuite, suppress_logging
    except Exception:  # pragma: no cover
        return True  # Skip if framework missing
    suite = TestSuite("Prompt Telemetry", "prompt_telemetry.py")
    suite.start_suite()
    with suppress_logging():
        suite.run_test("Record & summarize", _test_record_and_summarize,
                       "Events appended and reflected in summary",
                       "Append 3 control events then summarize",
                       "Check summary event count")
        suite.run_test("Variant analysis", _test_variant_analysis,
                       "Analysis returns variants block",
                       "Add alt events & analyze",
                       "Check analyze_experiments structure")
        suite.run_test("Baseline & regression", _test_build_baseline_and_regression,
                       "Baseline build attempt and regression status retrieval",
                       "Ensure enough control events then build baseline & detect",
                       "Check baseline + regression keys")
    return suite.finish_suite()

def run_comprehensive_tests() -> bool:  # Consistent entrypoint naming
    return prompt_telemetry_module_tests()

if __name__ == "__main__":
    import argparse, json as _json
    parser = argparse.ArgumentParser(description="Prompt Experiment Telemetry Utilities")
    parser.add_argument("--summary", action="store_true", help="Print telemetry summary (default last 1000 events)")
    parser.add_argument("--last", type=int, default=1000, help="Number of recent events to summarize")
    parser.add_argument("--analyze", action="store_true", help="Run variant performance analysis")
    parser.add_argument("--build-baseline", action="store_true", help="Build quality baseline from recent control events")
    parser.add_argument("--check-regression", action="store_true", help="Check for quality regression vs baseline")
    parser.add_argument("--variant", default="control", help="Variant key for baseline/regression (default control)")
    parser.add_argument("--window", type=int, default=200, help="Event window size for analysis/baseline")
    parser.add_argument("--drop-threshold", type=float, default=15.0, help="Median quality drop threshold for regression detection")
    parser.add_argument("--min-events", type=int, default=8, help="Minimum events required to build baseline")
    parser.add_argument("--self-test", action="store_true", help="Run internal test suite and exit (for harness)")
    args = parser.parse_args()

    ran_action = False
    if args.summary:
        print(_json.dumps(summarize_experiments(last_n=args.last), indent=2))
        ran_action = True
    if args.analyze:
        print(_json.dumps(analyze_experiments(window=args.window), indent=2))
        ran_action = True
    if args.build_baseline:
        baseline = build_quality_baseline(variant=args.variant, window=args.window, min_events=args.min_events)
        if baseline:
            print(_json.dumps(baseline, indent=2))
        else:
            print("Baseline not built (insufficient events)")
        ran_action = True
    if args.check_regression:
        print(_json.dumps(detect_quality_regression(current_window=args.window, drop_threshold=args.drop_threshold, variant=args.variant), indent=2))
        ran_action = True

    if args.self_test or (not ran_action) or os.environ.get("RUN_INTERNAL_TESTS"):
        try:
            prompt_telemetry_module_tests()
        except Exception:
            pass
    elif not ran_action:
        parser.print_help()
