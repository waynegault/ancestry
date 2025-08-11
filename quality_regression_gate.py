#!/usr/bin/env python3
"""Quality Regression Gate

Purpose:
  Optional CI/build gating script that fails (exit code 1) if the current
  prompt extraction quality shows a median drop >= configured threshold
  relative to a previously built baseline.

Usage:
  1. Build (or refresh) a baseline after you are satisfied with current quality:
       python prompt_telemetry.py --build-baseline --variant control --window 300 --min-events 8
  2. (CI) Run this gate to ensure no regression (returns non‑zero on regression):
       python quality_regression_gate.py

Environment Variables (override defaults):
  QUALITY_GATE_VARIANT          Variant label to check (default: control)
  QUALITY_GATE_WINDOW           Event window size to examine (default: 120)
  QUALITY_GATE_DROP_THRESHOLD   Median drop threshold (default: 15.0)
  QUALITY_GATE_VERBOSE          If set (to any value), prints detailed status

Exit Codes:
  0 = Passed (no baseline, no data, or no regression)
  1 = Regression detected (median drop >= threshold)
  2 = Unexpected internal error

Notes:
  - Gate is intentionally permissive when no baseline exists so first runs pass.
  - Baseline file: Logs/prompt_quality_baseline.json (created by prompt_telemetry)
  - Telemetry file: Logs/prompt_experiments.jsonl
"""
from __future__ import annotations

import os, sys, json
from typing import Any

try:
    from prompt_telemetry import detect_quality_regression  # type: ignore
except Exception:
    print("ERROR: Unable to import prompt_telemetry for regression detection", file=sys.stderr)
    sys.exit(2)


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def main() -> int:
    variant = os.environ.get("QUALITY_GATE_VARIANT", "control")
    window = int(os.environ.get("QUALITY_GATE_WINDOW", "120") or 120)
    drop_threshold = _get_env_float("QUALITY_GATE_DROP_THRESHOLD", 15.0)
    verbose = os.environ.get("QUALITY_GATE_VERBOSE") is not None

    try:
        result: dict[str, Any] = detect_quality_regression(
            current_window=window,
            drop_threshold=drop_threshold,
            variant=variant,
        )
    except Exception as e:
        print(f"ERROR: quality regression detection failed: {e}", file=sys.stderr)
        return 2

    status = result.get("status")
    regression = bool(result.get("regression"))

    if verbose:
        print(json.dumps({"gate_result": result, "variant": variant, "drop_threshold": drop_threshold}, indent=2))
    else:
        # Concise line
        if status == "ok":
            drop = result.get("drop")
            med_now = result.get("median_now")
            med_then = result.get("baseline_median")
            print(f"QualityGate status=ok variant={variant} median_now={med_now} baseline_median={med_then} drop={drop} threshold={drop_threshold} regression={regression}")
        else:
            print(f"QualityGate status={status} variant={variant} (no enforcement)")

    # Enforcement logic
    if status == "ok" and regression:
        return 1
    # lenient pass if no baseline / no data / ok without regression
    return 0


if __name__ == "__main__":
    sys.exit(main())
