#!/usr/bin/env python3
"""
Quality Regression Gate for CI/CD Pipeline

Blocks deployment if AI prompt quality drops more than 5 points from baseline.
Reads prompt_experiments.jsonl and compares median quality scores.

Exit codes:
  0 - No regression detected (quality maintained or improved)
  1 - Quality regression detected (>5 point drop from baseline)
  2 - Error (insufficient data, missing files, etc.)

Usage:
  python quality_regression_gate.py                    # Check against baseline
  python quality_regression_gate.py --generate-baseline # Create new baseline
  python quality_regression_gate.py --threshold 10     # Custom threshold
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional


def load_experiments(log_file: Path, days: int = 7) -> list[dict[str, Any]]:
    """Load recent experiment entries from JSONL file."""
    if not log_file.exists():
        return []

    # Use timezone-aware UTC cutoff so it compares cleanly with parsed ISO timestamps
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    experiments = []

    with log_file.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Parse timestamp
                ts_str = entry.get("timestamp_utc", "")
                if ts_str:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if ts >= cutoff:
                        experiments.append(entry)
                else:
                    # Include entries without timestamps
                    experiments.append(entry)
            except (json.JSONDecodeError, ValueError):
                continue

    return experiments


def calculate_median_quality(experiments: list[dict[str, Any]]) -> float | None:
    """Calculate median quality score from experiments."""
    scores = []
    for exp in experiments:
        if exp.get("parse_success") and "quality_score" in exp:
            score = exp["quality_score"]
            if isinstance(score, (int, float)) and 0 <= score <= 100:
                scores.append(float(score))

    if not scores:
        return None

    scores.sort()
    n = len(scores)
    if n % 2 == 0:
        return (scores[n // 2 - 1] + scores[n // 2]) / 2.0
    return scores[n // 2]


def load_baseline(baseline_file: Path) -> dict[str, Any]:
    """Load baseline quality metrics."""
    if not baseline_file.exists():
        return {}

    try:
        with baseline_file.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_baseline(baseline_file: Path, median_score: float, experiment_count: int) -> None:
    """Save new baseline quality metrics."""
    # Try to capture current git short SHA for provenance, if available
    git_sha = None
    try:
        git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                text=True,
            )
            .strip()
        )
    except Exception:
        git_sha = None

    baseline = {
        "median_quality_score": median_score,
        "experiment_count": experiment_count,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "baseline_id": f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{git_sha or 'nogit'}",
        "git_ref": git_sha,
        "description": "Baseline quality score for regression detection",
    }

    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    with baseline_file.open("w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)

    print(f"✅ Baseline saved to {baseline_file}")
    print(f"   Median quality score: {median_score:.1f}")
    print(f"   Based on {experiment_count} experiments")


def check_regression(
    current_median: float,
    baseline_median: float,
    threshold: float = 5.0
) -> tuple[bool, float]:
    """
    Check if current quality represents a regression.

    Returns:
        (is_regression, quality_drop)
    """
    quality_drop = baseline_median - current_median
    is_regression = quality_drop > threshold
    return is_regression, quality_drop


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911
    """Main entry point for quality regression gate."""
    parser = argparse.ArgumentParser(
        description="Quality Regression Gate - Block deployments on prompt quality drops"
    )
    parser.add_argument(
        "--generate-baseline",
        action="store_true",
        help="Generate new baseline from recent experiments"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Quality drop threshold (default: 5.0 points)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days of experiments to analyze (default: 7)"
    )
    parser.add_argument(
        "--experiments-file",
        type=Path,
        default=Path("Logs/prompt_experiments.jsonl"),
        help="Path to experiments JSONL file"
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=Path("Logs/quality_baseline.json"),
        help="Path to baseline JSON file"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a compact JSON summary to stdout (machine-readable)"
    )

    args = parser.parse_args(argv)

    # Load recent experiments
    experiments = load_experiments(args.experiments_file, args.days)

    if not experiments:
        if args.json:
            out = {
                "status": "error",
                "reason": "no_experiments",
                "experiments_file": str(args.experiments_file),
            }
            print(json.dumps(out, separators=(",", ":")))
            return 2
        print(f"❌ ERROR: No experiments found in {args.experiments_file}")
        print("   Cannot perform quality check without data.")
        return 2

    current_median = calculate_median_quality(experiments)

    if current_median is None:
        print("❌ ERROR: No valid quality scores in experiments")
        print(f"   Found {len(experiments)} experiments but none had quality_score.")
        return 2

    if not args.json:
        print(f"📊 Current Quality Metrics (last {args.days} days):")
        print(f"   Experiments analyzed: {len(experiments)}")
        print(f"   Median quality score: {current_median:.1f}")
    else:
        # In JSON mode we'll emit a compact summary later; nothing to print here now
        pass

    # Generate baseline mode
    if args.generate_baseline:
        save_baseline(args.baseline_file, current_median, len(experiments))
        if args.json:
            baseline_content = load_baseline(args.baseline_file)
            out = {
                "status": "baseline_generated",
                "median_quality_score": round(current_median, 3),
                "experiment_count": len(experiments),
                "baseline_file": str(args.baseline_file),
                "baseline_id": baseline_content.get("baseline_id"),
                "git_ref": baseline_content.get("git_ref"),
            }
            print(json.dumps(out, separators=(",", ":")))
        return 0

    # Check against baseline
    baseline = load_baseline(args.baseline_file)

    if not baseline or "median_quality_score" not in baseline:
        if args.json:
            out = {
                "status": "no_baseline",
                "baseline_file": str(args.baseline_file),
                "allow_deploy": True,
                "experiments_analyzed": len(experiments),
            }
            print(json.dumps(out, separators=(",", ":")))
            return 0
        print(f"\n⚠️  WARNING: No baseline found at {args.baseline_file}")
        print("   Run with --generate-baseline to create one.")
        print("   Allowing deployment (no baseline to compare against).")
        return 0

    baseline_median = baseline["median_quality_score"]
    baseline_date = baseline.get("generated_at", "unknown")

    if not args.json:
        print("\n📌 Baseline Quality Metrics:")
        print(f"   Median quality score: {baseline_median:.1f}")
        print(f"   Generated at: {baseline_date}")

    # Check for regression
    is_regression, quality_drop = check_regression(
        current_median, baseline_median, args.threshold
    )

    if args.json:
        out = {
            "status": "regression" if is_regression else "ok",
            "is_regression": bool(is_regression),
            "quality_drop": round(quality_drop, 3),
            "threshold": args.threshold,
            "current_median": round(current_median, 3),
            "baseline_median": round(baseline_median, 3),
            "experiments_analyzed": len(experiments),
            "baseline_generated_at": baseline_date,
            "baseline_id": baseline.get("baseline_id"),
            "git_ref": baseline.get("git_ref"),
        }
        print(json.dumps(out, separators=(",", ":")))
        return 1 if is_regression else 0

    print(f"\n{'='*60}")
    if is_regression:
        print("❌ QUALITY REGRESSION DETECTED!")
        print(f"   Quality dropped by {quality_drop:.1f} points")
        print(f"   Threshold: {args.threshold:.1f} points")
        print(f"   Current: {current_median:.1f} vs Baseline: {baseline_median:.1f}")
        print("\n🚫 BLOCKING DEPLOYMENT")
        print(f"   Review recent prompt changes in {args.experiments_file}")
        print("   Run 'python prompt_telemetry.py --stats' for details")
        print(f"{'='*60}")
        return 1
    improvement = current_median - baseline_median
    if improvement > 0:
        print("✅ QUALITY IMPROVED!")
        print(f"   Quality increased by {improvement:.1f} points")
    else:
        print("✅ QUALITY MAINTAINED")
        print(f"   Quality drop: {abs(quality_drop):.1f} points (within threshold)")
    print(f"   Current: {current_median:.1f} vs Baseline: {baseline_median:.1f}")
    print("\n✅ ALLOWING DEPLOYMENT")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
