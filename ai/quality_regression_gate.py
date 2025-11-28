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

# === PATH SETUP FOR PACKAGE IMPORTS ===
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import io
import json
import os
import subprocess
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional


def load_experiments(log_file: Path, days: int = 7) -> list[dict[str, Any]]:
    """Load recent experiment entries from JSONL file."""
    if not log_file.exists():
        return []

    # Use timezone-aware UTC cutoff so it compares cleanly with parsed ISO timestamps
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    experiments: list[dict[str, Any]] = []

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
    scores: list[float] = []
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
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
        ).strip()
    except Exception:
        git_sha = None

    timestamp = datetime.now(timezone.utc)

    baseline = {
        "median_quality_score": median_score,
        "experiment_count": experiment_count,
        "generated_at": timestamp.isoformat().replace("+00:00", "Z"),
        "baseline_id": f"{timestamp.strftime('%Y%m%d%H%M%S')}-{git_sha or 'nogit'}",
        "git_ref": git_sha,
        "description": "Baseline quality score for regression detection",
    }

    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    with baseline_file.open("w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)

    print(f"✅ Baseline saved to {baseline_file}")
    print(f"   Median quality score: {median_score:.1f}")
    print(f"   Based on {experiment_count} experiments")


def check_regression(current_median: float, baseline_median: float, threshold: float = 5.0) -> tuple[bool, float]:
    """
    Check if current quality represents a regression.

    Returns:
        (is_regression, quality_drop)
    """
    quality_drop = baseline_median - current_median
    is_regression = quality_drop > threshold
    return is_regression, quality_drop


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quality Regression Gate - Block deployments on prompt quality drops")
    parser.add_argument(
        "--generate-baseline", action="store_true", help="Generate new baseline from recent experiments"
    )
    parser.add_argument("--threshold", type=float, default=5.0, help="Quality drop threshold (default: 5.0 points)")
    parser.add_argument("--days", type=int, default=7, help="Number of days of experiments to analyze (default: 7)")
    parser.add_argument(
        "--experiments-file",
        type=Path,
        default=Path("Logs/prompt_experiments.jsonl"),
        help="Path to experiments JSONL file",
    )
    parser.add_argument(
        "--baseline-file", type=Path, default=Path("Data/quality_baseline.json"), help="Path to baseline JSON file"
    )
    parser.add_argument("--json", action="store_true", help="Emit a compact JSON summary to stdout (machine-readable)")
    return parser


def _handle_no_experiments(experiments: list[dict[str, Any]], args: argparse.Namespace) -> Optional[int]:
    if experiments:
        return None
    if args.json:
        out = {
            "status": "error",
            "reason": "no_experiments",
            "experiments_file": str(args.experiments_file),
        }
        print(json.dumps(out, separators=(",", ":")))
    else:
        print(f"❌ ERROR: No experiments found in {args.experiments_file}")
        print("   Cannot perform quality check without data.")
    return 2


def _handle_invalid_scores(current_median: Optional[float], experiments: list[dict[str, Any]]) -> Optional[int]:
    if current_median is not None:
        return None
    print("❌ ERROR: No valid quality scores in experiments")
    print(f"   Found {len(experiments)} experiments but none had quality_score.")
    return 2


def _print_current_metrics(args: argparse.Namespace, experiment_count: int, current_median: float) -> None:
    if args.json:
        return
    print(f"📊 Current Quality Metrics (last {args.days} days):")
    print(f"   Experiments analyzed: {experiment_count}")
    print(f"   Median quality score: {current_median:.1f}")


def _emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":")))


def _generate_baseline_flow(
    args: argparse.Namespace,
    current_median: float,
    experiments: list[dict[str, Any]],
) -> int:
    if args.json:
        with redirect_stdout(io.StringIO()):
            save_baseline(args.baseline_file, current_median, len(experiments))
    else:
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
        _emit_json(out)
    return 0


def _handle_missing_baseline(
    baseline: dict[str, Any],
    experiments: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Optional[int]:
    if baseline and "median_quality_score" in baseline:
        return None
    if args.json:
        out = {
            "status": "no_baseline",
            "baseline_file": str(args.baseline_file),
            "allow_deploy": True,
            "experiments_analyzed": len(experiments),
        }
        _emit_json(out)
    else:
        print(f"\n⚠️  WARNING: No baseline found at {args.baseline_file}")
        print("   Run with --generate-baseline to create one.")
        print("   Allowing deployment (no baseline to compare against).")
    return 0


def _evaluate_against_baseline(
    args: argparse.Namespace,
    experiments: list[dict[str, Any]],
    current_median: float,
    baseline: dict[str, Any],
) -> int:
    baseline_median = baseline["median_quality_score"]
    baseline_date = baseline.get("generated_at", "unknown")

    if not args.json:
        print("\n📌 Baseline Quality Metrics:")
        print(f"   Median quality score: {baseline_median:.1f}")
        print(f"   Generated at: {baseline_date}")

    is_regression, quality_drop = check_regression(
        current_median,
        baseline_median,
        args.threshold,
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
        _emit_json(out)
        return 1 if is_regression else 0

    print(f"\n{'=' * 60}")
    if is_regression:
        print("❌ QUALITY REGRESSION DETECTED!")
        print(f"   Quality dropped by {quality_drop:.1f} points")
        print(f"   Threshold: {args.threshold:.1f} points")
        print(f"   Current: {current_median:.1f} vs Baseline: {baseline_median:.1f}")
        print("\n🚫 BLOCKING DEPLOYMENT")
        print(f"   Review recent prompt changes in {args.experiments_file}")
        print("   Run 'python prompt_telemetry.py --stats' for details")
        print(f"{'=' * 60}")
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
    print(f"{'=' * 60}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point for quality regression gate."""
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    experiments = load_experiments(args.experiments_file, args.days)
    status = _handle_no_experiments(experiments, args)
    if status is not None:
        return status

    current_median = calculate_median_quality(experiments)
    status = _handle_invalid_scores(current_median, experiments)
    if status is not None:
        return status
    assert current_median is not None  # For type checkers

    _print_current_metrics(args, len(experiments), current_median)

    if args.generate_baseline:
        return _generate_baseline_flow(args, current_median, experiments)

    baseline = load_baseline(args.baseline_file)
    status = _handle_missing_baseline(baseline, experiments, args)
    if status is not None:
        return status
    assert baseline and "median_quality_score" in baseline

    return _evaluate_against_baseline(args, experiments, current_median, baseline)


# ==============================================
# Module Tests
# ==============================================


def _create_test_experiments_file(path: Path, scores: list[float]) -> None:
    """Create a JSONL experiments file containing the provided scores."""
    from datetime import datetime, timezone

    entries: list[str] = []
    now = datetime.now(timezone.utc).isoformat()
    for score in scores:
        entries.append(
            json.dumps(
                {
                    "timestamp_utc": now,
                    "parse_success": True,
                    "quality_score": score,
                }
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(entries), encoding="utf-8")


def _test_generate_baseline_json_output() -> None:
    """Ensure --generate-baseline emits JSON with baseline metadata."""
    import io
    from contextlib import redirect_stdout

    from testing.test_utilities import temp_directory

    with temp_directory(prefix="qrg-test-") as tmpdir:
        experiments = tmpdir / "prompt_experiments.jsonl"
        baseline = tmpdir / "quality_baseline.json"

        _create_test_experiments_file(experiments, [80.0, 85.0, 90.0])

        capture = io.StringIO()
        with redirect_stdout(capture):
            rc = main(
                [
                    "--experiments-file",
                    str(experiments),
                    "--baseline-file",
                    str(baseline),
                    "--generate-baseline",
                    "--json",
                ]
            )

        assert rc == 0, "Baseline generation should exit successfully"
        assert baseline.exists(), "Baseline file should be created"

        baseline_content = load_baseline(baseline)
        assert "median_quality_score" in baseline_content, "Baseline should record median score"
        assert "baseline_id" in baseline_content, "Baseline should include baseline_id"

        output = capture.getvalue().strip()
        if output:
            parsed = json.loads(output)
            assert parsed.get("status") == "baseline_generated"


def _test_regression_detection_json_mode() -> None:
    """Ensure JSON output reports regression status correctly."""
    import io
    from contextlib import redirect_stdout
    from datetime import datetime, timezone

    from testing.test_utilities import temp_directory

    with temp_directory(prefix="qrg-test-") as tmpdir:
        experiments = tmpdir / "prompt_experiments.jsonl"
        baseline = tmpdir / "quality_baseline.json"

        baseline_payload = {
            "median_quality_score": 90.0,
            "baseline_id": "test-1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        baseline.write_text(json.dumps(baseline_payload), encoding="utf-8")

        _create_test_experiments_file(experiments, [78.0, 80.0, 82.0])

        capture = io.StringIO()
        with redirect_stdout(capture):
            rc = main(
                [
                    "--experiments-file",
                    str(experiments),
                    "--baseline-file",
                    str(baseline),
                    "--json",
                ]
            )

        assert rc == 1, "Regression should exit with status code 1"

        output = capture.getvalue().strip()
        parsed = json.loads(output)
        assert parsed.get("status") == "regression", f"Unexpected JSON status: {parsed}"
        assert parsed.get("is_regression") is True
        assert parsed.get("quality_drop", 0) > 0


def quality_regression_gate_module_tests() -> bool:
    """Run quality regression gate unit tests."""
    from testing.test_framework import TestSuite, suppress_logging

    suite = TestSuite("Quality Regression Gate", "quality_regression_gate.py")
    suite.start_suite()

    with suppress_logging():
        suite.run_test(
            "Baseline generation JSON output",
            _test_generate_baseline_json_output,
            "--generate-baseline should emit JSON with baseline metadata",
        )
        suite.run_test(
            "Regression detection JSON output",
            _test_regression_detection_json_mode,
            "Regression run should emit JSON status and non-zero exit",
        )

    return suite.finish_suite()


try:
    from testing.test_utilities import create_standard_test_runner

    run_comprehensive_tests = create_standard_test_runner(quality_regression_gate_module_tests)
except ImportError:  # pragma: no cover - minimal environments may skip helper import

    def run_comprehensive_tests() -> bool:
        return quality_regression_gate_module_tests()


def _should_run_module_tests() -> bool:
    return os.environ.get("RUN_MODULE_TESTS") == "1"


if __name__ == "__main__":
    if _should_run_module_tests():
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    sys.exit(main())
