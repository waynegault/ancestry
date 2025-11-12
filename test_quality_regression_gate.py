#!/usr/bin/env python3
"""Tests for quality_regression_gate JSON output and timezone behavior."""

import io
import json
import tempfile
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from quality_regression_gate import load_baseline, main
from test_framework import TestSuite


def _create_experiments_file(path: Path, scores: list[float]) -> None:
    lines = []
    now = datetime.now(timezone.utc).isoformat()
    for s in scores:
        entry = {
            "timestamp_utc": now,
            "parse_success": True,
            "quality_score": s,
        }
        lines.append(json.dumps(entry))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def test_generate_baseline_and_json_output() -> None:
    tmpdir = Path(tempfile.mkdtemp(prefix="qrg-test-"))
    experiments = tmpdir / "prompt_experiments.jsonl"
    baseline = tmpdir / "quality_baseline.json"

    # Create experiments with median 85.0
    _create_experiments_file(experiments, [80.0, 85.0, 90.0])

    f = io.StringIO()
    with redirect_stdout(f):
        rc = main([
            "--experiments-file",
            str(experiments),
            "--baseline-file",
            str(baseline),
            "--generate-baseline",
            "--json",
        ])

    assert rc == 0, "generate-baseline should exit 0"
    assert baseline.exists(), "baseline file should be created"
    content = load_baseline(baseline)
    assert "median_quality_score" in content
    assert "baseline_id" in content


def test_regression_json_detection() -> None:
    tmpdir = Path(tempfile.mkdtemp(prefix="qrg-test-"))
    experiments = tmpdir / "prompt_experiments.jsonl"
    baseline = tmpdir / "quality_baseline.json"

    # Baseline 90
    baseline.write_text(json.dumps({"median_quality_score": 90.0, "baseline_id": "test-1", "generated_at": datetime.now(timezone.utc).isoformat()}))

    # New experiments median 80 -> regression
    _create_experiments_file(experiments, [78.0, 80.0, 82.0])

    f = io.StringIO()
    with redirect_stdout(f):
        rc = main([
            "--experiments-file",
            str(experiments),
            "--baseline-file",
            str(baseline),
            "--json",
        ])

    out = f.getvalue().strip()
    # Parse JSON output
    try:
        data = json.loads(out)
    except Exception as err:
        raise AssertionError(f"Output was not valid JSON: {out!r}") from err

    assert rc == 1, "Regression should return exit code 1"
    assert data.get("status") == "regression"
    assert data.get("is_regression") is True


def run_comprehensive_tests() -> bool:
    suite = TestSuite("Quality Regression Gate Tests", "test_quality_regression_gate.py")
    suite.start_suite()
    suite.run_test("Generate baseline JSON", test_generate_baseline_and_json_output, "Should generate baseline and include baseline_id")
    suite.run_test("Detect regression JSON", test_regression_json_detection, "Should detect regression and emit JSON status")
    return suite.finish_suite()


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
