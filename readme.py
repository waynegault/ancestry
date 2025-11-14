#!/usr/bin/env python3
"""CLI helper summarizing key project entry points.

This lightweight wrapper keeps the most frequently referenced
project facts close to the codebase and avoids hunting through the
long-form `readme.md` when a quick refresher is needed.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
README_PATH = PROJECT_ROOT / "readme.md"


def _build_summary() -> str:
    lines = [
        "Ancestry Research Automation quickstart:",
        "  - Activate the virtual environment: `.venv\\Scripts\\activate`",
        "  - Run the comprehensive suite: `python run_all_tests.py`",
        (
            "  - Module tests live beside their implementations; call "
            "`run_comprehensive_tests()` from any module for focused checks."
        ),
        (
            "  - Prometheus metrics exporter is available once `ObservabilityConfig` "
            "enables metrics; hit `/metrics` on the configured host to inspect the feed."
        ),
    ]
    return "\n".join(lines)


def _read_excerpt(lines: int = 25) -> str:
    if README_PATH.exists():
        with README_PATH.open(encoding="utf-8") as handle:
            head = [next(handle, "").rstrip("\n") for _ in range(lines)]
        return "\n".join(line for line in head if line)
    return "readme.md not found."


def main() -> int:
    parser = argparse.ArgumentParser(description="Show project quickstart information")
    parser.add_argument(
        "--show-readme",
        action="store_true",
        help="Print the first 25 non-empty lines from readme.md",
    )
    args = parser.parse_args()

    print(_build_summary())
    if args.show_readme:
        print("\n--- readme.md excerpt ---")
        print(_read_excerpt())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
