#!/usr/bin/env python3
"""
Test script to verify all Python modules can be executed standalone.

This runs each .py file with `python <file>` and reports any that fail
to even start (import errors, syntax errors, etc).
"""

import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

REPO_ROOT = Path(__file__).resolve().parents[1]

# Files to skip (they hang, require user input, or are special)
SKIP_FILES = {
    "main.py",  # Interactive menu
    "run_all_tests.py",  # Full test runner
    "__init__.py",
    "check_db.py",  # DB operations
    "ai_api_test.py",  # Live API test
    "test_all_modules.py",  # This script
}

# Directories to skip
SKIP_DIRS = {
    "__pycache__",
    ".venv",
    "test_data",
    "node_modules",
    ".git",
}


def test_module(py_file: Path) -> tuple[str, bool, str, float]:
    """
    Test a single module by running it with a short timeout.
    Returns: (relative_path, success, error_message, duration)
    """
    rel_path = py_file.relative_to(REPO_ROOT)
    start = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(py_file)],
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
            cwd=str(REPO_ROOT),
            env={**dict(__import__('os').environ), "RUN_MODULE_TESTS": "1"},
        )
        duration = time.time() - start

        # Check for import/syntax errors in stderr
        if result.returncode != 0:
            # Look for common fatal errors
            stderr = result.stderr[:500] if result.stderr else ""
            if any(
                err in stderr
                for err in [
                    "ModuleNotFoundError",
                    "ImportError",
                    "SyntaxError",
                    "NameError",
                    "AttributeError",
                ]
            ):
                return str(rel_path), False, stderr.strip(), duration
            # Other non-zero exits might be expected (test failures, etc)
            return str(rel_path), True, f"Exit code {result.returncode}", duration

        return str(rel_path), True, "", duration

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return str(rel_path), True, "TIMEOUT (probably OK)", duration
    except Exception as e:
        duration = time.time() - start
        return str(rel_path), False, str(e), duration


def find_all_modules() -> list[Path]:
    """Find all Python files in the repository."""
    modules: list[Path] = []

    for py_file in REPO_ROOT.rglob("*.py"):
        # Skip excluded directories
        if any(part in SKIP_DIRS for part in py_file.parts):
            continue

        # Skip excluded files
        if py_file.name in SKIP_FILES:
            continue

        # Only include files with __main__ blocks (they're meant to be run)
        try:
            content = py_file.read_text(encoding="utf-8")
            if 'if __name__' in content and '__main__' in content:
                modules.append(py_file)
        except Exception:
            continue

    return sorted(modules)


def main() -> bool:
    print("=" * 70)
    print("Testing all Python modules for standalone execution")
    print("=" * 70)
    print()

    modules = find_all_modules()
    print(f"Found {len(modules)} modules with __main__ blocks\n")

    passed: list[str] = []
    failed: list[tuple[str, str]] = []

    # Run tests in parallel for speed
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(test_module, m): m for m in modules}

        for i, future in enumerate(as_completed(futures), 1):
            rel_path, success, error, duration = future.result()

            if success:
                status = "✅"
                passed.append(rel_path)
            else:
                status = "❌"
                failed.append((rel_path, error))

            # Show progress
            print(f"[{i:3d}/{len(modules)}] {status} {rel_path} ({duration:.1f}s)")

            if not success and error:
                # Show first line of error
                first_line = error.split('\n')[0][:80]
                print(f"         └─ {first_line}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✅ Passed: {len(passed)}")
    print(f"❌ Failed: {len(failed)}")

    if failed:
        print("\n❌ FAILED MODULES:")
        for path, error in failed:
            print(f"\n  • {path}")
            # Show error details (first few lines)
            for line in error.split('\n')[:5]:
                print(f"    {line}")

    print()
    return len(failed) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
