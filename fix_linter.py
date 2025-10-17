#!/usr/bin/env python3
"""Quick script to run ruff --fix and show results."""
import subprocess
import sys


def main():
    """Run ruff check --fix."""
    print("Running ruff check --fix...")
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--fix", "."],
        capture_output=True,
        text=True, check=False
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    print(f"\nReturn code: {result.returncode}")

    # Now show remaining issues
    print("\n" + "="*60)
    print("Checking for remaining issues...")
    print("="*60)
    result2 = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "."],
        capture_output=True,
        text=True, check=False
    )

    print(result2.stdout)
    if result2.stderr:
        print("STDERR:", result2.stderr)

    return result2.returncode

if __name__ == "__main__":
    sys.exit(main())

