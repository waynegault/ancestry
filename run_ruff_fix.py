#!/usr/bin/env python3
"""Run ruff --fix and show results."""
import subprocess
import sys

# Run ruff check --fix
print("Running ruff check --fix...")
result = subprocess.run(
    [sys.executable, "-m", "ruff", "check", "--fix", "."],
    capture_output=True,
    text=True,
    cwd=r"C:\Users\wayne\GitHub\Python\Projects\Ancestry", check=False
)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

print(f"\nReturn code: {result.returncode}")

# Show remaining issues
print("\n" + "="*60)
print("Remaining issues:")
print("="*60)
result2 = subprocess.run(
    [sys.executable, "-m", "ruff", "check", "."],
    capture_output=True,
    text=True,
    cwd=r"C:\Users\wayne\GitHub\Python\Projects\Ancestry", check=False
)

print(result2.stdout)
if result2.stderr:
    print("STDERR:", result2.stderr)

sys.exit(result2.returncode)

