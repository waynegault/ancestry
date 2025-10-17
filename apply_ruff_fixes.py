#!/usr/bin/env python3
"""Apply ruff fixes and commit."""
import os
import subprocess
import sys

os.chdir(r"C:\Users\wayne\GitHub\Python\Projects\Ancestry")

print("="*60)
print("STEP 1: Running ruff check --fix")
print("="*60)
result = subprocess.run(
    [sys.executable, "-m", "ruff", "check", "--fix", "."],
    capture_output=False,  # Show output directly
    text=True, check=False
)

print(f"\nRuff fix return code: {result.returncode}")

print("\n" + "="*60)
print("STEP 2: Checking remaining issues")
print("="*60)
result2 = subprocess.run(
    [sys.executable, "-m", "ruff", "check", ".", "--statistics"],
    capture_output=False,
    text=True, check=False
)

print(f"\nRuff check return code: {result2.returncode}")

print("\n" + "="*60)
print("STEP 3: Git status")
print("="*60)
subprocess.run(["git", "status", "--short"], check=False)

print("\n" + "="*60)
print("Done! Now you can commit and push.")
print("="*60)

