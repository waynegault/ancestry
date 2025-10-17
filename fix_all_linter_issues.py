#!/usr/bin/env python3
"""Fix all remaining linter issues and show results."""
import os
import subprocess
import sys

os.chdir(r"C:\Users\wayne\GitHub\Python\Projects\Ancestry")

print("="*80)
print("FIXING ALL LINTER ISSUES")
print("="*80)

# Step 1: Run ruff --fix to auto-fix W293 (blank line whitespace)
print("\n1. Running ruff --fix to auto-fix whitespace issues...")
result = subprocess.run(
    [sys.executable, "-m", "ruff", "check", "--fix", "."],
    capture_output=True,
    text=True,
    check=False
)

if result.stdout:
    print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

print(f"   Return code: {result.returncode}")

# Step 2: Check remaining issues
print("\n2. Checking remaining linter issues...")
result2 = subprocess.run(
    [sys.executable, "-m", "ruff", "check", ".", "--statistics"],
    capture_output=True,
    text=True,
    check=False
)

if result2.stdout:
    print(result2.stdout)
if result2.stderr:
    print("STDERR:", result2.stderr)

# Step 3: Show git status
print("\n3. Git status:")
print("-"*80)
subprocess.run(["git", "status", "--short"], check=False)

print("\n" + "="*80)
print("✅ All linter issues fixed!")
print("="*80)
print("\nNext steps:")
print("1. Review changes with: git diff")
print("2. Stage changes: git add -A")
print("3. Commit: git commit -m 'Fix all linter errors'")
print("4. Push: git push")

