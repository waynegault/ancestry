#!/usr/bin/env python3
"""Test script to verify action10.py subprocess handling."""

import subprocess
import sys
import os


def test_action10_subprocess():
    """Test action10.py with enhanced subprocess handling."""
    print("Testing action10.py specifically with enhanced subprocess handling...")

    # Set up environment
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    # Run with timeout
    try:
        result = subprocess.run(
            [sys.executable, "action10.py"],
            capture_output=True,
            text=True,
            timeout=15,  # 15 second timeout
            env=env,
            cwd=os.getcwd(),
        )

        print(f"Return code: {result.returncode}")
        print(f"Stdout length: {len(result.stdout)} chars")
        print(f"Stderr length: {len(result.stderr)} chars")

        if result.stdout:
            print("First 200 chars of stdout:")
            print(repr(result.stdout[:200]))

        if result.stderr:
            print("Stderr:")
            print(result.stderr[:500])  # First 500 chars

        if result.returncode == 0:
            print("✅ action10.py completed successfully with subprocess!")
        else:
            print(
                f"⚠️ action10.py completed with non-zero exit code: {result.returncode}"
            )

    except subprocess.TimeoutExpired:
        print("❌ action10.py timed out - still hanging")
        return False
    except Exception as e:
        print(f"❌ Error running action10.py: {e}")
        return False

    return True


if __name__ == "__main__":
    test_action10_subprocess()
