#!/usr/bin/env python3
"""Debug subprocess execution for action10.py"""

import subprocess
import sys
import os
from pathlib import Path


def test_subprocess():
    """Test subprocess execution of action10.py"""
    current_dir = Path(__file__).parent
    print(f"Current directory: {current_dir}")
    print(f"Looking for: {current_dir / 'action10.py'}")
    print(f"File exists: {(current_dir / 'action10.py').exists()}")

    try:
        cmd = [sys.executable, "action10.py"]
        print(f"Running command: {' '.join(cmd)}")
        print(f"Working directory: {current_dir}")

        # Run with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=current_dir,
        )

        print(f"Return code: {result.returncode}")
        print(f"STDOUT length: {len(result.stdout)}")
        print(f"STDERR length: {len(result.stderr)}")

        if result.stdout:
            print("=== STDOUT (first 500 chars) ===")
            print(result.stdout[:500])

        if result.stderr:
            print("=== STDERR (first 500 chars) ===")
            print(result.stderr[:500])

    except subprocess.TimeoutExpired as e:
        print(f"TIMEOUT: {e}")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    test_subprocess()
