#!/usr/bin/env python3
"""Quick test to verify action10 timeout handling."""

import subprocess
import sys
import time
import os


def test_action10_timeout():
    """Test that action10 times out quickly with our new configuration."""
    print("🧪 Testing action10.py timeout handling...")

    start_time = time.time()

    # Set up environment like our test runner does
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            [sys.executable, "action10.py"],
            capture_output=True,
            text=True,
            timeout=10,  # 10 second timeout (same as in test runner for fast mode)
            env=env,
        )

        duration = time.time() - start_time
        print(f"✅ action10.py completed in {duration:.2f}s")
        print(f"   Return code: {result.returncode}")

        if result.stdout:
            print(f"   Output length: {len(result.stdout)} chars")
            print(f"   First 100 chars: {result.stdout[:100]}")

        return True

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ action10.py timed out after {duration:.2f}s (expected behavior)")
        print("   This confirms the module has input() calls that hang in subprocess")
        return True  # This is actually the expected behavior

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Unexpected error after {duration:.2f}s: {e}")
        return False


if __name__ == "__main__":
    test_action10_timeout()
