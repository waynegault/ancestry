#!/usr/bin/env python3
"""
Test runner that captures the output from relationship_utils test suite.
"""

import subprocess
import sys
import os


def run_tests():
    """Run the relationship_utils test suite and capture output."""
    try:
        # Change to the correct directory
        os.chdir(r"c:\Users\wayne\GitHub\Python\Projects\Ancestry")

        # Run the test file
        result = subprocess.run(
            [sys.executable, "relationship_utils.py"],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )

        print("=== STDOUT ===")
        print(result.stdout)

        if result.stderr:
            print("\n=== STDERR ===")
            print(result.stderr)

        print(f"\n=== RETURN CODE: {result.returncode} ===")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("❌ Test suite timed out after 2 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
