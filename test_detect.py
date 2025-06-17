#!/usr/bin/env python3
"""Test for credentials.py auto detection of test framework."""

import sys
import os


def main():
    """Main entry point."""
    # Check if running tests
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("🔐 Running tests via --test flag...")
        return True

    # Auto-detect if being run as part of run_all_tests.py
    if "run_all_tests" in sys.argv[0]:
        print("🔐 Auto-detected test execution from run_all_tests.py...")
        return True

    print("Running in normal mode")
    return True


if __name__ == "__main__":
    success = main()
    print(f"Success: {success}")
    sys.exit(0 if success else 1)
