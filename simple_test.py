#!/usr/bin/env python3
"""
Simple direct test for api_search_utils
"""

import sys


def main():
    print("Testing api_search_utils import...")

    try:
        import api_search_utils

        print("✓ Import successful")

        print("\nRunning self_test...")
        result = api_search_utils.self_test()
        print(f"\nTest result: {'SUCCESS' if result else 'FAILURE'}")

        return result

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    print(f"Final result: {success}")
    sys.exit(0 if success else 1)
