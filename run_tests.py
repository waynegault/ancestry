#!/usr/bin/env python3
"""
Simple test runner for gedcom_search_utils
"""

import gedcom_search_utils


def main():
    print("Running GEDCOM Search Utils Test Suite...")
    print("=" * 60)

    try:
        # Run the existing self-tests
        gedcom_search_utils.run_self_tests()
        print("\n" + "=" * 60)
        print("Test suite completed successfully!")

    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
