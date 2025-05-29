#!/usr/bin/env python3
"""
Minimal test to isolate the import hanging issue
"""

import sys
import time


def test_individual_imports():
    """Test each import individually"""
    imports_to_test = [
        "logging_config",
        "config",
        "utils",
        "api_utils",
        "relationship_utils",
    ]

    for module_name in imports_to_test:
        print(f"Testing import of {module_name}...")
        start_time = time.time()

        try:
            exec(f"import {module_name}")
            elapsed = time.time() - start_time
            print(f"✓ {module_name} imported successfully in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"✗ {module_name} failed after {elapsed:.2f}s: {e}")

        # Add a small delay between imports
        time.sleep(0.5)


if __name__ == "__main__":
    print("=== Testing Individual Imports ===")
    test_individual_imports()
