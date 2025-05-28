#!/usr/bin/env python3
"""
Sequential import test to identify hanging modules
"""

import sys
import time


def test_import(module_name, import_statement):
    """Test an import with a simple timer"""
    print(f"Testing import: {module_name}...")
    start_time = time.time()

    try:
        exec(import_statement)
        elapsed = time.time() - start_time
        print(f"✓ {module_name} imported successfully in {elapsed:.2f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ {module_name} failed in {elapsed:.2f}s: {e}")
        return False


print("Sequential Import Test")
print("=" * 40)

# Test basic imports first
test_import("json", "import json")
test_import("pathlib", "from pathlib import Path")
test_import("logging", "import logging")

# Test our local imports
test_import("logging_config", "from logging_config import logger")
test_import("config", "from config import config_instance")

# If config works, test the modules that depend on it
print("\nTesting modules that depend on config...")
test_import("utils", "from utils import format_name")
test_import("database", "from database import Person")

# Test the problematic modules
print("\nTesting potentially problematic modules...")
test_import("gedcom_utils (partial)", "from gedcom_utils import _normalize_id")

print("\nTest complete!")
