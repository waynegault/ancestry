#!/usr/bin/env python3
"""
Test imports with minimal dependencies
"""

import sys
import os

print("Testing minimal imports...")

# Test if we can import the individual pieces
print("\n1. Testing basic Python imports...")
try:
    import logging
    import re
    import time
    from pathlib import Path

    print("✓ Basic Python imports working")
except Exception as e:
    print(f"✗ Basic Python imports failed: {e}")

print("\n2. Testing third-party imports...")
try:
    from ged4py.parser import GedcomReader

    print("✓ ged4py import working")
except Exception as e:
    print(f"✗ ged4py import failed: {e}")

print("\n3. Testing local config import...")
try:
    # Just test if config can be imported without triggering full validation
    import config

    print("✓ config module imported")
except Exception as e:
    print(f"✗ config import failed: {e}")

print("\n4. Testing relationship_utils import...")
try:
    # Try importing relationship_utils which should have fewer dependencies
    import relationship_utils

    print("✓ relationship_utils imported successfully")
except Exception as e:
    print(f"✗ relationship_utils import failed: {e}")

print("\n5. Testing specific functions from relationship_utils...")
try:
    from relationship_utils import fast_bidirectional_bfs

    print("✓ fast_bidirectional_bfs imported successfully")
except Exception as e:
    print(f"✗ fast_bidirectional_bfs import failed: {e}")

print("\nDone!")
