#!/usr/bin/env python3
"""
Test script to check the remaining 3 issues identified in database.py
"""

import database

print("Testing database functionality...")

# Test 1: Transaction context manager
print("\n=== Test 1: SessionManager Import ===")
try:
    from database import SessionManager

    print("✓ SessionManager import successful")
except ImportError as e:
    print(f"✗ SessionManager import failed: {e}")

# Test 2: DnaMatch model uuid attribute
print("\n=== Test 2: DnaMatch UUID Attribute ===")
try:
    match = database.DnaMatch()
    if hasattr(match, "uuid"):
        print("✓ DnaMatch has uuid attribute")
    else:
        print("✗ DnaMatch missing uuid attribute")
        # Let's check what attributes it actually has
        attrs = [attr for attr in dir(match) if not attr.startswith("_")]
        print(f"  Available attributes: {attrs[:10]}...")  # Show first 10
except Exception as e:
    print(f"✗ DnaMatch test failed: {e}")

# Test 3: FamilyTree model uuid attribute
print("\n=== Test 3: FamilyTree UUID Attribute ===")
try:
    tree = database.FamilyTree()
    if hasattr(tree, "uuid"):
        print("✓ FamilyTree has uuid attribute")
    else:
        print("✗ FamilyTree missing uuid attribute")
        # Let's check what attributes it actually has
        attrs = [attr for attr in dir(tree) if not attr.startswith("_")]
        print(f"  Available attributes: {attrs[:10]}...")  # Show first 10
except Exception as e:
    print(f"✗ FamilyTree test failed: {e}")

print("\n=== Test Summary Complete ===")
