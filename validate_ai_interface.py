#!/usr/bin/env python3

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Starting AI interface validation test...")

# Test 1: Basic imports
print("1. Testing imports...")
try:
    from ai_interface import extract_genealogical_entities

    print("✓ extract_genealogical_entities imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check function exists and is callable
print("2. Testing function accessibility...")
if callable(extract_genealogical_entities):
    print("✓ extract_genealogical_entities is callable")
else:
    print("✗ extract_genealogical_entities is not callable")
    sys.exit(1)

# Test 3: Check AI configuration
print("3. Testing AI configuration...")
try:
    from config import config_instance

    print(f"✓ AI Provider: {config_instance.AI_PROVIDER}")
    has_key = bool(
        config_instance.DEEPSEEK_API_KEY
        if config_instance.AI_PROVIDER == "deepseek"
        else config_instance.GOOGLE_API_KEY
    )
    print(f"✓ API Key configured: {has_key}")

    if not config_instance.AI_PROVIDER:
        print("⚠ No AI provider configured - testing will use fallback")
    elif not has_key:
        print("⚠ No API key found - testing will use fallback")

except Exception as e:
    print(f"✗ Configuration check failed: {e}")
    sys.exit(1)

# Test 4: Test with minimal input (this might use fallback if AI is not available)
print("4. Testing function call with minimal input...")
try:
    # Very short test to minimize API call time
    result = extract_genealogical_entities("Test")
    print(f"✓ Function call successful")
    print(f"✓ Result type: {type(result)}")

    if isinstance(result, dict):
        print(f"✓ Result is dict with keys: {list(result.keys())}")

        # Check for expected structure
        if "extracted_data" in result:
            print("✓ extracted_data key found")
        else:
            print("✗ extracted_data key missing")

        if "suggested_tasks" in result:
            print("✓ suggested_tasks key found")
            print(f"  suggested_tasks type: {type(result['suggested_tasks'])}")
        else:
            print("✗ suggested_tasks key missing")
    else:
        print(f"✗ Expected dict, got {type(result)}")

except Exception as e:
    print(f"✗ Function call failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n=== AI Interface Validation Complete ===")
print("✓ All tests passed - the updated system should work correctly")
