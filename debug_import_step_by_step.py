# Test imports step by step to identify the hanging component
import sys
import time

print("Starting import test...")

# Test 1: Basic config
print("1. Testing config import...")
try:
    from config import get_config

    print("   ✓ Config imported successfully")
except Exception as e:
    print(f"   ✗ Config import failed: {e}")

# Test 2: Individual gedcom_utils components
print("2. Testing individual gedcom_utils imports...")
try:
    from gedcom_utils import calculate_match_score

    print("   ✓ calculate_match_score imported")
except Exception as e:
    print(f"   ✗ calculate_match_score failed: {e}")

try:
    from gedcom_utils import _normalize_id

    print("   ✓ _normalize_id imported")
except Exception as e:
    print(f"   ✗ _normalize_id failed: {e}")

try:
    from gedcom_utils import GedcomData

    print("   ✓ GedcomData imported")
except Exception as e:
    print(f"   ✗ GedcomData failed: {e}")

# Test 3: Individual relationship_utils components
print("3. Testing individual relationship_utils imports...")
try:
    from relationship_utils import fast_bidirectional_bfs

    print("   ✓ fast_bidirectional_bfs imported")
except Exception as e:
    print(f"   ✗ fast_bidirectional_bfs failed: {e}")

try:
    from relationship_utils import convert_gedcom_path_to_unified_format

    print("   ✓ convert_gedcom_path_to_unified_format imported")
except Exception as e:
    print(f"   ✗ convert_gedcom_path_to_unified_format failed: {e}")

try:
    from relationship_utils import format_relationship_path_unified

    print("   ✓ format_relationship_path_unified imported")
except Exception as e:
    print(f"   ✗ format_relationship_path_unified failed: {e}")

print("4. Testing action9 import with limited scope...")
try:
    # Import just the beginning of action9 to see if that works
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "action9", "action9_process_productive.py"
    )
    action9 = importlib.util.module_from_spec(spec)

    # Try to execute just the module loading without full execution
    print("   Module spec created successfully")

except Exception as e:
    print(f"   ✗ action9 module loading failed: {e}")

print("Test completed!")
