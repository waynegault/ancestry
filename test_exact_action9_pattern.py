# Test the exact import pattern that action9 uses
import sys
import time

print("Testing exact action9 import pattern...")

# Test the config import that action9 actually uses
print("1. Testing config_instance import...")
try:
    from config import config_instance

    print("   ✓ config_instance imported successfully")
except Exception as e:
    print(f"   ✗ config_instance import failed: {e}")

# Test all the imports that action9 does in the try block
print("2. Testing the exact gedcom_utils imports from action9...")
try:
    from gedcom_utils import (
        calculate_match_score,
        _normalize_id,
        GedcomData,
    )
    from relationship_utils import (
        fast_bidirectional_bfs,
        convert_gedcom_path_to_unified_format,
        format_relationship_path_unified,
    )

    print("   ✓ All primary imports successful")
    GEDCOM_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"   ✗ Primary imports failed: {e}")
    GEDCOM_UTILS_AVAILABLE = False

print("3. Testing relationship utilities imports...")
try:
    from gedcom_search_utils import get_gedcom_relationship_path
    from action11 import get_ancestry_relationship_path

    print("   ✓ Relationship utilities imported")
    RELATIONSHIP_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"   ✗ Relationship utilities failed: {e}")
    RELATIONSHIP_UTILS_AVAILABLE = False

print("4. Testing API utilities imports...")
try:
    from action11 import _process_and_score_suggestions

    print("   ✓ API utilities imported")
    API_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"   ✗ API utilities failed: {e}")
    API_UTILS_AVAILABLE = False

print("\nFlag Results:")
print(f"GEDCOM_UTILS_AVAILABLE: {GEDCOM_UTILS_AVAILABLE}")
print(f"RELATIONSHIP_UTILS_AVAILABLE: {RELATIONSHIP_UTILS_AVAILABLE}")
print(f"API_UTILS_AVAILABLE: {API_UTILS_AVAILABLE}")

print("\nNow testing action9 import directly...")
try:
    import action9_process_productive

    print("   ✓ action9_process_productive imported successfully!")
    print(
        f"   action9.GEDCOM_UTILS_AVAILABLE: {action9_process_productive.GEDCOM_UTILS_AVAILABLE}"
    )
    print(
        f"   action9.RELATIONSHIP_UTILS_AVAILABLE: {action9_process_productive.RELATIONSHIP_UTILS_AVAILABLE}"
    )
    print(
        f"   action9.API_UTILS_AVAILABLE: {action9_process_productive.API_UTILS_AVAILABLE}"
    )
except Exception as e:
    print(f"   ✗ action9_process_productive import failed: {e}")

print("Test completed!")
