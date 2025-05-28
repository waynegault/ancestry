#!/usr/bin/env python3
"""
Minimal test for availability flags that bypasses problematic imports
"""

print("Testing availability flags...")

# Test just the import part of action9_process_productive
try:
    # Test the GEDCOM utilities import
    from gedcom_utils import (
        calculate_match_score,
        _normalize_id,
        GedcomData,
    )

    print("✓ GEDCOM utilities imported successfully")
    GEDCOM_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"✗ GEDCOM utilities import failed: {e}")
    GEDCOM_UTILS_AVAILABLE = False

# Test the relationship utilities import
try:
    from relationship_utils import (
        fast_bidirectional_bfs,
        convert_gedcom_path_to_unified_format,
        format_relationship_path_unified,
    )

    print("✓ Relationship utilities imported successfully")
    RELATIONSHIP_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"✗ Relationship utilities import failed: {e}")
    RELATIONSHIP_UTILS_AVAILABLE = False

# Test specific functions from action11 and gedcom_search_utils
try:
    from gedcom_search_utils import get_gedcom_relationship_path
    from action11 import get_ancestry_relationship_path

    print("✓ Specific relationship functions imported successfully")
    RELATIONSHIP_FUNCS_AVAILABLE = True
except ImportError as e:
    print(f"✗ Specific relationship functions import failed: {e}")
    RELATIONSHIP_FUNCS_AVAILABLE = False

# Test API utilities import
try:
    from action11 import _process_and_score_suggestions

    print("✓ API utilities imported successfully")
    API_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"✗ API utilities import failed: {e}")
    API_UTILS_AVAILABLE = False

print("\n=== AVAILABILITY FLAGS TEST RESULTS ===")
print(f"GEDCOM_UTILS_AVAILABLE: {GEDCOM_UTILS_AVAILABLE}")
print(f"RELATIONSHIP_UTILS_AVAILABLE: {RELATIONSHIP_UTILS_AVAILABLE}")
print(f"RELATIONSHIP_FUNCS_AVAILABLE: {RELATIONSHIP_FUNCS_AVAILABLE}")
print(f"API_UTILS_AVAILABLE: {API_UTILS_AVAILABLE}")

if GEDCOM_UTILS_AVAILABLE and RELATIONSHIP_UTILS_AVAILABLE and API_UTILS_AVAILABLE:
    print("✓ All utilities are available - the fix should work!")
else:
    print("✗ Some utilities are missing - need to investigate further")

print("\nNow testing if our action9_process_productive fixes work...")

# Try to import just the flags from action9_process_productive
# But let's do it with a timeout mechanism to avoid hanging
import sys
import signal


def timeout_handler(signum, frame):
    raise TimeoutError("Import took too long")


# Set a timeout for the import (Unix only - won't work on Windows)
if sys.platform != "win32":
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)  # 10 second timeout

try:
    # Try to import the module
    print("Attempting to import action9_process_productive...")
    import action9_process_productive

    # Check the flags
    print(
        f"action9_process_productive.GEDCOM_UTILS_AVAILABLE: {action9_process_productive.GEDCOM_UTILS_AVAILABLE}"
    )
    print(
        f"action9_process_productive.RELATIONSHIP_UTILS_AVAILABLE: {action9_process_productive.RELATIONSHIP_UTILS_AVAILABLE}"
    )
    print(
        f"action9_process_productive.API_UTILS_AVAILABLE: {action9_process_productive.API_UTILS_AVAILABLE}"
    )

    if all(
        [
            action9_process_productive.GEDCOM_UTILS_AVAILABLE,
            action9_process_productive.RELATIONSHIP_UTILS_AVAILABLE,
            action9_process_productive.API_UTILS_AVAILABLE,
        ]
    ):
        print("🎉 SUCCESS! All availability flags are now True!")
    else:
        print("⚠️  Some flags are still False - our fix needs more work")

except TimeoutError:
    print("❌ Import timed out - there's likely a hanging import issue")
except Exception as e:
    print(f"❌ Import failed with error: {e}")
finally:
    if sys.platform != "win32":
        signal.alarm(0)  # Cancel the alarm

print("\nTest complete!")
