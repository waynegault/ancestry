#!/usr/bin/env python3

"""
test_action6_cache_integration.py - Verify action6_gather.py integrates with UnifiedCacheManager

Quick integration test to verify:
1. action6_gather imports successfully
2. Cache functions use UnifiedCacheManager
3. No references to old global_cache
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_action6_imports():
    """Test that action6_gather imports without errors."""
    try:
        import action6_gather
        assert hasattr(action6_gather, '_get_cached_profile')
        assert hasattr(action6_gather, '_cache_profile')
        assert hasattr(action6_gather, '_check_combined_details_cache')
        assert hasattr(action6_gather, '_cache_combined_details')
        print("✅ action6_gather imported successfully")
        print("✅ All cache functions exist")
        return True
    except Exception as e:
        print(f"❌ Failed to import action6_gather: {e}")
        return False

def test_cache_functions_work():
    """Test cache functions use UnifiedCacheManager."""
    try:
        from action6_gather import _cache_profile, _get_cached_profile
        from core.unified_cache_manager import get_unified_cache

        get_unified_cache()
        print("✅ UnifiedCacheManager initialized")

        # Test profile caching
        test_profile_id = "TEST_PROFILE_123"
        test_profile_data = {"last_logged_in_dt": None, "contactable": True}

        _cache_profile(test_profile_id, test_profile_data)
        print("✅ _cache_profile executed successfully")

        retrieved = _get_cached_profile(test_profile_id)
        assert retrieved is not None, "Cache lookup returned None"
        assert retrieved["contactable"], "Cache data corrupted"
        print("✅ _get_cached_profile retrieved data successfully")

        return True
    except Exception as e:
        print(f"❌ Cache function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_no_global_cache_references():
    """Verify no references to old global_cache in action6_gather."""
    try:
        from pathlib import Path
        with Path('action6_gather.py').open() as f:
            content = f.read()

        # Check for old global_cache references
        if 'global_cache' in content:
            print("❌ Found old 'global_cache' references in action6_gather.py")
            # Show first few occurrences
            lines = content.split('\n')
            count = 0
            for i, line in enumerate(lines, 1):
                if 'global_cache' in line and 'get_unified_cache' not in line:
                    print(f"  Line {i}: {line.strip()}")
                    count += 1
                    if count >= 3:
                        break
            return False

        print("✅ No global_cache references found (successfully migrated)")
        return True
    except Exception as e:
        print(f"❌ Failed to verify references: {e}")
        return False

def main():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("🧪 Action6 Cache Integration Tests")
    print("="*60 + "\n")

    tests = [
        ("Import Test", test_action6_imports),
        ("Cache Functions Test", test_cache_functions_work),
        ("Reference Check", test_no_global_cache_references),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        result = test_func()
        results.append(result)
        print()

    print("="*60)
    if all(results):
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("="*60)
        return 0
    print("❌ SOME TESTS FAILED")
    print("="*60)
    return 1

if __name__ == "__main__":
    sys.exit(main())
