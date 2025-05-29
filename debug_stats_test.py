#!/usr/bin/env python3

from api_cache import _api_cache_module
import json


def debug_statistics_test():
    """Debug the statistics collection test to identify the issue."""
    print("=" * 60)
    print("API CACHE STATISTICS COLLECTION DEBUG")
    print("=" * 60)

    try:
        # Get the stats
        stats = _api_cache_module.get_stats()
        print("✅ Statistics retrieved successfully")
        print("\nFull statistics data:")
        print(json.dumps(stats, indent=2, default=str))

        # Check required fields
        required_fields = [
            "module_name",
            "api_cache_expire",
            "db_cache_expire",
            "ai_cache_expire",
        ]
        print(f"\n📋 Checking required fields ({len(required_fields)} total):")

        for field in required_fields:
            present = field in stats
            status = "✅ PRESENT" if present else "❌ MISSING"
            value = stats.get(field, "NOT_FOUND")
            print(f"  {field}: {status} (value: {value})")

        # Calculate test result
        test_result = all(field in stats for field in required_fields)
        print(f"\n🏁 Test result: {'PASS' if test_result else 'FAIL'}")

        if not test_result:
            missing_fields = [f for f in required_fields if f not in stats]
            print(f"❌ Missing fields: {missing_fields}")
            print(f"📊 Available fields: {list(stats.keys())}")

        return test_result

    except Exception as e:
        print(f"❌ Error during statistics collection: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = debug_statistics_test()
    print(f"\nFinal result: {'SUCCESS' if result else 'FAILURE'}")
