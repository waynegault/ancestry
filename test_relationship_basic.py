#!/usr/bin/env python3

"""Quick test to check if relationship_utils works without config errors"""

print("=== Testing relationship_utils without config dependency ===")

try:
    print("1. Testing import...")
    import relationship_utils

    print("   ✅ Import successful")

    print("2. Testing format_name function...")
    result = relationship_utils.format_name("John /Smith/")
    print(f"   ✅ format_name('John /Smith/') = '{result}'")

    print("3. Testing _normalize_id function...")
    result = relationship_utils._normalize_id("@I123@")
    print(f"   ✅ _normalize_id('@I123@') = '{result}'")

    print("4. Testing fast_bidirectional_bfs with empty data...")
    result = relationship_utils.fast_bidirectional_bfs("test1", "test2", {}, {})
    print(f"   ✅ fast_bidirectional_bfs with empty data returns: {result}")

    print("\n✅ All basic tests passed - no config dependency issues!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()

print("\n=== Test complete ===")
