"""Quick test of unified cache manager."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.unified_cache_manager import get_unified_cache

cache = get_unified_cache()

# Test basic operations
cache.set("ancestry", "combined", "key1", {"data": "test"})
result = cache.get("ancestry", "combined", "key1")

if result == {"data": "test"}:
    print("✅ SUCCESS: Cache working!")
else:
    print("❌ FAILED: Cache test")
    sys.exit(1)

# Check stats
stats = cache.get_stats()
hit_rate = stats["global"]["hit_rate_percent"]
print(f"✅ Hit rate: {hit_rate:.1f}%")
print(f"✅ Cache size: {len(cache)} entries")

print("\n✅ All tests passed!")
