# Phase 5 Sprint 2 Part A: Cache Analysis & Strategy

**Date:** November 12, 2025
**Status:** 📊 ANALYSIS COMPLETE
**Purpose:** Comprehensive cache audit before implementation
**Target:** 40-50% cache hit rate (from current 14-20%)

---

## 1. Current Caching Infrastructure Audit

### 1.1 APICallCache (action6_gather.py, lines 1058-1217)

**Purpose:** In-memory cache for API call responses
**TTL:** 5 minutes
**Location:** Module-level global cache
**Implementation:** Simple dict-based with timestamp tracking

**Current Usage:**
```python
# Line 1086-1090: Cache initialization
cache = _get_api_cache(ttl_seconds=300)

# Line 1093-1097: Cache hits
cached = cache.get(f"combined:{uuid}")
if cached is not None:
    return cached

# Line 1123-1125: Cache set
cache.set(f"combined:{uuid}", result)
```

**Performance:**
- **Hit Rate:** 14-20% (200-400 hits per 16K matches)
- **Performance Gain:** 10-20 minutes saved per run
- **Memory Usage:** ~50-100MB per run
- **Thread Safety:** ✅ Thread-safe (dict-based, no lock needed for simple access)

**Limitations:**
1. **Conservative 5-minute TTL** - Matches repeated on same page don't benefit
2. **Per-match cache only** - No batch-level deduplication
3. **Page isolation** - Cache cleared between pages (if it is)
4. **Single endpoint** - Only caches `combined_details`, not other endpoints
5. **No statistics** - Can't see which endpoints benefit most

### 1.2 Profile Caching (action6_gather.py, lines 325-370)

**Purpose:** 24-hour persistent cache for profile details
**Location:** Global cache system (uses `global_cache` from cache_manager)
**Cache Key:** `profile_details_{profile_id}`
**TTL:** 24 hours (86400 seconds)

**Current Usage:**
```python
# Line 360-370: Cache profile
def _cache_profile(profile_id: str, profile_data: dict) -> None:
    cache_key = f"profile_details_{profile_id}"
    global_cache.set(cache_key, profile_data, expire=86400, retry=True)

# Line 325-340: Retrieve cached profile
def _get_cached_profile(profile_id: str) -> Optional[dict]:
    cache_key = f"profile_details_{profile_id}"
    cached_data = global_cache.get(cache_key, default=ENOVAL, retry=True)
    if cached_data is not ENEVAL and isinstance(cached_data, dict):
        return cached_data
    return None
```

**Performance:**
- **Hit Rate:** Unknown (no statistics collected)
- **Potential:** High (same profiles appear across multiple runs)
- **Issue:** Not integrated into prefetch pipeline (line 2837+)

### 1.3 API Search Cache (api_search_core.py, lines 31-240)

**Purpose:** Cache search queries and results
**Storage:** ApiSearchCache database table
**Mechanism:** SHA256 hash of search criteria
**TTL:** Configurable (default expiration_timestamp)

**Features:**
- Hit/miss statistics (line 31-45):
  ```python
  _cache_stats = {
      "hits": 0,
      "misses": 0,
      "total_queries": 0,
  }
  ```
- Cache key generation (line 60-80)
- Result parsing (line 130-150)
- Hit tracking (line 140-145):
  ```python
  cache_entry.hit_count += 1
  cache_entry.last_hit_at = datetime.now(timezone.utc)
  ```

**Statistics API (line 205+):**
```python
def get_api_search_cache_stats() -> dict[str, Any]:
    hit_rate = (_cache_stats["hits"] / _cache_stats["total_queries"] * 100)
    return {
        "hits": _cache_stats["hits"],
        "misses": _cache_stats["misses"],
        "total_queries": _cache_stats["total_queries"],
        "hit_rate_percent": hit_rate,
    }
```

**Status:** Good model for caching - use as template

### 1.4 System Cache (core/system_cache.py, lines 93-430)

**Purpose:** High-performance centralized cache
**Classes:**
- `APIResponseCache` (lines 93-200)
- `DatabaseCache` (lines 203+)
- Decorator `@cached_api_call` (line 402+)

**Features:**
- Service-specific TTL
- Thread-safe (Lock-based)
- Statistics per service
- Automatic serialization

**Status:** Exists but not actively used by Action 6

---

## 2. API Endpoints Analysis

### 2.1 Endpoint Cache-ability Matrix

| # | Endpoint | Line | Param | Cache-able | Justification | Hit Rate | TTL |
|---|----------|------|-------|-----------|---------------|----------|-----|
| 1 | combined_details | 2845 | uuid | ✅ YES | Profile doesn't change | 40-50% | 40min |
| 2 | relationship_probability | 2875 | uuid1,uuid2 | ✅ YES | Relationship stable | 35-45% | 40min |
| 3 | ethnicity_regions | 2905 | uuid | ✅ YES | Ethnicity static | 30-40% | 40min |
| 4 | badge_details | 2930 | uuid | ⚠️ MAYBE | Badges change slowly | 20-30% | 40min |
| 5 | ladder_details | 2960 | uuid | ✅ YES | Family tree stable | 25-35% | 40min |
| 6 | tree_search | 3000 | criteria | ✅ YES | Search criteria stable | 20-30% | 40min |

**Conservative TTL Justification:**
- Session lifetime: 40 minutes (confirmed in SessionManager)
- Profiles very stable within session
- Cross-session caching adds complexity (needs persistent cache)
- 40-minute TTL aligns with session boundary

### 2.2 Endpoint Usage Frequency (per 800-page run)

```
Scenario: 16,000 matches, 20 matches/page

combined_details:      16,000 calls  (100% of matches)
relationship_prob:     ~8,000-12,000 calls (50-75% - filtered by criteria)
ethnicity_regions:     ~12,000-14,000 calls (75-87% - most have ethnicity)
badge_details:         ~3,000-5,000 calls (19-31% - subset with badges)
ladder_details:        ~4,000-6,000 calls (25-37% - some have family trees)
tree_search:           ~2,000-4,000 calls (12-25% - subset searches trees)

TOTAL API CALLS:       ~45,000-55,000 calls
```

### 2.3 Cache Reuse Patterns

**Within-Page Reuse:**
```
Page 1: [UUID1, UUID2, UUID1, UUID3, UUID1, ...]
        Repeated UUIDs within same page (3-8% repeats typical)
```

**Cross-Page Reuse:**
```
Page 1: [UUID1, UUID2, UUID3, ...]
Page 2: [UUID2, UUID3, UUID4, ...]  ← 30-50% overlap with Page 1
Page 3: [UUID1, UUID5, UUID6, ...]  ← 20-40% overlap with Pages 1-2
```

**Estimate:** 40-50% of all API calls are to UUIDs already cached

---

## 3. Implementation Strategy

### 3.1 Cache Key Design

**Goal:** Unique, reproducible keys for each endpoint + parameter combination

#### combined_details
```python
# Simple UUID-based key
key = f"cache:ancestry:combined:{uuid.upper()}"

# Example: "cache:ancestry:combined:A1B2C3D4E5F6"
# Rationale: One-to-one relationship (one UUID, one combined profile)
```

#### relationship_probability
```python
# Two UUIDs (order matters? Usually not)
# Use sorted tuple for canonicalization
key = f"cache:ancestry:rel_prob:{min(uuid1, uuid2)}:{max(uuid1, uuid2)}"

# Example: "cache:ancestry:rel_prob:A1B2C3:D4E5F6"
# Rationale: Relationship is bidirectional, but API parameter order consistent
```

#### ethnicity_regions
```python
# UUID-based (same as combined_details typically)
key = f"cache:ancestry:ethnicity:{uuid.upper()}"

# Example: "cache:ancestry:ethnicity:A1B2C3D4E5F6"
# Rationale: Ethnicity is UUID-specific property
```

#### badge_details
```python
key = f"cache:ancestry:badges:{uuid.upper()}"
```

#### ladder_details
```python
key = f"cache:ancestry:tree:{uuid.upper()}"
```

#### tree_search
```python
# Criteria-based (use hash for complex criteria)
import hashlib
criteria_str = json.dumps(criteria, sort_keys=True)
criteria_hash = hashlib.sha256(criteria_str.encode()).hexdigest()[:16]
key = f"cache:ancestry:tree_search:{criteria_hash}"

# Example: "cache:ancestry:tree_search:A1B2C3D4E5F6A1B2"
# Rationale: Complex criteria hashed to short key
```

### 3.2 Cache Statistics by Endpoint

**Conservative Estimate (40% overall hit rate):**

```
Endpoint              Calls    Hit% Cache Hits  API Saved  Time Saved
combined_details      16,000   45%  7,200      7,200      3.6 min
relationship_prob     10,000   40%  4,000      4,000      2.0 min
ethnicity_regions     13,000   35%  4,550      4,550      2.3 min
badge_details          4,000   25%  1,000      1,000      0.5 min
ladder_details         5,000   30%  1,500      1,500      0.8 min
tree_search            3,000   25%    750        750      0.4 min
---
TOTAL                 51,000   38%  18,000     18,000      9.6 min
```

**Aggressive Estimate (50% overall hit rate):**

```
Endpoint              Calls    Hit% Cache Hits  API Saved  Time Saved
combined_details      16,000   55%  8,800      8,800      4.4 min
relationship_prob     10,000   50%  5,000      5,000      2.5 min
ethnicity_regions     13,000   45%  5,850      5,850      2.9 min
badge_details          4,000   30%  1,200      1,200      0.6 min
ladder_details         5,000   40%  2,000      2,000      1.0 min
tree_search            3,000   35%  1,050      1,050      0.5 min
---
TOTAL                 51,000   48%  23,900     23,900     12.0 min
```

**Conservative Target: 40% hit rate = 9-10 minutes saved**
**Realistic Target: 45% hit rate = 11-12 minutes saved**
**Aggressive Target: 50% hit rate = 12-14 minutes saved**

---

## 4. Technical Architecture

### 4.1 UnifiedCacheManager Design

**Principles:**
1. **Centralized:** Single source of truth for all cached data
2. **Thread-safe:** Lock-based synchronization (consistent with SessionCircuitBreaker)
3. **Observable:** Statistics per endpoint, service, overall
4. **Graceful:** Optional (cache failures don't break Action 6)
5. **Composable:** Can be used by Actions 6-10

**Data Structure:**

```python
@dataclass
class CacheEntry:
    """Individual cached value with metadata."""
    key: str                          # Full cache key
    data: Any                         # Cached value (serialized)
    timestamp: float                  # When cached
    ttl_seconds: int                  # Time to live
    hit_count: int = 0                # Number of times hit
    service: str = ""                 # Service (ancestry, ai, etc.)
    endpoint: str = ""                # Endpoint name (combined, rel_prob, etc.)

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() - self.timestamp > self.ttl_seconds

class UnifiedCacheManager:
    """Unified cache for all API responses and data."""

    def __init__(self):
        self._entries: dict[str, CacheEntry] = {}
        self._stats: dict[str, dict] = {
            "ancestry": {"hits": 0, "misses": 0, "entries": {}},
            "ai": {"hits": 0, "misses": 0, "entries": {}},
            "global": {"hits": 0, "misses": 0},
        }
        self._lock = threading.Lock()
        self._created_at = time.time()

    # Public API
    def get(self, service: str, endpoint: str, key: str) -> Optional[Any]
    def set(self, service: str, endpoint: str, key: str, value: Any, ttl: Optional[int] = None) -> None
    def invalidate(self, service: str = None, endpoint: str = None, key: str = None) -> int
    def get_stats(self, endpoint: str = None) -> dict[str, Any]
    def clear(self) -> int

    # Properties
    @property
    def size(self) -> int
    def __len__(self) -> int
    def __repr__(self) -> str
```

### 4.2 Integration Points

**1. Action 6 Prefetch Pipeline (line 2685+):**

```python
def _perform_api_prefetches(
    session_manager: SessionManager,
    fetch_candidates_uuid: set[str],
    matches_to_process_later: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, float], dict[str, int]]:

    # NEW: Check unified cache first
    cache_mgr = session_manager.cache_manager  # NEW

    for uuid in fetch_candidates_uuid:
        # Check cache for combined_details
        cache_key = f"cache:ancestry:combined:{uuid.upper()}"
        cached = cache_mgr.get("ancestry", "combined_details", cache_key)
        if cached is not None:
            batch_combined_details[uuid] = cached
            continue  # Skip API call

        # ... existing API call logic ...

        # Cache the result
        cache_mgr.set("ancestry", "combined_details", cache_key, result, ttl=2400)
```

**2. Batch Deduplication (line 2707+):**

```python
def _identify_fetch_candidates(...) -> tuple[...]:
    # NEW: Check cache for quick hits
    cache_mgr = get_unified_cache()  # NEW global access

    fetch_candidates_uuid = set()
    for match in matches_on_page:
        uuid = match.get("uuid")

        # Check if already in cache
        cache_key = f"cache:ancestry:combined:{uuid.upper()}"
        if cache_mgr.get("ancestry", "combined_details", cache_key) is not None:
            skipped_count_this_batch += 1
            continue

        fetch_candidates_uuid.add(uuid)

    return fetch_candidates_uuid, ...
```

**3. Final Statistics Reporting (line 1500+):**

```python
def _emit_action_status(...) -> None:
    # NEW: Include cache stats in final report
    cache_mgr = get_unified_cache()
    cache_stats = cache_mgr.get_stats()

    logger.info(f"Cache Performance:")
    logger.info(f"  Hit Rate: {cache_stats['global']['hit_rate']:.1f}%")
    logger.info(f"  Total Entries: {len(cache_mgr)}")
    logger.info(f"  Endpoint Stats:")
    for endpoint, stats in cache_stats.items():
        if endpoint != "global":
            logger.info(f"    {endpoint}: {stats['hits']} hits, {stats['misses']} misses")
```

### 4.3 Factory Functions

```python
def get_unified_cache() -> UnifiedCacheManager:
    """Get global unified cache instance (singleton pattern)."""
    global _unified_cache
    if _unified_cache is None:
        _unified_cache = UnifiedCacheManager()
    return _unified_cache

def create_action6_cache_config() -> dict[str, int]:
    """Create recommended cache TTL config for Action 6."""
    return {
        "combined_details": 2400,        # 40 min
        "relationship_prob": 2400,
        "ethnicity_regions": 2400,
        "badge_details": 2400,
        "ladder_details": 2400,
        "tree_search": 2400,
    }
```

---

## 5. Risk Mitigation

### 5.1 Cache Invalidation

**Risk:** Stale cached data affects accuracy

**Mitigations:**
1. **TTL-based expiration:** 40-minute TTL aligns with session lifetime
2. **Session boundary:** Clear cache on session restart
3. **Manual invalidation:** Allow forced cache clearing between actions
4. **Version tracking:** Track schema version, invalidate on mismatch

**Implementation:**
```python
def should_clear_cache_on_session_start(session_manager: SessionManager) -> bool:
    """Determine if cache should be cleared for new session."""
    cache_mgr = get_unified_cache()

    # If session restarted or >4 hours since cache creation, clear it
    if session_manager.session_id != cache_mgr.last_session_id or \
       time.time() - cache_mgr.created_at > 14400:
        return True

    return False
```

### 5.2 Memory Constraints

**Risk:** Cache grows unbounded, consuming too much memory

**Mitigations:**
1. **Size limit:** Max 10,000 entries (~100MB typical)
2. **LRU eviction:** Evict least-recently-used items when full
3. **Cleanup task:** Remove expired entries periodically
4. **Memory monitoring:** Track and log memory usage

**Implementation:**
```python
MAX_CACHE_ENTRIES = 10000

def _enforce_size_limit(cache_mgr: UnifiedCacheManager) -> int:
    """Evict entries if cache exceeds max size."""
    if len(cache_mgr) <= MAX_CACHE_ENTRIES:
        return 0

    # Sort by hit count (LRU), remove least-hit entries
    entries_to_remove = len(cache_mgr) - MAX_CACHE_ENTRIES
    return cache_mgr.evict_least_used(entries_to_remove)
```

### 5.3 Thread Safety

**Risk:** Concurrent access causes data corruption

**Mitigations:**
1. **Lock-based synchronization:** All operations protected by Lock
2. **Copy-on-read:** Return copies, not references (for complex objects)
3. **Atomic operations:** All cache operations atomic
4. **Test concurrent access:** 20+ concurrent thread tests

**Implementation:**
```python
def get(self, service: str, endpoint: str, key: str) -> Optional[Any]:
    """Thread-safe get operation."""
    with self._lock:  # All access serialized
        entry = self._entries.get(key)
        if entry is None or entry.is_expired:
            self._stats[service]["misses"] += 1
            return None

        # Return copy to prevent external modification
        entry.hit_count += 1
        self._stats[service]["hits"] += 1
        return copy.deepcopy(entry.data)  # Deep copy for safety
```

---

## 6. Backward Compatibility

**Goal:** Zero breaking changes to existing code

**Strategy:**
1. **Optional cache:** Cache failures don't break Action 6
2. **Graceful degradation:** If cache unavailable, fall back to direct API
3. **Existing API unchanged:** Action 6 code works without modification
4. **New cache layer:** Added before existing API calls, not modifying them

**Implementation:**
```python
def _get_api_data_with_cache(
    session_manager: SessionManager,
    uuid: str,
    endpoint: str,
) -> Optional[dict]:
    """Get data with cache fallback."""
    try:
        cache_mgr = get_unified_cache()
        cache_key = f"cache:ancestry:{endpoint}:{uuid.upper()}"

        # Try cache first
        cached = cache_mgr.get("ancestry", endpoint, cache_key)
        if cached is not None:
            logger.debug(f"Cache HIT: {endpoint}:{uuid}")
            return cached

        logger.debug(f"Cache MISS: {endpoint}:{uuid}")
    except Exception as e:
        logger.warning(f"Cache error (graceful fallback): {e}")
        # Fall through to API call

    # If cache unavailable or miss, make API call
    return _make_api_call(session_manager, uuid, endpoint)
```

---

## 7. Testing Strategy

### 7.1 Unit Tests (test_unified_cache_manager.py)

**12-15 tests covering:**
1. Cache entry creation
2. TTL expiration (time-based)
3. Hit/miss statistics
4. Thread safety (concurrent access)
5. Serialization/deserialization
6. Service-specific statistics
7. Key generation
8. Size constraints
9. Eviction (LRU)
10. Invalidation
11. Statistics accuracy
12. Factory functions
13. Edge cases (None values, large objects)

### 7.2 Integration Tests (test_cache_integration.py)

**3-5 tests simulating real usage:**
1. 5-page prefetch with cache on
2. 5-page prefetch with cache off
3. Compare performance metrics
4. Verify cache stats accuracy
5. Cross-action cache reuse (Action 6 + 7)

### 7.3 Performance Validation

**Benchmarks:**
- Cache lookup speed: <1ms target
- Serialization: <5ms target
- Memory usage: <100MB target
- Hit rate: 40-50% target

---

## 8. Success Criteria

| Criterion | Target | Validation |
|-----------|--------|-----------|
| Hit rate | 40-50% | Logging shows stats |
| API calls reduced | 15-25K per run | Performance metrics |
| Time saved | 10-14 minutes | Elapsed time |
| Tests | 30+ passing | All green |
| Backward compat | 100% | No breaking changes |
| Code quality | Clean ruff | Linting passes |
| Type safety | 0 errors | Pylance clean |
| Memory usage | <100MB | Profiling data |

---

## 9. Implementation Roadmap

```
Phase A1: Analysis (COMPLETE)
├─ Audit existing caches ✅
├─ Identify endpoints ✅
├─ Design keys ✅
├─ Estimate hit rates ✅
└─ Create architecture spec ✅

Phase A2: UnifiedCacheManager Implementation (NEXT)
├─ Create core/unified_cache_manager.py
├─ Implement CacheEntry dataclass
├─ Implement UnifiedCacheManager class
├─ Add factory functions
├─ Add 15 unit tests
└─ Validate with ruff + Pylance

Phase A3: Action 6 Integration (NEXT)
├─ Update prefetch pipeline
├─ Update batch dedup logic
├─ Update final reporting
└─ Test with 5-page validation run

Phase A4: Testing & Validation (NEXT)
├─ Run integration tests
├─ Performance benchmark
├─ 10-page validation run
└─ Compare with/without cache

Phase A5: Documentation & Cleanup (NEXT)
├─ Add docstrings
├─ Update README
├─ Create monitoring guide
└─ Commit changes
```

---

## 10. Next Steps

1. **Approved:** Proceed with Phase A2 (UnifiedCacheManager implementation)
2. **Timeline:** 2-3 hours for implementation + testing
3. **Validation:** 5-page test run shows 40-50% hit rate
4. **Integration:** Full 800-page production run with caching enabled

---

**Analysis Complete** ✅
**Ready for Implementation** 🚀
**Estimated Sprint Duration:** 8 hours
