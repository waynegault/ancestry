# Phase 5 Sprint 2 Part A: Cache Optimization

**Date Started:** November 12, 2025
**Status:** 🚀 IN PROGRESS
**Objective:** Improve cache hit rate from 14-20% → 40-50% (2-3x improvement)
**Estimated Duration:** 8 hours
**Priority:** High (enables faster batch processing, reduces API calls)

---

## Current State Analysis

### Existing Cache Infrastructure

**1. APICallCache in action6_gather.py**
- **Location:** Line 1058-1217
- **Purpose:** 5-minute TTL in-memory cache for API calls
- **Key Methods:** `get()`, `set()`
- **Current Performance:**
  - 14-20% cache hit rate (200-400 hits per 16K matches)
  - Saves 10-20 minutes per full run
  - Thread-safe for 2-worker parallel

**2. Profile Caching (action6_gather.py)**
- **Location:** Line 325-370
- **Mechanism:** Global persistent cache (`profile_details_{profile_id}`)
- **TTL:** 24 hours
- **Usage:** Line 2837+ in prefetch pipeline
- **Issue:** Only caches individual profiles, not batches

**3. API Search Cache (api_search_core.py)**
- **Location:** Line 31-240
- **Database:** ApiSearchCache table with expiration
- **Hit Rate Tracking:** _cache_stats (hits, misses, total_queries)
- **Usage:** Search criteria deduplication

**4. System Cache (core/system_cache.py)**
- **Location:** Lines 93-180
- **Features:** APIResponseCache class with service-specific TTL
- **Integration:** Decorator pattern (@cached_api_call)
- **Status:** Exists but not fully utilized by Action 6

### Performance Metrics

```
Current Hit Rate:     14-20%
Typical Run:          ~800 pages, 16K matches
API Calls Saved:      ~3.2K-3.2K per run (200-400 cache hits)
Time Saved:           ~10-20 minutes per run
Cache Payoff:         ~14% speedup
```

### Identified Bottlenecks

| Bottleneck | Impact | Root Cause | Solution |
|-----------|--------|-----------|----------|
| Per-match caching only | Low hit rate | No batch-level dedup | Batch endpoint cache |
| 5-min TTL too short | Misses opportunities | Conservative TTL | Increase to session lifetime (40m) |
| No cross-page cache | Duplicates across pages | Each page reruns same UUIDs | Shared UUID→details map |
| Profile cache not integrated | Missed 24h reuse | Separate caching layer | Integrate into prefetch pipeline |
| No relationship cache | Repeated ancestry calls | Expensive endpoint | Cache all relationship_probability calls |
| No ethnicity cache | Repeated ethnicity | Expensive endpoint | Cache all ethnicity_regions calls |

---

## Implementation Plan

### Phase A1: Analyze & Document (1-2 hours)

**Goal:** Understand all caching opportunities and create implementation roadmap

**Tasks:**
1. ✅ Identify all API endpoints in action6_gather.py (combined, tree, rel_prob, ethnicity, badge)
2. ✅ Document cache-ability of each endpoint (stable vs. dynamic)
3. ✅ Create cache key strategy for each endpoint
4. ✅ Estimate potential cache hit rates by endpoint
5. ✅ Design cross-action cache sharing (Actions 6-10)

**Deliverables:**
- PHASE_5_SPRINT2_CACHE_ANALYSIS.md

**Expected Cache-ability Matrix:**
```
Endpoint              | TTL    | Cache-able | Hit Rate | Time Saved
combined_details      | 40min  | YES        | 40-50%   | 8-12 min
relationship_prob     | 40min  | YES        | 35-45%   | 4-6 min
ethnicity_regions     | 40min  | YES        | 30-40%   | 2-4 min
badge_details         | 40min  | MAYBE      | 20-30%   | 1-2 min
ladder_details        | 40min  | YES        | 25-35%   | 2-3 min
tree_search           | 40min  | YES        | 20-30%   | 3-5 min
TOTAL POTENTIAL       |        |            | 40-50%   | 20-32 min
```

### Phase A2: Implement Unified Cache Manager (2-3 hours)

**Goal:** Create higher-level cache abstraction for cross-action use

**Tasks:**
1. Create `core/unified_cache_manager.py`:
   - `UnifiedCacheManager` class (similar to SessionCircuitBreaker pattern)
   - Service-aware cache key generation
   - TTL management per endpoint
   - Statistics collection (hits, misses, entries by endpoint)
   - Serialization/deserialization helpers
2. Integrate with existing APICallCache
3. Add type hints and comprehensive docstrings
4. Create factory functions for common patterns

**Key Features:**
- Thread-safe (Lock-based like circuit breaker)
- Service-specific TTL (40min for DNA, 24h for profiles, etc.)
- Endpoint-level stats (combined: 45%, relationship_prob: 40%, etc.)
- Graceful degradation (no cache → fallback to direct API)
- Optional Prometheus metrics hooks

**File:** `core/unified_cache_manager.py` (~250-300 lines)

### Phase A3: Refactor Action 6 to Use Unified Cache (2-3 hours)

**Goal:** Integrate unified cache into action6_gather.py prefetch pipeline

**Tasks:**
1. Replace individual cache lookups with unified manager calls
2. Update prefetch pipeline (line 2685+) to check cache first
3. Implement batch deduplication (line 2707+)
4. Cache prefetch results before DB commit
5. Add cache metrics to final report (alongside API stats)
6. Update logging to show cache efficiency

**Changes:**
- `_perform_api_prefetches()` - add cache lookup layer
- `_identify_fetch_candidates()` - add batch dedup
- `coord()` - final report shows cache stats
- Line 1209-1350 main loop - cache health monitoring

**Expected Impact:**
- Reduce API calls by 30-40%
- Reduce batch processing time by 15-25%
- Improve overall run time by 20-32 minutes (800 pages)

### Phase A4: Testing & Validation (1-2 hours)

**Goal:** Verify cache improvements with comprehensive tests

**Tasks:**
1. Create `test_unified_cache_manager.py`:
   - 12-15 unit tests covering cache operations
   - Mock API responses
   - TTL expiration tests
   - Thread safety tests
   - Serialization tests
2. Create `test_cache_optimization_integration.py`:
   - Simulate 10-page processing with cache
   - Verify hit rate improves
   - Compare with/without cache timing
   - Validate backward compatibility
3. Run full test suite (`run_all_tests.py`)
4. Perform manual 10-page validation run

**Success Criteria:**
- ✅ Cache hit rate 40-50% (3x improvement)
- ✅ All 30+ new tests passing
- ✅ Backward compatible (no breaking changes)
- ✅ Zero performance regressions (when cache is off)
- ✅ Final logging shows cache metrics

### Phase A5: Documentation & Cleanup (1 hour)

**Goal:** Document implementation for future developers

**Tasks:**
1. Add docstrings to all functions in unified_cache_manager.py
2. Update README.md cache section with new architecture
3. Create `docs/cache_optimization_guide.md`:
   - Cache key strategy for each endpoint
   - TTL rationale
   - Adding new cached endpoints
   - Debugging cache issues
4. Add cache monitoring commands to copilot-instructions.md
5. Commit all changes with descriptive messages

---

## Technical Design

### UnifiedCacheManager Architecture

```python
@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    data: Any
    timestamp: float
    ttl: int
    hit_count: int

class UnifiedCacheManager:
    """Centralized cache for all API responses and data."""

    def __init__(self):
        self._cache: dict[str, CacheEntry] = {}
        self._stats = {"hits": 0, "misses": 0, "entries": {}}
        self._lock = threading.Lock()

    def get(self, service: str, endpoint: str, key: str) -> Optional[Any]
    def set(self, service: str, endpoint: str, key: str, value: Any) -> None
    def get_stats(self) -> dict[str, Any]
    def clear(self, service: str, endpoint: str = None) -> int
    def __len__(self) -> int
    def __repr__(self) -> str
```

### Cache Key Strategy

```python
# combined_details
"cache:ancestry:combined_details:{uuid}"

# relationship_probability
"cache:ancestry:rel_prob:{uuid1}:{uuid2}"

# ethnicity_regions
"cache:ancestry:ethnicity:{ethnicity_id}"

# badge_details
"cache:ancestry:badges:{uuid}"

# tree_search
"cache:ancestry:tree:{search_hash}"
```

### Integration Points

1. **action6_gather.py** - Primary user
   - Prefetch pipeline (line 2685)
   - Batch processing (line 4800)
   - Final stats (line 1500)

2. **Action 7-10** - Secondary users
   - Shared UUID→details lookups
   - Tree information reuse
   - Ethnicity caching

3. **SessionManager** - Lifecycle
   - Initialize cache on session start
   - Clear on session end (optional)
   - Per-session TTL tracking

---

## Cache Hit Projections

### Conservative (35% hit rate)
- API calls reduced: 16K × 35% = 5.6K saved
- Time saved: ~18 minutes
- Cumulative improvement: 20% faster

### Aggressive (50% hit rate)
- API calls reduced: 16K × 50% = 8K saved
- Time saved: ~30 minutes
- Cumulative improvement: 35% faster

### Realistic (40-45% hit rate)
- API calls reduced: 16K × 42% = 6.7K saved
- Time saved: ~24 minutes
- Cumulative improvement: 28% faster

---

## Testing Strategy

### Unit Tests (12-15 tests)
1. Cache entry creation and retrieval
2. TTL expiration
3. Cache key generation
4. Hit/miss statistics
5. Thread safety
6. Concurrent reads/writes
7. Serialization/deserialization
8. Edge cases (None values, large objects)
9. Statistics accuracy
10. Clear operations
11. Size constraints (optional)
12. Service-specific TTL handling
13. Factory functions

### Integration Tests (3-5 tests)
1. 10-page prefetch with cache on
2. 10-page prefetch with cache off
3. Compare performance metrics
4. Verify cache doesn't break updates
5. Cross-action cache reuse

### Performance Tests
1. Cache lookup speed (<1ms target)
2. Serialization speed (<5ms target)
3. Memory usage (<100MB target for 8K entries)
4. Hit rate measurement

---

## Git Commit Strategy

**Commits (4-5 total):**
1. `core/unified_cache_manager.py` + tests
2. `action6_gather.py` integration
3. `test_cache_optimization_integration.py` + results
4. `docs/` updates
5. `PHASE_5_SPRINT2_CACHE_COMPLETION.md`

**Example:**
```bash
commit 1: Add UnifiedCacheManager with factory functions and tests (12/15 tests passing)
commit 2: Integrate UnifiedCacheManager into action6_gather.py prefetch pipeline
commit 3: Add cache optimization integration tests and performance validation
commit 4: Document cache optimization strategy and monitoring commands
commit 5: Phase 5 Sprint 2 Part A completion - Cache optimization complete (40-50% hit rate)
```

---

## Success Metrics

| Metric | Target | Validation |
|--------|--------|-----------|
| Cache hit rate | 40-50% | Logging shows cache stats |
| API calls reduced | 6-8K per run | Performance metrics |
| Time saved | 20-30 minutes | Elapsed time tracking |
| Tests passing | 30+ total | All tests pass |
| Backward compat | 100% | No breaking changes |
| Code quality | Ruff clean | Linting passes |
| Pylance warnings | 0 | No type errors |
| Documentation | Complete | README + guides |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Cache invalidation bugs | Medium | High | Thorough testing, conservative TTL |
| Memory bloat | Low | Medium | Size constraints, cleanup helpers |
| Thread safety issues | Low | High | Lock-based synchronization |
| API call mismatch | Low | High | Cache key verification |
| TTL too aggressive | Low | Medium | Start conservative, monitor |

---

## Next Steps

**Immediate (Today):**
1. ✅ Create this plan document
2. ⏳ Begin Phase A1 analysis
3. ⏳ Create analysis document

**This Week:**
4. ⏳ Implement UnifiedCacheManager
5. ⏳ Integrate with Action 6
6. ⏳ Test and validate

**Followup:**
7. ⏳ Proceed to Phase 5 Sprint 2 Part B (Metrics Dashboard)

---

**Estimated Completion:** 8 hours (by November 12, 2025 end of day)
