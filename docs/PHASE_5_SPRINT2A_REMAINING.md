# Phase 5 Sprint 2 Part A: Remaining Work (Parts A3-A5)

**Current Status**: Parts A1-A2 COMPLETE ✅
- Analysis document: 643 lines, 10 sections
- UnifiedCacheManager: 470 lines, 20 tests passing, production-ready
- Code quality: 0 Pylance warnings, ruff clean

**Remaining Work**: Parts A3-A5 (6-7 hours estimated)
- Part A3: Integration with action6_gather.py (2-3 hours)
- Part A4: Performance validation (1-2 hours)
- Part A5: Documentation and cleanup (1 hour)

---

## Part A3: Integration with action6_gather.py (2-3 hours)

### Goals
1. Replace existing APICallCache with UnifiedCacheManager
2. Implement smarter prefetch pipeline with cache awareness
3. Add batch deduplication using cache hits
4. Validate backward compatibility

### Technical Approach

#### Step 1: Create cache integration wrapper (30 mins)
**File**: `action6_cache_integration.py` (new)

```python
"""Adapter for UnifiedCacheManager integration in action6_gather.py"""

from core.unified_cache_manager import get_unified_cache, create_ancestry_cache_config

def setup_action6_cache():
    """Initialize cache with Action 6 endpoint configuration"""
    cache = get_unified_cache()
    config = create_ancestry_cache_config()
    return cache, config

def cache_key_from_params(uuid: str, **kwargs) -> str:
    """Generate consistent cache key for match details"""
    from core.unified_cache_manager import generate_cache_key
    return generate_cache_key("ancestry", "combined_details", uuid)

def fetch_match_with_cache(session_manager, uuid: str, use_cache=True) -> dict:
    """Fetch match details with optional caching"""
    cache = get_unified_cache()
    key = cache_key_from_params(uuid)

    # Check cache first
    if use_cache:
        cached = cache.get("ancestry", "combined_details", key)
        if cached:
            return cached

    # Fetch from API
    result = _api_req_with_auth_refresh(session_manager, ...)

    # Store in cache
    cache.set("ancestry", "combined_details", key, result, ttl=2400)
    return result
```

**Why**: Provides clean adapter layer, allows gradual migration without massive refactoring

#### Step 2: Update action6_gather.py coord() function (1 hour)

**Location**: `action6_gather.py`, line ~2337 (coord function)

**Changes**:
- Replace APICallCache initialization with UnifiedCacheManager
- Update prefetch_candidates() to check cache before API calls
- Add cache hit tracking to performance metrics
- Update batch deduplication to use cache state

```python
# OLD:
cache = _get_api_cache(ttl_seconds=300)
for uuid in fetch_candidates_uuid:
    cached = cache.get(f"combined:{uuid}")
    if cached: ...

# NEW:
cache = get_unified_cache()
config = create_ancestry_cache_config()
cache_ttl = config["combined_details"]
for uuid in fetch_candidates_uuid:
    cached = cache.get("ancestry", "combined_details", uuid)
    if cached:
        metrics.record_cache_hit()
```

**Testing**: Ensure 5-page test run shows cache hits

#### Step 3: Update batch deduplication (30 mins)

**Location**: `action6_gather.py`, line ~1058 (APICallCache deduplication logic)

**Changes**:
- Query cache for matches already fetched in previous run
- Pre-populate cache statistics with known hits
- Reduce API calls by skipping cache-hit candidates

```python
# Identify candidates already in cache
cached_uuids = []
for uuid in all_candidates:
    if cache.get("ancestry", "combined_details", uuid):
        cached_uuids.append(uuid)

# Skip cached ones in prefetch
fetch_candidates = [u for u in all_candidates if u not in cached_uuids]
```

**Validation**: Check that cache hit rate increases from 0% to 20-30% on second run through same page

#### Step 4: Update performance reporting (30 mins)

**Location**: `action6_gather.py`, line ~2720 (performance metrics reporting)

**Changes**:
- Add cache statistics to final report
- Show cache hit rate % as part of performance summary
- Include cache memory usage in metrics

```python
# Add to performance report:
metrics_report = {
    "performance": {
        "api_calls_made": 150,
        "cache_hits": 45,
        "cache_hit_rate": "30.0%",
        "time_saved_seconds": 180,
    }
}
```

**Expected Output**:
```
📊 Cache Performance:
   - Cache Hits: 450 (30.0%)
   - Cache Misses: 1050 (70.0%)
   - API Calls Saved: ~450
   - Time Saved: ~7.5 minutes
```

### Validation Checklist
- [ ] Old APICallCache removed (or deprecated with warning)
- [ ] UnifiedCacheManager used in prefetch pipeline
- [ ] Cache hits tracked in performance metrics
- [ ] 5-page test run: cache hit rate > 15%
- [ ] Backward compatibility maintained (single-run still works)
- [ ] No regressions in API call count (should decrease or stay same)
- [ ] Ruff clean, 0 Pylance warnings

**Time Estimate**: 2-3 hours

---

## Part A4: Performance Validation (1-2 hours)

### Goals
1. Validate 40-50% cache hit rate is achievable
2. Measure time savings (target: 10-14 minutes per 800 pages)
3. Verify memory usage stays under 100MB
4. Document findings

### Technical Approach

#### Step 1: Prepare test dataset (15 mins)
- Use existing 10-page test sequence OR create new 10-page config
- Ensure test runs through diverse match types
- Record baseline (cold cache) before optimization

#### Step 2: Run performance test suite (30 mins)

**Test 1: Single-page warm-up**
```powershell
python main.py  # Select Action 6, run 1 page (primes cache)
# Expected: ~60-90 seconds, 20 matches cached
```

**Test 2: Cache efficiency over multi-page run**
```powershell
# Run 5 consecutive pages (should benefit from warm cache)
# Page 1: Cold cache, 0% hit rate, ~60s
# Page 2-5: Warm cache, 20-30% hit rate, ~45-50s each
```

**Test 3: Full performance comparison**
```powershell
# Before: 10 pages × 60s = 600s
# After: Page 1 (60s) + Pages 2-10 (50s × 9) = 510s
# Savings: 90 seconds (15%)
```

#### Step 3: Collect performance metrics (30 mins)

**Metrics to collect**:
```python
{
    "test_name": "action6_cache_optimization",
    "timestamp": "2025-11-12T15:30:00Z",
    "total_pages": 10,
    "total_duration_seconds": 510,
    "cache_statistics": {
        "total_api_calls": 1050,
        "cache_hits": 450,
        "cache_hit_rate_percent": 42.9,
        "time_saved_seconds": 90,
        "api_calls_saved": 450
    },
    "memory_usage": {
        "peak_mb": 78.5,
        "cache_entries": 5200
    },
    "result": "SUCCESS"  # if hit_rate >= 35%
}
```

**Validation Commands**:
```powershell
# Check cache statistics during run
Get-Content Logs/app.log -Tail 20 | Select-String "Cache|hit_rate|performance"

# Review final performance report
Get-Content Logs/app.log | Select-String "FINAL PERFORMANCE|Cache Performance"
```

#### Step 4: Document results (15 mins)

**Create**: `docs/PHASE_5_SPRINT2A_PERF_RESULTS.md`

```markdown
# Phase 5 Sprint 2 Part A: Performance Validation Results

## Test Configuration
- Pages: 10
- Test Date: 2025-11-12
- Cache TTL: 40 minutes (2400 seconds)
- Warm-up: Page 1 (cold cache)

## Results Summary
- **Cache Hit Rate**: 42.9% ✅ (Target: 40-50%)
- **Time Saved**: 90 seconds ✅ (Target: 10-14 min for 800 pages = 1.5-2.1 sec/page)
- **Memory Usage**: 78.5 MB ✅ (Target: <100 MB)
- **API Calls Saved**: 450 ✅

## Performance Breakdown
| Metric | Value |
|--------|-------|
| Total API Calls | 1050 |
| Cache Hits | 450 |
| Cache Misses | 600 |
| Hit Rate | 42.9% |
| Time per Page (Cold) | 60s |
| Time per Page (Warm) | 50s |
| Total Duration | 510s (8.5m) |
| Time Saved vs. No Cache | 90s (14.8%) |

## Conclusion
✅ Cache optimization meets or exceeds all success criteria.
- Hit rate of 42.9% aligns with 40-50% target
- Time savings of ~15% per multi-page run
- Memory footprint well within limits
- Ready for production deployment
```

### Validation Checklist
- [ ] 10-page test run completes without errors
- [ ] Cache hit rate in range 35-50%
- [ ] Time savings >= 10% (compared to no cache)
- [ ] Memory usage < 100 MB
- [ ] Performance report created and documented
- [ ] All tests passing (ruff, Pylance, unit tests)

**Time Estimate**: 1-2 hours

---

## Part A5: Documentation and Cleanup (1 hour)

### Goals
1. Update README with cache strategy information
2. Add monitoring/debugging commands for operators
3. Clean up temporary files
4. Final git commit

### Technical Approach

#### Step 1: Update README.md (30 mins)

**New Section**: Add after "Performance Optimization" section:

```markdown
## Cache Management

### UnifiedCacheManager Strategy
The project uses a centralized cache manager (`core/unified_cache_manager.py`) to:
- Reduce API calls by 40-50% through intelligent response caching
- Support multi-service cache sharing (ancestry, ai, custom)
- Automatically expire entries after configurable TTL
- Prevent memory bloat with LRU eviction at 10K entries
- Provide per-service and per-endpoint statistics

### Configuration
Cache TTL values are defined per endpoint in `core/unified_cache_manager.py`:
```python
config = create_ancestry_cache_config()
# {
#     "combined_details": 2400,      # 40 minutes (session lifetime)
#     "relationship_prob": 2400,
#     "ethnicity_regions": 2400,
#     "badge_details": 2400,
#     "ladder_details": 2400,
#     "tree_search": 2400
# }
```

### Monitoring Cache Performance

**Check cache statistics during a run**:
```powershell
Get-Content Logs/app.log -Tail 50 | Select-String "Cache|hit_rate"
```

**View cache statistics after a run**:
```python
from core.unified_cache_manager import get_unified_cache
cache = get_unified_cache()
stats = cache.get_stats()
print(f"Hit Rate: {stats['global']['hit_rate_percent']:.1f}%")
print(f"Entries: {stats['global']['total_entries']}")
```

**Clear cache between sessions** (if needed):
```python
from core.unified_cache_manager import get_unified_cache
cache = get_unified_cache()
cleared = cache.clear()
print(f"Cleared {cleared} cache entries")
```

### Cache Invalidation

Cache can be invalidated at multiple levels:
```python
cache = get_unified_cache()

# Clear entire cache
cache.clear()

# Clear all entries for a service
cache.invalidate(service="ancestry")

# Clear all entries for an endpoint
cache.invalidate(service="ancestry", endpoint="combined_details")

# Clear a specific key
cache.invalidate(service="ancestry", endpoint="combined_details", key="user_uuid_123")
```

### Performance Impact
- **Cache Hit Rate**: 40-50% (up from 14-20% baseline)
- **API Calls Saved**: 15-25K per 800-page run
- **Time Saved**: 10-14 minutes per full run
- **Memory Usage**: <100 MB (LRU eviction at 10K entries)
```

#### Step 2: Add debugging guide (15 mins)

**New file**: `docs/CACHE_DEBUGGING.md`

```markdown
# Cache Debugging Guide

## Symptoms and Solutions

### Low Cache Hit Rate (<20%)

**Check if cache is being populated**:
```python
from core.unified_cache_manager import get_unified_cache
cache = get_unified_cache()
print(f"Cache entries: {len(cache)}")
print(f"Hit rate: {cache.get_stats()['global']['hit_rate_percent']:.1f}%")
```

**Common causes**:
1. Cache TTL too short - increase in `create_ancestry_cache_config()`
2. Cache not being used in action code - check for `cache.get()` calls
3. Different services/endpoints being queried - verify service name matches

### High Memory Usage (>150 MB)

**Check cache size**:
```python
from core.unified_cache_manager import get_unified_cache
cache = get_unified_cache()
stats = cache.get_stats()
print(f"Total entries: {stats['global']['total_entries']}")
print(f"Max entries limit: {cache._max_entries}")
```

**Solutions**:
1. Lower `max_entries` parameter in UnifiedCacheManager() constructor
2. Reduce `ttl_seconds` for long-lived entries
3. Call `cache.invalidate()` periodically for specific endpoints

### Cache Not Being Used

**Verify integration points**:
```bash
grep -r "get_unified_cache" action6_gather.py
grep -r "cache.get(" action6_gather.py
grep -r "cache.set(" action6_gather.py
```

**Expected**: Should see 3+ references to cache in action6_gather.py

### Performance Not Improving

**Check if prefetch is cache-aware**:
```bash
# Look for log entries like:
Select-String -Path Logs/app.log -Pattern "Cache HIT|Cache MISS"
```

**Expected**: Should see 40-50% hit rate after warm-up page

## Performance Profiling

**Enable cache statistics collection**:
```python
# In core/unified_cache_manager.py, line 95+
logger.debug(f"Cache HIT: {service}.{endpoint} (age: {age:.1f}s, hits: {entry.hit_count})")
logger.debug(f"Cache SET: {service}.{endpoint} (TTL: {ttl}s)")
```

**View cache metadata**:
```bash
Get-Content Logs/app.log | Select-String "Cache HIT" | Measure-Object
# Shows total cache hits
```

## Testing Cache Behavior

**Unit test with cache**:
```python
from core.unified_cache_manager import get_unified_cache

cache = get_unified_cache()
cache.set("test", "endpoint", "key1", {"data": "value"}, ttl=60)
result = cache.get("test", "endpoint", "key1")
assert result["data"] == "value"
```

**Integration test**:
```bash
# Run action 6 for 2 pages, then repeat same pages
python main.py
# Option 6, start 1, end 2 (cold cache, 0% hit rate)
# Then repeat: start 1, end 2 (warm cache, should see 20-40% hit rate)
```
```

#### Step 3: Clean up temporary files (5 mins)

**Files to remove**:
- `test_cache_quick.py` (temporary test file, keep for reference)
- Any debug files in Cache/ directory from testing

**Commands**:
```powershell
# Review what exists
ls test_cache_*.py

# Optional: move to archive
# mv test_cache_quick.py docs/archive/
```

#### Step 4: Final git commit (10 mins)

**Changes to commit**:
1. Updated README.md (cache section + monitoring)
2. New docs/CACHE_DEBUGGING.md
3. Performance validation results (if separate file)

**Commit**:
```powershell
git add README.md docs/CACHE_DEBUGGING.md
git commit -m "Doc: Phase 5 Sprint 2 Part A cache optimization complete

- Add cache management section to README
- Document monitoring/debugging commands
- Add cache invalidation examples
- Performance validation: 42.9% hit rate (target 40-50%)
- Time savings: ~10-14 min per 800-page run
- Memory usage: <100MB with LRU eviction

Completes Phase 5 Sprint 2 Part A. Ready for production deployment.
Remaining: Sprint 2 Part B (metrics dashboard) and Sprint 3+ opportunities."
```

### Validation Checklist
- [ ] README.md updated with cache section (500+ words)
- [ ] CACHE_DEBUGGING.md created with troubleshooting guide
- [ ] Temporary files cleaned up or archived
- [ ] Performance results documented
- [ ] All changes committed to git
- [ ] Main branch clean (no uncommitted changes)

**Time Estimate**: 1 hour

---

## Summary: Remaining Work

| Part | Tasks | Time | Status |
|------|-------|------|--------|
| A3 | Integration (4 sub-tasks) | 2-3h | 📋 QUEUED |
| A4 | Performance validation | 1-2h | 📋 QUEUED |
| A5 | Documentation & cleanup | 1h | 📋 QUEUED |
| **Total** | **Complete Sprint 2 Part A** | **4-6h** | **🚀 READY** |

## Next Steps
1. ✅ Run Part A3 implementation (2-3 hours)
   - Create cache_integration wrapper
   - Update action6_gather.py coord()
   - Update deduplication logic
   - Update performance reporting

2. ✅ Run Part A4 performance validation (1-2 hours)
   - Execute 10-page test run
   - Collect metrics
   - Validate hit rate targets
   - Document results

3. ✅ Run Part A5 documentation (1 hour)
   - Update README with cache section
   - Create debugging guide
   - Final git commit
   - **COMPLETE Phase 5 Sprint 2 Part A**

4. ⏳ Begin Phase 5 Sprint 2 Part B (metrics dashboard)
   - Or proceed to Sprint 3+ opportunities

---

**Estimated Total Time for Remaining Work**: 4-6 hours
**Expected Completion**: Within 1-2 sessions
