# Phase 2 Implementation Plan

## Overview
Addressing 4 key improvements for Action 6 (and other API-dependent actions):
1. **RateLimiter Reuse**: Eliminate creating new RateLimiter on every SessionManager
2. **Timestamp Logic Gate**: Reintroduce data freshness check before fetching
3. **Logging Consolidation**: Reduce verbose logs while keeping essential info
4. **Performance Optimization**: Increase RPS to 5.0, verify parallel processing & cookie caching

---

## Issue 1: RateLimiter Reuse ⚠️ CRITICAL

### Current Problem
- **Location**: `session_manager.py` line 307
- **Issue**: New `RateLimiter()` created every time `SessionManager()` instantiated
- **Impact**: 
  - Resets rate limiting state (delay, metrics, circuit breaker)
  - Creates new CircuitBreaker instance (logs initialization)
  - Loses adaptive delay tuning from previous requests
  - Inefficient: RateLimiter should be shared across session lifetime

### Current Flow
```
main.py:1689 → SessionManager() → RateLimiter() [NEW]
main.py:551  → SessionManager() [temp] → RateLimiter() [NEW]
main.py:892  → SessionManager() [temp] → RateLimiter() [NEW]
action6:45   → Uses session_manager.rate_limiter (shared)
```

### Solution: Singleton Pattern
```python
# In utils.py - Create global RateLimiter instance
_global_rate_limiter = None

def get_rate_limiter():
    """Get or create global RateLimiter instance."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter()
    return _global_rate_limiter

# In session_manager.py line 307
self.rate_limiter = get_rate_limiter()  # Reuse existing instance
```

### Benefits
- ✅ Preserves rate limiting state across sessions
- ✅ Maintains adaptive delay tuning
- ✅ Eliminates redundant CircuitBreaker initialization
- ✅ Reduces log spam (4 fewer logs per SessionManager creation)

---

## Issue 2: Timestamp Logic Gate 🔄 REINTRODUCE

### Current Problem
- **Location**: `action6_gather.py` - comparator logic exists but no timestamp check
- **Issue**: Fetches all data even if recently saved
- **Impact**: Unnecessary API calls, slower processing

### Solution: Add Timestamp Check
```python
# In action6_gather.py _process() method
def _should_fetch_details(match, session_manager):
    """Check if we should fetch details based on timestamp."""
    # Check if DnaMatch exists
    existing = session.query(DnaMatch).filter_by(
        person_id=match['personId']
    ).first()
    
    if not existing:
        return True  # New match, fetch details
    
    # Check if data is fresh (< 7 days old)
    if existing.last_updated:
        age_days = (datetime.now() - existing.last_updated).days
        if age_days < 7:
            return False  # Data is fresh, skip
    
    return True  # Data is stale, fetch
```

### Benefits
- ✅ Skips fetching if data is recent
- ✅ Reduces API calls by ~30-50%
- ✅ Faster processing on subsequent runs
- ✅ Configurable freshness threshold

---

## Issue 3: Logging Consolidation 📊 IN PROGRESS

### Changes Made
1. ✅ Circuit Breaker init: INFO → DEBUG
2. ✅ RateLimiter init: Removed verbose log

### Changes Needed
1. **Browser initialization**: 20+ logs → 3-5 logs
   - Consolidate: "Starting browser", "WebDriver init", "Browser ready"
   
2. **Cookie sync**: 3+ logs → 1 log per operation
   - Consolidate: "Syncing cookies" with count
   
3. **Navigation**: 5+ logs → 2 logs
   - Consolidate: "Navigating to X", "Navigation complete"

### Expected Impact
- **Before**: 12,700 lines per run
- **After**: ~8,000 lines per run (37% reduction)

---

## Issue 4: Performance Optimization ⚡

### A. Increase RPS to 5.0
**Current**: `requests_per_second: float = 0.4` (config_schema.py:413)
**Change**: `requests_per_second: float = 5.0`

**Rationale**:
- Current 0.4 RPS is extremely conservative
- Ancestry API typically allows 10-20 RPS
- 5.0 RPS is still safe with circuit breaker protection
- Expected speedup: ~12x faster (0.4 → 5.0)

**Implementation**:
```python
# config/config_schema.py line 413
requests_per_second: float = 5.0  # Increased from 0.4 (12x faster)
```

### B. Verify Parallel Processing
**Current**: `parallel_workers: int = 1` (config_schema.py)
**Status**: Code supports parallel but not enabled by default

**Verification**:
- Check if thread pool is actually used
- Monitor for concurrency errors
- Verify rate limiting works with parallel

### C. Verify Cookie Caching
**Current**: CSRF token cached for 300s (session_manager.py:190)
**Status**: Cookies synced multiple times per session

**Verification**:
- Check cookie sync frequency
- Measure time saved by caching
- Ensure cookies don't expire during session

---

## Implementation Order

### Phase 2.1: RateLimiter Reuse (HIGH PRIORITY)
1. Create `get_rate_limiter()` singleton in utils.py
2. Update SessionManager to use singleton
3. Test: Verify rate limiter state persists
4. Commit: "Implement RateLimiter singleton pattern"

### Phase 2.2: Timestamp Logic Gate (HIGH PRIORITY)
1. Add `last_updated` timestamp to DnaMatch model
2. Implement `_should_fetch_details()` check
3. Test: Verify skips recent data
4. Commit: "Add timestamp-based data freshness check"

### Phase 2.3: Logging Consolidation (MEDIUM PRIORITY)
1. Consolidate browser initialization logs
2. Consolidate cookie sync logs
3. Consolidate navigation logs
4. Test: Measure log file size reduction
5. Commit: "Consolidate verbose logging"

### Phase 2.4: Performance Optimization (MEDIUM PRIORITY)
1. Increase RPS to 5.0 in config
2. Verify parallel processing works
3. Verify cookie caching works
4. Test: Run Action 6 twice, measure performance
5. Commit: "Optimize RPS and verify parallel/caching"

---

## Testing Strategy

### Test 1: RateLimiter Reuse
```bash
python -c "
from core.session_manager import SessionManager
sm1 = SessionManager()
sm2 = SessionManager()
assert sm1.rate_limiter is sm2.rate_limiter, 'RateLimiter should be shared'
print('✅ RateLimiter reuse verified')
"
```

### Test 2: Timestamp Logic
```bash
# Run Action 6 twice
python main.py  # Select Action 6
python main.py  # Select Action 6 again
# Verify: Second run should skip more matches
```

### Test 3: Logging Reduction
```bash
# Compare log file sizes
wc -l Logs/app.log  # Before
# Run Action 6
wc -l Logs/app.log  # After
```

### Test 4: Performance
```bash
# Measure time for Action 6 with RPS=5.0
time python main.py  # Select Action 6
```

---

## Expected Outcomes

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| RateLimiter instances | 3+ per run | 1 global | 66% reduction |
| API calls (2nd run) | 200 | ~100 | 50% reduction |
| Log file size | 12,700 lines | 8,000 lines | 37% reduction |
| Action 6 duration | 27s | ~15s | 44% faster |
| RPS effective | 0.37/s | 5.0/s | 13x faster |

---

## Files to Modify
1. `utils.py` - Add RateLimiter singleton
2. `session_manager.py` - Use RateLimiter singleton
3. `database.py` - Add last_updated to DnaMatch
4. `action6_gather.py` - Add timestamp check
5. `config/config_schema.py` - Increase RPS to 5.0
6. Browser/cookie/nav logging - Consolidate logs

---

## Rollback Plan
If issues occur:
1. Revert RPS to 0.4 (conservative)
2. Disable parallel processing (set to 1)
3. Revert RateLimiter to per-session creation
4. Disable timestamp check (fetch all)

