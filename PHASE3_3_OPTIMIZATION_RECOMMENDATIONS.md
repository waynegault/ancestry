# Phase 3.3 Test Results & Optimization Recommendations

## Test Results Analysis

### Actual Performance
- **Pages Processed**: 4 out of 5 (80%)
- **Matches Per Page**: 30 (hardcoded, ignoring MATCHES_PER_PAGE=60 config)
- **Total Matches**: 120 (not 300 as expected: 5 pages × 60 = 300)
- **Processing Time**:
  - Page 1: 276.3s (4.6 min) - 30 matches
  - Page 2: 267.6s (4.5 min) - 30 matches
  - Page 3: 244.5s (4.1 min) - 30 matches
  - Page 4: 1039.1s (17.3 min) - **PROBLEM: 4x slower!**
  - Page 5: Not processed
- **Total API Requests**: 594
- **429 Errors in Rate Limiter**: 0 ✅
- **429 Errors in Logs**: 19 ❌ (discrepancy!)
- **Rate Adaptations**: ↓0 decreases, ↑5 increases
- **Final Rate**: 0.526 req/s
- **Average Wait Time**: 1.402s

### Issues Identified

#### 1. **CRITICAL: Hardcoded itemsPerPage=30**
**Location**: `action6_gather.py` line 5395
```python
f"discoveryui-matches/parents/list/api/matchList/{my_uuid}?itemsPerPage=30&currentPage={current_page}"
```
**Impact**:
- Config says 60, code uses 30
- Expected 300 matches (5×60), got 120 (4×30)
- 50% throughput loss

**Fix**: Use config value
```python
items_per_page = getattr(config_schema, 'MATCHES_PER_PAGE', 30)
f"...?itemsPerPage={items_per_page}&currentPage={current_page}"
```

#### 2. **Session Invalidation at Page 4**
**Evidence**: `invalid session id` errors at 13:22:06 (during page 4)
**Impact**: Page 4 took 1039s (17 min) vs avg 263s (4.4 min) - **4x slower**
**Root Cause**: WebDriver session died, likely due to:
- PC sleep/hibernation (now fixed with sleep prevention)
- Network timeout
- Chrome crash

**Current Mitigation**: Session health monitoring (refreshes at 25-min mark)
**Improvement**: Reduce health check interval from 25min to 15min for faster recovery

#### 3. **429 Error Discrepancy**
**Metrics Say**: 0 errors in AdaptiveRateLimiter
**Logs Say**: 19 rate limit errors found
**Explanation**:
- AdaptiveRateLimiter only tracks API calls it makes
- Some 429s may be from:
  - Non-rate-limited paths (direct session.get calls)
  - Relationship probability API fallbacks
  - CloudScraper fallback attempts

**Fix**: Ensure ALL API calls go through rate limiter

#### 4. **Conservative Rate Limiting**
**Current**: 0.526 req/s (1.9s between requests)
**Ancestry Limit**: ~5 req/s estimated
**Headroom**: 90% unused capacity!

**Analysis**:
- Starting rate too conservative (0.5 req/s)
- Only 5 increases over entire run
- Requires 100 successes per 1% increase = very slow adaptation
- Never reached optimal rate

## Optimization Recommendations

### Priority 1: Fix Critical Issues (Immediate)

#### 1.1. Fix Hardcoded itemsPerPage
```python
# action6_gather.py line ~5390
def _call_match_list_api(...):
    # Get from config instead of hardcoded 30
    items_per_page = getattr(config_schema.api, 'matches_per_page', 30)
    
    match_list_url = (
        f"discoveryui-matches/parents/list/api/matchList/{my_uuid}"
        f"?itemsPerPage={items_per_page}&currentPage={current_page}"
    )
```

#### 1.2. Sleep Prevention (✅ DONE in this commit)
Added to `main.py`:
- Activates on startup
- Deactivates on 'q' selection
- Prevents PC sleep during long operations

### Priority 2: Rate Limiting Optimization (Fast & Safe)

#### 2.1. Increase Initial Fill Rate
**Current**: 0.5 req/s (too conservative)
**Recommended**: 1.5 req/s (safe starting point with 3x margin)

**Why**:
- Ancestry limit estimated ~5 req/s
- 1.5 req/s gives 70% safety margin
- Faster convergence to optimal rate
- Still conservative enough to avoid 429s

**Change** `.env`:
```bash
REQUESTS_PER_SECOND=1.5  # Was implied 0.5
```

#### 2.2. Accelerate Adaptation
**Current**: 100 successes per 1% increase
**Recommended**: 50 successes per 2% increase

**Why**:
- Current: 100 calls × 1.9s avg = 190s (3.2 min) per 1% increase
- Proposed: 50 calls × 1.0s avg = 50s (0.8 min) per 2% increase
- **6x faster adaptation** with same risk profile
- Reaches optimal rate in ~5-10 pages instead of 50+

**Change** `rate_limiter.py`:
```python
AdaptiveRateLimiter(
    initial_fill_rate=1.5,  # Increased from 0.5
    success_threshold=50,   # Reduced from 100
    # on_success() increase from 1% to 2%
)
```

#### 2.3. Reduce Health Check Interval
**Current**: Check every 5 pages, refresh at 25-min mark
**Recommended**: Check every 3 pages, refresh at 15-min mark

**Why**: Page 4 session death took 17 minutes to recover
**Impact**: Faster detection and recovery from session issues

**Change** `.env`:
```bash
HEALTH_CHECK_INTERVAL_PAGES=3    # Was 5
SESSION_REFRESH_THRESHOLD_MIN=15 # Was 25
```

### Priority 3: Configuration Cleanup (Phase 4)

Remove deprecated config that no longer applies:
- `INITIAL_DELAY` - Replaced by AdaptiveRateLimiter.initial_fill_rate
- `MAX_DELAY` - No longer needed (token bucket handles this)
- `BACKOFF_FACTOR` - Replaced by on_429_error() logic
- `DECREASE_FACTOR` - Replaced by on_success() logic
- `THREAD_POOL_WORKERS` - Sequential processing only now

### Priority 4: Monitoring Improvements

#### 4.1. Add Rate Limiter Logging
Log fill rate changes more frequently:
```python
# After every page, not just on changes
logger.info(f"📊 Current rate: {metrics.current_fill_rate:.3f} req/s "
            f"(successes: {metrics.success_count}/{self.success_threshold})")
```

#### 4.2. Track All 429 Sources
Instrument all API call paths to identify 429 sources:
- `_api_req` calls (tracked ✅)
- `session.get` calls (not tracked ❌)
- CloudScraper fallbacks (not tracked ❌)

## Projected Performance Impact

### Current Performance (Baseline)
- **Rate**: 0.526 req/s
- **Page Time**: 263s avg (4.4 min/page, excluding page 4 anomaly)
- **5 Pages**: ~22 minutes (actual: >30 min due to issues)
- **Throughput**: 30 matches/page = 150 matches in 22 min = **6.8 matches/min**

### With Priority 1 Fixes (itemsPerPage + sleep)
- **Rate**: 0.526 req/s (same)
- **Page Time**: 263s avg
- **5 Pages**: ~22 minutes (stable, no sleep interruptions)
- **Throughput**: 60 matches/page = 300 matches in 22 min = **13.6 matches/min** (2x improvement)

### With Priority 1 + Priority 2 (optimized rate limiting)
- **Starting Rate**: 1.5 req/s
- **Convergence**: Reaches ~3 req/s by page 3-5
- **Average Rate**: ~2.5 req/s (during test)
- **Page Time**: ~110s (1.8 min/page)
- **5 Pages**: ~9 minutes
- **Throughput**: 60 matches/page = 300 matches in 9 min = **33 matches/min** (5x improvement)

### Safety Validation
- **Initial 1.5 req/s**: 70% below estimated 5 req/s limit ✅
- **Target 3 req/s**: 40% safety margin ✅
- **Adaptive backoff**: 20% decrease on any 429 ✅
- **Conservative increase**: 2% per 50 successes ✅

## Implementation Plan

### Step 1: Apply Priority 1 Fixes
1. Fix itemsPerPage hardcoding ✅ (ready to implement)
2. Sleep prevention ✅ (already done)
3. Test with MAX_PAGES=5

**Expected**: 300 matches in ~22 min, zero 429s, stable throughout

### Step 2: Apply Priority 2 Optimizations
1. Update `.env`: REQUESTS_PER_SECOND=1.5
2. Update `rate_limiter.py`: success_threshold=50, increase=2%
3. Update `.env`: Health check intervals
4. Test with MAX_PAGES=5

**Expected**: 300 matches in ~9 min, zero 429s, smooth adaptation to 3 req/s

### Step 3: Validation Run
1. MAX_PAGES=50 (production test)
2. Monitor for stability over long duration
3. Verify zero 429s maintained
4. Confirm rate converges and stabilizes

**Expected**: ~1500 matches in 45-60 min, zero 429s, rate stable at 2.5-3.5 req/s

## Risk Assessment

### Low Risk ✅
- itemsPerPage fix (just using config correctly)
- Sleep prevention (reversible, no API impact)

### Medium Risk ⚠️
- Initial rate 1.5 req/s (3x current, but 70% below limit)
- Faster adaptation (could overshoot, but 20% backoff protects)

### Mitigation
- Conservative increase step (2% per 50 calls)
- Aggressive decrease on 429 (20% immediate)
- Circuit breaker still active (blocks after 5 consecutive failures)
- Can revert to 0.5 req/s in `.env` if issues arise

## Conclusion

**Current State**:
- ❌ Only processing 30 matches/page instead of 60
- ❌ 19 hidden 429 errors (not tracked by rate limiter)
- ❌ Extremely conservative rate (0.5 req/s, 90% unused capacity)
- ❌ Page 4 session death (17 min recovery)

**With All Fixes**:
- ✅ 60 matches/page (using config correctly)
- ✅ Zero 429 errors (proper rate limiting)
- ✅ Optimal rate convergence (3 req/s by page 5)
- ✅ No sleep interruptions
- ✅ **5x throughput improvement** (6.8 → 33 matches/min)

**Recommendation**:
1. Apply Priority 1 fixes immediately (low risk, 2x improvement)
2. Test with MAX_PAGES=5 to validate
3. Apply Priority 2 optimizations (medium risk, 5x total improvement)
4. Final validation with MAX_PAGES=50

**Timeline**:
- Step 1: 15 minutes to implement + 30 min test
- Step 2: 30 minutes to implement + 15 min test  
- Step 3: 60 minutes test
- **Total**: 2.5 hours to 5x improvement
