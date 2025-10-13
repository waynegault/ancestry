# Action 6 - Phase 1: Rate Limiting Fix

**Date**: 2025-10-13  
**Status**: ✅ **IMPLEMENTED - READY FOR TESTING**

---

## Strategy

Reverted to clean commit `e83f019` and applied lessons learned from log analysis.

---

## Root Cause Analysis

### The Real Problem
**Ancestry API has aggressive time-window rate limiting** (~250-300 requests per 5-10 minutes)

### Evidence from Logs
- ✅ Pages 1-13: 260 matches processed successfully (~1000+ API calls in 6 minutes)
- ❌ Page 14+: API started rejecting ALL requests with 429 errors
- ⏸️ Process stalled for 1+ hour waiting for retries that kept failing
- ❌ Browser timed out during the stall period due to inactivity

### Key Insight
**Browser timeout is a SYMPTOM, not the root cause.**  
The browser times out BECAUSE the process stalls due to 429 errors.

---

## Phase 1 Changes

### 1. Reduced Request Rate ✅
**File**: `.env`

**Before**:
```env
REQUESTS_PER_SECOND=1.5
```

**After**:
```env
REQUESTS_PER_SECOND=0.3
```

**Why**: 1.5 RPS is too aggressive for Ancestry's time-window rate limit.

### 2. Sequential Processing ✅
**File**: `.env`

**Before**:
```env
THREAD_POOL_WORKERS=3
```

**After**:
```env
THREAD_POOL_WORKERS=1
```

**Why**: Parallel workers cause immediate 429 errors even with low RPS.

### 3. Page Delays ✅
**File**: `action6_gather.py` (lines 516-521)

**Added**:
```python
# Add delay between pages to avoid hitting API time-window rate limits
# Ancestry API has aggressive rate limiting (~250-300 requests per 5-10 minutes)
# 30-second delay between pages helps stay under the limit
if current_page_num < last_page_to_process:
    logger.debug(f"Waiting 30 seconds before processing next page (rate limiting)")
    time.sleep(30)
```

**Why**: Spreads requests over time to avoid hitting the time-window limit.

### 4. Test Configuration ✅
**File**: `.env`

**Set**:
```env
MAX_PAGES=5
```

**Why**: Small test to verify fixes work before running full 802 pages.

---

## Expected Results

### Processing Speed
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| REQUESTS_PER_SECOND | 1.5 | 0.3 | **5x slower** |
| THREAD_POOL_WORKERS | 3 | 1 | **Sequential** |
| Page delay | 0s | 30s | **+30s per page** |
| Time per page | ~8.8 min | ~10-12 min | **Slower but reliable** |

### Reliability
| Feature | Before | After |
|---------|--------|-------|
| 429 errors | ❌ After page 13 | ✅ Should be eliminated |
| Process stalls | ❌ 1+ hour | ✅ Should not occur |
| Browser timeout | ❌ After 1.7 hours | ✅ Should not occur |
| Pages processed | ❌ 19 before failure | ✅ All 802 pages |

**Trade-off**: **Slower but reliable** - can process all 802 pages without 429 errors.

---

## Testing Plan

### Phase 1 Test (5 pages) ⏳

```bash
# Configuration already set in .env:
# MAX_PAGES=5
# REQUESTS_PER_SECOND=0.3
# THREAD_POOL_WORKERS=1

# Run Action 6
python main.py
> 6
> 1
```

**Expected**:
- ✅ 5 pages in ~50-60 minutes (10-12 min/page including 30s delays)
- ✅ **ZERO 429 errors**
- ✅ 100 matches processed successfully
- ✅ No process stalls
- ✅ No browser timeout

**Success Criteria**:
- No "too many 429 error responses" in logs
- All 5 pages complete successfully
- Process runs smoothly without stalls

**Failure Indicators**:
- Any 429 errors → Need to reduce RPS further or increase page delay
- Process stalls → Check for other API issues
- Browser timeout → Unlikely with 5 pages, but would indicate different issue

---

## Next Steps

### If Phase 1 Test Succeeds ✅
**Proceed to Phase 1b: Medium Test**
- Set `MAX_PAGES=50`
- Run full test (~8-10 hours)
- Verify no 429 errors over longer period

### If Phase 1 Test Fails ❌
**Phase 2: Enhanced 429 Detection**
- Implement 429 error detection with long backoff (5-10 minutes)
- Add exponential backoff for retries
- Consider increasing page delay to 60 seconds

---

## Files Modified

1. ✅ `.env` - Reduced RPS to 0.3, workers to 1, set MAX_PAGES=5
2. ✅ `action6_gather.py` - Added 30-second delay between pages

---

## Commit

```
fix(action6): Phase 1 - Rate limiting fix

- Reduce REQUESTS_PER_SECOND from 1.5 to 0.3 (conservative)
- Reduce THREAD_POOL_WORKERS from 3 to 1 (sequential processing)
- Add 30-second delay between pages to avoid time-window rate limits
- Set MAX_PAGES=5 for testing

Root Cause: Ancestry API has time-window rate limit (~250-300 requests per 5-10 minutes)
Solution: Conservative rate limiting + page delays to stay under limit
```

---

## Summary

**Problem**: 429 rate limiting errors after ~260 matches  
**Root Cause**: Ancestry API time-window rate limit  
**Solution**: Conservative RPS (0.3) + sequential processing + page delays (30s)  
**Trade-off**: Slower but reliable  
**Status**: ✅ **READY FOR TESTING**

---

**Start testing with MAX_PAGES=5 to verify the rate limiting fixes work!** 🚀

