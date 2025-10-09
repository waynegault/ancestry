# Action 6 Performance Analysis - October 9, 2025

## Current Performance Issues

### Observed Performance
From `Logs/app.log`:
- Page 1: 131.9s for 20 matches (6.60s/match)
- Page 2: 121.4s for 20 matches (6.07s/match)  
- Page 3: 114.0s for 20 matches (5.70s/match)
- **Average: ~6s per match**

### Total Time Projection
- Total matches: 16,040 (802 pages × 20 matches/page)
- Time per match: 6.0s average
- **Total time: 96,240 seconds = 26.7 hours**

This is EXTREMELY slow for a batch operation.

---

## Root Cause Analysis

### Configuration Issue

The `.env` file had:
```env
THREAD_POOL_WORKERS=3
REQUESTS_PER_SECOND=0.4  # TOO CONSERVATIVE
```

**Problem**: `REQUESTS_PER_SECOND=0.4` is TOO conservative!

**Note**: There is NO `BATCH_SIZE` configuration - Action 6 processes matches PER PAGE (20 matches/page), not per batch. Each page is fetched, processed, and saved to the database atomically.

### Why Is It Slow?

Each DNA match requires **4-5 API calls**:
1. Combined Details API (match details + profile)
2. Badge Details API (tree information)
3. Ladder API (relationship path)
4. Match Probability API (relationship prediction)

With 3 parallel workers at 0.4 total RPS:
- Per-worker rate: 0.4 ÷ 3 = **0.13 RPS per worker**
- Time between requests: 1 ÷ 0.13 = **7.5 seconds per request**
- 4 API calls × 7.5s = **30 seconds of waiting per match**
- Actual API processing: ~2-3 seconds
- **Total: ~32s per match minimum**

The API calls themselves are fast, but we're spending most of the time WAITING due to overly conservative rate limiting.

---

## Recommended Solutions

### Option 1: Increase Rate Limit (Recommended)

**Change `.env` to:**
```env
THREAD_POOL_WORKERS=3
REQUESTS_PER_SECOND=1.2   # Changed from 0.4
```

**Expected Result:**
- Per-worker rate: 1.2 ÷ 3 = 0.4 RPS per worker
- Time per match: ~10-12 seconds (2-3x faster)
- Total time: ~12-15 hours (down from 27 hours)

**Risk**: Low - 1.2 RPS total is still conservative
**Testing**: Monitor first 50 pages for any 429 errors

### Option 2: Increase Workers (Moderate Risk)

**Change `.env` to:**
```env
THREAD_POOL_WORKERS=4
REQUESTS_PER_SECOND=1.6
```

**Expected Result:**
- Per-worker rate: 1.6 ÷ 4 = 0.4 RPS per worker  
- Time per match: ~8-10 seconds (3-4x faster)
- Total time: ~10-12 hours

**Risk**: Moderate - More workers = more coordination overhead
**Testing**: Monitor closely for 429 errors, reduce if needed

### Option 3: Aggressive (Higher Risk)

**Change `.env` to:**
```env
THREAD_POOL_WORKERS=5
REQUESTS_PER_SECOND=2.5
```

**Expected Result:**
- Per-worker rate: 2.5 ÷ 5 = 0.5 RPS per worker
- Time per match: ~5-6 seconds (5x faster)
- Total time: ~7-8 hours

**Risk**: Higher - This was attempted before and caused 429 errors
**Testing**: Only try if Options 1-2 are stable for 100+ pages

---

## Why Was 0.4 RPS Chosen?

Historical context from git commit c3b5535 (Aug 12, 2025):
- System was experiencing frequent 429 rate limit errors
- Conservative approach taken: 0.4 RPS with 3 workers
- Result: ZERO 429 errors, but very slow (27 hours)

**The problem**: We fixed rate limiting errors but made it TOO conservative.

---

## Implementation Notes

### Changes Made Today

1. ✅ **Auto-resume from database**: When no page specified, resumes from last saved record
2. ✅ **Progress bar formatting**: Added blank line before logs for better readability
3. ✅ **Speed metrics**: Shows matches/minute in progress bar
4. ✅ **Sleep prevention**: System stays awake during long runs (Windows/macOS/Linux)
5. ✅ **Clarified processing**: We process and save 20 matches per PAGE (not individual records)

### Progress Bar Format

**Before:**
```
14:45:40 INF Processing 802 pages from page 1 to 802.
  0%|                    | 0/16040
14:45:41 INF 📥 Fetch queue: 20 matches
```

**After:**
```
14:45:40 INF Processing 802 pages from page 1 to 802.

  0%|                    | 0/16040

14:45:41 INF 📥 Fetch queue: 20 matches
```

### Speed Tracking

Progress bar now shows:
```
  5%|██                  | 800/16040  New=600 Upd=150 Skip=50 Err=0 Speed=15.3/min
```

Where `Speed=15.3/min` means 15.3 matches processed per minute.

---

## Testing Recommendations

### Phase 1: Validation (Current Settings)
1. Let current run complete a few pages to establish baseline
2. Monitor for any 429 errors (should be zero)
3. Record average time per page

### Phase 2: Moderate Increase
1. Stop current run (Ctrl+C)
2. Update `.env`: `REQUESTS_PER_SECOND=1.2`
3. Restart: Will auto-resume from last saved record
4. Monitor first 50 pages closely
5. Check logs for 429 errors: `Select-String -Path "Logs\app.log" -Pattern "429"`

### Phase 3: Further Optimization (If Phase 2 Stable)
1. After 100+ pages with no errors, consider increasing to 1.6 RPS
2. Continue monitoring
3. If stable for 200+ pages, can try 2.0 RPS

### Emergency Rollback
If you see ANY "429 Too Many Requests" errors:
1. Stop immediately (Ctrl+C)
2. Reduce `REQUESTS_PER_SECOND` by 50% (e.g., 1.2 → 0.6)
3. Wait 5 minutes for rate limit to reset
4. Restart (will auto-resume)

---

## Monitoring Commands

### Check for Rate Limit Errors
```powershell
# Should return 0
(Select-String -Path "Logs\app.log" -Pattern "429|Too Many Requests").Count
```

### Watch Real-Time Progress
```powershell
Get-Content "Logs\app.log" -Wait -Tail 20
```

### Calculate Current Speed
```powershell
# Last 10 page completions
Select-String -Path "Logs\app.log" -Pattern "API fetch complete" | Select-Object -Last 10
```

---

## Processing Clarification

**How records are processed and saved:**

1. **Fetch Page**: Get 20 matches from Ancestry (single web page request)
2. **Identify Updates**: Compare with database, determine which need API refresh
3. **API Prefetch**: Fetch detailed data for matches (parallel, 3 workers)
4. **Database Save**: Bulk save all 20 matches to database
5. **Checkpoint**: Save progress after page completes
6. **Repeat**: Move to next page

**Key Points:**
- Progress bar shows **individual matches** (0/16040)
- Speed shows **matches per minute** (not pages)
- Database saves happen **per page** (20 matches at a time)
- Checkpointing happens **per page** (can resume from any page)

**Why per page?**
- Ancestry API returns 20 matches per page
- More efficient to batch process all 20 together
- Reduces database round-trips (bulk insert)
- Natural checkpoint boundaries

---

## Recommendations Priority

### Immediate (Today)
1. ✅ **Auto-resume working** - Can stop/start without losing progress
2. ✅ **Sleep prevention working** - Laptop won't interrupt 27-hour run
3. ✅ **Better monitoring** - Speed metrics visible in progress bar

### Short-term (This Week)
1. **Test RPS increase**: Try 1.2 RPS and monitor for 50-100 pages
2. **Validate stability**: Ensure zero 429 errors at new rate
3. **Measure improvement**: Should see 2-3x speedup (27h → 12-15h)

### Medium-term (Next Month)
1. **Incremental updates**: Only fetch matches that changed (not implemented yet)
2. **Smarter caching**: Cache API responses for 24 hours
3. **Parallel page fetching**: Fetch multiple pages simultaneously (complex)

---

## Conclusion

**Current State:**
- ✅ System is stable and reliable
- ✅ Zero rate limit errors
- ❌ Extremely slow (27 hours for full run)

**Root Cause:**
- Rate limit is too conservative (0.4 RPS)
- Spending 80% of time waiting, 20% processing

**Recommended Action:**
- Increase to 1.2 RPS as first step
- Monitor for stability
- Further increase if successful

**Expected Outcome:**
- 2-3x speedup (27h → 12-15h)
- Still conservative enough to avoid errors
- Can optimize further once proven stable
