# Action 6 Stress Testing - Finding Optimal Settings

## Objective
Determine the fastest stable configuration for Action 6 DNA match gathering by:
1. Starting with aggressive settings to find the breaking point (429 errors)
2. Dialing back to find the sweet spot (maximum speed without errors)
3. Validating stability over 50+ pages

## Test Configurations

### Test 1: AGGRESSIVE (Current - Find Breaking Point)
```env
PARALLEL_WORKERS=4
REQUESTS_PER_SECOND=2.0
INITIAL_DELAY=0.2
```

**Expected Outcome**: Should trigger 429 errors quickly (within 1-3 pages)
**What to Monitor**:
```powershell
# Watch for 429 errors in real-time
Get-Content Logs\app.log -Wait | Select-String "429|rate|worker|ERROR"

# After test, count 429 errors
(Select-String -Path Logs\app.log -Pattern "429 error").Count
```

**Results**:
- Pages completed before 429: _____
- Total 429 errors: _____
- Average throughput before failure: _____ matches/hour
- Notes:

---

### Test 2: MODERATE (After Test 1 fails)
```env
PARALLEL_WORKERS=2
REQUESTS_PER_SECOND=0.8
INITIAL_DELAY=0.3
```

**Expected Outcome**: Fewer 429 errors, but may still hit rate limits
**Results**:
- Pages completed before 429: _____
- Total 429 errors: _____
- Average throughput: _____ matches/hour
- Notes:

---

### Test 3: CONSERVATIVE-FAST (Target Sweet Spot)
```env
PARALLEL_WORKERS=2
REQUESTS_PER_SECOND=0.5
INITIAL_DELAY=0.4
```

**Expected Outcome**: Zero 429 errors over 50+ pages
**Validation**: Run for 50+ pages to confirm stability
**Results**:
- Pages completed: _____
- Total 429 errors: _____ (target: 0)
- Average throughput: _____ matches/hour
- Notes:

---

### Test 4: BASELINE (Original Conservative)
```env
PARALLEL_WORKERS=1
REQUESTS_PER_SECOND=0.3
INITIAL_DELAY=0.5
```

**Purpose**: Baseline comparison for throughput improvement
**Results**:
- Pages completed: _____
- Total 429 errors: _____ (should be 0)
- Average throughput: _____ matches/hour
- Notes:

---

## Testing Procedure

### Before Each Test
1. **Clear logs** to isolate test results:
   ```powershell
   Remove-Item Logs\app.log
   ```

2. **Update .env** with test configuration

3. **Note start time and page number**

### During Test
1. **Monitor in real-time**:
   ```powershell
   Get-Content Logs\app.log -Wait | Select-String "429|Progress:|ERROR"
   ```

2. **Watch for**:
   - `429 Too Many Requests` - Rate limit hit
   - `Progress: X/Y` - Throughput metrics
   - `ERROR` - Any other failures

### After Test (If 429 Errors)
1. **Count errors**:
   ```powershell
   (Select-String -Path Logs\app.log -Pattern "429 error").Count
   ```

2. **Review backoff behavior**:
   ```powershell
   Select-String -Path Logs\app.log -Pattern "429|backoff|delay" | Select-Object -Last 20
   ```

3. **Calculate throughput before failure**:
   - Check last "Progress:" log entry
   - Note: "Throughput: XX matches/hour"

### After Test (If No Errors)
1. **Verify zero 429s**:
   ```powershell
   (Select-String -Path Logs\app.log -Pattern "429 error").Count  # Should be 0
   ```

2. **Extract final performance**:
   ```powershell
   Select-String -Path Logs\app.log -Pattern "FINAL PERFORMANCE REPORT" -Context 0,10
   ```

3. **Calculate improvement**:
   - Compare to baseline throughput
   - Document speedup factor

---

## Performance Metrics to Track

### Primary Metrics
- **429 Error Count**: Target is ZERO
- **Pages Completed**: How many pages before failure (if any)
- **Throughput**: Matches/hour from performance logs
- **Average Page Time**: Seconds per page

### Secondary Metrics
- **API Response Time**: From performance logs (should be <1s)
- **Cache Hit Rate**: From APICallCache logs (14-20% is normal)
- **Circuit Breaker Trips**: Should be zero

---

## Decision Criteria

### ✅ STABLE Configuration (Use in Production)
- **Zero 429 errors** over 50+ pages
- **Consistent throughput** (no degradation)
- **No circuit breaker trips**
- **Graceful adaptive rate limiting** (delay decreases successfully)

### ⚠️ UNSTABLE Configuration (Dial Back)
- **Any 429 errors** (even 1 is too many)
- **Circuit breaker trips**
- **Increasing error rates** over time

### 📊 OPTIMAL Configuration (Sweet Spot)
- Meets all STABLE criteria
- **Highest throughput** without errors
- **Sustained performance** over 100+ pages

---

## Commands Reference

### Monitor Rate Limiting
```powershell
# Real-time monitoring
Get-Content Logs\app.log -Wait | Select-String "429|rate|Progress:"

# Check for ANY 429 errors after test
(Select-String -Path Logs\app.log -Pattern "429 error").Count

# Review rate limiter initialization
Select-String -Path Logs\app.log -Pattern "Thread-safe RateLimiter|parallel workers"

# Check final performance report
Select-String -Path Logs\app.log -Pattern "FINAL PERFORMANCE" -Context 0,15
```

### Database Checkpoint
```powershell
# Check where we left off (for resume)
Get-Content Cache\action6_checkpoint.json | ConvertFrom-Json | Format-List
```

### Quick Test Run
```powershell
# Run Action 6 for 5 pages only
# In main menu: "6 1" (start page 1, will process up to MAX_PAGES=1 if set)
# Or edit .env: MAX_PAGES=5 for 5-page test
```

---

## Expected Results

### Baseline (Current Conservative)
- **Workers**: 1 (sequential)
- **RPS**: 0.3
- **Throughput**: ~40-60 matches/hour (~40s per page)
- **Stability**: 100% (validated over 800+ pages)

### Target Optimized
- **Workers**: 2-3 (parallel)
- **RPS**: 0.5-0.8
- **Throughput**: 80-150 matches/hour (2-3x improvement)
- **Stability**: 100% (zero 429 errors over 50+ pages)

### Aggressive (Breaking Point)
- **Workers**: 4+
- **RPS**: 1.0+
- **Throughput**: 200+ matches/hour initially
- **Stability**: Will fail with 429 errors (expected)

---

## Notes & Observations

### Test 1 Notes
-

### Test 2 Notes
-

### Test 3 Notes
-

### Final Recommendation
**OPTIMAL SETTINGS**:
```env
PARALLEL_WORKERS=____
REQUESTS_PER_SECOND=____
INITIAL_DELAY=____
```

**Throughput Improvement**: __x faster than baseline

**Validation**: Stable over ____ pages with zero 429 errors
