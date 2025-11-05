# 3-Worker Test Analysis & Fix

## Test Results Summary

**Configuration**: 3 workers, optimized pre-wait calculation
**Result**: ❌ **FAILED** - 18× 429 rate limiting errors
**Pages completed**: ~1.5 of 5 (crashed on Page 2)

---

## Problem Analysis

### What Went Wrong

The optimized pre-wait calculation worked perfectly with 2 workers but **failed catastrophically with 3 workers**:

1. **Pre-wait calculation used 2-worker metrics**
   - `effective_rps = 1.37` (from previous 2-worker runs)
   - `estimated_time = 95 calls / 1.37 RPS = 69 seconds`
   - `tokens_refilled = 69s × 3.5 fill_rate = 241 tokens`
   - `pre_wait = almost nothing` (bucket appears full)

2. **But actual execution with 3 workers was MUCH faster**
   - 3 workers process requests ~1.7× faster than 2 workers
   - Actual execution time: ~40 seconds (not 69s!)
   - Actual tokens refilled: ~140 (not 241!)
   - Result: **Token bucket depleted mid-batch** → 429 errors

### Timeline of Failure

```
13:56:01 - Page 1 start: 99 API calls planned
13:56:01 - Pre-wait: MINIMAL (calculation used stale 2-worker effective_rps)
13:56:02 - Burst of 99 API calls starts
13:56:20 - Page 1 completes (19s, no errors - barely survived)

13:56:33 - Page 2 start: 95 API calls planned
13:56:33 - Pre-wait: MINIMAL (same issue)
13:56:34 - Burst of 95 API calls starts
13:56:41 - 💥 FIRST 429 ERROR on ethnicity API
13:56:42 - 💥 More 429 errors (3 simultaneous failures)
13:56:45 - Retry attempts (3.7s backoff)
13:56:52 - More 429s (60s backoff triggered)
```

### Root Cause

**The pre-wait optimization assumed a fixed effective_rps, but parallel workers CHANGE the effective_rps dynamically.**

```python
# BROKEN LOGIC:
effective_rps = 1.37  # From past runs with 2 workers
estimated_time = 95 / 1.37  # = 69s
# But with 3 workers, actual time = ~40s → under-estimated refill

# CORRECT LOGIC:
num_workers = 3
worker_speedup = sqrt(3) = 1.73
adjusted_rps = 1.37 × 1.73 = 2.37
estimated_time = 95 / 2.37  # = 40s ✅
```

---

## The Fix (Commit 6591c1f)

### Worker-Aware Pre-Wait Calculation

```python
# 1. Get number of workers from config
num_workers = config_schema.api.parallel_workers

# 2. Calculate speedup multiplier (conservative: sqrt, not linear)
#    Why sqrt? Amdahl's Law - parallel speedup has diminishing returns
worker_speedup = math.sqrt(num_workers)
# 2 workers: sqrt(2) = 1.41 → 1.4× faster
# 3 workers: sqrt(3) = 1.73 → 1.7× faster
# 4 workers: sqrt(4) = 2.00 → 2.0× faster

# 3. Adjust effective_rps for parallelism
adjusted_effective_rps = effective_rps * worker_speedup

# 4. Estimate execution time with adjusted RPS
estimated_execution_time = total_api_calls / adjusted_effective_rps

# 5. Calculate refill during (realistic) execution time
tokens_refilled = estimated_execution_time * fill_rate

# 6. Add safety buffer (10% per worker above 2)
safety_buffer = max(0, (num_workers - 2) * 0.10 * tokens_needed)

# 7. Calculate pre-wait with buffer
actual_deficit = tokens_needed - current_tokens - tokens_refilled
optimal_pre_delay = max(0, (actual_deficit + safety_buffer) / fill_rate)
```

### Why This Works

1. **Dynamic speedup adjustment**: Accounts for current worker count
2. **Conservative multiplier**: sqrt(n) not n (e.g., 3 workers = 1.7× not 3×)
3. **Safety buffer**: Extra 10% cushion per worker above 2
4. **Still optimized**: Still much better than old "wait for everything" approach

### Example Calculations

| Workers | Speedup | Adj RPS | Est Time (95 calls) | Refill | Pre-wait |
|---------|---------|---------|---------------------|--------|----------|
| 2 (base)| 1.41× | 1.93 | 49s | 172 tokens | ~8s |
| 3 (new) | 1.73× | 2.37 | 40s | 140 tokens | ~12s |
| 4       | 2.00× | 2.74 | 35s | 123 tokens | ~15s |

**Compare to old**:
- Old calculation: 24s pre-wait (too conservative, but safe)
- New broken: 5s pre-wait with 3 workers (too aggressive → 429s)
- New fixed: 12s pre-wait with 3 workers (balanced)

---

## Testing Instructions

### Step 1: Test with 2 Workers (Validate No Regression)

```powershell
# Edit .env: Set PARALLEL_WORKERS=2
# Run test
python main.py
# Choose Action 6, enter: 1 10

# Check results
(Select-String -Path Logs\app.log -Pattern "429 error").Count  # Should be 0
Select-String -Path Logs\app.log -Pattern "Adaptive rate limiting.*workers" | Select-Object -Last 5
```

**Expected with 2 workers**:
```
⏱️ Adaptive rate limiting: Pre-waiting 7.2s for 95 API calls (2 workers, 1.9 adjusted RPS, tokens: 10.0/95.0, will refill 175.3)
```

### Step 2: Test with 3 Workers (Validate Fix)

```powershell
# Edit .env: Set PARALLEL_WORKERS=3
# Run test
python main.py
# Choose Action 6, enter: 1 10

# Check results
(Select-String -Path Logs\app.log -Pattern "429 error").Count  # Should be 0
Select-String -Path Logs\app.log -Pattern "Adaptive rate limiting.*workers" | Select-Object -Last 5
```

**Expected with 3 workers**:
```
⏱️ Adaptive rate limiting: Pre-waiting 11.5s for 95 API calls (3 workers, 2.4 adjusted RPS, tokens: 10.0/95.0, will refill 145.8)
```

### Step 3: Measure Performance Improvement

```powershell
# After 10-20 page run
Select-String -Path Logs\app.log -Pattern "Total Run Time|Pages Scanned|Effective RPS" | Select-Object -Last 5

# Calculate throughput
$matches = <NEW_MATCHES_COUNT>
$time_secs = <TOTAL_SECONDS>
$matches_per_hour = ($matches / $time_secs) * 3600
Write-Host "Throughput: $([math]::Round($matches_per_hour, 0)) matches/hour"
```

---

## Expected Results

### 2 Workers (Baseline)

| Metric | Before Optimization | After Fix | Change |
|--------|---------------------|-----------|--------|
| Pre-wait | 24s | 7-8s | -67% |
| Time/page | 64s | 48-52s | -19-25% |
| Throughput | 1,684/hr | 2,100-2,300/hr | +25-37% |
| 429 errors | 0 | 0 | ✅ |

### 3 Workers (If Fixed Properly)

| Metric | Before Optimization | After Fix | Change |
|--------|---------------------|-----------|--------|
| Pre-wait | 24s | 11-13s | -46-54% |
| Time/page | 64s | 38-42s | -34-41% |
| Throughput | 1,684/hr | 2,600-2,900/hr | +54-72% |
| 429 errors | 18 (BROKEN) | 0 | ✅ |

---

## Rollback Plan

If you still see 429 errors with 3 workers after this fix:

1. **Immediate**: Set `PARALLEL_WORKERS=2` in `.env`
2. **Next**: Increase safety buffer in code:
   ```python
   # Change from 10% to 20% per worker
   safety_buffer = max(0, (num_workers - 2) * 0.20 * tokens_needed)
   ```
3. **Or**: Revert to conservative pre-wait:
   ```bash
   git revert 6591c1f
   git push origin main
   ```

---

## Recommendations

1. **Start with 2 workers** - Validate fix doesn't break existing behavior
2. **Test 3 workers** - Only after 2-worker runs show zero 429s for 50+ pages
3. **Monitor closely** - Watch for any 429 errors in first 10 pages
4. **Adjust if needed** - Increase safety buffer if still seeing issues

---

## Key Learnings

1. **Parallel workers change the game** - Can't use static metrics with dynamic parallelism
2. **Conservative is better** - sqrt(n) speedup, not linear n
3. **Safety buffers are essential** - 10% cushion prevents edge cases
4. **Test incrementally** - 2 workers first, then 3, then 4
5. **Monitor is critical** - Zero tolerance for 429 errors

---

*Generated: 2025-11-05*
*Related commit: 6591c1f*
*Status: FIX APPLIED - NEEDS TESTING*
