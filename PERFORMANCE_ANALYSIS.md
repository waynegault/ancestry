# Action 6 Performance Analysis

## Current Performance (2 Workers @ 2.5 RPS)

### Metrics from Latest Run
- **Duration**: 8 minutes 9 seconds (489 seconds)
- **Total API Requests**: 676
- **Matches Processed**: 200 (189 updated, 11 skipped)
- **Pages Processed**: 10
- **Zero Errors**: ✅ No 429 errors, no database errors, no attribute errors

### Rate Limiter Performance
- **Configured RPS**: 2.50/s
- **Effective RPS**: 1.29/s (51% utilization)
- **Average Wait Time**: 0.501s
- **Total Wait Time**: 338.37s (out of 489s total = 69% of time spent waiting)

### Throughput
- **Matches per minute**: ~24.5 (200 matches / 8.15 minutes)
- **API calls per match**: 3.38 (676 / 200)
- **Time per batch (10 matches)**: ~24 seconds average

## Performance Bottleneck Analysis

### Why Only 51% RPS Utilization?

The effective RPS (1.29/s) is only 51% of configured RPS (2.5/s). This indicates:

1. **Parallel Workers Underutilized**: With 2 workers, we should be able to make 2 concurrent requests
2. **Sequential Database Operations**: Database saves happen in main thread (by design for thread-safety)
3. **Batch Processing Overhead**: Time spent between batches (fetching match list, processing results)

### Time Breakdown (Estimated)
- **API Fetching**: ~338s (rate limiter wait time)
- **Database Operations**: ~100s (estimated - sequential saves)
- **Overhead**: ~51s (page navigation, batch setup, etc.)
- **Total**: 489s

## Optimization Opportunities

### Option 1: Increase Workers to 3
**Rationale**: Current workers are not saturating the rate limiter

**Expected Impact**:
- Effective RPS: 1.29 → ~1.9/s (75% utilization)
- Duration: 489s → ~380s (22% faster)
- Matches per minute: 24.5 → ~31.5

**Risk**: LOW - Rate limiter is thread-safe and proven at 2 workers

**Recommendation**: ✅ **SAFE TO TRY**

### Option 2: Increase RPS to 3.0
**Rationale**: We're only using 51% of current capacity

**Expected Impact**:
- Effective RPS: 1.29 → ~1.55/s (52% utilization at 3.0 RPS)
- Duration: 489s → ~450s (8% faster)
- Matches per minute: 24.5 → ~26.5

**Risk**: MEDIUM - Higher RPS increases 429 error risk

**Recommendation**: ⚠️ **Test with caution**

### Option 3: Increase Both (3 Workers @ 3.0 RPS)
**Rationale**: Maximize throughput while staying under API limits

**Expected Impact**:
- Effective RPS: 1.29 → ~2.25/s (75% utilization)
- Duration: 489s → ~320s (35% faster)
- Matches per minute: 24.5 → ~37.5

**Risk**: MEDIUM-HIGH - More aggressive, but still conservative

**Recommendation**: ⚠️ **Test incrementally**

### Option 4: Increase to 4 Workers @ 3.5 RPS
**Rationale**: Push closer to theoretical limits

**Expected Impact**:
- Effective RPS: 1.29 → ~2.6/s (74% utilization)
- Duration: 489s → ~280s (43% faster)
- Matches per minute: 24.5 → ~42.8

**Risk**: HIGH - Approaching API limits, may trigger 429 errors

**Recommendation**: ❌ **Too aggressive - not recommended yet**

## Recommended Testing Sequence

### Phase 1: Increase Workers Only ✅ RECOMMENDED
```ini
# .env
PARALLEL_WORKERS=3
REQUESTS_PER_SECOND=2.5
```

**Why**: Safest option - proven RPS, just adding one more worker

**Expected**: ~22% faster, zero 429 errors

**Test**: Run 10 pages, monitor for errors

### Phase 2: Increase RPS Slightly (if Phase 1 succeeds)
```ini
# .env
PARALLEL_WORKERS=3
REQUESTS_PER_SECOND=3.0
```

**Why**: Moderate increase, still conservative

**Expected**: ~35% faster than baseline

**Test**: Run 10 pages, monitor for 429 errors

### Phase 3: Push Further (if Phase 2 succeeds)
```ini
# .env
PARALLEL_WORKERS=4
REQUESTS_PER_SECOND=3.5
```

**Why**: Maximize throughput

**Expected**: ~43% faster than baseline

**Test**: Run 10 pages, watch for 429 errors

## Performance Targets

### Conservative (Phase 1)
- **Workers**: 3
- **RPS**: 2.5
- **Expected Duration**: ~6.5 minutes (for 10 pages)
- **Expected Throughput**: ~31 matches/minute
- **Risk**: Very Low

### Moderate (Phase 2)
- **Workers**: 3
- **RPS**: 3.0
- **Expected Duration**: ~5.5 minutes (for 10 pages)
- **Expected Throughput**: ~36 matches/minute
- **Risk**: Low

### Aggressive (Phase 3)
- **Workers**: 4
- **RPS**: 3.5
- **Expected Duration**: ~4.7 minutes (for 10 pages)
- **Expected Throughput**: ~42 matches/minute
- **Risk**: Medium

## Monitoring Checklist

For each test run, verify:
- [ ] Zero 429 errors in rate limiter metrics
- [ ] Zero database concurrency errors
- [ ] Zero attribute errors
- [ ] Effective RPS stays below configured RPS
- [ ] All batches commit successfully
- [ ] Final summary shows expected match counts

## Current Status

✅ **Baseline Established**: 2 workers @ 2.5 RPS = 24.5 matches/minute, zero errors

🎯 **Next Step**: Increase to 3 workers @ 2.5 RPS (Phase 1)

📊 **Goal**: Achieve 35-40 matches/minute with zero errors

