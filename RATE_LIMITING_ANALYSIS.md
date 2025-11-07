# Rate Limiting System Analysis and Redesign

**Date:** November 7, 2025  
**Author:** GitHub Copilot (Analysis Assistant)  
**Status:** Design Phase

## Executive Summary

Action 6 experiences three critical failures:
1. **Catastrophic hangs** (7+ hour API calls without timeout enforcement)
2. **Rate limiting oscillation** (three competing delay systems fighting each other)
3. **Session death without recovery** (browser dies, recovery fails)

This document provides root cause analysis, detailed design for unified rate limiting, and implementation roadmap.

---

## Part 1: Root Cause Analysis

### Issue 1: Timeout Protection Failure

**Symptom:** API call took 26,831 seconds (7.45 hours) without timeout

**Root Causes:**

1. **Thread-based timeout can't kill operations**
   ```python
   # error_handling.py line 961
   thread.daemon = True
   thread.start()
   thread.join(timeout)
   if thread.is_alive():
       raise NetworkTimeoutError(...)  # Thread keeps running!
   ```

   **Problem:** Python threads cannot be forcibly terminated. When timeout fires:
   - Decorator raises exception and returns
   - Background thread continues making HTTP request
   - Thread holds references to browser, session, sockets
   - Eventually OS kills browser (7 hours later)
   - Recovery mechanisms fail because session is corrupted

2. **Single-value timeout parameter**
   ```python
   # utils.py line 2321
   return req_session.request(**request_params)
   # request_params contains: timeout=30
   ```

   **Problem:** Single timeout value in requests doesn't protect against:
   - TCP connection hangs (connection "established" but no data)
   - OS-level network issues (socket in retransmit state)
   - Server accepting connection but never responding

   **Solution:** Tuple timeout `(connect_timeout, read_timeout)` provides dual protection:
   - `connect_timeout`: Max time to establish TCP connection
   - `read_timeout`: Max time waiting for response after connection

3. **No operation-level watchdog**
   - System lacks monitoring for hung operations
   - No circuit breaker for "too slow" operations
   - Session health checks don't detect hung API calls

### Issue 2: Rate Limiting Oscillation

**Symptom:** Continuous 429 errors despite adaptive delays increasing to 0.86s

**Root Causes:**

1. **Dual-delay stacking**
   ```python
   # utils.py line 1469 (RateLimiter.wait)
   if self.tokens >= 1.0:
       self.tokens -= 1.0
       base_sleep = self.current_delay  # ← PROBLEM: Extra delay
       sleep_duration = min(base_sleep * jitter_factor, self.max_delay)
   ```

   **Problem:** Token bucket should control rate, but adds current_delay on top:
   - Token bucket enforces: 1/fill_rate seconds between requests
   - PLUS current_delay (starts 0.1s, grows to 0.86s+)
   - Effective delay = max(1/fill_rate, current_delay) + jitter

   **Example with fill_rate=3.0:**
   - Token bucket alone: 0.33s between requests
   - With current_delay=0.86s: 0.86s between requests
   - Token bucket becomes irrelevant, current_delay dominates

2. **Competing increase/decrease mechanisms**
   ```python
   # On 429 error:
   def increase_delay(self):
       self.current_delay = min(
           self.current_delay * self.backoff_factor,  # 1.8x increase
           self.max_delay
       )
   
   # On every success:
   def decrease_delay(self):
       self.current_delay = self.current_delay * self.decrease_factor  # 0.98x decrease
   ```

   **Problem:** Creates oscillation:
   - 429 error → delay increases 1.8x (e.g., 0.1 → 0.18 → 0.32 → 0.86s)
   - Works for a few requests → decreases 0.98x per success
   - After ~35 successes, back to causing 429s
   - Cycle repeats endlessly, never reaching equilibrium

   **Logs show pattern:**
   ```
   Page 2: 429 → 0.10s → 0.18s → 0.32s
   Page 5: 429 → 0.10s → 0.18s → 0.32s  ← Reset by decrease_delay()
   Page 7: 429 → 0.10s → 0.18s → 0.32s
   ```

3. **Exponential backoff adds third delay system**
   ```python
   # utils.py line 2454
   sleep_time = min(
       current_delay * (backoff_factor ** (attempt - 1)),
       max_delay
   ) + random.uniform(0, 0.2)
   # First retry: 0.45s, Second retry: 10.87s
   ```

   **Problem:** Three delays stack:
   - Token bucket wait
   - PLUS current_delay
   - PLUS exponential backoff (on retry)

   **Total delay on 429 retry:**
   ```
   Token: 0.33s (1/fill_rate)
   + Current: 0.32s (adaptive increase)
   + Backoff: 10.87s (exponential)
   = 11.52 seconds for one retry!
   ```

### Issue 3: Configuration Confusion

**Current .env:**
```env
PARALLEL_WORKERS=1
REQUESTS_PER_SECOND=3
TOKEN_BUCKET_FILL_RATE=3.0
INITIAL_DELAY=0.3
```

**Problems:**

1. **Conflicting rate definitions**
   - REQUESTS_PER_SECOND=3 suggests 3 req/s
   - But TOKEN_BUCKET_FILL_RATE=3.0 (redundant?)
   - And INITIAL_DELAY=0.3 (3.33 req/s?)
   - Which one actually controls the rate?

2. **Formula unclear**
   - README says: effective_rps = workers × rps
   - With workers=1, rps=3: effective = 3.0 req/s
   - But token bucket + current_delay makes actual rate ~1.16 req/s
   - Users can't predict actual rate from config

3. **Still hitting 429s**
   - Logs show constant 429 errors
   - Suggests Ancestry's real limit < 1 req/s
   - Config claims "safe" but demonstrably isn't

---

## Part 2: Unified Rate Limiting Design

### Design Principles

1. **Single source of truth:** One parameter controls rate (fill_rate)
2. **Adaptive learning:** System adjusts fill_rate based on feedback
3. **Conservative speedup:** Requires many successes before speeding up
4. **Aggressive slowdown:** Decreases significantly on 429
5. **No oscillation:** Long stabilization period prevents fighting
6. **Separation of concerns:** Retry backoff separate from rate control

### New AdaptiveRateLimiter Class

```python
class AdaptiveRateLimiter:
    """
    Unified adaptive rate limiter using token bucket algorithm.
    
    Core Concept:
    - fill_rate (tokens/second) is the ONLY rate control
    - Token bucket handles bursts naturally
    - Adaptive adjustment modifies fill_rate based on API feedback
    - No extra delays, no competing mechanisms
    
    Adaptive Logic:
    - 429 error → decrease fill_rate by 20% (slow down significantly)
    - Success → increase fill_rate by 1% after 100 consecutive successes
    - Separate immediate retry backoff from system-wide rate
    
    Thread Safety:
    - All mutations protected by threading.Lock()
    - Safe for concurrent API calls from multiple threads
    """
    
    def __init__(
        self,
        initial_fill_rate: float = 0.5,  # Start conservative
        capacity: float = 10.0,          # Burst capacity
        min_fill_rate: float = 0.1,      # Never slower than 10s between requests
        max_fill_rate: float = 2.0,      # Never faster than 2 req/s
    ):
        # Token bucket state
        self.capacity = capacity
        self.fill_rate = initial_fill_rate
        self.tokens = capacity
        self.last_refill_time = time.monotonic()
        
        # Rate bounds
        self.min_fill_rate = min_fill_rate
        self.max_fill_rate = max_fill_rate
        
        # Adaptive learning state
        self.success_count = 0
        self.success_threshold = 100  # Successes required before speedup
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Metrics
        self._metrics = {
            'total_requests': 0,
            'total_wait_time': 0.0,
            'rate_decreases': 0,
            'rate_increases': 0,
            'error_429_count': 0,
        }
    
    def wait(self) -> float:
        """
        Wait according to token bucket algorithm.
        
        Returns:
            float: Time spent waiting (seconds)
        """
        with self._lock:
            self._refill_tokens()
            
            if self.tokens >= 1.0:
                # Token available, consume it
                self.tokens -= 1.0
                wait_time = 0.0
                logger.debug(
                    f"Token consumed: {self.tokens:.2f}/{self.capacity} remaining"
                )
            else:
                # Wait for token to generate
                wait_time = (1.0 - self.tokens) / self.fill_rate
                logger.debug(
                    f"Token bucket empty, waiting {wait_time:.3f}s for refill "
                    f"(rate: {self.fill_rate:.3f} req/s)"
                )
                time.sleep(wait_time)
                self._refill_tokens()
                self.tokens -= 1.0
            
            # Update metrics
            self._metrics['total_requests'] += 1
            self._metrics['total_wait_time'] += wait_time
            
            return wait_time
    
    def on_429_error(self) -> None:
        """
        Handle 429 rate limit error by decreasing fill_rate.
        
        Decreases by 20% to quickly back off from rate limit.
        Resets success counter to prevent premature speedup.
        """
        with self._lock:
            old_rate = self.fill_rate
            self.fill_rate = max(
                self.fill_rate * 0.80,  # 20% decrease
                self.min_fill_rate
            )
            self.success_count = 0  # Reset success streak
            
            # Update metrics
            self._metrics['error_429_count'] += 1
            self._metrics['rate_decreases'] += 1
            
            logger.warning(
                f"⚠️ 429 Rate Limit: Decreased rate from {old_rate:.3f} to "
                f"{self.fill_rate:.3f} req/s (20% reduction)"
            )
    
    def on_success(self) -> None:
        """
        Handle successful API call.
        
        Increases fill_rate by 1% only after 100 consecutive successes.
        This prevents oscillation and ensures stable operation.
        """
        with self._lock:
            self.success_count += 1
            
            if self.success_count >= self.success_threshold:
                old_rate = self.fill_rate
                self.fill_rate = min(
                    self.fill_rate * 1.01,  # 1% increase
                    self.max_fill_rate
                )
                self.success_count = 0  # Reset counter
                
                # Update metrics
                self._metrics['rate_increases'] += 1
                
                # Only log if rate actually changed
                if abs(old_rate - self.fill_rate) > 0.001:
                    logger.info(
                        f"✅ After {self.success_threshold} successes: "
                        f"Increased rate to {self.fill_rate:.3f} req/s "
                        f"(+1% from {old_rate:.3f})"
                    )
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time and fill_rate."""
        now = time.monotonic()
        elapsed = max(0.0, now - self.last_refill_time)
        tokens_to_add = elapsed * self.fill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill_time = now
    
    def get_metrics(self) -> dict:
        """Get current metrics for monitoring."""
        with self._lock:
            metrics = self._metrics.copy()
            metrics['current_fill_rate'] = self.fill_rate
            metrics['success_count'] = self.success_count
            metrics['tokens_available'] = self.tokens
            if metrics['total_requests'] > 0:
                metrics['avg_wait_time'] = (
                    metrics['total_wait_time'] / metrics['total_requests']
                )
            else:
                metrics['avg_wait_time'] = 0.0
            return metrics
```

### Integration Points

1. **Replace current RateLimiter in utils.py**
   - Keep singleton pattern via get_rate_limiter()
   - Update SessionManager to use new class
   - Remove old increase_delay/decrease_delay calls

2. **Update API call handling**
   ```python
   # Before API call
   rate_limiter.wait()
   
   # After success
   rate_limiter.on_success()
   
   # After 429 error
   rate_limiter.on_429_error()
   ```

3. **Keep exponential backoff for retries**
   ```python
   # Separate from rate limiting
   # Only for immediate retry of same request
   # Does NOT affect rate_limiter.fill_rate
   for attempt in range(1, max_retries + 1):
       response = make_request()
       if response.status_code == 429:
           backoff_time = min(0.5 * (2 ** attempt), 30)
           time.sleep(backoff_time)
           continue
   ```

### Expected Behavior

**Scenario 1: Clean Run (No 429s)**
```
Initial: fill_rate=0.5 req/s (2s between requests)
Request 1-100: All succeed, success_count increments
Request 101: success_count=100 → fill_rate=0.505 (+1%)
Request 201: success_count=100 → fill_rate=0.510 (+1%)
...continues gradually increasing until max_fill_rate=2.0
```

**Scenario 2: Hit Rate Limit**
```
Initial: fill_rate=1.0 req/s
Request 50: 429 error → fill_rate=0.8 (-20%), success_count=0
Request 51-150: All succeed
Request 151: success_count=100 → fill_rate=0.808 (+1%)
Request 251: success_count=100 → fill_rate=0.816 (+1%)
...gradually approaches stable rate without oscillation
```

**Scenario 3: Repeated 429s**
```
fill_rate=1.0 → 429 → 0.8 → 429 → 0.64 → 429 → 0.512 → stable
Aggressive slowdown quickly finds safe rate
```

---

## Part 3: Timeout Enforcement

### Problem Analysis

Current timeout=30 doesn't protect against:
- TCP-level hangs
- OS network issues  
- Server accepting but not responding

### Solution: Tuple Timeout

```python
# Before
timeout=30  # Single value for connect AND read

# After
timeout=(10, 30)  # (connect_timeout, read_timeout)
```

**Protections:**
- Connect timeout: 10s max to establish TCP connection
- Read timeout: 30s max waiting for response
- Total max time: 40s (10 + 30)

### Watchdog Timer

```python
class APICallWatchdog:
    """
    Monitors API calls and triggers emergency restart if hung.
    
    Purpose: Catch operations that bypass request timeouts
    (e.g., browser-based calls, TCP hangs, OS issues)
    """
    
    def __init__(self, timeout_seconds: float = 120):
        self.timeout_seconds = timeout_seconds
        self.timer: Optional[threading.Timer] = None
        self.is_active = False
        self._lock = threading.Lock()
    
    def start(self, api_name: str, callback: Callable) -> None:
        """Start watchdog timer for API call."""
        with self._lock:
            if self.is_active:
                logger.warning(
                    f"Watchdog already active, cancelling previous timer"
                )
                self.cancel()
            
            def timeout_handler():
                logger.critical(
                    f"🚨 WATCHDOG TIMEOUT: {api_name} exceeded "
                    f"{self.timeout_seconds}s limit"
                )
                callback()
            
            self.timer = threading.Timer(
                self.timeout_seconds,
                timeout_handler
            )
            self.timer.daemon = True
            self.timer.start()
            self.is_active = True
            
            logger.debug(
                f"Watchdog started for {api_name} "
                f"(timeout: {self.timeout_seconds}s)"
            )
    
    def cancel(self) -> None:
        """Cancel watchdog timer."""
        with self._lock:
            if self.timer:
                self.timer.cancel()
                self.timer = None
            self.is_active = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cancel()
        return False
```

**Usage:**
```python
def make_api_call_with_watchdog(api_name: str):
    watchdog = APICallWatchdog(timeout_seconds=120)
    
    def emergency_restart():
        session_manager._force_session_restart(
            reason=f"Watchdog timeout on {api_name}"
        )
    
    watchdog.start(api_name, emergency_restart)
    try:
        response = requests.get(url, timeout=(10, 30))
        return response
    finally:
        watchdog.cancel()
```

---

## Part 4: Implementation Roadmap

### Phase 1: Foundation (No Behavioral Changes)

**Goal:** Add infrastructure without changing existing behavior

1.1. Create AdaptiveRateLimiter class in new file `rate_limiter.py`
1.2. Add comprehensive tests for AdaptiveRateLimiter
1.3. Create APICallWatchdog class in `core/session_manager.py`
1.4. Add tests for APICallWatchdog
1.5. Run all existing tests - ensure 100% pass

**Success Criteria:**
- All new code has >90% test coverage
- All existing tests still pass
- No behavioral changes to Action 6
- Linting/Pylance clean

### Phase 2: Timeout Protection

**Goal:** Prevent catastrophic hangs

2.1. Add timeout tuples to utils._execute_api_request
2.2. Add timeout tuples to cloudscraper calls in action6_gather.py
2.3. Add watchdog to SessionManager._make_api_call wrapper
2.4. Implement_force_session_restart method
2.5. Add tests: mock 60s call, verify timeout at 40s
2.6. Run Action 6 with 10 pages, verify no hangs

**Success Criteria:**
- No API call exceeds 120s (watchdog timeout)
- Timeout tuples prevent TCP hangs
- Session restart works on timeout
- All tests pass

### Phase 3: Rate Limiter Migration

**Goal:** Switch to unified rate limiter

3.1. Add get_adaptive_rate_limiter() factory in rate_limiter.py
3.2. Update SessionManager to use AdaptiveRateLimiter
3.3. Update utils._apply_rate_limiting to call new limiter
3.4. Update utils._handle_status_code_response for 429s
3.5. Remove old RateLimiter.increase_delay calls
3.6. Keep old RateLimiter class (deprecated) for comparison
3.7. Run side-by-side comparison: 10 pages with old vs new

**Success Criteria:**
- New limiter produces same or fewer 429s
- No oscillation in logs
- fill_rate converges to stable value
- All tests pass

### Phase 4: Configuration Cleanup

**Goal:** Single clear configuration

4.1. Update .env with new simplified settings
4.2. Remove: TOKEN_BUCKET_CAPACITY, INITIAL_DELAY, BACKOFF_FACTOR
4.3. Keep: REQUESTS_PER_SECOND (becomes fill_rate)
4.4. Add validation: warn if rate > 1.0
4.5. Update config_schema.py with new structure
4.6. Update README with clear explanations

**Success Criteria:**
- Configuration is self-explanatory
- Users can predict actual rate from settings
- Validation prevents unsafe configs
- Documentation accurate

### Phase 5: Monitoring & Validation

**Goal:** Comprehensive observability

5.1. Add rate limiter metrics dashboard
5.2. Add 429 rate monitoring
5.3. Add API duration histogram
5.4. Create metrics CSV output
5.5. Run 200-page validation test
5.6. Analyze metrics, tune defaults

**Success Criteria:**
- <10 total 429 errors in 200 pages
- Zero session deaths
- Zero watchdog triggers
- fill_rate stabilizes

---

## Part 5: Testing Strategy

### Unit Tests

```python
# test_adaptive_rate_limiter.py

def test_token_bucket_enforces_rate():
    """Verify token bucket prevents requests faster than fill_rate."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=2.0, capacity=10.0)
    
    start = time.time()
    for _ in range(20):
        limiter.wait()
    elapsed = time.time() - start
    
    # Should take ~10 seconds (20 requests / 2.0 req/s)
    # Allow 20% margin for timing variance
    assert 8.0 <= elapsed <= 12.0, f"Expected ~10s, got {elapsed:.1f}s"

def test_429_decreases_rate():
    """Verify 429 error decreases fill_rate by 20%."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=1.0)
    
    assert limiter.fill_rate == 1.0
    limiter.on_429_error()
    assert limiter.fill_rate == 0.8  # 20% decrease
    limiter.on_429_error()
    assert limiter.fill_rate == 0.64  # 20% decrease again

def test_success_requires_100_before_increase():
    """Verify success doesn't speed up until 100 consecutive successes."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=1.0)
    
    # First 99 successes: no change
    for _ in range(99):
        limiter.on_success()
    assert limiter.fill_rate == 1.0
    
    # 100th success: increases by 1%
    limiter.on_success()
    assert limiter.fill_rate == 1.01

def test_429_resets_success_count():
    """Verify 429 error resets success counter."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=1.0)
    
    # 50 successes
    for _ in range(50):
        limiter.on_success()
    assert limiter.success_count == 50
    
    # 429 error resets
    limiter.on_429_error()
    assert limiter.success_count == 0

def test_watchdog_triggers_on_timeout():
    """Verify watchdog triggers callback after timeout."""
    triggered = threading.Event()
    
    def callback():
        triggered.set()
    
    watchdog = APICallWatchdog(timeout_seconds=0.5)
    watchdog.start("test_api", callback)
    
    # Wait for timeout
    triggered.wait(timeout=1.0)
    assert triggered.is_set(), "Watchdog should have triggered"
    
    watchdog.cancel()

def test_watchdog_cancels_on_success():
    """Verify watchdog doesn't trigger if cancelled."""
    triggered = threading.Event()
    
    def callback():
        triggered.set()
    
    watchdog = APICallWatchdog(timeout_seconds=0.5)
    watchdog.start("test_api", callback)
    
    # Cancel immediately
    watchdog.cancel()
    
    # Wait past timeout
    time.sleep(1.0)
    assert not triggered.is_set(), "Watchdog should not have triggered"
```

### Integration Tests

```python
# test_action6_with_new_rate_limiter.py

def test_action6_10_pages_no_429s():
    """Run Action 6 for 10 pages and verify no rate limit errors."""
    # Set conservative rate
    config_schema.api.requests_per_second = 0.5
    
    # Run Action 6
    result = gather_dna_matches(
        session_manager=session_manager,
        start_page=1,
        max_pages=10
    )
    
    # Check logs for 429 errors
    with open('Logs/app.log', 'r') as f:
        log_content = f.read()
    
    error_429_count = log_content.count('429')
    assert error_429_count == 0, f"Found {error_429_count} 429 errors"
    
    # Verify completion
    assert result['pages_processed'] == 10
    assert result['session_deaths'] == 0

def test_timeout_tuple_prevents_hang():
    """Verify timeout tuple prevents TCP-level hangs."""
    # Mock server that accepts connection but never responds
    mock_server = create_hanging_server()
    
    start = time.time()
    try:
        response = requests.get(
            mock_server.url,
            timeout=(5, 10)  # 5s connect, 10s read
        )
        pytest.fail("Should have timed out")
    except requests.Timeout:
        pass  # Expected
    
    elapsed = time.time() - start
    # Should timeout within 15s (5 + 10), not hang indefinitely
    assert elapsed < 20, f"Took {elapsed:.1f}s, should timeout faster"
```

### Regression Tests

```python
# test_rate_limiter_regression.py

def test_no_oscillation():
    """Verify new limiter doesn't oscillate like old one."""
    limiter = AdaptiveRateLimiter(initial_fill_rate=1.0)
    
    # Simulate pattern: 429, then successes
    limiter.on_429_error()  # → 0.8
    assert limiter.fill_rate == 0.8
    
    # 100 successes
    for _ in range(100):
        limiter.on_success()
    assert limiter.fill_rate == 0.808  # Only 1% increase
    
    # Should NOT cycle back to causing 429s
    # Old system would have increased 35x at 2% each = back to 1.0
```

---

## Part 6: Rollback Plan

### Pre-Implementation Checklist

- [ ] Create git branch: `feature/unified-rate-limiting`
- [ ] Document current commit: `git rev-parse HEAD`
- [ ] Backup .env file
- [ ] Backup database: `cp Data/ancestry.db Data/ancestry_backup_$(date +%Y%m%d).db`
- [ ] Run full test suite: `python run_all_tests.py`
- [ ] Document current metrics (capture baseline):
  - Average time per page
  - 429 error rate
  - Session death rate

### Rollback Procedure

If issues arise at any phase:

1. **Stop Action 6 immediately**
2. **Check logs for errors:** `tail -100 Logs/app.log`
3. **Revert code:**
   ```bash
   git reset --hard HEAD  # Discard uncommitted changes
   git checkout main      # Return to main branch
   ```
4. **Restore config:**
   ```bash
   cp .env.backup .env
   ```
5. **Verify rollback:**
   ```bash
   python run_all_tests.py
   python action6_gather.py  # Run 1 page test
   ```

### Phase-Specific Rollback

**Phase 1:** Just delete new files (rate_limiter.py), no other changes
**Phase 2:** Revert timeout changes, keep rest
**Phase 3:** Switch back to old RateLimiter in SessionManager
**Phase 4:** Restore old .env settings
**Phase 5:** Remove monitoring, keep fixes

---

## Part 7: Success Metrics

### Critical Metrics (Must Achieve)

1. **Zero catastrophic hangs**
   - No API call exceeds 120 seconds
   - Watchdog doesn't trigger in normal operation

2. **<10 429 errors per 1000 requests**
   - Error rate < 1%
   - Demonstrates stable rate limiting

3. **Zero session deaths**
   - No browser crashes
   - No recovery failures

4. **fill_rate convergence**
   - Stabilizes within 200 pages
   - Doesn't oscillate

### Performance Metrics (Target)

1. **Throughput: 1,500-2,000 matches/hour**
   - With fill_rate=0.5: ~1,500/hour
   - With fill_rate=1.0: ~2,000/hour

2. **Stability: 8+ hour runs**
   - Can process 16,000 matches overnight
   - No intervention required

3. **Predictability**
   - Time per page variance < 20%
   - Users can estimate completion time

### Code Quality Metrics

1. **Test coverage: >90%** for new code
2. **All Pylance errors resolved**
3. **All Ruff linting errors resolved**
4. **Function complexity < 10** (McCabe)
5. **Documentation complete** (docstrings, README, design doc)

---

## Conclusion

This design provides:
- **Root cause analysis** of all three critical failures
- **Unified rate limiting** that eliminates oscillation
- **Proper timeout enforcement** to prevent hangs
- **Comprehensive testing** strategy
- **Incremental implementation** roadmap
- **Rollback plan** for safety

Implementation will proceed phase by phase with testing at each step.

Next step: Begin Phase 1 - Create AdaptiveRateLimiter with tests.
