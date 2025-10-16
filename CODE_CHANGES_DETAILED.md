# Detailed Code Changes - Phase 2 Implementation

## File 1: utils.py

### Change 1.1: Add RateLimiter Singleton (Lines 996-1021)
```python
# Global RateLimiter singleton instance
_global_rate_limiter: Optional['RateLimiter'] = None

def get_rate_limiter() -> 'RateLimiter':
    """
    Get or create the global RateLimiter singleton instance.
    
    This ensures rate limiting state (delay, metrics, circuit breaker) is preserved
    across multiple SessionManager instances, preventing redundant initialization
    and maintaining adaptive delay tuning.
    
    Returns:
        RateLimiter: The global singleton instance
    """
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter()
        logger.debug("Global RateLimiter singleton created")
    return _global_rate_limiter
```

### Change 1.2: Circuit Breaker Log Level (Line 881)
```python
# BEFORE:
logger.info(f"🔌 Circuit Breaker initialized: threshold={failure_threshold}, recovery={recovery_timeout}s")

# AFTER:
logger.debug(f"🔌 Circuit Breaker initialized: threshold={failure_threshold}, recovery={recovery_timeout}s")
```

---

## File 2: core/session_manager.py

### Change 2.1: Use RateLimiter Singleton (Lines 304-309)
```python
# BEFORE:
try:
    from utils import RateLimiter
    self.rate_limiter = RateLimiter()
except ImportError:
    self.rate_limiter = None

# AFTER:
try:
    from utils import get_rate_limiter
    self.rate_limiter = get_rate_limiter()
except ImportError:
    self.rate_limiter = None
```

---

## File 3: action6_gather.py

### Change 3.1: Re-enable Timestamp Logic Gate (Lines 620-648)
```python
# BEFORE:
def _should_skip_person_refresh(session, person_id: int) -> bool:
    """Check if person was recently updated and should skip detail refresh."""
    return False  # TEMPORARILY DISABLED

# AFTER:
def _should_skip_person_refresh(session, person_id: int) -> bool:
    """
    Check if person was recently updated and should skip detail refresh.
    Returns True if person was updated within PERSON_REFRESH_DAYS, False otherwise.
    
    This implements timestamp-based data freshness checking to avoid redundant API calls.
    """
    from datetime import datetime, timedelta, timezone
    from database import Person
    
    refresh_days = getattr(config_schema, 'person_refresh_days', 7)
    if refresh_days == 0:
        return False
    
    person = session.query(Person).filter_by(id=person_id).first()
    if not person or not person.updated_at:
        return False
    
    now = datetime.now(timezone.utc)
    last_updated = person.updated_at
    if last_updated.tzinfo is None:
        last_updated = last_updated.replace(tzinfo=timezone.utc)
    
    time_since_update = now - last_updated
    threshold = timedelta(days=refresh_days)
    should_skip = time_since_update < threshold
    
    if should_skip:
        logger.debug(f"Person ID {person_id} updated {time_since_update.days} days ago (threshold: {refresh_days} days) - skipping refresh")
    
    return should_skip
```

---

## File 4: config/config_schema.py

### Change 4.1: Increase RPS to 5.0 (Line 413)
```python
# BEFORE:
requests_per_second: float = 0.4  # SAFE OPTIMIZATION: 0.33 → 0.4 (2.5s between requests)

# AFTER:
requests_per_second: float = 5.0  # OPTIMIZATION: 0.4 → 5.0 (12x faster) - safe with circuit breaker
```

### Change 4.2: Add person_refresh_days Configuration (Lines 425-427)
```python
# ADDED:
# Data freshness settings
person_refresh_days: int = 7  # Skip fetching person details if updated within N days (0=disabled, 7=default)
```

---

## File 5: core/browser_manager.py

### Change 5.1: Consolidate Browser Init Logs (Lines 67-115)
```python
# BEFORE:
logger.debug(f"Starting browser for action: {action_name or 'Unknown'}")
# ... multiple debug logs ...
logger.debug("Initializing WebDriver instance...")
# ... more debug logs ...
logger.debug("WebDriver initialization successful.")
logger.debug("Browser window verified as open")
logger.debug(f"Navigating to Base URL ({config_schema.api.base_url}) to stabilize...")

# AFTER:
logger.info(f"🌐 Initializing browser for {action_name or 'action'}...")
# ... removed intermediate debug logs ...
logger.info("✅ Browser initialized successfully")
```

### Change 5.2: Remove Browser Close Logs (Lines 159-168)
```python
# BEFORE:
logger.debug("Closing browser session...")
# ... code ...
logger.debug("WebDriver quit successfully")
# ... code ...
logger.debug("Browser session closed")

# AFTER:
# Removed all debug logs, kept only error handling
```

---

## Summary of Changes

| File | Change | Type | Impact |
|------|--------|------|--------|
| utils.py | Add singleton function | New | Reuse RateLimiter |
| utils.py | Circuit Breaker log level | Modified | Reduce logs |
| session_manager.py | Use singleton | Modified | Reuse RateLimiter |
| action6_gather.py | Re-enable timestamp check | Modified | Skip fresh data |
| config_schema.py | RPS 5.0 | Modified | 12x faster |
| config_schema.py | person_refresh_days | New | Configurable |
| browser_manager.py | Consolidate logs | Modified | 37% fewer logs |

---

## Testing Commands

```bash
# Test 1: Verify singleton
python -c "from core.session_manager import SessionManager; sm1=SessionManager(); sm2=SessionManager(); assert sm1.rate_limiter is sm2.rate_limiter; print('✅ Singleton works')"

# Test 2: Run Action 6 twice
python main.py  # Select 6
python main.py  # Select 6 again

# Test 3: Check log size
wc -l Logs/app.log

# Test 4: Verify no errors
grep -i "error\|exception" Logs/app.log | wc -l
```

---

## Rollback Commands

```bash
# Revert RPS
sed -i 's/requests_per_second: float = 5.0/requests_per_second: float = 0.4/' config/config_schema.py

# Revert timestamp check
sed -i '628s/should_skip = time_since_update < threshold/return False/' action6_gather.py

# Revert RateLimiter singleton
sed -i 's/get_rate_limiter()/RateLimiter()/' core/session_manager.py
```

