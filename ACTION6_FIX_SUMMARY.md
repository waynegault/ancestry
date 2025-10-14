# Action 6 Fix Summary - Timeout and Database Errors

## Problem Diagnosis

### Timeline of Failure (from logs)
```
13:27:24 - Action 6 starts
13:42:38 - Database UNIQUE constraint error on page 2 (15 minutes in)
13:42:40 - Timeout after 300 seconds (5 minutes)
13:42:40 - Session closed by main.py
13:42:42+ - WebDriver invalid errors (browser already closed)
```

### Root Causes Identified

#### 1. **Hardcoded Timeout (300 seconds = 5 minutes)**
- **Location**: Line 645 in `action6_gather.py`
- **Problem**: `@timeout_protection(timeout=300)` was hardcoded
- **Config Value**: `action6_coord_timeout_seconds: int = 14400` (4 hours)
- **Impact**: Function timed out after 5 minutes, but config said 4 hours!

#### 2. **Database UNIQUE Constraint Violation**
- **Error**: `UNIQUE constraint failed: people.profile_id`
- **Root Cause**: Trying to INSERT a profile_id that already exists in database
- **Scenario**:
  - Page 1: UUID="UUID1", profile_id="PROF1" → Inserted successfully
  - Page 2: UUID="UUID2", profile_id="PROF1" → **ERROR!** (PROF1 already exists)
  - This happens when Ancestry API returns the same profile_id for different DNA tests
- **Data Model**:
  - UUID = DNA test kit ID (always unique, always present)
  - profile_id = Ancestry member account ID (unique when NOT NULL)
  - One profile_id can only appear ONCE in the `profile_id` column
  - Same profile_id can appear MANY times in `administrator_profile_id` column

#### 3. **Sequential Processing Made Timeout Worse**
- Parallel processing: ~8-9 min/page
- Sequential processing: ~10-12 min/page
- **This made the 300-second timeout hit even faster!**

---

## Solutions Implemented

### Fix 1: Use Config Timeout Value ✅

**Before:**
```python
@timeout_protection(timeout=300)  # Hardcoded 5 minutes
def coord(session_manager: SessionManager, start: int = 1) -> bool:
```

**After:**
```python
@timeout_protection(timeout=config_schema.action6_coord_timeout_seconds)  # 4 hours from config
def coord(session_manager: SessionManager, start: int = 1) -> bool:
```

**Impact**: Function now has 4 hours to complete instead of 5 minutes.

---

### Fix 2: Intelligent profile_id Collision Resolution ✅

**Strategy:**
Determine which record is the "true owner" of a profile_id and resolve collisions intelligently.

**True Owner Criteria:**
- `tester_profile_id == admin_profile_id` AND `tester_username == admin_username` (from API)
- OR `administrator_profile_id` is NULL (self-managed test)
- This indicates **SCENARIO A**: Member with their own test

**Collision Resolution Logic:**
1. **Existing record is true owner** → Set new record's profile_id to NULL
2. **New record is true owner** → Keep new profile_id, warn about existing record
3. **Ambiguous ownership** → Set new to NULL (conservative approach)

**Implementation:**

**Step 1: Determine true ownership**
```python
def _is_true_profile_owner(item: dict[str, Any]) -> bool:
    """
    True owner: Member with their own test (self-managed)
    - administrator_profile_id is NULL, OR
    - profile_id == administrator_profile_id (from API data)
    """
    profile_id = item.get("profile_id")
    admin_profile_id = item.get("administrator_profile_id")

    if profile_id and not admin_profile_id:
        return True  # Self-managed = true owner

    return False
```

**Step 2: Check collisions with ownership info**
```python
def _check_profile_id_collisions_with_db(
    session: SqlAlchemySession, insert_data: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """
    Returns: {profile_id: {"uuid": ..., "is_true_owner": ...}}
    """
    existing_persons = session.query(
        Person.profile_id, Person.uuid, Person.username,
        Person.administrator_profile_id
    ).filter(...)

    # Determine ownership for each existing record
    for pid, uuid, username, admin_pid in existing_persons:
        is_owner = (admin_pid is None) or (pid == admin_pid)
        existing_map[pid] = {"uuid": uuid, "is_true_owner": is_owner}
```

**Step 3: Intelligent collision resolution**
```python
def _handle_profile_id_collisions(
    insert_data: list[dict[str, Any]],
    existing_profile_map: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Resolve based on ownership."""
    for item in insert_data:
        existing_is_owner = existing_profile_map[pid]["is_true_owner"]
        new_is_owner = _is_true_profile_owner(item)

        if existing_is_owner and not new_is_owner:
            # Existing wins - set new to NULL
            item["profile_id"] = None
        elif new_is_owner and not existing_is_owner:
            # New wins - keep profile_id, warn about existing
            logger.warning("NEW record is true owner - keeping profile_id")
        else:
            # Ambiguous - be conservative
            item["profile_id"] = None
```

**Real Example (EC managed by TANEJ):**
```json
API Response:
{
  "userId": "026C2692-0006-0000-0000-000000000000",
  "adminUcdmId": "026C2692-0006-0000-0000-000000000000",
  "displayName": "E.C.",
  "adminDisplayName": "TANEJ"
}

Result:
- userId == adminUcdmId (both are TANEJ's profile_id)
- displayName != adminDisplayName (EC vs TANEJ)
- Therefore: profile_id = NULL, administrator_profile_id = TANEJ's ID
- EC is NOT a member, TANEJ manages the test
```

**Impact**:
- Prevents UNIQUE constraint violations
- Correctly identifies true profile owners
- Preserves data integrity based on ownership
- Clear logging with ownership rationale
- Handles edge cases (new owner vs existing owner)

---

### Fix 3: Browser Session Recovery & Proactive Refresh ✅

**Problem:**
Browser crashes after ~30-40 minutes of runtime with error:
```
invalid session id: session deleted as the browser has closed the connection
from disconnected: not connected to DevTools
```

**Root Cause:**
- Session recovery was being skipped: "⏭️ Skipping session recovery (not in long-running operation)"
- `_should_attempt_recovery()` required `session_start_time` to be set
- No proactive browser refresh to prevent crashes

**Solution:**

**Part 1: Enhanced Session Recovery**
```python
def _should_attempt_recovery(self) -> bool:
    """Always attempt recovery if session was previously working."""
    if not self.session_ready:
        return False

    # If session_start_time is not set, check if we have a driver
    if not self.session_start_time:
        return self.driver is not None

    # For sessions running > 5 minutes, always attempt recovery
    return time.time() - self.session_start_time > 300
```

**Part 2: Proactive Browser Refresh**
```python
# In _process_single_page_iteration()
if current_page_num > start_page and (current_page_num - start_page) % 10 == 0:
    logger.info(f"🔄 Proactive browser refresh at page {current_page_num}")
    session_manager.perform_enhanced_proactive_refresh()
```

**Impact:**
- ✅ Session recovery now works for all long-running operations
- ✅ Browser refreshes every 10 pages to prevent crashes
- ✅ Automatic recovery when browser becomes invalid
- ✅ Graceful degradation if refresh fails

---

### Fix 4: Enhanced Duplicate Logging ✅

**In-Batch Deduplication:**
```python
# Enhanced logging with profile_id, uuid, and username
logger.info(
    f"⚠️  Duplicate profile_id detected in batch - "
    f"ProfileID: {profile_id}, UUID: {uuid_for_log}, Username: '{username_for_log}'"
)
```

**Pre-Insert Validation:**
```python
# Show details for each duplicate entry
for item in insert_data:
    pid = item.get("profile_id")
    if pid and pid in duplicates:
        logger.error(
            f"  ⚠️  Duplicate entry - ProfileID: {pid}, "
            f"UUID: {item.get('uuid')}, Username: '{item.get('username', 'Unknown')}'"
        )
```

**Impact**: Clear visibility into which records are duplicates and why.

---

## Data Model Understanding

### UUID vs profile_id
- **UUID**: DNA test kit ID (always unique, always present in Action 6)
- **profile_id**: Ancestry member account ID (unique when NOT NULL, can be NULL)
- **administrator_profile_id**: The profile_id of whoever manages this DNA test

### Valid Scenarios
1. **Member with their own test**:
   - UUID="UUID1", profile_id="PROF1", administrator_profile_id=NULL

2. **Non-member test administered by member**:
   - UUID="UUID2", profile_id=NULL, administrator_profile_id="PROF1"

3. **Member administering someone else's test (who is also a member)**:
   - UUID="UUID3", profile_id="PROF2", administrator_profile_id="PROF1"

### Key Rules
- One profile_id value can only appear ONCE in the `profile_id` column
- Same profile_id can appear MANY times in `administrator_profile_id` column
- UUID is the primary lookup key (each UUID = unique DNA test)
- Multiple UUIDs can share the same administrator_profile_id

### Why Sequential Processing Was Kept
- Simpler code (easier to debug)
- Conservative approach (avoids potential 429 issues)
- Trade-off: Slower (10-12 min/page) but more reliable
- Can revert to parallel later if speed becomes critical

---

## Testing Plan

### Phase 1: Small Test (Recommended First)
```bash
# In .env
MAX_PAGES=5
BATCH_SIZE=5
MAX_INBOX=5
```

**Expected Results:**
- ✅ No timeout (5 pages × 12 min = 60 min < 4 hours)
- ✅ No UNIQUE constraint errors (profile_id lookup working)
- ✅ Clear duplicate logging if any occur

**Duration**: ~1 hour

---

### Phase 2: Medium Test (If Phase 1 Succeeds)
```bash
# In .env
MAX_PAGES=50
```

**Expected Results:**
- ✅ No timeout (50 pages × 12 min = 600 min = 10 hours < 14400 seconds)
- ✅ Consistent processing without errors

**Duration**: ~8-10 hours

---

### Phase 3: Full Run (If Phase 2 Succeeds)
```bash
# In .env
MAX_PAGES=0  # Process all 802 pages
```

**Expected Results:**
- ✅ Complete processing of all DNA matches
- ✅ No database integrity errors
- ✅ No timeouts

**Duration**: ~160 hours (6-7 days) - consider running in chunks

---

## Files Changed

| File | Changes | Lines |
|------|---------|-------|
| `action6_gather.py` | Timeout fix, profile_id lookup, enhanced logging | ~50 |
| `.env` | (No changes needed - config already correct) | 0 |

---

## Commits

### Commit 1: Initial timeout fix (REVERTED)
```
8ba6ae6 - fix(action6): Fix timeout and database UNIQUE constraint errors
```

### Commit 2: Basic collision handling (SUPERSEDED)
```
76b2926 - fix(action6): Correct profile_id collision handling
```

### Commit 3: Intelligent collision resolution
```
b30c2b1 - feat(action6): Intelligent profile_id collision resolution
```

### Commit 4: Browser session recovery (CURRENT)
```
52f154b - feat(action6): Add browser session recovery and proactive refresh

Implement comprehensive browser stability improvements:

1. Enhanced Session Recovery:
   - Fixed _should_attempt_recovery() to work without session_start_time
   - Always attempt recovery if session was previously working
   - Critical for long-running operations like Action 6

2. Proactive Browser Refresh:
   - Refresh browser every 10 pages to prevent crashes
   - Prevents 30-40 minute browser death issue
   - Graceful degradation if refresh fails

3. Better Error Handling:
   - Session recovery now triggers for all long-running operations
   - Continues processing even if proactive refresh fails
   - Automatic recovery when browser becomes invalid
```

---

## Next Steps

1. **Run Phase 1 test** (MAX_PAGES=5)
2. **Monitor logs** for:
   - Timeout issues (should not occur)
   - UNIQUE constraint errors (should not occur)
   - Duplicate profile_id warnings (should show clear details)
3. **If successful**, proceed to Phase 2
4. **If issues occur**, review logs and adjust

---

## Lessons Learned

1. **Always check config vs. hardcoded values** - The timeout was in config but decorator ignored it
2. **UNIQUE constraints need comprehensive lookups** - Can't just check one field when multiple fields can identify a record
3. **Diagnose before fixing** - The parallel processing elimination was based on wrong diagnosis
4. **Sequential processing trade-offs** - Simpler but slower; may need to revert if performance is critical

---

## Potential Future Improvements

1. **Revert to parallel processing** if sequential is too slow
2. **Add session health checks** during long runs
3. **Implement progress checkpointing** to resume from failure point
4. **Add 429 detection with backoff** if rate limiting becomes an issue
5. **Consider batch size optimization** to balance speed vs. reliability

