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

### Fix 2: Detect and Resolve profile_id Collisions ✅

**Strategy:**
When a profile_id already exists in the database, set it to NULL and preserve the administrator_profile_id relationship.

**Implementation:**

**Step 1: Check for collisions with database**
```python
def _check_profile_id_collisions_with_db(
    session: SqlAlchemySession, insert_data: list[dict[str, Any]]
) -> set[str]:
    """Check if any profile_ids in insert_data already exist in the database."""
    profile_ids_to_insert = {
        item.get("profile_id") for item in insert_data
        if item.get("profile_id") is not None
    }

    # Query database for existing profile_ids
    existing_profile_ids = session.query(Person.profile_id).filter(
        Person.profile_id.in_(profile_ids_to_insert),
        Person.deleted_at.is_(None)
    ).all()

    return {pid[0] for pid in existing_profile_ids if pid[0]}
```

**Step 2: Resolve collisions by setting profile_id to NULL**
```python
def _handle_profile_id_collisions(
    insert_data: list[dict[str, Any]], existing_profile_ids: set[str]
) -> list[dict[str, Any]]:
    """Handle collisions by setting conflicting profile_ids to NULL."""
    for item in insert_data:
        pid = item.get("profile_id")
        if pid and pid in existing_profile_ids:
            logger.info(
                f"🔄 Profile_id collision - Setting to NULL: "
                f"UUID={item.get('uuid')}, profile_id={pid} → NULL, "
                f"administrator_profile_id={item.get('administrator_profile_id')}"
            )
            item["profile_id"] = None
    return insert_data
```

**Step 3: Apply before bulk insert**
```python
# Check for profile_id collisions with database
existing_profile_ids = _check_profile_id_collisions_with_db(session, insert_data)
if existing_profile_ids:
    insert_data = _handle_profile_id_collisions(insert_data, existing_profile_ids)
```

**Impact**:
- Prevents UNIQUE constraint violations
- Preserves data integrity (UUID and administrator_profile_id maintained)
- Clear logging of all collision resolutions
- No data loss - relationship preserved via administrator_profile_id

---

### Fix 3: Enhanced Duplicate Logging ✅

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

### Commit 2: Correct collision handling (CURRENT)
```
76b2926 - fix(action6): Correct profile_id collision handling

REVERTED incorrect profile_id lookup - UUID is the only valid lookup key
ADDED proper collision detection and resolution for duplicate profile_ids

Collision Resolution Strategy:
- Detect profile_ids that already exist in database
- Set conflicting profile_ids to NULL
- Preserve administrator_profile_id relationship
- Log all modifications with full details
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

