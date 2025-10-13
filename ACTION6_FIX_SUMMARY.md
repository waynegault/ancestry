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
- **Root Cause**: Lookup only checked UUID, not profile_id
- **Scenario**:
  - Person A: profile_id="ABC123", uuid="UUID1" (already in DB)
  - Person B: profile_id="ABC123", uuid="UUID2" (new match on page)
  - Lookup by UUID doesn't find Person A
  - Code tries to INSERT Person B with profile_id="ABC123" → UNIQUE constraint error!

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

### Fix 2: Query by BOTH UUID and profile_id ✅

**Before:**
```python
def _lookup_existing_persons(
    session: SqlAlchemySession, uuids_on_page: list[str]
) -> dict[str, Person]:
    # Only queried by UUID
    existing_persons = (
        session.query(Person)
        .filter(Person.uuid.in_(uuids_upper), Person.deleted_at.is_(None))
        .all()
    )
```

**After:**
```python
def _lookup_existing_persons(
    session: SqlAlchemySession, 
    uuids_on_page: list[str], 
    profile_ids_on_page: Optional[list[str]] = None
) -> dict[str, Person]:
    # Query by UUID OR profile_id
    filters = []
    if uuids_upper:
        filters.append(Person.uuid.in_(uuids_upper))
    if profile_ids_upper:
        filters.append(Person.profile_id.in_(profile_ids_upper))
    
    if filters:
        from sqlalchemy import or_
        query = query.filter(or_(*filters))
        existing_persons = query.all()
```

**Call Site Update:**
```python
# Extract both UUIDs and profile_ids from matches
uuids_on_page = [m["uuid"].upper() for m in matches_on_page if m.get("uuid")]
profile_ids_on_page = [m.get("profile_id") for m in matches_on_page if m.get("profile_id")]
existing_persons_map = _lookup_existing_persons(session, uuids_on_page, profile_ids_on_page)
```

**Impact**: 
- Finds existing persons even if they have different UUID but same profile_id
- Treats them as UPDATE instead of INSERT
- Prevents UNIQUE constraint violations

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

## Why Sequential Processing Was a Red Herring

### Original Hypothesis (WRONG)
- Thought parallel processing caused 429 API errors
- Eliminated ThreadPoolExecutor to fix rate limiting

### Reality
- 429 errors were NOT the problem in this run
- Database errors and timeout were the real issues
- Sequential processing made timeout problem WORSE (slower = hits timeout faster)

### Decision: Keep Sequential for Now
- Simpler code (easier to debug)
- Conservative approach (avoids potential 429 issues)
- Trade-off: Slower but more reliable
- Can revert to parallel later if needed

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

## Commit

```
8ba6ae6 - fix(action6): Fix timeout and database UNIQUE constraint errors

- Use config timeout (4 hours) instead of hardcoded 300 seconds
- Query existing persons by BOTH UUID and profile_id to prevent duplicates
- Enhanced duplicate logging with profile_id, uuid, and username
- Prevents UNIQUE constraint violations when same profile_id has different UUID
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

