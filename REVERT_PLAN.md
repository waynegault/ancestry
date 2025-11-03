# Action 6 Revert and Re-apply Plan

## Current State (Commit: 00992f5)
- ✅ Ethnicity data saves correctly
- ✅ Temporary backups removed
- ✅ Duplicate startup checks removed
- ✅ rate_limiter attribute added to SessionManager
- ❌ Action 6: Multiple progress bars, no batch/page summaries
- ❌ Actions 7 & 8: Missing check_browser_health method

## Target Revert Point
**Commit 1b10e73** - "✅ Action 6 successful test run with reverted 793d948 base"
- This was right after the surgical revert to 793d948
- Before we started removing "dead code"
- Action 6 worked perfectly: 40 new, 0 updated, 0 skipped, 0 errors

## What We Must Preserve (Re-apply After Revert)

### 1. Ethnicity Data Fix (Commit 83b1146)
**Files:** action6_gather.py, database.py

**action6_gather.py changes:**

**Location 1: Lines 2693-2725 (Bulk INSERT)**
```python
# Separate ethnicity columns from core data (bulk_insert_mappings doesn't handle dynamic columns)
core_insert_data = []
ethnicity_updates = []  # List of (people_id, ethnicity_dict) tuples

for insert_map in dna_insert_data:
    core_map = {k: v for k, v in insert_map.items() if not k.startswith("ethnicity_")}
    ethnicity_map = {k: v for k, v in insert_map.items() if k.startswith("ethnicity_")}
    core_insert_data.append(core_map)
    if ethnicity_map:
        ethnicity_updates.append((insert_map["people_id"], ethnicity_map))

# Bulk insert core data
session.bulk_insert_mappings(DnaMatch, core_insert_data)
session.flush()

# Apply ethnicity data via raw SQL UPDATE
if ethnicity_updates:
    from sqlalchemy import text
    for people_id, ethnicity_data in ethnicity_updates:
        set_clauses = ", ".join([f"{col} = :{col}" for col in ethnicity_data])
        sql = f"UPDATE dna_match SET {set_clauses} WHERE people_id = :people_id"
        params = {**ethnicity_data, "people_id": people_id}
        session.execute(text(sql), params)
    session.flush()
    logger.debug(f"Applied ethnicity data to {len(ethnicity_updates)} newly inserted DnaMatch records")
```

**Location 2: Lines 2727-2761 (Bulk UPDATE)**
```python
# Separate ethnicity columns from core data (bulk_update_mappings doesn't handle dynamic columns)
core_update_mappings = []
ethnicity_updates = []  # List of (id, ethnicity_dict) tuples

for update_map in dna_update_mappings:
    core_map = {k: v for k, v in update_map.items() if not k.startswith("ethnicity_")}
    ethnicity_map = {k: v for k, v in update_map.items() if k.startswith("ethnicity_")}
    core_update_mappings.append(core_map)
    if ethnicity_map:
        ethnicity_updates.append((update_map["id"], ethnicity_map))

# Bulk update core data
session.bulk_update_mappings(DnaMatch, core_update_mappings)
session.flush()

# Apply ethnicity data via raw SQL UPDATE
if ethnicity_updates:
    from sqlalchemy import text
    for match_id, ethnicity_data in ethnicity_updates:
        set_clauses = ", ".join([f"{col} = :{col}" for col in ethnicity_data])
        sql = f"UPDATE dna_match SET {set_clauses} WHERE id = :id"
        params = {**ethnicity_data, "id": match_id}
        session.execute(text(sql), params)
    session.flush()
    logger.debug(f"Applied ethnicity data to {len(ethnicity_updates)} updated DnaMatch records")
```

**Location 3: Lines 3671-3683 (Dict Comprehension Fix)**
```python
# Remove keys with None values *except* for predicted_relationship which we want to store as NULL if it's None
# Also keep internal keys like _operation and uuid
# IMPORTANT: Keep ethnicity columns even if value is 0 (0% is valid ethnicity data)
return {
    k: v
    for k, v in dna_dict_base.items()
    if v is not None
    or k == "predicted_relationship"  # Explicitly keep predicted_relationship even if None
    or k.startswith("_")  # Keep internal keys
    or k == "uuid"  # Keep uuid
    or k.startswith("ethnicity_")  # Keep ethnicity columns even if 0 or None
}
```

**database.py changes:**
- Lines 1571-1599: Debug logging for ethnicity operations (optional - can skip)

### 2. SessionManager rate_limiter Fix (Commit 00992f5)
**File:** core/session_manager.py
**Location:** Lines 162-168

```python
# Initialize rate limiter (use global singleton for all API calls)
try:
    from utils import get_rate_limiter
    self.rate_limiter = get_rate_limiter()
except ImportError:
    self.rate_limiter = None
```

### 3. Startup Checks Cleanup (Commit 7b1e549)
**File:** main.py
**Location:** Lines 2065-2077

Simplified `_check_startup_status()` to only check database (removed redundant cookie/token checks)

## Revert Steps

1. **Backup current state** (already done - commit 00992f5)

2. **Revert action6_gather.py to 1b10e73**
   ```bash
   git show 1b10e73:action6_gather.py > action6_gather.py
   ```

3. **Test Action 6** - verify it works and has proper formatting

4. **Re-apply ethnicity fix** - manually apply the 3 changes above

5. **Test ethnicity** - verify data saves correctly

6. **Commit** - "Revert Action 6 to 1b10e73 + re-apply ethnicity fix"

7. **Test Actions 7 & 8** - see if they work now

## Expected Outcome
- ✅ Action 6 works with proper batch/page summaries
- ✅ Ethnicity data saves correctly
- ✅ Actions 7 & 8 work (check_browser_health should be available)
- ✅ Clean, maintainable codebase

## Rollback Plan
If this doesn't work, revert to commit 00992f5:
```bash
git reset --hard 00992f5
```

