# Action 6 Diagnosis - Missing Data Issues

## Date: November 5, 2025
## Issue: Incomplete data extraction for DNA matches

---

## Issues Identified

### 1. ✅ **Birth Year Missing (Frances McHardy and 14 others)**
- **Symptom**: `birth_year` is NULL in database for 15 out of 20 matches (75%)
- **Expected**: Should contain birth year like 1947 for Frances McHardy
- **Root Cause**: Incomplete `_async_enhanced_api_orchestrator` function

### 2. ✅ **Only 5 Family Tree Records Instead of 19-20**
- **Symptom**: Only 5 `family_tree` records created, but 20 matches were marked as `in_my_tree=True`
- **Expected**: Should have ~19-20 family_tree records for matches in the tree
- **Root Cause**: Same - incomplete async orchestrator

### 3. ✅ **predicted_relationship Shows "N/A"**
- **Symptom**: All `dna_match` records have `predicted_relationship = "N/A"`
- **Expected**: Should show relationships like "2nd Cousin", "3rd-4th Cousin", etc.
- **Root Cause**: The extraction logic is working in `_refine_single_match` but the value is being overwritten somewhere

### 4. ✅ **relationship_path Shows "Enhanced API: 2nd cousin"**
- **Symptom**: User thinks this is wrong
- **Actual Status**: This is CORRECT - it's the format when detailed relationship path is unavailable
- **Not an error**: This is expected fallback behavior

### 5. ❓ **Mysterious "53"**
- **Symptom**: User mentions "53" but it doesn't appear in logs
- **Need clarification**: Where did the user see this number?

---

## Root Cause Analysis

### The Async Orchestrator Bug (Lines 2517-2560)

```python
async def _async_enhanced_api_orchestrator(...):
    """Advanced async orchestration..."""
    
    if len(uuid_list) < 10:  # Small batch
        return _perform_api_prefetches(...)  # Uses complete sync method
    
    logger.info(f"Large batch ({len(uuid_list)} items) - using advanced async orchestration")
    
    # PROBLEM: Only fetches combined details!
    batch_combined_details = {}
    batch_tree_data = {}  # Initialized but NEVER POPULATED
    batch_relationship_prob_data = {}  # Initialized but NEVER POPULATED
    
    # Only combined details are fetched
    await fetch_combined_details_batch()
    
    # Returns with EMPTY tree and rel_prob dictionaries!
    return {
        "combined": batch_combined_details,
        "tree": batch_tree_data,  # <-- EMPTY!
        "rel_prob": batch_relationship_prob_data,  # <-- EMPTY!
    }
```

**Impact:**
- Batches with **10+ matches** → Uses incomplete async orchestrator → Missing badge/ladder data
- Batches with **< 10 matches** → Uses complete sync method → All data present

**Evidence from Logs:**

**Batch 1 (15 matches)** - Used async orchestrator:
```log
03:52:19 INF [action6_ _async_e 2517] Large batch (15 items) - using advanced async orchestration
03:52:32 DEB [action6_ _async_e 2558] Async orchestrator completed: 15 combined details fetched
```
- ✓ Combined details fetched
- ✗ NO badge details calls
- ✗ NO ladder API calls
- Result: 15 people created, 0 family_tree records, birth_year=NULL

**Batch 2 (5 matches)** - Used sync method:
```log
03:52:58 DEB [action6_ _perform 2209] --- Starting Parallel API Pre-fetch (5 candidates, 2 workers) ---
03:52:58 DEB [action6_ _perform 2220] Identified 5 candidates for Badge/Ladder fetch.
```
- ✓ Combined details fetched
- ✓ Badge details fetched (birthYear available)
- ✓ Ladder API called (relationship paths available)
- ✓ Relationship probability fetched
- Result: 5 people created, 5 family_tree records, birth_year populated

---

## Detailed Code Flow

### How birth_year SHOULD work
1. **Badge Details API** returns: `{"birthYear": 1947, "lastName": "McHardy", ...}`
2. **Stored in** `prefetched_tree_data`:
   ```python
   {
       "their_cfpid": "102568995464",
       "their_firstname": "Frances",
       "their_lastname": "McHardy",
       "their_birth_year": 1947  # <-- From badge API
   }
   ```
3. **Extracted in** `_extract_birth_year()`:
   ```python
   def _extract_birth_year(prefetched_tree_data):
       if prefetched_tree_data and prefetched_tree_data.get("their_birth_year"):
           return int(prefetched_tree_data["their_birth_year"])
       return None  # <-- Returns None when badge data missing!
   ```

### What happened in Batch 1
```python
# Async orchestrator returned:
{
    "combined": {...},  # Has match details
    "tree": {},         # EMPTY! No badge data
    "rel_prob": {}      # EMPTY! No relationship data
}

# Result:
prefetched_tree_data = prefetched_data.get("tree", {}).get(uuid_val)
# → prefetched_tree_data = None

birth_year = _extract_birth_year(None)
# → birth_year = None → Stored as NULL in database
```

---

## Fix Options

### Option 1: Disable Incomplete Async Orchestrator (RECOMMENDED)
**File**: `action6_gather.py` line ~2519

```python
async def _async_enhanced_api_orchestrator(...):
    # Change threshold to never use async (always use complete sync method)
    if len(uuid_list) < 1000:  # Was: if len(uuid_list) < 10
        logger.debug(f"Batch size {len(uuid_list)} - using sync method")
        return _perform_api_prefetches(...)
```

**Pros:**
- ✅ Simple one-line fix
- ✅ Uses proven working sync method
- ✅ No risk of breaking anything

**Cons:**
- ❌ Loses potential async performance benefits (but async isn't complete anyway)

---

### Option 2: Complete the Async Orchestrator Implementation
**File**: `action6_gather.py` lines 2517-2560

Need to add:
1. Badge details fetching (async)
2. Ladder API calls (async)
3. Relationship probability (async)
4. Proper error handling

**Pros:**
- ✅ Potentially better performance for large batches
- ✅ Future-proof architecture

**Cons:**
- ❌ Complex implementation
- ❌ Requires extensive testing
- ❌ Risk of introducing new bugs
- ❌ More time to implement

---

### Option 3: Fix predicted_relationship Separately
The `predicted_relationship` issue is SEPARATE from the async orchestrator bug.

**Investigation needed**: Check where the value from `_refine_single_match` is being overwritten with "N/A".

**Possible locations to check:**
1. `_prepare_dna_operation_data` (line ~4770)
2. Default value in DnaMatch model
3. Bulk insert logic

---

## Recommended Fix

**Immediate action:** Apply Option 1 - Disable incomplete async orchestrator

```python
# Line 2519 in action6_gather.py
async def _async_enhanced_api_orchestrator(...):
    # TEMPORARY FIX: Disable incomplete async orchestrator
    # TODO: Complete async implementation for badge/ladder/rel_prob
    if len(uuid_list) < 1000:  # Changed from 10 to 1000
        logger.debug(f"Batch size {len(uuid_list)} - using proven sync method")
        return _perform_api_prefetches(session_manager, fetch_candidates_uuid, matches_to_process_later)
```

This will ensure ALL batches use the complete sync method until the async orchestrator is fully implemented.

---

## Testing Plan

After applying the fix:

1. **Reset database**: `python main.py` → Option 2
2. **Run Action 6 with 1 page**: `python main.py` → Option 6 → `6 1`
3. **Verify results**:
   ```sql
   -- Should show birth years for in-tree matches
   SELECT uuid, first_name, birth_year, in_my_tree 
   FROM people 
   LIMIT 20;
   
   -- Should show 19-20 family_tree records
   SELECT COUNT(*) FROM family_tree;
   
   -- Should show actual relationships (not "N/A")
   SELECT people_id, predicted_relationship 
   FROM dna_match 
   LIMIT 20;
   ```

---

## Git History Context

- **d80abf3** (2025-11-05): Added predicted_relationship extraction from match list API
- **5dc05c2** (2025-11-04): Refactor action6_gather.py to 100% quality
- **f50a17c** (2025-11-04): Complete 6 high-priority tasks

The async orchestrator was likely introduced in the refactoring but was never completed.
