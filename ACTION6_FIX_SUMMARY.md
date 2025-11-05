# Action 6 Data Integrity Fix Summary

## Issues Identified

### 1. Missing `predicted_relationship` in DNA Match Table ❌
**Problem**: All dna_match records show `predicted_relationship = "N/A"`  
**Root Cause**: `_refine_single_match()` function (line ~6020) was not extracting `predicted_relationship` from the Ancestry API match list response  
**Fix Applied**: Added extraction of `predicted_relationship` from `relationship_info.get("relationshipRange")` or `relationship_info.get("predictedRelationship")`

**Code Change** (action6_gather.py, line ~6022):
```python
# ADDED:
predicted_relationship = relationship_info.get("relationshipRange") or relationship_info.get("predictedRelationship")

# MODIFIED return dict to include:
"predicted_relationship": predicted_relationship,
```

### 2. Missing `birth_year` in People Table ❌  
**Problem**: All people records show `birth_year = NULL` even when data exists (e.g., Frances McHardy should be 1947)  
**Status**: **Needs Investigation** - The extraction logic at line 4499-4501 looks correct:
```python
if prefetched_tree_data and prefetched_tree_data.get("their_birth_year"):
    return int(prefetched_tree_data["their_birth_year"])
```
**Likely Cause**: 
- Badge API (line 6621) returns `person_badged.get("birthYear")` which might be NULL
- OR only 5/20 matches are in the user's tree (see Issue #4)

### 3. Incomplete `relationship_path` in Family Tree Table ⚠️
**Problem**: relationship_path shows "Enhanced API: 2nd cousin" instead of actual path like "You → Parent → Grandparent → ..."  
**Root Cause**: The ladder API or enhanced relationship API is not returning the full path, so the code falls back to the simplified format (lines 6702, 6715)  
**Status**: **Working As Designed** - This is a fallback when detailed path is unavailable

### 4. Only 5 Family Tree Records Instead of 19-20 ❌
**Problem**: Only 5 out of 20 matches have family_tree records  
**Root Cause**: Family tree data is only fetched for matches where `in_my_tree = true`  
**Evidence**: Line 2212-2218 in action6_gather.py:
```python
uuids_for_tree_badge_ladder = {
    match_data["uuid"]
    for match_data in matches_to_process_later
    if match_data.get("in_my_tree")  # <-- Only if in tree
    and match_data.get("uuid") in fetch_candidates_uuid
}
```
**Status**: **Working As Designed** - Only tree members get tree data

### 5. Request Counter Discrepancy (53 total, 52 successful) 🔍
**Problem**: Metrics show 53 total requests but only 52 successful  
**Root Cause**: The 303 See Other redirect (session expiration) is counted in circuit breaker's total_requests but the subsequent successful retry is counted in rate_limiter's successful_requests  
**Impact**: Minor metrics issue only, doesn't affect functionality

## Expected API Calls for 20 Matches

For 20 matches with 5 in-tree:
- **Match List API**: 1 call (fetches all 20 matches)
- **In-Tree Status API**: 1 call (batch check for all 20)  
- **Combined Details**: 20 calls (one per match for profile/relationship data)
- **Relationship Probability**: ~5 calls (high-priority matches only)
- **Badge Details**: 5 calls (only for in-tree matches)
- **Ladder Details**: 5 calls (only for in-tree matches)

**Total**: ~37 API calls  
**Actual**: 53 calls suggests some retries or additional lookups occurred

## Testing Plan

1. **Backup database**: `python main.py` → Option 3  
2. **Clear test data**: Delete the 20 test records from database
3. **Re-run Action 6 page 1**: `python main.py` → Option 6 → Enter `1`
4. **Verify fixes**:
   ```sql
   -- Check predicted_relationship
   SELECT people_id, predicted_relationship FROM dna_match LIMIT 5;
   
   -- Check birth_year  
   SELECT uuid, first_name, birth_year FROM people WHERE birth_year IS NOT NULL LIMIT 5;
   
   -- Count tree records
   SELECT COUNT(*) FROM family_tree;
   ```

## Next Steps

✅ **Fixed**: predicted_relationship extraction  
🔍 **Investigate**: Why birth_year is NULL (likely only populated for in-tree matches)  
📋 **Document**: Clarify that only 5/20 matches are in-tree (expected behavior)  
🧪 **Test**: Run Action 6 page 1 again and verify data integrity

## Questions for User

1. Are all 20 matches supposed to be in your tree, or just 5?  
2. Do you expect birth_year for non-tree matches?  
3. Is the simplified "Enhanced API: 2nd cousin" acceptable for relationship_path when detailed path unavailable?
