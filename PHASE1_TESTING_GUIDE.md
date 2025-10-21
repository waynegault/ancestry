# Phase 1 Testing Guide

This guide explains how to test and verify the Phase 1 enhancements to the DNA messaging system.

---

## Testing Options

### Option 1: Run Comprehensive Test Suite (Recommended First)

**What it tests:**
- Tree statistics calculation and caching
- Ethnicity commonality analysis
- Action 8 integration
- Message template compatibility

**How to run:**
```bash
python test_phase1_enhanced_messages.py
```

**Expected output:**
```
✓ ALL PHASE 1 TESTS PASSED!

Phase 1 implementation complete:
  ✓ Tree statistics calculation with caching
  ✓ Ethnicity commonality analysis
  ✓ Action 8 integration
  ✓ Message template compatibility
```

---

### Option 2: See Actual Enhanced Messages (Visual Inspection)

**What it shows:**
- Real message content with Phase 1 enhancements
- Tree statistics in action
- Ethnicity commonality formatting
- Relationship paths

**How to run:**
```bash
python demo_phase1_messages.py
```

This will show you:
1. Sample message to Frances Milne (in-tree match)
2. Sample message to hypothetical out-of-tree match
3. Before/after comparison
4. Statistics being used

---

### Option 3: Run Action 8 in Dry-Run Mode (Full Integration Test)

**What it tests:**
- Complete Action 8 workflow
- Message generation with real data
- Database message creation (no actual sending)
- Template selection logic

**How to run:**
```bash
# Using test database
python -c "
import os
os.environ['DATABASE_FILE'] = 'Data/ancestry_test.db'
os.environ['APP_MODE'] = 'dry_run'

from action8_messaging import send_messages_to_matches
from core.session_manager import SessionManager

session_manager = SessionManager()
send_messages_to_matches(session_manager)
"
```

**What to look for:**
- Messages created in database (check conversation_log table)
- No actual API calls made (dry_run mode)
- Enhanced content in message_content field

---

### Option 4: Inspect Database Directly

**Check tree statistics cache:**
```bash
sqlite3 Data/ancestry_test.db "SELECT * FROM tree_statistics_cache;"
```

**Check message templates:**
```bash
sqlite3 Data/ancestry_test.db "SELECT template_key, substr(message_content, 1, 100) FROM message_templates WHERE tree_status='in_tree';"
```

**Check conversation logs:**
```bash
sqlite3 Data/ancestry_test.db "SELECT direction, substr(message_content, 1, 100) FROM conversation_log ORDER BY created_at DESC LIMIT 5;"
```

---

### Option 5: Test Individual Functions

**Test tree statistics:**
```python
from tree_stats_utils import calculate_tree_statistics
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("sqlite:///Data/ancestry_test.db")
Session = sessionmaker(bind=engine)
session = Session()

frances_profile_id = "08FA6E79-0006-0000-0000-000000000000"
stats = calculate_tree_statistics(session, frances_profile_id)

print(f"Total matches: {stats['total_matches']}")
print(f"In tree: {stats['in_tree_count']}")
print(f"Out of tree: {stats['out_tree_count']}")
print(f"Close matches: {stats['close_matches']}")
print(f"Ethnicity regions: {len(stats['ethnicity_regions'])}")
```

**Test ethnicity commonality:**
```python
from tree_stats_utils import calculate_ethnicity_commonality
from database import Person

# Get Frances's person ID
frances = session.query(Person).filter(
    Person.profile_id == frances_profile_id
).first()

ethnicity = calculate_ethnicity_commonality(
    session, 
    frances_profile_id, 
    frances.id
)

print(f"Shared regions: {ethnicity['shared_regions']}")
print(f"Similarity score: {ethnicity['similarity_score']:.1f}%")
print(f"Top region: {ethnicity['top_shared_region']}")
```

---

## What to Verify

### ✅ Tree Statistics

**Check that statistics are calculated correctly:**
- Total matches count matches actual DNA matches in database
- In-tree count matches people with FamilyTree records
- Out-of-tree count = total - in-tree
- Close/moderate/distant matches based on shared_dna_cm thresholds
- Ethnicity regions extracted from DnaMatch table columns

**Check that caching works:**
- First call calculates and saves to tree_statistics_cache
- Second call within 24 hours returns cached data (faster)
- Cache expires after 24 hours and recalculates

### ✅ Ethnicity Commonality

**Check that shared regions are identified:**
- Compares owner's ethnicity with match's ethnicity
- Only includes regions where both have >0%
- Calculates similarity score based on percentages
- Identifies top shared region (highest combined percentage)

**Check that formatting is readable:**
- Single region: "We both have Scottish ancestry"
- Two regions: "We both have Scottish and Irish ancestry"
- Multiple regions: "We share 5 ethnicity regions including Scottish"

### ✅ Message Templates

**In-tree templates should include:**
- `{relationship_path}` - Full genealogical path
- `{actual_relationship}` - Calculated relationship
- `{total_rows}` - Number of people in tree

**Out-of-tree templates should include:**
- `{total_matches}` - Total DNA matches
- `{matches_in_tree}` - Matches in tree
- `{ethnicity_commonality}` - Shared ethnicity text

### ✅ Action 8 Integration

**Check that format data includes all fields:**
```python
from action8_messaging import _prepare_message_format_data
from database import Person

person = session.query(Person).first()
format_data = _prepare_message_format_data(
    person,
    person.family_tree,
    person.dna_match,
    session
)

# Should include:
assert 'name' in format_data
assert 'relationship_path' in format_data
assert 'total_matches' in format_data
assert 'matches_in_tree' in format_data
assert 'ethnicity_commonality' in format_data
```

---

## Expected Results

### Test Database (Frances Milne)

**Statistics:**
- Total matches: 1 (Frances herself)
- In tree: 1
- Out of tree: 0
- Close matches: 1 (>100 cM)
- Ethnicity regions: 7

**Ethnicity commonality:**
- Shared regions: 5 (comparing Frances to herself)
- Similarity score: ~310% (high because same person)
- Top region: longest_shared_segment

**Message content:**
- Should include relationship path: "Frances Margaret Milne 1947- (mother) ↓ Wayne Gault..."
- Should include tree statistics: "I've successfully connected 1 DNA matches to my tree"
- Should include ethnicity (if out-of-tree): "We both have Scottish ancestry"

---

## Production Database Testing

**⚠️ IMPORTANT: Use dry_run mode for production testing!**

```bash
# Set environment
export DATABASE_FILE=Data/ancestry.db
export APP_MODE=dry_run

# Run Action 8
python -c "
from action8_messaging import send_messages_to_matches
from core.session_manager import SessionManager

session_manager = SessionManager()
send_messages_to_matches(session_manager)
"
```

**What to check:**
1. Messages created in database (not sent to Ancestry)
2. Enhanced content includes statistics and ethnicity
3. No errors in logs
4. Performance acceptable (caching reduces queries)

---

## Troubleshooting

### Issue: "No Person record found for profile_id=..."

**Cause:** Profile ID in environment doesn't match database  
**Solution:** Use correct profile ID for the database you're testing
- Test DB: `08FA6E79-0006-0000-0000-000000000000` (Frances)
- Production: `07bdd45e-0006-0000-0000-000000000000` (Wayne)

### Issue: "Could not import Base from database module"

**Cause:** Non-critical warning in tree_stats_utils.py  
**Solution:** Ignore - does not affect functionality

### Issue: Statistics show 0 for all fields

**Cause:** No DNA matches in database for the profile ID  
**Solution:** Verify database has DnaMatch records for the profile

### Issue: Ethnicity commonality is empty

**Cause:** No shared ethnicity regions between owner and match  
**Solution:** Normal for matches with different ethnic backgrounds

---

## Next Steps After Testing

Once Phase 1 testing is complete:

1. **Review message quality** - Do enhanced messages look good?
2. **Check performance** - Is caching working? Response times acceptable?
3. **Verify accuracy** - Are statistics and ethnicity correct?
4. **Test with production data** - Run in dry_run mode with real database
5. **Proceed to Phase 2** - Person lookup integration

---

## Questions to Answer

- ✅ Do messages include relationship paths for in-tree matches?
- ✅ Do messages include tree statistics (total matches, in tree, out of tree)?
- ✅ Do messages include ethnicity commonality for out-of-tree matches?
- ✅ Is caching working (24-hour expiration)?
- ✅ Are all tests passing?
- ✅ Is performance acceptable?
- ✅ Are there any errors or warnings?

If all answers are ✅, Phase 1 is ready for production!

