# Fix Summary for Remaining Issues

## Issue 1: 220 Pylance Errors

**Status**: Investigating

The pyrightconfig.json is configured to silence most immaterial errors. The 220 errors you're seeing may be:
1. VS Code needs to be reloaded to pick up the pyrightconfig.json changes
2. Errors in files not yet checked
3. Legitimate type errors that need fixing

**Action**: Please provide a sample of the pylance errors you're seeing (file names and error types) so I can address them specifically.

## Issue 2: Action11 Test Gets Stuck

**Root Cause**: The action11 tests try to start a SessionManager which requires:
- Browser automation (Selenium/ChromeDriver)
- API authentication
- Network access to Ancestry.com

When run through `run_all_tests.py`, the test hangs waiting for browser/API initialization.

**Solution**: Modify action11 tests to skip live API tests when run through the test runner, similar to how other tests handle this.

## Issue 3: Scoring Mechanism Verification

**Current Status**:
- Both Action 10 and Action 11 use `calculate_match_score` from `gedcom_utils.py`
- Action 10 calls it via `calculate_match_score_cached` wrapper
- Action 11 imports it directly

**Scoring Weights** (from config_schema.common_scoring_weights):
```
contains_first_name: 25
contains_surname: 25
gender_match: 15
year_birth: 20
exact_birth_date: 0 (not used for Fraser - no exact date)
birth_place_contains: 25
bonus_birth_info: 15 (if birth year + place both match)
year_death: 0 (Fraser is alive)
exact_death_date: 0
death_place_contains: 0
bonus_death_info: 0
bonus_both_names_contain: 25
```

**Expected Score for Fraser Gault**:
- First name (fraser): 25
- Surname (gault): 25
- Gender (m): 15
- Birth year (1941): 20
- Birth place (Banff): 25
- Birth bonus (year + place): 15
- Both names bonus: 25
**Total: 150 points**

However, the test is showing 235 points, which suggests additional scoring is happening.

**Action Required**: Need to trace through the actual scoring to see what's adding the extra 85 points.

## Recommended Actions

1. **Pylance Errors**: 
   - Reload VS Code window
   - Run: `Ctrl+Shift+P` → "Developer: Reload Window"
   - If errors persist, provide sample errors for targeted fixes

2. **Action11 Test Hanging**:
   - Modify action11 tests to detect when running in test mode
   - Skip live API tests that require SessionManager
   - Keep unit tests that don't require network/browser

3. **Scoring Verification**:
   - Run action10 test manually to see actual field scores
   - Compare with action11 scoring
   - Update .env with correct expected score

