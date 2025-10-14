# Action 6 Diagnosis - 2025-10-14

## Executive Summary

**Test Run**: MAX_PAGES=5 (20:03:02 - 20:15:51)
**Result**: ✅ Pages 1 & 2 completed successfully, ❌ Page 3 failed mid-processing
**Database**: 40 people saved (20 from page 1, 20 from page 2)
**Root Cause**: Match Probability API started returning HTTP 303 redirects instead of JSON

---

## Problems Identified

### Problem 1: Match Probability API Returning 303 Redirects ❌ CRITICAL

**Evidence:**
```
20:15:38 DEB [action6_ _fetch_b 4913] <-- Match Probability API (Cloudscraper) Response Status: 303 See Other
20:15:38 ERR [action6_ _process 3544] Match Probability API (Cloudscraper): OK (303), but JSON decode FAILED
20:15:38 DEB [action6_ _process 3547] Response text: See Other
20:15:38 ERR [action6_ _fetch_b 4933] Error processing successful response: JSONDecodeError
20:15:41 ERR [utils    _handle_ 630 ] API Call failed after 3 retries for '_fetch_batch_relationship_prob'
```

**Pattern:**
- First 2 matches on page 3 worked fine (200 OK responses at 20:13:05, 20:13:08)
- Starting with 3rd match, API returns 303 redirects
- All subsequent calls fail with same error
- Log ends abruptly after multiple failures

**Impact:**
- Process likely crashed due to unhandled exception
- Page 3 was NOT saved to database
- No graceful degradation

**Root Cause:**
- API endpoint may have changed
- Session/cookies may have become invalid
- CSRF token may be stale (warning: "Using potentially stale CSRF from SessionManager")

---

### Problem 2: No Graceful Degradation for API Failures ❌ HIGH

**Evidence:**
- Match Probability API is marked as optional in code
- But failures cause process to crash instead of continuing
- No "Page 3 Batch Summary" in logs (compare to pages 1 & 2)

**Expected Behavior:**
- API failure should log warning
- Processing should continue without relationship probability data
- Page should still be saved to database

**Actual Behavior:**
- Process crashes/exits
- No completion message
- No final summary

---

### Problem 3: Proactive Browser Refresh Not Triggered ✅ EXPECTED

**Evidence:**
- No "🔄 Proactive browser refresh" messages in log
- This is expected since we only processed 2.5 pages (refresh triggers at page 10)

**Status:** Working as designed

---

### Problem 4: Session Recovery Not Needed ✅ EXPECTED

**Evidence:**
- No "Skipping session recovery" messages
- No "invalid session id" errors
- Browser session remained valid throughout

**Status:** Working as designed

---

## What Worked ✅

1. **Profile_id Collision Handling** - No UNIQUE constraint errors
2. **Timeout Fix** - No timeout errors (4-hour limit not reached)
3. **Pages 1 & 2** - Completed successfully with 40 people saved
4. **Database Operations** - All bulk operations successful
5. **Rate Limiting** - Token bucket working correctly
6. **Session Validity** - Browser session remained valid

---

## Recommended Fixes

### Fix 1: Make Match Probability API Truly Optional (HIGH PRIORITY)

**Current Code Issue:**
The API is supposed to be optional but failures cause crashes.

**Solution:**
1. Wrap Match Probability API calls in try/except
2. Log warning if it fails
3. Continue processing without relationship probability data
4. Set `relationship_prob` field to NULL in database

**Files to Modify:**
- `action6_gather.py` - `_fetch_batch_relationship_prob()` function
- Add better error handling around cloudscraper calls

---

### Fix 2: Investigate Match Probability API 303 Redirects (MEDIUM PRIORITY)

**Possible Causes:**
1. **Stale CSRF Token** - Warning shows "Using potentially stale CSRF from SessionManager"
2. **Session Expired** - Cookies may have become invalid
3. **API Endpoint Changed** - Ancestry may have updated the endpoint
4. **Rate Limiting** - API may be rejecting requests (though no 429 errors)

**Investigation Steps:**
1. Check if CSRF token is being refreshed properly
2. Verify cookie sync is working for cloudscraper
3. Test API endpoint manually with curl
4. Check if API requires different headers

---

### Fix 3: Add Better Error Logging for Process Crashes (LOW PRIORITY)

**Current Issue:**
- Log ends abruptly with no traceback
- No indication of why process stopped
- No final summary or cleanup

**Solution:**
1. Add top-level exception handler in `coord()` function
2. Log full traceback on unexpected errors
3. Always log final summary even on failure
4. Add signal handlers for Ctrl+C, system kill, etc.

---

## Testing Plan

### Phase 1: Fix Match Probability API Error Handling
1. Make API truly optional
2. Test with MAX_PAGES=5
3. Verify all 5 pages complete even if API fails
4. Verify 100 people saved to database

### Phase 2: Investigate 303 Redirects
1. Add debug logging for CSRF token refresh
2. Add debug logging for cookie sync
3. Test with MAX_PAGES=5
4. Check if relationship probability data is retrieved

### Phase 3: Full Test
1. Run with MAX_PAGES=50
2. Verify proactive refresh at page 10, 20, 30, 40, 50
3. Verify all pages complete successfully
4. Verify 1000 people saved to database

---

## Outstanding Tasks from Task List

### Task 1: Fix duplicate page processing bug ⏸️ ON HOLD
**Status:** Not observed in current test run
**Reason:** May have been fixed by sequential processing changes
**Action:** Monitor in future runs

### Task 2: Fix WebDriver session death and recovery skipping ✅ FIXED
**Status:** Fixed in commit 52f154b
**Evidence:** No session recovery issues in current run
**Action:** Continue monitoring

### Task 3: Test New Implementation ⏳ IN PROGRESS
**Status:** Currently testing with MAX_PAGES=5
**Result:** Partial success (2/5 pages completed)
**Action:** Fix Match Probability API issue and retest

---

## Log Analysis Summary

**Timeline:**
- 20:03:02 - Action 6 started
- 20:07:56 - Page 1 completed (20 people, 0 errors)
- 20:12:29 - Page 2 completed (20 people, 0 errors)
- 20:13:01 - Page 3 started processing
- 20:13:05 - First 2 matches processed successfully
- 20:15:38 - Match Probability API starts returning 303 redirects
- 20:15:51 - Log ends abruptly (process crashed/killed)

**Duration:** ~13 minutes for 2.5 pages
**Rate:** ~5 minutes per page (expected for sequential processing)

**Database State:**
- 40 people saved (pages 1 & 2)
- 0 UNIQUE constraint errors
- 0 duplicate UUIDs
- All records have proper profile_id assignment

---

## Conclusion

The Action 6 improvements are working well:
- ✅ Profile_id collision handling is perfect
- ✅ Timeout fix is working
- ✅ Sequential processing is stable
- ❌ Match Probability API needs better error handling

**Next Steps:**
1. Fix Match Probability API error handling (make it truly optional)
2. Investigate 303 redirect issue
3. Retest with MAX_PAGES=5
4. If successful, test with MAX_PAGES=50

