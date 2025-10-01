# Manual Testing Guide - Actions 5-11
**Date:** 2025-09-30  
**Purpose:** Guide for manually testing Actions 5-11 through main.py

---

## Prerequisites

Before testing, ensure:
1. ✅ All code quality improvements applied
2. ✅ All automated tests passing (385/388, 99.2%)
3. ✅ System in dry_run mode (safe for testing)
4. ⚠️ **You have valid Ancestry.com login credentials**

---

## Starting the Application

```bash
cd C:\Users\wayne\GitHub\Python\Projects\Ancestry
python main.py
```

The application will display a menu with options 0-11.

---

## Testing Sequence

### Step 1: Check Login Status (Action 5)

**Purpose:** Verify browser session and authentication

**Steps:**
1. Select option `5` from the menu
2. The system will:
   - Check if browser session is active
   - Verify cookies are valid
   - Confirm authentication status

**Expected Results:**
- ✅ "Login status: Active" or similar confirmation
- ✅ Profile ID displayed
- ✅ Session details shown

**If Login Required:**
- System will prompt you to log in
- Browser window will open
- Log in to Ancestry.com manually
- System will capture session

**Troubleshooting:**
- If login fails, check internet connection
- Verify Ancestry.com is accessible
- Check .env file has correct credentials

---

### Step 2: GEDCOM Report (Action 10)

**Purpose:** Test local GEDCOM file analysis (no login required)

**Steps:**
1. Select option `10` from the menu
2. Enter search criteria when prompted:
   - **Given Name:** Fraser
   - **Surname:** Gault
   - **Birth Year:** 1941
   - **Birth Place:** Banff
   - **Gender:** M

**Expected Results:**
- ✅ Search completes in ~30-60 seconds
- ✅ Fraser Gault found with score of 160 points
- ✅ Family details displayed:
  - Parents: James Gault, Dolly Clara Alexina Fraser
  - Siblings: 10 siblings listed
  - Spouse: Nellie Mason Smith
  - Children: David, Caroline, Barry
- ✅ Relationship path to tree owner shown

**Troubleshooting:**
- If GEDCOM file not found, check .env GEDCOM_FILE_PATH
- If search is slow, this is normal (14,640 individuals to search)
- If no results, verify search criteria matches .env test data

---

### Step 3: API Report (Action 11)

**Purpose:** Test live API search (requires login)

**Prerequisites:**
- ✅ Action 5 completed successfully (logged in)
- ✅ Browser session active

**Steps:**
1. Select option `11` from the menu
2. Enter same search criteria as Action 10:
   - **Given Name:** Fraser
   - **Surname:** Gault
   - **Birth Year:** 1941
   - **Birth Place:** Banff
   - **Gender:** M

**Expected Results:**
- ✅ API search completes quickly (< 10 seconds)
- ✅ Fraser Gault found via API
- ✅ Similar family details as Action 10
- ✅ Relationship path displayed
- ✅ Results should match Action 10 (same person, same family)

**Troubleshooting:**
- If "Session not ready" error, run Action 5 first
- If API errors, check rate limits in .env
- If no results, verify API access is working

---

### Step 4: Search Inbox (Action 7)

**Purpose:** Test inbox message processing

**Prerequisites:**
- ✅ Action 5 completed successfully (logged in)
- ✅ Browser session active

**Steps:**
1. Select option `7` from the menu
2. System will:
   - Fetch conversations from Ancestry inbox
   - Process messages
   - Classify message intents using AI
   - Update database

**Expected Results:**
- ✅ Conversations fetched successfully
- ✅ Progress bar shows processing
- ✅ Messages classified (productive/non-productive)
- ✅ Database updated with conversation logs
- ✅ Summary displayed:
  - Conversations processed
  - AI classifications
  - Database updates

**Troubleshooting:**
- If no conversations found, inbox may be empty
- If AI classification fails, check AI service configuration
- If database errors, check database connection

---

### Step 5: Process Productive Messages (Action 9)

**Purpose:** Test AI-powered message analysis and response generation

**Prerequisites:**
- ✅ Action 5 completed successfully (logged in)
- ✅ Action 7 completed (inbox processed)
- ✅ Productive messages exist in database

**Steps:**
1. Select option `9` from the menu
2. System will:
   - Find productive messages
   - Extract genealogical entities
   - Search GEDCOM/API for matches
   - Generate acknowledgment messages
   - Update database

**Expected Results:**
- ✅ Productive messages identified
- ✅ Genealogical data extracted
- ✅ Tree searches performed
- ✅ Acknowledgments generated
- ✅ Database updated
- ✅ Summary displayed:
  - Messages processed
  - Tasks created
  - Acknowledgments sent

**Troubleshooting:**
- If no productive messages, run Action 7 first
- If extraction fails, check AI service
- If tree search fails, verify GEDCOM file or API access

---

### Step 6: Send Messages (Action 8)

**Purpose:** Test automated messaging system (dry-run mode)

**Prerequisites:**
- ✅ Action 5 completed successfully (logged in)
- ✅ APP_MODE=dry_run in .env (safe testing)

**Steps:**
1. Select option `8` from the menu
2. System will:
   - Find eligible recipients
   - Load message templates
   - Generate personalized messages
   - **In dry-run mode:** Create messages but don't send
   - Update database

**Expected Results:**
- ✅ Recipients identified
- ✅ Messages generated from templates
- ✅ **Dry-run mode:** Messages logged but not sent
- ✅ Database updated with message records
- ✅ Summary displayed:
  - Messages prepared
  - Recipients processed
  - Status updates

**Troubleshooting:**
- If no recipients, check database for eligible matches
- If template errors, verify message_templates.csv exists
- If dry-run not working, check APP_MODE in .env

---

### Step 7: Gather DNA Matches (Action 6)

**Purpose:** Test DNA match gathering and processing

**Prerequisites:**
- ✅ Action 5 completed successfully (logged in)
- ✅ Browser session active

**Steps:**
1. Select option `6` from the menu
2. System will:
   - Navigate to DNA matches page
   - Scrape match data
   - Process and score matches
   - Update database

**Expected Results:**
- ✅ Browser navigates to DNA matches
- ✅ Matches scraped successfully
- ✅ Data processed and scored
- ✅ Database updated
- ✅ Summary displayed:
  - Matches gathered
  - New matches added
  - Existing matches updated

**Troubleshooting:**
- If navigation fails, check browser session
- If scraping fails, Ancestry.com may have changed layout
- If database errors, check database connection

---

## Testing Checklist

Use this checklist to track your testing progress:

- [ ] **Action 5:** Check Login Status
  - [ ] Login successful
  - [ ] Profile ID displayed
  - [ ] Session active

- [ ] **Action 10:** GEDCOM Report
  - [ ] Fraser Gault found
  - [ ] Score: 160 points
  - [ ] Family details correct
  - [ ] Relationship path shown

- [ ] **Action 11:** API Report
  - [ ] Fraser Gault found via API
  - [ ] Results match Action 10
  - [ ] Relationship path shown

- [ ] **Action 7:** Search Inbox
  - [ ] Conversations fetched
  - [ ] Messages classified
  - [ ] Database updated

- [ ] **Action 9:** Process Productive
  - [ ] Productive messages found
  - [ ] Entities extracted
  - [ ] Acknowledgments generated

- [ ] **Action 8:** Send Messages
  - [ ] Recipients identified
  - [ ] Messages generated
  - [ ] Dry-run mode working

- [ ] **Action 6:** Gather Matches
  - [ ] Matches scraped
  - [ ] Data processed
  - [ ] Database updated

---

## Expected Test Duration

- **Action 5:** 30 seconds - 2 minutes (depending on login)
- **Action 10:** 30-60 seconds (GEDCOM search)
- **Action 11:** 5-10 seconds (API search)
- **Action 7:** 1-5 minutes (depending on inbox size)
- **Action 9:** 2-10 minutes (depending on productive messages)
- **Action 8:** 1-5 minutes (depending on recipients)
- **Action 6:** 5-15 minutes (depending on match count)

**Total Estimated Time:** 15-45 minutes

---

## Safety Notes

1. **Dry-Run Mode:** System is in dry_run mode - no actual messages will be sent
2. **Conservative Limits:** Processing limits are set conservatively in .env
3. **Rate Limiting:** API calls are rate-limited to avoid hitting Ancestry limits
4. **Database Backups:** Consider backing up database before testing

---

## After Testing

1. Review logs in `Logs/` directory
2. Check database for updates
3. Verify no errors in console output
4. Report any issues or unexpected behavior

---

## Quick Reference

**Start Application:**
```bash
python main.py
```

**Test Sequence:**
1. Action 5 (Login)
2. Action 10 (GEDCOM - no login needed)
3. Action 11 (API)
4. Action 7 (Inbox)
5. Action 9 (Process Productive)
6. Action 8 (Send Messages)
7. Action 6 (Gather Matches)

**Exit Application:**
- Select option `0` or press `Ctrl+C`

---

**Guide Created:** 2025-09-30  
**System Status:** Production Ready + Code Quality Improvements  
**Mode:** dry_run (safe for testing)  
**Ready for Testing:** ✅ YES

