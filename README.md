# Ancestry DNA Match Automation

A comprehensive Python automation tool for managing Ancestry.com DNA matches, genealogical research, and family tree building.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Recent Fixes & Improvements](#recent-fixes--improvements)
- [Action 6: DNA Match Gathering](#action-6-dna-match-gathering)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Overview

This project automates the collection, analysis, and management of DNA matches from Ancestry.com. It provides tools for:
- Gathering DNA match data via API and web scraping
- Analyzing genealogical relationships
- Managing family trees and GEDCOM files
- Automated messaging to DNA matches
- Research prioritization and tracking

---

## Features

### Core Actions

1. **Action 2**: Database Reset - Clean slate for testing
2. **Action 6**: DNA Match Gathering - Collect match data from Ancestry
3. **Action 7**: Inbox Processing - Manage incoming messages
4. **Action 8**: Automated Messaging - Send personalized messages to matches
5. **Action 9**: Productive Match Processing - Analyze high-value matches
6. **Action 10**: GEDCOM-based Research - Cross-reference with family tree files
7. **Action 11**: API-based Research - Genealogical research via Ancestry API

### Key Capabilities

- **Intelligent Rate Limiting**: Token bucket algorithm prevents API throttling
- **Session Management**: Automatic browser session recovery and refresh
- **Database Management**: SQLite with SQLAlchemy ORM
- **Error Handling**: Comprehensive retry logic and graceful degradation
- **Caching**: Multi-layer caching for API responses and computed data
- **Logging**: Detailed debug logging with configurable levels

---

## Recent Fixes & Improvements

### Action 6 Stability Improvements (October 2025)

#### Problem 1: Match Probability API Failures ✅ FIXED
**Issue**: Process crashed when Match Probability API returned HTTP 303 redirects
**Fix (Commit 758cca8)**:
- Check for redirect status codes before parsing JSON
- Return None instead of raising exceptions for optional data
- Changed error logs to warnings
- Process continues even if API fails

#### Problem 2: Browser Session Crashes ✅ FIXED
**Issue**: Browser crashed after ~30-40 minutes with no recovery
**Fix (Commit 52f154b)**:
- Enhanced session recovery to work without session_start_time
- Added proactive browser refresh every 10 pages
- Automatic recovery when browser becomes invalid

#### Problem 3: Profile_id Collision Handling ✅ FIXED
**Issue**: UNIQUE constraint errors when same profile_id appeared multiple times
**Fix (Commit b30c2b1)**:
- Intelligent collision resolution based on ownership
- True owner detection logic
- Smart decision making for which record keeps profile_id

#### Problem 4: Timeout Issues ✅ FIXED
**Issue**: Hardcoded 5-minute timeout caused premature failures
**Fix (Commit 8ba6ae6)**:
- Use config timeout value (4 hours) instead of hardcoded 5 minutes

#### Problem 5: Parallel Processing Removed ✅ COMPLETE
**Issue**: ThreadPoolExecutor caused race conditions and complexity
**Fix (Commit series)**:
- Removed all parallel processing infrastructure
- Converted to simple sequential processing
- Improved stability and reliability

---

## Action 6: DNA Match Gathering

### Overview
Action 6 is the core data collection module that gathers DNA match information from Ancestry.com.

### How It Works

1. **Page-by-Page Processing**:
   - Fetches match list page (20 matches per page)
   - Retrieves detailed data for each match via API
   - Saves to database in batches
   - Waits 30 seconds between pages (rate limiting)

2. **Data Collected**:
   - Person details (name, profile_id, UUID)
   - DNA match details (shared cM, segments, confidence)
   - Family tree information
   - Relationship probability predictions
   - In-tree status and badges

3. **Performance**:
   - Sequential processing: ~5 minutes per page
   - Proactive browser refresh every 10 pages
   - Automatic session recovery on failures
   - Graceful degradation for optional APIs

### Configuration

```bash
# In .env file
MAX_PAGES=5              # Number of pages to process (0 = all)
BATCH_SIZE=5             # Matches per batch
MAX_PRODUCTIVE_TO_PROCESS=5
MAX_INBOX=5
```

### Testing Phases

**Phase 1: Small Test (5 pages)**
```bash
MAX_PAGES=5
# Expected: 100 people saved, ~25 minutes
```

**Phase 2: Medium Test (15 pages)**
```bash
MAX_PAGES=15
# Expected: 300 people saved, ~75 minutes, browser refresh at page 10
```

**Phase 3: Large Test (50 pages)**
```bash
MAX_PAGES=50
# Expected: 1000 people saved, ~4 hours, multiple browser refreshes
```

**Phase 4: Full Run (all pages)**
```bash
MAX_PAGES=0
# Expected: ~16,000 people saved, ~67 hours (3 days)
```

---

## Installation

### Prerequisites
- Python 3.11 or higher
- Google Chrome browser
- Ancestry.com account with DNA test results

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/waynegault/ancestry.git
cd ancestry
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your settings
```

4. **Set up credentials**:
- Add Ancestry.com cookies to `ancestry_cookies.json`
- Configure API credentials in `.env`

---

## Configuration

### Environment Variables (.env)

```bash
# Database
DATABASE_FILE=Data/ancestry.db

# API Settings
BASE_URL=https://www.ancestry.co.uk/
API_TIMEOUT=30

# Rate Limiting
REQUESTS_PER_SECOND=0.3
BASE_DELAY=1.0

# Action 6 Settings
MAX_PAGES=5
BATCH_SIZE=5
ACTION6_COORD_TIMEOUT_SECONDS=14400  # 4 hours

# Logging
LOG_LEVEL=INFO
LOG_FILE=Logs/app.log
```

### Configuration Schema

The project uses Pydantic for configuration validation. See `config/config_schema.py` for all available options.

---

## Usage

### Running the Application

```bash
python main.py
```

### Main Menu Options

1. **Check Login Status** - Verify Ancestry.com session
2. **Reset Database** - Clean slate for testing
3. **Backup Database** - Create backup before major operations
4. **Action 6: Gather DNA Matches** - Main data collection
5. **Action 7: Process Inbox** - Handle incoming messages
6. **Action 8: Send Messages** - Automated outreach
7. **Action 9: Process Productive Matches** - Analyze high-value matches
8. **Action 10: GEDCOM Research** - Cross-reference with family tree
9. **Action 11: API Research** - Genealogical research via API
10. **Toggle Debug Logging** - Switch between INFO and DEBUG levels

### Running Tests

```bash
# Run all tests
python run_all_tests.py

# Run specific action tests
python action6_gather.py  # Runs Action 6 tests
python action11.py        # Runs Action 11 tests
```

---

## Testing

### Test Framework

The project uses a custom test framework (`test_framework.py`) that provides:
- Test suite management
- Assertion helpers
- Logging suppression during tests
- Test result reporting

### Test Coverage

- **Action 6**: Sequential processing, database operations, API calls
- **Action 8**: Message sending, dry run mode, desist functionality
- **Action 10**: GEDCOM parsing, scoring, relationship detection
- **Action 11**: API research, scoring, relationship paths
- **Database**: CRUD operations, soft delete, cleanup
- **Session Management**: Browser lifecycle, recovery, validation

---

## Troubleshooting

### Common Issues

#### 1. Browser Crashes
**Symptom**: "invalid session id" errors
**Solution**: Proactive browser refresh is now automatic (every 10 pages)

#### 2. API Failures
**Symptom**: 303 redirects or JSON decode errors
**Solution**: Optional APIs now fail gracefully without crashing

#### 3. Database Errors
**Symptom**: UNIQUE constraint failed on profile_id
**Solution**: Intelligent collision resolution now handles this automatically

#### 4. Timeout Errors
**Symptom**: Process stops after 5 minutes
**Solution**: Config timeout is now 4 hours (configurable)

### Debug Logging

Enable debug logging from the main menu (Option 10) or set in `.env`:
```bash
LOG_LEVEL=DEBUG
```

### Log Files

- **app.log**: Main application log (all modules)
- Located in `Logs/` directory
- Rotates automatically when large

---

## Contributing

### Development Workflow

1. Create feature branch
2. Make changes
3. Run tests: `python run_all_tests.py`
4. Fix any pylance errors
5. Commit with descriptive message
6. Create pull request

### Code Quality

- Follow DRY (Don't Repeat Yourself) principles
- Use type hints for all functions
- Add docstrings for public APIs
- Keep functions under 50 lines when possible
- Fix all pylance errors before committing

### Testing Requirements

- All new features must have tests
- Tests must fail when conditions are not met (no fake passes)
- Use real API sessions for integration tests
- Test coverage should increase, not decrease

---

## License

This project is private and not licensed for public use.

---

## Contact

**Author**: Wayne Gault
**GitHub**: https://github.com/waynegault/ancestry
**Email**: 39455578+waynegault@users.noreply.github.com

---

## Acknowledgments

- Ancestry.com for providing the DNA matching platform
- SQLAlchemy for excellent ORM capabilities
- Selenium for browser automation
- Cloudscraper for Cloudflare bypass

---

## Appendix A: Detailed Fix History

### Action 6 Fix Timeline

#### October 13, 2025: Parallel Processing Elimination
**Problem**: ThreadPoolExecutor caused race conditions and complexity
**Solution**:
- Removed all parallel processing infrastructure
- Converted to sequential processing loops
- Updated configuration schema
- Updated documentation

**Files Modified**:
- `action6_gather.py` - Removed ThreadPoolExecutor, converted to sequential
- `config/config_schema.py` - Removed thread_pool_workers
- `.env` - Removed THREAD_POOL_WORKERS setting
- `.github/copilot-instructions.md` - Updated to reflect sequential processing

**Commit**: PARALLEL_PROCESSING_ELIMINATION_SUMMARY.md

---

#### October 13, 2025: Profile_id Collision Resolution
**Problem**: UNIQUE constraint errors when same profile_id appeared for different DNA tests

**Data Model Understanding**:
- **UUID**: DNA test Sample ID (always unique, always present)
- **profile_id**: Ancestry User Profile ID (unique when NOT NULL, can be NULL)
- **administrator_profile_id**: Profile ID of person managing the DNA kit

**Scenarios**:
- **Scenario A**: Member with own test → `profile_id="PROF1"`, `administrator_profile_id=NULL`
- **Scenario B**: Non-member test → `profile_id=NULL`, `administrator_profile_id="PROF1"`
- **Scenario C**: Member administering another's test → `profile_id="PROF2"`, `administrator_profile_id="PROF1"`

**Solution**:
1. **True Owner Detection**:
   - Member with own test: `tester_profile_id == admin_profile_id AND tester_username == admin_username`
   - Self-managed: `administrator_profile_id` is NULL

2. **Collision Resolution**:
   - Existing record is true owner → Set new record's profile_id to NULL
   - New record is true owner → Keep new profile_id, warn about existing
   - Ambiguous → Set new to NULL (conservative)

**Functions Added**:
- `_is_true_profile_owner()` - Ownership detection
- `_check_profile_id_collisions_with_db()` - Enhanced collision check
- `_handle_profile_id_collisions()` - Intelligent resolution

**Commit**: b30c2b1

---

#### October 14, 2025: Browser Session Recovery
**Problem**: Browser crashed after ~30-40 minutes with no recovery

**Root Cause**:
- `_should_attempt_recovery()` required `session_start_time` to be set
- Session recovery was being skipped for long-running operations

**Solution**:
1. Enhanced `_should_attempt_recovery()` to work without `session_start_time`
2. Added proactive browser refresh every 10 pages
3. Graceful degradation if refresh fails

**Code Changes**:
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

**Commit**: 52f154b

---

#### October 14, 2025: Match Probability API Fix
**Problem**: Process crashed when Match Probability API returned HTTP 303 redirects

**Evidence from Logs**:
```
20:15:38 DEB Response Status: 303 See Other
20:15:38 ERR JSON decode FAILED: Expecting value: line 1 column 1
20:15:41 ERR API Call failed after 3 retries
```

**Root Cause**:
- API endpoint may have changed or session expired
- Code tried to parse redirect response as JSON
- Exception propagated and crashed process
- No graceful degradation for optional API

**Solution**:
1. Check for redirect status codes (301, 302, 303, 307, 308) BEFORE parsing JSON
2. Return None instead of raising exception for redirects
3. Changed JSON decode errors from raising to returning None
4. Changed all exception handling to return None (this is optional data)
5. Changed error logs to warnings

**Code Changes**:
```python
# Check for redirects before parsing JSON
if response_rel.status_code in (301, 302, 303, 307, 308):
    logger.warning(
        f"{api_description}: Received redirect ({response_rel.status_code}). "
        f"Skipping relationship probability (optional data)."
    )
    return None

# Don't raise on JSON decode errors
try:
    data = response_rel.json()
except json.JSONDecodeError as json_err:
    logger.warning(
        f"{api_description}: JSON decode FAILED. "
        f"Skipping relationship probability."
    )
    return None  # Don't raise - just skip this optional data
```

**Commit**: 758cca8

---

## Appendix B: Test Results

### Test Run 1: MAX_PAGES=5 (Before Match Probability Fix)
**Date**: October 14, 2025 20:03:02 - 20:15:51
**Duration**: ~13 minutes
**Result**: ❌ Partial failure

**Pages Completed**: 2 out of 5
- Page 1: ✅ 20 people saved, 0 errors
- Page 2: ✅ 20 people saved, 0 errors
- Page 3: ❌ Started but crashed mid-processing

**Database**: 40 people saved
**Errors**: Match Probability API 303 redirects caused crash

**What Worked**:
- ✅ Profile_id collision handling (no UNIQUE constraint errors)
- ✅ Timeout protection (no timeout errors)
- ✅ Sequential processing (stable)
- ✅ Rate limiting (token bucket working)
- ✅ Database operations (all bulk operations successful)

**What Failed**:
- ❌ Match Probability API (303 redirects not handled)
- ❌ No graceful degradation (process crashed)

---

### Expected Test Run 2: MAX_PAGES=5 (After All Fixes)
**Status**: Ready to run
**Expected Result**: ✅ All 5 pages complete successfully

**Expected Outcomes**:
- ✅ 100 people saved to database
- ⚠️ Warnings for Match Probability API failures (expected, not errors)
- ✅ No process crashes
- ✅ All pages complete even if optional APIs fail

---

## Appendix C: Architecture & Design

### Database Schema

**Person Table**:
- `id` (Primary Key)
- `uuid` (UNIQUE, nullable) - DNA test Sample ID
- `profile_id` (UNIQUE, nullable) - Ancestry User Profile ID
- `administrator_profile_id` (nullable) - Profile ID of test manager
- `username` - Display name
- `status` - Enum (ACTIVE, INACTIVE, etc.)
- `deleted_at` - Soft delete timestamp

**DnaMatch Table**:
- `id` (Primary Key)
- `person_id` (Foreign Key → Person)
- `shared_cm` - Shared centimorgans
- `shared_segments` - Number of shared segments
- `confidence` - Match confidence level
- `relationship_prob` - Predicted relationship (optional)

**FamilyTree Table**:
- `id` (Primary Key)
- `person_id` (Foreign Key → Person)
- `tree_id` - Ancestry tree ID
- `tree_name` - Tree name
- `in_tree` - Boolean flag

### Session Management

**Components**:
- `SessionManager` - Main session coordinator
- `BrowserManager` - Chrome WebDriver lifecycle
- `APIManager` - API session and cookie management
- `DatabaseManager` - Database connection pooling

**Session Lifecycle**:
1. Initialize browser with saved cookies
2. Verify login status via API
3. Sync cookies between browser and API session
4. Retrieve CSRF token
5. Perform operations
6. Proactive refresh every 10 pages
7. Automatic recovery on failures

### Rate Limiting

**Token Bucket Algorithm**:
- Capacity: 10 tokens
- Refill rate: 0.3 tokens/second (configurable)
- Base delay: 1.0 seconds (configurable)
- Prevents API throttling and 429 errors

**Implementation**:
```python
def wait(self, cost: float = 1.0) -> None:
    """Wait for token availability and apply rate limiting."""
    self._refill_bucket()

    if self.tokens >= cost:
        self.tokens -= cost
        delay = self.base_delay + random.uniform(0, 0.5)
        time.sleep(delay)
    else:
        wait_time = (cost - self.tokens) / self.refill_rate
        time.sleep(wait_time)
        self._refill_bucket()
```

---

**Last Updated**: October 14, 2025
**Version**: 2.0 (Sequential Processing, Stable)

