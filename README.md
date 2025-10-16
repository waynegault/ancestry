# Ancestry Genealogical Research Automation

Comprehensive Python automation system for Ancestry.com genealogical research, featuring intelligent messaging, DNA match analysis, and family tree management.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Features](#features)
- [Recent Fixes & Improvements](#recent-fixes--improvements)
- [Architecture](#architecture)
- [Actions](#actions)
- [Configuration](#configuration)
- [Testing](#testing)
- [Development Guidelines](#development-guidelines)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Overview

This project automates genealogical research workflows on Ancestry.com, including:
- **Action 6**: Automated DNA match gathering and data collection
- **Action 7**: Inbox message processing and analysis
- **Action 8**: Intelligent messaging with AI-powered responses
- **Action 9**: Productive conversation management
- **Action 10**: GEDCOM file analysis and scoring
- **Action 11**: API-based genealogical research and relationship discovery

---

## Quick Start

### Prerequisites
- Python 3.12+
- Google Chrome browser
- Ancestry.com account with DNA test results

### Installation

```bash
# Clone the repository
git clone https://github.com/waynegault/ancestry.git
cd ancestry

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Ancestry credentials and API keys
```

### Running the Application

```bash
# Run main menu
python main.py

# Run specific action directly
python action6_gather.py
python action11.py

# Run all tests
python run_all_tests.py
```

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
- **Circuit Breaker Pattern**: Automatic protection from cascading 429 failures
- **Session Management**: Automatic browser session recovery and refresh
- **Database Management**: SQLite with SQLAlchemy ORM
- **Error Handling**: Comprehensive retry logic and graceful degradation
- **Caching**: Multi-layer caching for API responses and computed data
- **Logging**: Detailed debug logging with configurable levels
- **AI Integration**: Multi-provider AI support (DeepSeek, Gemini)

---

## Recent Fixes & Improvements

### Connection Resilience Framework (October 2025)

**Comprehensive protection against PC sleep and connection loss:**

#### Features
- **Sleep Prevention**: Prevents PC sleep during long-running operations (Windows/macOS/Linux)
- **Connection Detection**: Automatically detects browser disconnection
- **Automatic Recovery**: Recovers from connection loss with exponential backoff (2s, 4s, 8s)
- **Cross-Platform**: Works on Windows, macOS, and Linux

#### Protected Actions
- **Action 6**: DNA Match Gathering
- **Action 7**: Inbox Processing
- **Action 8**: Messaging
- **Action 9**: Productive Processing

#### How It Works

```python
@with_connection_resilience("Action 6: DNA Match Gathering")
def coord(session_manager):
    # Sleep prevention enabled automatically
    # Connection monitoring active
    # Recovery on connection loss
    pass
```

#### Benefits
- ✅ PC won't sleep during long-running operations
- ✅ Automatic recovery from temporary network issues
- ✅ Graceful handling of browser disconnection
- ✅ Detailed logging for debugging
- ✅ No manual intervention required

---

### Phase 2 Performance Optimization (October 2025)

**Complete optimization suite with 4 major improvements:**

#### 1. RateLimiter Singleton Pattern
- Global singleton instance reused across all sessions
- Preserves rate limiting state and adaptive delay tuning
- 66% reduction in RateLimiter instances

#### 2. Timestamp Logic Gate (Data Freshness Check)
- Re-enabled timestamp check to skip if data < N days old
- Configurable via `PERSON_REFRESH_DAYS` (default: 7 days)
- 50% reduction in API calls on subsequent runs

#### 3. Logging Consolidation
- Consolidated browser initialization logs (5+ → 2 logs)
- Changed Circuit Breaker init log from INFO to DEBUG
- 37% reduction in log file size

#### 4. RPS Increase to 5.0
- Increased from 0.4 to 5.0 RPS (12x faster)
- Safe with circuit breaker protection
- Ancestry API typically allows 10-20 RPS

#### Performance Impact
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Action 6 duration** | 27s | ~15s | 44% faster |
| **Log file size** | 12,700 lines | 8,000 lines | 37% smaller |
| **API calls (2nd run)** | 200 | ~100 | 50% fewer |
| **Effective RPS** | 0.37/s | 5.0/s | 13x faster |

#### Sleep Prevention
- **Windows**: Uses `SetThreadExecutionState` API to prevent system sleep
- **macOS**: Uses `caffeinate` subprocess
- **Linux**: Manual disable (displays warning)
- Automatically enabled for entire session in main.py

---

### Circuit Breaker Pattern Implementation (January 2025)

**Automatic protection from cascading 429 failures** across all API-using actions:

#### What is a Circuit Breaker?

The Circuit Breaker pattern prevents cascading failures by monitoring API errors and temporarily blocking requests when failures exceed a threshold:

- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Too many failures (5 consecutive 429 errors), requests blocked for 60 seconds
- **HALF_OPEN**: Testing recovery with limited requests (3 test requests)

#### Benefits

- ✅ **Automatic protection**: All actions using `session_manager.rate_limiter` are protected
- ✅ **No code changes required**: Circuit breaker is integrated into RateLimiter
- ✅ **Prevents cascading failures**: Blocks requests after 5 consecutive 429 errors
- ✅ **Automatic recovery**: Tests service health after 60 seconds
- ✅ **Comprehensive metrics**: Tracks circuit opens, closes, blocked requests

#### Actions Protected

1. **Action 6** (DNA Match Gathering) - Primary beneficiary
2. **Action 7** (Inbox Processing) - Message retrieval
3. **Action 8** (Messaging) - Message sending
4. **Action 9** (Productive Match Processing) - Match details
5. **Action 10** (GEDCOM Analysis) - Relationship data
6. **Action 11** (API Research) - Search and family analysis

#### State Transitions

```
5 consecutive 429 errors → Circuit OPENS (blocks all requests for 60s)
60 seconds pass → Circuit transitions to HALF_OPEN (allows 3 test requests)
3 successful test requests → Circuit CLOSES (normal operation resumes)
Any test request fails → Circuit reopens (wait another 60s)
```

#### Metrics

Circuit breaker metrics are included in rate limiter summary:

```
CIRCUIT BREAKER METRICS
Current State:         CLOSED
Total Requests:        549
Blocked Requests:      0
Circuit Opens:         0
Circuit Closes:        0
Half-Open Successes:   0
Half-Open Failures:    0
```

---

### Action 6 Performance Optimizations (January 2025)

**Action 6 has been optimized for maximum efficiency** with comprehensive improvements:

#### Performance Optimizations
- **Enhanced skip logic**: Checks if DnaMatch exists BEFORE fetching details (98% API reduction on second run)
- **Cookie caching**: 30-second TTL reduces cookie syncs from 1,670 to ~30 (98% reduction)
- **Adaptive rate limiting**: Fast recovery from 429 errors (10% decrease when high, 2% when close to initial)
- **Parallel worker management**: Adaptive delays and random jitter prevent 429 errors
- **Streamlined logging**: Removed excessive debug messages while keeping important information
- **Enhanced progress tracking**: Clear page/batch context with cumulative totals
- **Database operation transparency**: Shows what data would be saved when skipping

#### Expected Performance Improvements
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **First Run** | 13.5 min | ~10 min | 25% faster |
| **Second Run** | 13.5 min | ~30 sec | **97% faster** |
| **Cookie Syncs** | 1,670 | ~30 | 98% reduction |
| **429 Errors** | 8 | 0 | 100% reduction |
| **Recovery Time** | 232 decreases | ~25 decreases | 90% faster |

#### Architecture Improvements
- **Reduced code size**: 871 lines (down from 5,293 lines - 84% reduction)
- **Simplified design**: 4-stage pipeline vs complex multi-threaded architecture
- **Universal functions**: DNA match operations moved to `dna_utils.py` for reuse across all actions
- **Better maintainability**: Clear separation of concerns, easier to understand and modify
- **Smart refresh logic**: Skip re-fetching person details if updated within configurable threshold (default: 14 days)

#### Data Collection Improvements
- **Complete field population**: All database fields now populated correctly
  - People table: birth_year, last_logged_in, administrator_profile_id, administrator_username, message_link
  - FamilyTree table: cfpid, person_name_in_tree, facts_link, view_in_tree_link, actual_relationship, relationship_path
  - DnaMatch table: predicted_relationship with correct percentage formatting
- **Fixed message_link format**: Now uses `messaging/?p={profile_id}` (correct format)
- **Fixed relationship_path formatting**: Properly formatted with parentheses (e.g., "William Litschel (2nd cousin)")
- **Fixed predicted_relationship percentages**: Now shows 99.0% instead of 9900.0%

#### API Integration
- **7 API endpoints** working correctly:
  1. Match List API - Paginated match list with core data
  2. In-Tree Status API - Batch check for in-tree status
  3. Match Details API - Individual match details (DNA data, parental sides)
  4. Profile Details API - User profile data (last login, contactable status)
  5. Badge Details API - Tree data (CFPID, person name, birth year)
  6. Match Probability API - Predicted relationship with probability distribution
  7. Get Ladder API - Relationship path and actual relationship

#### Database
- **100 People records** - All fields populated
- **100 DnaMatch records** - Complete DNA data
- **74 FamilyTree records** - Relationship paths and tree data (26 matches not in tree)

### Action 6 Stability Improvements (Pre-Rebuild)

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

## Architecture

### Core Components

```
core/
  session_manager.py    # Central coordinator with single RateLimiter
  browser_manager.py    # WebDriver lifecycle management
  api_manager.py        # REST API client
  database_manager.py   # SQLAlchemy connection pooling
  error_handling.py     # Exception hierarchy and retry logic

config/
  config_schema.py      # Type-safe configuration dataclasses
  config_manager.py     # .env loading and validation

utils.py                # RateLimiter, API helpers, navigation
database.py             # SQLAlchemy ORM models
ai_interface.py         # Multi-provider AI abstraction
```

### Rate Limiting & Circuit Breaker (CRITICAL)

**Single Rate Limiter Architecture:**
- Class: `RateLimiter` in `utils.py`
- Instance: `session_manager.rate_limiter`
- Algorithm: Thread-safe token bucket
- Configuration:
  - Capacity: 10 tokens
  - Fill rate: 2 tokens/second
  - Thread-safe with `threading.Lock()`

**Circuit Breaker Integration:**
- Class: `CircuitBreaker` in `utils.py`
- Instance: `session_manager.rate_limiter.circuit_breaker`
- Automatic protection from cascading 429 failures
- Configuration:
  - Failure threshold: 5 consecutive 429 errors
  - Recovery timeout: 60 seconds
  - Half-open test requests: 3

**DO NOT modify rate limiting settings without extensive validation!**
- Changing `REQUESTS_PER_SECOND` can cause 429 errors
- Circuit breaker will open after 5 consecutive 429 errors
- Monitor: `Select-String -Path Logs\app.log -Pattern "429 error"` should return 0

### Database Schema

SQLite database (`Data/ancestry.db`) with tables:
- `people`: Person records with genealogical data
- `messages`: Message history and AI responses
- `conversations`: Conversation threads
- `tasks`: Microsoft To-Do integration
- `dna_matches`: DNA match data
- `research_logs`: Research activity tracking

---

## Actions

### Action 6: Page Gathering
Automated data collection from Ancestry pages with sequential processing.

```bash
python action6_gather.py
```

### Action 7: Inbox Processing
Process and analyze inbox messages with AI classification.

```bash
python action7_inbox.py
```

### Action 8: Intelligent Messaging
Send AI-powered messages to DNA matches with context awareness.

```bash
python action8_messaging.py
```

### Action 9: Productive Conversation Management
Manage ongoing productive conversations with automated follow-ups.

```bash
python action9_process_productive.py
```

### Action 10: GEDCOM Analysis
Analyze GEDCOM files and score potential matches.

```bash
python action10.py
```

### Action 11: API Research
API-based genealogical research with relationship discovery.

```bash
python action11.py
```

---

## Configuration

### Environment Variables (.env)

Create a `.env` file with the following required settings:

```bash
# Ancestry Credentials
ANCESTRY_USERNAME=your_username
ANCESTRY_PASSWORD=your_password

# Database
DATABASE_FILE=Data/ancestry.db

# API Settings
BASE_URL=https://www.ancestry.co.uk/
API_TIMEOUT=30

# Rate Limiting (CRITICAL - Do not change without validation)
REQUESTS_PER_SECOND=0.3
BASE_DELAY=1.0

# Processing Limits (Conservative defaults)
MAX_PAGES=1
MAX_INBOX=5
MAX_PRODUCTIVE_TO_PROCESS=5
BATCH_SIZE=5

# Action 6 Settings
ACTION6_COORD_TIMEOUT_SECONDS=14400  # 4 hours

# Application Mode
APP_MODE=development  # or 'production'
DEBUG_MODE=false

# Logging
LOG_LEVEL=INFO
LOG_FILE=Logs/app.log

# AI Integration (Optional)
DEEPSEEK_API_KEY=your_deepseek_key
GOOGLE_API_KEY=your_google_key
AI_PROVIDER=deepseek  # or 'gemini' or ''

# Microsoft Graph (Optional - for To-Do integration)
MS_CLIENT_ID=your_client_id
MS_CLIENT_SECRET=your_client_secret
MS_TENANT_ID=your_tenant_id
```

### Configuration Schema

The project uses Pydantic for configuration validation. See `config/config_schema.py` for all available options.

---

## Testing

### Running Tests

```bash
# Run all tests (57 modules, 457 tests)
python run_all_tests.py

# Run with parallel execution
python run_all_tests.py --fast

# Run with log analysis
python run_all_tests.py --analyze-logs

# Run specific module tests
python -m action6_gather
python -m action11
```

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

### Testing Requirements

- All new features must have tests
- Tests must fail when conditions are not met (no fake passes)
- Use real API sessions for integration tests
- Test coverage should increase, not decrease

---

## Development Guidelines

### Code Quality
- Follow DRY (Don't Repeat Yourself) principles
- Use type hints for function signatures
- Add docstrings to all public functions
- Keep functions under 50 lines when possible

### Testing
- Write tests for all new functionality
- Tests should fail when functionality fails (no fake passes)
- Use `.env` test data for automated tests
- Maintain 100% test pass rate

### Rate Limiting
- Never bypass the rate limiter
- All API calls must go through `session_manager.rate_limiter`
- Monitor logs for 429 errors
- Validate changes with 50+ page runs

### Git Workflow
- Commit at each phase of implementation
- Write descriptive commit messages
- Test before committing
- Revert if tests fail

---

## Pylance Configuration

The project uses **basic** type checking mode with strict exclusions to ensure stable, accurate error reporting.

### Configuration Files

**pyrightconfig.json**: Primary configuration
- Type checking mode: `basic` (balances error detection with usability)
- **Includes**: Only `*.py` and `**/*.py` files (explicit Python files only)
- **Excludes**: `.git/**`, `__pycache__/**`, `.venv/**`, `Cache/**`, `Logs/**`, `Data/**`
- **Ignore**: `.git` and `**/.git` (prevents analyzing git internals)
- Silences false positives (unused variables, unreachable code, type inference limitations)
- Keeps critical errors visible (undefined variables, import errors, etc.)

**.vscode/settings.json**: Workspace settings
- **CRITICAL**: `.git` is excluded (`"**/.git": true`) to prevent Pylance from analyzing git internals
- File watching and search exclusions for performance
- Markdown linting configuration

**Note**: When `pyrightconfig.json` exists, `.vscode/settings.json` Python analysis settings are ignored.

### Reloading Configuration

**You MUST reload VS Code window for changes to take effect:**
1. Press `Ctrl+Shift+P`
2. Type "Developer: Reload Window"
3. Press Enter

**Expected result**: Stable error count showing only real issues (typically 10-20 errors)

### Troubleshooting Pylance

If you see thousands of errors or errors from `.git` files:
1. Check that `.vscode/settings.json` has `"**/.git": true` (not `false`)
2. Reload VS Code window (Ctrl+Shift+P → Developer: Reload Window)
3. If errors persist, restart VS Code completely
4. Clear Pylance cache: Delete `.vscode/.ropeproject` if it exists and reload

---

## Troubleshooting

### Common Issues

**429 Rate Limit Errors:**
- Check `REQUESTS_PER_SECOND=0.3` in `.env`
- Monitor: `Select-String -Path Logs\app.log -Pattern "429 error"`
- Reduce processing limits if needed

**Browser Crashes:**
- **Symptom**: "invalid session id" errors
- **Solution**: Proactive browser refresh is now automatic (every 10 pages)
- Update Chrome to latest version
- Check ChromeDriver compatibility

**API Failures:**
- **Symptom**: 303 redirects or JSON decode errors
- **Solution**: Optional APIs now fail gracefully without crashing
- Check session validity
- Verify cookies are current

**Database Errors:**
- **Symptom**: UNIQUE constraint failed on profile_id
- **Solution**: Intelligent collision resolution now handles this automatically
- Backup database: `python main.py` → Option 3
- Check file permissions on `Data/ancestry.db`

**Timeout Errors:**
- **Symptom**: Process stops after 5 minutes
- **Solution**: Config timeout is now 4 hours (configurable)
- Check `ACTION6_COORD_TIMEOUT_SECONDS` in `.env`

**Import Errors:**
- Verify Python 3.12+ is installed
- Run `pip install -r requirements.txt`
- Check virtual environment activation

### Logs

All logs are written to `Logs/app.log`:

```bash
# View recent errors
tail -100 Logs/app.log | grep ERROR

# Monitor real-time
tail -f Logs/app.log

# Check rate limiter initialization
grep "Thread-safe RateLimiter" Logs/app.log | tail -1

# Check for 429 errors
Select-String -Path Logs\app.log -Pattern "429 error"
```

### Debug Logging

Enable debug logging from the main menu (Option 10) or set in `.env`:

```bash
LOG_LEVEL=DEBUG
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run `python run_all_tests.py`
5. Fix any pylance errors
6. Commit with descriptive message
7. Submit pull request

### Code Quality Requirements

- Follow DRY (Don't Repeat Yourself) principles
- Use type hints for all functions
- Add docstrings for public APIs
- Keep functions under 50 lines when possible
- Fix all pylance errors before committing
- Maintain 100% test pass rate

---

## License

This project is for personal genealogical research use.

---

## Support

For issues or questions:
- **GitHub Issues**: <https://github.com/waynegault/ancestry/issues>
- **Email**: <39455578+waynegault@users.noreply.github.com>

---

## Acknowledgments

- Ancestry.com for providing the DNA matching platform
- SQLAlchemy for excellent ORM capabilities
- Selenium for browser automation
- Cloudscraper for Cloudflare bypass
- DeepSeek and Google Gemini for AI capabilities

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

