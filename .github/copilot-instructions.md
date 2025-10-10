# Ancestry Research Automation Platform - AI Agent Instructions

## Project Overview
Python-based genealogical research automation for Ancestry.com featuring DNA match collection, AI-powered conversation analysis, personalized messaging, and automated task generation. Built with enterprise-grade architecture: SQLAlchemy ORM, Selenium WebDriver, multi-provider AI integration (Google Gemini, DeepSeek), and comprehensive quality assurance with 58 test modules.

## Quick Reference: Critical Commands

### Testing & Validation
```powershell
# Run all 58 test modules (sequential)
python run_all_tests.py

# Parallel execution for speed
python run_all_tests.py --fast

# With performance log analysis
python run_all_tests.py --analyze-logs

# Single module test
python -m action6_gather
```

### Rate Limiting Monitoring
```powershell
# Check for 429 errors (should return 0)
(Select-String -Path Logs\app.log -Pattern "429 error").Count

# Verify rate limiter initialization
Select-String -Path Logs\app.log -Pattern "Thread-safe RateLimiter" | Select-Object -Last 1

# Watch real-time API activity
Get-Content Logs\app.log -Wait | Select-String "429|rate|worker"

# Verify worker configuration
Select-String -Path Logs\app.log -Pattern "parallel workers" | Select-Object -Last 1
```

### Database Operations
```bash
# Backup before risky operations
python main.py  # Option 3: Backup Database

# Direct SQLite access
sqlite3 Data/ancestry.db
# .tables
# SELECT COUNT(*) FROM people;
# .quit
```

### Code Quality
```bash
# Auto-fix linting issues
ruff check --fix .

# Check only (no modifications)
ruff check .
```

## Critical Architecture Patterns

### Session Management & Resource Orchestration
- **SessionManager** (`core/session_manager.py`) is the central coordinator - ALL browser, database, and API operations flow through it
- **exec_actn()** (`main.py` lines 413-491) is the universal action wrapper:
  - Determines resource requirements (browser needed? DB only?)
  - Calls `session_manager.ensure_session_ready()` or `ensure_db_ready()` before execution
  - Handles logging, performance metrics, cleanup, and error recovery
  - ALL action functions (6-11) must work with this pattern - never manage resources directly
- **Never** directly instantiate WebDriver, create DB connections, or initialize API clients - SessionManager handles all lifecycle management

### Rate Limiting (CRITICAL - Zero Tolerance)
```python
# utils.py RateLimiter - Thread-safe token bucket algorithm
THREAD_POOL_WORKERS=1  # NEVER change without extensive validation
REQUESTS_PER_SECOND=0.4  # Empirically validated (429 errors = 72-second penalties)
```
- **RateLimiter** is instantiated ONCE by SessionManager, shared across all API calls
- Recent fix (Oct 2025): Added `threading.Lock()` for thread safety - validated zero 429 errors
- Changing worker count or RPS without 50+ page validation WILL break production
- Monitor: `Select-String -Path Logs\app.log -Pattern "429 error"` should return 0

### Database Schema (SQLAlchemy ORM)
```python
# database.py - UUID handling is critical
Person.uuid -> UPPERCASE storage (DNA test GUID, nullable for members without tests)
Person.profile_id -> User profile ID (nullable for non-member DNA testers)
Person.administrator_profile_id -> Kit manager for non-members (message routing)
```
- **UUID Case Sensitivity**: All lookups use `.upper()`, storage enforced uppercase (bug fix Oct 2025)
- **Soft Deletes**: Use `deleted_at` column, never hard delete (preserves history)
- **Relationship**: Person ← DnaMatch, FamilyTree, ConversationLog (one-to-many)

### Testing Infrastructure
```python
# test_framework.py - Standardized pattern in EVERY module
def module_tests() -> bool:
    """Module-specific test implementation"""
    suite = TestSuite("Module Name", "module_file.py")
    # Tests using suite.add_test(lambda: assertion, "test description")
    return suite.run_tests()

run_comprehensive_tests = create_standard_test_runner(module_tests)

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
```
- 58 test modules validated by `run_all_tests.py` (supports `--fast`, `--analyze-logs`)
- Tests embedded in source files (not separate test/ directory)
- Zero fake passes - every test must validate real behavior

## Action Modules (11 Total)

### Action 6: DNA Match Gathering (`action6_gather.py`)
- **coord()** function (line 2337+): Main orchestrator using ThreadPoolExecutor for parallel API fetches
- **Per-page processing**: 20 matches/page, auto-resume from database checkpoint
- Rate limiting applied: After each page AND inside each API fetch
- Performance: ~40-60s per page (1 worker), ~596 matches/hour throughput

### Action 7: Inbox Processing (`action7_inbox.py`)
- **InboxProcessor** class: Scrapes inbox HTML, extracts conversations
- AI classification: PRODUCTIVE / DESIST / OTHER using `ai_prompts.json` → `intent_classification`
- Entity extraction: Names, dates, places, relationships from messages

### Action 9: Task Generation (`action9_process_productive.py`)
- **process_productive_messages()** (line 1425+): Converts PRODUCTIVE conversations → MS To-Do tasks
- Uses `genealogical_task_templates.py` for 8 task categories (vital records, census, DNA, etc.)
- Quality scoring: 0-100 based on specificity (verbs, years, record types, locations)

## Configuration System

### Type-Safe Schema (`config/config_schema.py`)
```python
@dataclass
class APISettings:
    max_pages: int = 1  # Processing limit
    thread_pool_workers: int = 1  # CRITICAL: Do not change
    requests_per_second: float = 0.4  # CRITICAL: Empirically validated
```
- **ConfigManager** loads from `.env` → validates types → provides `config_schema` global
- Always use `config.api.thread_pool_workers` not hardcoded values
- Changes to critical settings require `validate_rate_limiting.py` validation

### Environment Variables (`.env`)
```env
# Never change without validation
THREAD_POOL_WORKERS=1
REQUESTS_PER_SECOND=0.4

# Safe to adjust for testing
MAX_PAGES=1
SKIP_LIVE_API_TESTS=true
```

## AI Integration & Quality Gates

### Multi-Provider Abstraction (`ai_interface.py`)
```python
def call_ai(prompt_key: str, context: dict, variant: Optional[str] = None) -> dict:
    """
    prompt_key: Lookup in ai_prompts.json (intent_classification, extraction_task, etc.)
    context: Variables for template substitution
    variant: Experimental prompt version for A/B testing
    """
```
- **Providers**: Google Gemini (primary), DeepSeek (fallback via provider abstraction)
- **Telemetry**: Every call logged to `Logs/prompt_experiments.jsonl` with quality metrics
- **Quality gates**: `quality_regression_gate.py` prevents deployment if median score drops >5 points

### Prompt Structure (`ai_prompts.json`)
```json
{
  "intent_classification": {
    "description": "Classify genealogy message actionability",
    "prompt": "Rules: 1) Output exactly one label: PRODUCTIVE/ENTHUSIASTIC/...",
    "variants": {"v2": "Updated classification logic..."}
  }
}
```

### Quality Scoring System (`extraction_quality.py`)
Computes quality scores (0-100) based on:
- **Entity richness** (names, dates, places, relationships) - up to 70 points
- **Task specificity** (verbs, years, record types, locations) - up to 30 points
- **Penalties**: Missing names (-20), no verbs (-10), filler words (-10)
- **Bonuses**: 5+ entities (+5), specific years (+10)

**Quality Thresholds**:
- 85-100: Excellent extraction
- 70-84: Good (production ready)
- 50-69: Acceptable (needs monitoring)
- <50: Poor (review prompt immediately)

### Telemetry & Quality Gates

**Prompt Telemetry** (`prompt_telemetry.py`):
- Tracks parse_success, quality_score, response_time for each AI call
- Stores in `Logs/prompt_experiments.jsonl` for analysis
- Alerts on quality degradation → `Logs/prompt_experiment_alerts.jsonl`

**Quality Regression Gate** (`quality_regression_gate.py`):
```bash
# Check for quality regression (CI/CD integration)
python quality_regression_gate.py
# Exit 1 if median score drops >5 points from baseline

# Generate new baseline after prompt improvements
python prompt_telemetry.py --baseline

# View current statistics
python prompt_telemetry.py --stats
```

**Monitoring Commands**:
```bash
# Review recent AI responses
Get-Content Logs\prompt_experiments.jsonl -Tail 20

# Check for quality alerts
Get-Content Logs\prompt_experiment_alerts.jsonl -Tail 10

# Analyze quality trends
python prompt_telemetry.py --stats
```

## Developer Workflows

### Running Tests
```powershell
python run_all_tests.py              # Sequential (all 58 modules)
python run_all_tests.py --fast       # Parallel execution
python run_all_tests.py --analyze-logs  # Performance analysis
python -m action6_gather             # Single module tests
```

### Debugging Rate Limiting
```powershell
# Check for errors (should be 0)
(Select-String -Path Logs\app.log -Pattern "429 error").Count

# Verify initialization
Select-String -Path Logs\app.log -Pattern "Thread-safe RateLimiter" | Select-Object -Last 1

# Watch real-time
Get-Content Logs\app.log -Wait | Select-String "429|rate|worker"
```

### Debugging AI Extraction Issues
```bash
# Check telemetry statistics
python prompt_telemetry.py --stats
# Review: median quality scores, parse success rate

# Review recent AI responses
Get-Content Logs\prompt_experiments.jsonl -Tail 20
# Check: parse_success: true/false, quality_score values

# Test specific prompt in isolation
python -c "from ai_interface import call_ai; print(call_ai('intent_classification', {'message': 'Test message'}))"

# Regenerate baseline after prompt improvements
python prompt_telemetry.py --baseline
```

### Debugging Database Issues
```bash
# Check connection pool status
python -c "from database import engine; print(engine.pool.status())"

# Query database directly
sqlite3 Data/ancestry.db
# .tables
# SELECT COUNT(*) FROM people;
# SELECT uuid, COUNT(*) FROM people GROUP BY uuid HAVING COUNT(*) > 1;
# .quit

# Backup before dangerous operations
python main.py  # Option 3: Backup Database
```

### Debugging Session Issues
```bash
# Check session validity and age
python -c "from core.session_manager import SessionManager; sm = SessionManager(); print(f'Valid: {sm.is_sess_valid()}, Age: {sm.session_age_seconds()}s')"

# Force session refresh
python main.py  # Option 5: Check Login Status

# Clear browser cache and cookies
Remove-Item -Recurse -Force Cache\*
```

### Adding New Action Functions
1. Create action function with signature: `def new_action(session_manager: SessionManager, *_) -> bool`
2. Add to `main.py` action handlers (lines ~1650-1720)
3. Use `exec_actn(new_action, session_manager, choice_str)` - never manage resources directly
4. Add tests using `TestSuite` pattern
5. Document in README.md under "Action Modules"

### Performance Profiling
```bash
# Enable DEBUG logging for detailed timing
# main.py: setup_logging(log_level="DEBUG")

# Monitor memory usage
python -c "import psutil; p = psutil.Process(); print(f'Memory: {p.memory_info().rss / 1024 / 1024:.1f} MB')"

# Analyze log file for slow operations
Select-String -Path Logs\app.log -Pattern "Duration:|Elapsed:" | Select-Object -Last 20

# Profile specific action
python -m cProfile -o profile.stats main.py
# Then analyze with: python -m pstats profile.stats
```

## Code Quality Standards

### Linting (Ruff)
```python
# Auto-fix before commit
ruff check --fix .

# Enforced rules: E722 (bare except), F821 (undefined), I001 (import sort)
# Disable per-file: # ruff: noqa: E722
```

### Type Hints (Pyright)
- Required for all new functions
- `pyrightconfig.json` configures standard checking
- Nullable types: Use `Optional[Type]` not `Type | None` for Python 3.9 compatibility

### Logging Discipline
```python
logger.info("User-visible milestones")    # Action starts/completes
logger.debug("Internal state details")    # Variable values, control flow
logger.warning("Recoverable issues")      # Retry scenarios, fallbacks
logger.error("Action failures")           # Explicit errors requiring intervention
```

## Error Handling & Recovery Patterns

### Error Hierarchy (`core/error_handling.py`)
```python
# Base exceptions
AncestryException -> Base for all project errors
  ├─ RetryableError -> Automatic retry logic applies
  │  ├─ APIRateLimitError -> 429 errors (retry_after parameter)
  │  ├─ NetworkTimeoutError -> Transient network issues
  │  └─ DatabaseConnectionError -> DB connection failures
  └─ FatalError -> DO NOT retry, fail immediately
     ├─ DataValidationError -> Invalid data format
     └─ ConfigurationError -> Missing/invalid config
```

### Recovery Decorators
```python
# Automatic retry with exponential backoff
@retry_on_failure(max_attempts=3, backoff_factor=2.0)
def fetch_api_data(url):
    # Automatically retries on RetryableError subclasses
    pass

# Graceful degradation with fallback
@graceful_degradation(fallback_value=None)
def get_optional_data():
    # Returns fallback_value on any exception
    pass

# Error context for detailed logging
@error_context("API Request")
def make_request():
    # Adds context to error logs automatically
    pass
```

### Circuit Breaker Pattern
```python
# Action 6 example (lines 120-290 in action6_gather.py)
circuit_breaker = SessionCircuitBreaker(threshold=5)

# Check before expensive operation
if circuit_breaker.is_tripped():
    logger.critical("🚨 Circuit breaker TRIPPED - aborting")
    return False

# Record success/failure
if session_valid:
    circuit_breaker.record_success()  # Resets failure count
else:
    if circuit_breaker.record_failure():  # Returns True if just tripped
        logger.error("Circuit breaker tripped after 5 failures")
        abort_remaining_work()
```

**Benefits**: Fails fast after 5 consecutive errors instead of 15,000+ wasted attempts

### 403 Auth Refresh Pattern
```python
# Automatic auth refresh on 403 errors (action6_gather.py lines 571-670)
response = _api_req_with_auth_refresh(
    session_manager=session_manager,
    url=api_url,
    method="GET"
)
# If 403: refreshes browser cookies, syncs to API session, retries once
```

### Session Health Monitoring
```python
# Proactive session refresh before expiry (action6_gather.py lines 698-768)
_check_session_health_proactive(session_manager, current_page)
# Checks every 5 pages
# Refreshes at 25-minute mark (40-min session lifetime with 15-min buffer)
# Prevents 403 errors during long-running operations
```

## Common Pitfalls & Solutions

### UNIQUE Constraint Violations
- **Root Cause**: UUID stored lowercase, lookup expects uppercase (fixed Oct 2025)
- **Solution**: Always use `.upper()` on UUIDs before DB operations
- **Example**: `person = session.query(Person).filter(Person.uuid == test_uuid.upper()).first()`

### SQLAlchemy Session Caching
- **Symptom**: Bulk insert followed by immediate lookup returns None
- **Solution**: Call `session.expire_all()` after bulk operations to force DB refresh
- **Location**: `action6_gather.py` after person batch inserts

### 429 Rate Limit Errors
- **Symptom**: "429 Too Many Requests" with 72-second backoff
- **Solution**: Verify `THREAD_POOL_WORKERS=1` in `.env`, run `validate_rate_limiting.py`
- **Never**: Increase workers without 50+ page validation showing zero 429s

### Session Not Ready Errors
- **Symptom**: "Cannot perform action: Session not ready"
- **Solution**: Ensure action uses `exec_actn()` wrapper, which calls `ensure_session_ready()`
- **Never**: Call browser/API operations directly without SessionManager initialization

### Low AI Quality Scores
- **Symptom**: Quality scores consistently <70 in telemetry logs
- **Diagnosis**: 
  ```bash
  python prompt_telemetry.py --stats  # Check median scores
  Get-Content Logs\prompt_experiments.jsonl -Tail 20  # Review recent extractions
  ```
- **Solution**: Review/update prompts in `ai_prompts.json`, add variants for A/B testing
- **Prevention**: Run `python quality_regression_gate.py` before deployment

### Session Expiry During Long Operations
- **Symptom**: 403 errors appearing after 40 minutes of action execution
- **Prevention**: Action 6 has proactive health monitoring (refreshes at 25-min mark)
- **Manual Fix**: Use `session_manager.refresh_browser_cookies()` + `sync_cookies_from_browser()`
- **Configuration**: Adjust `HEALTH_CHECK_INTERVAL_PAGES` in `.env` (default: 5)

## Performance Optimization

### Rate Limiting with Token Bucket Algorithm
```python
# utils.py RateLimiter (lines 835-1019)
class RateLimiter:
    def __init__(self):
        self.capacity = 10.0          # Burst capacity (tokens)
        self.fill_rate = 2.0          # Tokens per second
        self.tokens = self.capacity   # Current available tokens
        self._lock = threading.Lock() # Thread safety

    def wait(self):
        """Thread-safe token bucket algorithm"""
        with self._lock:
            # Wait until token available
            # Refill tokens based on elapsed time
            # Consume 1 token per request
```

**Key Features**:
- **Burst handling**: Initial 10 tokens allow fast startup
- **Adaptive backoff**: Increases delay on 429 errors (1.5x multiplier)
- **Gradual recovery**: Decreases delay on success (0.95x multiplier)
- **Max delay cap**: 15 seconds (prevents excessive waiting)

**Configuration** (`.env`):
```env
REQUESTS_PER_SECOND=0.4      # Fill rate for token bucket
INITIAL_DELAY=1.0            # Starting delay between requests
MAX_DELAY=15.0               # Maximum delay cap
BACKOFF_FACTOR=1.5           # Multiplier on 429 errors
DECREASE_FACTOR=0.95         # Gradual speedup on success
```

### Checkpoint System for Resume Capability
```python
# action6_gather.py (lines 848-957)
# Automatic checkpoint after each page
_save_checkpoint(current_page, total_pages, state)

# Resume logic on startup
checkpoint = _load_checkpoint()
resuming, start_page = _should_resume_from_checkpoint(checkpoint, requested_start_page)
```

**User Intent Handling**:
- `python main.py` → "6" (no page) → Auto-resume from checkpoint if exists
- `python main.py` → "6 1" (explicit page 1) → Ignore checkpoint, fresh start
- `python main.py` → "6 50" (explicit page) → Start from page 50, ignore checkpoint

**Checkpoint Data** (`Cache/action6_checkpoint.json`):
```json
{
  "version": "1.0",
  "timestamp": 1728123456.789,
  "current_page": 449,
  "total_pages": 800,
  "counters": {
    "total_new": 3245,
    "total_updated": 4123,
    "total_skipped": 1612,
    "total_errors": 0
  }
}
```

**Configuration**:
- `ENABLE_CHECKPOINTING=true` - Enable/disable (default: true)
- `CHECKPOINT_MAX_AGE_HOURS=24` - Expire old checkpoints (default: 24h)
- Atomic writes prevent corruption (write to temp → rename)

### API Call Caching & Deduplication
```python
# action6_gather.py APICallCache (lines 1058-1217)
cache = _get_api_cache(ttl_seconds=300)  # 5-minute TTL

# Check cache before API call
cached = cache.get(f"combined:{uuid}")
if cached is not None:
    return cached  # Cache hit - no API call

# Make API call and cache result
result = _api_req_with_auth_refresh(...)
cache.set(f"combined:{uuid}", result)
```

**Benefits**:
- **14-20% cache hit rate** in production
- **200-400 fewer API calls** per full run (16K matches)
- **10-20 minutes saved** on large batches
- Thread-safe for 2-worker parallel processing

**Deduplication**:
```python
# Within batch: remove duplicates before API calls
deduplicated, cache_hits = _deduplicate_api_requests(
    fetch_candidates_uuid, matches_to_process_later
)
logger.debug(f"🎯 Cache hits: {cache_hits}, Deduplicated: {len(fetch_candidates_uuid) - len(deduplicated)}")
```

### Performance Metrics Tracking
```python
# action6_gather.py PerformanceMetrics (lines 1086-1247)
metrics = PerformanceMetrics()

# Real-time progress (every 10 pages)
metrics.log_progress(current_page, total_pages)
# 📊 Progress: 200/800 (25.0%) | Elapsed: 4.6m | ETA: 13.8m | Avg: 1.4s/page | Rate: 43.9 pages/min

# Final report
metrics.log_final_summary()
# 📈 FINAL PERFORMANCE REPORT
# ⏱️  Total Duration: 18.2m
# 📊 Throughput: 43.9 pages/min, 880 matches/hour
# 🌐 API Response Time: 0.45s avg
# 💾 Cache Performance: 14.5% hit rate
```

### Sleep Prevention for Long Operations
```python
# utils.py - Prevents system sleep during batch operations
from utils import prevent_system_sleep, restore_system_sleep

sleep_state = prevent_system_sleep()
try:
    # Long-running operation (e.g., Action 6: 800 pages)
    process_all_matches()
finally:
    restore_system_sleep(sleep_state)
```

**Platform Support**:
- **Windows**: Uses `SetThreadExecutionState` API
- **macOS**: Uses `caffeinate` subprocess
- **Linux**: Manual disable (displays warning)

**Main.py Integration**: Automatically enabled for entire session (line 1803)

## File Organization

```
action6-11.py           # 11 action modules (gather, inbox, messaging, etc.)
main.py                 # Entry point with menu, exec_actn() pattern
database.py             # SQLAlchemy ORM models
utils.py                # RateLimiter, nav helpers, login flows
core/
  session_manager.py    # Central orchestrator (THE critical component)
  browser_manager.py    # WebDriver lifecycle
  api_manager.py        # REST client with rate limiting coordination
  database_manager.py   # Connection pooling
  error_handling.py     # Exception hierarchy, retry decorators, circuit breaker
config/
  config_schema.py      # Type-safe dataclass definitions
  config_manager.py     # .env loading and validation
ai_interface.py         # Multi-provider AI abstraction
ai_prompts.json         # Prompt library with versioning
extraction_quality.py   # Quality scoring for AI extractions
prompt_telemetry.py     # AI performance monitoring
quality_regression_gate.py  # Quality gate for CI/CD
test_framework.py       # TestSuite, assertion utilities
run_all_tests.py        # Test orchestrator with parallel execution
```

## Key Dependencies
- **selenium 4.31.0+**: Browser automation (ChromeDriver auto-updates via webdriver-manager)
- **SQLAlchemy 2.0.40+**: Database ORM with soft deletes, connection pooling
- **google-generativeai 0.8.4**: Google Gemini AI provider
- **requests 2.32.3+**: HTTP client for API calls (shares rate limiter with Selenium)
- **beautifulsoup4 4.13.3+**: HTML parsing for inbox scraping
- **tqdm 4.67.1+**: Progress bars with ETA calculation

## Important Invariants
1. **Single SessionManager Instance**: One per main.py execution, shared across all actions
2. **Thread-Safe Rate Limiter**: One DynamicRateLimiter instance, all API calls serialize through it
3. **Uppercase UUIDs**: All UUID storage and lookups use `.upper()` - no exceptions
4. **exec_actn() Wrapper**: ALL action functions invoked through this, never called directly
5. **Test Coverage**: New features require tests in same file using TestSuite pattern
6. **No Hardcoded Credentials**: All secrets in `.env` or encrypted via `credentials.py`

## Recent Critical Fixes (October 2025)
- **Rate Limiter Thread Safety**: Added `threading.Lock()` to DynamicRateLimiter (utils.py line 900)
- **UUID Case Sensitivity**: Standardized all lookups to uppercase (action6_gather.py)
- **Session Caching**: Added `session.expire_all()` after bulk inserts (action6_gather.py line 2780)
- **Configuration Externalization**: Moved THREAD_POOL_WORKERS and REQUESTS_PER_SECOND to `.env`
- **Checkpoint Logic**: Changed `start: int = 1` → `start: Optional[int] = None` for auto-resume (action6_gather.py line 2337)

## Questions to Ask When Unclear
1. "Does this action need browser automation?" → If yes, use `exec_actn()` with `ensure_session_ready()`
2. "Am I making API calls?" → Use `session_manager.api_manager`, never raw `requests.get()`
3. "Do I need a database transaction?" → Use `with db_transn(session)` context manager
4. "How do I test this?" → Add `run_comprehensive_tests()` using TestSuite pattern
5. "Is this rate limiting safe?" → Run `validate_rate_limiting.py` and monitor for 50+ pages

---

**When in doubt**: Check `README.md` (3500+ lines, comprehensive developer documentation), review similar patterns in existing action modules, or trace execution through SessionManager → exec_actn() → action function.
