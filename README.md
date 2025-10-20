# Ancestry Genealogical Research Automation

Comprehensive Python automation system for Ancestry.com genealogical research, featuring intelligent messaging, DNA match analysis, and family tree management.

---

## Overview

This project automates genealogical research workflows on Ancestry.com, including:
- **Action 6**: Automated DNA match gathering and data collection
- **Action 7**: Inbox message processing and analysis
- **Action 8**: Intelligent messaging with AI-powered responses
- **Action 9**: Productive conversation management
- **Action 10**: GEDCOM file analysis and scoring
- **Action 11**: API-based genealogical research and relationship discovery

**Current Status**: Phase 6 Complete - All systems ready for production messaging

---

## User Instructions

### Quick Start

#### Prerequisites
- Python 3.12+
- Google Chrome browser
- Ancestry.com account with DNA test results

#### Installation

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

#### Running the Application

```bash
# Run main menu
python main.py

# Run specific action directly
python action6_gather.py
python action11.py

# Run all tests
python run_all_tests.py
```

### Configuration

Create a `.env` file with required settings:

```bash
# Ancestry Credentials
ANCESTRY_USERNAME=your_username
ANCESTRY_PASSWORD=your_password

# Database
DATABASE_FILE=Data/ancestry.db

# Processing Limits (Conservative defaults)
MAX_PAGES=1
MAX_INBOX=5
MAX_PRODUCTIVE_TO_PROCESS=5
BATCH_SIZE=5

# Application Mode
APP_MODE=dry_run  # or 'production'

# Logging
LOG_LEVEL=INFO
LOG_FILE=Logs/app.log
```

### Actions Overview

| Action | Purpose | Usage |
|--------|---------|-------|
| **Action 6** | DNA Match Gathering | Collect match data from Ancestry |
| **Action 7** | Inbox Processing | Manage incoming messages |
| **Action 8** | Automated Messaging | Send personalized messages to matches |
| **Action 9** | Productive Processing | Analyze high-value matches |
| **Action 10** | GEDCOM Analysis | Cross-reference with family tree files |
| **Action 11** | API Research | Genealogical research via Ancestry API |

### Action 8: Messaging System

**Phase 6 Testing Complete** - All systems validated and ready for production.

**Dry Run Mode** (Safe Testing):
```bash
APP_MODE=dry_run
# Messages created in database but NOT sent to Ancestry
```

**Production Mode** (Real Sending):
```bash
APP_MODE=production
# Messages sent to Ancestry messaging system
```

**Features**:
- Differential messaging (in-tree vs out-of-tree templates)
- Automatic message sequencing (3 messages per match)
- Desist functionality (stop messaging on request)
- AI-powered personalization
- Comprehensive error handling

---

## Developer Instructions

### Architecture Overview

The system uses a modular, layered architecture with clear separation of concerns:

**Core Infrastructure** (`core/`):
- `session_manager.py`: Central coordinator managing browser, API, and database connections with integrated RateLimiter
- `browser_manager.py`: WebDriver lifecycle management with health checks and recovery
- `api_manager.py`: REST API client with cookie synchronization and retry logic
- `database_manager.py`: SQLAlchemy connection pooling with transaction management
- `error_handling.py`: Comprehensive exception hierarchy with retry decorators and circuit breaker

**Configuration** (`config/`):
- `config_schema.py`: Type-safe configuration dataclasses with validation
- `config_manager.py`: .env loading and environment variable management

**Core Utilities**:
- `utils.py`: RateLimiter (thread-safe token bucket), API helpers, navigation utilities
- `database.py`: SQLAlchemy ORM models with Enums for controlled vocabulary
- `ai_interface.py`: Multi-provider AI abstraction (Google Gemini, DeepSeek)

**Action Modules** (Autonomous workflows):
- `action6_gather.py`: DNA match collection with parallel processing
- `action7_inbox.py`: Message processing with conversation analysis
- `action8_messaging.py`: Personalized message sending with state machine
- `action9_process_productive.py`: High-value match analysis with AI integration
- `action10.py`: GEDCOM file analysis and scoring
- `action11.py`: API-based genealogical research

### Rate Limiting & Circuit Breaker

**Single Rate Limiter Architecture**:
- **Class**: `RateLimiter` in `utils.py` - implements thread-safe token bucket algorithm
- **Instance**: `session_manager.rate_limiter` - shared across all API calls
- **Algorithm**: Token bucket with adaptive delays for parallel workers
- **Configuration**:
  - Capacity: 10 tokens (burst allowance)
  - Fill rate: 2 tokens/second (0.5s per request baseline)
  - Thread-safe with `threading.Lock()` for concurrent access
  - Adaptive delay: multiplied by sqrt(parallel_workers) to prevent thundering herd

**Why This Design**:
- Single rate limiter prevents cascading 429 errors across all API calls
- Token bucket allows burst traffic while maintaining average rate
- Adaptive delays for parallel workers prevent synchronized request storms
- Thread-safe implementation supports concurrent batch processing

**Circuit Breaker Integration**:
- Automatic protection from cascading 429 failures
- Failure threshold: 5 consecutive 429 errors
- Recovery timeout: 60 seconds
- Half-open test requests: 3 (validates recovery before resuming)

**CRITICAL**: Do not modify rate limiting settings without extensive validation! Changes affect all API interactions and can trigger 429 rate limit errors.

### Database Schema

SQLite database (`Data/ancestry.db`) with tables:
- `people`: Person records with genealogical data
- `dna_match`: DNA match data with ethnicity tracking
- `conversation_log`: Message history and AI responses
- `message_template`: Message templates for different scenarios
- `family_tree`: Family tree information

### Testing

```bash
# Run all tests (513 tests across 63 modules)
python run_all_tests.py

# Run specific module tests
python action6_gather.py
python action11.py
```

**Test Requirements**:
- All new features must have tests
- Tests must fail when conditions are not met (no fake passes)
- Use real API sessions for integration tests
- Maintain 100% test pass rate

### Code Quality Standards

**Commenting Philosophy**:
- Comments should explain **WHY**, not **WHAT** (code shows what it does)
- Use section headers with `===` markers for logical organization
- Include docstrings for all public functions with Args, Returns, and Raises
- Remove commented-out code - use git history if you need it back
- Add comments for non-obvious behavior or complex logic

**Code Style**:
- Follow DRY (Don't Repeat Yourself) principles - extract common patterns
- Use type hints for all function signatures (required for new code)
- Add docstrings to all public functions
- Keep functions under 50 lines when possible
- Fix all Pylance errors before committing

**Testing Requirements**:
- All new features must have tests
- Tests must fail when conditions are not met (no fake passes)
- Use real API sessions for integration tests
- Maintain 100% test pass rate

**Linting & Type Checking**:
- Ruff for code style: `ruff check --fix .`
- Pyright for type hints: configured in `pyrightconfig.json`
- Intentional ignores documented in `.ruff.toml`

### Git Workflow

- Commit at each phase of implementation
- Write descriptive commit messages
- Test before committing
- Revert if tests fail

### Important Implementation Details & Gotchas

**Session Management**:
- SessionManager is a singleton-like pattern - reuse the same instance across actions
- Always call `session_manager.ensure_session_ready()` before API calls
- Browser recovery is automatic but can take 30-60 seconds
- Cookies are synced from browser to requests session once per session

**Database Operations**:
- Use `commit_bulk_data()` for batch inserts/updates (more efficient than individual commits)
- Always use context managers for database sessions to ensure cleanup
- Enums (PersonStatusEnum, MessageDirectionEnum) provide controlled vocabulary
- Foreign key constraints are enforced - respect the schema

**API Rate Limiting**:
- Rate limiter is shared across all API calls - don't bypass it
- 429 errors trigger circuit breaker (5 consecutive failures = 60s timeout)
- Parallel workers automatically adjust rate limiting (adaptive delays)
- Always use `session_manager.rate_limiter.wait()` before API calls

**Error Handling**:
- Use decorators for automatic retry: `@retry_on_failure()`, `@circuit_breaker()`
- Catch specific exceptions (APIError, DatabaseError) not generic Exception
- Log errors with context using `@error_context()` decorator
- Recovery attempts are logged - check logs for failure reasons

**Testing**:
- Tests must use real API sessions (no mocking for integration tests)
- Tests should fail when conditions not met (strict validation)
- Use `suppress_logging()` context manager to reduce test output noise
- Run `python run_all_tests.py` before committing

---

## Future Development Ideas

1. **Enhanced DNA Ethnicity Tracking**: Expand ethnicity region tracking with visualization
2. **Parallel Processing**: Implement safe parallel processing for faster data collection
3. **Machine Learning**: Add ML-based relationship prediction
4. **Web Dashboard**: Create web UI for monitoring and management
5. **Mobile App**: Mobile companion app for on-the-go research
6. **Advanced Filtering**: More sophisticated match filtering and scoring
7. **Export Formats**: Support for additional genealogy file formats
8. **API Caching**: Intelligent caching layer for API responses

---

## Appendix A: Chronology of Changes

### Phase 7: Code Review & Quality Improvements (October 20, 2025)
- ✅ Fixed all Pylance errors (unused imports, type hints, duplicate code)
- ✅ Removed commented-out code blocks
- ✅ Enhanced commenting for clarity and WHY explanations
- ✅ Updated README with implementation details and gotchas
- ✅ Improved code organization and documentation
- ✅ All 513+ tests passing

### Phase 6: Final Validation (October 19, 2025)
- ✅ Validated all systems with Wayne's account
- ✅ Confirmed dry_run mode working correctly
- ✅ All 14,792 candidates processed without errors
- ✅ 44,376 messages in database (all OUT direction)
- ✅ System ready for production messaging

### Phase 5: Frances McHardy Testing (October 19, 2025)
- ✅ Tested with real Frances McHardy from production database
- ✅ Created 3 test messages (In_Tree templates)
- ✅ Verified template differentiation logic
- ✅ Confirmed message sequencing works

### Phase 4: Desist Functionality (October 18, 2025)
- ✅ Tested desist message sending
- ✅ Verified 7 templates working correctly
- ✅ Confirmed no errors during processing

### Phase 3: Differential Messaging (October 18, 2025)
- ✅ Processed 14,735 messages
- ✅ Verified in-tree vs out-of-tree template selection
- ✅ Confirmed message creation and database storage

### Phase 2: Dry-Run Testing (October 18, 2025)
- ✅ Fixed [Errno 22] error
- ✅ Validated dry_run mode functionality
- ✅ Confirmed messages created but not sent

### Phase 1: Message Template Review (October 18, 2025)
- ✅ Reviewed all message templates
- ✅ Verified template adequacy
- ✅ Confirmed template structure

---

## Appendix B: Technical Specifications

### API Endpoints

**Action 6 - DNA Match Gathering**:
1. Match List API - Paginated match list with core data
2. In-Tree Status API - Batch check for in-tree status
3. Match Details API - Individual match details
4. Profile Details API - User profile data
5. Badge Details API - Tree data
6. Match Probability API - Predicted relationship
7. Get Ladder API - Relationship path

**Action 8 - Messaging**:
- Message sending endpoint
- Conversation retrieval endpoint
- Message history endpoint

**Action 11 - API Research**:
- Person search endpoint
- Family analysis endpoint
- Relationship ladder endpoint

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Action 6 First Run** | ~10 minutes (100 people) |
| **Action 6 Second Run** | ~30 seconds (cached) |
| **Action 7 First Run** | ~4.8 hours (15,000 conversations) |
| **Action 7 Second Run** | ~5-10 minutes (95% skip rate) |
| **Action 8 Processing** | ~0.86 candidates/second |
| **Database Size** | ~60 MB (15,000 conversations) |

### Environment Variables

See `.env.example` for complete list. Key variables:
- `ANCESTRY_USERNAME` - Ancestry login
- `ANCESTRY_PASSWORD` - Ancestry password
- `DATABASE_FILE` - SQLite database path
- `APP_MODE` - dry_run or production
- `LOG_LEVEL` - DEBUG, INFO, WARNING, ERROR
- `REQUESTS_PER_SECOND` - Rate limiting (default: 0.3)

---

**Last Updated**: October 20, 2025
**Status**: Production Ready - Phase 7 Code Review Complete
