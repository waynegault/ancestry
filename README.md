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

**Current Status**: Phase 5 IN PROGRESS - Research Assistant Features (P5.1 Complete: Source Citation Extraction)

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
| **Action 10** | GEDCOM Analysis | Cross-reference with local family tree files (NO API) |
| **Action 11** | API Research | Genealogical research via Ancestry API |

### Action 10: GEDCOM File Analysis

**Pure Local Analysis** - NO API calls, NO internet required:
- Analyzes local GEDCOM (.ged) family tree files
- Searches for individuals matching specific criteria
- Calculates relationship paths using bidirectional BFS
- Uses universal scoring system (shared with Action 11)
- Multi-level caching for 6x performance improvement

**Performance**:
- First run (no cache): ~32 seconds (14,640 individuals)
- Subsequent runs (disk cache): ~5-6 seconds (6x faster)
- Memory cache: Instant access

**Usage**:
```bash
python main.py
# Select option 10
# Enter search criteria (name, birth year, place, etc.)
# View top matches with family details and relationship paths
```

**Features**:
- Universal scoring algorithm (identical to Action 11)
- Relationship path calculation (uncle, cousin, etc.)
- Family member display (parents, siblings, spouses, children)
- Intelligent caching with automatic invalidation
- No session/API requirements (db_ready only)

### Action 8: Intelligent Messaging System

**Phase 4 (P4.1) Complete** - AI-powered conversational messaging with adaptive timing.

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
- `action10.py`: GEDCOM file analysis (NO API, pure local)
- `action11.py`: API-based genealogical research

**Supporting Modules**:
- `gedcom_utils.py`: GEDCOM parsing, relationship path finding (bidirectional BFS)
- `gedcom_cache.py`: Multi-level caching (memory → disk → file)
- `relationship_utils.py`: Relationship determination and path formatting
- `search_criteria_utils.py`: Unified search criteria for Actions 10 & 11
- `universal_scoring.py`: Shared scoring algorithm across actions

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

**GEDCOM Caching System**:
- **Memory Cache**: Instant access for same-session requests (fastest)
- **Disk Cache**: Persistent cache using DiskCache library (~6x faster than file parsing)
- **File Parsing**: Full GEDCOM parse when cache is stale or missing (~32 seconds for 14,640 individuals)
- **Automatic Invalidation**: Cache key includes file modification time - automatically reloads when GEDCOM file changes
- **Cache Contents**: Stores processed_data_cache, id_to_parents, id_to_children (NOT indi_index - unpicklable)
- **Rebuild Strategy**: indi_index rebuilt from reader when loading from cache (fast - just indexing, no data extraction)
- **User Visibility**: Clear messages show cache source ("Using GEDCOM cache" vs "Using GEDCOM file")

**Why This Design**:
- Individual objects from ged4py cannot be pickled (contain BinaryFileCR file reader references)
- Caching processed data structures avoids expensive re-extraction
- Rebuilding indi_index is fast because it only indexes existing records
- File modification time ensures cache stays synchronized with GEDCOM file
- Multi-level caching provides optimal performance for different scenarios

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

### Phase 5: Research Assistant Features (October 21, 2025) 🚧 IN PROGRESS

**P5.1: Source Citation Extraction** ✅ COMPLETE
- ✅ Added TAG_SOUR and TAG_TITL constants for GEDCOM source tags
- ✅ Implemented _extract_sources_from_event() function
  - Extracts source citations from GEDCOM event records (BIRT, DEAT, etc.)
  - Returns list of source titles/descriptions
  - Handles missing sources gracefully
- ✅ Implemented get_person_sources() function
  - Extracts all sources for a person by event type
  - Returns dict: {'birth': [...], 'death': [...], 'other': [...]}
  - Validates individual records before extraction
- ✅ Implemented format_source_citations() function
  - Formats sources for human-readable display
  - Single source: "documented in 1881 Scotland Census (birth)"
  - Multiple sources: "documented in A (birth) and B (death)"
  - Three+ sources: "documented in A, B, and C"
- ✅ Added test_source_citation_extraction() with 7 comprehensive tests
- ✅ Added test_source_citation_demonstration() showing complete workflow
- ✅ All 17 gedcom_utils tests passing
- ✅ Quality: 100.0/100 across all 65 modules
- ✅ Usage: `sources = get_person_sources(individual); citation = format_source_citations(sources)`
- ✅ Ready for integration into Action 8/9 responses

**P5.2: Research Suggestion Generation** ✅ COMPLETE
- ✅ Created research_suggestions.py module
- ✅ Implemented generate_research_suggestions() function
  - Generates relevant Ancestry collection suggestions based on locations
  - Suggests specific record types based on common ancestors
  - Provides research strategies based on relationship context
  - Returns formatted message ready for use in conversations
- ✅ Added ANCESTRY_COLLECTIONS mapping for Scotland, England, Ireland, Canada, USA
- ✅ Added TIME_PERIOD_COLLECTIONS for 1800s and 1900s
- ✅ Implemented helper functions to reduce complexity:
  - _extract_location_collections()
  - _extract_time_period_collections()
  - _generate_record_types()
  - _generate_strategies()
  - _format_research_suggestion_message()
- ✅ Added 4 comprehensive tests
- ✅ All tests passing
- ✅ Quality: 100.0/100 across all 66 modules
- ✅ Usage: `result = generate_research_suggestions(ancestors, locations, periods, relationship)`
- ✅ Ready for integration into Action 8/9 responses

**P5.3-P5.10: Remaining Tasks** 🚧 PENDING
- P5.3: Enhanced MS To-Do task creation
- P5.4: Relationship diagram generation
- P5.5: Record sharing capabilities
- P5.6: Research guidance AI prompt
- P5.7-P5.10: Testing and completion

### Phase 4: Adaptive Messaging & Intelligent Dialogue (October 21, 2025) ✅ COMPLETE

**P4.1: Engagement-Based Follow-Up Timing** ✅ COMPLETE
- ✅ Implemented adaptive timing using engagement score + login activity
- ✅ Four timing tiers: High (7 days), Medium (14 days), Low (21 days), None (30 days)
- ✅ Configuration added to .env (9 new settings)
- ✅ calculate_adaptive_interval() function with tier logic
- ✅ Updated _check_message_interval() to use adaptive timing
- ✅ 5 comprehensive tests + demonstration script
- ✅ NOTE: Adaptive timing only applies in PRODUCTION mode (testing/dry_run use fixed intervals)

**P4.2: Status Change Detection** ✅ COMPLETE
- ✅ Implemented detect_status_change_to_in_tree() function
- ✅ Detects when DNA matches are recently added to family tree (within 7 days)
- ✅ Added STATUS_CHANGE_RECENT_DAYS=7 configuration
- ✅ Helper functions: _is_tree_creation_recent(), _has_message_after_tree_creation()
- ✅ 4 comprehensive tests (all passing)
- ✅ Complexity reduced to <11 via helper function extraction

**P4.3: Automatic Message Cancellation** ✅ COMPLETE
- ✅ Implemented cancel_pending_messages_on_status_change() function
- ✅ Updates conversation_state.next_action to 'status_changed'
- ✅ Sets next_action_date to NULL
- ✅ Prevents sending outdated out-of-tree messages when match added to tree
- ✅ 2 comprehensive tests (all passing)

**P4.4: Status Change Message Template** ✅ COMPLETE
- ✅ Created 'In_Tree-Status_Change_Update' template in database
- ✅ Category: 'status_change', Tree Status: 'in_tree'
- ✅ Subject: "Update: Found Our Family Connection!"
- ✅ Concise message announcing connection found with relationship_path
- ✅ 1 comprehensive test (passing)

**P4.5: Conversation Continuity (Cancel on Reply)** ✅ COMPLETE
- ✅ Implemented cancel_pending_on_reply() function
- ✅ Cancels pending follow-up messages when recipient replies
- ✅ Switches conversation_phase to 'active_dialogue'
- ✅ Updates next_action to 'await_reply'
- ✅ Idempotent operation (safe to call multiple times)
- ✅ 3 comprehensive tests (all passing)

**P4.6: Determine Next Action Function** ✅ COMPLETE
- ✅ Implemented determine_next_action() function
- ✅ Analyzes conversation state to determine next action and timing
- ✅ Returns tuple: (action, datetime)
- ✅ Actions: status_changed, await_reply, research_needed, send_follow_up, no_action
- ✅ Logic flow: status change → active dialogue → research needed → adaptive timing
- ✅ 5 comprehensive tests (all passing)

**P4.7: Conversation Flow Logging** ✅ COMPLETE
- ✅ Implemented log_conversation_state_change() function
- ✅ Logs all state transitions: phase changes, next_action updates, cancellations, status changes
- ✅ Includes: person ID, username, old/new values, engagement score, timestamp
- ✅ Integrated into: cancel_pending_messages_on_status_change(), cancel_pending_on_reply(), determine_next_action()
- ✅ Logging format: 🔄 Conversation state change for [username] (ID [id]): [type] '[old]' → '[new]' (engagement: [score])
- ✅ 2 comprehensive tests (all passing)

**P4.8-P4.10: Comprehensive Testing** ✅ COMPLETE
- ✅ P4.8: Adaptive timing tested with 5 tests covering all engagement tiers
- ✅ P4.9: Status change workflow tested with 4 tests
- ✅ P4.10: Conversation continuity tested with 3 tests
- ✅ Total: 23 Phase 4 tests, all passing
- ✅ Quality: 100.0/100 across all 65 modules
- ✅ Complexity: All functions <11

**Phase 4 Summary**:
- ✅ 7 new functions implemented
- ✅ 23 comprehensive tests (all passing)
- ✅ 10 new .env configuration settings
- ✅ Intelligent action determination based on conversation state
- ✅ Adaptive timing based on engagement and login activity
- ✅ Status change detection and handling
- ✅ Conversation flow logging for monitoring
- ✅ All complexity <11, quality 100.0/100

### Phase 3: Conversational Dialogue Engine (October 21, 2025) ✅ COMPLETE
- ✅ Created genealogical_dialogue_response AI prompt (v1.0.0)
- ✅ Created engagement_assessment AI prompt (v1.0.0)
- ✅ Implemented generate_contextual_response() function
- ✅ Multi-person lookup and response capabilities
- ✅ Conversation phase tracking (initial_outreach, active_dialogue, research_exchange)
- ✅ Engagement scoring implementation (0-100 scale)
- ✅ Reduced complexity in ai_interface.py (_format_dialogue_prompt < 11)
- ✅ Comprehensive test infrastructure created and validated

### Phase 2: Person Lookup Integration (October 21, 2025) ✅ COMPLETE
- ✅ Created ConversationState database table
- ✅ Created PersonLookupResult dataclass in person_lookup_utils.py
- ✅ Enhanced AI entity extraction to capture detailed person objects
- ✅ Implemented lookup_mentioned_people() using Action 10 GEDCOM search
- ✅ Integrated lookup results into AI response generation pipeline
- ✅ Added conversation state tracking with phase detection
- ✅ Engagement scoring (0-100) based on conversation quality
- ✅ 6 comprehensive tests for PersonLookupResult functionality

### Phase 1: Enhanced Message Content (October 21, 2025) ✅ COMPLETE
- ✅ Implemented tree statistics calculation and caching
- ✅ Added ethnicity commonality calculation
- ✅ Database caching with 24-hour expiration (TreeStatisticsCache table)
- ✅ Updated message templates with enhanced content
- ✅ Comprehensive testing in dry_run mode
- ✅ All 9 Phase 1 tasks completed

### Phase 8: Action 10 GEDCOM Caching & Optimization (October 20, 2025)
- ✅ Implemented multi-level GEDCOM caching (memory → disk → file)
- ✅ Fixed disk cache serialization (Individual objects cannot be pickled)
- ✅ Added automatic cache invalidation based on file modification time
- ✅ Improved performance: 32s → 5-6s (6x faster with disk cache)
- ✅ Fixed Action 10 session requirement bug (was requiring session_ready, now db_ready)
- ✅ Added clear user-visible cache messages
- ✅ Verified NO API calls in Action 10 (pure local GEDCOM analysis)
- ✅ Fixed all Pylance errors in action10.py, gedcom_utils.py, relationship_utils.py
- ✅ Removed unused parameters and imports
- ✅ Updated README with comprehensive Action 10 and caching documentation

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
1. TreesUI List API - `trees/{tree_id}/persons` - Person search within a tree
2. Facts User API - `family-tree/person/facts/user/{owner_profile_id}/tree/{tree_id}/person/{person_id}` - Detailed person information
3. GetLadder API - `family-tree/person/tree/{tree_id}/person/{person_id}/getladder` - Relationship path calculation (tree-based)
4. Discovery Relationship API - `discoveryui-matchingservice/api/relationship` - Relationship path calculation (profile-based, fallback)

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Action 6 First Run** | ~10 minutes (100 people) |
| **Action 6 Second Run** | ~30 seconds (cached) |
| **Action 7 First Run** | ~4.8 hours (15,000 conversations) |
| **Action 7 Second Run** | ~5-10 minutes (95% skip rate) |
| **Action 8 Processing** | ~0.86 candidates/second |
| **Action 10 First Run** | ~32 seconds (14,640 individuals, no cache) |
| **Action 10 Disk Cache** | ~5-6 seconds (6x faster) |
| **Action 10 Memory Cache** | Instant (same session) |
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

**Last Updated**: October 21, 2025
**Status**: Phase 4 (P4.1) Complete - Intelligent Conversational Messaging with Adaptive Timing
