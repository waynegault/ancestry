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

### Architecture

**Core Components**:
```
core/
  session_manager.py    # Central coordinator with RateLimiter
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

### Rate Limiting & Circuit Breaker

**Single Rate Limiter Architecture**:
- Class: `RateLimiter` in `utils.py`
- Instance: `session_manager.rate_limiter`
- Algorithm: Thread-safe token bucket
- Configuration:
  - Capacity: 10 tokens
  - Fill rate: 2 tokens/second
  - Thread-safe with `threading.Lock()`

**Circuit Breaker Integration**:
- Automatic protection from cascading 429 failures
- Failure threshold: 5 consecutive 429 errors
- Recovery timeout: 60 seconds
- Half-open test requests: 3

**CRITICAL**: Do not modify rate limiting settings without extensive validation!

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

### Code Quality

- Follow DRY (Don't Repeat Yourself) principles
- Use type hints for function signatures
- Add docstrings to all public functions
- Keep functions under 50 lines when possible
- Fix all pylance errors before committing

### Git Workflow

- Commit at each phase of implementation
- Write descriptive commit messages
- Test before committing
- Revert if tests fail

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

**Last Updated**: October 19, 2025
**Status**: Production Ready
