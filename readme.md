# Ancestry Genealogical Research Automation

Comprehensive Python automation system for Ancestry.com genealogical research, featuring intelligent messaging, DNA match analysis, and family tree management.

## Overview

This project automates genealogical research workflows on Ancestry.com, including:
- **Action 6**: Automated page gathering and data collection
- **Action 7**: Inbox message processing and analysis
- **Action 8**: Intelligent messaging with AI-powered responses
- **Action 9**: Productive conversation management
- **Action 10**: GEDCOM file analysis and scoring
- **Action 11**: API-based genealogical research and relationship discovery

## Quick Start

### Prerequisites
- Python 3.12+
- Chrome browser
- Ancestry.com account

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

### Configuration

Create a `.env` file with the following required settings:

```bash
# Ancestry Credentials
ANCESTRY_USERNAME=your_username
ANCESTRY_PASSWORD=your_password

# Rate Limiting (CRITICAL - Do not change without validation)
THREAD_POOL_WORKERS=1
REQUESTS_PER_SECOND=0.4

# Processing Limits (Conservative defaults)
MAX_PAGES=1
MAX_INBOX=5
MAX_PRODUCTIVE_TO_PROCESS=5
BATCH_SIZE=5

# Application Mode
APP_MODE=development  # or 'production'
DEBUG_MODE=false

# AI Integration (Optional)
DEEPSEEK_API_KEY=your_deepseek_key
GOOGLE_API_KEY=your_google_key
AI_PROVIDER=deepseek  # or 'gemini' or ''

# Microsoft Graph (Optional - for To-Do integration)
MS_CLIENT_ID=your_client_id
MS_CLIENT_SECRET=your_client_secret
MS_TENANT_ID=your_tenant_id
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

### Global Session Pattern (Single Source of Truth)

- Only main.py creates SessionManager and registers it globally via session_utils.set_global_session(session_manager)
- All actions/scripts must obtain the session via session_utils.get_global_session() and authenticate with session_utils.get_authenticated_session()
- No module-level or ad-hoc SessionManager() creation anywhere else in the codebase
- Recent refactor highlights:
  - action11.py: removed module-level SessionManager creation; uses only the global session
  - action8_messaging.py: template loading and helpers now rely on the global session
  - scripts/test_editrelationships_shape.py: uses the global session (requires main.py to register it first)

### Rate Limiting (CRITICAL)

**Single Rate Limiter Architecture:**
- Class: `RateLimiter` in `utils.py`
- Instance: `session_manager.rate_limiter`
- Algorithm: Thread-safe token bucket
- Configuration:
  - Capacity: 10 tokens
  - Fill rate: 2 tokens/second
  - Thread-safe with `threading.Lock()`

**DO NOT modify rate limiting settings without extensive validation!**
- Changing `THREAD_POOL_WORKERS` or `REQUESTS_PER_SECOND` can cause 429 errors
- 429 errors result in 72-second penalties
- Monitor: `Select-String -Path Logs\app.log -Pattern "429 error"` should return 0

### Database Schema

SQLite database (`Data/ancestry.db`) with tables:
- `people`: Person records with genealogical data
- `messages`: Message history and AI responses
- `conversations`: Conversation threads
- `tasks`: Microsoft To-Do integration
- `dna_matches`: DNA match data
- `research_logs`: Research activity tracking

## Vision and Roadmap (Consolidated)

Create an AI-powered genealogical research assistant that conducts intelligent, contextually-aware conversations with DNA matches, automatically researching family connections and providing substantive genealogical insights while respecting user preferences and managing conversation lifecycle.

Core capabilities to deliver incrementally:
- Intelligent initial outreach: in-tree messages include relationship path; out-of-tree messages include tree statistics and DNA commonality
- Conversational dialogue engine: detect people mentioned in messages; look them up via Action 10 (GEDCOM) and Action 11 (API); generate contextual replies
- Adaptive sequencing: follow-up timing adapts to engagement and status changes (out_tree -> in_tree)
- Do-not-contact management: detect and honor desist immediately; cancel scheduled messages
- Research assistant features: source citations, research suggestions, relationship diagrams, record sharing
- Conversation state management: track engagement_score, last_topic, pending_questions, ai_summary

Phase status snapshot:
- Phase 1 (Enhanced content): Complete foundations; relationship path and stats support available
- Phase 2 (Person lookup integration): Partially implemented in Action 11; needs Action 9 dialogue glue
- Phase 3 (Dialogue engine): Implemented core engagement assessment and conversation state fields
- Phase 4 (Adaptive messaging): Partially implemented; follow-up adaptation queued
- Phase 5 (Research assistant): Enrichment policy and formatting in place (Action 9)
- Phase 6 (Monitoring & analytics): Planned
- Phase 7 (Local LLM): Provider and tests implemented; see Local LLM Integration

## Developer Instructions (Key Topics)

- Architecture and global session pattern: see sections above
- Actions 6–11: see per-action sections below
- Testing and quality: run_all_tests.py is authoritative; tests must fail on genuine failures
- Pylance/linters: fix errors, do not suppress; reduce function complexity (target <10) and keep functions short
- Technical specs: see Appendix B (Action 11 endpoints, display rules, logging, AI provider config)

## Local LLM Integration (Phase 7 – Real Use)

Local models are supported via LM Studio (OpenAI-compatible server). Configure and validate as follows:

1. .env settings
```bash
AI_PROVIDER="local_llm"
LOCAL_LLM_API_KEY="lm-studio"        # LM Studio default
LOCAL_LLM_MODEL="qwen3-4b-2507"      # Or any model you load in LM Studio
LOCAL_LLM_BASE_URL="http://localhost:1234/v1"
```

1. Start LM Studio (GUI)
- Open LM Studio
- Load your chosen instruct model (e.g., qwen3-4b-2507 or Llama/Mistral)
- Click "Start Server"; ensure it shows Running at http://localhost:1234/v1

1. Validate with the real tests
```bash
python test_local_llm.py
```
- Test 1 (Direct connection) should PASS in <5s typical
- Test 2 (Configuration) should PASS with AI_PROVIDER=local_llm
- Test 3 (Genealogical prompt) should PASS and mention at least 3 of: census, birth, marriage, death, record, search, scotland

1. Use in normal actions
- Keep LM Studio running
- Run Action 8 or 9 normally; AI calls will route to the local model

Troubleshooting
- Ensure the server status is Running and the model is loaded
- Verify .env AI_PROVIDER/local LLM vars are set exactly
- Restart LM Studio and re-run tests if connection errors persist

## Future Developer Ideas

- Engagement-based follow-up scheduler with activity heuristics
- Conversation analytics dashboard with trends and funnel metrics
- A/B testing for template variants and enrichment choices
- Automatic fallback to DeepSeek on local model failures
- Prompt variant optimization for local models to improve speed/quality
- Relationship diagram inline images using lightweight SVG rendering

## Actions

### Action 6: Page Gathering
Automated data collection from Ancestry pages with parallel processing.

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

#### Action 11 API sources and parsing notes

- Primary family endpoint: /family-tree/person/addedit/user/{owner_profile_id}/tree/{tree_id}/person/{person_id}/editrelationships
  - Response shape: { cssBundleUrl, jsBundleUrl, data }, where data is a JSON STRING that must be json.loads(...) into { userId, treeId, personId, person, urls, res }
  - The family arrays live under parsed_data.person: fathers[], mothers[], spouses[], children[] (children may be nested arrays per spouse)
  - res contains UI/localization strings, not family data
- Relationship ladder endpoint: /family-tree/person/card/user/{user_id}/tree/{tree_id}/person/{person_id}/kinship/relationladderwithlabels
- Design decisions:
  - Siblings are not displayed in Action 11 (per requirements); parents, spouses, and children are displayed
  - Session authentication occurs once via session_utils.get_authenticated_session; no redundant re-login or cookie syncs

```bash
python action10.py
```

### Action 11: API Research
API-based genealogical research with relationship discovery.

```bash
python action11.py
```

## Testing

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

## Troubleshooting

### Common Issues

**429 Rate Limit Errors:**
- Check `THREAD_POOL_WORKERS=1` in `.env`
- Check `REQUESTS_PER_SECOND=0.4` in `.env`
- Monitor: `Select-String -Path Logs\app.log -Pattern "429 error"`

**Import Errors:**
- Verify Python 3.12+ is installed
- Run `pip install -r requirements.txt`
- Check virtual environment activation

**Browser Issues:**
- Update Chrome to latest version
- Run `python -c "from core.browser_manager import BrowserManager; BrowserManager()"`
- Check ChromeDriver compatibility

**Database Errors:**
- Backup database: `python main.py` → Option 3
- Check file permissions on `Data/ancestry.db`
- Verify SQLite installation

### Logs

All logs are written to `Logs/app.log`:

```bash
# View recent errors
tail -100 Logs/app.log | grep ERROR

# Monitor real-time
tail -f Logs/app.log

# Check rate limiter initialization
grep "Thread-safe RateLimiter" Logs/app.log | tail -1
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run `python run_all_tests.py`
5. Submit pull request

## License

This project is for personal genealogical research use.

## Support

For issues or questions:
- GitHub Issues: https://github.com/waynegault/ancestry/issues
- Email: 39455578+waynegault@users.noreply.github.com

---

## Appendices

### Appendix A: Chronology of Changes

2025-10-24
- Eliminated module-level SessionManager creation in action11.py; switched to global session usage only
- Consolidated Action 11 authentication to a single path via session_utils.get_authenticated_session()
- Removed redundant login/cookie helper functions from action11.py
- Updated action8_messaging.py to load templates and get session via session_utils.get_global_session()
- Updated scripts/test_editrelationships_shape.py to require/use the global session
- Deferred Action 8 template loading to runtime (lazy initialization); eliminated import-time CRITICALs
- Added de-duplication for gender inference DEBUG logs in relationship_utils
- Reduced INFO banner noise in session_utils.get_authenticated_session; cache-hit now DEBUG
- Enhanced API call logging with durations in api_utils.call_enhanced_api
- Reworded cookie skip message in core/session_validator for clarity
- Temporarily disabled startup clear-screen in main.py to expose early errors
- Tests: run_all_tests.py passed (72/72 modules) after refactor

2025-10-26
- INFO logging tidy-up: removed "UUID: Not yet set" banner; if MY_UUID exists in .env we log it as "UUID (from .env) — will verify"; otherwise keep at DEBUG only
- Global session banner shows once on first auth; subsequent get_authenticated_session() calls reuse cache without re-printing banners
- Cookie sync logging clarified: "initial sync" first time, else "age Xs; threshold Ys"; removed misleading "cache expired, last sync ..." phrasing
- API Request Cookies logging compressed to count only (no full key dump)
- Edit Relationships API response logging summarized (type + top keys) instead of full structure dump
- Cookie-check message clarified: "Skipping essential cookies check (expected): …"
- Action 11: added success/error summary with duration at completion
- Action 8: lazy-load templates at runtime; no import-time CRITICALs
- Relationship logs: de-duplicated repeated gender inference debug lines
- Tests: run_all_tests.py passed (72/72 modules) in fast mode
- Re-enabled startup clear-screen in main.py (now that early error review is complete)
- Restored VISION_INTELLIGENT_DNA_MESSAGING.md to repository root from commit 2aee7d6 (temporary for implementation reference; will consolidate back into readme.md at completion per single-doc rule)

2025-10-27
- Complexity reduction: session_utils.get_authenticated_session split into `_assert_global_session_exists` and `_pre_auth_logging`; type-safety improvements
- Complexity reduction: api_search_utils.get_api_family_details decomposed into helpers for structure logging, section extraction, and per-relationship parsing
- Complexity reduction: action7_inbox extracted `_assess_and_upsert`; reduced analytics method branching
- Complexity reduction: action9_process_productive extracted `_formatting_fallback` to simplify exception path
- Documentation: Consolidated vision into readme.md; added Local LLM integration steps; cleaned markdown lint issues
- Policy: Removed messages.json earlier; VISION_INTELLIGENT_DNA_MESSAGING.md will be removed after verification (single-file docs policy)

2025-10-23
- Switched Action 11 family extraction to the Edit Relationships endpoint and parsed nested data['person'] correctly
- Suppressed verbose raw path logging in Action 11; kept concise debug metrics

### Appendix B: Technical Specifications

- Session Architecture
  - Exactly one SessionManager instance created by main.py
  - Registered globally via session_utils.set_global_session()
  - Consumers must call session_utils.get_authenticated_session(action_name=...) before API usage

- Action 11 Endpoints
  - Edit Relationships: /family-tree/person/addedit/user/{owner_profile_id}/tree/{tree_id}/person/{person_id}/editrelationships
    - Response: { cssBundleUrl, jsBundleUrl, data } where data is a JSON string; parse with json.loads
    - Family arrays: parsed['person'] → fathers[], mothers[], spouses[], children[]
  - Relationship Ladder: /family-tree/person/card/user/{user_id}/tree/{tree_id}/person/{person_id}/kinship/relationladderwithlabels

- Display Rules
  - Parents, spouses, children shown; siblings intentionally omitted in Action 11

- AI Providers and Local LLM
  - ai_provider: one of ["deepseek", "gemini", "local_llm"]
  - LOCAL_LLM_* when ai_provider=local_llm: LOCAL_LLM_API_KEY, LOCAL_LLM_MODEL, LOCAL_LLM_BASE_URL
  - Default base URL: http://localhost:1234/v1 (LM Studio)

- Conversation State Model (Phase 3)
  - Fields: ai_summary (text), engagement_score (int 0-100), last_topic (text), pending_questions (text/json), conversation_phase (enum)
  - Updated via Action 7 analytics pipeline and assess_engagement()

- Enrichment Policy (Phase 5)
  - Flag: enable_task_enrichment (bool)
  - When enabled, Action 9 appends enrichment lines (to-do links, research prompts, record sharing hints)

- Logging
  - Single header/footer, info-level friendly
  - No raw HTML dumps; log length/count metrics instead

- Testing
  - run_all_tests.py is the canonical runner; tests should fail on genuine failures
  - API-dependent tests assume live authentication through the global session
