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

### Appendix A: Test Coverage Report

For a comprehensive analysis of test coverage across all Python modules, see [test_coverage_report.md](test_coverage_report.md).

**Summary:**
- **Total Modules Analyzed:** 82
- **Modules with Tests:** 80 (97.6%)
- **Total Public Functions:** 1,048
- **Total Test Functions:** 1,033
- **Test Pass Rate:** 100%

### Appendix B: Test Quality Analysis

For a detailed assessment of test quality, duplication analysis, and recommendations, see [test_quality_analysis.md](test_quality_analysis.md).

**Key Findings:**
- ✅ All tests validate real functionality (no smoke tests)
- ✅ Tests use live authentication via `get_authenticated_session()`
- ✅ Comprehensive coverage of core functionality, edge cases, and error handling
- ✅ Excellent DRY adherence with centralized test utilities
- ✅ Code quality: 100.0/100 across all modules

### Appendix C: Test Review Summary

For an executive summary of the comprehensive test review completed on 2025-10-23, see [TEST_REVIEW_SUMMARY.md](TEST_REVIEW_SUMMARY.md).

**Highlights:**
- Fixed all linter issues (21 auto-fixed, 4 manual fixes)
- Reduced complexity in 3 functions (database.py, conversation_analytics.py, core/session_manager.py)
- Achieved 100% code quality score across all 82 modules
- Confirmed minimal test duplication with excellent DRY principles
- Overall Grade: A+
