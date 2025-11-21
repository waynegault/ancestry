# Maintainer Handoff Guide

## 1. System Overview

The Ancestry Research Automation Platform is a Python-based system designed to automate genealogical research tasks on Ancestry.com. It leverages Selenium for browser automation, SQLAlchemy for database persistence, and various AI providers (Gemini, DeepSeek) for intelligent data processing.

### Key Architecture Patterns

- **Session Management**: `core/session_manager.py` is the central coordinator. It manages the Selenium WebDriver lifecycle, authentication state, and API rate limiting. All actions must request resources through the `SessionManager`.
- **Action Architecture**: The system is organized into "Actions" (e.g., `action6_gather.py`, `action7_inbox.py`). Each action is a standalone module invoked by `main.py` via the `exec_actn` wrapper, which handles error recovery and logging.
- **Unified Caching**: `core/unified_cache_manager.py` and `cache.py` provide a multi-tiered caching strategy (memory + disk) to minimize API calls and improve performance.
- **Error Handling**: `core/error_handling.py` provides centralized retry logic (`@api_retry`, `@selenium_retry`) and error classification.

## 2. Critical Components

### Action 6: DNA Match Gathering (`action6_gather.py`)
- **Purpose**: Harvests DNA matches and their details.
- **Complexity**: High (9000+ lines). Contains logic for pagination, API fetching, ethnicity enrichment, and database persistence.
- **Key Risks**: Rate limiting (429 errors) and session expiry during long runs.
- **Mitigation**: Uses `RateLimiter` (sequential processing) and proactive session health checks.

### Action 10: GEDCOM Analysis (`action10.py`)
- **Purpose**: Compares GEDCOM file data with Ancestry API search results.
- **Workflow**: Loads GEDCOM -> Searches local tree -> Falls back to API if no local match -> Displays side-by-side comparison.
- **Key Dependencies**: `gedcom_utils.py`, `api_search_core.py`.

### Database (`database.py`, `core/database_manager.py`)
- **Schema**: SQLite database (`Data/ancestry.db`).
- **ORM**: SQLAlchemy.
- **Migrations**: Managed by `core/schema_migrator.py`. Always run migrations after pulling code changes.

## 3. Maintenance Workflows

### Testing
The project maintains a strict 100% pass rate policy.
```powershell
# Run all tests
python run_all_tests.py

# Run integration tests
python run_all_tests.py --integration

# Run fast tests (skip slow ones)
python run_all_tests.py --fast
```

### Code Quality
Enforce style and type safety before committing.
```powershell
# Linting
ruff check .

# Type Checking
npx pyright
```

### Schema Migrations
When modifying `database.py` models, create a new migration script in `core/migrations/` (if applicable) or update `core/schema_migrator.py`.
```powershell
# Check migration status
python core/schema_migrator.py --list

# Apply migrations
python core/schema_migrator.py --apply
```

### Documentation
- **`docs/code_graph.json`**: Represents the codebase structure. Update it using `scripts/maintain_code_graph.py` if you remove/rename modules.
- **`readme.md`**: Keep the "Recent Improvements" section up to date.

## 4. Troubleshooting Common Issues

### Rate Limiting (429 Too Many Requests)
- **Symptom**: Logs show "429 error" and long backoff times.
- **Cause**: `REQUESTS_PER_SECOND` in `.env` is too high.
- **Fix**: Reduce `REQUESTS_PER_SECOND` (default 0.3) and ensure `THREAD_POOL_WORKERS` is set to 1 (sequential processing).

### Session Expiry / Auth Failures
- **Symptom**: "Session not ready" or 401/403 errors in logs.
- **Cause**: Browser cookies expired or Ancestry forced a logout.
- **Fix**: The system attempts auto-recovery. If that fails, restart the script to trigger a fresh login. Check `Logs/app.log` for "Session death cascade".

### Database Locks
- **Symptom**: "database is locked" errors.
- **Cause**: Concurrent writes or open transactions.
- **Fix**: Ensure all DB operations use `with db_transn(session):` or `with session_scope():`. Avoid manual `commit()` calls inside loops.

## 5. Future Roadmap & Technical Debt

### High Priority
- **Refactor `action6_gather.py`**: Split the monolithic file into smaller, focused modules (e.g., `action6/fetcher.py`, `action6/processor.py`).
- **Resolve Import Cycles**: `cache.py` and `gedcom_cache.py` have circular dependencies. Refactor to use dependency injection or move shared types to a separate module.

### Medium Priority
- **Strict Type Checking**: Move `pyrightconfig.json` towards "strict" mode by fixing remaining `Any` types.
- **Expand Integration Tests**: Cover more complex workflows like Action 8 (Messaging) in `test_integration_workflow.py`.

### Low Priority
- **UI Modernization**: The CLI menu (`main.py`) is functional but could be replaced with a TUI (Text User Interface) library like `textual` for better UX.
