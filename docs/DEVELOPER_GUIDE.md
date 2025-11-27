# Developer Guide

This guide covers architecture patterns, testing conventions, and development workflows for the Ancestry Research Automation Platform.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Testing Framework](#testing-framework)
4. [Code Quality Standards](#code-quality-standards)
5. [Common Patterns](#common-patterns)
6. [Debugging Guide](#debugging-guide)

---

## Architecture Overview

### Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Entry Point (main.py)                   │
├─────────────────────────────────────────────────────────────┤
│                   Action Modules (action6-11.py)            │
├─────────────────────────────────────────────────────────────┤
│                    Core Services Layer                       │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │SessionManager│ │ APIManager   │ │  DatabaseManager     │ │
│  │              │ │              │ │                      │ │
│  └──────────────┘ └──────────────┘ └──────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                  Infrastructure Layer                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │RateLimiter   │ │ ErrorHandling│ │  Correlation IDs     │ │
│  └──────────────┘ └──────────────┘ └──────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Data Layer (SQLAlchemy)                   │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **SessionManager as Central Orchestrator**: All browser, database, and API operations flow through SessionManager
2. **exec_actn() Wrapper**: Universal action wrapper in main.py handles resource management
3. **Rate Limiting First**: All API calls serialize through AdaptiveRateLimiter
4. **Type Safety**: Comprehensive type hints with Pyright validation
5. **Test Coverage**: Each module has embedded tests using TestSuite pattern

---

## Core Components

### SessionManager (`core/session_manager.py`)

The central coordinator for all resources:

```python
from core.session_manager import SessionManager

# Create session manager
session_manager = SessionManager()

# Ensure browser session is ready
session_manager.ensure_session_ready()

# Ensure database is ready (browser optional)
session_manager.ensure_db_ready()

# Check session validity
if session_manager.is_sess_valid():
    # Session is authenticated and within lifetime
    pass

# Get session age
age_seconds = session_manager.session_age_seconds()
```

**Key Methods:**
- `ensure_session_ready()` - Initialize browser, login, sync cookies
- `ensure_db_ready()` - Initialize database connection
- `is_sess_valid()` - Check if session is authenticated
- `session_age_seconds()` - Get session age for timeout management

### Session Management (`session_utils.py`)

Dependency injection-based session access with backward compatibility:

```python
from session_utils import (
    register_session_manager,  # Register at startup
    get_session_manager,       # Get session (preferred)
    is_session_available,      # Check availability
    requires_session,          # Decorator
    SessionNotAvailableError,  # Exception
)

# At startup (core/lifecycle.py does this automatically)
session_manager = SessionManager()
register_session_manager(session_manager)

# Access anywhere in code
sm = get_session_manager()
if sm and is_session_available():
    # Use session...
    pass

# Decorator for functions requiring session
@requires_session()
def my_function():
    sm = get_session_manager()
    # Use sm...

# Auto-inject session as first argument
@requires_session(inject_session=True)
def process_matches(session_manager: SessionManager, matches: list):
    # session_manager is automatically injected
    pass
```

**Key Features:**
- Uses DI container (`core/dependency_injection.py`)
- Modern DI pattern: use `get_session_manager()` and `register_session_manager()`
- Thread-safe access
- Clear error messages via `SessionNotAvailableError`

### APIManager (`core/api_manager.py`)

Unified API request handling with retry and rate limiting:

```python
from core.api_manager import APIManager, RequestConfig, RetryPolicy

api_manager = APIManager(session_manager)

# Make a request with retry
config = RequestConfig(
    url="https://api.example.com/data",
    method="POST",
    json_data={"key": "value"},
    retry_policy=RetryPolicy.RESILIENT,  # 3 retries with backoff
)

result = api_manager.request(config)
if result.success:
    data = result.json
```

**Retry Policies:**
- `RetryPolicy.NONE` - No retries
- `RetryPolicy.API` - 2 retries, API-optimized delays
- `RetryPolicy.RESILIENT` - 3 retries, longer backoff

### AdaptiveRateLimiter (`rate_limiter.py`)

Token bucket rate limiter with adaptive tuning:

```python
from rate_limiter import AdaptiveRateLimiter, get_adaptive_rate_limiter

# Get global instance (singleton)
limiter = get_adaptive_rate_limiter()

# Wait before making request
limiter.wait()

# Report success (gradual rate increase after threshold)
limiter.on_success()

# Report 429 error (immediate rate decrease)
limiter.on_429_error()

# Check observability
print(limiter.get_status_message())  # "⚡ Rate: 0.50 req/s | Tokens: 10.0/10.0"
print(limiter.get_health_status())   # "optimal", "degraded", "throttled", "critical"
```

**Configuration via .env:**
```env
REQUESTS_PER_SECOND=0.4      # Initial fill rate
MAX_RATE_LIMIT=5.0           # Maximum rate
MIN_RATE_LIMIT=0.1           # Minimum rate floor
```

### Error Handling (`core/error_handling.py`)

Domain-specific exception hierarchy:

```python
from core.error_handling import (
    AncestryError,           # Base exception
    RetryableError,          # Should retry
    APIRateLimitError,       # 429 error with retry_after
    NetworkTimeoutError,     # Transient network issue
    FatalError,              # DO NOT retry
    DataValidationError,     # Invalid data
    retry_on_failure,        # Decorator
    graceful_degradation,    # Fallback decorator
)

# Automatic retry with exponential backoff
@retry_on_failure(max_attempts=3, backoff_factor=2.0)
def fetch_data():
    pass

# Graceful fallback
@graceful_degradation(fallback_value=[])
def get_optional_data():
    pass
```

### Correlation IDs (`core/correlation.py`)

Request tracking across operations:

```python
from core.correlation import correlation_context, get_current_correlation_id

# Create correlation context for a request
with correlation_context(operation="fetch_matches", user_id="user123") as ctx:
    # All logs within this context include correlation_id
    logger.info("Processing")  # Includes [correlation_id] prefix

    # Nested operations inherit the context
    process_data()

    # Access correlation ID if needed
    print(f"Request ID: {ctx.correlation_id}")
```

---

## Testing Framework

### Test Pattern

Every module uses the standardized TestSuite pattern:

```python
from test_framework import TestSuite
from test_utilities import create_standard_test_runner

def module_tests() -> bool:
    """Module-specific test implementation."""
    suite = TestSuite("Module Name", "module_file.py")

    # Test 1: Basic functionality
    suite.run_test(
        test_name="Basic initialization",
        test_func=_test_initialization,
        test_summary="Verify component initializes correctly",
        functions_tested="ComponentClass.__init__",
        method_description="Create component with valid parameters",
        expected_outcome="Component created with correct state",
    )

    return suite.finish_suite()

# Create standardized test runner
run_comprehensive_tests = create_standard_test_runner(module_tests)

def _test_initialization() -> None:
    """Test basic initialization."""
    component = ComponentClass()
    assert component.is_valid, "Component should be valid after init"

if __name__ == "__main__":
    import sys
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
```

### Test Utilities (`test_utilities.py`)

Common test decorators and factories:

```python
from test_utilities import (
    with_temp_database,
    with_mock_session,
    with_test_config,
    create_test_match,
    create_test_person,
)

# Isolated database for each test
@with_temp_database
def test_database_operation(db_session):
    person = create_test_person()
    db_session.add(person)
    db_session.commit()

# Mock session manager
@with_mock_session
def test_api_call(mock_session):
    mock_session.api_manager.request.return_value = {"data": "test"}

# Temporary config override
@with_test_config({"MAX_PAGES": 5})
def test_with_custom_config():
    pass
```

### Running Tests

```powershell
# Run all 118 modules (sequential)
python run_all_tests.py

# Parallel execution (faster)
python run_all_tests.py --fast

# Single module
python -m rate_limiter

# Quick unit tests only
python scripts/run_tests_fast.py
```

---

## Code Quality Standards

### Linting (Ruff)

```powershell
# Check and auto-fix
ruff check --fix .

# Check only
ruff check .
```

### Type Checking (Pyright)

Configuration in `pyrightconfig.json`:
- Standard type checking mode
- Python 3.13 target
- Strict mode disabled (too many legacy violations)

### Pre-commit Hooks

Automatically runs on commit:
- Ruff linting
- Large file detection
- Private key detection
- Trailing whitespace removal
- End of file fixing

---

## Common Patterns

### Action Function Pattern

All action functions follow this signature:

```python
def action_function(session_manager: SessionManager, *args) -> bool:
    """
    Perform action with session manager.

    Args:
        session_manager: Centralized session manager
        *args: Additional arguments passed from menu

    Returns:
        True if action completed successfully
    """
    # Use exec_actn() wrapper in main.py - never call directly
    pass
```

### Database Transaction Pattern

```python
from database import db_transn

with db_transn(session) as transaction:
    # All operations in transaction
    person = Person(name="Test")
    transaction.add(person)
    # Auto-commits on exit, rollback on exception
```

### UUID Handling

All UUIDs are stored uppercase:

```python
# Always uppercase for storage/lookup
uuid = raw_uuid.upper()
person = session.query(Person).filter(Person.uuid == uuid).first()
```

### Checkpoint Pattern (Action 6)

```python
# Save checkpoint after each page
_save_checkpoint(current_page, total_pages, state)

# Resume from checkpoint
checkpoint = _load_checkpoint()
if checkpoint:
    start_page = checkpoint["current_page"]
```

---

## Debugging Guide

### Rate Limiting Issues

```powershell
# Check for 429 errors (should be 0)
(Select-String -Path Logs\app.log -Pattern "429 error").Count

# Verify rate limiter initialization
Select-String -Path Logs\app.log -Pattern "RateLimiter" | Select-Object -Last 5

# Check rate limiter health
python -c "from rate_limiter import get_adaptive_rate_limiter; print(get_adaptive_rate_limiter().get_health_status())"
```

### Session Issues

```powershell
# Check session validity
python -c "from core.session_manager import SessionManager; sm = SessionManager(); print(f'Valid: {sm.is_sess_valid()}')"

# Check session age
python -c "from core.session_manager import SessionManager; sm = SessionManager(); print(f'Age: {sm.session_age_seconds()}s')"
```

### Database Issues

```powershell
# Check connection pool
python -c "from database import engine; print(engine.pool.status())"

# Direct SQLite access
sqlite3 Data/ancestry.db ".tables"
```

### AI Extraction Issues

```powershell
# Check telemetry statistics
python prompt_telemetry.py --stats

# Review recent AI responses
Get-Content Logs\prompt_experiments.jsonl -Tail 20
```

---

## File Organization

```
├── main.py                 # Entry point, exec_actn() wrapper
├── action6-11.py          # Action modules
├── database.py            # SQLAlchemy ORM models
├── rate_limiter.py        # Adaptive rate limiting
├── core/
│   ├── session_manager.py # Central orchestrator
│   ├── api_manager.py     # Unified API handling
│   ├── error_handling.py  # Exception hierarchy
│   ├── correlation.py     # Request tracking
│   └── ...
├── config/
│   ├── config_schema.py   # Type-safe configuration
│   └── config_manager.py  # .env loading
├── ai_interface.py        # Multi-provider AI abstraction
├── test_framework.py      # TestSuite implementation
├── test_utilities.py      # Test decorators and factories
└── docs/
    └── DEVELOPER_GUIDE.md # This file
```

---

## Contributing

1. **Create feature branch** from `main`
2. **Write tests first** using TestSuite pattern
3. **Run full test suite** before committing: `python run_all_tests.py`
4. **Ensure code quality**: `ruff check --fix .`
5. **Update documentation** if adding new patterns

---

## Quick Reference

| Task | Command |
|------|---------|
| Run all tests | `python run_all_tests.py` |
| Run tests fast | `python run_all_tests.py --fast` |
| Run single module | `python -m module_name` |
| Lint and fix | `ruff check --fix .` |
| Check types | `pyright` |
| Database backup | `python main.py` → Option 3 |
