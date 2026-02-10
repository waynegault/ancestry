# Core Package Review Report

**Scope**: Every `.py` file in `core/` (53 files, ~43,000 lines)
**Categories**: Duplication, Consolidation, Complexity, Test Quality, Linting/Types

---

## Executive Summary

The `core/` package has **significant structural duplication** across 5 major areas that inflate maintenance burden and create confusion about which abstraction to use. Several large files (4000+ lines) need decomposition. Test quality is uneven — some modules have strong behavioral tests, while others rely on shallow existence checks or trivial dataclass assertions. Two parallel exception hierarchies coexist with no clear migration path.

**Top priority findings:**
1. **Two competing exception hierarchies** in `error_handling.py` (39 classes, 2318 lines)
2. **Three `RequestConfig` dataclasses** across three files
3. **Two circuit breaker implementations** that overlap
4. **Five+ overlapping cache abstractions** across 8 files
5. **`utils.py` is a 4082-line monolith** that should be decomposed

---

## 1. Duplication

### 1A. Duplicate `RequestConfig` (HIGH)

| File | Class | Fields |
|------|-------|--------|
| `common_params.py:321` | `RequestConfig` | url, method, headers, referer_url, use_csrf_token, add_default_origin, timeout, allow_redirects, data, json_data, json, force_text_response, cookie_jar |
| `api_manager.py:73` | `RequestConfig` | url, method, headers, params, data, json_data, timeout, auth, cookies, verify_ssl, allow_redirects, stream |
| `utils.py:160` | `ApiRequestConfig` | url, method, data, json_data, json, headers, referer_url, use_csrf_token, add_default_origin, timeout, cookie_jar, allow_redirects, force_text_response, max_retries, initial_delay, backoff_factor, max_delay, retry_status_codes, retry_policy, jitter_seconds, api_description, attempt, session_manager |

**Impact**: Consumers must know which `RequestConfig` to import and where. Fields overlap ~70% but diverge just enough to cause confusion.
**Fix**: Consolidate into a single `RequestConfig` in `common_params.py` with all needed fields. Remove the other two. Use optional fields for retry/session concerns.

### 1B. Duplicate Circuit Breaker (HIGH)

| File | Class | Notes |
|------|-------|-------|
| `error_handling.py:518` | `CircuitBreaker` | Generic implementation with `call()` wrapper, uses `CircuitState` enum |
| `circuit_breaker.py:55` | `SessionCircuitBreaker` | Session-specific with Prometheus metrics, uses string-based `CircuitBreakerState` |

Both use `CircuitBreakerOpenError` from `error_handling.py`. Both implement `record_success()`, `record_failure()`, `reset()`, state tracking, thread safety via Lock, and recovery timeouts.

**Impact**: Two parallel state machines with slightly different APIs. `error_handling.py:CircuitBreaker` uses `call()` to wrap function execution; `SessionCircuitBreaker` uses `is_tripped()` for manual checks. Consumer must know which to use.
**Fix**: Keep `SessionCircuitBreaker` (richer, metrics-integrated) and deprecate `error_handling.py:CircuitBreaker`. Or merge features into one class with optional metrics.

### 1C. Duplicate Exception Hierarchies (HIGH)

`error_handling.py` contains **two parallel exception trees**:

**Tree 1** (line 610+): `AncestryError` → `RetryableError` / `FatalError`
- `APIRateLimitError`, `NetworkTimeoutError`, `DatabaseConnectionError`
- `DataValidationError`, `MissingConfigError`, `AuthenticationExpiredError`, `BrowserSessionError`, `MaxApiFailuresExceededError`, `ConfigurationError`

**Tree 2** (line 737+): `AppError(Exception)` — completely separate base
- `AuthenticationError(AppError)`, `ValidationError(AppError)`, `DatabaseError(AppError)`, `NetworkError(AppError)`, `BrowserError(AppError)`, `APIError(AppError)`

These cover the same domain categories (auth, database, network, browser, API, validation) with different base classes, different metadata structures, and different naming. Both are exported.

**Impact**: Callers must guess which hierarchy to catch. `except DatabaseConnectionError` won't catch `DatabaseError` and vice versa.
**Fix**: Pick one hierarchy and migrate. `AncestryError` tree is simpler and used in practice. Add `AppError`'s metadata fields (severity, category, user_message) to `AncestryError` subclasses if needed, then remove `AppError` subtree.

### 1D. Duplicate Retry Configuration (MEDIUM)

| File | Class/Enum | Purpose |
|------|-----------|---------|
| `error_handling.py:101` | `RetryConfig` dataclass | max_retries, backoff_factor, max_delay, jitter, retry_on |
| `error_handling.py:115` | `RetryPolicyProfile` enum | AGGRESSIVE / STANDARD / CONSERVATIVE / API / GENTLE / NONE |
| `error_handling.py:133` | `RetryDecoratorSettings` dataclass | max_attempts, delay, backoff_factor, max_delay, jitter_range, retryable_exceptions |
| `common_params.py:44` | `RetryContext` dataclass | attempt, max_attempts, max_delay, backoff_factor |
| `utils.py:160` | `ApiRequestConfig` | max_retries, initial_delay, backoff_factor, max_delay, retry_policy, jitter_seconds |

Five overlapping retry configuration structures.
**Fix**: Consolidate to `RetryConfig` + `RetryPolicyProfile` in `error_handling.py`. Remove `RetryContext` and inline retry fields from `ApiRequestConfig`.

### 1E. Duplicate Cache Abstractions (HIGH)

| File | Lines | Key Classes | Purpose |
|------|-------|-------------|---------|
| `cache_backend.py` | 623 | `CacheBackend` (Protocol), `CacheStats`, `CacheHealth`, `CacheFactory` | Abstract interface |
| `cache/interface.py` | 517 | `CacheConfig`, `CacheEntry`, `InvalidationPattern`, `Cache` (Protocol) | Another abstract interface |
| `cache/adapters.py` | 912 | `MemoryCache`, `DiskCacheAdapter`, `TTLCache`, `NullCache` | Concrete implementations |
| `unified_cache_manager.py` | 1202 | `UnifiedCacheManager`, `UnifiedCacheBackendAdapter(CacheBackend)` | High-level orchestrator |
| `session_cache.py` | 816 | `SessionComponentCache`, `OptimizedSessionState` | Session-specific cache |
| `system_cache.py` | 985 | `APIResponseCache`, `DatabaseQueryCache`, `MemoryOptimizer` | System-wide caches |
| `cache_registry.py` | 361 | `CacheRegistry`, `CacheComponent` | Registry of all caches |
| `caching_bootstrap.py` | 134 | `ensure_caching_initialized()` | Bootstrap |

**8 files, ~4550 lines** for caching. Two separate Protocol definitions (`CacheBackend` in `cache_backend.py` and `Cache` in `cache/interface.py`) define similar contracts. `CacheEntry` is defined in both `cache/interface.py` and `unified_cache_manager.py`.

**Fix**:
1. Pick ONE cache protocol (recommend `cache/interface.py` since it's richer)
2. Merge `cache_backend.py:CacheStats/CacheHealth` into `cache/interface.py`
3. Delete `cache_backend.py`
4. Consider merging `session_cache.py` + `system_cache.py` into `unified_cache_manager.py`

### 1F. Duplicate Protocol Definitions (MEDIUM)

| File | Protocols |
|------|-----------|
| `protocols.py` | `RateLimiterProtocol`, `SessionManagerProtocol`, `SessionManagerLike`, `DatabaseSessionProtocol`, `LoggerProtocol`, `CacheProtocol`, `SupportsBrowserConsoleLogs` |
| `type_definitions.py` | `Loggable`, `Cacheable`, `Scoreable` (+ `APIResponse`, `SearchCriteria` TypedDicts) |
| `action_runner.py` | `DatabaseManagerProtocol`, `BrowserManagerProtocol`, `APIManagerProtocol` (inline) |

`APIResponse` is defined in BOTH `protocols.py:207` and `type_definitions.py:125` — different fields.
`SearchCriteria` exists in both `type_definitions.py:99` (TypedDict) and `common_params.py:306` (dataclass) — different structures.
`CacheProtocol` in `protocols.py` overlaps with `Cache` Protocol in `cache/interface.py`.

**Fix**: Consolidate all protocol definitions into `protocols.py`. Move TypedDicts to `type_definitions.py`. Remove duplicates.

### 1G. Logging Utilities Delegation in `utils.py` (LOW)

`utils.py` lines 148-155 re-exports 7 functions from `logging_utils.py` as module-level aliases:
```python
log_action_configuration = _log_action_configuration_impl
log_starting_position = _log_starting_position_impl
...
```
These are pure pass-through re-exports. Callers should import from `logging_utils.py` directly.

---

## 2. Consolidation Opportunities

### 2A. `utils.py` — Decompose the 4082-Line Monolith (CRITICAL)

`utils.py` is the largest file after `database.py`. It contains:
- `ApiRequestConfig` dataclass (duplicate, covered above)
- `parse_cookie()`, `fast_json_loads()` — pure utility functions
- `load_login_cookies()` — session management
- `ordinal_case()`, `format_name()` — text formatting
- `retry()`, `time_wait()` — decorators
- `get_rate_limiter()` — rate limiter factory
- `_prepare_base_headers()`, `_prepare_api_headers()`, `_add_origin_header()`, `_parse_csrf_token()`, `_add_csrf_token_header()`, `_add_user_id_header()` — API header preparation (~200 lines)
- `make_ube()` — browser utility
- `handle_two_fa()`, `enter_creds()`, `consent()` — login flows (~800 lines)
- `log_in()`, `login_status()` — session login (~500 lines)
- `nav_to_page()` — navigation (~100 lines)
- `prevent_system_sleep()`, `restore_system_sleep()` — OS integration
- Module tests (~300 lines)

**Proposed decomposition:**

| New Module | Functions to Move | Est. Lines |
|-----------|------------------|------------|
| `core/auth.py` | `handle_two_fa`, `enter_creds`, `consent`, `log_in`, `login_status`, `load_login_cookies` | ~1500 |
| `core/api_headers.py` | `_prepare_base_headers`, `_prepare_api_headers`, `_add_origin_header`, `_parse_csrf_token`, `_add_csrf_token_header`, `_add_user_id_header` | ~250 |
| `core/text_utils.py` | `ordinal_case`, `format_name`, `parse_cookie`, `fast_json_loads` | ~200 |
| `core/os_utils.py` | `prevent_system_sleep`, `restore_system_sleep` | ~100 |

Keep `retry()`, `time_wait()`, `get_rate_limiter()`, `nav_to_page()` in `utils.py` as they're widely imported.

### 2B. `error_handling.py` — Two Systems in One File (HIGH)

2318 lines containing:
- `CircuitBreaker` + `CircuitState` + `CircuitBreakerConfig` — belongs in `circuit_breaker.py`
- `EnhancedErrorRecovery` + `RecoveryStrategy` + `RecoveryContext` — recovery framework
- TWO exception hierarchies (see 1C above)
- `ErrorHandler` (ABC) + `DatabaseErrorHandler`, `NetworkErrorHandler`, `BrowserErrorHandler` — handler pattern
- `ErrorHandlerRegistry` — handler registry
- `ErrorContext` — context manager for error tracking
- `ErrorRecoveryManager` — manages circuit breakers and strategies
- Recovery functions: `ancestry_session_recovery`, `ancestry_api_recovery`, `ancestry_database_recovery`

**Fix**: After resolving the duplicate exception hierarchy (1C), consider splitting:
1. Keep exceptions + retry config in `error_handling.py`
2. Move `CircuitBreaker` to `circuit_breaker.py` (merge with `SessionCircuitBreaker`)
3. Move `ErrorHandler*` + `ErrorHandlerRegistry` to `error_handlers.py`
4. Move `ErrorRecoveryManager` + recovery functions to `error_recovery.py`

### 2C. Session-Related Files — 6 Files for One Concept (MEDIUM)

| File | Lines | Purpose |
|------|-------|---------|
| `session_manager.py` | 3128 | Main SessionManager class |
| `session_mixins.py` | 651 | `SessionHealthMixin`, `SessionIdentifierMixin` |
| `session_validator.py` | 965 | Session validation logic |
| `session_utils.py` | 668 | Global session helpers, `_AuthCache` |
| `session_guards.py` | 259 | Navigation guards, decorators |
| `session_cache.py` | 816 | Session-specific caching |

Total: ~6487 lines across 6 files for session management. The mixins pattern (`SessionHealthMixin`, `SessionIdentifierMixin`) is appropriate, but `session_guards.py` (259 lines, 6 functions) could be folded into `session_validator.py`. `session_utils.py` contains global singleton management (`get_session_manager()`, `set_session_manager()`) that could live in `session_manager.py`.

### 2D. `database.py` — 4562 Lines (MEDIUM)

Largest file in the project. Contains ALL SQLAlchemy models, enums, utility functions, and tests. Should be split:
- `core/models/` package with one file per 2-3 related models
- Keep `database.py` as a re-export hub

### 2E. `metrics_collector.py` vs `metrics_integration.py` (LOW)

`metrics_collector.py` (635 lines): Full metrics framework with `MetricType`, `MetricPoint`, `WindowedStats`, `ServiceMetrics`, `MetricsSnapshot`, `MetricRegistry`.
`metrics_integration.py` (255 lines): `APICallMetricsContext` — a single context manager.

These could be merged, or `metrics_integration.py` renamed to make the relationship clear.

---

## 3. Excess Complexity

### 3A. `error_handling.py` — 39 Classes (CRITICAL)

39 classes in a single 2318-line file is excessive. The two parallel exception hierarchies (1C) mean ~20 exception classes that overlap in domain coverage. The file also contains circuit breaker, recovery manager, error handlers, error context, and recovery strategies — all separate concerns.

### 3B. Caching — 8 Files, 2 Protocols (HIGH)

The caching subsystem has grown organically into 8 files with 2 parallel Protocol definitions, 4 concrete cache implementations, 3 domain-specific cache wrappers, a registry, and a bootstrap module. A new developer cannot easily determine which cache to use.

### 3C. `action_registry.py` — Hardcoded Action Definitions (MEDIUM)

Lines 300-900+ contain inline definitions of every action with metadata. This is a ~600-line data table embedded in code. Consider moving to a declarative format (YAML/JSON) or using a decorator-based registration pattern where each action module registers itself.

### 3D. `dependency_injection.py` — Custom DI Framework (LOW)

861 lines implementing a full DI container with singleton/transient/factory patterns, `ServiceRegistry`, `Injectable`, `inject` decorator. This is a well-known wheel that libraries like `dependency-injector` or `python-inject` handle. The custom implementation works but adds maintenance burden. Not a critical issue — just worth noting.

### 3E. `schema_migrator.py` — Full Migration Framework (LOW)

1111 lines implementing database migrations from scratch. Standard tools like Alembic exist for this with SQLAlchemy. This is another custom wheel, though it may serve specific needs.

---

## 4. Test Quality

### Smoke Tests / Fake Tests (Need Improvement)

| File | Issue | Details |
|------|-------|---------|
| `__main__.py` (178 lines) | **Pure import checks** | All tests verify `import X works` — no behavioral validation |
| `common_params.py` (7 tests) | **Trivial dataclass assertions** | All tests just create a dataclass and assert fields match constructor args. Example: `ctx = RetryContext(attempt=1, max_attempts=3); assert ctx.attempt == 1` — this tests Python's `@dataclass`, not project code |
| `session_mixins.py` (1 test) | **`_test_module_integrity` — empty** | Single test: `return True`. 651 lines of mixin code with zero behavioral tests |
| `selenium_utils.py` (1 test) | **`_test_module_integrity` — empty** | Same pattern: `return True` |
| `type_definitions.py` (1 test) | **`_test_module_integrity` — empty** | Same pattern |
| `api_manager.py` (13 tests) | **Many shallow** | Several tests check `hasattr(mgr, 'request')`, `isinstance(mgr, APIManager)`, method existence rather than calling methods with test data |
| `browser_manager.py` (13 tests) | **Mixed** | Some good (cookie timeout, state management), some shallow (initialization checks, method availability) |
| `error_handling.py` | **`callable()` checks** | 3 tests just verify `callable(ancestry_session_recovery)` — verifies the function exists, not that it works |
| `error_handling.py` | **Signature inspection** | 3 more tests inspect function signatures with `inspect.signature()` — this tests Python's reflection, not recovery logic |
| `workflow_actions.py` (7 tests) | **Decorator existence** | 3 tests just check `hasattr(func, '__wrapped__')` — verifies a decorator was applied, not that the decorated function works correctly |
| `protocols.py` (5 tests) | **TypedDict creation** | Tests create dicts and assert values — tests Python's TypedDict mechanism, not project logic |

### Good Tests (Examples to Follow)

| File | Tests | Why Good |
|------|-------|----------|
| `action_registry.py` (14 tests) | Register/retrieve, duplicate prevention, category grouping, singleton | Tests real behavior with assertions on outcomes |
| `cancellation.py` (6 tests) | Request/clear/check with threading | Tests concurrency behavior |
| `correlation.py` (7 tests) | Nesting, metadata, elapsed time, filter | Tests state management across contexts |
| `config_validation.py` (7 tests) | Mock-based with realistic inputs | Tests validation logic against boundary cases |
| `circuit_breaker.py` | State transitions, threshold behavior, recovery | Tests the state machine |
| `pii_redaction.py` (8 tests) | Email, UUID, phone, name redaction | Tests regex patterns against various input |
| `opt_out_detection.py` | Indicator detection, analysis | Tests domain logic |
| `validation_factory.py` (6 tests) | Composite validators, required fields, type checks | Tests real validation behavior |
| `schema_migrator.py` (10 tests) | Ordering, rollback, dependency validation | Tests migration state machine |

### Missing Test Coverage

| File | Lines | Tests | Gap |
|------|-------|-------|-----|
| `session_manager.py` | 3128 | Has `module_tests()` | Session lifecycle (start→validate→refresh→close) not tested end-to-end |
| `session_mixins.py` | 651 | 1 (empty) | Zero coverage of `SessionHealthMixin` (health monitoring, heartbeat, death cascade) or `SessionIdentifierMixin` (profile ID/CSRF extraction) |
| `utils.py` | 4082 | 9 tests | Login flow (`log_in`, `login_status`, `handle_two_fa`) completely untested in module tests |
| `database.py` | 4562 | Unknown | Needs review — 4500 lines with complex ORM models |
| `unified_cache_manager.py` | 1202 | Unknown | Large caching orchestrator |
| `lifecycle.py` | 799 | 5 tests | Tests are structural/startup checks |
| `logging_config.py` | 810 | None visible | No `module_tests` function found |
| `metrics_integration.py` | 255 | None visible | No `module_tests` function found |

---

## 5. Linting / Type Issues

### 5A. `utils.py` — `# End of` Comments (LOW)

Extensive use of `# End of if`, `# End of for`, `# End of try/except`, `# End of wrapper`, `# End of decorator`, `# End of retry` etc. These are noise — Python's indentation makes block boundaries clear. ~50+ such comments throughout the file.

### 5B. Conditional `SessionManager` Import in `utils.py` (MEDIUM)

```python
if TYPE_CHECKING:
    from core.session_manager import SessionManager
else:
    SessionManager = None
```

Setting `SessionManager = None` at module level means runtime `isinstance(x, SessionManager)` checks would fail. This pattern works for annotations but is fragile. Consider using string annotations (`"SessionManager"`) consistently or restructuring to eliminate the circular import.

### 5C. `sys.path` Manipulation (MEDIUM)

Multiple files manipulate `sys.path` at module level:
- `cache/adapters.py:27`: `sys.path.insert(0, parent_dir)`
- `protocols.py:497`: `sys.path.insert(0, _parent_dir)`

This is fragile and can mask import issues. Prefer proper package installation (`pip install -e .`) or `__init__.py` configuration.

### 5D. `protocols.py` — Duplicate Path Insertion

Lines 496-499 duplicate `sys.path` insertion that's already done at the top of many other modules. Also, `_parent_dir` is computed but potentially already in path.

### 5E. `type_definitions.py` — Constants Mixed with Types

`type_definitions.py` contains `STATUS_SUCCESS`, `STATUS_ERROR`, `GENDER_MALE`, `CLASSIFICATION_PRODUCTIVE`, `TASK_CATEGORIES` — these are runtime constants, not type definitions. They should live in a `constants.py` file.

### 5F. `action_runner.py` — Inline Protocol Definitions

Three Protocol classes (`DatabaseManagerProtocol`, `BrowserManagerProtocol`, `APIManagerProtocol`) are defined inline rather than in `protocols.py`. These should be moved to `protocols.py` for discoverability.

### 5G. `error_handling.py` — 39 Classes with `Optional` Typing

Some error classes use `**kwargs: Any` pass-through pattern:
```python
class APIRateLimitError(RetryableError):
    def __init__(self, message: str = "API rate limit exceeded", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = kwargs.get('retry_after', 60)
```
This loses type safety — `retry_after` is typed `Any` at runtime. Better to use explicit keyword arguments with proper types.

### 5H. `circuit_breaker.py` — String-Based State vs Enum

`CircuitBreakerState` uses class-level string constants instead of `Enum`:
```python
class CircuitBreakerState:
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"
```
While `error_handling.py:CircuitState` uses proper `Enum`. These should be consistent.

---

## 6. File-by-File Summary

| File | Lines | Issues | Priority |
|------|-------|--------|----------|
| `utils.py` | 4082 | Monolith, duplicate RequestConfig, end-of comments | CRITICAL |
| `error_handling.py` | 2318 | 39 classes, 2 exception trees, duplicate CB, kwargs typing | CRITICAL |
| `database.py` | 4562 | Monolith | HIGH |
| `session_manager.py` | 3128 | Large but well-structured via mixins | MEDIUM |
| `api_manager.py` | 1392 | Duplicate RequestConfig, legacy+new request paths, shallow tests | HIGH |
| `common_params.py` | 519 | Duplicate RequestConfig/SearchCriteria/RetryContext, trivial tests | HIGH |
| `cache_backend.py` | 623 | Overlaps with cache/interface.py | HIGH |
| `cache/interface.py` | 517 | Overlaps with cache_backend.py | HIGH |
| `unified_cache_manager.py` | 1202 | Part of cache sprawl | MEDIUM |
| `session_cache.py` | 816 | Part of cache sprawl | MEDIUM |
| `system_cache.py` | 985 | Part of cache sprawl | MEDIUM |
| `circuit_breaker.py` | 792 | Overlaps error_handling.py CB | HIGH |
| `protocols.py` | 503 | Duplicate APIResponse, CacheProtocol, sys.path hack | MEDIUM |
| `type_definitions.py` | 244 | Constants mixed with types, duplicate APIResponse/SearchCriteria | MEDIUM |
| `session_mixins.py` | 651 | Zero behavioral tests | MEDIUM |
| `action_runner.py` | 653 | Inline Protocols should move to protocols.py | LOW |
| `action_registry.py` | 951 | Hardcoded action data; good tests | LOW |
| `rate_limiter.py` | 2144 | Large but focused; has tests | LOW |
| `dependency_injection.py` | 860 | Custom DI — works but high maintenance | LOW |
| `schema_migrator.py` | 1111 | Custom migrations — works but Alembic exists; good tests | LOW |
| `browser_manager.py` | 685 | Mixed test quality | LOW |
| `logging_config.py` | 810 | No module_tests | MEDIUM |
| `metrics_integration.py` | 255 | No module_tests | MEDIUM |
| `validation_factory.py` | 605 | Good tests | OK |
| `cancellation.py` | 207 | Clean, good tests | OK |
| `correlation.py` | 400 | Clean, good tests | OK |
| `config_validation.py` | 317 | Good tests | OK |
| `pii_redaction.py` | 513 | Good tests | OK |
| `app_mode_policy.py` | 230 | Good tests | OK |

---

## Recommended Action Plan

### Phase 1 — Eliminate Duplication (1-2 weeks)
1. Consolidate `RequestConfig` into one definition in `common_params.py`
2. Pick one exception hierarchy in `error_handling.py`, deprecate the other
3. Merge `CircuitBreaker` from `error_handling.py` into `circuit_breaker.py`
4. Consolidate `APIResponse`, `SearchCriteria` duplicates
5. Pick one cache Protocol, remove the other

### Phase 2 — Decompose Monoliths (2-3 weeks)
1. Split `utils.py` into `auth.py`, `api_headers.py`, `text_utils.py`, `os_utils.py`
2. Split `error_handling.py` into focused modules
3. Consider splitting `database.py` into a models subpackage

### Phase 3 — Improve Test Quality (Ongoing)
1. Replace all `_test_module_integrity` → `return True` stubs with real tests
2. Replace trivial dataclass assertion tests in `common_params.py` with tests of functions that USE those dataclasses
3. Add behavioral tests for `session_mixins.py` (health monitoring, identifier extraction)
4. Add `module_tests()` to `logging_config.py` and `metrics_integration.py`
5. Convert `callable()` checks in `error_handling.py` to actual recovery-logic tests
