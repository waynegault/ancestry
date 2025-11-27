# Production Readiness: Remaining Action Items

This document contains actionable items for the Ancestry Research Automation Platform codebase.

## Priority Legend
- 🔴 **HIGH** - Should be addressed before production deployment
- 🟠 **MEDIUM** - Should be addressed for code quality, can be done post-launch
- 🟢 **LOW** - Nice-to-have improvements, can be deferred

---

## 1. Large File Opportunities

### 🟢 LOW: File Size Reduction

| File | Lines | Suggestion |
|------|-------|------------|
| `utils.py` | ~4700 | Consider extracting login/consent functions to `auth_utils.py` |
| `core/session_manager.py` | ~3600 | Consider extracting cookie management to separate module |
| `core/error_handling.py` | ~2029 | Split into: `exceptions.py`, `retry.py`, `circuit_breaker.py`, `decorators.py` |

Only split if maintainability becomes an issue.

---

## 2. Error Handling Improvements

### ✅ ~~MEDIUM: Inconsistent Error Path Testing~~

**Status:** COMPLETED

Error scenarios are now well-tested across critical modules including:
- 429 rate limiting responses (with retry logic)
- Browser crashes and session recovery
- Circuit breaker patterns
- API authentication refresh on 403 errors

---

## 3. Architecture Improvements

### ✅ ~~MEDIUM: Dependency Graph Cleanup and Import Ordering~~

**Status:** COMPLETED

**Implemented:**
1. ✅ Created `scripts/analyze_imports.py` - generates dependency graph, detects 17 circular import cycles
2. ✅ Created `core/type_definitions.py` - shared TypedDicts, Protocols, and type aliases
3. CI check can use: `python scripts/analyze_imports.py --baseline-cycles 17`

**Analysis Results:**
- 144 modules analyzed, 831 import edges
- 17 circular import cycles detected (mostly in test utilities)
- Top imported modules: `typing` (107), `logging` (96), `database` (48)

### 🟢 LOW: Extract Business Logic from UI/Orchestration

**Problem:** Action files mix orchestration, business logic, API calls, and UI concerns.

**Suggested Approach:**
Create `actions/gather/` package structure:
- `coordinator.py` - high-level orchestration
- `api_client.py` - API interactions
- `data_processor.py` - business logic
- `persistence.py` - database operations

### 🟢 LOW: Unified Cache Interface with Clear Invalidation

**Problem:** Multiple cache systems with inconsistent interfaces.

**Suggested Approach:**
1. Create `core/cache/interface.py` with `Cache` protocol
2. Implement adapters: `DiskCache`, `MemoryCache`, `DatabaseCache`
3. Add explicit TTL and cache versioning

### 🟢 LOW: Browser Automation Service Layer

**Problem:** Selenium code mixed with business logic.

**Suggested Approach:**
1. Create `services/browser_service.py` with high-level operations
2. Implement page object pattern
3. Create mock browser service for testing

### 🟢 LOW: Runtime Feature Toggle Framework

**Problem:** Features hardcoded on/off in code.

**Suggested Approach:**
Create `core/feature_flags.py` with `FeatureFlags` class for A/B testing and gradual rollout.

### 🟢 LOW: Database Schema Evolution System

**Problem:** Current `schema_migrator.py` is basic; no rollback capability.

**Suggested Approach:**
Enhance migration system with forward/backward migrations, dependencies, and dry-run mode.

---

## 4. Observability & Monitoring

### 🟢 LOW: Production Performance Monitoring

**Problem:** No visibility into production performance.

**Suggested Approach:**
Integrate APM (opentelemetry or sentry) with automatic span tracking.

---

## 5. Testing Strategy

### ✅ ~~MEDIUM: Multi-Layer Test Strategy~~

**Status:** IMPLEMENTED (pytest markers)

**Implementation:**
- Added pytest markers in `pytest.ini`: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`, `@pytest.mark.slow`
- Tests remain embedded in source files (project convention)
- Run by type: `pytest -m unit` for fast unit tests only

**Usage:**
```bash
# Run only fast unit tests
pytest -m unit

# Run integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

**Note:** Most existing tests are integration tests by nature (touching DB, files). Marker assignment is gradual - start with new tests.

### ✅ ~~MEDIUM: Inconsistent Test Framework Usage~~

**Status:** COMPLETED

**File:** `core/dependency_injection.py` - Already uses a hybrid approach with `dependency_injection_module_tests()` wrapping unittest.TestCase tests using TestSuite for standardized output.

This pattern (unittest.TestCase wrapped by TestSuite) is acceptable for modules that benefit from unittest's setup/teardown fixtures while maintaining consistent test output.

### ✅ ~~MEDIUM: Tests Using Mocking Without Real Assertions~~

**Status:** REVIEWED & ACCEPTABLE

**Analysis:** Upon review, the tests in these modules DO have proper assertions:
- `core/database_manager.py` - Tests use mocks for stubbing but assert real behavior (connection counts, session lifecycle)
- `core/config_validation.py` - Tests assert formatted values and function call behavior, not just mock.called
- `core/analytics_helpers.py` - Tests verify correct function resolution from dynamic imports with identity assertions

Mocking module imports is appropriate for testing dynamic import logic.

### ✅ ~~LOW: Missing Tests for `ai/prompts.py`~~

**Status:** COMPLETED

Added 8 comprehensive tests for prompt loading/validation:
- `supports_json_prompts` returns bool
- `load_prompts` returns dict structure
- `get_prompt` handles missing keys gracefully
- `get_prompt_with_experiment` fallback behavior
- `get_prompt_version` handles missing keys
- `record_extraction_experiment_event` no-raise guarantee
- Valid prompts structure validation
- String return for valid keys

---

## 6. Future Enhancements

### ⚠️ LOW (v2.0): Async/Await for I/O Operations

**Problem:** Sequential API calls waste time; Action 6 takes hours for large datasets.

**Suggested Approach:**
Migrate to aiohttp and implement async database operations.

**Note:** Large change, consider for v2.0 after current architecture stabilizes.

---

## Summary

| Category | Count | Priority Breakdown |
|----------|-------|-------------------|
| Large Files | 1 item | 1 LOW |
| Error Handling | ~~1 item~~ | ✅ COMPLETED |
| Architecture | 6 items | ~~1 MEDIUM~~, 5 LOW (1 COMPLETED) |
| Observability | 1 item | 1 LOW |
| Testing Strategy | 4 items | ~~3 MEDIUM~~, ~~1 LOW~~ (4 COMPLETED) |
| Future | 1 item | v2.0 |

**Total Remaining Items:** 8 actionable items (6 completed this session)
**Critical Issues:** 0 (All HIGH priority items completed!)
