# Production Readiness: Remaining Action Items

This document contains actionable items for the Ancestry Research Automation Platform codebase.

## Priority Legend
- 🔴 **HIGH** - Should be addressed before production deployment
- 🟠 **MEDIUM** - Should be addressed for code quality, can be done post-launch
- 🟢 **LOW** - Nice-to-have improvements, can be deferred

---

## 1. Test Quality Improvements

### 🟠 MEDIUM: Remaining Tests to Review

| File | Issue |
|------|-------|
| `config/config_schema.py` | Some config tests only check existence |
| `database.py` | Some tests swallow exceptions with `pass` |

### 🟠 MEDIUM: Missing Behavior Verification

| File | Lines | Issue |
|------|-------|-------|
| `action8_messaging.py` | 4080-4090 | Performance tracking attributes created but usage not verified |
| `action9_process_productive.py` | 3831-3835 | Retry helper behavior not tested |
| `action10.py` | 2019-2053 | Config effects on behavior not tested |

---

## 2. Large File Opportunities

### 🟢 LOW: File Size Reduction

| File | Lines | Suggestion |
|------|-------|------------|
| `utils.py` | ~4700 | Consider extracting login/consent functions to `auth_utils.py` |
| `core/session_manager.py` | ~3600 | Consider extracting cookie management to separate module |
| `core/error_handling.py` | ~2029 | Split into: `exceptions.py`, `retry.py`, `circuit_breaker.py`, `decorators.py` |

Only split if maintainability becomes an issue.

---

## 3. Error Handling Improvements

### 🟠 MEDIUM: Inconsistent Error Path Testing

Happy paths are well-tested, but error scenarios (429 responses, browser crashes, session recovery) have limited coverage.

**Suggested Approach:**
Add error scenario tests to critical modules:
```python
def _test_429_handling():
    mock_response = Mock()
    mock_response.status_code = 429
    mock_response.headers = {"Retry-After": "60"}
    with patch('requests.get', return_value=mock_response):
        result = make_api_call()
    assert result is None
```

---

## 4. Architecture Improvements

### 🟠 MEDIUM: Dependency Graph Cleanup and Import Ordering

**Problem:** Circular imports cause fragility and initialization issues.

**Suggested Approach:**
1. Create `scripts/analyze_imports.py` to generate dependency graph
2. Move shared types/protocols to `core/types.py`
3. Add CI check that fails on new circular imports

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

## 5. Observability & Monitoring

### 🟢 LOW: Production Performance Monitoring

**Problem:** No visibility into production performance.

**Suggested Approach:**
Integrate APM (opentelemetry or sentry) with automatic span tracking.

---

## 6. Testing Strategy

### 🟠 MEDIUM: Multi-Layer Test Strategy

**Problem:** No clear distinction between unit (fast, isolated) and integration (slow) tests.

**Suggested Approach:**
1. Create `tests/` directory structure: `tests/unit/`, `tests/integration/`, `tests/e2e/`
2. Add pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`
3. Run times: unit <1s each, integration <30s each

### 🟠 MEDIUM: Inconsistent Test Framework Usage

**File:** `core/dependency_injection.py`

**Issue:** Uses `unittest.TestCase` instead of the project's `TestSuite` pattern.

**Suggested Approach:**
Convert to use `TestSuite` pattern OR document that `unittest.TestCase` is acceptable for pure unit tests.

### 🟠 MEDIUM: Tests Using Mocking Without Real Assertions

**Problem:** Some tests mock all dependencies and only verify the mock was called.

**Affected Files:**
- `core/database_manager.py`
- `core/config_validation.py`
- `core/analytics_helpers.py`

### 🟢 LOW: Missing Tests for `ai/prompts.py`

Add tests for prompt loading/validation.

---

## 7. Future Enhancements

### ⚠️ LOW (v2.0): Async/Await for I/O Operations

**Problem:** Sequential API calls waste time; Action 6 takes hours for large datasets.

**Suggested Approach:**
Migrate to aiohttp and implement async database operations.

**Note:** Large change, consider for v2.0 after current architecture stabilizes.

---

## Summary

| Category | Count | Priority Breakdown |
|----------|-------|-------------------|
| Test Quality | 2 items | 2 MEDIUM |
| Large Files | 1 item | 1 LOW |
| Error Handling | 1 item | 1 MEDIUM |
| Architecture | 6 items | 1 MEDIUM, 5 LOW |
| Observability | 1 item | 1 LOW |
| Testing Strategy | 4 items | 3 MEDIUM, 1 LOW |
| Future | 1 item | v2.0 |

**Total Remaining Items:** 16 actionable items
**Critical Issues:** 0 (All HIGH priority items completed!)
