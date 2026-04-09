# Implementation Roadmap - Ancestry Project

**Created:** 2026-04-09
**Status:** In Progress
**Scope:** Code quality, bug fixes, efficiency improvements, feature completion

---

## Priority Classification

- **P0 (Critical):** Bugs, broken functionality, security issues
- **P1 (High):** Code quality, maintainability, performance
- **P2 (Medium):** Completeness, documentation, best practices
- **P3 (Low):** Nice-to-have, refactoring, optimization

---

## Findings & Action Items

### P0: Critical Bugs & Broken Functionality

#### 1. Dockerfile references non-existent requirements-dev.txt ✅ FIXED
- **Location:** Dockerfile lines 17, 20
- **Issue:** `COPY requirements.txt requirements-dev.txt ./` fails because `requirements-dev.txt` doesn't exist
- **Impact:** Docker build will fail
- **Action:** Created `requirements-dev.txt` with development dependencies; updated Dockerfile to install only production dependencies
- **Status:** ✅ FIXED (2026-04-09)
- **Commit:** 4dc12e3
- **Additional Fix:** Fixed MCP server bug (SharedMatch.people_id -> person_id) discovered during testing

#### 2. Massive file sizes impede maintainability
- **Location:** Multiple files
- **Files affected:**
  - `ai/ai_interface.py`: ~4822 lines
  - `actions/action7_inbox.py`: ~4577 lines
  - `core/database.py`: ~4564 lines
  - `core/session_manager.py`: ~3135 lines
  - `run_all_tests.py`: ~2423 lines
  - `core/rate_limiter.py`: ~2145 lines
- **Issue:** Files exceed reasonable size for comprehension and maintenance
- **Impact:** High cognitive load, difficult to test, merge conflicts likely
- **Action:** Refactor into smaller, focused modules (target: <500 lines per file)
- **Status:** 🔴 TODO (incremental refactoring)

---

### P1: Code Quality & Maintainability

#### 3. Widespread star imports create namespace pollution
- **Location:** Multiple modules
- **Issue:** `from module import *` patterns make code harder to audit and understand
- **Impact:** Implicit dependencies, unclear symbol origins, harder refactoring
- **Action:** Replace with explicit imports; update `.ruff.toml` to enforce
- **Status:** 🟡 TODO

#### 4. Heavy reliance on global state
- **Location:** Multiple modules (`_action_registry`, `config_schema`, `_PROVIDER_ADAPTERS`, `_LIMITER_STATE`, etc.)
- **Issue:** Module-level globals and singleton patterns make testing difficult
- **Impact:** Hard to test in isolation, state management complexity
- **Action:** Introduce dependency injection, use explicit state passing
- **Status:** 🟡 TODO (long-term refactoring)

#### 5. Circular import dependencies
- **Location:** Throughout codebase
- **Issue:** Extensive use of `importlib.import_module()`, `TYPE_CHECKING` guards, deferred imports
- **Impact:** Fragile import order, runtime errors possible, unclear dependency graph
- **Action:** Restructure modules to eliminate circular dependencies
- **Status:** 🟡 TODO

#### 6. Pyright "strict" mode is effectively disabled
- **Location:** `pyrightconfig.json`
- **Issue:** Claims "strict" mode but disables most strict checks (`reportUnknownVariableType: "none"`, etc.)
- **Impact:** False sense of type safety, actual type checking is minimal
- **Action:** Either enable true strict mode incrementally, or change to "basic"/"off" to be honest
- **Status:** 🟡 TODO

#### 7. Duplicate code across modules
- **Location:** `ai/ai_interface.py`, `ai_api_test.py`, and others
- **Issue:** `_normalize_grok_*` functions and other utilities duplicated
- **Impact:** Maintenance burden, inconsistency risk
- **Action:** Extract to shared utility modules
- **Status:** 🟡 TODO

#### 8. Extensive error suppression
- **Location:** Throughout codebase
- **Issue:** Widespread `cast()`, `pragma: no cover`, silent exception swallowing
- **Impact:** Masks real issues, hard to debug production problems
- **Action:** Replace with proper error handling and logging
- **Status:** 🟡 TODO

---

### P2: Completeness & Best Practices

#### 9. No .env.example file
- **Location:** Project root
- **Issue:** README says "create .env manually" but no template provided
- **Impact:** Difficult onboarding, configuration errors
- **Action:** Create `.env.example` with all required/optional variables documented
- **Status:** 🟢 TODO

#### 10. Test coverage not actually measured
- **Location:** `pytest.ini`, `run_all_tests.py`
- **Issue:** Claims "100% test pass rate" but this is pass rate, not code coverage; pytest-cov installed but not configured
- **Impact:** Unknown actual code coverage, false confidence
- **Action:** Configure pytest-cov in pytest.ini, add coverage targets
- **Status:** 🟢 TODO

#### 11. Windows-specific assumptions limit portability
- **Location:** `main.py`, various modules
- **Issue:** `os.system("cls" if os.name == "nt" else "clear")`, pywin32, console focus manipulation
- **Impact:** Docker support undermined, cross-platform use limited
- **Action:** Abstract platform-specific operations, use cross-platform alternatives
- **Status:** 🟢 TODO

#### 12. Hardcoded magic numbers and strings
- **Location:** Throughout codebase
- **Issue:** Cookie names, API endpoints, token values scattered as literals
- **Impact:** Hard to maintain, inconsistency risk
- **Action:** Centralize into constants modules
- **Status:** 🟢 TODO

#### 13. Embedded testing pattern mixes tests with production code
- **Location:** 16+ modules contain `run_comprehensive_tests()` functions
- **Issue:** Tests embedded in production modules instead of separate test files
- **Impact:** Unclear separation, harder to run isolated test suites
- **Action:** Gradually migrate embedded tests to dedicated test files in `tests/`
- **Status:** 🟢 TODO (long-term)

---

### P3: Optimizations & Nice-to-Have

#### 14. Action runner could benefit from better argument validation
- **Location:** `core/action_runner.py`, `main.py`
- **Issue:** Basic validation, could be more robust
- **Action:** Add comprehensive validation layer
- **Status:** ⚪ TODO

#### 15. Logging could be more structured
- **Location:** Throughout codebase
- **Issue:** Mixed logging patterns, inconsistent formats
- **Action:** Standardize on structured logging (JSON for production)
- **Status:** ⚪ TODO

#### 16. Performance monitoring incomplete
- **Location:** `performance/` directory
- **Issue:** Health monitor exists but not all actions instrumented
- **Action:** Add performance hooks to all major actions
- **Status:** ⚪ TODO

---

## Implementation Plan

### Phase 1: Critical Fixes (P0)
1. ✅ **Fix Dockerfile** - Create requirements-dev.txt or fix Dockerfile
2. 🔄 **Begin file refactoring** - Start with largest files, extract logical modules

### Phase 2: Code Quality (P1)
3. Replace star imports with explicit imports
4. Address global state patterns
5. Resolve circular imports
6. Fix pyright configuration honesty
7. Deduplicate code
8. Improve error handling

### Phase 3: Completeness (P2)
9. Create .env.example
10. Configure proper test coverage measurement
11. Improve cross-platform support
12. Centralize constants
13. Migrate embedded tests

### Phase 4: Optimization (P3)
14. Enhance argument validation
15. Standardize logging
16. Complete performance monitoring

---

## Progress Tracking

**Completed:** 0/16
**In Progress:** 0/16
**Remaining:** 16/16

---

## Notes

- This analysis is based on comprehensive codebase review
- 231 Python files, ~106k lines of code (excluding dependencies)
- Complex, production-grade system with sophisticated architecture
- Many strengths: rate limiting, caching, observability, safety controls
- Primary concerns: file sizes, global state, duplicate code, configuration gaps

---

*Last updated: 2026-04-09*
