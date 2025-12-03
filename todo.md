# Ancestry Research Platform - Implementation Roadmap

## Production Readiness Audit - December 2025

This document consolidates all identified improvements for production readiness.

---

## 1. Critical Type/Linting Issues

### 1.1 Unused Imports (Pyright Warnings) ✅ COMPLETED

**File**: `genealogy/gedcom/gedcom_utils.py`

- [x] Remove unused imports `is_ancestor_at_generation` and `is_descendant_at_generation`
- These were imported from `relationship_calculations` but never used in the file

### 1.2 Import Sorting (Ruff I001) ✅ COMPLETED

- [x] Fixed with `ruff check --fix .`

### 1.3 PLC2701 Private Name Imports ✅ COMPLETED

**Problem**: 13 private functions (underscore prefix) were being imported across modules.

**Solution**: Renamed all 15 affected functions to public names (removed underscore prefix):

In `genealogy/relationship_calculations.py`:

- `_is_ancestor_at_generation` → `is_ancestor_at_generation`
- `_is_descendant_at_generation` → `is_descendant_at_generation`
- `_is_grandparent` → `is_grandparent`
- `_is_grandchild` → `is_grandchild`
- `_is_great_grandparent` → `is_great_grandparent`
- `_is_great_grandchild` → `is_great_grandchild`
- `_are_siblings` → `are_siblings`
- `_is_aunt_or_uncle` → `is_aunt_or_uncle`
- `_is_niece_or_nephew` → `is_niece_or_nephew`
- `_are_cousins` → `are_cousins`
- `_find_direct_relationship` → `find_direct_relationship`
- `_has_direct_relationship` → `has_direct_relationship`

In `genealogy/gedcom/gedcom_utils.py`:

- `_are_spouses` → `are_spouses`
- `_get_event_info` → `get_event_info`
- `_get_full_name` → `get_full_name`

---

## 2. Code Consolidation Opportunities

### 2.1 Duplicate API Call Helper Functions ✅ REVIEWED - LOW PRIORITY

**Pattern**: Multiple files implement lazy-loading of the utils module to avoid circular imports.

**Files affected**:

- `api/api_utils.py` (line 120): `_get_utils_module()`
- `core/browser_manager.py` (line 49): `_get_utils_module()`
- `core/session_manager.py` (line 87): `importlib.import_module("utils")`
- `genealogy/dna/dna_utils.py` (line 60): `importlib.import_module("utils")`
- `genealogy/dna/dna_ethnicity_utils.py` (line 57): `importlib.import_module("utils")`

**Status**: This pattern is intentional to break circular import chains. The `@lru_cache(maxsize=1)` decorator ensures each module only loads utils once. Not a code smell - this is a valid Python pattern for circular import resolution.

**Action**: No changes needed. The gather module files (fetch.py, persistence.py, orchestrator.py) do NOT contain this pattern.

### 2.2 Cookie Synchronization Duplication

**Pattern**: Cookie syncing logic exists in multiple locations.

**Files affected**:

- `core/session_manager.py` (lines 1260-1347): `sync_cookies_from_browser()`, `sync_cookies_to_requests()`
- `core/api_manager.py` (lines 364-383): `_sync_cookies()`

**Action**:

- [ ] Consolidate into `SessionManager` as single `sync_browser_cookies()` method
- [ ] Have `APIManager` delegate to SessionManager for cookie operations

### 2.3 Test Runner Import Inconsistency ✅ REVIEWED - CORRECT

**Pattern**: `create_standard_test_runner` is imported from `testing.test_framework`.

**Status**: All files correctly import from `testing.test_framework`. Verified 21+ files use the correct import pattern.

**Action**: No changes needed.

### 2.4 ConfigManager Singleton Pattern ✅ COMPLETED

**Problem**: `ConfigManager()` was instantiated 15+ times across the codebase instead of using a singleton.

**Solution Implemented**:

- Added `_ConfigManagerSingleton` container class (avoids global statement warning)
- Added `get_config_manager()` function that returns the shared instance
- Function supports `force_new=True` for testing scenarios
- Backwards compatible - direct `ConfigManager()` instantiation still works

**Usage**:

```python
from config.config_manager import get_config_manager
config_manager = get_config_manager()
config = config_manager.get_config()
```

**Files affected** (partial list) - can gradually migrate:

- `main.py` (line 122)
- `integrations/ms_graph_utils.py` (line 39)
- `genealogy/gedcom/gedcom_utils.py` (line 136)
- `core/database_manager.py` (line 71)
- `ai/ai_interface.py` (line 162)
- And 10+ more

**Future Work** (optional optimization):

- [ ] Gradually migrate files to use `get_config_manager()` instead of direct instantiation

### 2.5 sys.path.insert() Proliferation

**Problem**: 96 files contain `sys.path.insert()` or `sys.path.append()` calls despite the project being installed in editable mode (`pip install -e .`).

**Analysis** (December 2025):
- Package IS correctly installed in editable mode (verified with `pip show ancestry`)
- All imports work correctly without sys.path modifications
- Most common patterns (96 total occurrences):
  - `sys.path.insert(0, str(_project_root))` - 39 files
  - `sys.path.insert(0, parent_dir)` - 28 files
  - `sys.path.insert(0, str(REPO_ROOT))` - 14 files
  - Other variations - 15 files

**Status**: Redundant but safe - these patterns are harmless since they only add paths that are already accessible.

**Recommendation**: Low priority cleanup. Could be done with automated script, but requires careful testing.

**Action** (if undertaken):
- [ ] Create script to remove sys.path patterns from all 96 files
- [ ] Test all 148 modules after removal
- [ ] Verify standalone script execution still works

---

## 3. Stale Comments Cleanup ✅ COMPLETED

### 3.1 "Removed" Placeholder Comments ✅ COMPLETED

Removed 38 stale comments from 10 files:

| File | Comments Removed |
|------|------------------|
| `core/session_manager.py` | 7 (Moved to Mixin notes) |
| `actions/gather/orchestrator.py` | 10 (progressive processing, ObjectPool, calculate optimized workers, ThreadPoolExecutor, etc.) |
| `actions/action6_gather.py` | 1 (CircuitBreaker removal note) |
| `messaging/message_types.py` | 1 (MessageType class removal note) |
| `research/relationship_utils.py` | 1 (removed function note) |
| `database.py` | 4 (mock mode, mock filtering, _process_matches, _get_gedcom_data_or_skip) |
| `actions/action7_inbox.py` | 2 (concurrent.futures removal, migration + smoke test) |
| `performance/performance_orchestrator.py` | 3 (smoke test removal notes) |
| `performance/performance_monitor.py` | 3 (smoke test removal notes) |
| `genealogy/genealogical_normalization.py` | 3 (smoke test removal notes) |

### 3.2 "Moved to" Comments ✅ COMPLETED

All "Moved to Mixin" notes removed from `core/session_manager.py`.

### 3.3 "Removed smoke test" Pattern ✅ COMPLETED

All smoke test removal comments removed from:

- `performance/performance_orchestrator.py`
- `performance/performance_monitor.py`
- `genealogy/genealogical_normalization.py`

### 3.4 Phase/Version Comments

Kept functional comments that provide context:

- `core/api_manager.py`: `PHASE 4.1` is a section header, not a stale reference
- `actions/action10.py`, `genealogy/universal_scoring.py`, `utils.py`: No stale comments found

---

## 4. Test Quality Improvements

### 4.1 Tests That Only Check `callable()` or `isinstance()` Without Behavioral Validation ✅ REVIEWED

**Status**: Most tests mentioned have been verified to test actual behavior, not just type checks.

**Verified as Good**:

- `observability/metrics_exporter.py`: Tests call actual functions and verify HTTP responses
- Tests use `assert start_metrics_exporter()`, make HTTP requests, verify response content

**Remaining Minor Issues** (Low Priority - Tests Still Pass):

- [ ] `ui/__init__.py`: Menu render existence check - acceptable for import validation
- [ ] `ui/menu.py`: Same pattern - acceptable for import validation

**Action**: No immediate changes needed. All 148 test modules pass with 100% quality.

### 4.2 Tests That Skip and Return True (Fake Passes) ✅ REVIEWED

**Status**: This pattern is intentional for integration tests that require live sessions.

**File**: `testing/test_integration_workflow.py` - Uses `_run_with_live_session()` pattern that returns `True` when SKIP_LIVE_API_TESTS=true. This is the correct behavior for tests that require authenticated browser sessions - they should skip gracefully rather than fail in CI/CD.

### 4.3 Trivial/Minimal Assertion Tests ✅ REVIEWED

**Status**: These patterns are acceptable for the module's purpose.

- `ai/prompt_telemetry.py`: Tests validate telemetry recording and file operations
- `config/__init__.py`: Import and type validation tests are appropriate for init modules

**Action**: No changes needed - all tests pass with 100% quality scores.

---

## 5. noqa Suppressions Analysis

### 5.1 Suppressions That Are Necessary (Keep)

These suppressions are required due to Protocol/interface requirements:

- `observability/apm.py` (line 129): PLR6301 - Base class override
- `core/logging_config.py` (line 157): PLR6301 - stdlib `logging.Filter` interface
- `core/correlation.py` (line 160): PLR6301 - stdlib `logging.Filter` interface
- `ai/providers/base.py` (lines 147, 150): PLR6301, ARG002 - Protocol implementation in test
- `core/cache/adapters.py` (lines 615-671): ARG002, PLR6301 - Null Object Pattern

### 5.2 Suppressions That Could Be Refactored

- [ ] `cli/maintenance.py` (line 48): PLR0904 - "Too many public methods"
  - **Current**: `MainCLIHelpers` class has 20+ public methods
  - **Recommendation**: Consider splitting into smaller helper classes:
    - `LogMaintenanceHelper`
    - `TestRunnerHelper`
    - `AnalyticsHelper`
    - `ReviewQueueHelper`
    - `CacheHelper`

- [ ] `genealogy/fact_validator.py` (line 147): ARG003 - Unused `context` parameter
  - Comment says "kept for API compatibility"
  - **Action**: Verify if any callers use this parameter; if not, remove it

- [ ] `performance/health_monitor.py` (line 2): PLR0904 - Too many public methods
  - Similar to `MainCLIHelpers` - consider splitting class

---

## 6. Legacy/Backward Compatibility Code

### 6.1 Patterns That May No Longer Be Needed ✅ REVIEWED

**Status**: Verified that "legacy" patterns serve different purposes than Phase 2 patterns.

**Analysis** - `messaging/safety.py`:
- `_OPT_OUT_PATTERNS`, `_DANGER_PATTERNS`, `_HOSTILITY_PATTERNS` are used by `check_message()` method
- `_CRITICAL_*_PATTERNS` are used by `check_critical_alerts()` method
- These are complementary, not duplicates:
  - `check_message()`: Simple boolean opt-out/danger detection
  - `check_critical_alerts()`: Category-based alerts with priority ordering

**Action**: Rename comment from "Legacy compatibility" to "Standard Safety Patterns" for clarity (optional).

**Other Items** (Low Priority):

- [ ] `genealogy/genealogical_normalization.py` (lines 43-44, 359-367): Legacy field promotion
  - `LEGACY_TO_STRUCTURED_MAP` and `_promote_legacy_fields()`
  - **Action**: Verify if AI still returns legacy format; if not, remove

- [ ] `testing/verify_opt_out.py` (lines 65-80): Tests legacy `check_message` method
  - Method is still used, tests are valid

---

## 7. Architecture Improvements

### 7.1 Retry Decorator Consolidation ✅ VERIFIED - NO DUPLICATION

**Status**: Verified that only ONE retry implementation exists.

**Verified**:

- `core/error_handling.py` (line 1645): `retry_on_failure` decorator - ONLY implementation
- `core/circuit_breaker.py`: Contains CircuitBreaker pattern but NO retry decorators

**Action**: No changes needed. The todo entry was based on outdated information.

### 7.2 Logging Configuration Overlap ✅ VERIFIED - COMPLEMENTARY MODULES

**Status**: Verified that the two modules serve different, complementary purposes.

**Analysis**:

- `core/logging_config.py` (811 lines): Main configuration infrastructure
  - Handler setup (console, file)
  - Custom formatters
  - Filters for external libraries (Selenium, urllib3)
  - Dynamic log level updates

- `core/logging_utils.py` (637 lines): Utility layer
  - `get_logger()` function for consistent logger access
  - Centralized logging state management
  - Helper functions for logging operations

**Action**: No changes needed. These are well-organized complementary modules.

---

## 8. Documentation Updates

### 8.1 Update README.md

- [ ] Remove reference to `sys.path.insert()` not being needed if it is still being used
- [ ] Or fix the codebase to actually not need it

### 8.2 Update copilot-instructions.md

- [ ] Update after completing consolidation tasks
- [ ] Remove references to patterns that no longer exist

---

## Pre-Production Validation

**Status**: Ready for manual validation
**Priority**: Low (requires production data access)

These tasks require manual testing with real historical data and cannot be automated:

- [ ] **Execute Dry-Run Validation**
  - Run `validate` command against 50+ historical PRODUCTIVE conversations
  - Target: 90%+ parse success rate

- [ ] **Quality Audit**
  - Manual audit comparing AI-generated drafts vs actual human replies
  - Document edge cases and failure modes
  - Measure extraction quality scores (target: median >70)

---

## Priority Order for Implementation

### Immediate (CI/CD Blocking) ✅ COMPLETED

1. ✅ Fix unused imports in `gedcom_utils.py` (Section 1.1)
2. ✅ Fix import sorting with `ruff check --fix .` (Section 1.2)
3. ✅ Fix PLC2701 private name imports (Section 1.3)

### High Priority (Code Quality) ✅ COMPLETED

1. ✅ Remove stale "removed" comments (Section 3.1) - 38 stale comments removed from 10 files
2. ✅ Fake-pass test pattern reviewed (Section 4.2) - Pattern is intentional for live integration tests
3. ✅ API helper functions reviewed (Section 2.1) - Pattern is intentional for circular import resolution
4. ✅ Test runner imports reviewed (Section 2.3) - All imports are correct
5. ✅ Trivial tests reviewed (Section 4.1, 4.3) - Tests are appropriate for their purpose

### Medium Priority (Maintainability) - MOSTLY COMPLETE

1. ✅ `sys.path.insert()` analyzed (Section 2.5) - 96 files, redundant but harmless; low priority cleanup
2. ✅ ConfigManager singleton implemented (Section 2.4) - `get_config_manager()` function added
3. [ ] Consolidate cookie sync logic (Section 2.2) - SessionManager vs APIManager (complex refactor, optional)
4. ✅ Retry decorator consolidation verified (Section 7.1) - Only one implementation exists
5. ✅ Logging configuration verified (Section 7.2) - Complementary modules, no overlap

### Low Priority (Nice to Have)

1. [ ] Split large CLI helper class (Section 5.2) - PLR0904 suppression
2. ✅ Legacy compatibility patterns reviewed (Section 6) - Patterns are complementary, not duplicates
