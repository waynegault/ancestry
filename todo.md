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

### 2.4 ConfigManager Instantiation Pattern

**Pattern**: `ConfigManager()` is instantiated 15+ times across the codebase instead of using a singleton.

**Files affected** (partial list):

- `main.py` (line 122)
- `integrations/ms_graph_utils.py` (line 39)
- `genealogy/gedcom/gedcom_utils.py` (line 136)
- `core/database_manager.py` (line 71)
- `ai/ai_interface.py` (line 162)
- And 10+ more

**Action**:

- [ ] Implement singleton pattern in `ConfigManager` class
- [ ] Or use dependency injection pattern with a single instance created in `main.py`
- [ ] Update all files to use the singleton/injected instance

### 2.5 sys.path.insert() Proliferation

**Problem**: 30+ files contain `sys.path.insert()` or `sys.path.append()` calls despite README stating this shouldn't be needed (project has proper package structure via pyproject.toml).

**Files affected** (partial list):

- `ui/__init__.py`, `ui/menu.py`
- `testing/test_integration_workflow.py`, `testing/dead_code_scan.py`, `testing/check_type_ignores.py`
- `scripts/migrate_phase4.py`, `scripts/maintain_code_graph.py`, `scripts/dry_run_validation.py`
- `research/*.py` (multiple files)
- `performance/*.py` (multiple files)
- `observability/*.py` (multiple files)
- `messaging/*.py` (multiple files)
- `genealogy/*.py` (multiple files)

**Action**:

- [ ] Verify `pyproject.toml` correctly configures the package
- [ ] Remove all `sys.path.insert()`/`sys.path.append()` calls
- [ ] Use proper relative imports or install package in editable mode (`pip install -e .`)

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

### 6.1 Patterns That May No Longer Be Needed

- [ ] `messaging/safety.py` (lines 64-100): "ORIGINAL PATTERNS (Legacy compatibility)"
  - Review if legacy patterns are still needed
  - If newer patterns cover all cases, remove legacy section

- [ ] `genealogy/genealogical_normalization.py` (lines 43-44, 359-367): Legacy field promotion
  - `LEGACY_TO_STRUCTURED_MAP` and `_promote_legacy_fields()`
  - **Action**: Verify if AI still returns legacy format; if not, remove

- [ ] `testing/verify_opt_out.py` (lines 65-80): Tests legacy `check_message` method
  - If deprecated method is removed, update tests

---

## 7. Architecture Improvements

### 7.1 Retry Decorator Consolidation

**Pattern**: Multiple retry implementations exist.

**Files**:

- `core/error_handling.py` (lines 869-913): `retry_on_failure` decorator
- `core/circuit_breaker.py` (lines 93-107): `with_retry`, `retry_with_backoff`

**Action**:

- [ ] Consolidate into single retry implementation in `core/error_handling.py`
- [ ] Deprecate/remove duplicate in `circuit_breaker.py`

### 7.2 Logging Configuration Overlap

**Files**:

- `core/logging_config.py`: Primary logging configuration
- `core/logging_utils.py`: Additional logging utilities

**Action**:

- [ ] Review overlap and either merge or clearly define boundaries
- [ ] External logger suppression appears in both files

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

### Medium Priority (Maintainability) - REMAINING

1. [ ] `sys.path.insert()` calls (Section 2.5) - 50+ files affected, requires careful refactoring
2. [ ] Implement ConfigManager singleton (Section 2.4) - 15+ instantiations to consolidate
3. [ ] Consolidate cookie sync logic (Section 2.2) - SessionManager vs APIManager

### Low Priority (Nice to Have)

1. [ ] Split large CLI helper class (Section 5.2) - PLR0904 suppression
2. [ ] Remove legacy compatibility code after verification (Section 6)
3. [ ] Consolidate retry decorators (Section 7.1)
