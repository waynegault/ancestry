# Codebase Review Master Todo *(Updated 2025-11-17)*

All open work is captured in the single checklist below. Address items in priority order unless a dependency is noted.

## Open Items (Identified 2025-11-17 Audit)

- [x] **Main Menu Session Guard Helper**
  `main.require_interactive_session` now wraps Actions 7–9 and funnels all readiness checks through `_ensure_interactive_session_ready()`, so future login/session policy tweaks only touch one decorator.
- [x] **Action 6 Start Page Alignment**
  ✅ `main.gather_dna_matches` and the Action 6 orchestration pipeline now treat `None` as “resume from checkpoint,” with menu parsing and coord/start-page validation wired into the new checkpoint helpers.
- [x] **Main.py Pylance Issues**
  Added typed helper loaders for GEDCOM/API display helpers, analytics logging, and the Windows console-focus function so `main.py` no longer needs `pyright` suppressions; `npx pyright` now reports 0 issues.
- [x] **Action 6 Regression Tests**
  Converted the timeout/selenium policy, duplicate handling, and final summary regression tests to use real assertions plus new IntegrityError coverage, so `_test_error_handling()` now fails fast if the critical safeguards regress.
- [x] **Unified Error-Handling Stack**
  Merged the enhanced recovery decorators into `core/error_handling.py`, removed `core/enhanced_error_recovery.py`, and updated Action 6/7/8 plus archival scripts to import from the unified module.

## High Priority Technical Debt

- [ ] **Cache Module Consolidation** (High Priority)
  Consolidate 7 cache-related modules into unified architecture
  - Target: Single cache manager with specialized implementations
  - Estimated Impact: 30% reduction in cache-related code
  - Files: cache.py, cache_manager.py, gedcom_cache.py, performance_cache.py, core/session_cache.py, core/system_cache.py, core/unified_cache_manager.py

- [ ] **Error Handling Deduplication** (High Priority)
  Remove duplicate error handling between error_handling.py and core/error_handling.py
  - Keep: core/error_handling.py (more comprehensive)
  - Migrate and remove: error_handling.py
  - Update all imports across codebase

- [ ] **Database Module Consolidation** (Medium Priority)
  Complete migration from database.py to core/database_manager.py
  - Verify all functionality moved to core module
  - Update remaining references
  - Archive or remove legacy database.py

- [ ] **Function Decomposition** (Medium Priority)
  Break down large functions with multiple responsibilities:
  - main.exec_actn (220+ lines)
  - main.run_gedcom_then_api_fallback (large with inline helpers)
  - Extract common guard patterns from Actions 7/8/9

- [ ] **Test Infrastructure Modernization** (Medium Priority)
  Migrate inline tests to dedicated test directory structure
  - Create tests/ directory mirroring source structure
  - Maintain test framework but separate test code from production
  - Improve test discoverability and IDE integration

- [ ] **Linting Configuration Hardening** (Low Priority, High Impact)
  Gradually enable stricter linting and type checking
  - Phase 1: Enable reportReturnType, reportUnusedVariable, reportDuplicateImport
  - Phase 2: Enable PLR rules (cyclomatic complexity, too many arguments)
  - Phase 3: Enable comprehensive type checking
  - Document exemptions with justification

- [ ] **Import Standardization Audit** (Low Priority)
  Verify all modules use standard_imports.setup_module() correctly
  - Found duplicate in gedcom_cache.py (lines 46-76)
  - Scan for other instances of duplicate setup_module calls
