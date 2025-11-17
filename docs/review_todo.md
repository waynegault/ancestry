# Codebase Review Master Todo *(Updated 2025-11-17)*

All open work is captured in the single checklist below. Address items in priority order unless a dependency is noted.

## High Priority Technical Debt

- [x] **Cache Module Consolidation** (High Priority)
  Consolidate 7 cache-related modules into unified architecture
  - Completed Nov 20, 2025: Added `core/cache_registry.py`, onboarded tree statistics cache, exposed performance cache stats, and switched menu/performance dashboards to the registry output for one-click warm/clear operations.

- [x] **Database Module Consolidation** (Medium Priority)
  Complete migration from database.py to core/database_manager.py
  - Verify all functionality moved to core module
  - Update remaining references
  - Archive or remove legacy database.py
  - Completed Nov 20, 2025: Moved `db_transn` and backup pipelines into `core/database_manager.py`, updated action/main imports, and left `database.py` as an ORM-focused shim that simply re-exports those helpers for backward compatibility.

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
