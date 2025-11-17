# Codebase Review Master Todo *(Updated 2025-11-17)*

All open work is captured in the single checklist below. Address items in priority order unless a dependency is noted.

## High Priority Technical Debt

- [x] **Function Decomposition** (Medium Priority)
  Completed 2025-11-17:
  - main.exec_actn now delegates logging/analytics/cleanup to `_ActionExecutionContext` helpers
  - main.run_gedcom_then_api_fallback orchestrates via new comparison input/execution/render helpers
  - Actions 7/8/9 share `_ensure_navigation_ready` for driver/nav guard logic

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
