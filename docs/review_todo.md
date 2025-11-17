# Codebase Review Master Todo *(Updated 2025-11-17)*

All open work is captured in the single checklist below. Address items in priority order unless a dependency is noted.

## High Priority Technical Debt

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
