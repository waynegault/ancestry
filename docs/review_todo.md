# Codebase Review Master Todo *(Updated 2025-11-17)*

All open work is captured in the single checklist below. Address items in priority order unless a dependency is noted.

## High Priority Technical Debt

- [x] **Linting Configuration Hardening** (Low Priority, High Impact)
  Completed 2025-11-17:
  - Phase 1 enabled `reportReturnType` and `reportUnusedVariable` inside `pyrightconfig.json`
  - Repository is clean under the stricter settings (`npx pyright` reports 0 warnings)
  - Phase 2+ (PLR rules, comprehensive type checking) remain open for future tightening

- [x] **Import Standardization Audit** (Low Priority)
  Completed 2025-11-17:
  - Removed duplicate `setup_module` invocations from `gedcom_cache.py`, `utils.py`, `config/config_manager.py`, `core/browser_manager.py`, `core/database_manager.py`, and `core/session_manager.py`
  - Verified remaining modules reference `standard_imports.setup_module()` exactly once (examples in `standard_imports.py` are documentation/tests)
