# Codebase Review Master Todo *(Updated 2025-11-17)*

All open work is captured in the single checklist below. Address items in priority order unless a dependency is noted.

- [x] **Tighten Lint & Type Enforcement** (Est. 3h)
  Gradually remove Ruff ignores (start with F821/E722), then raise Pyright severities from warnings to errors once violations are cleared.

  **Completed:**
  - [x] Fixed markdown indentation in review_todo.md (MD007/MD005)
  - [x] Removed unused `contextlib` import from config_manager.py
  - [x] Fixed LOG_DIRECTORY constant redefinition in logging_config.py
  - [x] Added type annotations (Any) to 7 person_record parameters in gedcom_intelligence.py
  - [x] Fixed 5 unused 'functions' variables in message_personalization.py tests (marked with underscore)
  - [x] All Pylance errors and warnings resolved
  - [x] F821 and E722 Ruff checks passing

- [ ] **Comment & Docstring Spot Check** (Est. 1h per sprint)
  After each merge, scan new comments/docstrings for tone/verbosity alignment and adjust immediately.

- [ ] **Schema Versioning & Lightweight Migrations** (Est. 2h)
  Introduce SQLite schema version stamps plus a minimal migration runner to unblock future columns without manual SQL.

- [ ] **Data Integrity Checker** (Est. 3h)
  Schedule periodic audits for soft deletes, UUID uniqueness, and cross-table consistency; emit alerts to Logs and Grafana.

- [ ] **Knowledge Graph & README Export** (Est. 1h)
  regenerate the code graph/README snapshots and commit the artifacts so downstream consumers have the latest state.

- [ ] **Maintainer Handoff Brief** (Est. 1h)
  Summarize outcomes, unresolved risks, and recommended sequencing for the next maintainer before project pause/transition.
