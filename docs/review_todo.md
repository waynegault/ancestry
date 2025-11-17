# Codebase Review Master Todo *(Updated 2025-11-17)*

All open work is captured in the single checklist below. Address items in priority order unless a dependency is noted.

- [x] **Performance Profiling Utilities** (Est. 3h)
   Package reusable cProfile/timing helpers (decorators + CLI switches) for long-running actions so perf data is easy to capture without manual scripts.
  - ✅ Created `performance_profiling.py` with cProfile integration
  - ✅ Implemented `@profile_with_cprofile` decorator for full cProfile profiling with .stats and .txt output
  - ✅ Implemented `@time_function` decorator for lightweight timing without profiling overhead
  - ✅ Added `enable_profiling_from_cli()` for CLI flag support (--profile, --profile-output)
  - ✅ Added `ProfileConfig` dataclass for centralized configuration
  - ✅ Generated both machine-readable (.stats) and human-readable (.txt) reports
  - ✅ All 7 comprehensive tests passing (decorators, CLI integration, report generation)

- [ ] **Schema Versioning & Lightweight Migrations** (Est. 2h)
   Introduce SQLite schema version stamps plus a minimal migration runner to unblock future columns without manual SQL.

- [ ] **Data Integrity Checker** (Est. 3h)
   Schedule periodic audits for soft deletes, UUID uniqueness, and cross-table consistency; emit alerts to Logs and Grafana.

- [ ] **Strengthen Assertions** (Est. 2h)
   Deepen coverage in `gedcom_intelligence.py` and `message_personalization.py` with edge-case fixtures, explicit assertion messaging, and negative-path tests.

- [ ] **Tighten Lint & Type Enforcement** (Est. 3h)
   Gradually remove Ruff ignores (start with F821/E722), then raise Pyright severities from warnings to errors once violations are cleared.

- [ ] **Comment & Docstring Spot Check** (Est. 1h per sprint)
   After each merge, scan new comments/docstrings for tone/verbosity alignment and adjust immediately.

- [ ] **Regression Guardrails** (Ongoing)
   Enforce the policy: run `python run_all_tests.py --fast` and `ruff check .` for every behavioral change before pushing.

- [ ] **Knowledge Graph & README Export** (Est. 1h)
   regenerate the code graph/README snapshots and commit the artifacts so downstream consumers have the latest state.

- [ ] **Maintainer Handoff Brief** (Est. 1h)
   Summarize outcomes, unresolved risks, and recommended sequencing for the next maintainer before project pause/transition.

- [ ] **Workflow Replay Capability** (Est. 4h)
   Capture anonymized session traces (API inputs/outputs, key browser steps) and provide a replay harness that stubs external calls for offline debugging. Success = `replay/` toolkit plus documentation on capturing runs.
