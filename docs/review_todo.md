# Codebase Review Master Todo *(Updated 2025-11-16)*

All open work is captured in the single checklist below. Address items in priority order unless a dependency is noted.

- [x] **Logging Standardization** (Est. 2h)
   Normalize log levels, prefixes, and emoji usage across action modules and shared utilities so operators can grep consistently. Success = shared helper in `logging_utils.py` plus updated calls in Actions 6–10.
  - ✅ Added `log_action_banner` helper in `core/logging_utils.py` with start/success/failure stages and detail formatting.
  - ✅ Actions 6–10 and shared `utils.log_action_status()` now emit standardized lifecycle banners with consistent emoji/prefix payloads.

- [x] **Dead Code Cleanup** (Est. 2h)
   Audit for legacy helpers/tests left from Phase 5 refactors and remove or quarantine them. Prioritize `connection_resilience.py`, legacy browser recovery code, and redundant cache utilities.
  - ✅ Archived 5 standalone diagnostic scripts to `scripts/archive/`: ai_api_test.py, analyze_test_quality.py, comprehensive_auth_tests.py, lm_studio_manager.py, standardize_test_runners.py
  - ✅ Verified connection_resilience.py, diagnose_chrome.py, grafana_checker.py, gedcom_cache.py are actively used in production

- [ ] **Centralize Test Utilities** (Est. 3h)
   Finish moving duplicated helpers into `test_utilities.py`, including migrating remaining temp-file usage in `logging_config.py`, `diagnose_chrome.py`, and `config/config_manager.py`.

- [ ] **Performance Profiling Utilities** (Est. 3h)
   Package reusable cProfile/timing helpers (decorators + CLI switches) for long-running actions so perf data is easy to capture without manual scripts.

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
