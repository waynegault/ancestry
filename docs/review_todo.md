# Codebase Review Master Todo *(Updated 2025-11-16)*

All open work is captured in the single checklist below. Address items in priority order unless a dependency is noted.

## ✅ Completed Since 2025-11-16

- [x] **AI Quality Telemetry Enhancements** – Prompt telemetry now records provider metadata and sanitized scoring inputs, supports `--provider` filtering across all CLI modes, and emits automatic regression alerts when rolling medians drop ≥7.5 points. See `prompt_telemetry.py` for schema and alert handling details.

- [x] **Comprehensive Retry Strategy** (Est. 3h)
   Unify the API and Selenium retry decorators so both channels share tuned attempt counts, jitter, and stop conditions derived from recent telemetry. Success = a single configuration surface plus regression coverage in `action6_gather.py`, `action7_inbox.py`, and `core/session_manager.py`.
   *2025-11-17*: Added `api_retry`/`selenium_retry` helpers wired to `config_schema.retry_policies`, migrated Action 6/7 + SessionManager call sites, and shipped regression tests that assert helper usage and telemetry-aligned settings.

- [x] **Session State Machine** (Est. 4h)
   Formalize explicit lifecycle states for `SessionManager` (e.g., UNINITIALIZED → READY → DEGRADED) with guard methods that callers can interrogate instead of ad-hoc readiness checks. Success = state diagram in code comments and enforcement inside `exec_actn()`.
   *2025-11-18*: Added `SessionLifecycleState` enum with diagram, lifecycle guard helpers, and `guard_action()` enforcement inside `exec_actn()`. Readiness now transitions through RECOVERING → READY and degrades safely, with regression coverage in `core/session_manager.py` module tests.

- [ ] **Logging Standardization** (Est. 2h)
   Normalize log levels, prefixes, and emoji usage across action modules and shared utilities so operators can grep consistently. Success = shared helper in `logging_utils.py` plus updated calls in Actions 6–10.

- [ ] **Workflow Replay Capability** (Est. 4h)
   Capture anonymized session traces (API inputs/outputs, key browser steps) and provide a replay harness that stubs external calls for offline debugging. Success = `replay/` toolkit plus documentation on capturing runs.

- [ ] **Dead Code Cleanup** (Est. 2h)
   Audit for legacy helpers/tests left from Phase 5 refactors and remove or quarantine them. Prioritize `connection_resilience.py`, legacy browser recovery code, and redundant cache utilities.

- [ ] **Performance Profiling Utilities** (Est. 3h)
   Package reusable cProfile/timing helpers (decorators + CLI switches) for long-running actions so perf data is easy to capture without manual scripts.

- [ ] **Schema Versioning & Lightweight Migrations** (Est. 2h)
   Introduce SQLite schema version stamps plus a minimal migration runner to unblock future columns without manual SQL.

- [ ] **Data Integrity Checker** (Est. 3h)
   Schedule periodic audits for soft deletes, UUID uniqueness, and cross-table consistency; emit alerts to Logs and Grafana.

- [ ] **Centralize Test Utilities** (Est. 3h)
   Finish moving duplicated helpers into `test_utilities.py`, including migrating remaining temp-file usage in `logging_config.py`, `diagnose_chrome.py`, and `config/config_manager.py`.

- [ ] **Strengthen Assertions** (Est. 2h)
   Deepen coverage in `gedcom_intelligence.py` and `message_personalization.py` with edge-case fixtures, explicit assertion messaging, and negative-path tests.

- [ ] **Enforce Test Quality Gates** (Est. 3h)
   Wire `analyze_test_quality.py` into smoke-test workflows so suites fail fast when signal quality drops below threshold.

- [ ] **Separate Unit vs Integration Tests** (Est. 4–6h)
   Tag tests with `test_type`, add `run_unit_tests.py` / `run_integration_tests.py`, and update CI docs so fast-feedback paths skip live-session dependencies.

- [ ] **Tighten Lint & Type Enforcement** (Est. 3h)
   Gradually remove Ruff ignores (start with F821/E722), then raise Pyright severities from warnings to errors once violations are cleared.

- [ ] **Comment & Docstring Spot Check** (Est. 1h per sprint)
   After each merge, scan new comments/docstrings for tone/verbosity alignment and adjust immediately.

- [ ] **Regression Guardrails** (Ongoing)
   Enforce the policy: run `python run_all_tests.py --fast` and `ruff check .` for every behavioral change before pushing.

- [ ] **Knowledge Graph & README Export** (Est. 1h)
   Once Phase 5 closes, regenerate the code graph/README snapshots and commit the artifacts so downstream consumers have the latest state.

- [ ] **Maintainer Handoff Brief** (Est. 1h)
   Summarize outcomes, unresolved risks, and recommended sequencing for the next maintainer before project pause/transition.
