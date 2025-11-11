# Codebase Review Master To-Do

> Working document for the comprehensive architecture & documentation audit. Tasks are grouped by phase. Update inline as work progresses.

## Phase 0 · Ground Rules & Tooling
- [x] Confirm and document governing instructions, coding standards, and rate limiting invariants.
- [x] Stand up shared note-taking artifacts (graph schema, README outline, findings log).
- [x] Enumerate automation available via MCP tools; script reusable queries where helpful.
  - Tooling inventory: `read_file`, `apply_patch`, `grep_search`, `file_search`, `mcp_pylance_mcp_s_*` (syntax check, run code snippets, env info), `runTests`, `run_in_terminal`, `list_dir`, `semantic_search`.
  - Reusable queries: standard AST def enumerator for module coverage, globbed `grep_search` for call-site discovery, `semantic_search` for cross-module dependency hints, and `runTests` targeting module-specific test suites.

## Phase 1 · Discovery & Baseline Capture
- [x] Generate repository inventory (modules, tests, data assets) for quick reference.
- [x] Capture current README structure/content for gap analysis.
- [x] Collect existing architecture notes from `.github/copilot-instructions.md`, `README`, and inline comments.
- [x] Snapshot key configuration (.env, config_schema) and operational scripts.

## Phase 2 · Knowledge Graph Design & Population
- [x] Define node/edge taxonomy (modules, classes, functions, workflows, dependencies, quality flags).
- [x] Implement storage strategy (e.g., MCP memory entries per file or serialized JSON artifact in `docs/`).
- [x] Establish per-file review template (purpose, method, data flow, quality notes, refactor ideas).
- [ ] Populate graph iteratively as each file is reviewed; version changes where appropriate.

## Phase 3 · Systematic Code Walkthrough
- [ ] Determine traversal order (entry points → core managers → actions → utils → tests → scripts).
- [ ] For each file:
  - [ ] Read top-to-bottom, summarizing responsibilities and interactions.
  - [ ] Evaluate and update comments/docstrings for clarity and accuracy.
  - [ ] Flag deprecated, redundant, or duplicate logic.
  - [ ] Record quality reservations, extension ideas, and dependency notes in the graph.
  - [ ] Note testing coverage or gaps.
- [ ] Track progress checklist for all modules; ensure no files skipped.

## Phase 4 · Opportunity Synthesis
- [ ] Analyze graph data to surface systemic risks, dead code, duplication, or modernization targets.
- [ ] Draft prioritized improvement backlog (risk, impact, effort).
- [ ] Identify quick wins vs. long-term initiatives.

## Phase 5 · README Overhaul
- [ ] Map current content to requested structure; note missing information.
- [ ] Draft new sections:
  - Executive Summary / Overview (purpose)
  - Quickstart (installation, configuration, actions, usage tips)
  - Architecture (core components, global session pattern, code quality/testing pattern, rate limiting & timeout protection, database schema, API endpoints, AI infrastructure)
  - Functions (Actions 0-10)
  - Troubleshooting
  - Future Developer Ideas (code improvement, speeding up, new capabilities)
  - Development Guidelines (code quality, testing, rate limiting, git workflow)
  - Contributing
  - License
  - Support
  - Appendices (Chronology of Changes, Technical Specifications, Test Review Summary)
- [ ] Validate cross-references and command snippets.
- [ ] Incorporate improvement opportunities identified in Phase 4.

## Phase 6 · Validation & Finalization
- [ ] Spot-check updated comments/docstrings for tone and brevity (per instructions).
- [ ] Run targeted tests or linting if behavioral code changed.
- [ ] Export/commit graph artifact and README updates.
- [ ] Summarize outcomes, open questions, and recommended next steps.

---

### Progress Log
- _2025-11-10:_ Roadmap established; master to-do scaffolding created.
- _2025-11-10:_ Completed first-pass analysis of `main.py`; populated graph nodes and flagged duplicated session guards for Actions 7-9 wrappers.
- _2025-11-10:_ Began `core/session_manager.py` review (init → cookie sync); added method nodes and noted coupling/monitoring concerns.
- _2025-11-10:_ Finished `core/session_manager.py` audit including health validation, identifier lookups, and watchdog helpers; expanded graph with recovery patterns and telemetry opportunities.
- _2025-11-10:_ Reviewed `core/api_manager.py`; documented cookie sync flow, identifier caching weaknesses, and added APIManager nodes/edges to graph.
- _2025-11-10:_ Audited `core/browser_manager.py`; captured lifecycle nuances and wired BrowserManager nodes with SessionManager dependencies in the graph.
- _2025-11-10:_ Studied `core/database_manager.py`; summarized pooling telemetry, async helpers, and linked DatabaseManager nodes plus readiness edges in the graph.
- _2025-11-10:_ Documented `core/enhanced_error_recovery.py`; added nodes for retry decorators, enums, and hooked wrap relationships for API/DB/File presets into the knowledge graph.
- _2025-11-10:_ Analyzed `core/error_handling.py`; mapped AppError hierarchy, decorator exports, and added SessionManager dependency edge to the graph.
- _2025-11-10:_ Reviewed `core/session_validator.py`; charted readiness stages, skip heuristics, and cookie/CSRF dependencies in the knowledge graph and progress log.
- _2025-11-10:_ Mapped login/navigation utilities in `utils.py`; captured log_in, login_status, and nav_to_page flows with new graph nodes and SessionValidator dependency edges.
- _2025-11-10:_ Captured SessionManager rate limiter configuration helpers and sleep-prevention utilities, linking `utils.get_rate_limiter`, `prevent_system_sleep`, and `restore_system_sleep` into the dependency graph.
- _2025-11-10:_ Documented `utils._api_req` pipeline (config dataclass, retry loop, response handlers) and recorded RetryContext usage plus SessionManager limiter interactions in the graph.
- _2025-11-10:_ Added header preparation, cookie sync, and rate limiter wait helpers (`_prepare_api_headers`, `_sync_cookies_for_request`, `_apply_rate_limiting`, etc.) to the knowledge graph with updated dependency edges.
- _2025-11-10:_ Captured exponential backoff calculators (`_calculate_retry_sleep_time`, `_calculate_sleep_time`) and wired them to retry handlers for full API request pipeline coverage.
- _2025-11-10:_ Documented legacy and API-specific retry decorators (`retry`, `retry_api`) plus helper functions, adding configuration and call edges in the knowledge graph.
- _2025-11-10:_ Added browser guard and wait instrumentation decorators (`ensure_browser_open`, `time_wait`) alongside driver extraction helpers to the knowledge graph and progress log.
- _2025-11-10:_ Captured the `CircuitBreaker` class with method-level nodes and state-transition edges, highlighting recovery flows for 429 suppression.
- _2025-11-10:_ Documented `utils` login and 2FA helpers (consent dismissal, credential entry, 2FA polling/verification) with corresponding call edges linking the flow end-to-end in `docs/code_graph.json`.
- _2025-11-10:_ Added login status fallbacks and navigation resilience helpers (API/UI probes, nav retries, unavailability handling) to the knowledge graph, completing call wiring for `nav_to_page` and `login_status` ecosystems.
- _2025-11-10:_ Captured logging/summary utilities, cookie persistence helpers, name formatting pipeline, UBE generators, and module test harness in the graph, wiring action6 final reporting to shared logging nodes.
- _2025-11-11:_ Documented `BrowserManager._load_saved_cookies`, linked its invocation chain to `utils._load_login_cookies`, and completed cookie persistence coverage in the graph.
- _2025-11-11:_ Cataloged MCP tooling repertoire and reusable queries (AST enumerator, grep patterns, targeted `runTests`) to support ongoing audit automation.
- _2025-11-11:_ Generated `docs/repo_inventory.md` to capture the repository baseline and prepared to continue Phase 1 discovery tasks.
- _2025-11-11:_ Captured README baseline structure in `docs/readme_snapshot.md`, summarized architecture directives in `docs/architecture_notes_snapshot.md`, and recorded configuration expectations in `docs/config_snapshot.md` to complete Phase 1 discovery tasks.
- _2025-11-11:_ Expanded `docs/code_graph_plan.md` with controlled node/edge vocabularies, storage/validation workflow, and a per-file review template to kick off Phase 2.
- _2025-11-11:_ Began Phase 3 walkthrough with `action6_gather.py`; documented core workflow and helper nodes in `docs/code_graph.json`, highlighting batching complexity and final logging structure.
- _2025-11-11:_ Completed Action 6 prefetch planning coverage; added knowledge graph nodes/edges for priority classification, ethnicity gating, sequential API prefetch helpers, and ladder enrichment flows.
- _2025-11-11:_ Mapped Action 6 data-preparation and bulk database helpers, capturing per-match prep, integrity protections, and DNA/FamilyTree pipelines within `docs/code_graph.json`.
- _2025-11-11:_ Added Action 7 inbox follow-up heuristic coverage to `docs/code_graph.json` and confirmed payload spot-checks for PRODUCTIVE inbound/outbound scenarios.
- _2025-11-11:_ Completed Action 7 follow-up pipeline mapping (conversation phase, inbound/outbound handling, reminder creation) in `docs/code_graph.json`, capturing commit/error recovery helpers.
- _2025-11-11:_ Began Action 8 messaging coverage; documented orchestration entrypoint and candidate processing loop nodes/edges in `docs/code_graph.json` for Phase 3 walkthrough.
- _2025-11-11:_ Expanded Action 8 coverage to include per-person processing helpers (template selection, mode filtering, send/simulate, counter aggregation) in `docs/code_graph.json`.
- _2025-11-11:_ Completed Action 8 documentation sweep with initialization, batching, commit, cleanup, and sequencing helpers wired into `docs/code_graph.json`.
- _2025-11-11:_ Added `run_all_tests.py` orchestration coverage to `docs/code_graph.json`, documenting environment setup, execution paths, reporting, and log analysis helpers.
- _2025-11-11:_ Documented `analytics.py` logging and reporting helpers, adding nodes for transient extras handling, event logging, and weekly summary generation in `docs/code_graph.json`.
- _2025-11-11:_ Documented `analyze_test_quality.py` by mapping `TestQualityAnalyzer` class and method heuristics into `docs/code_graph.json`, including script entry edges and call relationships for assertion detection helpers.
- _2025-11-11:_ Captured `code_quality_checker.py` coverage with file, class, method, and script nodes outlining quality heuristics, CLI flow, and in-module tests within `docs/code_graph.json`.
- _2025-11-11:_ Added `test_utilities.py` nodes covering core factories, validator helpers, module tests, and CLI entry wiring inside `docs/code_graph.json`, highlighting run_comprehensive_tests linkage.
- _2025-11-11:_ Added `action9_process_productive.py` coverage, documenting orchestration functions, processing state classes, and batch commit flow in `docs/code_graph.json`.
- _2025-11-11:_ Added `action10.py` nodes for the GEDCOM analysis entrypoint, input helpers, scoring pipeline, and presentation functions within `docs/code_graph.json`.
- _2025-11-11:_ Captured `search_criteria_utils.py` as the shared criteria prompt layer, adding helper/test nodes and Action 10 linkage in `docs/code_graph.json`.
- _2025-11-11:_ Documented `genealogy_presenter.py`, mapping presenter helpers and unified output flow into `docs/code_graph.json` with internal call edges.
- _2025-11-11:_ Added `relationship_utils.py` coverage, including BFS cache, GEDCOM/API converters, and cross-module dependencies in `docs/code_graph.json`.
- _2025-11-11:_ Documented `gedcom_utils.py`, capturing normalization helpers, GedcomData cache methods, scoring pipeline, and Action 10 linkages inside `docs/code_graph.json`.
