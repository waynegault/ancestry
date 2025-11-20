# Codebase Review Master Todo

Top 10 improvements (refreshed Nov 19 2025, ordered from highest to lowest priority)

1a. (Priority 1) Finish migrating to the unified error-handling stack — ✅ Completed Nov 21 2025
`action7_inbox.py` and `action6_gather.py` now import `api_retry` straight from `core.error_handling`, every decorator application points to the centralized resilience stack, and the legacy `utils.retry_api` implementation plus helper functions were removed. The action modules no longer need interim `DecoratorFactory` casts, giving us a single audited retry pipeline.

1b.  Monitor `core/error_handling.api_retry` telemetry for the next few production runs and regenerate any derived documentation (e.g., `docs/code_graph.json`) so they stop referencing `utils.retry_api`.

2. (Priority 2) Wire the action registry into menu/dispatch flows
`core/action_registry.py` already provides a dataclass-driven catalog of action metadata, browser requirements, and confirmation prompts, but `main.py` still hard-codes the menu (`main.py` §486‑538) and browser/state gating logic (`main.py` §612‑760). Any change requires touching multiple scattered lists, defeating the purpose of the registry.

   Recommendation: Have `main.menu()` and `exec_actn()` consume `core.action_registry.get_action_registry()` so menu rendering, argument validation, and state requirements come from one source of truth, then delete the parallel hard-coded arrays.

3. (Priority 3) Add retention policies for the on-disk Cache/ hierarchy
Several subsystems write directly under `Cache/` without TTL or size limits: `performance_cache.PerformanceCache` creates per-key `.pkl` files in `Cache/performance` and never deletes them (`performance_cache.py` §111‑134, §282‑327), while the health monitor and session tooling accumulate JSON blobs in `Cache/session_checkpoints` and `Cache/session_state` (`health_monitor.py` §1008‑1250). Diskcache handles its own shards, but these bespoke writers will grow indefinitely and are invisible to Grafana.

   Recommendation: introduce a shared retention service (age- or size-based) for `Cache/performance`, session checkpoints, and other ad-hoc directories, and surface the metrics through `core/cache_registry` so operators can monitor cleanup effectiveness.

4. (Priority 4) Add database schema versioning
Schema changes are still applied by calling `Base.metadata.create_all(engine)` at runtime with no migration history (`database.py` §3325‑3365). Any column change requires manual SQL or a destructive reset, which is risky now that production data lives in `Data/ancestry.db`.

   Recommendation: add a lightweight migration runner (Alembic or a custom version table) that tracks applied revisions, ships with roll-forward/rollback scripts, and integrates with the existing `database_manager` utilities.

5. (Priority 5) Break main.py into focused modules
`main.py` has ballooned to 3,841 lines and still mixes menu rendering, caching bootstrap, exec_actn orchestration, analytics launchers, and CLI helpers (`main.py` §1‑3300). This makes regression isolation difficult and slows onboarding.

   Recommendation: extract the menu UI into `ui/menu.py`, move exec_actn helpers into `core/action_runner.py`, and keep `main.py` as a thin CLI entrypoint that wires SessionManager + the registry together.

6. (Priority 6) Stop suppressing configuration warnings
Startup immediately sets `SUPPRESS_CONFIG_WARNINGS=1` (`main.py` §12‑16) and `_should_suppress_config_warnings()` defaults to hiding validation issues whenever tests or scripts run (`main.py` §336‑344). The same flag is re-applied inside `core/session_manager.py` when tests execute (§18‑44). As a result, misconfigured `.env` values never reach the operator.

   Recommendation: remove the blanket suppression, surface the warnings emitted by `ConfigManager`, and treat noisy modules as bugs to fix rather than silencing them globally.

7. (Priority 7) Improve type-safety coverage
`pyrightconfig.json` still disables most of the "unknown type" diagnostics (`reportUnknownParameterType`, `reportUnknownVariableType`, `reportUnknownMemberType`, etc. are all set to "none" in lines 32‑48) even though the codebase leans on heavy `Any` usage (e.g., `action6_gather.PageProcessingMetrics` stores `dict[str, Any]`).

   Recommendation: re-enable the unknown-type checks incrementally (start with warnings), add missing annotations in hot files (`action6_gather.py`, `main.py`, `action7_inbox.py`), and gate new modules on `reportGeneralTypeIssues` = error once coverage improves.

8. (Priority 8) Remove legacy warning suppression from SessionManager
The top of `core/session_manager.py` still redirects `sys.stderr`, installs global `warnings.filterwarnings`, and notes "OBSOLETE" in comments (§18‑43), yet the code path runs every time tests execute. This hides legitimate RuntimeWarnings and complicates debugging.

   Recommendation: delete the stderr redirection/suppression block, ensure tests run via `python core/session_manager.py` work without it, and lean on targeted `warnings.catch_warnings` in the rare helpers that truly need it.

9. (Priority 9) Add genuine integration/e2e tests
`run_all_tests.py` executes 58 module-level suites, but there are no scenarios that exercise the end-to-end workflow (`exec_actn` → SessionManager → action chain). The README even references `python -m test_action6_cache_integration` (`readme.md` §446‑451), yet that module does not exist in the repo.

   Recommendation: create an integration test package that spins up an in-memory SQLite DB plus mocked API responses, runs Action 6 → 7 → 9 via the real SessionManager, and asserts database plus telemetry side effects. Wire it into `run_all_tests.py --integration` so regressions surface before production runs.

10. (Priority 10) Continue profiling Action 6 throughput
Action 6 tracks rich telemetry (`PageProcessingMetrics` in `action6_gather.py` §25‑85 and per-page duration logging in `_prepare_bulk_db_data` §3330‑3380), yet the pipeline remains strictly sequential: every match runs `_process_single_match_for_bulk` one at a time, prefetched data is handled in serial (`_perform_api_prefetches` §2960‑3120), and bulk commits still block on `commit_bulk_data`. Real-world runs continue to hover around 40‑60 s per page under safe rate limits.

   Recommendation: use the existing metrics to identify dominant stages, experiment with batched SQLAlchemy `bulk_save_objects`, prefetch caching, or overlapping I/O (while honoring the 0.3 RPS limiter), and set an explicit 30 s/page SLO to measure progress.
