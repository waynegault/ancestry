# Codebase Review Master Todo

Top 10 improvements (refreshed Nov 19 2025, ordered from highest to lowest priority)

<!-- markdownlint-disable MD029 -->

1a. (Priority 1) Finish migrating to the unified error-handling stack — ✅ Completed Nov 21 2025
`action7_inbox.py` and `action6_gather.py` now import `api_retry` straight from `core.error_handling`, every decorator application points to the centralized resilience stack, and the legacy `utils.retry_api` implementation plus helper functions were removed. The action modules no longer need interim `DecoratorFactory` casts, giving us a single audited retry pipeline.

1b.  Monitor `core/error_handling.api_retry` telemetry for the next few production runs and regenerate any derived documentation (e.g., `docs/code_graph.json`) so they stop referencing `utils.retry_api`.

2. (Priority 2) Wire the action registry into menu/dispatch flows — ✅ Completed Nov 23 2025
`main.menu()` now renders every primary, meta, and test action straight from `core.action_registry`, and the registry’s metadata drives browser requirements, config injection, input hints, and session gating. The legacy `MENU_OPTIONS`, manual `input().startswith()` parsing, and bespoke `_handle_*` helpers were deleted, so adding a new action only requires a registry entry plus a function binding.

3. (Priority 3) Add retention policies for the on-disk Cache/ hierarchy — ✅ Completed Nov 23 2025
New `cache_retention.CacheRetentionService` enforces age/size/count policies for `Cache/performance`, `Cache/session_checkpoints`, and `Cache/session_state`, publishing per-target metrics via the cache registry. Automatic sweeps run hourly (and immediately when performance cache initializes or session checkpoints/state write to disk), and operators can trigger ad-hoc cleanup through the Cache Statistics screen or `registry.clear("cache_retention")`.

4. (Priority 4) Add database schema versioning — ✅ Completed Nov 24 2025
`core/schema_migrator.py` now registers structured migrations (starting with `0001_baseline`), persists applied versions in the new `schema_migrations` table, and exposes both programmatic helpers plus a CLI (`python core/schema_migrator.py --list/--apply`) for operators. `DatabaseManager` automatically invokes the migrator after ensuring tables exist, and the `migrate-db` meta action in `main.py` surfaces migration status from the menu, so schema upgrades can run without manual SQL or destructive resets.

5. (Priority 5) Break main.py into focused modules
`main.py` is still ~3,500 lines and mixes menu rendering, caching bootstrap, analytics launchers, and CLI helpers (`main.py` §1‑3300). On Nov 25 we finally moved the `exec_actn` orchestration (state detection, analytics logging, metrics) into `core/action_runner.py` and now configure it via `configure_action_runner()` so `main.py` imports shared helpers instead of maintaining a private copy.

   Recommendation: continue trimming `main.py` by peeling off the remaining subsystems (caching bootstrap, CLI utilities, test harness) into dedicated modules so the entrypoint just wires SessionManager + the registry together. Once that is done, reassess whether further menu slimming is needed beyond the existing `ui/menu.py` renderer.

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

<!-- markdownlint-enable MD029 -->
