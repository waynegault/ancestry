# Action 6 Decomposition Plan (Phase 1)

> Tracking TODO: "Decompose `action6_gather.py` (9280 lines) into a package" – this document captures the discovery work requested in Phase 1.

## Current `coord()` Flow (today)

| Stage | Responsibilities | Key helpers in `action6_gather.py` | Candidate destination |
| --- | --- | --- | --- |
| Session bootstrap | Validate that `SessionManager` already owns a live driver/db session, hydrate state/checkpoints, log banner | `_validate_session_state`, `_build_coord_state`, `_determine_start_page`, `_log_action_start` | `actions/gather/orchestrator.py` (pairs with `actions/gather/checkpoint.py`)
| Initial fetch & pagination | Navigate to DNA matches, call TreesUI list API for page `start`, determine total pages/run window, cache first page payload | `_handle_initial_fetch`, `_navigate_and_get_initial_page_data`, `_determine_page_processing_range` | `actions/gather/orchestrator.py` + `actions/gather/fetch.py`
| Main loop control | Walk page range, enforce session/health checks, checkpoint progress, rate-limit between pages | `_main_page_processing_loop`, `_process_single_page`, `_check_and_handle_session_health`, `_try_fast_skip_page`, `_apply_rate_limiting`, `persist_checkpoint` | `actions/gather/orchestrator.py` (loop) + `actions/gather/checkpoint.py` (persistence)
| Page fetch + normalization | Acquire DB session, fetch list view data, merge extra columns (in-tree cache, relationship labels), fast-skip unchanged pages | `_handle_page_fetch_and_validation`, `_get_database_session_with_retry`, `_fetch_page_matches`, `get_matches`, `_refine_matches_for_page`, `_load_in_tree_status_from_cache`, `_decorate_relationships` | `actions/gather/fetch.py`
| Prefetch detail APIs | Decide which UUIDs need detail fetches, deduplicate cached payloads, call `/details`, `/profiles/details`, `badgeDetails`, `getLadder`, `ethnicity` APIs sequentially | `_identify_fetch_candidates`, `_perform_api_prefetches`, `_Action6PrefetchPlan` helpers, `_deduplicate_api_requests`, `_get_combined_match_details` | `actions/gather/fetch.py`
| Persistence/batch commit | Compare incoming matches to DB, generate INSERT/UPDATE payloads for `Person`, `DnaMatch`, `FamilyTree`, split ethnicity updates, update caches | `_do_batch`, `_process_dna_data_*`, `_persist_new_persons`, `_persist_match_updates`, `_bulk_upsert_family_trees`, `_resolve_person_ids`, `_apply_ethnicity_updates` | `actions/gather/persistence.py`
| Telemetry & status | Track per-page metrics, emit timing summaries, performance snapshots, final action banners, rate limiter stats | `PageProcessingMetrics`, `_log_page_completion_summary`, `_accumulate_page_metrics`, `_emit_final_summary`, `_emit_timing_breakdown`, `_emit_rate_limiter_metrics`, `_emit_action_status` | `actions/gather/metrics.py`
| Cleanup | Finalize checkpoint, log run summary, bubble keyboard interrupts | `finalize_checkpoint_after_run`, `_log_final_results`, `_emit_action_status`, `coord()` finally block | `actions/gather/orchestrator.py` + `actions/gather/metrics.py`

### Notes on helper clustering
- **Checkpoint/Resume** logic already lives in `actions/gather/checkpoint.py`; `_build_coord_state` and `_determine_start_page` can delegate to that service instead of duplicating logic inside `action6_gather.py`.
- **Prefetch planning** is encapsulated by `Action6PrefetchPlan` + `_perform_api_prefetches` (~400 lines). This is an ideal seam for `actions/gather/fetch.py` so `orchestrator` only needs a `FetchCoordinator` interface.
- **Batch commits** are isolated under `_do_batch` and the helpers it calls (`_prepare_batch`, `_process_single_match_for_bulk`, `_persist_*`). These can move to `actions/gather/persistence.py` once the orchestrator receives a minimal contract like `BatchResult = persistence.process_page(matches, prefetched_data, session_manager)`.
- **Metrics** already use the `PageProcessingMetrics` dataclass. Extracting this (plus `_log_timing_breakdown*`, `_compose_progress_snapshot`, `_log_page_completion_summary`) allows the orchestrator to remain <500 lines and focus purely on control flow.

## External dependencies to preserve

Category | Examples | Notes
--- | --- | ---
Session orchestration | `SessionManager` (`ensure_session_ready`, `get_db_conn`, `return_session`, rate limiter) | New modules must accept a `SessionManager` handle instead of importing browser/db utilities directly to avoid cycles.
Database layer | `db_transn`, SQLAlchemy `Session`, models (`Person`, `DnaMatch`, `FamilyTree`) | Persistence module will be the only layer talking to SQLAlchemy and should expose pure-Python DTOs to orchestrator.
Utilities | `utils._api_req`, `utils.log_starting_position`, `utils.log_final_summary`, `_api_req_with_auth_refresh` | These imports must stay centralized; extracting fetch logic gives us a single place to wrap `_api_req`.
Cache | `cache.cache` (diskcache), `core.unified_cache_manager`, `Cache/action6_checkpoint.json` | Both fetch and persistence layers rely on caching; plan to inject cache handles (so modules stay testable) and keep `checkpoint` module owning disk IO.
Telemetry | `health_monitor`, `performance_monitor.track_api_performance`, `rate_limiter.persist_rate_limiter_state` | `actions/gather/metrics.py` will wrap these dependencies, so orchestrator just forwards events.
Config | `config_schema` (batch size, max pages, checkpoint toggles) | Provide typed config blobs to modules rather than importing globally to avoid surprises during testing.
API helpers | `api_utils` (`call_relation_ladder_with_labels_api`, `get_relationship_path_data`) and `relationship_utils` conversions | Keep these as pure helper imports inside the new `fetch` module to prevent a cross-import between orchestrator and `api_utils`.
Selectors/Selenium | `my_selectors`, `selenium_utils.get_driver_cookies` | Only the fetch module needs DOM selectors; orchestrator should stay Selenium-agnostic beyond `SessionManager` health checks.

## Proposed package structure (Phase 2 target)

```
actions/gather/
├── orchestrator.py      # coord(), _main_page_processing_loop, state machine, health checks
├── fetch.py             # list view fetch, detail prefetching, dedupe/cache integration
├── persistence.py       # DB lookups, batch writes, ethnicity/tree helpers
├── metrics.py           # PageProcessingMetrics + logging/telemetry/reporting helpers
├── checkpoint.py        # already live (resume plan + file IO)
└── __init__.py          # lightweight exports for main.py/action6 entrypoint
```

Each module should expose a thin interface back to `action6_gather.py` during transition, e.g. `from actions.gather.orchestrator import run_action6` so we can peel functionality without breaking CLI entry points or tests.

## Immediate next steps (Phase 2 prep)
1. **Create orchestrator/fetch/persistence/metrics skeletons** matching the responsibilities above and re-export them via `actions/gather/__init__.py`.
2. **Lift shared dataclasses/state** (`PageProcessingMetrics`, `Action6State`, `GatherCheckpointPlan`) into their destination modules so they can be imported without reaching into the monolith.
3. **Evolve `coord()` into a façade** that calls `actions.gather.orchestrator.run(session_manager, start_page)`; the monolith keeps existing helpers until moved, but this isolates all new work in the package.
4. **Add focused `TestSuite` modules** for each new file as soon as skeletons land (can start with smoke tests asserting wiring and dependency injection), preventing regressions while pieces are relocated.

This plan completes the Phase 1 discovery requirement: we now have a documented map of the existing flow, named seams for each helper cluster, and an explicit dependency list to avoid import cycles when the code moves during Phases 2–4.
