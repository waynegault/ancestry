# Action 6 Gather Refactor – Discovery Notes

_Last updated: 2025-11-23_

## Current State Summary
- **File**: `action6_gather.py`
- **Size**: ~9,280 LOC with 45 top-level helpers.
- **Purpose**: End-to-end orchestration for DNA match gathering (pagination, API fetches, DB persistence, telemetry, checkpointing, and circuit-breaker safety).
- **Pain Points**: monolithic `coord()` logic, tight coupling between API fetch, data normalization, persistence, and telemetry, making the flow hard to reason about and nearly impossible to unit test without live runs.

## High-Level Execution Flow (`coord()`)
1. **Session + State Prep**: `_validate_session_state()`, `_build_coord_state()`, `_log_action_start()` consume `SessionManager`, `config_schema`, and optional start overrides. If checkpoints exist, metadata is loaded via `_load_checkpoint()`.
2. **Page Planning**: `_execute_coord_run()` calls `_prepare_page_range()` to clamp the request, returning `(last_page, total_pages)` along with initial match payload stored in `state["matches_on_current_page"]`.
3. **Main Loop**: `_main_page_processing_loop()` iterates pages and delegates to `_process_single_page()`, which orchestrates fetch → normalization → persistence, updating cumulative counters in `state["counters"]`.
4. **Page Fetch**:
   - `_fetch_page_matches()` obtains the summary list view (`_get_matches_page()` → `api_utils.get_matches`), applies `_normalize_match_payload()` and `_filter_matches_by_threshold()`.
   - `_fetch_detail_for_candidates()` drives `_fetch_match_detail()` per UUID with rate-limiting & retry. It depends on `_apply_rate_limiter_delay()`, `_api_req_with_auth_refresh()`, and `SessionManager.api_manager`.
5. **Persistence Phase**:
   - `_lookup_existing_persons()` preloads DB rows via SQLAlchemy (joins `Person`, `DnaMatch`, `FamilyTree`).
   - `_identify_fetch_candidates()` and `_merge_detail_payloads()` queue enriched matches.
   - `_perform_person_upserts()` and `_commit_batch_results()` insert/update via `db_transn()` plus bulk utilities such as `_bulk_insert_persons()`, `_bulk_insert_dna_matches()`, `_bulk_insert_family_tree()`.
6. **Telemetry & Checkpointing**:
   - `PerformanceMetrics` + `_log_progress()` track throughput and ETA, writing to `Logs`.
   - `_save_checkpoint()`, `_finalize_checkpoint_after_run()`, `_remove_stale_checkpoint_files()` persist JSON checkpoints under `Cache/action6_checkpoint.json` (dependency on `cache_manager.atomic_write_json`).
7. **Safety Systems**: `_handle_coord_failure()`, `_record_critical_api_failure()`, `_reset_failure_counters()`, plus `SessionCircuitBreaker` integration prevent runaway API retries.

## Proposed Package Decomposition (Phase 2 Targets)
| Future Module | Responsibilities | Candidate Helpers / Types |
| --- | --- | --- |
| `actions/gather/orchestrator.py` | Public `coord()` entry, state building, loop control, keyboard interrupt handling. | `_validate_session_state`, `_build_coord_state`, `_execute_coord_run`, `_main_page_processing_loop`, `_process_single_page`, `_handle_coord_failure` |
| `actions/gather/fetch.py` | API pagination + detail retrieval with rate limiting. | `_get_matches_page`, `_fetch_page_matches`, `_fetch_detail_for_candidates`, `_fetch_match_detail`, `_apply_rate_limiter_delay`, `_handle_api_rate_limit` |
| `actions/gather/checkpoint.py` | Resume metadata load/save, age validation, cleanup. | `_load_checkpoint`, `_save_checkpoint`, `_finalize_checkpoint_after_run`, `_remove_stale_checkpoint_files`, `_checkpoint_metadata_from_state` |
| `actions/gather/persistence.py` | DB lookups, diffing, bulk inserts/updates, commit summaries. | `_lookup_existing_persons`, `_identify_fetch_candidates`, `_merge_detail_payloads`, `_perform_person_upserts`, `_commit_batch_results`, `_bulk_insert_*` helpers |
| `actions/gather/metrics.py` | Performance counters, logging, telemetry exports. | `PerformanceMetrics`, `_log_progress`, `_record_batch_metrics`, `_log_final_results` |

_Notes_: The package can expose a thin `actions.gather.coord()` re-export so existing imports stay stable during Phase 3.

## Key Dependencies to Preserve
- **SessionManager**: Provides `api_manager`, `rate_limiter`, Selenium session state, database handles (`get_db_conn()`), and circuit-breaker utilities.
- **Database Models**: `Person`, `DnaMatch`, `FamilyTree`, `ConversationLog` via SQLAlchemy (module `database.py`).
- **Utilities**:
  - `utils.RateLimiter`, `_api_req_with_auth_refresh`, `_handle_rate_limit_error`.
  - `cache_manager.atomic_write_json` and `cache_manager.safe_json_load` for checkpointing.
  - `memory_utils.prevent_system_sleep` guard around long runs.
- **Config Schema**: `config_schema.api.*` (rate limiting, page limits), `config_schema.batch_size`, `config_schema.enable_action6_checkpointing`.
- **Logging & Telemetry**: `logging_config`, `prompt_telemetry` (quality gating), `performance_monitor` hooks.

## Near-Term Migration Plan (Phase 1 Deliverables)
1. **Tag Helper Regions**: Comments already mark sections (e.g., "Batch Processing Logic", "Checkpoint Handling"). During Phase 2 we can move those sections file-by-file into the new modules while leaving import shims (e.g., `from actions.gather.fetch import fetch_page_matches`).
2. **Acyclic Imports**: New modules should only depend on `SessionManager` via typing imports to avoid circular references (`if TYPE_CHECKING`). Shared constants (e.g., `_MATCH_BATCH_SIZE`) should live in `actions/gather/__init__.py` or a dedicated `constants.py` to avoid repetition.
3. **Testing Strategy**: Each module gets its own `module_tests()` (mirroring current style) focused on pure helpers. For orchestration, create a thin harness that injects fake `SessionManager` + stubbed API responses so we can verify pagination without live calls.
4. **Documentation**: Once skeleton exists, update `README`'s Action 6 section plus `docs/action_modules.md` so future contributors understand the sub-packages.

With these notes captured, we can begin Phase 2 by creating the package and relocating checkpoint + metrics helpers first (lowest risk) while keeping `coord()` untouched.
