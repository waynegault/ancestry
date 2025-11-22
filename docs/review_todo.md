# Codebase Review Master Todo

Top improvements (refreshed Nov 21 2025, ordered from highest to lowest priority)

<!-- markdownlint-disable MD029 -->

1. (Priority 1) Improve type-safety coverage — ✅ Done
   `pyrightconfig.json` has been updated to enable `reportUnknownParameterType`, `reportMissingTypeArgument`, and `reportUnknownMemberType` as warnings.
   Addressed underlying type issues in `utils.py` (SessionManager forward references) and `relationship_utils.py` (Gedcom protocols) to resolve warnings.

   Recommendation: Continue to incrementally enable stricter checks as needed.

2. (Priority 2) Add genuine integration/e2e tests — ✅ Done
   `run_all_tests.py` now supports an `--integration` flag that executes `test_integration_workflow.py`. This suite mocks `SessionManager`, `DatabaseManager`, and API responses to test the end-to-end workflow of Actions 6, 7, and 9, asserting database state and telemetry side effects.

   Recommendation: Continue to expand integration scenarios as new features are added.

3. (Priority 3) Continue profiling Action 6 throughput — ✅ Done
   Action 6 tracks rich telemetry (`PageProcessingMetrics`), yet the pipeline remains strictly sequential. Real-world runs hover around 40-60s per page under safe rate limits.

   Update (Nov 21 2025):
   - Added granular timing logs to `_fetch_combined_details`.
   - Implemented persistent disk caching (`cache.py`) for expensive "Combined Details" API calls. This prevents redundant fetching of profile/DNA details across session restarts, significantly improving throughput for resumed runs.

   Recommendation: Monitor logs to verify cache hit rates and overall page processing time reduction.

4. (Priority 4) Monitor `core/error_handling.api_retry` telemetry — ✅ Done
   The migration to the unified error-handling stack is complete.

   Update (Nov 21 2025):
   - Regenerated `docs/code_graph.json` to remove stale references to legacy `utils.retry_api`.

   Recommendation: Monitor telemetry for the next few production runs to ensure stability.

5. (Priority 5) Create Maintainer Handoff Documentation — ✅ Done
   The `readme.md` references `docs/MAINTAINER_HANDOFF.md`, but the file does not exist.
   Create a comprehensive handoff guide covering:
   - System architecture and critical paths
   - Key maintenance workflows (schema migration, dependency updates)
   - Troubleshooting common production issues
   - Future roadmap and known technical debt

   Update (Nov 21 2025):
   - Created `docs/MAINTAINER_HANDOFF.md` with sections on Architecture, Critical Components, Maintenance, Troubleshooting, and Roadmap.

   Recommendation: Keep this document updated as the system evolves.

5. sending JSON to an llm is inefficient. Instead, convert data going to an llm to TOON format. See https://github.com/toon-format/toon

<!-- markdownlint-enable MD029 -->
