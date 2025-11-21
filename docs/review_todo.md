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

3. (Priority 3) Continue profiling Action 6 throughput
   Action 6 tracks rich telemetry (`PageProcessingMetrics`), yet the pipeline remains strictly sequential. Real-world runs hover around 40-60s per page under safe rate limits.

   Recommendation: Use existing metrics to identify dominant stages. Experiment with batched SQLAlchemy `bulk_save_objects`, prefetch caching, or overlapping I/O (while honoring the 0.3 RPS limiter). Set an explicit 30s/page SLO.

4. (Priority 4) Monitor `core/error_handling.api_retry` telemetry
   The migration to the unified error-handling stack is complete.

   Recommendation: Monitor telemetry for the next few production runs to ensure stability. Regenerate derived documentation (e.g., `docs/code_graph.json`) to remove references to legacy `utils.retry_api`.

<!-- markdownlint-enable MD029 -->
