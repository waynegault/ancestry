# Codebase Review Master Todo

Top improvements (refreshed Nov 21 2025, ordered from highest to lowest priority)

<!-- markdownlint-disable MD029 -->

1. (Priority 1) Improve type-safety coverage — 🚧 In Progress
   `pyrightconfig.json` has been updated to enable `reportUnknownParameterType`, `reportMissingTypeArgument`, and `reportUnknownMemberType` as warnings.

   Recommendation: Continue to incrementally enable stricter checks. Address the underlying type issues in `utils.py` and `relationship_utils.py` to resolve the remaining warnings.

2. (Priority 2) Add genuine integration/e2e tests
   `run_all_tests.py` executes 58 module-level suites, but there are no scenarios that exercise the end-to-end workflow (`exec_actn` → SessionManager → action chain). The README references `python -m test_action6_cache_integration`, yet that module does not exist.

   Recommendation: Create an integration test package that spins up an in-memory SQLite DB plus mocked API responses, runs Action 6 → 7 → 9 via the real SessionManager, and asserts database plus telemetry side effects. Wire it into `run_all_tests.py --integration`.

3. (Priority 3) Continue profiling Action 6 throughput
   Action 6 tracks rich telemetry (`PageProcessingMetrics`), yet the pipeline remains strictly sequential. Real-world runs hover around 40-60s per page under safe rate limits.

   Recommendation: Use existing metrics to identify dominant stages. Experiment with batched SQLAlchemy `bulk_save_objects`, prefetch caching, or overlapping I/O (while honoring the 0.3 RPS limiter). Set an explicit 30s/page SLO.

4. (Priority 4) Monitor `core/error_handling.api_retry` telemetry
   The migration to the unified error-handling stack is complete.

   Recommendation: Monitor telemetry for the next few production runs to ensure stability. Regenerate derived documentation (e.g., `docs/code_graph.json`) to remove references to legacy `utils.retry_api`.

<!-- markdownlint-enable MD029 -->
