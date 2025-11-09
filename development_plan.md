# Development Plan

## Context
- Action 6 throughput tuning remains the biggest lever for large batch DNA match harvesting.
- The current architecture enforces sequential processing with a single shared `RateLimiter` (0.3 requests/second) to guarantee zero 429 errors.
- Phase 3.3 testing highlighted older pain points (hardcoded `itemsPerPage`, hidden 429s, session drops). Most foundational fixes are now checked in, but verification and monitoring still matter.
- Action 9 recently gained AI skip heuristics, memoization, and tighter context windows; focus now shifts back to gathering performance, end-to-end dialogue quality, and operational telemetry.

## Recently Completed Foundations
- `itemsPerPage` now respects `config_schema.api.matches_per_page`; default set to 30 with config override.
- System sleep prevention is hooked into `main.py` so long DNA runs keep the machine awake.
- `RateLimiter` in `utils.py` is thread-safe, shared via `SessionManager`, and drives **strictly sequential** API access.
- AI messaging pipeline (Action 9) now bypasses low-value messages, trims context to 60 words, and enforces a 1,800 token cap for local models.

## Immediate Priorities (P0)
1. **Rate Limiter Instrumentation**
   - Audit remaining direct HTTP calls (`requests.Session.get`, CloudScraper fallbacks) and route them through `session_manager.api_manager` so every request is rate limited and logged.
   - Expand logging around `RateLimiter` metrics to surface the effective delay, 429 recoveries, and adaptive backoff decisions after each page processed.

2. **Session Health Safeguards**
   - Review `HEALTH_CHECK_INTERVAL_PAGES` and proactive refresh timing to confirm we consistently refresh before the 40-minute session limit when processing large batches.
   - Add regression tests that simulate forced session expiry and ensure the circuit breaker short-circuits remaining work after five consecutive failures.

3. **Validation Loop**
   - Run `validate_rate_limiting.py` and a 5-page Action 6 smoke test whenever the rate limiting logic is touched.
   - Document the expected throughput and acceptable variance ranges in `Logs/performance` summaries so future runs can be compared quickly.

## Near-Term Objectives (P1)
1. **Dialogue Engine Glue (Action 9 ↔ Action 10)**
   - Pull Action 10 person lookups into Action 9 reply generation so responses include contextual records by default.
   - Persist conversation state (entities mentioned, pending questions) to drive richer follow-up tasks.

2. **Adaptive Follow-Up Scheduling**
   - Implement engagement-based timing that downgrades or accelerates follow-ups using the conversation score and status transitions (OUT_OF_TREE → IN_TREE → DESIST).

3. **Telemetry & Quality Monitoring**
   - Automate weekly rollups of `Logs/prompt_experiments.jsonl` and `Logs/prompt_experiment_alerts.jsonl`.
   - Wire `quality_regression_gate.py` into CI so prompt regressions block deployment by default.

## Longer-Term Roadmap (P2+)
- Conversation analytics dashboard with funnels, engagement trends, and AI quality metrics.
- A/B harness for prompt variants and enrichment strategies, including automatic DeepSeek fallback scoring.
- Inline relationship diagrams using lightweight SVG rendering to embed in outbound messages.
- Research assistant upgrades: record suggestions, citation synthesis, and Microsoft To-Do task prioritization improvements.


Other considerations:

## Comprehensive Test Analysis & Development Plan

After thoroughly analyzing the codebase, I've completed a comprehensive review of the test framework, utilities, and specific module tests. Here are my findings and recommendations:

### 🔍 __Current Test State Analysis__

__Test Quality Assessment:__

- __Overall Statistics__: 87 modules analyzed, 32 with tests, 241 total test functions
- __Quality Issues Found__: 0 smoke tests, 0 tests with no assertions, 0 tests always returning True
- __Test Framework__: Well-structured with `TestSuite` class providing standardized testing
- __Test Discovery__: Automated discovery via `run_comprehensive_tests()` function detection

__Linting Status:__

- __Critical Errors__: 0 (All blocking rules E722,F821,F811,F823 passed)
- __Minor Issues__: 11 non-blocking issues found (PLR0911, UP045, RUF059, etc.)
- __Fixable Issues__: 1 issue can be auto-fixed with `--fix` option

### 🎯 __Key Findings__

#### __Strengths Identified:__

1. __Comprehensive Test Framework__: Modular `TestSuite` with standardized reporting
2. __Performance Monitoring__: Built-in performance metrics and optimization suggestions
3. __Error Handling__: Robust error handling with circuit breaker patterns
4. __Session Management__: Advanced session caching and health monitoring
5. __Rate Limiting__: Sophisticated adaptive rate limiting system

#### __Areas for Improvement:__

__1. Test Duplication Opportunities:__

- Multiple modules have similar test patterns for basic functionality
- Common test utilities could be consolidated into shared helpers
- Authentication test patterns repeated across action modules

__2. Authentication Requirements:__

- Tests properly handle live authenticated sessions via `SKIP_LIVE_API_TESTS` environment variable
- Session management tests include comprehensive authentication scenarios
- CSRF token caching optimization prevents authentication-related performance issues

__3. Test Pattern Consolidation:__

- Standardized test structure across all modules using `TestSuite` framework
- Consistent error handling and logging patterns
- Performance testing integrated into test suite

### 📋 __Development Tasks Identified__

#### __High Priority (Quality Issues):__

1. __Fix Minor Linting Issues__: Address 11 non-blocking linting warnings
2. __Consolidate Test Utilities__: Create shared test helper functions
3. __Optimize Test Performance__: Reduce test execution time through parallelization

#### __Medium Priority (Enhancements):__

1. __Enhance Authentication Testing__: Add more comprehensive session validation tests
2. __Improve Error Reporting__: Better test failure diagnostics and reporting
3. __Add Integration Tests__: Cross-module integration testing

#### __Low Priority (Maintenance):__

1. __Documentation Updates__: Improve test documentation and examples
2. __Test Data Management__: Standardize test data creation and cleanup
3. __Monitoring Integration__: Enhanced test result monitoring

### 🚀 __Recommended Next Steps__

1. __Immediate Actions__:

   - Fix the 1 auto-fixable linting issue
   - Address the 11 minor linting warnings
   - Consolidate duplicate test patterns

2. __Quality Improvements__:

   - Enhance test coverage for edge cases
   - Improve test execution performance
   - Add more comprehensive authentication testing

3. __Long-term Enhancements__:

   - Implement cross-module integration tests
   - Add performance regression testing
   - Enhance test result analytics

### 📊 __Quality Metrics__

- __Test Coverage__: High (32/87 modules have comprehensive tests)
- __Test Quality__: Excellent (0 smoke tests, all tests have proper assertions)
- __Performance__: Good (parallel execution available with `--fast` flag)
- __Maintainability__: Good (standardized test framework)

The codebase demonstrates excellent test quality with a sophisticated testing framework. The main opportunities lie in consolidating duplicate patterns and addressing minor linting issues rather than fundamental test quality problems.


## Reference Checklist
- Keep `.env` aligned with `config/config_schema.py` defaults (notably `REQUESTS_PER_SECOND=0.3`, `MAX_PAGES`, `MATCHES_PER_PAGE`).
- Use `exec_actn()` for every action entry point; never instantiate sessions, browsers, or API clients directly.
- Maintain uppercase UUID handling for all DNA kit lookups.
- Run `python run_all_tests.py` before large refactors; individual modules can be validated with `python -m <module>`.
- When adjusting prompts, pair the change with `python prompt_telemetry.py --stats` and refresh the quality baseline if median scores improve sustainably.
