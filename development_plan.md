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

## Reference Checklist
- Keep `.env` aligned with `config/config_schema.py` defaults (notably `REQUESTS_PER_SECOND=0.3`, `MAX_PAGES`, `MATCHES_PER_PAGE`).
- Use `exec_actn()` for every action entry point; never instantiate sessions, browsers, or API clients directly.
- Maintain uppercase UUID handling for all DNA kit lookups.
- Run `python run_all_tests.py` before large refactors; individual modules can be validated with `python -m <module>`.
- When adjusting prompts, pair the change with `python prompt_telemetry.py --stats` and refresh the quality baseline if median scores improve sustainably.
