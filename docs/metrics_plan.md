# Sprint 2B Metrics & Observability Plan

> Design document for introducing structured metrics collection and visualization without violating existing rate limiting or session orchestration invariants.

## Objectives
- Establish a unified metrics schema that captures latency, throughput, cache efficacy, and session health across the automation platform.
- Provide a Prometheus-compatible exporter that surfaces real-time metrics without adding external dependencies for default runs.
- Supply developers with a reproducible Grafana dashboard showcasing key indicators (API latency distributions, cache hit rate, action throughput, circuit breaker activity).
- Preserve the single `SessionManager` orchestration model and the sequential rate limiter guarantees while emitting observability data.

## Scope
- **In scope**: Core managers (`core/session_manager.py`, `core/api_manager.py`, `core/unified_cache_manager.py`), cache/reporting helpers (`performance_monitor.py`), and action throughput counters for Actions 6-9.
- **Out of scope** (for this sprint): External alerting integrations, persistent time-series storage, production deployment automation.

## Metrics Taxonomy
| Category | Metric | Cardinality Controls | Notes |
| --- | --- | --- | --- |
| API | `ancestry_api_latency_seconds` (histogram) | Labels: `endpoint`, `status_family` | Timed around `_api_req_with_auth_refresh` wrapper. Bucket strategy: `[0.5, 1, 2, 4, 8, 16]` seconds. |
| API | `ancestry_api_requests_total` (counter) | Labels: `endpoint`, `method`, `result` | Increment before/after request; `result` values: `success`, `retry`, `failure`, `cached`. |
| Cache | `ancestry_cache_hit_ratio` (gauge) | Labels: `service`, `endpoint` | Updated on `UnifiedCacheManager.get/set`; emits ratio per update cycle. |
| Cache | `ancestry_cache_operations_total` (counter) | Labels: `service`, `endpoint`, `operation` | Operations: `get`, `set`, `invalidate`, `expire`. |
| Session | `ancestry_session_uptime_seconds` (gauge) | No labels | Derived from `SessionManager.session_age_seconds()`. |
| Session | `ancestry_session_refresh_total` (counter) | Labels: `reason` | Reasons: `proactive`, `api_forced`, `browser_error`. |
| Throughput | `ancestry_action_processed_total` (counter) | Labels: `action`, `result` | Hooked into action completion callbacks; `result`: `success`, `failure`, `skipped`. |
| Circuit Breaker | `ancestry_circuit_breaker_state` (gauge) | Labels: `breaker` | Values: `0` (closed), `1` (open), `0.5` (half-open). |
| Circuit Breaker | `ancestry_circuit_breaker_trip_total` (counter) | Labels: `breaker` | Incremented on trip events. |
| Rate Limiter | `ancestry_rate_limiter_delay_seconds` (histogram) | No labels | Captures actual wait durations applied before API calls. |
| System | `ancestry_worker_thread_count` (gauge) | No labels | Reuse data from `performance_monitor.py`; provides parity with existing monitoring. |

## Registry Architecture
1. **Metrics Registry Module**: Create `observability/metrics_registry.py` exporting a singleton registry with helper functions:
   - `get_registry()`: returns the Prometheus registry instance.
   - `metrics()` namespace providing typed helpers for each metric (e.g., `metrics.api_latency.observe(endpoint, status_family, seconds)`).
   - Gate registry creation on `PROMETHEUS_METRICS_ENABLED` config flag (default: `false`).
2. **Config Integration**: Extend `config/config_schema.py` with `ObservabilitySettings` dataclass containing:
   - `enable_prometheus_metrics: bool = False`
   - `metrics_export_port: int = 9000`
   - `metrics_namespace: str = "ancestry"`
3. **Exporter**: Add `observability/metrics_exporter.py` implementing a lightweight HTTP server (using `prometheus_client.start_http_server`) launched by `SessionManager.ensure_session_ready()` when metrics are enabled.

## Instrumentation Plan
- **core/api_manager.py**
  - Wrap `_api_req_with_auth_refresh` to record latency histograms and request counters.
  - Record rate limiter wait durations via hooks in the shared `RateLimiter.wait()` method (requires non-invasive callback registration).
- **core/session_manager.py**
  - Emit session uptime gauge during periodic health checks (`_update_session_metrics` helper).
  - Increment session refresh counters inside `refresh_browser_cookies` and other refresh pathways.
  - Publish circuit breaker gauges via `SessionCircuitBreaker` callbacks.
- **core/unified_cache_manager.py**
  - After every cache get/set/invalidate, update counters and recompute per-endpoint hit ratio.
  - Guard computations with lightweight locks to avoid contention.
- **performance_monitor.py**
  - Reuse existing system metrics (thread count, memory); provide adapter that feeds Prometheus gauges when enabled.
- **Action Modules (6-9)**
  - Inject action-level counters at orchestrator boundaries (e.g., `action6_gather.coord`, `action7_inbox.process_inbox`).
  - Avoid per-record labels; aggregate by action and result status only.

## Data Flow Overview
```
Action / Manager ──▶ metrics_registry helpers ──▶ Prometheus registry ──▶ HTTP exporter ──▶ (optional) Grafana dashboards
```

## Rate Limiting Safeguards
- Metrics emissions must not introduce additional API calls or blocking operations.
- Prometheus exporter runs on a background thread with daemon flag to avoid blocking shutdown.
- Cache hit ratio calculations reuse in-memory statistics; no extra database reads.
- Guard instrumentation with feature flag checks to avoid overhead when disabled.

## Deliverables
1. `observability/metrics_registry.py` + tests covering singleton behavior and metric helper registration.
2. `observability/metrics_exporter.py` + smoke tests verifying exporter lifecycle.
3. Config updates with validation in `config_manager.py` and default `.env.example` entries.
4. Hook updates across targeted modules with unit tests (mock registry assertions) and documentation in README + `docs/monitoring.md` (new).
5. Example Grafana dashboard JSON in `docs/grafana/ancestry_overview.json` with panels for the primary metrics.

## Testing Strategy
- Unit tests for registry helpers ensuring metric names/labels match the schema.
- Integration tests that enable metrics in a controlled environment and assert that the exporter exposes expected sample values (using `requests.get` against localhost during tests with short timeouts).
- Update `run_all_tests.py` to skip exporter tests unless `SKIP_LIVE_API_TESTS` is `false` or metrics flag is enabled.

## Documentation Checklist
- Update `README.md` Observability section with setup instructions.
- Add `docs/monitoring.md` covering metric semantics, exporter usage, and Grafana setup.
- Extend `docs/review_todo.md` Phase 6 tasks to include metrics validation once implemented.

## Open Questions
1. Should cache hit ratio gauges be derived metrics (computed periodically) or directly maintained on each operation? (_Current plan: compute on operation for freshness_.)
2. Do we expose per-endpoint histograms for API latency, or aggregate across categories to limit cardinality? (_Plan: per-endpoint with sanitized labels; revisit if Prometheus load increases._)
3. How do we reconcile `performance_monitor.py` existing alerting with Prometheus alerts? (_Decision deferred; keep both until Prometheus adoption proves stable._)
