# Phase 4 Preparation — Concurrency Evaluation and Optimization Targets

Date: 2025-08-18
Owner: Augment Agent
Scope: Evaluate if/where concurrency is necessary post-Phase 3; identify safe patterns and performance hotspots from benchmark.

## Decision Gates
- Gate 1: Is end-to-end throughput currently insufficient for target SLAs? If no, keep single-threaded and stop here.
- Gate 2: Can we gain throughput with I/O concurrency (not parallel browsers) without increasing failure modes? If yes, prefer I/O concurrency inside a single BrowserActor.
- Gate 3: If multiple browsers are required, can we isolate them via process boundaries (preferred) over threads (avoid shared state)?

## Recommended Patterns (if needed)
- Actor Model (preferred):
  - Single BrowserActor owns WebDriver and exposes a message queue (Queue/asyncio).
  - All browser ops happen on the actor thread; callers post messages and await responses.
  - Strict single-writer principle eliminates race conditions around driver.
- Supervisor Tree:
  - SessionSupervisor monitors the BrowserActor; restarts actor on fault with exponential backoff and health checks.
- Work Dispatcher (if multi-browser is justified):
  - Dispatcher assigns independent work to independent BrowserActor processes via IPC.
  - No shared WebDriver; each actor has isolated resources.

## Strawman Interfaces (sketch)

- BrowserActor public API (sync sketch with thread + Queue):
  - start(), stop(), submit(task: BrowserTask) -> Future[Result]
  - Internal: worker loop consumes tasks; only it touches driver

- BrowserTask types:
  - Navigate(url), GetCookies(names), ExecuteJS(script), Screenshot(), ExportCookies()

- SessionSupervisor:
  - start_actor(), ensure_actor_live(), restart_on_fault(), health() -> status

## Safety Invariants
- Only the actor thread touches `driver`.
- Global locks are not used; isolation via message passing prevents priority inversions.
- All messages are idempotent or have compensating actions.
- Backpressure: bounded queues and timeouts; drop or defer low-priority tasks under load.

## Benchmark Hotspots (from --benchmark)
- action11.py: 38.2s
- core/session_manager.py: 20.35s
- core/browser_manager.py: 8.49s

Initial hypotheses and low-risk optimizations:
1) action11.py (API search/research)
   - Add API-level filtering before scoring (verify already prioritized; audit for redundant passes)
   - Cache repeated kinship ladder fetches per person within the run
   - Ensure pagination limits respect .env conservative caps (MAX_PAGES=1, etc.)
2) core/session_manager.py
   - Reduce redundant auth checks when stable (increase check interval dynamically)
   - Coalesce network probes on success (exponential widen until error)
3) core/browser_manager.py
   - Avoid unnecessary nav_to_page calls during ensure_driver_live when already on base URL
   - Lazily create tabs only when needed

## Proposed Next Steps
- N1: Add a feature flag “actor_mode=false” and scaffold BrowserActor (no-op tasks route to direct calls when disabled). Keep disabled by default.
- N2: Implement minimal BrowserActor skeleton + tests (no concurrency yet, just structure) to reduce risk when/if enabled later.
- N3: Apply two safe micro-optimizations:
  - SessionManager: dynamic auth check interval after repeated successes
  - BrowserManager: skip base URL navigation if already there (cheap check)
- N4: Re-run benchmarks; if action11 remains the bottleneck, profile query path and add a per-run cache for kinship ladder.

## Risks and Mitigations
- Risk: Actor scaffolding introduces code paths not used now —
  - Mitigation: Feature-flag disabled by default; unit tests cover equivalence when disabled.
- Risk: Over-optimization of checks can hide real regressions —
  - Mitigation: Keep early-warning windows; only widen intervals after proven stability.

## Ask
- Approve creation of feature-flagged BrowserActor scaffolding and the two micro-optimizations (N3) prior to enabling any concurrency.

