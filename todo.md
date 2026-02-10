# Technical Debt & Architectural Improvements

> Last updated: July 14, 2025
> Items below require multi-file refactoring and are deferred from immediate implementation.

## Priority 1: Large File Decomposition

### 1.1 Split `utils.py` (4082 lines)
- Extract date/time utilities → `core/date_utils.py`
- Extract string formatting helpers → `core/string_utils.py`
- Extract browser/cookie utilities → `browser/cookie_utils.py`
- Extract rate limiting (if still in utils) → already in `core/rate_limiter.py`
- Target: `utils.py` < 2000 lines

### 1.2 Split `error_handling.py` (2318 lines, 39 classes)
- Extract exception hierarchy → `core/exceptions.py`
- Extract retry decorators → `core/retry.py`
- Extract `CircuitBreaker` class → already in `core/circuit_breaker.py`
- Consolidate competing `AncestryError` vs `AppError` hierarchies into one
- Target: `error_handling.py` < 800 lines

### 1.3 Split `gedcom_utils.py` (2795 lines)
- Extract GEDCOM parsing helpers → `genealogy/gedcom/gedcom_parser.py`
- Extract event info extraction → `genealogy/gedcom/gedcom_events.py`
- Extract ID validation/fixing → `genealogy/gedcom/gedcom_ids.py`
- Target: `gedcom_utils.py` < 1200 lines

### 1.4 Split `relationship_utils.py` (2218 lines)
- Extract path formatting → `research/relationship_formatting.py`
- Extract graph traversal → `research/relationship_graph.py`
- Target: `relationship_utils.py` < 1000 lines

## Priority 2: Code Consolidation

### 2.1 Replace ~25 proxy classes in `metrics_registry.py` with factory
- Current: Each metric type has a manual proxy class with identical structure
- Target: Generic `MetricProxy` factory that creates proxies dynamically
- Benefit: ~500 lines of boilerplate removed

### 2.2 Consolidate triplicate Person dataclasses
- `core/common_params.MatchIdentifiers` / `CandidatePerson` in semantic_search / `PersonInfo` in various modules
- Create one canonical `PersonSummary` and migrate callers

### 2.3 Consolidate date utility duplication
- Date parsing exists in 4+ files: `utils.py`, `action6_gather.py`, `gedcom_utils.py`, `relationship_utils.py`
- Centralize in `core/date_utils.py`

### 2.4 Consolidate ~5 overlapping cache abstractions (~4711 lines)
- `caching/cache.py`, `core/cache_backend.py`, `core/cache_registry.py`, `core/unified_cache_manager.py`, `core/caching_bootstrap.py`
- Define clear responsibilities and eliminate overlap

## Priority 3: Test Improvements

### 3.1 Replace any remaining fake/stub tests
- Check `performance_logging.py` tests for fakes (reported in review)
- Ensure all tests validate real behavior

### 3.2 Add integration tests for Action 16 (Unified Send)
- End-to-end test for priority-based message routing
- Verify one-message-per-person constraint

## Priority 4: Operational

### 4.1 CI/CD Pipeline
- Set up GitHub Actions workflow for `ruff check`, test suite, quality gate
- Add `quality_regression_gate.py` to CI/CD

### 4.2 Dead code audit
- Run periodic `vulture` or similar dead code scanner
- Remove identified dead code

### 4.3 Shadow mode validation for Action 16
- Run Action 16 in shadow mode (log-only) for 1 week
- Compare outputs with legacy Actions 8+9+11
- Verify zero regressions before full deployment
