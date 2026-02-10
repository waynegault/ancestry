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

### 2.2 ~~Consolidate triplicate Person dataclasses~~ — No change needed (Feb 2025)
- **Assessed 14 Person-like classes** across the codebase (dataclasses + TypedDicts)
- **9 of 14 have zero external importers** — purely module-internal data shapes
- **Field sets differ meaningfully**: `GedcomPerson` has family structure (parents/children/spouses), `PersonLookupResult` has 20 fields with relationship paths and source tracking, `MatchIdentifiers` is a minimal gather-time struct, `ClusterAnchor` carries triangulation-specific fields, etc.
- **Mixed type systems**: 4 are TypedDicts (`PersonInfo`, `DNAMatchInfo`, `PersonData`, `MatchData`), 10 are dataclasses — merging across type systems would be disruptive
- **Closest pair** (`CandidatePerson` ⊂ `PersonSearchResult`) is already co-located in genealogy subsystem; `CandidatePerson` is intentionally a simpler internal subset
- **Conclusion**: Each module owns its own data shape. A canonical `PersonSummary` would be either too minimal or too bloated, and existing classes would remain anyway for backward compatibility. This is working as designed.

### 2.3 ~~Consolidate date utility duplication~~ — Minimal, no change needed (Feb 2026)
- **Assessed `_format_date()` in `api/api_search_utils.py`**: takes integer day/month/year → display string ("15 Mar 1850"). Purpose is API response formatting.
- **GEDCOM `_parse_date()` in `gedcom_events.py`**: parses freeform date strings → `datetime`. Purpose is GEDCOM data ingestion.
- **`api/api_utils.py` and `api/api_search_core.py`** already delegate date parsing via lazy imports to the GEDCOM module — no duplication there.
- **Conclusion**: API display formatting ≠ GEDCOM date parsing. Functions serve different domains with different signatures and return types. No `core/date_utils.py` needed.

### 2.4 ~~Consolidate ~5 overlapping cache abstractions~~ — Assessed, dead code tagged (Feb 2026)
- **`caching/cache.py`** (1748 lines): `IntelligentCacheWarmer` and `CacheDependencyTracker` have **zero external importers** — tagged with `# TODO: candidate for removal` comments. Removing them would save ~260 lines.
- **`core/caching_bootstrap.py`** (139 lines): Only imported by `main.py` (`ensure_caching_initialized`). Small, self-contained with own tests. Merging into `cache_registry.py` would mix initialization concerns with registry concerns — kept separate.
- **`core/cache_backend.py`** — Protocol/interface definition (keep as-is)
- **`core/unified_cache_manager.py`** — Active replacement layer (keep as-is)
- **`core/cache_registry.py`** — Coordinator (keep as-is)
- **Remaining action**: Remove `IntelligentCacheWarmer` + `CacheDependencyTracker` + their global instances/getters when convenient (~260 lines savings)

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
- [x] Shadow mode infrastructure added to `UnifiedSendProcessor` (Action 16)
- [x] Config settings: `SHADOW_MODE_ENABLED` (legacy comparisons), `ACTION16_SHADOW_MODE` (log-only sends)
- [x] Shadow log output: `Logs/action16_shadow_decisions.jsonl` (JSON, comparison-ready)
- [ ] Run Action 16 in shadow mode (`ACTION16_SHADOW_MODE=true`) for 1 week
- [ ] Compare outputs with legacy Actions 8+9+11 (`python -m messaging.shadow_mode_analyzer --report`)
- [ ] Verify zero regressions before full deployment
