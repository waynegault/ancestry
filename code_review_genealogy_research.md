# Code Review: `genealogy/` and `research/` Packages

**Date**: 2025-01-28
**Scope**: 34 files across `genealogy/` (20 files) and `research/` (14 files)
**Total LOC**: ~24,500 lines

---

## EXECUTIVE SUMMARY

Overall code quality is **solid** — well-structured dataclasses, meaningful tests in most modules, and good separation of concerns. However, there are **systemic patterns** that inflate code volume by an estimated 15-20% and create maintenance drag:

| Category | Count | Severity |
|---|---|---|
| Duplicate dataclass definitions | 3 triplicate sets | HIGH |
| Fake/no-op test functions | 20+ across codebase | HIGH |
| Bloated module docstrings | 4 files | MEDIUM |
| `Optional` imported but unused (PEP 604 style used) | 13 files | LOW |
| Identical `__init__.py` boilerplate | 4 files in scope | LOW |
| Duplicate confidence-level mappings | 3+ locations | MEDIUM |
| Stub/placeholder implementations | 5 methods in gedcom_intelligence | MEDIUM |
| Overlapping conflict detection logic | 2 modules | HIGH |

---

## CROSS-CUTTING ISSUES (Systemic)

### 1. TRIPLICATE Person-Result Dataclasses — HIGH

Three nearly identical dataclasses exist for representing a person search result:

| Dataclass | File | Key Fields |
|---|---|---|
| `PersonSearchResult` | `genealogy/tree_query_service.py:21` | person_id, name, birth_year, birth_place, death_year, death_place, confidence, match_score, alternatives |
| `PersonLookupResult` | `research/person_lookup_utils.py:27` | person_id, name, first_name, last_name, birth_year, birth_place, death_year, death_place, confidence, match_score, source |
| `CandidatePerson` | `genealogy/semantic_search.py:69` | person_id, name, birth_year, birth_place, death_year, match_score |

**Recommendation**: Create a single `PersonResult` base dataclass in a shared module (e.g., `core/type_definitions.py` or a new `genealogy/models.py`). Subclass only where genuinely different fields are needed. `CandidatePerson` is essentially a subset and can be replaced entirely.

### 2. Fake `_test_module_integrity()` Functions — HIGH

**20+ files** across the codebase contain this pattern:

```python
def _test_module_integrity() -> bool:
    "Test that module can be imported and definitions are valid."
    return True
```

**In scope**: `genealogy/__init__.py`, `genealogy/dna/__init__.py`, `genealogy/gedcom/__init__.py`, `genealogy/triangulation.py`, `genealogy/research_service.py`, `research/__init__.py`

These provide **zero test coverage** — they always pass regardless of module state. They exist only to satisfy the `run_all_tests.py` test runner requirement.

**Recommendation**: Either replace with actual import-and-validate tests (check that key classes/functions are importable and have correct signatures) or remove them and have `run_all_tests.py` skip modules without real `module_tests()` functions.

### 3. `Optional` Imported but `X | None` Syntax Used — LOW

13 files in `genealogy/` import `Optional` from `typing` but use PEP 604 `X | None` syntax throughout:

- `universal_scoring.py`, `triangulation.py`, `tree_stats_utils.py`, `tree_query_service.py`, `semantic_search.py`, `research_service.py`, `gedcom_search_utils.py`, `gedcom_intelligence.py`, `gedcom_cache.py`, `fact_validator.py`, `dna_utils.py`, `dna_gedcom_crossref.py`, `dna_ethnicity_utils.py`

**Recommendation**: Remove unused `Optional` imports. Run `ruff check --fix .` — rule `F401` will catch these.

### 4. Identical `__init__.py` Boilerplate — LOW

Four `__init__.py` files use identical lazy-import + fake-test boilerplate (~47 lines each):
- `genealogy/__init__.py`
- `genealogy/dna/__init__.py`
- `genealogy/gedcom/__init__.py`
- `research/__init__.py`

**Recommendation**: Extract a helper function `create_lazy_package(__name__, submodules)` into `testing/test_utilities.py` to DRY this up. The `_test_module_integrity` can be dropped (see issue #2).

### 5. Bloated Marketing-Style Module Docstrings — MEDIUM

Four files have 35-45 line docstrings that read like promotional copy rather than developer documentation:

| File | Docstring Lines | Example Excerpt |
|---|---|---|
| `gedcom/gedcom_utils.py` | ~40 | "Provides a comprehensive suite of..." |
| `gedcom/gedcom_search_utils.py` | ~40 | "enterprise-grade GEDCOM search..." |
| `gedcom/gedcom_cache.py` | ~45 | "sophisticated hierarchical caching..." |
| `dna/dna_gedcom_crossref.py` | ~45 | "Advanced DNA-GEDCOM Integration..." |

**Recommendation**: Trim to 5-10 lines covering: (1) what the module does, (2) key classes/functions, (3) usage example if warranted. Delete the "Features:", "Architecture:", "Integration:" sections.

### 6. Duplicate Conflict Detection — HIGH

Two separate modules handle data conflict detection with overlapping scope:

| Module | Class | Purpose |
|---|---|---|
| `research/conflict_detector.py` | `ConflictDetector` | Field-level comparison, DB conflict records, resolution workflow |
| `genealogy/fact_validator.py` | `FactValidator` | Fact extraction, validation pipeline, conflict types |

Both implement:
- String similarity comparison (`SequenceMatcher`)
- Date comparison with tolerance
- Severity/priority classification
- Database persistence for conflicts

**Recommendation**: Consolidate into a single conflict detection service. `FactValidator` focuses on extraction-to-validation pipeline; `ConflictDetector` focuses on DB record comparison. They should share a common comparison engine rather than reimplementing `SequenceMatcher`-based comparison independently.

---

## FILE-BY-FILE REVIEW

### `genealogy/__init__.py` (47 lines)

| Aspect | Assessment |
|---|---|
| Duplication | Identical boilerplate with dna/__init__.py, gedcom/__init__.py |
| Tests | Fake `_test_module_integrity()` → always True |
| Type issues | None |
| Action | Remove fake test; extract lazy-import helper |

---

### `genealogy/universal_scoring.py` (600 lines)

| Aspect | Assessment |
|---|---|
| Duplication | `_get_confidence_level()` (L270-290) duplicates confidence mapping found in `triangulation.py` and `person_lookup_utils.py` |
| Tests | **GOOD** — 15 embedded test functions with real assertions |
| Complexity | `apply_universal_scoring()` is well-decomposed |
| Type issues | `Optional` imported but `X \| None` used throughout |
| Action | Unify confidence-level mapping into shared utility; remove `Optional` import |

---

### `genealogy/triangulation.py` (130 lines)

| Aspect | Assessment |
|---|---|
| Duplication | Confidence mapping (`min_cm` → confidence level, L90-100) duplicates `universal_scoring._get_confidence_level()` |
| Tests | Fake `_test_module_integrity()` — but `test_triangulation.py` provides real tests |
| Type issues | `Optional` imported but never used |
| Complexity | Low, appropriate |
| Action | Use shared confidence mapping; remove `Optional` import; remove fake test |

---

### `genealogy/tree_stats_utils.py` (1054 lines)

| Aspect | Assessment |
|---|---|
| Duplication | `excluded_cols` set defined twice: once in `_calculate_ethnicity_distribution()` (~L350) and again in `_get_ethnicity_columns()` (~L480) |
| Tests | **GOOD** — 11 real tests with DB integration; `_test_statistics_functions_available` uses `inspect.signature` validation |
| Complexity | File is large but well-decomposed into focused helper functions |
| Type issues | `Optional` imported but unused |
| Action | Extract `excluded_cols` to module-level constant; remove `Optional` import |

---

### `genealogy/tree_query_service.py` (981 lines)

| Aspect | Assessment |
|---|---|
| Duplication | `PersonSearchResult` dataclass (L21) largely duplicates `PersonLookupResult` in research/ |
| Tests | **GOOD** — 9 tests covering dataclasses, ordinal generation, prompt string formatting |
| Complexity | `get_full_name` import inside `_get_family_member_details` (repeated 4x) — should be at class init |
| Type issues | `Optional` imported but unused; repeated `from genealogy.gedcom import gedcom_utils` inside methods |
| Action | Consolidate person result dataclasses; hoist repeated imports to class-level; remove `Optional` |

**Specific issue**: `gedcom_utils` is imported inside `get_person_details()`, `get_family_members()`, `_get_family_member_details()`, `get_common_ancestors()`, `_collect_parents()` etc. This should be done once at class initialization or as a lazy property.

---

### `genealogy/semantic_search.py` (755 lines)

| Aspect | Assessment |
|---|---|
| Duplication | `CandidatePerson` dataclass (L69) is essentially a subset of `PersonSearchResult` |
| Tests | **EXCELLENT** — 5 thorough tests including stub injection, ambiguity testing, prompt string formatting |
| Complexity | Well-structured with clear intent classification flow |
| Type issues | `Optional` imported but unused |
| Action | Replace `CandidatePerson` with shared person result type; remove `Optional` |

---

### `genealogy/research_service.py` (360 lines)

| Aspect | Assessment |
|---|---|
| Duplication | `_calculate_match_score_cached()` wraps `calculate_match_score` from gedcom_utils — same caching pattern exists in `gedcom_search_utils._calculate_cached_score()` |
| Tests | Fake `_test_module_integrity()` — but `test_research_service.py` provides real mock-based tests |
| Complexity | Appropriate |
| Type issues | `Optional` imported but unused |
| Action | Remove fake test; consolidate score caching; remove `Optional` |

---

### `genealogy/relationship_calculations.py` (332 lines)

| Aspect | Assessment |
|---|---|
| Duplication | None significant |
| Tests | **EXCELLENT** — Real tests with family tree fixture, thorough coverage of all relationship functions |
| Complexity | Pure functions, well-designed |
| Type issues | Clean |
| Action | **No changes needed** — this is a model module |

---

### `genealogy/genealogy_presenter.py` (358 lines)

| Aspect | Assessment |
|---|---|
| Duplication | `display_family_members()` re-exported in `research/search_criteria_utils.py` via protocol pattern |
| Tests | **GOOD** — Real tests with assertion validation |
| Complexity | Uses `print()` directly — consider returning strings for testability |
| Type issues | Clean |
| Action | Minor: consider returning formatted strings instead of printing directly |

---

### `genealogy/genealogical_normalization.py` (698 lines)

| Aspect | Assessment |
|---|---|
| Duplication | US state abbreviation dict (~L400-450) — could be a shared constant |
| Tests | Tests exist but are **shallow** — verify structure correctness but don't validate normalization accuracy deeply |
| Complexity | `_validate_location()` has many branches but each is simple |
| Type issues | Clean (uses `X \| None` properly) |
| Action | Deepen test assertions; extract US states to shared constants module |

---

### `genealogy/fact_validator.py` (1192 lines)

| Aspect | Assessment |
|---|---|
| Duplication | Overlaps significantly with `research/conflict_detector.py` — both implement SequenceMatcher comparison, date tolerance, severity levels |
| Tests | **EXCELLENT** — 13+ real tests covering fact creation, conflict detection, date normalization, factory methods, DB integration |
| Complexity | Large but well-structured with clear pipeline: extract → validate → save |
| Type issues | `Optional` imported but unused |
| Specific issue | `_get_suggested_facts_from_db()` has unreachable `return existing` after the `except` block (L507) |
| Action | Fix unreachable code; consolidate comparison logic with conflict_detector; remove `Optional` |

---

### `genealogy/dna/dna_utils.py` (643 lines)

| Aspect | Assessment |
|---|---|
| Tests | **GOOD** — 7 real tests covering CSRF extraction, URL construction, header building, cache keys |
| Complexity | Well-decomposed with clear helper functions for cache, API, cookie sync |
| Type issues | `Optional` imported but unused |
| Action | Remove `Optional` import |

---

### `genealogy/dna/dna_ethnicity_utils.py` (848 lines)

| Aspect | Assessment |
|---|---|
| Tests | Need deeper review (only first 200 lines read) |
| Type issues | `Optional` imported but unused |
| Action | Remove `Optional` import |

---

### `genealogy/dna/dna_gedcom_crossref.py` (807 lines)

| Aspect | Assessment |
|---|---|
| Duplication | **BLOATED DOCSTRING** — ~45 lines of marketing-style text |
| Tests | Need deeper review |
| Type issues | `Optional` imported but unused |
| Action | Trim docstring to 5-10 lines; remove `Optional` |

---

### `genealogy/dna/__init__.py` (47 lines)

| Aspect | Assessment |
|---|---|
| Duplication | Identical boilerplate with genealogy/__init__.py |
| Tests | Fake `_test_module_integrity()` |
| Action | Extract lazy-import helper; remove fake test |

---

### `genealogy/gedcom/gedcom_utils.py` (2795 lines)

| Aspect | Assessment |
|---|---|
| Duplication | **BLOATED DOCSTRING** (~40 lines); `get_full_name()` has 4 extraction strategies — well-factored but verbose |
| Complexity | **VERY HIGH** at 2795 lines — this is the largest file in scope. Date parsing alone spans ~200 lines. |
| Tests | Need to check test section (beyond read range) |
| Type issues | Uses `Optional` import |
| Action | **Split this file**: Extract date parsing to `gedcom_date_utils.py`, name handling to `gedcom_name_utils.py`, and scoring to existing `universal_scoring.py`. Target: no file >800 lines. Trim docstring. |

---

### `genealogy/gedcom/gedcom_search_utils.py` (1374 lines)

| Aspect | Assessment |
|---|---|
| Duplication | **BLOATED DOCSTRING** (~40 lines); `_GedcomDataCache` duplicates caching in `gedcom_cache.py` — two competing caching systems |
| Complexity | Score caching in `_calculate_cached_score()` duplicates `research_service._calculate_match_score_cached()` |
| Tests | Need deeper review |
| Type issues | `Optional` imported but unused |
| Action | Consolidate caching with `gedcom_cache.py`; trim docstring; remove `Optional` |

**Critical**: `_GedcomDataCache` (simple class-level cache) and `gedcom_cache.py` (multi-level memory+disk cache via `GedcomCacheModule`) both cache GEDCOM data. `gedcom_search_utils.get_gedcom_data()` even imports from `gedcom_cache.py` as a fallback. This dual-caching creates confusion about which cache is authoritative.

---

### `genealogy/gedcom/gedcom_cache.py` (1198 lines)

| Aspect | Assessment |
|---|---|
| Duplication | **BLOATED DOCSTRING** (~45 lines); memory cache management (`_MEMORY_CACHE` dict) partially overlaps with `_GedcomDataCache` in search_utils |
| Complexity | Multi-level caching (memory → disk → file) is well-structured but creates complexity when combined with the simpler cache in search_utils |
| Tests | Need deeper review |
| Type issues | `Optional` imported but unused |
| Specific issue | `load_gedcom_with_aggressive_caching()` has a misplaced docstring fragment (L453-455: "if isinstance..." appears mid-docstring) |
| Action | Fix misplaced code in docstring; consolidate with search_utils caching; trim docstring; remove `Optional` |

**Bug**: Around line 453, there's a code fragment embedded inside the docstring of `load_gedcom_with_aggressive_caching()`:
```python
        if isinstance(disk_cached, dict) and "id_to_spouses" not in disk_cached:
                logger.debug("Disk GEDCOM cache missing spouse map; refreshing cached entry with rebuilt data.")
                _store_gedcom_in_disk_cache(gedcom_data, disk_cache_key)
        GedcomData instance or None if loading fails
```
This looks like code that was accidentally moved into the docstring during refactoring.

---

### `genealogy/gedcom/gedcom_intelligence.py` (950 lines)

| Aspect | Assessment |
|---|---|
| Duplication | None significant |
| Tests | **EXCELLENT** — 10 real tests including edge cases (circular relationships, Unicode names, large datasets, invalid input) |
| Complexity | Reasonable for the scope |
| Type issues | `Optional` imported but unused |
| **Stub methods** | `_extract_birth_year()`, `_extract_death_year()`, `_extract_birth_place()`, `_extract_death_place()` are **all stubs returning None**. This means `_analyze_date_consistency()`, `_analyze_location_patterns()`, and `_analyze_relationship_conflicts()` **never detect anything**. |
| Action | **Implement the extract methods** using `gedcom_utils.get_event_info()` — this is the most impactful fix in the entire review. Remove `Optional` import. |

**Critical**: The entire GEDCOM intelligence analysis pipeline is non-functional because the extraction methods are stubs. The `_analyze_date_consistency()` method checks birth/death years, but `_extract_birth_year()` always returns `None`, so no conflicts are ever found. The test `test_gap_detection_with_mocked_birth_year` works around this by subclassing with overridden extractors, which proves the framework works but hides the production gap.

---

### `genealogy/gedcom/__init__.py` (47 lines)

| Aspect | Assessment |
|---|---|
| Duplication | Identical boilerplate |
| Tests | Fake `_test_module_integrity()` |
| Action | Extract lazy-import helper; remove fake test |

---

### `genealogy/test_triangulation.py` (67 lines)

| Aspect | Assessment |
|---|---|
| Tests | **GOOD** — Real mock-based tests validating parameter passing and filtering logic |
| Action | **No changes needed** |

---

### `genealogy/test_research_service.py` (100 lines)

| Aspect | Assessment |
|---|---|
| Tests | **GOOD** — Real mock-based tests covering load success/failure, search scenarios |
| Action | **No changes needed** |

---

### `research/__init__.py` (55 lines)

| Aspect | Assessment |
|---|---|
| Duplication | Identical boilerplate with genealogy __init__.py files |
| Tests | Fake `_test_module_integrity()` |
| Action | Extract lazy-import helper; remove fake test |

---

### `research/conflict_detector.py` (587 lines)

| Aspect | Assessment |
|---|---|
| Duplication | Overlaps with `genealogy/fact_validator.py` — both use `SequenceMatcher`, similarity thresholds, severity enums |
| Tests | **GOOD** — 8 real tests covering exact match, conflicts, normalization, similarity, severity ordering |
| Complexity | Well-structured; `ConflictSeverity` enum duplicates `ConflictSeverityEnum` in `core/database.py` |
| Type issues | `Optional` imported — some used as `Optional[str]`, some as `str \| None` (inconsistent within file) |
| Specific issue | `ConflictSeverity` enum (module-level) mirrors `ConflictSeverityEnum` (DB-level) but values may diverge |
| Action | Consolidate comparison logic with fact_validator; use DB severity enum directly; standardize type annotations |

---

### `research/triangulation_intelligence.py` (969 lines)

| Aspect | Assessment |
|---|---|
| Duplication | `ConfidenceLevel` enum (L12-17) duplicates confidence mappings in `universal_scoring.py` and `triangulation.py` |
| Tests | Need deeper review (only first 200 lines read) |
| Complexity | Multiple dataclasses (`Evidence`, `TriangulationHypothesis`, `ClusterAnchor`, etc.) — appropriate for the domain |
| Action | Use shared confidence level definitions |

---

### `research/search_criteria_utils.py` (521 lines)

| Aspect | Assessment |
|---|---|
| Duplication | Re-exports `display_family_members` and `present_post_selection` from `genealogy_presenter.py` via Protocol — this indirection adds complexity without clear benefit |
| Tests | Need deeper review |
| Action | Direct imports instead of Protocol re-export pattern |

---

### `research/research_suggestions.py` (790 lines)

| Aspect | Assessment |
|---|---|
| Complexity | Mostly large constant dictionaries (`ANCESTRY_COLLECTIONS`, `TIME_PERIOD_COLLECTIONS`, `ETHNICITY_RESEARCH_SUGGESTIONS`) — appropriate as reference data |
| Tests | Need deeper review |
| Action | Consider moving large constant dicts to a JSON/YAML config file for easier editing |

---

### `research/research_prioritization.py` (1206 lines)

| Aspect | Assessment |
|---|---|
| Complexity | Large file with `IntelligentResearchPrioritizer` class |
| Tests | Need deeper review |
| Action | Consider splitting if > 800 lines policy applies |

---

### `research/research_guidance_prompts.py` (597 lines)

| Aspect | Assessment |
|---|---|
| Complexity | AI prompt builder functions — appropriate |
| Tests | Need deeper review |
| Action | No immediate action needed |

---

### `research/relationship_utils.py` (2218 lines)

| Aspect | Assessment |
|---|---|
| Complexity | **VERY HIGH** at 2218 lines — second largest file in scope |
| Duplication | `RelationshipPathCache` (L365-450) is a manual LRU cache — could use `functools.lru_cache` or the project's existing `caching/` system |
| Tests | Need deeper review of test section |
| Notable | `fast_bidirectional_bfs()` is well-implemented with cache, timeout, node limit, path scoring |
| Action | **Split this file**: Extract `RelationshipPathCache` to `caching/`, HTML parsing to separate module, path formatting to `relationship_diagram.py`. Target: < 800 lines per file. |

---

### `research/relationship_diagram.py` (468 lines)

| Aspect | Assessment |
|---|---|
| Tests | **EXCELLENT** — 11 real tests covering all diagram styles, edge cases, message formatting |
| Complexity | Appropriate — 3 diagram styles is clean |
| Action | **No changes needed** — this is a well-written module |

---

### `research/record_sharing.py` (539 lines)

| Aspect | Assessment |
|---|---|
| Tests | Need deeper review |
| Action | No immediate issues identified from review |

---

### `research/predictive_gaps.py` (827 lines)

| Aspect | Assessment |
|---|---|
| Complexity | `PredictiveGapDetector` class with gap detection — appropriate |
| Tests | Need deeper review |
| Action | No immediate issues identified from review |

---

### `research/person_lookup_utils.py` (539 lines)

| Aspect | Assessment |
|---|---|
| Duplication | `PersonLookupResult` dataclass duplicates `PersonSearchResult` in tree_query_service.py |
| Tests | **EXCELLENT** — 6 real tests covering creation, AI formatting, factory methods, confidence scoring |
| Complexity | Clean factory pattern (`create_result_from_gedcom`, `create_result_from_api`, `create_not_found_result`) |
| Action | Consolidate with `PersonSearchResult`; keep factory methods |

---

## PRIORITY ACTION ITEMS

### P0 — Bugs/Non-functional Code

1. **`gedcom_intelligence.py` stub extractors** — `_extract_birth_year()`, `_extract_death_year()`, `_extract_birth_place()`, `_extract_death_place()` all return `None`. The entire intelligence analysis pipeline is silently non-functional. Implement using `gedcom_utils.get_event_info()`.

2. **`gedcom_cache.py` misplaced code in docstring** — Code fragment embedded in `load_gedcom_with_aggressive_caching()` docstring around line 453. This code won't execute.

3. **`fact_validator.py` unreachable code** — `return existing` at end of `_get_suggested_facts_from_db()` (L507) is unreachable after `except` block that already returns `[]`.

### P1 — High-Impact Consolidation

4. **Unify person result dataclasses** — Merge `PersonSearchResult`, `PersonLookupResult`, `CandidatePerson` into a single base type. Estimated savings: ~200 lines, eliminated mapping code.

5. **Consolidate conflict detection** — `fact_validator.py` and `conflict_detector.py` should share a comparison engine. Extract `ComparisonEngine` with `compare_strings()`, `compare_dates()`, `compare_locations()`.

6. **Split oversized files** — `gedcom_utils.py` (2795 lines) and `relationship_utils.py` (2218 lines) should each be split into 3-4 focused modules.

### P2 — Code Hygiene

7. **Remove 20+ fake `_test_module_integrity()` functions** — Replace with real import validation or skip.

8. **Remove unused `Optional` imports** from 13 files — `ruff check --fix .` handles this.

9. **Trim 4 bloated docstrings** — gedcom_utils, gedcom_search_utils, gedcom_cache, dna_gedcom_crossref.

10. **Consolidate dual GEDCOM caching** — `_GedcomDataCache` in search_utils vs `GedcomCacheModule` in gedcom_cache. Pick one authoritative cache.

### P3 — Nice-to-Have

11. **Extract `__init__.py` lazy-import helper** to eliminate 4× boilerplate.

12. **Move `research_suggestions.py` constants** to JSON/YAML config.

13. **Replace manual `RelationshipPathCache`** with project's caching infrastructure or `functools.lru_cache`.

---

## TEST QUALITY SUMMARY

| Rating | Files |
|---|---|
| **EXCELLENT** (thorough, real assertions, edge cases) | `relationship_calculations.py`, `semantic_search.py`, `fact_validator.py`, `gedcom_intelligence.py`, `relationship_diagram.py`, `person_lookup_utils.py`, `tree_query_service.py` |
| **GOOD** (real assertions, adequate coverage) | `universal_scoring.py`, `tree_stats_utils.py`, `genealogy_presenter.py`, `dna_utils.py`, `conflict_detector.py`, `test_triangulation.py`, `test_research_service.py` |
| **FAKE** (always return True, no real testing) | `genealogy/__init__.py`, `dna/__init__.py`, `gedcom/__init__.py`, `research/__init__.py`, `triangulation.py` (has real tests in separate file), `research_service.py` (has real tests in separate file) |
| **SHALLOW** (test structure not behavior) | `genealogical_normalization.py` |

---

## METRICS

| Metric | Value |
|---|---|
| Files reviewed | 34 |
| Total lines | ~24,500 |
| Critical bugs found | 3 |
| Duplicate dataclass sets | 3 |
| Files with unused `Optional` import | 13 |
| Fake test functions | 6 in scope (20+ project-wide) |
| Files > 1000 lines | 6 (`gedcom_utils`: 2795, `relationship_utils`: 2218, `gedcom_search_utils`: 1374, `research_prioritization`: 1206, `gedcom_cache`: 1198, `fact_validator`: 1192) |
| Estimated reducible LOC | 2,000-3,000 (through consolidation and deduplication) |
