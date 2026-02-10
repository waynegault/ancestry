# Code Review: `messaging/` and `genealogy/` Directories

**Date**: June 2025
**Scope**: All `.py` files in `messaging/` (14 files) and `genealogy/` (27 files including `dna/` and `gedcom/` subpackages)
**Focus Areas**: Duplication, consolidation opportunities, complexity, test quality, linting/type issues

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Cross-Cutting Issues](#cross-cutting-issues)
3. [Messaging Module Review](#messaging-module-review)
4. [Genealogy Module Review](#genealogy-module-review)
5. [Prioritized Action Plan](#prioritized-action-plan)

---

## Executive Summary

| Metric | Messaging | Genealogy | Total |
|--------|-----------|-----------|-------|
| Files reviewed | 14 | 27 | 41 |
| Total lines | ~7,700 | ~13,500 | ~21,200 |
| Critical issues | 3 | 4 | 7 |
| Moderate issues | 8 | 9 | 17 |
| Test quality issues | 3 | 5 | 8 |
| Duplication clusters | 3 | 4 | 7 |

**Top 3 systemic issues:**
1. **SafetyCheckResult name collision** — 3 different classes with the same name across `safety.py`, `send_audit.py`, and `send_orchestrator.py`
2. **Date normalization duplication** — Nearly identical date parsing/extraction logic in `fact_validator.py`, `genealogical_normalization.py`, `message_personalization.py`, and `gedcom_utils.py`
3. **Placeholder implementations** — `gedcom_intelligence.py` and `dna_gedcom_crossref.py` contain large amounts of scaffold code with stub methods returning `None`

---

## Cross-Cutting Issues

### CC-1: `_extract_year_from_date()` Duplication (HIGH)

The same year-extraction logic appears in at least 4 locations:

| Location | Function | Line |
|----------|----------|------|
| `genealogical_normalization.py` | `_extract_year_from_date()` | ~280 |
| `fact_validator.py` | `ExtractedFact._normalize_date()` | ~165 |
| `message_personalization.py` | `_extract_year_from_date()` (static) | ~766 |
| `gedcom_utils.py` | `_extract_year_fallback()` | ~517 |

**Recommendation**: Create a shared `genealogy/date_utils.py` module with canonical date parsing, year extraction, and normalization. All modules import from there.

### CC-2: Deduplication Helpers Duplicated (LOW)

| Location | Function |
|----------|----------|
| `genealogical_normalization.py` | `_dedupe_list_str()` |
| `tree_stats_utils.py` | `_dedupe_preserve_order()` |
| `messaging/workflow_helpers.py` | `_deduplicate_values()` (similar pattern) |

**Recommendation**: Consolidate into a single utility in `core/` or `utils.py`.

### CC-3: Bloated Module Docstrings (LOW)

Several `gedcom/` and `dna/` files have 30-50 line marketing-style docstrings (e.g., `gedcom_cache.py`, `gedcom_search_utils.py`, `dna_gedcom_crossref.py`). These add no technical value and make navigation harder.

**Recommendation**: Trim to ≤10 lines describing purpose, key classes, and dependencies.

---

## Messaging Module Review

### `messaging/__init__.py` (124 lines)
- **Status**: Clean. Comprehensive `__all__` with proper public API.
- **No issues found.**

---

### `messaging/empathetic_responses.py` (775 lines)

- **Quality**: Good. Well-structured escalation detection with `EscalationCategory` enum and template system.
- **Tests**: 11 tests — good coverage.
- **No significant issues.**

---

### `messaging/message_types.py` (297 lines)

- **Quality**: Good. Clean state machine with transition table.
- **Tests**: 7 tests — adequate.
- **No significant issues.**

---

### `messaging/person_eligibility.py` (534 lines)

- **Quality**: Good. `PersonEligibilityChecker` with proper lazy imports to avoid circular deps.
- **Tests**: 8 tests — adequate.
- **Minor**: `IneligibilityReason` has 9 values — consider documenting which are soft vs hard rejections.

---

### `messaging/safety.py` (520 lines) — ISSUES FOUND

**Issue S-1: `field(default_factory=list)` on non-dataclass (MODERATE)**
- `_OPT_OUT_PATTERNS` at line ~72 uses `field(default_factory=list)` but `SafetyGuard` is **not** a dataclass. This is a runtime no-op — the field descriptor object is assigned directly rather than triggering dataclass machinery.
- **Fix**: Replace with a plain `list()` or `[]`.

**Issue S-2: Duplicated safety patterns (MODERATE)**
- Legacy patterns and Phase 2 patterns contain overlapping regex entries (e.g., `\bsuicide\b`, `\bpolice\b`, `\blawyer\b` appear in both sets).
- **Fix**: Merge into a single pattern set or explicitly document the two tiers and ensure no overlap.

**Issue S-3: Inconsistent test runner import path (LOW)**
- Uses a slightly different import path for `create_standard_test_runner` compared to other files.
- **Fix**: Standardize to `from testing.test_utilities import create_standard_test_runner`.

- **Tests**: 15 tests — good coverage, but no tests verify that the duplicated patterns don't cause double-flagging.

---

### `messaging/send_audit.py` (765 lines) — ISSUES FOUND

**Issue SA-1: `SafetyCheckResult` name collision (CRITICAL)**
- Defines a `SafetyCheckResult` class that collides with identically-named classes in `safety.py` and `send_orchestrator.py`. These are 3 different classes with different fields.
- **Impact**: Any code that imports from multiple messaging modules may get the wrong class. Refactoring or renaming is blocked by ambiguity.
- **Fix**: Rename to `AuditSafetyCheckResult` in `send_audit.py` and `OrchestratorSafetyCheckResult` in `send_orchestrator.py`, or consolidate to a single shared class.

- **Tests**: 8 tests — adequate.

---

### `messaging/send_metrics.py` (405 lines)

- **Quality**: Good. Clean Prometheus-compatible metrics with normalization helpers.
- **Tests**: 8 tests — adequate.
- **No significant issues.**

---

### `messaging/send_orchestrator.py` (1362 lines) — ISSUES FOUND

**Issue SO-1: `SafetyCheckResult` collision (see SA-1) (CRITICAL)**
- Third definition of `SafetyCheckResult` — see SA-1 above.

**Issue SO-2: Unused computed variable (LOW)**
- `_generate_desist_acknowledgement()` at line ~780 computes `person_name` but never uses it.
- **Fix**: Remove the unused variable or use it in the generated message.

**Issue SO-3: Tests only validate dataclass creation (MODERATE)**
- Tests create dataclass/enum instances but do NOT test actual orchestration logic (`send()`, `run_safety_checks()`, `_make_decision()`).
- **Impact**: The most complex logic in the module (decision tree, safety pipeline) has zero test coverage.
- **Fix**: Add integration tests with mocked database and session objects that exercise the send pipeline.

---

### `messaging/shadow_mode_analyzer.py` (510 lines)

**Issue SM-1: Tests don't test comparison logic (MODERATE)**
- 7 tests only validate dataclass creation, not the actual comparison/analysis functions.
- **Fix**: Add tests that compare orchestrator vs legacy decisions and verify discrepancy detection.

---

### `messaging/template_selector.py` (583 lines)

- **Quality**: Good. `TemplateVariant` enum with 6 values, clean selector logic.
- **Tests**: 8 tests — adequate.
- **No significant issues.**

---

### `messaging/workflow_helpers.py` (719 lines)

- **Quality**: Good. Well-tested utility functions.
- **Tests**: 13 tests — among the best test coverage in messaging/.
- **No significant issues.**

---

### `messaging/inbound.py` (868 lines)

**Issue IN-1: `_record_messaging_counter()` helper duplicates pattern (LOW)**
- The metrics counter recording pattern is repeated across multiple messaging modules. Could be consolidated.

- **Tests**: Bridge to unittest in `test_inbound.py`.

---

### `messaging/test_inbound.py` (290 lines) — ISSUE FOUND

**Issue TI-1: Double test registration (LOW)**
- Has its own standalone runner AND `inbound.py` wraps the same tests via `create_standard_test_runner`.
- **Impact**: Tests may run twice during full suite execution.
- **Fix**: Remove the standalone runner in `test_inbound.py` or the wrapper in `inbound.py`.

---

### `messaging/message_personalization.py` (2401 lines) — ISSUES FOUND

**Issue MP-1: File too large — candidate for splitting (HIGH)**
- At 2401 lines, this is the largest file in `messaging/`. Contains three distinct concerns:
  1. `MessagePersonalizer` class (~1200 lines) with 30+ personalization methods
  2. `MessageEffectivenessTracker` class (~400 lines) with effectiveness tracking
  3. Test suite (~500 lines)
- **Fix**: Split into `message_personalizer.py`, `message_effectiveness.py`, and move tests to `test_message_personalization.py`.

**Issue MP-2: 30+ repetitive static methods (MODERATE)**
- Methods like `_format_shared_ancestors()`, `_create_genealogical_context()`, `_format_occupations()`, etc. all follow the identical pattern:
  ```python
  @staticmethod
  def _method_name(data: Any) -> str:
      if not isinstance(data, dict):
          return ""
      # extract from dict with .get() calls
      # format string
      return result
  ```
- **Fix**: Extract a generic `_extract_and_format(data, keys, template)` helper to reduce repetition. Or use a registry/dispatch pattern keyed by personalization type.

**Issue MP-3: `_extract_year_from_date()` duplication (see CC-1)**

- **Tests**: 20 tests — comprehensive, including edge cases (null, Unicode, extremely long inputs). Good quality.

---

## Genealogy Module Review

### `genealogy/__init__.py` (45 lines)
- **Status**: Clean. Lazy imports for submodules.
- **No issues found.**

---

### `genealogy/fact_validator.py` (1192 lines) — ISSUES FOUND

**Issue FV-1: `extract_facts_from_ai_response()` duplicates `ExtractedFact.from_conversation()` (HIGH)**
- Standalone function `extract_facts_from_ai_response()` at line ~1050 largely duplicates the `ExtractedFact.from_conversation()` class method at line ~220. Both extract facts from AI JSON responses using the same field mappings.
- **Fix**: Have `extract_facts_from_ai_response()` delegate to `ExtractedFact.from_conversation()` or vice versa.

**Issue FV-2: `_compare_dates()` logic gap (MODERATE)**
- Lines ~597-600: Returns `MINOR_CONFLICT` for both 1-2 year AND 3-5 year differences. The `MAJOR_CONFLICT_YEAR_DIFF = 5` threshold is effectively unused for the 3-5 year range — those should arguably be `MODERATE_CONFLICT` or similar.
- **Fix**: Add a `MODERATE_CONFLICT` level or adjust thresholds so 3-5 years returns `MAJOR_CONFLICT`.

**Issue FV-3: Date normalization duplication (see CC-1)**
- `ExtractedFact._normalize_date()` and `_extract_date_qualifier()` duplicate logic in `genealogical_normalization.py`.

**Issue FV-4: Unreachable code (LOW)**
- `_get_suggested_facts_from_db()` at line ~528: has `return existing` after the except block that may be unreachable depending on control flow.

- **Tests**: 13 tests — good coverage including factory methods, conflict detection, AI response extraction.

---

### `genealogy/genealogical_normalization.py` (699 lines)

- **Quality**: Good. Conservative normalization approach, well-documented.
- **Tests**: 6 tests — adequate.
- **Duplication**: Date helpers overlap with `fact_validator.py` (see CC-1).
- **Has `_run_basic_tests()` fallback** when TestSuite unavailable — acceptable defensive pattern.

---

### `genealogy/genealogy_presenter.py` (~310 lines)

- **Quality**: Good. Clean presentation logic with deduplication.
- **Tests**: 5 tests with `_FallbackTestSuite` class.
- **No significant issues.**

---

### `genealogy/relationship_calculations.py` (~280 lines)

- **Quality**: Excellent. Pure functions, no side effects, clean separation of concerns.
- **Tests**: 1 comprehensive test covering all relationship types.
- **This is a model module** — other modules should emulate this design.

---

### `genealogy/research_service.py` (~320 lines) — ISSUE FOUND

**Issue RS-1: No meaningful tests (CRITICAL)**
- The only test is `_test_module_integrity()` which returns `True` unconditionally.
- `ResearchService` has methods (`search_people()`, `get_relationship_path()`) with zero test coverage.
- **Mitigated by**: `test_research_service.py` exists with 4 proper mocked tests. But the module's own `run_comprehensive_tests` runs the empty test.
- **Fix**: Replace `_test_module_integrity` with a reference to `test_research_service.research_service_tests`.

---

### `genealogy/test_research_service.py` (~110 lines)

- **Quality**: Good. 4 properly mocked tests for GEDCOM loading and search.
- **No issues.**

---

### `genealogy/semantic_search.py` (756 lines)

- **Quality**: Very good. Conservative design with fail-closed-to-clarification approach. Well-structured dataclasses with `to_dict()` and `to_prompt_string()`.
- **Tests**: 5 tests including stub-based integration tests for candidate retrieval, ambiguity detection, and prompt formatting. Good quality.
- **No significant issues.**

---

### `genealogy/tree_query_service.py` (981 lines)

- **Quality**: Good. Lazy initialization, fuzzy matching with scoring, BFS relationship pathfinding.
- **Tests**: 9 tests — good coverage of dataclasses, ordinal generation, prompt string formatting, surname extraction.
- **Minor**: Tests don't exercise `find_person()` or `explain_relationship()` with real GEDCOM data (only dataclass construction).

---

### `genealogy/tree_stats_utils.py` (1054 lines) — ISSUES FOUND

**Issue TS-1: `_dedupe_preserve_order()` duplicates `genealogical_normalization._dedupe_list_str()` (LOW)**
- See CC-2.

**Issue TS-2: `excluded_cols` set defined twice (LOW)**
- The set of excluded ethnicity columns is defined in both `_calculate_ethnicity_distribution()` and `_get_ethnicity_columns()`.
- **Fix**: Define once as a module constant.

- **Tests**: 11 tests — good coverage including cache hits, match count validation, timestamp format, ethnicity structure. Tests use live database which is appropriate for this module.

---

### `genealogy/triangulation.py` (~140 lines)

- **Quality**: Clean service class with proper DB query usage.
- **Tests**: Module-level test is `_test_module_integrity()` returning `True` — **no real tests**.
- **Mitigated by**: `test_triangulation.py` has 1 proper mocked test for filtering logic.
- **Fix**: Wire `test_triangulation.triangulation_tests` as the module's test runner.

---

### `genealogy/test_triangulation.py` (~80 lines)

- **Quality**: Good. 1 test with proper mocking that validates `min_cm` and `min_confidence` parameter handling.
- **No issues.**

---

### `genealogy/universal_scoring.py` (600 lines)

- **Quality**: Good. Clean scoring abstraction that unifies Action 10 and Action 11 scoring.
- **Tests**: 15 tests — excellent coverage including exact/partial/no match, criteria validation, confidence levels, display bonuses.
- **Minor**: `format_scoring_breakdown()` accepts `_search_criteria` (unused parameter, prefixed with underscore). Consider removing if truly unused.

---

### `genealogy/dna/__init__.py` (45 lines)
- Lazy imports. No issues.

---

### `genealogy/dna/dna_utils.py` (643 lines)

- **Quality**: Good. Well-structured with clear separation of navigation, CSRF, in-tree status, and match list functions.
- **Tests**: 7 tests — all unit tests for URL/header/cache key construction. Adequate for utility functions.
- **No significant issues.**

---

### `genealogy/dna/dna_ethnicity_utils.py` (847 lines)

- **Quality**: Good. API integration for ethnicity region management with proper session handling.
- **Tests**: 3 unit tests (column sanitization, percentage extraction, regression guard) + 3 live API tests (skippable). Appropriate split.
- **Minor**: `LIVE_API_TESTS` tuple at line ~750 is defined but only used if tests are run directly — consider documenting this more clearly.

---

### `genealogy/dna/dna_gedcom_crossref.py` (807 lines) — ISSUES FOUND

**Issue DG-1: Extensive placeholder implementations (HIGH)**
- Multiple methods have `# Placeholder implementation` or return `None` unconditionally:
  - `_extract_birth_year()` (line ~490) — always returns `None`
  - `_extract_death_year()` (line ~498) — always returns `None`
  - `_extract_birth_place()` (line ~503) — always returns `None`
  - `_extract_death_place()` (line ~507) — always returns `None`
  - `_is_plausible_relationship_match()` (line ~545) — always returns `True`
- **Impact**: The cross-reference analysis claims to analyze people but cannot actually extract basic data from GEDCOM records. The entire module may produce misleading results.
- **Fix**: Implement these using the existing `gedcom_utils.py` functions, or mark the module as experimental/incomplete in its docstring.

**Issue DG-2: `DNAMatch` name collision with `core/database.py` (MODERATE)**
- `dna_gedcom_crossref.py` defines a `DNAMatch` dataclass (line ~65) while `core/database.py` likely has a `DnaMatch` SQLAlchemy model. Different casing but confusingly similar.
- **Fix**: Rename to `DNACrossRefMatch` or `DNAMatchInput`.

- **Tests**: 5 tests — test the actual cross-reference flow, confidence boosting, conflict detection, and relationship parsing. Good quality for the non-placeholder parts.

---

### `genealogy/gedcom/__init__.py` (45 lines)
- Lazy imports. No issues.

---

### `genealogy/gedcom/gedcom_cache.py` (1198 lines)

- **Quality**: Solid multi-level caching system (memory + disk) with `GedcomCacheModule` implementing `BaseCacheModule`.
- **Architecture**: Good use of `get_unified_cache_key()` and `warm_cache_with_data()` from caching framework.
- **Minor**: `LazyGedcomData` class at line ~108 creates a new `GedcomReader` on every `.data` access — should cache the result.
- **Tests**: Not visible in the first 400 lines read; likely in remainder of file.

---

### `genealogy/gedcom/gedcom_intelligence.py` (951 lines) — ISSUES FOUND

**Issue GI-1: Placeholder extract methods — same pattern as DG-1 (HIGH)**
- `_extract_birth_year()` (line ~463), `_extract_death_year()` (line ~473), `_extract_birth_place()` (line ~482), `_extract_death_place()` (line ~491) all return `None`.
- **Impact**: Gap detection (`_analyze_family_completeness`) and date consistency checks (`_analyze_date_consistency`) cannot function properly since they depend on these methods.
- **Fix**: Use `gedcom_utils._parse_date()` and the existing GEDCOM extraction infrastructure.

**Issue GI-2: `_generate_ai_insights()` doesn't actually call AI (LOW)**
- Despite the name and module description claiming "AI-powered analysis", the method only performs simple arithmetic (completeness %) and static recommendations.
- **Fix**: Either integrate with `ai_interface.call_ai()` or rename to `_generate_analysis_insights()`.

---

### `genealogy/gedcom/gedcom_search_utils.py` (1374 lines)

- **Quality**: Comprehensive search infrastructure with multi-level GEDCOM caching, criterion matching, and scoring integration.
- **Architecture**: `_GedcomDataCache` class manages singleton GEDCOM data with fallback loading strategies.
- **Uses config-based scoring weights** rather than hardcoded values — good practice.
- **Minor**: `normalize_gedcom_id()` at line ~71 calls `_get_gedcom_utils_module()._normalize_id` — imports a private function via string lookup. Should use `gedcom_utils.normalize_id` directly.

---

### `genealogy/gedcom/gedcom_utils.py` (2797 lines)

- **Quality**: Core GEDCOM processing engine. Well-structured with many small helper functions.
- **Size**: At 2797 lines, this is the largest file in the reviewed scope. Contains:
  - ID normalization and validation (~100 lines)
  - Name extraction with 4 fallback strategies (~200 lines)
  - Date parsing with dateparser + strptime + regex fallback (~200 lines)
  - Event extraction and individual processing (~300 lines)
  - `GedcomData` class with family map building (~800 lines)
  - Scoring system (`calculate_match_score`) (~400 lines)
  - Display formatting (~200 lines)
  - Tests (~400 lines)
- **Recommendation**: Consider splitting into `gedcom_parsing.py` (ID/name/date extraction), `gedcom_data.py` (GedcomData class), and `gedcom_scoring.py` (match scoring). Keep `gedcom_utils.py` as a re-export facade.

---

## Prioritized Action Plan

### Priority 1 — Critical (Fix Now)

| # | Issue | Files | Impact |
|---|-------|-------|--------|
| 1 | **SafetyCheckResult** name collision (3 classes, same name) | `safety.py`, `send_audit.py`, `send_orchestrator.py` | Import confusion, maintenance burden |
| 2 | **Placeholder GEDCOM extractors** always return `None` | `gedcom_intelligence.py`, `dna_gedcom_crossref.py` | Entire analysis modules non-functional |
| 3 | **`send_orchestrator.py` zero test coverage** of actual orchestration logic | `send_orchestrator.py` | Riskiest module has no real tests |
| 4 | **`research_service.py` wired to empty test** | `research_service.py` | Appears tested but isn't |

### Priority 2 — High (Fix This Sprint)

| # | Issue | Files | Impact |
|---|-------|-------|--------|
| 5 | **Date utility duplication** across 4+ files | Multiple (see CC-1) | Maintenance burden, divergent behavior |
| 6 | **`message_personalization.py` at 2401 lines** | `message_personalization.py` | Hard to navigate, review, test |
| 7 | **`extract_facts_from_ai_response()` duplicates factory method** | `fact_validator.py` | Two paths for same logic will diverge |
| 8 | **`_compare_dates()` logic gap** (3-5 year range) | `fact_validator.py` | Incorrect conflict severity assignment |

### Priority 3 — Moderate (Backlog)

| # | Issue | Files | Impact |
|---|-------|-------|--------|
| 9 | `field()` on non-dataclass | `safety.py` line ~72 | Runtime no-op, confusing |
| 10 | Duplicated safety patterns | `safety.py` | Potential double-flagging |
| 11 | `shadow_mode_analyzer.py` tests don't test analysis | `shadow_mode_analyzer.py` | False confidence in test suite |
| 12 | Double test registration | `test_inbound.py` / `inbound.py` | Tests run twice |
| 13 | Unused `person_name` variable | `send_orchestrator.py` ~line 780 | Dead code |
| 14 | `_dedupe` helper duplication | `tree_stats_utils.py`, `genealogical_normalization.py` | Minor maintenance burden |
| 15 | `excluded_cols` defined twice | `tree_stats_utils.py` | Minor maintenance burden |
| 16 | `DNAMatch` name collision | `dna_gedcom_crossref.py` vs `core/database.py` | Confusion |
| 17 | Bloated module docstrings | `gedcom_cache.py`, `gedcom_search_utils.py`, etc. | Navigation difficulty |
| 18 | `gedcom_utils.py` at 2797 lines | `gedcom_utils.py` | Hard to navigate |

### Priority 4 — Low (Nice to Have)

| # | Issue | Files | Impact |
|---|-------|-------|--------|
| 19 | `normalize_gedcom_id()` uses private import | `gedcom_search_utils.py` | Fragile coupling |
| 20 | `LazyGedcomData.data` doesn't cache result | `gedcom_cache.py` | Re-parses on every access |
| 21 | `_generate_ai_insights()` doesn't call AI | `gedcom_intelligence.py` | Misleading name |
| 22 | Metrics counter pattern duplicated | `inbound.py` + others | Minor repetition |

---

## Module Quality Summary

### Messaging — Test Quality Matrix

| File | Tests | Coverage Quality | Grade |
|------|-------|-----------------|-------|
| `empathetic_responses.py` | 11 | Good | A |
| `message_types.py` | 7 | Adequate | B |
| `person_eligibility.py` | 8 | Adequate | B |
| `safety.py` | 15 | Good (missing overlap test) | B+ |
| `send_audit.py` | 8 | Adequate | B |
| `send_metrics.py` | 8 | Adequate | B |
| `send_orchestrator.py` | ~8 | **Dataclass-only, no logic tests** | **D** |
| `shadow_mode_analyzer.py` | 7 | **Dataclass-only, no analysis tests** | **D** |
| `template_selector.py` | 8 | Adequate | B |
| `workflow_helpers.py` | 13 | Good | A |
| `inbound.py` / `test_inbound.py` | 5 | Good (proper mocks) | A- |
| `message_personalization.py` | 20 | Comprehensive | A |

### Genealogy — Test Quality Matrix

| File | Tests | Coverage Quality | Grade |
|------|-------|-----------------|-------|
| `fact_validator.py` | 13 | Good | A- |
| `genealogical_normalization.py` | 6 | Adequate | B |
| `genealogy_presenter.py` | 5 | Adequate | B |
| `relationship_calculations.py` | 1 (comprehensive) | Excellent | A+ |
| `research_service.py` | 0 real (empty) | **Non-existent** | **F** |
| `test_research_service.py` | 4 | Good (proper mocks) | A- |
| `semantic_search.py` | 5 | Good (stub integration) | A- |
| `tree_query_service.py` | 9 | Good | A- |
| `tree_stats_utils.py` | 11 | Good (live DB) | A |
| `triangulation.py` | 0 real (empty) | **Non-existent** | **F** |
| `test_triangulation.py` | 1 | Adequate (mocked) | B |
| `universal_scoring.py` | 15 | Excellent | A+ |
| `dna_utils.py` | 7 | Adequate (unit) | B |
| `dna_ethnicity_utils.py` | 3+3 live | Good split | B+ |
| `dna_gedcom_crossref.py` | 5 | Good (non-stub parts) | B+ |
| `gedcom_intelligence.py` | ? | Unknown (not fully read) | ? |
| `gedcom_cache.py` | ? | Unknown (not fully read) | ? |
| `gedcom_search_utils.py` | ? | Unknown (not fully read) | ? |
| `gedcom_utils.py` | ? | Unknown (not fully read) | ? |

---

## Well-Designed Modules (Positive Highlights)

These modules demonstrate patterns worth emulating:

1. **`relationship_calculations.py`** — Pure functions, no side effects, single responsibility, comprehensive test
2. **`semantic_search.py`** — Conservative fail-closed design, proper dataclass hierarchy, `to_prompt_string()` pattern
3. **`universal_scoring.py`** — Clean abstraction layer unifying two action modules, 15 tests
4. **`workflow_helpers.py`** — Well-tested utility functions, 13 tests with good edge case coverage
5. **`message_personalization.py` tests** — Comprehensive edge cases including Unicode, null, extremely long inputs
