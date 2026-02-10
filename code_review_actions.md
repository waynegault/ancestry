# Code Review: `actions/` and `actions/gather/` Packages

**Date:** 2025-01-27
**Scope:** 22 files across `actions/` (12 files) and `actions/gather/` (10 files)
**Total Lines:** ~27,600

---

## Executive Summary

The `actions/` package is the single largest concentration of business logic in the codebase. Overall quality is solid — the module-level test framework is consistently applied, the SessionManager pattern is respected, and critical actions (6, 7, 11) have well-structured orchestration. However, there are several high-severity issues that merit immediate attention:

| Category | Critical | High | Medium | Low |
|---|---|---|---|---|
| Code Duplication | 1 | 2 | 3 | — |
| Consolidation Opportunities | — | 3 | 4 | 2 |
| Excess Complexity | — | 2 | 5 | 3 |
| Test Quality | 1 | 3 | 4 | — |
| Linting / Type Annotations | — | 1 | 6 | 5 |

---

## 1. CRITICAL & HIGH-SEVERITY FINDINGS

### 1.1 `needs_ethnicity_refresh()` — Duplicated with Divergent Logic ⛔ CRITICAL

**Files:** `gather/persistence.py:304` and `gather/api_implementations.py:242`

Two completely different implementations exist under the same name:

| | `persistence.py` | `api_implementations.py` |
|---|---|---|
| **Semantics** | Time-based: refresh if `ethnicity_updated_at` is >7 days old | Column-based: refresh if any ethnicity column is `None` |
| **`None` input** | Returns `True` (needs refresh) | Returns `False` (skip) |
| **Used by** | `persistence.py:240` (flags matches during batch prep) | Not directly imported anywhere in `gather/` |
| **Exported** | Yes (via `gather/__init__.py`) | No |

**Impact:** The `api_implementations.py` version is dead or shadows the canonical one. If any caller imports from the wrong module, the opposite behavior occurs on `None` input.

**Recommendation:** Delete `api_implementations.py:needs_ethnicity_refresh`. Keep the `persistence.py` version as the single source of truth. If column-presence checking is needed, create a separate `has_ethnicity_data()` function.

---

### 1.2 Circular Import: `api_implementations.py` → `action6_gather.py` ⛔ HIGH

**File:** `gather/api_implementations.py:284`

```python
# Inside _fetch_match_details_api():
from actions.action6_gather import _parse_details_response
```

A `gather/` submodule reaches back into the *parent* action file to import a private function. This is a layering violation — the `gather/` package was created to decompose `action6_gather.py`, but the dependency runs backward.

The surrounding comments (`"I missed _parse_details_response!"`) confirm this was an accidental omission during the refactor.

**Recommendation:** Move `_parse_details_response` into `gather/api_implementations.py` (or a shared `gather/parsers.py` module). This eliminates the circular dependency and completes the refactor.

---

### 1.3 Fake/Smoke Tests in `performance_logging.py` ⛔ HIGH

**File:** `gather/performance_logging.py:129-171`

Three of four test functions follow this anti-pattern:

```python
def _test_log_api_performance() -> None:
    log_api_performance(...)  # call the function
    assert True, "log_api_performance should execute without errors"

def _test_duration_messages() -> None:
    _log_api_duration_message(...)
    assert True, "Duration messages should handle all thresholds"

def _test_batch_processing_message() -> None:
    ...
    assert True, "Batch processing messages should handle escalating durations"
```

These tests only verify "doesn't crash." They validate zero behavior — the functions could return garbage or log nothing and these would still pass.

**Recommendation:**
- Capture logger output with `unittest.mock.patch('logging.Logger.info')` and assert expected log messages or call counts.
- For `_log_api_duration_message`, verify the tiered threshold messages (e.g., >5s → "🐢 SLOW", >15s → "🔴 CRITICAL").

---

### 1.4 `_get_api_headers()` Wrapped in `@api_retry` ⛔ HIGH

**File:** `gather/api_implementations.py`

```python
@api_retry
def _get_api_headers() -> dict[str, str]:
    return {"User-Agent": "...", ...}
```

This function builds a static dict from constants. It can never fail, never makes network calls, and never needs retry logic. The decorator adds overhead (try/except/sleep setup) on every call for zero benefit.

**Recommendation:** Remove the `@api_retry` decorator from `_get_api_headers()`.

---

### 1.5 Decorator Stack on `_validate_session_state` ⛔ HIGH

**File:** `gather/orchestrator.py:765-769`

```python
@with_enhanced_recovery(max_attempts=3, base_delay=2.0, max_delay=60.0)
@selenium_retry()
@circuit_breaker(failure_threshold=3, recovery_timeout=60)
@timeout_protection(timeout=900)
@error_context("DNA match gathering coordination")
def _validate_session_state(self, ...):
```

Five stacked decorators create:
1. **Retry multiplication**: `with_enhanced_recovery` (3 attempts) × `selenium_retry` (default 3 attempts) = up to 9 retries before the circuit breaker even engages.
2. **Unpredictable timeout behavior**: `timeout_protection(900)` wraps the entire retry chain. A 15-minute timeout on session validation is excessive.
3. **Debugging opacity**: Stack traces pass through 5 wrapper layers, making root cause analysis difficult.

**Recommendation:** Consolidate to at most 2 decorators. Use `@with_enhanced_recovery` for retry logic and `@error_context` for logging. The circuit breaker is already managed at the `GatherOrchestrator` level.

---

## 2. CODE DUPLICATION

### 2.1 Redundant `plan or checkpoint_settings()` Pattern — MEDIUM

**File:** `gather/checkpoint.py`

Every public function repeats:
```python
def load_checkpoint(plan: GatherCheckpointPlan | None = None) -> dict[str, Any] | None:
    plan = plan or checkpoint_settings()
    ...

def persist_checkpoint(plan: GatherCheckpointPlan | None = None, ...) -> None:
    plan = plan or checkpoint_settings()
    ...
```

This appears in `load_checkpoint`, `persist_checkpoint`, `write_checkpoint_state`, `clear_checkpoint`, `finalize_checkpoint_after_run` — 5 functions total.

**Recommendation:** Make `GatherCheckpointService` a proper class that receives `plan` once in `__init__`, or use `functools.partial` to bind the plan. This removes 5 redundant `plan or checkpoint_settings()` lines.

---

### 2.2 Person Blocking Logic Duplicated — MEDIUM

**Files:** `action11_send_approved_drafts.py:131-153` vs `action16_unified_send.py` (via `should_allow_outbound_to_person`)

`action11` implements its own `_person_blocks_outbound()` and `_conversation_state_blocks_outbound()` functions that duplicate the intent of `core.app_mode_policy.should_allow_outbound_to_person()`, which `action16` already uses exclusively.

`action11` checks: DESIST/ARCHIVE/BLOCKED status, contactable flag, automation_enabled flag, conversation state (OPT_OUT, HUMAN_REVIEW, PAUSED, DESIST, safety_flag).

`should_allow_outbound_to_person()` handles the same concerns centrally.

**Recommendation:** Migrate `action11` to use `should_allow_outbound_to_person()` exclusively. The `_person_blocks_outbound` and `_conversation_state_blocks_outbound` functions in action11 can be removed.

---

### 2.3 Error Categorization Duplicated — MEDIUM

`action11_send_approved_drafts.py` defines `SendErrorCategory` and `categorize_send_error()`. `action8_messaging.py` has `ErrorCategorizer` class. Both classify exceptions into the same buckets (network, auth, rate_limit, api_error, etc.) with similar string-matching logic.

**Recommendation:** Consolidate into `core/error_handling.py` as a shared utility. Both actions can import from there.

---

## 3. CONSOLIDATION OPPORTUNITIES

### 3.1 `action16_unified_send.py` Should Replace `action8` + `action9` + `action11` — HIGH

Action 16 was designed as a unified replacement for running Actions 8, 9, and 11 separately. However, all four actions still exist as independent code paths. Action 16 reimplements candidate gathering logic from:
- `action8_messaging.py` (template sequences)
- `action9_process_productive.py` (AI replies)
- `action11_send_approved_drafts.py` (approved drafts)

**Recommendation:** Actions 8, 9, and 11 should either:
1. Delegate their outbound logic to Action 16's `UnifiedSendProcessor`, or
2. Be deprecated with Action 16 as the canonical send path.

Currently maintaining four independent send paths increases the risk of divergent behavior.

---

### 3.2 `gather/fetch.py` Is a Premature Abstraction — HIGH

**File:** `gather/fetch.py` (61 lines)

This file contains:
- `GatherFetchPlan` dataclass with 2 fields (`total_match_count`, `matches_per_page`)
- `GatherFetchService` class with one method (`build_plan`) that returns a `GatherFetchPlan`
- 2 trivial tests

The entire file could be replaced by a 3-line dataclass in `orchestrator.py`. The "service" class adds no behavior — `build_plan()` just returns a dataclass with the same values passed in.

**Recommendation:** Inline `GatherFetchPlan` into `orchestrator.py` or `checkpoint.py`. Delete `fetch.py`.

---

### 3.3 `action14_research_tools.py` Is a Pure Delegation Wrapper — HIGH

**File:** `action14_research_tools.py` (93 lines)

The entire file delegates to `cli.research_tools.run_interactive_menu()`:

```python
def run_research_tools(session_manager, *_):
    run_interactive_menu()
    return True
```

The remaining ~50 lines are tests to verify this delegation works.

**Recommendation:** Register `cli.research_tools.run_interactive_menu` directly in the action registry instead of maintaining this wrapper file. If the wrapper exists only for the `SessionManager` signature contract, consider a lambda or `functools.partial`.

---

### 3.4 `gather/rate_persistence.py` Could Be Absorbed — MEDIUM

**File:** `gather/rate_persistence.py` (121 lines)

Contains a single function `persist_rates_periodically()` and a module-level `_persistence_state` dict. This is tightly coupled to `orchestrator.py` and adds file-count overhead for minimal abstraction value.

**Recommendation:** Move `persist_rates_periodically()` into `orchestrator.py` as a private method of `GatherOrchestrator`.

---

## 4. EXCESS COMPLEXITY

### 4.1 `action6_gather.py` Remains at 4,525+ Lines — HIGH

Despite the `gather/` subpackage extraction (checkpoint, metrics, orchestrator, persistence, prefetch, etc.), `action6_gather.py` still contains 4,525+ lines. This suggests the refactor is incomplete — significant logic (match processing, caching, navigation, URL construction) still lives in the parent file.

**Recommendation:** Continue the decomposition:
- Move `_parse_details_response` to `gather/api_implementations.py` (also fixes the circular import)
- Move URL/navigation helpers to `gather/navigation.py`
- Move cache helper functions to `caching/` or `gather/cache_helpers.py`
- Target: `action6_gather.py` should be <500 lines, primarily containing `coord()` and the action entry point

---

### 4.2 `action8_messaging.py` at 5,776+ Lines — MEDIUM

Largest file in the entire project. Contains template management, sentiment analysis, message personalization, A/B testing, error categorization, resource management, and send logic all in one file.

**Recommendation:** Extract into subpackage:
- `messaging/templates.py` — template loading and management
- `messaging/personalization.py` — sentiment adaptation, personalization
- `messaging/error_categorizer.py` → shared with action11
- Keep `action8_messaging.py` as the orchestrator (<500 lines)

---

### 4.3 `action13_triangulation.py` — Menu Rendering Mixed with Business Logic — MEDIUM

The file mixes UI concerns (menu rendering with emoji, `input()` calls, `print()` formatting) with service orchestration. The `_render_triangulation_menu`, `_handle_search_target`, `_handle_enter_id`, etc. are pure UI functions interleaved with domain logic.

**Recommendation:** Move UI functions to `ui/` or `cli/triangulation_menu.py`. Keep `run_triangulation_analysis` as the action entry point delegating to both the menu and the `TriangulationService`.

---

### 4.4 `gather/persistence.py` at 1,515 Lines — MEDIUM

This file handles Person CRUD, DnaMatch CRUD, ethnicity data, deduplication, bulk operations, and ID mapping. It's doing too many things.

**Recommendation:** Split into:
- `gather/person_persistence.py` — Person create/update bulk operations
- `gather/match_persistence.py` — DnaMatch create/update
- `gather/ethnicity.py` — ethnicity refresh logic and `needs_ethnicity_refresh()`

---

### 4.5 `action15_tree_updates.py` — Hardcoded Cutoff Date Math — LOW

**File:** `action_review.py:453` (via `batch_expire_old_items`)

```python
cutoff = cutoff.replace(day=cutoff.day - max_age_days)
```

This will throw `ValueError` when `max_age_days > cutoff.day` (e.g., day=15, max_age_days=30). The conditional guard only handles one branch.

**Recommendation:** Use `timedelta`:
```python
cutoff = datetime.now(UTC) - timedelta(days=max_age_days)
```

---

## 5. TEST QUALITY ASSESSMENT

### Per-File Test Quality Matrix

| File | Tests | Quality | Issues |
|---|---|---|---|
| `gather/checkpoint.py` | 5 | ✅ GOOD | Real tempdir round-trips, expiry validation |
| `gather/rate_persistence.py` | 2 | ✅ GOOD | Interval logic + graceful handling tested |
| `gather/metrics.py` | 5 | ✅ GOOD | Validates snapshot math, accumulation |
| `gather/performance_logging.py` | 4 | ⛔ FAKE | 3 of 4 tests are `assert True` after calling functions |
| `gather/prefetch.py` | 4 | ⚠️ MEDIUM | Config clamping good; hooks contract weakly tested |
| `gather/persistence.py` | 4 | ⚠️ MEDIUM | Summary totals validated; `needs_ethnicity_refresh` tests exist but test the wrong copy |
| `gather/fetch.py` | 2 | ⚠️ TRIVIAL | Tests only verify dataclass defaults |
| `gather/orchestrator.py` | 6 | ✅ GOOD | State init, checkpoint round-trip, resume detection, retry alignment |
| `gather/api_implementations.py` | 3 | ⚠️ MEDIUM | Normalization tests good; no API interaction tests |
| `action6_gather.py` | 10+ | ✅ GOOD | Cache helpers, navigation, edge cases, error constructors |
| `action7_inbox.py` | 15+ | ✅ GOOD | Conversation parsing, AI classification, follow-up management, phase tracking |
| `action8_messaging.py` | 15+ | ✅ GOOD | Template loading, circuit breaker, error handling, status detection |
| `action9_process_productive.py` | 15+ | ✅ GOOD | Quality scoring, task creation, AI processing, fact validation |
| `action10.py` | 12+ | ✅ GOOD | Search, display, row builders all tested with mocks |
| `action11_send_approved_drafts.py` | 3 | ✅ EXCELLENT | Real in-memory DB, stub send_fn, verifies DraftReply status, ConversationLog, ConversationMetrics, EngagementTracking all in one test |
| `action12_shared_matches.py` | 2 | ⚠️ MEDIUM | Mock-based, verifies URL construction but no DB persistence test |
| `action13_triangulation.py` | 2 | ⚠️ WEAK | Service init + empty result. Menu test calls `_render_triangulation_menu` but can't test interactively |
| `action14_research_tools.py` | 3 | ✅ GOOD | Delegation + error path + None handling |
| `action15_tree_updates.py` | 4 | ⚠️ WEAK | Only tests dataclass defaults and enum values. `_test_dry_run_mode_prevents_mutations` creates mocks but just asserts `summary.applied == 0` on a fresh `TreeUpdateSummary()` — doesn't actually call `run_tree_updates` |
| `action16_unified_send.py` | 4 | ✅ GOOD | Priority ordering, candidate sorting, defaults, empty-candidate processing |
| `action_review.py` | 8 | ✅ GOOD | Display formatting, diff generation, summary creation, enum values |

---

### 5.1 Tests That Need Live authenticated Sessions

The following tests would benefit from integration tests with a live `SessionManager`:

| File | Test Candidate | Reason |
|---|---|---|
| `action6_gather.py` | `_test_initial_navigation_threads_start_page` | Currently mocks browser; real navigation validation needed |
| `action7_inbox.py` | `_test_fetch_first_page_conversations` | Mocks HTML; real inbox scraping would catch selector drift |
| `action8_messaging.py` | `_test_main_function_with_dry_run` | Mocks API; real dry-run send validation |
| `action12_shared_matches.py` | `test_url_construction` | Verifies URL but never exercises real API path |
| `action15_tree_updates.py` | N/A | Needs real dry-run test against actual `SuggestedFact` + tree service |

**Recommendation:** Create a `tests/integration/` directory with tests marked `@pytest.mark.live_session` for CI exclusion.

---

### 5.2 Specific Test Improvements Needed

**`gather/performance_logging.py`** — Replace all `assert True` with real assertions:
```python
def _test_log_api_performance() -> None:
    with patch.object(logger, 'info') as mock_info:
        log_api_performance(1.5, "test_endpoint", True)
        mock_info.assert_called()
        assert "test_endpoint" in str(mock_info.call_args)
```

**`action15_tree_updates.py`** — `_test_dry_run_mode_prevents_mutations` doesn't test anything real:
```python
# Current: creates a fresh summary and asserts its defaults are 0. This is a no-op test.
summary = TreeUpdateSummary()
assert summary.applied == 0  # This always passes.
```
Should actually call `run_tree_updates(mode=TreeUpdateMode.DRY_RUN)` with an in-memory DB and verify no DB writes occurred.

**`action13_triangulation.py`** — Missing test: `_test_menu_rendering` was declared in the grep but excluded from `module_tests()`. Add it back or remove the dead function.

---

## 6. LINTING & TYPE ANNOTATION ISSUES

### 6.1 f-strings in Logger Calls — MEDIUM (Widespread)

**Files:** `gather/persistence.py` (30+ instances), `gather/rate_persistence.py`, `gather/prefetch.py`, `gather/api_implementations.py`, `action12_shared_matches.py`, `action13_triangulation.py`

```python
# Anti-pattern (eager evaluation even if log level disabled):
logger.debug(f"De-duplicating {len(person_creates_raw)} raw person creates...")

# Correct (lazy evaluation):
logger.debug("De-duplicating %d raw person creates...", len(person_creates_raw))
```

**Impact:** Minor performance cost for debug-level messages in production. The codebase is inconsistent — `action11` uses `%s` formatting correctly, while `persistence.py` uses f-strings throughout.

**Recommendation:** Run a codemod to convert `logger.*(f"...")` to `logger.*("...", ...)` format throughout `actions/`. Ruff rule `G004` can enforce this.

---

### 6.2 Unused Imports — LOW

| File | Import | Status |
|---|---|---|
| `action12_shared_matches.py:4` | `from datetime import timezone` | Unused (uses `UTC` instead) |
| `action13_triangulation.py:21` | `from typing import Optional` | Unused (uses `X \| None` syntax) |
| `action16_unified_send.py:35` | `from sqlalchemy import and_, not_, or_` | `not_` and `or_` appear unused |
| `action11_send_approved_drafts.py:21` | `from typing import Optional` | Unused |

**Recommendation:** Run `ruff check --select F401 actions/` to auto-detect and `--fix` to remove.

---

### 6.3 `from unittest import mock` at Module Level — LOW

**File:** `gather/metrics.py`

```python
from unittest import mock  # Module-level import of test dependency
```

This import runs in production. It should be inside the test functions.

**Recommendation:** Move to inside `_test_*` functions or guard with `if TYPE_CHECKING`.

---

### 6.4 Type Annotations Missing on Key Functions — LOW

| File | Function | Issue |
|---|---|---|
| `action_review.py` | `batch_expire_old_items` | Return type `tuple[int, int]` correct, but `cutoff` has a runtime bug (see §4.5) |
| `gather/persistence.py` | `_handle_unique_constraint_violation` | Missing return type annotation |
| `gather/api_implementations.py:284` | `_parse_details_response` import | Imported as `Any` — loses type information |

---

### 6.5 Hardcoded User-Agent String — LOW

**File:** `gather/api_implementations.py`

```python
def _get_api_headers() -> dict[str, str]:
    return {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)..."}
```

Hardcoded browser UA string. Should be centralized in `config/` or `api/api_constants.py`.

---

## 7. ADDITIONAL OBSERVATIONS

### 7.1 `action_review.py` — `ReviewQueue` Uses Only Static Methods

Every method on `ReviewQueue` is `@staticmethod`. The class has no instance state (the `batch_size` in `__init__` is never used by any static method). This is a namespace, not a class.

**Recommendation:** Either:
1. Convert to a module with top-level functions, or
2. Make the methods instance methods that use `self.batch_size` and `self.db_session`

---

### 7.2 `action12_shared_matches.py` — DB Session Management Anti-Pattern

```python
# Lines 75-90: Gets a new DB session for EACH match in the loop
for i, (match_id, match_uuid, person_name) in enumerate(candidates, 1):
    success = _fetch_and_store_shared_matches(...)
    if success:
        db_session = session_manager.db_manager.get_session()  # New session per iteration!
        with db_transn(db_session) as session:
            ...
```

Creating a new session per loop iteration is wasteful and risks connection pool exhaustion.

**Recommendation:** Use a single session context for the entire batch, or use `session_manager.db_manager.get_session_context()` as a context manager.

---

### 7.3 `gather/orchestrator.py` — Hardcoded 800s Session Refresh Threshold

```python
def _attempt_proactive_session_refresh(self, ...):
    if session_age > 800:  # Hardcoded seconds
        ...
```

The 800-second threshold should be configurable via `config_schema.session.refresh_threshold_seconds` to allow tuning without code changes.

---

## 8. PRIORITIZED ACTION ITEMS

### Immediate (Bug-Risk)
1. **Delete duplicate `needs_ethnicity_refresh`** from `api_implementations.py` — conflicting semantics on `None` input
2. **Move `_parse_details_response`** to `gather/` to fix circular import
3. **Fix `batch_expire_old_items` date math** — use `timedelta` instead of day arithmetic

### This Sprint
4. **Replace fake tests** in `performance_logging.py` with real assertions
5. **Remove `@api_retry`** from `_get_api_headers()`
6. **Reduce decorator stack** on `_validate_session_state` from 5 to 2
7. **Fix `action15` dry-run test** to actually call the function under test

### Next Sprint
8. **Consolidate error categorization** between action8 and action11
9. **Migrate action11 blocking logic** to `should_allow_outbound_to_person()`
10. **Inline `gather/fetch.py`** into orchestrator
11. **Move f-string logging** to %-formatting (enable Ruff G004)
12. **Remove unused imports** with `ruff check --select F401 --fix`

### Backlog
13. Continue `action6_gather.py` decomposition (target <500 lines)
14. Extract `action8_messaging.py` into subpackage
15. Consolidate send paths (Actions 8/9/11 → Action 16)
16. Create `tests/integration/` for live session tests
17. Convert `ReviewQueue` statics to module functions or proper instance methods
