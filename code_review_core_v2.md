# Code Review Report: `core/` Module Group

**Reviewer**: GitHub Copilot
**Date**: 2025-07-01
**Scope**: 20 files in `core/` directory
**Focus**: Duplication, consolidation, complexity, test quality, linting/type issues

---

## Executive Summary

The `core/` package contains **significant cache-layer fragmentation** — 7 files defining overlapping cache abstractions, protocols, data classes, and adapters. There are also concrete consolidation opportunities in validation, DI boilerplate, and test infrastructure. Test quality is generally good (real assertions, real behavior validation), with a few exceptions.

### Key Findings by Severity

| Severity | Count | Category |
|----------|-------|----------|
| **HIGH** | 3 | Cache layer duplication/fragmentation |
| **MEDIUM** | 5 | Code smell / excess complexity |
| **LOW** | 8 | Minor linting, type annotation, style issues |

---

## Per-File Review

---

### 1. `core/draft_content.py` — 198 lines

**Duplication**: None
**Complexity**: Low — well-scoped, single-responsibility
**Test Quality**: **Good** — 3 tests verify round-trip append/strip, missing end marker safety, legacy appendix stripping. All test real behavior.
**Linting/Type Issues**: None detected. Clean use of `dataclass(frozen=True, slots=True)`, modern `X | None` unions.

**Verdict**: ✅ No issues. Model file for the codebase.

---

### 2. `core/dependency_injection.py` — 862 lines

**Duplication**:
- Lines 846–862: Redundant `sys.path` manipulation at bottom of file duplicates lines 22–24. The `try/except ImportError` fallback that does `sys.path.insert(0, project_root)` in both branches is dead code.
- `unittest.TestCase` tests (17 methods) duplicate what `TestSuite` pattern could cover. This is the only file mixing `unittest.TestCase` *and* `TestSuite`.

**Complexity**:
- `_create_instance()` (lines 226–265): Silently swallows dependency resolution failures via fallback to parameterless constructor. This hides real bugs. Should log at WARNING minimum or fail loudly in non-production.
- `DIScope.__exit__` directly mutates private `_services`, `_factories`, `_singletons`, `_interfaces` dicts — fragile if container internals change.

**Test Quality**: **Acceptable** — Tests cover singleton, transient, factory, instance registration, resolution errors, scoping, and integration. However, `test_imports_and_availability` (line 762) is a smoke test (just checks `assertIsNotNone`) — adds no value.

**Linting/Type Issues**:
- Line 313: `inject[T]` uses PEP 695 type parameter syntax — requires Python 3.12+. Document or gate behind version check.
- `Optional` import (line 20) is unused — the file uses `X | None` consistently.
- `Union` import (line 20) is unused.

**Actionable Items**:
1. Remove duplicate `sys.path` block at bottom of file
2. Remove unused `Optional`, `Union` imports
3. Delete `test_imports_and_availability` smoke test
4. Consider migrating `TestDIContainer` to `TestSuite` pattern for consistency

---

### 3. `core/config_validation.py` — 318 lines

**Duplication**: None significant
**Complexity**: Low
**Test Quality**: **Good** — 6 tests covering warning emission, suppression detection (env + argv), missing schema, success path, and import error handling. All use proper mocking.

**Linting/Type Issues**:
- `check_rate_limiting_settings()` (line 35): Empty function body — `config` param is reassigned to `_` for "API compatibility". Should use `_config` parameter name or delete the function entirely if it does nothing.
- `builtins` import only used in tests — move inside test function.
- `from unittest.mock import patch` imported at module level but only used in tests.

**Actionable Items**:
1. Remove or deprecate `check_rate_limiting_settings()` — it's a no-op
2. Move test-only imports inside test functions

---

### 4. `core/caching_bootstrap.py` — 107 lines

**Duplication**:
- `_CachingState` + `_get_caching_state()` reinvents a module-level singleton pattern that could be a simple module-level `_initialized = False` flag.

**Complexity**: Slightly over-engineered for what it does — `_get_caching_state` stores state as a function attribute (line 20: `_get_caching_state._state`), which is a code smell. A plain module-level variable would be simpler and more Pythonic.

**Test Quality**: **Good** — 2 tests verify missing module handling and stateful initialization idempotence.

**Linting/Type Issues**:
- Line 20: `_get_caching_state._state = state` — type checkers will flag setting arbitrary attributes on functions. Use a module-level variable instead.

**Actionable Items**:
1. Replace `_CachingState` class + `_get_caching_state()` with a simple module-level `_initialized: bool = False`

---

### 5. `core/cache_registry.py` — 362 lines ⚠️ HIGH: Consolidation Target

**Duplication**:
- `CacheComponent` duplicates the concept in `CacheFactory` (from `cache_backend.py`). Both maintain a registry of named cache backends with stats/clear/warm operations.
- `_lazy_call()` pattern (line 187) duplicates lazy import patterns used in `analytics_helpers.py`.
- `_unified_cache_stats()` and `_unified_cache_clear()` duplicate the `UnifiedCacheBackendAdapter` pattern in `unified_cache_manager.py`.

**Complexity**: Moderate — the lazy import pattern is appropriate here but the class duplicates `CacheFactory`.

**Test Quality**: **Good** — 3 tests with proper stub registry validation.

**Consolidation Opportunity**:
- **`CacheRegistry` should absorb or delegate to `CacheFactory`** from `cache_backend.py`. Currently they serve parallel purposes — two separate registries for the same concept.

**Actionable Items**:
1. Merge `CacheFactory` into `CacheRegistry` or make one delegate to the other
2. Remove `_unified_cache_stats`/`_unified_cache_clear` static methods (use `CacheComponent` lazy_call pattern instead)

---

### 6. `core/cache_backend.py` — 623 lines ⚠️ HIGH: Consolidation Target

**Duplication**:
- `CacheBackend` protocol duplicates `Cache` protocol in `core/cache/interface.py`. Both define `get()`, `set()`, `delete()`, `clear()`, `get_stats()`, `get_health()`. The interface version adds `exists()`, `invalidate()`, `invalidate_pattern()`, `invalidate_all()`, `keys()`, `get_entry()`.
- `CacheStats` dataclass here overlaps conceptually with stats returned by `UnifiedCacheManager.get_stats()` (dict-based).
- `CacheHealth` dataclass is defined here but also returned by concrete implementations in `cache/adapters.py` and `unified_cache_manager.py`.
- `CacheFactory` class (lines 274–400) duplicates `CacheRegistry` in `cache_registry.py`.
- `ScopedCacheBackend` protocol is not used anywhere in the reviewed files.

**Test Quality**: **Good** — 6 tests covering stats calculations, serialization, protocol checks, and factory operations.

**Linting/Type Issues**:
- `Optional` import (line 18) is unused — file uses `X | None`.
- `_test_cache_factory_operations`: `TestCache` class (lines 530–565) is a near-exact duplicate of `MockCache` class (lines 487–520). Extract to shared test fixture.

**Actionable Items**:
1. **Merge `CacheBackend` and `Cache` protocols into ONE canonical protocol** — the `Cache` protocol in `core/cache/interface.py` is the richer version; promote it and deprecate `CacheBackend`
2. Remove `ScopedCacheBackend` if unused
3. Merge `CacheFactory` into `CacheRegistry`
4. Deduplicate `MockCache`/`TestCache` in tests

---

### 7. `core/app_mode_policy.py` — 194 lines

**Duplication**: None
**Complexity**: Low — clean decision tree with clear mode handling
**Test Quality**: **Good** — 3 tests cover testing allowlist by name, by profile ID override, and production blocking. Tests properly save/restore `config_schema` state.

**Linting/Type Issues**:
- `Optional` import (line 8) is unused — uses `X | None`.

**Actionable Items**:
1. Remove unused `Optional` import

---

### 8. `core/approval_queue.py` — 1580 lines

**Duplication**:
- `_get_owner_profile_id()` (lines 188–210): Tries 3 different sources for the owner's profile ID. This pattern appears in other modules too. Should be centralized.
- Multiple `from core.database import ...` lazy imports inside methods — same imports repeated in 6+ methods.
- `_is_cooldown_expired` and `_is_within_daily_limit` both query `ConversationState` separately — could be combined into a single query.

**Complexity**: **High** — 1580 lines is too large for a single module. The `ApprovalQueueService` class handles: queueing, validation, moderation, priority calculation, auto-approval logic, approval/rejection, statistics, expiry, and queue retrieval. Should be split.

**Test Quality**: Tests were partially reviewed (lines 1–800). The reviewed tests cover queue validation, AI moderation integration, and auto-approval logic properly.

**Linting/Type Issues**:
- `Optional` import (line 23) unused — uses `X | None`.
- `_ai_reasoning` parameter (line 230) uses underscore prefix suggesting it's unused, but it IS used in the metadata block. Remove the underscore prefix — these are public API parameters with private naming convention.
- Same issue with `_context_summary`, `_research_suggestions`, `_research_metadata`.

**Actionable Items**:
1. **Split into submodules**: `approval_queue.py` (core service), `auto_approval.py` (auto-approve logic), `approval_stats.py` (statistics)
2. Remove underscore prefix from `_ai_reasoning`, `_context_summary`, `_research_suggestions`, `_research_metadata` parameters
3. Remove unused `Optional` import
4. Centralize `_get_owner_profile_id()` into a shared utility
5. Combine `_is_cooldown_expired` + `_is_within_daily_limit` into single ConversationState query

---

### 9. `core/analytics_helpers.py` — 175 lines

**Duplication**:
- `_build_fake_module()` + `_module_patch()` test helpers duplicate patterns that could be in `testing/test_utilities.py`.

**Complexity**: Low — clean lazy-import wrappers
**Test Quality**: **Good** — 5 tests verify callable resolution, analytics setter invocation, missing dependency handling, and metrics factory.

**Linting/Type Issues**:
- `Optional` import (line 7) unused.
- Bottom: `import sys as _sys` — unnecessary alias.

**Actionable Items**:
1. Remove unused `Optional` import
2. Replace `import sys as _sys` with `import sys`

---

### 10. `core/background_scheduler.py` — 567 lines

**Duplication**: None significant
**Complexity**: Moderate — well-structured scheduler with proper thread safety
**Test Quality**: **Good** — 8 tests covering initialization, registration, enable/disable, start/stop lifecycle, execution tracking, error handling, singleton pattern, and multi-task status. Uses `time.sleep()` for timing-dependent tests — acceptable for scheduler testing.

**Linting/Type Issues**:
- `Optional` import (line 21) unused.
- Line 487: `global _scheduler` — modifying module global in test is fragile. Reset via a dedicated `_reset_for_testing()` function instead.
- `lambda: None` on line 417 — Ruff `E731` may flag this. Use `def noop(): pass` instead.

**Actionable Items**:
1. Remove unused `Optional` import
2. Add `_reset_for_testing()` method instead of directly mutating global

---

### 11. `core/workflow_actions.py` — 619 lines

**Duplication**: **Significant pattern duplication**
- `send_messages_action`, `send_approved_drafts_action`, `run_unified_send_action` all follow identical pattern:
  1. Log start
  2. Navigate to base URL
  3. Call lazy-imported function
  4. Check result
  5. Log success/failure
  6. Return bool
- `_run_action6_gather`, `_run_action7_inbox`, `_run_action8_send_messages`, `_run_action9_process_productive` — identical try/except/log pattern repeated 4 times.

**Complexity**: Medium — mostly boilerplate that could be factored into a generic wrapper.

**Test Quality**: **Acceptable** — 7 tests mostly check decorator application and session guards. `_test_srch_inbox_actn_decorator_applied` and similar just verify `result is False` for None session — these are thin but valid guard tests.

**Linting/Type Issues**:
- Multiple `import sys` statements at bottom of file in `if __name__` block — `sys` already imported conditionally.

**Actionable Items**:
1. **Extract a generic `_run_action_with_navigation()` helper** that takes: action label, lazy getter, session_manager, optional URL/selector → eliminates 4 duplicated wrappers
2. **Extract a generic `_action_with_base_nav()` decorator/wrapper** for `send_messages_action`, `send_approved_drafts_action`, `run_unified_send_action` → eliminates 3 duplicated functions

---

### 12. `core/venv_bootstrap.py` — 62 lines

**Duplication**: None
**Complexity**: Low — simple utility
**Test Quality**: **No tests** — this file has no `module_tests()` function. Given it re-execs the Python process, unit testing is tricky but at minimum the venv detection logic should be tested.

**Linting/Type Issues**: Clean.

**Actionable Items**:
1. Add tests for venv detection logic (the `in_venv` check) without actually re-executing

---

### 13. `core/validation_factory.py` — 607 lines

**Duplication**:
- `validate_api_prerequisites()` re-checks session manager via `validate_session_manager()` then does additional work — could chain validators using the composite pattern defined later in the same file.

**Complexity**: Low — factory pattern well-applied
**Test Quality**: **Good** — 6 tests covering required fields, types, ranges, session manager, person data, and composite validators. All test real validation behavior.

**Linting/Type Issues**:
- `Optional` import (line 9) is unused.

**Actionable Items**:
1. Remove unused `Optional` import
2. Consider using `create_composite_validator` internally for `validate_api_prerequisites`

---

### 14. `core/type_definitions.py` — 187 lines

**Duplication**:
- `CacheKey = str` defined here AND in `core/cache/interface.py` (line 35).
- `CacheTTL = int` defined here AND in `core/cache/interface.py` (line 36).

**Complexity**: Low — pure type definitions
**Test Quality**: **Fake test** — `_test_module_integrity()` just returns `True`. Validates nothing. For a type definitions file this is somewhat acceptable, but the function name is misleading.

**Linting/Type Issues**:
- `Union` import (line 7) is unused.

**Actionable Items**:
1. Remove duplicate `CacheKey`/`CacheTTL` definitions — canonical home should be `core/type_definitions.py`, and `core/cache/interface.py` should import from there
2. Either delete the smoke test or add actual validation (e.g., verify TypedDict fields, verify constants are defined)
3. Remove unused `Union` import

---

### 15. `core/unified_cache_manager.py` — 1203 lines ⚠️ HIGH: Consolidation Target

**Duplication**:
- `CacheEntry` dataclass (line 37) duplicates `CacheEntry` in `core/cache/interface.py` — different fields (`data` vs `value`, `ttl_seconds` vs `ttl`, no `version`/`size_bytes`). Two incompatible `CacheEntry` classes in the same project.
- `UnifiedCacheBackendAdapter` (lines 498–538) wraps `UnifiedCacheManager` to conform to `CacheBackend` protocol — this adapter is only needed because the cache protocols are fragmented.
- `generate_cache_key()` (line 570) duplicates key generation logic in `system_cache.py` (`_get_api_cache_key`, `_get_query_cache_key`).
- Hit rate calculation repeated inline in `get_stats()` — already provided by `CacheStats.hit_rate` property.
- `_get_default_ttl()` hardcodes TTL values that overlap with `create_ancestry_cache_config()` in the same file.

**Complexity**: **High** — 1203 lines for a cache manager. The class does: get/set/delete, invalidation (4 strategies), statistics tracking, LRU eviction, deep copy isolation, key generation, protocol adaptation, and 15+ test functions.

**Test Quality**: **Excellent** — 15 tests covering: entry dataclass, deep copy isolation, service auto-creation, overwrite behavior, clear operation, singleton, realistic access patterns, basic operations, statistics tracking, TTL expiration, invalidation, size limits, key generation, thread safety, and endpoint statistics. Best-tested file in the review.

**Linting/Type Issues**:
- `Optional` import (line 18) unused.
- Debug `print()` statements in `_test_cache_size_limit_enforcement`.

**Actionable Items**:
1. **Unify `CacheEntry` definitions** — one canonical `CacheEntry` for the project
2. Remove `UnifiedCacheBackendAdapter` by having `UnifiedCacheManager` implement the unified protocol directly
3. Extract `generate_cache_key()` to a shared cache utils module
4. Remove debug `print()` statements from tests
5. Remove inline hit rate calculations where `CacheStats.hit_rate` is available

---

### 16. `core/system_cache.py` — 987 lines

**Duplication**:
- `APIResponseCache._get_api_cache_key()` duplicates `generate_cache_key()` from `unified_cache_manager.py`.
- `DatabaseQueryCache._get_query_cache_key()` — another key generation variant.
- `MemoryOptimizer.optimize_memory()` iterates `gc.get_objects()` and checks for dead `weakref.ref` objects — this is a **dangerous** pattern that can cause issues with the GC. The `del obj` inside the loop doesn't actually free the weakref.
- `clear_system_caches()` manually iterates cache keys with prefix matching — this is what `invalidate_pattern()` in the unified cache should do.
- The `if __name__ == "__main__"` block appears **twice** (line 902 and at end of file).

**Complexity**: **High** — 987 lines covering API caching, DB query caching, memory optimization, decorators, stats, clearing, warming, and tests. Too many concerns.

**Test Quality**: **Mixed** — Some tests (`_test_database_query_cache_initialization`, `_test_memory_optimizer_initialization`) just check attributes exist — these are weak. The decorator tests and performance tests are better.

**Linting/Type Issues**:
- `Optional` import (line 48) unused.
- `Union` import (line 48) unused.
- `weakref` import used dangerously in `optimize_memory()`.
- `cast(Any, psutil)` — unnecessary cast.
- Accessing `_api_cache._lock` and `_db_cache._lock` directly in decorators — violates encapsulation.

**Actionable Items**:
1. **Move `APIResponseCache` and `DatabaseQueryCache` to use `UnifiedCacheManager`** instead of maintaining separate cache implementations
2. Remove `MemoryOptimizer.optimize_memory()` weakref cleanup — it's ineffective and risky
3. Remove duplicate `if __name__ == "__main__"` block
4. Replace attribute-checking tests with behavioral tests
5. Remove unused `Optional`, `Union` imports
6. Fix encapsulation violation — add public methods instead of accessing `_lock` directly

---

### 17. `core/cache/interface.py` — 517 lines

**Duplication**:
- `CacheConfig` dataclass has significant overlap with constructor params of `UnifiedCacheManager` and `MemoryCache`.
- `CacheEntry` duplicates `CacheEntry` in `unified_cache_manager.py` (different field names/semantics).
- `Cache` protocol duplicates `CacheBackend` protocol in `cache_backend.py` — `Cache` is the superset with `exists()`, `invalidate_pattern()`, `keys()`, `get_entry()`.
- `CacheKey` and `CacheTTL` type aliases duplicate definitions in `core/type_definitions.py`.

**Test Quality**: **Good** — 10 tests covering config defaults/custom, entry expiration/age, all invalidation pattern types, and protocol structural check.

**Linting/Type Issues**:
- `Optional` import (line 25) unused.
- `Path` import (line 24) unused.
- `sys` import (line 22) unused.

**Actionable Items**:
1. **This should be THE canonical cache interface** — merge `CacheBackend` from `cache_backend.py` into this
2. Import `CacheKey`/`CacheTTL` from `core/type_definitions.py` instead of redefining
3. Remove unused imports (`Optional`, `Path`, `sys`)

---

### 18. `core/cache/adapters.py` — 912 lines

**Duplication**:
- `MemoryCache` here vs `UnifiedCacheManager` — both are in-memory thread-safe caches with LRU eviction, TTL support, and stats tracking. **Two parallel implementations of the same thing.**

**Complexity**: Moderate — individual classes are well-scoped.

**Test Quality**: **Good** — 10 tests covering basic operations, TTL, LRU eviction, pattern invalidation, statistics, namespace, TTL cleanup, null cache, disk adapter fallback, and entry metadata.

**Linting/Type Issues**:
- `Optional` import (line 17) unused.

**Actionable Items**:
1. **Choose ONE in-memory cache implementation** — either `MemoryCache` (richer protocol) or `UnifiedCacheManager` (production-used). The other should be deprecated.
2. Remove unused `Optional` import

---

### 19. `core/__init__.py` — 308 lines

**Duplication**:
- `DummyComponent` fallback class (lines 88–97) creates 20+ dummy objects for error fallback — this is a lot of dead-code scaffolding for an unlikely import failure scenario.

**Complexity**: Medium — the try/except import block with 35+ fallback dummy assignments is verbose.

**Test Quality**: **Good** — 4 tests verify package structure, component imports, DI imports, and error handling imports.

**Linting/Type Issues**:
- The `DummyComponent` assignments will cause type errors downstream if any code actually receives them — `DummyComponent("SessionManager")` is not `type[SessionManager]`.

**Actionable Items**:
1. Consider removing the `DummyComponent` fallback — if core package imports fail, the application can't work anyway. Let it crash with a clear `ImportError`.

---

### 20. `core/__main__.py` — 179 lines

**Duplication**: Test functions duplicate checks already done in `core/__init__.py` tests.
**Complexity**: Low
**Test Quality**: **Weak** — Tests just verify modules can be imported and classes are not None. These are smoke tests that duplicate `__init__.py` coverage.

**Linting/Type Issues**: Clean.

**Actionable Items**:
1. Consider removing duplicate import tests — `core/__init__.py` already tests this

---

## Cross-File Consolidation Opportunities

### 🔴 CRITICAL: Cache Layer Unification

The project has **7 cache-related files** with significant overlap:

| File | Purpose | Lines |
|------|---------|-------|
| `cache_backend.py` | Protocol + CacheStats + CacheHealth + CacheFactory | 623 |
| `cache_registry.py` | Registry of cache components | 362 |
| `cache/interface.py` | Protocol + CacheConfig + CacheEntry + InvalidationPattern | 517 |
| `cache/adapters.py` | MemoryCache, TTLCache, DiskCacheAdapter, NullCache | 912 |
| `unified_cache_manager.py` | UnifiedCacheManager (another in-memory cache) | 1203 |
| `system_cache.py` | APIResponseCache, DatabaseQueryCache, MemoryOptimizer | 987 |
| `caching_bootstrap.py` | Bootstrap helper | 107 |

**Total: 4,711 lines** across 7 files for caching functionality.

**Duplicated Concepts**:
- **3 `CacheEntry` classes** (unified_cache_manager, cache/interface, implicit in system_cache)
- **2 cache protocols** (`CacheBackend` in cache_backend.py, `Cache` in cache/interface.py)
- **2 cache registries** (`CacheFactory` in cache_backend.py, `CacheRegistry` in cache_registry.py)
- **2 in-memory caches** (`MemoryCache` in adapters.py, `UnifiedCacheManager`)
- **3 key generation functions** (unified_cache_manager, system_cache API, system_cache DB)
- **2 `CacheKey`/`CacheTTL` definitions** (type_definitions.py, cache/interface.py)

**Recommended Consolidation**:

1. **Single protocol**: Merge `CacheBackend` into `Cache` (keep richer interface from `cache/interface.py`)
2. **Single CacheEntry**: Canonical definition in `cache/interface.py`, delete others
3. **Single registry**: Merge `CacheFactory` into `CacheRegistry`
4. **Single in-memory cache**: Keep `MemoryCache` from `adapters.py` (cleaner protocol conformance), make `UnifiedCacheManager` use it internally or replace
5. **Single key generation**: Extract to shared utility
6. **Estimated reduction**: ~1,500–2,000 lines removable through consolidation

---

### 🟡 MEDIUM: Unused Import Pattern

**10 of 20 files** have unused `Optional` imports. The codebase has migrated to `X | None` syntax but hasn't cleaned up old imports.

**Affected files**: `dependency_injection.py`, `cache_backend.py`, `cache_registry.py`, `app_mode_policy.py`, `analytics_helpers.py`, `background_scheduler.py`, `validation_factory.py`, `unified_cache_manager.py`, `system_cache.py`, `cache/interface.py`, `cache/adapters.py`

**Fix**: `ruff check --fix .` should resolve all of these automatically.

---

### 🟡 MEDIUM: `workflow_actions.py` Action Wrapper Duplication

4 `_run_action*` functions and 3 `*_action` functions follow identical patterns (navigate → call → log → return). A generic helper would reduce ~150 lines to ~30.

---

### 🟢 LOW: Test Quality Issues

| File | Issue |
|------|-------|
| `type_definitions.py` | `_test_module_integrity()` returns `True` unconditionally — fake test |
| `__main__.py` | Import-only tests duplicate `__init__.py` coverage |
| `dependency_injection.py` | `test_imports_and_availability` is import smoke test (low value) |
| `system_cache.py` | `_test_*_initialization` tests only check attribute existence |

---

## Summary of Actionable Items (Prioritized)

### Priority 1: Cache Consolidation (HIGH impact, ~1500 LOC reduction)
- [ ] Merge `CacheBackend` protocol into `Cache` protocol (keep `cache/interface.py` as canonical)
- [ ] Unify `CacheEntry` to single canonical definition in `cache/interface.py`
- [ ] Merge `CacheFactory` into `CacheRegistry`
- [ ] Evaluate replacing `UnifiedCacheManager` internal storage with `MemoryCache` from adapters
- [ ] Move `system_cache.py` decorators to use unified cache layer
- [ ] Extract `generate_cache_key()` to shared location
- [ ] Remove `ScopedCacheBackend` protocol if unused

### Priority 2: Code Quality (MEDIUM impact)
- [ ] Split `approval_queue.py` (1580 lines) into submodules
- [ ] Extract generic action wrapper in `workflow_actions.py` (~150 LOC reduction)
- [ ] Fix `approval_queue.py` parameter naming (`_ai_reasoning` → `ai_reasoning`)
- [ ] Remove `MemoryOptimizer.optimize_memory()` weakref cleanup (ineffective, risky)
- [ ] Remove `DummyComponent` fallback in `__init__.py`

### Priority 3: Cleanup (LOW impact)
- [ ] Run `ruff check --fix .` to remove 10+ unused `Optional`/`Union` imports
- [ ] Remove duplicate `sys.path` blocks in `dependency_injection.py`
- [ ] Remove/fix `check_rate_limiting_settings()` no-op in `config_validation.py`
- [ ] Replace `_CachingState` with module-level flag in `caching_bootstrap.py`
- [ ] Delete `test_imports_and_availability` smoke test in `dependency_injection.py`
- [ ] Fix duplicate `if __name__ == "__main__"` in `system_cache.py`
- [ ] Add tests for `venv_bootstrap.py`
- [ ] Fix or delete `_test_module_integrity` fake test in `type_definitions.py`
- [ ] Remove debug `print()` statements from `unified_cache_manager.py` tests
- [ ] Move test-only imports inside test functions in `config_validation.py`
