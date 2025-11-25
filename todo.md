# Ancestry Automation - Technical Roadmap

> **Quality guard**: Run `ruff check` and `pyright` after every milestone. Never introduce `# noqa` or `type: ignore` suppressions.

---

## Session Summary (November 29, 2025)

### ✅ Completed This Session

1. **AI Interface Decomposition - Steps 1-3 (Track 6)**
   - Created `ai/providers/base.py` with `ProviderRequest`/`ProviderResponse` dataclasses plus adapter registration helpers, then added `ai/providers/deepseek.py` and `ai/providers/gemini.py` so each provider encapsulates availability checks, request construction, and response normalization.
   - Added `ai/prompts.py` to centralize JSON prompt loading, experiment routing, and telemetry forwarding; `ai_interface.py` now imports prompts/providers from this package and records experiment payloads through a single helper.
   - Refactored `ai_interface.py` to register adapters via `_register_provider()`, route calls through `_route_ai_provider_call()`, and treat provider errors uniformly; telemetry payloads now serialize through the new prompt helper, removing the tightly coupled direct call.
   - Verified coverage with `python run_all_tests.py` (115 modules, 100% quality) and `npx pyright` (clean) after the refactor.

2. **Cache Backend Protocol Compliance & Pyright Zero-Warning Gate**
   - Wrapped `cache.py`, `performance_cache.py`, and `core/unified_cache_manager.py` implementations with lightweight `CacheBackend` adapters so the `CacheFactory` only registers protocol-compatible objects; added typed recommendation lists and stricter `_max_size` handling in `PerformanceCache`.
   - Hardened `core/cache_backend.py` to safely probe optional `get_stats_typed()` implementations and aligned the in-module `TestCache` helpers with the protocol signature.
   - Result: `npx pyright` now reports **0 warnings** without suppressions, eliminating the previous cache backend incompatibilities.

3. **AI Provider Failover Orchestration - Track 6 Step 4**
   - Added `ai_provider_fallbacks` config plus `AI_PROVIDER_FALLBACKS` env parsing so operators can define an explicit failover list (default: deepseek → gemini → moonshot → local_llm → grok → inception → tetrate).
   - Replaced the moonshot-only chain with adapter-aware routing in `ai_interface._call_ai_model()`, automatically skipping providers that lack credentials or dependencies and reusing the configured order for every action.
   - Added `_test_configurable_provider_failover()` to the module TestSuite to simulate adapter failures and prove the orchestration falls back cleanly without leaving cached state behind.

### Assessment: Did We Do the Right Things?

**Yes** – Track 6 now has a concrete provider/prompt abstraction with passing tests, and the cache infrastructure finally satisfies both the protocol contract and Pyright's structural typing, giving us a clean lint/type baseline for the next todo items.

---

## Session Summary (November 28, 2025)

### ✅ Completed This Session

1. **API Utils Consolidation - Steps 2-3 Complete (Track 5)**
   - **Step 2**: Extended `core/api_manager.py` with unified `request()` method
     - Added `RequestConfig` and `RequestResult` dataclasses
     - Added `RetryPolicy` enum (STANDARD, AGGRESSIVE, MINIMAL)
     - Integrated rate limiting via `session_manager.rate_limiter`
     - Reduced complexity from 20 → 9 (threshold 10)
     - Achieved 100% code quality (was 94.2/100)
   - **Step 3**: Migrated all 14 api_utils.py call sites to APIManager.request()
     - Created adapter layer with `_call_via_api_manager()` and `_call_api_request_unified()`
     - Added feature flag `USE_API_MANAGER_REQUEST = True` for gradual migration
     - All 14 call_* functions now route through APIManager.request()
     - Zero breaking changes - all existing imports work unchanged
   - **Testing**: All 115 modules pass at 100% quality, 939 tests, 100% success rate

### Assessment: Did We Do the Right Things?

**Yes** - Track 5 Steps 2-3 successfully unified API request handling:
- Eliminated code duplication across 14 API endpoint functions
- Single source of truth for rate limiting, retries, cookie sync
- Adapter pattern enables zero-downtime migration
- All consumers (api_search_utils, api_search_core, actions 6/8/9) now benefit from unified logic

---

## Previous Session Summary (November 25, 2025)

### ✅ Completed That Session

1. **Fixed Circular Import in `actions/__init__.py`**
   - Changed direct import of `coord` from `action6_gather` to lazy import using `__getattr__`
   - Prevents ImportError when `action6_gather` imports from `actions.gather.metrics`
   - Uses `TYPE_CHECKING` for static type analysis
   - All 115 test modules pass

2. **API Utils Consolidation - Step 1 Inventory (Track 5)**
   - Catalogued all 14 `call_*` API functions in `api_utils.py` (3780 lines)
   - Documented core request logic in `utils.py` (`_api_req`, `_api_req_impl`, `ApiRequestConfig`)
   - Identified 14 import sites across 6 modules
   - Current `api_manager.py` has connection pooling and cookie sync but no unified `request()` method

### Assessment: Did We Do the Right Things?

**Yes** - Track 5 Step 1 provides foundation for API consolidation:
- Complete function inventory enables targeted migration
- Understanding of request flow guides `APIManager.request()` design
- Import site mapping ensures no consumers are missed

---

## Previous Session Summary (November 27, 2025)

### ✅ Completed That Session

1. **Cache Stack Unification - Step 3 Protocol Integration (Track 4)**
   - Updated `cache.py` BaseCacheModule to implement CacheBackend protocol methods
   - Updated `core/unified_cache_manager.py` UnifiedCacheManager with protocol methods
   - Updated `performance_cache.py` PerformanceCache with protocol methods
   - All 3 primary cache implementations now register with `CacheFactory`
   - `CacheFactory.get_all_stats()` aggregates stats from all backends
   - Enhanced `get_all_stats()` to support both `get_stats_typed()` and legacy `get_stats()` patterns
   - 115 test modules pass with 100% success rate

2. **Cache Stack Unification - Step 4 Assessment (Track 4)**
   - Analyzed module dependencies - all 6 cache modules still serve distinct purposes:
     - `cache.py` - Core diskcache persistence with BaseCacheModule
     - `cache_manager.py` - Higher-level API caching coordination
     - `core/cache_registry.py` - Registry pattern for cache orchestration
     - `core/session_cache.py` - Session-specific caching
     - `core/unified_cache_manager.py` - In-memory service/endpoint-aware caching
     - `performance_cache.py` - Specialized GEDCOM caching with adaptive sizing
   - No modules can be removed - each serves unique functionality
   - **Achievement**: Unified monitoring via CacheFactory without consolidation

### Assessment: Did We Do the Right Things?

**Yes** - Track 4 now complete with practical outcome:
- CacheBackend protocol enables unified stats collection
- CacheFactory provides single point for cache monitoring
- Import cycles partially addressed through protocol-based dependency inversion
- All existing functionality preserved with zero breaking changes

---

## Previous Session Summary (November 26, 2025)

### ✅ Completed That Session

1. **Cache Stack Unification - Steps 1-2 (Track 4)**
   - Analyzed all 6 cache modules (5,678 lines total)
   - Identified 28 import sites and 3 import cycles
   - Created `core/cache_backend.py` with `CacheBackend` protocol
   - Defined `CacheStats` and `CacheHealth` dataclasses
   - Added `CacheFactory` for centralized backend registration
   - 6 tests covering stats, serialization, protocol checks

2. **Test Coverage Gaps (Track 7) - COMPLETE**
   - Added TestSuite tests to 4 remaining modules
   - Fixed circular import in `workflow_actions.py` using lazy imports
   - All test modules pass with 100% success rate

---

## Previous Session Summary (November 25, 2025)

### ✅ Completed This Session

1. **Messaging Workflow Helpers Consolidation**
   - Created `messaging/workflow_helpers.py` with 12 helper functions + 13 tests
   - Extracted `build_safe_column_value`, `cancel_pending_messages_on_status_change`, `cancel_pending_on_reply`, `log_conversation_state_change` plus 8 additional helpers
   - Removed ~200+ lines of duplicated code from Actions 7, 8, 9
   - All action test suites pass (Action 7: 23/23, Action 8: 36/36, Action 9: 14/14)

2. **Message Types Module Extraction**
   - Created `messaging/message_types.py` with 7 tests
   - Extracted `MESSAGE_TYPES`, `MESSAGE_TRANSITION_TABLE`, `determine_next_message_type()` from Action 8
   - Added utility functions: `is_terminal_message_type()`, `get_message_type_category()`
   - Removed ~80 lines of code from `action8_messaging.py`
   - Updated `messaging/__init__.py` to export all message type constants

3. **Messaging Package Documentation**
   - Documented `messaging/message_types.py` in README.md
   - Assessed template loading - not extracted (tight DB coupling by design)
   - Marked Messaging/Inbox Helpers Extraction track as COMPLETE

4. **Pylance Configuration**
   - Added `python.analysis.extraPaths` to `.vscode/settings.json`
   - Fixed import resolution errors across messaging package

5. **Action 10 GEDCOM/API Consolidation** (prior session, verified working)
   - Merged `action10_wrapper.py` into `action10.py`
   - Exposed `ComparisonConfig`/`ComparisonResults` as public dataclasses
   - Ported 11 regression tests into action10 module
   - Updated `main.py` to import from `action10` directly
   - Deleted `action10_wrapper.py`

### Assessment: Did We Do the Right Things?

**Yes** - The changes align with the documented refactoring goals:
- Reduced code duplication across messaging actions
- Improved testability with dedicated module tests
- Maintained backward compatibility (all existing tests pass)
- Followed the established patterns (TestSuite, standard runner, etc.)

---

## Active Refactoring Tracks

### 1. ✅ Action 6 Decomposition [COMPLETE]

> **Goal**: Break 9280-line monolith into testable packages

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ | Discovery and dependency mapping |
| Phase 2 | ✅ | Package skeleton (`actions/gather/`) |
| Phase 3 | ✅ | Incremental module extraction |
| Phase 4 | ✅ | Cleanup and documentation |

**Extracted modules**:
- `actions/gather/metrics.py` - telemetry + logging
- `actions/gather/prefetch.py` - API prefetch orchestration
- `actions/gather/persistence.py` - batch persistence helpers
- `actions/gather/checkpoint.py` - checkpoint/resume logic
- `actions/gather/orchestrator.py` - main coord() delegation

---

### 2. ✅ Messaging/Inbox Helpers Extraction [COMPLETE]

> **Goal**: Reduce Actions 7–9 to <2k lines each via shared `messaging/` package

| Step | Status | Description |
|------|--------|-------------|
| Step 1 | ✅ | `build_safe_column_value` + enum coercion helpers |
| Step 2 | ✅ | Conversation state & engagement timing helpers |
| Step 3 | ✅ | Nav-guard helpers (already in `core/session_guards.py`) |
| Step 4 | ✅ | Message type constants + state machine |
| Step 5 | ✅ | Documentation in README.md |

**Completed helpers** (in `messaging/workflow_helpers.py`):
- `safe_column_value()`, `build_safe_column_value()` - enum-aware column extraction
- `cancel_pending_messages_on_status_change()`, `cancel_pending_on_reply()` - state management
- `log_conversation_state_change()` - audit logging
- `calculate_days_since_login()`, `determine_engagement_tier()`, `calculate_adaptive_interval()` - timing
- `is_tree_creation_recent()`, `has_message_after_tree_creation()`, `detect_status_change_to_in_tree()` - tree detection
- `calculate_follow_up_action()` - workflow decisions

**Completed constants** (in `messaging/message_types.py`):
- `MESSAGE_TYPES`, `MESSAGE_TYPES_ACTION8` - message type constants
- `MESSAGE_TRANSITION_TABLE` - state machine transitions
- `CORE_REQUIRED_TEMPLATE_KEYS` - validation keys
- `determine_next_message_type()` - state machine logic
- `is_terminal_message_type()`, `get_message_type_category()` - utility functions

**Not extracted** (by design):
- Template loading (`load_message_templates()`) - Tight DB coupling via `session_manager.get_db_conn_context()`. Not worth extracting as it requires session infrastructure.

---

### 3. ✅ Action 10 Consolidation [COMPLETE]

> **Goal**: Merge `action10_wrapper.py` back into `action10.py`

- [x] Merged comparison pipeline helpers
- [x] Exposed `ComparisonConfig`/`ComparisonResults` dataclasses
- [x] Ported TestSuite coverage (11 tests)
- [x] Updated main.py imports
- [x] Deleted wrapper file

---

## Future Refactoring Priorities

### 4. Cache Stack Unification [COMPLETE] ✅

> **Goal**: Unified cache monitoring via CacheBackend protocol
> **Outcome**: Single `CacheFactory` aggregates stats from all cache backends

**Problem Addressed**: 6 cache modules with overlapping functionality lacked unified monitoring.

**Solution Implemented**:
- Created `CacheBackend` protocol with standardized `get_stats()`, `get_health()`, `warm()` methods
- Created `CacheStats` and `CacheHealth` dataclasses for consistent metrics
- Created `CacheFactory` for centralized backend registration
- Ported 3 primary backends: `disk_cache`, `unified_cache`, `performance_cache`

**Cache Modules (6 total, ~5,678 lines)** - All preserved with protocol integration:
| Module | Lines | CacheFactory Name | Status |
|--------|-------|-------------------|--------|
| `cache.py` | 1624 | `disk_cache` | ✅ Protocol methods added |
| `core/unified_cache_manager.py` | 1060 | `unified_cache` | ✅ Protocol methods added |
| `performance_cache.py` | 946 | `performance_cache` | ✅ Protocol methods added |
| `cache_manager.py` | 835 | N/A | Coordinator, uses cache.py |
| `core/cache_registry.py` | 369 | N/A | Registry pattern overlay |
| `core/session_cache.py` | 844 | N/A | Extends BaseCacheModule |

| Step | Status | Description |
|------|--------|-------------|
| 1 | ✅ | Inventory all cache entry points and consumers |
| 2 | ✅ | Design `CacheBackend` protocol in `core/cache_backend.py` |
| 3 | ✅ | Port consumers with protocol methods and CacheFactory registration |
| 4 | ✅ | Assessed removal - modules serve distinct purposes, kept all |

---

### 5. ✅ API Utils Consolidation [COMPLETE]

> **Goal**: Single HTTP pipeline via `core/api_manager.py`
> **Effort**: ~2-3 sessions | **Priority**: Medium | **Status**: ✅ COMPLETE

**Problem**: `api_utils.py` (3780 lines) duplicates rate limiting, retry, and cookie sync logic that should live in `SessionManager.api_manager`.

**Solution**: Extended `APIManager.request()` with unified configuration, migrated all 14 call_* functions via adapter pattern with zero breaking changes.

#### Step 1: Consumer Inventory (COMPLETE)

**API Functions in `api_utils.py` (14 total)**:
| Function | Used By | Purpose |
|----------|---------|---------|
| `call_suggest_api` | api_search_utils | Search suggestions |
| `call_facts_user_api` | api_search_utils | User facts |
| `call_getladder_api` | api_search_utils | Relationship ladder HTML |
| `call_discovery_relationship_api` | api_search_core | Discovery relationships |
| `call_treesui_list_api` | api_search_core | Tree UI list |
| `call_newfamilyview_api` | api_search_utils | New family view |
| `call_relation_ladder_with_labels_api` | api_search_core, action6_gather | Relationship labels |
| `call_send_message_api` | action8_messaging, action9 | Send messages |
| `call_profile_details_api` | Various | Profile details |
| `call_header_trees_api_for_tree_id` | Various | Tree ID lookup |
| `call_tree_owner_api` | Various | Tree owner lookup |
| `call_enhanced_api` | Various | Enhanced API wrapper |
| `call_edit_relationships_api` | api_search_utils | Edit relationships |
| `call_relationship_ladder_api` | Various | Relationship ladder |

**Core Request Logic in `utils.py`**:
- `_api_req()` - Legacy positional parameter wrapper (line 2442)
- `_api_req_impl()` - Main implementation with `ApiRequestConfig` (line 2493)
- `_prepare_api_request()` - Request preparation (line 1768)
- `_execute_api_request()` - Request execution with retries (line 1856)
- `ApiRequestConfig` dataclass - Configuration container (line 477)

#### Step 2: Extend APIManager (COMPLETE)

**New `core/api_manager.py` additions (1350+ lines)**:
- `RequestConfig` dataclass - Unified request configuration with sensible defaults
- `RequestResult` dataclass - Structured response with `is_json`, `json`, `text` helpers
- `RetryPolicy` enum - Predefined policies: `NONE`, `API` (3 retries), `RESILIENT` (5 retries)
- `APIManager.request()` - Unified entry point integrating:
  - Rate limiting via `AdaptiveRateLimiter` parameter
  - Cookie synchronization via `BrowserManager` parameter
  - Configurable retry with exponential backoff and jitter
  - Comprehensive metrics recording

**Tests Added (4 new tests, 13 total)**:
- `_test_request_config_dataclass` - Validates defaults and retry policy presets
- `_test_request_result_dataclass` - Validates helper properties
- `_test_request_method_available` - Validates method signature
- `_test_retry_policy_enum` - Validates enum values

**Consumers (14 import sites)**:
- `api_search_utils.py` - 4 imports
- `api_search_core.py` - 2 imports
- `action6_gather.py` - 2 imports (lazy)
- `action8_messaging.py` - 1 import
- `action9_process_productive.py` - 1 import (lazy)
- `core/session_manager.py` - 2 imports (lazy)

| Step | Status | Description |
|------|--------|-------------|
| 1 | ✅ | Catalogue all `_api_req` consumers (Actions 6–10, messaging, telemetry) |
| 2 | ✅ | Extend `APIManager` with unified `request()`, retry policies, rate limiting |
| 3 | ✅ | Migrate callers to `session_manager.api_manager.request()` via adapter pattern |
| 4 | ✅ | All 14 call_* functions now route through APIManager.request() with feature flag |

**Outcome**: All 115 modules pass at 100% quality (939 tests, 100% success rate). Zero breaking changes - existing imports work unchanged.

---

### 6. AI Interface Decomposition [IN PROGRESS]

> **Goal**: `ai/` package with provider adapters, prompt templating, telemetry
> **Effort**: ~2 sessions | **Priority**: Medium

**Progress (Nov 29)**:
- Added `ai/providers/base.py` with adapter registration helpers plus DeepSeek/Gemini provider modules and prompt loader `ai/prompts.py`.
- Refactored `ai_interface.py` to route provider calls through adapters, load prompts via the new helper, and standardize telemetry payloads.
- Achieved clean `ruff` + `pyright` runs after the refactor; provider modules and prompt loader include their own guardrails.

| Step | Status | Description |
|------|--------|-------------|
| 1 | ✅ | Create `ai/providers/` with Gemini, DeepSeek adapters |
| 2 | ✅ | Create `ai/prompts.py` for template loading and substitution |
| 3 | ✅ | Migrate prompt loading/telemetry entry points in `ai_interface.py` |
| 4 | ✅ | Unified provider failover abstraction (configurable fallback order + adapter-aware routing) |

---

### 7. Test Coverage Gaps [COMPLETE]

> **Goal**: TestSuite-based tests for all modules

**All modules now have TestSuite-based tests** (114 modules with 100% success rate)

**Completed this session**:
- [x] `core/workflow_actions.py` - 5 tests for session guards and decorators (lazy imports to avoid circular dependencies)
- [x] `observability/__init__.py` - 5 tests for metrics functions and Prometheus availability
- [x] `ui/menu.py` - 6 tests for menu rendering helpers
- [x] `ui/__init__.py` - 2 tests for package exports

**Completed** (prior sessions):
- [x] `test_integration_workflow.py`, `config/__init__.py`, `config/__main__.py`
- [x] `core/session_guards.py`, `core/config_validation.py`, `core/analytics_helpers.py`
- [x] `core/maintenance_actions.py`, `cli/maintenance.py`, `cli/__init__.py`
- [x] `scripts/check_type_ignores.py`, `scripts/maintain_code_graph.py`, `scripts/dead_code_scan.py`
- [x] `ai_api_test.py`, `gedcom_utils.py`, `action10.py`

---

## Completed Milestones (Archive)

- [x] Replace smoke tests with behavior assertions in `core/error_handling.py`, `gedcom_search_utils.py`, `relationship_utils.py`, `selenium_utils.py`
- [x] Convert `test_integration_workflow.py` to standard TestSuite pattern
- [x] Create `live_session_fixture()` in `test_utilities.py` for authenticated session reuse

---

## Quick Reference: Validation Commands

```powershell
# Run all tests (sequential)
python run_all_tests.py

# Run specific module tests
python -m action7_inbox
python -m action8_messaging
python -m action9_process_productive
python -m messaging.workflow_helpers

# Lint check
ruff check .

# Type check
pyright

# Check for 429 rate limit errors
(Select-String -Path Logs\app.log -Pattern "429 error").Count
```

---

## Recommended Next Steps

1. **Start cache unification** - High-impact fix for import cycle issues (Track 4)
2. **API consolidation** - After cache is stable, tackle HTTP pipeline (Track 5)
3. **AI Interface decomposition** - Provider adapters and prompt templating (Track 6)
