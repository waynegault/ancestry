# Ancestry Automation - Technical Roadmap

> **Quality guard**: Run `ruff check` and `pyright` after every milestone. Never introduce `# noqa` or `type: ignore` suppressions.

---

## Session Summary (November 25, 2025)

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

### 4. Cache Stack Unification [NOT STARTED] ⚠️ High Impact

> **Goal**: Single cache backend replacing 6 overlapping modules
> **Effort**: ~3-4 sessions | **Priority**: High (import cycles cause issues)

**Problem**: `cache.py`, `cache_manager.py`, `core/unified_cache_manager.py`, `core/cache_registry.py`, `core/session_cache.py`, `performance_cache.py` have import cycles and duplicate functionality.

| Step | Description |
|------|-------------|
| 1 | Inventory all cache entry points and consumers |
| 2 | Design `CacheBackend` protocol in `core/cache_backend.py` |
| 3 | Port consumers module-by-module with regression tests |
| 4 | Remove redundant modules, update telemetry dashboards |

---

### 5. API Utils Consolidation [NOT STARTED]

> **Goal**: Single HTTP pipeline via `core/api_manager.py`
> **Effort**: ~2-3 sessions | **Priority**: Medium

**Problem**: `api_utils.py` (3779 lines) duplicates rate limiting, retry, and cookie sync logic that should live in `SessionManager.api_manager`.

| Step | Description |
|------|-------------|
| 1 | Catalogue all `_api_req` consumers (Actions 6–10, messaging, telemetry) |
| 2 | Extend `APIManager` with parameterized endpoints, streaming, retry policies |
| 3 | Migrate callers to `session_manager.api_manager.request()` |
| 4 | Trim `api_utils.py` to parsing/transform helpers only |

---

### 6. AI Interface Decomposition [NOT STARTED]

> **Goal**: `ai/` package with provider adapters, prompt templating, telemetry
> **Effort**: ~2 sessions | **Priority**: Medium

**Problem**: `ai_interface.py` (3k lines) mixes provider selection, prompt templates, and telemetry.

| Step | Description |
|------|-------------|
| 1 | Create `ai/providers/` with Gemini, DeepSeek adapters |
| 2 | Create `ai/prompts.py` for template loading and substitution |
| 3 | Migrate `ai_prompt_utils.py`, `message_personalization.py`, `prompt_telemetry.py` |
| 4 | Unified provider failover abstraction |

---

### 7. Test Coverage Gaps [ONGOING]

> **Goal**: TestSuite-based tests for all modules

**Modules needing tests**:
- [ ] `core/workflow_actions.py`
- [ ] `observability/__init__.py`
- [ ] `ui/menu.py`
- [ ] `ui/__init__.py`

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
4. **Test coverage gaps** - Add tests for remaining modules (Track 7)
