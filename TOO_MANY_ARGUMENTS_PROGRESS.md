# Too-Many-Arguments Refactoring Progress

## Overview
This document tracks the progress of eliminating PLR0913 (too-many-arguments) violations across the codebase.

## Initial State
- **Total Violations**: 123 violations across 15+ files
- **Goal**: Reduce function parameters to ≤5 by using dataclasses, configuration objects, and **kwargs

## Refactoring Strategy
1. **Create Parameter Grouping Dataclasses**: Group related parameters into dataclasses
2. **Use Configuration Objects**: Pass configuration objects instead of individual settings
3. **Apply **kwargs for Optional Parameters**: Use **kwargs for optional/rarely-used parameters
4. **Extract to Methods**: Convert functions to class methods to use self for shared state

## Violations by File (Top 15)
| File | Violations | Status |
|------|------------|--------|
| action6_gather.py | 21 → 20 | 🔄 In Progress (1 fixed) |
| action8_messaging.py | 17 | ⏳ Pending |
| utils.py | 16 | ⏳ Pending |
| action7_inbox.py | 10 | ⏳ Pending |
| gedcom_utils.py | 10 | ⏳ Pending |
| action11.py | 8 | ⏳ Pending |
| relationship_utils.py | 6 | ⏳ Pending |
| api_utils.py | 5 | ⏳ Pending |
| action9_process_productive.py | 4 | ⏳ Pending |
| api_search_utils.py | 4 | ⏳ Pending |
| adaptive_rate_limiter.py | 3 | ⏳ Pending |
| ai_interface.py | 3 | ⏳ Pending |
| action10.py | 2 | ⏳ Pending |
| core/progress_indicators.py | 2 | ⏳ Pending |
| core/session_validator.py | 2 | ⏳ Pending |

## Completed Refactorings

### action6_gather.py (21 → 20 violations)

**Dataclasses Created**:
1. `BatchCounters` - Groups batch processing counters (new, updated, skipped, errors)
2. `MatchIdentifiers` - Groups match identification parameters (uuid, username, in_my_tree, log_ref_short)

**Functions Refactored**:
1. `_update_state_after_batch` (6 → 3 parameters)
   - Before: `(state, page_new, page_updated, page_skipped, page_errors, progress_bar)`
   - After: `(state, counters: BatchCounters, progress_bar)`
   - Reduction: 3 parameters (50%)

**Code Example**:
```python
# Before
def _update_state_after_batch(
    state: Dict[str, Any],
    page_new: int,
    page_updated: int,
    page_skipped: int,
    page_errors: int,
    progress_bar
):
    state["total_new"] += page_new
    state["total_updated"] += page_updated
    state["total_skipped"] += page_skipped
    state["total_errors"] += page_errors
    # ...

# After
@dataclass
class BatchCounters:
    new: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0

def _update_state_after_batch(
    state: Dict[str, Any],
    counters: BatchCounters,
    progress_bar
):
    state["total_new"] += counters.new
    state["total_updated"] += counters.updated
    state["total_skipped"] += counters.skipped
    state["total_errors"] += counters.errors
    # ...

# Call site
counters = BatchCounters(new=page_new, updated=page_updated, skipped=page_skipped, errors=page_errors)
_update_state_after_batch(state, counters, progress_bar)
```

## Metrics
- **Violations Fixed**: 1 / 123 (0.8%)
- **Violations Remaining**: 122 / 123 (99.2%)
- **Files Completed**: 0 / 15 (0%)
- **Files In Progress**: 1 / 15 (6.7%)

## Testing Status
- **All Tests Passing**: 468/468 (100%) ✅
- **Quality Score**: 98.9/100 ✅

## Remaining Work

### High Priority (Most Violations)
1. **action6_gather.py** (20 remaining)
   - Functions with 7-9 parameters
   - Opportunity to use MatchIdentifiers dataclass
   - Consider creating ProcessingContext class

2. **action8_messaging.py** (17 violations)
   - Message template parameters
   - Recipient information
   - Consider MessageConfig dataclass

3. **utils.py** (16 violations)
   - Session management parameters
   - WebDriver configuration
   - Consider SessionConfig and DriverConfig classes

### Medium Priority (5-10 Violations)
4. **action7_inbox.py** (10 violations)
5. **gedcom_utils.py** (10 violations)
6. **action11.py** (8 violations)
7. **relationship_utils.py** (6 violations)
8. **api_utils.py** (5 violations)

### Low Priority (2-4 Violations)
9. **action9_process_productive.py** (4 violations)
10. **api_search_utils.py** (4 violations)
11. **adaptive_rate_limiter.py** (3 violations)
12. **ai_interface.py** (3 violations)
13. **action10.py** (2 violations)
14. **core/progress_indicators.py** (2 violations)
15. **core/session_validator.py** (2 violations)

## Estimated Effort
- **Total Violations**: 123
- **Average Time per Violation**: 5-10 minutes
- **Total Estimated Time**: 10-20 hours
- **Recommended Approach**: Tackle files in priority order, create reusable dataclasses

## Next Steps
1. Continue with action6_gather.py (19 more violations)
2. Create additional dataclasses as patterns emerge
3. Move to action8_messaging.py after completing action6
4. Consider creating a shared `common_params.py` module for widely-used dataclasses

## Notes
- This is a large refactoring task that will require multiple sessions
- Each file may need custom dataclasses based on its specific parameter patterns
- Some functions may benefit from being converted to class methods
- Consider breaking this into phases: Phase 1 (files with 10+ violations), Phase 2 (files with 5-9), Phase 3 (files with 2-4)

