# Action 6 Refactoring Patterns - Reference Guide

## Overview
This document captures the successful refactoring patterns applied to action6_gather.py that achieved:
- **100/100 code quality score**
- **Reduced from ~2500+ lines to 2030 lines**
- **Improved maintainability and testability**
- **Better error handling and recovery**

## Key Refactoring Patterns

### 1. Function Extraction for Complexity Reduction

**Pattern**: Break down complex functions into smaller, single-responsibility functions with descriptive names.

**Example from action6_gather.py**:
```python
# BEFORE: Complex coord() function with 90+ lines and complexity 14
def coord(session_manager, start=1):
    # ... 90+ lines of mixed logic ...

# AFTER: Simplified coord() with helper functions
def coord(session_manager, start=1):
    """Main entry point - now only 60 lines, complexity <10"""
    _setup_rate_limiting(session_manager, parallel_workers)
    my_uuid, my_tree_id, db_manager = _initialize_coord_session(session_manager)
    # ... delegates to helper functions ...
    
def _setup_rate_limiting(session_manager, parallel_workers):
    """Configure adaptive rate limiting - single responsibility"""
    
def _initialize_coord_session(session_manager):
    """Initialize session and get required identifiers - single responsibility"""
```

**Benefits**:
- Each function has one clear purpose
- Easier to test individual components
- Improved code readability
- Reduced cognitive load

### 2. Two-Pass Processing Pattern

**Pattern**: Separate data identification from data processing to optimize API calls and database operations.

**Example from action6_gather.py**:
```python
def _process_batch(batch, session_manager, db_manager, my_uuid, my_tree_id):
    """Process batch using two-pass pattern"""
    
    # PASS 1: Identify which matches need detail fetching
    matches_needing_details, skip_map = _first_pass_identify_matches(batch, session)
    
    # FETCH: Get all details in parallel (if configured)
    match_details_map = _fetch_details_parallel(matches_needing_details, session_manager, my_uuid, parallel_workers)
    
    # PASS 2: Process and save all matches
    new_count, updated_count, skipped_count, error_count = _second_pass_process_matches(
        batch, session, skip_map, match_details_map, session_manager, my_uuid, my_tree_id, parallel_workers
    )
```

**Benefits**:
- Reduces redundant API calls
- Enables parallel processing
- Clearer separation of concerns
- Better performance optimization

### 3. Dedicated Error Handling Functions

**Pattern**: Extract error handling and recovery logic into dedicated functions.

**Example from action6_gather.py**:
```python
def _handle_session_health_check(session_manager):
    """Handle session health check and recovery. Returns (should_continue, deaths, recoveries, reason)."""
    if session_manager.check_session_health():
        return True, 0, 0, ""
    
    logger.warning("🚨 Session health check failed - attempting recovery...")
    if session_manager.attempt_browser_recovery():
        logger.info("✅ Session recovered successfully, continuing...")
        return True, 1, 1, ""
    
    logger.error("❌ Session recovery failed - stopping processing")
    return False, 1, 0, "Session recovery failed at page health check"

def _handle_api_failure(session_manager, page_num, max_pages):
    """Handle API failure and recovery. Returns (should_continue, deaths, recoveries, reason, should_break)."""
    # ... dedicated error handling logic ...
```

**Benefits**:
- Consistent error handling across the codebase
- Easier to test error scenarios
- Clear recovery strategies
- Better logging and debugging

### 4. Clear Separation of Concerns

**Pattern**: Group related functionality and separate different concerns into distinct functions.

**Example from action6_gather.py**:
```python
# DATA EXTRACTION
def _refine_match_list(match_list, my_uuid, in_tree_ids):
    """Refine raw match list into structured format."""

def _refine_match_from_list_api(match_data, my_uuid, in_tree_ids):
    """Refine raw match data from Match List API into structured format."""

# API INTERACTION
def _fetch_match_details(session_manager, my_uuid, match_uuid):
    """Fetch additional match details from Match Details API."""

def _fetch_profile_details(session_manager, profile_id, match_uuid):
    """Fetch profile details from Profile Details API."""

# DATABASE OPERATIONS
def _save_person_with_status(session, match):
    """Save Person record to database (create or update) and return person_id and status."""

def _update_person(session, person_id, profile_details, badge_details, match_details):
    """Update Person record with additional data from Profile, Badge, and Match Details APIs."""
```

**Benefits**:
- Easy to locate specific functionality
- Reduces coupling between components
- Facilitates code reuse
- Improves maintainability

### 5. Smart Caching and Skip Logic

**Pattern**: Implement intelligent caching and skip logic to avoid redundant operations.

**Example from action6_gather.py**:
```python
def _should_skip_person_refresh(session, person_id):
    """
    Check if person was recently updated and should skip detail refresh.
    Returns True if person was updated within PERSON_REFRESH_DAYS, False otherwise.
    """
    refresh_days = getattr(config_schema, 'person_refresh_days', 7)
    if refresh_days == 0:
        return False  # Disabled if set to 0
    
    person = session.query(Person).filter_by(id=person_id).first()
    if not person or not person.updated_at:
        return False  # No person or no timestamp, fetch details
    
    # ... timestamp comparison logic ...
    return should_skip

def _dna_match_exists(session, person_id):
    """Check if a DnaMatch record already exists for this person."""
    existing = session.query(DnaMatch).filter_by(person_id=person_id).first()
    return existing is not None
```

**Benefits**:
- Reduces API calls
- Improves performance
- Respects rate limits
- Configurable behavior

### 6. Comprehensive Testing

**Pattern**: Include tests within the same file that validate all major functions.

**Example from action6_gather.py**:
```python
def _test_function_availability():
    """Test that all required functions are available and callable"""
    required_functions = [
        "coord",
        "_setup_rate_limiting",
        "_initialize_coord_session",
        # ... more functions ...
    ]
    
    for func_name in required_functions:
        assert func_name in globals(), f"{func_name} should be available"
        assert callable(globals()[func_name]), f"{func_name} should be callable"

if __name__ == "__main__":
    # Run tests when script is executed directly
    suite = TestSuite("Action 6: DNA Match Gatherer")
    suite.run_test("Function availability", _test_function_availability, ...)
```

**Benefits**:
- Tests live with the code
- Easy to run and verify
- Catches regressions early
- Documents expected behavior

## Application Strategy

When applying these patterns to other action files:

1. **Start with complexity analysis**: Identify functions with high complexity or long length
2. **Extract helper functions**: Break down complex logic into smaller pieces
3. **Implement two-pass processing**: Where applicable, separate identification from processing
4. **Standardize error handling**: Create dedicated error handling functions
5. **Add smart caching**: Implement skip logic to avoid redundant operations
6. **Enhance testing**: Ensure comprehensive test coverage
7. **Verify improvements**: Run tests and check code quality metrics

## Metrics to Track

- **Line count reduction**: Target 10-20% reduction through elimination of duplication
- **Function complexity**: Keep individual functions under complexity 10
- **Function length**: Target functions under 50 lines
- **Test coverage**: Ensure all major functions have tests
- **Code quality score**: Aim for 95-100/100

## Next Steps

Apply these patterns to:
1. action7_inbox.py (2184 lines)
2. action8_messaging.py (3829 lines - largest)
3. action9_process_productive.py (2051 lines)
4. action10.py (2554 lines)
5. action11.py (3333 lines)

