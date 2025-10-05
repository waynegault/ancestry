# action10_module_tests() Refactoring Guide

**Function**: `action10_module_tests()`  
**File**: `action10.py`  
**Current**: 885 lines, complexity 49  
**Target**: <100 lines, complexity <10  
**Estimated Effort**: 16-20 hours

---

## 📋 CURRENT STRUCTURE ANALYSIS

### **Function Location**
- **Start**: Line 1578
- **End**: Line 2465
- **Total Lines**: 885

### **Nested Test Functions** (12 total)

| # | Function Name | Lines | Start | End | Purpose |
|---|---------------|-------|-------|-----|---------|
| 1 | `test_module_initialization` | 63 | 1601 | 1663 | Test function availability |
| 2 | `test_config_defaults` | 57 | 1665 | 1721 | Test configuration defaults |
| 3 | `test_sanitize_input` | 35 | 1723 | 1757 | Test input sanitization |
| 4 | `test_get_validated_year_input_patch` | 45 | 1759 | 1803 | Test year input validation |
| 5 | `test_fraser_gault_scoring_algorithm` | 35 | 1805 | 1839 | Test scoring algorithm |
| 6 | `test_display_relatives_fraser` | 67 | 1841 | 1905 | Test display_relatives function |
| 7 | `test_analyze_top_match_fraser` | 90 | 1908 | 1997 | Test analyze_top_match function |
| 8 | `test_real_search_performance_and_accuracy` | 90 | 2000 | 2089 | Test search performance |
| 9 | `test_family_relationship_analysis` | 70 | 2092 | 2161 | Test family relationships |
| 10 | `test_relationship_path_calculation` | 118 | 2164 | 2281 | Test relationship paths |
| 11 | `test_main_patch` | 13 | 2284 | 2296 | Test main function |
| 12 | `test_fraser_gault_comprehensive` | 160 | 2298 | 2457 | Comprehensive Fraser test |

### **Helper Functions** (Already Extracted)
- `_register_input_validation_tests()` - Line 1481
- `_register_scoring_tests()` - Line 1499
- `_register_relationship_tests()` - Line 1510
- `_setup_test_environment()` - Line 1537
- `_teardown_test_environment()` - Line 1548
- `_debug_wrapper()` - Line 1553
- `_load_test_person_data_from_env()` - Line 1556
- `_get_gedcom_data_or_skip()` - Line 1528

### **Current Registration Pattern**
```python
# Lines 2460-2462
_register_input_validation_tests(suite, debug_wrapper, test_sanitize_input, test_get_validated_year_input_patch)
_register_scoring_tests(suite, debug_wrapper, test_fraser_gault_scoring_algorithm)
_register_relationship_tests(suite, debug_wrapper, test_family_relationship_analysis, test_relationship_path_calculation)
```

---

## 🔧 STEP-BY-STEP REFACTORING INSTRUCTIONS

### **STEP 1: Create Baseline (5 minutes)**

```bash
# Run tests and save baseline
python run_all_tests.py > baseline_action10_before.txt

# Check current quality
python code_quality_checker.py | grep -A 10 "action10.py"

# Create git checkpoint
git add -A
git commit -m "Checkpoint: Before action10_module_tests refactoring"
```

---

### **STEP 2: Extract Test Functions (12-14 hours)**

Extract each function one at a time, following this pattern:

#### **2.1: Extract test_module_initialization (30 min)**

**Location to insert**: Line 1577 (just before `action10_module_tests()`)

**Code to extract** (Lines 1601-1663):
```python
def test_module_initialization() -> None:
    """Test that all required Action 10 functions are available and callable"""
    required_functions = [
        "main",
        "load_gedcom_data",
        "filter_and_score_individuals",
        "analyze_top_match",
        "get_user_criteria",
        "display_top_matches",
        "display_relatives",
        "validate_config",
        "calculate_match_score_cached",
        "sanitize_input",
        "parse_command_line_args",
    ]

    print(f"📋 Testing availability of {len(required_functions)} core functions:")
    for func_name in required_functions:
        print(f"   • {func_name}")

    try:
        found_functions = []
        callable_functions = []

        for func_name in required_functions:
            if func_name in globals():
                found_functions.append(func_name)
                if callable(globals()[func_name]):
                    callable_functions.append(func_name)
                    print(f"   ✅ {func_name}: Found and callable")
                else:
                    print(f"   ❌ {func_name}: Found but not callable")
            else:
                print(f"   ❌ {func_name}: Not found")

        # Test configuration
        config_available = config_schema is not None
        config_has_api = (
            hasattr(config_schema, "api") if config_available else False
        )

        print("📊 Results:")
        print(
            f"   Functions found: {len(found_functions)}/{len(required_functions)}"
        )
        print(
            f"   Functions callable: {len(callable_functions)}/{len(found_functions)}"
        )
        print(f"   Config available: {config_available}")
        print(f"   Config has API: {config_has_api}")

        assert len(found_functions) == len(
            required_functions
        ), f"Missing functions: {set(required_functions) - set(found_functions)}"
        assert len(callable_functions) == len(
            found_functions
        ), f"Non-callable functions: {set(found_functions) - set(callable_functions)}"
        assert config_available, "Configuration schema not available"

        return True
    except (NameError, AssertionError) as e:
        print(f"❌ Module initialization failed: {e}")
        return True  # Skip if config is missing in test env
```

**Actions**:
1. Copy the function above
2. Insert at line 1577 (before `action10_module_tests()`)
3. Remove indentation (unindent by 4 spaces)
4. Delete the nested function from inside `action10_module_tests()` (lines 1601-1663)
5. Test: `python action10.py`

#### **2.2: Extract test_config_defaults (30 min)**

**Location to insert**: After `test_module_initialization()`

**Code to extract** (Lines 1665-1721):
```python
def test_config_defaults() -> None:
    """Test that configuration defaults are loaded correctly"""
    print("📋 Testing configuration default values:")

    try:
        # Get actual values
        date_flexibility_value = (
            config_schema.date_flexibility if config_schema else 2
        )
        scoring_weights = (
            dict(config_schema.common_scoring_weights) if config_schema else {}
        )

        # Expected values
        expected_date_flexibility = 5.0
        expected_weight_keys = [
            "contains_first_name",
            "contains_surname",
            "bonus_both_names_contain",
            "exact_birth_date",
            "birth_year_match",
            "year_birth",
            "gender_match",
        ]

        print(
            f"   • Date flexibility: Expected {expected_date_flexibility}, Got {date_flexibility_value}"
        )
        print(f"   • Scoring weights type: {type(scoring_weights).__name__}")
        print(f"   • Scoring weights count: {len(scoring_weights)} keys")

        # Check key scoring weights
        for key in expected_weight_keys:
            weight = scoring_weights.get(key, "MISSING")
            print(f"   • {key}: {weight}")

        print("📊 Results:")
        print(
            f"   Date flexibility correct: {date_flexibility_value == expected_date_flexibility}"
        )
        print(f"   Scoring weights is dict: {isinstance(scoring_weights, dict)}")
        print(
            f"   Has required weight keys: {all(key in scoring_weights for key in expected_weight_keys)}"
        )

        assert (
            date_flexibility_value == expected_date_flexibility
        ), f"Date flexibility should be {expected_date_flexibility}, got {date_flexibility_value}"
        assert isinstance(
            scoring_weights, dict
        ), f"Scoring weights should be dict, got {type(scoring_weights)}"
        assert len(scoring_weights) > 0, "Scoring weights should not be empty"

        return True
    except Exception as e:
        print(f"❌ Config defaults test failed: {e}")
        return True
```

**Actions**:
1. Copy the function above
2. Insert after `test_module_initialization()`
3. Remove indentation (unindent by 4 spaces)
4. Delete the nested function from inside `action10_module_tests()`
5. Test: `python action10.py`

#### **2.3-2.12: Repeat for Remaining Functions**

Follow the same pattern for:
- `test_sanitize_input` (lines 1723-1757)
- `test_get_validated_year_input_patch` (lines 1759-1803)
- `test_fraser_gault_scoring_algorithm` (lines 1805-1839)
- `test_display_relatives_fraser` (lines 1841-1905)
- `test_analyze_top_match_fraser` (lines 1908-1997)
- `test_real_search_performance_and_accuracy` (lines 2000-2089)
- `test_family_relationship_analysis` (lines 2092-2161)
- `test_relationship_path_calculation` (lines 2164-2281)
- `test_main_patch` (lines 2284-2296)
- `test_fraser_gault_comprehensive` (lines 2298-2457)

**Commit after every 3 functions**:
```bash
git add action10.py
git commit -m "action10: Extract test functions 1-3 to module level"
```

---

### **STEP 3: Update Registration Functions (1-2 hours)**

The registration functions already exist but need to be updated to use module-level functions.

**Current** (lines 1481-1496):
```python
def _register_input_validation_tests(suite: Any, debug_wrapper: Callable, test_sanitize_input: Callable, test_get_validated_year_input_patch: Callable) -> None:
    """Register input validation and parsing tests."""
    suite.run_test(
        "Input Sanitization",
        debug_wrapper(test_sanitize_input),
        ...
    )
```

**After** (simplified):
```python
def _register_input_validation_tests(suite: Any, debug_wrapper: Callable) -> None:
    """Register input validation and parsing tests."""
    suite.run_test(
        "Input Sanitization",
        debug_wrapper(test_sanitize_input),  # Now references module-level function
        ...
    )
```

Update all three registration functions to remove function parameters.

---

### **STEP 4: Simplify Main Function (1 hour)**

**Current** (885 lines):
```python
def action10_module_tests() -> bool:
    """Comprehensive test suite for action10.py"""
    import builtins
    import os
    import time
    from pathlib import Path
    
    # ... 12 nested function definitions ...
    
    _register_input_validation_tests(suite, debug_wrapper, test_sanitize_input, test_get_validated_year_input_patch)
    _register_scoring_tests(suite, debug_wrapper, test_fraser_gault_scoring_algorithm)
    _register_relationship_tests(suite, debug_wrapper, test_family_relationship_analysis, test_relationship_path_calculation)
    
    return suite.finish_suite()
```

**Target** (<100 lines):
```python
def action10_module_tests() -> bool:
    """Comprehensive test suite for action10.py - streamlined orchestrator"""
    original_gedcom, suite = _setup_test_environment()
    debug_wrapper = _debug_wrapper
    
    # Register all tests
    _register_input_validation_tests(suite, debug_wrapper)
    _register_scoring_tests(suite, debug_wrapper)
    _register_relationship_tests(suite, debug_wrapper)
    
    _teardown_test_environment(original_gedcom)
    return suite.finish_suite()
```

---

### **STEP 5: Move Imports to Module Level (30 min)**

Move these imports from inside the function to module level (around line 100):

```python
import builtins  # For input patching in tests
```

---

### **STEP 6: Validation (30 min)**

```bash
# Run tests
python action10.py

# Run full test suite
python run_all_tests.py > baseline_action10_after.txt

# Compare
diff baseline_action10_before.txt baseline_action10_after.txt

# Check quality
python code_quality_checker.py | grep -A 10 "action10.py"
```

**Expected Results**:
- All tests pass ✅
- Quality score: 89.2 → 98+ ✅
- Complexity: 49 → <10 ✅
- Lines: 885 → <100 ✅

---

### **STEP 7: Final Commit (15 min)**

```bash
git add action10.py
git commit -m "Refactor action10_module_tests: Extract 12 test functions to module level

- Extracted 12 nested test functions to module level
- Reduced complexity from 49 to <10
- Reduced function length from 885 to <100 lines
- All tests passing (100% pass rate maintained)
- Quality score improved from 89.2 to 98+

Benefits:
- Individual tests can now be run independently
- Better test failure diagnostics
- Improved code organization
- Follows established TestSuite pattern
- Reduced technical debt

Closes task: ppQhCXnJtfYTcPB6Qmqh4f"
```

---

## ⚠️ COMMON PITFALLS

1. **Indentation**: Remember to unindent by 4 spaces when extracting
2. **Imports**: Some tests use `builtins.input` - ensure it's imported at module level
3. **Scope**: Tests use `globals()` - this still works at module level
4. **Order**: Extract functions in order to avoid line number confusion
5. **Testing**: Test after each extraction to catch issues early

---

## 🎯 SUCCESS CHECKLIST

- [ ] All 12 test functions extracted to module level
- [ ] All nested function definitions removed from main function
- [ ] Registration functions updated
- [ ] Main function simplified to <100 lines
- [ ] All imports moved to module level
- [ ] All tests still passing
- [ ] Quality score improved
- [ ] Complexity reduced to <10
- [ ] Git commits created
- [ ] Task marked as COMPLETE

---

**Estimated Time**: 16-20 hours total
**Recommended Approach**: Work in 2-hour sessions, commit frequently

