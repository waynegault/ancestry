# Monolithic Test Function Refactoring Plan

**Created**: 2025-10-05  
**Status**: Ready for Execution  
**Estimated Total Effort**: 40-60 hours across 6 functions

---

## 🎯 OBJECTIVE

Refactor 6 monolithic test functions (3,600+ lines total) into modular, maintainable test suites following the TestSuite pattern used throughout the codebase.

---

## 📊 FUNCTIONS TO REFACTOR (Priority Order)

| # | Function | File | Lines | Complexity | Priority | Effort |
|---|----------|------|-------|------------|----------|--------|
| 1 | `action10_module_tests()` | action10.py | 885 | 49 | 🔴 CRITICAL | 16-20h |
| 2 | `credential_manager_module_tests()` | config/credential_manager.py | 615 | 17 | 🔴 HIGH | 10-12h |
| 3 | `main_module_tests()` | main.py | 540 | 13 | 🟡 MEDIUM | 8-10h |
| 4 | `action8_messaging_tests()` | action8_messaging.py | 537 | 26 | 🔴 HIGH | 8-10h |
| 5 | `genealogical_task_templates_module_tests()` | genealogical_task_templates.py | 485 | 19 | 🟡 MEDIUM | 6-8h |
| 6 | `security_manager_module_tests()` | security_manager.py | 485 | - | 🟡 MEDIUM | 6-8h |

---

## 🔧 REFACTORING PATTERN (Standard Approach)

### **Before (Monolithic)**
```python
def module_tests() -> bool:
    """Monolithic test function"""
    suite = TestSuite("Module", "module.py")
    suite.start_suite()
    
    # 12 nested test functions defined here
    def test_feature_1():
        # 50 lines of test code
        pass
    
    def test_feature_2():
        # 60 lines of test code
        pass
    
    # ... 10 more nested functions
    
    # Register all tests
    suite.run_test("Feature 1", test_feature_1, ...)
    suite.run_test("Feature 2", test_feature_2, ...)
    
    return suite.finish_suite()
```

### **After (Modular)**
```python
# Module-level test functions
def test_feature_1() -> None:
    """Test feature 1 functionality"""
    # 50 lines of test code
    pass

def test_feature_2() -> None:
    """Test feature 2 functionality"""
    # 60 lines of test code
    pass

# ... 10 more module-level functions

def module_tests() -> bool:
    """Streamlined test orchestrator"""
    suite = TestSuite("Module", "module.py")
    suite.start_suite()
    
    # Register all tests (clean and simple)
    suite.run_test("Feature 1", test_feature_1, ...)
    suite.run_test("Feature 2", test_feature_2, ...)
    
    return suite.finish_suite()
```

---

## 📋 DETAILED REFACTORING STEPS (Per Function)

### **Phase 1: Preparation (30 minutes)**

1. **Create baseline**
   ```bash
   python run_all_tests.py > baseline_before_refactor.txt
   git add -A
   git commit -m "Baseline before refactoring [function_name]"
   ```

2. **Identify all nested test functions**
   - Search for `def test_` inside the monolithic function
   - Count total nested functions
   - Document each function's purpose

3. **Review dependencies**
   - Check for shared variables between tests
   - Identify helper functions that need extraction
   - Note any test-specific imports

### **Phase 2: Extract Test Functions (2-4 hours per function)**

1. **Extract first test function**
   - Copy nested function to module level (before monolithic function)
   - Remove one level of indentation
   - Add proper type hints: `-> None`
   - Ensure docstring is present

2. **Update function to use module-level scope**
   - Replace `globals()` with module-level references if needed
   - Ensure all imports are available at module level
   - Test that function works independently

3. **Repeat for all nested functions**
   - Extract one function at a time
   - Test after each extraction
   - Commit after every 2-3 extractions

### **Phase 3: Simplify Main Function (1-2 hours)**

1. **Remove nested function definitions**
   - Delete all `def test_*()` blocks from inside monolithic function
   - Keep only the registration calls

2. **Clean up imports**
   - Move test-specific imports to module level if needed
   - Remove unused imports from inside function

3. **Simplify structure**
   - Main function should only:
     - Create TestSuite
     - Call suite.start_suite()
     - Register tests with suite.run_test()
     - Call suite.finish_suite()

### **Phase 4: Validation (30 minutes)**

1. **Run tests**
   ```bash
   python run_all_tests.py > baseline_after_refactor.txt
   ```

2. **Compare results**
   ```bash
   diff baseline_before_refactor.txt baseline_after_refactor.txt
   ```
   - Test count should be identical
   - All tests should still pass
   - Quality score should improve

3. **Check quality metrics**
   ```bash
   python code_quality_checker.py
   ```
   - Function should no longer appear in violations
   - Complexity should be <10
   - Lines should be <400

### **Phase 5: Commit (15 minutes)**

1. **Final commit**
   ```bash
   git add -A
   git commit -m "Refactor [function_name]: Extract [N] test functions to module level

   - Extracted [N] nested test functions to module level
   - Reduced complexity from [X] to [Y]
   - Reduced function length from [X] to [Y] lines
   - All tests passing (100% pass rate maintained)
   - Quality score improved from [X] to [Y]
   
   Closes task: [task_id]"
   ```

2. **Update task list**
   - Mark task as COMPLETE
   - Update quality metrics in tracking document

---

## 🎯 FUNCTION-SPECIFIC DETAILS

### **1. action10_module_tests() - 885 lines, complexity 49**

**Location**: `action10.py:1578-2465`

**Nested Functions** (12 total):
1. `test_module_initialization()` - Lines 1601-1663
2. `test_config_defaults()` - Lines 1665-1721
3. `test_sanitize_input()` - Lines 1723-1757
4. `test_get_validated_year_input_patch()` - Lines 1759-1803
5. `test_fraser_gault_scoring_algorithm()` - Lines 1805-1839
6. `test_display_relatives_fraser()` - Lines 1841-1905
7. `test_analyze_top_match_fraser()` - Lines 1908-1997
8. `test_real_search_performance_and_accuracy()` - Lines 2000-2089
9. `test_family_relationship_analysis()` - Lines 2092-2161
10. `test_relationship_path_calculation()` - Lines 2164-2281
11. `test_main_patch()` - Lines 2284-2296
12. `test_fraser_gault_comprehensive()` - Lines 2298-2457

**Helper Functions Already Extracted**:
- `_register_input_validation_tests()` - Line 1481
- `_register_scoring_tests()` - Line 1499
- `_register_relationship_tests()` - Line 1510

**Special Considerations**:
- Uses `builtins.input` patching (needs to be at module level)
- Uses `mock_logger_context` from test_framework
- Loads GEDCOM data from .env configuration
- Has performance testing components

**Estimated Effort**: 16-20 hours

---

### **2. credential_manager_module_tests() - 615 lines, complexity 17**

**Location**: `config/credential_manager.py:558-1172`

**Nested Functions** (11 total):
1. `test_initialization()`
2. `test_environment_loading()`
3. `test_credential_validation()`
4. `test_credential_access()`
5. `test_credential_storage()`
6. `test_api_key_retrieval()`
7. `test_credential_removal()`
8. `test_credential_listing()`
9. `test_security_integration()`
10. `test_cache_management()`
11. `test_error_handling()`

**Special Considerations**:
- Security-critical testing
- Uses `suppress_logging()` context manager
- Tests encryption/decryption
- Mock security manager needed

**Estimated Effort**: 10-12 hours

---

### **3. main_module_tests() - 540 lines, complexity 13**

**Location**: `main.py:1757-2297`

**Nested Functions** (Approximately 15-20):
- Module initialization tests
- Menu system tests
- Action dispatch tests
- Configuration tests
- Error handling tests
- Cleanup procedure tests

**Special Considerations**:
- Central menu system testing
- Uses `inspect.getsource()` for code analysis
- Tests exception handling coverage

**Estimated Effort**: 8-10 hours

---

### **4. action8_messaging_tests() - 537 lines, complexity 26**

**Location**: `action8_messaging.py` (line numbers TBD)

**Special Considerations**:
- High complexity (26) indicates complex test logic
- Messaging functionality testing
- May involve database interactions

**Estimated Effort**: 8-10 hours

---

### **5. genealogical_task_templates_module_tests() - 485 lines, complexity 19**

**Location**: `genealogical_task_templates.py:607-1093`

**Special Considerations**:
- Genealogical research testing
- Template validation
- Task generation testing

**Estimated Effort**: 6-8 hours

---

### **6. security_manager_module_tests() - 485 lines**

**Location**: `security_manager.py` (line numbers TBD)

**Special Considerations**:
- Security-critical testing
- Encryption/decryption testing
- Credential management

**Estimated Effort**: 6-8 hours

---

## ✅ SUCCESS CRITERIA (Per Function)

- [ ] All nested test functions extracted to module level
- [ ] Main function reduced to <100 lines
- [ ] Complexity reduced to <10
- [ ] All tests still passing (100% pass rate)
- [ ] Quality score improved
- [ ] Git commit created with detailed message
- [ ] Task marked as COMPLETE

---

## 📈 PROGRESS TRACKING

| Function | Status | Start Date | End Date | Actual Effort | Quality Before | Quality After |
|----------|--------|------------|----------|---------------|----------------|---------------|
| action10_module_tests | ⏳ Not Started | - | - | - | 89.2/100 | - |
| credential_manager_module_tests | ⏳ Not Started | - | - | - | 88.2/100 | - |
| main_module_tests | ⏳ Not Started | - | - | - | 89.1/100 | - |
| action8_messaging_tests | ⏳ Not Started | - | - | - | 89.5/100 | - |
| genealogical_task_templates_module_tests | ⏳ Not Started | - | - | - | 88.2/100 | - |
| security_manager_module_tests | ⏳ Not Started | - | - | - | 94.0/100 | - |

---

## 🎉 EXPECTED OUTCOMES

### **Quality Improvements**
- Average quality score: 97.3 → 99.5+ (target: 100)
- All test functions: <400 lines ✅
- All test functions: complexity <10 ✅
- Improved debuggability: Can run individual tests ✅
- Better test failure diagnostics ✅

### **Maintainability Improvements**
- Easier to add new tests
- Easier to debug test failures
- Better code organization
- Follows established patterns
- Reduced technical debt

---

**Next Steps**: Start with action10_module_tests() following the detailed steps above.

