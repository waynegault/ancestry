# 📋 REFACTORING TASK LIST

**Last Updated**: 2025-10-04  
**Strategy**: Focus on toughest challenges first (lowest quality scores)  
**Target**: Get all files above 70/100 quality threshold

---

## 🎯 PRIORITY 1: CRITICAL FILES (0-25 Quality Score)

### **Task 1.1: utils.py (10.3/100) - 16 violations**
**Current Issues**:
- Function `main` is too long (572 lines) - **TEST FUNCTION, ACCEPTABLE**
- Function `main` is too complex (complexity: 36) - **TEST FUNCTION, ACCEPTABLE**
- Function `mock_start_sess` missing type hints
- Function `mock_ensure_session_ready` missing type hints
- 13 more violations

**Recommended Actions**:
1. Add type hints to `mock_start_sess` function
2. Add type hints to `mock_ensure_session_ready` function
3. Review remaining 11 violations and address systematically
4. Note: `main` function is a test function - complexity acceptable

**Estimated Impact**: +5-10 quality points

---

### **Task 1.2: action6_gather.py (22.0/100) - 14 violations**
**Current Issues**:
- Function `_main_page_processing_loop` is too complex (complexity: 12)
- Function `_prepare_bulk_db_data` is too complex (complexity: 15)
- Function `_execute_bulk_db_operations` is too complex (complexity: <10 now, verify)
- 11 more violations

**Recommended Actions**:
1. Verify `_execute_bulk_db_operations` complexity is now <10
2. Refactor `_prepare_bulk_db_data` (complexity 15 → <10)
   - Extract data preparation logic into helpers
   - Separate person, DNA match, and family tree preparation
3. Refactor `_main_page_processing_loop` (complexity 12 → <10)
   - Extract page processing steps into helpers
4. Address remaining violations

**Estimated Impact**: +10-15 quality points

---

### **Task 1.3: api_utils.py (23.4/100) - 14 violations**
**Current Issues**:
- Function `_extract_event_from_api_details` is too complex (complexity: 13)
- Function `parse_ancestry_person_details` is too complex (complexity: 16)
- Function `call_suggest_api` is too complex (complexity: 12)
- 11 more violations

**Recommended Actions**:
1. Refactor `parse_ancestry_person_details` (complexity 16 → <10)
   - Extract person detail parsing logic into helpers
   - Separate name, event, and relationship parsing
2. Refactor `_extract_event_from_api_details` (complexity 13 → <10)
   - Extract event extraction logic into helpers
3. Refactor `call_suggest_api` (complexity 12 → <10)
   - Extract API call logic into helpers
4. Address remaining violations

**Estimated Impact**: +10-15 quality points

---

### **Task 1.4: main.py (24.4/100) - 12 violations**
**Current Issues**:
- Function `validate_action_config` is too complex (complexity: 12)
- Function `run_core_workflow_action` missing type hints
- Function `backup_db_actn` missing type hints
- 9 more violations

**Recommended Actions**:
1. Refactor `validate_action_config` (complexity 12 → <10)
   - Extract validation logic into helpers
   - Separate config validation steps
2. Add type hints to `run_core_workflow_action`
3. Add type hints to `backup_db_actn`
4. Address remaining violations

**Estimated Impact**: +10-15 quality points

---

### **Task 1.5: gedcom_utils.py (24.7/100) - 13 violations**
**Current Issues**:
- Function `_get_full_name` is too complex (complexity: 11)
- Function `fast_bidirectional_bfs` is too complex (complexity: 12)
- Function `explain_relationship_path` is too complex (complexity: 26) **MASSIVE**
- 10 more violations

**Recommended Actions**:
1. Refactor `explain_relationship_path` (complexity 26 → <10) **HIGH PRIORITY**
   - Extract relationship explanation logic into helpers
   - Separate path analysis, generation formatting
2. Refactor `fast_bidirectional_bfs` (complexity 12 → <10)
   - Extract BFS search logic into helpers
3. Refactor `_get_full_name` (complexity 11 → <10)
   - Already has good helpers, may need minor adjustments
4. Address remaining violations

**Estimated Impact**: +15-20 quality points

---

## 🎯 PRIORITY 2: MODERATE FILES (25-40 Quality Score)

### **Task 2.1: session_manager.py (28.0/100) - 13 violations**
**Current Issues**:
- Function `session_manager_module_tests` is too complex (complexity: 14)
- Function `_verify_session_continuity` is too complex (complexity: 13)
- Function `get_cookies` is too complex (complexity: 14)
- 10 more violations

**Recommended Actions**:
1. Refactor `get_cookies` (complexity 14 → <10)
2. Refactor `_verify_session_continuity` (complexity 13 → <10)
3. Refactor `session_manager_module_tests` (complexity 14 → <10)
   - Note: This is a test function, may be acceptable
4. Address remaining violations

**Estimated Impact**: +10-15 quality points

---

### **Task 2.2: action7_inbox.py (Current: Unknown, previously 82.3/100)**
**Status**: Previously improved significantly
**Recommended Actions**: Verify current status and address any remaining issues

---

### **Task 2.3: action8_messaging.py (Current: Unknown, previously 73.4/100)**
**Status**: Previously improved significantly
**Recommended Actions**: Verify current status and address any remaining issues

---

## 🎯 PRIORITY 3: FILES NEAR THRESHOLD (40-70 Quality Score)

### **Task 3.1: Review files in 40-70 range**
**Recommended Actions**:
1. Run quality checker to identify files in this range
2. Prioritize files closest to 70 threshold for quick wins
3. Focus on adding missing type hints
4. Address minor complexity issues

**Estimated Impact**: Move 5-10 files above 70 threshold

---

## 🎯 PRIORITY 4: SYSTEMATIC IMPROVEMENTS

### **Task 4.1: Type Hint Coverage**
**Current**: 98.2%
**Target**: 99%+

**Recommended Actions**:
1. Identify all functions missing type hints
2. Add type hints systematically
3. Focus on public API functions first

---

### **Task 4.2: Test Function Complexity**
**Issue**: Some test functions have high complexity (acceptable)

**Recommended Actions**:
1. Document which functions are test functions
2. Consider excluding test functions from complexity checks
3. Or refactor test functions if they're too complex to maintain

---

### **Task 4.3: Documentation**
**Recommended Actions**:
1. Ensure all refactored functions have clear docstrings
2. Document helper function purposes
3. Update README with refactoring progress

---

## 📊 PROGRESS TRACKING

### **Completed This Session**:
- ✅ session_manager.py: 5.5 → 28.0/100 (+22.5 points)
- ✅ action6_gather.py: 21.5 → 22.0/100 (+0.5 points)
- ✅ gedcom_utils.py: 18.6 → 24.7/100 (+6.1 points)
- ✅ main.py: 17.8 → 24.4/100 (+6.6 points)
- ✅ utils.py: 0.0 → 10.3/100 (+10.3 points)
- ✅ relationship_utils.py: 59.5 → 65.8/100 (+6.3 points)
- ✅ error_handling.py: 8.5 → Improved

### **Next Session Goals**:
- 🎯 Get action6_gather.py above 30/100
- 🎯 Get api_utils.py above 30/100
- 🎯 Get main.py above 30/100
- 🎯 Get gedcom_utils.py above 30/100
- 🎯 Get session_manager.py above 40/100

---

## 🔧 REFACTORING WORKFLOW

For each function to refactor:

1. **Analyze**: Understand the function's purpose and complexity
2. **Plan**: Identify logical sections that can be extracted
3. **Extract**: Create focused helper functions (SRP)
4. **Refactor**: Update main function to use helpers
5. **Test**: Run all tests to ensure no regressions
6. **Commit**: Commit changes with descriptive message
7. **Verify**: Check quality score improvement

---

## 📈 SUCCESS METRICS

- **Primary**: Average quality score > 85/100
- **Secondary**: All files > 70/100
- **Tertiary**: Type hint coverage > 99%
- **Critical**: 100% test success rate maintained

---

**End of Task List**

