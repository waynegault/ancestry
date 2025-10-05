# Monolithic Test Refactoring - Quick Checklist

**Use this checklist for each function you refactor**

---

## 📋 PRE-REFACTORING CHECKLIST

- [ ] Read the detailed plan in `MONOLITHIC_TEST_REFACTORING_PLAN.md`
- [ ] Read function-specific guide (e.g., `ACTION10_REFACTORING_GUIDE.md`)
- [ ] Ensure you have 2-4 hours of uninterrupted time
- [ ] Close all other applications to avoid distractions

---

## 🔧 REFACTORING CHECKLIST (Per Function)

### **Phase 1: Baseline (5 min)**
- [ ] Run: `python run_all_tests.py > baseline_before.txt`
- [ ] Run: `python code_quality_checker.py > quality_before.txt`
- [ ] Commit: `git commit -m "Checkpoint: Before [function_name] refactoring"`
- [ ] Note current metrics:
  - Lines: _______
  - Complexity: _______
  - Quality Score: _______

### **Phase 2: Extract Functions (2-4 hours)**
- [ ] Extract function 1 to module level
- [ ] Remove indentation (unindent by 4 spaces)
- [ ] Add type hint: `-> None`
- [ ] Delete nested function from main
- [ ] Test: `python [module].py`
- [ ] Commit: `git commit -m "Extract test function 1"`

- [ ] Extract function 2 to module level
- [ ] Remove indentation
- [ ] Add type hint
- [ ] Delete nested function
- [ ] Test
- [ ] Commit

- [ ] Extract function 3 to module level
- [ ] Remove indentation
- [ ] Add type hint
- [ ] Delete nested function
- [ ] Test
- [ ] Commit: `git commit -m "Extract test functions 1-3"`

**Repeat for all remaining functions...**

### **Phase 3: Update Registration (30 min)**
- [ ] Update registration helper functions
- [ ] Remove function parameters from registration calls
- [ ] Test: `python [module].py`
- [ ] Commit: `git commit -m "Update test registration functions"`

### **Phase 4: Simplify Main (30 min)**
- [ ] Remove all nested function definitions
- [ ] Move imports to module level
- [ ] Simplify main function to <100 lines
- [ ] Test: `python [module].py`
- [ ] Commit: `git commit -m "Simplify main test function"`

### **Phase 5: Validation (30 min)**
- [ ] Run: `python run_all_tests.py > baseline_after.txt`
- [ ] Run: `python code_quality_checker.py > quality_after.txt`
- [ ] Compare: `diff baseline_before.txt baseline_after.txt`
- [ ] Verify: All tests still passing
- [ ] Verify: Test count unchanged
- [ ] Verify: Quality score improved
- [ ] Note new metrics:
  - Lines: _______
  - Complexity: _______
  - Quality Score: _______

### **Phase 6: Final Commit (15 min)**
- [ ] Create detailed commit message (see template below)
- [ ] Commit: `git commit -m "[detailed message]"`
- [ ] Update task status to COMPLETE
- [ ] Update progress tracking document

---

## 📝 COMMIT MESSAGE TEMPLATE

```
Refactor [function_name]: Extract [N] test functions to module level

- Extracted [N] nested test functions to module level
- Reduced complexity from [OLD] to [NEW]
- Reduced function length from [OLD] to [NEW] lines
- All tests passing (100% pass rate maintained)
- Quality score improved from [OLD] to [NEW]

Benefits:
- Individual tests can now be run independently
- Better test failure diagnostics
- Improved code organization
- Follows established TestSuite pattern
- Reduced technical debt

Test Functions Extracted:
1. test_[name_1]
2. test_[name_2]
3. test_[name_3]
... [list all]

Closes task: [task_id]
```

---

## ⚠️ TROUBLESHOOTING

### **Problem: Tests fail after extraction**
**Solution**: 
- Check indentation (should be 0 spaces at module level)
- Check imports (move to module level if needed)
- Check variable scope (use module-level variables)

### **Problem: Function not found**
**Solution**:
- Ensure function is defined before it's called
- Check function name spelling
- Ensure function is at module level (not nested)

### **Problem: Import errors**
**Solution**:
- Move test-specific imports to module level
- Check for circular imports
- Ensure all dependencies are available

### **Problem: Quality score didn't improve**
**Solution**:
- Run quality checker again
- Check that function length actually decreased
- Check that complexity actually decreased
- Ensure all nested functions were extracted

---

## 🎯 SUCCESS CRITERIA

After refactoring, the function should meet ALL of these:

- [ ] **Lines**: <100 (was 400-900)
- [ ] **Complexity**: <10 (was 13-49)
- [ ] **Tests**: 100% passing (maintained)
- [ ] **Quality**: Improved by 5-10 points
- [ ] **Structure**: Only orchestration code in main function
- [ ] **Modularity**: All test functions at module level
- [ ] **Documentation**: All functions have docstrings
- [ ] **Type Hints**: All functions have `-> None`

---

## 📊 PROGRESS TRACKER

| Function | Status | Date | Time | Lines Before | Lines After | Complexity Before | Complexity After |
|----------|--------|------|------|--------------|-------------|-------------------|------------------|
| action10_module_tests | ⏳ | - | - | 885 | - | 49 | - |
| credential_manager_module_tests | ⏳ | - | - | 615 | - | 17 | - |
| main_module_tests | ⏳ | - | - | 540 | - | 13 | - |
| action8_messaging_tests | ⏳ | - | - | 537 | - | 26 | - |
| genealogical_task_templates_module_tests | ⏳ | - | - | 485 | - | 19 | - |
| security_manager_module_tests | ⏳ | - | - | 485 | - | - | - |

**Legend**: ⏳ Not Started | 🔄 In Progress | ✅ Complete

---

## 💡 TIPS FOR SUCCESS

1. **Work in small chunks**: Extract 2-3 functions, then commit
2. **Test frequently**: Run tests after each extraction
3. **Take breaks**: This is tedious work - take a 5-minute break every hour
4. **Use search**: Use Ctrl+F to find function definitions quickly
5. **Copy-paste carefully**: Ensure you copy the entire function
6. **Check indentation**: Use your editor's "show whitespace" feature
7. **Commit often**: Better to have many small commits than one large one
8. **Read error messages**: They usually tell you exactly what's wrong

---

## 🚀 GETTING STARTED

**For action10_module_tests():**
1. Open `ACTION10_REFACTORING_GUIDE.md`
2. Follow Step 1: Create Baseline
3. Follow Step 2.1: Extract first function
4. Test and commit
5. Repeat for remaining functions

**Estimated time per function**: 30-45 minutes  
**Total estimated time**: 16-20 hours  
**Recommended sessions**: 8-10 sessions of 2 hours each

---

## ✅ FINAL CHECKLIST

After completing ALL 6 functions:

- [ ] All 6 monolithic test functions refactored
- [ ] All tests passing (100% pass rate)
- [ ] Quality score: 97.3 → 99.5+
- [ ] All tasks marked COMPLETE
- [ ] Progress tracking updated
- [ ] Git history clean and well-documented
- [ ] Celebrate! 🎉

---

**Remember**: This is a marathon, not a sprint. Take your time, test frequently, and commit often!

