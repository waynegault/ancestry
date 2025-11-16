# Code Quality & Testing Review Tasks

## Overview
This document tracks technical debt and quality improvement tasks for the Ancestry Genealogical Research Automation project. Tasks are organized by priority and impact on code quality, maintainability, and reliability.

---

## 🎯 High Priority

### 1. Test Quality Enforcement (PARTIALLY COMPLETE)
**Status**: 🟡 In Progress
**Current State**:
- ✅ `analyze_test_quality.py` exists and analyzes test quality across all modules
- ✅ `run_all_tests.py` discovers and runs 57+ test modules with comprehensive reporting
- ✅ `test_framework.py` provides standardized `TestSuite` class and utilities
- ✅ `test_utilities.py` provides centralized DRY utilities (`EmptyTestService`, `mock_func`, etc.)
- ✅ Most modules use `create_standard_test_runner()` pattern from `test_utilities.py`

**Remaining Work**:
- Integrate `analyze_test_quality.py` as a **pre-test gate** in `run_all_tests.py`
  - Block test execution if quality score < threshold (e.g., 70%)
  - Add `--skip-quality-gate` flag for emergency use
- Add quality metrics to test summary report:
  ```
  📊 Test Quality: 85% (smoke: 5, no assertions: 2, always-true: 1)
  ```
- Document quality standards in `readme.md` testing section

**Impact**: HIGH - Prevents regression in test quality and enforces meaningful tests

---

### 2. Standardize Entry Points (MOSTLY COMPLETE)
**Status**: 🟢 Near Complete  
**Current State**:
- ✅ ~45+ modules use standardized `run_comprehensive_tests = create_standard_test_runner(module_tests)` pattern
- ⚠️ ~5-10 modules still use legacy manual `def run_comprehensive_tests():` implementations
  - Examples: `rate_limiter.py`, `relationship_diagram.py`, `record_sharing.py`, `research_guidance_prompts.py`

**Remaining Work**:
- Audit remaining manual implementations:
  ```bash
  # Find legacy patterns
  ruff check . --select "# Manual run_comprehensive_tests"
  ```
- Refactor to use `create_standard_test_runner()` from `test_utilities.py`
- Add linting rule to enforce pattern (custom ruff rule or pre-commit hook)

**Impact**: MEDIUM - Improves consistency and reduces maintenance burden

---

### 3. Strengthen Test Assertions
**Status**: 🔴 Not Started  
**Modules Flagged**: `gedcom_intelligence.py`, `message_personalization.py`, and others per `analyze_test_quality.py`

**Tasks**:
1. Run quality analysis to identify weak tests:
   ```bash
   python analyze_test_quality.py
   ```
2. For each module with quality issues:
   - Replace smoke tests (always-pass) with real assertions
   - Add assertions to test functions lacking validation
   - Remove tests that validate trivial behavior (e.g., `assert True`)
3. Target minimum quality score of 85% per module

**Quality Criteria**:
- All tests must have at least 1 meaningful assertion
- Avoid `assert True`, `assert isinstance(x, object)` (trivial)
- Test real behavior, not just function existence

**Impact**: HIGH - Ensures tests actually validate correctness

---

## 🔧 Medium Priority

### 4. Separate Unit vs Integration Tests
**Status**: 🔴 Not Started  
**Current State**: Tests are embedded in source modules using `if __name__ == "__main__"` pattern

**Proposed Structure**:
```
tests/
  unit/
    test_utils.py
    test_session_utils.py
    test_gedcom_utils.py
    ...
  integration/
    test_action6_gather.py
    test_action7_inbox.py
    test_api_search_core.py (requires live session)
    ...
  conftest.py  # Shared fixtures for live sessions
  helpers/
    session_fixtures.py  # Live session management
    mock_factories.py    # Test data generators
```

**Benefits**:
- Clear separation of fast unit tests vs slow integration tests
- Enables `pytest -m unit` for quick validation during development
- Shared fixtures reduce duplication in integration tests

**Migration Plan**:
1. Create `tests/` directory structure
2. Move tests from modules to appropriate `test_*.py` files
3. Keep `if __name__ == "__main__"` for backward compatibility (calls test file)
4. Update `run_all_tests.py` to discover tests in both locations

**Impact**: MEDIUM - Improves test organization and developer experience

---

### 5. Consolidate Temp File/Dir Helpers
**Status**: 🔴 Not Started  
**Current Duplication**: Multiple modules create ad-hoc temporary files/directories

**Proposed Solution** (in `test_utilities.py`):
```python
@contextmanager
def temp_file(suffix: str = ".txt", content: Optional[str] = None) -> Iterator[Path]:
    """Context manager for temporary file with auto-cleanup."""
    temp = Path(tempfile.mktemp(suffix=suffix))
    try:
        if content:
            temp.write_text(content)
        yield temp
    finally:
        temp.unlink(missing_ok=True)

@contextmanager
def temp_directory() -> Iterator[Path]:
    """Context manager for temporary directory with auto-cleanup."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
```

**Usage**:
```python
from test_utilities import temp_file, temp_directory

def test_file_processing():
    with temp_file(".json", '{"key": "value"}') as f:
        result = process_file(f)
        assert result == expected
```

**Impact**: LOW-MEDIUM - Reduces code duplication and improves test reliability

---

## 📋 Low Priority / Maintenance

### 6. Code Quality Enforcement
**Status**: 🟢 Partially Complete  
**Current State**:
- ✅ Ruff configured with comprehensive rule set (`.ruff.toml`)
  - E, F, W (pycodestyle), I (isort), B (bugbear), PL (pylint), etc.
- ✅ Pyright configured with `basic` mode (`pyrightconfig.json`)
  - Critical errors enabled (undefined variables, type mismatches)
  - False positives suppressed (unused variables, unreachable code)
- ⚠️ Project-wide ignore patterns reduce strictness (intentional for pragmatism)

**Current Suppressions** (from `.ruff.toml`):
```toml
ignore = [
    "E501",      # line length (120 chars)
    "E402",      # imports after code (setup_module pattern)
    "F401",      # unused imports (re-exports)
    "PLR0913",   # many arguments (config classes)
    "PLR0915",   # many statements (main functions)
    ...
]
```

**Discussion**:
- **Goal**: 100% quality without suppressions is aspirational but impractical for a 20K+ LOC codebase
- **Reality**: Strategic suppressions enable practical development without sacrificing core quality
- **Recommendation**:
  - Keep current enforcement level (strict on critical issues, pragmatic on stylistic ones)
  - Consider tightening specific rules incrementally (e.g., enable `N806` for new code only)
  - Document rationale for each suppression in `.ruff.toml` (already done)

**Impact**: LOW - Current configuration is reasonable; avoid perfectionism paralysis

---

### 7. Test Coverage Analysis
**Status**: 🔴 Not Started  
**Current State**: No automated coverage tracking

**Proposed Enhancement**:
```bash
# Install coverage.py
pip install coverage pytest-cov

# Run tests with coverage
pytest --cov=. --cov-report=html --cov-report=term

# Add to run_all_tests.py
python run_all_tests.py --coverage  # Generate coverage report
```

**Target Metrics**:
- Core modules (session_manager, database, utils): 80%+ coverage
- Action modules (6-10): 70%+ coverage
- Utility modules: 60%+ coverage

**Impact**: LOW - Nice-to-have, but not blocking current development

---

## 📊 Progress Summary

| Task | Priority | Status | Completion |
|------|----------|--------|------------|
| Test Quality Enforcement | High | 🟡 In Progress | 70% |
| Standardize Entry Points | High | 🟢 Near Complete | 85% |
| Strengthen Assertions | High | 🔴 Not Started | 0% |
| Separate Unit/Integration | Medium | 🔴 Not Started | 0% |
| Consolidate Temp Helpers | Medium | 🔴 Not Started | 0% |
| Code Quality Enforcement | Low | 🟢 Complete | 95% |
| Test Coverage Analysis | Low | 🔴 Not Started | 0% |

**Overall Progress**: 36% complete (3.5/7 tasks)

---

## 🚀 Next Steps

1. **Immediate** (This Week):
   - Integrate `analyze_test_quality.py` into `run_all_tests.py` as a pre-test gate
   - Refactor remaining 5-10 modules to use `create_standard_test_runner()`

2. **Short-term** (Next 2 Weeks):
   - Run `analyze_test_quality.py` and create task list for modules with quality < 70%
   - Strengthen assertions in high-priority modules (action6, action7, database)

3. **Medium-term** (Next Month):
   - Design and prototype `tests/` directory structure
   - Implement temp file/directory helpers in `test_utilities.py`

4. **Long-term** (Backlog):
   - Migrate tests to separate `tests/` directory
   - Add coverage tracking to CI/CD pipeline

---

_Last updated on 2025-11-16 by waynegault_
_Next review: 2025-11-30_
