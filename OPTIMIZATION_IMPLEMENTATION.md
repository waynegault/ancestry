# Test Performance Optimization - Implementation Guide

## Summary

The 3 slowest tests account for **718 seconds (12 minutes)** of test time:
1. **session_manager.py**: 610s - Creates 3 real browsers
2. **code_similarity_classifier.py**: 63s - Scans entire codebase
3. **action10.py**: 45s - Loads full GEDCOM files

## Root Cause Analysis

### 1. Session Manager (610s)

**Problem**: Line 3094-3111 in `core/session_manager.py`
```python
def _test_initialization_performance() -> bool:
    for _i in range(3):
        session_manager = SessionManager()  # ← Starts real browser!
        session_managers.append(session_manager)
```

Each `SessionManager()` call:
- Initializes DatabaseManager
- Initializes BrowserManager → **Starts ChromeDriver** (30-60s each!)
- Initializes APIManager
- Initializes SessionValidator

**3 browsers × 60s = 180s minimum**

Plus the 724-page simulation test adds more time.

### 2. Code Similarity Classifier (63s)

**Problem**: Lines 882-892 in `code_similarity_classifier.py`
```python
clsfr = CodeSimilarityClassifier(Path())
funcs = clsfr.scan()  # ← Scans ALL Python files!
pairs = clsfr.find_similar(min_ratio=0.86, max_hamming=10)  # ← O(n²) comparison!
```

- Scans 63+ Python modules
- Parses AST for each file
- Extracts hundreds of functions
- Compares all pairs for similarity

### 3. Action10 (45s)

**Problem**: Line 1439 in `action10.py`
```python
disable_mock_mode()  # ← Forces real GEDCOM loading!
```

- Loads full GEDCOM file (potentially thousands of individuals)
- Parses and indexes all data
- Runs scoring on real data
- Multiple tests each reload the data

---

## Optimization Strategy

### Phase 1: Mock Browser Initialization (Saves ~540s)

**File**: `core/session_manager.py`

**Change**: Mock browser initialization in performance test

```python
def _test_initialization_performance() -> bool:
    """Test SessionManager initialization performance (mocked browser for speed)."""
    import time
    from unittest.mock import patch, MagicMock
    
    # Mock browser initialization to avoid slow ChromeDriver startup
    with patch('core.browser_manager.BrowserManager.start_browser', return_value=True), \
         patch('core.browser_manager.BrowserManager.__init__', return_value=None):
        
        session_managers = []
        start_time = time.time()
        for _i in range(3):
            session_manager = SessionManager()
            session_managers.append(session_manager)
        end_time = time.time()
        total_time = end_time - start_time
        
        # With mocked browser, should be very fast
        max_time = 2.0  # Reduced from 5.0s since browser is mocked
        assert (
            total_time < max_time
        ), f"3 optimized initializations took {total_time:.3f}s, should be under {max_time}s"
        
        for sm in session_managers:
            with contextlib.suppress(Exception):
                sm.close_sess(keep_db=True)
        return True
```

**Expected**: 180s → 2s (99% reduction)

---

### Phase 2: Skip Slow Tests in Fast Mode (Saves ~60s)

**File**: `run_all_tests.py`

**Change**: Add environment variable for fast mode

```python
def main() -> bool:
    # ... existing code ...
    
    # Set environment variable to skip live API tests that require browser/network
    os.environ["SKIP_LIVE_API_TESTS"] = "true"
    
    # NEW: Set environment variable to skip slow simulation tests
    os.environ["SKIP_SLOW_TESTS"] = "true"
    
    # ... rest of code ...
```

**File**: `core/session_manager.py`

**Change**: Skip 724-page simulation in fast mode

```python
def test_724_page_workload_simulation():
    """Simulate 724-page workload with realistic error injection."""
    # Skip in fast mode
    if os.getenv("SKIP_SLOW_TESTS", "false").lower() == "true":
        logger.info("Skipping 724-page workload simulation (SKIP_SLOW_TESTS=true)")
        return True
    
    # ... rest of simulation code ...
```

**Expected**: 60s → 0s (100% reduction)

---

### Phase 3: Cache Code Similarity Results (Saves ~60s)

**File**: `code_similarity_classifier.py`

**Change**: Add caching and fast mode skip

```python
def code_similarity_classifier_module_tests() -> bool:
    """Minimal, strict tests that validate this module actually analyzes code."""
    import os
    
    # Skip expensive similarity detection in fast mode
    skip_slow = os.getenv("SKIP_SLOW_TESTS", "false").lower() == "true"
    
    try:
        clsfr = CodeSimilarityClassifier(Path())
        funcs = clsfr.scan()
        
        # Expect a nontrivial codebase: require at least 40 functions discovered
        assert len(funcs) >= 40, f"Too few functions discovered: {len(funcs)}"
        
        # Sanity check of fields
        sample = funcs[min(5, len(funcs) - 1)]
        assert sample.module_path.endswith(".py"), "module_path should be a .py file"
        assert sample.qualname and isinstance(sample.qualname, str)
        assert sample.loc >= 1 and sample.lineno >= 1
        
        # Skip expensive similarity detection in fast mode
        if not skip_slow:
            # Similarity should find at least a few candidates in a real repo
            pairs = clsfr.find_similar(min_ratio=0.86, max_hamming=10)
            # Not required to be many, but expect at least 1 in a sizable repo
            assert len(pairs) >= 1, "No similar function pairs found — unexpected in this repo"
        
        # JSON serialization (fast)
        blob = clsfr.to_json()
        assert "functions" in blob and "similar_pairs" in blob and "clusters" in blob
        
        return True
    except AssertionError as e:
        print(f"TEST FAILURE (code_similarity_classifier): {e}")
        return False
    except Exception as e:
        print(f"TEST ERROR (code_similarity_classifier): {e}")
        return False
```

**Expected**: 63s → 5s (92% reduction)

---

### Phase 4: Use Minimal Test GEDCOM (Saves ~35s)

**File**: Create `test_data/minimal_test.ged`

**Content**: Minimal GEDCOM with just Fraser Gault and immediate family

```gedcom
0 HEAD
1 SOUR Ancestry Test Data
1 GEDC
2 VERS 5.5.1
0 @I1@ INDI
1 NAME Fraser /Gault/
1 SEX M
1 BIRT
2 DATE 1941
2 PLAC Banff, Banffshire, Scotland
0 @I2@ INDI
1 NAME Margaret /Gault/
1 SEX F
1 FAMS @F1@
0 @F1@ FAM
1 HUSB @I1@
1 WIFE @I2@
0 TRLR
```

**File**: `action10.py`

**Change**: Use minimal GEDCOM for tests

```python
def action10_module_tests() -> bool:
    """Comprehensive test suite for action10.py"""
    import builtins
    import time
    import os
    
    from test_framework import (
        Colors,
        TestSuite,
        clean_test_output,
        format_score_breakdown_table,
        format_search_criteria,
        format_test_section_header,
    )
    
    # Use minimal test GEDCOM for faster tests
    original_gedcom = os.getenv("GEDCOM_FILE_PATH")
    test_gedcom = "test_data/minimal_test.ged"
    
    # Only use minimal GEDCOM if it exists and we're in fast mode
    if os.path.exists(test_gedcom) and os.getenv("SKIP_SLOW_TESTS", "false").lower() == "true":
        os.environ["GEDCOM_FILE_PATH"] = test_gedcom
        logger.info(f"Using minimal test GEDCOM: {test_gedcom}")
    
    try:
        # PHASE 4.2: Disable mock mode - use real GEDCOM data for testing
        disable_mock_mode()
        
        # ... rest of tests ...
        
    finally:
        # Restore original GEDCOM path
        if original_gedcom:
            os.environ["GEDCOM_FILE_PATH"] = original_gedcom
        else:
            os.environ.pop("GEDCOM_FILE_PATH", None)
```

**Expected**: 45s → 10s (78% reduction)

---

## Implementation Checklist

### Step 1: Update run_all_tests.py
- [ ] Add `os.environ["SKIP_SLOW_TESTS"] = "true"` in main()
- [ ] Test that environment variable is set correctly

### Step 2: Update core/session_manager.py
- [ ] Mock browser initialization in `_test_initialization_performance()`
- [ ] Add skip check to `test_724_page_workload_simulation()`
- [ ] Reduce max_time from 5.0s to 2.0s in performance test
- [ ] Test that mocked version works correctly

### Step 3: Update code_similarity_classifier.py
- [ ] Add skip check for similarity detection
- [ ] Test that basic scanning still works
- [ ] Verify JSON serialization still tested

### Step 4: Create minimal test GEDCOM
- [ ] Create `test_data/` directory if it doesn't exist
- [ ] Create `minimal_test.ged` with Fraser Gault data
- [ ] Update action10.py to use minimal GEDCOM in fast mode
- [ ] Test that action10 tests still pass

### Step 5: Verify Results
- [ ] Run `python run_all_tests.py` and measure total time
- [ ] Verify all tests still pass
- [ ] Check that slow tests are actually being skipped
- [ ] Measure time for each of the 3 optimized tests

---

## Expected Results

| Test | Before | After | Reduction |
|------|--------|-------|-----------|
| session_manager.py | 610s | 5s | 99.2% |
| code_similarity_classifier.py | 63s | 5s | 92.1% |
| action10.py | 45s | 10s | 77.8% |
| **Total** | **718s** | **20s** | **97.2%** |

**Overall test suite improvement**: From ~12 minutes to ~30 seconds

---

## Rollback Plan

If optimizations cause issues:

1. **Remove SKIP_SLOW_TESTS from run_all_tests.py**
2. **Revert session_manager.py changes** (git checkout)
3. **Revert code_similarity_classifier.py changes** (git checkout)
4. **Revert action10.py changes** (git checkout)

All changes are isolated and can be reverted independently.

---

## Next Steps

Would you like me to:
1. Implement all optimizations now?
2. Implement them one at a time for testing?
3. Start with just the session_manager optimization (biggest impact)?

Please advise on your preferred approach.

