# Too-Many-Arguments Refactoring Plan

**Created**: 2025-10-06  
**Status**: PLANNING PHASE  
**Scope**: 120+ functions across 20+ files  
**Estimated Effort**: 20-30 hours

---

## ⚠️ CRITICAL ASSESSMENT

This is the **BIGGEST refactoring challenge** in the codebase. It requires:

1. **Architectural changes** - Not just code cleanup
2. **Extensive testing** - Every change must be validated
3. **Coordinated refactoring** - Functions call each other
4. **Backward compatibility** - Can't break existing code

### Why This Is Complex

The too-many-arguments violations are NOT isolated issues. They form **call chains**:

```
_api_req (16 args)
  ↓ calls
_execute_request_with_retries (21 args)
  ↓ calls
_process_request_attempt (23 args)
  ↓ calls
_prepare_api_request (16 args)
```

Refactoring ONE function requires refactoring ALL functions in the chain.

---

## 📊 VIOLATION BREAKDOWN

### By File (Sorted by Severity)

| File | Functions | Worst Args | Total Violations |
|------|-----------|------------|------------------|
| utils.py | 18 | 23 | High Priority |
| action8_messaging.py | 17 | 18 | High Priority |
| action6_gather.py | 21 | 10 | Medium Priority |
| action7_inbox.py | 10 | 12 | Medium Priority |
| action11.py | 8 | 10 | Medium Priority |
| gedcom_utils.py | 5 | 9 | Low Priority |
| relationship_utils.py | 6 | 8 | Low Priority |
| action10.py | 2 | 8 | Low Priority |
| run_all_tests.py | 4 | 9 | Low Priority |
| Others | 29+ | 6-7 | Low Priority |

### By Function Type

| Type | Count | Strategy |
|------|-------|----------|
| API Request Functions | 15 | ApiRequestConfig dataclass |
| Test Helper Functions | 40+ | TestConfig dataclass |
| Database Functions | 10 | DbOperationConfig dataclass |
| Message Functions | 20 | MessageConfig dataclass |
| Processing Functions | 35+ | ProcessingConfig dataclass |

---

## 🎯 RECOMMENDED PHASED APPROACH

### Phase 1: Proof of Concept (4-6 hours)
**Goal**: Validate the approach with ONE file

**Target**: utils.py API request chain
- Create `ApiRequestConfig` dataclass ✅ (already done)
- Refactor `_prepare_api_request` (16 args → 1)
- Refactor `_process_request_attempt` (23 args → 1)
- Refactor `_execute_request_with_retries` (21 args → 1)
- Refactor `_api_req` (16 args → 1)
- Update all callers in utils.py
- Run tests to verify no regressions

**Success Criteria**:
- All tests pass
- No functionality changes
- Code is cleaner and more maintainable

### Phase 2: Extend to Test Functions (6-8 hours)
**Goal**: Refactor test helper functions

**Target**: utils.py test functions
- Create `TestMessageConfig` dataclass
- Refactor test functions with 10+ args
- Update test callers

### Phase 3: Action Modules (8-12 hours)
**Goal**: Refactor action modules

**Targets**:
- action8_messaging.py (17 functions)
- action6_gather.py (21 functions)
- action7_inbox.py (10 functions)
- action11.py (8 functions)

### Phase 4: Remaining Files (4-6 hours)
**Goal**: Complete the refactoring

**Targets**:
- gedcom_utils.py
- relationship_utils.py
- action10.py
- run_all_tests.py
- Other files

---

## 🔧 REFACTORING PATTERNS

### Pattern 1: Configuration Dataclass

**Before**:
```python
def _process_request_attempt(
    session_manager: SessionManager,
    driver: DriverType,
    url: str,
    method: str,
    api_description: str,
    attempt: int,
    headers: Optional[Dict[str, str]],
    referer_url: Optional[str],
    use_csrf_token: bool,
    add_default_origin: bool,
    timeout: Optional[int],
    cookie_jar: Optional[RequestsCookieJar],
    allow_redirects: bool,
    data: Optional[Dict],
    json_data: Optional[Dict],
    json: Optional[Dict],
    force_text_response: bool,
    retry_status_codes: List[int],
    retries_left: int,
    max_retries: int,
    current_delay: float,
    backoff_factor: float,
    max_delay: float,
) -> tuple[Optional[Any], bool, int, float, Optional[Exception]]:
    pass
```

**After**:
```python
@dataclass
class ApiRequestConfig:
    url: str
    driver: DriverType
    session_manager: SessionManager
    method: str = "GET"
    # ... all other parameters with defaults
    
def _process_request_attempt(
    config: ApiRequestConfig,
    retries_left: int,
    current_delay: float,
) -> tuple[Optional[Any], bool, int, float, Optional[Exception]]:
    # Access config.url, config.method, etc.
    pass
```

### Pattern 2: Builder Pattern (for complex construction)

```python
class ApiRequestBuilder:
    def __init__(self, url: str, session_manager: SessionManager):
        self.config = ApiRequestConfig(url=url, session_manager=session_manager)
    
    def with_method(self, method: str) -> 'ApiRequestBuilder':
        self.config.method = method
        return self
    
    def with_headers(self, headers: Dict[str, str]) -> 'ApiRequestBuilder':
        self.config.headers = headers
        return self
    
    def build(self) -> ApiRequestConfig:
        return self.config

# Usage
config = (ApiRequestBuilder(url, session_manager)
    .with_method("POST")
    .with_headers({"Content-Type": "application/json"})
    .build())
```

---

## ⚠️ RISKS AND MITIGATION

### Risk 1: Breaking Changes
**Mitigation**: 
- Keep old function signatures temporarily
- Add deprecation warnings
- Provide migration guide

### Risk 2: Test Failures
**Mitigation**:
- Run full test suite after each function
- Git commit after each successful refactoring
- Revert immediately if tests fail

### Risk 3: Performance Impact
**Mitigation**:
- Dataclasses are lightweight
- No runtime performance impact expected
- Benchmark critical paths if needed

### Risk 4: Incomplete Refactoring
**Mitigation**:
- Track progress in this document
- Use linter to verify completion
- Don't merge until 100% complete

---

## 📋 PROGRESS TRACKING

### Phase 1: Proof of Concept
- [x] Create ApiRequestConfig dataclass
- [ ] Refactor _prepare_api_request
- [ ] Refactor _process_request_attempt
- [ ] Refactor _execute_request_with_retries
- [ ] Refactor _api_req
- [ ] Update all callers in utils.py
- [ ] Run tests and verify

### Phase 2: Test Functions
- [ ] Create TestMessageConfig dataclass
- [ ] Refactor test functions (0/40+)
- [ ] Update test callers
- [ ] Run tests and verify

### Phase 3: Action Modules
- [ ] action8_messaging.py (0/17)
- [ ] action6_gather.py (0/21)
- [ ] action7_inbox.py (0/10)
- [ ] action11.py (0/8)
- [ ] Run tests and verify

### Phase 4: Remaining Files
- [ ] gedcom_utils.py (0/5)
- [ ] relationship_utils.py (0/6)
- [ ] action10.py (0/2)
- [ ] run_all_tests.py (0/4)
- [ ] Other files (0/29+)
- [ ] Run tests and verify

---

## 🎯 DECISION POINT

**Question for User**: Should I proceed with Phase 1 (Proof of Concept)?

**Phase 1 Details**:
- Time: 4-6 hours
- Scope: utils.py API request chain only (4 functions)
- Risk: Low (isolated to one file)
- Benefit: Validates approach before full commitment

**Options**:
1. ✅ **Proceed with Phase 1** - Refactor utils.py API chain as proof of concept
2. ⏸️ **Pause** - Review plan and discuss approach first
3. 🔄 **Alternative** - Focus on easier tasks first (global statements, type hints)

**Recommendation**: Start with Phase 1 to validate the approach. If successful, continue with remaining phases. If issues arise, we can adjust the strategy.

---

## 📝 NOTES

- The `ApiRequestConfig` dataclass has been created in utils.py (lines 81-122)
- This is ready to use for Phase 1 refactoring
- All tests currently pass (100% pass rate)
- Quality score is 98.2/100

**Next Step**: Await user decision on proceeding with Phase 1.

