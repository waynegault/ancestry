# Complete List of Functions Requiring Refactoring
**Date**: 2025-10-04  
**Analysis**: Quality metrics, complexity analysis, linting violations  
**Focus**: Biggest challenges (complexity >10, quality <70/100)

---

## 🚨 CRITICAL PRIORITY FUNCTIONS

### run_all_tests.py
- **main()** - Complexity: 39 (Target: <10)
  - Central test orchestration
  - Needs extraction of 5+ helper functions

### action10.py
- **action10_module_tests()** - Lines: 917, Complexity: 49
  - Largest monolithic test function
  - Break into 20-30 individual tests

### action6_gather.py (Quality: 28.7/100)
- **_main_page_processing_loop()** - Complexity: 12
- **_prepare_bulk_db_data()** - Complexity: 11
- **_do_batch()** - Complexity: 11
- Plus 10 additional violations

---

## 🔥 HIGH PRIORITY FUNCTIONS

### credential_manager.py
- **credential_manager_module_tests()** - Lines: 615, Complexity: 17

### gedcom_utils.py (Quality: 56.5/100)
- **_check_relationship_type()** - Complexity: 23 ⚠️ CRITICAL
- **_get_full_name()** - Complexity: 11
- **_are_spouses()** - Complexity: 11
- Plus 5 additional violations

### main.py
- **_dispatch_menu_action()** - Complexity: 23 ⚠️ CRITICAL
- **main_module_tests()** - Lines: 540, Complexity: 13

### api_utils.py (Quality: 34.8/100)
- **call_getladder_api()** - Complexity: 15
- **_extract_event_from_api_details()** - Complexity: 13
- **call_suggest_api()** - Complexity: 12
- Plus 9 additional violations

### action8_messaging.py
- **action8_messaging_tests()** - Lines: 537, Complexity: 26

---

## ⚙️ MEDIUM PRIORITY FUNCTIONS

### session_manager.py (Quality: 44.7/100)
- **session_manager_module_tests()** - Complexity: 14
- **_sync_cookies()** - Complexity: 13
- **my_profile_id** - Missing type hints
- Plus 7 additional violations

### database.py (Quality: 49.2/100)
- **create_or_update_family_tree()** - Complexity: 14
- **create_or_update_person()** - Complexity: 14
- **_compare_field_values()** - Complexity: 12
- Plus 6 additional violations

### credentials.py (Quality: 48.1/100)
- **setup_credentials()** - Complexity: 16
- **main()** - Complexity: 16
- **credentials_module_tests()** - Complexity: 13
- Plus 5 additional violations

### genealogical_task_templates.py
- **genealogical_task_templates_module_tests()** - Lines: 485, Complexity: 19

### security_manager.py
- **security_manager_module_tests()** - Lines: 485

### extraction_quality.py
- **compute_task_quality()** - Complexity: 17

### relationship_utils.py (Quality: 72.2/100)
- **_determine_gedcom_relationship()** - Complexity: 12
- **_parse_discovery_relationship()** - Complexity: 12
- **_get_relationship_term()** - Complexity: 12
- Plus 2 additional violations

### action9_process_productive.py (Quality: 57.2/100)
- **_create_ms_tasks()** - Complexity: 11
- **__post_init__** (multiple) - Missing type hints
- Plus 4 additional violations

### message_personalization.py (Quality: 82.2/100)
- **_create_occupation_social_context()** - Complexity: 13
- **_create_generational_gap_analysis()** - Complexity: 12
- **__init__** - Missing type hints

### ms_graph_utils.py (Quality: 74.1/100)
- **acquire_token_device_flow()** - Complexity: 14
- **get_todo_list_id()** - Complexity: 14
- **create_todo_task()** - Complexity: 11

### action7_inbox.py
- **_get_all_conversations_api()** - Complexity: 12
- **_determine_fetch_need()** - Complexity: 13
- **_process_conversations_in_batch()** - Complexity: 13

### config_manager.py
- **run_setup_wizard()** - Complexity: 12
- **validate_system_requirements()** - Complexity: 11
- **config_manager_module_tests()** - Complexity: 11

### ai_interface.py
- **test_ai_functionality()** - Complexity: 19
- **test_configuration()** - Complexity: 12
- **test_prompt_loading()** - Complexity: 11

### gedcom_cache.py
- **load_gedcom_with_aggressive_caching()** - Complexity: 11
- **get_health_status()** - Complexity: 11

### config_schema.py
- **validate()** - Complexity: 16
- **_test_rate_limiting_configuration()** - Complexity: 11

### database_manager.py
- **_initialize_engine_and_session()** - Complexity: 13
- **get_session_context()** - Complexity: 13

### api_search_utils.py
- **_parse_lifespan()** - Complexity: 11
- **get_api_relationship_path()** - Complexity: 11

### universal_scoring.py
- **validate_search_criteria()** - Complexity: 12

### session_validator.py
- **perform_readiness_checks()** - Complexity: 13

### logging_utils.py
- **logging_utils_module_tests()** - Complexity: 14

### logging_config.py
- **format()** - Complexity: 11

### health_monitor.py
- **restore_from_checkpoint()** - Complexity: 14

### research_prioritization.py
- **_calculate_conflict_priority_score()** - Complexity: 14

### action11.py
- **action11_module_tests()** - Complexity: 16

### cache.py
- **enforce_cache_size_limit()** - Complexity: 11
- **cache_module_tests()** - Complexity: 15

### config.py
- **config_module_tests()** - Complexity: 11

### adaptive_rate_limiter.py
- **test_regression_prevention_rate_limiter_caching()** - Complexity: 11

---

## 🏗️ ARCHITECTURAL ISSUES (Systemic)

### Superfluous-else-return (71 violations)
**Impact**: Code smell, unnecessary nesting  
**Solution**: Use early returns, eliminate else blocks  
**Files Affected**: Multiple across codebase

### Global-statement (47 violations)
**Impact**: Architectural issue, tight coupling  
**Solution**: Dependency injection, class attributes, function parameters  
**Files Affected**: Multiple across codebase

### Too-many-return-statements (45 violations)
**Impact**: Complexity, difficult to follow logic  
**Solution**: Consolidate return logic, use result variables  
**Files Affected**: Multiple across codebase

---

## 📊 SUMMARY STATISTICS

### By Severity
- **Critical Functions**: 15 functions
- **High Priority Functions**: 25 functions
- **Medium Priority Functions**: 40+ functions
- **Architectural Issues**: 163 violations

### By Type
- **Monolithic Test Functions**: 6 functions (3,600+ lines)
- **High Complexity Functions**: 35+ functions (complexity >10)
- **Missing Type Hints**: 10+ functions
- **Low Quality Modules**: 6 modules (<60/100)

### Total Effort Estimate
- **Critical**: 48-60 hours
- **High**: 58-72 hours
- **Medium**: 72-96 hours
- **Architectural**: 28-40 hours
- **TOTAL**: 206-268 hours

---

## ✅ REFACTORING APPROACH

### For High Complexity Functions
1. Extract helper functions for distinct responsibilities
2. Use early returns to reduce nesting
3. Consolidate error handling
4. Add comprehensive type hints
5. Write unit tests for extracted functions

### For Monolithic Test Functions
1. Break into individual test functions
2. Use TestSuite pattern from test_framework
3. Enable individual test execution
4. Improve test failure diagnostics
5. Follow naming convention: test_<feature>_<scenario>

### For Low Quality Modules
1. Address all complexity violations first
2. Add missing type hints
3. Refactor duplicate code
4. Improve error handling
5. Run quality checks after each change

---

**All functions listed above have been added to Augment Tasks for tracking.**
**This list focuses exclusively on BIGGEST CHALLENGES requiring significant refactoring effort.**

