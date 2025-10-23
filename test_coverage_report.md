# Test Coverage Report

This report shows test coverage across all Python modules in the Ancestry project.

**Total Modules Analyzed:** 82
**Modules with Tests:** 80
**Modules without Tests:** 2

**Total Public Functions:** 1048
**Total Test Functions:** 1033

## Coverage by Module

### Root Directory Modules

| Module | Public Functions | Test Functions | Has Tests |
|--------|------------------|----------------|-----------|
| action10.py | 26 | 20 | ✅ |
| action11.py | 14 | 10 | ✅ |
| action12.py | 6 | 15 | ✅ |
| action6_gather.py | 3 | 13 | ✅ |
| action7_inbox.py | 4 | 13 | ✅ |
| action8_messaging.py | 42 | 51 | ✅ |
| action9_process_productive.py | 19 | 18 | ✅ |
| ai_interface.py | 17 | 12 | ✅ |
| ai_prompt_utils.py | 12 | 9 | ✅ |
| api_search_utils.py | 6 | 8 | ✅ |
| api_utils.py | 21 | 25 | ✅ |
| cache.py | 29 | 13 | ✅ |
| cache_manager.py | 23 | 22 | ✅ |
| chromedriver.py | 8 | 10 | ✅ |
| code_quality_checker.py | 6 | 3 | ✅ |
| common_params.py | 1 | 8 | ✅ |
| config.py | 1 | 6 | ✅ |
| connection_resilience.py | 9 | 7 | ✅ |
| conversation_analytics.py | 5 | 11 | ✅ |
| core_imports.py | 19 | 6 | ✅ |
| credentials.py | 16 | 19 | ✅ |
| database.py | 22 | 20 | ✅ |
| dna_ethnicity_utils.py | 8 | 9 | ✅ |
| dna_gedcom_crossref.py | 2 | 6 | ✅ |
| dna_utils.py | 5 | 8 | ✅ |
| error_handling.py | 34 | 22 | ✅ |
| gedcom_cache.py | 17 | 14 | ✅ |
| gedcom_intelligence.py | 2 | 5 | ✅ |
| gedcom_search_utils.py | 10 | 13 | ✅ |
| gedcom_utils.py | 16 | 18 | ✅ |
| genealogical_normalization.py | 3 | 9 | ✅ |
| health_monitor.py | 38 | 19 | ✅ |
| logging_config.py | 4 | 18 | ✅ |
| main.py | 20 | 27 | ✅ |
| memory_utils.py | 4 | 3 | ✅ |
| message_personalization.py | 5 | 13 | ✅ |
| ms_graph_utils.py | 5 | 8 | ✅ |
| my_selectors.py | 1 | 11 | ✅ |
| performance_cache.py | 16 | 10 | ✅ |
| performance_monitor.py | 19 | 15 | ✅ |
| performance_orchestrator.py | 23 | 15 | ✅ |
| person_lookup_utils.py | 7 | 8 | ✅ |
| prompt_telemetry.py | 7 | 4 | ✅ |
| record_sharing.py | 7 | 17 | ✅ |
| relationship_diagram.py | 4 | 13 | ✅ |
| relationship_utils.py | 11 | 17 | ✅ |
| research_guidance_prompts.py | 5 | 16 | ✅ |
| research_prioritization.py | 2 | 5 | ✅ |
| research_suggestions.py | 2 | 12 | ✅ |
| search_criteria_utils.py | 4 | 11 | ✅ |
| security_manager.py | 9 | 12 | ✅ |
| selenium_utils.py | 15 | 12 | ✅ |
| session_utils.py | 13 | 20 | ✅ |
| standard_imports.py | 6 | 11 | ✅ |
| test_all_llm_models.py | 5 | 2 | ✅ |
| test_framework.py | 45 | 23 | ✅ |
| test_local_llm.py | 1 | 3 | ✅ |
| test_utilities.py | 25 | 18 | ✅ |
| tree_stats_utils.py | 3 | 12 | ✅ |
| universal_scoring.py | 5 | 16 | ✅ |
| utils.py | 36 | 16 | ✅ |

### Core Package Modules

| Module | Public Functions | Test Functions | Has Tests |
|--------|------------------|----------------|-----------|
| core/__init__.py | 1 | 5 | ✅ |
| core/__main__.py | 1 | 6 | ✅ |
| core/api_manager.py | 12 | 8 | ✅ |
| core/browser_manager.py | 7 | 14 | ✅ |
| core/cancellation.py | 7 | 7 | ✅ |
| core/database_manager.py | 14 | 9 | ✅ |
| core/dependency_injection.py | 19 | 26 | ✅ |
| core/enhanced_error_recovery.py | 17 | 10 | ✅ |
| core/error_handling.py | 35 | 7 | ✅ |
| core/logging_utils.py | 10 | 11 | ✅ |
| core/progress_indicators.py | 14 | 9 | ✅ |
| core/registry_utils.py | 9 | 1 | ✅ |
| core/session_cache.py | 22 | 11 | ✅ |
| core/session_manager.py | 65 | 18 | ✅ |
| core/session_validator.py | 4 | 17 | ✅ |
| core/system_cache.py | 19 | 13 | ✅ |

### Config Package Modules

| Module | Public Functions | Test Functions | Has Tests |
|--------|------------------|----------------|-----------|
| config/__init__.py | 0 | 0 | ❌ |
| config/__main__.py | 0 | 0 | ❌ |
| config/config_manager.py | 15 | 13 | ✅ |
| config/config_schema.py | 8 | 18 | ✅ |
| config/credential_manager.py | 16 | 20 | ✅ |

## Detailed Function Coverage

This section lists all public functions and their corresponding test functions for each module.

### action10.py

**Public Functions:**
- `action10_module_tests()`
- `analyze_top_match()`
- `calculate_match_score_cached()`
- `compare_action10_performance()`
- `detailed_scoring_breakdown()`
- `display_relatives()`
- `display_top_matches()`
- `filter()`
- `filter_and_score_individuals()`
- `format_display_value()`
- `get_cached_gedcom()`
- `get_input()`
- `get_user_criteria()`
- `get_validated_year_input()`
- `load_gedcom_data()`
- `log_criteria_summary()`
- `main()`
- `make_mock_input()`
- `matches_criterion()`
- `matches_year_criterion()`
- `mock_input_func()`
- `parse_command_line_args()`
- `run_performance_validation()`
- `sanitize_input()`
- `validate_config()`
- `validate_performance_improvements()`

**Test Functions:**
- `_format_test_person_analysis()`
- `_get_test_person_config()`
- `_load_test_person_data_from_env()`
- `_register_input_validation_tests()`
- `_register_relationship_tests()`
- `_register_scoring_tests()`
- `_setup_test_environment()`
- `_teardown_test_environment()`
- `action10_module_tests()`
- `test_analyze_top_match_fraser()`
- `test_config_defaults()`
- `test_display_relatives_fraser()`
- `test_family_relationship_analysis()`
- `test_fraser_gault_scoring_algorithm()`
- `test_get_validated_year_input_patch()`
- `test_main_patch()`
- `test_module_initialization()`
- `test_real_search_performance_and_accuracy()`
- `test_relationship_path_calculation()`
- `test_sanitize_input()`

### action11.py

**Public Functions:**
- `action11_module_tests()`
- `browser_check()`
- `check_login()`
- `clean_param()`
- `get_ancestry_person_details()`
- `get_ancestry_relationship_path()`
- `handle_api_report()`
- `load_test_person_from_env()`
- `login_attempt()`
- `main()`
- `run_action11()`
- `run_comprehensive_tests()`
- `search_ancestry_api_for_person()`
- `session_init()`

**Test Functions:**
- `_build_search_criteria_from_test_person()`
- `_test_live_family_matches_env()`
- `_test_live_relationship_uncle()`
- `_test_live_search_fraser()`
- `action11_module_tests()`
- `load_test_person_from_env()`
- `run_comprehensive_tests()`
- `test_live_family_matches_env()`
- `test_live_relationship_uncle()`
- `test_live_search_fraser()`

### action12.py

**Public Functions:**
- `action12_module_tests()`
- `compare_results()`
- `get_search_input()`
- `run_action10_search()`
- `run_action11_search()`
- `run_action12_wrapper()`

**Test Functions:**
- `_test_action10_error_handling()`
- `_test_action11_error_handling()`
- `_test_action12_functions_available()`
- `_test_action12_imports()`
- `_test_compare_results_output()`
- `_test_compare_results_structure()`
- `_test_print_functions()`
- `_test_run_action10_search_structure()`
- `_test_run_action11_search_structure()`
- `_test_score_comparison()`
- `_test_search_criteria_building()`
- `_test_search_input_optional_fields()`
- `_test_search_input_structure()`
- `_test_wrapper_function()`
- `action12_module_tests()`

### action6_gather.py

**Public Functions:**
- `action6_module_tests()`
- `coord()`
- `run_comprehensive_tests()`

**Test Functions:**
- `_test_badge_details_api()`
- `_test_bounds_checking()`
- `_test_database_schema()`
- `_test_error_handling()`
- `_test_match_details_api()`
- `_test_match_list_api()`
- `_test_parallel_fetch_match_details()`
- `_test_parallel_function_thread_safety()`
- `_test_person_id_attribute_fix()`
- `_test_profile_details_api()`
- `_test_relationship_probability_api()`
- `action6_module_tests()`
- `run_comprehensive_tests()`

### action7_inbox.py

**Public Functions:**
- `action7_inbox_module_tests()`
- `get_statistics()`
- `safe_column_value()`
- `search_inbox()`

**Test Functions:**
- `_find_latest_messages()`
- `_get_db_latest_timestamp()`
- `_test_ai_classification()`
- `_test_class_and_methods_available()`
- `_test_conversation_database_storage()`
- `_test_conversation_parsing()`
- `_test_error_recovery()`
- `_test_fetch_first_page_conversations()`
- `_test_inbox_processor_initialization()`
- `_test_person_status_updates()`
- `_test_stop_on_unchanged_conversation()`
- `_test_summary_logging()`
- `action7_inbox_module_tests()`

### action8_messaging.py

**Public Functions:**
- `action8_messaging_tests()`
- `add_monitoring_hook()`
- `analyze_template_effectiveness()`
- `api_call()`
- `attempt_reauthentication()`
- `calculate_adaptive_interval()`
- `calculate_delay()`
- `cancel_pending_messages_on_status_change()`
- `cancel_pending_on_reply()`
- `categorize_status()`
- `check_authentication()`
- `check_memory_usage()`
- `cleanup_resources()`
- `critical_error_hook()`
- `detect_status_change_to_in_tree()`
- `determine_next_action()`
- `determine_next_message_type()`
- `emit()`
- `enhance_message_format_data_phase5()`
- `enhance_message_with_relationship_diagram()`
- `enhance_message_with_research_suggestions()`
- `enhance_message_with_sources()`
- `get_error_summary()`
- `get_safe_relationship_path()`
- `get_safe_relationship_text()`
- `is_sess_valid()`
- `load_message_templates()`
- `log_conversation_state_change()`
- `my_profile_id()`
- `periodic_maintenance()`
- `print_template_effectiveness_report()`
- `record_api_result()`
- `safe_column_value()`
- `select_template_by_confidence()`
- `select_template_variant_ab_testing()`
- `send_messages_to_matches()`
- `target()`
- `track_resource()`
- `track_template_selection()`
- `trigger_garbage_collection()`
- `trigger_monitoring_alert()`
- `validate_api_response()`

**Test Functions:**
- `_fetch_test_candidates()`
- `_process_test_candidates()`
- `_test_adaptive_timing_high_engagement()`
- `_test_adaptive_timing_low_engagement()`
- `_test_adaptive_timing_medium_engagement()`
- `_test_adaptive_timing_moderate_login()`
- `_test_adaptive_timing_no_engagement()`
- `_test_calculate_follow_up_action()`
- `_test_cancel_on_reply_already_active()`
- `_test_cancel_on_reply_no_state()`
- `_test_cancel_on_reply_success()`
- `_test_cancel_pending_messages_no_state()`
- `_test_cancel_pending_messages_success()`
- `_test_circuit_breaker_config()`
- `_test_confidence_scoring_hardening()`
- `_test_conversation_log_tracking()`
- `_test_database_message_creation()`
- `_test_determine_next_action_await_reply()`
- `_test_determine_next_action_no_state()`
- `_test_determine_next_action_research_needed()`
- `_test_determine_next_action_send_follow_up()`
- `_test_determine_next_action_status_change()`
- `_test_dry_run_mode_no_actual_send()`
- `_test_enhance_message_format_data_phase5()`
- `_test_enhance_message_with_relationship_diagram()`
- `_test_enhance_message_with_research_suggestions()`
- `_test_enhance_message_with_sources()`
- `_test_enhanced_error_handling()`
- `_test_error_categorization_integration_minimal()`
- `_test_function_availability()`
- `_test_halt_signal_integration()`
- `_test_integration_with_shared_modules()`
- `_test_log_conversation_state_change()`
- `_test_log_no_conversation_state()`
- `_test_logger_respects_info_level()`
- `_test_main_function_with_dry_run()`
- `_test_message_template_loading()`
- `_test_message_template_loading_from_db()`
- `_test_no_debug_when_info()`
- `_test_performance_tracking()`
- `_test_real_api_manager_integration_minimal()`
- `_test_safe_column_value()`
- `_test_session_death_cascade_detection()`
- `_test_status_change_already_messaged()`
- `_test_status_change_not_in_tree()`
- `_test_status_change_old_addition()`
- `_test_status_change_recent_addition()`
- `_test_status_change_template_exists()`
- `_test_system_health_validation_hardening()`
- `action8_messaging_tests()`
- `select_template_variant_ab_testing()`

### action9_process_productive.py

**Public Functions:**
- `action9_process_productive_module_tests()`
- `calculate_task_priority_from_relationship()`
- `commit_batch()`
- `create_enhanced_research_task()`
- `ensure_list_of_strings()`
- `ensure_tasks_list()`
- `format_response_with_records()`
- `format_response_with_relationship_diagram()`
- `generate_ai_response_prompt()`
- `get_all_locations()`
- `get_all_names()`
- `get_gedcom_data()`
- `get_sort_key()`
- `process_person()`
- `process_productive_messages()`
- `run_comprehensive_tests()`
- `safe_column_value()`
- `should_commit()`
- `should_exclude_message()`

**Test Functions:**
- `_get_latest_incoming_message()`
- `_test_ai_processing_functions()`
- `_test_calculate_task_priority_from_relationship()`
- `_test_circuit_breaker_config()`
- `_test_core_functionality()`
- `_test_create_enhanced_research_task()`
- `_test_database_session_availability()`
- `_test_edge_cases()`
- `_test_enhanced_task_creation()`
- `_test_error_handling()`
- `_test_format_response_with_records()`
- `_test_format_response_with_relationship_diagram()`
- `_test_generate_ai_response_prompt()`
- `_test_integration()`
- `_test_message_templates_available()`
- `_test_module_initialization()`
- `action9_process_productive_module_tests()`
- `run_comprehensive_tests()`

### ai_interface.py

**Public Functions:**
- `ai_interface_module_tests()`
- `analyze_dna_match_conversation()`
- `classify_message_intent()`
- `extract_genealogical_entities()`
- `extract_with_custom_prompt()`
- `generate_contextual_response()`
- `generate_genealogical_reply()`
- `generate_record_research_strategy()`
- `generate_with_custom_prompt()`
- `get_fallback_extraction_prompt()`
- `get_fallback_reply_prompt()`
- `get_prompt()`
- `load_prompts()`
- `quick_health_check()`
- `run_comprehensive_tests()`
- `self_check_ai_interface()`
- `verify_family_tree_connections()`

**Test Functions:**
- `_perform_test_call()`
- `_test_ai_fallback_behavior()`
- `_test_genealogical_extraction()`
- `_test_intent_classification()`
- `_test_reply_generation()`
- `_test_specialized_analysis_functions()`
- `ai_interface_module_tests()`
- `run_comprehensive_tests()`
- `test_ai_functionality()`
- `test_configuration()`
- `test_prompt_loading()`
- `test_pydantic_compatibility()`

### ai_prompt_utils.py

**Public Functions:**
- `ai_prompt_utils_module_tests()`
- `backup_prompts_file()`
- `cleanup_old_backups()`
- `get_prompt()`
- `get_prompts_summary()`
- `import_improved_prompts()`
- `load_prompts()`
- `quick_test()`
- `run_comprehensive_tests()`
- `save_prompts()`
- `update_prompt()`
- `validate_prompt_structure()`

**Test Functions:**
- `_test_backup_functionality()`
- `_test_error_handling()`
- `_test_import_functionality()`
- `_test_prompt_operations()`
- `_test_prompt_validation()`
- `_test_prompts_loading()`
- `ai_prompt_utils_module_tests()`
- `quick_test()`
- `run_comprehensive_tests()`

### api_search_utils.py

**Public Functions:**
- `api_search_utils_module_tests()`
- `clean_param()`
- `get_api_family_details()`
- `get_api_relationship_path()`
- `run_comprehensive_tests()`
- `search_api_for_criteria()`

**Test Functions:**
- `_test_core_functionality()`
- `_test_edge_cases()`
- `_test_error_handling()`
- `_test_integration()`
- `_test_module_initialization()`
- `_test_performance()`
- `api_search_utils_module_tests()`
- `run_comprehensive_tests()`

### api_utils.py

**Public Functions:**
- `api_utils_module_tests()`
- `call_discovery_relationship_api()`
- `call_edit_relationships_api()`
- `call_enhanced_api()`
- `call_facts_user_api()`
- `call_getladder_api()`
- `call_header_trees_api_for_tree_id()`
- `call_profile_details_api()`
- `call_relationship_ladder_api()`
- `call_send_message_api()`
- `call_suggest_api()`
- `call_tree_owner_api()`
- `call_treesui_list_api()`
- `can_make_request()`
- `dict()`
- `from_dict()`
- `get_relationship_path_data()`
- `parse_ancestry_person_details()`
- `print_group()`
- `record_request()`
- `wait_time_until_available()`

**Test Functions:**
- `_run_core_functionality_tests()`
- `_run_edge_case_tests()`
- `_run_error_handling_tests()`
- `_run_initialization_tests()`
- `_run_integration_tests()`
- `_run_performance_tests()`
- `api_utils_module_tests()`
- `test_api_response_parsing()`
- `test_config_integration()`
- `test_configuration_errors()`
- `test_data_processing_efficiency()`
- `test_data_validation_errors()`
- `test_datetime_handling()`
- `test_empty_data_handling()`
- `test_invalid_json_handling()`
- `test_json_parsing_performance()`
- `test_logger_initialization()`
- `test_logging_integration()`
- `test_module_imports()`
- `test_network_error_simulation()`
- `test_optional_dependencies()`
- `test_person_detail_parsing()`
- `test_special_characters()`
- `test_url_construction()`
- `test_url_encoding_performance()`

### cache.py

**Public Functions:**
- `add_dependency()`
- `cache_file_based_on_mtime()`
- `cache_module_tests()`
- `cache_result()`
- `check_cache_size_before_add()`
- `clear()`
- `clear_cache()`
- `close_cache()`
- `decorator()`
- `enforce_cache_size_limit()`
- `get_cache_coordination_stats()`
- `get_cache_dependency_tracker()`
- `get_cache_entry_count()`
- `get_cache_stats()`
- `get_dependency_chain()`
- `get_health_status()`
- `get_intelligent_cache_warmer()`
- `get_module_name()`
- `get_stats()`
- `get_unified_cache_key()`
- `get_warming_candidates()`
- `invalidate_cache_pattern()`
- `invalidate_related_caches()`
- `invalidate_with_dependencies()`
- `record_cache_access()`
- `warm()`
- `warm_cache_with_data()`
- `warm_predictive_cache()`
- `wrapper()`

**Test Functions:**
- `_test_basic_cache_operations()`
- `_test_cache_clearing()`
- `_test_cache_decorator()`
- `_test_cache_expiration()`
- `_test_cache_initialization()`
- `_test_cache_interfaces()`
- `_test_cache_performance()`
- `_test_cache_size_management()`
- `_test_complex_data_types()`
- `_test_error_handling()`
- `_test_health_monitoring()`
- `cache_module_tests()`
- `test_function()`

### cache_manager.py

**Public Functions:**
- `cache_api_response()`
- `cache_component()`
- `cache_manager_module_tests()`
- `cache_session_component()`
- `cached_api_call()`
- `cached_session_component()`
- `create_api_cache_key()`
- `decorator()`
- `get_api_cache_stats()`
- `get_cached_api_response()`
- `get_cached_component()`
- `get_cached_session_component()`
- `get_comprehensive_stats()`
- `get_memory_usage_mb()`
- `get_module_name()`
- `get_session_cache_stats()`
- `get_stats()`
- `get_system_cache_stats()`
- `get_unified_cache_manager()`
- `optimize_memory()`
- `warm_all_caches()`
- `warm_system_caches()`
- `wrapper()`

**Test Functions:**
- `_test_access_control()`
- `_test_api_integration()`
- `_test_audit_logging()`
- `_test_cache_invalidation()`
- `_test_cache_manager_initialization()`
- `_test_cache_operations()`
- `_test_cache_performance()`
- `_test_cache_statistics()`
- `_test_concurrent_access()`
- `_test_configuration_loading()`
- `_test_data_corruption_handling()`
- `_test_data_encryption()`
- `_test_database_integration()`
- `_test_environment_adaptation()`
- `_test_error_handling()`
- `_test_eviction_policies()`
- `_test_feature_toggles()`
- `_test_memory_management()`
- `_test_performance_monitoring()`
- `_test_recovery_mechanisms()`
- `_test_session_management()`
- `cache_manager_module_tests()`

### chromedriver.py

**Public Functions:**
- `chromedriver_module_tests()`
- `cleanup_webdrv()`
- `close_tabs()`
- `init_webdvr()`
- `main()`
- `reset_preferences_file()`
- `run_all_tests()`
- `set_win_size()`

**Test Functions:**
- `chromedriver_module_tests()`
- `run_all_tests()`
- `test_chrome_options_creation()`
- `test_chrome_process_cleanup()`
- `test_chromedriver_initialization()`
- `test_cleanup()`
- `test_driver_initialization()`
- `test_preferences_file()`
- `test_preferences_file_reset()`
- `test_webdriver_initialization()`

### code_quality_checker.py

**Public Functions:**
- `check_directory()`
- `check_file()`
- `code_quality_checker_module_tests()`
- `generate_report()`
- `quality_score()`
- `type_hint_coverage()`

**Test Functions:**
- `code_quality_checker_module_tests()`
- `test_checker_initialization()`
- `test_quality_metrics()`

### common_params.py

**Public Functions:**
- `run_comprehensive_tests()`

**Test Functions:**
- `_test_dataclass_defaults()`
- `_test_graph_context_initialization()`
- `_test_match_identifiers_initialization()`
- `_test_progress_indicator_config_initialization()`
- `_test_request_config_initialization()`
- `_test_retry_context_initialization()`
- `_test_search_criteria_initialization()`
- `run_comprehensive_tests()`

### config.py

**Public Functions:**
- `config_module_tests()`

**Test Functions:**
- `_test_config_manager_functionality()`
- `_test_config_manager_import()`
- `_test_config_schema_validity()`
- `_test_global_instances()`
- `_test_module_exports()`
- `config_module_tests()`

### config/config_manager.py

**Public Functions:**
- `config_manager_module_tests()`
- `export_config()`
- `get_api_config()`
- `get_cache_config()`
- `get_config()`
- `get_database_config()`
- `get_environment_config()`
- `get_logging_config()`
- `get_security_config()`
- `get_selenium_config()`
- `load_config()`
- `reload_config()`
- `run_setup_wizard()`
- `validate_config()`
- `validate_system_requirements()`

**Test Functions:**
- `_load_testing_config_from_env()`
- `_test_config_access()`
- `_test_config_access_performance()`
- `_test_config_error_handling()`
- `_test_config_file_integration()`
- `_test_config_loading()`
- `_test_config_manager_initialization()`
- `_test_config_validation()`
- `_test_environment_integration()`
- `_test_invalid_config_data()`
- `_test_missing_config_handling()`
- `_test_requests_per_second_loading()`
- `config_manager_module_tests()`

### config/config_schema.py

**Public Functions:**
- `add_environment_rule()`
- `add_rule()`
- `config_schema_module_tests()`
- `from_dict()`
- `get_connection_string()`
- `to_dict()`
- `validate()`
- `validate_path_exists()`

**Test Functions:**
- `_test_api_config()`
- `_test_cache_config()`
- `_test_config_schema_creation()`
- `_test_config_schema_from_dict()`
- `_test_config_schema_to_dict()`
- `_test_config_schema_validation()`
- `_test_database_config()`
- `_test_edge_cases()`
- `_test_function_structure()`
- `_test_import_dependencies()`
- `_test_integration()`
- `_test_logging_config()`
- `_test_max_pages_configuration()`
- `_test_performance()`
- `_test_rate_limiting_configuration()`
- `_test_security_config()`
- `_test_selenium_config()`
- `config_schema_module_tests()`

### config/credential_manager.py

**Public Functions:**
- `clear_cache()`
- `credential_manager_module_tests()`
- `export_for_backup()`
- `get_ancestry_credentials()`
- `get_api_key()`
- `get_credential()`
- `get_credential_status()`
- `has_credential()`
- `load_credentials()`
- `migrate_from_environment()`
- `remove_credential()`
- `run_all_tests()`
- `run_test()`
- `start_suite()`
- `store_credentials()`
- `validate_credentials()`

**Test Functions:**
- `_get_credential_manager_tests()`
- `_get_test_framework()`
- `_test_ancestry_credentials()`
- `_test_api_key_retrieval()`
- `_test_cache_management()`
- `_test_credential_access()`
- `_test_credential_status()`
- `_test_credential_validation()`
- `_test_environment_loading()`
- `_test_error_handling()`
- `_test_export_functionality()`
- `_test_function_structure()`
- `_test_import_dependencies()`
- `_test_initialization()`
- `_test_integration()`
- `_test_performance()`
- `_test_security_manager_integration()`
- `credential_manager_module_tests()`
- `run_all_tests()`
- `run_test()`

### connection_resilience.py

**Public Functions:**
- `decorator()`
- `handle_connection_loss()`
- `monitored_func()`
- `run_comprehensive_tests()`
- `start_resilience_mode()`
- `stop_resilience_mode()`
- `with_connection_resilience()`
- `with_periodic_health_check()`
- `wrapper()`

**Test Functions:**
- `_test_decorator_parameters()`
- `_test_decorators_are_callable()`
- `_test_resilience_manager_initialization()`
- `_test_resilience_manager_max_attempts()`
- `_test_resilience_manager_recovery_backoff()`
- `_test_resilience_manager_state_transitions()`
- `run_comprehensive_tests()`

### conversation_analytics.py

**Public Functions:**
- `conversation_analytics_module_tests()`
- `get_overall_analytics()`
- `print_analytics_dashboard()`
- `record_engagement_event()`
- `update_conversation_metrics()`

**Test Functions:**
- `_test_database_models_available()`
- `_test_engagement_score_delta_calculation()`
- `_test_get_overall_analytics_empty()`
- `_test_print_analytics_dashboard()`
- `_test_record_engagement_event()`
- `_test_research_outcomes_tracking()`
- `_test_template_tracking()`
- `_test_tree_impact_tracking()`
- `_test_update_conversation_metrics_existing()`
- `_test_update_conversation_metrics_new()`
- `conversation_analytics_module_tests()`

### core/__init__.py

**Public Functions:**
- `core_package_module_tests()`

**Test Functions:**
- `core_package_module_tests()`
- `test_component_imports()`
- `test_dependency_injection_imports()`
- `test_error_handling_imports()`
- `test_package_structure()`

### core/__main__.py

**Public Functions:**
- `run_comprehensive_tests()`

**Test Functions:**
- `_test_core_browser_manager_availability()`
- `_test_core_database_manager_availability()`
- `_test_core_package_imports()`
- `_test_core_package_structure()`
- `_test_core_session_manager_availability()`
- `run_comprehensive_tests()`

### core/api_manager.py

**Public Functions:**
- `api_manager_module_tests()`
- `clear_identifiers()`
- `get_csrf_token()`
- `get_profile_id()`
- `get_uuid()`
- `has_essential_identifiers()`
- `make_api_request()`
- `requests_session()`
- `reset_logged_flags()`
- `retrieve_all_identifiers()`
- `sync_cookies_from_browser()`
- `verify_api_login_status()`

**Test Functions:**
- `_test_api_manager_initialization()`
- `_test_api_request_methods()`
- `_test_config_integration()`
- `_test_connection_error_handling()`
- `_test_identifier_management()`
- `_test_invalid_response_handling()`
- `_test_session_reuse_efficiency()`
- `api_manager_module_tests()`

### core/browser_manager.py

**Public Functions:**
- `close_browser()`
- `create_new_tab()`
- `ensure_driver_live()`
- `get_cookies()`
- `is_session_valid()`
- `run_comprehensive_tests()`
- `start_browser()`

**Test Functions:**
- `_test_browser_manager_initialization()`
- `_test_close_browser_no_driver()`
- `_test_configuration_access()`
- `_test_cookie_check_invalid_session()`
- `_test_cookie_check_prevents_long_waits()`
- `_test_cookie_timeout_custom()`
- `_test_cookie_timeout_default()`
- `_test_ensure_driver_not_needed()`
- `_test_exception_handling()`
- `_test_initialization_performance()`
- `_test_method_availability()`
- `_test_session_validation_no_driver()`
- `_test_state_management()`
- `run_comprehensive_tests()`

### core/cancellation.py

**Public Functions:**
- `cancel_scope()`
- `check_cancel()`
- `clear_cancel()`
- `is_cancel_requested()`
- `request_cancel()`
- `run_comprehensive_tests()`
- `set_cancel()`

**Test Functions:**
- `_test_cancel_state_thread_safety()`
- `_test_cancellation_state_initialization()`
- `_test_clear_cancel()`
- `_test_multiple_cancel_requests()`
- `_test_request_cancel_with_scope()`
- `_test_request_cancel_without_scope()`
- `run_comprehensive_tests()`

### core/database_manager.py

**Public Functions:**
- `add_to_batch()`
- `batch_operation_context()`
- `close_connections()`
- `commit_batch()`
- `database_manager_module_tests()`
- `enable_sqlite_settings()`
- `ensure_ready()`
- `execute_query_with_timing()`
- `get_db_path()`
- `get_performance_stats()`
- `get_session()`
- `get_session_context()`
- `is_ready()`
- `return_session()`

**Test Functions:**
- `database_manager_module_tests()`
- `test_connection_pooling()`
- `test_database_manager_initialization()`
- `test_database_readiness()`
- `test_engine_session_creation()`
- `test_error_handling_recovery()`
- `test_session_context_management()`
- `test_session_lifecycle()`
- `test_transaction_isolation()`

### core/dependency_injection.py

**Public Functions:**
- `clear()`
- `clear_all_containers()`
- `clear_container()`
- `configure_dependencies()`
- `decorator()`
- `dependency_injection_module_tests()`
- `factory()`
- `get_container()`
- `get_registration_info()`
- `get_service()`
- `inject()`
- `is_registered()`
- `register_factory()`
- `register_instance()`
- `register_singleton()`
- `register_transient()`
- `resolve()`
- `setUp()`
- `wrapper()`

**Test Functions:**
- `dependency_injection_module_tests()`
- `test_clear()`
- `test_configure_dependencies()`
- `test_container_state_management()`
- `test_create_instance()`
- `test_di_resolution_error()`
- `test_di_scope()`
- `test_function()`
- `test_get_registration_info()`
- `test_get_service_convenience()`
- `test_global_container()`
- `test_imports_and_availability()`
- `test_inject_decorator()`
- `test_injectable_class()`
- `test_integration()`
- `test_is_registered()`
- `test_register_factory()`
- `test_register_instance()`
- `test_register_singleton()`
- `test_register_transient()`
- `test_resolve_factory()`
- `test_resolve_instance()`
- `test_resolve_singleton()`
- `test_resolve_transient()`
- `test_service_registry()`
- `test_type_annotations()`

### core/enhanced_error_recovery.py

**Public Functions:**
- `add_error()`
- `create_user_guidance()`
- `decorator()`
- `get_backoff_delay()`
- `get_recovery_stats()`
- `handle_partial_success()`
- `is_circuit_open()`
- `record_failure()`
- `record_success()`
- `run_comprehensive_tests()`
- `should_retry()`
- `update_stats()`
- `with_api_recovery()`
- `with_database_recovery()`
- `with_enhanced_recovery()`
- `with_file_recovery()`
- `wrapper()`

**Test Functions:**
- `_test_api_recovery_decorator()`
- `_test_database_recovery_decorator()`
- `_test_error_context_add_error()`
- `_test_error_context_initialization()`
- `_test_error_severity_enum()`
- `_test_exponential_backoff_calculation()`
- `_test_file_recovery_decorator()`
- `_test_recovery_strategy_enum()`
- `_test_user_guidance_creation()`
- `run_comprehensive_tests()`

### core/error_handling.py

**Public Functions:**
- `ancestry_api_recovery()`
- `ancestry_database_recovery()`
- `ancestry_session_recovery()`
- `call()`
- `can_handle()`
- `circuit_breaker()`
- `decorator()`
- `error_context()`
- `error_handler()`
- `error_handling_module_tests()`
- `failing_func()`
- `get_circuit_breaker()`
- `get_error_handler()`
- `get_health_status()`
- `get_stats()`
- `graceful_degradation()`
- `handle()`
- `handle_error()`
- `network_error_func()`
- `register_error_handler()`
- `register_handler()`
- `register_recovery_strategy()`
- `reset()`
- `reset_all_circuit_breakers()`
- `retry_on_failure()`
- `safe_execute()`
- `safe_func()`
- `successful_func()`
- `target()`
- `timeout_handler()`
- `timeout_protection()`
- `to_dict()`
- `with_circuit_breaker()`
- `with_recovery()`
- `wrapper()`

**Test Functions:**
- `error_handling_module_tests()`
- `test_circuit_breaker()`
- `test_error_context()`
- `test_error_handling()`
- `test_error_recovery()`
- `test_error_types()`
- `test_function_availability()`

### core/logging_utils.py

**Public Functions:**
- `debug_if_enabled()`
- `debug_lazy()`
- `decorator()`
- `ensure_no_duplicate_handlers()`
- `get_app_logger()`
- `get_logger()`
- `logger()`
- `logging_utils_module_tests()`
- `suppress_external_loggers()`
- `wrapper()`

**Test Functions:**
- `_test_app_logger_convenience()`
- `_test_centralized_logging_setup()`
- `_test_debug_decorator()`
- `_test_duplicate_handler_prevention()`
- `_test_external_logger_suppression()`
- `_test_get_logger_functionality()`
- `_test_logging_import_fallback()`
- `_test_optimized_logger_functionality()`
- `logging_utils_module_tests()`
- `test_debug_function()`
- `test_msg_func()`

### core/progress_indicators.py

**Public Functions:**
- `create_progress_indicator()`
- `decorator()`
- `elapsed_seconds()`
- `eta_seconds()`
- `finish()`
- `items_per_second()`
- `log_milestone()`
- `memory_usage_mb()`
- `run_comprehensive_tests()`
- `set_total()`
- `start()`
- `update()`
- `with_progress()`
- `wrapper()`

**Test Functions:**
- `_test_progress_decorator_creation()`
- `_test_progress_indicator_creation()`
- `_test_progress_stats_elapsed_time()`
- `_test_progress_stats_eta_calculation()`
- `_test_progress_stats_eta_no_total()`
- `_test_progress_stats_initialization()`
- `_test_progress_stats_items_per_second()`
- `run_comprehensive_tests()`
- `test_func()`

### core/registry_utils.py

**Public Functions:**
- `auto_register_module()`
- `create_registration_report()`
- `get()`
- `get_stats()`
- `is_available()`
- `performance_register()`
- `register()`
- `register_module()`
- `run_comprehensive_tests()`

**Test Functions:**
- `run_comprehensive_tests()`

### core/session_cache.py

**Public Functions:**
- `cache_component()`
- `cache_session_state()`
- `cached_api_manager()`
- `cached_browser_manager()`
- `cached_database_manager()`
- `cached_session_component()`
- `cached_session_validator()`
- `clear()`
- `clear_session_cache()`
- `create_expensive_component()`
- `create_test_component()`
- `decorator()`
- `get_cached_component()`
- `get_cached_session_state()`
- `get_health_status()`
- `get_module_name()`
- `get_session_cache_stats()`
- `get_stats()`
- `session_cache_module_tests()`
- `warm()`
- `warm_session_cache()`
- `wrapper()`

**Test Functions:**
- `_test_cache_expiration()`
- `_test_cache_stats()`
- `_test_cached_session_component_decorator()`
- `_test_clear_session_cache()`
- `_test_component_caching_and_retrieval()`
- `_test_config_hash_generation()`
- `_test_session_component_cache_initialization()`
- `_test_warm_session_cache()`
- `create_test_component()`
- `session_cache_module_tests()`
- `test_session_cache_performance()`

### core/session_manager.py

**Public Functions:**
- `attempt_browser_recovery()`
- `attempt_cascade_recovery()`
- `browser_needed()`
- `cached_api_manager()`
- `cached_browser_manager()`
- `cached_database_manager()`
- `cached_session_validator()`
- `cancel_all_operations()`
- `check_automatic_intervention()`
- `check_browser_health()`
- `check_cascade_before_operation()`
- `check_js_errors()`
- `check_session_health()`
- `cleanup()`
- `clear_session_cache()`
- `clear_session_caches()`
- `close_browser()`
- `close_sess()`
- `cls_db_conn()`
- `csrf_token()`
- `driver()`
- `driver_live()`
- `emergency_shutdown()`
- `ensure_db_ready()`
- `ensure_session_ready()`
- `force_cookie_resync()`
- `get_cookies()`
- `get_csrf()`
- `get_db_conn()`
- `get_db_conn_context()`
- `get_my_profile_id()`
- `get_my_tree_id()`
- `get_my_uuid()`
- `get_session_performance_stats()`
- `get_session_summary()`
- `get_tree_owner()`
- `increment_page_count()`
- `invalidate_csrf_cache()`
- `is_emergency_shutdown()`
- `is_ready()`
- `is_sess_valid()`
- `is_session_death_cascade()`
- `make_tab()`
- `monitor_js_errors()`
- `my_profile_id()`
- `my_tree_id()`
- `my_uuid()`
- `perform_proactive_browser_refresh()`
- `perform_proactive_refresh()`
- `process_pages()`
- `requests_session()`
- `reset_session_health_monitoring()`
- `restart_sess()`
- `return_session()`
- `scraper()`
- `session_age_seconds()`
- `session_manager_module_tests()`
- `should_halt_operations()`
- `should_proactive_browser_refresh()`
- `should_proactive_refresh()`
- `start_browser()`
- `start_sess()`
- `tree_owner_name()`
- `validate_system_health()`
- `verify_sess()`

**Test Functions:**
- `_test_724_page_workload_simulation()`
- `_test_authentication_state()`
- `_test_browser_navigation()`
- `_test_browser_operations()`
- `_test_component_delegation()`
- `_test_component_manager_availability()`
- `_test_cookie_access()`
- `_test_database_operations()`
- `_test_error_handling()`
- `_test_initialization_performance()`
- `_test_javascript_execution()`
- `_test_post_replacement_cookies()`
- `_test_property_access()`
- `_test_regression_prevention_csrf_optimization()`
- `_test_regression_prevention_initialization_stability()`
- `_test_regression_prevention_property_access()`
- `_test_session_manager_initialization()`
- `session_manager_module_tests()`

### core/session_validator.py

**Public Functions:**
- `perform_readiness_checks()`
- `session_validator_module_tests()`
- `validate_session_cookies()`
- `verify_login_status()`

**Test Functions:**
- `_test_full_validation_workflow()`
- `_test_general_exception_handling()`
- `_test_initialization_performance()`
- `_test_invalid_browser_session()`
- `_test_login_verification()`
- `_test_login_verification_failure()`
- `_test_readiness_checks_success()`
- `_test_session_validator_initialization()`
- `_test_should_skip_cookie_check_action6()`
- `_test_should_skip_cookie_check_action7()`
- `_test_should_skip_cookie_check_action8()`
- `_test_should_skip_cookie_check_action9()`
- `_test_should_skip_cookie_check_case_insensitive()`
- `_test_should_skip_cookie_check_no_action()`
- `_test_should_skip_cookie_check_unknown_action()`
- `_test_webdriver_exception_handling()`
- `session_validator_module_tests()`

### core/system_cache.py

**Public Functions:**
- `cache_api_response()`
- `cache_query_result()`
- `cached_api_call()`
- `cached_database_query()`
- `clear_system_caches()`
- `decorator()`
- `get_api_cache_stats()`
- `get_cached_api_response()`
- `get_cached_query_result()`
- `get_memory_usage_mb()`
- `get_system_cache_stats()`
- `memory_intensive_function()`
- `memory_optimized()`
- `mock_api_call()`
- `mock_db_query()`
- `optimize_memory()`
- `system_cache_module_tests()`
- `warm_system_caches()`
- `wrapper()`

**Test Functions:**
- `_test_api_cache_key_generation()`
- `_test_api_response_cache_initialization()`
- `_test_api_response_caching_and_retrieval()`
- `_test_cached_api_call_decorator()`
- `_test_cached_database_query_decorator()`
- `_test_clear_system_caches()`
- `_test_database_query_cache_initialization()`
- `_test_memory_optimized_decorator()`
- `_test_memory_optimizer_initialization()`
- `_test_system_cache_stats()`
- `_test_warm_system_caches()`
- `system_cache_module_tests()`
- `test_system_cache_performance()`

### core_imports.py

**Public Functions:**
- `auto_register_module()`
- `call_function()`
- `cleanup_registry()`
- `core_imports_module_tests()`
- `decorator()`
- `ensure_imports()`
- `get_available_functions()`
- `get_function()`
- `get_import_stats()`
- `get_logger()`
- `get_project_root()`
- `get_stats()`
- `import_context()`
- `is_function_available()`
- `register_function()`
- `register_many()`
- `safe_execute()`
- `standardize_module_imports()`
- `wrapper()`

**Test Functions:**
- `_test_auto_registration()`
- `_test_context_manager()`
- `_test_function_registration()`
- `_test_import_standardization()`
- `_test_performance_caching()`
- `core_imports_module_tests()`

### credentials.py

**Public Functions:**
- `check_and_install_dependencies()`
- `check_status()`
- `create_test_credential_file()`
- `credentials_module_tests()`
- `delete_all_credentials()`
- `display_main_menu()`
- `edit_credential_types()`
- `export_credentials()`
- `fake_input()`
- `import_from_env()`
- `main()`
- `remove_credential()`
- `run()`
- `setup_credentials()`
- `setup_test_credentials()`
- `view_credentials()`

**Test Functions:**
- `_should_run_tests()`
- `_test_check_status_with_missing_credentials()`
- `_test_edit_credential_types_error_handling()`
- `_test_encryption()`
- `_test_load_credential_types_with_invalid_json()`
- `_test_load_credential_types_with_invalid_structure()`
- `_test_load_credential_types_with_missing_file()`
- `_test_load_credential_types_with_valid_file()`
- `_test_manager_initialization()`
- `_test_manager_initialization_with_security_unavailable()`
- `_test_menu_methods()`
- `_test_security_availability()`
- `_test_setup_credentials_permission_error()`
- `create_test_credential_file()`
- `credentials_module_tests()`
- `setup_test_credentials()`
- `test_load_credential_types_with_invalid_json()`
- `test_load_credential_types_with_invalid_structure()`
- `test_load_credential_types_with_valid_file()`

### database.py

**Public Functions:**
- `backup_database()`
- `cleanup_soft_deleted_records()`
- `commit_bulk_data()`
- `create_or_update_dna_match()`
- `create_or_update_family_tree()`
- `create_or_update_person()`
- `create_person()`
- `database_module_tests()`
- `db_transn()`
- `delete_database()`
- `delete_person()`
- `enable_sqlite_settings_standalone()`
- `exclude_deleted_persons()`
- `find_existing_person()`
- `get_person_and_dna_match()`
- `get_person_by_profile_id()`
- `get_person_by_profile_id_and_username()`
- `get_person_by_uuid()`
- `hard_delete_person()`
- `run_comprehensive_tests()`
- `soft_delete_person()`
- `uuid()`

**Test Functions:**
- `_create_and_verify_test_person()`
- `_create_test_persons_for_cleanup()`
- `_soft_delete_test_persons_with_timestamps()`
- `_test_configuration_error_handling()`
- `_test_database_base_setup()`
- `_test_database_model_definitions()`
- `_test_database_utilities()`
- `_test_enum_definitions()`
- `_test_enum_edge_cases()`
- `_test_import_error_handling()`
- `_test_model_attributes()`
- `_test_model_creation_performance()`
- `_test_model_instantiation_edge_cases()`
- `_test_model_relationships()`
- `_test_schema_integration()`
- `_test_transaction_context_manager()`
- `database_module_tests()`
- `run_comprehensive_tests()`
- `test_cleanup_soft_deleted_records()`
- `test_soft_delete_functionality()`

### dna_ethnicity_utils.py

**Public Functions:**
- `dna_ethnicity_utils_module_tests()`
- `extract_match_ethnicity_percentages()`
- `fetch_ethnicity_comparison()`
- `fetch_ethnicity_region_names()`
- `fetch_tree_owner_ethnicity_regions()`
- `load_ethnicity_metadata()`
- `run_comprehensive_tests()`
- `sanitize_column_name()`

**Test Functions:**
- `_setup_test_session()`
- `_test_ethnicity_comparison()`
- `_test_extract_match_ethnicity_percentages()`
- `_test_region_names_fetch()`
- `_test_sanitize_column_name()`
- `_test_tree_owner_ethnicity_fetch()`
- `_validate_test_prerequisites()`
- `dna_ethnicity_utils_module_tests()`
- `run_comprehensive_tests()`

### dna_gedcom_crossref.py

**Public Functions:**
- `analyze_dna_gedcom_connections()`
- `dna_gedcom_crossref_module_tests()`

**Test Functions:**
- `dna_gedcom_crossref_module_tests()`
- `test_conflict_identification_out_of_range_cm()`
- `test_dna_gedcom_crossref()`
- `test_name_match_and_confidence_boost()`
- `test_relationship_distance_parser()`
- `test_verification_opportunity_threshold()`

### dna_utils.py

**Public Functions:**
- `fetch_in_tree_status()`
- `fetch_match_list_page()`
- `get_csrf_token_for_dna_matches()`
- `nav_to_dna_matches_page()`
- `run_comprehensive_tests()`

**Test Functions:**
- `_test_cache_key_construction()`
- `_test_cookie_names_for_csrf()`
- `_test_csrf_token_extraction()`
- `_test_csrf_token_url_decoding()`
- `_test_dna_matches_url_construction()`
- `_test_match_list_api_url_construction()`
- `_test_match_list_headers_construction()`
- `run_comprehensive_tests()`

### error_handling.py

**Public Functions:**
- `ancestry_api_recovery()`
- `ancestry_database_recovery()`
- `ancestry_session_recovery()`
- `calculate_delay()`
- `call()`
- `capture_environment()`
- `check_failure_patterns()`
- `circuit_breaker()`
- `decorator()`
- `error_context()`
- `error_handling_module_tests()`
- `execute_with_recovery()`
- `failing_operation()`
- `fast_operation()`
- `get_all_stats()`
- `get_circuit_breaker()`
- `get_stats()`
- `graceful_degradation()`
- `log_failure_warnings()`
- `nested_error_function()`
- `record_attempt()`
- `register_recovery_strategy()`
- `reset()`
- `reset_all_circuit_breakers()`
- `retry_on_failure()`
- `run_comprehensive_tests()`
- `should_retry()`
- `target()`
- `thread_operation()`
- `timeout_protection()`
- `to_dict()`
- `with_circuit_breaker()`
- `with_recovery()`
- `wrapper()`

**Test Functions:**
- `_test_circuit_breaker_config()`
- `_test_circuit_breaker_edge_cases()`
- `_test_circuit_breaker_performance()`
- `_test_circuit_breaker_states()`
- `_test_config_integration()`
- `_test_error_context_edge_cases()`
- `_test_error_handling_basics()`
- `_test_error_handling_performance()`
- `_test_error_recovery_edge_cases()`
- `_test_error_types()`
- `_test_failure_pattern_monitoring()`
- `_test_failure_warning_logging()`
- `_test_function_availability()`
- `_test_logging_integration()`
- `_test_memory_error_handling()`
- `_test_module_imports()`
- `_test_recursive_error_handling()`
- `_test_retry_strategy_performance()`
- `_test_threading_integration()`
- `_test_timeout_error_handling()`
- `error_handling_module_tests()`
- `run_comprehensive_tests()`

### gedcom_cache.py

**Public Functions:**
- `cache_gedcom_processed_data()`
- `clear()`
- `clear_gedcom_cache()`
- `clear_memory_cache()`
- `data()`
- `demonstrate_gedcom_cache_usage()`
- `gedcom_cache_module_tests()`
- `get_gedcom_cache_health()`
- `get_gedcom_cache_info()`
- `get_gedcom_cache_stats()`
- `get_health_status()`
- `get_module_name()`
- `get_stats()`
- `load_gedcom_with_aggressive_caching()`
- `preload_gedcom_cache()`
- `warm()`
- `warm_gedcom_cache()`

**Test Functions:**
- `gedcom_cache_module_tests()`
- `test_cache_health_status()`
- `test_cache_invalidation_file_modification()`
- `test_cache_key_generation()`
- `test_cache_performance_metrics()`
- `test_cache_statistics_collection()`
- `test_cache_validation_integrity()`
- `test_cached_data_retrieval()`
- `test_gedcom_cache_initialization()`
- `test_gedcom_parsing_caching()`
- `test_memory_cache_expiration()`
- `test_memory_cache_operations()`
- `test_memory_management_cleanup()`
- `test_multifile_cache_management()`

### gedcom_intelligence.py

**Public Functions:**
- `analyze_gedcom_data()`
- `gedcom_intelligence_module_tests()`

**Test Functions:**
- `gedcom_intelligence_module_tests()`
- `test_ai_insights_structure()`
- `test_gap_detection_with_mocked_birth_year()`
- `test_gedcom_intelligence()`
- `test_recommendation_balance_logic()`

### gedcom_search_utils.py

**Public Functions:**
- `gedcom_search_module_tests()`
- `get_cached_gedcom_data()`
- `get_gedcom_data()`
- `get_gedcom_family_details()`
- `get_gedcom_relationship_path()`
- `load_gedcom_data()`
- `matches_criterion()`
- `matches_year_criterion()`
- `search_gedcom_for_criteria()`
- `set_cached_gedcom_data()`

**Test Functions:**
- `gedcom_search_module_tests()`
- `test_criterion_matching()`
- `test_edge_cases()`
- `test_error_recovery()`
- `test_family_details()`
- `test_function_availability()`
- `test_gedcom_operations()`
- `test_invalid_data_handling()`
- `test_memory_efficiency()`
- `test_performance()`
- `test_relationship_paths()`
- `test_search_criteria()`
- `test_year_criterion()`

### gedcom_utils.py

**Public Functions:**
- `build_caches()`
- `calculate_match_score()`
- `explain_relationship_path()`
- `extract_and_fix_id()`
- `fast_bidirectional_bfs()`
- `find_individual_by_id()`
- `format_full_life_details()`
- `format_life_dates()`
- `format_relative_info()`
- `format_source_citations()`
- `from_cache()`
- `gedcom_module_tests()`
- `get_person_sources()`
- `get_processed_indi_data()`
- `get_related_individuals()`
- `get_relationship_path()`

**Test Functions:**
- `gedcom_module_tests()`
- `test_bfs_pathfinding()`
- `test_date_parsing()`
- `test_error_recovery()`
- `test_event_extraction()`
- `test_external_integration()`
- `test_function_availability()`
- `test_id_normalization()`
- `test_individual_detection()`
- `test_invalid_data_handling()`
- `test_large_dataset_performance()`
- `test_life_dates_formatting()`
- `test_memory_optimization()`
- `test_name_extraction()`
- `test_relationship_explanation()`
- `test_sibling_detection()`
- `test_source_citation_demonstration()`
- `test_source_citation_extraction()`

### genealogical_normalization.py

**Public Functions:**
- `genealogical_normalization_module_tests()`
- `normalize_ai_response()`
- `normalize_extracted_data()`

**Test Functions:**
- `_run_basic_tests()`
- `_test_ai_response_normalization()`
- `_test_container_structure()`
- `_test_edge_cases()`
- `_test_extracted_data_normalization()`
- `_test_function_availability()`
- `_test_legacy_field_promotion()`
- `_test_list_deduplication()`
- `genealogical_normalization_module_tests()`

### health_monitor.py

**Public Functions:**
- `auto_checkpoint()`
- `begin_safety_test()`
- `calculate_health_score()`
- `create_recovery_checkpoint()`
- `create_session_checkpoint()`
- `enable_session_state_persistence()`
- `end_safety_test()`
- `get_error_rate_statistics()`
- `get_health_dashboard()`
- `get_health_monitor()`
- `get_health_status()`
- `get_intervention_status()`
- `get_performance_recommendations()`
- `get_performance_stats()`
- `get_recommended_actions()`
- `get_session_recovery_status()`
- `health_monitor_tests()`
- `initialize_health_monitoring()`
- `integrate_with_action6()`
- `integrate_with_session_manager()`
- `is_enhanced_monitoring_active()`
- `list_available_checkpoints()`
- `log_health_summary()`
- `optimize_for_long_session()`
- `persist_session_state_to_disk()`
- `predict_session_death_risk()`
- `record_api_response_time()`
- `record_error()`
- `record_page_processing_time()`
- `recover_session_state_from_disk()`
- `reset_intervention_flags()`
- `restore_from_checkpoint()`
- `should_emergency_halt()`
- `should_immediate_intervention()`
- `status()`
- `update_metric()`
- `update_session_metrics()`
- `update_system_metrics()`

**Test Functions:**
- `_test_alert_system()`
- `_test_auto_checkpoint_functionality()`
- `_test_checkpoint_management()`
- `_test_dashboard_generation()`
- `_test_global_instance()`
- `_test_health_monitor_initialization()`
- `_test_health_score_calculation()`
- `_test_integration_helpers()`
- `_test_long_session_resource_management()`
- `_test_memory_pressure_monitoring()`
- `_test_metric_updates()`
- `_test_performance_tracking()`
- `_test_resource_constraint_handling()`
- `_test_risk_prediction()`
- `_test_session_checkpoint_creation()`
- `_test_session_state_persistence()`
- `begin_safety_test()`
- `end_safety_test()`
- `health_monitor_tests()`

### logging_config.py

**Public Functions:**
- `filter()`
- `format()`
- `logging_config_module_tests()`
- `setup_logging()`

**Test Functions:**
- `logging_config_module_tests()`
- `test_default_configuration()`
- `test_directory_creation()`
- `test_external_library_logging()`
- `test_filter_integration()`
- `test_formatter_application()`
- `test_handler_configuration()`
- `test_handler_performance()`
- `test_invalid_file_path()`
- `test_invalid_log_level()`
- `test_log_level_setting()`
- `test_logger_creation()`
- `test_logging_speed()`
- `test_missing_directory()`
- `test_multiple_handlers()`
- `test_permission_errors()`
- `test_reinitialize_logging()`
- `test_setup_logging()`

### main.py

**Public Functions:**
- `all_but_first_actn()`
- `backup_db_actn()`
- `check_login_actn()`
- `clear_log_file()`
- `ensure_caching_initialized()`
- `exec_actn()`
- `gather_dna_matches()`
- `initialize_aggressive_caching()`
- `main()`
- `main_module_tests()`
- `menu()`
- `process_productive_messages_action()`
- `reset_db_actn()`
- `restore_db_actn()`
- `run_action11_wrapper()`
- `run_comprehensive_tests()`
- `run_core_workflow_action()`
- `send_messages_action()`
- `srch_inbox_actn()`
- `validate_action_config()`

**Test Functions:**
- `_handle_test_options()`
- `_run_all_tests()`
- `_run_main_tests()`
- `_test_action_function_availability()`
- `_test_action_integration()`
- `_test_cleanup_procedures()`
- `_test_clear_log_file_function()`
- `_test_configuration_availability()`
- `_test_configuration_integration()`
- `_test_database_integration()`
- `_test_database_operations()`
- `_test_edge_case_handling()`
- `_test_error_handling_structure()`
- `_test_exception_handling_coverage()`
- `_test_function_call_performance()`
- `_test_import_error_handling()`
- `_test_import_performance()`
- `_test_logging_integration()`
- `_test_main_function_structure()`
- `_test_memory_efficiency()`
- `_test_menu_system_components()`
- `_test_module_initialization()`
- `_test_reset_db_actn_integration()`
- `_test_session_manager_integration()`
- `_test_validate_action_config()`
- `main_module_tests()`
- `run_comprehensive_tests()`

### memory_utils.py

**Public Functions:**
- `acquire()`
- `fast_json_loads()`
- `memory_utils_module_tests()`
- `release()`

**Test Functions:**
- `_test_fast_json_loads()`
- `_test_object_pool()`
- `memory_utils_module_tests()`

### message_personalization.py

**Public Functions:**
- `create_personalized_message()`
- `get_optimization_recommendations()`
- `get_template_effectiveness_score()`
- `message_personalization_module_tests()`
- `track_message_response()`

**Test Functions:**
- `_apply_ab_testing()`
- `_test_dna_context_creation()`
- `_test_effectiveness_tracker_initialization()`
- `_test_location_context_formatting_empty()`
- `_test_personalization_config()`
- `_test_personalization_registry()`
- `_test_personalizer_initialization()`
- `_test_shared_ancestors_formatting_empty()`
- `message_personalization_module_tests()`
- `test_fallback_template_path()`
- `test_location_context_limit()`
- `test_message_personalization()`
- `test_shared_ancestors_formatting()`

### ms_graph_utils.py

**Public Functions:**
- `acquire_token_device_flow()`
- `create_todo_task()`
- `get_todo_list_id()`
- `ms_graph_utils_module_tests()`
- `save_cache_on_exit()`

**Test Functions:**
- `ms_graph_utils_module_tests()`
- `test_core_functionality()`
- `test_edge_cases()`
- `test_enhanced_task_creation()`
- `test_error_handling()`
- `test_initialization()`
- `test_integration()`
- `test_performance()`

### my_selectors.py

**Public Functions:**
- `my_selectors_module_tests()`

**Test Functions:**
- `my_selectors_module_tests()`
- `test_css_format()`
- `test_error_selectors()`
- `test_login_selectors()`
- `test_performance()`
- `test_placeholder_selectors()`
- `test_selector_accessibility()`
- `test_selector_definitions()`
- `test_selector_integrity()`
- `test_selector_organization()`
- `test_special_characters()`

### performance_cache.py

**Public Functions:**
- `cache_gedcom_results()`
- `cache_stats()`
- `clear_performance_cache()`
- `create_mock_filter_criteria()`
- `create_mock_gedcom_data()`
- `create_mock_scoring_criteria()`
- `decorator()`
- `fast_test_cache()`
- `get()`
- `get_cache_stats()`
- `invalidate_dependencies()`
- `performance_cache_module_tests()`
- `progressive_processing()`
- `set()`
- `warm_performance_cache()`
- `wrapper()`

**Test Functions:**
- `fast_test_cache()`
- `performance_cache_module_tests()`
- `test_cache_expiration()`
- `test_cache_health_status()`
- `test_cache_key_generation()`
- `test_cache_performance_metrics()`
- `test_cache_statistics_collection()`
- `test_memory_cache_operations()`
- `test_memory_management_cleanup()`
- `test_performance_cache_initialization()`

### performance_monitor.py

**Public Functions:**
- `error_function()`
- `export_report()`
- `generate_performance_dashboard()`
- `get_performance_dashboard()`
- `get_report()`
- `get_system_health_score()`
- `monitor_performance()`
- `performance_monitor_module_tests()`
- `profile()`
- `profile_function()`
- `record_metric()`
- `start_advanced_monitoring()`
- `start_monitoring()`
- `stop_advanced_monitoring()`
- `stop_monitoring()`
- `track_api_performance()`
- `validate_configuration()`
- `validate_system_configuration()`
- `wrapper()`

**Test Functions:**
- `_run_basic_tests()`
- `_test_advanced_monitoring()`
- `_test_alert_generation()`
- `_test_configuration_validation()`
- `_test_error_handling()`
- `_test_function_availability()`
- `_test_function_profiling()`
- `_test_global_performance_functions()`
- `_test_memory_monitoring()`
- `_test_metric_recording_and_retrieval()`
- `_test_performance_monitor_initialization()`
- `_test_performance_optimization()`
- `_test_performance_statistics()`
- `performance_monitor_module_tests()`
- `test_function()`

### performance_orchestrator.py

**Public Functions:**
- `add_to_batch()`
- `decorator()`
- `get_batch_for_execution()`
- `get_cached_import()`
- `get_global_optimizer()`
- `get_memory_info()`
- `get_optimization_stats()`
- `get_optimization_suggestions()`
- `get_performance_report()`
- `is_memory_pressure_high()`
- `monitor_memory_pressure()`
- `optimize_common_patterns()`
- `optimize_memory_usage()`
- `optimize_on_high_usage()`
- `optimize_performance()`
- `optimize_slow_imports()`
- `performance_orchestrator_module_tests()`
- `run_comprehensive_optimization()`
- `should_execute_batch()`
- `track_module_load()`
- `track_query()`
- `track_query_performance()`
- `wrapper()`

**Test Functions:**
- `_run_basic_tests()`
- `_test_api_batch_coordination()`
- `_test_comprehensive_performance_optimization()`
- `_test_error_handling_and_resilience()`
- `_test_function_availability()`
- `_test_global_optimization_functions()`
- `_test_memory_optimization_techniques()`
- `_test_memory_pressure_monitoring()`
- `_test_module_load_optimization()`
- `_test_optimization_decorators()`
- `_test_performance_metrics()`
- `_test_query_optimization_patterns()`
- `_test_query_optimizer_functionality()`
- `performance_orchestrator_module_tests()`
- `test_function()`

### person_lookup_utils.py

**Public Functions:**
- `create_not_found_result()`
- `create_result_from_api()`
- `create_result_from_gedcom()`
- `format_for_ai()`
- `person_lookup_utils_module_tests()`
- `run_comprehensive_tests()`
- `to_dict()`

**Test Functions:**
- `_test_confidence_scoring()`
- `_test_create_not_found_result()`
- `_test_create_result_from_api()`
- `_test_create_result_from_gedcom()`
- `_test_format_for_ai()`
- `_test_person_lookup_result_creation()`
- `person_lookup_utils_module_tests()`
- `run_comprehensive_tests()`

### prompt_telemetry.py

**Public Functions:**
- `analyze_experiments()`
- `build_quality_baseline()`
- `detect_quality_regression()`
- `load_quality_baseline()`
- `prompt_telemetry_module_tests()`
- `record_extraction_experiment_event()`
- `summarize_experiments()`

**Test Functions:**
- `_test_build_baseline_and_regression()`
- `_test_record_and_summarize()`
- `_test_variant_analysis()`
- `prompt_telemetry_module_tests()`

### record_sharing.py

**Public Functions:**
- `create_record_sharing_message()`
- `extract_record_url()`
- `format_multiple_records()`
- `format_record_reference()`
- `format_record_with_link()`
- `record_sharing_module_tests()`
- `run_comprehensive_tests()`

**Test Functions:**
- `_test_empty_multiple_records()`
- `_test_minimal_record_formatting()`
- `_test_record_url_extraction_missing()`
- `_test_record_url_extraction_present()`
- `_test_record_with_date_only()`
- `_test_record_with_link_no_url()`
- `_test_record_with_place_only()`
- `_test_sharing_message_minimal()`
- `_test_single_record_in_list()`
- `record_sharing_module_tests()`
- `run_comprehensive_tests()`
- `test_basic_record_formatting()`
- `test_complete_sharing_message()`
- `test_multiple_records_formatting()`
- `test_record_with_url()`
- `test_record_without_source()`
- `test_url_extraction()`

### relationship_diagram.py

**Public Functions:**
- `format_relationship_for_message()`
- `generate_relationship_diagram()`
- `relationship_diagram_module_tests()`
- `run_comprehensive_tests()`

**Test Functions:**
- `_test_all_diagram_styles()`
- `_test_empty_relationship_path()`
- `_test_format_relationship_complex()`
- `_test_format_relationship_empty_path()`
- `_test_format_relationship_simple()`
- `_test_invalid_diagram_style()`
- `_test_single_step_relationship()`
- `relationship_diagram_module_tests()`
- `run_comprehensive_tests()`
- `test_compact_diagram()`
- `test_horizontal_diagram()`
- `test_message_formatting()`
- `test_vertical_diagram()`

### relationship_utils.py

**Public Functions:**
- `convert_api_path_to_unified_format()`
- `convert_discovery_api_path_to_unified_format()`
- `convert_gedcom_path_to_unified_format()`
- `explain_relationship_path()`
- `fast_bidirectional_bfs()`
- `format_api_relationship_path()`
- `format_name()`
- `format_relationship_path_unified()`
- `get_element_by_id()`
- `relationship_module_tests()`
- `relationship_utils_module_tests()`

**Test Functions:**
- `_run_basic_functionality_tests()`
- `_run_conversion_tests()`
- `_run_validation_tests()`
- `_test_api_relationship_formatting()`
- `_test_discovery_api_conversion()`
- `_test_gedcom_path_conversion()`
- `_test_general_api_conversion()`
- `_test_unified_path_formatting()`
- `relationship_module_tests()`
- `relationship_utils_module_tests()`
- `test_bfs_pathfinding()`
- `test_bidirectional_bfs()`
- `test_error_handling()`
- `test_function_availability()`
- `test_name_formatting()`
- `test_performance()`
- `test_relationship_terms()`

### research_guidance_prompts.py

**Public Functions:**
- `create_brick_wall_analysis_prompt()`
- `create_conversation_response_prompt()`
- `create_research_guidance_prompt()`
- `research_guidance_prompts_module_tests()`
- `run_comprehensive_tests()`

**Test Functions:**
- `_test_brick_wall_analysis_prompt_structure()`
- `_test_complete_brick_wall_analysis_prompt()`
- `_test_complete_conversation_response_prompt()`
- `_test_complete_research_guidance_prompt()`
- `_test_conversation_response_prompt_structure()`
- `_test_minimal_brick_wall_analysis_prompt()`
- `_test_minimal_conversation_response_prompt()`
- `_test_minimal_research_guidance_prompt()`
- `_test_research_guidance_prompt_structure()`
- `research_guidance_prompts_module_tests()`
- `run_comprehensive_tests()`
- `test_basic_research_guidance_prompt()`
- `test_brick_wall_analysis_prompt()`
- `test_conversation_response_prompt()`
- `test_prompt_with_available_records()`
- `test_prompt_with_common_ancestors()`

### research_prioritization.py

**Public Functions:**
- `prioritize_research_tasks()`
- `research_prioritization_module_tests()`

**Test Functions:**
- `research_prioritization_module_tests()`
- `test_cluster_generation_and_efficiency()`
- `test_dna_verification_task_creation()`
- `test_priority_scoring_and_ranking()`
- `test_research_prioritization()`

### research_suggestions.py

**Public Functions:**
- `generate_research_suggestions()`
- `research_suggestions_tests()`

**Test Functions:**
- `_test_basic_research_suggestions()`
- `_test_complete_research_suggestions()`
- `_test_empty_input()`
- `_test_formatted_message_structure()`
- `_test_location_collections_extraction()`
- `_test_multiple_locations()`
- `_test_record_types_generation()`
- `_test_result_limits()`
- `_test_strategies_generation()`
- `_test_time_period_collections()`
- `_test_time_period_collections_extraction()`
- `research_suggestions_tests()`

### search_criteria_utils.py

**Public Functions:**
- `display_family_members()`
- `get_unified_search_criteria()`
- `mock_input()`
- `search_criteria_utils_module_tests()`

**Test Functions:**
- `_test_create_date_object()`
- `_test_display_family_members()`
- `_test_format_years_display()`
- `_test_get_unified_search_criteria_cancelled()`
- `_test_get_unified_search_criteria_minimal()`
- `_test_get_unified_search_criteria_valid()`
- `_test_parse_gender_input()`
- `_test_parse_year_input()`
- `_test_print_functions()`
- `_test_sanitize_input()`
- `search_criteria_utils_module_tests()`

### security_manager.py

**Public Functions:**
- `decrypt_credentials()`
- `delete_credentials()`
- `encrypt_credentials()`
- `get_credential()`
- `migrate_env_credentials()`
- `prompt_for_credentials()`
- `security_manager_module_tests()`
- `setup_secure_credentials()`
- `validate_credentials()`

**Test Functions:**
- `_test_credential_deletion()`
- `_test_credential_encryption_decryption()`
- `_test_credential_validation()`
- `_test_error_handling()`
- `_test_file_security()`
- `_test_full_workflow()`
- `_test_individual_credential_retrieval()`
- `_test_master_key_operations()`
- `_test_multiple_instances()`
- `_test_performance()`
- `_test_security_manager_instantiation()`
- `security_manager_module_tests()`

### selenium_utils.py

**Public Functions:**
- `close_tabs()`
- `export_cookies()`
- `extract_attribute()`
- `extract_text()`
- `force_user_agent()`
- `get_driver_cookies()`
- `get_element_text()`
- `is_browser_open()`
- `is_elem_there()`
- `is_element_visible()`
- `safe_click()`
- `scroll_to_element()`
- `selenium_module_tests()`
- `selenium_utils_module_tests()`
- `wait_for_element()`

**Test Functions:**
- `_test_element_text()`
- `_test_force_user_agent()`
- `_test_function_availability()`
- `_test_performance()`
- `_test_safe_execution()`
- `selenium_module_tests()`
- `selenium_utils_module_tests()`
- `test_element_text()`
- `test_force_user_agent()`
- `test_function_availability()`
- `test_performance()`
- `test_safe_execution()`

### session_utils.py

**Public Functions:**
- `authenticate_session()`
- `check_and_login()`
- `clear_cached_session()`
- `close_cached_session()`
- `create_and_start_session()`
- `ensure_session_for_tests()`
- `ensure_session_for_tests_sm_only()`
- `get_authenticated_session()`
- `session_utils_module_tests()`
- `slow_login_check()`
- `slow_start()`
- `start_session()`
- `validate_session_ready()`

**Test Functions:**
- `_test_authenticate_session_already_logged_in()`
- `_test_authenticate_session_login_failure()`
- `_test_authenticate_session_login_required()`
- `_test_authenticate_session_timeout()`
- `_test_clear_cached_session()`
- `_test_close_cached_session()`
- `_test_create_and_start_session_failure()`
- `_test_create_and_start_session_success()`
- `_test_create_and_start_session_timeout()`
- `_test_ensure_session_for_tests_sm_only_wrapper()`
- `_test_ensure_session_for_tests_wrapper()`
- `_test_get_authenticated_session_cached()`
- `_test_get_authenticated_session_new()`
- `_test_session_caching_behavior()`
- `_test_validate_session_ready_no_uuid()`
- `_test_validate_session_ready_not_ready()`
- `_test_validate_session_ready_success()`
- `ensure_session_for_tests()`
- `ensure_session_for_tests_sm_only()`
- `session_utils_module_tests()`

### standard_imports.py

**Public Functions:**
- `get_standard_logger()`
- `get_unified_test_framework()`
- `safe_import()`
- `safe_import_from()`
- `setup_module()`
- `standard_imports_module_tests()`

**Test Functions:**
- `_test_core_imports_availability()`
- `_test_function_registration()`
- `_test_import_standardization()`
- `_test_logger_creation()`
- `_test_module_cleanup()`
- `_test_module_setup()`
- `_test_performance()`
- `_test_safe_imports()`
- `_test_standard_library_availability()`
- `get_unified_test_framework()`
- `standard_imports_module_tests()`

### test_all_llm_models.py

**Public Functions:**
- `calculate_quality_score()`
- `main()`
- `print_final_comparison()`
- `print_model_instructions()`
- `wait_for_user_confirmation()`

**Test Functions:**
- `test_all_models()`
- `test_model()`

### test_framework.py

**Public Functions:**
- `add_warning()`
- `assert_valid_config()`
- `assert_valid_function()`
- `blue()`
- `bold()`
- `clean_test_output()`
- `cleanup_test_environment()`
- `clear()`
- `colorize()`
- `create_isolated_test_environment()`
- `create_mock_data()`
- `create_standardized_test_data()`
- `create_test_data_factory()`
- `critical()`
- `cyan()`
- `database_rollback_test()`
- `debug()`
- `demo_tests()`
- `error()`
- `finish_suite()`
- `format_score_breakdown_table()`
- `format_search_criteria()`
- `format_test_result()`
- `format_test_section_header()`
- `get_messages()`
- `get_test_mode()`
- `gray()`
- `green()`
- `has_ansi_codes()`
- `info()`
- `magenta()`
- `mock_logger_context()`
- `red()`
- `restore_debug_logging()`
- `run_test()`
- `standardized_test_wrapper()`
- `start_suite()`
- `strip_ansi_codes()`
- `suppress_debug_logging()`
- `suppress_logging()`
- `underline()`
- `warning()`
- `white()`
- `wrapper()`
- `yellow()`

**Test Functions:**
- `_test_colors()`
- `_test_context_managers()`
- `_test_icons()`
- `_test_mock_data()`
- `_test_standardized_data_factory()`
- `_test_test_suite_creation()`
- `clean_test_output()`
- `cleanup_test_environment()`
- `create_isolated_test_environment()`
- `create_standardized_test_data()`
- `create_test_data_factory()`
- `database_rollback_test()`
- `demo_tests()`
- `format_test_result()`
- `format_test_section_header()`
- `get_test_mode()`
- `run_test()`
- `standardized_test_wrapper()`
- `test_colors()`
- `test_framework_module_tests()`
- `test_function_availability()`
- `test_icons()`
- `test_mock_data()`

### test_local_llm.py

**Public Functions:**
- `main()`

**Test Functions:**
- `test_configuration()`
- `test_genealogical_prompt()`
- `test_lm_studio_direct()`

### test_utilities.py

**Public Functions:**
- `assert_database_state()`
- `assert_function_behavior()`
- `create_composite_validator()`
- `create_file_extension_validator()`
- `create_method_delegator()`
- `create_parameterized_test_function()`
- `create_property_delegator()`
- `create_range_validator()`
- `create_standard_test_runner()`
- `create_string_validator()`
- `create_test_function()`
- `create_test_session()`
- `create_type_validator()`
- `decorated_safe_func()`
- `delegator()`
- `dummy_test()`
- `get_test_function()`
- `getter()`
- `mock_api_response()`
- `run_comprehensive_tests()`
- `run_parameterized_tests()`
- `safe_func()`
- `sample_function()`
- `temp_function()`
- `validator()`

**Test Functions:**
- `_test_assert_function_behavior()`
- `_test_basic_functions()`
- `_test_factory_functions()`
- `_test_function_registry()`
- `_test_mock_api_response()`
- `_test_runner_factory()`
- `create_parameterized_test_function()`
- `create_standard_test_runner()`
- `create_test_function()`
- `create_test_session()`
- `dummy_test()`
- `get_test_function()`
- `run_comprehensive_tests()`
- `run_parameterized_tests()`
- `test_func()`
- `test_func_with_param()`
- `test_function()`
- `test_utilities_module_tests()`

### tree_stats_utils.py

**Public Functions:**
- `calculate_ethnicity_commonality()`
- `calculate_tree_statistics()`
- `tree_stats_utils_module_tests()`

**Test Functions:**
- `_test_calculate_ethnicity_commonality_with_no_data()`
- `_test_calculate_tree_statistics_with_invalid_profile()`
- `_test_calculate_tree_statistics_with_valid_profile()`
- `_test_empty_ethnicity_commonality_structure()`
- `_test_empty_statistics_structure()`
- `_test_statistics_cache_hit()`
- `_test_statistics_ethnicity_regions_structure()`
- `_test_statistics_functions_available()`
- `_test_statistics_match_counts()`
- `_test_statistics_timestamp_format()`
- `_test_statistics_with_tree_owner()`
- `tree_stats_utils_module_tests()`

### universal_scoring.py

**Public Functions:**
- `apply_universal_scoring()`
- `calculate_display_bonuses()`
- `format_scoring_breakdown()`
- `universal_scoring_module_tests()`
- `validate_search_criteria()`

**Test Functions:**
- `_test_confidence_levels()`
- `_test_criteria_validation_gender()`
- `_test_criteria_validation_invalid_year()`
- `_test_criteria_validation_names()`
- `_test_criteria_validation_years()`
- `_test_display_bonuses_action10_format()`
- `_test_display_bonuses_action11_format()`
- `_test_display_bonuses_no_bonus()`
- `_test_scoring_breakdown_format()`
- `_test_universal_scoring_basic()`
- `_test_universal_scoring_exact_match()`
- `_test_universal_scoring_max_results()`
- `_test_universal_scoring_multiple_candidates()`
- `_test_universal_scoring_no_match()`
- `_test_universal_scoring_partial_match()`
- `universal_scoring_module_tests()`

### utils.py

**Public Functions:**
- `call()`
- `check_circuit_breaker()`
- `consent()`
- `decorator()`
- `decrease_delay()`
- `ensure_browser_open()`
- `enter_creds()`
- `format_name()`
- `get_metrics()`
- `get_rate_limiter()`
- `get_state()`
- `handle_two_fa()`
- `increase_delay()`
- `is_throttled()`
- `log_in()`
- `login_status()`
- `main()`
- `make_ube()`
- `nav_to_page()`
- `ordinal_case()`
- `parse_cookie()`
- `prevent_system_sleep()`
- `print_metrics_summary()`
- `record_failure()`
- `record_success()`
- `reset()`
- `reset_delay()`
- `reset_metrics()`
- `restore_system_sleep()`
- `retry()`
- `retry_api()`
- `run_comprehensive_tests()`
- `time_wait()`
- `utils_module_tests()`
- `wait()`
- `wrapper()`

**Test Functions:**
- `run_comprehensive_tests()`
- `test_api_request_function()`
- `test_check_for_unavailability()`
- `test_circuit_breaker()`
- `test_decorators()`
- `test_format_name()`
- `test_func()`
- `test_login_status_function()`
- `test_module_registration()`
- `test_ordinal_case()`
- `test_parse_cookie()`
- `test_performance_validation()`
- `test_rate_limiter()`
- `test_session_manager()`
- `test_sleep_prevention()`
- `utils_module_tests()`
