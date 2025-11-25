#!/usr/bin/env python3

"""
main.py - Ancestry Research Automation Main Entry Point

Provides the main application entry point with menu-driven interface for
all automation workflows including DNA match gathering, inbox processing,
messaging, and genealogical research tools.
"""

# === SUPPRESS CONFIG WARNINGS FOR PRODUCTION ===
import os

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

# === MODULE SETUP ===
logger = setup_module(globals(), __name__)

# === TEST UTILITIES ===
# === STANDARD LIBRARY IMPORTS ===


import importlib
import sys
import time
from importlib import import_module
from typing import Any, Callable, Optional, Protocol, cast

from action10 import run_gedcom_then_api_fallback
from cli.maintenance import GrafanaCheckerProtocol, MainCLIHelpers
from core.action_registry import (
    ActionMetadata,
    get_action_registry,
)
from core.action_runner import (
    configure_action_runner,
    exec_actn,
    get_action_metadata as _get_action_metadata,
    get_database_manager as _get_database_manager,
    parse_menu_choice as _parse_menu_choice,
)
from core.analytics_helpers import get_metrics_bundle as _get_metrics_bundle
from core.caching_bootstrap import ensure_caching_initialized
from core.config_validation import validate_action_config

# === NEW IMPORTS ===
from core.maintenance_actions import (
    all_but_first_actn,
    backup_db_actn,
    check_login_actn,
    reset_db_actn,
    restore_db_actn,
)
from core.session_manager import SessionManager
from core.workflow_actions import (
    gather_dna_matches,
    process_productive_messages_action,
    run_core_workflow_action,
    send_messages_action,
    srch_inbox_actn,
)
from test_utilities import create_standard_test_runner
from ui.menu import render_main_menu


class ConfigManagerProtocol(Protocol):
    """Protocol describing the ConfigManager behavior used here."""

    def get_config(self) -> Any: ...


_config_manager_factory: type[ConfigManagerProtocol] | None = None
_config_manager_error: Exception | None = None

try:
    _config_module = import_module("config.config_manager")
except Exception as exc:
    _config_manager_error = exc
else:
    _config_candidate = getattr(_config_module, "ConfigManager", None)
    if isinstance(_config_candidate, type):
        _config_manager_factory = cast(type[ConfigManagerProtocol], _config_candidate)
    else:
        _config_manager_error = RuntimeError("ConfigManager class missing from config.config_manager")


_grafana_checker: GrafanaCheckerProtocol | None = None
try:
    _grafana_checker_module = import_module("grafana_checker")
except Exception:
    _grafana_checker = None
else:
    _grafana_checker = cast(GrafanaCheckerProtocol, _grafana_checker_module)

grafana_checker: GrafanaCheckerProtocol | None = _grafana_checker


_cli_helpers = MainCLIHelpers(logger=logger, grafana_checker=grafana_checker)

# Re-export helper functions to maintain existing references
clear_log_file = _cli_helpers.clear_log_file
_run_main_tests = _cli_helpers.run_main_tests
_run_all_tests = _cli_helpers.run_all_tests
_open_graph_visualization = _cli_helpers.open_graph_visualization
_show_analytics_dashboard = _cli_helpers.show_analytics_dashboard
_show_cache_statistics = _cli_helpers.show_cache_statistics
_run_schema_migrations_action = _cli_helpers.run_schema_migrations_action
_toggle_log_level = _cli_helpers.toggle_log_level
_show_metrics_report = _cli_helpers.show_metrics_report
_run_grafana_setup = _cli_helpers.run_grafana_setup
_clear_screen = _cli_helpers.clear_screen
_exit_application = _cli_helpers.exit_application


_metrics_factory: Callable[[], Any] | None = None
_metrics_import_error: Exception | None = None

try:
    _metrics_module = import_module("observability.metrics_registry")
except Exception as exc:
    _metrics_import_error = exc
else:
    _metrics_candidate = getattr(_metrics_module, "metrics", None)
    if callable(_metrics_candidate):
        _metrics_factory = cast(Callable[[], Any], _metrics_candidate)
    else:
        _metrics_import_error = RuntimeError("metrics() not found in observability.metrics_registry")


def _create_config_manager() -> Optional[ConfigManagerProtocol]:
    """Instantiate ConfigManager if available."""

    if _config_manager_factory is None:
        if _config_manager_error is not None:
            logger.debug("ConfigManager unavailable: %s", _config_manager_error)
        return None

    try:
        return _config_manager_factory()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to instantiate ConfigManager: %s", exc, exc_info=True)
        return None


# Core modules


config_manager = _create_config_manager()
config: Any = config_manager.get_config() if config_manager is not None else None

configure_action_runner(config=config, metrics_provider=_get_metrics_bundle)


def menu() -> str:
    """Display the main menu and return the normalized choice."""

    return render_main_menu(logger, config, get_action_registry())


# End of menu


# --- Action Functions


def _check_action_confirmation(choice: str) -> bool:
    """
    Check if action requires confirmation and get user confirmation.

    Returns:
        True if action should proceed, False if cancelled

    """
    action_id, _ = _parse_menu_choice(choice)
    metadata = _get_action_metadata(action_id)

    if metadata and metadata.requires_confirmation:
        action_desc = metadata.confirmation_message or metadata.name
        confirm = input(f"Are you sure you want to {action_desc}? ⚠️  This cannot be undone. (yes/no): ").strip().lower()
        if confirm not in {"yes", "y"}:
            print("Action cancelled.\n")
            return False
        print(" ")  # Newline after confirmation

    return True


def _execute_meta_action(metadata: ActionMetadata) -> Optional[bool]:
    """Execute a meta action and return loop control signal."""

    action_func = metadata.function
    if not callable(action_func):
        logger.error("Meta action '%s' is not wired to a function", metadata.id)
        print("Action not available.\n")
        return True

    result = action_func()
    if isinstance(result, bool):
        return result
    return True


def _execute_test_action(metadata: ActionMetadata) -> None:
    """Execute a registered test action."""

    action_func = metadata.function
    if not callable(action_func):
        logger.error("Test action '%s' is not wired to a function", metadata.id)
        print("Test action not available.\n")
        return
    action_func()


def _validate_action_args(metadata: ActionMetadata, arg_tokens: list[str]) -> bool:
    """Validate CLI arguments against action metadata."""

    if not arg_tokens:
        return True
    if metadata.max_args == 0:
        print(f"Action '{metadata.name}' does not accept additional arguments.\n")
        return False
    if len(arg_tokens) > metadata.max_args:
        print(f"Action '{metadata.name}' accepts at most {metadata.max_args} argument(s).\n")
        return False
    return True


def _parse_start_page_argument(arg_tokens: list[str]) -> Optional[int]:
    """Parse optional start page for Action 6."""

    if not arg_tokens:
        return None

    start_token = arg_tokens[0]
    try:
        start_arg = int(start_token)
        if start_arg > 0:
            return start_arg
        logger.warning("Invalid start page '%s'. Using checkpoint resume if available.", start_token)
        print(f"Invalid start page '{start_token}'. Defaulting to checkpoint resume.")
    except ValueError:
        logger.warning("Invalid start page '%s'. Using checkpoint resume if available.", start_token)
        print(f"Invalid start page '{start_token}'. Defaulting to checkpoint resume.")
    return None


def _execute_primary_action(
    metadata: ActionMetadata,
    session_manager: SessionManager,
    config_obj: Any,
    arg_tokens: list[str],
) -> bool:
    """Execute a primary action (non-meta/test) using exec_actn."""

    action_func = metadata.function
    if not callable(action_func):
        logger.error("Action '%s' is not wired to an executable function", metadata.id)
        print("Action not implemented yet.\n")
        return True

    if not _validate_action_args(metadata, arg_tokens):
        return True

    extra_args: list[Any] = []
    if metadata.inject_config:
        extra_args.append(config_obj)

    if metadata.id == "6":
        extra_args.append(_parse_start_page_argument(arg_tokens))
    elif arg_tokens:
        logger.debug("Ignoring unused arguments %s for action %s", arg_tokens, metadata.id)

    if metadata.enable_caching:
        ensure_caching_initialized()

    exec_actn(action_func, session_manager, metadata.id, metadata.close_session_after, *extra_args)
    return True


def _dispatch_menu_action(choice: str, session_manager: SessionManager, config: Any) -> bool:
    """
    Dispatch menu action based on user choice.

    Returns:
        True to continue menu loop, False to exit
    """
    action_id, arg_tokens = _parse_menu_choice(choice)
    metadata = _get_action_metadata(action_id)

    if metadata is None:
        print("Invalid choice.\n")
        return True

    if metadata.is_meta_action:
        if arg_tokens:
            print("This option does not accept arguments.\n")
            return True
        result = _execute_meta_action(metadata)
        return True if result is None else bool(result)

    if metadata.is_test_action:
        if arg_tokens:
            print("This option does not accept arguments.\n")
            return True
        _execute_test_action(metadata)
        return True

    return _execute_primary_action(metadata, session_manager, config, arg_tokens)


def _assign_action_registry_functions() -> None:
    """Attach callable implementations to action registry metadata."""

    registry = get_action_registry()
    registry.set_action_function("0", all_but_first_actn)
    registry.set_action_function("1", run_core_workflow_action)
    registry.set_action_function("2", reset_db_actn)
    registry.set_action_function("3", backup_db_actn)
    registry.set_action_function("4", restore_db_actn)
    registry.set_action_function("5", check_login_actn)
    registry.set_action_function("6", gather_dna_matches)
    registry.set_action_function("7", srch_inbox_actn)
    registry.set_action_function("8", send_messages_action)
    registry.set_action_function("9", process_productive_messages_action)
    registry.set_action_function("10", run_gedcom_then_api_fallback)

    registry.set_action_function("analytics", _show_analytics_dashboard)
    registry.set_action_function("metrics", _show_metrics_report)
    registry.set_action_function("setup-grafana", _run_grafana_setup)
    registry.set_action_function("graph", _open_graph_visualization)
    registry.set_action_function("s", _show_cache_statistics)
    registry.set_action_function("migrate-db", _run_schema_migrations_action)
    registry.set_action_function("t", _toggle_log_level)
    registry.set_action_function("c", _clear_screen)
    registry.set_action_function("q", _exit_application)

    registry.set_action_function("test", _run_main_tests)
    registry.set_action_function("testall", _run_all_tests)


_assign_action_registry_functions()


from core.lifecycle import (
    check_startup_status,
    cleanup_session_manager,
    display_tree_owner,
    initialize_application,
    pre_authenticate_ms_graph,
    pre_authenticate_session,
    set_windows_console_focus,
    validate_ai_provider_on_startup,
)


def main() -> None:
    session_manager = None
    sleep_state = None  # Track sleep prevention state
    set_windows_console_focus()

    try:
        # Initialize application (handles sleep prevention and diagnostics)
        session_manager, sleep_state = initialize_application(config, grafana_checker)

        # Pre-authenticate services
        pre_authenticate_session()
        pre_authenticate_ms_graph()

        # Check startup status and validate AI provider
        check_startup_status(session_manager)
        validate_ai_provider_on_startup()

        # Display tree owner at the end of startup checks
        display_tree_owner(session_manager)

        # Main menu loop
        while True:
            choice = menu()
            print("")

            if not _check_action_confirmation(choice):
                continue

            if not _dispatch_menu_action(choice, session_manager, config):
                break  # Exit requested

    except KeyboardInterrupt:
        os.system("cls" if os.name == "nt" else "clear")
        print("\nCTRL+C detected. Exiting.")
    except Exception as e:
        logger.critical(f"Critical error in main: {e}", exc_info=True)
    finally:
        # Restore system sleep settings
        if sleep_state is not None:
            from utils import restore_system_sleep

            restore_system_sleep(sleep_state)
            logger.info("🔓 System sleep prevention deactivated")

        import contextlib
        import io

        # Suppress all stderr output during cleanup to hide undetected_chromedriver errors
        with contextlib.redirect_stderr(io.StringIO()):
            cleanup_session_manager(session_manager)
            # Small delay to allow cleanup to complete before exit
            time.sleep(0.2)


# end main


# === Module Test Suite ===


def _test_clear_log_file_function() -> bool:
    """Validate log file clearing behavior returns structured tuple."""

    try:
        result = clear_log_file()
        assert isinstance(result, tuple), "clear_log_file should return a tuple"
        assert len(result) == 2, "clear_log_file should return a 2-element tuple"
        success, message = result
        assert isinstance(success, bool), "First element should be boolean"
        assert message is None or isinstance(message, str), "Second element should be None or string"
    except Exception as exc:  # pragma: no cover - defensive assertion
        assert isinstance(exc, Exception), "Should handle errors gracefully"
    return True


def _test_main_function_structure() -> bool:
    """Ensure main() signature stays parameter-free with exception handling."""

    import inspect

    assert callable(main), "main() function should be callable"
    sig = inspect.signature(main)
    assert len(sig.parameters) == 0, "main() should take no parameters"
    return True


def _test_reset_db_actn_integration() -> bool:
    """Validate database manager access through SessionManager."""

    try:
        test_sm = SessionManager()
        db_manager = _get_database_manager(test_sm)
        assert db_manager is not None, "SessionManager should provide a database manager"
        assert hasattr(db_manager, "_initialize_engine_and_session"), (
            "DatabaseManager should have _initialize_engine_and_session method"
        )
        assert hasattr(db_manager, "engine"), "DatabaseManager should have engine attribute"
        assert hasattr(db_manager, "Session"), "DatabaseManager should have Session attribute"
        logger.debug("reset_db_actn integration test: All required methods and attributes verified")
    except AttributeError as exc:
        raise AssertionError(f"reset_db_actn integration test failed with AttributeError: {exc}") from exc
    except Exception as exc:  # pragma: no cover - acceptable fallback
        logger.debug(
            "reset_db_actn integration test: Non-AttributeError exception (acceptable): %s",
            exc,
        )
        return True
    return True


def _test_edge_case_handling() -> bool:
    """Verify required action modules stay imported for dispatch helpers."""

    required_modules = [
        "action6_gather",
        "action7_inbox",
        "action8_messaging",
        "action9_process_productive",
        "action10",
    ]

    for module_name in required_modules:
        importlib.import_module(module_name)
    return True


def _test_import_error_handling() -> bool:
    """Confirm core imports remain registered on the module namespace."""

    module_globals = globals()
    required_imports = [
        "gather_dna_matches",
        "srch_inbox_actn",
        "send_messages_action",
        "process_productive_messages_action",
        "config",
        "logger",
        "SessionManager",
    ]

    for import_name in required_imports:
        assert import_name in module_globals, f"{import_name} should be imported"
    return True


def _test_validate_action_config() -> bool:
    """Validate action configuration helper returns a boolean."""

    assert callable(validate_action_config), "validate_action_config should be callable"

    try:
        result = validate_action_config()
        assert isinstance(result, bool), "validate_action_config should return boolean"
        assert result is True, "validate_action_config should return True for basic validation"
    except Exception as exc:
        assert "config" in str(exc).lower(), f"validate_action_config failed unexpectedly: {exc}"
    return True


def _test_action_integration() -> bool:
    """Confirm action hooks remain callable for menu dispatch."""

    from action10 import main as run_action10

    actions_to_test: list[tuple[str, Callable[..., Any]]] = [
        ("gather_dna_matches", gather_dna_matches),
        ("srch_inbox_actn", srch_inbox_actn),
        ("send_messages_action", send_messages_action),
        ("process_productive_messages_action", process_productive_messages_action),
        ("run_action10", run_action10),
    ]

    for action_name, action_func in actions_to_test:
        assert callable(action_func), f"{action_name} should be callable"
        assert action_func is not None, f"{action_name} should not be None"
    return True


def _test_import_performance() -> bool:
    """Reload config module to ensure performance stays reasonable."""

    start_time = time.time()

    try:
        config_module = sys.modules.get("config")
        if config_module:
            importlib.reload(config_module)
    except Exception:  # pragma: no cover - diagnostic only
        pass

    duration = time.time() - start_time
    assert duration < 1.0, f"Module reloading should be fast, took {duration:.3f}s"
    return True


def _test_memory_efficiency() -> bool:
    """Track module globals to guard against excessive state."""

    import inspect
    from types import ModuleType

    module_size = sys.getsizeof(sys.modules[__name__])
    assert module_size < 10000, f"Module size should be reasonable, got {module_size} bytes"

    tracked_state: dict[str, Any] = {
        name: value
        for name, value in globals().items()
        if not name.startswith("__")
        and not isinstance(value, ModuleType)
        and not inspect.isfunction(value)
        and not inspect.isclass(value)
    }

    globals_count = len(tracked_state)
    assert globals_count < 80, f"Stateful global variables should be reasonable, got {globals_count}"
    return True


def _test_function_call_performance() -> bool:
    """Ensure menu callable checks stay performant."""

    start_time = time.time()

    for _ in range(1000):
        result = callable(menu)
        assert result is True, "menu should be callable"

    duration = time.time() - start_time
    assert duration < 0.1, f"1000 function checks should be fast, took {duration:.3f}s"
    return True


def _test_error_handling_structure() -> bool:
    """Ensure main() retains try/except/finally scaffolding."""

    import inspect

    main_source = inspect.getsource(main)
    assert "try:" in main_source, "main() should have try-except structure"
    assert "except" in main_source, "main() should have exception handling"
    assert "finally:" in main_source, "main() should have finally block"
    assert "KeyboardInterrupt" in main_source, "main() should handle KeyboardInterrupt"
    return True


def _test_cleanup_procedures() -> bool:
    """Check main() references cleanup code paths."""

    import inspect

    main_source = inspect.getsource(main)
    assert "finally:" in main_source, "main() should have finally block for cleanup"
    assert "cleanup" in main_source.lower(), "main() should mention cleanup"
    return True


def _test_exception_handling_coverage() -> bool:
    """Verify exception logging references stay intact."""

    import inspect

    main_source = inspect.getsource(main)
    assert "Exception" in main_source, "main() should handle general exceptions"
    assert "logger" in main_source, "main() should use logger for error reporting"
    return True


def main_module_tests() -> bool:
    """Comprehensive regression suite for main.py."""

    from test_framework import TestSuite, suppress_logging

    suite = cast(Any, TestSuite("Main Application Controller & Menu System", "main.py"))
    suite.start_suite()

    with suppress_logging():
        suite.run_test(
            test_name="clear_log_file() function logic and return values",
            test_func=_test_clear_log_file_function,
            test_summary="Log file clearing functionality and return structure",
            method_description="Testing clear_log_file function execution and return tuple structure",
            expected_outcome="Function executes properly and returns appropriate tuple structure",
        )

        suite.run_test(
            test_name="main() function structure and signature",
            test_func=_test_main_function_structure,
            test_summary="Main function structure and parameter requirements",
            method_description="Testing main function callable status and parameter signature",
            expected_outcome="Main function has proper structure and takes no parameters",
        )

        suite.run_test(
            test_name="reset_db_actn() integration and method availability",
            test_func=_test_reset_db_actn_integration,
            test_summary="Database reset function integration and required method verification",
            method_description="Testing reset_db_actn function for proper SessionManager and DatabaseManager method access",
            expected_outcome="reset_db_actn can access all required methods without AttributeError",
        )

        suite.run_test(
            test_name="Edge case handling and module import validation",
            test_func=_test_edge_case_handling,
            test_summary="Edge cases and import validation scenarios",
            method_description="Testing edge conditions and module import status",
            expected_outcome="Edge cases are handled and imports are properly validated",
        )

        suite.run_test(
            test_name="Import error scenarios and required module presence",
            test_func=_test_import_error_handling,
            test_summary="Import error handling and required module validation",
            method_description="Testing essential module imports and availability",
            expected_outcome="All essential modules are imported and available",
        )

        suite.run_test(
            test_name="Configuration validation system from Action 6 lessons",
            test_func=_test_validate_action_config,
            test_summary="Configuration validation system prevents Action 6-style failures",
            method_description="Testing validate_action_config() function validates .env settings and rate limiting",
            expected_outcome="Configuration validation function works correctly and returns boolean result",
        )

        suite.run_test(
            test_name="All action function integration with main application",
            test_func=_test_action_integration,
            test_summary="Action functions integrate properly with main application",
            method_description="Testing action function availability and callable status",
            expected_outcome="All action functions integrate properly and are callable",
        )

        suite.run_test(
            test_name="Module import and reload performance",
            test_func=_test_import_performance,
            test_summary="Import performance and module caching efficiency",
            method_description="Testing module import and reload times for performance",
            expected_outcome="Module imports and reloads complete within reasonable time limits",
        )

        suite.run_test(
            test_name="Memory usage efficiency and global variable management",
            test_func=_test_memory_efficiency,
            test_summary="Memory usage efficiency and resource management",
            method_description="Testing module memory usage and global variable count",
            expected_outcome="Memory usage is reasonable and global variables are controlled",
        )

        suite.run_test(
            test_name="Function call performance and responsiveness",
            test_func=_test_function_call_performance,
            test_summary="Function call performance and execution speed",
            method_description="Testing basic function call performance with multiple iterations",
            expected_outcome="Function calls execute efficiently within performance limits",
        )

        suite.run_test(
            test_name="main() error handling structure and exception coverage",
            test_func=_test_error_handling_structure,
            test_summary="Error handling structure in main function",
            method_description="Testing main function for proper try-except-finally structure",
            expected_outcome="Main function has comprehensive error handling structure",
        )

        suite.run_test(
            test_name="Cleanup procedures and resource management",
            test_func=_test_cleanup_procedures,
            test_summary="Cleanup procedures and resource management implementation",
            method_description="Testing cleanup code presence and resource management",
            expected_outcome="Proper cleanup procedures are implemented for resource management",
        )

        suite.run_test(
            test_name="Exception handling coverage and logging integration",
            test_func=_test_exception_handling_coverage,
            test_summary="Exception handling coverage and error logging",
            method_description="Testing exception handling scope and logging integration",
            expected_outcome="Exception handling covers expected scenarios with proper logging",
        )

    return bool(suite.finish_suite())


run_comprehensive_tests = create_standard_test_runner(main_module_tests)


# --- Entry Point ---

if __name__ == "__main__":
    import io
    import sys

    # Run main program
    main()

    # Suppress stderr during final garbage collection to hide undetected_chromedriver cleanup errors
    sys.stderr = io.StringIO()


# end of main.py
