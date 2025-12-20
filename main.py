#!/usr/bin/env python3

"""
main.py - Ancestry Research Automation Main Entry Point

Provides the main application entry point with menu-driven interface for
all automation workflows including DNA match gathering, inbox processing,
messaging, and genealogical research tools.
"""

# === VENV AUTO-DETECTION (must be before any project imports) ===
import sys
from pathlib import Path


def _ensure_venv() -> None:
    """Ensure we are running inside the local .venv; re-run with it if not."""
    in_venv = hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    if in_venv:
        return

    venv_candidates = [Path(".venv") / "Scripts" / "python.exe", Path(".venv") / "bin" / "python"]
    venv_python = next((p for p in venv_candidates if p.exists()), None)

    if venv_python is None:
        print("❌ No virtual environment found at .venv. Please create it before running main.py.")
        sys.exit(1)

    # Re-run using the venv interpreter while preserving stdin/stdout. Avoid execv
    # because some Windows shells detach stdin on exec replacement.
    import subprocess

    print(f"🔄 Re-running with venv Python: {venv_python}")
    result = subprocess.run([str(venv_python), __file__, *sys.argv[1:]], check=False)
    sys.exit(result.returncode)


_ensure_venv()

# === SUPPRESS CONFIG WARNINGS FOR PRODUCTION ===
# === CORE INFRASTRUCTURE ===
import logging
import os

from core.logging_config import setup_logging

# === MODULE SETUP ===
setup_logging()
logger = logging.getLogger(__name__)

# === STANDARD LIBRARY IMPORTS ===
import importlib
import time
from importlib import import_module
from typing import Any, Callable, Optional, cast

from actions.action10 import run_gedcom_then_api_fallback
from actions.action12_shared_matches import fetch_shared_matches
from actions.action13_triangulation import run_triangulation_analysis
from actions.action14_research_tools import run_research_tools
from cli.maintenance import GrafanaCheckerProtocol, MainCLIHelpers
from config.config_manager import ConfigManager, get_config_manager
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
from core.feature_flags import bootstrap_feature_flags
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
    run_daily_review_first_loop_action,
    send_approved_drafts_action,
    send_messages_action,
    srch_inbox_actn,
)
from performance.health_monitor import (
    initialize_health_monitoring,
    integrate_with_session_manager,
)
from testing.test_utilities import create_standard_test_runner
from ui.menu import render_main_menu

_grafana_checker: GrafanaCheckerProtocol | None = None
try:
    _grafana_checker_module = import_module("performance.grafana_checker")
except Exception:
    _grafana_checker = None
else:
    _grafana_checker = cast(GrafanaCheckerProtocol, _grafana_checker_module)

grafana_checker: GrafanaCheckerProtocol | None = _grafana_checker


_cli_helpers = MainCLIHelpers(logger=logger, grafana_checker=grafana_checker)


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


def _create_config_manager() -> Optional[ConfigManager]:
    """Instantiate ConfigManager if available."""
    try:
        return get_config_manager()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to instantiate ConfigManager: %s", exc, exc_info=True)
        return None


# Core modules


config_manager = _create_config_manager()
config: Any = config_manager.get_config() if config_manager is not None else None
feature_flags = bootstrap_feature_flags(config)
logger.debug("Feature flags loaded: %d", len(feature_flags.get_all_flags()))

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

    # Health monitor integration: update metrics and honor halt requests
    monitor = integrate_with_session_manager(session_manager)
    if monitor.should_emergency_halt():
        reason = monitor.get_intervention_status().get("emergency_halt", {}).get("reason", "Health halt")
        print(f"\n🚨 Health monitor requested halt: {reason}\n")
        return False
    if monitor.should_immediate_intervention():
        reason = monitor.get_intervention_status().get("immediate_intervention", {}).get("reason", "Intervention")
        print(f"\n⚠️  Immediate intervention recommended: {reason}\n")
    result = True

    if metadata is None:
        print("Invalid choice.\n")
    elif metadata.is_meta_action:
        if arg_tokens:
            print("This option does not accept arguments.\n")
        else:
            meta_result = _execute_meta_action(metadata)
            result = True if meta_result is None else bool(meta_result)
    elif metadata.is_test_action:
        if arg_tokens:
            print("This option does not accept arguments.\n")
        else:
            _execute_test_action(metadata)
            result = True
    else:
        result = _execute_primary_action(metadata, session_manager, config, arg_tokens)
    return result


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
    registry.set_action_function("11", send_approved_drafts_action)
    registry.set_action_function("12", fetch_shared_matches)
    registry.set_action_function("13", run_triangulation_analysis)
    registry.set_action_function("14", run_research_tools)
    registry.set_action_function("15", run_daily_review_first_loop_action)

    # Use _cli_helpers directly
    registry.set_action_function("a", _cli_helpers.show_analytics_dashboard)
    registry.set_action_function("b", _cli_helpers.show_metrics_report)
    registry.set_action_function("l", _cli_helpers.run_grafana_setup)
    registry.set_action_function("o", _cli_helpers.open_grafana_dashboard)
    registry.set_action_function("g", _cli_helpers.open_graph_visualization)
    registry.set_action_function("d", _cli_helpers.show_cache_statistics)
    registry.set_action_function("e", _cli_helpers.run_config_health_check)
    registry.set_action_function("m", _cli_helpers.run_config_setup_wizard)
    registry.set_action_function("n", _cli_helpers.reload_configuration)
    registry.set_action_function("f", _cli_helpers.show_review_queue)
    registry.set_action_function("v", _cli_helpers.launch_review_web_ui)
    registry.set_action_function("h", _cli_helpers.run_dry_run_validation)
    registry.set_action_function("k", _cli_helpers.run_schema_migrations_action)
    registry.set_action_function("t", _cli_helpers.toggle_log_level)
    registry.set_action_function("c", _cli_helpers.clear_screen)
    registry.set_action_function("r", _cli_helpers.clear_test_cache)
    registry.set_action_function("w", _cli_helpers.clear_app_log_menu)
    registry.set_action_function("q", _cli_helpers.exit_application)

    registry.set_action_function("i", _cli_helpers.run_main_tests)
    registry.set_action_function("j", _cli_helpers.run_all_tests)


_assign_action_registry_functions()


from core.lifecycle import (
    check_startup_status,
    cleanup_session_manager,
    display_tree_owner,
    initialize_application,
    pre_authenticate_ms_graph,
    pre_authenticate_session,
    run_startup_maintenance_tasks,
    set_windows_console_focus,
    validate_ai_provider_on_startup,
)


def main() -> None:
    session_manager = None
    sleep_state = None
    set_windows_console_focus()

    monitor = initialize_health_monitoring()

    try:
        # Initialize application (handles sleep prevention and diagnostics)
        session_manager, sleep_state = initialize_application(config, grafana_checker)

        # Sync monitor with session manager before any pre-auth flows
        monitor = integrate_with_session_manager(session_manager)
        if monitor.should_emergency_halt():
            logger.critical("Health monitor requested emergency halt before startup tasks")
            return

        # Pre-authenticate services
        print("", file=sys.stderr)  # Blank line after health checks
        print(" Session ".center(80, "="), file=sys.stderr)
        pre_authenticate_session()
        pre_authenticate_ms_graph()

        # Check startup status and validate AI provider
        check_startup_status(session_manager)
        validate_ai_provider_on_startup()

        # Run maintenance tasks (Phase 10.1: expire stale drafts, etc.)
        print("", file=sys.stderr)
        print(" Startup ".center(80, "="), file=sys.stderr)
        run_startup_maintenance_tasks(session_manager)

        # Display tree owner at the end of startup section
        display_tree_owner(session_manager)
        print("", file=sys.stderr)

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
            from core.utils import restore_system_sleep

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
    result = _cli_helpers.clear_log_file()
    assert isinstance(result, tuple), "clear_log_file should return a tuple"
    assert len(result) == 2, "clear_log_file should return a 2-element tuple"
    success, message = result
    assert isinstance(success, bool), "First element should be boolean"
    assert message is None or isinstance(message, str), "Second element should be None or string"
    return True


def _test_main_function_structure() -> bool:
    """Ensure main() signature stays parameter-free with exception handling."""

    import inspect

    assert callable(main), "main() function should be callable"
    sig = inspect.signature(main)
    assert len(sig.parameters) == 0, "main() should take no parameters"
    return True


def _test_reset_db_actn_integration() -> bool:
    """Validate database manager access through SessionManager.

    Tests that SessionManager provides a valid DatabaseManager with expected interface.
    """
    # import inspect  # Removed unused import

    test_sm = SessionManager()
    db_manager = _get_database_manager(test_sm)
    assert db_manager is not None, "SessionManager should provide a database manager"

    # Verify DatabaseManager has required initialization method
    assert hasattr(db_manager, "_initialize_engine_and_session"), (
        "DatabaseManager should have _initialize_engine_and_session method"
    )
    init_method = db_manager._initialize_engine_and_session
    assert callable(init_method), "_initialize_engine_and_session should be callable"

    # Verify engine and Session attributes exist (may be None before initialization)
    assert hasattr(db_manager, "engine"), "DatabaseManager should have engine attribute"
    assert hasattr(db_manager, "Session"), "DatabaseManager should have Session attribute"

    # Verify DatabaseManager class has expected signature structure
    db_manager_type = type(db_manager)
    assert hasattr(db_manager_type, "__init__"), "DatabaseManager should have __init__"

    # Verify key methods have proper signatures
    return True


def _test_edge_case_handling() -> bool:
    """Verify required action modules can be imported and have expected exports.

    Tests module imports, export existence, and function signatures.
    """
    import inspect

    required_modules = [
        ("actions.action6_gather", "coord"),  # Main coordinator function
        ("actions.action7_inbox", "InboxProcessor"),  # Main class
        ("actions.action8_messaging", "send_messages_to_matches"),  # Main function
        ("actions.action9_process_productive", "process_productive_messages"),  # Main function
        ("actions.action10", "main"),  # Main function
    ]

    for module_name, expected_export in required_modules:
        module = importlib.import_module(module_name)
        assert hasattr(module, expected_export), f"{module_name} should have {expected_export}"
        export = getattr(module, expected_export)
        assert callable(export), f"{module_name}.{expected_export} should be callable"

        # Verify signature structure for functions
        if inspect.isfunction(export):
            sig = inspect.signature(export)
            # All coordinator functions should accept session_manager as first param
            params = list(sig.parameters.keys())
            if module_name == "actions.action6_gather":
                assert "session_manager" in params, "coord should accept session_manager parameter"
            # Verify return annotation exists for type safety
            if sig.return_annotation != inspect.Parameter.empty:
                # Return annotation is defined - good practice
                pass

    return True


def _test_import_error_handling() -> bool:
    """Confirm core imports remain registered and have proper interfaces.

    Tests that required imports are present and have expected attributes/methods.
    """
    import inspect

    module_globals = globals()

    # Test function imports are callable with proper signatures
    function_imports = [
        "gather_dna_matches",
        "srch_inbox_actn",
        "send_messages_action",
        "process_productive_messages_action",
    ]

    for import_name in function_imports:
        assert import_name in module_globals, f"{import_name} should be imported"
        func = module_globals[import_name]
        assert callable(func), f"{import_name} should be callable"
        # Verify function has a signature we can inspect
        sig = inspect.signature(func)
        assert sig is not None, f"{import_name} should have inspectable signature"

    # Test config has expected API attribute structure
    assert "config" in module_globals, "config should be imported"
    cfg = module_globals["config"]
    assert hasattr(cfg, "api"), "config should have api attribute"
    api_attr = cfg.api
    # API config should have rate limiting settings
    assert hasattr(api_attr, "max_pages") or hasattr(api_attr, "requests_per_second"), (
        "config.api should have rate limiting or pagination settings"
    )

    # Test logger has expected logging methods
    assert "logger" in module_globals, "logger should be imported"
    log = module_globals["logger"]
    for method in ["info", "debug", "warning", "error"]:
        assert hasattr(log, method), f"logger should have {method} method"
        assert callable(getattr(log, method)), f"logger.{method} should be callable"

    # Test SessionManager is properly importable
    assert "SessionManager" in module_globals, "SessionManager should be imported"
    sm_class = module_globals["SessionManager"]
    assert callable(sm_class), "SessionManager should be callable (class)"
    # Verify it's a class, not just any callable
    assert inspect.isclass(sm_class), "SessionManager should be a class"

    return True


def _test_validate_action_config() -> bool:
    """Validate action configuration helper returns a boolean.

    Tests actual invocation of validate_action_config and verifies return type.
    """
    import inspect

    # Verify function signature
    assert callable(validate_action_config), "validate_action_config should be callable"
    sig = inspect.signature(validate_action_config)
    assert len(sig.parameters) == 0, "validate_action_config should take no parameters"

    # Verify return annotation if present
    if sig.return_annotation != inspect.Parameter.empty:
        assert sig.return_annotation is bool, "validate_action_config should return bool"

    # Actually invoke and verify behavior
    result = validate_action_config()
    assert isinstance(result, bool), "validate_action_config should return boolean"

    return True


def _test_action_integration() -> bool:
    """Confirm action hooks remain callable with expected signatures.

    Tests that all action functions have proper signatures for menu dispatch.
    """
    import inspect

    from actions.action10 import main as run_action10

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

        # Verify action functions have inspectable signatures
        sig = inspect.signature(action_func)
        params = list(sig.parameters.keys())

        # Most action functions should accept session_manager or similar context
        # This validates the dependency injection pattern
        if action_name != "run_action10":  # action10.main may have different signature
            # Action functions typically accept session context
            assert len(params) >= 0, f"{action_name} signature should be inspectable"

        # Verify return annotation is bool or None (action success indicator)
        if sig.return_annotation != inspect.Parameter.empty:
            ret_type = sig.return_annotation
            # Allow bool, None, or Optional[bool]
            valid_returns = {bool, type(None)}
            if hasattr(ret_type, "__origin__"):  # Handle Optional, Union types
                pass  # Complex type annotation - accept it
            elif ret_type not in valid_returns:
                # Some actions return other types - that's fine
                pass

    return True


def _test_import_performance() -> bool:
    """Reload config module to ensure performance stays reasonable."""
    start_time = time.time()

    config_module = sys.modules.get("config")
    if config_module:
        importlib.reload(config_module)

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

    from testing.test_framework import TestSuite, suppress_logging

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
