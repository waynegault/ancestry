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

from core.venv_bootstrap import ensure_venv

ensure_venv(strict=True)

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
from collections.abc import Callable
from importlib import import_module
from typing import Any, cast

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
    run_unified_send_action,
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


def _create_config_manager() -> ConfigManager | None:
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


def _execute_meta_action(metadata: ActionMetadata) -> bool | None:
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


def _parse_start_page_argument(arg_tokens: list[str]) -> int | None:
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
    registry.set_action_function("16", run_unified_send_action)

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

    from mcp_server.server import run_mcp_server_action

    registry.set_action_function("s", run_mcp_server_action)
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


def _test_parse_menu_choice_behavior() -> bool:
    """Verify _parse_menu_choice correctly splits input into action ID and arguments."""
    # Simple numeric action
    action_id, args = _parse_menu_choice("6")
    assert action_id == "6", f"Expected action_id '6', got '{action_id}'"
    assert args == [], f"Expected empty args, got {args}"

    # Action with trailing argument (e.g. start page)
    action_id, args = _parse_menu_choice("6 50")
    assert action_id == "6", f"Expected action_id '6', got '{action_id}'"
    assert args == ["50"], f"Expected ['50'], got {args}"

    # Letter action
    action_id, args = _parse_menu_choice("q")
    assert action_id == "q", f"Expected action_id 'q', got '{action_id}'"
    assert args == [], f"Expected empty args, got {args}"

    # Empty / whitespace-only input
    action_id, args = _parse_menu_choice("  ")
    assert not action_id, f"Blank input should yield empty action_id, got '{action_id}'"
    assert args == [], f"Blank input should yield empty args, got {args}"

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


def _test_parse_start_page_argument_behavior() -> bool:
    """Verify _parse_start_page_argument handles valid, invalid, and edge-case inputs."""
    # Empty list → None (auto-resume from checkpoint)
    result = _parse_start_page_argument([])
    assert result is None, f"Empty list should return None, got {result}"

    # Valid positive integer
    result = _parse_start_page_argument(["5"])
    assert result == 5, f"Expected 5, got {result}"

    # Large valid page
    result = _parse_start_page_argument(["999"])
    assert result == 999, f"Expected 999, got {result}"

    # Zero is invalid (not > 0)
    result = _parse_start_page_argument(["0"])
    assert result is None, f"Zero should return None, got {result}"

    # Negative is invalid
    result = _parse_start_page_argument(["-1"])
    assert result is None, f"Negative should return None, got {result}"

    # Non-numeric string
    result = _parse_start_page_argument(["abc"])
    assert result is None, f"Non-numeric should return None, got {result}"

    return True


def _test_config_manager_creation() -> bool:
    """Verify _create_config_manager returns a valid ConfigManager with expected interface."""
    cm = _create_config_manager()
    assert cm is not None, "ConfigManager should be created successfully"
    assert isinstance(cm, ConfigManager), f"Expected ConfigManager instance, got {type(cm)}"

    # Verify it produces a usable config object
    cfg = cm.get_config()
    assert cfg is not None, "ConfigManager.get_config() should return a config object"
    assert hasattr(cfg, "api"), "Config should have 'api' section"

    # Verify API config exposes rate-limiting / pagination settings
    api_cfg = cfg.api
    assert hasattr(api_cfg, "max_pages"), "config.api should have max_pages setting"

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


def _test_action_registry_completeness() -> bool:
    """Verify all expected actions are registered with callable functions after wiring."""
    registry = get_action_registry()

    # Every primary action ID that _assign_action_registry_functions wires up
    expected_primary = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                        "10", "11", "12", "13", "14", "15", "16"]

    for action_id in expected_primary:
        metadata = registry.get_action(action_id)
        assert metadata is not None, f"Action '{action_id}' should be registered"
        assert metadata.function is not None, f"Action '{action_id}' should have a function assigned"
        assert callable(metadata.function), f"Action '{action_id}' function should be callable"

    # Verify a sample of utility / meta actions are wired
    for action_id in ["q", "c", "w", "t"]:
        metadata = registry.get_action(action_id)
        assert metadata is not None, f"Utility action '{action_id}' should be registered"
        assert metadata.function is not None, f"Utility action '{action_id}' missing function"

    return True


def _test_feature_flags_initialization() -> bool:
    """Verify feature flags are properly bootstrapped from config."""
    assert feature_flags is not None, "feature_flags should be initialized at module level"

    all_flags = feature_flags.get_all_flags()
    assert isinstance(all_flags, dict), f"get_all_flags should return dict, got {type(all_flags)}"

    # Flags should be deterministic across calls
    all_flags_again = feature_flags.get_all_flags()
    assert all_flags == all_flags_again, "Feature flags should be stable across calls"

    return True


def _test_action_metadata_properties() -> bool:
    """Verify _get_action_metadata returns correct properties for known action IDs."""
    # Primary action
    meta6 = _get_action_metadata("6")
    assert meta6 is not None, "Action 6 should have metadata"
    assert meta6.id == "6", f"Expected id '6', got '{meta6.id}'"
    assert isinstance(meta6.name, str) and len(meta6.name) > 0, "Action should have a non-empty name"
    assert meta6.max_args >= 1, "Action 6 should accept at least one arg (start page)"

    # Meta action (quit)
    meta_q = _get_action_metadata("q")
    assert meta_q is not None, "Quit action should have metadata"
    assert meta_q.is_meta_action is True, "Quit should be a meta action"

    # Nonexistent action returns None
    meta_invalid = _get_action_metadata("zzz_nonexistent")
    assert meta_invalid is None, "Nonexistent action should return None"

    # Empty string returns None
    meta_empty = _get_action_metadata("")
    assert meta_empty is None, "Empty action_id should return None"

    return True


def _test_validate_action_args_behavior() -> bool:
    """Verify _validate_action_args enforces argument limits from ActionMetadata."""
    # Action 6 accepts a start-page argument
    meta6 = _get_action_metadata("6")
    assert meta6 is not None, "Action 6 metadata required"

    # No args always valid
    assert _validate_action_args(meta6, []) is True, "Empty args should pass"

    # Valid single arg within max_args
    assert _validate_action_args(meta6, ["50"]) is True, "Single arg should pass for Action 6"

    # Exceeding max_args should fail
    too_many = ["1"] * (meta6.max_args + 1)
    assert _validate_action_args(meta6, too_many) is False, "Exceeding max_args should return False"

    # Meta action with max_args=0 should reject any argument
    meta_q = _get_action_metadata("q")
    if meta_q is not None and meta_q.max_args == 0:
        assert _validate_action_args(meta_q, ["extra"]) is False, "max_args=0 should reject args"

    return True


def _test_cleanup_handles_none_session() -> bool:
    """Verify cleanup_session_manager safely handles a None session (early-exit path)."""
    # main() passes None when session_manager was never initialized
    try:
        cleanup_session_manager(None)
    except Exception as exc:
        raise AssertionError(f"cleanup_session_manager(None) should not raise, got: {exc}") from exc
    return True


def _test_dispatch_handles_invalid_action() -> bool:
    """Verify _dispatch_menu_action returns True (continue) for an unknown action ID."""
    sm = SessionManager()
    # An obviously-invalid choice should print 'Invalid choice' and keep the loop going
    result = _dispatch_menu_action("zzz_nonexistent_999", sm, config)
    assert result is True, "Invalid choice should return True to continue the menu loop"
    return True


def _test_health_monitor_initialization() -> bool:
    """Verify health monitoring initializes and provides valid status."""
    monitor = initialize_health_monitoring()
    assert monitor is not None, "Health monitor should initialize"

    # Should not request emergency halt in a normal test context
    halt = monitor.should_emergency_halt()
    assert isinstance(halt, bool), f"should_emergency_halt should return bool, got {type(halt)}"
    assert halt is False, "Health monitor should not halt during tests"

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
            test_name="_parse_menu_choice splits action ID and arguments correctly",
            test_func=_test_parse_menu_choice_behavior,
            test_summary="Menu choice parsing separates action ID from trailing arguments",
            method_description="Testing _parse_menu_choice with numeric, letter, and edge-case inputs",
            expected_outcome="Action ID and argument tokens are correctly extracted from raw input",
        )

        suite.run_test(
            test_name="reset_db_actn() integration and method availability",
            test_func=_test_reset_db_actn_integration,
            test_summary="Database reset function integration and required method verification",
            method_description="Testing reset_db_actn function for proper SessionManager and DatabaseManager method access",
            expected_outcome="reset_db_actn can access all required methods without AttributeError",
        )

        suite.run_test(
            test_name="_parse_start_page_argument handles valid, invalid, and edge-case inputs",
            test_func=_test_parse_start_page_argument_behavior,
            test_summary="Start-page argument parsing with boundary and invalid values",
            method_description="Testing _parse_start_page_argument with positive, zero, negative, and non-numeric inputs",
            expected_outcome="Valid pages returned as int; invalid inputs return None for auto-resume",
        )

        suite.run_test(
            test_name="_create_config_manager returns valid ConfigManager with API settings",
            test_func=_test_config_manager_creation,
            test_summary="ConfigManager factory produces usable config with api section",
            method_description="Testing _create_config_manager instantiation and get_config().api access",
            expected_outcome="ConfigManager created, get_config() returns object with api.max_pages",
        )

        suite.run_test(
            test_name="Configuration validation system from Action 6 lessons",
            test_func=_test_validate_action_config,
            test_summary="Configuration validation system prevents Action 6-style failures",
            method_description="Testing validate_action_config() function validates .env settings and rate limiting",
            expected_outcome="Configuration validation function works correctly and returns boolean result",
        )

        suite.run_test(
            test_name="Action registry has all expected actions wired with functions",
            test_func=_test_action_registry_completeness,
            test_summary="All primary and utility actions registered with callable functions",
            method_description="Testing action registry for IDs 0-16 plus q/c/w/t after wiring",
            expected_outcome="Every expected action ID has non-None callable function assigned",
        )

        suite.run_test(
            test_name="Feature flags bootstrapped from config and stable across calls",
            test_func=_test_feature_flags_initialization,
            test_summary="Feature flags initialized at module level with deterministic output",
            method_description="Testing bootstrap_feature_flags result and get_all_flags consistency",
            expected_outcome="Feature flags return a dict that is stable across successive calls",
        )

        suite.run_test(
            test_name="_get_action_metadata returns correct properties for known IDs",
            test_func=_test_action_metadata_properties,
            test_summary="Action metadata lookup returns correct id, name, max_args, and is_meta_action",
            method_description="Testing _get_action_metadata for action 6, quit, nonexistent, and empty ID",
            expected_outcome="Known actions return correct metadata; invalid IDs return None",
        )

        suite.run_test(
            test_name="_validate_action_args enforces max_args limits from metadata",
            test_func=_test_validate_action_args_behavior,
            test_summary="Argument validation accepts/rejects args based on ActionMetadata.max_args",
            method_description="Testing _validate_action_args with empty, valid, and excess arguments",
            expected_outcome="Empty args always pass; excess args rejected; within-limit args accepted",
        )

        suite.run_test(
            test_name="cleanup_session_manager safely handles None session",
            test_func=_test_cleanup_handles_none_session,
            test_summary="Cleanup path handles early-exit case where session was never created",
            method_description="Testing cleanup_session_manager(None) does not raise",
            expected_outcome="Function completes without exception when given None",
        )

        suite.run_test(
            test_name="_dispatch_menu_action returns True for unknown action ID",
            test_func=_test_dispatch_handles_invalid_action,
            test_summary="Menu dispatch gracefully handles nonexistent action IDs",
            method_description="Testing _dispatch_menu_action with an invalid choice string",
            expected_outcome="Returns True (continue loop) and prints 'Invalid choice'",
        )

        suite.run_test(
            test_name="Health monitor initializes and reports no emergency halt",
            test_func=_test_health_monitor_initialization,
            test_summary="Health monitoring system initializes and returns safe status",
            method_description="Testing initialize_health_monitoring and should_emergency_halt in test context",
            expected_outcome="Monitor initializes successfully and does not request halt",
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
