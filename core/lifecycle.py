#!/usr/bin/env python3

"""
core/lifecycle.py - Application Lifecycle Management

Handles application startup, initialization, and shutdown procedures.
"""

import os
import sys
import tempfile
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional, cast
from unittest import mock

# Ensure project root is on sys.path when running as a script
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.action_registry import get_action_registry
from core.action_runner import (
    get_api_manager,
    get_browser_manager,
    get_database_manager,
)
from core.config_validation import validate_action_config
from core.session_manager import SessionManager
from logging_config import setup_logging
from standard_imports import setup_module
from test_framework import TestSuite
from test_utilities import create_standard_test_runner

logger = setup_module(globals(), __name__)


def get_windows_console_handles() -> tuple[Optional[Any], Optional[Any]]:
    """Return kernel32/user32 handles when available on Windows."""

    if os.name != "nt":
        return None, None

    try:
        import ctypes
    except Exception:
        return None, None

    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return None, None

    kernel32 = getattr(windll, "kernel32", None)
    user32 = getattr(windll, "user32", None)
    return kernel32, user32


def set_windows_console_focus() -> None:
    """Ensure terminal window has focus on Windows."""

    kernel32, user32 = get_windows_console_handles()
    if kernel32 is None or user32 is None:
        return

    try:
        console_window = kernel32.GetConsoleWindow()
    except Exception:
        return

    if not console_window:
        return

    try:
        user32.SetForegroundWindow(console_window)
        user32.ShowWindow(console_window, 9)  # SW_RESTORE
    except Exception:
        logger.debug("Unable to focus Windows console window", exc_info=True)


def print_config_error_message() -> None:
    """Print detailed configuration error message and exit."""
    logger.critical("Configuration validation failed - unable to proceed")
    print("\n❌ CONFIGURATION ERROR:")
    print("   Critical configuration validation failed.")
    print("   This usually means missing credentials or configuration files.")
    print("")
    print("💡 SOLUTION:")
    print("   1. Copy .env.example to .env and add your credentials")
    print("   2. Ensure all required environment variables are set")

    print("\n📚 For detailed instructions:")
    print("   See ENV_IMPORT_GUIDE.md or readme.md")

    print("\nExiting application...")
    sys.exit(1)


def check_startup_status(session_manager: SessionManager) -> None:
    """
    Check and display database connection status at startup.
    Authentication status is already displayed by the authentication steps above.
    """
    # Check database connection
    try:
        db_manager = get_database_manager(session_manager)
        if db_manager and db_manager.ensure_ready():
            logger.info("✅ Database connection OK")
        else:
            logger.warning("⚠️ Database connection not available")
    except Exception as e:
        logger.warning(f"⚠️ Database connection check failed: {e}")

    # CRITICAL: Proactive cookie sync during warmup
    # Ensures fresh cookies for ALL actions before menu display
    # Prevents 303 redirects from stale cookies across all actions
    try:
        api_manager = get_api_manager(session_manager)
        browser_manager = get_browser_manager(session_manager)
        if session_manager.is_sess_valid() and api_manager and browser_manager:
            logger.debug("Syncing browser cookies to API session during warmup...")
            synced = api_manager.sync_cookies_from_browser(browser_manager, session_manager=session_manager)
            if synced:
                logger.info("✅ Cookies refreshed and OK")
            else:
                logger.debug("Cookie sync returned False (may be normal)")
    except Exception as cookie_err:
        logger.debug(f"Cookie sync during warmup failed (non-fatal): {cookie_err}")


def _check_lm_studio_running() -> bool:
    """Check if LM Studio process is running.

    Returns:
        True if LM Studio is running, False otherwise
    """
    import psutil

    for proc in cast(Any, psutil).process_iter(['name']):
        try:
            proc_name = proc.info['name'].lower()
            if 'lm studio' in proc_name or 'lmstudio' in proc_name:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def _validate_local_llm_config(config_schema: Any) -> bool:
    """Validate local LLM configuration with auto-start support.

    Returns:
        True if validation passed, False otherwise
    """
    from openai import OpenAI

    api_key = config_schema.api.local_llm_api_key
    model_name = config_schema.api.local_llm_model
    base_url = config_schema.api.local_llm_base_url

    if not all([api_key, model_name, base_url]):
        logger.warning("⚠️ Local LLM configuration incomplete - AI features may not work")
        return False

    if not _check_lm_studio_running():
        logger.debug("LM Studio process not detected; manager will attempt startup if needed")

    # Auto-start LM Studio if not running (or verify it's ready)
    try:
        lm_module = import_module("lm_studio_manager")
        manager_factory = cast(
            Callable[[Any], Any],
            getattr(lm_module, "create_manager_from_config"),
        )
        lm_manager = manager_factory(config_schema)
        success, error_msg = lm_manager.ensure_ready()

        if not success:
            logger.warning(f"⚠️ LM Studio not ready: {error_msg}")
            return False
        logger.info(f"✅ LM Studio ready; verifying model '{model_name}' is loaded")

    except Exception as e:
        logger.warning(f"⚠️ Failed to start/verify LM Studio: {e}")
        logger.warning("   Please start LM Studio manually and load a model")
        return False

    # Check if model is loaded
    try:
        client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        ai_module = import_module("ai_interface")
        validate_llm_loaded = cast(
            Callable[[Any, str], tuple[Optional[str], Optional[str]]],
            getattr(ai_module, "_validate_local_llm_model_loaded"),
        )
        actual_model_name, error_msg = validate_llm_loaded(client, model_name)

        if error_msg:
            logger.warning(f"⚠️ {error_msg}")
            logger.warning("   Please load the model in LM Studio before using AI features")
            return False
        logger.info(f"✅ Local LLM {actual_model_name} OK")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Could not validate Local LLM: {e}")
        logger.warning("   AI features may not work until model is loaded in LM Studio")
        return False


def _validate_cloud_provider(provider_name: str, api_key: Any, model: Any) -> bool:
    """Validate cloud AI provider configuration.

    Args:
        provider_name: Name of the provider (DeepSeek, Gemini, etc.)
        api_key: API key for the provider
        model: Model name

    Returns:
        True if configured, False otherwise
    """
    if api_key and model:
        logger.info(f"✅ {provider_name} configured: {model}")
        return True
    logger.warning(f"⚠️ {provider_name} configuration incomplete")
    return False


def validate_ai_provider_on_startup() -> None:
    """
    Validate AI provider configuration on startup.

    For local_llm: Checks if model is loaded in LM Studio
    For cloud providers: Just logs the configuration
    """
    try:
        config_module = import_module("config")
        config_schema = getattr(config_module, "config_schema", None)
        if not config_schema:
            return
    except ImportError:
        return

    ai_provider = config_schema.ai_provider.lower() if config_schema.ai_provider else ""

    if not ai_provider:
        logger.debug("No AI provider configured - AI features will be disabled")
        return

    logger.debug(f"Validating AI provider: {ai_provider}")

    if ai_provider == "local_llm":
        _validate_local_llm_config(config_schema)
    elif ai_provider == "deepseek":
        _validate_cloud_provider("DeepSeek", config_schema.api.deepseek_api_key, config_schema.api.deepseek_ai_model)
    elif ai_provider == "gemini":
        _validate_cloud_provider("Gemini", config_schema.api.google_api_key, config_schema.api.google_ai_model)
    elif ai_provider == "moonshot":
        _validate_cloud_provider("Moonshot", config_schema.api.moonshot_api_key, config_schema.api.moonshot_ai_model)
    elif ai_provider == "inception":
        _validate_cloud_provider("Inception", config_schema.api.inception_api_key, config_schema.api.inception_ai_model)
    elif ai_provider == "tetrate":
        _validate_cloud_provider("Tetrate", config_schema.api.tetrate_api_key, config_schema.api.tetrate_ai_model)
    else:
        logger.warning(f"⚠️ Unknown AI provider: {ai_provider}")


def _get_tree_name_from_config(config: Any) -> str:
    """Return the configured tree name from API config, or 'Unknown Tree'."""

    api_cfg = getattr(config, "api", None)
    name = getattr(api_cfg, "tree_name", None) if api_cfg is not None else None
    return name or "Unknown Tree"


def display_tree_owner(session_manager: SessionManager) -> None:
    """Display tree owner name at the end of startup checks."""
    try:
        # Get tree owner if available - this will log it via session_manager
        api_manager = get_api_manager(session_manager) if session_manager else None
        if api_manager:
            owner_name = api_manager.tree_owner_name
            if owner_name:
                print("")
                logger.info(f"Tree owner name: {owner_name}")

            # Display tree ID if available
            tree_id = api_manager.my_tree_id
            if tree_id:
                from config.config_manager import ConfigManager

                cfg = ConfigManager().get_config()
                tree_name = _get_tree_name_from_config(cfg)
                logger.info(f"Found tree ID '{tree_id}' for tree '{tree_name}'")
    except Exception:
        pass  # Silently ignore - not critical for startup


def _test_get_tree_name_from_config_uses_api_tree_name() -> None:
    """Test that _get_tree_name_from_config prefers api.tree_name and falls back safely."""

    from types import SimpleNamespace

    # When api.tree_name is set, it should be returned
    config_with_tree = SimpleNamespace(api=SimpleNamespace(tree_name="Test Tree From Config"))
    result = _get_tree_name_from_config(config_with_tree)
    assert result == "Test Tree From Config", (
        f"_get_tree_name_from_config should return api.tree_name when configured, got '{result}' instead."
    )

    # When api exists but tree_name is empty/None, fall back to 'Unknown Tree'
    config_without_name = SimpleNamespace(api=SimpleNamespace(tree_name=None))
    result_fallback = _get_tree_name_from_config(config_without_name)
    assert result_fallback == "Unknown Tree", (
        "_get_tree_name_from_config should fall back to 'Unknown Tree' when api.tree_name is not set, "
        f"got '{result_fallback}' instead."
    )


def initialize_application(config: Any, grafana_checker: Any = None) -> tuple["SessionManager", Any]:
    """Initialize application logging, configuration, and sleep prevention."""
    print("")

    _clear_startup_log_file()

    setup_logging()
    validate_action_config()

    sleep_state = _enable_system_sleep_prevention()

    print(" Checks ".center(80, "="))
    _log_sleep_prevention_status(sleep_state)

    _log_action_registry_status()
    _run_chrome_diagnostics()
    _check_grafana_status(grafana_checker)

    if config is None:
        print_config_error_message()

    session_manager = SessionManager()

    from session_utils import register_session_manager

    register_session_manager(session_manager)
    logger.debug("✅ SessionManager registered via DI container")

    return session_manager, sleep_state


def _clear_startup_log_file() -> None:
    """Clear the main application log file before logging is initialized."""
    try:
        log_dir = Path(os.getenv("LOG_DIR", "Logs"))
        if not log_dir.is_absolute():
            log_dir = (Path(__file__).parent / log_dir).resolve()
        log_file = os.getenv("LOG_FILE", "app.log")
        log_path = log_dir / log_file
        if log_path.exists():
            log_path.write_text("", encoding="utf-8")
    except Exception:
        # Silently ignore issues clearing the log file; startup should continue
        pass


def _enable_system_sleep_prevention() -> Any:
    """Enable system sleep prevention and return the resulting state object."""
    sleep_state: Any = None
    try:
        from utils import prevent_system_sleep

        sleep_state = prevent_system_sleep()
    except Exception as sleep_err:
        logger.warning(f"⚠️ System sleep prevention unavailable: {sleep_err}")
    return sleep_state


def _log_sleep_prevention_status(sleep_state: Any) -> None:
    """Log whether system sleep prevention is active based on state object."""
    if sleep_state is not None:
        logger.info("✅ System sleep prevention active")
    else:
        logger.info("⚠️ System sleep prevention inactive")


def _test_clear_startup_log_file_truncates_existing_log() -> bool:
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        log_path = log_dir / "startup.log"
        log_path.write_text("data", encoding="utf-8")
        with mock.patch.dict(os.environ, {"LOG_DIR": str(log_dir), "LOG_FILE": "startup.log"}):
            _clear_startup_log_file()

        assert not log_path.read_text(encoding="utf-8")
    return True


def _test_check_grafana_status_logs_branches() -> bool:
    ready_status = {
        "ready": True,
        "installed": True,
        "running": True,
        "sqlite_plugin": True,
        "plugins_accessible": True,
    }
    ready_checker = SimpleNamespace(check_grafana_status=lambda: ready_status)
    with mock.patch.object(logger, "info") as info_log:
        _check_grafana_status(ready_checker)
    info_log.assert_called_with("✅ Grafana ready (http://localhost:3000)")

    not_ready_status = {
        "ready": False,
        "installed": True,
        "running": False,
        "sqlite_plugin": False,
        "plugins_accessible": False,
    }
    pending_checker = SimpleNamespace(check_grafana_status=lambda: not_ready_status)
    with mock.patch.object(logger, "info") as info_log:
        _check_grafana_status(pending_checker)
    info_log.assert_called_with("⚠️  Grafana installed but not fully configured (run 'setup-grafana' from menu)")
    return True


def _test_log_sleep_prevention_status_reports_correctly() -> bool:
    active_state = object()
    with mock.patch.object(logger, "info") as info_log:
        _log_sleep_prevention_status(active_state)
        _log_sleep_prevention_status(None)

    info_log.assert_any_call("✅ System sleep prevention active")
    info_log.assert_any_call("⚠️ System sleep prevention inactive")
    return True


def _log_action_registry_status() -> None:
    """Log the number of registered actions in the action registry."""
    logger.info("✅ Action registry initialized (%d actions)", len(get_action_registry().get_all_actions()))


def _run_chrome_diagnostics() -> None:
    """Run Chrome/ChromeDriver diagnostics in silent mode and log the outcome."""
    try:
        from diagnose_chrome import run_silent_diagnostic

        success, message = run_silent_diagnostic()
        if success:
            logger.info("✅ Chrome/ChromeDriver OK")
        else:
            logger.warning(f"⚠️  Chrome diagnostic issue: {message}")
    except Exception as diag_error:
        logger.warning(f"Chrome diagnostics failed to run: {diag_error}")


def _check_grafana_status(grafana_checker: Any) -> None:
    """Check and log Grafana installation/ready status if a checker is provided."""
    if not grafana_checker:
        return

    try:
        grafana_status = grafana_checker.check_grafana_status()
        if grafana_status["ready"]:
            logger.info("✅ Grafana ready (http://localhost:3000)")
        elif grafana_status["installed"]:
            logger.info("⚠️  Grafana installed but not fully configured (run 'setup-grafana' from menu)")
        else:
            logger.info("💡 Grafana not installed (run 'setup-grafana' from menu for automated setup)")
    except Exception as grafana_error:
        logger.debug(f"Grafana check skipped: {grafana_error}")


def pre_authenticate_session() -> None:
    """
    Pre-authenticate the global session with proper browser startup.
    CRITICAL: This ensures the global session is fully authenticated and ready
    before the menu displays, preventing authentication delays during actions.
    """
    try:
        from session_utils import get_authenticated_session

        # Authenticate session - this will start browser if needed
        logger.debug("Pre-authenticating global session for immediate availability...")
        session_manager, uuid = get_authenticated_session(action_name="Main Menu Initialization", skip_csrf=False)

        # Verify session is actually ready
        if session_manager and session_manager.session_ready:
            logger.info("✅ Global session authenticated and ready")
            logger.debug(f"   UUID: {uuid}")
        else:
            logger.warning("⚠️ Session authentication incomplete - will retry during action")

    except Exception as e:
        logger.warning(f"⚠️ Session pre-authentication failed: {e}")
        logger.warning("   Session will be authenticated when first action requires it")


def pre_authenticate_ms_graph() -> None:
    """Pre-authenticate MS Graph for MS To-Do integration."""
    try:
        from ms_graph_utils import acquire_token_device_flow

        logger.debug("Attempting MS Graph authentication at startup...")
        ms_token = acquire_token_device_flow()
        if ms_token:
            logger.info("✅ MS Graph authenticated OK")
        else:
            logger.debug("MS Graph authentication skipped or failed (will retry during Action 9 if needed)")
    except Exception as e:
        logger.debug(f"MS Graph authentication failed at startup: {e}")


def cleanup_session_manager(session_manager: Optional[Any]) -> None:
    """Clean up session manager on shutdown with proper resource ordering.

    Args:
        session_manager: SessionManager instance to clean up
    """

    if session_manager is not None:
        try:
            import contextlib
            import gc
            import io

            # Suppress stderr during cleanup to hide undetected_chromedriver errors
            with contextlib.redirect_stderr(io.StringIO()):
                session_manager.close_sess(keep_db=False)

            # Clear reference immediately to prevent delayed destruction
            session_manager = None

            # Force garbage collection while process is still active
            # This ensures Chrome cleanup happens before Windows handles are invalidated
            gc.collect()

            logger.debug("✅ Session manager closed cleanly")
        except Exception as final_close_e:
            logger.debug(f"⚠️  Cleanup warning (non-critical): {final_close_e}")

    print("\nExit.")


def lifecycle_module_tests() -> bool:
    """Comprehensive tests for core/lifecycle.py."""

    suite = TestSuite("Application Lifecycle Management", "core/lifecycle.py")
    suite.start_suite()

    suite.run_test(
        "Tree name helper uses api.tree_name",
        _test_get_tree_name_from_config_uses_api_tree_name,
        test_summary="Ensure _get_tree_name_from_config reads api.tree_name and falls back to 'Unknown Tree'",
        functions_tested="_get_tree_name_from_config",
        method_description=(
            "Constructs lightweight config objects with and without api.tree_name and verifies the helper "
            "returns the configured name or the expected fallback."
        ),
        expected_outcome=(
            "When api.tree_name is set, the helper returns it; when missing, it returns 'Unknown Tree' without errors."
        ),
    )

    suite.run_test(
        "Clear startup log file truncates existing file",
        _test_clear_startup_log_file_truncates_existing_log,
        test_summary="Ensures _clear_startup_log_file wipes configured log file contents",
        functions_tested="_clear_startup_log_file",
        expected_outcome="Existing log file contents are removed without raising exceptions.",
    )

    suite.run_test(
        "Grafana status logs ready state",
        _test_check_grafana_status_logs_branches,
        test_summary="Ensures _check_grafana_status logs appropriate status messages",
        functions_tested="_check_grafana_status",
        expected_outcome="Ready and installed-but-not-ready states trigger the correct log messages.",
    )

    suite.run_test(
        "Sleep prevention status logging",
        _test_log_sleep_prevention_status_reports_correctly,
        test_summary="Ensures _log_sleep_prevention_status reports both active and inactive states",
        functions_tested="_log_sleep_prevention_status",
        expected_outcome="Logger receives ✅ message when active and ⚠️ message when inactive.",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(lifecycle_module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
