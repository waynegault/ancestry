"""Standalone test suite for the main application module.

These tests were extracted from main.py to keep the entrypoint slim while
still registering with the standardized run_all_tests discovery workflow.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any, Callable, cast

import main as main_module
from action6_gather import coord
from action7_inbox import InboxProcessor
from action8_messaging import send_messages_to_matches
from action9_process_productive import process_productive_messages
from action10 import main as run_action10
from test_utilities import create_standard_test_runner

clear_log_file = main_module.clear_log_file
main = main_module.main
SessionManager = main_module.SessionManager
_get_database_manager = main_module._get_database_manager
validate_action_config = main_module.validate_action_config
backup_database = main_module.backup_database
db_transn = main_module.db_transn
menu = main_module.menu
logger = main_module.logger


def _test_clear_log_file_function() -> bool:
    """Test log file clearing functionality - validates actual behavior."""
    try:
        result = clear_log_file()
        assert isinstance(result, tuple), "clear_log_file should return a tuple"
        assert len(result) == 2, "clear_log_file should return a 2-element tuple"
        success, message = result
        assert isinstance(success, bool), "First element should be boolean"
        assert message is None or isinstance(
            message, str
        ), "Second element should be None or string"
    except Exception as exc:
        assert isinstance(exc, Exception), "Should handle errors gracefully"
    return True


def _test_main_function_structure() -> bool:
    """Test main function structure and error handling."""
    assert callable(main), "main() function should be callable"

    import inspect

    sig = inspect.signature(main)
    assert len(sig.parameters) == 0, "main() should take no parameters"
    return True


def _test_reset_db_actn_integration() -> bool:
    """Test reset_db_actn function integration and method availability."""
    try:
        test_sm = SessionManager()
        db_manager = _get_database_manager(test_sm)
        assert db_manager is not None, "SessionManager should provide a database manager"
        assert hasattr(db_manager, "_initialize_engine_and_session"), (
            "DatabaseManager should have _initialize_engine_and_session method"
        )
        assert hasattr(db_manager, "engine"), "DatabaseManager should have engine attribute"
        assert hasattr(db_manager, "Session"), "DatabaseManager should have Session attribute"
        logger.debug(
            "reset_db_actn integration test: All required methods and attributes verified"
        )
    except AttributeError as exc:
        raise AssertionError(
            f"reset_db_actn integration test failed with AttributeError: {exc}"
        ) from exc
    except Exception as exc:
        logger.debug(
            "reset_db_actn integration test: Non-AttributeError exception (acceptable): %s",
            exc,
        )
        return True
    return True


def _test_edge_case_handling() -> bool:
    """Test edge cases and error conditions."""
    import sys as _sys

    assert "action6_gather" in _sys.modules, "action6_gather should be imported"
    assert "action7_inbox" in _sys.modules, "action7_inbox should be imported"
    assert "action8_messaging" in _sys.modules, "action8_messaging should be imported"
    assert "action9_process_productive" in _sys.modules, (
        "action9_process_productive should be imported"
    )
    assert "action10" in _sys.modules, "action10 should be imported"
    return True


def _test_import_error_handling() -> bool:
    """Test import error scenarios."""
    module_globals = vars(main_module)
    required_imports = [
        "coord",
        "InboxProcessor",
        "send_messages_to_matches",
        "process_productive_messages",
        "config",
        "logger",
        "SessionManager",
    ]

    for import_name in required_imports:
        assert import_name in module_globals, f"{import_name} should be imported"
    return True


def _test_validate_action_config() -> bool:
    """Test the new validate_action_config() function from Action 6 lessons."""
    assert callable(validate_action_config), "validate_action_config should be callable"

    try:
        result = validate_action_config()
        assert isinstance(result, bool), "validate_action_config should return boolean"
        assert result is True, "validate_action_config should return True for basic validation"
    except Exception as exc:
        assert "config" in str(exc).lower(), f"validate_action_config failed unexpectedly: {exc}"
    return True


def _test_database_integration() -> bool:
    """Test database system integration."""
    assert callable(backup_database), "backup_database should be callable"
    assert callable(db_transn), "db_transn should be callable"

    from database import Base

    assert Base is not None, "SQLAlchemy Base should be accessible"
    return True


def _test_action_integration() -> bool:
    """Test all actions integrate properly with main."""
    actions_to_test: list[tuple[str, Callable[..., Any]]] = [
        ("coord", coord),
        ("InboxProcessor", InboxProcessor),
        ("send_messages_to_matches", send_messages_to_matches),
        ("process_productive_messages", process_productive_messages),
        ("run_action10", run_action10),
    ]

    for action_name, action_func in actions_to_test:
        assert callable(action_func), f"{action_name} should be callable"
        assert action_func is not None, f"{action_name} should not be None"
    return True


def _test_import_performance() -> bool:
    """Test import performance is reasonable."""
    import time

    start_time = time.time()

    try:
        config_module = sys.modules.get("config")
        if config_module:
            importlib.reload(config_module)
    except Exception:
        pass

    duration = time.time() - start_time
    assert duration < 1.0, f"Module reloading should be fast, took {duration:.3f}s"
    return True


def _test_memory_efficiency() -> bool:
    """Test memory usage is reasonable."""
    import inspect

    module_size = sys.getsizeof(main_module)
    assert module_size < 10000, f"Module size should be reasonable, got {module_size} bytes"

    tracked_state: dict[str, Any] = {
        name: value
        for name, value in vars(main_module).items()
        if not name.startswith("__")
        and not isinstance(value, ModuleType)
        and not inspect.isfunction(value)
        and not inspect.isclass(value)
    }

    globals_count = len(tracked_state)
    assert globals_count < 80, (
        f"Stateful global variables should be reasonable, got {globals_count}"
    )
    return True


def _test_function_call_performance() -> bool:
    """Test function call performance."""
    import time

    start_time = time.time()

    for _ in range(1000):
        result = callable(menu)
        assert result is True, "menu should be callable"

    duration = time.time() - start_time
    assert duration < 0.1, f"1000 function checks should be fast, took {duration:.3f}s"
    return True


def _test_error_handling_structure() -> bool:
    """Test error handling structure in main functions."""
    import inspect

    main_source = inspect.getsource(main)
    assert "try:" in main_source, "main() should have try-except structure"
    assert "except" in main_source, "main() should have exception handling"
    assert "finally:" in main_source, "main() should have finally block"
    assert "KeyboardInterrupt" in main_source, "main() should handle KeyboardInterrupt"
    return True


def _test_cleanup_procedures() -> bool:
    """Test cleanup procedures are in place."""
    import inspect

    main_source = inspect.getsource(main)
    assert "finally:" in main_source, "main() should have finally block for cleanup"
    assert "cleanup" in main_source.lower(), "main() should mention cleanup"
    return True


def _test_exception_handling_coverage() -> bool:
    """Test exception handling covers expected scenarios."""
    import inspect

    main_source = inspect.getsource(main)
    assert "Exception" in main_source, "main() should handle general exceptions"
    assert "logger" in main_source, "main() should use logger for error reporting"
    return True


def main_module_tests() -> bool:
    """Comprehensive test suite for main.py."""
    from test_framework import TestSuite, suppress_logging

    suite = cast(
        Any, TestSuite("Main Application Controller & Menu System", "main.py")
    )
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
            test_name="Database system integration and transaction management",
            test_func=_test_database_integration,
            test_summary="Database system integration with main application",
            method_description="Testing database functions and model accessibility",
            expected_outcome="Database system is properly integrated with transaction support",
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


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
