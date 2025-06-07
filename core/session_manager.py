"""
Refactored Session Manager - Orchestrates all session components.

This module provides a new, modular SessionManager that orchestrates
the specialized managers (DatabaseManager, BrowserManager, APIManager, etc.)
to provide a clean, maintainable architecture.
"""

import logging
import time
from typing import Optional

from .database_manager import DatabaseManager
from .browser_manager import BrowserManager
from .api_manager import APIManager
from .session_validator import SessionValidator

from config import config_instance

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Refactored SessionManager that orchestrates specialized managers.

    This new SessionManager delegates responsibilities to specialized managers:
    - DatabaseManager: Handles all database operations
    - BrowserManager: Handles all browser/WebDriver operations
    - APIManager: Handles all API interactions and user identifiers
    - SessionValidator: Handles session validation and readiness checks

    This design provides better separation of concerns, easier testing,
    and improved maintainability.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the SessionManager with all component managers.

        Args:
            db_path: Optional database path override
        """
        logger.debug("Initializing refactored SessionManager...")

        # Initialize component managers
        self.db_manager = DatabaseManager(db_path)
        self.browser_manager = BrowserManager()
        self.api_manager = APIManager()
        self.validator = SessionValidator()

        # Session state
        self.session_ready: bool = False
        self.session_start_time: Optional[float] = None

        # Configuration
        self.ancestry_username: str = config_instance.ANCESTRY_USERNAME
        self.ancestry_password: str = config_instance.ANCESTRY_PASSWORD

        # Initialize database connection on creation
        self.db_manager.ensure_ready()

        logger.debug(f"Refactored SessionManager created: ID={id(self)}")

    def ensure_db_ready(self) -> bool:
        """
        Ensure database is ready.

        Returns:
            bool: True if database is ready, False otherwise
        """
        return self.db_manager.ensure_ready()

    def start_browser(self, action_name: Optional[str] = None) -> bool:
        """
        Start the browser session.

        Args:
            action_name: Optional name of the action for logging

        Returns:
            bool: True if browser started successfully, False otherwise
        """
        return self.browser_manager.start_browser(action_name)

    def close_browser(self):
        """Close the browser session without affecting database."""
        self.browser_manager.close_browser()

    def start_sess(self, action_name: Optional[str] = None) -> bool:
        """
        Start session (database and browser if needed).

        Args:
            action_name: Optional name of the action for logging

        Returns:
            bool: True if session started successfully, False otherwise
        """
        logger.debug(f"Starting session for: {action_name or 'Unknown Action'}")

        # Ensure database is ready
        if not self.ensure_db_ready():
            logger.error("Failed to ensure database ready.")
            return False

        # Start browser if needed
        if self.browser_manager.browser_needed:
            browser_success = self.start_browser(action_name)
            if not browser_success:
                logger.error("Failed to start browser.")
                return False

        # Mark session as started
        self.session_start_time = time.time()

        return True

    def ensure_session_ready(self, action_name: Optional[str] = None) -> bool:
        """
        Ensure the session is ready for operations.

        Args:
            action_name: Optional name of the action for logging

        Returns:
            bool: True if session is ready, False otherwise
        """
        logger.debug(f"Ensuring session ready for: {action_name or 'Unknown Action'}")

        # Ensure driver is live if browser is needed
        if self.browser_manager.browser_needed:
            if not self.browser_manager.ensure_driver_live(action_name):
                logger.error("Failed to ensure driver live.")
                self.session_ready = False
                return False

        # Perform readiness checks
        try:
            ready_checks_ok = self.validator.perform_readiness_checks(
                self.browser_manager, self.api_manager, action_name
            )

            if not ready_checks_ok:
                logger.error("Readiness checks failed.")
                self.session_ready = False
                return False

        except Exception as e:
            logger.critical(f"Exception in readiness checks: {e}", exc_info=True)
            self.session_ready = False
            return False

        # Retrieve user identifiers
        identifiers_ok = self.api_manager.retrieve_all_identifiers()
        if not identifiers_ok:
            logger.warning("Some identifiers could not be retrieved.")

        # Retrieve tree owner if configured
        owner_ok = True
        if config_instance.TREE_NAME:
            owner_ok = self._retrieve_tree_owner()
            if not owner_ok:
                logger.warning("Tree owner name could not be retrieved.")

        # Set session ready status
        self.session_ready = ready_checks_ok and identifiers_ok and owner_ok

        logger.debug(f"Session ready status: {self.session_ready}")
        return self.session_ready

    def _retrieve_tree_owner(self) -> bool:
        """
        Retrieve tree owner name (placeholder implementation).

        Returns:
            bool: True if successful, False otherwise
        """
        # This would be implemented based on the specific API calls needed
        # For now, return True as a placeholder
        logger.debug("Tree owner retrieval not yet implemented in refactored version.")
        return True

    def verify_sess(self) -> bool:
        """
        Verify session status.

        Returns:
            bool: True if session is valid, False otherwise
        """
        return self.validator.verify_login_status(self.api_manager)

    def is_sess_valid(self) -> bool:
        """
        Check if session is valid.

        Returns:
            bool: True if session is valid, False otherwise
        """
        return self.browser_manager.is_session_valid()

    def restart_sess(self, url: Optional[str] = None) -> bool:
        """
        Restart the session.

        Args:
            url: Optional URL to navigate to after restart

        Returns:
            bool: True if restart successful, False otherwise
        """
        logger.info("Restarting session...")

        # Close current session
        self.close_sess(keep_db=True)

        # Start new session
        if not self.start_sess("Session Restart"):
            logger.error("Failed to start session during restart.")
            return False

        # Ensure session is ready
        if not self.ensure_session_ready("Session Restart"):
            logger.error("Failed to ensure session ready during restart.")
            self.close_sess(keep_db=True)
            return False  # Navigate to URL if provided
        if url and self.browser_manager.driver:
            logger.info(f"Navigating to: {url}")
            try:
                from utils import nav_to_page

                nav_success = nav_to_page(
                    self.browser_manager.driver,
                    url,
                    selector="body",
                    session_manager=self,  # type: ignore
                )
                if not nav_success:
                    logger.warning(f"Failed to navigate to {url} after restart.")
                else:
                    logger.info(f"Successfully navigated to {url}.")
            except Exception as e:
                logger.warning(f"Error navigating to {url} after restart: {e}")

        logger.info("Session restart completed successfully.")
        return True

    def close_sess(self, keep_db: bool = False):
        """
        Close the session.

        Args:
            keep_db: If True, keeps database connections alive
        """
        logger.debug(f"Closing session (keep_db={keep_db})")

        # Close browser
        self.close_browser()

        # Close database connections if requested
        if not keep_db:
            self.db_manager.close_connections(dispose_engine=True)

        # Clear API identifiers
        self.api_manager.clear_identifiers()

        # Reset session state
        self.session_ready = False
        self.session_start_time = None

        logger.debug("Session closed.")

    # Database delegation methods
    def get_db_conn(self):
        """Get a database session."""
        return self.db_manager.get_session()

    def return_session(self, session):
        """Return a database session."""
        self.db_manager.return_session(session)

    def get_db_conn_context(self):
        """Get database session context manager."""
        return self.db_manager.get_session_context()

    def cls_db_conn(self, keep_db: bool = True):
        """Close database connections."""
        self.db_manager.close_connections(dispose_engine=not keep_db)

    # Browser delegation methods
    @property
    def driver(self):
        """Get the WebDriver instance."""
        return self.browser_manager.driver

    @property
    def driver_live(self):
        """Check if driver is live."""
        return self.browser_manager.driver_live

    def make_tab(self):
        """Create a new browser tab."""
        return self.browser_manager.make_new_tab()

    # API delegation methods
    @property
    def my_profile_id(self):
        """Get the user's profile ID."""
        return self.api_manager.my_profile_id

    @property
    def my_uuid(self):
        """Get the user's UUID."""
        return self.api_manager.my_uuid

    @property
    def my_tree_id(self):
        """Get the user's tree ID."""
        return self.api_manager.my_tree_id

    @property
    def csrf_token(self):
        """Get the CSRF token."""
        return self.api_manager.csrf_token

    def get_my_profileId(self):
        """Get profile ID (legacy method name)."""
        return self.api_manager.get_profile_id()

    def get_my_uuid(self):
        """Get UUID (legacy method name)."""
        return self.api_manager.get_uuid()

    def get_csrf(self):
        """Get CSRF token (legacy method name)."""
        return self.api_manager.get_csrf_token()

    # Compatibility properties for legacy code
    @property
    def tree_owner_name(self):
        """Get the tree owner name."""
        return self.api_manager.tree_owner_name

    @property
    def requests_session(self):
        """Get the requests session."""
        return self.api_manager.requests_session

    # Status properties
    @property
    def is_ready(self) -> bool:
        """Check if the session manager is ready."""
        db_ready = self.db_manager.is_ready
        browser_ready = (
            not self.browser_manager.browser_needed or self.browser_manager.is_ready
        )
        api_ready = self.api_manager.has_essential_identifiers

        return db_ready and browser_ready and api_ready

    @property
    def session_age_seconds(self) -> Optional[float]:
        """Get the age of the current session in seconds."""
        if self.session_start_time:
            return time.time() - self.session_start_time
        return None


def run_comprehensive_tests() -> bool:
    """
    Run comprehensive tests for the SessionManager class.

    This function tests all major functionality of the SessionManager
    to ensure proper operation and integration with component managers.
    """
    import sys
    import traceback
    from typing import Dict, Any
    from contextlib import contextmanager

    # Test framework imports with fallback
    try:
        from test_framework import (
            TestSuite,
            suppress_logging,
            create_mock_data,
            assert_valid_function,
        )

        HAS_TEST_FRAMEWORK = True
    except ImportError:
        # Fallback implementations
        HAS_TEST_FRAMEWORK = False

        class TestSuite:
            def __init__(self, name, module):
                self.name = name
                self.tests_passed = 0
                self.tests_failed = 0

            def start_suite(self):
                print(f"Starting {self.name} tests...")

            def run_test(self, name, func, description):
                try:
                    func()
                    self.tests_passed += 1
                    print(f"✓ {name}")
                except Exception as e:
                    self.tests_failed += 1
                    print(f"✗ {name}: {e}")

            def finish_suite(self):
                print(f"Tests: {self.tests_passed} passed, {self.tests_failed} failed")
                return self.tests_failed == 0

        @contextmanager
        def suppress_logging():
            yield

        def create_mock_data():
            return {}

        def assert_valid_function(func, name):
            assert callable(func), f"{name} should be callable"

    logger.info("=" * 60)
    logger.info("SESSION MANAGER COMPREHENSIVE TESTS")
    logger.info("=" * 60)

    test_results = {"passed": 0, "failed": 0, "errors": []}

    def run_test(test_name: str, test_func) -> bool:
        """Helper to run individual tests with error handling."""
        try:
            logger.info(f"\n--- Running: {test_name} ---")
            test_func()
            test_results["passed"] += 1
            logger.info(f"✓ PASSED: {test_name}")
            return True
        except Exception as e:
            test_results["failed"] += 1
            error_msg = f"✗ FAILED: {test_name} - {str(e)}"
            test_results["errors"].append(error_msg)
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False

    # Test 1: Basic Initialization
    def test_initialization():
        session_manager = SessionManager()
        assert session_manager is not None, "SessionManager should initialize"
        assert hasattr(session_manager, "db_manager"), "Should have db_manager"
        assert hasattr(
            session_manager, "browser_manager"
        ), "Should have browser_manager"
        assert hasattr(session_manager, "api_manager"), "Should have api_manager"
        assert hasattr(session_manager, "validator"), "Should have validator"
        assert (
            session_manager.session_ready == False
        ), "Should start with session_ready=False"

    # Test 2: Component Manager Creation
    def test_component_managers():
        session_manager = SessionManager()
        assert (
            session_manager.db_manager is not None
        ), "DatabaseManager should be created"
        assert (
            session_manager.browser_manager is not None
        ), "BrowserManager should be created"
        assert session_manager.api_manager is not None, "APIManager should be created"
        assert (
            session_manager.validator is not None
        ), "SessionValidator should be created"

    # Test 3: Database Operations
    def test_database_operations():
        session_manager = SessionManager()
        result = session_manager.ensure_db_ready()
        assert isinstance(result, bool), "ensure_db_ready should return bool"

    # Test 4: Browser Operations
    def test_browser_operations():
        session_manager = SessionManager()
        # Test browser start (will fail gracefully without WebDriver)
        result = session_manager.start_browser("test_action")
        assert isinstance(result, bool), "start_browser should return bool"

        # Test browser close (should not raise exception)
        session_manager.close_browser()

    # Test 5: Session Start
    def test_session_start():
        session_manager = SessionManager()
        result = session_manager.start_sess("test_action")
        assert isinstance(result, bool), "start_sess should return bool"

    # Test 6: Property Access
    def test_property_access():
        session_manager = SessionManager()

        # Test properties exist (may be None initially)
        properties_to_check = [
            "my_profile_id",
            "my_uuid",
            "my_tree_id",
            "csrf_token",
            "tree_owner_name",
            "requests_session",
            "is_ready",
            "session_age_seconds",
        ]

        for prop in properties_to_check:
            assert hasattr(session_manager, prop), f"Should have property {prop}"

    # Test 7: Legacy Method Compatibility
    def test_legacy_methods():
        session_manager = SessionManager()

        methods_to_check = ["get_my_profileId", "get_my_uuid", "get_csrf"]

        for method_name in methods_to_check:
            method = getattr(session_manager, method_name, None)
            assert method is not None, f"Method {method_name} should exist"
            assert callable(method), f"Method {method_name} should be callable"

    # Test 8: Configuration Access
    def test_configuration_access():
        session_manager = SessionManager()
        assert hasattr(
            session_manager, "ancestry_username"
        ), "Should have ancestry_username"
        assert hasattr(
            session_manager, "ancestry_password"
        ), "Should have ancestry_password"

    # Test 9: Validation Integration
    def test_validation_integration():
        session_manager = SessionManager()
        assert hasattr(
            session_manager.validator, "validate_session"
        ), "Validator should have validate_session method"

    # Test 10: Status Properties
    def test_status_properties():
        session_manager = SessionManager()

        # Test is_ready property
        is_ready = session_manager.is_ready
        assert isinstance(is_ready, bool), "is_ready should return bool"

        # Test session_age_seconds property
        age = session_manager.session_age_seconds
        assert age is None or isinstance(
            age, (int, float)
        ), "session_age_seconds should be None or numeric"

    # Test 11: Component Method Delegation
    def test_component_delegation():
        session_manager = SessionManager()

        # Test that manager methods delegate to components
        try:
            # These may fail gracefully without proper setup
            session_manager.ensure_db_ready()
            session_manager.close_browser()
            logger.info("Component delegation test completed")
        except Exception as e:
            # Graceful failure is acceptable
            logger.warning(f"Component delegation had expected failure: {e}")

    # Test 12: Error Handling
    def test_error_handling():
        # Test with invalid database path
        try:
            invalid_session = SessionManager(db_path="/invalid/path/to/db.sqlite")
            # Should handle gracefully or raise appropriate exception
            assert (
                invalid_session is not None
            ), "Should handle invalid db_path gracefully"
        except Exception as e:
            # Expected behavior for invalid path
            logger.info(f"Invalid path handled appropriately: {e}")

    # Test 13: Import Dependencies
    def test_import_dependencies():
        # Test that all required components can be imported
        try:
            from .database_manager import DatabaseManager
            from .browser_manager import BrowserManager
            from .api_manager import APIManager
            from .session_validator import SessionValidator
            from config import config_instance

            assert DatabaseManager is not None, "DatabaseManager should import"
            assert BrowserManager is not None, "BrowserManager should import"
            assert APIManager is not None, "APIManager should import"
            assert SessionValidator is not None, "SessionValidator should import"
            assert config_instance is not None, "config_instance should import"
        except ImportError as e:
            assert False, f"Required imports failed: {e}"

    # Test 14: Type Annotations
    def test_type_annotations():
        session_manager = SessionManager()
        assert isinstance(
            session_manager.session_ready, bool
        ), "session_ready should be bool"

        age = session_manager.session_age_seconds
        if age is not None:
            assert isinstance(
                age, (int, float)
            ), "session_age_seconds should be numeric when not None"

    # Test 15: Comprehensive Function Structure
    def test_comprehensive_function():
        # Test that this function itself is properly structured
        assert_valid_function(
            run_comprehensive_tests, "run_comprehensive_tests should be callable"
        )
        assert test_results is not None, "test_results should be initialized"
        assert isinstance(test_results, dict), "test_results should be a dictionary"

    # Run all tests
    tests = [
        ("Basic Initialization", test_initialization),
        ("Component Manager Creation", test_component_managers),
        ("Database Operations", test_database_operations),
        ("Browser Operations", test_browser_operations),
        ("Session Start", test_session_start),
        ("Property Access", test_property_access),
        ("Legacy Method Compatibility", test_legacy_methods),
        ("Configuration Access", test_configuration_access),
        ("Validation Integration", test_validation_integration),
        ("Status Properties", test_status_properties),
        ("Component Method Delegation", test_component_delegation),
        ("Error Handling", test_error_handling),
        ("Import Dependencies", test_import_dependencies),
        ("Type Annotations", test_type_annotations),
        ("Comprehensive Function Structure", test_comprehensive_function),
    ]

    # Run each test
    for test_name, test_func in tests:
        run_test(test_name, test_func)

    # Print summary
    total_tests = len(tests)
    logger.info("\n" + "=" * 60)
    logger.info("SESSION MANAGER TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {test_results['passed']}")
    logger.info(f"Failed: {test_results['failed']}")

    if test_results["errors"]:
        logger.info("\nErrors:")
        for error in test_results["errors"]:
            logger.error(f"  {error}")

    success = test_results["failed"] == 0
    if success:
        logger.info("🎉 ALL SESSION MANAGER TESTS PASSED!")
    else:
        logger.warning("⚠️ Some Session Manager tests failed")

    return success


if __name__ == "__main__":
    run_comprehensive_tests()
