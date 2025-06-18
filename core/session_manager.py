"""
Refactored Session Manager - Orchestrates all session components.

This module provides a new, modular SessionManager that orchestrates
the specialized managers (DatabaseManager, BrowserManager, APIManager, etc.)
to provide a clean, maintainable architecture.
"""

import logging
import time
from typing import Optional

from core.database_manager import DatabaseManager
from core.browser_manager import BrowserManager
from core.api_manager import APIManager
from core.session_validator import SessionValidator

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
        self.db_manager.close_connections(
            dispose_engine=not keep_db
        )  # Browser delegation methods

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
        return self.browser_manager.create_new_tab()

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

    # Status properties    @property
    def is_ready(self) -> bool:
        """Check if the session manager is ready."""
        db_ready = self.db_manager.is_ready
        browser_ready = (
            not self.browser_manager.browser_needed or self.browser_manager.driver_live
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
    Comprehensive test suite for session_manager.py with real functionality testing.
    Tests initialization, core functionality, edge cases, integration, performance, and error handling.
    """
    from test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite(
            "Session Manager & Component Coordination", "session_manager.py"
        )
        suite.start_suite()

        # INITIALIZATION TESTS
        def test_session_manager_initialization():
            """Test SessionManager initialization and component creation."""
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
            return True

        suite.run_test(
            "SessionManager Initialization",
            test_session_manager_initialization,
            "SessionManager creates successfully with all component managers (database, browser, API, validator)",
            "Instantiate SessionManager and verify all component managers are created and session_ready starts as False",
            "Test SessionManager initialization and component manager creation",
        )

        def test_component_manager_availability():
            """Test that all component managers are properly created."""
            session_manager = SessionManager()
            assert (
                session_manager.db_manager is not None
            ), "DatabaseManager should be created"
            assert (
                session_manager.browser_manager is not None
            ), "BrowserManager should be created"
            assert (
                session_manager.api_manager is not None
            ), "APIManager should be created"
            assert (
                session_manager.validator is not None
            ), "SessionValidator should be created"
            return True

        suite.run_test(
            "Component Manager Availability",
            test_component_manager_availability,
            "All component managers (database, browser, API, validator) are properly instantiated and accessible",
            "Verify that all component managers exist and are not None after SessionManager creation",
            "Test component manager instantiation and availability",
        )

        # CORE FUNCTIONALITY TESTS
        def test_database_operations():
            """Test database operation delegation."""
            session_manager = SessionManager()
            result = session_manager.ensure_db_ready()
            assert isinstance(result, bool), "ensure_db_ready should return bool"
            return True

        suite.run_test(
            "Database Operations",
            test_database_operations,
            "Database operations are properly delegated to DatabaseManager and return expected types",
            "Call ensure_db_ready() and verify it returns a boolean result",
            "Test database operation delegation and return type validation",
        )

        def test_browser_operations():
            """Test browser operation delegation."""
            session_manager = SessionManager()
            # Test browser start (will fail gracefully without WebDriver)
            result = session_manager.start_browser("test_action")
            assert isinstance(result, bool), "start_browser should return bool"

            # Test browser close (should not raise exception)
            session_manager.close_browser()
            return True

        suite.run_test(
            "Browser Operations",
            test_browser_operations,
            "Browser operations are properly delegated to BrowserManager without errors",
            "Call start_browser() and close_browser() and verify proper delegation and error handling",
            "Test browser operation delegation and graceful error handling",
        )

        # EDGE CASES TESTS
        def test_property_access():
            """Test property access and delegation."""
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
                # Should not raise AttributeError
                assert hasattr(session_manager, prop), f"Property {prop} should exist"
            return True

        suite.run_test(
            "Property Access",
            test_property_access,
            "All expected properties are accessible without AttributeError",
            "Access various session properties and verify they exist (even if None)",
            "Test property access and delegation to component managers",
        )

        # INTEGRATION TESTS
        def test_component_delegation():
            """Test that method calls are properly delegated to component managers."""
            session_manager = SessionManager()

            # Test database delegation
            db_result = session_manager.ensure_db_ready()
            assert isinstance(db_result, bool), "Database delegation should work"

            # Test browser delegation
            browser_result = session_manager.start_browser("test")
            assert isinstance(browser_result, bool), "Browser delegation should work"
            return True

        suite.run_test(
            "Component Method Delegation",
            test_component_delegation,
            "Method calls are properly delegated to appropriate component managers",
            "Call methods that should be delegated and verify they execute without errors",
            "Test delegation pattern between SessionManager and component managers",
        )

        # PERFORMANCE TESTS
        def test_initialization_performance():
            """Test SessionManager initialization performance."""
            import time

            start_time = time.time()
            for _ in range(3):  # Fewer iterations due to heavier initialization
                session_manager = SessionManager()
            end_time = time.time()

            total_time = end_time - start_time
            assert (
                total_time < 15.0
            ), f"3 initializations took {total_time:.3f}s, should be under 15s"
            return True

        suite.run_test(
            "Initialization Performance",
            test_initialization_performance,
            "3 SessionManager initializations complete in under 15 seconds",
            "Create 3 SessionManager instances and measure total time",
            "Test performance of SessionManager initialization with all component managers",
        )

        # ERROR HANDLING TESTS
        def test_error_handling():
            """Test graceful error handling for invalid operations."""
            session_manager = SessionManager()
            try:
                # These operations should handle errors gracefully
                session_manager.ensure_db_ready()
                session_manager.start_browser("test_action")
                session_manager.close_browser()

                # Property access should not raise errors
                _ = session_manager.session_ready
                _ = session_manager.is_ready
            except Exception as e:
                assert False, f"SessionManager should handle operations gracefully: {e}"
            return True

        suite.run_test(
            "Error Handling",
            test_error_handling,
            "SessionManager handles various operations gracefully without raising exceptions",
            "Perform various operations and property access and verify no exceptions are raised",
            "Test error handling and graceful degradation for session operations",
        )

        return suite.finish_suite()


if __name__ == "__main__":
    run_comprehensive_tests()
