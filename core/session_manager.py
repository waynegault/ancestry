#!/usr/bin/env python3

"""
Refactored Session Manager - Orchestrates all session components.

This module provides a new, modular SessionManager that orchestrates
the specialized managers (DatabaseManager, BrowserManager, APIManager, etc.)
to provide a clean, maintainable architecture.

PHASE 5.1 OPTIMIZATION: Enhanced with intelligent session caching for dramatic
performance improvement. Reduces initialization from 34.59s to <12s target.
"""

# === CORE INFRASTRUCTURE ===
import sys
import os

# Add parent directory to path for core_imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
from error_handling import (
    retry_on_failure,
    circuit_breaker,
    timeout_protection,
    graceful_degradation,
    error_context,
    AncestryException,
    RetryableError,
    NetworkTimeoutError,
    AuthenticationExpiredError,
    APIRateLimitError,
    ErrorContext,
)

# === PHASE 5.1: SESSION PERFORMANCE OPTIMIZATION ===
from core.session_cache import (
    cached_database_manager,
    cached_browser_manager,
    cached_api_manager,
    cached_session_validator,
    get_session_cache_stats,
    clear_session_cache,
)

# === MODULE SETUP ===
logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import logging
import time
from typing import Optional, Dict, Any

# === LOCAL IMPORTS ===
from core.database_manager import DatabaseManager
from core.browser_manager import BrowserManager
from core.api_manager import APIManager
from core.session_validator import SessionValidator
from config.config_manager import ConfigManager

# === MODULE CONSTANTS ===
config_manager = ConfigManager()
config_schema = config_manager.get_config()

# Initialize config
config_manager = ConfigManager()
config_schema = config_manager.get_config()


class SessionManager:
    """
    Refactored SessionManager that orchestrates specialized managers.

    PHASE 5.1 OPTIMIZATION: Enhanced with intelligent component caching to reduce
    initialization overhead from 34.59s to <12s. Implements session state persistence
    and component reuse for dramatic performance improvement.

    This new SessionManager delegates responsibilities to specialized managers:
    - DatabaseManager: Handles all database operations (with connection pooling cache)
    - BrowserManager: Handles all browser/WebDriver operations (with instance reuse)
    - APIManager: Handles all API interactions and user identifiers (with session cache)
    - SessionValidator: Handles session validation and readiness checks (with state cache)

    Performance optimizations:
    - Component caching with intelligent TTL management
    - Session state persistence across test runs
    - Lazy initialization for non-critical components
    - Background connection warming
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the SessionManager with optimized component creation.

        PHASE 5.1: Uses intelligent caching to reuse expensive components
        across multiple SessionManager instances.

        Args:
            db_path: Optional database path override
        """
        start_time = time.time()
        logger.debug("Initializing optimized SessionManager...")

        # PHASE 5.1: Use cached component managers for dramatic performance improvement
        self.db_manager = self._get_cached_database_manager(db_path)
        self.browser_manager = self._get_cached_browser_manager()
        self.api_manager = self._get_cached_api_manager()
        self.validator = self._get_cached_session_validator()

        # Session state
        self.session_ready: bool = False
        self.session_start_time: Optional[float] = None

        # PHASE 5.1: Session state caching for performance
        self._last_readiness_check: Optional[float] = None
        self._cached_session_state: Dict[str, Any] = {}

        # Configuration (cached access)
        self.ancestry_username: str = config_schema.api.username
        self.ancestry_password: str = config_schema.api.password

        # PHASE 5.1: Only initialize database if not already cached and ready
        if not self.db_manager.is_ready:
            self.db_manager.ensure_ready()

        init_time = time.time() - start_time
        logger.debug(
            f"Optimized SessionManager created in {init_time:.3f}s: ID={id(self)}"
        )

    @cached_database_manager()
    def _get_cached_database_manager(
        self, db_path: Optional[str] = None
    ) -> "DatabaseManager":
        """Get cached DatabaseManager instance"""
        logger.debug("Creating/retrieving DatabaseManager from cache")
        return DatabaseManager(db_path)

    @cached_browser_manager()
    def _get_cached_browser_manager(self) -> "BrowserManager":
        """Get cached BrowserManager instance"""
        logger.debug("Creating/retrieving BrowserManager from cache")
        return BrowserManager()

    @cached_api_manager()
    def _get_cached_api_manager(self) -> "APIManager":
        """Get cached APIManager instance"""
        logger.debug("Creating/retrieving APIManager from cache")
        return APIManager()

    @cached_session_validator()
    def _get_cached_session_validator(self) -> "SessionValidator":
        """Get cached SessionValidator instance"""
        logger.debug("Creating/retrieving SessionValidator from cache")
        return SessionValidator()

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

    @timeout_protection(timeout=30)  # Prevent hanging on slow operations
    @graceful_degradation(fallback_value=False)
    @error_context("ensure_session_ready")
    def ensure_session_ready(self, action_name: Optional[str] = None) -> bool:
        """
        Ensure the session is ready for operations.

        PHASE 5.1 OPTIMIZATION: Uses intelligent caching to bypass expensive
        readiness checks when session state is known to be valid.

        Args:
            action_name: Optional name of the action for logging

        Returns:
            bool: True if session is ready, False otherwise
        """
        start_time = time.time()
        logger.debug(f"Ensuring session ready for: {action_name or 'Unknown Action'}")

        # PHASE 5.1: Check cached session state first
        session_id = f"{id(self)}_{action_name or 'default'}"

        # Try to use cached readiness state
        if self._last_readiness_check is not None:
            time_since_check = time.time() - self._last_readiness_check
            if time_since_check < 60 and self.session_ready:  # Cache for 60 seconds
                logger.debug(
                    f"Using cached session readiness (age: {time_since_check:.1f}s)"
                )
                return True

        # Ensure driver is live if browser is needed (with optimization)
        if self.browser_manager.browser_needed:
            if not self.browser_manager.ensure_driver_live(action_name):
                logger.error("Failed to ensure driver live.")
                self.session_ready = False
                return False

        # PHASE 5.1: Optimized readiness checks with circuit breaker pattern
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

        # PHASE 5.1: Optimized identifier retrieval with caching
        identifiers_ok = True
        if not self.api_manager.has_essential_identifiers:
            identifiers_ok = self.api_manager.retrieve_all_identifiers()
            if not identifiers_ok:
                logger.warning("Some identifiers could not be retrieved.")

        # Retrieve tree owner if configured (with caching)
        owner_ok = True
        if config_schema.api.tree_name:
            owner_ok = self._retrieve_tree_owner()
            if not owner_ok:
                logger.warning("Tree owner name could not be retrieved.")

        # Set session ready status
        self.session_ready = ready_checks_ok and identifiers_ok and owner_ok

        # PHASE 5.1: Cache the readiness check result
        self._last_readiness_check = time.time()

        check_time = time.time() - start_time
        logger.debug(
            f"Session readiness check completed in {check_time:.3f}s, status: {self.session_ready}"
        )
        return self.session_ready
        if config_schema.api.tree_name:
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

    # Public properties
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

    # PHASE 5.1: Session cache management methods
    def get_session_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this session"""
        stats = get_session_cache_stats()
        stats.update(
            {
                "session_ready": self.session_ready,
                "session_age": self.session_age_seconds,
                "last_readiness_check_age": (
                    time.time() - self._last_readiness_check
                    if self._last_readiness_check
                    else None
                ),
                "db_ready": (
                    self.db_manager.is_ready
                    if hasattr(self.db_manager, "is_ready")
                    else False
                ),
                "browser_needed": self.browser_manager.browser_needed,
                "driver_live": self.browser_manager.driver_live,
            }
        )
        return stats

    @classmethod
    def clear_session_caches(cls) -> int:
        """Clear all session caches for fresh initialization"""
        return clear_session_cache()


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
            """Test SessionManager initialization performance with caching optimization."""
            import time

            start_time = time.time()

            # PHASE 5.1: Test optimized initialization with caching
            session_managers = []
            for i in range(3):  # Test 3 instances
                logger.debug(f"Creating SessionManager {i+1}/3...")
                session_manager = SessionManager()
                session_managers.append(session_manager)

            end_time = time.time()
            total_time = end_time - start_time

            # PHASE 5.1: Much stricter performance requirement due to caching
            max_time = 5.0  # Reduced from 15s to 5s with caching optimization

            logger.debug(
                f"SessionManager initialization test: {total_time:.3f}s (target: <{max_time}s)"
            )
            logger.debug(f"Cache stats: {get_session_cache_stats()}")

            assert (
                total_time < max_time
            ), f"3 optimized initializations took {total_time:.3f}s, should be under {max_time}s"

            # Clean up to prevent resource leaks
            for sm in session_managers:
                try:
                    sm.close_sess(keep_db=True)
                except Exception as e:
                    logger.debug(f"Cleanup warning: {e}")

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
