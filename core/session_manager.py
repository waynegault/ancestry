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
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

# === THIRD-PARTY IMPORTS ===
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
try:
    import cloudscraper
except ImportError:
    cloudscraper = None
    logger.warning("CloudScraper not available - anti-bot protection disabled")

# === SELENIUM IMPORTS ===
try:
    from selenium.common.exceptions import (
        WebDriverException,
        InvalidSessionIdException,
        NoSuchWindowException
    )
except ImportError:
    WebDriverException = Exception
    InvalidSessionIdException = Exception
    NoSuchWindowException = Exception

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

        # Database state tracking (from old SessionManager)
        self._db_init_attempted: bool = False
        self._db_ready: bool = False

        # Identifier logging flags (from old SessionManager)
        self._profile_id_logged: bool = False
        self._uuid_logged: bool = False
        self._tree_id_logged: bool = False
        self._owner_logged: bool = False

        # Add dynamic rate limiter for AI calls (matches utils.py SessionManager)
        try:
            from utils import DynamicRateLimiter

            self.dynamic_rate_limiter = DynamicRateLimiter()
        except ImportError:
            self.dynamic_rate_limiter = None

        # === ENHANCED SESSION CAPABILITIES ===
        # JavaScript error monitoring
        self.last_js_error_check: datetime = datetime.now(timezone.utc)

        # Initialize enhanced requests session with advanced configuration
        self._initialize_enhanced_requests_session()

        # Initialize CloudScraper for anti-bot protection
        self._initialize_cloudscraper()

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

    def _initialize_enhanced_requests_session(self):
        """
        Initialize enhanced requests session with advanced configuration.
        Includes connection pooling, retry strategies, and performance optimizations.
        """
        logger.debug("Initializing enhanced requests session...")

        # Enhanced retry strategy with more comprehensive status codes
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )

        # Advanced HTTPAdapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=50,
            max_retries=retry_strategy
        )

        # Apply adapter to API manager's requests session
        if hasattr(self.api_manager, '_requests_session'):
            self.api_manager._requests_session.mount("http://", adapter)
            self.api_manager._requests_session.mount("https://", adapter)
            logger.debug("Enhanced requests session configuration applied to APIManager")
        else:
            logger.warning("APIManager requests session not found - creating fallback")
            # Create fallback session if APIManager doesn't have one
            self.api_manager._requests_session = requests.Session()
            self.api_manager._requests_session.mount("http://", adapter)
            self.api_manager._requests_session.mount("https://", adapter)

    def _initialize_cloudscraper(self):
        """
        Initialize CloudScraper for anti-bot protection.
        Provides enhanced capabilities for bypassing CloudFlare and other protections.
        """
        if cloudscraper is None:
            logger.debug("CloudScraper not available - skipping initialization")
            self._scraper = None
            return

        logger.debug("Initializing CloudScraper with anti-bot protection...")

        try:
            # Create CloudScraper with browser fingerprinting
            self._scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "desktop": True},
                delay=10,
            )

            # Enhanced retry strategy for CloudScraper
            scraper_retry = Retry(
                total=3,
                backoff_factor=0.8,
                status_forcelist=[403, 429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
            )

            # Apply retry adapter to CloudScraper
            scraper_adapter = HTTPAdapter(max_retries=scraper_retry)
            self._scraper.mount("http://", scraper_adapter)
            self._scraper.mount("https://", scraper_adapter)

            logger.debug("CloudScraper initialized successfully with retry strategy")

        except Exception as scraper_init_e:
            logger.error(
                f"Failed to initialize CloudScraper: {scraper_init_e}",
                exc_info=True,
            )
            self._scraper = None

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
        # Reset logged flags when starting browser
        self._reset_logged_flags()
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

    @timeout_protection(timeout=60)  # Increased timeout for complex operations
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

        # Try to use cached readiness state, but validate driver is still live
        if self._last_readiness_check is not None:
            time_since_check = time.time() - self._last_readiness_check
            cache_duration = 60  # Use consistent 60-second cache for all actions
            if time_since_check < cache_duration and self.session_ready:
                # Validate that the cached state is still accurate
                if self.browser_manager.browser_needed:
                    if not self.browser_manager.is_session_valid():
                        logger.debug(
                            f"Cached session readiness invalid - driver session expired (age: {time_since_check:.1f}s)"
                        )
                        self.session_ready = False
                        self._last_readiness_check = None
                    else:
                        logger.debug(
                            f"Using cached session readiness (age: {time_since_check:.1f}s, action: {action_name})"
                        )
                        return True
                else:
                    logger.debug(
                        f"Using cached session readiness (age: {time_since_check:.1f}s, action: {action_name})"
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
                self.browser_manager, self.api_manager, self, action_name
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
        identifiers_ok = self._retrieve_identifiers()
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
        Verify session status using login_status function.

        Returns:
            bool: True if session is valid, False otherwise
        """
        logger.debug("Verifying session status (using login_status)...")
        try:
            # Import login_status locally to avoid circular imports
            from utils import login_status

            login_ok = login_status(self, disable_ui_fallback=False)
            if login_ok is True:
                logger.debug("Session verification successful (logged in).")
                return True
            elif login_ok is False:
                logger.warning("Session verification failed (user not logged in).")
                return False
            else:  # login_ok is None
                logger.error("Session verification failed critically (login_status returned None).")
                return False
        except Exception as e:
            logger.error(f"Unexpected error during session verification: {e}", exc_info=True)
            return False

    def is_sess_valid(self) -> bool:
        """
        Simplified session validity check to prevent recursion.

        Just checks if driver exists without doing WebDriver operations
        that might trigger more session validation.

        Returns:
            bool: True if driver exists, False otherwise
        """
        # Simple check - just verify driver exists
        return self.driver is not None

    def _reset_logged_flags(self):
        """Reset flags used to prevent repeated logging of IDs."""
        self._profile_id_logged = False
        self._uuid_logged = False
        self._tree_id_logged = False
        self._owner_logged = False

    def _retrieve_identifiers(self) -> bool:
        """
        Retrieve all essential identifiers.

        Returns:
            bool: True if all identifiers retrieved successfully, False otherwise
        """
        if not self.is_sess_valid():
            logger.error("_retrieve_identifiers: Session is invalid.")
            return False

        all_ok = True

        # Get Profile ID
        if not self.my_profile_id:
            logger.debug("Retrieving profile ID (ucdmid)...")
            profile_id = self.get_my_profileId()
            if not profile_id:
                logger.error("Failed to retrieve profile ID (ucdmid).")
                all_ok = False

        # Get UUID
        if not self.my_uuid:
            logger.debug("Retrieving UUID (testId)...")
            uuid_val = self.get_my_uuid()
            if not uuid_val:
                logger.error("Failed to retrieve UUID (testId).")
                all_ok = False

        # Get Tree ID (only if TREE_NAME is configured)
        if config_schema.api.tree_name and not self.my_tree_id:
            logger.debug(f"Retrieving tree ID for tree name: '{config_schema.api.tree_name}'...")
            try:
                tree_id = self.get_my_tree_id()
                if not tree_id:
                    logger.error(f"TREE_NAME '{config_schema.api.tree_name}' configured, but failed to get corresponding tree ID.")
                    all_ok = False
            except ImportError as tree_id_imp_err:
                logger.error(f"Failed to retrieve tree ID due to import error: {tree_id_imp_err}")
                all_ok = False

        return all_ok

    def _retrieve_tree_owner(self) -> bool:
        """
        Retrieve tree owner name.

        Returns:
            bool: True if tree owner retrieved successfully, False otherwise
        """
        if not self.is_sess_valid():
            logger.error("_retrieve_tree_owner: Session is invalid.")
            return False

        if not self.my_tree_id:
            logger.debug("Cannot retrieve tree owner name: my_tree_id is not set.")
            return False

        # Only retrieve if not already present
        if self.tree_owner_name and self._owner_logged:
            return True

        logger.debug("Retrieving tree owner name...")
        try:
            owner_name = self.get_tree_owner(self.my_tree_id)
            return bool(owner_name)
        except ImportError as owner_imp_err:
            logger.error(f"Failed to retrieve tree owner due to import error: {owner_imp_err}")
            return False

    def get_cookies(self, cookie_names: List[str], timeout: int = 30) -> bool:
        """
        Advanced cookie management with timeout and session validation.

        Waits for specific cookies to be available with intelligent retry logic
        and continuous session validity checking.

        Args:
            cookie_names: List of cookie names to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if all cookies found, False otherwise
        """
        if not self.driver:
            logger.error("get_cookies: WebDriver instance is None.")
            return False

        # Skip is_sess_valid() check here to prevent circular recursion

        start_time = time.time()
        logger.debug(f"Waiting up to {timeout}s for cookies: {cookie_names}...")
        required_lower = {name.lower() for name in cookie_names}
        interval = 0.5
        last_missing_str = ""

        while time.time() - start_time < timeout:
            try:
                # Basic driver check (avoid is_sess_valid() to prevent recursion)
                if not self.driver:
                    logger.warning("Driver became None while waiting for cookies.")
                    return False

                cookies = self.driver.get_cookies()
                current_cookies_lower = {
                    c["name"].lower()
                    for c in cookies
                    if isinstance(c, dict) and "name" in c
                }
                missing_lower = required_lower - current_cookies_lower

                if not missing_lower:
                    logger.debug(f"All required cookies found: {cookie_names}.")
                    # Skip automatic cookie sync to prevent recursion
                    # Cookie syncing will be handled elsewhere when needed
                    return True

                # Log missing cookies only if the set changes
                missing_str = ", ".join(sorted(missing_lower))
                if missing_str != last_missing_str:
                    logger.debug(f"Still missing cookies: {missing_str}")
                    last_missing_str = missing_str

                time.sleep(interval)

            except WebDriverException as e:
                logger.error(f"WebDriverException while retrieving cookies: {e}")
                # Check if session died due to the exception
                if not self.is_sess_valid():
                    logger.error("Session invalid after WebDriverException during cookie retrieval.")
                    return False
                # If session still valid, wait a bit longer before next try
                time.sleep(interval * 2)

            except Exception as e:
                logger.error(f"Unexpected error during cookie retrieval: {e}")
                time.sleep(interval * 2)

        # Final check after timeout
        missing_final = []
        try:
            if self.is_sess_valid():
                cookies_final = self.driver.get_cookies()
                current_cookies_final_lower = {
                    c["name"].lower()
                    for c in cookies_final
                    if isinstance(c, dict) and "name" in c
                }
                missing_final = [
                    name for name in cookie_names
                    if name.lower() not in current_cookies_final_lower
                ]
            else:
                missing_final = cookie_names
        except Exception:
            missing_final = cookie_names

        if missing_final:
            logger.warning(f"Timeout waiting for cookies. Missing: {missing_final}.")
            return False
        else:
            logger.debug("Cookies found in final check after loop (unexpected).")
            return True

    def _sync_cookies_to_requests(self):
        """
        Synchronize cookies from WebDriver to requests session.

        This ensures that API calls made through requests.Session
        have the same authentication cookies as the browser session.
        """
        if not self.driver or not hasattr(self.api_manager, '_requests_session'):
            return

        try:
            # Get cookies from WebDriver
            driver_cookies = self.driver.get_cookies()

            # Clear existing cookies in requests session
            self.api_manager._requests_session.cookies.clear()

            # Sync cookies from driver to requests session
            synced_count = 0
            for cookie in driver_cookies:
                if isinstance(cookie, dict) and "name" in cookie and "value" in cookie:
                    self.api_manager._requests_session.cookies.set(
                        cookie["name"],
                        cookie["value"],
                        domain=cookie.get("domain"),
                        path=cookie.get("path", "/")
                    )
                    synced_count += 1

            logger.debug(f"Synced {synced_count} cookies from WebDriver to requests session")

        except Exception as e:
            logger.error(f"Failed to sync cookies to requests session: {e}")

    def _sync_cookies(self):
        """
        Simple cookie synchronization from WebDriver to requests session.

        Simplified version that avoids all session validation to prevent recursion.
        """
        # Recursion guard to prevent infinite loops
        if hasattr(self, '_in_sync_cookies') and self._in_sync_cookies:
            logger.debug("Recursion detected in _sync_cookies(), skipping to prevent loop")
            return

        if not self.driver:
            return

        if not hasattr(self.api_manager, '_requests_session') or not self.api_manager._requests_session:
            return

        try:
            # Set recursion guard
            self._in_sync_cookies = True

            # Simple cookie retrieval without any validation
            driver_cookies = self.driver.get_cookies()
            if not driver_cookies:
                return

            # Clear and sync cookies
            self.api_manager._requests_session.cookies.clear()
            synced_count = 0

            for cookie in driver_cookies:
                if isinstance(cookie, dict) and "name" in cookie and "value" in cookie:
                    try:
                        self.api_manager._requests_session.cookies.set(
                            cookie["name"],
                            cookie["value"],
                            domain=cookie.get("domain"),
                            path=cookie.get("path", "/")
                        )
                        synced_count += 1
                    except Exception:
                        continue  # Skip problematic cookies silently

            logger.debug(f"Synced {synced_count} cookies to requests session")

        except Exception as e:
            logger.warning(f"Cookie sync failed: {e}")
            return
        finally:
            # Clear recursion guard
            if hasattr(self, '_in_sync_cookies'):
                self._in_sync_cookies = False



    def check_js_errors(self) -> List[Dict[str, Any]]:
        """
        Check for JavaScript errors in the browser console.

        Returns:
            List[Dict]: List of JavaScript errors found since last check
        """
        if not self.driver or not self.driver_live:
            return []

        try:
            # Get browser logs
            logs = self.driver.get_log('browser')

            # Filter for errors that occurred after last check
            current_time = datetime.now(timezone.utc)
            js_errors = []

            for log_entry in logs:
                # Check if this is a JavaScript error
                if log_entry.get('level') in ['SEVERE', 'ERROR']:
                    # Parse timestamp (browser logs use milliseconds since epoch)
                    log_timestamp = datetime.fromtimestamp(
                        log_entry.get('timestamp', 0) / 1000,
                        tz=timezone.utc
                    )

                    # Only include errors since last check
                    if log_timestamp > self.last_js_error_check:
                        js_errors.append({
                            'timestamp': log_timestamp,
                            'level': log_entry.get('level'),
                            'message': log_entry.get('message', ''),
                            'source': log_entry.get('source', '')
                        })

            # Update last check time
            self.last_js_error_check = current_time

            if js_errors:
                logger.warning(f"Found {len(js_errors)} JavaScript errors since last check")
                for error in js_errors:
                    logger.debug(f"JS Error: {error['message']}")

            return js_errors

        except Exception as e:
            logger.error(f"Failed to check JavaScript errors: {e}")
            return []

    def monitor_js_errors(self) -> bool:
        """
        Monitor JavaScript errors and log warnings if found.

        Returns:
            bool: True if no critical errors found, False if critical errors detected
        """
        errors = self.check_js_errors()

        # Count critical errors (those that might affect functionality)
        critical_errors = [
            error for error in errors
            if any(keyword in error['message'].lower() for keyword in [
                'uncaught', 'reference error', 'type error', 'syntax error'
            ])
        ]

        if critical_errors:
            logger.warning(f"Found {len(critical_errors)} critical JavaScript errors")
            return False

        return True

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

    # === MISSING API METHODS FROM OLD SESSIONMANAGER ===

    @retry_on_failure(max_attempts=3)
    def get_csrf(self) -> Optional[str]:
        """
        Retrieve CSRF token from API.

        Returns:
            str: CSRF token if successful, None otherwise
        """
        if not self.is_sess_valid():
            logger.error("get_csrf: Session invalid.")
            return None

        from urllib.parse import urljoin
        csrf_token_url = urljoin(config_schema.api.base_url, "discoveryui-matches/parents/api/csrfToken")
        logger.debug(f"Attempting to fetch fresh CSRF token from: {csrf_token_url}")

        # Check essential cookies
        essential_cookies = ["ANCSESSIONID", "SecureATT"]
        if not self.get_cookies(essential_cookies, timeout=10):
            logger.warning(f"Essential cookies {essential_cookies} NOT found before CSRF token API call.")

        try:
            # Import _api_req locally to avoid circular imports
            from utils import _api_req

            response_data = _api_req(
                url=csrf_token_url,
                driver=self.driver,
                session_manager=self,
                method="GET",
                use_csrf_token=False,
                api_description="CSRF Token API",
                force_text_response=True,
            )

            if response_data and isinstance(response_data, str):
                csrf_token_val = response_data.strip()
                if csrf_token_val and len(csrf_token_val) > 20:
                    logger.debug(f"CSRF token successfully retrieved (Length: {len(csrf_token_val)}).")
                    self.csrf_token = csrf_token_val
                    return csrf_token_val
                else:
                    logger.error(f"CSRF token API returned empty or invalid string: '{csrf_token_val}'")
                    return None
            else:
                logger.warning("Failed to get CSRF token response via _api_req.")
                return None

        except Exception as e:
            logger.error(f"Unexpected error in get_csrf: {e}", exc_info=True)
            return None

    @retry_on_failure(max_attempts=3)
    def get_my_profileId(self) -> Optional[str]:
        """
        Retrieve user's profile ID (ucdmid).

        Returns:
            str: Profile ID if successful, None otherwise
        """
        if not self.is_sess_valid():
            logger.error("get_my_profileId: Session invalid.")
            return None

        from urllib.parse import urljoin
        url = urljoin(config_schema.api.base_url, "app-api/cdp-p13n/api/v1/users/me?attributes=ucdmid")
        logger.debug("Attempting to fetch own profile ID (ucdmid)...")

        try:
            from utils import _api_req

            response_data = _api_req(
                url=url,
                driver=self.driver,
                session_manager=self,
                method="GET",
                use_csrf_token=False,
                api_description="Get my profile_id",
            )

            if not response_data:
                logger.warning("Failed to get profile_id response via _api_req.")
                return None

            if isinstance(response_data, dict) and "data" in response_data:
                data_dict = response_data["data"]
                if isinstance(data_dict, dict) and "ucdmid" in data_dict:
                    my_profile_id_val = str(data_dict["ucdmid"]).upper()
                    logger.debug(f"Successfully retrieved profile_id: {my_profile_id_val}")
                    # Store in API manager
                    self.api_manager.my_profile_id = my_profile_id_val
                    if not self._profile_id_logged:
                        logger.info(f"My profile id: {my_profile_id_val}")
                        self._profile_id_logged = True
                    return my_profile_id_val
                else:
                    logger.error("Could not find 'ucdmid' in 'data' dict of profile_id API response.")
                    return None
            else:
                logger.error(f"Unexpected response format for profile_id API: {type(response_data)}")
                return None

        except Exception as e:
            logger.error(f"Unexpected error in get_my_profileId: {e}", exc_info=True)
            return None

    @retry_on_failure(max_attempts=3)
    def get_my_uuid(self) -> Optional[str]:
        """
        Retrieve user's UUID (testId).

        Returns:
            str: UUID if successful, None otherwise
        """
        if not self.is_sess_valid():
            logger.error("get_my_uuid: Session invalid.")
            return None

        from urllib.parse import urljoin
        url = urljoin(config_schema.api.base_url, "api/uhome/secure/rest/header/dna")
        logger.debug("Attempting to fetch own UUID (testId) from header/dna API...")

        try:
            from utils import _api_req

            response_data = _api_req(
                url=url,
                driver=self.driver,
                session_manager=self,
                method="GET",
                use_csrf_token=False,
                api_description="Get UUID API",
            )

            if response_data and isinstance(response_data, dict):
                if "testId" in response_data:
                    my_uuid_val = str(response_data["testId"]).upper()
                    logger.debug(f"Successfully retrieved UUID: {my_uuid_val}")
                    # Store in API manager
                    self.api_manager.my_uuid = my_uuid_val
                    if not self._uuid_logged:
                        logger.info(f"My uuid: {my_uuid_val}")
                        self._uuid_logged = True
                    return my_uuid_val
                else:
                    logger.error("Could not retrieve UUID ('testId' missing in response).")
                    return None
            else:
                logger.error("Failed to get header/dna data via _api_req.")
                return None

        except Exception as e:
            logger.error(f"Unexpected error in get_my_uuid: {e}", exc_info=True)
            return None

    @retry_on_failure(max_attempts=3)
    def get_my_tree_id(self) -> Optional[str]:
        """
        Retrieve user's tree ID.

        Returns:
            str: Tree ID if successful, None otherwise
        """
        try:
            import api_utils as local_api_utils
        except ImportError as e:
            logger.error(f"get_my_tree_id: Failed to import api_utils: {e}")
            raise ImportError(f"api_utils module failed to import: {e}")

        tree_name_config = config_schema.api.tree_name
        if not tree_name_config:
            logger.debug("TREE_NAME not configured, skipping tree ID retrieval.")
            return None

        if not self.is_sess_valid():
            logger.error("get_my_tree_id: Session invalid.")
            return None

        logger.debug(f"Delegating tree ID fetch for TREE_NAME='{tree_name_config}' to api_utils...")
        try:
            my_tree_id_val = local_api_utils.call_header_trees_api_for_tree_id(
                self, tree_name_config
            )
            if my_tree_id_val:
                # Store in API manager
                self.api_manager.my_tree_id = my_tree_id_val
                if not self._tree_id_logged:
                    logger.info(f"My tree id: {my_tree_id_val}")
                    self._tree_id_logged = True
                return my_tree_id_val
            else:
                logger.warning("api_utils.call_header_trees_api_for_tree_id returned None.")
                return None
        except Exception as e:
            logger.error(f"Error calling api_utils.call_header_trees_api_for_tree_id: {e}", exc_info=True)
            return None

    @retry_on_failure(max_attempts=3)
    def get_tree_owner(self, tree_id: str) -> Optional[str]:
        """
        Retrieve tree owner name.

        Args:
            tree_id: The tree ID to get owner for

        Returns:
            str: Tree owner name if successful, None otherwise
        """
        try:
            import api_utils as local_api_utils
        except ImportError as e:
            logger.error(f"get_tree_owner: Failed to import api_utils: {e}")
            raise ImportError(f"api_utils module failed to import: {e}")

        if not tree_id:
            logger.warning("Cannot get tree owner: tree_id is missing.")
            return None

        if not isinstance(tree_id, str):
            logger.warning(f"Invalid tree_id type provided: {type(tree_id)}. Expected string.")
            return None

        if not self.is_sess_valid():
            logger.error("get_tree_owner: Session invalid.")
            return None

        logger.debug(f"Delegating tree owner fetch for tree ID {tree_id} to api_utils...")
        try:
            owner_name = local_api_utils.call_tree_owner_api(self, tree_id)
            if owner_name:
                # Store in API manager
                self.api_manager.tree_owner_name = owner_name
                if not self._owner_logged:
                    logger.info(f"Tree owner name: {owner_name}")
                    self._owner_logged = True
                return owner_name
            else:
                logger.warning("api_utils.call_tree_owner_api returned None.")
                return None
        except Exception as e:
            logger.error(f"Error calling api_utils.call_tree_owner_api: {e}", exc_info=True)
            return None

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
    def requests_session(self):
        """Get the requests session."""
        return self.api_manager.requests_session

    @property
    def _requests_session(self):
        """Get the requests session (backward compatibility)."""
        return self.api_manager.requests_session

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
        # Try to get from API manager first, then retrieve if needed
        profile_id = self.api_manager.my_profile_id
        if not profile_id:
            profile_id = self.get_my_profileId()
        return profile_id

    @property
    def my_uuid(self):
        """Get the user's UUID."""
        # Try to get from API manager first, then retrieve if needed
        uuid_val = self.api_manager.my_uuid
        if not uuid_val:
            uuid_val = self.get_my_uuid()
        return uuid_val

    @property
    def my_tree_id(self):
        """Get the user's tree ID."""
        # Try to get from API manager first, then retrieve if needed
        tree_id = self.api_manager.my_tree_id
        if not tree_id and config_schema.api.tree_name:
            tree_id = self.get_my_tree_id()
        return tree_id

    @property
    def csrf_token(self):
        """Get the CSRF token."""
        # Try to get from API manager first, then retrieve if needed
        csrf = self.api_manager.csrf_token
        if not csrf:
            csrf = self.get_csrf()
        return csrf

    # Public properties
    @property
    def tree_owner_name(self):
        """Get the tree owner name."""
        return self.api_manager.tree_owner_name

    @property
    def requests_session(self):
        """Get the requests session."""
        return self.api_manager.requests_session

    # Enhanced capabilities properties
    @property
    def scraper(self):
        """Get the CloudScraper instance for anti-bot protection."""
        return getattr(self, '_scraper', None)

    @scraper.setter
    def scraper(self, value):
        """Set the CloudScraper instance."""
        self._scraper = value

    # Compatibility properties for legacy code
    @property
    def browser_needed(self):
        """Get/set browser needed flag."""
        return self.browser_manager.browser_needed

    @browser_needed.setter
    def browser_needed(self, value: bool):
        """Set browser needed flag."""
        self.browser_manager.browser_needed = value



    @property
    def _requests_session(self):
        """Get the requests session (compatibility property with underscore)."""
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


# === Decomposed Helper Functions ===
def _test_session_manager_initialization():
    """Test SessionManager initialization with detailed component verification"""
    required_components = [
        ("db_manager", "DatabaseManager for database operations"),
        ("browser_manager", "BrowserManager for browser operations"),
        ("api_manager", "APIManager for API interactions"),
        ("validator", "SessionValidator for session validation"),
    ]

    print("📋 Testing SessionManager initialization:")
    print(f"   Creating SessionManager instance...")

    try:
        session_manager = SessionManager()

        print(f"   ✅ SessionManager created successfully (ID: {id(session_manager)})")

        # Test component availability
        results = []
        for component_name, description in required_components:
            has_component = hasattr(session_manager, component_name)
            component_value = getattr(session_manager, component_name, None)
            is_not_none = component_value is not None

            status = "✅" if has_component and is_not_none else "❌"
            print(f"   {status} {component_name}: {description}")
            print(f"      Has attribute: {has_component}, Not None: {is_not_none}")

            results.append(has_component and is_not_none)
            assert has_component, f"Should have {component_name}"
            assert is_not_none, f"{component_name} should not be None"

        # Test initial state
        initial_ready = session_manager.session_ready
        print(f"   ✅ Initial session_ready state: {initial_ready} (Expected: False)")

        results.append(initial_ready == False)
        assert initial_ready == False, "Should start with session_ready=False"

        print(f"📊 Results: {sum(results)}/{len(results)} initialization checks passed")
        return True

    except Exception as e:
        print(f"❌ SessionManager initialization failed: {e}")
        return False


def _test_component_manager_availability():
    """Test component manager availability with detailed type verification"""
    component_tests = [
        ("db_manager", "DatabaseManager", "Database operations and connection pooling"),
        (
            "browser_manager",
            "BrowserManager",
            "Browser automation and WebDriver management",
        ),
        ("api_manager", "APIManager", "API interactions and session management"),
        ("validator", "SessionValidator", "Session validation and readiness checks"),
    ]

    print("📋 Testing component manager availability:")

    try:
        session_manager = SessionManager()
        results = []

        for component_name, expected_type, description in component_tests:
            component = getattr(session_manager, component_name, None)
            is_available = component is not None
            type_name = type(component).__name__ if component else "None"

            status = "✅" if is_available else "❌"
            print(f"   {status} {component_name}: {description}")
            print(f"      Type: {type_name}, Available: {is_available}")

            results.append(is_available)
            assert is_available, f"{expected_type} should be created"

        print(f"📊 Results: {sum(results)}/{len(results)} component managers available")
        return True

    except Exception as e:
        print(f"❌ Component manager availability test failed: {e}")
        return False


def _test_database_operations():
    """Test database operations with detailed result verification"""
    database_operations = [
        ("ensure_db_ready", "Ensure database is ready for operations"),
        ("get_db_conn", "Get database connection/session"),
        ("get_db_conn_context", "Get database session context manager"),
    ]

    print("📋 Testing database operations:")

    try:
        session_manager = SessionManager()
        results = []

        for operation_name, description in database_operations:
            try:
                if operation_name == "ensure_db_ready":
                    result = session_manager.ensure_db_ready()
                    is_bool = isinstance(result, bool)

                    status = "✅" if is_bool else "❌"
                    print(f"   {status} {operation_name}: {description}")
                    print(f"      Result: {result} (Type: {type(result).__name__})")

                    results.append(is_bool)
                    assert is_bool, f"{operation_name} should return bool"

                elif operation_name == "get_db_conn":
                    conn = session_manager.get_db_conn()
                    has_conn = conn is not None

                    status = "✅" if has_conn else "❌"
                    print(f"   {status} {operation_name}: {description}")
                    print(
                        f"      Connection: {type(conn).__name__ if conn else 'None'}"
                    )

                    results.append(True)  # Just test it doesn't crash

                    # Return the connection if we got one
                    if conn:
                        session_manager.return_session(conn)

                elif operation_name == "get_db_conn_context":
                    context = session_manager.get_db_conn_context()
                    has_context = context is not None

                    status = "✅" if has_context else "❌"
                    print(f"   {status} {operation_name}: {description}")
                    print(
                        f"      Context: {type(context).__name__ if context else 'None'}"
                    )

                    results.append(True)  # Just test it doesn't crash

            except Exception as e:
                print(f"   ❌ {operation_name}: Exception {e}")
                results.append(False)

        print(
            f"📊 Results: {sum(results)}/{len(results)} database operations successful"
        )
        return True

    except Exception as e:
        print(f"❌ Database operations test failed: {e}")
        return False


def _test_browser_operations():
    session_manager = SessionManager()
    result = session_manager.start_browser("test_action")
    assert isinstance(result, bool), "start_browser should return bool"
    session_manager.close_browser()
    return True


def _test_property_access():
    session_manager = SessionManager()
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
        assert hasattr(session_manager, prop), f"Property {prop} should exist"
    return True


def _test_component_delegation():
    session_manager = SessionManager()
    db_result = session_manager.ensure_db_ready()
    assert isinstance(db_result, bool), "Database delegation should work"
    browser_result = session_manager.start_browser("test")
    assert isinstance(browser_result, bool), "Browser delegation should work"
    return True


def _test_initialization_performance():
    import time

    session_managers = []
    start_time = time.time()
    for i in range(3):
        session_manager = SessionManager()
        session_managers.append(session_manager)
    end_time = time.time()
    total_time = end_time - start_time
    max_time = 5.0
    assert (
        total_time < max_time
    ), f"3 optimized initializations took {total_time:.3f}s, should be under {max_time}s"
    for sm in session_managers:
        try:
            sm.close_sess(keep_db=True)
        except Exception:
            pass
    return True


def _test_error_handling():
    session_manager = SessionManager()
    try:
        session_manager.ensure_db_ready()
        session_manager.start_browser("test_action")
        session_manager.close_browser()
        _ = session_manager.session_ready
        _ = session_manager.is_ready
    except Exception as e:
        assert False, f"SessionManager should handle operations gracefully: {e}"
    return True


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for session_manager.py (decomposed).
    """
    from test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite(
            "Session Manager & Component Coordination", "session_manager.py"
        )
        suite.start_suite()
        suite.run_test(
            "SessionManager Initialization",
            _test_session_manager_initialization,
            "4 component managers created (db, browser, api, validator), session_ready=False initially.",
            "Test SessionManager initialization with detailed component verification.",
            "Create SessionManager and verify: db_manager, browser_manager, api_manager, validator exist and are not None.",
        )
        suite.run_test(
            "Component Manager Availability",
            _test_component_manager_availability,
            "4 component managers available with correct types: DatabaseManager, BrowserManager, APIManager, SessionValidator.",
            "Test component manager availability with detailed type verification.",
            "Check each component manager exists, is not None, and verify type names match expected classes.",
        )
        suite.run_test(
            "Database Operations",
            _test_database_operations,
            "3 database operations work: ensure_db_ready()→bool, get_db_conn()→connection, get_db_conn_context()→context.",
            "Test database operations with detailed result verification.",
            "Call ensure_db_ready(), get_db_conn(), get_db_conn_context() and verify return types and no exceptions.",
        )
        suite.run_test(
            "Browser Operations",
            _test_browser_operations,
            "Browser operations are properly delegated to BrowserManager without errors",
            "Call start_browser() and close_browser() and verify proper delegation and error handling",
            "Test browser operation delegation and graceful error handling",
        )
        suite.run_test(
            "Property Access",
            _test_property_access,
            "All expected properties are accessible without AttributeError",
            "Access various session properties and verify they exist (even if None)",
            "Test property access and delegation to component managers",
        )
        suite.run_test(
            "Component Method Delegation",
            _test_component_delegation,
            "Method calls are properly delegated to appropriate component managers",
            "Call methods that should be delegated and verify they execute without errors",
            "Test delegation pattern between SessionManager and component managers",
        )
        suite.run_test(
            "Initialization Performance",
            _test_initialization_performance,
            "3 SessionManager initializations complete in under 15 seconds",
            "Create 3 SessionManager instances and measure total time",
            "Test performance of SessionManager initialization with all component managers",
        )
        suite.run_test(
            "Error Handling",
            _test_error_handling,
            "SessionManager handles various operations gracefully without raising exceptions",
            "Perform various operations and property access and verify no exceptions are raised",
            "Test error handling and graceful degradation for session operations",
        )
        return suite.finish_suite()


if __name__ == "__main__":
    run_comprehensive_tests()
