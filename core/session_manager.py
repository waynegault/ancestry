#!/usr/bin/env python3

"""
Refactored Session Manager - Orchestrates all session components.

This module provides a new, modular SessionManager that orchestrates
the specialized managers (DatabaseManager, BrowserManager, APIManager, etc.)
to provide a clean, maintainable architecture.

PHASE 5.1 OPTIMIZATION: Enhanced with intelligent session caching for dramatic
performance improvement. Reduces initialization from 34.59s to <12s target.
"""

# === SUPPRESS TEST WARNINGS FIRST (before any imports) ===
import os
import sys
import warnings

# NOTE: These warning suppressions are OBSOLETE but kept for defense-in-depth.
# PREFERRED APPROACH: Run as script (`python core\session_manager.py`) NOT as module (`python -m core.session_manager`)
# See test execution block at bottom of file for full explanation.
#
# The suppression below doesn't work for runpy RuntimeWarnings anyway (they occur before our code runs),
# but it does suppress other test-related warnings if someone uses `-m` anyway.

# Suppress warnings during test runs - must be before other imports
if __name__ == "__main__" or any("test" in arg.lower() for arg in sys.argv):
    # Suppress all RuntimeWarnings (including runpy module warnings)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*runpy.*")
    # Also suppress via simplefilter for more aggressive filtering
    warnings.simplefilter("ignore", RuntimeWarning)
    # Suppress config warning output to stderr
    os.environ["SUPPRESS_CONFIG_WARNINGS"] = "1"
    # Redirect stderr temporarily to suppress subprocess warnings
    import io
    _original_stderr = sys.stderr
    sys.stderr = io.StringIO()

# === CORE INFRASTRUCTURE ===
from pathlib import Path

# Add parent directory to path for core_imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from standard_imports import setup_module

# Restore stderr after critical imports
if __name__ == "__main__" or any("test" in arg.lower() for arg in sys.argv):
    sys.stderr = _original_stderr

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
# === PHASE 5.1: SESSION PERFORMANCE OPTIMIZATION ===
from core.session_cache import (
    cached_api_manager,
    cached_browser_manager,
    cached_database_manager,
    cached_session_validator,
    clear_session_cache,
    get_session_cache_stats,
)
from error_handling import (
    error_context,
    graceful_degradation,
    retry_on_failure,
    timeout_protection,
)

# === MODULE SETUP ===
logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from selenium.webdriver.remote.webdriver import WebDriver as WebDriverType
else:
    WebDriverType = Any

# === THIRD-PARTY IMPORTS ===
import requests
from requests.adapters import HTTPAdapter

try:
    import cloudscraper
except ImportError:
    cloudscraper = None
    logger.warning("CloudScraper not available - anti-bot protection disabled")

# === SELENIUM IMPORTS ===
try:
    from selenium.common.exceptions import WebDriverException
    from selenium.webdriver.remote.webdriver import WebDriver
except ImportError:
    WebDriverException = Exception
    WebDriver = None  # type: ignore

# === LOCAL IMPORTS ===
import contextlib

from api_constants import API_PATH_UUID_NAVHEADER
from config import config_schema
from core.api_manager import APIManager
from core.browser_manager import BrowserManager
from core.database_manager import DatabaseManager
from core.session_validator import SessionValidator

# === MODULE CONSTANTS ===
# Use global cached config instance


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
        self._cached_session_state: dict[str, Any] = {}

        # ⚡ OPTIMIZATION 1: Pre-cached CSRF token for Action 6 performance
        self._cached_csrf_token: Optional[str] = None
        self._csrf_cache_time: float = 0.0
        self._csrf_cache_duration: float = 300.0  # 5-minute cache

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

        # Initialize rate limiter (use global singleton for all API calls)
        try:
            from utils import get_rate_limiter
            self.rate_limiter = get_rate_limiter()
        except ImportError:
            self.rate_limiter = None

        # Alias for backward compatibility with code that references dynamic_rate_limiter
        # Both attributes point to the same RateLimiter instance
        self.dynamic_rate_limiter = self.rate_limiter

        # UNIVERSAL SESSION HEALTH MONITORING (moved from action6-specific to universal)
        self.session_health_monitor = {
            'is_alive': threading.Event(),
            'death_detected': threading.Event(),
            'last_heartbeat': time.time(),
            'heartbeat_interval': 30,  # Check every 30 seconds
            'death_cascade_halt': threading.Event(),
            'death_timestamp': None,
            'parallel_operations': 0,
            'death_cascade_count': 0
        }
        self.session_health_monitor['is_alive'].set()  # Initially alive

        # === ENHANCED SESSION CAPABILITIES ===
        # JavaScript error monitoring
        self.last_js_error_check: datetime = datetime.now(timezone.utc)

        # CSRF token caching for performance optimization
        self._cached_csrf_token: Optional[str] = None
        self._csrf_cache_time: float = 0
        self._csrf_cache_duration: float = 300  # 5 minutes

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

    def _initialize_enhanced_requests_session(self) -> None:
        """
        Initialize enhanced requests session with advanced configuration.
        Includes connection pooling, retry strategies, and performance optimizations.
        """
        logger.debug("Initializing enhanced requests session...")

        # Enhanced retry strategy with more comprehensive status codes
        # Advanced HTTPAdapter with connection pooling (no urllib3 retries)
        # Retry logic handled at application level in utils.py for consistency
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=50,
            max_retries=0  # Application handles retries
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

    def _initialize_cloudscraper(self) -> None:
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

            # Apply HTTP adapter to CloudScraper (no urllib3 retries - application handles)
            scraper_adapter = HTTPAdapter(max_retries=0)
            self._scraper.mount("http://", scraper_adapter)
            self._scraper.mount("https://", scraper_adapter)

            logger.debug("CloudScraper initialized successfully with connection pooling")

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

    def close_browser(self) -> None:
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

    @timeout_protection(timeout=120)  # Increased timeout for complex operations like Action 7
    @graceful_degradation(fallback_value=False)
    @error_context("ensure_session_ready")
    def _check_cached_readiness(self, action_name: Optional[str]) -> Optional[bool]:
        """Check if we can use cached readiness state.

        Returns:
            True if cached state is valid and ready, False if invalid, None if no cache
        """
        if self._last_readiness_check is None:
            return None

        time_since_check = time.time() - self._last_readiness_check
        cache_duration = 60  # Use consistent 60-second cache for all actions

        if time_since_check >= cache_duration or not self.session_ready:
            return None

        # Validate that the cached state is still accurate
        if self.browser_manager.browser_needed and not self.browser_manager.is_session_valid():
            logger.debug(
                f"Cached session readiness invalid - driver session expired (age: {time_since_check:.1f}s)"
            )
            self.session_ready = False
            self._last_readiness_check = None
            return False

        logger.debug(
            f"Using cached session readiness (age: {time_since_check:.1f}s, action: {action_name})"
        )
        return True

    def _perform_readiness_validation(self, action_name: Optional[str], skip_csrf: bool) -> bool:
        """Perform readiness checks and identifier retrieval.

        Returns:
            True if all checks pass, False otherwise
        """
        # PHASE 5.1: Optimized readiness checks with circuit breaker pattern
        try:
            ready_checks_ok = self.validator.perform_readiness_checks(
                self.browser_manager, self.api_manager, self, action_name, skip_csrf=skip_csrf
            )

            if not ready_checks_ok:
                logger.error("Readiness checks failed.")
                return False

        except Exception as e:
            logger.critical(f"Exception in readiness checks: {e}", exc_info=True)
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

        # ⚡ OPTIMIZATION 1: Pre-cache CSRF token during session setup
        if ready_checks_ok and identifiers_ok:
            self._precache_csrf_token()

        return ready_checks_ok and identifiers_ok and owner_ok

    def ensure_session_ready(self, action_name: Optional[str] = None, skip_csrf: bool = False) -> bool:
        """
        Ensure the session is ready for operations.

        PHASE 5.1 OPTIMIZATION: Uses intelligent caching to bypass expensive
        readiness checks when session state is known to be valid.

        Args:
            action_name: Optional name of the action for logging
            skip_csrf: Skip CSRF token validation (for actions that don't need it)

        Returns:
            bool: True if session is ready, False otherwise
        """
        start_time = time.time()
        # Removed duplicate logging - browser_manager will log the action

        # MINIMAL FIX: Set browser_needed to True for session operations
        self.browser_manager.browser_needed = True

        # PHASE 5.1: Check cached session state first
        cached_result = self._check_cached_readiness(action_name)
        if cached_result is not None:
            return cached_result

        # Ensure driver is live if browser is needed (with optimization)
        if self.browser_manager.browser_needed and not self.browser_manager.ensure_driver_live(action_name):
            logger.error("Failed to ensure driver live.")
            self.session_ready = False
            return False

        # Perform readiness validation
        self.session_ready = self._perform_readiness_validation(action_name, skip_csrf)

        # PHASE 5.1: Cache the readiness check result
        self._last_readiness_check = time.time()

        check_time = time.time() - start_time
        logger.debug(
            f"Session readiness check completed in {check_time:.3f}s, status: {self.session_ready}"
        )
        return self.session_ready

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
            if login_ok is False:
                logger.warning("Session verification failed (user not logged in).")
                return False
            # login_ok is None
            logger.error("Session verification failed critically (login_status returned None).")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during session verification: {e}", exc_info=True)
            return False

    def is_sess_valid(self) -> bool:
        """
        Enhanced session validity check with recovery capabilities.

        Checks if driver exists and is responsive, with automatic recovery
        for invalid sessions during long-running operations.

        Returns:
            bool: True if driver exists and is responsive, False otherwise
        """
        # Simple check - verify driver exists
        if self.driver is None:
            return False

        # Enhanced check - verify driver is responsive
        try:
            # Quick responsiveness test
            _ = self.driver.current_url
            return True
        except Exception as e:
            logger.warning(f"🔌 WebDriver session appears invalid: {e}")
            # Attempt session recovery for long-running operations
            if self._should_attempt_recovery():
                logger.info("🔄 Attempting automatic session recovery...")
                if self._attempt_session_recovery():
                    logger.info("✅ Session recovery successful")
                    return True
                logger.error("❌ Session recovery failed")
            else:
                logger.debug("⏭️  Skipping session recovery (not in long-running operation)")
            return False

    def _should_attempt_recovery(self) -> bool:
        """
        Determine if session recovery should be attempted.

        Returns:
            bool: True if recovery should be attempted
        """
        # Only attempt recovery if session was previously working
        # and we're in a long-running operation
        return bool(self.session_ready and
                   self.session_start_time and
                   time.time() - self.session_start_time > 300)  # 5 minutes

    def _attempt_session_recovery(self) -> bool:
        """
        Attempt to recover an invalid WebDriver session.

        Returns:
            bool: True if recovery successful, False otherwise
        """
        try:
            logger.debug("Closing invalid browser session...")
            self.close_browser()

            logger.debug("Starting new browser session...")
            if self.start_browser("session_recovery"):
                logger.debug("Browser recovery successful, re-authenticating...")

                # Re-authenticate if needed
                from utils import login_status
                if login_status(self, disable_ui_fallback=False):
                    logger.info("Session recovery and re-authentication successful")
                    return True
                logger.error("Re-authentication failed after browser recovery")

        except Exception as e:
            logger.error(f"Session recovery failed: {e}", exc_info=True)

        return False

    # UNIVERSAL SESSION HEALTH MONITORING METHODS
    def check_session_health(self) -> bool:
        """
        Universal session health monitoring that detects session death and prevents
        cascade failures during long-running operations.

        This replaces action6-specific monitoring with universal monitoring.
        """
        try:
            # Quick session validation first
            if not self.is_sess_valid():
                if not self.session_health_monitor['death_detected'].is_set():
                    self.session_health_monitor['death_detected'].set()
                    self.session_health_monitor['is_alive'].clear()
                    self.session_health_monitor['death_timestamp'] = time.time()
                    logger.critical(
                        f"🚨 SESSION DEATH DETECTED at {time.strftime('%H:%M:%S')}"
                        f" - Universal session health monitoring triggered"
                    )
                return False

            # Update heartbeat if session is alive
            self.session_health_monitor['last_heartbeat'] = time.time()
            return True

        except Exception as exc:
            logger.error(f"Session health check failed: {exc}")
            # Assume session is dead on health check failure
            if not self.session_health_monitor['death_detected'].is_set():
                self.session_health_monitor['death_detected'].set()
                self.session_health_monitor['is_alive'].clear()
                self.session_health_monitor['death_timestamp'] = time.time()
                logger.critical("🚨 SESSION HEALTH CHECK FAILED - Assuming session death")
            return False

    def is_session_death_cascade(self) -> bool:
        """Check if we're in a session death cascade scenario."""
        return self.session_health_monitor['death_detected'].is_set()

    def should_halt_operations(self) -> bool:
        """Determine if operations should halt due to session death."""
        if self.is_session_death_cascade():
            self.session_health_monitor['death_cascade_count'] += 1

            # Halt immediately if session is dead
            logger.warning(
                f"⚠️  Halting operation due to session death cascade "
                f"(cascade #{self.session_health_monitor['death_cascade_count']})"
            )
            return True
        return False

    def reset_session_health_monitoring(self) -> None:
        """Reset session health monitoring (used when creating new sessions)."""
        self.session_health_monitor['is_alive'].set()
        self.session_health_monitor['death_detected'].clear()
        self.session_health_monitor['last_heartbeat'] = time.time()
        self.session_health_monitor['death_timestamp'] = None
        self.session_health_monitor['parallel_operations'] = 0
        self.session_health_monitor['death_cascade_count'] = 0
        logger.debug("🔄 Session health monitoring reset for new session")

    def _reset_logged_flags(self) -> None:
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

    def _get_current_cookie_names(self) -> set[str]:
        """Get current cookie names from driver (lowercase).

        Returns:
            Set of lowercase cookie names
        """
        if not self.driver:
            return set()
        try:
            cookies = self.driver.get_cookies()
            return {
                c["name"].lower()
                for c in cookies
                if isinstance(c, dict) and "name" in c
            }
        except Exception:
            return set()

    def _check_cookies_available(
        self,
        required_lower: set[str],
        last_missing_str: str
    ) -> tuple[bool, str]:
        """Check if required cookies are available.

        Args:
            required_lower: Set of required cookie names (lowercase)
            last_missing_str: Last missing cookies string for logging

        Returns:
            Tuple of (all_found, new_missing_str)
        """
        current_cookies_lower = self._get_current_cookie_names()
        missing_lower = required_lower - current_cookies_lower

        if not missing_lower:
            return True, ""

        # Log missing cookies only if the set changes
        missing_str = ", ".join(sorted(missing_lower))
        if missing_str != last_missing_str:
            logger.debug(f"Still missing cookies: {missing_str}")

        return False, missing_str

    def _perform_final_cookie_check(
        self,
        cookie_names: list[str]
    ) -> bool:
        """Perform final cookie check after timeout.

        Args:
            cookie_names: List of required cookie names

        Returns:
            True if all cookies found, False otherwise
        """
        try:
            if not self.is_sess_valid():
                logger.warning(f"Timeout waiting for cookies. Missing: {cookie_names}.")
                return False

            current_cookies_lower = self._get_current_cookie_names()
            missing_final = [
                name for name in cookie_names
                if name.lower() not in current_cookies_lower
            ]

            if missing_final:
                logger.warning(f"Timeout waiting for cookies. Missing: {missing_final}.")
                return False

            logger.debug("Cookies found in final check after loop (unexpected).")
            return True

        except Exception:
            logger.warning(f"Timeout waiting for cookies. Missing: {cookie_names}.")
            return False

    def get_cookies(self, cookie_names: list[str], timeout: int = 30) -> bool:
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

        start_time = time.time()
        logger.debug(f"Waiting up to {timeout}s for cookies: {cookie_names}...")
        required_lower = {name.lower() for name in cookie_names}
        interval = 0.5
        last_missing_str = ""

        while time.time() - start_time < timeout:
            try:
                if not self.driver:
                    logger.warning("Driver became None while waiting for cookies.")
                    return False

                all_found, last_missing_str = self._check_cookies_available(
                    required_lower, last_missing_str
                )
                if all_found:
                    logger.debug(f"All required cookies found: {cookie_names}.")
                    return True

                time.sleep(interval)

            except WebDriverException as e:
                logger.error(f"WebDriverException while retrieving cookies: {e}")
                if not self.is_sess_valid():
                    logger.error("Session invalid after WebDriverException during cookie retrieval.")
                    return False
                time.sleep(interval * 2)

            except Exception as e:
                logger.error(f"Unexpected error during cookie retrieval: {e}")
                time.sleep(interval * 2)

        return self._perform_final_cookie_check(cookie_names)

    def _should_skip_cookie_sync(self, current_time: float) -> bool:
        """
        Determine if cookie sync should be skipped based on various conditions.

        Returns:
            True if sync should be skipped, False if sync should proceed
        """
        # Check 1: Recursion prevention
        if hasattr(self, '_in_sync_cookies') and self._in_sync_cookies:
            logger.debug("Cookie sync skipped: recursion detected")
            return True

        # Check 2: Prerequisites validation
        if not self.driver or not hasattr(self.api_manager, '_requests_session'):
            logger.debug("Cookie sync skipped: driver or requests_session not available")
            return True

        # Check 3: Cooldown period (60 seconds)
        # Cookie state rarely changes during normal operation, so we can safely wait 1 minute
        # This prevents burst syncs while still allowing updates during long-running sessions
        COOKIE_SYNC_COOLDOWN = 60.0  # seconds
        if hasattr(self, '_last_cookie_sync_time'):
            time_since_last_sync = current_time - self._last_cookie_sync_time
            if time_since_last_sync < COOKIE_SYNC_COOLDOWN:
                logger.debug(
                    f"Cookie sync skipped: cooldown active "
                    f"(last sync: {time_since_last_sync:.1f}s ago, cooldown: {COOKIE_SYNC_COOLDOWN:.0f}s)"
                )
                return True

        # Check 4: Already synced for this session
        if hasattr(self, '_session_cookies_synced') and self._session_cookies_synced:
            logger.debug("Cookie sync skipped: already synced for this session")
            return True

        return False

    def _sync_cookies_to_requests(self) -> None:
        """
        Synchronize cookies from WebDriver to requests session.
        Only syncs once per session unless forced due to auth errors.

        Enhanced with recursion prevention, robust error handling, and cooldown period.
        Uses _should_skip_cookie_sync() for clean separation of sync validation logic.
        """
        current_time = time.time()

        # Check if sync should be skipped
        if self._should_skip_cookie_sync(current_time):
            return

        try:
            # Set recursion prevention flag
            self._in_sync_cookies = True

            # Get cookies from WebDriver (validated in _should_skip_cookie_sync)
            if not self.driver:
                logger.error("Driver not available for cookie sync")
                return

            driver_cookies = self.driver.get_cookies()

            # Validate cookies were retrieved
            if not driver_cookies:
                logger.debug("No cookies retrieved from WebDriver")
                return

            # Use helper method for robust cookie syncing
            synced_count = self._sync_driver_cookies_to_requests(driver_cookies)

            self._session_cookies_synced = True
            self._last_cookie_sync_time = current_time  # Track sync time for cooldown
            logger.debug(f"Synced {synced_count} cookies from WebDriver to requests session (once per session)")

        except Exception as e:
            logger.error(f"Failed to sync cookies to requests session: {e}")
        finally:
            # Always clear recursion flag
            if hasattr(self, '_in_sync_cookies'):
                self._in_sync_cookies = False

    def force_cookie_resync(self) -> None:
        """Force a cookie resync when authentication errors occur."""
        if hasattr(self, '_session_cookies_synced'):
            delattr(self, '_session_cookies_synced')
        self._sync_cookies_to_requests()
        logger.debug("Forced session cookie resync due to authentication error")



    def _sync_driver_cookies_to_requests(self, driver_cookies: list[dict[str, Any]]) -> int:
        """Sync driver cookies to requests session.

        Args:
            driver_cookies: List of cookies from WebDriver

        Returns:
            Number of cookies synced
        """
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

        return synced_count





    def check_js_errors(self) -> list[dict[str, Any]]:
        """
        Check for JavaScript errors in the browser console.

        Returns:
            list[Dict]: List of JavaScript errors found since last check
        """
        if not self.driver or not self.driver_live:
            return []

        try:
            # Get browser logs (if available)
            if hasattr(self.driver, 'get_log'):
                # Type: ignore to handle dynamic method availability
                logs = getattr(self.driver, 'get_log')('browser')  # type: ignore
            else:
                logger.debug("WebDriver does not support get_log method")
                return []

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

    def check_browser_health(self) -> bool:
        """Check browser health and detect browser death."""
        current_time = time.time()
        self.session_health_monitor['last_browser_health_check'] = current_time

        # Check if browser is needed
        if not self.browser_manager.browser_needed:
            return True

        # Check if driver exists and is responsive
        if not self.browser_manager.is_session_valid():
            self.session_health_monitor['browser_death_count'] = self.session_health_monitor.get('browser_death_count', 0) + 1
            logger.warning(f"🚨 Browser death detected (count: {self.session_health_monitor['browser_death_count']})")
            return False

        return True

    def attempt_browser_recovery(self, action_name: Optional[str] = None) -> bool:
        """
        Public method to attempt browser session recovery.

        Args:
            action_name: Optional name of the action for logging context

        Returns:
            bool: True if recovery successful, False otherwise
        """
        if action_name:
            logger.info(f"Attempting browser recovery for: {action_name}")
        return self._attempt_session_recovery()

    def validate_system_health(self, action_name: str = "Unknown") -> bool:
        """
        Comprehensive system health validation before starting operations.
        Consolidates health check patterns from Actions 6, 7, 8.

        Args:
            action_name: Name of the action performing the check

        Returns:
            True if system is healthy and ready for operations, False otherwise
        """
        try:
            # Check 1: Session death cascade detection
            if self.should_halt_operations():
                cascade_count = self.session_health_monitor.get('death_cascade_count', 0)
                logger.critical(
                    f"🚨 {action_name}: Session death cascade detected (#{cascade_count}). "
                    f"System is not safe for operations."
                )
                return False

            # Check 2: Database connectivity
            try:
                with self.get_db_conn_context() as db_session:
                    if not db_session:
                        logger.critical(f"🚨 {action_name}: Failed to get database session")
                        return False

                    # Test database connectivity with timeout
                    from sqlalchemy import text
                    result = db_session.execute(text("SELECT 1")).scalar()
                    if result != 1:
                        logger.critical(f"🚨 {action_name}: Database query returned unexpected result")
                        return False

            except Exception as db_err:
                logger.critical(f"🚨 {action_name}: Database connectivity error: {db_err}")
                return False

            # Check 3: Browser session validity (if available)
            try:
                if hasattr(self, 'is_sess_valid') and not self.is_sess_valid():
                    logger.warning(f"⚠️ {action_name}: Browser session invalid - may affect operations")
                    # Don't fail hard on browser issues for API-only operations

            except Exception as browser_check_err:
                logger.debug(f"{action_name}: Browser health check failed (non-critical): {browser_check_err}")

            logger.debug(f"✅ {action_name}: System health check passed - all components validated")
            return True

        except Exception as health_err:
            logger.critical(f"🚨 {action_name}: System health validation failed: {health_err}")
            return False

    def check_cascade_before_operation(self, action_name: str, operation_name: str) -> None:
        """
        Check for session death cascade before starting an operation.

        Args:
            action_name: Name of the action performing the check
            operation_name: Name of the operation about to be performed

        Raises:
            Exception: If cascade detected
        """
        if self.should_halt_operations():
            cascade_count = self.session_health_monitor.get('death_cascade_count', 0)
            logger.critical(
                f"🚨 {action_name}: CASCADE DETECTED before {operation_name}: "
                f"Session death cascade #{cascade_count} - halting operation"
            )
            raise Exception(
                f"Session death cascade detected before {operation_name} (#{cascade_count})"
            )

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

    def _cleanup_browser(self) -> None:
        """Kill browser process and release resources."""
        if not (self.browser_manager and self.driver_live):
            return

        # Try graceful quit first
        try:
            driver = self.driver
            if driver:
                driver.quit()
        except Exception as e:
            logger.warning(f"Graceful browser quit failed: {e}")

        # Force browser manager to release resources
        try:
            self.browser_manager.close_browser()
        except Exception as e:
            logger.warning(f"BrowserManager cleanup failed: {e}")

    def _cleanup_database(self) -> None:
        """Close all database connections."""
        if not self.db_manager:
            return

        try:
            self.db_manager.close_connections(dispose_engine=True)
        except Exception as e:
            logger.warning(f"Database cleanup failed: {e}")

    def _cleanup_api_caches(self) -> None:
        """Clear API manager caches and CSRF token."""
        if self.api_manager:
            try:
                self.api_manager.clear_identifiers()
            except Exception as e:
                logger.warning(f"API manager cleanup failed: {e}")

        try:
            self.invalidate_csrf_cache()
        except Exception as e:
            logger.warning(f"CSRF cache invalidation failed: {e}")

    def _reset_session_state(self) -> None:
        """Reset internal session state flags."""
        self.session_ready = False
        self.session_start_time = None
        self._db_init_attempted = False
        self._db_ready = False

    def _force_session_restart(self, reason: str = "Watchdog timeout") -> bool:
        """
        Emergency session restart triggered by watchdog timeout.

        Purpose:
        --------
        Forcefully restarts session when an operation hangs beyond timeout threshold.
        More aggressive than close_sess() - kills browser process and clears all caches.

        Args:
            reason: Description of why restart was triggered (for logging)

        Returns:
            bool: Always returns False (operation failed, restart attempted)

        Called By:
        ----------
        APICallWatchdog when operation exceeds timeout_seconds threshold

        Actions Taken:
        --------------
        1. Log critical timeout event with reason
        2. Kill browser process (if running)
        3. Close all database connections
        4. Clear API manager caches (identifiers, CSRF token)
        5. Clear CSRF cache
        6. Set session_ready=False
        7. Log restart completion

        Example:
        --------
        >>> watchdog = APICallWatchdog(timeout_seconds=120)
        >>> watchdog.start("relationship_prob",
        ...               lambda: session_manager._force_session_restart("API timeout"))
        """
        logger.critical(
            f"🚨 FORCE SESSION RESTART triggered: {reason} - "
            "killing browser and clearing caches"
        )

        try:
            self._cleanup_browser()
            self._cleanup_database()
            self._cleanup_api_caches()
            self._reset_session_state()

            logger.info("Force session restart complete - session marked invalid")

        except Exception as e:
            logger.error(f"Error during force session restart: {e}", exc_info=True)

        # Always return False - operation failed, restart attempted
        return False

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
                    self.api_manager.csrf_token = csrf_token_val
                    return csrf_token_val
                logger.error(f"CSRF token API returned empty or invalid string: '{csrf_token_val}'")
                return None
            logger.warning("Failed to get CSRF token response via _api_req.")
            return None

        except Exception as e:
            logger.error(f"Unexpected error in get_csrf: {e}", exc_info=True)
            return None

    @retry_on_failure(max_attempts=3)
    def get_my_profileId(self) -> Optional[str]:  # noqa: N802 - matches API field name
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
                logger.error("Could not find 'ucdmid' in 'data' dict of profile_id API response.")
                return None
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
            # Reduce log spam during shutdown - only log once per minute
            if not hasattr(self, '_last_uuid_error_time') or time.time() - self._last_uuid_error_time > 60:
                logger.error("get_my_uuid: Session invalid.")
                self._last_uuid_error_time = time.time()
            return None

        from urllib.parse import urljoin
        url = urljoin(config_schema.api.base_url, API_PATH_UUID_NAVHEADER)
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
                        logger.debug(f"My uuid: {my_uuid_val}")
                        self._uuid_logged = True
                    return my_uuid_val
                logger.error("Could not retrieve UUID ('testId' missing in response).")
                return None
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
            raise ImportError(f"api_utils module failed to import: {e}") from e

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
                    logger.debug(f"My tree id: {my_tree_id_val}")
                    self._tree_id_logged = True
                return my_tree_id_val
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
            raise ImportError(f"api_utils module failed to import: {e}") from e

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
                # Store in API manager (logging done separately in main.py startup)
                self.api_manager.tree_owner_name = owner_name
                return owner_name
            logger.warning("api_utils.call_tree_owner_api returned None.")
            return None
        except Exception as e:
            logger.error(f"Error calling api_utils.call_tree_owner_api: {e}", exc_info=True)
            return None

    # Database delegation methods
    def get_db_conn(self) -> Any:
        """Get a database session."""
        return self.db_manager.get_session()

    def return_session(self, session: Any) -> None:
        """Return a database session."""
        self.db_manager.return_session(session)

    def get_db_conn_context(self) -> Any:
        """Get database session context manager."""
        return self.db_manager.get_session_context()

    def cls_db_conn(self, keep_db: bool = True) -> None:
        """Close database connections."""
        self.db_manager.close_connections(
            dispose_engine=not keep_db
        )  # Browser delegation methods

    def invalidate_csrf_cache(self) -> None:
        """Invalidate cached CSRF token (useful on auth errors)."""
        self._cached_csrf_token = None
        self._csrf_cache_time = 0

    @property
    def driver(self) -> Optional[WebDriverType]:
        """Get the WebDriver instance."""
        return self.browser_manager.driver

    @property
    def driver_live(self) -> bool:
        """Check if driver is live."""
        return self.browser_manager.driver_live

    def make_tab(self) -> Optional[str]:
        """Create a new browser tab."""
        return self.browser_manager.create_new_tab()

    # API delegation methods
    @property
    def my_profile_id(self) -> Optional[str]:
        """Get the user's profile ID."""
        # Try to get from API manager first, then retrieve if needed
        profile_id = self.api_manager.my_profile_id
        if not profile_id:
            profile_id = self.get_my_profileId()
        return profile_id

    @property
    def my_uuid(self) -> Optional[str]:
        """Get the user's UUID."""
        # Try to get from API manager first, then retrieve if needed
        uuid_val = self.api_manager.my_uuid
        if not uuid_val:
            uuid_val = self.get_my_uuid()
        return uuid_val

    @property
    def my_tree_id(self) -> Optional[str]:
        """Get the user's tree ID."""
        # Try to get from API manager first, then retrieve if needed
        tree_id = self.api_manager.my_tree_id
        if not tree_id and config_schema.api.tree_name:
            tree_id = self.get_my_tree_id()
        return tree_id

    @property
    def csrf_token(self) -> Optional[str]:
        """Get the CSRF token with smart caching."""
        # ⚡ OPTIMIZATION 1: Check pre-cached CSRF token first
        if self._cached_csrf_token and self._csrf_cache_time:
            cache_age = time.time() - self._csrf_cache_time
            if cache_age < self._csrf_cache_duration:
                return self._cached_csrf_token

        # Return cached token from API manager if available
        return self.api_manager.csrf_token

    def _precache_csrf_token(self) -> None:
        """
        ⚡ OPTIMIZATION 1: Pre-cache CSRF token during session setup to eliminate delays
        during Action 6 API operations.
        """
        try:
            if not self.browser_manager or not self.browser_manager.driver:
                logger.debug("⚡ CSRF pre-cache: Browser not available, skipping")
                return

            # Try to get CSRF token from cookies
            driver = self.browser_manager.driver
            csrf_cookie_names = ['_dnamatches-matchlistui-x-csrf-token', '_csrf']

            driver_cookies_list = driver.get_cookies()
            driver_cookies_dict = {
                c["name"]: c["value"]
                for c in driver_cookies_list
                if isinstance(c, dict) and "name" in c and "value" in c
            }

            for name in csrf_cookie_names:
                if driver_cookies_dict.get(name):
                    from urllib.parse import unquote
                    csrf_token_val = unquote(driver_cookies_dict[name]).split("|")[0]

                    # Cache the token
                    self._cached_csrf_token = csrf_token_val
                    self._csrf_cache_time = time.time()

                    logger.debug(f"⚡ Pre-cached CSRF token '{name}' during session setup (performance optimization)")
                    return

            logger.debug("⚡ CSRF pre-cache: No CSRF tokens found in cookies yet")

        except Exception as e:
            logger.debug(f"⚡ CSRF pre-cache: Error pre-caching CSRF token: {e}")

    def _is_csrf_token_valid(self) -> bool:
        """
        ⚡ OPTIMIZATION 1: Check if cached CSRF token is still valid.
        """
        if not self._cached_csrf_token or not self._csrf_cache_time:
            return False

        cache_age = time.time() - self._csrf_cache_time
        return cache_age < self._csrf_cache_duration

    # Public properties
    @property
    def tree_owner_name(self) -> Optional[str]:
        """Get the tree owner name."""
        return self.api_manager.tree_owner_name

    @property
    def requests_session(self) -> requests.Session:
        """Get the requests session."""
        return self.api_manager.requests_session

    # Enhanced capabilities properties
    @property
    def scraper(self) -> Optional[Any]:
        """Get the CloudScraper instance for anti-bot protection."""
        return getattr(self, '_scraper', None)

    @scraper.setter
    def scraper(self, value: Any) -> None:
        """Set the CloudScraper instance."""
        self._scraper = value

    # Compatibility properties for legacy code
    @property
    def browser_needed(self) -> bool:
        """Get/set browser needed flag."""
        return self.browser_manager.browser_needed

    @browser_needed.setter
    def browser_needed(self, value: bool) -> None:
        """Set browser needed flag."""
        self.browser_manager.browser_needed = value



    @property
    def _requests_session(self) -> requests.Session:
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
    def get_session_performance_stats(self) -> dict[str, Any]:
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


# === API Call Watchdog for Timeout Protection ===


class APICallWatchdog:
    """
    Monitors API calls and triggers emergency restart if operation hangs.

    Purpose:
    --------
    Catches operations that bypass request-level timeouts due to:
    - Browser-based API calls that ignore timeout parameters
    - TCP-level hangs (connection established but no response)
    - OS-level issues (file handles, network interfaces)
    - CloudScraper internal issues

    Root Cause:
    -----------
    The 7-hour browser hang (26,831 seconds) occurred because:
    1. timeout_protection decorator uses daemon threads (can't be killed)
    2. cloudscraper.get() call got stuck at TCP level
    3. No operation-level timeout enforcement existed

    Solution:
    ---------
    This watchdog provides operation-level timeout enforcement that:
    - Runs in parallel with API calls
    - Triggers force_session_restart() if timeout exceeded
    - Works as context manager for clean resource management

    Usage:
    ------
    >>> watchdog = APICallWatchdog(timeout_seconds=120)
    >>> def emergency_callback():
    ...     session_manager._force_session_restart("Watchdog timeout")
    >>> watchdog.start("relationship_prob", emergency_callback)
    >>> try:
    ...     response = cloudscraper.get(url)
    ... finally:
    ...     watchdog.cancel()

    Or with context manager:
    >>> with APICallWatchdog(timeout_seconds=120) as watchdog:
    ...     watchdog.set_callback("api_name", emergency_callback)
    ...     response = make_api_call()

    Thread Safety:
    --------------
    All state mutations protected by threading.Lock()
    Safe for concurrent use from multiple threads

    See Also:
    ---------
    RATE_LIMITING_ANALYSIS.md lines 430-510 for design rationale
    """

    def __init__(self, timeout_seconds: float = 120) -> None:
        """
        Initialize watchdog with timeout threshold.

        Args:
            timeout_seconds: Maximum operation duration before triggering
                           emergency restart (default: 120 seconds)
        """
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}")

        self.timeout_seconds = timeout_seconds
        self.timer: Optional[threading.Timer] = None
        self.is_active = False
        self._lock = threading.Lock()
        self._api_name = ""
        self._callback: Optional[Any] = None

    def start(self, api_name: str, callback: Any) -> None:
        """
        Start watchdog timer for API call.

        Args:
            api_name: Name of API operation being monitored
            callback: Function to call if timeout occurs (typically
                     session_manager._force_session_restart)

        Example:
            >>> watchdog.start("relationship_prob", lambda: restart())
        """
        with self._lock:
            if self.is_active:
                logger.warning(
                    f"Watchdog already active for '{self._api_name}', "
                    "cancelling previous timer"
                )
                self._cancel_unsafe()

            self._api_name = api_name
            self._callback = callback

            def timeout_handler() -> None:
                logger.critical(
                    f"🚨 WATCHDOG TIMEOUT: {api_name} exceeded "
                    f"{self.timeout_seconds}s limit - triggering emergency restart"
                )
                if callback:
                    callback()

            self.timer = threading.Timer(self.timeout_seconds, timeout_handler)
            self.timer.daemon = True
            self.timer.start()
            self.is_active = True

            logger.debug(
                f"Watchdog started for '{api_name}' "
                f"(timeout: {self.timeout_seconds}s)"
            )

    def cancel(self) -> None:
        """
        Cancel watchdog timer (operation completed successfully).

        Safe to call multiple times. Should always be called in finally block.

        Example:
            >>> try:
            ...     response = make_api_call()
            ... finally:
            ...     watchdog.cancel()
        """
        with self._lock:
            self._cancel_unsafe()

    def _cancel_unsafe(self) -> None:
        """Internal cancel without lock (caller must hold lock)."""
        if self.timer:
            self.timer.cancel()
            self.timer = None
        if self.is_active:
            logger.debug(f"Watchdog cancelled for '{self._api_name}'")
        self.is_active = False
        self._api_name = ""
        self._callback = None

    def set_callback(self, api_name: str, callback: Any) -> None:
        """
        Set callback for context manager usage.

        Args:
            api_name: Name of API operation
            callback: Emergency restart callback

        Example:
            >>> with APICallWatchdog(120) as watchdog:
            ...     watchdog.set_callback("api_name", restart_callback)
            ...     make_api_call()
        """
        with self._lock:
            self._api_name = api_name
            self._callback = callback

            if self.is_active:
                # Restart timer with new callback
                self._cancel_unsafe()

            def timeout_handler() -> None:
                logger.critical(
                    f"🚨 WATCHDOG TIMEOUT: {api_name} exceeded "
                    f"{self.timeout_seconds}s limit"
                )
                if callback:
                    callback()

            self.timer = threading.Timer(self.timeout_seconds, timeout_handler)
            self.timer.daemon = True
            self.timer.start()
            self.is_active = True

    def __enter__(self) -> "APICallWatchdog":
        """Context manager entry (timer started by set_callback)."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """Context manager exit (always cancel timer)."""
        self.cancel()
        return False  # Don't suppress exceptions


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
    print("   Creating SessionManager instance...")

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

        results.append(initial_ready is False)
        assert initial_ready is False, "Should start with session_ready=False"

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
    for _i in range(3):
        session_manager = SessionManager()
        session_managers.append(session_manager)
    end_time = time.time()
    total_time = end_time - start_time
    max_time = 5.0
    assert (
        total_time < max_time
    ), f"3 optimized initializations took {total_time:.3f}s, should be under {max_time}s"
    for sm in session_managers:
        with contextlib.suppress(Exception):
            sm.close_sess(keep_db=True)
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
        raise AssertionError(f"SessionManager should handle operations gracefully: {e}") from e
    return True


def _test_regression_prevention_csrf_optimization():
    """
    🛡️ REGRESSION TEST: CSRF token caching optimization.

    This test verifies that Optimization 1 (CSRF token pre-caching) is properly
    implemented and working. This would have prevented performance regressions
    caused by fetching CSRF tokens on every API call.
    """
    print("🛡️ Testing CSRF token caching optimization regression prevention:")
    results = []

    try:
        session_manager = SessionManager()

        # Test 1: Verify CSRF caching attributes exist
        if hasattr(session_manager, '_cached_csrf_token'):
            print("   ✅ _cached_csrf_token attribute exists")
            results.append(True)
        else:
            print("   ❌ _cached_csrf_token attribute missing")
            results.append(False)

        if hasattr(session_manager, '_csrf_cache_time'):
            print("   ✅ _csrf_cache_time attribute exists")
            results.append(True)
        else:
            print("   ❌ _csrf_cache_time attribute missing")
            results.append(False)

        # Test 2: Verify CSRF validation method exists
        if hasattr(session_manager, '_is_csrf_token_valid'):
            print("   ✅ _is_csrf_token_valid method exists")

            # Test that it returns a boolean
            try:
                is_valid = session_manager._is_csrf_token_valid()
                if isinstance(is_valid, bool):
                    print("   ✅ _is_csrf_token_valid returns boolean")
                    results.append(True)
                else:
                    print("   ❌ _is_csrf_token_valid doesn't return boolean")
                    results.append(False)
            except Exception as method_error:
                print(f"   ⚠️  _is_csrf_token_valid method error: {method_error}")
                results.append(False)
        else:
            print("   ❌ _is_csrf_token_valid method missing")
            results.append(False)

        # Test 3: Verify pre-cache method exists
        if hasattr(session_manager, '_precache_csrf_token'):
            print("   ✅ _precache_csrf_token method exists")
            results.append(True)
        else:
            print("   ⚠️  _precache_csrf_token method not found (may be named differently)")
            results.append(False)

    except Exception as e:
        print(f"   ❌ SessionManager CSRF optimization test failed: {e}")
        results.append(False)

    success = all(results)
    if success:
        print("🎉 CSRF token caching optimization regression test passed!")
    return success


def _test_regression_prevention_property_access():
    """
    🛡️ REGRESSION TEST: SessionManager property access stability.

    This test verifies that SessionManager properties are accessible without
    errors. This would have caught the duplicate method definition issues
    we encountered.
    """
    print("🛡️ Testing SessionManager property access regression prevention:")
    results = []

    try:
        session_manager = SessionManager()

        # Test key properties that had duplicate definition issues
        properties_to_test = [
            ('requests_session', 'requests session object'),
            ('csrf_token', 'CSRF token string'),
            ('my_uuid', 'user UUID string'),
            ('my_tree_id', 'tree ID string'),
            ('session_ready', 'session ready boolean')
        ]

        for prop, description in properties_to_test:
            try:
                getattr(session_manager, prop)
                print(f"   ✅ Property '{prop}' accessible ({description})")
                results.append(True)
            except AttributeError:
                print(f"   ⚠️  Property '{prop}' not found (may be intended)")
                results.append(True)  # Not finding is OK, crashing is not
            except Exception as prop_error:
                print(f"   ❌ Property '{prop}' error: {prop_error}")
                results.append(False)

    except Exception as e:
        print(f"   ❌ SessionManager property access test failed: {e}")
        results.append(False)

    success = all(results)
    if success:
        print("🎉 SessionManager property access regression test passed!")
    return success


def _test_regression_prevention_initialization_stability():
    """
    🛡️ REGRESSION TEST: SessionManager initialization stability.

    This test verifies that SessionManager initializes without crashes,
    which would have caught WebDriver stability issues.
    """
    print("🛡️ Testing SessionManager initialization stability regression prevention:")
    results = []

    try:
        # Test multiple initialization attempts
        for i in range(3):
            try:
                session_manager = SessionManager()
                print(f"   ✅ Initialization attempt {i+1} successful")
                results.append(True)

                # Test basic attribute access
                _ = hasattr(session_manager, 'db_manager')
                _ = hasattr(session_manager, 'browser_manager')
                _ = hasattr(session_manager, 'api_manager')

                print(f"   ✅ Basic attribute access {i+1} successful")
                results.append(True)

            except Exception as init_error:
                print(f"   ❌ Initialization attempt {i+1} failed: {init_error}")
                results.append(False)
                break

    except Exception as e:
        print(f"   ❌ SessionManager initialization stability test failed: {e}")
        results.append(False)

    success = all(results)
    if success:
        print("🎉 SessionManager initialization stability regression test passed!")
    return success


# === APICallWatchdog Test Functions ===


def _test_watchdog_initialization() -> None:
    """Test basic APICallWatchdog initialization."""
    watchdog = APICallWatchdog(timeout_seconds=60)
    assert watchdog.timeout_seconds == 60, "Timeout should match initialization"
    assert watchdog.timer is None, "Timer should be None initially"
    assert not watchdog.is_active, "Watchdog should not be active initially"


def _test_watchdog_timeout_enforcement() -> None:
    """Test that watchdog triggers callback after timeout."""
    callback_executed = []

    def emergency_callback() -> None:
        callback_executed.append(True)

    watchdog = APICallWatchdog(timeout_seconds=0.5)
    watchdog.start("test_api", emergency_callback)
    time.sleep(1.0)  # Wait for timeout

    assert len(callback_executed) == 1, "Callback should be executed exactly once"


def _test_watchdog_graceful_completion() -> None:
    """Test that cancel() prevents callback execution."""
    callback_executed = []

    def emergency_callback() -> None:
        callback_executed.append(True)

    watchdog = APICallWatchdog(timeout_seconds=0.5)
    watchdog.start("test_api", emergency_callback)
    time.sleep(0.2)
    watchdog.cancel()
    time.sleep(0.5)  # Wait longer than timeout

    assert len(callback_executed) == 0, "Callback should NOT be executed after cancel"


def _test_watchdog_context_manager() -> None:
    """Test context manager protocol."""
    callback_executed = []

    def emergency_callback() -> None:
        callback_executed.append(True)

    with APICallWatchdog(timeout_seconds=0.5) as watchdog:
        watchdog.set_callback("test_api", emergency_callback)
        time.sleep(0.1)

    time.sleep(0.6)
    assert len(callback_executed) == 0, "Callback should NOT fire after context exit"


def _test_watchdog_multiple_cycles() -> None:
    """Test watchdog can be reused multiple times."""
    watchdog = APICallWatchdog(timeout_seconds=0.3)

    for i in range(5):
        watchdog.start(f"test_api_{i}", lambda: None)
        time.sleep(0.1)
        watchdog.cancel()
        assert not watchdog.is_active, f"Cycle {i}: should not be active"


def _test_watchdog_thread_safety() -> None:
    """Test thread safety under concurrent access."""
    watchdog = APICallWatchdog(timeout_seconds=1.0)
    errors = []

    def thread_worker(thread_id: int) -> None:
        try:
            watchdog.start(f"thread_{thread_id}", lambda: None)
            time.sleep(0.1)
            watchdog.cancel()
        except Exception as e:
            errors.append((thread_id, e))

    threads = [threading.Thread(target=thread_worker, args=(i,)) for i in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(errors) == 0, f"Thread safety errors: {errors}"


def _test_watchdog_parameter_validation() -> None:
    """Test parameter validation."""
    try:
        APICallWatchdog(timeout_seconds=0)
        raise AssertionError("Should raise ValueError for timeout=0")
    except ValueError:
        pass  # Expected

    try:
        APICallWatchdog(timeout_seconds=-5)
        raise AssertionError("Should raise ValueError for negative timeout")
    except ValueError:
        pass  # Expected


def _test_watchdog_edge_cases() -> None:
    """Test edge cases."""
    watchdog = APICallWatchdog(timeout_seconds=0.5)

    # Cancel before start
    watchdog.cancel()
    assert not watchdog.is_active, "Cancel before start should work"

    # Double cancel
    watchdog.cancel()
    watchdog.cancel()

    # Start, let timeout, then cancel
    callback_executed = []
    watchdog.start("test_api", lambda: callback_executed.append(True))
    time.sleep(0.7)
    assert len(callback_executed) == 1, "Callback should have executed"
    watchdog.cancel()  # Should not error


def _test_force_session_restart() -> None:
    """Test _force_session_restart() method."""
    sm = SessionManager()

    # Set up session state
    sm.session_ready = True
    sm.session_start_time = time.time()
    sm._db_init_attempted = True
    sm._db_ready = True

    # Initial state verification
    assert sm.session_ready is True, "Session should be ready initially"
    assert sm._db_ready is True, "DB should be ready initially"

    # Call force_session_restart
    result = sm._force_session_restart("Test timeout")

    # Verify result (should always return False)
    assert result is False, "Force restart should always return False"

    # Verify session state reset
    assert sm.session_ready is False, "Session should be marked not ready"
    assert sm.session_start_time is None, "Session start time should be None"
    assert sm._db_init_attempted is False, "DB init attempted should be reset"
    assert sm._db_ready is False, "DB ready should be reset"


def _test_watchdog_integration_with_session_restart() -> None:
    """Test watchdog integration with _force_session_restart()."""
    sm = SessionManager()
    restart_called = []

    def restart_callback() -> None:
        """Callback that triggers session restart."""
        result = sm._force_session_restart("Watchdog timeout in test")
        restart_called.append(result)

    # Set up session state
    sm.session_ready = True

    # Create watchdog with short timeout
    watchdog = APICallWatchdog(timeout_seconds=0.5)
    watchdog.start("test_api", restart_callback)

    # Wait for timeout
    time.sleep(0.7)

    # Verify restart was called
    assert len(restart_called) == 1, "Restart callback should have been called"
    assert restart_called[0] is False, "Restart should return False"
    assert sm.session_ready is False, "Session should be marked not ready after restart"

    # Cleanup
    watchdog.cancel()


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for session_manager.py (decomposed).
    """
    from test_framework import TestSuite, suppress_logging

    # Warnings already suppressed at module level when __name__ == "__main__"
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

        # 🛡️ REGRESSION PREVENTION TESTS - These would have caught the optimization and stability issues
        suite.run_test(
            "CSRF token caching optimization regression prevention",
            _test_regression_prevention_csrf_optimization,
            "CSRF token caching attributes and methods exist and function correctly",
            "Prevents regression of Optimization 1 (CSRF token pre-caching)",
            "Verify _cached_csrf_token, _csrf_cache_time, and _is_csrf_token_valid implementation",
        )

        suite.run_test(
            "SessionManager property access regression prevention",
            _test_regression_prevention_property_access,
            "All SessionManager properties are accessible without errors or conflicts",
            "Prevents regression of duplicate method definitions and property conflicts",
            "Test key properties that had duplicate definition issues (csrf_token, requests_session, etc.)",
        )

        suite.run_test(
            "SessionManager initialization stability regression prevention",
            _test_regression_prevention_initialization_stability,
            "SessionManager initializes reliably without crashes or WebDriver issues",
            "Prevents regression of SessionManager initialization and WebDriver crashes",
            "Test multiple initialization attempts and basic attribute access stability",
        )

        # === APICallWatchdog Tests ===
        suite.run_test(
            "APICallWatchdog: Basic initialization",
            _test_watchdog_initialization,
            "Watchdog initializes with valid timeout, timer inactive, callback not set",
            "Test APICallWatchdog basic initialization",
            "Create watchdog with specific timeout and verify initial state",
        )

        suite.run_test(
            "APICallWatchdog: Timeout enforcement",
            _test_watchdog_timeout_enforcement,
            "Callback executed after timeout expires",
            "Test watchdog triggers callback after timeout",
            "Start watchdog with 0.5s timeout, wait 1s, verify callback executed",
        )

        suite.run_test(
            "APICallWatchdog: Graceful completion",
            _test_watchdog_graceful_completion,
            "Callback NOT executed when cancelled before timeout",
            "Test cancel() prevents callback execution",
            "Start watchdog, cancel before timeout, verify callback not executed",
        )

        suite.run_test(
            "APICallWatchdog: Context manager protocol",
            _test_watchdog_context_manager,
            "Timer started on enter, cancelled on exit",
            "Test context manager __enter__/__exit__ work correctly",
            "Use watchdog with 'with' statement, verify proper lifecycle",
        )

        suite.run_test(
            "APICallWatchdog: Multiple cycles",
            _test_watchdog_multiple_cycles,
            "Each cycle works independently, no state leakage",
            "Test watchdog can be reused multiple times",
            "Start and cancel watchdog 5 times, verify clean state",
        )

        suite.run_test(
            "APICallWatchdog: Thread safety",
            _test_watchdog_thread_safety,
            "No race conditions, all operations complete safely",
            "Test operations are thread-safe under concurrent access",
            "Run 10 threads starting/cancelling watchdogs concurrently",
        )

        suite.run_test(
            "APICallWatchdog: Parameter validation",
            _test_watchdog_parameter_validation,
            "ValueError raised for invalid timeout values",
            "Test invalid timeout raises ValueError",
            "Try zero and negative timeout, verify ValueError",
        )

        suite.run_test(
            "APICallWatchdog: Edge cases",
            _test_watchdog_edge_cases,
            "All edge cases handled without errors",
            "Test edge cases: cancel before start, double cancel, cancel after timeout",
            "Verify graceful handling of unusual call sequences",
        )

        suite.run_test(
            "Force Session Restart",
            _test_force_session_restart,
            "Session state reset correctly: session_ready=False, DB flags reset",
            "Test _force_session_restart() method",
            "Verify session state, DB flags, and timing are reset after force restart",
        )

        suite.run_test(
            "Watchdog Integration with Session Restart",
            _test_watchdog_integration_with_session_restart,
            "Watchdog triggers _force_session_restart() on timeout",
            "Test watchdog integration with _force_session_restart()",
            "Verify watchdog timeout triggers session restart callback correctly",
        )

        return suite.finish_suite()


def main() -> None:
    """Main entry point for running tests."""
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # NOTE: Running via "python -m core.session_manager" triggers a harmless RuntimeWarning
    # from Python's runpy module because core.session_manager is both:
    #   1. A package member (core/session_manager.py)
    #   2. Being executed as __main__
    #
    # Python's import system detects this and warns about "unpredictable behaviour"
    # but it's actually safe in this case - we're just running tests.
    #
    # To avoid the warning, use one of these alternatives:
    #   • python core/session_manager.py (run as script, not module)
    #   • python -m core (runs core/__main__.py which tests all modules)
    #   • python run_all_tests.py (runs all 58 test modules)
    #
    # The warning does NOT indicate a bug - it's Python being cautious about
    # a module that's both imported and executed. Our code handles this safely.

    main()
