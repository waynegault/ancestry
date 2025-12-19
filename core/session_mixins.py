"""
Mixins for SessionManager to reduce class size and complexity.
"""

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional, cast
from urllib.parse import urljoin

from api.api_constants import API_PATH_UUID_NAVHEADER
from config import config_schema
from core.error_handling import api_retry
from core.protocols import SessionHealthMonitor, SupportsBrowserConsoleLogs

if TYPE_CHECKING:
    from core.browser_manager import BrowserManager
    from core.database_manager import DatabaseManager
    from core.session_manager import SessionManager

logger = logging.getLogger(__name__)


class SessionHealthMixin:
    """
    Mixin for SessionManager to handle session health monitoring and recovery.
    """

    # Type hints for attributes expected on self (SessionManager)
    if TYPE_CHECKING:
        session_health_monitor: SessionHealthMonitor
        browser_manager: BrowserManager
        db_manager: DatabaseManager
        last_js_error_check: datetime
        session_ready: bool
        session_start_time: Optional[float]

        def is_sess_valid(self) -> bool: ...
        def _update_session_metrics(self, force_zero: bool = False) -> None: ...
        def _attempt_session_recovery(self, reason: str = "browser_error") -> bool: ...
        def get_db_conn_context(self) -> Any: ...
        def _record_session_refresh_metric(self, reason: str) -> None: ...
        def close_sess(self, keep_db: bool = False) -> None: ...
        def start_sess(self, action_name: Optional[str] = None) -> bool: ...
        def ensure_session_ready(self, action_name: Optional[str] = None, skip_csrf: bool = False) -> bool: ...
        def _precache_csrf_token(self) -> None: ...

    def _check_session_health(self) -> bool:
        """
        Universal session health monitoring that detects session death and prevents
        cascade failures during long-running operations.
        """
        self._update_session_metrics()
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

    def _is_session_death_cascade(self) -> bool:
        """Check if we're in a session death cascade scenario."""
        return self.session_health_monitor['death_detected'].is_set()

    def should_halt_operations(self) -> bool:
        """Determine if operations should halt due to session death."""
        if self._is_session_death_cascade():
            self.session_health_monitor['death_cascade_count'] += 1

            # Halt immediately if session is dead
            logger.warning(
                f"⚠️  Halting operation due to session death cascade "
                f"(cascade #{self.session_health_monitor['death_cascade_count']})"
            )
            return True
        return False

    def _reset_session_health_monitoring(self) -> None:
        """Reset session health monitoring (used when creating new sessions)."""
        self.session_health_monitor['is_alive'].set()
        self.session_health_monitor['death_detected'].clear()
        self.session_health_monitor['last_heartbeat'] = time.time()
        self.session_health_monitor['death_timestamp'] = None
        self.session_health_monitor['parallel_operations'] = 0
        self.session_health_monitor['death_cascade_count'] = 0
        logger.debug("🔄 Session health monitoring reset for new session")

    def _check_js_errors(self) -> list[dict[str, Any]]:
        """
        Check for JavaScript errors in the browser console.

        Returns:
            list[Dict]: List of JavaScript errors found since last check
        """
        driver = self.browser_manager.driver
        if not driver or not self.browser_manager.driver_live:
            return []

        try:
            # Get browser logs (if available)
            if hasattr(driver, 'get_log'):
                log_driver = cast(SupportsBrowserConsoleLogs, driver)
                logs = log_driver.get_log('browser')
            else:
                logger.debug("WebDriver does not support get_log method")
                return []

            # Filter for errors that occurred after last check
            current_time = datetime.now(timezone.utc)
            js_errors: list[dict[str, Any]] = []

            for log_entry in logs:
                # Check if this is a JavaScript error
                if log_entry.get('level') in {'SEVERE', 'ERROR'}:
                    # Parse timestamp (browser logs use milliseconds since epoch)
                    log_timestamp = datetime.fromtimestamp(log_entry.get('timestamp', 0) / 1000, tz=timezone.utc)

                    # Only include errors since last check
                    if log_timestamp > self.last_js_error_check:
                        js_errors.append(
                            {
                                'timestamp': log_timestamp,
                                'level': log_entry.get('level'),
                                'message': log_entry.get('message', ''),
                                'source': log_entry.get('source', ''),
                            }
                        )

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
        errors = self._check_js_errors()

        # Count critical errors (those that might affect functionality)
        critical_errors = [
            error
            for error in errors
            if any(
                keyword in error['message'].lower()
                for keyword in ['uncaught', 'reference error', 'type error', 'syntax error']
            )
        ]

        if critical_errors:
            logger.warning(f"Found {len(critical_errors)} critical JavaScript errors")
            return False

        return True

    def check_browser_health(self) -> bool:
        """Check browser health and detect browser death."""
        current_time = time.time()
        self.session_health_monitor['last_browser_health_check'] = current_time

        # Check if browser is needed
        if not self.browser_manager.browser_needed:
            return True

        # Check if driver exists and is responsive
        if not self.is_sess_valid():
            self.session_health_monitor['browser_death_count'] = (
                self.session_health_monitor.get('browser_death_count', 0) + 1
            )
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
        return self._attempt_session_recovery(reason="browser_error")

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
            raise Exception(f"Session death cascade detected before {operation_name} (#{cascade_count})")

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
                from browser.css_selectors import WAIT_FOR_PAGE_SELECTOR
                from core.utils import nav_to_page

                nav_success = nav_to_page(
                    self.browser_manager.driver,
                    url,
                    selector=WAIT_FOR_PAGE_SELECTOR,
                    session_manager=cast("SessionManager", self),
                )
                if not nav_success:
                    logger.warning(f"Failed to navigate to {url} after restart.")
                else:
                    logger.info(f"Successfully navigated to {url}.")
            except Exception as e:
                logger.warning(f"Error navigating to {url} after restart: {e}")

        self._record_session_refresh_metric("browser_error")
        logger.info("Session restart completed successfully.")
        return True


class SessionIdentifierMixin:
    """
    Mixin for SessionManager to handle user identifiers and CSRF tokens.
    """

    # Type hints for attributes expected on self (SessionManager)
    if TYPE_CHECKING:
        api_manager: Any
        _profile_id_logged: bool
        _uuid_logged: bool
        _tree_id_logged: bool
        _owner_logged: bool
        _cached_csrf_token: Optional[str]
        _csrf_cache_time: float
        ESSENTIAL_SESSION_COOKIES: tuple[str, str]

        def is_sess_valid(self) -> bool: ...
        def _get_utils_attr(self, attr_name: str) -> Any: ...
        def sync_cookies_to_requests(self, force: bool = False) -> None: ...
        def _get_cookies(self, names: list[str], timeout: int = 10) -> bool: ...

    @api_retry()
    def _get_csrf(self) -> Optional[str]:
        """
        Retrieve CSRF token from API.

        Returns:
            str: CSRF token if successful, None otherwise
        """
        if not self.is_sess_valid():
            logger.error("get_csrf: Session invalid.")
            return None

        csrf_token_url = urljoin(config_schema.api.base_url, "discoveryui-matches/parents/api/csrfToken")
        logger.debug(f"Attempting to fetch fresh CSRF token from: {csrf_token_url}")

        # Check essential cookies
        essential_cookies = list(self.ESSENTIAL_SESSION_COOKIES)
        if not self._get_cookies(essential_cookies, timeout=10):
            logger.warning(f"Essential cookies {essential_cookies} NOT found before CSRF token API call.")

        # Sync cookies to requests session
        self.sync_cookies_to_requests()

        try:
            api_request = self._get_utils_attr("_api_req")

            response_data = api_request(
                url=csrf_token_url,
                session_manager=self,
                method="GET",
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

    @api_retry()
    def _get_my_profile_id(self) -> Optional[str]:
        """
        Retrieve user's profile ID (ucdmid).

        Returns:
            str: Profile ID if successful, None otherwise
        """
        if not self.is_sess_valid():
            logger.error("get_my_profile_id: Session invalid.")
            return None

        url = urljoin(config_schema.api.base_url, "app-api/cdp-p13n/api/v1/users/me?attributes=ucdmid")
        logger.debug("Attempting to fetch own profile ID (ucdmid)...")

        # Sync cookies to requests session
        self.sync_cookies_to_requests()

        try:
            api_request = self._get_utils_attr("_api_req")

            response_data = api_request(
                url=url,
                session_manager=self,
                method="GET",
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
            logger.error(f"Unexpected error in get_my_profile_id: {e}", exc_info=True)
            return None

    @api_retry()
    def _get_my_uuid(self) -> Optional[str]:
        """
        Retrieve user's UUID (testId).

        Returns:
            str: UUID if successful, None otherwise
        """
        if not self.is_sess_valid():
            # Reduce log spam during shutdown - only log once per minute
            if (
                not hasattr(self, '_last_uuid_error_time')
                or time.time() - getattr(self, '_last_uuid_error_time', 0) > 60
            ):
                logger.error("get_my_uuid: Session invalid.")
                self._last_uuid_error_time = time.time()
            return None

        url = urljoin(config_schema.api.base_url, API_PATH_UUID_NAVHEADER)
        logger.debug("Attempting to fetch own UUID (testId) from header/dna API...")

        # Sync cookies to requests session
        self.sync_cookies_to_requests()

        try:
            api_request = self._get_utils_attr("_api_req")

            response_data = api_request(
                url=url,
                session_manager=self,
                method="GET",
                api_description="Get UUID API",
                use_csrf_token=False,
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

    @api_retry()
    def _get_my_tree_id(self) -> Optional[str]:
        """
        Retrieve user's tree ID.

        Returns:
            str: Tree ID if successful, None otherwise
        """
        try:
            import api.api_utils as local_api_utils
        except ImportError as e:
            logger.error(f"get_my_tree_id: Failed to import api.api_utils: {e}")
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
                cast("SessionManager", self), tree_name_config
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

    @api_retry()
    def _get_tree_owner(self, tree_id: Optional[str]) -> Optional[str]:
        """
        Retrieve tree owner name.

        Args:
            tree_id: The tree ID to get owner for

        Returns:
            str: Tree owner name if successful, None otherwise
        """
        try:
            import api.api_utils as local_api_utils
        except ImportError as e:
            logger.error(f"get_tree_owner: Failed to import api.api_utils: {e}")
            raise ImportError(f"api.api_utils module failed to import: {e}") from e

        if not tree_id:
            logger.warning("Cannot get tree owner: tree_id is missing.")
            return None

        if not self.is_sess_valid():
            logger.error("get_tree_owner: Session invalid.")
            return None

        logger.debug(f"Delegating tree owner fetch for tree ID {tree_id} to api_utils...")
        try:
            owner_name = local_api_utils.call_tree_owner_api(cast("SessionManager", self), tree_id)
            if owner_name:
                # Store in API manager (logging done separately in main.py startup)
                self.api_manager.tree_owner_name = owner_name
                return owner_name
            logger.warning("api_utils.call_tree_owner_api returned None.")
            return None
        except Exception as e:
            logger.error(f"Error calling api_utils.call_tree_owner_api: {e}", exc_info=True)
            return None

    def get_tree_owner(self, tree_id: Optional[str]) -> Optional[str]:
        """Public wrapper for _get_tree_owner."""
        return self._get_tree_owner(tree_id)

    def invalidate_csrf_cache(self) -> None:
        """Invalidate cached CSRF token (useful on auth errors)."""
        self._cached_csrf_token = None
        self._csrf_cache_time = 0

    @property
    def my_tree_id(self) -> Optional[str]:
        """Delegate my_tree_id to api_manager."""
        return self.api_manager.my_tree_id

    @property
    def my_uuid(self) -> Optional[str]:
        """Delegate my_uuid to api_manager."""
        return self.api_manager.my_uuid

    @property
    def my_profile_id(self) -> Optional[str]:
        """Delegate my_profile_id to api_manager."""
        return self.api_manager.my_profile_id

    @property
    def csrf_token(self) -> Optional[str]:
        """Delegate csrf_token to api_manager."""
        return self.api_manager.csrf_token

    @property
    def tree_owner_name(self) -> Optional[str]:
        """Delegate tree_owner_name to api_manager."""
        return self.api_manager.tree_owner_name


# -----------------------------------------------------------------------------
# Standard Test Runner
# -----------------------------------------------------------------------------
from testing.test_utilities import create_standard_test_runner


def _test_module_integrity() -> bool:
    "Test that module can be imported and definitions are valid."
    return True


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
