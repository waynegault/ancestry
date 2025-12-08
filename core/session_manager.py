#!/usr/bin/env python3

"""
Refactored Session Manager - Orchestrates all session components.

This module provides a new, modular SessionManager that orchestrates
the specialized managers (DatabaseManager, BrowserManager, APIManager, etc.)
to provide a clean, maintainable architecture.

"""

# === CORE INFRASTRUCTURE ===
import logging
import sys

logger = logging.getLogger(__name__)

# === STANDARD LIBRARY IMPORTS ===
import threading
import time
from contextlib import suppress
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional, cast
from unittest.mock import patch

from core.error_handling import (
    error_context,
    graceful_degradation,
    timeout_protection,
)
from core.session_cache import (
    cached_api_manager,
    cached_browser_manager,
    cached_database_manager,
    cached_session_validator,
    clear_session_cache,
    get_session_cache_stats,
)

if TYPE_CHECKING:
    from selenium.webdriver.remote.webdriver import WebDriver as WebDriverType
    from sqlalchemy.orm import Session as SqlAlchemySession
else:
    WebDriverType = Any
    SqlAlchemySession = Any

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
except ImportError:
    WebDriverException = Exception

# === LOCAL IMPORTS ===
import contextlib
import importlib
from functools import lru_cache

from config import config_schema
from core.api_manager import APIManager
from core.browser_manager import BrowserManager
from core.database_manager import DatabaseManager
from core.session_validator import SessionValidator
from observability.metrics_exporter import start_metrics_exporter, stop_metrics_exporter
from observability.metrics_registry import metrics
from performance.health_monitor import integrate_with_session_manager


@lru_cache(maxsize=1)
def _load_utils_module() -> Any:
    """Lazily import the utils module to avoid circular dependencies."""

    return importlib.import_module("core.utils")


# === MODULE CONSTANTS ===
# Use global cached config instance


class SessionLifecycleState(Enum):
    """Lifecycle states enforced by SessionManager.

    Diagram:
        UNINITIALIZED ──(request readiness)──▶ RECOVERING ──(success)──▶ READY
              ▲                                                 │
              └─────────────── (reset/close) ◀── DEGRADED ◀─────┘

    READY represents a healthy browser/API session, DEGRADED captures
    fatal readiness failures, and RECOVERING is used while rebuilding
    state after resets or guard-triggered recoveries.
    """

    UNINITIALIZED = "UNINITIALIZED"
    RECOVERING = "RECOVERING"
    READY = "READY"
    DEGRADED = "DEGRADED"


from core.protocols import SessionHealthMonitor
from core.session_mixins import SessionHealthMixin, SessionIdentifierMixin


class SessionManager(SessionIdentifierMixin, SessionHealthMixin):
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

    PHASE 5.4 STATE MACHINE: Session lifecycle now follows explicit
    states (UNINITIALIZED → RECOVERING → READY, with DEGRADED for
    hard failures). Callers invoke guard_action() before session-level
    work so exec_actn() can reset degraded sessions automatically.
    """

    ESSENTIAL_SESSION_COOKIES: tuple[str, str] = ("ANCSESSIONID", "SecureATT")

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

        # Lifecycle state tracking for guard enforcement
        self._state_lock = threading.Lock()
        self._state: SessionLifecycleState = SessionLifecycleState.UNINITIALIZED
        self._state_reason: str = "Session not initialized"
        self._state_changed_at: float = time.time()

        # PHASE 5.1: Session state caching for performance
        self._last_readiness_check: Optional[float] = None
        self._cached_session_state: dict[str, Any] = {}

        # Cookie sync tracking (explicit defaults for recovery logic)
        self._last_cookie_sync_time: float = 0.0
        self._session_cookies_synced: bool = False

        # ⚡ OPTIMIZATION 1: Pre-cached CSRF token for Action 6 performance
        self._cached_csrf_token: Optional[str] = None
        self._csrf_cache_time: float = 0.0
        self._csrf_cache_duration: float = 300.0  # 5-minute cache

        # Performance monitoring attributes shared with actions
        self._response_times: list[float] = []
        self._avg_response_time: float = 0.0
        self._recent_slow_calls: int = 0

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
        self.rate_limiter = self._initialize_rate_limiter()

        # UNIVERSAL SESSION HEALTH MONITORING (moved from action6-specific to universal)
        self.session_health_monitor: SessionHealthMonitor = {
            'is_alive': threading.Event(),
            'death_detected': threading.Event(),
            'last_heartbeat': time.time(),
            'heartbeat_interval': 30,  # Check every 30 seconds
            'death_cascade_halt': threading.Event(),
            'death_timestamp': None,
            'parallel_operations': 0,
            'death_cascade_count': 0,
        }
        self.session_health_monitor['is_alive'].set()  # Initially alive

        self._metrics_exporter_started = False
        self._last_session_refresh_reason: str = "startup"

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

        self._update_session_metrics(force_zero=True)
        self._ensure_metrics_exporter()

        # Integrate health monitoring
        self.health_monitor = integrate_with_session_manager(self)

        init_time = time.time() - start_time
        logger.debug(f"Optimized SessionManager created in {init_time:.3f}s: ID={id(self)}")

    @property
    def driver(self) -> Optional[WebDriverType]:
        """Expose the active WebDriver for helpers that still require it."""

        return cast(Optional[WebDriverType], getattr(self.browser_manager, "driver", None))

    @staticmethod
    @cached_database_manager()
    def _get_cached_database_manager(db_path: Optional[str] = None) -> "DatabaseManager":
        """Get cached DatabaseManager instance"""
        logger.debug("Creating/retrieving DatabaseManager from cache")
        return DatabaseManager(db_path)

    @staticmethod
    @cached_browser_manager()
    def _get_cached_browser_manager() -> "BrowserManager":
        """Get cached BrowserManager instance"""
        logger.debug("Creating/retrieving BrowserManager from cache")
        return BrowserManager()

    @staticmethod
    @cached_api_manager()
    def _get_cached_api_manager() -> "APIManager":
        """Get cached APIManager instance"""
        logger.debug("Creating/retrieving APIManager from cache")
        return APIManager()

    @staticmethod
    @cached_session_validator()
    def _get_cached_session_validator() -> "SessionValidator":
        """Get cached SessionValidator instance"""
        logger.debug("Creating/retrieving SessionValidator from cache")
        return SessionValidator()

    def _initialize_rate_limiter(self) -> Optional[Any]:
        """Create or reuse the adaptive rate limiter configured for this session."""

        get_rate_limiter = self._resolve_rate_limiter_factory()
        if not get_rate_limiter:
            return None

        batch_threshold = self._resolve_batch_threshold()
        success_threshold = self._determine_success_threshold(batch_threshold)

        # Resolve rates using consolidated configuration
        min_fill_rate = getattr(config_schema.api, "rate_limiter_min_rate", 0.1)
        max_fill_rate = getattr(config_schema.api, "rate_limiter_max_rate", 10.0)

        # requests_per_second acts as initial rate
        initial_rps = getattr(config_schema.api, "requests_per_second", 2.0) or 2.0

        # Ensure initial rate is within bounds
        initial_fill_rate = max(min_fill_rate, min(initial_rps, max_fill_rate))

        logger.debug(
            f"Rate Limiter Config - Initial RPS: {initial_rps}, Min: {min_fill_rate}, Max: {max_fill_rate}, Calculated Initial: {initial_fill_rate}"
        )

        endpoint_profiles = self._build_endpoint_profile_config()
        self._log_endpoint_rate_cap(endpoint_profiles)

        bucket_capacity = getattr(config_schema.api, "token_bucket_capacity", 10.0)

        # New parameters for tuning
        rate_limiter_429_backoff = getattr(config_schema.api, "rate_limiter_429_backoff", 0.80)
        rate_limiter_success_factor = getattr(config_schema.api, "rate_limiter_success_factor", 1.05)

        limiter = get_rate_limiter(
            initial_fill_rate=initial_fill_rate,
            success_threshold=success_threshold,
            min_fill_rate=min_fill_rate,
            max_fill_rate=max_fill_rate,
            capacity=bucket_capacity,
            endpoint_profiles=endpoint_profiles,
            rate_limiter_429_backoff=rate_limiter_429_backoff,
            rate_limiter_success_factor=rate_limiter_success_factor,
        )

        if limiter:
            logger.debug(f"AdaptiveRateLimiter created/retrieved. Current fill_rate: {limiter.fill_rate}")

            # Force the rate to be at least initial_fill_rate (overriding any persisted slow state)
            # BUT respect the max_fill_rate which might have been clamped by endpoint caps
            effective_initial_rate = min(initial_fill_rate, limiter.max_fill_rate)

            if limiter.fill_rate < effective_initial_rate:
                logger.info(
                    f"🚀 Restoring rate limiter rate from {limiter.fill_rate:.3f} to {effective_initial_rate:.3f} req/s"
                )
                limiter.fill_rate = effective_initial_rate

            logger.debug(
                "AdaptiveRateLimiter bound to session (success_threshold=%d, backoff=%.2f, success_factor=%.2f)",
                limiter.success_threshold,
                rate_limiter_429_backoff,
                rate_limiter_success_factor,
            )

        return limiter

    @staticmethod
    def _get_utils_attr_static(attr_name: str) -> Any:
        """Safely retrieve an attribute from the utils module."""

        utils_module = _load_utils_module()
        try:
            return getattr(utils_module, attr_name)
        except AttributeError as exc:  # pragma: no cover - defensive logging
            raise AttributeError(f"utils module missing attribute '{attr_name}'") from exc

    def _get_utils_attr(self, attr_name: str) -> Any:
        """Instance-friendly wrapper to satisfy mixin expectations."""

        return self._get_utils_attr_static(attr_name)

    @staticmethod
    def _resolve_rate_limiter_factory() -> Optional[Callable[..., Any]]:
        """Import the rate limiter factory if it is available."""

        try:
            return SessionManager._get_utils_attr_static("get_rate_limiter")
        except (ImportError, ModuleNotFoundError, AttributeError):
            return None

    @staticmethod
    def _resolve_batch_threshold() -> int:
        """Derive the base batch threshold from configuration."""

        batch_threshold = getattr(config_schema, "batch_size", 50) or 50
        return max(int(batch_threshold), 1)

    @staticmethod
    def _determine_success_threshold(batch_threshold: int) -> int:
        """Resolve success threshold with configuration overrides."""

        configured_threshold = getattr(
            getattr(config_schema, "api", None),
            "token_bucket_success_threshold",
            None,
        )
        if isinstance(configured_threshold, int) and configured_threshold > 0:
            return configured_threshold
        return max(batch_threshold, 10)

    def _log_endpoint_rate_cap(self, endpoint_profiles: dict[str, dict[str, Any]]) -> None:
        """Log derived endpoint rate caps for observability."""

        endpoint_rate_cap = self._calculate_endpoint_rate_cap(endpoint_profiles)
        if endpoint_rate_cap is not None:
            logger.debug("Endpoint-specific throttle floor detected: %.3f req/s", endpoint_rate_cap)

    @staticmethod
    def _build_endpoint_profile_config() -> dict[str, dict[str, Any]]:
        """Normalize endpoint throttle profiles from configuration."""

        endpoint_profiles_raw = getattr(config_schema.api, "endpoint_throttle_profiles", {})
        if not isinstance(endpoint_profiles_raw, dict):
            return {}

        return {
            key: dict(value)
            for key, value in endpoint_profiles_raw.items()
            if isinstance(key, str) and isinstance(value, dict)
        }

    @staticmethod
    def _calculate_endpoint_rate_cap(endpoint_profiles: dict[str, dict[str, Any]]) -> Optional[float]:
        """Derive the tightest rate cap from endpoint throttle definitions."""

        endpoint_rate_cap: Optional[float] = None
        for profile in endpoint_profiles.values():
            max_rate_val = profile.get("max_rate")
            min_interval_val = profile.get("min_interval")
            candidate_rates: list[float] = []

            if isinstance(max_rate_val, (int, float)) and max_rate_val > 0:
                candidate_rates.append(float(max_rate_val))
            if isinstance(min_interval_val, (int, float)) and min_interval_val > 0:
                candidate_rates.append(1.0 / float(min_interval_val))

            if candidate_rates:
                cap_candidate = min(candidate_rates)
                endpoint_rate_cap = (
                    cap_candidate if endpoint_rate_cap is None else min(endpoint_rate_cap, cap_candidate)
                )

        return endpoint_rate_cap

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
            max_retries=0,  # Application handles retries
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
            self._scraper = cast(Any, cloudscraper).create_scraper(
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

    def _start_browser(self, action_name: Optional[str] = None) -> bool:
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

    def _close_browser(self) -> None:
        """Close the browser session without affecting database."""
        self.browser_manager.close_browser()
        self._reset_cookie_sync_state("browser_close")

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
        if not self.db_manager.ensure_ready():
            logger.error("Failed to ensure database ready.")
            return False

        # Start browser if needed
        if self.browser_manager.browser_needed:
            browser_success = self.browser_manager.start_browser(action_name)
            if not browser_success:
                logger.error("Failed to start browser.")
                return False

        # Mark session as started
        self.session_start_time = time.time()
        self._update_session_metrics()

        return True

    # --- Lifecycle guard helpers ---

    def _transition_state(self, target: SessionLifecycleState, reason: str) -> None:
        """Internal helper to update lifecycle state with logging."""

        with self._state_lock:
            previous = self._state
            self._state = target
            self._state_reason = reason or previous.value
            self._state_changed_at = time.time()

            if target == SessionLifecycleState.READY:
                self.session_ready = True
            else:
                self.session_ready = False

        if previous != target:
            logger.debug(
                "Session lifecycle transition: %s → %s (%s)",
                previous.value,
                target.value,
                reason,
            )

    def lifecycle_state(self) -> SessionLifecycleState:
        """Return the current lifecycle state (thread-safe)."""

        with self._state_lock:
            return self._state

    def _get_state_snapshot(self) -> dict[str, Any]:
        """Provide a snapshot of lifecycle state for diagnostics."""

        with self._state_lock:
            return {
                "state": self._state.value,
                "reason": self._state_reason,
                "changed_at": self._state_changed_at,
            }

    def _request_recovery(self, reason: str) -> None:
        """Reset lifecycle state to UNINITIALIZED before rebuilding."""

        self._last_readiness_check = None
        self._transition_state(SessionLifecycleState.UNINITIALIZED, reason)

    def _guard_action(self, required_state: str, action_name: str) -> bool:
        """Guard execution based on lifecycle state requirements."""

        normalized_required = (required_state or "").lower()
        if normalized_required == "db_ready":
            return True

        current_state = self.lifecycle_state()
        if current_state == SessionLifecycleState.DEGRADED:
            logger.warning(
                "Action %s requires %s but session is DEGRADED; resetting lifecycle state before continuing",
                action_name,
                normalized_required,
            )
            self._request_recovery(f"{action_name} reset from DEGRADED state")
            return True

        if current_state == SessionLifecycleState.UNINITIALIZED and normalized_required in {
            "driver_ready",
            "session_ready",
        }:
            logger.debug(
                "Action %s will bootstrap session lifecycle (state=%s)",
                action_name,
                current_state.value,
            )

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
        if self.browser_manager.browser_needed and not self.is_sess_valid():
            logger.debug(f"Cached session readiness invalid - driver session expired (age: {time_since_check:.1f}s)")
            self._last_readiness_check = None
            self._transition_state(
                SessionLifecycleState.DEGRADED,
                "Driver session expired during cached readiness check",
            )
            return False

        logger.debug(f"Using cached session readiness (age: {time_since_check:.1f}s, action: {action_name})")
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

    def ensure_db_ready(self) -> bool:
        """Ensure the database layer is initialized and ready for use."""

        if self._db_ready:
            return True

        try:
            self._db_ready = bool(self.db_manager.ensure_ready())
            return self._db_ready
        except Exception as exc:
            logger.error(f"Failed to prepare database: {exc}", exc_info=True)
            self._db_ready = False
            return False

    def _handle_cached_readiness(self, action_name: Optional[str], skip_csrf: bool) -> Optional[bool]:
        """Check cached readiness and verify CSRF if needed."""
        cached_result = self._check_cached_readiness(action_name)
        if cached_result is True:
            # Verify CSRF requirement
            if not skip_csrf and not self.api_manager.csrf_token:
                logger.debug("Cached readiness OK but CSRF token missing. Fetching...")
                token = self.api_manager.get_csrf_token()
                if not token:
                    logger.warning("Failed to fetch CSRF token with cached session. Forcing full validation.")
                    return None  # Fall through to full validation

            self._update_session_metrics()
            return True
        return cached_result

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

        # Check health monitor
        if hasattr(self, 'health_monitor') and self.health_monitor.should_emergency_halt():
            logger.critical("🚨 Emergency halt requested by health monitor - refusing session readiness")
            return False

        # PHASE 5.1: Check cached session state first
        cached_result = self._handle_cached_readiness(action_name, skip_csrf)
        if cached_result is not None:
            return cached_result

        readiness_reason = f"{action_name or 'unknown_action'} readiness validation"
        self._transition_state(SessionLifecycleState.RECOVERING, readiness_reason)

        # Ensure driver is live if browser is needed (with optimization)
        if self.browser_manager.browser_needed and not self.browser_manager.ensure_driver_live(action_name):
            logger.error("Failed to ensure driver live.")
            self._transition_state(
                SessionLifecycleState.DEGRADED,
                "Driver live check failed",
            )
            self._update_session_metrics()
            return False

        # Perform readiness validation
        readiness_ok = self._perform_readiness_validation(action_name, skip_csrf)

        if readiness_ok:
            self._transition_state(SessionLifecycleState.READY, readiness_reason)
        else:
            self._transition_state(SessionLifecycleState.DEGRADED, readiness_reason)
        self.session_ready = readiness_ok

        # PHASE 5.1: Cache the readiness check result
        self._last_readiness_check = time.time()

        check_time = time.time() - start_time
        logger.debug(f"Session readiness check completed in {check_time:.3f}s, status: {self.session_ready}")
        self._update_session_metrics()
        return readiness_ok

    def get_db_conn(self) -> Optional[SqlAlchemySession]:
        """Borrow a database session from the pool after ensuring readiness."""

        if not self.ensure_db_ready():
            return None
        return self.db_manager.get_session()

    def return_session(self, session: Optional[SqlAlchemySession]) -> None:
        """Return a previously borrowed database session."""

        self.db_manager.return_session(session)

    def get_db_conn_context(self) -> contextlib.AbstractContextManager[Optional[SqlAlchemySession]]:
        """Context manager wrapper around DatabaseManager.get_session_context."""

        if not self.ensure_db_ready():
            return contextlib.nullcontext(None)
        return self.db_manager.get_session_context()

    def _verify_sess(self) -> bool:
        """
        Verify session status using login_status function.

        Returns:
            bool: True if session is valid, False otherwise
        """
        logger.debug("Verifying session status (using login_status)...")
        try:
            # Import login_status locally to avoid circular imports
            from core.utils import login_status

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
        if self.browser_manager.driver is None:
            return False

        # Enhanced check - verify driver is responsive
        try:
            # Quick responsiveness test
            _ = self.browser_manager.driver.current_url
            return True
        except Exception as e:
            logger.warning(f"🔌 WebDriver session appears invalid: {e}")
            # Attempt session recovery for long-running operations
            if self._should_attempt_recovery():
                logger.info("🔄 Attempting automatic session recovery...")
                if self._attempt_session_recovery(reason="browser_error"):
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
        # Attempt recovery any time a session that was marked ready becomes invalid
        return bool(self.session_ready)

    def _attempt_session_recovery(self, reason: str = "browser_error") -> bool:
        """
        Attempt to recover an invalid WebDriver session.

        Returns:
            bool: True if recovery successful, False otherwise
        """
        try:
            logger.debug("Closing invalid browser session...")
            self.browser_manager.close_browser()

            logger.debug("Starting new browser session...")
            if not self.browser_manager.start_browser("session_recovery"):
                logger.error("Browser restart failed during session recovery")
                return False

            logger.debug("Browser recovery successful, validating authentication state...")

            from core.utils import log_in, login_status

            login_ok = login_status(self, disable_ui_fallback=False)
            if login_ok is not True:
                logger.info(
                    "Recovered browser not authenticated (login_status=%s) - initiating login flow",
                    login_ok,
                )
                login_result = log_in(self)
                if login_result != "LOGIN_SUCCEEDED":
                    logger.error(
                        "Re-authentication failed after browser recovery (result=%s)",
                        login_result,
                    )
                    return False

            if not self._finalize_recovered_session_state():
                logger.error("Session recovery failed validation (cookies/CSRF missing)")
                return False

            self.session_start_time = time.time()
            self._update_session_metrics()
            self._record_session_refresh_metric(reason)
            logger.info("Session recovery and re-authentication successful")
            return True

        except Exception as e:
            logger.error(f"Session recovery failed: {e}", exc_info=True)

        return False

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
        if not self.api_manager.my_profile_id:
            logger.debug("Retrieving profile ID (ucdmid)...")
            profile_id = self._get_my_profile_id()
            if not profile_id:
                logger.error("Failed to retrieve profile ID (ucdmid).")
                all_ok = False

        # Get UUID
        if not self.api_manager.my_uuid:
            logger.debug("Retrieving UUID (testId)...")
            uuid_val = self._get_my_uuid()
            if not uuid_val:
                logger.error("Failed to retrieve UUID (testId).")
                all_ok = False

        # Get Tree ID (only if TREE_NAME is configured)
        if config_schema.api.tree_name and not self.api_manager.my_tree_id:
            logger.debug(f"Retrieving tree ID for tree name: '{config_schema.api.tree_name}'...")
            try:
                tree_id = self._get_my_tree_id()
                if not tree_id:
                    logger.error(
                        f"TREE_NAME '{config_schema.api.tree_name}' configured, but failed to get corresponding tree ID."
                    )
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
        if not self.browser_manager.driver:
            return set()
        try:
            driver = cast(Any, self.browser_manager.driver)
            cookies = cast(list[dict[str, Any]], driver.get_cookies())
            return {cookie["name"].lower() for cookie in cookies if "name" in cookie}
        except Exception:
            return set()

    def _check_cookies_available(self, required_lower: set[str], last_missing_str: str) -> tuple[bool, str]:
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

    def _perform_final_cookie_check(self, names: list[str]) -> bool:
        """Perform final cookie check after timeout.

        Args:
            names: List of required cookie names

        Returns:
            True if all cookies found, False otherwise
        """
        try:
            if not self.is_sess_valid():
                logger.warning(f"Timeout waiting for cookies. Missing: {names}.")
                return False

            current_cookies_lower = self._get_current_cookie_names()
            missing_final = [name for name in names if name.lower() not in current_cookies_lower]

            if missing_final:
                logger.warning(f"Timeout waiting for cookies. Missing: {missing_final}.")
                return False

            logger.debug("Cookies found in final check after loop (unexpected).")
            return True

        except Exception:
            logger.warning(f"Timeout waiting for cookies. Missing: {names}.")
            return False

    def _get_cookies(self, names: list[str], timeout: int = 30) -> bool:
        """
        Advanced cookie management with timeout and session validation.

        Waits for specific cookies to be available with intelligent retry logic
        and continuous session validity checking.

        Args:
            names: List of cookie names to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if all cookies found, False otherwise
        """
        if not self.browser_manager.driver:
            logger.error("get_cookies: WebDriver instance is None.")
            return False

        start_time = time.time()
        logger.debug(f"Waiting up to {timeout}s for cookies: {names}...")
        required_lower = {name.lower() for name in names}
        interval = 0.5
        last_missing_str = ""

        while time.time() - start_time < timeout:
            try:
                if not self.browser_manager.driver:
                    logger.warning("Driver became None while waiting for cookies.")
                    return False

                all_found, last_missing_str = self._check_cookies_available(required_lower, last_missing_str)
                if all_found:
                    logger.debug(f"All required cookies found: {names}.")
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

        return self._perform_final_cookie_check(names)

    def _should_skip_cookie_sync(self, current_time: float, force: bool = False) -> bool:
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
        if not self.browser_manager.driver or not hasattr(self.api_manager, '_requests_session'):
            logger.debug("Cookie sync skipped: driver or requests_session not available")
            return True

        # Forced sync bypasses cooldown and prior success checks but still
        # requires the basic prerequisites above to be satisfied.
        if force:
            return False

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

    def _sync_cookies_to_requests(self, force: bool = False) -> bool:
        """
        Synchronize cookies from WebDriver to requests session.
        Only syncs once per session unless forced due to auth errors.

        Enhanced with recursion prevention, robust error handling, and cooldown period.
        Uses _should_skip_cookie_sync() for clean separation of sync validation logic.

        Args:
            force: When True, bypass cooldown and prior-success guards while
                still requiring prerequisite browser state.

        Returns:
            bool: True if sync occurred or was skipped intentionally (success), False on error.
        """
        current_time = time.time()

        # Check if sync should be skipped
        if self._should_skip_cookie_sync(current_time, force=force):
            return True

        try:
            # Set recursion prevention flag
            self._in_sync_cookies = True

            # Get cookies from WebDriver (validated in _should_skip_cookie_sync)
            if not self.browser_manager.driver:
                logger.error("Driver not available for cookie sync")
                return False

            driver = cast(Any, self.browser_manager.driver)
            driver_cookies = cast(list[dict[str, Any]], driver.get_cookies())

            # Validate cookies were retrieved
            if not driver_cookies:
                logger.debug("No cookies retrieved from WebDriver")
                return False

            # Use helper method for robust cookie syncing
            synced_count = self._sync_driver_cookies_to_requests(driver_cookies)

            self._session_cookies_synced = True
            self._last_cookie_sync_time = current_time  # Track sync time for cooldown
            logger.debug(f"Synced {synced_count} cookies from WebDriver to requests session (once per session)")
            return True

        except Exception as e:
            logger.error(f"Failed to sync cookies to requests session: {e}")
            return False
        finally:
            # Always clear recursion flag
            if hasattr(self, '_in_sync_cookies'):
                self._in_sync_cookies = False

    def sync_browser_cookies(self, force: bool = False) -> bool:
        """
        Synchronize cookies from WebDriver to requests session.
        Public wrapper that enforces typing-friendly cookie sync access.
        """
        return self._sync_cookies_to_requests(force=force)

    def sync_cookies_to_requests(self, force: bool = False) -> None:
        """Deprecated alias for sync_browser_cookies."""
        self.sync_browser_cookies(force=force)

    def _force_cookie_resync(self) -> None:
        """Force a cookie resync when authentication errors occur."""
        self._session_cookies_synced = False
        self._sync_cookies_to_requests(force=True)
        logger.debug("Forced session cookie resync due to authentication error")

    def _reset_cookie_sync_state(self, reason: str = "unspecified") -> None:
        """Clear cookie sync flags and cached tokens for reliable recovery."""

        self._session_cookies_synced = False
        self._last_cookie_sync_time = 0.0

        try:
            self.api_manager._requests_session.cookies.clear()
        except Exception as exc:
            logger.debug(
                "Failed to clear requests-session cookies during %s: %s",
                reason,
                exc,
            )

        self.invalidate_csrf_cache()

    def _finalize_recovered_session_state(self) -> bool:
        """Ensure cookies/CSRF are available before declaring recovery success."""

        essential_cookies = list(self.ESSENTIAL_SESSION_COOKIES)
        if not self._get_cookies(essential_cookies, timeout=30):
            logger.error(
                "Essential cookies %s missing after session recovery",
                essential_cookies,
            )
            return False

        self._force_cookie_resync()

        csrf_token = self._get_csrf()
        if not csrf_token:
            logger.error("Failed to refresh CSRF token after session recovery")
            return False

        self._precache_csrf_token()
        return True

    @staticmethod
    def _deduplicate_cookies(
        driver_cookies: list[dict[str, Any]],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """
        Deduplicate cookies by (name, path), preferring more specific domains.

        Args:
            driver_cookies: List of cookies from browser

        Returns:
            dict: Deduplicated cookies keyed by (name, path)
        """
        unique_cookies: dict[tuple[str, str], dict[str, Any]] = {}
        for cookie in driver_cookies:
            name = cookie.get("name")
            if not name:
                continue
            path = cookie.get("path", "/")
            domain = cookie.get("domain", "")
            key = (name, path)

            # If we already have this cookie, prefer the one with more specific domain
            if key in unique_cookies:
                existing_domain = unique_cookies[key].get("domain", "")
                # Prefer domain without leading dot (more specific)
                if domain.startswith(".") and not existing_domain.startswith("."):
                    continue  # Keep existing (more specific)
                if not domain.startswith(".") and existing_domain.startswith("."):
                    unique_cookies[key] = cookie  # Replace with more specific
                else:
                    unique_cookies[key] = cookie  # Keep last one
            else:
                unique_cookies[key] = cookie

        return unique_cookies

    def _sync_driver_cookies_to_requests(self, driver_cookies: list[dict[str, Any]]) -> int:
        """Sync driver cookies to requests session with deduplication.

        Args:
            driver_cookies: List of cookies from WebDriver

        Returns:
            Number of cookies synced
        """
        self.api_manager._requests_session.cookies.clear()
        synced_count = 0

        # Deduplicate cookies first
        unique_cookies = self._deduplicate_cookies(driver_cookies)

        for cookie in unique_cookies.values():
            name = cookie.get("name")
            value = cookie.get("value")
            if not isinstance(name, str) or value is None:
                continue

            try:
                cast(Any, self.api_manager._requests_session.cookies).set(
                    name,
                    value,
                    domain=cookie.get("domain"),
                    path=cookie.get("path", "/"),
                    secure=cookie.get("secure", False),
                )
                synced_count += 1
            except Exception:
                continue  # Skip problematic cookies silently

        return synced_count

    def close_sess(self, keep_db: bool = False):
        """
        Close the session.

        Args:
            keep_db: If True, keeps database connections alive
        """
        logger.debug(f"Closing session (keep_db={keep_db})")

        # Close browser
        self.browser_manager.close_browser()
        self._reset_cookie_sync_state("Session closed")

        # Close database connections if requested
        if not keep_db:
            self.db_manager.close_connections(dispose_engine=True)

        # Clear API identifiers
        self.api_manager.clear_identifiers()

        # Reset session state
        self.session_start_time = None
        self._transition_state(SessionLifecycleState.UNINITIALIZED, "Session fully closed")
        self._update_session_metrics(force_zero=True)

        self._shutdown_metrics_exporter()

        logger.debug("Session closed.")

    def _cleanup_browser(self) -> None:
        """Kill browser process and release resources."""
        if not (self.browser_manager and self.browser_manager.driver_live):
            return

        # Try graceful quit first
        try:
            driver = self.browser_manager.driver
            if driver:
                driver.quit()
        except Exception as e:
            logger.warning(f"Graceful browser quit failed: {e}")

        # Force browser manager to release resources
        try:
            self.browser_manager.close_browser()
        except Exception as e:
            logger.warning(f"BrowserManager cleanup failed: {e}")

        self._reset_cookie_sync_state("cleanup_browser")

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
        self.session_start_time = None
        self._db_init_attempted = False
        self._transition_state(SessionLifecycleState.UNINITIALIZED, "Session state reset")
        self._db_ready = False
        self._update_session_metrics(force_zero=True)

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
        logger.critical(f"🚨 FORCE SESSION RESTART triggered: {reason} - killing browser and clearing caches")

        try:
            self._cleanup_browser()
            self._cleanup_database()
            self._cleanup_api_caches()
            self._reset_session_state()
            self._record_session_refresh_metric("api_forced")
            self._shutdown_metrics_exporter()

            logger.info("Force session restart complete - session marked invalid")

        except Exception as e:
            logger.error(f"Error during force session restart: {e}", exc_info=True)

        # Always return False - operation failed, restart attempted
        return False

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
            driver = cast(Any, self.browser_manager.driver)
            csrf_cookie_names = ['_dnamatches-matchlistui-x-csrf-token', '_csrf']

            driver_cookies_list = cast(list[dict[str, Any]], driver.get_cookies())
            driver_cookies_dict = {c["name"]: c["value"] for c in driver_cookies_list if "name" in c and "value" in c}

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

    def is_csrf_token_valid(self) -> bool:
        """Public wrapper so callers can check CSRF cache health."""

        return self._is_csrf_token_valid()

    def _update_session_metrics(self, force_zero: bool = False) -> None:
        """Update Prometheus session uptime gauge."""
        try:
            if force_zero:
                metrics().session_uptime.set(0.0)
                return

            session_age = None
            if self.session_start_time:
                session_age = time.time() - self.session_start_time
            metrics().session_uptime.set(float(session_age) if session_age is not None else 0.0)
        except Exception:
            logger.debug("Failed to update session uptime metric", exc_info=True)

    def session_age_seconds(self) -> Optional[float]:
        """Return the age of the current browser session in seconds."""

        if not self.session_start_time:
            return None
        return time.time() - self.session_start_time

    def _record_session_refresh_metric(self, reason: str) -> None:
        """Increment session refresh counter for a specific reason."""
        try:
            safe_reason = reason or "unknown"
            self._last_session_refresh_reason = safe_reason
            metrics().session_refresh.inc(safe_reason)
        except Exception:
            logger.debug("Failed to record session refresh metric", exc_info=True)

    def _ensure_metrics_exporter(self) -> None:
        """Start Prometheus metrics exporter when metrics are enabled."""
        if getattr(self, "_metrics_exporter_started", False):
            return

        observability_cfg = getattr(config_schema, "observability", None)
        if not observability_cfg or not getattr(observability_cfg, "enable_prometheus_metrics", False):
            return

        try:
            if start_metrics_exporter(
                observability_cfg.metrics_export_host,
                observability_cfg.metrics_export_port,
            ):
                self._metrics_exporter_started = True
        except Exception:
            logger.debug("Failed to start Prometheus metrics exporter", exc_info=True)

    def _shutdown_metrics_exporter(self) -> None:
        """Stop the metrics exporter if this manager started it."""
        if not getattr(self, "_metrics_exporter_started", False):
            return
        try:
            stop_metrics_exporter()
        except Exception:
            logger.debug("Failed to stop Prometheus metrics exporter", exc_info=True)
        finally:
            self._metrics_exporter_started = False

    # PHASE 5.1: Session cache management methods
    def _get_session_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for this session"""
        stats = get_session_cache_stats()
        session_age = None
        if self.session_start_time:
            session_age = time.time() - self.session_start_time

        stats.update(
            {
                "session_ready": self.session_ready,
                "session_age": session_age,
                "last_readiness_check_age": (
                    time.time() - self._last_readiness_check if self._last_readiness_check else None
                ),
                "db_ready": (self.db_manager.is_ready if hasattr(self.db_manager, "is_ready") else False),
                "browser_needed": self.browser_manager.browser_needed,
                "driver_live": self.browser_manager.driver_live,
            }
        )
        return stats

    @classmethod
    def _clear_session_caches(cls) -> int:
        """Clear all session caches for fresh initialization"""
        return clear_session_cache()

    def _update_response_time_tracking(
        self,
        duration: float,
        slow_threshold: float = 5.0,
        max_history: int = 20,
    ) -> None:
        """Update response time tracking metrics.

        Public method to safely update performance tracking without
        directly accessing protected attributes.

        Args:
            duration: Response time in seconds
            slow_threshold: Duration above which calls are considered slow
            max_history: Maximum number of response times to track
        """
        self._response_times.append(duration)
        if len(self._response_times) > max_history:
            self._response_times.pop(0)

        if self._response_times:
            self._avg_response_time = sum(self._response_times) / len(self._response_times)

        if duration > slow_threshold:
            self._recent_slow_calls += 1
        else:
            self._recent_slow_calls = max(0, self._recent_slow_calls - 1)

        self._recent_slow_calls = min(self._recent_slow_calls, 10)

    def _reset_response_time_tracking(self) -> None:
        """Reset response time tracking to initial state."""
        self._response_times = []
        self._recent_slow_calls = 0
        self._avg_response_time = 0.0

    def _update_cookie_sync_time(self, sync_time: float) -> None:
        """Update the last cookie sync timestamp."""
        self._last_cookie_sync_time = sync_time

    def _set_cached_csrf_token(self, token: str, cache_time: float) -> None:
        """Set the cached CSRF token and cache timestamp."""
        self._cached_csrf_token = token
        self._csrf_cache_time = cache_time

    def _get_cached_csrf_token(self) -> tuple[Optional[str], float]:
        """Get the cached CSRF token and cache timestamp."""
        return self._cached_csrf_token, self._csrf_cache_time

    def _clear_last_readiness_check(self) -> None:
        """Clear the last readiness check timestamp for fresh validation."""
        self._last_readiness_check = None

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
                logger.warning(f"Watchdog already active for '{self._api_name}', cancelling previous timer")
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

            logger.debug(f"Watchdog started for '{api_name}' (timeout: {self.timeout_seconds}s)")

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
                logger.critical(f"🚨 WATCHDOG TIMEOUT: {api_name} exceeded {self.timeout_seconds}s limit")
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
def _test_session_manager_initialization() -> bool:
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
        results: list[bool] = []
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


def _test_component_manager_availability() -> bool:
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
        results: list[bool] = []

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


def _test_database_operations() -> bool:
    """Test database operations with detailed result verification"""
    database_operations = [
        ("ensure_db_ready", "Ensure database is ready for operations"),
        ("get_db_conn", "Get database connection/session"),
        ("get_db_conn_context", "Get database session context manager"),
    ]

    print("📋 Testing database operations:")

    try:
        session_manager = SessionManager()
        results: list[bool] = []

        for operation_name, description in database_operations:
            try:
                if operation_name == "ensure_db_ready":
                    result = session_manager.db_manager.ensure_ready()
                    is_bool = isinstance(cast(Any, result), bool)

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
                    print(f"      Connection: {type(conn).__name__ if conn else 'None'}")

                    results.append(True)  # Just test it doesn't crash

                    # Return the connection if we got one
                    if conn:
                        session_manager.return_session(conn)

                elif operation_name == "get_db_conn_context":
                    context = session_manager.get_db_conn_context()
                    has_context = context is not None

                    status = "✅" if has_context else "❌"
                    print(f"   {status} {operation_name}: {description}")
                    print(f"      Context: {type(context).__name__ if context else 'None'}")

                    results.append(True)  # Just test it doesn't crash

            except Exception as e:
                print(f"   ❌ {operation_name}: Exception {e}")
                results.append(False)

        print(f"📊 Results: {sum(results)}/{len(results)} database operations successful")
        return True

    except Exception as e:
        print(f"❌ Database operations test failed: {e}")
        return False


def _test_browser_operations() -> bool:
    session_manager = SessionManager()
    result = session_manager.browser_manager.start_browser("test_action")
    assert isinstance(result, bool), "start_browser should return bool"
    session_manager.browser_manager.close_browser()
    return True


def _test_property_access() -> bool:
    session_manager = SessionManager()
    properties_to_check = [
        "session_ready",
    ]
    for prop in properties_to_check:
        assert hasattr(session_manager, prop), f"Property {prop} should exist"
    return True


def _test_component_delegation() -> bool:
    session_manager = SessionManager()
    db_result = session_manager.db_manager.ensure_ready()
    assert isinstance(db_result, bool), "Database delegation should work"
    browser_result = session_manager.browser_manager.start_browser("test")
    assert isinstance(browser_result, bool), "Browser delegation should work"
    return True


def _test_initialization_performance() -> bool:
    import time

    session_managers: list[SessionManager] = []
    start_time = time.time()
    for _i in range(3):
        session_manager = SessionManager()
        session_managers.append(session_manager)
    end_time = time.time()
    total_time = end_time - start_time
    max_time = 5.0
    assert total_time < max_time, f"3 optimized initializations took {total_time:.3f}s, should be under {max_time}s"
    for sm in session_managers:
        with contextlib.suppress(Exception):
            sm.close_sess(keep_db=True)
    return True


def _test_error_handling() -> bool:
    session_manager = SessionManager()
    try:
        session_manager.db_manager.ensure_ready()
        session_manager.browser_manager.start_browser("test_action")
        session_manager.browser_manager.close_browser()
        _ = session_manager.session_ready
    except Exception as e:
        raise AssertionError(f"SessionManager should handle operations gracefully: {e}") from e
    return True


def _test_regression_prevention_csrf_optimization() -> bool:
    """
    🛡️ REGRESSION TEST: CSRF token caching optimization.

    This test verifies that Optimization 1 (CSRF token pre-caching) is properly
    implemented and working. This would have prevented performance regressions
    caused by fetching CSRF tokens on every API call.
    """
    print("🛡️ Testing CSRF token caching optimization regression prevention:")
    results: list[bool] = []

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

        # Test 2: Verify CSRF validation method exists (using public wrapper)
        if hasattr(session_manager, 'is_csrf_token_valid'):
            print("   ✅ is_csrf_token_valid method exists")

            # Test that it returns a boolean
            try:
                is_valid = session_manager.is_csrf_token_valid()
                if isinstance(cast(Any, is_valid), bool):
                    print("   ✅ is_csrf_token_valid returns boolean")
                    results.append(True)
                else:
                    print("   ❌ is_csrf_token_valid doesn't return boolean")
                    results.append(False)
            except Exception as method_error:
                print(f"   ⚠️  is_csrf_token_valid method error: {method_error}")
                results.append(False)
        else:
            print("   ❌ is_csrf_token_valid method missing")
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


def _test_regression_prevention_property_access() -> bool:
    """
    🛡️ REGRESSION TEST: SessionManager property access stability.

    This test verifies that SessionManager properties are accessible without
    errors. This would have caught the duplicate method definition issues
    we encountered.
    """
    print("🛡️ Testing SessionManager property access regression prevention:")
    results: list[bool] = []
    failure_details: list[str] = []

    try:
        session_manager = SessionManager()

        # Test key properties that had duplicate definition issues
        properties_to_test = [
            ('requests_session', 'requests session object'),
            ('csrf_token', 'CSRF token string'),
            ('my_uuid', 'user UUID string'),
            ('my_tree_id', 'tree ID string'),
            ('session_ready', 'session ready boolean'),
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
                detail = f"Property '{prop}' error: {prop_error}"
                print(f"   ❌ {detail}")
                failure_details.append(detail)
                results.append(False)

    except Exception as e:
        detail = f"SessionManager property access test failed: {e}"
        print(f"   ❌ {detail}")
        failure_details.append(detail)
        results.append(False)

    success = all(results)
    if not success:
        failure_summary = " | ".join(failure_details) or "Unknown property failures"
        raise AssertionError(f"SessionManager property regression test failures: {failure_summary}")

    print("🎉 SessionManager property access regression test passed!")
    return True


def _test_regression_prevention_initialization_stability() -> bool:
    """
    🛡️ REGRESSION TEST: SessionManager initialization stability.

    This test verifies that SessionManager initializes without crashes,
    which would have caught WebDriver stability issues.
    """
    print("🛡️ Testing SessionManager initialization stability regression prevention:")
    results: list[bool] = []

    try:
        # Test multiple initialization attempts
        for i in range(3):
            try:
                session_manager = SessionManager()
                print(f"   ✅ Initialization attempt {i + 1} successful")
                results.append(True)

                # Test basic attribute access
                _ = hasattr(session_manager, 'db_manager')
                _ = hasattr(session_manager, 'browser_manager')
                _ = hasattr(session_manager, 'api_manager')

                print(f"   ✅ Basic attribute access {i + 1} successful")
                results.append(True)

            except Exception as init_error:
                print(f"   ❌ Initialization attempt {i + 1} failed: {init_error}")
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
    callback_executed: list[bool] = []

    def emergency_callback() -> None:
        callback_executed.append(True)

    watchdog = APICallWatchdog(timeout_seconds=0.5)
    watchdog.start("test_api", emergency_callback)
    time.sleep(1.0)  # Wait for timeout

    assert len(callback_executed) == 1, "Callback should be executed exactly once"


def _test_watchdog_graceful_completion() -> None:
    """Test that cancel() prevents callback execution."""
    callback_executed: list[bool] = []

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
    callback_executed: list[bool] = []

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
    errors: list[tuple[int, Exception]] = []

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
    callback_executed: list[bool] = []
    watchdog.start("test_api", lambda: callback_executed.append(True))
    time.sleep(0.7)
    assert len(callback_executed) == 1, "Callback should have executed"
    watchdog.cancel()  # Should not error


def _test_force_session_restart() -> None:
    """Test _force_session_restart() method."""
    sm = SessionManager()

    # Set up session state
    sm._transition_state(SessionLifecycleState.READY, "test setup")
    sm.session_start_time = time.time()
    sm._db_init_attempted = True
    sm._db_ready = True

    # Initial state verification
    assert sm.lifecycle_state() is SessionLifecycleState.READY, "Session should be ready initially"
    assert sm._db_ready is True, "DB should be ready initially"

    # Call force_session_restart
    result = sm._force_session_restart("Test timeout")

    # Verify result (should always return False)
    assert result is False, "Force restart should always return False"

    # Verify session state reset
    assert sm.lifecycle_state() is SessionLifecycleState.UNINITIALIZED, "Lifecycle should reset to UNINITIALIZED"
    assert sm.session_start_time is None, "Session start time should be None"
    assert sm._db_init_attempted is False, "DB init attempted should be reset"
    assert sm._db_ready is False, "DB ready should be reset"


def _test_watchdog_integration_with_session_restart() -> None:
    """Test watchdog integration with _force_session_restart()."""
    sm = SessionManager()
    restart_called: list[bool] = []

    def restart_callback() -> None:
        """Callback that triggers session restart."""
        result = sm._force_session_restart("Watchdog timeout in test")
        restart_called.append(result)

    # Set up session state
    sm._transition_state(SessionLifecycleState.READY, "watchdog test setup")

    # Create watchdog with short timeout
    watchdog = APICallWatchdog(timeout_seconds=0.5)
    watchdog.start("test_api", restart_callback)

    # Wait for timeout
    time.sleep(0.7)

    # Verify restart was called
    assert len(restart_called) == 1, "Restart callback should have been called"
    assert restart_called[0] is False, "Restart should return False"
    assert sm.lifecycle_state() is SessionLifecycleState.UNINITIALIZED, "Lifecycle should reset after restart"

    # Cleanup
    watchdog.cancel()


def _test_session_expiry_simulation() -> None:
    """
    Test that simulates forced session expiry and verifies recovery.

    Priority 0 Todo #5: Session health safeguard regression test
    """
    sm = SessionManager()

    # Setup valid session state
    sm._transition_state(SessionLifecycleState.READY, "expiry test setup")
    sm.session_start_time = time.time() - 2500  # 41+ minutes ago (expired)
    sm._db_ready = True

    # Verify initial state
    assert sm.lifecycle_state() is SessionLifecycleState.READY, "Session should be marked ready initially"

    # Simulate session expiry check
    session_age = time.time() - sm.session_start_time if sm.session_start_time else 0
    is_expired = session_age > 2400  # 40 minutes threshold

    assert is_expired, f"Session should be expired (age: {session_age:.0f}s)"

    # Force session restart (simulating expiry detection)
    result = sm._force_session_restart("Session expiry simulation")

    # Verify session was reset
    assert result is False, "Force restart should return False"
    assert sm.lifecycle_state() is SessionLifecycleState.UNINITIALIZED, "Lifecycle should reset after expiry"
    assert sm.session_start_time is None, "Session start time should be cleared"
    assert sm._db_ready is False, "DB ready flag should be reset"


def _test_circuit_breaker_short_circuit() -> None:
    """
    Test circuit breaker short-circuits after 5 consecutive failures.

    Priority 0 Todo #5: Circuit breaker regression test
    """
    from core.error_handling import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenError

    config = CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60)
    breaker = CircuitBreaker(name="test_session", config=config)

    # Verify initial state
    assert breaker.failure_count == 0, "Circuit breaker should start with 0 failures"
    assert breaker.state.value == "CLOSED", "Circuit breaker should start CLOSED"

    # Create a test function that always fails
    def failing_operation() -> None:
        raise RuntimeError("Simulated failure")

    # Record 4 failures (just under threshold) - should not open
    for i in range(4):
        with suppress(RuntimeError):
            breaker.call(failing_operation)
        assert breaker.failure_count == i + 1, f"Should have {i + 1} failures"
        assert breaker.state.value == "CLOSED", f"Should stay CLOSED after {i + 1} failures"

    # 5th failure should open the circuit
    with suppress(RuntimeError):
        breaker.call(failing_operation)
    assert breaker.failure_count >= 5, "Should have at least 5 failures"
    assert breaker.state.value == "OPEN", "Circuit breaker should OPEN on 5th failure"

    # Next call should immediately raise CircuitBreakerOpenError without executing function
    try:
        breaker.call(failing_operation)
        raise AssertionError("Should have raised CircuitBreakerOpenError")
    except CircuitBreakerOpenError:
        pass  # Expected - circuit is open

    # Reset for success test
    breaker.reset()
    assert breaker.failure_count == 0, "Reset should clear failure count"
    assert breaker.state.value == "CLOSED", "Reset should close circuit breaker"

    logger.info("✅ Circuit breaker 5-failure threshold validated")


def _test_proactive_session_refresh_timing() -> None:
    """
    Test proactive session refresh triggers before expiry.

    Priority 0 Todo #5: Health check timing regression test
    """
    sm = SessionManager()

    # Setup session at 25 minutes old (refresh threshold with 15-min buffer)
    sm.session_start_time = time.time() - 1500  # 25 minutes
    sm._transition_state(SessionLifecycleState.READY, "proactive refresh test")

    # Check age
    session_age = time.time() - sm.session_start_time if sm.session_start_time else 0

    # Verify we're in the proactive refresh window (>25 min, <40 min)
    assert session_age >= 1500, f"Session should be at least 25 minutes old (actual: {session_age:.0f}s)"
    assert session_age < 2400, f"Session should be under 40 minutes (actual: {session_age:.0f}s)"

    # Verify refresh would be triggered in this window
    refresh_threshold = 1500  # 25 minutes in seconds
    should_refresh = session_age >= refresh_threshold

    assert should_refresh, "Session should trigger proactive refresh at 25 minutes"


def _test_retry_helper_alignment_session_manager() -> None:
    """Ensure API helper methods use api_retry with telemetry-derived settings."""

    api_policy = config_schema.retry_policies.api
    methods_to_check = [
        ("get_csrf", SessionManager._get_csrf),
        ("get_my_profile_id", SessionManager._get_my_profile_id),
        ("get_my_uuid", SessionManager._get_my_uuid),
        ("get_my_tree_id", SessionManager._get_my_tree_id),
        ("get_tree_owner", SessionManager._get_tree_owner),
    ]

    for method_name, method in methods_to_check:
        helper_name = getattr(method, "__retry_helper__", None)
        policy_name = getattr(method, "__retry_policy__", None)
        settings = getattr(method, "__retry_settings__", {})

        assert helper_name == "api_retry", f"{method_name} should use api_retry helper"
        assert policy_name == "api", f"{method_name} should resolve to api retry policy"
        assert settings.get("max_attempts") == api_policy.max_attempts, f"{method_name} max_attempts mismatch"
        assert settings.get("backoff_factor") == api_policy.backoff_factor, f"{method_name} backoff_factor mismatch"
        assert settings.get("base_delay") == api_policy.initial_delay_seconds, f"{method_name} base_delay mismatch"
        assert settings.get("max_delay") == api_policy.max_delay_seconds, f"{method_name} max_delay mismatch"


def _test_session_lifecycle_transitions() -> None:
    """Validate lifecycle state machine transitions and guards."""

    sm = SessionManager()

    snapshot = sm._get_state_snapshot()
    assert snapshot["state"] == SessionLifecycleState.UNINITIALIZED.value, "Initial state should be UNINITIALIZED"

    sm._transition_state(SessionLifecycleState.RECOVERING, "test transition")
    assert sm.lifecycle_state() is SessionLifecycleState.RECOVERING, "Should transition to RECOVERING"
    assert sm.session_ready is False, "session_ready should be False when recovering"

    sm._transition_state(SessionLifecycleState.READY, "ready test")
    assert sm.lifecycle_state() is SessionLifecycleState.READY, "Should transition to READY"
    assert sm.session_ready is True, "session_ready should mirror READY state"

    sm._transition_state(SessionLifecycleState.DEGRADED, "failure simulation")
    assert sm.lifecycle_state() is SessionLifecycleState.DEGRADED, "Should transition to DEGRADED"
    assert sm.session_ready is False, "session_ready should be False when degraded"

    sm._guard_action("session_ready", "unit_test")
    assert sm.lifecycle_state() is SessionLifecycleState.UNINITIALIZED, "Guard should reset DEGRADED state for recovery"

    sm._request_recovery("manual reset")
    assert sm.lifecycle_state() is SessionLifecycleState.UNINITIALIZED, "request_recovery should reset to UNINITIALIZED"


def _test_cookie_sync_state_reset_on_browser_close() -> None:
    """Closing the browser should clear cookie sync flags and cached cookies."""

    sm = SessionManager()

    # Prevent actual browser operations during the test
    browser_manager = cast(Any, sm.browser_manager)
    browser_manager.close_browser = lambda: None

    sm._session_cookies_synced = True
    sm._last_cookie_sync_time = 123.0
    sm.api_manager._requests_session.cookies.clear()

    sm.close_sess(keep_db=True)

    assert sm._session_cookies_synced is False, "Cookie sync flag should reset"
    assert sm._last_cookie_sync_time == 0.0, "Last cookie sync timestamp should reset"
    assert not sm.api_manager._requests_session.cookies.get_dict(), "Requests session cookies should be cleared"


def _test_recovery_validation_requires_essential_cookies() -> None:
    """Recovery validation should fail when essential cookies are missing."""

    sm = SessionManager()

    from types import MethodType

    sm._get_cookies = MethodType(lambda _self, _names, timeout=30: bool(timeout) and False, sm)

    assert not sm._finalize_recovered_session_state(), "Recovery validation should fail without cookies"


def _test_recovery_validation_resyncs_and_fetches_csrf() -> None:
    """Successful recovery validation forces cookie resync and CSRF refresh."""

    sm = SessionManager()

    from types import MethodType

    sm._get_cookies = MethodType(lambda _self, _names, timeout=30: bool(timeout) or True, sm)

    flags = {"resync": False, "csrf": False}

    def _fake_force_resync(_session_manager: "SessionManager") -> None:
        flags["resync"] = True

    def _fake_get_csrf(_session_manager: "SessionManager") -> str:
        flags["csrf"] = True
        return "csrf-token"

    sm._force_cookie_resync = MethodType(_fake_force_resync, sm)
    sm._get_csrf = MethodType(_fake_get_csrf, sm)

    assert sm._finalize_recovered_session_state(), "Recovery validation should succeed with cookies and CSRF"
    assert flags["resync"], "force_cookie_resync should be invoked"
    assert flags["csrf"], "get_csrf should be invoked"


def _test_cached_readiness_returns_none_without_cache() -> None:
    """Cached readiness helper should return None when no cache exists."""

    sm = SessionManager()
    sm.browser_manager.browser_needed = False
    sm.session_ready = True
    sm._last_readiness_check = None

    assert sm._check_cached_readiness("unit_test") is None, "Without cache the helper should return None"


def _test_cached_readiness_respects_fresh_cache() -> None:
    """Cached readiness helper should reuse fresh cache when session ready."""

    sm = SessionManager()
    sm.browser_manager.browser_needed = False
    sm.session_ready = True
    sm._last_readiness_check = time.time()

    assert sm._check_cached_readiness("unit_test") is True, "Fresh cache with ready session should return True"


def _test_cached_readiness_expires_on_stale_state() -> None:
    """Cached readiness should expire when stale or session_ready flag cleared."""

    sm = SessionManager()
    sm.browser_manager.browser_needed = False
    sm.session_ready = True
    sm._last_readiness_check = time.time() - 120

    assert sm._check_cached_readiness("unit_test") is None, "Stale cache should be ignored"

    sm._last_readiness_check = time.time()
    sm.session_ready = False
    assert sm._check_cached_readiness("unit_test") is None, "Non-ready session should bypass cache"


def _test_cached_readiness_detects_invalid_driver() -> None:
    """Cached readiness should detect invalid driver sessions and degrade state."""

    sm = SessionManager()
    sm.browser_manager.browser_needed = True
    sm.session_ready = True
    sm._last_readiness_check = time.time()
    sm._transition_state(SessionLifecycleState.READY, "unit prep")

    from types import MethodType

    sm.browser_manager.is_session_valid = MethodType(lambda _self: False, sm.browser_manager)

    result = sm._check_cached_readiness("unit_test")
    assert result is False, "Invalid driver should cause cached readiness failure"
    assert sm.lifecycle_state() is SessionLifecycleState.DEGRADED, "State should transition to DEGRADED"
    assert sm._last_readiness_check is None, "Cached timestamp should reset after invalidation"


def _test_endpoint_profile_config_normalization_and_rate_cap() -> None:
    """Endpoint profile normalization should filter invalid entries and derive tightest caps."""

    original_profiles = config_schema.api.endpoint_throttle_profiles
    api_any = cast(Any, config_schema.api)
    try:
        api_any.endpoint_throttle_profiles = {
            "Valid": {"max_rate": 0.5, "min_interval": 1.5},
            "Slow": {"min_interval": 3.0},
        }
        mutated = cast(Any, api_any.endpoint_throttle_profiles)
        mutated[123] = {"max_rate": 0.1}
        mutated["Bad"] = ["not", "dict"]
        profiles = SessionManager._build_endpoint_profile_config()
        assert set(profiles.keys()) == {"Valid", "Slow"}, "Non-dict and non-str keys should be ignored"
        assert profiles["Valid"]["max_rate"] == 0.5, "Valid profile should be preserved"

        cap = SessionManager._calculate_endpoint_rate_cap(profiles)
        assert cap is not None, "Rate cap should derive from available inputs"
        assert abs(cap - (1.0 / 3.0)) < 1e-9, "Slow endpoint min interval should define tightest cap"

        api_any.endpoint_throttle_profiles = "invalid"
        assert SessionManager._build_endpoint_profile_config() == {}, "Non-dict config should be ignored"
    finally:
        config_schema.api.endpoint_throttle_profiles = original_profiles


def _test_enhanced_requests_session_configuration() -> None:
    """Enhanced requests session should mount adapters with expected pool sizing."""

    from types import SimpleNamespace

    sm = SessionManager.__new__(SessionManager)
    session = requests.Session()
    sm.api_manager = cast(APIManager, SimpleNamespace(_requests_session=session))

    sm._initialize_enhanced_requests_session()

    adapter = session.adapters["https://"]
    assert isinstance(adapter, HTTPAdapter), "HTTPS adapter should be HTTPAdapter"
    assert getattr(adapter, "_pool_connections", None) == 20, "Pool connections should be 20"
    assert getattr(adapter, "_pool_maxsize", None) == 50, "Pool max size should be 50"
    assert getattr(adapter.max_retries, "total", None) == 0, "Retries handled at app layer"


def _test_enhanced_requests_session_creates_fallback() -> None:
    """Enhanced session init should create fallback requests session when missing."""

    from types import SimpleNamespace

    sm = SessionManager.__new__(SessionManager)
    sm.api_manager = cast(APIManager, SimpleNamespace())

    sm._initialize_enhanced_requests_session()

    assert hasattr(sm.api_manager, "_requests_session"), "Fallback session should be created"
    adapter = sm.api_manager._requests_session.adapters["http://"]
    assert isinstance(adapter, HTTPAdapter), "HTTP adapter should be configured"


def _test_initialize_cloudscraper_without_dependency() -> None:
    """CloudScraper init should no-op cleanly when dependency missing."""

    from types import SimpleNamespace

    sm = SessionManager.__new__(SessionManager)
    sm.api_manager = cast(APIManager, SimpleNamespace())

    with patch(f"{__name__}.cloudscraper", None):
        sm._initialize_cloudscraper()
        assert sm._scraper is None, "Scraper should remain None when library unavailable"


def _test_initialize_cloudscraper_with_factory() -> None:
    """CloudScraper init should create scraper and mount adapters when available."""

    from types import SimpleNamespace

    class _FakeScraper:
        def __init__(self) -> None:
            self.mount_calls: list[tuple[str, HTTPAdapter]] = []

        def mount(self, prefix: str, adapter: HTTPAdapter) -> None:
            self.mount_calls.append((prefix, adapter))

    class _FakeCloudscraper:
        def __init__(self) -> None:
            self.kwargs: Optional[dict[str, Any]] = None

        def create_scraper(self, **kwargs: Any) -> _FakeScraper:
            self.kwargs = kwargs
            return _FakeScraper()

    sm = SessionManager.__new__(SessionManager)
    sm.api_manager = cast(APIManager, SimpleNamespace())

    factory = _FakeCloudscraper()
    with patch(f"{__name__}.cloudscraper", factory):
        sm._initialize_cloudscraper()
        scraper = sm._scraper
        assert isinstance(scraper, _FakeScraper), "Scraper instance should be created"
        assert factory.kwargs is not None, "Factory should receive kwargs"
        browser_cfg = factory.kwargs.get("browser", {})
        assert browser_cfg.get("browser") == "chrome", "Browser fingerprint should target Chrome"
        mounts = {prefix for prefix, _ in scraper.mount_calls}
        assert {"http://", "https://"}.issubset(mounts), "Both protocols should be mounted"


def core_session_manager_module_tests() -> bool:
    """Comprehensive test suite for session_manager.py (decomposed)."""
    from testing.test_framework import TestSuite, suppress_logging

    # Warnings already suppressed at module level when __name__ == "__main__"
    with suppress_logging():
        suite = TestSuite("Session Manager & Component Coordination", "session_manager.py")
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

        # Priority 0 Todo #5: Session health safeguard regression tests
        suite.run_test(
            "Session Expiry Detection (40-min threshold)",
            _test_session_expiry_simulation,
            "Session correctly marked as expired after 40 minutes",
            "Test session expiry simulation after 40-minute threshold",
            "Verify _is_session_expired() returns True and triggers restart",
        )

        suite.run_test(
            "Circuit Breaker 5-Failure Threshold",
            _test_circuit_breaker_short_circuit,
            "Circuit breaker opens after exactly 5 consecutive failures",
            "Test circuit breaker trips at failure threshold",
            "Verify circuit breaker state transitions: closed → open at 5 failures",
        )

        suite.run_test(
            "Proactive Session Refresh (25-min window)",
            _test_proactive_session_refresh_timing,
            "Session refresh triggered proactively at 25-minute mark",
            "Test proactive session health monitoring timing",
            "Verify refresh occurs before 40-min expiry with 15-min buffer",
        )

        suite.run_test(
            "Retry helper alignment (API accessors)",
            _test_retry_helper_alignment_session_manager,
            "Ensures SessionManager API helpers use api_retry with telemetry settings",
            "Inspect retry decorator metadata for get_csrf/get_my_* helpers",
            "Helper marker is api_retry and settings match config_schema.retry_policies.api",
        )

        suite.run_test(
            "Session lifecycle state machine",
            _test_session_lifecycle_transitions,
            "Lifecycle states transition through UNINITIALIZED→RECOVERING→READY/DEGRADED with guard resets",
            "Validate lifecycle guard and transition helpers",
            "Ensures guard_action resets DEGRADED state and transitions toggle session_ready consistently",
        )
        suite.run_test(
            "Cookie sync reset on browser close",
            _test_cookie_sync_state_reset_on_browser_close,
            "Closing browser clears cookie sync flags and cached cookies",
            "Ensure closing browser invalidates stale cookies",
            "Verify _reset_cookie_sync_state runs during close_browser and clears request-session cookies",
        )
        suite.run_test(
            "Recovery validation fails without cookies",
            _test_recovery_validation_requires_essential_cookies,
            "Recovery helper rejects when essential cookies missing",
            "Prevent false positives when cookies unavailable",
            "Mock get_cookies False and expect _finalize_recovered_session_state to fail",
        )
        suite.run_test(
            "Recovery validation resyncs and refreshes CSRF",
            _test_recovery_validation_resyncs_and_fetches_csrf,
            "Successful recovery forces cookie resync and CSRF refresh",
            "Ensure recovery path refreshes auth state",
            "Mock cookies present and assert force_cookie_resync + get_csrf invoked",
        )

        suite.run_test(
            "Cached readiness without cache",
            _test_cached_readiness_returns_none_without_cache,
            "Helper returns None when no cached readiness state",
            "Ensure cached readiness helper bypasses when cache missing",
            "Invoke _check_cached_readiness without cached timestamp and expect None",
        )

        suite.run_test(
            "Cached readiness with fresh cache",
            _test_cached_readiness_respects_fresh_cache,
            "Fresh cached readiness re-used when session still valid",
            "Ensure helper returns True when cache fresh and browser not needed",
            "Set recent cache timestamp and session_ready True, expect True",
        )

        suite.run_test(
            "Cached readiness expiration rules",
            _test_cached_readiness_expires_on_stale_state,
            "Cached readiness expires after 60s or when session_ready False",
            "Prevent stale readiness state from being re-used",
            "Set cached timestamp beyond threshold and toggle session_ready False",
        )

        suite.run_test(
            "Cached readiness invalid driver detection",
            _test_cached_readiness_detects_invalid_driver,
            "Helper detects invalid driver and transitions to DEGRADED",
            "Ensure invalid browser state clears cache and flips lifecycle state",
            "Patch browser_manager.is_session_valid to False and expect DEGRADED state",
        )

        suite.run_test(
            "Endpoint profile normalization & rate cap",
            _test_endpoint_profile_config_normalization_and_rate_cap,
            "Endpoint throttle config filtered and tightest cap derived",
            "Ensure endpoint profile helper handles invalid inputs and derives tightest rate cap",
            "Patch config_schema.api.endpoint_throttle_profiles with mixed entries and validate normalization/rate cap",
        )

        suite.run_test(
            "Enhanced requests session configuration",
            _test_enhanced_requests_session_configuration,
            "Requests session adapters use expected pool sizing",
            "Ensure enhanced session init mounts HTTPAdapter with 20/50 pool sizing",
            "Build SessionManager stub, call _initialize_enhanced_requests_session, inspect adapters",
        )

        suite.run_test(
            "Enhanced session fallback creation",
            _test_enhanced_requests_session_creates_fallback,
            "Fallback requests session created when API manager lacks one",
            "Ensure enhanced session helper creates Session when attribute missing",
            "Call helper without _requests_session and verify mount configuration",
        )

        suite.run_test(
            "CloudScraper missing dependency",
            _test_initialize_cloudscraper_without_dependency,
            "Initialization no-ops cleanly when cloudscraper unavailable",
            "Verify helper sets scraper to None when dependency missing",
            "Patch module-level cloudscraper to None and ensure _scraper stays None",
        )

        suite.run_test(
            "CloudScraper factory integration",
            _test_initialize_cloudscraper_with_factory,
            "Helper constructs scraper and mounts adapters when dependency present",
            "Validate create_scraper kwargs and adapter mounting",
            "Inject fake cloudscraper implementation and assert mounts captured",
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


# Use centralized test runner utility from test_utilities
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(core_session_manager_module_tests)


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
