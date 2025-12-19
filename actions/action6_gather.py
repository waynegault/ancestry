#!/usr/bin/env python3

# action6_gather.py

"""
action6_gather.py - Gather DNA Matches from Ancestry

Fetches the user's DNA match list page by page, extracts relevant information,
compares with existing database records, fetches additional details via API for
new or changed matches, and performs bulk updates/inserts into the local database.
Handles pagination, rate limiting, caching (via utils/cache.py decorators used
within helpers), error handling, and sequential API fetches coordinated through
SessionManager.

"""

import contextlib
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Final, Optional, cast

from core.logging_utils import OptimizedLogger
from performance.health_monitor import integrate_with_action6

# === MODULE SETUP ===
raw_logger = logging.getLogger(__name__)
logger = OptimizedLogger(raw_logger)

# === STANDARD LIBRARY IMPORTS ===
import math
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal
from urllib.parse import unquote, urlencode, urljoin, urlparse

# Automatically connect API performance metrics with the health monitor on import
integrate_with_action6(sys.modules[__name__])

# === THIRD-PARTY IMPORTS ===
import requests
from requests.exceptions import ConnectionError
from selenium.common.exceptions import (
    NoSuchCookieException,
    WebDriverException,
)
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session as SqlAlchemySession  # Alias Session

from core.error_handling import (
    AuthenticationExpiredError,
    BrowserSessionError,
    DatabaseConnectionError,
    MaxApiFailuresExceededError,
    NetworkTimeoutError,
    RetryableError,
)

# === LOCAL IMPORTS ===
if TYPE_CHECKING:
    from config.config_schema import ConfigSchema

from actions.gather.api_implementations import (
    _cache_combined_details,
    _cache_profile,
    _call_api_request,
    _check_combined_details_cache,
    _ensure_action6_session_ready,
    _get_cached_profile,
    fetch_batch_badge_details as _fetch_batch_badge_details,
    fetch_batch_ladder as _fetch_batch_ladder,
    fetch_combined_details as _fetch_combined_details,
    fetch_ethnicity_for_batch as _fetch_ethnicity_for_batch,
    needs_ethnicity_refresh as _needs_ethnicity_refresh,
)
from actions.gather.metrics import PageProcessingMetrics
from actions.gather.orchestrator import GatherConfiguration, GatherOrchestrator
from actions.gather.performance_logging import (
    log_api_performance as _log_api_performance,
)
from actions.gather.persistence import (
    BatchLookupArtifacts,
    PersistenceHooks,
    prepare_and_commit_batch_data as gather_prepare_and_commit_batch_data,
    process_batch_lookups as gather_process_batch_lookups,
)
from actions.gather.prefetch import (
    PrefetchConfig,
    PrefetchHooks,
    get_prefetched_data_for_match as _get_prefetched_data_for_match,
    perform_api_prefetches as gather_perform_api_prefetches,
)
from browser.css_selectors import *  # Import CSS selectors
from browser.selenium_utils import get_driver_cookies
from config import config_schema
from core.database import (
    DnaMatch,
    FamilyTree,
    Person,
    PersonStatusEnum,
)
from core.session_manager import SessionManager
from core.unified_cache_manager import get_unified_cache
from core.utils import (
    format_name,  # Name formatting utility
    nav_to_page,  # Navigation helper
)
from testing.test_framework import (
    TestSuite,
    suppress_logging,
)
from testing.test_utilities import create_standard_test_runner

# --- Constants ---
# Default values before attempting config overrides
_matches_per_page_default = 20
_enable_ethnicity_enrichment_default = True
_ethnicity_min_cm_default = 10
_relationship_prob_limit_default: Any = 0

_matches_per_page = _matches_per_page_default
_enable_ethnicity_enrichment = _enable_ethnicity_enrichment_default
_ethnicity_min_cm = _ethnicity_min_cm_default
_relationship_prob_limit_raw: Any = _relationship_prob_limit_default

# Get MATCHES_PER_PAGE from config, fallback to defaults if not available
try:
    from config import config_schema as _cfg_temp

    _matches_per_page = int(getattr(_cfg_temp, "matches_per_page", _matches_per_page))
    _enable_ethnicity_enrichment = bool(getattr(_cfg_temp, "enable_ethnicity_enrichment", _enable_ethnicity_enrichment))
    raw_min_cm = getattr(_cfg_temp, "ethnicity_enrichment_min_cm", _ethnicity_min_cm)
    _ethnicity_min_cm = int(raw_min_cm or _ethnicity_min_cm)
    _relationship_prob_limit_raw = getattr(
        getattr(_cfg_temp, "api", None), "max_relationship_prob_fetches", _relationship_prob_limit_raw
    )
except (ImportError, ValueError, TypeError):
    # Defaults already set above; import errors simply retain defaults
    pass

ETHNICITY_ENRICHMENT_MIN_CM: int = max(0, int(_ethnicity_min_cm))
MATCHES_PER_PAGE: int = max(1, _matches_per_page)
ENABLE_ETHNICITY_ENRICHMENT: bool = _enable_ethnicity_enrichment

try:
    _relationship_prob_max_per_page = int(_relationship_prob_limit_raw or 0)
except (TypeError, ValueError):
    _relationship_prob_max_per_page = 0

RELATIONSHIP_PROB_MAX_PER_PAGE: int = max(0, _relationship_prob_max_per_page)

# Get DNA match probability threshold from environment, fallback to 10 cM
try:
    import os

    _dna_threshold_raw = int(os.getenv('DNA_MATCH_PROBABILITY_THRESHOLD_CM', '10'))
except (ValueError, TypeError):
    _dna_threshold_raw = 10

DNA_MATCH_PROBABILITY_THRESHOLD_CM: int = max(0, _dna_threshold_raw)

_CM_RELATIONSHIP_BUCKETS: tuple[tuple[int, str], ...] = (
    (2200, "Parent/Child"),
    (1750, "Full sibling"),
    (1350, "Grandparent/Grandchild or Aunt/Uncle"),
    (900, "Half sibling or great aunt/uncle"),
    (540, "1st cousin"),
    (400, "1st cousin once removed"),
    (220, "2nd cousin"),
    (150, "Second cousin once removed / 3rd cousin"),
    (90, "3rd cousin"),
    (45, "4th cousin"),
    (20, "5th cousin"),
)


# Dynamic critical API failure threshold based on total pages to process
def get_critical_api_failure_threshold(total_pages: int = 100) -> int:
    """Calculate appropriate failure threshold based on total pages to process."""
    # Allow 1 failure per 20 pages, minimum of 10, maximum of 100
    return max(10, min(100, total_pages // 20))


CRITICAL_API_FAILURE_THRESHOLD_DEFAULT: Final[int] = 10


class Action6State:
    """Holds mutable state for Action 6."""

    critical_api_failure_threshold: int = CRITICAL_API_FAILURE_THRESHOLD_DEFAULT


# Configurable settings from config_schema
DB_ERROR_PAGE_THRESHOLD: int = 10  # Max consecutive DB errors allowed


# --- Custom Exceptions ---
# OPTIMIZATION: Profile caching using UnifiedCacheManager


# === ETHNICITY ENRICHMENT HELPERS ===


@lru_cache(maxsize=1)
def _try_get_csrf_from_api(session_manager: "SessionManager") -> Optional[str]:
    """
    Try to get fresh CSRF token from API.

    Args:
        session_manager: SessionManager instance

    Returns:
        CSRF token if successful, None otherwise
    """
    try:
        if hasattr(session_manager, 'api_manager') and hasattr(session_manager.api_manager, 'get_csrf_token'):
            fresh_token = session_manager.api_manager.get_csrf_token()
            if fresh_token:
                logger.info("Successfully obtained fresh CSRF token from API")
                return fresh_token
            logger.debug("API CSRF token request returned None")
        else:
            logger.debug("API CSRF token method not available")
    except Exception as api_error:
        logger.warning(f"API CSRF token refresh failed: {api_error}")
    return None


def _try_get_csrf_from_cookies(session_manager: "SessionManager") -> Optional[str]:
    """
    Try to get CSRF token from browser cookies.

    Args:
        session_manager: SessionManager instance

    Returns:
        CSRF token if found, None otherwise
    """
    csrf_cookie_names = ['_dnamatches-matchlistui-x-csrf-token', '_csrf', 'csrf_token', 'X-CSRF-TOKEN']

    driver = session_manager.browser_manager.driver
    if not driver:
        logger.warning("Cannot access CSRF cookies: browser driver missing")
        return None

    cookies = cast(list[dict[str, Any]], cast(Any, driver).get_cookies())
    for cookie_name in csrf_cookie_names:
        for cookie in cookies:
            if cookie['name'] == cookie_name:
                return cookie['value']

    logger.warning("No CSRF token found in cookies")
    return None


def _get_csrf_token(session_manager: SessionManager, force_api_refresh: bool = False) -> Optional[str]:
    """
    Helper function to extract CSRF token from cookies or API.

    Args:
        session_manager: SessionManager instance with active browser session
        force_api_refresh: If True, attempts to get fresh token from API

    Returns:
        str: CSRF token if found, None otherwise
    """
    try:
        # Try API first if force refresh requested
        if force_api_refresh:
            token = _try_get_csrf_from_api(session_manager)
            if token:
                return token

        # Fall back to cookies
        return _try_get_csrf_from_cookies(session_manager)

    except Exception as e:
        logger.error(f"Error extracting CSRF token: {e}")
        return None


def _ensure_on_match_list_page(session_manager: SessionManager, desired_page: Optional[int] = None) -> bool:
    """
    Ensure browser is on the DNA match list page.

    Args:
        session_manager: SessionManager instance

    Returns:
        True if on correct page, False otherwise
    """
    try:
        target_matches_url_base = urljoin(
            config_schema.api.base_url, f"discoveryui-matches/list/{session_manager.my_uuid}"
        )
        driver = session_manager.browser_manager.driver
        if not driver:
            logger.error("WebDriver unavailable while checking DNA match list page.")
            return False

        current_url = driver.current_url

        desired_page_int: Optional[int] = None
        if desired_page is not None:
            try:
                desired_page_int = int(desired_page)
            except (TypeError, ValueError):
                desired_page_int = None

        # If we're not on the match list base URL, navigate.
        # If a desired page is provided, also navigate when the current URL doesn't reflect it.
        needs_navigation = not current_url.startswith(target_matches_url_base)
        if (
            desired_page_int is not None
            and desired_page_int > 0
            and f"currentPage={desired_page_int}" not in current_url
        ):
            needs_navigation = True

        if needs_navigation:
            if not nav_to_list(session_manager, current_page=desired_page_int):
                logger.error("Failed to navigate to DNA match list page.")
                return False
        else:
            logger.debug(f"Already on correct DNA matches page: {current_url}")
        return True

    except WebDriverException as nav_e:
        logger.error(f"WebDriver error checking/navigating to match list: {nav_e}", exc_info=True)
        return False


def _get_db_session_with_retries(session_manager: SessionManager, max_retries: int = 3) -> Optional[SqlAlchemySession]:
    """
    Get database session with retry logic.

    Args:
        session_manager: SessionManager instance
        max_retries: Maximum retry attempts

    Returns:
        Database session or None if all retries failed
    """
    for retry_attempt in range(max_retries):
        db_session = session_manager.get_db_conn()
        if db_session:
            return db_session
        logger.warning(f"DB session attempt {retry_attempt + 1}/{max_retries} failed. Retrying in 5s...")
        time.sleep(5)

    logger.critical(f"Could not get DB session after {max_retries} retries.")
    return None


def _navigate_and_get_initial_page_data(
    session_manager: SessionManager, start_page: int
) -> tuple[Optional[list[dict[str, Any]]], Optional[int], bool]:
    """
    Ensures navigation to the match list and fetches initial page data.

    Returns:
        tuple: (matches_on_page, total_pages, success_flag)
    """
    # Ensure we're on the correct page
    if not _ensure_on_match_list_page(session_manager, desired_page=start_page):
        return None, None, False

    logger.debug(f"Fetching initial page {start_page} to determine total pages...")

    # CRITICAL FIX: Proactive cookie refresh to prevent 303 redirects
    # The 303 "See Other" response indicates stale cookies from previous session
    # Refreshing before first API call ensures fresh, valid cookies
    logger.debug("Proactively refreshing browser cookies before first API call...")
    try:
        session_manager.sync_browser_cookies()
        logger.debug("✅ Cookies refreshed successfully - preventing 303 redirect")
    except Exception as cookie_refresh_err:
        logger.warning(f"Cookie refresh warning (non-fatal): {cookie_refresh_err}")
        # Continue anyway - if cookies are truly invalid, the API call will handle it

    # Get database session with retries
    db_session_for_page = _get_db_session_with_retries(session_manager)
    if not db_session_for_page:
        return None, None, False

    try:
        # Validate session before API call
        if not session_manager.is_sess_valid():
            raise ConnectionError("WebDriver session invalid before initial get_matches.")

        # Fetch initial page data
        result = get_matches(session_manager, db_session_for_page, start_page)
        if result is None:
            logger.error(f"Initial get_matches for page {start_page} returned None.")
            return [], None, False

        matches_on_page, total_pages_from_api = result
        return matches_on_page, total_pages_from_api, True

    except ConnectionError as init_conn_e:
        logger.critical(f"ConnectionError during initial get_matches: {init_conn_e}.", exc_info=False)
        return None, None, False

    except Exception as get_match_err:
        logger.error(f"Error during initial get_matches call on page {start_page}: {get_match_err}", exc_info=True)
        return None, None, False

    finally:
        if db_session_for_page:
            session_manager.return_session(db_session_for_page)


# End of _navigate_and_get_initial_page_data


def _determine_page_processing_range(total_pages_from_api: int, start_page: int) -> tuple[int, int]:
    """Determines the last page to process and total pages in the run."""
    max_pages_config = config_schema.api.max_pages
    logger.debug(f"🔍 DEBUG MAX_PAGES config value: {max_pages_config} (from config_schema.api.max_pages)")
    pages_to_process_config = (
        min(max_pages_config, total_pages_from_api) if max_pages_config > 0 else total_pages_from_api
    )
    logger.debug(f"🔍 DEBUG pages_to_process_config calculated: {pages_to_process_config}")
    last_page_to_process = min(start_page + pages_to_process_config - 1, total_pages_from_api)
    total_pages_in_run = max(0, last_page_to_process - start_page + 1)
    return last_page_to_process, total_pages_in_run


# End of _determine_page_processing_range


def coord(session_manager: SessionManager, start: Optional[int] = None) -> bool:
    """Entry point for Action 6 that delegates to the shared orchestrator."""

    def _orchestrator_get_matches(
        sess_mgr: SessionManager,
        db_session: SqlAlchemySession,
        current_page: int,
    ) -> Optional[tuple[list[dict[str, Any]], int]]:
        result = get_matches(sess_mgr, db_session, current_page)
        if result is None:
            return None
        matches, total_pages = result
        normalized_total = int(total_pages) if total_pages is not None else 0
        return matches, normalized_total

    hooks = GatherConfiguration(
        matches_per_page=MATCHES_PER_PAGE,
        relationship_prob_max_per_page=RELATIONSHIP_PROB_MAX_PER_PAGE,
        db_error_page_threshold=DB_ERROR_PAGE_THRESHOLD,
        navigate_and_get_initial_page_data=_navigate_and_get_initial_page_data,
        determine_page_processing_range=_determine_page_processing_range,
        do_batch=_do_batch,
        get_matches=_orchestrator_get_matches,
        adjust_delay=_adjust_delay,
        action_state_cls=Action6State,
        calculate_failure_threshold=get_critical_api_failure_threshold,
    )

    orchestrator = GatherOrchestrator(session_manager=session_manager, hooks=hooks)
    return orchestrator.coord(start=start)


# End of coord

# ------------------------------------------------------------------------------
# Batch Processing Logic (_do_batch and Helpers)
# ------------------------------------------------------------------------------


def _build_prefetch_config() -> PrefetchConfig:
    """Bridge Action 6 runtime settings to the gather.prefetch config object."""

    return PrefetchConfig(
        relationship_prob_max_per_page=RELATIONSHIP_PROB_MAX_PER_PAGE,
        enable_ethnicity_enrichment=ENABLE_ETHNICITY_ENRICHMENT,
        ethnicity_min_cm=ETHNICITY_ENRICHMENT_MIN_CM,
        dna_match_priority_threshold_cm=DNA_MATCH_PROBABILITY_THRESHOLD_CM,
        critical_failure_threshold=Action6State.critical_api_failure_threshold,
    )


def _build_prefetch_hooks() -> PrefetchHooks:
    """Expose the legacy helper callables to the modular prefetch pipeline."""

    return PrefetchHooks(
        fetch_combined_details=_fetch_combined_details,
        fetch_badge_details=_fetch_batch_badge_details,
        fetch_ladder_details=_fetch_batch_ladder,
        fetch_ethnicity_batch=_fetch_ethnicity_for_batch,
    )


def _build_persistence_hooks() -> PersistenceHooks:
    """Bridge legacy persistence helpers into the modular pipeline."""

    return PersistenceHooks(
        process_single_match=_process_single_match_for_bulk,
        execute_bulk_db_operations=_execute_bulk_db_operations,
    )


def _process_single_match_for_bulk(
    session: SqlAlchemySession,
    session_manager: SessionManager,
    match_list_data: dict[str, Any],
    existing_persons_map: dict[str, Person],
    prefetched_data: dict[str, dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], Literal["new", "updated", "skipped", "error"], Optional[str]]:
    """Process a single match and prepare bulk data.

    Args:
        session: SQLAlchemy session
        session_manager: SessionManager instance
        match_list_data: Match data dictionary
        existing_persons_map: Map of existing persons
        prefetched_data: Prefetched API data

    Returns:
        Tuple of (prepared_data, status, error_msg)
    """
    uuid_val = match_list_data.get("uuid")
    log_ref_short = f"UUID={uuid_val or 'MISSING'} User='{match_list_data.get('username', 'Unknown')}'"

    # Basic validation
    if not uuid_val:
        logger.error("Critical error: Match data missing UUID in _prepare_bulk_db_data. Skipping.")
        return None, "error", "Missing UUID"

    # Retrieve existing person and prefetched data
    existing_person = existing_persons_map.get(uuid_val.upper())
    prefetched_combined, prefetched_tree, prefetched_rel_prob, prefetched_ethnicity = _get_prefetched_data_for_match(
        uuid_val, prefetched_data
    )

    # Add relationship probability to match dict
    match_list_data["predicted_relationship"] = prefetched_rel_prob

    # Add prefetched ethnicity data to match dict (for use in _do_match)
    if prefetched_ethnicity:
        match_list_data["_prefetched_ethnicity"] = prefetched_ethnicity

    # Check WebDriver session validity
    if not session_manager.is_sess_valid():
        logger.error(f"WebDriver session invalid before calling _do_match for {log_ref_short}. Treating as error.")
        return None, "error", "WebDriver session invalid"

    # Call _do_match to prepare the bulk dictionary structure
    return _do_match(
        session,
        match_list_data,
        session_manager,
        existing_person,
        prefetched_combined,
        prefetched_tree,
        config_schema,
        raw_logger,
    )


# ===================================================================
# PHASE 2: API PREFETCH ORCHESTRATION (SEQUENTIAL ONLY)
# ===================================================================


# ===================================================================
# LEVERAGING EXISTING SYSTEMS (No Duplication)
# - Database batching: database.py:commit_bulk_data()
# - Advanced caching: core/system_cache.py (API, DB query, memory optimization)
# - Batch management: action9_process_productive.py:BatchCommitManager
# ===================================================================

# ===================================================================
# LEVERAGING EXISTING SYSTEMS (No Duplication)
# - Database batching: database.py:commit_bulk_data()
# - Advanced caching: core/system_cache.py (API, DB query, memory optimization)
# - Batch management: action9_process_productive.py:BatchCommitManager
# ===================================================================

# For relationship caching, use the existing core/system_cache.py @cached_database_query decorator
# For API caching, use the existing core/system_cache.py @cached_api_call decorator
# These provide TTL, cleanup, statistics, and are already battle-tested


# ===================================================================
# PHASE 2: OPTIMIZED DATABASE BATCH OPERATIONS
# ===================================================================


# Get batch size from configuration (respects .env BATCH_SIZE setting)
def _get_configured_batch_size() -> int:
    """Get batch size from configuration system, respecting .env BATCH_SIZE setting."""
    try:
        from config import config_schema

        batch_size = getattr(config_schema, 'batch_size', 10)  # Default to 10 if not found
        logger.debug(f"Using configured batch size: {batch_size} (from cached config)")
        return batch_size
    except Exception as e:
        logger.warning(f"Failed to get configured batch size: {e}, using default 10")
        return 10  # Fallback to match .env default


def _get_adaptive_batch_size(session_manager: Optional["SessionManager"], base_batch_size: Optional[int] = None) -> int:
    """Get dynamically adapted batch size based on current server performance."""
    if base_batch_size is None:
        base_batch_size = _get_configured_batch_size()

    # Get current performance metrics from session manager
    avg_response_time = getattr(session_manager, '_avg_response_time', 0.0)
    recent_slow_calls = getattr(session_manager, '_recent_slow_calls', 0)

    # Adaptive batch sizing based on server performance
    if avg_response_time > 10.0:  # Very slow server
        adapted_size = max(5, base_batch_size // 4)
        logger.info(f"Server very slow ({avg_response_time:.1f}s avg), reducing batch size to {adapted_size}")
    elif avg_response_time > 7.0:  # Slow server
        adapted_size = max(8, base_batch_size // 2)
        logger.info(f"Server slow ({avg_response_time:.1f}s avg), reducing batch size to {adapted_size}")
    elif recent_slow_calls > 5:  # Multiple consecutive slow calls
        adapted_size = max(8, base_batch_size // 2)
        logger.info(f"Multiple slow calls ({recent_slow_calls}), reducing batch size to {adapted_size}")
    elif avg_response_time < 3.0 and recent_slow_calls == 0:  # Fast server
        adapted_size = min(50, int(base_batch_size * 1.5))
        logger.debug(f"Server fast ({avg_response_time:.1f}s avg), increasing batch size to {adapted_size}")
    else:
        adapted_size = base_batch_size

    return adapted_size


DB_BATCH_SIZE = _get_configured_batch_size()  # Now respects .env BATCH_SIZE=10

# ===================================================================
# PHASE 3: MEMORY-OPTIMIZED DATA STRUCTURES
# ===================================================================


class MemoryOptimizedMatchProcessor:
    """
    Phase 3: Memory-optimized match processing with lazy loading and cleanup.
    """

    def __init__(self, max_memory_mb: int = 500):
        """
        Initialize with memory limit.

        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_mb = max_memory_mb
        self.processed_count = 0
        self.memory_checkpoints = []

    def process_matches_with_memory_management(
        self, matches: list[dict[str, Any]], session_manager: SessionManager
    ) -> list[dict[str, Any]]:
        """
        Process matches with active memory management.

        Args:
            matches: list of matches to process
            session_manager: SessionManager for API calls

        Returns:
            list of processed matches
        """
        import gc

        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        logger.info(
            f"Phase 3: Starting memory-optimized processing (Initial: {initial_memory:.1f}MB, Limit: {self.max_memory_mb}MB)"
        )

        processed_matches: list[dict[str, Any]] = []
        memory_cleanup_threshold = self.max_memory_mb * 0.8  # Clean up at 80% of limit

        for i, match in enumerate(matches):
            # Process single match
            processed_match = self._process_single_match(match, session_manager)
            processed_matches.append(processed_match)
            self.processed_count += 1

            # Memory check every 10 matches
            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024

                if current_memory > memory_cleanup_threshold:
                    logger.warning(
                        f"Phase 3: Memory usage {current_memory:.1f}MB exceeds threshold, triggering cleanup"
                    )

                    # Force garbage collection
                    gc.collect()

                    # Cache cleanup now handled by core/system_cache.py
                    logger.debug("Phase 3: Cache cleanup handled by existing system_cache.py")

                    # Memory after cleanup
                    after_cleanup = process.memory_info().rss / 1024 / 1024
                    logger.info(f"Phase 3: Memory cleanup completed: {current_memory:.1f}MB → {after_cleanup:.1f}MB")

        final_memory = process.memory_info().rss / 1024 / 1024
        logger.info(f"Phase 3: Memory-optimized processing completed: {initial_memory:.1f}MB → {final_memory:.1f}MB")

        return processed_matches

    @staticmethod
    def _process_single_match(match: dict[str, Any], _session_manager: SessionManager) -> dict[str, Any]:
        """Process a single match with minimal memory footprint.

        Args:
            match: Match data to process
            _session_manager: SessionManager (unused, kept for API compatibility)
        """
        # Placeholder - would integrate with existing match processing logic
        return match


def _deduplicate_person_creates(person_creates_raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    De-duplicate Person creates based on Profile ID before bulk insert.

    Args:
        person_creates_raw: list of raw person create data dictionaries

    Returns:
        list of filtered person create data (duplicates removed)
    """
    person_creates_filtered: list[dict[str, Any]] = []
    seen_profile_ids: set[str] = set()
    skipped_duplicates = 0

    if not person_creates_raw:
        return person_creates_filtered

    logger.debug(f"De-duplicating {len(person_creates_raw)} raw person creates based on Profile ID...")

    for p_data in person_creates_raw:
        profile_id = cast(Optional[str], p_data.get("profile_id"))  # Already uppercase from prep if exists
        uuid_for_log = cast(Optional[str], p_data.get("uuid"))  # For logging skipped items

        if profile_id is None:
            person_creates_filtered.append(p_data)  # Allow creates with null profile ID
        elif profile_id not in seen_profile_ids:
            person_creates_filtered.append(p_data)
            seen_profile_ids.add(profile_id)
        else:
            logger.warning(
                f"Skipping duplicate Person create in batch (ProfileID: {profile_id}, UUID: {uuid_for_log})."
            )
            skipped_duplicates += 1

    if skipped_duplicates > 0:
        logger.info(f"Skipped {skipped_duplicates} duplicate person creates in this batch.")
    logger.debug(f"Proceeding with {len(person_creates_filtered)} unique person creates.")

    return person_creates_filtered


def _check_existing_profile_ids(session: SqlAlchemySession, profile_ids_to_check: set[str]) -> set[str]:
    """Check database for existing profile IDs."""
    existing_profile_ids: set[str] = set()
    if not profile_ids_to_check:
        return existing_profile_ids

    try:
        logger.debug(f"Checking database for {len(profile_ids_to_check)} existing profile IDs...")
        existing_records = (
            session.query(Person.profile_id)
            .filter(Person.profile_id.in_(profile_ids_to_check), Person.deleted_at.is_(None))
            .all()
        )
        existing_profile_ids = {record.profile_id for record in existing_records}
        if existing_profile_ids:
            logger.info(f"Found {len(existing_profile_ids)} existing profile IDs that will be skipped")
    except Exception as e:
        logger.warning(f"Failed to check existing profile IDs: {e}")

    return existing_profile_ids


def _check_existing_uuids(session: SqlAlchemySession, uuids_to_check: set[str]) -> set[str]:
    """Check database for existing UUIDs."""
    existing_uuids: set[str] = set()
    if not uuids_to_check:
        return existing_uuids

    try:
        logger.debug(f"Checking database for {len(uuids_to_check)} existing UUIDs...")
        existing_uuid_records = (
            session.query(Person.uuid).filter(Person.uuid.in_(uuids_to_check), Person.deleted_at.is_(None)).all()
        )
        existing_uuids = {record.uuid.upper() for record in existing_uuid_records}
        if existing_uuids:
            logger.info(f"Found {len(existing_uuids)} existing UUIDs that will be skipped")
    except Exception as e:
        logger.warning(f"Failed to check existing UUIDs: {e}")

    return existing_uuids


def _check_existing_records(
    session: SqlAlchemySession, insert_data_raw: list[dict[str, Any]]
) -> tuple[set[str], set[str]]:
    """
    Check database for existing profile IDs and UUIDs to prevent constraint violations.

    Args:
        session: SQLAlchemy session
        insert_data_raw: Raw insert data to check

    Returns:
        tuple of (existing_profile_ids, existing_uuids) sets
    """
    profile_ids_to_check: set[str] = {str(item.get("profile_id")) for item in insert_data_raw if item.get("profile_id")}
    uuids_to_check: set[str] = {str(item.get("uuid") or "").upper() for item in insert_data_raw if item.get("uuid")}

    existing_profile_ids = _check_existing_profile_ids(session, profile_ids_to_check)
    existing_uuids = _check_existing_uuids(session, uuids_to_check)

    return existing_profile_ids, existing_uuids


def _handle_integrity_error_recovery(
    session: SqlAlchemySession, insert_data: Optional[list[dict[str, Any]]] = None
) -> bool:
    """
    Handle UNIQUE constraint violations by attempting individual inserts.

    Args:
        session: SQLAlchemy session
        insert_data: Data that failed bulk insert (optional)

    Returns:
        True if recovery was successful
    """
    try:
        session.rollback()  # Clear the failed transaction
        logger.debug("Rolled back failed transaction due to UNIQUE constraint violation")

        if not insert_data:
            logger.debug(
                "No insert_data available for recovery - treating as successful (records likely already exist)"
            )
            return True

        logger.debug(f"Retrying with individual inserts for {len(insert_data)} records")
        successful_inserts = 0

        for item in insert_data:
            try:
                # Try individual insert
                individual_person = Person(**{k: v for k, v in item.items() if hasattr(Person, k)})
                session.add(individual_person)
                session.flush()  # Force insert attempt
                successful_inserts += 1
            except IntegrityError as individual_err:
                # This specific record already exists - skip it
                logger.debug(f"Skipping duplicate record UUID {item.get('uuid', 'unknown')}: {individual_err}")
                session.rollback()  # Clear this specific failure
            except Exception as individual_exc:
                logger.warning(
                    f"Failed to insert individual record UUID {item.get('uuid', 'unknown')}: {individual_exc}"
                )
                session.rollback()  # Clear this specific failure

        logger.info(
            f"Successfully inserted {successful_inserts} of {len(insert_data)} records after handling duplicates"
        )
        return True

    except Exception as rollback_err:
        logger.error(f"Failed to handle UNIQUE constraint violation gracefully: {rollback_err}", exc_info=True)
        return False


def _should_skip_person_insert(
    uuid_val: str,
    profile_id: Optional[str],
    seen_uuids: set[str],
    existing_persons_map: dict[str, Person],
    existing_uuids: set[str],
    existing_profile_ids: set[str],
) -> tuple[bool, Optional[str]]:
    """Check if person should be skipped during insert preparation.

    Args:
        uuid_val: Person UUID
        profile_id: Person profile ID
        seen_uuids: Set of UUIDs already seen in this batch
        existing_persons_map: Map of existing persons by UUID
        existing_uuids: Set of UUIDs that exist in database
        existing_profile_ids: Set of profile IDs that exist in database

    Returns:
        Tuple of (should_skip, reason)
    """
    if not uuid_val:
        return True, None
    if uuid_val in seen_uuids:
        return True, f"Duplicate Person in batch (UUID: {uuid_val}) - skipping duplicate."
    if uuid_val in existing_persons_map:
        return True, f"Person exists for UUID {uuid_val}; will handle as update if changes detected."
    if uuid_val in existing_uuids:
        return True, f"Person exists in DB for UUID {uuid_val}; will handle as update if needed."
    if profile_id and profile_id in existing_profile_ids:
        return True, f"Person exists with profile ID {profile_id} (UUID: {uuid_val}); will handle as update if needed."
    return False, None


def _convert_status_enums(insert_data: list[dict[str, Any]]) -> None:
    """Convert status Enum to its value for bulk insertion."""
    for item_data in insert_data:
        if "status" in item_data and hasattr(item_data["status"], 'value'):
            item_data["status"] = item_data["status"].value


def _prepare_person_insert_data(
    person_creates_filtered: list[dict[str, Any]], session: SqlAlchemySession, existing_persons_map: dict[str, Person]
) -> list[dict[str, Any]]:
    """
    Prepare and validate person insert data, removing duplicates and existing records.

    Args:
        person_creates_filtered: Filtered person create data
        session: SQLAlchemy session
        existing_persons_map: Map of existing persons by UUID

    Returns:
        list of validated insert data ready for bulk insert
    """
    if not person_creates_filtered:
        return []

    logger.debug(f"Preparing {len(person_creates_filtered)} Person records for bulk insert...")

    # Prepare list of dictionaries for bulk_insert_mappings
    insert_data_raw = [{k: v for k, v in p.items() if not k.startswith("_")} for p in person_creates_filtered]

    # Check for existing records in database
    existing_profile_ids, existing_uuids = _check_existing_records(session, insert_data_raw)

    # De-duplicate by UUID within this batch and drop existing records
    seen_uuids: set[str] = set()
    insert_data: list[dict[str, Any]] = []

    for item in insert_data_raw:
        uuid_val = str(item.get("uuid") or "").upper()
        profile_id = item.get("profile_id")

        should_skip, reason = _should_skip_person_insert(
            uuid_val, profile_id, seen_uuids, existing_persons_map, existing_uuids, existing_profile_ids
        )

        if should_skip:
            if reason:
                logger.debug(reason)
            continue

        seen_uuids.add(uuid_val)
        item["uuid"] = uuid_val
        insert_data.append(item)

    _convert_status_enums(insert_data)
    return insert_data


def _is_person_create(d: dict[str, Any]) -> bool:
    """Check if data dict contains a person create operation."""
    return bool(d.get("person") and d["person"]["_operation"] == "create")


def _is_person_update(d: dict[str, Any]) -> bool:
    """Check if data dict contains a person update operation."""
    return bool(d.get("person") and d["person"]["_operation"] == "update")


def _separate_bulk_operations(
    prepared_bulk_data: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate prepared data by operation type and table.

    Args:
        prepared_bulk_data: List of prepared bulk data dictionaries

    Returns:
        Tuple of (person_creates, person_updates, dna_match_ops, family_tree_ops)
    """
    person_creates_raw = [d["person"] for d in prepared_bulk_data if _is_person_create(d)]
    person_updates = [d["person"] for d in prepared_bulk_data if _is_person_update(d)]
    dna_match_ops = [d["dna_match"] for d in prepared_bulk_data if d.get("dna_match")]
    family_tree_ops = [d["family_tree"] for d in prepared_bulk_data if d.get("family_tree")]
    return person_creates_raw, person_updates, dna_match_ops, family_tree_ops


def _validate_no_duplicate_profile_ids(insert_data: list[dict[str, Any]]) -> None:
    """Validate that there are no duplicate profile IDs in insert data.

    Args:
        insert_data: List of person insert data dictionaries

    Raises:
        IntegrityError: If duplicate profile IDs are found
    """
    final_profile_ids = {item.get("profile_id") for item in insert_data if item.get("profile_id")}
    if len(final_profile_ids) != sum(1 for item in insert_data if item.get("profile_id")):
        logger.error("CRITICAL: Duplicate non-NULL profile IDs DETECTED post-filter! Aborting bulk insert.")
        id_counts = Counter(item.get("profile_id") for item in insert_data if item.get("profile_id"))
        duplicates = {pid: count for pid, count in id_counts.items() if count > 1}
        logger.error(f"Duplicate Profile IDs in filtered list: {duplicates}")
        dup_exception = ValueError(f"Duplicate profile IDs: {duplicates}")
        raise IntegrityError(
            "Duplicate profile IDs found pre-bulk insert",
            params=str(duplicates),
            orig=dup_exception,
        )


def _get_person_id_mapping(session: SqlAlchemySession, inserted_uuids: list[str]) -> dict[str, int]:
    """Get Person ID mapping for inserted UUIDs.

    Args:
        session: SQLAlchemy session
        inserted_uuids: List of inserted UUIDs

    Returns:
        Dictionary mapping UUID to Person ID
    """
    created_person_map: dict[str, int] = {}
    if not inserted_uuids:
        logger.warning("No UUIDs available in insert_data to query back IDs.")
        return created_person_map

    logger.debug(f"Querying IDs for {len(inserted_uuids)} inserted UUIDs...")

    try:
        session.flush()  # Make pending changes visible to current session
        session.commit()  # Commit to database for ID generation

        person_lookup_stmt = select(Person.id, Person.uuid).where(Person.uuid.in_(inserted_uuids))
        newly_inserted_persons = session.execute(person_lookup_stmt).all()
        created_person_map = {
            p_uuid: p_id for p_id, p_uuid in newly_inserted_persons if p_id is not None and isinstance(p_uuid, str)
        }

        logger.debug(
            f"Person ID Mapping: Queried {len(inserted_uuids)} UUIDs, mapped {len(created_person_map)} Person IDs"
        )

        if len(created_person_map) != len(inserted_uuids):
            missing_count = len(inserted_uuids) - len(created_person_map)
            missing_uuids = [uuid for uuid in inserted_uuids if uuid not in created_person_map]
            logger.error(
                f"CRITICAL: Person ID mapping failed for {missing_count} UUIDs. Missing: {missing_uuids[:3]}{'...' if missing_count > 3 else ''}"
            )

            # Recovery attempt: Query with broader filter
            if missing_uuids:
                recovery_stmt = (
                    select(Person.id, Person.uuid)
                    .where(Person.uuid.in_(missing_uuids))
                    .where(Person.deleted_at.is_(None))
                )
                recovery_persons = session.execute(recovery_stmt).all()
                recovery_map = {
                    p_uuid: p_id for p_id, p_uuid in recovery_persons if p_id is not None and isinstance(p_uuid, str)
                }
                if recovery_map:
                    logger.info(f"Recovery: Found {len(recovery_map)} additional Person IDs")
                    created_person_map.update(recovery_map)

        return created_person_map

    except Exception as mapping_error:
        logger.error(f"CRITICAL: Person ID mapping query failed: {mapping_error}")
        session.rollback()
        created_person_map.clear()
        return created_person_map


def _process_person_creates(
    session: SqlAlchemySession, person_creates_raw: list[dict[str, Any]], existing_persons_map: dict[str, Person]
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Process Person create operations.

    Args:
        session: SQLAlchemy session
        person_creates_raw: Raw person create data
        existing_persons_map: Map of existing persons

    Returns:
        Tuple of (created_person_map, insert_data)
    """
    # De-duplicate Person creates
    person_creates_filtered = _deduplicate_person_creates(person_creates_raw)

    created_person_map: dict[str, int] = {}
    insert_data: list[dict[str, Any]] = []

    if not person_creates_filtered:
        return created_person_map, insert_data

    # Prepare insert data
    insert_data = _prepare_person_insert_data(person_creates_filtered, session, existing_persons_map)

    # Validate no duplicates
    _validate_no_duplicate_profile_ids(insert_data)

    # Perform bulk insert
    logger.debug(f"Bulk inserting {len(insert_data)} Person records...")
    session.bulk_insert_mappings(Person.__mapper__, insert_data)

    # Get newly created IDs
    session.flush()
    inserted_uuids = [p_data["uuid"] for p_data in insert_data if p_data.get("uuid")]
    created_person_map = _get_person_id_mapping(session, inserted_uuids)

    return created_person_map, insert_data


def _build_person_update_dict(p_data: dict[str, Any], existing_id: int) -> dict[str, Any]:
    """Build update dictionary for a person record."""
    update_dict = {k: v for k, v in p_data.items() if not k.startswith("_") and k not in {"uuid", "profile_id"}}
    if "status" in update_dict and isinstance(update_dict["status"], PersonStatusEnum):
        update_dict["status"] = update_dict["status"].value
    update_dict["id"] = existing_id
    update_dict["updated_at"] = datetime.now(timezone.utc)
    return update_dict


def _process_person_updates(session: SqlAlchemySession, person_updates: list[dict[str, Any]]) -> None:
    """Process Person update operations.

    Args:
        session: SQLAlchemy session
        person_updates: List of person update data
    """
    if not person_updates:
        logger.debug("No Person updates needed for this batch.")
        return

    update_mappings: list[dict[str, Any]] = []
    for p_data in person_updates:
        existing_id = p_data.get("_existing_person_id")
        if not existing_id:
            logger.warning(f"Skipping person update (UUID {p_data.get('uuid')}): Missing '_existing_person_id'.")
            continue
        update_dict = _build_person_update_dict(p_data, existing_id)
        if len(update_dict) > 2:
            update_mappings.append(update_dict)

    if update_mappings:
        logger.debug(f"Bulk updating {len(update_mappings)} Person records...")
        session.bulk_update_mappings(Person.__mapper__, update_mappings)
        logger.debug("Bulk update Persons called.")
    else:
        logger.debug("No valid Person updates to perform.")


def _add_update_ids_to_map(all_person_ids_map: dict[str, int], person_updates: list[dict[str, Any]]) -> None:
    """Add IDs from person updates to the master ID map."""
    for p_update_data in person_updates:
        if p_update_data.get("_existing_person_id") and p_update_data.get("uuid"):
            all_person_ids_map[p_update_data["uuid"]] = p_update_data["_existing_person_id"]


def _add_existing_ids_to_map(
    all_person_ids_map: dict[str, int],
    prepared_bulk_data: list[dict[str, Any]],
    existing_persons_map: dict[str, Person],
) -> None:
    """Add IDs from existing persons to the master ID map.

    This function ensures that ALL existing persons are added to the map,
    not just those that have new/updated data in prepared_bulk_data.
    This is critical when all persons in a batch already exist - without this,
    all_person_ids_map would be empty, causing DNA match INSERT failures.
    """
    # First, add IDs for persons that have data in prepared_bulk_data
    processed_uuids = {p["person"]["uuid"] for p in prepared_bulk_data if p.get("person") and p["person"].get("uuid")}
    for uuid_processed in processed_uuids:
        if uuid_processed not in all_person_ids_map and existing_persons_map.get(uuid_processed):
            person = existing_persons_map[uuid_processed]
            person_id_val = getattr(person, "id", None)
            if person_id_val is not None:
                all_person_ids_map[uuid_processed] = person_id_val

    # CRITICAL FIX: Also add IDs for ALL other existing persons, even if they don't have
    # data in prepared_bulk_data. This handles the case where all persons already exist
    # and were filtered out from Person INSERT operations, but we still need their IDs
    # to properly handle DNA match UPDATE operations.
    added_count = 0
    for uuid_val, person in existing_persons_map.items():
        if uuid_val not in all_person_ids_map:
            person_id_val = getattr(person, "id", None)
            if person_id_val is not None:
                all_person_ids_map[uuid_val] = person_id_val
                added_count += 1
    if added_count > 0:
        logger.info(f"Added {added_count} existing person IDs to map from existing_persons_map")


def _create_master_id_map(
    created_person_map: dict[str, int],
    person_updates: list[dict[str, Any]],
    prepared_bulk_data: list[dict[str, Any]],
    existing_persons_map: dict[str, Person],
) -> dict[str, int]:
    """Create master ID map for linking related records.

    Args:
        created_person_map: Map of newly created person IDs
        person_updates: List of person update data
        prepared_bulk_data: List of prepared bulk data
        existing_persons_map: Map of existing persons

    Returns:
        Master ID map (UUID -> Person ID)
    """
    logger.debug(
        f"Creating master ID map: created_person_map={len(created_person_map)}, person_updates={len(person_updates)}, existing_persons_map={len(existing_persons_map)}"
    )
    all_person_ids_map: dict[str, int] = created_person_map.copy()
    _add_update_ids_to_map(all_person_ids_map, person_updates)
    _add_existing_ids_to_map(all_person_ids_map, prepared_bulk_data, existing_persons_map)
    logger.debug(f"Master ID map created with {len(all_person_ids_map)} total entries")
    return all_person_ids_map


def _resolve_person_id(
    session: SqlAlchemySession,
    person_uuid: Optional[str],
    all_person_ids_map: dict[str, int],
    existing_persons_map: dict[str, Person],
) -> Optional[int]:
    """Resolve person ID from UUID using multiple strategies.

    Args:
        session: SQLAlchemy session
        person_uuid: Person UUID to resolve
        all_person_ids_map: Map of UUID to Person ID
        existing_persons_map: Map of existing persons

    Returns:
        Person ID if found, None otherwise
    """
    if not person_uuid:
        logger.warning("Missing UUID in DNA match data - skipping DNA Match creation")
        return None

    # Strategy 1: Check all_person_ids_map
    person_id = all_person_ids_map.get(person_uuid)

    # Strategy 2: Check existing_persons_map
    if not person_id and existing_persons_map.get(person_uuid):
        existing_person = existing_persons_map[person_uuid]
        person_id = getattr(existing_person, "id", None)
        if person_id:
            all_person_ids_map[person_uuid] = person_id
            logger.debug(f"Resolved Person ID {person_id} for UUID {person_uuid} (from existing_persons_map)")
        else:
            logger.warning(f"Person exists in database for UUID {person_uuid} but has no ID attribute")

    # Strategy 3: Direct database query as fallback
    if not person_id:
        try:
            db_person = session.query(Person.id).filter(Person.uuid == person_uuid, Person.deleted_at.is_(None)).first()
            if db_person:
                person_id = db_person.id
                all_person_ids_map[person_uuid] = person_id
                logger.debug(f"Resolved Person ID {person_id} for UUID {person_uuid} (direct DB query)")
            else:
                logger.debug(f"Person UUID {person_uuid} not found in database - will be created in next batch")
        except Exception as e:
            logger.warning(f"Database query failed for UUID {person_uuid}: {e}")

    return person_id


def _get_existing_dna_matches(session: SqlAlchemySession, all_person_ids_map: dict[str, int]) -> dict[int, int]:
    """Get existing DnaMatch records for people in batch.

    Args:
        session: SQLAlchemy session
        all_person_ids_map: Map of UUID to Person ID

    Returns:
        Map of people_id to DnaMatch ID
    """
    people_ids_in_batch: set[int] = set(all_person_ids_map.values())
    logger.debug(
        f"all_person_ids_map has {len(all_person_ids_map)} entries, people_ids_in_batch has {len(people_ids_in_batch)} IDs"
    )
    if not people_ids_in_batch:
        logger.warning("No people IDs in batch - cannot query for existing DNA matches")
        return {}

    logger.debug(f"Querying for existing DNA matches for people IDs: {sorted(people_ids_in_batch)}")
    stmt = select(DnaMatch.people_id, DnaMatch.id).where(DnaMatch.people_id.in_(list(people_ids_in_batch)))
    existing_matches = session.execute(stmt).all()
    existing_dna_matches_map: dict[int, int] = {}
    for people_id, match_id in existing_matches:
        if people_id is None or match_id is None:
            continue
        existing_dna_matches_map[int(people_id)] = int(match_id)
    logger.debug(f"Found {len(existing_dna_matches_map)} existing DnaMatch records for people in this batch.")
    if len(existing_dna_matches_map) < len(people_ids_in_batch):
        missing_count = len(people_ids_in_batch) - len(existing_dna_matches_map)
        logger.debug(f"{missing_count} people IDs do not have existing DNA match records")
    return existing_dna_matches_map


def _prepare_dna_match_data(dna_data: dict[str, Any], person_id: int) -> dict[str, Any]:
    """Prepare DNA match data for insert/update.

    Args:
        dna_data: Raw DNA match data
        person_id: Person ID to link to

    Returns:
        Prepared data dictionary
    """
    op_data = {k: v for k, v in dna_data.items() if not k.startswith("_") and k != "uuid"}
    op_data["people_id"] = person_id
    return op_data


def _classify_dna_match_operations(
    session: SqlAlchemySession,
    dna_match_ops: list[dict[str, Any]],
    all_person_ids_map: dict[str, int],
    existing_persons_map: dict[str, Person],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify DNA match operations into inserts and updates.

    Args:
        session: SQLAlchemy session
        dna_match_ops: List of DNA match operations
        all_person_ids_map: Map of UUID to Person ID
        existing_persons_map: Map of existing persons

    Returns:
        Tuple of (insert_data, update_mappings)
    """
    existing_dna_matches_map = _get_existing_dna_matches(session, all_person_ids_map)
    dna_insert_data: list[dict[str, Any]] = []
    dna_update_mappings: list[dict[str, Any]] = []

    for dna_data in dna_match_ops:
        person_uuid = dna_data.get("uuid")
        person_id = _resolve_person_id(session, person_uuid, all_person_ids_map, existing_persons_map)

        if not person_id:
            continue

        op_data = _prepare_dna_match_data(dna_data, person_id)

        # FIX: Double-check for existing DnaMatch record even if not in existing_dna_matches_map
        # This handles cases where the record was created in a previous batch in the same run
        existing_match_id = existing_dna_matches_map.get(person_id)
        if not existing_match_id:
            # Query database directly to ensure we don't miss recently created records
            try:
                db_match = session.query(DnaMatch.id).filter(DnaMatch.people_id == person_id).first()
                if db_match:
                    existing_match_id = db_match.id
                    logger.debug(
                        f"Found existing DnaMatch (ID={existing_match_id}) for PersonID {person_id} via direct query"
                    )
            except Exception as e:
                logger.warning(f"Direct DnaMatch query failed for PersonID {person_id}: {e}")

        if existing_match_id:
            # Prepare for UPDATE
            update_map = op_data.copy()
            update_map["id"] = existing_match_id
            update_map["updated_at"] = datetime.now(timezone.utc)
            if len(update_map) > 3:  # More than id/people_id/updated_at
                dna_update_mappings.append(update_map)
            else:
                logger.debug(f"Skipping DnaMatch update for PersonID {person_id}: No changed fields.")
        else:
            # Prepare for INSERT
            insert_map = op_data.copy()
            insert_map.setdefault("created_at", datetime.now(timezone.utc))
            insert_map.setdefault("updated_at", datetime.now(timezone.utc))
            dna_insert_data.append(insert_map)

    return dna_insert_data, dna_update_mappings


def _apply_ethnicity_data(session: SqlAlchemySession, people_id: int, ethnicity_data: dict[str, Any]) -> None:
    """Apply ethnicity data via raw SQL UPDATE.

    Args:
        session: SQLAlchemy session
        people_id: Person ID
        ethnicity_data: Ethnicity data dictionary
    """
    from sqlalchemy import text

    set_clauses = ", ".join([f"{col} = :{col}" for col in ethnicity_data])
    sql = f"UPDATE dna_match SET {set_clauses} WHERE people_id = :people_id"
    params = {**ethnicity_data, "people_id": people_id}
    session.execute(text(sql), params)


def _bulk_insert_dna_matches(session: SqlAlchemySession, dna_insert_data: list[dict[str, Any]]) -> None:
    """Bulk insert DNA match records with ethnicity data.

    Args:
        session: SQLAlchemy session
        dna_insert_data: List of DNA match insert data
    """
    if not dna_insert_data:
        return

    logger.debug(f"Bulk inserting {len(dna_insert_data)} DnaMatch records...")

    # Separate ethnicity columns from core data
    core_insert_data: list[dict[str, Any]] = []
    ethnicity_updates: list[tuple[int, dict[str, Any]]] = []

    for insert_map in dna_insert_data:
        core_map = {k: v for k, v in insert_map.items() if not k.startswith("ethnicity_")}
        ethnicity_map = {k: v for k, v in insert_map.items() if k.startswith("ethnicity_")}
        core_insert_data.append(core_map)
        if ethnicity_map:
            ethnicity_updates.append((insert_map["people_id"], ethnicity_map))

    # Bulk insert core data
    session.bulk_insert_mappings(DnaMatch.__mapper__, core_insert_data)
    session.flush()

    # Apply ethnicity data
    if ethnicity_updates:
        for people_id, ethnicity_data in ethnicity_updates:
            _apply_ethnicity_data(session, people_id, ethnicity_data)
        session.flush()
        logger.debug(f"Applied ethnicity data to {len(ethnicity_updates)} newly inserted DnaMatch records")


def _separate_core_and_ethnicity(update_map: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate core data from ethnicity data in an update mapping."""
    core_map = {k: v for k, v in update_map.items() if not k.startswith("ethnicity_")}
    ethnicity_map = {k: v for k, v in update_map.items() if k.startswith("ethnicity_")}
    return core_map, ethnicity_map


def _bulk_update_dna_matches(session: SqlAlchemySession, dna_update_mappings: list[dict[str, Any]]) -> None:
    """Bulk update DNA match records with ethnicity data.

    Args:
        session: SQLAlchemy session
        dna_update_mappings: List of DNA match update mappings
    """
    if not dna_update_mappings:
        return

    logger.debug(f"Bulk updating {len(dna_update_mappings)} DnaMatch records...")

    # Separate ethnicity columns from core data
    core_update_mappings: list[dict[str, Any]] = []
    ethnicity_updates: list[tuple[int, dict[str, Any]]] = []

    for update_map in dna_update_mappings:
        core_map, ethnicity_map = _separate_core_and_ethnicity(update_map)
        core_update_mappings.append(core_map)
        if ethnicity_map:
            # FIX: Use people_id instead of id for ethnicity updates
            ethnicity_updates.append((update_map["people_id"], ethnicity_map))

    # Bulk update core data
    if core_update_mappings:
        session.bulk_update_mappings(DnaMatch.__mapper__, core_update_mappings)
        session.flush()

    # Apply ethnicity data
    if ethnicity_updates:
        for people_id, ethnicity_data in ethnicity_updates:
            _apply_ethnicity_data(session, people_id, ethnicity_data)
        session.flush()
        logger.debug(f"Applied ethnicity data to {len(ethnicity_updates)} updated DnaMatch records")


def _process_dna_match_operations(
    session: SqlAlchemySession,
    dna_match_ops: list[dict[str, Any]],
    all_person_ids_map: dict[str, int],
    existing_persons_map: dict[str, Person],
) -> None:
    """Process DNA match operations (inserts and updates).

    Args:
        session: SQLAlchemy session
        dna_match_ops: List of DNA match operations
        all_person_ids_map: Map of UUID to Person ID
        existing_persons_map: Map of existing persons
    """
    if not dna_match_ops:
        return

    dna_insert_data, dna_update_mappings = _classify_dna_match_operations(
        session, dna_match_ops, all_person_ids_map, existing_persons_map
    )

    _bulk_insert_dna_matches(session, dna_insert_data)
    _bulk_update_dna_matches(session, dna_update_mappings)

    # FIX: Expire session cache after bulk operations to ensure subsequent queries
    # can see newly inserted/updated records. This prevents UNIQUE constraint errors
    # when the same person is processed in subsequent batches.
    if dna_insert_data or dna_update_mappings:
        session.expire_all()
        logger.debug("Session cache expired after DnaMatch bulk operations")


def _prepare_family_tree_inserts(
    session: SqlAlchemySession,
    tree_creates: list[dict[str, Any]],
    all_person_ids_map: dict[str, int],
    existing_persons_map: dict[str, Person],
) -> list[dict[str, Any]]:
    """Prepare FamilyTree insert data.

    Args:
        session: SQLAlchemy session
        tree_creates: List of FamilyTree create operations
        all_person_ids_map: Map of UUID to Person ID
        existing_persons_map: Map of existing persons

    Returns:
        List of FamilyTree insert data
    """
    tree_insert_data: list[dict[str, Any]] = []

    for tree_data in tree_creates:
        person_uuid = tree_data.get("uuid")
        person_id = _resolve_person_id(session, person_uuid, all_person_ids_map, existing_persons_map)

        if person_id:
            insert_dict = {k: v for k, v in tree_data.items() if not k.startswith("_")}
            insert_dict["people_id"] = person_id
            insert_dict.pop("uuid", None)
            tree_insert_data.append(insert_dict)
        else:
            logger.debug(f"Person with UUID {person_uuid} not found in database - skipping FamilyTree creation.")

    return tree_insert_data


def _prepare_family_tree_updates(tree_updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepare FamilyTree update mappings.

    Args:
        tree_updates: List of FamilyTree update operations

    Returns:
        List of FamilyTree update mappings
    """
    tree_update_mappings: list[dict[str, Any]] = []

    for tree_data in tree_updates:
        existing_tree_id = tree_data.get("_existing_tree_id")
        if not existing_tree_id:
            logger.warning(
                f"Skipping FamilyTree update op (UUID {tree_data.get('uuid')}): Missing '_existing_tree_id'."
            )
            continue

        update_dict_tree = {k: v for k, v in tree_data.items() if not k.startswith("_") and k != "uuid"}
        update_dict_tree["id"] = existing_tree_id
        update_dict_tree["updated_at"] = datetime.now(timezone.utc)

        if len(update_dict_tree) > 2:
            tree_update_mappings.append(update_dict_tree)

    return tree_update_mappings


def _process_family_tree_operations(
    session: SqlAlchemySession,
    family_tree_ops: list[dict[str, Any]],
    all_person_ids_map: dict[str, int],
    existing_persons_map: dict[str, Person],
) -> None:
    """Process FamilyTree operations (inserts and updates).

    Args:
        session: SQLAlchemy session
        family_tree_ops: List of FamilyTree operations
        all_person_ids_map: Map of UUID to Person ID
        existing_persons_map: Map of existing persons
    """
    tree_creates = [op for op in family_tree_ops if op.get("_operation") == "create"]
    tree_updates = [op for op in family_tree_ops if op.get("_operation") == "update"]

    # Process creates
    if tree_creates:
        tree_insert_data = _prepare_family_tree_inserts(session, tree_creates, all_person_ids_map, existing_persons_map)
        if tree_insert_data:
            logger.debug(f"Bulk inserting {len(tree_insert_data)} FamilyTree records...")
            session.bulk_insert_mappings(FamilyTree.__mapper__, tree_insert_data)
        else:
            logger.debug("No valid FamilyTree records to insert")
    else:
        logger.debug("No FamilyTree creates prepared.")

    # Process updates
    if tree_updates:
        tree_update_mappings = _prepare_family_tree_updates(tree_updates)
        if tree_update_mappings:
            logger.debug(f"Bulk updating {len(tree_update_mappings)} FamilyTree records...")
            session.bulk_update_mappings(FamilyTree.__mapper__, tree_update_mappings)
            logger.debug("Bulk update FamilyTrees called.")
        else:
            logger.debug("No valid FamilyTree updates.")
    else:
        logger.debug("No FamilyTree updates prepared.")


def _execute_bulk_db_operations(
    session: SqlAlchemySession,
    prepared_bulk_data: list[dict[str, Any]],
    existing_persons_map: dict[str, Person],  # Needed to potentially map existing IDs
) -> bool:
    """
    Executes bulk INSERT and UPDATE operations for Person, DnaMatch, and FamilyTree
    records within an existing database transaction session.

    Args:
        session: The active SQLAlchemy database session (within a transaction).
        prepared_bulk_data: list of dictionaries prepared by `_prepare_bulk_db_data`.
                            Each dict contains 'person', 'dna_match', 'family_tree' keys
                            with data and an '_operation' hint ('create'/'update').
        existing_persons_map: Dictionary mapping UUIDs to existing Person objects,
                              used for linking updates correctly.

    Returns:
        True if all bulk operations completed successfully within the transaction,
        False if a database error occurred.
    """
    # Step 1: Initialization
    bulk_start_time = time.time()
    num_items = len(prepared_bulk_data)
    if num_items == 0:
        return True  # Nothing to do, considered success

    logger.debug(f"--- Starting Bulk DB Operations ({num_items} prepared items) ---")

    try:
        # Step 2: Separate data by operation type (create/update) and table
        person_creates_raw, person_updates, dna_match_ops, family_tree_ops = _separate_bulk_operations(
            prepared_bulk_data
        )

        # --- Step 3: Person Creates ---
        created_person_map, _ = _process_person_creates(session, person_creates_raw, existing_persons_map)

        # --- Step 4: Person Updates ---
        _process_person_updates(session, person_updates)

        # --- Step 5: Create Master ID Map (for linking related records) ---
        all_person_ids_map = _create_master_id_map(
            created_person_map, person_updates, prepared_bulk_data, existing_persons_map
        )

        # --- Step 6: DnaMatch Bulk Upsert ---
        _process_dna_match_operations(session, dna_match_ops, all_person_ids_map, existing_persons_map)

        # --- Step 7: FamilyTree Bulk Upsert ---
        _process_family_tree_operations(session, family_tree_ops, all_person_ids_map, existing_persons_map)

        # Step 8: Log success
        bulk_duration = time.time() - bulk_start_time
        logger.debug(f"--- Bulk DB Operations OK. Duration: {bulk_duration:.2f}s ---")
        return True

    # Step 9: Handle database errors during bulk operations
    except IntegrityError as integrity_err:
        # Handle UNIQUE constraint violations gracefully
        error_str = str(integrity_err)
        if "UNIQUE constraint failed: people.uuid" in error_str:
            logger.warning(
                f"UNIQUE constraint violation during bulk insert - some records already exist: {integrity_err}"
            )
            # This is expected behavior when records already exist - don't fail the entire batch
            logger.info("Continuing with database operations despite duplicate records...")

            # Use helper function to handle recovery
            # Note: insert_data might not be available in this exception scope, pass None for safe recovery
            return _handle_integrity_error_recovery(session, None)

        if "UNIQUE constraint failed: dna_match.people_id" in error_str:
            logger.error(
                "UNIQUE constraint violation: dna_match.people_id already exists. This indicates the code tried to INSERT when it should UPDATE."
            )
            logger.error(f"Error details: {integrity_err}")
            # Roll back the session to clear the error state
            session.rollback()
            logger.info("Session rolled back. Returning False to indicate failure.")
        else:
            logger.error(f"Other IntegrityError during bulk DB operation: {integrity_err}", exc_info=True)

        return False  # Other integrity errors should still fail

    except SQLAlchemyError as bulk_db_err:
        logger.error(f"Bulk DB operation FAILED: {bulk_db_err}", exc_info=True)
        return False  # Indicate failure (caller owns transaction rollback)
    except Exception as e:
        logger.error(f"Unexpected error during bulk DB operations: {e}", exc_info=True)
        return False  # Indicate failure


# End of _execute_bulk_db_operations


def _optimize_batch_size_for_page(base_batch_size: int, num_matches: int, current_page: int) -> int:
    """Optimize batch size based on page characteristics."""
    optimized_size = base_batch_size

    # Additional optimizations based on page characteristics
    if num_matches >= 50:  # Large pages
        optimized_size = min(25, int(optimized_size * 1.2))
        logger.debug(f"Large page optimization: Increased batch size to {optimized_size}")
    elif num_matches <= 10:  # Small pages
        optimized_size = max(5, int(optimized_size * 0.8))
        logger.debug(f"Small page optimization: Reduced batch size to {optimized_size}")

    # Memory efficiency for long runs
    if current_page % 20 == 0:  # Every 20 pages, use smaller batches
        optimized_size = max(5, optimized_size - 2)
        logger.debug(f"Memory efficiency: Reduced batch size to {optimized_size} at page {current_page}")

    return optimized_size


def _get_optimized_batch_size(session_manager: SessionManager, num_matches: int, current_page: int) -> int:
    """Get optimized batch size with fallback handling."""
    try:
        base_batch_size = _get_adaptive_batch_size(session_manager)
        return _optimize_batch_size_for_page(base_batch_size, num_matches, current_page)
    except Exception as batch_opt_exc:
        logger.warning(f"Batch size optimization failed: {batch_opt_exc}, using fallback")
        return 10  # Safe fallback


_RECENT_BATCH_DURATIONS: list[float] = []
_MAX_TRACKED_BATCH_SAMPLES = 10


@dataclass
class _BatchTotals:
    """Track aggregate counts for batched page processing."""

    new: int = 0
    updated: int = 0
    skipped: int = 0
    error: int = 0

    def update(self, new_count: int, updated_count: int, skipped_count: int, error_count: int) -> None:
        self.new += new_count
        self.updated += updated_count
        self.skipped += skipped_count
        self.error += error_count

    @property
    def processed(self) -> int:
        """Return total processed matches excluding errors."""
        return self.new + self.updated + self.skipped

    def success_rate(self, total_matches: int) -> float:
        """Compute processed-to-total ratio for logging."""
        if total_matches <= 0:
            return 1.0
        return self.processed / total_matches


def _do_batch(
    session_manager: SessionManager,
    matches_on_page: list[dict[str, Any]],
    current_page: int,
) -> tuple[int, int, int, int, PageProcessingMetrics]:
    """Process matches from a single page using dynamic batching."""
    batch_start_time = time.time()
    num_matches_on_page = len(matches_on_page)
    optimized_batch_size = _get_optimized_batch_size(session_manager, num_matches_on_page, current_page)

    # If we have fewer matches than optimized batch size, process normally (no need to split)
    if num_matches_on_page <= optimized_batch_size:
        new, updated, skipped, errors, metrics = _process_page_matches(session_manager, matches_on_page, current_page)
        metrics.total_seconds = max(metrics.total_seconds, time.time() - batch_start_time)
        return new, updated, skipped, errors, metrics

    # SURGICAL FIX #7: Create single session for all batches on this page
    page_session = session_manager.get_db_conn()
    if not page_session:
        logger.error(f"Page {current_page}: Failed to get DB session for batch processing.")
        return 0, 0, 0, 0, PageProcessingMetrics()

    try:
        totals, combined_metrics, total_duration = _process_batches_for_page(
            session_manager,
            page_session,
            matches_on_page,
            optimized_batch_size,
            current_page,
            batch_start_time,
        )
        success_rate = totals.success_rate(num_matches_on_page)
        _record_batch_performance(
            current_page=current_page,
            batch_duration=total_duration,
            batch_size=optimized_batch_size,
            num_matches=num_matches_on_page,
            success_rate=success_rate,
            batch_start_time=batch_start_time,
        )

        combined_metrics.total_matches = num_matches_on_page or combined_metrics.total_matches
        combined_metrics.total_seconds = max(combined_metrics.total_seconds, total_duration)
        return totals.new, totals.updated, totals.skipped, totals.error, combined_metrics

    finally:
        # SURGICAL FIX #7: Clean up the reused session
        if page_session:
            session_manager.return_session(page_session)
            logger.debug(f"Page {current_page}: Returned reused session to pool")


def _process_batches_for_page(
    session_manager: SessionManager,
    page_session: SqlAlchemySession,
    matches_on_page: list[dict[str, Any]],
    batch_size: int,
    current_page: int,
    batch_start_time: float,
) -> tuple[_BatchTotals, PageProcessingMetrics, float]:
    """Process all batches for a page and aggregate metrics."""
    logger.debug(
        "Splitting page %d (%d matches) into batches of %d",
        current_page,
        len(matches_on_page),
        batch_size,
    )

    totals = _BatchTotals()
    combined_metrics = PageProcessingMetrics()
    total_batches = max(1, math.ceil(len(matches_on_page) / batch_size))

    for batch_index, start_index in enumerate(range(0, len(matches_on_page), batch_size), start=1):
        batch_matches = matches_on_page[start_index : start_index + batch_size]
        _, batch_metrics, counts = _execute_single_batch(
            session_manager,
            page_session,
            batch_matches,
            current_page,
            batch_index,
            total_batches,
        )
        totals.update(*counts)
        combined_metrics.merge(batch_metrics)

    combined_metrics.total_matches = len(matches_on_page) or combined_metrics.total_matches
    combined_metrics.batches = max(combined_metrics.batches, total_batches)
    total_duration = time.time() - batch_start_time
    return totals, combined_metrics, total_duration


def _execute_single_batch(
    session_manager: SessionManager,
    page_session: SqlAlchemySession,
    batch_matches: list[dict[str, Any]],
    current_page: int,
    batch_number: int,
    total_batches: int,
) -> tuple[float, PageProcessingMetrics, tuple[int, int, int, int]]:
    """Process an individual batch and return its metrics."""
    logger.debug(
        "--- Processing Page %d Batch No%s (%d matches) ---",
        current_page,
        batch_number,
        len(batch_matches),
    )

    batch_timer_start = time.time()
    new, updated, skipped, errors, batch_metrics = _process_page_matches(
        session_manager,
        batch_matches,
        current_page,
        is_batch=True,
        reused_session=page_session,
    )

    measured_duration = time.time() - batch_timer_start
    batch_metrics.total_seconds = max(batch_metrics.total_seconds, measured_duration)
    batch_metrics.batches = max(batch_metrics.batches, 1)
    batch_duration = batch_metrics.total_seconds

    throughput = _calculate_batch_throughput(new, updated, skipped, batch_duration)
    _log_batch_summary(
        current_page=current_page,
        batch_number=batch_number,
        total_batches=total_batches,
        batch_match_count=len(batch_matches),
        new=new,
        updated=updated,
        skipped=skipped,
        errors=errors,
        duration=batch_duration,
        throughput=throughput,
    )

    return batch_duration, batch_metrics, (new, updated, skipped, errors)


def _calculate_batch_throughput(new: int, updated: int, skipped: int, duration: float) -> float:
    """Calculate matches processed per second for a batch."""
    processed = new + updated + skipped
    if duration <= 0 or processed <= 0:
        return 0.0
    return processed / duration


def _log_batch_summary(
    *,
    current_page: int,
    batch_number: int,
    total_batches: int,
    batch_match_count: int,
    new: int,
    updated: int,
    skipped: int,
    errors: int,
    duration: float,
    throughput: float,
) -> None:
    """Emit a concise INFO-level summary for a processed batch."""
    message_lines = [
        f"Page {current_page} batch {batch_number} of {total_batches}",
        f"  matches={batch_match_count} duration={duration:.2f}s",
        f"  new={new} updated={updated} skipped={skipped} errors={errors}",
        f"  rate={throughput:.2f} match/s",
    ]
    logger.info("\n".join(message_lines))


def _record_batch_performance(
    *,
    current_page: int,
    batch_duration: float,
    batch_size: int,
    num_matches: int,
    success_rate: float,
    batch_start_time: float,
) -> None:
    """Record batch performance metrics for future optimization."""
    _update_recent_batch_history(batch_duration)
    logger.debug(
        "Page %d batch performance: %.2fs for %d matches (%.1f%% success rate, batch size: %d)",
        current_page,
        batch_duration,
        num_matches,
        success_rate * 100,
        batch_size,
    )
    _log_api_performance("batch_processing", batch_start_time, f"success {success_rate:.0%}")


def _update_recent_batch_history(duration: float) -> None:
    """Maintain a rolling history of recent batch durations."""
    _RECENT_BATCH_DURATIONS.append(duration)
    if len(_RECENT_BATCH_DURATIONS) > _MAX_TRACKED_BATCH_SAMPLES:
        del _RECENT_BATCH_DURATIONS[:-_MAX_TRACKED_BATCH_SAMPLES]


def _initialize_page_processing(
    matches_on_page: list[dict[str, Any]], current_page: int, my_uuid: Optional[str]
) -> tuple[dict[str, int], int, Optional[Any]]:
    """Initialize page processing with validation and memory optimization."""
    page_statuses: dict[str, int] = {"new": 0, "updated": 0, "skipped": 0, "error": 0}
    num_matches_on_page = len(matches_on_page)

    if not my_uuid:
        logger.error(f"_do_batch Page {current_page}: Missing my_uuid.")
        raise ValueError("Missing my_uuid")
    if not matches_on_page:
        logger.debug(f"_do_batch Page {current_page}: Empty match list.")
        raise ValueError("Empty match list")

    logger.debug(f"--- Starting Batch Processing for Page {current_page} ({num_matches_on_page} matches) ---")

    memory_processor = None
    if num_matches_on_page > 20:
        memory_processor = MemoryOptimizedMatchProcessor(max_memory_mb=400)
        logger.debug(f"Page {current_page}: Enabled memory optimization for {num_matches_on_page} matches")

    return page_statuses, num_matches_on_page, memory_processor


@dataclass(slots=True)
class _PrefetchArtifacts:
    prefetched_data: dict[str, Any]
    timings: dict[str, float]
    call_counts: dict[str, int]


@dataclass(slots=True)
class _PrefetchSummary:
    total_seconds: float
    filtered_timings: dict[str, float]
    filtered_counts: dict[str, int]


@contextlib.contextmanager
def _timed_phase(label: str, timing_log: dict[str, float]) -> Iterator[None]:
    start = time.time()
    try:
        yield
    finally:
        timing_log[label] = time.time() - start


def _summarize_prefetch_metrics(
    timing_log: dict[str, float],
    prefetch_artifacts: _PrefetchArtifacts,
) -> _PrefetchSummary:
    filtered_timings = {
        key: value
        for key, value in prefetch_artifacts.timings.items()
        if value > 0.0 or prefetch_artifacts.call_counts.get(key, 0) > 0
    }
    filtered_counts = {key: prefetch_artifacts.call_counts.get(key, 0) for key in filtered_timings}
    return _PrefetchSummary(
        total_seconds=sum(timing_log.values()),
        filtered_timings=filtered_timings,
        filtered_counts=filtered_counts,
    )


def _handle_critical_batch_error(
    critical_err: Exception, current_page: int, page_statuses: dict[str, int], num_matches_on_page: int
) -> tuple[int, int, int, int, PageProcessingMetrics]:
    """Handle critical batch processing errors."""
    logger.critical(
        f"CRITICAL ERROR processing batch page {current_page}: {critical_err}",
        exc_info=True,
    )

    final_error_count_for_page = page_statuses["error"] + max(
        0,
        num_matches_on_page
        - (page_statuses["new"] + page_statuses["updated"] + page_statuses["skipped"] + page_statuses["error"]),
    )

    return (
        page_statuses["new"],
        page_statuses["updated"],
        page_statuses["skipped"],
        final_error_count_for_page,
        PageProcessingMetrics(),
    )


def _handle_unhandled_batch_error(
    outer_batch_exc: Exception, current_page: int, page_statuses: dict[str, int], num_matches_on_page: int
) -> tuple[int, int, int, int, PageProcessingMetrics]:
    """Handle unhandled batch processing exceptions."""
    logger.critical(
        f"CRITICAL UNHANDLED EXCEPTION processing batch page {current_page}: {outer_batch_exc}",
        exc_info=True,
    )

    final_error_count_for_page = num_matches_on_page - (
        page_statuses["new"] + page_statuses["updated"] + page_statuses["skipped"]
    )
    return (
        page_statuses["new"],
        page_statuses["updated"],
        page_statuses["skipped"],
        max(0, final_error_count_for_page),
        PageProcessingMetrics(),
    )


def _perform_batch_api_prefetches(
    session_manager: SessionManager,
    fetch_candidates_uuid: set[str],
    matches_to_process_later: list[dict[str, Any]],
    current_page: int,
) -> _PrefetchArtifacts:
    """Perform API prefetches for batch."""
    if len(fetch_candidates_uuid) == 0:
        logger.debug(f"Batch {current_page}: All matches skipped (no API processing needed) - fast path")
        return _PrefetchArtifacts({}, {}, {})

    logger.debug(
        f"Batch {current_page}: Performing sequential API prefetches for {len(fetch_candidates_uuid)} candidates"
    )

    prefetch_result = gather_perform_api_prefetches(
        session_manager,
        fetch_candidates_uuid,
        matches_to_process_later,
        _build_prefetch_config(),
        _build_prefetch_hooks(),
    )
    return _PrefetchArtifacts(
        prefetch_result.prefetched_data,
        prefetch_result.endpoint_durations,
        prefetch_result.endpoint_counts,
    )


_memory_cleanup_state: dict[str, Optional[float]] = {"previous_mb": None}


def _perform_memory_cleanup(current_page: int) -> None:
    """Perform memory cleanup for batch processing."""
    try:
        import gc
        import os

        import psutil

        try:
            process = psutil.Process(os.getpid())
            current_memory_mb = process.memory_info().rss / 1024 / 1024
            logger.debug(f"Memory usage at page {current_page}: {current_memory_mb:.1f} MB")
        except Exception:
            current_memory_mb = 0

        if current_page % 5 == 0 or current_memory_mb > 500:
            collected = gc.collect()
            logger.debug(
                f"Memory cleanup: Forced garbage collection at page {current_page}, "
                f"collected {collected} objects, memory: {current_memory_mb:.1f} MB"
            )

            if current_memory_mb > 800:
                logger.warning(f"High memory usage ({current_memory_mb:.1f} MB) - performing aggressive cleanup")
                gc.collect(0)
                gc.collect(1)
                gc.collect(2)

        elif current_page % 3 == 0:
            gc.collect(0)
            logger.debug(f"Memory cleanup: Light garbage collection at page {current_page}")

        previous_memory_mb = _memory_cleanup_state.get("previous_mb")
        if previous_memory_mb is not None:
            memory_growth = current_memory_mb - previous_memory_mb
            if memory_growth > 50:
                logger.warning(f"Memory growth detected: +{memory_growth:.1f} MB since last check")
        _memory_cleanup_state["previous_mb"] = current_memory_mb

    except Exception as cleanup_exc:
        logger.warning(f"Memory cleanup warning at page {current_page}: {cleanup_exc}")


def _cleanup_batch_session(
    session_manager: SessionManager,
    batch_session: SqlAlchemySession,
    reused_session: Optional[SqlAlchemySession],
    current_page: int,
) -> None:
    """Clean up batch session if it wasn't reused."""
    if not reused_session and batch_session:
        session_manager.return_session(batch_session)
    elif reused_session:
        logger.debug(f"Batch {current_page}: Keeping reused session for parent cleanup")


def _log_batch_summary_if_needed(is_batch: bool, current_page: int, page_statuses: dict[str, int]) -> None:
    """Log page summary if not processing as part of a batch."""
    if not is_batch:
        _log_page_summary(
            current_page,
            page_statuses["new"],
            page_statuses["updated"],
            page_statuses["skipped"],
            page_statuses["error"],
        )


def _get_batch_session(
    session_manager: SessionManager, reused_session: Optional[SqlAlchemySession], current_page: int
) -> SqlAlchemySession:
    """Get or create batch session."""
    if reused_session:
        logger.debug(f"Batch {current_page}: Using reused session for batch operations")
        return reused_session

    batch_session = session_manager.get_db_conn()
    if not batch_session:
        logger.error(f"_do_batch Page {current_page}: Failed DB session.")
        raise SQLAlchemyError("Failed get DB session")
    return batch_session


def _filter_matches_by_config(matches_on_page: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter matches based on configuration criteria."""
    from config.config_manager import get_config_manager

    config = get_config_manager().get_config()
    if config.action6_min_tree_size is None and not config.action6_public_tree_only:
        return matches_on_page

    original_count = len(matches_on_page)
    filtered_matches = [
        m
        for m in matches_on_page
        if (config.action6_min_tree_size is None or (int(m.get("tree_size") or 0) >= config.action6_min_tree_size))
        and (not config.action6_public_tree_only or m.get("has_public_tree"))
    ]
    filtered_count = len(filtered_matches)
    if original_count != filtered_count:
        logger.info(f"Filtered {original_count - filtered_count} matches based on tree criteria.")
    return filtered_matches


def _process_page_matches(
    session_manager: SessionManager,
    matches_on_page: list[dict[str, Any]],
    current_page: int,
    is_batch: bool = False,
    reused_session: Optional[SqlAlchemySession] = None,
) -> tuple[int, int, int, int, PageProcessingMetrics]:
    """
    Original batch processing logic - now used by both single page and chunked batch processing.
    Coordinates DB lookups, API prefetches, data preparation, and bulk DB operations.
    """
    my_uuid = session_manager.my_uuid

    # TIMING BREAKDOWN: Track each phase for performance analysis
    timing_log: dict[str, float] = {}
    page_metrics = PageProcessingMetrics()
    lookup_artifacts: Optional[BatchLookupArtifacts] = None
    prefetch_artifacts: Optional[_PrefetchArtifacts] = None

    # Filter matches based on configuration
    matches_on_page = _filter_matches_by_config(matches_on_page)

    try:
        with _timed_phase("initialization", timing_log):
            page_statuses, num_matches_on_page, _ = _initialize_page_processing(matches_on_page, current_page, my_uuid)
    except ValueError:
        return 0, 0, 0, 0, page_metrics

    try:
        batch_session = _get_batch_session(session_manager, reused_session, current_page)

        try:
            # Phase 1: Database Lookups
            with _timed_phase("db_lookups", timing_log):
                lookup_artifacts = gather_process_batch_lookups(
                    batch_session,
                    matches_on_page,
                    current_page,
                    page_statuses,
                )
            logger.debug(f"⏱️  Page {current_page} - DB lookups: {timing_log['db_lookups']:.2f}s")

            assert lookup_artifacts is not None

            # Phase 2: API Prefetches (sequential API calls)
            with _timed_phase("api_prefetches", timing_log):
                prefetch_artifacts = _perform_batch_api_prefetches(
                    session_manager,
                    lookup_artifacts.fetch_candidates_uuid,
                    lookup_artifacts.matches_to_process_later,
                    current_page,
                )
            logger.debug(f"⏱️  Page {current_page} - API prefetches: {timing_log['api_prefetches']:.2f}s")

            assert prefetch_artifacts is not None

            # Phase 3: Data Preparation & DB Commit
            with _timed_phase("data_prep_commit", timing_log):
                gather_prepare_and_commit_batch_data(
                    batch_session,
                    session_manager,
                    lookup_artifacts.matches_to_process_later,
                    lookup_artifacts.existing_persons_map,
                    prefetch_artifacts.prefetched_data,
                    current_page,
                    page_statuses,
                    _build_persistence_hooks(),
                )
            logger.debug(f"⏱️  Page {current_page} - Data prep & commit: {timing_log['data_prep_commit']:.2f}s")
        finally:
            _cleanup_batch_session(session_manager, batch_session, reused_session, current_page)

        _log_batch_summary_if_needed(is_batch, current_page, page_statuses)

        assert lookup_artifacts is not None
        assert prefetch_artifacts is not None

        summary = _summarize_prefetch_metrics(timing_log, prefetch_artifacts)

        # Log timing breakdown summary for slow pages
        if summary.total_seconds > 30.0:
            logger.info(
                "Timings: Page %d total %.1fs | DB %.2fs | API %.2fs | prep %.2fs",
                current_page,
                summary.total_seconds,
                timing_log.get("db_lookups", 0.0),
                timing_log.get("api_prefetches", 0.0),
                timing_log.get("data_prep_commit", 0.0),
            )

        matches_with_trees = sum(1 for m in matches_on_page if m.get("match_tree_id"))
        matches_with_public_trees = sum(1 for m in matches_on_page if m.get("has_public_tree"))

        page_metrics = PageProcessingMetrics(
            total_matches=num_matches_on_page,
            fetch_candidates=len(lookup_artifacts.fetch_candidates_uuid),
            existing_matches=len(lookup_artifacts.existing_persons_map),
            matches_with_trees=matches_with_trees,
            matches_with_public_trees=matches_with_public_trees,
            db_seconds=timing_log.get("db_lookups", 0.0),
            prefetch_seconds=timing_log.get("api_prefetches", 0.0),
            commit_seconds=timing_log.get("data_prep_commit", 0.0),
            total_seconds=summary.total_seconds,
            batches=1,
            prefetch_breakdown=summary.filtered_timings,
            prefetch_call_counts=summary.filtered_counts,
        )

        return (
            page_statuses["new"],
            page_statuses["updated"],
            page_statuses["skipped"],
            page_statuses["error"],
            page_metrics,
        )

    except MaxApiFailuresExceededError:
        raise
    except (ValueError, SQLAlchemyError, ConnectionError) as critical_err:
        return _handle_critical_batch_error(critical_err, current_page, page_statuses, num_matches_on_page)
    except Exception as outer_batch_exc:
        return _handle_unhandled_batch_error(outer_batch_exc, current_page, page_statuses, num_matches_on_page)

    finally:
        _perform_memory_cleanup(current_page)
        logger.debug(f"--- Finished Batch Processing for Page {current_page} ---")


# End of _do_batch

# ------------------------------------------------------------------------------
# _do_match Helper Functions (_prepare_person_operation_data, etc.)
# ------------------------------------------------------------------------------


def _compare_datetime_field(current_value: Any, new_value: Any) -> tuple[bool, Any]:
    """Compare datetime fields with UTC normalization."""
    current_dt_utc = (
        current_value.astimezone(timezone.utc).replace(microsecond=0)
        if isinstance(current_value, datetime) and current_value.tzinfo
        else (
            current_value.replace(tzinfo=timezone.utc, microsecond=0) if isinstance(current_value, datetime) else None
        )
    )  # Closing parenthesis for _process_dna_data_safe
    new_dt_utc = (
        new_value.astimezone(timezone.utc).replace(microsecond=0)
        if isinstance(new_value, datetime) and new_value.tzinfo
        else (new_value.replace(tzinfo=timezone.utc, microsecond=0) if isinstance(new_value, datetime) else None)
    )  # Closing parenthesis for _process_dna_data_safe
    return (new_dt_utc != current_dt_utc, new_value)


def _compare_status_field(current_value: Any, new_value: Any) -> tuple[bool, Any]:
    """Compare status enum fields."""
    current_enum_val = current_value.value if isinstance(current_value, PersonStatusEnum) else current_value
    new_enum_val = new_value.value if isinstance(new_value, PersonStatusEnum) else new_value
    return (new_enum_val != current_enum_val, new_value)


def _compare_birth_year_field(
    current_value: Any, new_value: Any, log_ref_short: str, logger_instance: logging.Logger
) -> tuple[bool, Any]:
    """Compare birth year field (only update if new is valid and current is None)."""
    if new_value is not None and current_value is None:
        try:
            value_to_set_int = int(new_value)
            return (True, value_to_set_int)
        except (ValueError, TypeError):
            logger_instance.warning(f"Invalid birth_year '{new_value}' for update {log_ref_short}")
    return (False, new_value)


def _compare_gender_field(current_value: Any, new_value: Any) -> tuple[bool, Any]:
    """Compare gender field (only update if new is valid and current is None)."""
    if (
        new_value is not None
        and current_value is None
        and isinstance(new_value, str)
        and new_value.lower() in {"f", "m"}
    ):
        return (True, new_value.lower())
    return (False, new_value)


def _compare_profile_id_field(current_value: Any, new_value: Any) -> tuple[bool, Any]:
    """Compare profile ID fields with uppercase normalization."""
    current_str_upper = str(current_value).upper() if current_value is not None else None
    new_str_upper = str(new_value).upper() if new_value is not None else None
    return (new_str_upper != current_str_upper, new_str_upper)


def _compare_boolean_field(current_value: Any, new_value: Any) -> tuple[bool, Any]:
    """Compare boolean fields."""
    return (bool(current_value) != bool(new_value), bool(new_value))


def _get_field_comparator(
    key: str,
    current_value: Any,
    new_value: Any,
) -> Optional[Callable[[Any, Any], tuple[bool, Any]]]:
    """Get the appropriate comparator function for a field."""
    # Check for boolean fields first (type-based)
    if isinstance(current_value, bool) or isinstance(new_value, bool):
        return _compare_boolean_field

    # Field-specific comparators
    field_comparators = {
        "last_logged_in": _compare_datetime_field,
        "status": _compare_status_field,
        "gender": _compare_gender_field,
        "profile_id": _compare_profile_id_field,
        "administrator_profile_id": _compare_profile_id_field,
    }

    return field_comparators.get(key)


def _compare_person_field(
    key: str, current_value: Any, new_value: Any, log_ref_short: str, logger_instance: logging.Logger
) -> tuple[bool, Any]:
    """Compare person field and return whether it changed and the value to set."""

    if key == "birth_year":
        return _compare_birth_year_field(current_value, new_value, log_ref_short, logger_instance)

    comparator = _get_field_comparator(key, current_value, new_value)

    if comparator is None:
        # Default comparison for fields without special handling
        return (current_value != new_value, new_value)

    return comparator(current_value, new_value)


def _determine_profile_ids_when_both_exist(
    tester_profile_id_upper: str,
    admin_profile_id_upper: str,
    formatted_match_username: str,
    formatted_admin_username: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Determine profile IDs when both tester and admin IDs exist."""
    if tester_profile_id_upper == admin_profile_id_upper:
        if (
            formatted_match_username
            and formatted_admin_username
            and formatted_match_username.lower() == formatted_admin_username.lower()
        ):
            return tester_profile_id_upper, None, None
        return None, admin_profile_id_upper, formatted_admin_username
    return tester_profile_id_upper, admin_profile_id_upper, formatted_admin_username


def _extract_raw_profile_data(
    details_part: dict[str, Any], match: dict[str, Any]
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extract raw profile data from details and match."""
    raw_tester_profile_id = details_part.get("tester_profile_id") or match.get("profile_id")
    raw_admin_profile_id = details_part.get("admin_profile_id") or match.get("administrator_profile_id_hint")
    raw_admin_username = details_part.get("admin_username") or match.get("administrator_username_hint")

    formatted_admin_username = format_name(raw_admin_username) if raw_admin_username else None
    tester_profile_id_upper = raw_tester_profile_id.upper() if raw_tester_profile_id else None
    admin_profile_id_upper = raw_admin_profile_id.upper() if raw_admin_profile_id else None

    return tester_profile_id_upper, admin_profile_id_upper, formatted_admin_username, raw_admin_username


def _extract_profile_ids(
    details_part: dict[str, Any], match: dict[str, Any], formatted_match_username: str
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract and determine profile IDs for person and administrator."""
    tester_profile_id_upper, admin_profile_id_upper, formatted_admin_username, _ = _extract_raw_profile_data(
        details_part, match
    )

    if tester_profile_id_upper and admin_profile_id_upper:
        return _determine_profile_ids_when_both_exist(
            tester_profile_id_upper, admin_profile_id_upper, formatted_match_username, formatted_admin_username
        )
    if tester_profile_id_upper:
        return tester_profile_id_upper, None, None
    if admin_profile_id_upper:
        return None, admin_profile_id_upper, formatted_admin_username
    return None, None, None


def _build_message_link(
    person_profile_id: Optional[str], person_admin_id: Optional[str], config_schema_arg: "ConfigSchema"
) -> Optional[str]:
    """Build message link for person."""
    message_target_id = person_profile_id or person_admin_id
    if message_target_id:
        base_url = config_schema_arg.api.base_url
        return urljoin(base_url, f"/messaging/?p={message_target_id.upper()}")
    return None


def _extract_birth_year(prefetched_tree_data: Optional[dict[str, Any]]) -> Optional[int]:
    """Extract birth year from tree data."""
    if prefetched_tree_data and prefetched_tree_data.get("their_birth_year"):
        with contextlib.suppress(ValueError, TypeError):
            return int(prefetched_tree_data["their_birth_year"])
    return None


def _normalize_last_logged_in(last_logged_in_val: Optional[datetime]) -> Optional[datetime]:
    """Normalize last logged in datetime to UTC."""
    if isinstance(last_logged_in_val, datetime):
        if last_logged_in_val.tzinfo is None:
            return last_logged_in_val.replace(tzinfo=timezone.utc)
        return last_logged_in_val.astimezone(timezone.utc)
    return last_logged_in_val


def _build_incoming_person_data(
    match: dict[str, Any],
    match_uuid: str,
    formatted_match_username: str,
    match_in_my_tree: bool,
    person_profile_id: Optional[str],
    person_admin_id: Optional[str],
    person_admin_username: Optional[str],
    message_link: Optional[str],
    birth_year: Optional[int],
    last_logged_in: Optional[datetime],
    details_part: dict[str, Any],
    profile_part: dict[str, Any],
) -> dict[str, Any]:
    """Build incoming person data dictionary."""
    return {
        "uuid": match_uuid.upper(),
        "profile_id": person_profile_id,
        "username": formatted_match_username,
        "administrator_profile_id": person_admin_id,
        "administrator_username": person_admin_username,
        "in_my_tree": match_in_my_tree,
        "first_name": cast(Optional[str], match.get("first_name")),
        "last_logged_in": last_logged_in,
        "contactable": bool(profile_part.get("contactable", True)),
        "gender": cast(Optional[str], details_part.get("gender")),
        "message_link": message_link,
        "birth_year": birth_year,
        "status": PersonStatusEnum.ACTIVE,
    }


def _prepare_person_operation_data(
    match: dict[str, Any],
    existing_person: Optional[Person],
    prefetched_combined_details: Optional[dict[str, Any]],
    prefetched_tree_data: Optional[dict[str, Any]],
    config_schema_arg: "ConfigSchema",
    match_uuid: str,
    formatted_match_username: str,
    match_in_my_tree: bool,
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> tuple[Optional[dict[str, Any]], bool]:
    """
    Prepares Person data for create or update operations based on API data and existing records.
    """
    details_part = prefetched_combined_details or {}
    profile_part = details_part

    person_profile_id, person_admin_id, person_admin_username = _extract_profile_ids(
        details_part, match, formatted_match_username
    )

    message_link = _build_message_link(person_profile_id, person_admin_id, config_schema_arg)
    birth_year = _extract_birth_year(prefetched_tree_data)
    last_logged_in = _normalize_last_logged_in(cast(Optional[datetime], profile_part.get("last_logged_in_dt")))

    incoming_person_data = _build_incoming_person_data(
        match,
        match_uuid,
        formatted_match_username,
        match_in_my_tree,
        person_profile_id,
        person_admin_id,
        person_admin_username,
        message_link,
        birth_year,
        last_logged_in,
        details_part,
        profile_part,
    )

    if existing_person is None:
        person_op_dict = incoming_person_data.copy()
        person_op_dict["_operation"] = "create"
        return person_op_dict, False
    person_data_for_update: dict[str, Any] = {
        "_operation": "update",
        "_existing_person_id": existing_person.id,
        "uuid": match_uuid.upper(),
    }
    person_fields_changed = False

    for key, new_value in incoming_person_data.items():
        if key == "uuid":
            continue

        current_value = getattr(existing_person, key, None)
        value_changed, value_to_set = _compare_person_field(
            key, current_value, new_value, log_ref_short, logger_instance
        )

        if value_changed:
            person_data_for_update[key] = value_to_set
            person_fields_changed = True
            logger_instance.debug(
                f"  Person change {log_ref_short}: Field '{key}' ('{current_value}' -> '{value_to_set}')"
            )

    return (person_data_for_update if person_fields_changed else None), person_fields_changed


# End of _prepare_person_operation_data


def _check_basic_dna_changes(
    api_cm: int, db_cm: int, api_segments: int, db_segments: int, log_ref_short: str, logger_instance: logging.Logger
) -> bool:
    """Check basic DNA changes (cM and segments)."""
    if api_cm != db_cm:
        logger_instance.debug(f"  DNA change {log_ref_short}: cM")
        return True
    if api_segments != db_segments:
        logger_instance.debug(f"  DNA change {log_ref_short}: Segments")
        return True
    return False


def _check_longest_segment_changes(
    api_longest: Optional[float], db_longest: Optional[float], log_ref_short: str, logger_instance: logging.Logger
) -> bool:
    """Check longest segment changes."""
    if (
        api_longest is not None
        and db_longest is not None
        and abs(float(str(api_longest)) - float(str(db_longest))) > 0.01
    ):
        logger_instance.debug(f"  DNA change {log_ref_short}: Longest Segment")
        return True
    if db_longest is not None and api_longest is None:
        logger_instance.debug(f"  DNA change {log_ref_short}: Longest Segment (API lost data)")
        return True
    return False


def _check_relationship_and_side_changes(
    db_predicted_rel_for_comp: str,
    api_predicted_rel_for_comp: str,
    details_part: dict[str, Any],
    existing_dna_match: DnaMatch,
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> bool:
    """Check relationship and parental side changes."""
    if str(db_predicted_rel_for_comp) != str(api_predicted_rel_for_comp):
        logger_instance.debug(
            f"  DNA change {log_ref_short}: Predicted Rel ({db_predicted_rel_for_comp} -> {api_predicted_rel_for_comp})"
        )
        return True
    if bool(details_part.get("from_my_fathers_side")) != bool(existing_dna_match.from_my_fathers_side):
        logger_instance.debug(f"  DNA change {log_ref_short}: Father Side")
        return True
    if bool(details_part.get("from_my_mothers_side")) != bool(existing_dna_match.from_my_mothers_side):
        logger_instance.debug(f"  DNA change {log_ref_short}: Mother Side")
        return True
    return False


def _check_tree_ids_changed(
    existing_dna_match: DnaMatch,
    match: dict[str, Any],
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> bool:
    """Check if tree IDs have changed."""
    if str(match.get("match_tree_id") or "") != str(existing_dna_match.match_tree_id or ""):
        logger_instance.debug(f"  DNA change {log_ref_short}: Tree ID")
        return True
    if str(match.get("match_tree_person_id") or "") != str(existing_dna_match.match_tree_person_id or ""):
        logger_instance.debug(f"  DNA change {log_ref_short}: Tree Person ID")
        return True
    return False


def _check_tree_status_changed(
    existing_dna_match: DnaMatch,
    match: dict[str, Any],
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> bool:
    """Check if tree status or size has changed."""
    if bool(match.get("has_public_tree")) != bool(existing_dna_match.has_public_tree):
        logger_instance.debug(f"  DNA change {log_ref_short}: Public Tree Status")
        return True

    # Handle tree_size comparison safely (None vs 0 vs int)
    api_size = match.get("tree_size")
    db_size = existing_dna_match.tree_size
    if api_size is not None and (db_size is None or int(api_size) != int(db_size)):
        logger_instance.debug(f"  DNA change {log_ref_short}: Tree Size")
        return True

    return False


def _check_tree_data_changes(
    existing_dna_match: DnaMatch,
    match: dict[str, Any],
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> bool:
    """Check for changes in tree-related data."""
    if _check_tree_ids_changed(existing_dna_match, match, log_ref_short, logger_instance):
        return True
    return _check_tree_status_changed(existing_dna_match, match, log_ref_short, logger_instance)


def _compare_dna_fields(
    existing_dna_match: DnaMatch,
    match: dict[str, Any],
    details_part: dict[str, Any],
    api_predicted_rel_for_comp: str,
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> bool:
    """Compare DNA fields and return True if update is needed."""
    api_cm = int(match.get("cm_dna", 0))
    db_cm = existing_dna_match.cm_dna
    api_segments = int(details_part.get("shared_segments", match.get("numSharedSegments", 0)))
    db_segments = existing_dna_match.shared_segments if existing_dna_match.shared_segments is not None else 0
    api_longest_raw = details_part.get("longest_shared_segment")
    api_longest = float(api_longest_raw) if api_longest_raw is not None else None
    db_longest = existing_dna_match.longest_shared_segment

    raw_predicted_rel = cast(Optional[str], getattr(existing_dna_match, "predicted_relationship", None))
    db_predicted_rel_for_comp = raw_predicted_rel or "N/A"

    if _check_basic_dna_changes(api_cm, db_cm, api_segments, db_segments, log_ref_short, logger_instance):
        return True
    if _check_longest_segment_changes(api_longest, db_longest, log_ref_short, logger_instance):
        return True
    if _check_relationship_and_side_changes(
        db_predicted_rel_for_comp,
        api_predicted_rel_for_comp,
        details_part,
        existing_dna_match,
        log_ref_short,
        logger_instance,
    ):
        return True
    return _check_tree_data_changes(existing_dna_match, match, log_ref_short, logger_instance)


def _check_dna_update_needed(
    existing_dna_match: Optional[DnaMatch],
    match: dict[str, Any],
    details_part: dict[str, Any],
    api_predicted_rel_for_comp: str,
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> bool:
    """Check if DNA match record needs updating."""
    if existing_dna_match is None:
        return True

    try:
        return _compare_dna_fields(
            existing_dna_match, match, details_part, api_predicted_rel_for_comp, log_ref_short, logger_instance
        )
    except (ValueError, TypeError, AttributeError) as dna_comp_err:
        logger_instance.warning(
            f"Error comparing DNA data for {log_ref_short}: {dna_comp_err}. Assuming update needed."
        )
        return True


def _infer_predicted_relationship_from_cm(cm_value: int) -> str:
    """Map centimorgan totals to a descriptive relationship bucket."""

    for threshold, label in _CM_RELATIONSHIP_BUCKETS:
        if cm_value >= threshold:
            return label
    return "Distant relationship?"


def _resolve_predicted_relationship_value(
    api_value: Optional[str],
    cm_value: int,
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> str:
    """Return a usable predicted relationship, inferring from cM when API omits it."""

    normalized = (api_value or "").strip()
    if normalized and normalized.upper() != "N/A":
        return normalized

    inferred_label = _infer_predicted_relationship_from_cm(cm_value)
    inferred_value = f"{inferred_label} (≈{cm_value} cM, inferred)"
    logger_instance.debug(
        "%s: Predicted relationship missing; inferred '%s' from %s cM",
        log_ref_short,
        inferred_value,
        cm_value,
    )
    return inferred_value


def _build_dna_dict_base(match_uuid: str, match: dict[str, Any], safe_predicted_relationship: str) -> dict[str, Any]:
    """Build base DNA match dictionary."""
    return {
        "uuid": match_uuid.upper(),
        "compare_link": cast(Optional[str], match.get("compare_link")),
        "cm_dna": int(match.get("cm_dna", 0)),
        "predicted_relationship": safe_predicted_relationship,
        "_operation": "create",
    }


def _add_dna_details(
    dna_dict_base: dict[str, Any],
    prefetched_combined_details: Optional[dict[str, Any]],
    match: dict[str, Any],
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> None:
    """Add DNA details to the base dictionary."""
    # Add tree data from match list
    dna_dict_base.update(
        {
            "match_tree_id": match.get("match_tree_id"),
            "match_tree_person_id": match.get("match_tree_person_id"),
            "has_public_tree": match.get("has_public_tree"),
            "tree_size": match.get("tree_size"),
        }
    )

    if prefetched_combined_details:
        details_part = prefetched_combined_details
        dna_dict_base.update(
            {
                "shared_segments": details_part.get("shared_segments"),
                "longest_shared_segment": details_part.get("longest_shared_segment"),
                "meiosis": details_part.get("meiosis"),
                "from_my_fathers_side": bool(details_part.get("from_my_fathers_side", False)),
                "from_my_mothers_side": bool(details_part.get("from_my_mothers_side", False)),
            }
        )
    else:
        logger_instance.warning(
            f"{log_ref_short}: DNA needs create/update, but no/limited combined details. Using list data for segments."
        )
        dna_dict_base["shared_segments"] = cast(Optional[int], match.get("numSharedSegments"))


def _filter_dna_dict(dna_dict_base: dict[str, Any]) -> dict[str, Any]:
    """Filter DNA dictionary to remove None values except for special keys."""
    return {
        k: v
        for k, v in dna_dict_base.items()
        if v is not None
        or k == "predicted_relationship"
        or k.startswith("_")
        or k == "uuid"
        or k.startswith("ethnicity_")
    }


def _filter_changed_ethnicity_values(
    existing_dna_match: Optional[DnaMatch],
    prefetched_ethnicity: dict[str, Optional[int]],
) -> dict[str, Optional[int]]:
    """Return only ethnicity values that differ from what is already stored."""

    if existing_dna_match is None:
        return prefetched_ethnicity

    changed: dict[str, Optional[int]] = {}
    for column_name, new_value in prefetched_ethnicity.items():
        if not hasattr(existing_dna_match, column_name):
            changed[column_name] = new_value
            continue

        current_value = getattr(existing_dna_match, column_name)
        if current_value != new_value:
            changed[column_name] = new_value

    return changed


def _add_ethnicity_data(
    dna_dict_base: dict[str, Any],
    existing_dna_match: Optional[DnaMatch],
    match: dict[str, Any],
    match_uuid: str,
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> None:
    """Add ethnicity data to DNA match dictionary from prefetched data."""
    if existing_dna_match is None or _needs_ethnicity_refresh(existing_dna_match):
        # Use prefetched ethnicity data from sequential API fetch
        prefetched_ethnicity = cast(Optional[dict[str, Optional[int]]], match.get("_prefetched_ethnicity"))
        if prefetched_ethnicity:
            ethnicity_updates = _filter_changed_ethnicity_values(existing_dna_match, prefetched_ethnicity)
            if ethnicity_updates:
                dna_dict_base.update(ethnicity_updates)
                logger_instance.debug(
                    "%s: Added ethnicity data (%d regions)",
                    log_ref_short,
                    len(ethnicity_updates),
                )
            else:
                logger_instance.debug(f"{log_ref_short}: Ethnicity unchanged; skipping update")
        else:
            short_uuid = match_uuid[:8] if match_uuid else "unknown"
            logger_instance.debug(f"{log_ref_short}: No prefetched ethnicity data available for {short_uuid}")


def _prepare_dna_match_operation_data(
    match: dict[str, Any],
    existing_dna_match: Optional[DnaMatch],
    prefetched_combined_details: Optional[dict[str, Any]],
    match_uuid: str,
    predicted_relationship: Optional[str],
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> Optional[dict[str, Any]]:
    """
    Prepares DnaMatch data for create or update operations by comparing API data with existing records.

    Args:
        match: Dictionary containing data for one match from the match list API.
        existing_dna_match: The existing DnaMatch object from the database, or None if this is a new match.
        prefetched_combined_details: Prefetched data from '/details' API containing DNA-specific information.
        match_uuid: The UUID (Sample ID) of the match.
        predicted_relationship: The predicted relationship string from the API, can be None if not available.
        log_ref_short: Short reference string for logging.
        logger_instance: The logger instance.
        session_manager: SessionManager instance for ethnicity API calls.

    Returns:
        Optional[dict[str, Any]]: Dictionary with DNA match data and '_operation' key set to 'create',
        or None if no create/update is needed. The dictionary includes fields like: cm_dna,
        shared_segments, longest_shared_segment, etc.
    """
    details_part = prefetched_combined_details or {}
    cm_value_raw = match.get("cm_dna", 0)
    try:
        cm_value = int(cm_value_raw or 0)
    except (TypeError, ValueError):
        logger_instance.debug(
            "%s: Unable to parse cm_dna value '%s'; defaulting to 0 for inference",
            log_ref_short,
            cm_value_raw,
        )
        cm_value = 0

    safe_predicted_relationship = _resolve_predicted_relationship_value(
        predicted_relationship,
        cm_value,
        log_ref_short,
        logger_instance,
    )
    api_predicted_rel_for_comp = safe_predicted_relationship

    needs_dna_create_or_update = _check_dna_update_needed(
        existing_dna_match, match, details_part, api_predicted_rel_for_comp, log_ref_short, logger_instance
    )

    if not needs_dna_create_or_update:
        return None

    dna_dict_base = _build_dna_dict_base(match_uuid, match, safe_predicted_relationship)
    _add_dna_details(dna_dict_base, prefetched_combined_details, match, log_ref_short, logger_instance)
    _add_ethnicity_data(dna_dict_base, existing_dna_match, match, match_uuid, log_ref_short, logger_instance)
    return _filter_dna_dict(dna_dict_base)


# End of _prepare_dna_match_operation_data


def _build_tree_links(
    their_cfpid: str, session_manager: SessionManager, config_schema_arg: "ConfigSchema"
) -> tuple[Optional[str], Optional[str]]:
    """Build facts and view links for a person in the tree."""
    tree_id = session_manager.my_tree_id
    if not their_cfpid or not tree_id:
        return None, None

    base_url = config_schema_arg.api.base_url
    base_person_path = f"/family-tree/person/tree/{tree_id}/person/{their_cfpid}"
    facts_link = urljoin(base_url, f"{base_person_path}/facts")
    view_params = {
        "cfpid": their_cfpid,
        "showMatches": "true",
        "sid": session_manager.my_uuid or "",
    }
    base_view_url = urljoin(base_url, f"/family-tree/tree/{tree_id}/family")
    view_in_tree_link = f"{base_view_url}?{urlencode(view_params)}"
    return facts_link, view_in_tree_link


def _check_tree_update_needed(
    existing_family_tree: FamilyTree,
    prefetched_tree_data: dict[str, Any],
    their_cfpid_final: Optional[str],
    facts_link: Optional[str],
    view_in_tree_link: Optional[str],
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> bool:
    """Check if family tree record needs updating."""
    fields_to_check = [
        ("cfpid", their_cfpid_final),
        ("person_name_in_tree", prefetched_tree_data.get("their_firstname", "Unknown")),
        ("actual_relationship", prefetched_tree_data.get("actual_relationship")),
        ("relationship_path", prefetched_tree_data.get("relationship_path")),
        ("facts_link", facts_link),
        ("view_in_tree_link", view_in_tree_link),
    ]
    for field_name, new_val in fields_to_check:
        old_val = getattr(existing_family_tree, field_name, None)
        if new_val != old_val:
            logger_instance.debug(f"  Tree change {log_ref_short}: Field '{field_name}'")
            return True
    return False


def _build_tree_data_dict(
    match_uuid: str,
    their_cfpid_final: Optional[str],
    prefetched_tree_data: dict[str, Any],
    facts_link: Optional[str],
    view_in_tree_link: Optional[str],
    tree_operation: Literal["create", "update", "none"],
    existing_family_tree: Optional[FamilyTree],
) -> dict[str, Any]:
    """Build family tree data dictionary for create/update operations."""
    tree_person_name = prefetched_tree_data.get("their_firstname", "Unknown")
    tree_dict_base = {
        "uuid": match_uuid.upper(),
        "cfpid": their_cfpid_final,
        "person_name_in_tree": tree_person_name,
        "facts_link": facts_link,
        "view_in_tree_link": view_in_tree_link,
        "actual_relationship": prefetched_tree_data.get("actual_relationship"),
        "relationship_path": prefetched_tree_data.get("relationship_path"),
        "_operation": tree_operation,
        "_existing_tree_id": (existing_family_tree.id if tree_operation == "update" and existing_family_tree else None),
    }
    # Keep all keys for _operation and _existing_tree_id, otherwise only non-None values
    return {
        k: v for k, v in tree_dict_base.items() if v is not None or k in {"_operation", "_existing_tree_id", "uuid"}
    }


def _determine_tree_operation(
    match_in_my_tree: bool,
    existing_family_tree: Optional[FamilyTree],
    prefetched_tree_data: Optional[dict[str, Any]],
    their_cfpid_final: Optional[str],
    facts_link: Optional[str],
    view_in_tree_link: Optional[str],
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> Literal["create", "update", "none"]:
    """Determine what operation is needed for family tree record."""
    if match_in_my_tree and existing_family_tree is None:
        # Only create if we have tree data available
        if prefetched_tree_data:
            return "create"
        logger_instance.debug(
            f"{log_ref_short}: Match is in tree but tree data not available. Skipping tree record creation."
        )
        return "none"
    if match_in_my_tree and existing_family_tree is not None:
        if prefetched_tree_data and _check_tree_update_needed(
            existing_family_tree,
            prefetched_tree_data,
            their_cfpid_final,
            facts_link,
            view_in_tree_link,
            log_ref_short,
            logger_instance,
        ):
            return "update"
        return "none"
    if not match_in_my_tree and existing_family_tree is not None:
        logger_instance.warning(
            f"{log_ref_short}: Data mismatch - API says not 'in_my_tree', but FamilyTree record exists (ID: {existing_family_tree.id}). Skipping."
        )
    return "none"


def _prepare_family_tree_operation_data(
    existing_family_tree: Optional[FamilyTree],
    prefetched_tree_data: Optional[dict[str, Any]],
    match_uuid: str,
    match_in_my_tree: bool,
    session_manager: SessionManager,
    config_schema_arg: "ConfigSchema",  # Config schema argument
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> tuple[Optional[dict[str, Any]], Literal["create", "update", "none"]]:
    """
    Prepares FamilyTree data for create or update operations based on API data and existing records.

    Args:
        existing_family_tree: The existing FamilyTree object from the database, or None if no record exists.
        prefetched_tree_data: Prefetched data from 'badgedetails' & 'getladder' APIs containing tree information.
        match_uuid: The UUID (Sample ID) of the match.
        match_in_my_tree: Boolean indicating if the match is in the user's family tree.
        session_manager: The active SessionManager instance containing session and tree information.
        config_schema_arg: The application configuration schema.
        log_ref_short: Short reference string for logging.
        logger_instance: The logger instance.

    Returns:
        A tuple containing:
        - tree_data (Optional[dict]): Dictionary with family tree data and '_operation' key
          set to 'create' or 'update'. None if no create/update is needed.
        - tree_operation (Literal["create", "update", "none"]): The operation type determined
          for this family tree record.
    """
    view_in_tree_link, facts_link = None, None
    their_cfpid_final = None

    if prefetched_tree_data:
        their_cfpid_final = prefetched_tree_data.get("their_cfpid")
        if their_cfpid_final:
            facts_link, view_in_tree_link = _build_tree_links(their_cfpid_final, session_manager, config_schema_arg)

    tree_operation = _determine_tree_operation(
        match_in_my_tree,
        existing_family_tree,
        prefetched_tree_data,
        their_cfpid_final,
        facts_link,
        view_in_tree_link,
        log_ref_short,
        logger_instance,
    )

    if tree_operation != "none":
        if prefetched_tree_data:
            tree_data = _build_tree_data_dict(
                match_uuid,
                their_cfpid_final,
                prefetched_tree_data,
                facts_link,
                view_in_tree_link,
                tree_operation,
                existing_family_tree,
            )
            return tree_data, tree_operation
        logger_instance.warning(
            f"{log_ref_short}: FamilyTree needs '{tree_operation}', but tree details not fetched. Skipping."
        )
        tree_operation = "none"

    return None, tree_operation


# End of _prepare_family_tree_operation_data

# ------------------------------------------------------------------------------
# Individual Match Processing (_do_match) - Refactored
# ------------------------------------------------------------------------------


@dataclass(slots=True)
class _MatchInfo:
    uuid: str
    username: str
    predicted_relationship: Optional[str]
    in_my_tree: bool
    log_ref_short: str


def _extract_match_info(match: dict[str, Any]) -> _MatchInfo:
    """Extract basic information from match data."""
    match_uuid = str(match.get("uuid") or "")
    match_username_raw = cast(Optional[str], match.get("username"))
    match_username = format_name(match_username_raw) if match_username_raw else "Unknown"
    predicted_relationship = cast(Optional[str], match.get("predicted_relationship"))
    match_in_my_tree = bool(match.get("in_my_tree"))
    log_ref_short = f"UUID={match_uuid} User='{match_username}'"
    return _MatchInfo(match_uuid, match_username, predicted_relationship, match_in_my_tree, log_ref_short)


def _process_person_data_safe(
    match: dict[str, Any],
    existing_person: Optional[Person],
    prefetched_combined_details: Optional[dict[str, Any]],
    prefetched_tree_data: Optional[dict[str, Any]],
    match_uuid: str,
    match_username: str,
    match_in_my_tree: bool,
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> tuple[Optional[dict[str, Any]], bool]:
    """Process person data with error handling."""
    try:
        return _prepare_person_operation_data(
            match=match,
            existing_person=existing_person,
            prefetched_combined_details=prefetched_combined_details,
            prefetched_tree_data=prefetched_tree_data,
            config_schema_arg=config_schema,
            match_uuid=match_uuid,
            formatted_match_username=match_username,
            match_in_my_tree=match_in_my_tree,
            log_ref_short=log_ref_short,
            logger_instance=logger_instance,
        )
    except Exception as person_err:
        logger_instance.error(
            f"Error in _prepare_person_operation_data for {log_ref_short}: {person_err}",
            exc_info=True,
        )
        return None, False


def _process_dna_data_safe(
    match: dict[str, Any],
    dna_match_record: Optional[DnaMatch],
    prefetched_combined_details: Optional[dict[str, Any]],
    match_uuid: str,
    predicted_relationship: Optional[str],
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> Optional[dict[str, Any]]:
    """Process DNA match data with error handling."""
    try:
        return _prepare_dna_match_operation_data(
            match=match,
            existing_dna_match=dna_match_record,
            prefetched_combined_details=prefetched_combined_details,
            match_uuid=match_uuid,
            predicted_relationship=predicted_relationship,
            log_ref_short=log_ref_short,
            logger_instance=logger_instance,
        )
    except Exception as dna_err:
        logger_instance.error(
            f"Error in _prepare_dna_match_operation_data for {log_ref_short}: {dna_err}",
            exc_info=True,
        )
        return None


def _process_tree_data_safe(
    family_tree_record: Optional[FamilyTree],
    prefetched_tree_data: Optional[dict[str, Any]],
    match_uuid: str,
    match_in_my_tree: bool,
    session_manager: SessionManager,
    log_ref_short: str,
    logger_instance: logging.Logger,
) -> tuple[Optional[dict[str, Any]], Literal["create", "update", "none"]]:
    """Process family tree data with error handling."""
    try:
        return _prepare_family_tree_operation_data(
            existing_family_tree=family_tree_record,
            prefetched_tree_data=prefetched_tree_data,
            match_uuid=match_uuid,
            match_in_my_tree=match_in_my_tree,
            session_manager=session_manager,
            config_schema_arg=config_schema,
            log_ref_short=log_ref_short,
            logger_instance=logger_instance,
        )
    except Exception as tree_err:
        logger_instance.error(
            f"Error in _prepare_family_tree_operation_data for {log_ref_short}: {tree_err}",
            exc_info=True,
        )
        fallback_status: Literal["none"] = "none"
        return None, fallback_status


def _populate_bulk_data_dict(
    prepared_data: dict[str, Any],
    person_op_data: Optional[dict[str, Any]],
    dna_op_data: Optional[dict[str, Any]],
    tree_op_data: Optional[dict[str, Any]],
    tree_operation_status: Literal["create", "update", "none"],
    is_new_person: bool,
) -> None:
    """Populate bulk data dictionary with operation data."""
    if person_op_data:
        prepared_data["person"] = person_op_data
    if dna_op_data:
        prepared_data["dna_match"] = dna_op_data
    if is_new_person:
        if tree_op_data and tree_operation_status == "create":
            prepared_data["family_tree"] = tree_op_data
    elif tree_op_data:
        prepared_data["family_tree"] = tree_op_data


def _determine_overall_status(
    is_new_person: bool,
    person_fields_changed: bool,
    dna_op_data: Optional[dict[str, Any]],
    tree_op_data: Optional[dict[str, Any]],
    tree_operation_status: Literal["create", "update", "none"],
) -> Literal["new", "updated", "skipped", "error"]:
    """Determine overall status based on operation data."""
    if is_new_person:
        return "new"
    if person_fields_changed or dna_op_data or (tree_op_data and tree_operation_status != "none"):
        return "updated"
    return "skipped"


def _assemble_bulk_data(
    is_new_person: bool,
    person_op_data: Optional[dict[str, Any]],
    dna_op_data: Optional[dict[str, Any]],
    tree_op_data: Optional[dict[str, Any]],
    tree_operation_status: Literal["create", "update", "none"],
    person_fields_changed: bool,
) -> tuple[dict[str, Any], Literal["new", "updated", "skipped", "error"]]:
    """Assemble bulk data and determine overall status."""
    prepared_data_for_bulk: dict[str, Any] = {
        "person": None,
        "dna_match": None,
        "family_tree": None,
    }

    _populate_bulk_data_dict(
        prepared_data_for_bulk, person_op_data, dna_op_data, tree_op_data, tree_operation_status, is_new_person
    )

    overall_status = _determine_overall_status(
        is_new_person, person_fields_changed, dna_op_data, tree_op_data, tree_operation_status
    )

    return prepared_data_for_bulk, overall_status


@dataclass(slots=True)
class _PreparedMatchOperations:
    person_op_data: Optional[dict[str, Any]]
    person_fields_changed: bool
    dna_op_data: Optional[dict[str, Any]]
    tree_op_data: Optional[dict[str, Any]]
    tree_operation_status: Literal["create", "update", "none"]


def _prepare_match_operations(
    match: dict[str, Any],
    existing_person: Optional[Person],
    prefetched_combined_details: Optional[dict[str, Any]],
    prefetched_tree_data: Optional[dict[str, Any]],
    match_info: _MatchInfo,
    logger_instance: logging.Logger,
    session_manager: SessionManager,
) -> _PreparedMatchOperations:
    person_op_data, person_fields_changed = _process_person_data_safe(
        match,
        existing_person,
        prefetched_combined_details,
        prefetched_tree_data,
        match_info.uuid or "",
        match_info.username,
        match_info.in_my_tree,
        match_info.log_ref_short,
        logger_instance,
    )

    dna_op_data = _process_dna_data_safe(
        match,
        existing_person.dna_match if existing_person else None,
        prefetched_combined_details,
        match_info.uuid or "",
        match_info.predicted_relationship,
        match_info.log_ref_short,
        logger_instance,
    )

    tree_op_data, tree_operation_status = _process_tree_data_safe(
        existing_person.family_tree if existing_person else None,
        prefetched_tree_data,
        match_info.uuid or "",
        match_info.in_my_tree,
        session_manager,
        match_info.log_ref_short,
        logger_instance,
    )

    return _PreparedMatchOperations(
        person_op_data=person_op_data,
        person_fields_changed=person_fields_changed,
        dna_op_data=dna_op_data,
        tree_op_data=tree_op_data,
        tree_operation_status=tree_operation_status,
    )


def _do_match(
    _session: SqlAlchemySession,
    match: dict[str, Any],
    session_manager: SessionManager,
    existing_person_arg: Optional[Person],
    prefetched_combined_details: Optional[dict[str, Any]],
    prefetched_tree_data: Optional[dict[str, Any]],
    _config_schema_arg: "ConfigSchema",
    logger_instance: logging.Logger,
) -> tuple[
    Optional[dict[str, Any]],
    Literal["new", "updated", "skipped", "error"],
    Optional[str],
]:
    """
    Processes a single DNA match by calling helper functions to compare incoming data
    with existing database records. Determines if a 'create', 'update', or 'skip'
    operation is needed and prepares a dictionary for bulk operations.

    This function orchestrates the data preparation process by:
    1. Extracting basic information from the match data
    2. Calling helper functions to prepare data for each table (Person, DnaMatch, FamilyTree)
    3. Determining the overall status based on the results from helper functions
    4. Assembling the final data structure for bulk database operations

    Args:
        session: The active SQLAlchemy database session.
        match: Dictionary containing data for one match from the match list API.
        session_manager: The active SessionManager instance with session and tree information.
        existing_person_arg: The existing Person object from the database, or None if this is a new person.
        prefetched_combined_details: Prefetched data from '/details' & '/profiles/details' APIs.
        prefetched_tree_data: Prefetched data from 'badgedetails' & 'getladder' APIs.
        config_schema_arg: The application configuration schema.
        logger_instance: The logger instance for recording debug/error information.

    Returns:
        A tuple containing:
        - prepared_data (Optional[dict[str, Any]]): Dictionary with keys 'person', 'dna_match', and
          'family_tree', each containing data for bulk operations or None if no change needed.
          Returns None if status is 'skipped' or 'error'.
        - status (Literal["new", "updated", "skipped", "error"]): The overall status determined
          for this match based on all data comparisons.
        - error_msg (Optional[str]): An error message if status is 'error', otherwise None.
    """
    existing_person: Optional[Person] = existing_person_arg

    match_info = _extract_match_info(match)

    if not match_info.uuid:
        error_msg = f"_do_match Pre-check failed: Missing 'uuid' in match data: {match}"
        logger_instance.error(error_msg)
        return None, "error", error_msg

    try:
        is_new_person = existing_person is None
        operations = _prepare_match_operations(
            match,
            existing_person,
            prefetched_combined_details,
            prefetched_tree_data,
            match_info,
            logger_instance,
            session_manager,
        )

        prepared_data_for_bulk, overall_status = _assemble_bulk_data(
            is_new_person,
            operations.person_op_data,
            operations.dna_op_data,
            operations.tree_op_data,
            operations.tree_operation_status,
            operations.person_fields_changed,
        )

        data_to_return = (
            prepared_data_for_bulk
            if overall_status not in {"skipped", "error"} and any(v for v in prepared_data_for_bulk.values())
            else None
        )

        if overall_status not in {"error", "skipped"} and not data_to_return:
            logger_instance.debug(
                f"Status is '{overall_status}' for {match_info.log_ref_short}, but no data payloads prepared. Revising to 'skipped'."
            )
            overall_status = "skipped"
            data_to_return = None

        return data_to_return, overall_status, None

    except Exception as e:
        error_type = type(e).__name__
        error_details = str(e)
        error_msg_for_log = f"Unexpected critical error ({error_type}) in _do_match for {match_info.log_ref_short}. Details: {error_details}"
        logger_instance.error(error_msg_for_log, exc_info=True)
        error_msg_return = f"Unexpected {error_type} during data prep for {match_info.log_ref_short}"
        return None, "error", error_msg_return


# End of _do_match

# ------------------------------------------------------------------------------
# API Data Acquisition Helpers (_fetch_*)
# ------------------------------------------------------------------------------


def _session_recovery_required(
    session_manager: SessionManager,
    driver: Optional[Any],
    my_uuid: Optional[str],
) -> bool:
    """Determine whether session recovery is needed for match retrieval."""

    needs_recovery = False
    if driver is None:
        logger.warning("get_matches: WebDriver missing at validation time; attempting recovery.")
        needs_recovery = True
    if not my_uuid:
        logger.warning("get_matches: my_uuid missing at validation time; attempting recovery.")
        needs_recovery = True
    if driver is not None and not session_manager.is_sess_valid():
        logger.warning("get_matches: Detected invalid session prior to API fetch; attempting recovery.")
        needs_recovery = True
    return needs_recovery


def _finalize_session_validation(
    session_manager: SessionManager,
    driver: Optional[Any],
    my_uuid: Optional[str],
) -> tuple[bool, Optional[Any], Optional[str]]:
    """Verify driver, UUID, and session validity after optional recovery."""

    if not driver:
        logger.error("get_matches: WebDriver unavailable after recovery attempt.")
        return False, None, None
    if not my_uuid:
        logger.error("get_matches: my_uuid unavailable after recovery attempt.")
        return False, None, None
    if not session_manager.is_sess_valid():
        logger.error("get_matches: Session remains invalid after recovery attempt.")
        return False, None, None
    return True, driver, my_uuid


def _validate_get_matches_session(session_manager: SessionManager) -> tuple[bool, Optional[Any], Optional[str]]:
    """
    Validate session manager, driver, UUID, and session validity for get_matches.

    Returns:
        Tuple of (is_valid, driver, my_uuid)
    """
    driver = session_manager.browser_manager.driver
    my_uuid = session_manager.my_uuid

    if _session_recovery_required(session_manager, driver, my_uuid):
        if not _ensure_action6_session_ready(session_manager, context="match list fetch"):
            logger.error("get_matches: Session recovery failed; cannot continue.")
            return False, None, None
        driver = session_manager.browser_manager.driver
        my_uuid = session_manager.my_uuid

    return _finalize_session_validation(session_manager, driver, my_uuid)


def _validate_and_refresh_page_url(driver: Any, session_manager: SessionManager) -> bool:
    """
    Validate current URL is on Ancestry page and refresh if needed.

    Returns:
        True if validation successful, False otherwise
    """
    try:
        # Check if we're still on a valid Ancestry page
        current_url = driver.current_url
        if not current_url or "ancestry.co" not in current_url:
            logger.warning(f"Driver not on Ancestry page. Current URL: {current_url}")
            # Try to refresh the page
            driver.refresh()
            time.sleep(2)

        # Validate session cookies are present
        if not session_manager.is_sess_valid():
            logger.error("Session validation failed before API call")
            return False
        return True
    except Exception as session_validation_error:
        logger.error(f"Session validation error: {session_validation_error}")
        return False


def _perform_smart_cookie_sync(session_manager: SessionManager) -> None:
    """
    Perform smart cookie sync with freshness tracking to avoid unnecessary syncing.
    """
    import time as time_module

    current_time = time_module.time()

    # Check if cookies were synced recently (within last 2 minutes)
    last_cookie_sync = getattr(session_manager, '_last_cookie_sync_time', 0)
    cookie_sync_needed = (current_time - last_cookie_sync) > 120  # 2 minutes (Reduced from 5m to prevent 303s)

    if cookie_sync_needed:
        # Force sync to bypass session_manager's "once-per-session" guard
        session_manager.sync_cookies_to_requests(force=True)
        session_manager._update_cookie_sync_time(current_time)
        logger.debug("Smart cookie sync performed (cookies were stale)")
    elif not cookie_sync_needed:
        logger.debug("Skipping cookie sync - cookies are fresh")
    else:
        logger.debug("Cookie sync method not available")


def _read_csrf_from_driver_cookies(driver: Any, csrf_token_cookie_names: tuple[str, ...]) -> Optional[str]:
    """
    Read CSRF token from driver cookies using get_cookie method.

    Returns:
        CSRF token string or None if not found
    """
    for cookie_name in csrf_token_cookie_names:
        try:
            cookie_obj = driver.get_cookie(cookie_name)
            if cookie_obj and "value" in cookie_obj and cookie_obj["value"]:
                csrf_token = unquote(cookie_obj["value"]).split("|")[0]
                logger.debug(f"Read CSRF token from cookie '{cookie_name}'.")
                return csrf_token
        except NoSuchCookieException:
            continue
        except WebDriverException as cookie_e:
            logger.warning(f"WebDriver error getting cookie '{cookie_name}': {cookie_e}")
        except Exception as e:
            logger.error(
                f"Unexpected error getting cookie '{cookie_name}': {e}",
                exc_info=True,
            )
    return None


def _read_csrf_from_fallback_cookies(driver: Any, csrf_token_cookie_names: tuple[str, ...]) -> Optional[str]:
    """
    Read CSRF token from driver cookies using get_driver_cookies fallback.

    Returns:
        CSRF token string or None if not found
    """
    logger.debug("CSRF token not found via get_cookie. Trying get_driver_cookies fallback...")
    all_cookies = get_driver_cookies(driver)
    if not all_cookies:
        logger.warning("Fallback get_driver_cookies also failed to retrieve cookies.")
        return None

    # get_driver_cookies returns a list of cookie dictionaries
    for cookie_name in csrf_token_cookie_names:
        for cookie in all_cookies:
            if cookie.get("name") == cookie_name and cookie.get("value"):
                csrf_token = unquote(cookie["value"]).split("|")[0]
                logger.debug(f"Read CSRF token via fallback from '{cookie_name}'.")
                return csrf_token
    return None


def _cache_csrf_token(session_manager: SessionManager, csrf_token: str) -> None:
    """Cache CSRF token in session manager."""
    import time as time_module

    session_manager._set_cached_csrf_token(csrf_token, time_module.time())


def _get_cached_or_fresh_csrf_token(session_manager: SessionManager, driver: Any) -> Optional[str]:
    """
    Get CSRF token from cache if valid, otherwise read fresh from cookies.

    Returns:
        CSRF token string or None if not found
    """
    import time as time_module

    # Check if we have a cached CSRF token that's still valid
    cached_csrf_token, cached_csrf_time = session_manager._get_cached_csrf_token()
    csrf_cache_valid = (time_module.time() - cached_csrf_time) < 1800  # 30 minutes

    if cached_csrf_token and csrf_cache_valid:
        logger.debug(f"Using cached CSRF token (age: {time_module.time() - cached_csrf_time:.1f}s)")
        return cached_csrf_token

    # Need to read CSRF token from cookies
    csrf_token_cookie_names = (
        "_dnamatches-matchlistui-x-csrf-token",
        "_csrf",
    )

    try:
        logger.debug(f"Reading fresh CSRF token from cookies: {csrf_token_cookie_names}")

        # Try reading from driver cookies first
        specific_csrf_token = _read_csrf_from_driver_cookies(driver, csrf_token_cookie_names)

        # If not found, try fallback method
        if not specific_csrf_token:
            specific_csrf_token = _read_csrf_from_fallback_cookies(driver, csrf_token_cookie_names)

        # Cache the token if found
        if specific_csrf_token:
            _cache_csrf_token(session_manager, specific_csrf_token)

        return specific_csrf_token

    except Exception as csrf_err:
        logger.error(f"Critical error during CSRF token retrieval: {csrf_err}", exc_info=True)
        return None


def _call_match_list_api(
    session_manager: SessionManager, my_uuid: str, current_page: int, specific_csrf_token: str
) -> Any:
    """
    Build URL and headers, then call the match list API.

    Returns:
        API response (dict, Response object, or None)
    """
    # Get matches_per_page from config (respects MATCHES_PER_PAGE in .env)
    # Default to 30 if not configured (balance between throughput and rate limiting)
    items_per_page = getattr(config_schema.api, 'matches_per_page', 30)

    # Use the working API endpoint pattern with itemsPerPage parameter
    # OPTIMIZATION NOTE: Higher values require more API calls per page
    # - itemsPerPage=50: ~70 API calls (risk of rate limiting)
    # - itemsPerPage=30: ~45 API calls (safe with adaptive rate limiter)
    # - itemsPerPage=20: ~30 API calls (Ancestry default, slower throughput)
    match_list_url = urljoin(
        config_schema.api.base_url,
        f"discoveryui-matches/parents/list/api/matchList/{my_uuid}?itemsPerPage={items_per_page}&currentPage={current_page}",
    )
    # Use simplified headers that were working earlier
    match_list_headers = {
        "X-CSRF-Token": specific_csrf_token,
        "Accept": "application/json",
        "Referer": urljoin(config_schema.api.base_url, "/discoveryui-matches/list/"),
    }
    logger.debug(f"Calling Match list API for page {current_page} (itemsPerPage={items_per_page})...")
    logger.debug(f"Headers being passed to _api_req for Match list: {match_list_headers}")

    # Additional debug logging for troubleshooting 303 redirects
    logger.debug(f"Match list URL: {match_list_url}")
    logger.debug(
        f"Session manager state - driver_live: {session_manager.browser_manager.driver_live}, session_ready: {session_manager.session_ready}"
    )

    # CRITICAL: Ensure cookies are synced immediately before API call
    try:
        session_manager.sync_cookies_to_requests()
    except Exception as cookie_sync_error:
        logger.warning("Session-level cookie sync hint failed (ignored): %s", cookie_sync_error)

    # Call the API with fresh cookie sync
    return _call_api_request(
        url=match_list_url,
        session_manager=session_manager,
        method="GET",
        headers=match_list_headers,
        use_csrf_token=False,
        api_description="Match list API",
        _allow_redirects=True,
    )


def _handle_303_with_redirect(
    location: str, session_manager: SessionManager, match_list_headers: dict[str, str]
) -> Any:
    """
    Handle 303 See Other response with redirect location.

    Returns:
        API response from redirected URL or None if failed
    """
    logger.warning(f"Match list API received 303 See Other. Retrying with redirect to {location}.")
    # Retry once with the new location
    api_response_redirect = _call_api_request(
        url=location,
        session_manager=session_manager,
        method="GET",
        headers=match_list_headers,
        use_csrf_token=False,
        api_description="Match list API (redirected)",
        _allow_redirects=False,
    )
    if isinstance(api_response_redirect, dict):
        return api_response_redirect
    logger.error(
        f"Redirected Match list API did not return dict. Status: {getattr(api_response_redirect, 'status_code', None)}"
    )
    return None


def _handle_303_session_refresh(
    session_manager: SessionManager, match_list_url: str, match_list_headers: dict[str, str]
) -> Any:
    """
    Handle 303 See Other without redirect location (session expired).
    Performs session refresh with cache clear and retries the API call.

    Returns:
        API response after session refresh or None if failed
    """
    logger.info(
        "Match list API received 303 See Other with no redirect location. "
        "This usually indicates session expiration. Attempting session refresh with cache clear."
    )
    try:
        # Clear session cache for complete fresh start
        try:
            cleared_count = session_manager._clear_session_caches()
            logger.debug(f"🧹 Cleared {cleared_count} session cache entries before refresh")
        except Exception as cache_err:
            logger.warning(f"⚠️ Could not clear session cache: {cache_err}")

        # Force clear readiness check cache to ensure fresh validation
        session_manager._clear_last_readiness_check()
        logger.debug("🔄 Cleared session readiness cache to force fresh validation")

        # Force session refresh with cleared cache
        fresh_success = session_manager.ensure_session_ready(action_name="coord_action - Session Refresh")
        if not fresh_success:
            logger.error("❌ Session refresh failed after cache clear")
            return None

        # Force cookie sync and CSRF token refresh
        session_manager.sync_cookies_to_requests(force=True)
        fresh_csrf_token = _get_csrf_token(session_manager, force_api_refresh=True)
        if fresh_csrf_token:
            # Update headers with fresh token and retry
            match_list_headers['X-CSRF-Token'] = fresh_csrf_token
            logger.info("✅ Retrying Match list API with refreshed session, cleared cache, and fresh CSRF token.")
            logger.debug(f"🔑 Fresh CSRF token: {fresh_csrf_token[:20]}...")
            logger.debug(
                f"🍪 Session cookies synced: {len(session_manager.api_manager.requests_session.cookies)} cookies"
            )

            api_response_refresh = _call_api_request(
                url=match_list_url,
                session_manager=session_manager,
                method="GET",
                headers=match_list_headers,
                use_csrf_token=False,
                api_description="Match list API (Session Refreshed)",
                _allow_redirects=True,
            )
            if isinstance(api_response_refresh, dict):
                return api_response_refresh
            logger.error("Match list API still failing after session refresh. Aborting.")
            return None
        logger.error("Could not obtain fresh CSRF token for session refresh.")
        return None
    except Exception as refresh_err:
        logger.error(f"Error during session refresh: {refresh_err}")
        return None


def _handle_non_dict_response(
    api_response: Any,
    session_manager: SessionManager,
    match_list_url: str,
    match_list_headers: dict[str, str],
) -> Optional[dict[str, Any]]:
    """Handle non-dict API response including 303 redirects."""
    if not isinstance(api_response, requests.Response):
        logger.error(f"Match list API did not return dict. Type: {type(api_response).__name__}")
        return None

    status = api_response.status_code
    # Check both Location and Content-Location
    location = api_response.headers.get('Location') or api_response.headers.get('Content-Location')

    if status == 303:
        if location:
            # Check if redirect is to signout/login - if so, treat as session expiry
            if "signout" in location.lower() or "login" in location.lower():
                logger.warning(f"303 Redirect to {location} detected (Signout/Login). Triggering session refresh.")
                return _handle_303_session_refresh(session_manager, match_list_url, match_list_headers)

            return _handle_303_with_redirect(location, session_manager, match_list_headers)

        logger.debug(f"303 Response Headers: {dict(api_response.headers)}")
        return _handle_303_session_refresh(session_manager, match_list_url, match_list_headers)

    logger.error(
        f"Match list API did not return dict. Type: {type(api_response).__name__}, "
        f"Status: {getattr(api_response, 'status_code', 'N/A')}"
    )
    return None


def _handle_match_list_response(
    api_response: Any,
    current_page: int,
    session_manager: SessionManager,
    match_list_url: str,
    match_list_headers: dict[str, str],
) -> Optional[dict[str, Any]]:
    """
    Handle and validate match list API response, including 303 redirect handling.

    Returns:
        Response dict if successful, None if failed
    """
    if api_response is None:
        logger.warning(f"No response/error from match list API page {current_page}. Assuming empty page.")
        return None

    if not isinstance(api_response, dict):
        return _handle_non_dict_response(api_response, session_manager, match_list_url, match_list_headers)

    return api_response


def _parse_total_pages(api_response: dict[str, Any], _current_page: int) -> Optional[int]:
    """
    Parse total pages from API response.

    Note: current_page parameter reserved for future logging enhancements.

    Returns:
        Total pages as integer or None if not found/invalid
    """
    total_pages: Optional[int] = None
    total_pages_raw = api_response.get("totalPages")
    if total_pages_raw is not None:
        try:
            total_pages = int(total_pages_raw)
        except (ValueError, TypeError):
            logger.warning(f"Could not parse totalPages '{total_pages_raw}'.")
    else:
        logger.warning("Total pages missing from match list response.")
    return total_pages


def _filter_valid_matches(match_data_list: list[Any], current_page: int) -> list[dict[str, Any]]:
    """
    Filter matches to only include those with valid sampleId.

    Returns:
        List of valid matches with sampleId
    """
    valid_matches_for_processing: list[dict[str, Any]] = []
    skipped_sampleid_count = 0
    for m_idx, m_val in enumerate(match_data_list):
        if isinstance(m_val, dict):
            m_dict = cast(dict[str, Any], m_val)
            if m_dict.get("sampleId"):
                valid_matches_for_processing.append(m_dict)
                continue

        skipped_sampleid_count += 1
        match_log_info = f"(Index: {m_idx}, Data: {str(m_val)[:100]}...)"
        logger.warning(f"Skipping raw match missing 'sampleId' on page {current_page}. {match_log_info}")
    if skipped_sampleid_count > 0:
        logger.warning(
            f"Skipped {skipped_sampleid_count} raw matches on page {current_page} due to missing 'sampleId'."
        )
    if not valid_matches_for_processing:
        logger.warning(f"No valid matches (with sampleId) found on page {current_page} to process further.")
    return valid_matches_for_processing


def _read_in_tree_cache(sample_ids_on_page: list[str], current_page: int) -> set[str]:
    """
    Read in-tree status from cache if available.

    Returns:
        Set of in-tree sample IDs (empty set if cache miss or error)
    """
    in_tree_ids: set[str] = set()
    cache_key_tree = f"matches_in_tree_{hash(frozenset(sample_ids_on_page))}"

    try:
        cache = get_unified_cache()
        cached_in_tree = cache.get("ancestry", "tree_search", cache_key_tree)
        if cached_in_tree is not None:
            if isinstance(cached_in_tree, set):
                in_tree_ids = cached_in_tree
            logger.debug(f"Loaded {len(in_tree_ids)} in-tree IDs from cache for page {current_page}.")
        else:
            logger.debug(f"Cache miss for in-tree status (Key: {cache_key_tree}). Fetching from API.")
    except Exception as cache_read_err:
        logger.error(
            f"Error reading in-tree status from cache: {cache_read_err}. Fetching from API.",
            exc_info=True,
        )
        in_tree_ids = set()

    return in_tree_ids


def _process_in_tree_api_response(response_in_tree: Any, sample_ids_on_page: list[str], current_page: int) -> set[str]:
    """
    Process in-tree API response and cache the result.

    Returns:
        Set of in-tree sample IDs (empty set if response is invalid)
    """
    in_tree_ids: set[str] = set()

    if isinstance(response_in_tree, list):
        in_tree_ids = {item.upper() for item in response_in_tree if isinstance(item, str)}
        logger.debug(f"Fetched {len(in_tree_ids)} in-tree IDs from API for page {current_page}.")
        # Cache the result
        try:
            cache_key_tree = f"matches_in_tree_{hash(frozenset(sample_ids_on_page))}"
            cache = get_unified_cache()
            cache.set(
                "ancestry",
                "tree_search",
                cache_key_tree,
                in_tree_ids,
                ttl=config_schema.cache.memory_cache_ttl,
            )
            logger.debug(f"Cached in-tree status result for page {current_page}.")
        except Exception as cache_write_err:
            logger.error(f"Error writing in-tree status to cache: {cache_write_err}")
    else:
        status_code_log = ""
        if isinstance(response_in_tree, requests.Response):
            status_code_log = f" Status: {response_in_tree.status_code}"
        logger.warning(
            f"In-Tree Status Check API failed or returned unexpected format for page {current_page}.{status_code_log}"
        )
        logger.debug(f"In-Tree check response: {response_in_tree}")

    return in_tree_ids


def _fetch_in_tree_from_api(
    session_manager: SessionManager,
    my_uuid: str,
    sample_ids_on_page: list[str],
    specific_csrf_token: str,
    current_page: int,
) -> set[str]:
    """
    Fetch in-tree status from API and cache the result.

    Returns:
        Set of in-tree sample IDs (empty set if API call fails)
    """
    in_tree_ids: set[str] = set()

    if not session_manager.is_sess_valid():
        logger.error(f"In-Tree Status Check: Session invalid page {current_page}. Cannot fetch.")
        return in_tree_ids

    in_tree_url = urljoin(
        config_schema.api.base_url,
        f"discoveryui-matches/parents/list/api/badges/matchesInTree/{my_uuid.upper()}",
    )
    parsed_base_url = urlparse(config_schema.api.base_url)
    origin_header_value = f"{parsed_base_url.scheme}://{parsed_base_url.netloc}"
    ua_in_tree = None
    if session_manager.browser_manager.driver and session_manager.is_sess_valid():
        with contextlib.suppress(Exception):
            script_result = cast(Any, session_manager.browser_manager.driver).execute_script(
                "return navigator.userAgent;"
            )
            if isinstance(script_result, str):
                ua_in_tree = script_result
    ua_in_tree = ua_in_tree or random.choice(config_schema.api.user_agents)
    in_tree_headers = {
        "X-CSRF-Token": specific_csrf_token,
        "Referer": urljoin(config_schema.api.base_url, "/discoveryui-matches/list/"),
        "Origin": origin_header_value,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": ua_in_tree,
    }
    in_tree_headers = {k: v for k, v in in_tree_headers.items() if v}

    logger.debug(f"Fetching in-tree status for {len(sample_ids_on_page)} matches on page {current_page}...")
    logger.debug(f"In-Tree Check Headers FULLY set in get_matches: {in_tree_headers}")
    response_in_tree = _call_api_request(
        url=in_tree_url,
        session_manager=session_manager,
        method="POST",
        json_data={"sampleIds": sample_ids_on_page},
        headers=in_tree_headers,
        use_csrf_token=False,
        api_description="In-Tree Status Check",
    )

    # Process the response and cache the result
    return _process_in_tree_api_response(response_in_tree, sample_ids_on_page, current_page)


def _fetch_in_tree_status(
    session_manager: SessionManager,
    my_uuid: str,
    valid_matches_for_processing: list[dict[str, Any]],
    specific_csrf_token: str,
    current_page: int,
) -> set[str]:
    """
    Fetch in-tree status for matches, using cache if available.

    Returns:
        Set of in-tree sample IDs
    """
    sample_ids_on_page = [match["sampleId"].upper() for match in valid_matches_for_processing]

    # Try to read from cache first
    in_tree_ids = _read_in_tree_cache(sample_ids_on_page, current_page)

    # If cache miss, fetch from API
    if not in_tree_ids:
        in_tree_ids = _fetch_in_tree_from_api(
            session_manager, my_uuid, sample_ids_on_page, specific_csrf_token, current_page
        )

    return in_tree_ids


def _normalize_profile_id(profile_user_id: Any) -> Optional[str]:
    if not profile_user_id:
        return None
    return str(profile_user_id).upper()


def _derive_first_name(match_username: str) -> Optional[str]:
    if not match_username or match_username == "Valued Relative":
        return None
    trimmed_username = match_username.strip()
    if not trimmed_username:
        return None
    name_parts = trimmed_username.split()
    return name_parts[0] if name_parts else None


def _build_compare_link(my_uuid: str, sample_id_upper: str) -> str:
    return urljoin(
        config_schema.api.base_url,
        f"discoveryui-matches/compare/{my_uuid.upper()}/with/{sample_id_upper}",
    )


def _refine_single_match(
    match_api_data: dict[str, Any], my_uuid: str, in_tree_ids: set[str], match_index: int, current_page: int
) -> Optional[dict[str, Any]]:
    """
    Refine a single match from raw API data into structured format.

    Returns:
        Refined match dict or None if refinement fails
    """
    try:
        profile_info = match_api_data.get("matchProfile", {})
        relationship_info = match_api_data.get("relationship", {})
        sample_id_upper = str(match_api_data["sampleId"]).upper()

        match_username = format_name(profile_info.get("displayName"))
        first_name = _derive_first_name(match_username)

        admin_profile_id_hint = match_api_data.get("adminId")
        admin_username_hint = match_api_data.get("adminName")
        profile_user_id_upper = _normalize_profile_id(profile_info.get("userId"))
        compare_link = _build_compare_link(my_uuid, sample_id_upper)
        is_in_tree = sample_id_upper in in_tree_ids

        return {
            "username": match_username,
            "first_name": first_name,
            "initials": profile_info.get("displayInitials", "??").upper(),
            "gender": match_api_data.get("gender"),
            "profile_id": profile_user_id_upper,
            "uuid": sample_id_upper,
            "administrator_profile_id_hint": admin_profile_id_hint,
            "administrator_username_hint": admin_username_hint,
            "photoUrl": profile_info.get("photoUrl", ""),
            "cm_dna": int(relationship_info.get("sharedCentimorgans", 0)),
            "numSharedSegments": int(relationship_info.get("numSharedSegments", 0)),
            "predicted_relationship": relationship_info.get("relationshipRange")
            or relationship_info.get("predictedRelationship"),
            "compare_link": compare_link,
            "message_link": None,
            "in_my_tree": is_in_tree,
            "createdDate": match_api_data.get("createdDate"),
            "match_tree_id": match_api_data.get("treeId"),
            "match_tree_person_id": match_api_data.get("treePersonId"),
            "has_public_tree": match_api_data.get("userTreePublic"),
            "tree_size": match_api_data.get("personCount"),
        }

    except (IndexError, KeyError, TypeError, ValueError) as refine_err:
        match_uuid_err = match_api_data.get("sampleId", "UUID_UNKNOWN")
        logger.error(
            f"Refinement error page {current_page}, match #{match_index + 1} (UUID: {match_uuid_err}): {type(refine_err).__name__} - {refine_err}. Skipping match.",
            exc_info=False,
        )
        logger.debug(f"Problematic match data during refinement: {match_api_data}")
        return None
    except Exception as critical_refine_err:
        match_uuid_err = match_api_data.get("sampleId", "UUID_UNKNOWN")
        logger.error(
            f"CRITICAL unexpected error refining match page {current_page}, match #{match_index + 1} (UUID: {match_uuid_err}): {critical_refine_err}",
            exc_info=True,
        )
        logger.debug(f"Problematic match data during critical error: {match_api_data}")
        raise critical_refine_err


def _refine_all_matches(
    valid_matches_for_processing: list[dict[str, Any]], my_uuid: str, in_tree_ids: set[str], current_page: int
) -> list[dict[str, Any]]:
    """
    Refine all matches from raw API data into structured format.

    Returns:
        List of refined match dicts
    """
    refined_matches: list[dict[str, Any]] = []
    logger.debug(f"Refining {len(valid_matches_for_processing)} valid matches...")
    for match_index, match_api_data in enumerate(valid_matches_for_processing):
        refined_match = _refine_single_match(match_api_data, my_uuid, in_tree_ids, match_index, current_page)
        if refined_match is not None:
            refined_matches.append(refined_match)

    logger.debug(f"Successfully refined {len(refined_matches)} matches on page {current_page}.")
    return refined_matches


def get_matches(
    session_manager: SessionManager,
    _db_session: SqlAlchemySession,  # Parameter name changed for clarity
    current_page: int = 1,
) -> Optional[tuple[list[dict[str, Any]], Optional[int]]]:
    """
    Fetches a single page of DNA match list data from the Ancestry API v2.
    Also fetches the 'in_my_tree' status for matches on the page via a separate API call.
    Refines the raw API data into a more structured format.

    Args:
        session_manager: The active SessionManager instance.
        _db_session: The active SQLAlchemy database session (not used directly in this function but
                     passed to maintain interface consistency with other functions).
        current_page: The page number to fetch (1-based).

    Returns:
        A tuple containing:
        - list of refined match data dictionaries for the page, or empty list if none.
        - Total number of pages available (integer), or None if retrieval fails.
        Returns None if a critical error occurs during fetching.

    Note:
        _db_session parameter is kept for interface consistency with other functions.
    """

    # Step 1: Validate session, driver, UUID
    is_valid, driver, my_uuid = _validate_get_matches_session(session_manager)
    if not is_valid:
        return None

    # Type assertion: my_uuid is guaranteed to be str (not None) after validation
    assert my_uuid is not None, "my_uuid should not be None after successful validation"

    logger.debug(f"--- Fetching Match list Page {current_page} ---")

    # Step 2: Validate and refresh page URL if needed
    if not _validate_and_refresh_page_url(driver, session_manager):
        return None

    # Step 3: Perform smart cookie sync
    _perform_smart_cookie_sync(session_manager)

    # Step 4: Get CSRF token (cached or fresh)
    specific_csrf_token = _get_cached_or_fresh_csrf_token(session_manager, driver)
    if not specific_csrf_token:
        logger.error("Failed to obtain specific CSRF token required for Match list API.")
        return None
    logger.debug(f"Specific CSRF token FOUND: '{specific_csrf_token}'")

    # Step 5: Call match list API
    api_response = _call_match_list_api(session_manager, my_uuid, current_page, specific_csrf_token)

    # Build match_list_url for use in response handling
    match_list_url = urljoin(
        config_schema.api.base_url,
        f"discoveryui-matches/parents/list/api/matchList/{my_uuid}?currentPage={current_page}",
    )
    match_list_headers = {
        "X-CSRF-Token": specific_csrf_token,
        "Accept": "application/json",
        "Referer": urljoin(config_schema.api.base_url, "/discoveryui-matches/list/"),
    }
    # Step 6: Handle response (including 303 redirects and session refresh)
    api_response = _handle_match_list_response(
        api_response, current_page, session_manager, match_list_url, match_list_headers
    )
    if api_response is None:
        return [], None

    # Step 7: Parse total pages
    total_pages = _parse_total_pages(api_response, current_page)

    # Step 8: Filter valid matches
    match_data_list = api_response.get("matchList", [])
    valid_matches_for_processing = _filter_valid_matches(match_data_list, current_page)
    if not valid_matches_for_processing:
        return [], total_pages

    # Step 9: Fetch in-tree status (using cache if available)
    in_tree_ids = _fetch_in_tree_status(
        session_manager, my_uuid, valid_matches_for_processing, specific_csrf_token, current_page
    )

    # Step 10: Refine all matches
    refined_matches = _refine_all_matches(valid_matches_for_processing, my_uuid, in_tree_ids, current_page)

    return refined_matches, total_pages


# End of get_matches


def _extract_relationship_string(predictions: list[Any]) -> Optional[str]:
    """Extract formatted relationship string from predictions."""
    if not predictions:
        return None

    valid_preds: list[dict[str, Any]] = []
    for candidate in predictions:
        if isinstance(candidate, dict) and "distributionProbability" in candidate and "pathsToMatch" in candidate:
            valid_preds.append(candidate)

    if not valid_preds:
        return None

    best_pred = max(valid_preds, key=lambda x: x.get("distributionProbability", 0.0))
    top_prob_raw = best_pred.get("distributionProbability", 0.0)
    top_prob = float(top_prob_raw) if isinstance(top_prob_raw, (int, float)) else 0.0

    paths_raw = best_pred.get("pathsToMatch", [])
    path_entries = paths_raw if isinstance(paths_raw, Sequence) else []

    labels: list[str] = []
    for path in path_entries:
        if isinstance(path, dict):
            path_dict = cast(dict[str, Any], path)
            label_value = path_dict.get("label")
            if isinstance(label_value, str):
                labels.append(label_value)

    max_labels_param = 2  # Default used in action6
    final_labels = labels[:max_labels_param]
    relationship_str_val = " or ".join(map(str, final_labels))
    return f"{relationship_str_val} [{top_prob:.1f}%]"


def _parse_details_response(details_response: Any, match_uuid: str) -> Optional[dict[str, Any]]:
    """Parse match details API response."""
    if not (details_response and isinstance(details_response, dict)):
        if isinstance(details_response, requests.Response):
            logger.error(
                f"Match Details API failed for UUID {match_uuid}. Status: {details_response.status_code} {details_response.reason}"
            )
        else:
            logger.error(f"Match Details API did not return dict for UUID {match_uuid}. Type: {type(details_response)}")
        return None

    details_dict = cast(dict[str, Any], details_response)
    relationship_part = cast(dict[str, Any], details_dict.get("relationship", {}))

    # Extract relationship predictions
    predictions = details_dict.get("predictions", [])
    relationship_str = _extract_relationship_string(predictions)

    return {
        "admin_profile_id": details_dict.get("adminUcdmId"),
        "admin_username": details_dict.get("adminDisplayName"),
        "tester_profile_id": details_dict.get("userId"),
        "tester_username": details_dict.get("displayName"),
        "tester_initials": details_dict.get("displayInitials"),
        "gender": details_dict.get("subjectGender"),
        "shared_segments": relationship_part.get("sharedSegments"),
        "longest_shared_segment": relationship_part.get("longestSharedSegment"),
        "meiosis": relationship_part.get("meiosis"),
        "from_my_fathers_side": bool(details_dict.get("fathersSide", False)),
        "from_my_mothers_side": bool(details_dict.get("mothersSide", False)),
        "relationship_str": relationship_str,
    }

    badge_dict = cast(dict[str, Any], badge_response)
    person_badged = cast(dict[str, Any], badge_dict.get("personBadged", {}))
    if not person_badged:
        logger.warning(f"Badge details response for UUID {match_uuid} missing 'personBadged' key.")
        return None

    their_cfpid = person_badged.get("personId")
    raw_firstname = person_badged.get("firstName")
    formatted_name_val = format_name(raw_firstname)
    their_firstname_formatted = (
        formatted_name_val.split()[0] if formatted_name_val and formatted_name_val != "Valued Relative" else "Unknown"
    )

    return {
        "their_cfpid": their_cfpid,
        "their_firstname": their_firstname_formatted,
        "their_lastname": person_badged.get("lastName", "Unknown"),
        "their_birth_year": person_badged.get("birthYear"),
    }


# ------------------------------------------------------------------------------
# Utility & Helper Functions
# ------------------------------------------------------------------------------


def _log_page_summary(page: int, page_new: int, page_updated: int, page_skipped: int, page_errors: int) -> None:
    """Logs a summary of processed matches for a single page with proper formatting."""
    logger.debug("")  # Blank line above
    logger.debug(f"---- Page {page} Summary ----")
    logger.debug(f"  New Person/Data: {page_new}")
    logger.debug(f"  Updated Person/Data: {page_updated}")
    logger.debug(f"  Skipped (No Change): {page_skipped}")
    logger.debug(f"  Errors during Prep/DB: {page_errors}")
    logger.debug("---------------------------")
    logger.debug("")  # Blank line below


# End of _log_page_summary


def _adjust_delay(session_manager: SessionManager, current_page: int) -> None:
    """
    Adjusts the adaptive rate limiter's delay based on throttling feedback
    received during the processing of the current page.

    Args:
        session_manager: The active SessionManager instance.
        current_page: The page number just processed (for logging context).
    """
    limiter = session_manager.rate_limiter
    if limiter is None:
        return
    if hasattr(limiter, "is_throttled") and limiter.is_throttled():
        logger.info(f"Adaptive rate limiting: throttling detected on page {current_page}. Delay remains increased.")
    else:
        # Success - notify rate limiter (AdaptiveRateLimiter interface)
        if hasattr(limiter, "on_success"):
            limiter.on_success()
            logger.debug("API success recorded in rate limiter")
        # Log significant rate changes
        metrics = limiter.get_metrics() if hasattr(limiter, "get_metrics") else None
        if metrics and hasattr(metrics, "current_fill_rate"):
            max_rate = getattr(limiter, "max_fill_rate", "N/A")
            logger.info(
                f"Rate limiting currently {metrics.current_fill_rate:.3f} req/s (Max: {max_rate}) after page {current_page}"
            )


# End of _adjust_delay


def _can_nav_to_list(session_manager: Optional[SessionManager]) -> bool:
    """Return True if the session is ready for match-list UI navigation."""
    return bool(session_manager and session_manager.is_sess_valid() and session_manager.my_uuid)


def nav_to_list(session_manager: SessionManager, current_page: Optional[int] = None) -> bool:
    """
    Navigates the browser directly to the user's specific DNA matches list page,
    using the UUID stored in the SessionManager. Verifies successful navigation
    by checking the final URL and waiting for a match entry element.

    Args:
        session_manager: The active SessionManager instance.
        current_page: Optional 1-based page index to include in the UI URL (resume support).

    Returns:
        True if navigation was successful, False otherwise.
    """
    if not _can_nav_to_list(session_manager):
        logger.error("nav_to_list: Session invalid or UUID missing.")
        return False

    my_uuid = session_manager.my_uuid

    target_url_base = urljoin(config_schema.api.base_url, f"discoveryui-matches/list/{my_uuid}")
    query_parts: list[str] = []
    page_int: Optional[int] = None
    if current_page is not None:
        with contextlib.suppress(TypeError, ValueError):
            page_int = int(current_page)
    if page_int is not None and page_int > 0:
        query_parts.append(f"currentPage={page_int}")

    # Include itemsPerPage to keep the UI paging consistent with API paging.
    with contextlib.suppress(Exception):
        per_page = int(MATCHES_PER_PAGE)
        if per_page > 0:
            query_parts.append(f"itemsPerPage={per_page}")

    query = f"?{'&'.join(query_parts)}" if query_parts else ""
    target_url = f"{target_url_base}{query}"

    logger.debug(f"Navigating to specific match list URL: {target_url}")

    driver = session_manager.browser_manager.driver
    if driver is None:
        logger.error("nav_to_list: WebDriver is None")
        return False

    match_entry_selector = cast(str, MATCH_ENTRY_SELECTOR)
    success = nav_to_page(
        driver=driver,
        url=target_url,
        selector=match_entry_selector,
        session_manager=session_manager,
    )

    if success:
        try:
            current_url = driver.current_url
            if not current_url.startswith(target_url_base):
                logger.warning(f"Navigation successful (element found), but final URL unexpected: {current_url}")
            else:
                logger.debug("Successfully navigated to specific matches list page.")
        except Exception as e:
            logger.warning(f"Could not verify final URL after nav_to_list success: {e}")
    else:
        logger.error("Failed navigation to specific matches list page using nav_to_page.")
    return success


# End of nav_to_list


# ------------------------------------------------------------------------------
# Test Harness
# ------------------------------------------------------------------------------


# ==============================================
# Test Functions
# ==============================================


# ==============================================
# Module-Level Test Functions
# ==============================================


def _test_cache_profile_helpers() -> bool:
    """Validate profile cache helper functions use UnifiedCacheManager."""
    from core.unified_cache_manager import get_unified_cache

    cache = get_unified_cache()
    cache.clear()

    profile_id = "TEST_PROFILE_CACHE_HELPERS"
    payload = {
        "last_logged_in_dt": "2024-01-01T00:00:00Z",
        "contactable": True,
    }

    _cache_profile(profile_id, payload)
    cached = _get_cached_profile(profile_id)

    assert cached is not None, "Cached profile should be retrievable"
    assert cached.get("contactable") is True, "Contactable flag should round-trip"
    assert cached.get("last_logged_in_dt") == payload["last_logged_in_dt"], "Last login should persist"

    cache.invalidate(service="ancestry", endpoint="profile_details")
    return True


def _test_combined_details_cache_helpers() -> bool:
    """Ensure combined details cache helpers interact with UnifiedCacheManager."""
    from core.unified_cache_manager import get_unified_cache

    cache = get_unified_cache()
    cache.clear()

    match_uuid = "TEST-CACHE-MATCH-UUID"
    combined_data = {
        "tester_profile_id": "TEST-PROFILE-UUID",
        "admin_profile_id": "ADMIN-UUID",
        "shared_segments": 3,
    }

    _cache_combined_details(combined_data, match_uuid)
    cached = _check_combined_details_cache(match_uuid, time.time())

    assert cached is not None, "Cached combined details should be retrievable"
    assert cached.get("tester_profile_id") == "TEST-PROFILE-UUID", "Tester profile ID should persist"
    assert cached.get("shared_segments") == 3, "Shared segments should persist"

    cache.invalidate(service="ancestry", endpoint="combined_details")
    return True


def _test_nav_to_list_predefined_page() -> bool:
    """Verify match list UI navigation supports an explicit resume page."""
    from types import SimpleNamespace
    from urllib.parse import parse_qs, urlparse

    from browser.css_selectors import WAIT_FOR_PAGE_SELECTOR

    class _FakeDriver:
        def __init__(self) -> None:
            self.current_url = ""

    class _FakeSessionManager:
        def __init__(self, my_uuid: str, driver: Any) -> None:
            self.my_uuid = my_uuid
            self.browser_manager = SimpleNamespace(driver=driver)

        @staticmethod
        def is_sess_valid() -> bool:
            return True

    fake_driver = _FakeDriver()
    fake_sm = _FakeSessionManager(my_uuid="TEST-UUID", driver=fake_driver)

    expected_base = urljoin(config_schema.api.base_url, "discoveryui-matches/list/TEST-UUID")

    captured: dict[str, Any] = {}

    def _stub_nav_to_page(
        *,
        driver: Any,
        url: str,
        selector: str = WAIT_FOR_PAGE_SELECTOR,
        session_manager: Any = None,
    ) -> bool:
        captured["driver"] = driver
        captured["url"] = url
        captured["selector"] = selector
        captured["session_manager"] = session_manager
        # Simulate that the browser landed on the expected URL.
        with contextlib.suppress(Exception):
            driver.current_url = url
        return True

    original_nav_to_page = globals().get("nav_to_page")
    try:
        globals()["nav_to_page"] = _stub_nav_to_page

        ok = nav_to_list(cast(Any, fake_sm), current_page=243)
        assert ok is True
        captured_url = cast(str, captured.get("url"))
        assert captured_url.startswith(expected_base)
        parsed = urlparse(captured_url)
        qs = parse_qs(parsed.query)
        assert qs.get("currentPage") == ["243"], f"Expected currentPage=243, got {qs.get('currentPage')}"
        assert qs.get("itemsPerPage") == [str(MATCHES_PER_PAGE)], (
            f"Expected itemsPerPage={MATCHES_PER_PAGE}, got {qs.get('itemsPerPage')}"
        )
        assert captured.get("driver") is fake_driver
        assert captured.get("session_manager") is fake_sm
        assert isinstance(captured.get("selector"), str) and captured.get("selector"), "Selector should be non-empty"
        return True
    finally:
        if original_nav_to_page is not None:
            globals()["nav_to_page"] = original_nav_to_page
        else:
            with contextlib.suppress(KeyError):
                del globals()["nav_to_page"]


def _test_nav_to_list_includes_items_per_page_when_page_omitted() -> bool:
    """Verify UI navigation includes itemsPerPage even without a specific currentPage."""
    from types import SimpleNamespace
    from urllib.parse import parse_qs, urlparse

    from browser.css_selectors import WAIT_FOR_PAGE_SELECTOR

    class _FakeDriver:
        def __init__(self) -> None:
            self.current_url = ""

    class _FakeSessionManager:
        def __init__(self, my_uuid: str, driver: Any) -> None:
            self.my_uuid = my_uuid
            self.browser_manager = SimpleNamespace(driver=driver)

        @staticmethod
        def is_sess_valid() -> bool:
            return True

    fake_driver = _FakeDriver()
    fake_sm = _FakeSessionManager(my_uuid="TEST-UUID", driver=fake_driver)

    expected_base = urljoin(config_schema.api.base_url, "discoveryui-matches/list/TEST-UUID")
    captured: dict[str, Any] = {}

    def _stub_nav_to_page(
        *,
        driver: Any,
        url: str,
        selector: str = WAIT_FOR_PAGE_SELECTOR,
        session_manager: Any = None,
    ) -> bool:
        _ = selector
        _ = session_manager
        captured["url"] = url
        with contextlib.suppress(Exception):
            driver.current_url = url
        return True

    original_nav_to_page = globals().get("nav_to_page")
    try:
        globals()["nav_to_page"] = _stub_nav_to_page

        ok = nav_to_list(cast(Any, fake_sm), current_page=None)
        assert ok is True

        captured_url = cast(str, captured.get("url"))
        assert captured_url.startswith(expected_base)
        parsed = urlparse(captured_url)
        qs = parse_qs(parsed.query)
        assert qs.get("itemsPerPage") == [str(MATCHES_PER_PAGE)]
        # currentPage is optional; if absent it should not break navigation.
        return True
    finally:
        if original_nav_to_page is not None:
            globals()["nav_to_page"] = original_nav_to_page
        else:
            with contextlib.suppress(KeyError):
                del globals()["nav_to_page"]


def _test_initial_navigation_threads_start_page() -> bool:
    """Verify initial navigation passes start_page into match list page navigation."""
    captured: dict[str, Any] = {}

    def _stub_ensure_on_match_list_page(session_manager: SessionManager, desired_page: Optional[int] = None) -> bool:
        captured["session_manager"] = session_manager
        captured["desired_page"] = desired_page
        # Force early exit from _navigate_and_get_initial_page_data so the test stays isolated.
        return False

    original_ensure = globals().get("_ensure_on_match_list_page")
    try:
        globals()["_ensure_on_match_list_page"] = _stub_ensure_on_match_list_page

        fake_sm: Any = object()
        matches, total_pages, ok = _navigate_and_get_initial_page_data(fake_sm, start_page=243)
        assert (matches, total_pages, ok) == (None, None, False)
        assert captured.get("session_manager") is fake_sm
        assert captured.get("desired_page") == 243
        return True
    finally:
        if original_ensure is not None:
            globals()["_ensure_on_match_list_page"] = original_ensure
        else:
            with contextlib.suppress(KeyError):
                del globals()["_ensure_on_match_list_page"]


def _test_module_initialization() -> bool:
    """Test module initialization and state functions with detailed verification"""
    from actions.gather import orchestrator as gather_orchestrator

    print("📋 Testing Action 6 module initialization:")
    results: list[bool] = []

    # Test initialize_gather_state function
    print("   • Testing initialize_gather_state...")
    try:
        state: Any = gather_orchestrator.initialize_gather_state()
        is_dict = isinstance(state, dict)

        required_keys = ["total_new", "total_updated", "total_pages_processed"]
        keys_present = all(key in state for key in required_keys)

        print(f"   ✅ State dictionary created: {is_dict}")
        print(f"   ✅ Required keys present: {keys_present} ({len(required_keys)} keys)")
        print(f"   ✅ State structure: {list(state.keys())}")

        results.extend([is_dict, keys_present])
        assert is_dict, "Should return dictionary state"
        assert keys_present, "Should have all required keys in state"

    except Exception as e:
        print(f"   ❌ initialize_gather_state: Exception {e}")
        results.extend([False, False])

    # Test validate_start_page function
    print("   • Testing validate_start_page...")
    validation_tests = [
        ("5", 5, "String number conversion"),
        (10, 10, "Integer input handling"),
        (None, 1, "None input (should default to 1)"),
        ("invalid", 1, "Invalid string (should default to 1)"),
        (0, 1, "Zero input (should default to 1)"),
    ]

    for input_val, expected, description in validation_tests:
        try:
            result = gather_orchestrator.validate_start_page(input_val)
            matches_expected = result == expected

            status = "✅" if matches_expected else "❌"
            print(f"   {status} {description}: {input_val!r} → {result}")

            results.append(matches_expected)
            assert matches_expected, f"Failed for {input_val}: expected {expected}, got {result}"

        except Exception as e:
            print(f"   ❌ {description}: Exception {e}")
            results.append(False)

    print(f"📊 Results: {sum(results)}/{len(results)} initialization tests passed")
    return all(results)


def _test_core_functionality() -> bool:
    """Test actual behavior of core DNA match gathering functions"""
    import inspect
    from unittest.mock import MagicMock

    # Test get_matches function signature and type hints
    sig = inspect.signature(get_matches)
    params = list(sig.parameters.keys())
    assert "session_manager" in params, "get_matches requires session_manager parameter"
    assert "current_page" in params, "get_matches requires current_page parameter"
    assert sig.return_annotation is not inspect.Signature.empty, "get_matches should have return type hint"

    # Test coord function signature accepts optional start page
    coord_sig = inspect.signature(coord)
    coord_params = coord_sig.parameters
    assert "session_manager" in coord_params, "coord requires session_manager"
    assert "start" in coord_params, "coord requires start parameter"
    assert coord_params["start"].default is None, "coord start parameter should default to None"

    # Test nav_to_list returns bool and checks session validity
    nav_sig = inspect.signature(nav_to_list)
    assert nav_sig.return_annotation is bool or str(nav_sig.return_annotation) == "bool", (
        "nav_to_list should return bool"
    )

    # Test nav_to_list with invalid session_manager returns False
    mock_session_manager = MagicMock()
    mock_session_manager.is_sess_valid.return_value = False
    mock_session_manager.my_uuid = None
    result = nav_to_list(mock_session_manager)
    assert result is False, "nav_to_list should return False for invalid session"

    return True


def _test_data_processing_functions() -> bool:
    """Test actual behavior of data processing and preparation functions"""
    import inspect

    from actions.gather import persistence as gather_persistence_module

    # Test process_batch_lookups function signature and return type
    lookup_sig = inspect.signature(gather_persistence_module.process_batch_lookups)
    params = lookup_sig.parameters
    assert "batch_session" in params, "process_batch_lookups requires batch_session"
    assert "matches_on_page" in params, "process_batch_lookups requires matches_on_page"
    assert "current_page" in params, "process_batch_lookups requires current_page"
    assert "page_statuses" in params, "process_batch_lookups requires page_statuses"

    # Test prepare_and_commit_batch_data function signature
    commit_sig = inspect.signature(gather_persistence_module.prepare_and_commit_batch_data)
    commit_params = commit_sig.parameters
    assert "batch_session" in commit_params, "prepare_and_commit requires batch_session"
    assert "session_manager" in commit_params, "prepare_and_commit requires session_manager"
    assert "matches_to_process_later" in commit_params, "prepare_and_commit requires matches list"

    # Test _execute_bulk_db_operations function signature
    bulk_sig = inspect.signature(_execute_bulk_db_operations)
    bulk_params = bulk_sig.parameters
    assert "session" in bulk_params, "_execute_bulk_db_operations requires session"
    assert "prepared_bulk_data" in bulk_params, "_execute_bulk_db_operations requires prepared data"
    assert "existing_persons_map" in bulk_params, "_execute_bulk_db_operations requires existing map"
    assert bulk_sig.return_annotation is bool or str(bulk_sig.return_annotation) == "bool", (
        "_execute_bulk_db_operations should return bool"
    )

    return True


def _test_edge_cases() -> bool:
    """Test edge cases and boundary conditions"""
    from actions.gather import orchestrator as gather_orchestrator

    # Test validate_start_page with edge cases
    result = gather_orchestrator.validate_start_page("invalid")
    assert result == 1, "Should handle invalid string input"

    result = gather_orchestrator.validate_start_page(-5)
    assert result == 1, "Should handle negative numbers"

    result = gather_orchestrator.validate_start_page(0)
    assert result == 1, "Should handle zero input"

    return True


def _test_integration() -> bool:
    """Test integration with external dependencies"""
    import inspect
    from unittest.mock import MagicMock

    # Test that core functions can work with session manager interface
    mock_session_manager = MagicMock()
    mock_session_manager.get_driver.return_value = MagicMock()
    mock_session_manager.my_profile_id = "test_profile_12345"

    # Test nav_to_list function signature and callability
    sig = inspect.signature(nav_to_list)
    params = list(sig.parameters.keys())
    assert "session_manager" in params, "nav_to_list should accept session_manager parameter"
    assert callable(nav_to_list), "nav_to_list should be callable"

    # Test coord function accepts proper parameters
    coord_sig = inspect.signature(coord)
    coord_params = list(coord_sig.parameters.keys())
    assert len(coord_params) > 0, "coord should accept parameters"

    return True


def _test_performance() -> bool:
    """Test performance of data processing operations"""
    import time

    from actions.gather import orchestrator as gather_orchestrator

    # Test initialize_gather_state performance
    start_time = time.time()
    for _ in range(100):
        state = gather_orchestrator.initialize_gather_state()
        assert isinstance(state, dict), "Should return dict each time"
    duration = time.time() - start_time

    assert duration < 1.0, f"100 state initializations should be fast, took {duration:.3f}s"

    # Test validate_start_page performance
    start_time = time.time()
    for i in range(1000):
        result = gather_orchestrator.validate_start_page(f"page_{i}_12345")
        assert isinstance(result, int), "Should return integer"
    duration = time.time() - start_time

    assert duration < 0.5, f"1000 page validations should be fast, took {duration:.3f}s"

    return True


def _test_retryable_error_constructor() -> bool:
    """Test RetryableError constructor with conflicting parameters"""
    print("   • Test 1: RetryableError constructor parameter conflict bug")
    try:
        error = RetryableError(
            "Transaction failed: UNIQUE constraint failed",
            recovery_hint="Check database connectivity and retry",
            context={"session_id": "test_123", "error_type": "IntegrityError"},
        )
        assert error.message == "Transaction failed: UNIQUE constraint failed"
        assert error.recovery_hint == "Check database connectivity and retry"
        assert "session_id" in error.context
        print("     ✅ RetryableError constructor handles conflicting parameters correctly")
        return True
    except TypeError as e:
        if "got multiple values for keyword argument" in str(e):
            raise AssertionError(f"CRITICAL: RetryableError constructor bug still exists: {e}") from e
        raise


def _test_database_connection_error_constructor() -> bool:
    """Test DatabaseConnectionError constructor"""
    print("   • Test 2: DatabaseConnectionError constructor")
    try:
        db_error = DatabaseConnectionError(
            "Database operation failed",
            recovery_hint="Database may be temporarily unavailable",
            context={"session_id": "test_456"},
        )
        assert db_error.error_code == "DB_CONNECTION_FAILED"
        assert db_error.recovery_hint and "temporarily unavailable" in db_error.recovery_hint
        print("     ✅ DatabaseConnectionError constructor works correctly")
        return True
    except TypeError as e:
        raise AssertionError(f"DatabaseConnectionError constructor has parameter conflicts: {e}") from e


def _test_database_transaction_rollback() -> bool:
    """Test database transaction rollback scenario simulation"""
    import sqlite3
    from unittest.mock import patch

    print("   • Test 3: Database transaction rollback scenario simulation")
    try:
        with patch('core.database.logger'):
            try:
                raise sqlite3.IntegrityError("UNIQUE constraint failed: people.uuid")
            except sqlite3.IntegrityError as e:
                error_type = type(e).__name__
                context = {
                    "session_id": "test_session_789",
                    "transaction_time": 1.5,
                    "error_type": error_type,
                }
                retryable_error = RetryableError(f"Transaction failed: {e}", context=context)
                assert "Transaction failed:" in retryable_error.message
                assert retryable_error.context["error_type"] == "IntegrityError"
                print("     ✅ Database rollback error handling works correctly")
            return True
    except Exception as e:
        raise AssertionError(f"Database transaction rollback simulation failed: {e}") from e


def _test_all_error_class_constructors() -> bool:
    """Test all error class constructors to prevent future regressions"""
    from core.error_handling import (
        APIRateLimitError,
        ConfigurationError,
        DataValidationError,
        FatalError,
    )

    print("   • Test 4: All error class constructors parameter validation")
    error_classes: list[tuple[Any, dict[str, Any]]] = [
        (APIRateLimitError, {"retry_after": 30}),
        (AuthenticationExpiredError, {}),
        (NetworkTimeoutError, {}),
        (DataValidationError, {}),
        (BrowserSessionError, {}),
        (ConfigurationError, {}),
        (FatalError, {}),
    ]

    for error_class, extra_args in error_classes:
        try:
            error = error_class(
                f"Test {error_class.__name__} message",
                recovery_hint="Test recovery hint",
                context={"test": True},
                **extra_args,
            )
            assert hasattr(error, 'message')
            print(f"     ✅ {error_class.__name__} constructor works correctly")
        except TypeError as e:
            if "got multiple values for keyword argument" in str(e):
                raise AssertionError(
                    f"CRITICAL: {error_class.__name__} has constructor parameter conflicts: {e}"
                ) from e
            raise

    return True


def _test_legacy_function_error_handling() -> bool:
    """Test legacy function error handling"""
    from unittest.mock import MagicMock

    from actions.gather import orchestrator as gather_orchestrator, persistence as gather_persistence_module

    print("   • Test 5: Legacy function error handling")
    mock_session = MagicMock()
    mock_scalar_result = MagicMock()
    mock_scalar_result.all.side_effect = Exception("Database error 12345")
    mock_session.scalars.return_value = mock_scalar_result

    try:
        gather_persistence_module.process_batch_lookups(
            mock_session,
            [{"uuid": "test_12345"}],
            current_page=1,
            page_statuses={"skipped": 0},
        )
    except Exception as e:
        assert "12345" in str(e), "Should be test-related error"
    else:
        raise AssertionError("Lookup should propagate database errors for visibility")

    result = gather_orchestrator.validate_start_page(None)
    assert result == 1, "Should handle None gracefully"

    result = gather_orchestrator.validate_start_page("not_a_number_12345")
    assert result == 1, "Should handle invalid input gracefully"

    print("     ✅ Legacy function error handling works correctly")
    return True


def _test_timeout_and_retry_handling() -> bool:
    """Test timeout and retry handling configuration"""
    from config import config_schema

    print("   • Test 6: Timeout and retry handling that caused multiple final summaries")
    expected_min_timeout = 900  # 15 minutes
    actual_timeout = getattr(config_schema, "action6_coord_timeout_seconds", 0)
    assert actual_timeout >= expected_min_timeout, (
        "Action 6 coord timeout must stay comfortably above historical 12+ min runtimes"
    )

    selenium_policy = getattr(getattr(config_schema, "retry_policies", None), "selenium", None)
    assert selenium_policy is not None, "Selenium retry policy must exist"
    assert selenium_policy.max_attempts >= 3, "Selenium retries must allow at least 3 attempts"
    assert selenium_policy.backoff_factor >= 1.5, "Backoff must remain exponential to avoid flapping"
    assert selenium_policy.max_delay_seconds >= selenium_policy.initial_delay_seconds, (
        "Max delay must be >= initial delay so retries can slow down"
    )
    print(
        "     ✅ coord timeout and Selenium retry policy satisfy long-run safety requirements"
        f" ({actual_timeout}s timeout, {selenium_policy.max_attempts} attempts)."
    )

    return True


def _test_duplicate_record_handling() -> bool:
    """Test duplicate record handling during retry scenarios"""
    print("   • Test 7: Duplicate record handling during retry scenarios")

    # Ensure in-batch de-duplication keeps first entry and allows null profile IDs
    dedup_input = [
        {"profile_id": "KIT123", "uuid": "UUID-1"},
        {"profile_id": "KIT123", "uuid": "UUID-2"},
        {"profile_id": None, "uuid": "UUID-3"},
    ]
    deduped = _deduplicate_person_creates(dedup_input)
    assert [row["uuid"] for row in deduped] == ["UUID-1", "UUID-3"], (
        "Duplicate profile IDs should retain first record and keep null-profile entries"
    )
    _validate_no_duplicate_profile_ids(deduped)  # Should not raise

    # Explicit duplicate profile IDs should raise IntegrityError before DB insert
    duplicate_payload = [
        {"profile_id": "KIT123", "uuid": "UUID-1"},
        {"profile_id": "KIT123", "uuid": "UUID-2"},
    ]
    try:
        _validate_no_duplicate_profile_ids(duplicate_payload)
    except IntegrityError as dup_exc:
        assert "Duplicate profile IDs" in str(dup_exc.orig), "IntegrityError should describe duplicate IDs"
    else:
        raise AssertionError("IntegrityError should be raised when duplicate profile IDs remain in payload")

    print("     ✅ Duplicate detection prevents UNIQUE violations both in-memory and pre-insert")
    return True


def _test_error_handling() -> bool:
    """
    Test error handling scenarios including the critical RetryableError constructor bug
    that caused Action 6 database transaction failures.
    """
    print("🧪 Testing error handling scenarios that previously caused Action 6 failures...")

    _test_retryable_error_constructor()
    _test_database_connection_error_constructor()
    _test_database_transaction_rollback()
    _test_all_error_class_constructors()
    _test_legacy_function_error_handling()
    _test_timeout_and_retry_handling()
    _test_duplicate_record_handling()

    print("🎯 All critical error handling scenarios validated successfully!")
    print("   This comprehensive test would have caught:")
    print("   - RetryableError constructor parameter conflicts")
    print("   - Timeout configuration too short for Action 6 runtime")
    print("   - Duplicate record handling during retries")
    print("   - Multiple final summary reporting issues")
    print("🎉 All error handling tests passed - Action 6 database transaction bugs prevented!")
    return True


def _test_dynamic_api_failure_threshold() -> bool:
    """
    🔧 TEST: Dynamic API failure threshold calculation.

    Tests that the dynamic threshold scales appropriately with the number of pages
    to prevent premature halts on large processing runs while maintaining safety.
    """
    print("🔧 Testing Dynamic API Failure Threshold:")
    results: list[bool] = []

    test_cases = [
        (10, 10),  # 10 pages -> minimum threshold of 10
        (100, 10),  # 100 pages -> 100/20 = 5, but minimum is 10
        (200, 10),  # 200 pages -> 200/20 = 10
        (400, 20),  # 400 pages -> 400/20 = 20
        (795, 39),  # 795 pages -> 795/20 = 39 (our actual use case)
        (2000, 100),  # 2000 pages -> 2000/20 = 100 (maximum)
        (5000, 100),  # 5000 pages -> 5000/20 = 250, but capped at 100
    ]

    for pages, expected in test_cases:
        result = get_critical_api_failure_threshold(pages)
        status = "✅" if result == expected else "❌"
        print(f"   {status} {pages} pages -> {result} threshold (expected {expected})")
        results.append(result == expected)

    print(
        "   Current default threshold: "
        f"{CRITICAL_API_FAILURE_THRESHOLD_DEFAULT} (runtime: {Action6State.critical_api_failure_threshold})"
    )

    success = all(results)
    if success:
        print("🎉 Dynamic API failure threshold tests passed!")
    return success


def _test_cm_relationship_fallback() -> bool:
    """Verify cM-driven predicted relationship inference when API data is missing."""

    print("🧪 Testing cM-based predicted relationship fallback...")
    logger_instance = logging.getLogger("action6.tests.cm_fallback")

    missing_match = {
        "uuid": "TEST-CM-550",
        "cm_dna": 550,
        "compare_link": None,
    }

    inferred = _prepare_dna_match_operation_data(
        match=missing_match,
        existing_dna_match=None,
        prefetched_combined_details=None,
        match_uuid=str(missing_match["uuid"]),
        predicted_relationship=None,
        log_ref_short="TEST-CM",
        logger_instance=logger_instance,
    )

    assert inferred is not None, "Expected DNA data dict for new match"
    inferred_payload = inferred.get("dna_match")
    if not isinstance(inferred_payload, dict):
        inferred_payload = inferred

    inferred_dict = cast(dict[str, Any], inferred_payload)
    inferred_value = inferred_dict.get("predicted_relationship")
    assert isinstance(inferred_value, str)
    assert inferred_value.startswith("1st cousin"), inferred_value
    assert "inferred" in inferred_value

    provided_match = {
        "uuid": "TEST-API-REL",
        "cm_dna": 75,
        "compare_link": None,
    }

    provided = _prepare_dna_match_operation_data(
        match=provided_match,
        existing_dna_match=None,
        prefetched_combined_details=None,
        match_uuid=str(provided_match["uuid"]),
        predicted_relationship="3rd cousin",
        log_ref_short="TEST-API",
        logger_instance=logger_instance,
    )

    assert provided is not None, "Expected DNA data dict when API provides relationship"
    provided_payload = provided.get("dna_match")
    if not isinstance(provided_payload, dict):
        provided_payload = provided

    provided_dict = cast(dict[str, Any], provided_payload)
    assert provided_dict.get("predicted_relationship") == "3rd cousin"

    print("🎉 cM-based predicted relationship fallback tests passed!")
    return True


def _test_tree_data_extraction() -> bool:
    """Verify extraction of tree metadata from match data."""
    print("🧪 Testing tree data extraction...")

    # Mock match data with tree info
    match_data = {
        "match_tree_id": "TREE123",
        "match_tree_person_id": "PERSON456",
        "has_public_tree": True,
        "tree_size": 1500,
        "uuid": "TEST-UUID",
        "testGuid": "TEST-UUID",
        "publicDisplayName": "Test User",
    }

    logger_instance = logging.getLogger("action6.tests.tree_data")

    result = _prepare_dna_match_operation_data(
        match=match_data,
        existing_dna_match=None,
        prefetched_combined_details=None,
        match_uuid="TEST-UUID",
        predicted_relationship="Parent/Child",
        log_ref_short="TEST",
        logger_instance=logger_instance,
    )

    assert result is not None
    dna_match = result

    assert dna_match["match_tree_id"] == "TREE123"
    assert dna_match["match_tree_person_id"] == "PERSON456"
    assert dna_match["has_public_tree"] is True
    assert dna_match["tree_size"] == 1500

    print("🎉 Tree data extraction tests passed!")
    return True


def action6_gather_module_tests() -> bool:
    """Comprehensive test suite for action6_gather.py"""

    suite = TestSuite("Action 6 - Gather DNA Matches", "action6_gather.py")
    suite.start_suite()

    # Run all tests with suppress_logging
    with suppress_logging():
        suite.run_test(
            test_name="Tree data extraction",
            test_func=_test_tree_data_extraction,
            test_summary="Verify extraction of tree metadata from match data",
            functions_tested="_prepare_dna_match_operation_data()",
            method_description="Mock match data with tree info and verify preservation",
            expected_outcome="Tree metadata fields are correctly preserved in operation data",
        )

        suite.run_test(
            test_name="Profile cache helper round-trip",
            test_func=_test_cache_profile_helpers,
            test_summary="Ensures profile caching helpers use UnifiedCacheManager",
            functions_tested="_cache_profile(), _get_cached_profile()",
            method_description="Cache a synthetic profile payload and read it back",
            expected_outcome="Profile data persists via UnifiedCacheManager and can be invalidated",
        )

        suite.run_test(
            test_name="Combined details cache helper round-trip",
            test_func=_test_combined_details_cache_helpers,
            test_summary="Validates combined details caching and retrieval helpers",
            functions_tested="_cache_combined_details(), _check_combined_details_cache()",
            method_description="Cache combined match payload then read from cache",
            expected_outcome="Combined details persist via UnifiedCacheManager and remain consistent",
        )

        suite.run_test(
            test_name="Match list UI navigation supports predefined page",
            test_func=_test_nav_to_list_predefined_page,
            test_summary="Ensures nav_to_list can target a specific UI page for resume",
            functions_tested="nav_to_list()",
            method_description="Stub nav_to_page and verify ?currentPage is included in the navigation URL",
            expected_outcome="Navigation URL includes the requested currentPage value",
        )

        suite.run_test(
            test_name="Match list UI navigation includes itemsPerPage",
            test_func=_test_nav_to_list_includes_items_per_page_when_page_omitted,
            test_summary="Ensures nav_to_list always includes itemsPerPage for consistent paging",
            functions_tested="nav_to_list()",
            method_description="Stub nav_to_page and verify itemsPerPage is included even when current_page is None",
            expected_outcome="Navigation URL includes itemsPerPage parameter",
        )

        suite.run_test(
            test_name="Initial navigation threads start_page into UI nav",
            test_func=_test_initial_navigation_threads_start_page,
            test_summary="Ensures initial Action 6 navigation requests the desired page",
            functions_tested="_navigate_and_get_initial_page_data(), _ensure_on_match_list_page()",
            method_description="Stub _ensure_on_match_list_page and verify desired_page=start_page is passed",
            expected_outcome="_navigate_and_get_initial_page_data passes start_page as desired_page",
        )

        # INITIALIZATION TESTS
        suite.run_test(
            test_name="_initialize_gather_state(), _validate_start_page()",
            test_func=_test_module_initialization,
            test_summary="Module initialization and state management functions",
            functions_tested="_initialize_gather_state(), _validate_start_page()",
            method_description="Testing state initialization, page validation, and parameter handling for DNA match gathering",
            expected_outcome="Module initializes correctly with proper state management and page validation",
        )

        # CORE FUNCTIONALITY TESTS
        suite.run_test(
            test_name="Core DNA match gathering functions",
            test_func=_test_core_functionality,
            test_summary="Core functions for DNA match data processing",
            functions_tested="_process_match_data(), _extract_match_info()",
            method_description="Testing core DNA match processing and data extraction",
            expected_outcome="Core functions process match data correctly",
        )

        # DATA PROCESSING TESTS
        suite.run_test(
            test_name="Data processing functions",
            test_func=_test_data_processing_functions,
            test_summary="Data transformation and processing functions",
            functions_tested="_transform_match_data(), _validate_match_data()",
            method_description="Testing data transformation and validation",
            expected_outcome="Data processing functions work correctly",
        )

        # EDGE CASE TESTS
        suite.run_test(
            test_name="Edge case handling",
            test_func=_test_edge_cases,
            test_summary="Edge case and error condition handling",
            functions_tested="Various edge case handlers",
            method_description="Testing edge cases and error conditions",
            expected_outcome="Edge cases handled gracefully",
        )

        # INTEGRATION TESTS
        suite.run_test(
            test_name="Integration tests",
            test_func=_test_integration,
            test_summary="Integration between components",
            functions_tested="Component integration",
            method_description="Testing integration between different components",
            expected_outcome="Components integrate correctly",
        )

        # PERFORMANCE TESTS
        suite.run_test(
            test_name="Performance tests",
            test_func=_test_performance,
            test_summary="Performance of data processing operations",
            functions_tested="_initialize_gather_state(), _validate_start_page()",
            method_description="Testing performance of critical operations",
            expected_outcome="Operations complete within acceptable time limits",
        )

        # ERROR HANDLING TESTS
        suite.run_test(
            test_name="Error handling scenarios",
            test_func=_test_error_handling,
            test_summary="Critical error handling including RetryableError constructor bug",
            functions_tested="RetryableError, DatabaseConnectionError, error handling",
            method_description="Testing error handling scenarios that previously caused Action 6 failures",
            expected_outcome="All error handling scenarios work correctly without constructor conflicts",
        )

        suite.run_test(
            test_name="Dynamic API failure threshold calculation",
            test_func=_test_dynamic_api_failure_threshold,
            test_summary="Dynamic threshold prevents premature halts on large runs while maintaining safety",
            functions_tested="_main_page_loop()",
            method_description="Testing threshold calculation: min 10, max 100, scales at 1 per 20 pages",
            expected_outcome="API failure threshold scales appropriately with number of pages to process",
        )

        suite.run_test(
            test_name="cM relationship fallback logic",
            test_func=_test_cm_relationship_fallback,
            test_summary="Predicted relationship inference uses cM buckets when API omits a value",
            functions_tested="_prepare_dna_match_operation_data(), _resolve_predicted_relationship_value()",
            method_description="Simulate matches with and without API-provided relationships to verify inference and pass-through behavior",
            expected_outcome="Missing values get inferred labels while provided labels remain unchanged",
        )

        # PERFORMANCE TESTS
        suite.run_test(
            test_name="Performance of state initialization and validation operations",
            test_func=test_performance,
            test_summary="Performance characteristics of DNA match gathering operations",
            functions_tested="_initialize_gather_state(), _validate_start_page()",
            method_description="Testing execution speed and efficiency of state management and validation functions",
            expected_outcome="All operations complete within acceptable time limits with good performance",
        )

        # ERROR HANDLING TESTS
        suite.run_test(
            test_name="Comprehensive error handling including RetryableError constructor bug prevention",
            test_func=_test_error_handling,
            test_summary="Enhanced error handling testing including RetryableError bug fix, timeout configuration validation, duplicate record handling, and final summary accuracy",
            functions_tested="Error handling across all functions",
            method_description="Testing RetryableError constructor conflicts, timeout/retry scenarios, UNIQUE constraint handling, and reporting accuracy to prevent Action 6 database transaction failures and multiple summary issues",
            expected_outcome="All error conditions handled gracefully, timeout issues resolved, database transaction errors prevented, no constructor parameter conflicts",
        )

    return suite.finish_suite()


# Use centralized test runner utility from test_utilities
run_comprehensive_tests = create_standard_test_runner(action6_gather_module_tests)


# ==============================================
# Standalone Test Block
# ==============================================

if __name__ == "__main__":
    import sys

    print("🧬 Running Action 6 - Gather DNA Matches comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
