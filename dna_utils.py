#!/usr/bin/env python3

"""
dna_utils.py - Universal DNA Match Utilities

Provides universal functions for DNA match operations that can be used
across all scripts (action6, action6b, and future DNA-related actions).

Functions:
- nav_to_dna_matches_page() - Navigate to DNA matches page
- get_csrf_token_for_dna_matches() - Get CSRF token from browser cookies
- fetch_in_tree_status() - Fetch in-tree status for sample IDs
- fetch_match_list_page() - Fetch match list from API
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import contextlib
import random
from typing import Any, Optional
from urllib.parse import unquote, urljoin, urlparse

# === THIRD-PARTY IMPORTS ===
import requests
from diskcache.core import ENOVAL
from selenium.common.exceptions import NoSuchCookieException, WebDriverException

# === LOCAL IMPORTS ===
from cache import cache as global_cache
from config import config_schema
from core.session_manager import SessionManager
from my_selectors import MATCH_ENTRY_SELECTOR
from selenium_utils import get_driver_cookies
from utils import _api_req, nav_to_page

# ============================================================================
# NAVIGATION FUNCTIONS
# ============================================================================

def nav_to_dna_matches_page(session_manager: SessionManager) -> bool:
    """
    Navigate to the user's DNA matches list page.

    Args:
        session_manager: Active SessionManager instance

    Returns:
        True if navigation successful, False otherwise
    """
    if (
        not session_manager
        or not session_manager.is_sess_valid()
        or not session_manager.my_uuid
    ):
        logger.error("nav_to_dna_matches_page: Session invalid or UUID missing.")
        return False

    my_uuid = session_manager.my_uuid
    target_url = urljoin(
        config_schema.api.base_url, f"discoveryui-matches/list/{my_uuid}"
    )
    logger.debug(f"Navigating to DNA matches page: {target_url}")

    driver = session_manager.driver
    if driver is None:
        logger.error("nav_to_dna_matches_page: WebDriver is None")
        return False

    success = nav_to_page(
        driver=driver,
        url=target_url,
        selector=MATCH_ENTRY_SELECTOR,  # type: ignore
        session_manager=session_manager,
    )

    if success:
        try:
            current_url = driver.current_url
            if not current_url.startswith(target_url):
                logger.warning(
                    f"Navigation successful (element found), but final URL unexpected: {current_url}"
                )
            else:
                logger.debug("Successfully navigated to DNA matches page.")
        except Exception as e:
            logger.warning(f"Could not verify final URL after navigation: {e}")
    else:
        logger.error("Failed to navigate to DNA matches page.")

    return success


# ============================================================================
# CSRF TOKEN FUNCTIONS
# ============================================================================

def _try_get_csrf_from_driver_cookies(
    driver: Any,
    cookie_names: tuple[str, ...]
) -> Optional[str]:
    """
    Fallback method to get CSRF token using get_driver_cookies.

    Returns:
        CSRF token if found, None otherwise
    """
    logger.debug(
        "CSRF token not found via get_cookie. Trying get_driver_cookies fallback..."
    )
    all_cookies = get_driver_cookies(driver)
    if not all_cookies:
        logger.warning(
            "Fallback get_driver_cookies also failed to retrieve cookies."
        )
        return None

    for cookie_name in cookie_names:
        for cookie in all_cookies:
            if cookie.get("name") == cookie_name and cookie.get("value"):
                token = unquote(cookie["value"]).split("|")[0]
                logger.debug(
                    f"Read CSRF token via fallback from '{cookie_name}'."
                )
                return token

    return None


def get_csrf_token_for_dna_matches(driver: Any) -> Optional[str]:
    """
    Retrieve CSRF token from browser cookies for DNA match list API.

    Args:
        driver: Selenium WebDriver instance

    Returns:
        CSRF token string if found, None otherwise
    """
    csrf_token_cookie_names = (
        "_dnamatches-matchlistui-x-csrf-token",
        "_csrf",
    )
    specific_csrf_token: Optional[str] = None

    try:
        logger.debug(f"Attempting to read CSRF cookies: {csrf_token_cookie_names}")

        # Try direct cookie access first
        for cookie_name in csrf_token_cookie_names:
            try:
                cookie_obj = driver.get_cookie(cookie_name)
                if cookie_obj and "value" in cookie_obj and cookie_obj["value"]:
                    specific_csrf_token = unquote(cookie_obj["value"]).split("|")[0]
                    logger.debug(f"Read CSRF token from cookie '{cookie_name}'.")
                    break
            except NoSuchCookieException:
                continue
            except WebDriverException as cookie_e:
                logger.warning(
                    f"WebDriver error getting cookie '{cookie_name}': {cookie_e}"
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error getting cookie '{cookie_name}': {e}",
                    exc_info=True,
                )

        # Fallback to get_driver_cookies if direct access failed
        if not specific_csrf_token:
            specific_csrf_token = _try_get_csrf_from_driver_cookies(
                driver, csrf_token_cookie_names
            )

        if not specific_csrf_token:
            logger.error(
                "Failed to obtain specific CSRF token required for Match List API."
            )
            return None

        logger.debug(f"Specific CSRF token FOUND: '{specific_csrf_token}'")
        return specific_csrf_token

    except Exception as csrf_err:
        logger.error(
            f"Critical error during CSRF token retrieval: {csrf_err}", exc_info=True
        )
        return None


# ============================================================================
# IN-TREE STATUS FUNCTIONS
# ============================================================================

def _try_get_in_tree_from_cache(
    cache_key: str,
    current_page: int,
) -> Optional[set[str]]:
    """
    Try to get in-tree status from cache.

    Returns:
        Set of in-tree IDs if found in cache, None otherwise
    """
    try:
        if global_cache is not None:
            cached_in_tree = global_cache.get(cache_key, default=ENOVAL, retry=True)
            if cached_in_tree is not ENOVAL:
                if isinstance(cached_in_tree, set):
                    logger.debug(
                        f"Loaded {len(cached_in_tree)} in-tree IDs from cache for page {current_page}."
                    )
                    return cached_in_tree
            else:
                logger.debug(
                    f"Cache miss for in-tree status (Key: {cache_key}). Fetching from API."
                )
    except Exception as cache_read_err:
        logger.error(
            f"Error reading in-tree status from cache: {cache_read_err}. Fetching from API.",
            exc_info=True,
        )
    return None


def _save_in_tree_to_cache(
    cache_key: str,
    in_tree_ids: set[str],
    current_page: int,
) -> None:
    """Save in-tree status to cache."""
    try:
        if global_cache is not None:
            global_cache.set(
                cache_key,
                in_tree_ids,
                expire=config_schema.cache.memory_cache_ttl,
                retry=True,
            )
        logger.debug(f"Cached in-tree status result for page {current_page}.")
    except Exception as cache_write_err:
        logger.error(f"Error writing in-tree status to cache: {cache_write_err}")


def _fetch_in_tree_from_api(
    driver: Any,
    session_manager: SessionManager,
    my_uuid: str,
    sample_ids_on_page: list[str],
    specific_csrf_token: str,
    current_page: int,
) -> set[str]:
    """
    Fetch in-tree status from API.

    Returns:
        Set of sample IDs that are in the user's tree
    """
    in_tree_url = urljoin(
        config_schema.api.base_url,
        f"discoveryui-matches/parents/list/api/badges/matchesInTree/{my_uuid.upper()}",
    )
    parsed_base_url = urlparse(config_schema.api.base_url)
    origin_header_value = f"{parsed_base_url.scheme}://{parsed_base_url.netloc}"

    ua_in_tree = None
    if driver and session_manager.is_sess_valid():
        with contextlib.suppress(Exception):
            ua_in_tree = driver.execute_script("return navigator.userAgent;")
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

    logger.debug(
        f"Fetching in-tree status for {len(sample_ids_on_page)} matches on page {current_page}..."
    )

    response_in_tree = _api_req(
        url=in_tree_url,
        driver=driver,
        session_manager=session_manager,
        method="POST",
        json_data={"sampleIds": sample_ids_on_page},
        headers=in_tree_headers,
        use_csrf_token=False,
        api_description="In-Tree Status Check",
    )

    if isinstance(response_in_tree, list):
        in_tree_ids = {item.upper() for item in response_in_tree if isinstance(item, str)}
        logger.debug(f"Fetched {len(in_tree_ids)} in-tree IDs from API for page {current_page}.")
        return in_tree_ids

    status_code_log = (
        f" Status: {response_in_tree.status_code}"  # type: ignore
        if isinstance(response_in_tree, requests.Response)
        else ""
    )
    logger.warning(
        f"In-Tree Status Check API failed or returned unexpected format for page {current_page}.{status_code_log}"
    )
    return set()


def fetch_in_tree_status(
    driver: Any,
    session_manager: SessionManager,
    my_uuid: str,
    sample_ids_on_page: list[str],
    specific_csrf_token: str,
    current_page: int,
) -> set[str]:
    """
    Fetch in-tree status for a list of sample IDs, with caching.

    Args:
        driver: Selenium WebDriver instance
        session_manager: Active SessionManager instance
        my_uuid: User's UUID
        sample_ids_on_page: List of sample IDs to check
        specific_csrf_token: CSRF token for API request
        current_page: Current page number (for logging)

    Returns:
        Set of sample IDs that are in the user's tree
    """
    cache_key_tree = f"matches_in_tree_{hash(frozenset(sample_ids_on_page))}"

    # Try to get from cache first
    cached_result = _try_get_in_tree_from_cache(cache_key_tree, current_page)
    if cached_result is not None:
        return cached_result

    # Fetch from API if cache miss or error
    if not session_manager.is_sess_valid():
        logger.error(
            f"In-Tree Status Check: Session invalid page {current_page}. Cannot fetch."
        )
        return set()

    in_tree_ids = _fetch_in_tree_from_api(
        driver, session_manager, my_uuid, sample_ids_on_page, specific_csrf_token, current_page
    )

    # Cache the result if we got data
    if in_tree_ids:
        _save_in_tree_to_cache(cache_key_tree, in_tree_ids, current_page)

    return in_tree_ids


# ============================================================================
# MATCH LIST API FUNCTIONS
# ============================================================================

def _sync_cookies_to_session(driver: Any, session_manager: SessionManager) -> None:
    """Sync browser cookies to requests session before API call."""
    try:
        logger.debug("Syncing browser cookies to API session before Match List API call...")
        browser_cookies = driver.get_cookies()
        logger.debug(f"Retrieved {len(browser_cookies)} cookies from browser")

        # Clear and re-sync all cookies to ensure fresh state
        if hasattr(session_manager, 'requests_session') and session_manager.requests_session:
            session_manager.requests_session.cookies.clear()
            for cookie in browser_cookies:
                session_manager.requests_session.cookies.set(
                    cookie['name'],
                    cookie['value'],
                    domain=cookie.get('domain', ''),
                    path=cookie.get('path', '/')
                )
            logger.debug(f"Synced {len(browser_cookies)} cookies to requests session")
        else:
            logger.warning("No requests session available for cookie sync")
    except Exception as cookie_sync_error:
        logger.error(f"Cookie sync failed: {cookie_sync_error}")


def fetch_match_list_page(
    driver: Any,
    session_manager: SessionManager,
    my_uuid: str,
    current_page: int,
    csrf_token: str
) -> Optional[dict[str, Any]]:
    """
    Fetch match list data for a specific page from the API.

    Args:
        driver: Selenium WebDriver instance
        session_manager: Active SessionManager instance
        my_uuid: User's UUID
        current_page: Page number to fetch
        csrf_token: CSRF token for API request

    Returns:
        API response dict if successful, None otherwise
    """
    # Build API URL
    match_list_url = urljoin(
        config_schema.api.base_url,
        f"discoveryui-matches/parents/list/api/matchList/{my_uuid}?currentPage={current_page}",
    )

    # Build headers
    match_list_headers = {
        "X-CSRF-Token": csrf_token,
        "Accept": "application/json",
        "Referer": urljoin(config_schema.api.base_url, "/discoveryui-matches/list/"),
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "priority": "u=1, i",
    }

    logger.debug(f"Calling Match List API for page {current_page}...")
    logger.debug(f"Headers being passed to _api_req for Match List: {match_list_headers}")

    # Sync cookies before API call
    _sync_cookies_to_session(driver, session_manager)

    # Call the API
    return _api_req(
        url=match_list_url,
        driver=driver,
        session_manager=session_manager,
        method="GET",
        headers=match_list_headers,
        use_csrf_token=False,
        api_description="Match List API",
        allow_redirects=True,
    )

