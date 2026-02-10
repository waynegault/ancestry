#!/usr/bin/env python3

"""Cookie management utilities for parsing, persistence, synchronization, and UBE header generation."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from selenium.common.exceptions import NoSuchCookieException, WebDriverException
from selenium.webdriver.remote.webdriver import WebDriver

from browser.selenium_utils import DriverProtocol

if TYPE_CHECKING:
    from core.session_manager import SessionManager

# === MODULE SETUP ===
logger = logging.getLogger(__name__)

# Type alias matching core/utils.py
DriverType = WebDriver | None


# ------------------------------------------------------------------------------------
# Cookie Parsing
# ------------------------------------------------------------------------------------


def parse_cookie(cookie_string: str) -> dict[str, str]:
    """
    Parses a raw HTTP cookie string into a dictionary of key-value pairs.
    Handles empty keys and values.
    """
    cookies: dict[str, str] = {}
    parts = cookie_string.split(";")
    for raw_part in parts:
        part = raw_part.strip()
        if not part:
            continue
        # End of if
        if "=" in part:
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            cookies[key] = value
        else:
            logger.debug(f"Skipping cookie part without '=': '{part}'")
        # End of if/else
    # End of for
    return cookies


# End of parse_cookie


# ------------------------------------------------------------------------------------
# Cookie Persistence Functions
# ------------------------------------------------------------------------------------


def _get_cookie_file_path() -> Path:
    """Get the path to the cookie file."""
    return Path("ancestry_cookies.json")


def _save_login_cookies(session_manager: SessionManager) -> bool:
    """Save login cookies to file for session persistence."""
    try:
        driver = session_manager.browser_manager.driver
        if driver is None:
            logger.debug("Cannot save cookies: No driver available")
            return False

        # Cast to Protocol to ensure types
        driver_proto = cast(DriverProtocol, driver)
        cookies = driver_proto.get_cookies()
        if not cookies:
            logger.debug("No cookies to save")
            return False

        cookies_file = _get_cookie_file_path()

        with cookies_file.open("w", encoding="utf-8") as f:
            json.dump(cookies, f, indent=2)

        logger.info("💾 Saved %d cookies to %s", len(cookies), cookies_file)
        return True

    except Exception as e:
        logger.warning("Failed to save cookies: %s", e)
        return False


def _load_login_cookies(session_manager: SessionManager) -> bool:
    """Load saved login cookies from file."""
    try:
        driver = session_manager.browser_manager.driver
        if driver is None:
            logger.debug("Cannot load cookies: No driver available")
            return False

        # Cast to Protocol to ensure types
        driver_proto = cast(DriverProtocol, driver)

        cookies_file = _get_cookie_file_path()

        if not cookies_file.exists():
            logger.debug(f"No saved cookies file found at: {cookies_file}")
            return False

        with cookies_file.open(encoding="utf-8") as f:
            cookies: list[dict[str, Any]] = json.load(f)

        if not cookies:
            logger.debug("No cookies in saved file")
            return False

        # Add each cookie to the driver
        loaded_count = 0
        for cookie in cookies:
            try:
                # Remove expiry if it's in the past (causes Selenium errors)
                if "expiry" in cookie:
                    if cookie["expiry"] < time.time():
                        del cookie["expiry"]

                # Cast cookie values to object to match Protocol
                cookie_obj: dict[str, object] = cast(dict[str, object], cookie)
                driver_proto.add_cookie(cookie_obj)
                loaded_count += 1
            except Exception as cookie_err:
                logger.debug(f"Failed to add cookie {cookie.get('name', 'unknown')}: {cookie_err}")
                continue

        logger.debug(f"🍪 Loaded {loaded_count}/{len(cookies)} cookies from {cookies_file}")
        return loaded_count > 0

    except Exception as e:
        logger.warning("Failed to load cookies: %s", e)
        return False


def load_login_cookies(session_manager: SessionManager) -> bool:
    """Public alias so static analysers see the loader being referenced."""
    return _load_login_cookies(session_manager)


# End of cookie persistence functions


# ------------------------------------------------------------------------------------
# Cookie Sync Functions
# ------------------------------------------------------------------------------------

# Cookie cache to reduce excessive synchronization
_cookie_sync_cache: dict[str, float] = {"last_sync_time": 0.0, "sync_interval": 30.0}  # Sync every 30 seconds max


def _should_skip_cookie_sync(force_sync: bool, time_since_last_sync: float) -> bool:
    """
    Determine if cookie sync can be skipped based on cache.

    Args:
        force_sync: Whether sync is forced
        time_since_last_sync: Time since last sync in seconds

    Returns:
        bool: True if sync can be skipped, False otherwise
    """
    return not force_sync and time_since_last_sync < _cookie_sync_cache["sync_interval"]


def _validate_driver_for_sync(
    driver: DriverType, session_manager: SessionManager, api_description: str, attempt: int
) -> bool:
    """
    Validate that driver is available for cookie sync.

    Args:
        driver: The WebDriver instance
        session_manager: The session manager instance
        api_description: Description of the API being called
        attempt: The current attempt number

    Returns:
        bool: True if driver is valid, False otherwise
    """
    driver_is_valid = driver and session_manager.browser_manager.driver
    if not driver_is_valid and attempt == 1:
        logger.warning(
            f"[{api_description}] Browser session invalid or driver None (Attempt {attempt}). "
            "Runtime headers might be incomplete/stale."
        )
    return bool(driver_is_valid)


def _perform_cookie_sync(
    session_manager: SessionManager,
    api_description: str,
    attempt: int,
    force_sync: bool,
    time_since_last_sync: float,
    current_time: float,
) -> bool:
    """
    Perform the actual cookie synchronization.

    Args:
        session_manager: The session manager instance
        api_description: Description of the API being called
        attempt: The current attempt number
        force_sync: Whether sync is forced
        time_since_last_sync: Time since last sync in seconds
        current_time: Current timestamp

    Returns:
        bool: True if sync successful, False otherwise
    """
    # Only log on first attempt or if forced to reduce verbosity
    if attempt == 1 or force_sync:
        logger.debug(
            f"[{api_description}] Syncing cookies from browser "
            f"(cache expired, last sync {time_since_last_sync:.1f}s ago)"
        )

    # Prefer SessionManager.sync_browser_cookies()
    if hasattr(session_manager, "sync_browser_cookies"):
        try:
            session_manager.sync_browser_cookies()
            sync_success = True
        except Exception as e:
            logger.warning(f"[{api_description}] SessionManager cookie sync failed: {e}")
            sync_success = False
    else:
        # Fallback to legacy APIManager method
        api_manager = getattr(session_manager, "api_manager", None)
        browser_manager = getattr(session_manager, "browser_manager", None)
        sync_method = getattr(api_manager, "sync_cookies_from_browser", None)
        if not browser_manager or not callable(sync_method):
            logger.warning(f"[{api_description}] Cookie sync requested but browser/API managers are unavailable")
            return False

        sync_success = sync_method(browser_manager, session_manager=session_manager)

    if sync_success:
        # Update cache timestamp
        _cookie_sync_cache["last_sync_time"] = current_time
        if attempt == 1 or force_sync:
            logger.debug(f"[{api_description}] Cookie sync successful")
        return True

    logger.warning(f"[{api_description}] Cookie sync failed (Attempt {attempt}).")
    return False


def _sync_cookies_for_request(
    session_manager: SessionManager,
    driver: DriverType,
    api_description: str,
    attempt: int = 1,
    force_sync: bool = False,
) -> bool:
    """
    Synchronizes cookies from the WebDriver to the requests session.

    Uses caching to reduce excessive cookie synchronization - only syncs if:
    1. force_sync=True (e.g., after session recovery)
    2. More than 30 seconds since last sync
    3. First attempt of a request

    Args:
        session_manager: The session manager instance
        driver: The WebDriver instance
        api_description: Description of the API being called
        attempt: The current attempt number
        force_sync: Force cookie sync regardless of cache

    Returns:
        True if cookies were synced successfully, False otherwise
    """
    # Check if we can skip cookie sync (use cached cookies)
    current_time = time.time()
    time_since_last_sync = current_time - _cookie_sync_cache["last_sync_time"]

    if _should_skip_cookie_sync(force_sync, time_since_last_sync):
        return True

    # Validate driver
    if not _validate_driver_for_sync(driver, session_manager, api_description, attempt):
        return False

    # Perform cookie synchronization
    try:
        return _perform_cookie_sync(
            session_manager, api_description, attempt, force_sync, time_since_last_sync, current_time
        )
    except Exception as e:
        logger.error(f"[{api_description}] Exception during cookie sync (Attempt {attempt}): {e}", exc_info=True)
        return False


# End of _sync_cookies_for_request


# ------------------------------------------------------------------------------------
# UBE Header Generation
# ------------------------------------------------------------------------------------


def _validate_driver_session(driver: DriverType) -> bool:
    """Validate that driver session is active."""
    if not driver:
        return False
    try:
        _ = driver.title  # Quick check for session validity
        return True
    except WebDriverException as e:
        logger.warning(f"Cannot generate UBE header: Session invalid/unresponsive ({type(e).__name__}).")
        return False


def _extract_cookie_value(cookie_obj: Mapping[str, Any] | None) -> str | None:
    """Return cookie value when available."""
    if not isinstance(cookie_obj, Mapping):
        return None
    value = cookie_obj.get("value")
    return str(value) if value is not None else None


def _build_cookie_lookup(driver: WebDriver) -> dict[str, str]:
    """Create a name->value mapping for all cookies in the driver."""
    # Cast to Protocol to ensure types
    driver_proto = cast(DriverProtocol, driver)
    cookies_raw = driver_proto.get_cookies()
    cookies_dict: dict[str, str] = {}
    for cookie in cookies_raw:
        name = cookie.get("name")
        value = cookie.get("value")
        if isinstance(name, str) and value is not None:
            cookies_dict[name] = str(value)
    return cookies_dict


def _get_ancsessionid_cookie(driver: DriverType) -> str | None:
    """Get ANCSESSIONID cookie value from driver."""
    if not driver:
        return None
    try:
        # Cast to Protocol to ensure types
        driver_proto = cast(DriverProtocol, driver)
        direct_value = _extract_cookie_value(driver_proto.get_cookie("ANCSESSIONID"))
        if direct_value:
            return direct_value

        cookies_dict = _build_cookie_lookup(driver)
        ancsessionid = cookies_dict.get("ANCSESSIONID")
        if not ancsessionid:
            logger.warning("ANCSESSIONID cookie not found. Cannot generate UBE header.")
            return None
        return ancsessionid
    except (NoSuchCookieException, WebDriverException) as cookie_e:
        logger.warning(f"Error getting ANCSESSIONID cookie for UBE header: {cookie_e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting ANCSESSIONID for UBE: {e}", exc_info=True)
        return None


def _build_ube_payload(ancsessionid: str) -> dict[str, str]:
    """Build UBE data payload."""
    event_id = "00000000-0000-0000-0000-000000000000"
    correlated_id = str(uuid.uuid4())
    screen_name_standard = "ancestry : uk : en : dna-matches-ui : match-list : 1"
    screen_name_legacy = "ancestry uk : dnamatches-matchlistui : list"
    user_consent = (
        "necessary|preference|performance|analytics1st|analytics3rd|advertising1st|advertising3rd|attribution3rd"
    )

    return {
        "eventId": event_id,
        "correlatedScreenViewedId": correlated_id,
        "correlatedSessionId": ancsessionid,
        "screenNameStandard": screen_name_standard,
        "screenNameLegacy": screen_name_legacy,
        "userConsent": user_consent,
        "vendors": "adobemc",
        "vendorConfigurations": "{}",
    }


def _encode_ube_payload(ube_data: dict[str, str]) -> str | None:
    """Encode UBE payload to base64."""
    try:
        json_payload = json.dumps(ube_data, separators=(",", ":")).encode("utf-8")
        return base64.b64encode(json_payload).decode("utf-8")
    except (json.JSONDecodeError, TypeError, binascii.Error) as encode_e:
        logger.error(f"Error encoding UBE header data: {encode_e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error encoding UBE header: {e}", exc_info=True)
        return None


def make_ube(driver: DriverType) -> str | None:
    """Generate UBE header for Ancestry API requests."""
    if not _validate_driver_session(driver):
        return None

    ancsessionid = _get_ancsessionid_cookie(driver)
    if not ancsessionid:
        return None

    ube_data = _build_ube_payload(ancsessionid)
    return _encode_ube_payload(ube_data)


# ------------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------------


def _test_parse_cookie() -> None:
    """Test cookie parsing with various cookie string formats"""
    test_cases = [
        (
            "session_id=abc123; path=/; domain=.example.com",
            {"session_id": "abc123", "path": "/", "domain": ".example.com"},
            "Standard cookie string",
        ),
        ("", {}, "Empty string"),
        ("single=value", {"single": "value"}, "Single cookie"),
        ("a=1; b=2; c=3", {"a": "1", "b": "2", "c": "3"}, "Multiple cookies"),
        (
            "invalid_part; valid=test",
            {"valid": "test"},
            "Mixed valid/invalid parts",
        ),
    ]

    print("📋 Testing cookie parsing with various formats:")
    results: list[bool] = []

    for cookie_str, expected, description in test_cases:
        try:
            result = parse_cookie(cookie_str)
            matches_expected = result == expected

            status = "✅" if matches_expected else "❌"
            print(f"   {status} {description}")
            print(f"      Input: '{cookie_str}'")
            print(f"      Output: {result}")
            print(f"      Expected: {expected}")

            results.append(matches_expected)
            assert matches_expected, f"Should match expected result for '{cookie_str}'"

        except Exception as e:
            print(f"   ❌ {description}: Exception {e}")
            results.append(False)

    print(f"📊 Results: {sum(results)}/{len(results)} cookie parsing tests passed")


def module_tests() -> bool:
    """Run cookie_utils tests using standardized TestSuite format."""
    from testing.test_framework import TestSuite

    suite = TestSuite("Cookie Utilities", "cookie_utils.py")

    suite.run_test("Cookie Parsing", _test_parse_cookie, "Parse cookie strings with various formats")

    return suite.finish_suite()


from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(module_tests)

if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
