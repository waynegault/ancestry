"""Page navigation utilities with retry logic, URL validation, and error recovery."""

# === SESSION MANAGER IMPORT ===
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.session_manager import SessionManager

# === STANDARD LIBRARY IMPORTS ===
import contextlib
import logging
import random
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, cast
from urllib.parse import urljoin, urlparse, urlunparse

# === THIRD-PARTY IMPORTS ===
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    UnexpectedAlertPresentException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support.wait import WebDriverWait

# === LOCAL IMPORTS ===
from browser.css_selectors import (
    PAGE_NO_LONGER_AVAILABLE_SELECTOR,
    TEMP_UNAVAILABLE_SELECTOR,
    TWO_STEP_VERIFICATION_HEADER_SELECTOR,
    USERNAME_INPUT_SELECTOR,
    WAIT_FOR_PAGE_SELECTOR,
)
from browser.selenium_utils import DriverProtocol, is_browser_open, is_elem_there
from config import config_schema
from core.common_params import NavigationConfig
from core.selenium_utils import (
    wait_until_present as _wait_until_present,
    wait_until_visible as _wait_until_visible,
)
from testing.test_utilities import create_standard_test_runner

# === MODULE SETUP ===
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------
# Navigation Functions
# ------------------------------------------------------------------------------------


def _validate_nav_inputs(url: str) -> bool:
    """Validate navigation inputs."""
    if not url:
        logger.error(f"Navigation failed: Target URL '{url}' is invalid.")
        return False
    return True


def _parse_and_normalize_url(url: str) -> str | None:
    """Parse and normalize URL to base form."""
    try:
        target_url_parsed = urlparse(url)
        return urlunparse(
            (
                target_url_parsed.scheme,
                target_url_parsed.netloc,
                target_url_parsed.path.rstrip("/"),
                "",
                "",
                "",
            )
        ).rstrip("/")
    except ValueError as url_parse_err:
        logger.error(f"Failed to parse target URL '{url}': {url_parse_err}")
        return None


def _check_browser_session(
    driver: WebDriver,
    session_manager: Optional["SessionManager"],
    attempt: int,
) -> WebDriver | None:
    """Check browser session and restart if needed. Returns driver or None if failed."""
    if not is_browser_open(driver):
        logger.error(f"Navigation failed (Attempt {attempt}): Browser session invalid before nav.")
        if session_manager:
            logger.warning("Attempting session restart...")
            if session_manager.restart_sess():
                logger.info("Session restarted. Retrying navigation...")
                driver_instance = session_manager.browser_manager.driver
                if driver_instance is not None:
                    return driver_instance
                logger.error("Session restart reported success but driver is still None.")
                return None
            logger.error("Session restart failed. Cannot navigate.")
            return None
        logger.error("Session invalid and no SessionManager provided for restart.")
        return None
    return driver


def _document_ready(driver: WebDriver) -> bool:
    """Return True once the current document is fully loaded."""

    # Cast to Protocol to ensure types
    driver_proto = cast(DriverProtocol, driver)
    state = driver_proto.execute_script("return document.readyState")
    return isinstance(state, str) and state in {"complete", "interactive"}


def _execute_navigation(driver: WebDriver, url: str, page_timeout: int) -> None:
    """Execute navigation and wait for page ready state."""
    logger.debug(f"🌐 Navigating to URL: {url}")

    driver.get(url)
    logger.debug("driver.get() completed, waiting for page ready state...")
    waiter: WebDriverWait[Any] = WebDriverWait(driver, page_timeout)
    waiter.until(_document_ready)
    time.sleep(random.uniform(0.5, 1.5))

    # Some Ancestry pages show a cookie/privacy overlay that can prevent key
    # content from becoming visible. Best-effort dismiss it on every navigation.
    with contextlib.suppress(Exception):
        from browser.login import consent

        consent(driver)


def _dump_navigation_debug_artifacts(driver: WebDriver, selector: str) -> None:
    """Best-effort dump of page HTML/screenshot for diagnosing selector failures."""
    try:
        logs_dir = Path("Logs")
        logs_dir.mkdir(exist_ok=True)
        ts = int(time.time())
        safe_selector = re.sub(r"[^A-Za-z0-9_-]+", "_", selector)[:40] or "body"
        base = f"nav_timeout_{ts}_{safe_selector}"
        html_path = logs_dir / f"{base}.html"
        png_path = logs_dir / f"{base}.png"

        with contextlib.suppress(Exception):
            html = driver.page_source
            html_path.write_text(html or "", encoding="utf-8")

        with contextlib.suppress(Exception):
            # Use getattr to avoid Pylance warning about partially-typed selenium stub
            save_fn: Callable[[str], bool] = getattr(driver, "save_screenshot")
            save_fn(str(png_path))

        logger.warning(f"Saved navigation debug artifacts: {html_path} and {png_path}")
    except Exception as dump_err:
        logger.debug(f"Failed to dump navigation debug artifacts: {dump_err}")


def _get_landed_url_base(driver: WebDriver, attempt: int) -> str | None:
    """Get and normalize the current URL after navigation."""
    try:
        landed_url = driver.current_url
        landed_url_parsed = urlparse(landed_url)
        landed_url_base = urlunparse(
            (
                landed_url_parsed.scheme,
                landed_url_parsed.netloc,
                landed_url_parsed.path.rstrip("/"),
                "",
                "",
                "",
            )
        ).rstrip("/")
        logger.debug(f"Landed on URL base: {landed_url_base}")
        return landed_url_base
    except WebDriverException as e:
        logger.error(f"Failed to get current URL after get() (Attempt {attempt}): {e}. Retrying.")
        return None


def _check_for_mfa_page(driver: WebDriver) -> bool:
    """Check if currently on MFA page."""
    try:
        _wait_until_visible(
            WebDriverWait(driver, 1),
            (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR),
        )
        return True
    except (TimeoutException, NoSuchElementException):
        # MFA page not detected
        return False
    except WebDriverException as e:
        logger.warning(f"WebDriverException checking for MFA header: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking for MFA page: {e}", exc_info=True)
        return False


def _check_for_login_page(driver: WebDriver, target_url_base: str, signin_page_url_base: str) -> bool:
    """Check if currently on login page (only if not intentionally navigating there)."""
    if target_url_base == signin_page_url_base:
        return False
    try:
        _wait_until_visible(
            WebDriverWait(driver, 1),
            (By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR),
        )
        return True
    except (TimeoutException, NoSuchElementException):
        # Login page not detected
        return False
    except WebDriverException as e:
        logger.warning(f"WebDriverException checking for Login username input: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking for Login page: {e}", exc_info=True)
        return False


def _handle_login_redirect(session_manager: "SessionManager") -> str:
    """Handle unexpected login page redirect. Returns 'retry' or 'fail'."""
    from browser.login import log_in, login_status

    logger.warning("Landed on Login page unexpectedly. Checking login status first...")
    login_stat = login_status(session_manager, disable_ui_fallback=True)
    if login_stat is True:
        logger.info("Login status OK after landing on login page redirect. Retrying original navigation.")
        return "retry"

    logger.info("Not logged in according to API. Attempting re-login...")
    login_result_str = log_in(session_manager)
    if login_result_str == "LOGIN_SUCCEEDED":
        logger.info("Re-login successful. Retrying original navigation...")
        return "retry"

    logger.error(f"Re-login attempt failed ({login_result_str}). Cannot complete navigation.")
    return "fail"


def _check_signin_redirect(
    target_url_base: str,
    landed_url_base: str,
    signin_page_url_base: str,
    session_manager: Optional["SessionManager"],
) -> bool:
    """Check if redirect from signin to base URL is valid. Returns True if valid redirect."""
    # When navigating to signin page and already logged in, Ancestry redirects to homepage
    # Compare landed URL to base URL (scheme + netloc + path, normalized)
    from browser.login import login_status

    base_url_normalized = _parse_and_normalize_url(config_schema.api.base_url) or ""
    is_signin_to_base_redirect = target_url_base == signin_page_url_base and landed_url_base == base_url_normalized
    if is_signin_to_base_redirect:
        logger.debug("Redirected from signin page to base URL. Verifying login status...")
        time.sleep(1)
        if session_manager and login_status(session_manager, disable_ui_fallback=True) is True:
            logger.info(
                "Redirect after signin confirmed as logged in. Considering original navigation target 'signin' successful."
            )
            return True
    return False


def _handle_url_mismatch(
    driver: WebDriver,
    landed_url_base: str,
    target_url_base: str,
    unavailability_selectors: dict[str, tuple[str, int]],
) -> str:
    """Handle landing on unexpected URL. Returns 'continue', 'fail', or 'success'."""
    logger.warning(f"Navigation landed on unexpected URL base: '{landed_url_base}' (Expected: '{target_url_base}')")
    action, wait_time = _check_for_unavailability(driver, unavailability_selectors)
    if action == "skip":
        logger.error("Page no longer available message found. Skipping.")
        return "fail"
    if action == "refresh":
        logger.info(f"Temporary unavailability message found. Waiting {wait_time}s and retrying...")
        time.sleep(wait_time)
        return "continue"
    logger.warning("Wrong URL, no specific unavailability message found. Retrying navigation.")
    return "continue"


def _wait_for_element(
    driver: WebDriver,
    selector: str,
    element_timeout: int,
    unavailability_selectors: dict[str, tuple[str, int]],
) -> str:
    """Wait for target element. Returns 'success', 'fail', or 'continue'."""
    wait_selector = selector if selector else "body"
    logger.debug(f"On correct URL base. Waiting up to {element_timeout}s for selector: '{wait_selector}'")
    try:
        _wait_until_visible(
            WebDriverWait(driver, element_timeout),
            (By.CSS_SELECTOR, wait_selector),
        )
        logger.debug(f"Navigation successful and element '{wait_selector}' found.")
        return "success"
    except TimeoutException:
        # Many modern pages render lists via virtualization/shadow DOM or keep
        # elements present-but-not-visible briefly; a presence check is a safer
        # readiness signal than strict visibility.
        with contextlib.suppress(TimeoutException):
            _wait_until_present(
                WebDriverWait(driver, element_timeout),
                (By.CSS_SELECTOR, wait_selector),
            )
            logger.debug(f"Element '{wait_selector}' present but not visible; treating navigation as successful.")
            return "success"

        current_url_on_timeout = "Unknown"
        with contextlib.suppress(Exception):
            current_url_on_timeout = driver.current_url
        logger.warning(
            f"Timeout waiting for selector '{wait_selector}' at {current_url_on_timeout} (URL base was correct)."
        )

        _dump_navigation_debug_artifacts(driver, wait_selector)

        action, wait_time = _check_for_unavailability(driver, unavailability_selectors)
        if action == "skip":
            return "fail"
        if action == "refresh":
            time.sleep(wait_time)
            return "continue"
        logger.warning("Timeout on selector, no unavailability message. Retrying navigation.")
        return "continue"
    except WebDriverException as el_wait_err:
        logger.error(f"WebDriverException waiting for selector '{wait_selector}': {el_wait_err}")
        return "continue"


def _handle_navigation_alert(driver: WebDriver, attempt: int) -> str:
    """Handle unexpected alert. Returns 'continue' or 'fail'."""
    alert_text = "N/A"
    with contextlib.suppress(AttributeError):
        # Try to get alert text if available
        alert = driver.switch_to.alert
        alert_text = alert.text
    logger.warning(f"Unexpected alert detected (Attempt {attempt}): {alert_text}")
    try:
        driver.switch_to.alert.accept()
        logger.info("Accepted unexpected alert.")
        return "continue"
    except Exception as accept_e:
        logger.error(f"Failed to accept unexpected alert: {accept_e}")
        return "fail"


def _handle_webdriver_exception(
    driver: WebDriver,
    session_manager: Optional["SessionManager"],
) -> tuple[str, WebDriver | None]:
    """Handle WebDriver exception. Returns (action, driver) where action is 'continue' or 'fail'."""
    if session_manager and not is_browser_open(driver):
        logger.error("WebDriver session invalid after exception. Attempting restart...")
        if session_manager.restart_sess():
            logger.info("Session restarted. Retrying navigation...")
            driver_instance = session_manager.browser_manager.driver
            if driver_instance is not None:
                return ("continue", driver_instance)
            return ("fail", None)
        logger.error("Session restart failed. Cannot complete navigation.")
        return ("fail", None)
    logger.warning("WebDriverException occurred, session seems valid or no restart possible. Waiting before retry.")
    time.sleep(random.uniform(2, 4))
    return ("continue", driver)


def _check_url_mismatch_and_handle(
    driver: WebDriver,
    landed_url_base: str,
    target_url_base: str,
    signin_page_url_base: str,
    unavailability_selectors: dict[str, tuple[str, int]],
    session_manager: Optional["SessionManager"],
) -> tuple[str | None, WebDriver | None]:
    """
    Check for URL mismatch and handle appropriately.
    Returns (action, driver) where action is 'success', 'fail', 'continue', or None.
    """
    if landed_url_base == target_url_base:
        return (None, driver)

    # Check if signin redirect is acceptable
    if _check_signin_redirect(target_url_base, landed_url_base, signin_page_url_base, session_manager):
        return ("success", driver)

    # Handle URL mismatch
    mismatch_action = _handle_url_mismatch(driver, landed_url_base, target_url_base, unavailability_selectors)
    if mismatch_action == "fail":
        return ("fail", driver)
    if mismatch_action == "continue":
        return ("continue", driver)

    return (None, driver)


def _validate_post_navigation(
    driver: WebDriver,
    landed_url_base: str,
    target_url_base: str,
    signin_page_url_base: str,
    selector: str,
    element_timeout: int,
    unavailability_selectors: dict[str, tuple[str, int]],
    session_manager: Optional["SessionManager"],
) -> tuple[str, WebDriver | None]:
    """
    Validate navigation after landing on page.
    Returns (action, driver) where action is 'success', 'fail', or 'continue'.
    """
    result_action = "continue"
    result_driver = driver

    # Check for MFA page
    if _check_for_mfa_page(driver):
        logger.error("Landed on MFA page unexpectedly during navigation. Navigation failed.")
        result_action = "fail"
    # Check for Login page
    elif _check_for_login_page(driver, target_url_base, signin_page_url_base):
        if session_manager:
            login_action = _handle_login_redirect(session_manager)
            if login_action == "retry":
                result_action = "continue"
            elif login_action in {"fail", "no_manager"}:
                result_action = "fail"
        else:
            logger.error("Landed on login page but no SessionManager provided.")
            result_action = "fail"
    else:
        # Check for URL mismatch
        url_check_result, updated_driver = _check_url_mismatch_and_handle(
            driver, landed_url_base, target_url_base, signin_page_url_base, unavailability_selectors, session_manager
        )
        if url_check_result:
            result_action = url_check_result
            result_driver = updated_driver if updated_driver else driver
        else:
            # --- Final Check: Element on Page ---
            element_result = _wait_for_element(driver, selector, element_timeout, unavailability_selectors)
            if element_result == "success":
                result_action = "success"
            elif element_result == "fail":
                result_action = "fail"
            elif element_result == "continue":
                result_action = "continue"

    return (result_action, result_driver)


def _perform_navigation_attempt(
    driver: WebDriver,
    nav_config: NavigationConfig,
    session_manager: Optional["SessionManager"],
    attempt: int,
) -> tuple[str, WebDriver | None]:
    """
    Perform a single navigation attempt.
    Returns (action, driver) where action is 'success', 'fail', 'continue', or 'retry'.
    """
    result_action = "continue"
    result_driver = driver

    try:
        # --- Pre-Navigation Checks ---
        driver_check = _check_browser_session(driver, session_manager, attempt)
        if driver_check is None:
            result_action = "fail"
        elif driver_check != driver:
            result_action = "retry"
            result_driver = driver_check
        else:
            # --- Navigation Execution ---
            _execute_navigation(driver, nav_config.url, nav_config.page_timeout)

            # --- Post-Navigation Checks ---
            landed_url_base = _get_landed_url_base(driver, attempt)
            if landed_url_base is None:
                result_action = "continue"
            else:
                # Validate post-navigation state
                result_action, result_driver = _validate_post_navigation(
                    driver,
                    landed_url_base,
                    nav_config.target_url_base,
                    nav_config.signin_page_url_base,
                    nav_config.selector,
                    nav_config.element_timeout,
                    nav_config.unavailability_selectors,
                    session_manager,
                )

    except UnexpectedAlertPresentException:
        alert_action = _handle_navigation_alert(driver, attempt)
        result_action = "fail" if alert_action == "fail" else "continue"

    except WebDriverException as wd_e:
        logger.error(f"WebDriverException during navigation (Attempt {attempt}): {wd_e}", exc_info=False)
        wd_action, new_driver = _handle_webdriver_exception(driver, session_manager)
        result_action = "fail" if wd_action == "fail" else "continue"
        result_driver = new_driver if new_driver else driver

    except Exception as e:
        logger.error(f"Unexpected error during navigation (Attempt {attempt}): {e}", exc_info=True)
        time.sleep(random.uniform(2, 4))
        result_action = "continue"

    return (result_action, result_driver)


def nav_to_page(
    driver: WebDriver,
    url: str,
    selector: str = WAIT_FOR_PAGE_SELECTOR,  # CSS selector to wait for as indication of page load success
    session_manager: Optional["SessionManager"] = None,  # Pass SessionManager for context/restart
) -> bool:
    """
    Navigates the WebDriver to a given URL, waits for a specific element,
    and handles common issues like redirects, unavailability messages, and session restarts.

    Args:
        driver: The WebDriver instance.
        url: The target URL to navigate to.
        selector: A CSS selector for an element expected on the target page.
                  Defaults to 'body'. Used to verify successful navigation.
        session_manager: The active SessionManager instance (optional, needed for restart).

    Returns:
        True if navigation succeeded and the selector was found, False otherwise.
    """
    # Validate inputs
    if not _validate_nav_inputs(url):
        return False

    # Parse and normalize URL
    target_url_base = _parse_and_normalize_url(url)
    if target_url_base is None:
        return False

    # Get configuration
    max_attempts = config_schema.api.max_retries
    page_timeout = config_schema.selenium.page_load_timeout
    element_timeout = config_schema.selenium.explicit_wait

    # Define common problematic URLs/selectors
    signin_page_url_base = urljoin(config_schema.api.base_url, "account/signin").rstrip("/")
    # Selectors for known 'unavailable' pages
    unavailability_selectors: dict[str, tuple[str, int]] = {
        TEMP_UNAVAILABLE_SELECTOR: ("refresh", 5),
        PAGE_NO_LONGER_AVAILABLE_SELECTOR: ("skip", 0),
    }

    for attempt in range(1, max_attempts + 1):
        logger.debug(f"Navigation Attempt {attempt}/{max_attempts} to: {url}")

        nav_config = NavigationConfig(
            url=url,
            selector=selector,
            target_url_base=target_url_base,
            signin_page_url_base=signin_page_url_base,
            unavailability_selectors=unavailability_selectors,
            page_timeout=page_timeout,
            element_timeout=element_timeout,
        )
        action, new_driver = _perform_navigation_attempt(driver, nav_config, session_manager, attempt)

        if action == "success":
            return True
        if action == "fail":
            return False
        if action in {"retry", "continue"}:
            driver = new_driver if new_driver else driver
            continue

    # --- Failed After All Attempts ---
    logger.critical(f"Navigation to '{url}' failed permanently after {max_attempts} attempts.")
    try:
        logger.error(f"Final URL after failure: {driver.current_url}")
    except Exception:
        logger.error("Could not retrieve final URL after failure.")
    # End of try/except
    return False


# End of nav_to_page


def _check_for_unavailability(
    driver: WebDriver,
    selectors: dict[str, tuple[str, int]],
) -> tuple[str | None, int]:
    """Checks if known 'page unavailable' messages are present using provided selectors."""
    # Check if driver is usable
    if not is_browser_open(driver):
        logger.warning("Cannot check for unavailability: driver session invalid.")
        return None, 0
    # End of if

    for msg_selector, (action, wait_time) in selectors.items():
        # Use selenium_utils helper 'is_elem_there' with a very short wait
        # Assume is_elem_there is imported
        if is_elem_there(driver, msg_selector, By.CSS_SELECTOR, wait=1):
            logger.warning(
                f"Unavailability message found matching selector: '{msg_selector}'. Action: {action}, Wait: {wait_time}s"
            )
            return action, wait_time  # Return action (refresh/skip) and wait time
        # End of if
    # End of for

    # Return default (no action, zero wait) if no matching selectors found
    return None, 0


# End of _check_for_unavailability


# ------------------------------------------------------------------------------------
# Module Tests
# ------------------------------------------------------------------------------------


def module_tests() -> bool:
    """Run navigation module tests using standardized TestSuite format."""
    from testing.test_framework import TestSuite

    suite = TestSuite("Page Navigation", "browser/navigation.py")

    # Navigation functions are tightly coupled to browser/session state;
    # verify that the public API surface is importable and callable.
    suite.run_test(
        "Navigation API Surface",
        lambda: (
            callable(nav_to_page)
            and callable(_validate_nav_inputs)
            and callable(_parse_and_normalize_url)
            and callable(_check_for_unavailability)
        ),
        "Verify all navigation functions are importable and callable",
    )

    return suite.finish_suite()


# Standardized test runner (recommended pattern)
run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
