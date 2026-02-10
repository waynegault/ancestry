#!/usr/bin/env python3

"""Login, authentication, 2FA handling, and consent management for Ancestry.com."""

# === SESSION MANAGER IMPORT ===
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.session_manager import SessionManager

# === STANDARD LIBRARY IMPORTS ===
import logging
import random
import time
from typing import Any, cast
from urllib.parse import urljoin

# === THIRD-PARTY IMPORTS ===
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    ElementNotInteractableException,
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait

# === LOCAL IMPORTS ===
from browser.cookie_utils import _save_login_cookies
from browser.css_selectors import (
    CONFIRMED_LOGGED_IN_SELECTOR,
    CONSENT_ACCEPT_BUTTON_SELECTOR,
    COOKIE_BANNER_SELECTOR,
    FAILED_LOGIN_SELECTOR,
    GENERIC_ALERT_SELECTOR,
    GENERIC_ERROR_ELEMENTS_SELECTOR,
    LOG_IN_BUTTON_SELECTOR,
    PASSWORD_INPUT_SELECTOR,
    SIGN_IN_BUTTON_SELECTOR,
    TWO_FA_BODY_SELECTOR,
    TWO_FA_CODE_INPUT_SELECTOR,
    TWO_FA_EMAIL_SELECTOR,
    TWO_FA_SMS_METHOD_BUTTON_SELECTOR,
    TWO_FA_SMS_SELECTOR,
    TWO_FA_VERIFICATION_H1_TWO_STEP_XPATH,
    TWO_FA_VERIFICATION_H1_XPATH,
    TWO_FA_VERIFICATION_H2_TWO_STEP_XPATH,
    TWO_FA_VERIFICATION_H2_XPATH,
    TWO_STEP_VERIFICATION_HEADER_SELECTOR,
    USERNAME_INPUT_SELECTOR,
)
from browser.selenium_utils import DriverProtocol, WebElementProtocol, is_browser_open, is_elem_there
from config import config_schema
from core.selenium_utils import (
    wait_until_clickable as _wait_until_clickable,
    wait_until_not_present as _wait_until_not_present,
    wait_until_not_visible as _wait_until_not_visible,
    wait_until_present as _wait_until_present,
    wait_until_visible as _wait_until_visible,
)
from core.utils import retry, time_wait
from testing.test_utilities import create_standard_test_runner

# === MODULE SETUP ===
logger = logging.getLogger(__name__)

# === TYPE ALIASES ===
Locator = tuple[str, str]


# ----------------------------------------------------------------------------
# 2FA Helper Functions
# ----------------------------------------------------------------------------


def _wait_for_2fa_header(element_wait: "WebDriverWait[Any]", session_manager: "SessionManager") -> bool:
    """Wait for 2FA page header to appear."""
    try:
        logger.debug(f"Waiting for 2FA page header using selector: '{TWO_STEP_VERIFICATION_HEADER_SELECTOR}'")
        _wait_until_visible(element_wait, (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR))
        logger.debug("2FA page header detected.")
        return True
    except TimeoutException:
        logger.debug("Did not detect 2FA page header within timeout.")
        if login_status(session_manager, disable_ui_fallback=True) is True:
            logger.info("User appears logged in after checking for 2FA page. Assuming 2FA handled/skipped.")
            return True
        logger.warning("Assuming 2FA not required or page didn't load correctly (header missing).")
        return False
    except WebDriverException as e:
        logger.error(f"WebDriverException waiting for 2FA header: {e}")
        return False


def _find_sms_button_with_selectors(selector_wait: "WebDriverWait[Any]") -> WebElement | None:
    """Try multiple selectors to find the SMS button."""
    sms_selectors = [
        "button[data-method='sms']",
        TWO_FA_SMS_SELECTOR,
        "//button[contains(., 'Text message')]",
        "//div[contains(text(), 'Text message')]",
        "//button[contains(text(), 'Text message')]",
    ]

    for idx, selector in enumerate(sms_selectors):
        try:
            logger.debug(f"[DEBUG] Trying selector {idx + 1}/{len(sms_selectors)}: {selector}")
            locator: Locator = (By.XPATH, selector) if selector.startswith("//") else (By.CSS_SELECTOR, selector)
            sms_button = _wait_until_clickable(selector_wait, locator)
            logger.debug(f"✓ Found SMS button with selector: {selector}")
            return sms_button
        except (TimeoutException, NoSuchElementException) as e:
            logger.debug(f"SMS button not found with selector {idx + 1}: {selector} - {type(e).__name__}")
        except Exception as e:
            logger.debug(f"Error trying selector {idx + 1} ({selector}): {e}")

    return None


def _debug_log_page_buttons(driver: WebDriver) -> None:
    """Log all buttons on the page for debugging."""
    try:
        all_buttons: list[WebElement] = driver.find_elements(By.TAG_NAME, "button")
        logger.debug(f"[DEBUG] Found {len(all_buttons)} buttons on page")
        for i, btn in enumerate(all_buttons[:10]):
            try:
                btn_text = btn.text.strip()
                btn_proto = cast(WebElementProtocol, btn)
                btn_classes = btn_proto.get_attribute("class")
                btn_data_method = btn_proto.get_attribute("data-method")
                if btn_text or btn_data_method:
                    logger.debug(
                        f"  Button {i}: text='{btn_text}', class='{btn_classes}', data-method='{btn_data_method}'"
                    )
            except Exception:
                pass
    except Exception as debug_err:
        logger.debug(f"[DEBUG] Error listing buttons: {debug_err}")


def _perform_sms_button_click(driver: WebDriver, sms_button: WebElement) -> bool:
    """Attempt to click the SMS button with fallback strategies."""
    try:
        # Cast to Protocol to ensure types
        driver_proto = cast(DriverProtocol, driver)
        driver_proto.execute_script("arguments[0].click();", sms_button)
        logger.debug("SMS button clicked via JavaScript.")
        print("  ✓ SMS verification code requested. Check your phone!", flush=True)
        time.sleep(3)
        return _wait_for_code_input_field(driver)
    except WebDriverException as click_err:
        logger.error(f"Error clicking SMS button via JS: {click_err}")
        try:
            sms_button.click()
            logger.debug("SMS button clicked via standard click.")
            print("  ✓ SMS verification code requested. Check your phone!", flush=True)
            time.sleep(3)
            return _wait_for_code_input_field(driver)
        except Exception as std_err:
            logger.error(f"Standard click also failed: {std_err}")
            print("  ✗ Failed to click SMS button. Please click 'Text message' manually.", flush=True)
            return False


def _click_sms_button(driver: WebDriver) -> bool:
    """Try clicking the SMS 'Send Code' button."""
    print("  📱 Looking for SMS/Text message option...", flush=True)
    logger.debug(f"Looking for 2FA SMS button with selector: '{TWO_FA_SMS_SELECTOR}'")

    selector_wait = WebDriverWait(driver, 5)
    sms_button = _find_sms_button_with_selectors(selector_wait)

    if not sms_button:
        logger.warning("Could not find SMS button with any selector.")
        _debug_log_page_buttons(driver)
        print("  ⚠️  SMS button not found. Please click 'Text message' manually.", flush=True)
        return False

    return _perform_sms_button_click(driver, sms_button)


def _wait_for_code_input_field(driver: WebDriver) -> bool:
    """Wait for 2FA code input field to appear after clicking Send Code."""
    try:
        logger.debug(f"Waiting for 2FA code input field: '{TWO_FA_CODE_INPUT_SELECTOR}'")
        _wait_until_visible(WebDriverWait(driver, 10), (By.CSS_SELECTOR, TWO_FA_CODE_INPUT_SELECTOR))
        logger.debug("Code input field appeared after clicking 'Send Code'.")
        return True
    except TimeoutException:
        logger.debug("Code input field not visible yet - will wait for manual entry")
        return False
    except WebDriverException as e_input:
        logger.error(f"Error waiting for 2FA code input field: {e_input}. Check selector: {TWO_FA_CODE_INPUT_SELECTOR}")
        return False


def _wait_for_user_2fa_action(driver: WebDriver, code_entry_timeout: int) -> bool:
    """Wait for user to manually enter 2FA code and submit."""
    logger.info(f"Waiting up to {code_entry_timeout}s for user to manually enter 2FA code and submit...")
    start_time = time.time()
    check_interval = 0.5  # Check every half second for responsive detection

    while time.time() - start_time < code_entry_timeout:
        try:
            # Check if the 2FA header is GONE (indicates page change/submission)
            wait = WebDriverWait(driver, check_interval)
            _wait_until_not_visible(wait, (By.CSS_SELECTOR, TWO_STEP_VERIFICATION_HEADER_SELECTOR))
            logger.info("2FA page elements disappeared, assuming user submitted code.")
            return True
        except TimeoutException:
            time.sleep(2)  # Check every 2 seconds
        except NoSuchElementException:
            logger.info("2FA header element no longer present.")
            return True
        except WebDriverException as e:
            logger.error(f"WebDriver error checking for 2FA header during wait: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking for 2FA header during wait: {e}")
            return False

    logger.error(f"Timed out ({code_entry_timeout}s) waiting for user 2FA action (page did not change).")
    return False


def _verify_2fa_completion(session_manager: "SessionManager") -> bool:
    """Verify that 2FA was completed successfully by checking login status."""
    logger.info("Re-checking login status after potential 2FA submission...")
    time.sleep(1)  # Allow page to settle
    final_status = login_status(session_manager, disable_ui_fallback=False)
    if final_status is True:
        logger.info("User completed 2FA successfully (login confirmed after page change).")
        # Save cookies after successful 2FA login
        _save_login_cookies(session_manager)
        return True
    logger.error("2FA page disappeared, but final login status check failed or returned False.")
    return False


@time_wait("Handle 2FA Page")
def handle_two_fa(session_manager: "SessionManager") -> bool:
    if session_manager.browser_manager.driver is None:
        logger.error("handle_two_fa: SessionManager driver is None. Cannot proceed.")
        return False

    driver = session_manager.browser_manager.driver
    element_wait = WebDriverWait(driver, config_schema.selenium.explicit_wait)

    try:
        print("Two-factor authentication required. Please check your email or phone for a verification code.")
        logger.debug("Handling Two-Factor Authentication (2FA)...")

        # Wait for 2FA page header
        if not _wait_for_2fa_header(element_wait, session_manager):
            return False

        # Handle cookie consent banner if present
        logger.debug("Checking for cookie consent banner on 2FA page...")
        if not consent(driver):
            logger.warning("Failed to handle consent banner on 2FA page, but continuing anyway.")

        # Try clicking SMS button (non-critical, continue if fails)
        _click_sms_button(driver)

        # Wait for user action
        code_entry_timeout = config_schema.selenium.two_fa_code_entry_timeout
        user_action_detected = _wait_for_user_2fa_action(driver, code_entry_timeout)

        # Verify completion
        if user_action_detected:
            return _verify_2fa_completion(session_manager)
        return False

    except WebDriverException as e:
        logger.error(f"WebDriverException during 2FA handling: {e}")
        if not is_browser_open(driver):
            logger.error("Session invalid after WebDriverException during 2FA.")
        # End of if
        return False
    except Exception as e:
        logger.error(f"Unexpected error during 2FA handling: {e}", exc_info=True)
        return False
    # End of try/except for overall 2FA handling

    # Should not be reachable unless an early return was missed
    # return False


# End of handle_two_fa

# ----------------------------------------------------------------------------
# Credential Entry Helper Functions
# ----------------------------------------------------------------------------


def _clear_input_field(driver: WebDriver, input_element: WebElement, field_name: str) -> bool:
    """Clear an input field robustly."""
    try:
        input_element.click()
        time.sleep(0.1)
        input_element.clear()
        time.sleep(0.1)
        # Cast to Protocol to ensure types
        driver_proto = cast(DriverProtocol, driver)
        driver_proto.execute_script("arguments[0].value = '';", input_element)
        time.sleep(0.1)
        return True
    except (ElementNotInteractableException, StaleElementReferenceException) as e:
        logger.warning(f"Issue clicking/clearing {field_name} field ({type(e).__name__}). Proceeding cautiously.")
        return True
    except WebDriverException as e:
        logger.error(f"WebDriverException clicking/clearing {field_name}: {e}. Aborting.")
        return False


def _enter_username(driver: WebDriver, element_wait: "WebDriverWait[Any]") -> bool:
    """Enter username into the login form."""
    logger.debug(f"Waiting for username input: '{USERNAME_INPUT_SELECTOR}'...")
    username_input = _wait_until_visible(element_wait, (By.CSS_SELECTOR, USERNAME_INPUT_SELECTOR))
    logger.debug("Username input field found.")

    if not _clear_input_field(driver, username_input, "username"):
        return False

    ancestry_username = config_schema.api.username
    if not ancestry_username:
        raise ValueError("ANCESTRY_USERNAME configuration is missing.")

    # Retry loop to ensure text is actually entered
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        logger.debug(f"Entering username (attempt {attempt}/{max_attempts})...")

        # Use JavaScript to set value directly for reliability
        # Cast to Protocol to ensure types
        driver_proto = cast(DriverProtocol, driver)
        driver_proto.execute_script("arguments[0].value = arguments[1];", username_input, ancestry_username)
        time.sleep(0.2)

        # Trigger input event to ensure page JavaScript recognizes the change
        driver_proto.execute_script(
            """
            var element = arguments[0];
            var event = new Event('input', { bubbles: true });
            element.dispatchEvent(event);
        """,
            username_input,
        )
        time.sleep(0.1)

        # Verify the value was set
        username_input_proto = cast(WebElementProtocol, username_input)
        current_value = username_input_proto.get_attribute("value")
        if current_value == ancestry_username:
            logger.debug(f"Username successfully entered and verified (attempt {attempt}).")
            print(f"  ✓ Username entered: {ancestry_username[:3]}***", flush=True)
            time.sleep(random.uniform(0.2, 0.4))
            return True
        logger.warning(
            f"Username verification failed (attempt {attempt}): expected '{ancestry_username}', got '{current_value}'"
        )
        if attempt < max_attempts:
            time.sleep(0.5)

    logger.error(f"Failed to enter username after {max_attempts} attempts")
    return False


def _find_next_button_with_selectors(short_wait: "WebDriverWait[Any]") -> WebElement | None:
    """Try multiple selectors to find the Next button."""
    next_selectors = [
        SIGN_IN_BUTTON_SELECTOR,
        "button[type='submit']",
        "button.ancBtn",
        "//button[contains(text(), 'Next')]",
        "//button[contains(text(), 'Continue')]",
    ]

    for idx, selector in enumerate(next_selectors):
        try:
            logger.debug(f"[DEBUG] Trying Next button selector {idx + 1}/{len(next_selectors)}: {selector}")
            locator: Locator = (By.XPATH, selector) if selector.startswith("//") else (By.CSS_SELECTOR, selector)
            next_button = _wait_until_clickable(short_wait, locator)
            logger.debug(f"✓ Found Next button with selector: {selector}")
            return next_button
        except (TimeoutException, NoSuchElementException):
            logger.debug(f"Next button not found with selector {idx + 1}: {selector}")
        except Exception as e:
            logger.debug(f"Error trying selector {idx + 1}: {e}")

    return None


def _debug_log_page_state_after_next_click(driver: WebDriver) -> None:
    """Log page state after Next button click for debugging."""
    try:
        current_url = driver.current_url
        page_title = driver.title
        logger.debug(f"[DEBUG] After Next click - URL: {current_url}")
        logger.debug(f"[DEBUG] After Next click - Title: {page_title}")
    except Exception as debug_err:
        logger.debug(f"[DEBUG] Error checking page after Next: {debug_err}")


def _debug_log_signin_page_buttons(driver: WebDriver) -> None:
    """Log all buttons on signin page for debugging."""
    try:
        all_buttons: list[WebElement] = driver.find_elements(By.TAG_NAME, "button")
        logger.debug(f"[DEBUG] Found {len(all_buttons)} buttons on signin page")
        for i, btn in enumerate(all_buttons[:5]):
            try:
                btn_text = btn.text.strip()
                btn_proto = cast(WebElementProtocol, btn)
                btn_id = btn_proto.get_attribute("id")
                btn_class = btn_proto.get_attribute("class")
                if btn_text or btn_id:
                    logger.debug(f"  Button {i}: text='{btn_text}', id='{btn_id}', class='{btn_class}'")
            except Exception:
                pass
    except Exception as debug_err:
        logger.debug(f"[DEBUG] Error listing buttons: {debug_err}")


def _perform_next_button_click(driver: WebDriver, next_button: WebElement) -> bool:
    """Attempt to click the Next button with fallback strategies."""
    try:
        # Cast to Protocol to ensure types
        driver_proto = cast(DriverProtocol, driver)
        driver_proto.execute_script("arguments[0].click();", next_button)
        logger.debug("Next button clicked via JavaScript.")
        logger.info("Next button clicked, waiting for password field to appear...")
        time.sleep(random.uniform(2.0, 3.0))
        _debug_log_page_state_after_next_click(driver)
        return True
    except WebDriverException as js_err:
        logger.warning(f"JS click failed: {js_err}, trying standard click...")
        try:
            next_button.click()
            logger.debug("Next button clicked successfully (standard click).")
            time.sleep(random.uniform(2.0, 3.0))
            return True

        except (ElementClickInterceptedException, ElementNotInteractableException) as e:
            logger.error(f"Both click methods failed: {e}")
            return True  # Continue anyway - password field might be visible


def _click_next_button(driver: WebDriver, short_wait: "WebDriverWait[Any]") -> bool:
    """Click the Next button in two-step login flow."""
    logger.debug("Looking for Next/Continue button after username...")

    next_button = _find_next_button_with_selectors(short_wait)

    if not next_button:
        raise TimeoutException("Next button not found with any selector")

    try:
        logger.debug("Next button found, attempting to click...")
        return _perform_next_button_click(driver, next_button)
    except TimeoutException:
        logger.debug(
            "Next button not found with selector '{SIGN_IN_BUTTON_SELECTOR}', assuming single-step login (password field already visible)."
        )
        _debug_log_signin_page_buttons(driver)
        return True
    except WebDriverException as e:
        logger.warning(f"Error finding Next button: {e}. Continuing anyway.")
        return True


def _wait_for_password_field(driver: WebDriver, element_wait: "WebDriverWait[Any]") -> WebElement | None:
    """Wait for the password field, retrying once with a longer timeout."""
    try:
        logger.debug("[PASSWORD_ENTRY] Waiting for password field to appear...")
        password_input = _wait_until_visible(element_wait, (By.CSS_SELECTOR, PASSWORD_INPUT_SELECTOR))
        logger.info("[PASSWORD_ENTRY] Password field found on first attempt")
        return password_input
    except TimeoutException:
        logger.warning(
            "[PASSWORD_ENTRY] Password field did not appear with standard wait. Retrying with longer wait..."
        )
        try:
            password_input = _wait_until_visible(WebDriverWait(driver, 30), (By.CSS_SELECTOR, PASSWORD_INPUT_SELECTOR))
            logger.info("[PASSWORD_ENTRY] Password field found on retry with longer wait")
            return password_input
        except TimeoutException:
            logger.error("[PASSWORD_ENTRY] CRITICAL: Password field never appeared even after 30s wait!")
            return None
    except Exception as exc:
        logger.error(f"[PASSWORD_ENTRY] Unexpected error waiting for password field: {exc}")
        return None


def _attempt_password_entry(
    driver: WebDriver,
    password_input: WebElement,
    ancestry_password: str,
    attempt: int,
    max_attempts: int,
) -> bool:
    """Try to set and verify the password value once."""
    logger.info(f"[PASSWORD_ENTRY] Attempt {attempt}/{max_attempts}: Entering password...")

    logger.debug("[PASSWORD_ENTRY] Setting password value via JavaScript...")
    # Cast to Protocol to ensure types
    driver_proto = cast(DriverProtocol, driver)
    driver_proto.execute_script("arguments[0].value = arguments[1];", password_input, ancestry_password)
    time.sleep(0.2)

    logger.debug("[PASSWORD_ENTRY] Triggering input event...")
    driver_proto.execute_script(
        """
            var element = arguments[0];
            var event = new Event('input', { bubbles: true });
            element.dispatchEvent(event);
        """,
        password_input,
    )
    time.sleep(0.1)

    logger.debug("[PASSWORD_ENTRY] Verifying password was set...")
    password_input_proto = cast(WebElementProtocol, password_input)
    current_value = password_input_proto.get_attribute("value")
    current_length = len(current_value) if current_value else 0
    expected_length = len(ancestry_password)
    logger.info(f"[PASSWORD_ENTRY] Verification: current_length={current_length}, expected_length={expected_length}")

    if current_value and current_length == expected_length:
        logger.info(f"[PASSWORD_ENTRY] ✅ Password successfully entered and verified (attempt {attempt})")
        print("  ✓ Password entered successfully", flush=True)
        time.sleep(random.uniform(0.3, 0.6))
        return True

    logger.warning(
        f"[PASSWORD_ENTRY] ❌ Password verification failed (attempt {attempt}): length mismatch (expected {expected_length}, got {current_length})"
    )
    return False


def _sleep_between_password_attempts(attempt: int, max_attempts: int) -> None:
    """Sleep between password attempts when another retry is allowed."""
    if attempt < max_attempts:
        logger.info("[PASSWORD_ENTRY] Retrying in 0.5s...")
        time.sleep(0.5)


def _enter_password(driver: WebDriver, element_wait: "WebDriverWait[Any]") -> tuple[bool, WebElement | None]:
    """Enter password into the login form."""
    logger.info("[PASSWORD_ENTRY] Starting password entry process...")
    logger.debug(f"Waiting for password input: '{PASSWORD_INPUT_SELECTOR}'...")

    password_input = _wait_for_password_field(driver, element_wait)
    if password_input is None:
        return False, None

    logger.debug("Password input field found.")
    logger.debug("[PASSWORD_ENTRY] Clearing password field...")
    if not _clear_input_field(driver, password_input, "password"):
        logger.error("[PASSWORD_ENTRY] Failed to clear password field")
        return False, None
    logger.debug("[PASSWORD_ENTRY] Password field cleared successfully")

    ancestry_password = config_schema.api.password
    if not ancestry_password:
        logger.error("[PASSWORD_ENTRY] CRITICAL: Password configuration is missing!")
        raise ValueError("ANCESTRY_PASSWORD configuration is missing.")

    logger.info(f"[PASSWORD_ENTRY] Password loaded from config (length: {len(ancestry_password)})")

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            if _attempt_password_entry(driver, password_input, ancestry_password, attempt, max_attempts):
                return True, password_input
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"[PASSWORD_ENTRY] Exception during attempt {attempt}: {type(exc).__name__} - {exc}")
            if attempt == max_attempts:
                logger.error("[PASSWORD_ENTRY] All attempts exhausted due to exceptions")
                return False, None
            time.sleep(0.5)
            continue

        _sleep_between_password_attempts(attempt, max_attempts)

    logger.error(f"[PASSWORD_ENTRY] ❌ FAILED to enter password after {max_attempts} attempts")
    return False, None


# ----------------------------------------------------------------------------
# Sign-In Button Click Helpers
# ----------------------------------------------------------------------------


def _try_standard_click(sign_in_button: WebElement) -> bool:
    """Try standard click on sign in button."""
    try:
        logger.debug("Attempting standard click on sign in button...")
        sign_in_button.click()
        logger.debug("Standard click executed.")
        return True
    except (ElementClickInterceptedException, ElementNotInteractableException, StaleElementReferenceException) as e:
        logger.warning(f"Standard click failed ({type(e).__name__}).")
        return False
    except WebDriverException as e:
        logger.error(f"WebDriver error during standard click: {e}.")
        return False


def _try_javascript_click(driver: WebDriver, sign_in_button: WebElement) -> bool:
    """Try JavaScript click on sign in button."""
    try:
        logger.debug("Attempting JavaScript click on sign in button...")
        # Cast to Protocol to ensure types
        driver_proto = cast(DriverProtocol, driver)
        driver_proto.execute_script("arguments[0].click();", sign_in_button)
        logger.info("JavaScript click executed.")
        return True
    except WebDriverException as js_click_e:
        logger.error(f"Error during JavaScript click: {js_click_e}")
        return False


def _try_return_key_fallback(password_input: WebElement) -> bool:
    """Try sending RETURN key to password field as fallback."""
    try:
        logger.warning("Attempting fallback: Sending RETURN key to password field.")
        password_input.send_keys(Keys.RETURN)
        logger.info("Fallback RETURN key sent to password field.")
        return True
    except StaleElementReferenceException:
        # Stale element means page already changed (form likely submitted) - this is success!
        logger.info("Form already submitted (stale element) - treating as successful login.")
        return True
    except (WebDriverException, ElementNotInteractableException) as key_e:
        logger.error(f"Failed to send RETURN key: {key_e}")
        return False


def _click_sign_in_button(driver: WebDriver, password_input: WebElement) -> bool:
    """Click the sign in button or use fallback methods."""
    # Try to locate sign in button
    try:
        logger.debug(f"Waiting for sign in button presence: '{SIGN_IN_BUTTON_SELECTOR}'...")
        _wait_until_present(WebDriverWait(driver, 3), (By.CSS_SELECTOR, SIGN_IN_BUTTON_SELECTOR))
        logger.debug("Waiting for sign in button clickability...")
        sign_in_button = _wait_until_clickable(
            WebDriverWait(driver, 3),
            (By.CSS_SELECTOR, SIGN_IN_BUTTON_SELECTOR),
        )
        logger.debug("Sign in button located and deemed clickable.")
    except TimeoutException:
        logger.debug("Sign in button not found, using RETURN key fallback (normal on some pages)")
        return _try_return_key_fallback(password_input)
    except WebDriverException as find_e:
        logger.error(f"Unexpected WebDriver error finding sign in button: {find_e}")
        return False

    if not sign_in_button:
        return False

    # Try standard click first
    if _try_standard_click(sign_in_button):
        return True

    # Try JavaScript click if standard click failed
    logger.warning("Standard click failed. Trying JS click...")
    if _try_javascript_click(driver, sign_in_button):
        return True

    # Try RETURN key as final fallback
    logger.warning("Both standard and JS clicks failed.")
    return _try_return_key_fallback(password_input)


def enter_creds(driver: WebDriver) -> bool:
    """Enter credentials and sign in to Ancestry."""
    element_wait = WebDriverWait(driver, config_schema.selenium.explicit_wait)
    short_wait = WebDriverWait(driver, 3)  # Reduced from 5s to 3s for faster response
    time.sleep(random.uniform(0.5, 1.0))

    result = False

    try:
        logger.debug("Entering Credentials and Signing In...")

        # Enter username
        if _enter_username(driver, element_wait) and _click_next_button(driver, short_wait):
            # Enter password
            password_ok, password_input = _enter_password(driver, element_wait)
            if password_ok and password_input is not None:
                # Click sign in button
                result = _click_sign_in_button(driver, password_input)

    except (TimeoutException, NoSuchElementException) as e:
        logger.error(f"Timeout or Element not found finding username/password field: {e}")
    except ValueError as ve:
        logger.critical(f"Configuration Error: {ve}")
        raise ve
    except WebDriverException as e:
        logger.error(f"WebDriver error entering credentials: {e}")
        if not is_browser_open(driver):
            logger.error("Session invalid during credential entry.")
    except Exception as e:
        logger.error(f"Unexpected error entering credentials: {e}", exc_info=True)

    return result


# End of enter_creds

# ----------------------------------------------------------------------------
# Consent Banner Functions
# ----------------------------------------------------------------------------


def _find_consent_banner(driver: WebDriver) -> WebElement | None:
    """Find the cookie consent banner if present."""
    logger.debug(f"Checking for cookie consent overlay: '{COOKIE_BANNER_SELECTOR}'")
    try:
        # Reduce wait to 1 second for faster dismissal
        overlay_element = _wait_until_present(
            WebDriverWait(driver, 1),
            (By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR),
        )
        logger.debug("Cookie consent overlay DETECTED.")
        return overlay_element
    except TimeoutException:
        logger.debug("Cookie consent overlay not found. Assuming no consent needed.")
        return None
    except WebDriverException as e:
        logger.error(f"Error checking for consent banner: {e}")
        raise


def _click_accept_button_standard(driver: WebDriver, accept_button: WebElement) -> bool:
    """Try standard click on accept button."""
    try:
        accept_button.click()
        logger.info("Clicked accept button successfully.")
        # Reduce verification wait to 1 second for faster dismissal
        _wait_until_not_present(
            WebDriverWait(driver, 1),
            (By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR),
        )
        logger.debug("Consent overlay gone after clicking accept button.")
        return True
    except ElementClickInterceptedException:
        logger.warning("Click intercepted for accept button, trying JS click...")
        return False
    except (TimeoutException, NoSuchElementException):
        logger.debug("Consent overlay likely gone after standard click (verification timed out/not found).")
        return True
    except WebDriverException as click_err:
        logger.error(f"Error during standard click on accept button: {click_err}. Trying JS click...")
        return False


def _click_accept_button_js(driver: WebDriver, accept_button: WebElement) -> bool:
    """Try JavaScript click on accept button."""
    try:
        logger.debug("Attempting JS click on accept button...")
        # Cast to Protocol to ensure types
        driver_proto = cast(DriverProtocol, driver)
        driver_proto.execute_script("arguments[0].click();", accept_button)
        logger.info("Clicked accept button via JS successfully.")
        # Reduce verification wait to 1 second for faster dismissal
        _wait_until_not_present(
            WebDriverWait(driver, 1),
            (By.CSS_SELECTOR, COOKIE_BANNER_SELECTOR),
        )
        logger.debug("Consent overlay gone after JS clicking accept button.")
        return True
    except (TimeoutException, NoSuchElementException):
        logger.debug("Consent overlay likely gone after JS click (verification timed out/not found).")
        return True
    except WebDriverException as js_click_err:
        logger.error(f"Failed JS click for accept button: {js_click_err}")
        return False


def _handle_consent_button(driver: WebDriver) -> bool:
    """Handle clicking the consent accept button."""
    logger.debug(f"Trying specific accept button: '{CONSENT_ACCEPT_BUTTON_SELECTOR}'")
    try:
        # Reduce wait to 1 second for faster dismissal
        accept_button = _wait_until_clickable(
            WebDriverWait(driver, 1),
            (By.CSS_SELECTOR, CONSENT_ACCEPT_BUTTON_SELECTOR),
        )
        logger.info("Found specific clickable accept button.")

        # Try standard click first
        if _click_accept_button_standard(driver, accept_button):
            return True

        # Try JS click as fallback
        return _click_accept_button_js(driver, accept_button)

    except TimeoutException:
        logger.warning(f"Specific accept button '{CONSENT_ACCEPT_BUTTON_SELECTOR}' not found or not clickable.")
        return False
    except WebDriverException as find_err:
        logger.error(f"Error finding/clicking specific accept button: {find_err}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error handling consent button: {e}", exc_info=True)
        return False


def consent(driver: WebDriver) -> bool:
    """Handles the cookie consent banner if present. Fast dismissal with no retries."""
    # Find consent banner
    try:
        overlay_element = _find_consent_banner(driver)
    except WebDriverException:
        return False

    # No banner found
    if overlay_element is None:
        return True

    # Try to handle the consent button
    if _handle_consent_button(driver):
        return True

    # If button click failed
    logger.error("Could not remove consent overlay via button click.")
    return False


# End of consent

# ----------------------------------------------------------------------------
# Login Flow Functions
# ----------------------------------------------------------------------------


def _check_initial_login_status(session_manager: "SessionManager") -> str | None:
    """Check if already logged in before attempting login."""
    initial_status = login_status(session_manager, disable_ui_fallback=True)
    if initial_status is True:
        print("Already logged in. No need to sign in again.")
        return "LOGIN_SUCCEEDED"
    return None


def _navigate_to_signin(driver: Any, session_manager: "SessionManager", signin_url: str) -> str | None:
    """Navigate to sign-in page and verify."""
    from core.utils import nav_to_page

    if not nav_to_page(driver, signin_url, USERNAME_INPUT_SELECTOR, session_manager):
        logger.debug("Navigation to sign-in page failed/redirected. Checking login status...")
        current_status = login_status(session_manager, disable_ui_fallback=True)
        if current_status is True:
            logger.info("Detected as already logged in during navigation attempt. Login considered successful.")
            return "LOGIN_SUCCEEDED"
        logger.error("Failed to navigate to login page (and not logged in).")
        return "LOGIN_FAILED_NAVIGATION"
    logger.debug("Successfully navigated to sign-in page.")
    return None


def _check_for_login_errors(driver: Any) -> str | None:
    """Check for specific or generic login error messages."""
    try:
        WebDriverWait(driver, 1).until(
            expected_conditions.presence_of_element_located((By.CSS_SELECTOR, FAILED_LOGIN_SELECTOR))
        )
        logger.error("Login failed: Specific 'Invalid Credentials' alert detected.")
        return "LOGIN_FAILED_BAD_CREDS"
    except TimeoutException:
        try:
            alert_element = WebDriverWait(driver, 0.5).until(
                expected_conditions.presence_of_element_located((By.CSS_SELECTOR, GENERIC_ALERT_SELECTOR))
            )
            alert_text = alert_element.text if alert_element and alert_element.text else "Unknown error"
            logger.error(f"Login failed: Generic alert found: '{alert_text}'.")
            return "LOGIN_FAILED_ERROR_DISPLAYED"
        except TimeoutException:
            logger.error("Login failed: Credential entry failed, but no specific or generic alert found.")
            return "LOGIN_FAILED_CREDS_ENTRY"
        except WebDriverException as alert_err:
            logger.warning(f"Error checking for generic login error message: {alert_err}")
            return "LOGIN_FAILED_CREDS_ENTRY"
    except WebDriverException as alert_err:
        logger.warning(f"Error checking for specific login error message: {alert_err}")
        return "LOGIN_FAILED_CREDS_ENTRY"


def _try_2fa_selectors(driver: Any) -> bool:
    """Try multiple selectors to detect 2FA page."""
    twofa_selectors = [
        ("css", TWO_STEP_VERIFICATION_HEADER_SELECTOR),
        ("css", TWO_FA_BODY_SELECTOR),
        ("xpath", TWO_FA_VERIFICATION_H1_XPATH),
        ("xpath", TWO_FA_VERIFICATION_H2_XPATH),
        ("xpath", TWO_FA_VERIFICATION_H1_TWO_STEP_XPATH),
        ("xpath", TWO_FA_VERIFICATION_H2_TWO_STEP_XPATH),
        ("css", TWO_FA_SMS_METHOD_BUTTON_SELECTOR),
        ("css", TWO_FA_EMAIL_SELECTOR),
    ]

    for idx, (selector_type, selector) in enumerate(twofa_selectors):
        try:
            logger.debug(f"[DEBUG] Trying 2FA selector {idx + 1}/{len(twofa_selectors)}: ({selector_type}) {selector}")
            by_type = By.XPATH if selector_type == "xpath" else By.CSS_SELECTOR
            WebDriverWait(driver, 2).until(expected_conditions.visibility_of_element_located((by_type, selector)))
            logger.debug(f"✓ Found 2FA page with selector: ({selector_type}) {selector}")
            return True
        except TimeoutException:
            logger.debug(f"2FA page not found with selector {idx + 1}: ({selector_type}) {selector}")
        except Exception as e:
            logger.debug(f"Error trying selector {idx + 1} (({selector_type}) {selector}): {e}")

    return False


def _debug_log_page_headers(driver: Any) -> None:
    """Log page header elements for debugging."""
    try:
        body_classes = driver.find_element(By.TAG_NAME, "body").get_attribute("class")
        logger.debug(f"[DEBUG] Body classes: {body_classes}")

        h1_elements = driver.find_elements(By.TAG_NAME, "h1")
        h2_elements = driver.find_elements(By.TAG_NAME, "h2")
        logger.debug(f"[DEBUG] Found {len(h1_elements)} h1 elements, {len(h2_elements)} h2 elements")

        for h1 in h1_elements[:3]:
            if h1.is_displayed():
                logger.debug(f"[DEBUG] h1 text: '{h1.text}'")

        for h2 in h2_elements[:3]:
            if h2.is_displayed():
                logger.debug(f"[DEBUG] h2 text: '{h2.text}'")
    except Exception as debug_err:
        logger.debug(f"[DEBUG] Error checking page elements: {debug_err}")


def _detect_2fa_page(driver: Any) -> tuple[bool, str | None]:
    """Detect if 2FA page is present."""
    try:
        logger.info("Checking for two-step verification page (waiting up to 15 seconds)...")
        twofa_found = _try_2fa_selectors(driver)

        if twofa_found:
            logger.info("Two-step verification page detected.")
            print("\n🔐 Two-factor authentication required...")
            return True, None

    except Exception as outer_err:
        logger.debug(f"Unexpected error in 2FA detection: {outer_err}")

    logger.debug("Two-step verification page not detected with any selector.")
    _debug_log_page_headers(driver)
    print("  (i) No 2FA prompt detected", flush=True)
    return False, None


def _handle_2fa_flow(session_manager: "SessionManager") -> str:
    """Handle 2FA verification flow."""
    if handle_two_fa(session_manager):
        logger.info("Two-step verification handled successfully.")
        if login_status(session_manager) is True:
            print("\n✓ Two-factor authentication completed successfully!")
            return "LOGIN_SUCCEEDED"
        logger.error("Login status check failed AFTER successful 2FA handling report.")
        return "LOGIN_FAILED_POST_2FA_VERIFY"
    logger.error("Two-step verification handling failed.")
    return "LOGIN_FAILED_2FA_HANDLING"


def _verify_login_no_2fa(driver: Any, session_manager: "SessionManager", signin_url: str) -> str:
    """Verify login when no 2FA is detected."""
    logger.debug("Checking login status directly (no 2FA detected)...")
    login_check_result = login_status(session_manager, disable_ui_fallback=False)

    if login_check_result is True:
        print("\n✓ Login successful!")
        # Save cookies after successful login
        _save_login_cookies(session_manager)
        # CRITICAL FIX: Sync browser cookies to API requests session
        session_manager.sync_cookies_to_requests()
        return "LOGIN_SUCCEEDED"

    if login_check_result is False:
        print("\n✗ Login failed. Please check your credentials.")
        logger.error("Direct login check failed. Checking for error messages again...")

        # Check for errors again
        error_result = _check_for_login_errors(driver)
        if error_result:
            return error_result

        # Check if still on login page
        try:
            if driver.current_url.startswith(signin_url):
                logger.error("Login failed: Still on login page (post-check), no error message found.")
                return "LOGIN_FAILED_STUCK_ON_LOGIN"
            logger.error("Login failed: Status False, no 2FA, no error msg, not on login page.")
            return "LOGIN_FAILED_UNKNOWN"
        except WebDriverException:
            logger.error("Login failed: Status False, WebDriverException getting URL.")
            return "LOGIN_FAILED_WEBDRIVER"

    logger.error("Login failed: Critical error during final login status check.")
    return "LOGIN_FAILED_STATUS_CHECK_ERROR"


# Login Main Function
def _debug_log_page_errors(driver: Any) -> None:
    """Log any visible error messages on the page."""
    try:
        error_elements = driver.find_elements(By.CSS_SELECTOR, GENERIC_ERROR_ELEMENTS_SELECTOR)
        if error_elements:
            for elem in error_elements[:3]:
                if elem.is_displayed():
                    error_text = elem.text.strip()
                    if error_text:
                        logger.warning(f"[DEBUG] Visible error on page: {error_text}")
    except Exception as e:
        logger.debug(f"[DEBUG] Error checking for error messages: {e}")


def _debug_log_post_credentials_state(driver: Any) -> None:
    """Log page state after credential submission."""
    try:
        current_url = driver.current_url
        page_title = driver.title
        logger.info(f"[DEBUG] After credentials - URL: {current_url}")
        logger.info(f"[DEBUG] After credentials - Page title: {page_title}")
        _debug_log_page_errors(driver)
    except Exception as e:
        logger.debug(f"[DEBUG] Error capturing page state: {e}")


def _handle_credentials_entry(driver: Any) -> str | None:
    """Handle credential entry and check for errors. Returns error status if failed."""
    if not enter_creds(driver):
        logger.error("Failed during credential entry or submission.")
        error_result = _check_for_login_errors(driver)
        return error_result if error_result else "LOGIN_FAILED_CREDS_ENTRY"
    return None


def _execute_login_flow(
    driver: Any,
    session_manager: "SessionManager",
    signin_url: str,
) -> str:
    """Execute the main login flow. Returns status string."""
    # Navigate to sign-in page
    nav_result = _navigate_to_signin(driver, session_manager, signin_url)
    if nav_result:
        return nav_result

    # Handle consent banner
    if not consent(driver):
        logger.warning("Failed to handle consent banner, login might be impacted.")

    # Enter credentials
    creds_error = _handle_credentials_entry(driver)
    if creds_error:
        return creds_error

    # Wait for page change and log state
    logger.debug("Credentials submitted. Waiting for potential page change...")
    time.sleep(random.uniform(3.0, 5.0))
    _debug_log_post_credentials_state(driver)

    # Check for 2FA
    two_fa_present, early_result = _detect_2fa_page(driver)
    if early_result:
        return early_result

    # Handle 2FA or verify login
    if two_fa_present:
        return _handle_2fa_flow(session_manager)
    return _verify_login_no_2fa(driver, session_manager, signin_url)


def _handle_login_exception(e: Exception, driver: Any) -> str:
    """Handle exceptions during login. Returns error status string."""
    if isinstance(e, TimeoutException):
        logger.error(f"Timeout during login process: {e}", exc_info=False)
        return "LOGIN_FAILED_TIMEOUT"
    if isinstance(e, WebDriverException):
        logger.error(f"WebDriverException during login: {e}", exc_info=False)
        if not is_browser_open(driver):
            logger.error("Session became invalid during login.")
        return "LOGIN_FAILED_WEBDRIVER"
    logger.error(f"An unexpected error occurred during login: {e}", exc_info=True)
    return "LOGIN_FAILED_UNEXPECTED"


def log_in(session_manager: "SessionManager") -> str:
    """
    Automates the login process: navigates to signin, handles consent,
    enters credentials, handles 2FA, and verifies final login status.

    Returns:
        A status string indicating success ("LOGIN_SUCCEEDED") or failure type.
    """
    driver = session_manager.browser_manager.driver
    if not driver:
        logger.error("Login failed: WebDriver not available in SessionManager.")
        return "LOGIN_ERROR_NO_DRIVER"

    # Check if already logged in
    initial_result = _check_initial_login_status(session_manager)
    if initial_result:
        return initial_result

    signin_url = urljoin(config_schema.api.base_url, "account/signin")

    try:
        return _execute_login_flow(driver, session_manager, signin_url)
    except Exception as e:
        return _handle_login_exception(e, driver)


# End of log_in

# ----------------------------------------------------------------------------
# Login Status Check Functions
# ----------------------------------------------------------------------------


def _validate_login_status_inputs(session_manager: "SessionManager") -> bool | None:
    """Validate session manager for login status check."""
    if not hasattr(session_manager, 'is_sess_valid'):
        logger.error(f"Invalid argument: Expected SessionManager-like object, got {type(session_manager)}.")
        return None

    # Check for browser access (either via browser_manager or direct driver)
    has_browser_manager = hasattr(session_manager, 'browser_manager')
    has_driver = hasattr(session_manager, 'driver')

    if not (has_browser_manager or has_driver):
        logger.error(
            f"Invalid argument: Expected SessionManager-like object with browser access, got {type(session_manager)}."
        )
        return None

    if not session_manager.is_sess_valid():
        logger.debug("Session is invalid, user cannot be logged in.")
        return False

    # Check driver availability
    driver = None
    if has_browser_manager:
        driver = session_manager.browser_manager.driver
    elif has_driver:
        driver = session_manager.driver

    if driver is None:
        logger.error("Login status check: Driver is None within SessionManager.")
        return None

    return True  # Valid


def _perform_api_login_check(session_manager: "SessionManager") -> bool | None:
    """Perform API-based login status check."""
    logger.debug("Performing primary API-based login status check...")
    try:
        session_manager.sync_cookies_to_requests()
        api_check_result = session_manager.api_manager.verify_api_login_status()

        if api_check_result is True:
            logger.debug("API login check confirmed user is logged in.")
            return True
        if api_check_result is False:
            logger.debug("API login check confirmed user is NOT logged in.")
            return False

        logger.warning("API login check returned ambiguous result (None).")
        return None
    except Exception as e:
        logger.error(f"Exception during API login check: {e}", exc_info=True)
        return None


def _check_ui_login_indicators(driver: Any) -> bool | None:
    """Check UI for login indicators."""
    # Check for logged-in element
    logged_in_selector = CONFIRMED_LOGGED_IN_SELECTOR
    logger.debug(f"Checking for logged-in indicator: '{logged_in_selector}'")
    ui_element_present = is_elem_there(driver, logged_in_selector, By.CSS_SELECTOR, wait=3)

    if ui_element_present:
        logger.debug("UI check: Logged-in indicator found. User is logged in.")
        return True

    # Check for login button
    login_button_selector = LOG_IN_BUTTON_SELECTOR
    logger.debug(f"Checking for login button: '{login_button_selector}'")
    login_button_present = is_elem_there(driver, login_button_selector, By.CSS_SELECTOR, wait=3)

    if login_button_present:
        logger.debug("UI check: Login button found. User is NOT logged in.")
        return False

    return None  # Inconclusive


def _perform_ui_login_check_with_navigation(driver: Any) -> bool:
    """Perform UI login check with navigation fallback."""
    result = _check_ui_login_indicators(driver)
    if result is not None:
        return result

    # Navigate to base URL for clearer check
    logger.debug("UI check inconclusive. Navigating to base URL for clearer check...")
    try:
        current_url = driver.current_url
        base_url = config_schema.api.base_url

        if not current_url.startswith(base_url):
            driver.get(base_url)
            time.sleep(2)

            result = _check_ui_login_indicators(driver)
            if result is not None:
                return result
    except Exception as nav_e:
        logger.warning(f"Error during navigation for secondary UI check: {nav_e}")

    # Default to False for security
    logger.debug("UI check still inconclusive after all checks. Defaulting to NOT logged in for security.")
    return False


# Login Status Check Function
@retry(max_retries=2)
def login_status(session_manager: "SessionManager", disable_ui_fallback: bool = False) -> bool | None:
    """
    Checks if the user is currently logged in. Prioritizes API check, with optional UI fallback.

    Args:
        session_manager: The session manager instance
        disable_ui_fallback: If True, only use API check and never fall back to UI check

    Returns:
        True if logged in, False if not logged in, None if the check fails critically.
    """
    # Validate inputs
    validation_result = _validate_login_status_inputs(session_manager)
    if validation_result is not True:
        return validation_result

    driver = None
    if hasattr(session_manager, 'browser_manager'):
        driver = session_manager.browser_manager.driver
    elif hasattr(session_manager, 'driver'):
        driver = session_manager.driver

    # Perform API check
    api_check_result = _perform_api_login_check(session_manager)
    if api_check_result is not None:
        return api_check_result

    # If API check is ambiguous and UI fallback is disabled, return None
    if disable_ui_fallback:
        logger.warning("API login check was ambiguous and UI fallback is disabled. Status unknown.")
        return None

    # Perform UI check
    logger.debug("Performing fallback UI-based login status check...")
    try:
        return _perform_ui_login_check_with_navigation(driver)
    except WebDriverException as e:
        logger.error(f"WebDriverException during UI login_status check: {e}")
        if not is_browser_open(driver):
            logger.warning("Browser appears to be closed. Closing session.")
            session_manager.close_sess()
        return None
    except Exception as e:
        logger.error(f"Unexpected error during UI login_status check: {e}", exc_info=True)
        return None


# End of login_status


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


def _test_login_status_function() -> None:
    """Test login_status prioritizes API check with UI fallback when needed."""
    import sys
    from unittest.mock import MagicMock, patch

    session_manager = MagicMock()
    session_manager.is_sess_valid.return_value = True
    session_manager.browser_manager.driver = MagicMock()
    session_manager.sync_cookies_to_requests = MagicMock()
    session_manager.api_manager.verify_api_login_status.return_value = True

    result = login_status(session_manager, disable_ui_fallback=True)
    assert result is True, "login_status should return API result when definitive"

    session_manager.api_manager.verify_api_login_status.return_value = None
    with patch.object(
        sys.modules[__name__],
        "_perform_ui_login_check_with_navigation",
        return_value=False,
    ) as mock_ui_check:
        result = login_status(session_manager, disable_ui_fallback=False)
    mock_ui_check.assert_called_once()
    assert result is False, "login_status should return UI fallback result when API ambiguous"


def module_tests() -> bool:
    """Run login module tests using standardized TestSuite format."""
    from testing.test_framework import TestSuite

    suite = TestSuite("Login & Authentication", "browser/login.py")

    suite.run_test(
        "Login Status Function",
        _test_login_status_function,
        "Verify login_status prioritizes API check with UI fallback",
    )

    return suite.finish_suite()


# Standardized test runner (recommended pattern)
run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
