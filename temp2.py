# test.py

# --- Keep all existing imports ---
import logging
import sys
import os
import json
import time
from typing import Optional, Dict, Any, List
from urllib.parse import urljoin, unquote
import requests
import cloudscraper
from requests import Response as RequestsResponse
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.common.exceptions import (
    WebDriverException,
    NoSuchCookieException,
    JavascriptException,
    TimeoutException,
)
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from dotenv import load_dotenv

try:
    from utils import (
        SessionManager,
        nav_to_page,
    )  # Assuming nav_to_page is needed standalone
    from config import config_instance
    from logging_config import setup_logging, logger
except ImportError as e:
    print(f"Error importing necessary modules: {e}", file=sys.stderr)
    print(
        "Ensure utils.py, config.py, logging_config.py are accessible.", file=sys.stderr
    )
    sys.exit(1)

# --- Configuration ---
load_dotenv()
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
CURRENT_PAGE_TO_TEST = 1
UI_TIMEOUT = 60
REQUEST_TIMEOUT = 120

# --- Consistent UA & Headers ---
LATEST_CHROME_VERSION = "125"
CONSISTENT_USER_AGENT = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{LATEST_CHROME_VERSION}.0.0.0 Safari/537.36"
CONSISTENT_PLATFORM = '"Windows"'
CONSISTENT_SEC_CH_UA = f'"Google Chrome";v="{LATEST_CHROME_VERSION}", "Not-A.Brand";v="8", "Chromium";v="{LATEST_CHROME_VERSION}"'

# Cookie names
ALT_CSRF_COOKIE_NAME = "_dnamatches-matchlistui-x-csrf-token"
STANDARD_CSRF_COOKIE_NAME = "_csrf"


# --- FULL test_matchlist FUNCTION ---
def test_matchlist():
    """
    Performs test using cloudscraper after navigating to UI page:
    1. Starts session (which should store UBE header).
    2. Navigates to match list UI page.
    3. Waits for match entries.
    4. Retrieves page-specific CSRF token cookie.
    5. Retrieves the stored ancestry-context-ube header from SessionManager.
    6. Calls API via cloudscraper with specific headers & cookies.
    """
    global logger
    try:
        logger = setup_logging(log_level=LOG_LEVEL)
    except NameError:
        logging.basicConfig(level=logging.getLevelName(LOG_LEVEL))
        logger = logging.getLogger(__name__)
        logger.warning("setup_logging function not found, using basicConfig.")
    except Exception as log_err:
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        logger.error(f"Error setting up logging: {log_err}", exc_info=True)

    logger.info(
        f"--- Starting Standalone MatchList API Test (Cloudscraper + UBE Header V2 Debug) ---"
    )  # Added V2 Debug

    session_manager: Optional[SessionManager] = None
    success = False
    api_response: Optional[RequestsResponse] = None  # Store the raw response object
    driver: Optional[WebDriver] = None

    try:
        # 1. Initialize SessionManager
        logger.info("Initializing SessionManager...")
        session_manager = SessionManager()
        logger.info(
            f"SessionManager instance created in test.py: ID={id(session_manager)}"
        )
        # Explicitly initialize attribute here for clarity before start_sess modifies it
        session_manager.last_ancestry_context_ube = None
        logger.info("SessionManager initialized.")

        # 2. Start Session & Set UA
        logger.info("Starting WebDriver session and logging in...")
        start_ok = session_manager.start_sess(
            action_name="Cloudscraper+UBE Test V3 Debug"
        )  # New action name
        driver = session_manager.driver if start_ok else None
        if not start_ok or not driver:
            logger.critical("SessionManager.start_sess() FAILED. Aborting test.")
            if session_manager:
                session_manager.close_sess()  # Ensure cleanup
            return False

        # === ADDED DEBUGGING BLOCK ===
        logger.info("--- Post start_sess Attribute Check ---")
        logger.info(
            f"Checking SessionManager instance: ID={id(session_manager)}"
        )  # Check ID again
        ube_check_value = getattr(
            session_manager, "last_ancestry_context_ube", "!!! ATTRIBUTE NOT FOUND !!!"
        )
        logger.info(
            f"Value of last_ancestry_context_ube: {ube_check_value[:30] if isinstance(ube_check_value, str) else ube_check_value}..."
        )
        logger.info(f"Type of last_ancestry_context_ube: {type(ube_check_value)}")
        logger.info(f"UUID value from manager: {session_manager.my_uuid}")
        logger.info(
            f"CSRF value from manager: {session_manager.csrf_token[:10] if session_manager.csrf_token else None}..."
        )
        logger.info("--- End Post start_sess Check ---")
        # ===========================

        # Apply CDP override (Best effort)
        try:
            logger.info(
                f"Attempting to override WebDriver UA via CDP to: {CONSISTENT_USER_AGENT}"
            )
            driver.execute_cdp_cmd(
                "Network.setUserAgentOverride", {"userAgent": CONSISTENT_USER_AGENT}
            )
            logger.info("WebDriver UA override attempted via CDP.")
            current_ua = driver.execute_script("return navigator.userAgent;")
            logger.debug(f"WebDriver UA after override attempt: {current_ua}")
            if CONSISTENT_USER_AGENT not in current_ua:
                logger.warning(
                    f"WebDriver UA may not have been overridden successfully. Current: {current_ua}"
                )
        except Exception as cdp_err:
            logger.warning(
                f"Could not set User-Agent via CDP: {cdp_err}", exc_info=False
            )

        logger.info("Session started and login successful.")

        # 3. Get User UUID (Check after successful start)
        my_uuid = session_manager.my_uuid
        if not my_uuid:
            # Should not happen if start_sess succeeded and previous check passed, but safety first
            logger.critical("User UUID check failed after start_sess. Aborting.")
            return False
        logger.info(f"User UUID confirmed: {my_uuid}")

        # --- Get UBE header directly after start_sess ---
        logger.info("Retrieving last used UBE header from SessionManager...")
        last_ube_header = getattr(session_manager, "last_ancestry_context_ube", None)

        if not last_ube_header:
            logger.critical(
                "FAILED: last_ancestry_context_ube is None after successful start_sess. Check attribute storage/retrieval."
            )
            try:
                logger.debug(f"Attributes on session_manager: {dir(session_manager)}")
            except:
                pass
            return False
        logger.info(
            f"Using ancestry-context-ube header from SessionManager: {last_ube_header[:20]}..."
        )

        # 4. Navigate to the Match List UI Page & Wait
        match_list_ui_url = urljoin(
            config_instance.BASE_URL, "/discoveryui-matches/list/"
        )
        logger.info(f"Navigating WebDriver to Match List UI: {match_list_ui_url}")
        nav_ok = False
        try:
            driver.get(match_list_ui_url)
            wait_selector = "ui-custom[type='match-entry']"
            logger.debug(
                f"Waiting up to {UI_TIMEOUT}s for selector to be visible: '{wait_selector}'"
            )
            wait = WebDriverWait(driver, UI_TIMEOUT)
            wait.until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, wait_selector))
            )
            logger.info(
                "Successfully navigated to Match List UI page and found match entries."
            )
            logger.debug("Pausing for 5 seconds after matches visible...")
            time.sleep(5)
            logger.debug("Pause complete.")
            nav_ok = True
        except TimeoutException:
            logger.error(
                f"Timed out waiting for Match List UI page elements ('{wait_selector}') to be visible.",
                exc_info=False,
            )
            try:
                logger.debug(
                    "Page source at timeout:\n" + driver.page_source[:2000] + "..."
                )
            except Exception:
                pass
        except Exception as nav_err:
            logger.error(
                f"Unexpected error navigating to or validating Match List UI page: {nav_err}",
                exc_info=True,
            )

        if not nav_ok:
            logger.error(
                "Aborting test due to failure navigating to or loading the Match List UI."
            )
            return False

        # 5. Get CSRF Token (Check cookies after successful navigation)
        logger.debug("Attempting to retrieve cookies after UI navigation and wait...")
        try:
            all_cookies_after_nav = driver.get_cookies()
            logger.debug(
                f"All cookies found ({len(all_cookies_after_nav)}): {[c['name'] for c in all_cookies_after_nav]}"
            )
        except Exception as get_cookie_err:
            logger.error(f"Error retrieving cookies after navigation: {get_cookie_err}")
            all_cookies_after_nav = []

        csrf_token_to_use = None
        csrf_token_source = "None"
        # Try page-specific cookie
        try:
            cookie_obj = driver.get_cookie(ALT_CSRF_COOKIE_NAME)
            if cookie_obj and cookie_obj.get("value"):
                page_specific_csrf = unquote(cookie_obj["value"])
                if "|" in page_specific_csrf:
                    page_specific_csrf = page_specific_csrf.split("|")[0]
                csrf_token_to_use = page_specific_csrf
                csrf_token_source = f"Cookie: {ALT_CSRF_COOKIE_NAME}"
                logger.info(
                    f"Found page-specific CSRF token from '{ALT_CSRF_COOKIE_NAME}': {csrf_token_to_use[:10]}..."
                )
            else:
                logger.debug(
                    f"Page-specific CSRF cookie '{ALT_CSRF_COOKIE_NAME}' not found or empty."
                )
        except NoSuchCookieException:
            logger.debug(
                f"Page-specific CSRF cookie '{ALT_CSRF_COOKIE_NAME}' not found."
            )
        except Exception as cookie_err:
            logger.warning(
                f"Error retrieving page-specific CSRF cookie '{ALT_CSRF_COOKIE_NAME}': {cookie_err}"
            )

        # Try standard cookie if page-specific not found
        if not csrf_token_to_use:
            logger.info(f"Checking standard '{STANDARD_CSRF_COOKIE_NAME}' cookie.")
            try:
                cookie_obj = driver.get_cookie(STANDARD_CSRF_COOKIE_NAME)
                if cookie_obj and cookie_obj.get("value"):
                    csrf_token_to_use = cookie_obj["value"]
                    csrf_token_source = f"Cookie: {STANDARD_CSRF_COOKIE_NAME}"
                    logger.info(
                        f"Found standard CSRF token from '{STANDARD_CSRF_COOKIE_NAME}': {csrf_token_to_use[:10]}..."
                    )
                else:
                    logger.warning(
                        f"Standard CSRF cookie '{STANDARD_CSRF_COOKIE_NAME}' not found or empty."
                    )
            except NoSuchCookieException:
                logger.warning(
                    f"Standard CSRF cookie '{STANDARD_CSRF_COOKIE_NAME}' not found."
                )
            except Exception as cookie_err:
                logger.warning(
                    f"Error retrieving standard CSRF cookie '{STANDARD_CSRF_COOKIE_NAME}': {cookie_err}"
                )

        # Fallback to API-retrieved token stored in SessionManager
        if not csrf_token_to_use:
            logger.warning(
                "No CSRF token found in WebDriver cookies. Using token stored in SessionManager (from API fallback)."
            )
            csrf_token_to_use = session_manager.csrf_token
            if csrf_token_to_use:
                csrf_token_source = "SessionManager Cache (API Fallback)"
            else:
                logger.critical(
                    "Failed to obtain any CSRF token (from cookies or SessionManager cache). Aborting."
                )
                return False
        logger.info(
            f"Using CSRF token: {csrf_token_to_use[:10]}... (Source: {csrf_token_source})"
        )

        # 6. Construct Target URL
        target_api_url = urljoin(
            config_instance.BASE_URL,
            f"discoveryui-matches/parents/list/api/matchList/{my_uuid}?currentPage={CURRENT_PAGE_TO_TEST}",
        )
        logger.info(f"Target API URL for cloudscraper: {target_api_url}")

        # 7. Construct Explicit Headers for Cloudscraper Request
        referer_url = match_list_ui_url

        explicit_headers_for_request = {
            "User-Agent": CONSISTENT_USER_AGENT,
            "accept": "application/json",
            "Referer": referer_url,
            "x-csrf-token": csrf_token_to_use,
            "ancestry-context-ube": last_ube_header,
            "sec-ch-ua": CONSISTENT_SEC_CH_UA,
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": CONSISTENT_PLATFORM,
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
        logger.info(
            f"Explicit headers being sent: {list(explicit_headers_for_request.keys())}"
        )
        logger.debug(
            f"Headers dictionary preview: {json.dumps({k: (v[:20]+'...' if isinstance(v, str) and len(v)>50 else v) for k, v in explicit_headers_for_request.items()}, indent=2)}"
        )

        # 8. Get Cookies from WebDriver again
        logger.info("Retrieving final cookies from WebDriver...")
        webdriver_cookies = driver.get_cookies()
        if not webdriver_cookies:
            logger.error(
                "Failed to retrieve cookies from WebDriver before making request."
            )
            return False
        logger.info(f"Retrieved {len(webdriver_cookies)} cookies from WebDriver.")

        # 9. Create cloudscraper instance and make request
        logger.info("Creating cloudscraper instance...")
        scraper = cloudscraper.create_scraper(delay=5)

        logger.info("Calling API using cloudscraper instance with explicit headers...")
        try:
            logger.info("Transferring WebDriver cookies to cloudscraper session...")
            for cookie in webdriver_cookies:
                scraper.cookies.set(
                    cookie["name"],
                    cookie["value"],
                    domain=cookie.get("domain"),
                    path=cookie.get("path", "/"),
                )
            logger.info("Finished transferring cookies.")

            api_response = scraper.request(
                method="GET",
                url=target_api_url,
                headers=explicit_headers_for_request,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=False,
            )
            logger.info(
                f"<-- Cloudscraper Response Status: {api_response.status_code} {api_response.reason}"
            )
            logger.info(
                f"<-- Cloudscraper Response URL (Final URL): {api_response.url}"
            )
            logger.debug(
                f"<-- Cloudscraper Response Headers:\n{json.dumps(dict(api_response.headers), indent=2)}"
            )

        except requests.exceptions.RequestException as req_exc:
            logger.error(f"Cloudscraper requests call failed: {req_exc}", exc_info=True)
            api_response = getattr(req_exc, "response", None)
        except cloudscraper.exceptions.CloudflareException as cf_exc:
            logger.error(
                f"Cloudscraper detected Cloudflare challenge or issue: {cf_exc}",
                exc_info=True,
            )
            api_response = getattr(cf_exc, "response", None)
        except Exception as cs_exc:
            logger.error(f"Cloudscraper general error: {cs_exc}", exc_info=True)
            api_response = None

        # 10. Analyze Response
        logger.info("--- API Response Analysis ---")
        if api_response is not None:
            content_type = api_response.headers.get("content-type", "").lower()
            if api_response.status_code == 200 and "application/json" in content_type:
                try:
                    response_data = api_response.json()
                    logger.info(f"SUCCESS: API returned 200 OK and valid JSON.")
                    total_pages = response_data.get("totalPages")
                    current_page = response_data.get("currentPage")
                    total_matches = response_data.get("totalMatchCount")
                    matches = response_data.get("matchList", [])
                    logger.info(
                        f"Data Preview - Total Pages: {total_pages}, Current Page: {current_page}, Total Matches: {total_matches}"
                    )
                    logger.info(f"Matches received: {len(matches)}")
                    if matches:
                        logger.debug(
                            f"First match data preview:\n{json.dumps(matches[0], indent=2)}"
                        )
                    success = True
                except json.JSONDecodeError as json_err:
                    logger.error(
                        f"FAILED: API returned 200 OK but JSON decoding failed: {json_err}",
                        exc_info=True,
                    )
                    logger.debug(
                        f"Response text preview: {api_response.text[:1000]}..."
                    )
                    success = False
            elif api_response.status_code == 303:
                logger.error(
                    f"FAILED: API returned status {api_response.status_code} {api_response.reason}. Redirect detected."
                )
                location = api_response.headers.get("Location")
                logger.error(f"Redirect Location: {location}")
                logger.debug("Cookies SENT (from prepared request's Cookie header):")
                if (
                    hasattr(api_response, "request")
                    and api_response.request is not None
                    and hasattr(api_response.request, "headers")
                ):
                    sent_cookie_header = api_response.request.headers.get(
                        "Cookie", "Not available"
                    )
                    logger.debug(
                        f"{sent_cookie_header[:100]}...{sent_cookie_header[-100:]}"
                        if len(sent_cookie_header) > 200
                        else sent_cookie_header
                    )
                logger.debug("Request Headers SENT (from prepared request):")
                if (
                    hasattr(api_response, "request")
                    and api_response.request is not None
                ):
                    for k, v in api_response.request.headers.items():
                        if k.lower() != "cookie":
                            logger.debug(f"  {k}: {v}")
                success = False
            else:
                logger.error(
                    f"FAILED: API returned non-OK status: {api_response.status_code} {api_response.reason}"
                )
                logger.error(f"Content-Type: {content_type}")
                try:
                    logger.error(
                        f"Response Text Preview: {api_response.text[:1000]}..."
                    )
                except Exception:
                    pass
                success = False
        else:
            logger.error(
                "FAILED: No response object received from cloudscraper call (exception likely occurred)."
            )
            success = False
        logger.info("--- End API Response Analysis ---")

    except Exception as e:
        logger.critical(
            f"An unexpected error occurred during the test setup or execution: {e}",
            exc_info=True,
        )
        success = False
    finally:
        # 11. Cleanup
        if session_manager and driver:
            logger.info("Closing session manager (includes WebDriver)...")
            session_manager.close_sess()
            logger.info("Session closed.")
        elif driver:
            logger.info("Closing WebDriver instance directly...")
            try:
                driver.quit()
            except Exception as q_err:
                logger.error(f"Error quitting WebDriver: {q_err}")
        logger.info(
            f"--- Standalone MatchList API Test (Cloudscraper + UBE Header V2 Debug) {'PASSED' if success else 'FAILED'} ---"
        )

    return success


if __name__ == "__main__":
    test_matchlist()
