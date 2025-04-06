#!/usr/bin/env python3

# Test2.py 

import logging
import sys
import os
import json
import time
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urljoin, unquote
import requests
import cloudscraper
from requests import Response as RequestsResponse
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.common.exceptions import WebDriverException, NoSuchCookieException, TimeoutException
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from dotenv import load_dotenv

# Configuration imports
try:
    from utils import SessionManager
    from config import config_instance
    from logging_config import setup_logging
except ImportError as e:
    print(f"Critical import error: {e}", file=sys.stderr)
    sys.exit(1)

# Constants
DEFAULT_TIMEOUT = 60
REQUEST_TIMEOUT = 120
NAVIGATION_DELAY = 5  # Seconds to wait after page load
LATEST_CHROME_VERSION = "125"
CSRF_COOKIE_NAMES = ("_dnamatches-matchlistui-x-csrf-token", "_csrf")

# User Agent Configuration
USER_AGENT = (
    f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    f"(KHTML, like Gecko) Chrome/{LATEST_CHROME_VERSION}.0.0.0 Safari/537.36"
)
SEC_CH_UA = (
    f'"Google Chrome";v="{LATEST_CHROME_VERSION}", '
    '"Not-A.Brand";v="8", "Chromium";v="{LATEST_CHROME_VERSION}"'
)

class TestConfiguration:
    """Container for test configuration parameters"""
    def __init__(self):
        load_dotenv()
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.current_page = 1
        self.base_url = config_instance.BASE_URL
        self.match_list_path = "/discoveryui-matches/list/"
        self.api_path_template = "/discoveryui-matches/parents/list/api/matchList/{uuid}?currentPage={page}"

def configure_logging(config: TestConfiguration) -> logging.Logger:
    """Initialize and return configured logger instance"""
    try:
        logger = setup_logging(log_level=config.log_level)
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.warning("Using fallback logging configuration: %s", e)
    return logger

def initialize_session(logger: logging.Logger) -> Tuple[Optional[SessionManager], Optional[WebDriver]]:
    """Initialize and validate session manager with WebDriver"""
    try:
        logger.info("Initializing SessionManager")
        session_manager = SessionManager()
        logger.debug("SessionManager instance created: ID=%s", id(session_manager))
        
        logger.info("Starting WebDriver session")
        start_ok, driver = session_manager.start_sess(action_name="Cloudscraper+UBE Test")
        
        if not start_ok or not driver:
            logger.error("Session initialization failed")
            return None, None
            
        return session_manager, driver
        
    except Exception as e:
        logger.critical("Session initialization failed: %s", e, exc_info=True)
        return None, None

def handle_webdriver_configuration(driver: WebDriver, logger: logging.Logger) -> None:
    """Configure WebDriver settings including User Agent"""
    try:
        logger.info("Configuring WebDriver User Agent")
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": USER_AGENT})
        current_ua = driver.execute_script("return navigator.userAgent;")
        
        if USER_AGENT not in current_ua:
            logger.warning("UserAgent override may have failed. Current UA: %s", current_ua)
            
    except WebDriverException as e:
        logger.warning("WebDriver configuration partially failed: %s", e)

def navigate_to_target_page(driver: WebDriver, url: str, logger: logging.Logger) -> bool:
    """Navigate to target page and validate successful load"""
    try:
        logger.info("Navigating to %s", url)
        driver.get(url)
        
        # Wait for page elements
        WebDriverWait(driver, DEFAULT_TIMEOUT).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, "ui-custom[type='match-entry']"))
        )
        
        logger.debug("Navigation successful, waiting %ds for stabilization", NAVIGATION_DELAY)
        time.sleep(NAVIGATION_DELAY)
        return True
        
    except TimeoutException:
        logger.error("Timeout waiting for page elements")
        return False
    except WebDriverException as e:
        logger.error("Navigation failed: %s", e)
        return False

def retrieve_csrf_token(driver: WebDriver, logger: logging.Logger) -> Optional[str]:
    """Retrieve CSRF token from available cookie sources"""
    for cookie_name in CSRF_COOKIE_NAMES:
        try:
            cookie = driver.get_cookie(cookie_name)
            if cookie and cookie.get('value'):
                token = unquote(cookie['value']).split('|')[0]
                logger.info("Found CSRF token in %s: %s...", cookie_name, token[:10])
                return token
        except NoSuchCookieException:
            continue
        except Exception as e:
            logger.warning("Error retrieving %s cookie: %s", cookie_name, e)
    
    logger.warning("No valid CSRF cookies found")
    return None

def prepare_api_request(
    session_manager: SessionManager,
    csrf_token: str,
    config: TestConfiguration
) -> Tuple[str, Dict[str, str]]:
    """Prepare API request URL and headers"""
    api_url = urljoin(
        config.base_url,
        config.api_path_template.format(uuid=session_manager.my_uuid, page=config.current_page)
    )
    
    headers = {
        "User-Agent": USER_AGENT,
        "accept": "application/json",
        "Referer": urljoin(config.base_url, config.match_list_path),
        "x-csrf-token": csrf_token,
        "sec-ch-ua": SEC_CH_UA,
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    }
    
    return api_url, headers

def execute_api_request(
    url: str,
    headers: Dict[str, str],
    cookies: List[Dict[str, str]],
    logger: logging.Logger
) -> Optional[RequestsResponse]:
    """Execute API request using cloudscraper with provided configuration"""
    try:
        scraper = cloudscraper.create_scraper(
            delay=10,
            browser={'custom': USER_AGENT}
        )
        
        # Set cookies from WebDriver
        for cookie in cookies:
            scraper.cookies.set(
                cookie['name'], cookie['value'],
                domain=cookie.get('domain'),
                path=cookie.get('path', '/')
            )
        
        response = scraper.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=False
        )
        
        logger.info("API response: %d %s", response.status_code, response.reason)
        return response
        
    except requests.RequestException as e:
        logger.error("Request failed: %s", e)
    except Exception as e:
        logger.error("An error occurred during the request: %s", e)
    return None

def analyze_response(response: RequestsResponse, logger: logging.Logger) -> bool:
    """Analyze and validate API response"""
    if response.status_code != 200:
        logger.error("API request failed with status: %d", response.status_code)
        return False
        
    try:
        data = response.json()
        logger.info("Received %d matches (Page %d/%d)",
                   len(data.get('matchList', [])),
                   data.get('currentPage', 0),
                   data.get('totalPages', 0))
        return True
    except json.JSONDecodeError:
        logger.error("Invalid JSON response")
    return False

def test_matchlist() -> bool:
    """Main test execution flow"""
    config = TestConfiguration()
    logger = configure_logging(config)
    
    logger.info("=== Starting MatchList API Test ===")
    
    session_manager, driver = initialize_session(logger)
    if not session_manager or not driver:
        logger.error("Aborting test due to initialization failure")
        return False

    success = False
    try:
        # WebDriver configuration
        handle_webdriver_configuration(driver, logger)
        
        # Page navigation
        target_url = urljoin(config.base_url, config.match_list_path)
        if not navigate_to_target_page(driver, target_url, logger):
            return False
            
        # Token retrieval
        csrf_token = retrieve_csrf_token(driver, logger) or session_manager.csrf_token
        if not csrf_token:
            logger.error("No valid CSRF token available")
            return False
            
        # Request preparation
        api_url, headers = prepare_api_request(session_manager, csrf_token, config)
        cookies = driver.get_cookies()
        
        # Request execution
        response = execute_api_request(api_url, headers, cookies, logger)
        if not response:
            return False
            
        # Response analysis
        success = analyze_response(response, logger)
        
    except Exception as e:
        logger.critical("Unhandled exception in test flow: %s", e, exc_info=True)
    finally:
        if session_manager:
            logger.info("Cleaning up session resources")
            session_manager.close_sess()
            
    logger.info("=== Test %s ===", "PASSED" if success else "FAILED")
    return success

if __name__ == "__main__":
    sys.exit(0 if test_matchlist() else 1)