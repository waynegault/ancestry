#!/usr/bin/env python3

# config.py

import os
import logging
from typing import Optional, Dict, Any, Tuple, List
from dotenv import load_dotenv
from selenium.webdriver.support.wait import WebDriverWait
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse

load_dotenv()
logger = logging.getLogger("logger")


class BaseConfig:
    """Base configuration class with helper methods."""

    def _get_env_var(self, key: str) -> Optional[str]:
        value = os.getenv(key)
        # Removed debug log for brevity, caller handles defaults
        return value

    # End of _get_env_var

    def _get_int_env(self, key: str, default: int) -> int:
        value_str = self._get_env_var(key)
        if value_str is None:
            return default
        try:
            return int(value_str)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid int '{value_str}' for '{key}'. Using default: {default}"
            )
            return default

    # End of _get_int_env

    def _get_float_env(self, key: str, default: float) -> float:
        value_str = self._get_env_var(key)
        if value_str is None:
            return default
        try:
            return float(value_str)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid float '{value_str}' for '{key}'. Using default: {default}"
            )
            return default

    # End of _get_float_env

    def _get_string_env(self, key: str, default: str) -> str:
        value = self._get_env_var(key)
        if value is None:
            return default
        return str(value)

    # End of _get_string_env

    def _get_bool_env(self, key: str, default: bool) -> bool:
        value_str = self._get_env_var(key)
        if value_str is None:
            return default
        return value_str.lower() in ("true", "1", "t", "y", "yes")

    # End of _get_bool_env

    def _get_json_env(self, key: str, default: Any) -> Any:
        value_str = self._get_env_var(key)
        if value_str:
            try:
                return json.loads(value_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON '{key}': {e}. Using default: {default}")
                return default
        else:
            return default

    # End of _get_json_env

    def _get_path_env(self, key: str, default: Optional[str]) -> Optional[Path]:
        value_str = self._get_env_var(key)
        if value_str is None:
            if default is None:
                return None
            else:
                value_str = default
        try:
            path_obj = Path(value_str).resolve() if value_str else None
            return path_obj
        except Exception as e:
            logger.warning(f"Error resolving path '{value_str}' for '{key}': {e}")
            return Path(value_str) if value_str else None

    # End of _get_path_env


# End of BaseConfig


class Config_Class(BaseConfig):
    """Main configuration class loading settings."""

    # --- Constants / Fixed Settings ---
    INITIAL_DELAY: float = 0.5
    MAX_DELAY: float = 60.0
    BACKOFF_FACTOR: float = 1.8
    DECREASE_FACTOR: float = 0.98
    LOG_LEVEL: str = "INFO"
    RETRY_STATUS_CODES: Tuple[int, ...] = (429, 500, 502, 503, 504)
    DB_POOL_SIZE = 10
    MESSAGE_TRUNCATION_LENGTH: int = 100
    CHECK_JS_ERRORS_ACTN_6: bool = False
    USER_AGENTS: list[str] = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    ]

    # --- AI Context Control Defaults (Added) ---
    AI_CONTEXT_MESSAGES_COUNT: int = 7
    AI_CONTEXT_MESSAGE_MAX_WORDS: int = 500

    def __init__(self):
        self._load_values()

    # End of __init__

    def _load_values(self):
        logger.debug("Loading configuration settings...")
        # === Credentials & Identifiers ===
        self.ANCESTRY_USERNAME: str = self._get_string_env("ANCESTRY_USERNAME", "")
        self.ANCESTRY_PASSWORD: str = self._get_string_env("ANCESTRY_PASSWORD", "")
        self.TREE_NAME: str = self._get_string_env("TREE_NAME", "")
        self.MY_PROFILE_ID: Optional[str] = self._get_string_env("MY_PROFILE_ID", "")
        # === Paths & Files ===
        log_dir_name = self._get_string_env("LOG_DIR", "Logs")
        data_dir_name = self._get_string_env("DATA_DIR", "Data")
        cache_dir_name = self._get_string_env("CACHE_DIR", "Cache")
        self.LOG_DIR: Path = Path(log_dir_name)
        self.DATA_DIR: Path = Path(data_dir_name)
        self.CACHE_DIR: Path = Path(cache_dir_name)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.DATABASE_FILE: Path = self.DATA_DIR / self._get_string_env(
            "DATABASE_FILE", "ancestry_data.db"
        )
        self.GEDCOM_FILE_PATH: Optional[Path] = self._get_path_env(
            "GEDCOM_FILE_PATH", None
        )
        self.CACHE_DIR_PATH: Path = self.CACHE_DIR
        # === URLs ===
        self.BASE_URL: str = self._get_string_env(
            "BASE_URL", "https://www.ancestry.co.uk/"
        )
        self.ALTERNATIVE_API_URL: Optional[str] = self._get_env_var(
            "ALTERNATIVE_API_URL"
        )
        self.API_BASE_URL_PATH: str = self._get_string_env(
            "API_BASE_URL_PATH", "/api/v2/"
        )
        if self.BASE_URL and self.API_BASE_URL_PATH:
            self.API_BASE_URL = urljoin(self.BASE_URL, self.API_BASE_URL_PATH)
        else:
            self.API_BASE_URL = None
        # === Application Behavior ===
        self.APP_MODE: str = self._get_string_env("APP_MODE", "dry_run")
        self.MAX_PAGES: int = self._get_int_env("MAX_PAGES", 0)
        self.MAX_RETRIES: int = self._get_int_env("MAX_RETRIES", 5)
        self.MAX_INBOX: int = self._get_int_env("MAX_INBOX", 0)
        self.BATCH_SIZE: int = self._get_int_env("BATCH_SIZE", 50)
        # === Database ===
        self.DB_POOL_SIZE: int = self._get_int_env("DB_POOL_SIZE", self.DB_POOL_SIZE)
        # === Caching ===
        self.CACHE_TIMEOUT: int = self._get_int_env("CACHE_TIMEOUT", 3600)
        # --- Rate Limiter Values ---
        self.INITIAL_DELAY = self._get_float_env("INITIAL_DELAY", self.INITIAL_DELAY)
        self.MAX_DELAY = self._get_float_env("MAX_DELAY", self.MAX_DELAY)
        self.BACKOFF_FACTOR = self._get_float_env("BACKOFF_FACTOR", self.BACKOFF_FACTOR)
        self.DECREASE_FACTOR = self._get_float_env(
            "DECREASE_FACTOR", self.DECREASE_FACTOR
        )
        # --- RETRY_STATUS_CODES ---
        retry_codes_env = self._get_json_env(
            "RETRY_STATUS_CODES", self.RETRY_STATUS_CODES
        )
        if isinstance(retry_codes_env, (list, tuple)) and all(
            isinstance(code, int) for code in retry_codes_env
        ):
            self.RETRY_STATUS_CODES = tuple(retry_codes_env)
        else:
            logger.warning(
                f"RETRY_STATUS_CODES invalid/not set. Using default: {self.RETRY_STATUS_CODES}"
            )
        # === AI Configuration ===
        self.AI_PROVIDER: str = self._get_string_env("AI_PROVIDER", "").lower()
        self.DEEPSEEK_API_KEY: str = self._get_string_env("DEEPSEEK_API_KEY", "")
        self.DEEPSEEK_AI_MODEL: str = self._get_string_env(
            "DEEPSEEK_AI_MODEL", "deepseek-chat"
        )
        self.DEEPSEEK_AI_BASE_URL: Optional[str] = self._get_string_env(
            "DEEPSEEK_AI_BASE_URL", "https://api.deepseek.com"
        )
        self.GOOGLE_API_KEY: str = self._get_string_env("GOOGLE_API_KEY", "")
        self.GOOGLE_AI_MODEL: str = self._get_string_env(
            "GOOGLE_AI_MODEL", "gemini-1.5-flash-latest"
        )
        # === AI Context Control (Load from Env or use default) ===
        self.AI_CONTEXT_MESSAGES_COUNT = self._get_int_env(
            "AI_CONTEXT_MESSAGES_COUNT", self.AI_CONTEXT_MESSAGES_COUNT
        )
        self.AI_CONTEXT_MESSAGE_MAX_WORDS = self._get_int_env(
            "AI_CONTEXT_MESSAGE_MAX_WORDS", self.AI_CONTEXT_MESSAGE_MAX_WORDS
        )
        logger.info(
            f"AI Context: Last {self.AI_CONTEXT_MESSAGES_COUNT} messages, Max {self.AI_CONTEXT_MESSAGE_MAX_WORDS} words/msg."
        )
        # Log selected provider info
        logger.info(f"Selected AI Provider: '{self.AI_PROVIDER}'")
        if self.AI_PROVIDER == "deepseek":
            logger.info(
                f"  DeepSeek: Model='{self.DEEPSEEK_AI_MODEL}', BaseURL='{self.DEEPSEEK_AI_BASE_URL}', Key Loaded={'Yes' if self.DEEPSEEK_API_KEY else 'No'}"
            )
        elif self.AI_PROVIDER == "gemini":
            logger.info(
                f"  Google: Model='{self.GOOGLE_AI_MODEL}', Key Loaded={'Yes' if self.GOOGLE_API_KEY else 'No'}"
            )
        elif self.AI_PROVIDER:
            logger.warning(f"AI_PROVIDER '{self.AI_PROVIDER}' not recognized.")
        else:
            logger.warning("AI_PROVIDER not set. AI disabled.")
        # === API Headers ===
        parsed_base_url = urlparse(self.BASE_URL)
        origin_header_value = f"{parsed_base_url.scheme}://{parsed_base_url.netloc}"
        self.API_CONTEXTUAL_HEADERS: Dict[str, Dict[str, Optional[str]]] = {
            "CSRF Token API": {
                "Accept": "application/json",
                "Referer": urljoin(self.BASE_URL, "/discoveryui-matches/list/"),
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
            },
            "Get my profile_id": {
                "Accept": "application/json, text/plain, */*",
                "ancestry-clientpath": "p13n-js",
                "Referer": self.BASE_URL,
            },
            "Get UUID API": {"Accept": "application/json", "Referer": self.BASE_URL},
            "API Login Verification (header/dna)": {
                "Accept": "application/json",
                "Referer": self.BASE_URL,
            },
            "Header Trees API": {"Accept": "*/*", "Referer": self.BASE_URL},
            "Tree Owner Name API": {
                "Accept": "application/json, text/plain, */*",
                "ancestry-clientpath": "Browser:meexp-uhome",
                "Referer": self.BASE_URL,
            },
            "Get Ladder API": {"Accept": "*/*"},
            "Match List API": {
                "Accept": "application/json",
                "Referer": urljoin(self.BASE_URL, "/discoveryui-matches/list/"),
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "cache-control": "no-cache",
                "pragma": "no-cache",
                "priority": "u=1, i",
            },
            "In-Tree Status Check": {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Referer": urljoin(self.BASE_URL, "/discoveryui-matches/list/"),
                "Origin": origin_header_value,
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
            },
            "Match Probability API": {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Referer": urljoin(self.BASE_URL, "/discoveryui-matches/list/"),
                "Origin": origin_header_value,
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
            },
            "Badge Details API": {
                "Accept": "application/json",
                "Referer": urljoin(self.BASE_URL, "/discoveryui-matches/list/"),
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
            },
            "Match Details API": {
                "Accept": "application/json",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
            },
            "Profile Details API": {
                "accept": "application/json",
                "ancestry-clientpath": "express-fe",
                "cache-control": "no-cache",
                "pragma": "no-cache",
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-origin",
                "priority": "u=1, i",
            },
            "Create Conversation API": {
                "Accept": "*/*",
                "Content-Type": "application/json",
                "ancestry-clientpath": "express-fe",
                "Origin": origin_header_value,
                "Referer": urljoin(self.BASE_URL, "/messaging/"),
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "cache-control": "no-cache",
                "pragma": "no-cache",
                "priority": "u=1, i",
            },
            "Send Message API (Existing Conv)": {
                "Accept": "*/*",
                "Content-Type": "application/json",
                "ancestry-clientpath": "express-fe",
                "Origin": origin_header_value,
                "Referer": urljoin(self.BASE_URL, "/messaging/"),
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "cache-control": "no-cache",
                "pragma": "no-cache",
                "priority": "u=1, i",
            },
            "Fetch Conversation Messages": {
                "accept": "*/*",
                "ancestry-clientpath": "express-fe",
                "referer": urljoin(self.BASE_URL, "/messaging/"),
            },
        }
        # --- Log critical values ---
        logger.info(
            f"Config Loaded: BASE_URL='{self.BASE_URL}', DB='{self.DATABASE_FILE.name}', TREE='{self.TREE_NAME or 'N/A'}'"
        )
        if not self.ANCESTRY_USERNAME or not self.ANCESTRY_PASSWORD:
            logger.warning("Ancestry credentials missing!")
        logger.debug("Config loading complete.\n")

    # End of _load_values


# End of Config_Class


class SeleniumConfig(BaseConfig):
    """Configuration specific to Selenium WebDriver settings."""

    # --- Defaults ---
    HEADLESS_MODE: bool = False
    PROFILE_DIR: str = "Default"
    DEBUG_PORT: int = 9516
    CHROME_MAX_RETRIES: int = 3
    CHROME_RETRY_DELAY: int = 5
    IMPLICIT_WAIT: int = 0
    ELEMENT_TIMEOUT: int = 20
    PAGE_TIMEOUT: int = 40
    ASYNC_SCRIPT_TIMEOUT: int = 60
    LOGGED_IN_CHECK_TIMEOUT: int = 15
    MODAL_TIMEOUT: int = 10
    DNA_LIST_PAGE_TIMEOUT: int = 30
    NEW_TAB_TIMEOUT: int = 15
    TWO_FA_CODE_ENTRY_TIMEOUT: int = 300
    API_TIMEOUT: int = 60

    def __init__(self):
        self._load_values()
        logger.debug("Selenium configuration loaded.")

    # End of __init__
    def _load_values(self):
        self.CHROME_DRIVER_PATH: Optional[Path] = self._get_path_env(
            "CHROME_DRIVER_PATH", None
        )
        self.CHROME_BROWSER_PATH: Optional[Path] = self._get_path_env(
            "CHROME_BROWSER_PATH", None
        )
        default_user_data_str = str(Path.home() / ".ancestry_chrome_data")
        self.CHROME_USER_DATA_DIR: Optional[Path] = self._get_path_env(
            "CHROME_USER_DATA_DIR", default_user_data_str
        )
        if self.CHROME_USER_DATA_DIR:
            self.CHROME_USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.HEADLESS_MODE = self._get_bool_env("HEADLESS_MODE", self.HEADLESS_MODE)
        self.PROFILE_DIR = self._get_string_env("PROFILE_DIR", self.PROFILE_DIR)
        self.DEBUG_PORT = self._get_int_env("DEBUG_PORT", self.DEBUG_PORT)
        self.CHROME_MAX_RETRIES = self._get_int_env(
            "CHROME_MAX_RETRIES", self.CHROME_MAX_RETRIES
        )
        self.CHROME_RETRY_DELAY = self._get_int_env(
            "CHROME_RETRY_DELAY", self.CHROME_RETRY_DELAY
        )
        self.ELEMENT_TIMEOUT = self._get_int_env(
            "ELEMENT_TIMEOUT", self.ELEMENT_TIMEOUT
        )
        self.PAGE_TIMEOUT = self._get_int_env("PAGE_TIMEOUT", self.PAGE_TIMEOUT)
        self.ASYNC_SCRIPT_TIMEOUT = self._get_int_env(
            "ASYNC_SCRIPT_TIMEOUT", self.ASYNC_SCRIPT_TIMEOUT
        )
        self.LOGGED_IN_CHECK_TIMEOUT = self._get_int_env(
            "LOGGED_IN_CHECK_TIMEOUT", self.LOGGED_IN_CHECK_TIMEOUT
        )
        self.MODAL_TIMEOUT = self._get_int_env("MODAL_TIMEOUT", self.MODAL_TIMEOUT)
        self.DNA_LIST_PAGE_TIMEOUT = self._get_int_env(
            "DNA_LIST_PAGE_TIMEOUT", self.DNA_LIST_PAGE_TIMEOUT
        )
        self.NEW_TAB_TIMEOUT = self._get_int_env(
            "NEW_TAB_TIMEOUT", self.NEW_TAB_TIMEOUT
        )
        self.TWO_FA_CODE_ENTRY_TIMEOUT = self._get_int_env(
            "TWO_FA_CODE_ENTRY_TIMEOUT", self.TWO_FA_CODE_ENTRY_TIMEOUT
        )
        self.API_TIMEOUT = self._get_int_env("API_TIMEOUT", self.API_TIMEOUT)
        logger.debug(f"ChromeDriver Path: {self.CHROME_DRIVER_PATH or 'Auto-detect'}")
        logger.debug(
            f"Chrome Browser Path: {self.CHROME_BROWSER_PATH or 'System default'}"
        )
        logger.debug(f"Chrome User Data Dir: {self.CHROME_USER_DATA_DIR or 'Not Set'}")
        logger.debug(f"Chrome Profile Dir: {self.PROFILE_DIR}")
        logger.debug(f"Headless Mode: {self.HEADLESS_MODE}")
        logger.debug(f"Element Timeout: {self.ELEMENT_TIMEOUT}s")
        logger.debug(f"Page Timeout: {self.PAGE_TIMEOUT}s")
        logger.debug(f"Async Script Timeout: {self.ASYNC_SCRIPT_TIMEOUT}s")
        logger.info(f"API Request Timeout (requests lib): {self.API_TIMEOUT}s")

    # --- WebDriverWait Factory Methods ---
    def default_wait(self, driver, timeout: Optional[int] = None) -> WebDriverWait:
        return WebDriverWait(
            driver, timeout if timeout is not None else self.ELEMENT_TIMEOUT
        )

    def page_load_wait(self, driver, timeout: Optional[int] = None) -> WebDriverWait:
        return WebDriverWait(
            driver, timeout if timeout is not None else self.PAGE_TIMEOUT
        )

    def short_wait(self, driver, timeout: int = 5) -> WebDriverWait:
        return WebDriverWait(driver, timeout)

    def long_wait(self, driver, timeout: Optional[int] = None) -> WebDriverWait:
        return WebDriverWait(
            driver, timeout if timeout is not None else self.TWO_FA_CODE_ENTRY_TIMEOUT
        )

    def logged_in_check_wait(self, driver) -> WebDriverWait:
        return WebDriverWait(driver, self.LOGGED_IN_CHECK_TIMEOUT)

    def element_wait(self, driver) -> WebDriverWait:
        return WebDriverWait(driver, self.ELEMENT_TIMEOUT)

    def page_wait(self, driver) -> WebDriverWait:
        return WebDriverWait(driver, self.PAGE_TIMEOUT)

    def modal_wait(self, driver) -> WebDriverWait:
        return WebDriverWait(driver, self.MODAL_TIMEOUT)

    def dna_list_page_wait(self, driver) -> WebDriverWait:
        return WebDriverWait(driver, self.DNA_LIST_PAGE_TIMEOUT)

    def new_tab_wait(self, driver) -> WebDriverWait:
        return WebDriverWait(driver, self.NEW_TAB_TIMEOUT)


# End of SeleniumConfig

# --- Singleton Instances ---
config_instance = Config_Class()
selenium_config = SeleniumConfig()

# --- End of config.py ---
