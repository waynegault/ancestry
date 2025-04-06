# File: config.py

#!/usr/bin/env python3

# config.py

import os
import logging
from typing import Optional, Dict, Any, Tuple, List  # Added List
from dotenv import load_dotenv  # Keep import here
from selenium.webdriver.support.wait import WebDriverWait
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse

load_dotenv()
logger = logging.getLogger("logger")


class BaseConfig:
    """
    Base configuration class with simplified helper methods for reading environment variables.
    Defaults are now handled primarily within each specific getter method.
    """

    def _get_env_var(self, key: str) -> Optional[str]:
        """
        Retrieves an environment variable directly using os.getenv.
        Returns None if not found. Does NOT handle defaults here.
        """
        value = os.getenv(key)
        if value is None:
            # Log the absence at DEBUG level, let caller handle defaults/warnings
            logger.debug(f"Environment variable '{key}' not found in environment.")
        return value

    # end _get_env_var

    # --- GETTERS Handle Defaults Internally ---
    def _get_int_env(self, key: str, default: int) -> int:
        """Gets an integer environment variable, falling back to default."""
        value_str = self._get_env_var(key)  # Get value or None
        if value_str is None:
            logger.debug(
                f"Environment variable '{key}' not set, using default: {default}"
            )
            return default
        try:
            return int(value_str)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid integer value '{value_str}' for '{key}'. Using default: {default}"
            )
            return default

    # end _get_int_env

    def _get_float_env(self, key: str, default: float) -> float:
        """Gets a float environment variable, falling back to default."""
        value_str = self._get_env_var(key)  # Get value or None
        if value_str is None:
            logger.debug(
                f"Environment variable '{key}' not set, using default: {default}"
            )
            return default
        try:
            return float(value_str)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid float value '{value_str}' for '{key}'. Using default: {default}"
            )
            return default

    # end _get_float_env

    def _get_string_env(self, key: str, default: str) -> str:
        """Gets a string environment variable, falling back to default."""
        value = self._get_env_var(key)  # Get value or None
        if value is None:
            # Only log if the default itself isn't an empty string or similar 'falsy' default
            if default:
                logger.debug(
                    f"Environment variable '{key}' not set, using default: '{default}'"
                )
            else:
                logger.debug(
                    f"Environment variable '{key}' not set, using default (empty string or similar)."
                )
            return default
        return str(value)  # Should already be string, but ensures type

    # end  _get_string_env

    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Gets a boolean environment variable, falling back to default."""
        value_str = self._get_env_var(key)  # Get value or None
        if value_str is None:
            logger.debug(
                f"Environment variable '{key}' not set, using default: {default}"
            )
            return default
        # Check against truthy values
        return value_str.lower() in ("true", "1", "t", "y", "yes")

    # end _get_bool_env

    def _get_json_env(self, key: str, default: Any) -> Any:
        """Gets a JSON environment variable, falling back to default."""
        value_str = self._get_env_var(key)  # Get value or None
        if value_str:  # If found in environment
            try:
                return json.loads(value_str)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Invalid JSON format for '{key}': {e}. Using default: {default}"
                )
                return default
        else:  # Not set in environment, use the provided default
            logger.debug(
                f"Environment variable '{key}' not set or empty, using default."
            )
            return default

    # end _get_json_env

    def _get_path_env(self, key: str, default: Optional[str]) -> Optional[Path]:
        """Gets a Path environment variable, falling back to default."""
        value_str = self._get_env_var(key)  # Get value or None
        if value_str is None:
            if default is None:
                logger.debug(
                    f"Environment variable '{key}' not set and no default path provided."
                )
                return None
            else:
                logger.debug(
                    f"Environment variable '{key}' not set, using default path: '{default}'"
                )
                value_str = default
        # Convert to Path if we have a string value
        try:
            # Use resolve() to get absolute path, handle potential errors
            path_obj = Path(value_str).resolve() if value_str else None
            return path_obj
        except TypeError:
            logger.warning(
                f"Could not create Path object from value '{value_str}' for key '{key}'. Returning None."
            )
            return None
        except (
            Exception
        ) as e:  # Catch potential resolution errors (e.g., invalid characters)
            logger.warning(
                f"Error resolving path '{value_str}' for key '{key}': {e}. Returning Path object without resolving."
            )
            try:
                # Fallback to creating Path without resolving if resolve fails
                return Path(value_str) if value_str else None
            except Exception:
                logger.error(
                    f"Could not create Path object even without resolving for '{value_str}'. Returning None."
                )
                return None

    # end _get_path_env


#
# End of BaseConfig class


class Config_Class(BaseConfig):
    """
    Main configuration class loading settings from environment variables.
    Provides defaults and basic validation for application settings.
    Accessed via the 'config_instance' object.
    """

    # --- Constants / Fixed Settings ---
    # --- Rate Limiter Defaults (Tune via .env if needed) ---
    INITIAL_DELAY: float = 0.5  # Start slightly faster
    MAX_DELAY: float = 60.0
    BACKOFF_FACTOR: float = 1.8  # Less aggressive backoff
    DECREASE_FACTOR: float = 0.98  # Slower decrease when things are good
    LOG_LEVEL: str = "INFO"  # Default log level if not set otherwise
    RETRY_STATUS_CODES: Tuple[int, ...] = (
        429,
        500,
        502,
        503,
        504,
    )  # Use tuple for immutability
    DB_POOL_SIZE = 120
    MESSAGE_TRUNCATION_LENGTH: int = 100

    # --- Feature Flags ---
    CHECK_JS_ERRORS_ACTN_6: bool = False  # Keep default as False unless needed

    # --- User Agents ---
    USER_AGENTS: list[str] = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    ]

    def __init__(self):
        """Initializes Config_Class by loading values from environment."""
        # Call _load_values first, which now also constructs API_BASE_URL
        self._load_values()

        # Note: self.CACHE_DIR is now the primary Path object for the cache directory,
        # set within _load_values(). Other modules like cache.py and utils.py
        # should now directly use config_instance.CACHE_DIR.

    # end __init__

    def _load_values(self):
        """Load and validate settings from .env file or environment variables."""
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
        logger.debug(f"Log directory set to: {self.LOG_DIR.resolve()}")
        logger.debug(f"Data directory set to: {self.DATA_DIR.resolve()}")
        logger.debug(f"Cache directory set to: {self.CACHE_DIR.resolve()}")
        self.DATABASE_FILE: Path = self.DATA_DIR / self._get_string_env(
            "DATABASE_FILE", "ancestry_data.db"
        )
        self.GEDCOM_FILE_PATH: Optional[Path] = self._get_path_env(
            "GEDCOM_FILE_PATH", None
        )
        self.CACHE_DIR_PATH: Path = self.CACHE_DIR
        logger.debug(f"Database file path: {self.DATABASE_FILE.resolve()}")
        if self.GEDCOM_FILE_PATH:
            logger.debug(f"Gedcom file path: {self.GEDCOM_FILE_PATH.resolve()}")
        else:
            logger.debug("Gedcom file path: Not set")

        # === URLs ===
        self.BASE_URL: str = self._get_string_env(
            "BASE_URL", "https://www.ancestry.co.uk/"
        )
        self.ALTERNATIVE_API_URL: Optional[str] = self._get_env_var(
            "ALTERNATIVE_API_URL"
        )
        self.API_BASE_URL_PATH: str = self._get_string_env(
            "API_BASE_URL_PATH", "/api/v2/"
        )  # Example default

        # --- MOVED API_BASE_URL Construction Here ---
        if self.BASE_URL and self.API_BASE_URL_PATH:  # Check dependencies are loaded
            self.API_BASE_URL = urljoin(self.BASE_URL, self.API_BASE_URL_PATH)
        elif self.BASE_URL:
            logger.debug(
                "API_BASE_URL_PATH not set, API_BASE_URL will not be constructed."
            )
            self.API_BASE_URL = None
        else:
            logger.error(
                "BASE_URL not loaded correctly, cannot construct API_BASE_URL."
            )
            self.API_BASE_URL = None
        # --- END MOVE ---

        # === Application Behavior ===
        self.APP_MODE: str = self._get_string_env("APP_MODE", "dry_run")
        self.MAX_PAGES: int = self._get_int_env("MAX_PAGES", 0)
        self.MAX_RETRIES: int = self._get_int_env("MAX_RETRIES", 5)
        self.MAX_INBOX: int = self._get_int_env("MAX_INBOX", 0)
        self.BATCH_SIZE: int = self._get_int_env("BATCH_SIZE", 50)

        # === Database ===
        self.DB_POOL_SIZE: int = self._get_int_env(
            "DB_POOL_SIZE", self.DB_POOL_SIZE
        )  # Use class default

        # === Caching ===
        self.CACHE_TIMEOUT: int = self._get_int_env(
            "CACHE_TIMEOUT", 3600
        )  # 1 hour default

        # --- Load Rate Limiter Values from Env or use Class Defaults ---
        # *** Values now use tuned class defaults ***
        self.INITIAL_DELAY = self._get_float_env("INITIAL_DELAY", self.INITIAL_DELAY)
        self.MAX_DELAY = self._get_float_env("MAX_DELAY", self.MAX_DELAY)
        self.BACKOFF_FACTOR = self._get_float_env("BACKOFF_FACTOR", self.BACKOFF_FACTOR)
        self.DECREASE_FACTOR = self._get_float_env(
            "DECREASE_FACTOR", self.DECREASE_FACTOR
        )
        logger.debug(
            f"Rate Limiter Config: Initial={self.INITIAL_DELAY}s, Max={self.MAX_DELAY}s, "
            f"Backoff={self.BACKOFF_FACTOR}x, Decrease={self.DECREASE_FACTOR}x"
        )

        # --- Load RETRY_STATUS_CODES ---
        retry_codes_from_env = self._get_json_env(
            "RETRY_STATUS_CODES", self.RETRY_STATUS_CODES
        )
        if isinstance(retry_codes_from_env, (list, tuple)):
            if all(isinstance(code, int) for code in retry_codes_from_env):
                self.RETRY_STATUS_CODES = tuple(retry_codes_from_env)
                logger.debug(
                    f"Using RETRY_STATUS_CODES from environment: {self.RETRY_STATUS_CODES}"
                )
            else:
                logger.warning(
                    f"Invalid non-integer value in RETRY_STATUS_CODES env var. Using default: {self.RETRY_STATUS_CODES}"
                )
        else:
            logger.warning(
                f"RETRY_STATUS_CODES env var is not valid JSON list/tuple. Using default: {self.RETRY_STATUS_CODES}"
            )

        # === API Headers ===
        parsed_base_url = urlparse(self.BASE_URL)
        origin_header_value = f"{parsed_base_url.scheme}://{parsed_base_url.netloc}"
        # Header definitions remain the same as before...
        self.API_CONTEXTUAL_HEADERS: Dict[str, Dict[str, Optional[str]]] = {
            # --- Core/Session Related ---
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
            "Get UUID API": {
                "Accept": "application/json",
                "Referer": self.BASE_URL,
            },
            "API Login Verification (header/dna)": {  # Used by SessionManager._verify_api_login_status
                "Accept": "application/json",
                "Referer": self.BASE_URL,
            },
            # --- Tree Related ---
            "Header Trees API": {
                "Accept": "*/*",
                "Referer": self.BASE_URL,
            },
            "Tree Owner Name API": {
                "Accept": "application/json, text/plain, */*",
                "ancestry-clientpath": "Browser:meexp-uhome",
                "Referer": self.BASE_URL,
            },
            "Get Ladder API": {  # Note: Referer added dynamically in _get_relShip
                "Accept": "*/*",  # Expects JSONP, Accept */* is safe
            },
            # --- Match List/Processing Related ---
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
            "Badge Details API": {  # Called by _get_tree
                "Accept": "application/json",
                "Referer": urljoin(self.BASE_URL, "/discoveryui-matches/list/"),
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
            },
            "Match Details API": {  # Called by _get_match_details_and_admin (/details endpoint)
                "Accept": "application/json",
                # Referer added dynamically based on compare page URL
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
            },
            "Profile Details API": {  # Called by _get_match_details_and_admin (/profiles/details endpoint)
                "accept": "application/json",
                "ancestry-clientpath": "express-fe",
                # "ancestry-userid": Added dynamically
                "cache-control": "no-cache",
                "pragma": "no-cache",
                # Referer added dynamically
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-origin",
                "priority": "u=1, i",
            },
            # --- Messaging Related ---
            "Create Conversation API": {
                "Accept": "*/*",
                "Content-Type": "application/json",
                "ancestry-clientpath": "express-fe",
                # "ancestry-userid": handled dynamically
                "Origin": origin_header_value,
                "Referer": urljoin(self.BASE_URL, "/messaging/"),  # Corrected path
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
                # "ancestry-userid": handled dynamically
                "Origin": origin_header_value,
                "Referer": urljoin(self.BASE_URL, "/messaging/"),  # Corrected path
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "cache-control": "no-cache",
                "pragma": "no-cache",
                "priority": "u=1, i",
            },
            "Fetch Conversation Messages API (action8 confirmation)": {
                "accept": "*/*",
                "ancestry-clientpath": "express-fe",
                "cache-control": "no-cache",
                "pragma": "no-cache",
                "referer": urljoin(self.BASE_URL, "/messaging/"),  # Corrected path
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-origin",
                "priority": "u=1, i",
            },
        }

        # --- Log loaded critical values ---
        # Now safe to access self.API_BASE_URL here
        logger.info(
            f"Configuration Loaded: BASE_URL='{self.BASE_URL}', DB='{self.DATABASE_FILE.name}', TREE_NAME='{self.TREE_NAME or 'Not Set'}'"
        )
        if not self.ANCESTRY_USERNAME or not self.ANCESTRY_PASSWORD:
            logger.warning(
                "Ancestry username or password not found in configuration! Required for login."
            )
        logger.debug(
            f"API Base Path: '{self.API_BASE_URL_PATH or 'Not Set'}' -> API Base URL: '{self.API_BASE_URL or 'Not Constructed'}'"
        )
        logger.debug(f"Alternative API URL: '{self.ALTERNATIVE_API_URL or 'Not Set'}'")
        logger.debug(
            f"Max Pages to Process: {self.MAX_PAGES if self.MAX_PAGES > 0 else 'All'}"
        )
        logger.debug(f"Batch Size (if used): {self.BATCH_SIZE}")
        # >>> ADDED Log <<<
        logger.debug(f"Message Truncation Length: {self.MESSAGE_TRUNCATION_LENGTH}")

        logger.debug("Config settings loading complete.\n")

    # end _load_values


#
# End of Config_Class

# ... (rest of config.py, including SeleniumConfig and instances, remains the same) ...


class SeleniumConfig(BaseConfig):
    """
    Configuration specific to Selenium WebDriver settings.
    Accessed via the 'selenium_config' object.
    """

    # --- Define defaults directly as class attributes ---
    HEADLESS_MODE: bool = False
    PROFILE_DIR: str = "Default"
    DEBUG_PORT: int = 9516  # Default debug port, can be overridden
    CHROME_MAX_RETRIES: int = 3
    CHROME_RETRY_DELAY: int = 5
    # --- Implicit Wait is generally discouraged with explicit waits ---
    IMPLICIT_WAIT: int = 0
    # --- Explicit Wait Timeouts ---
    ELEMENT_TIMEOUT: int = 20  # Default for finding elements
    PAGE_TIMEOUT: int = 40  # Default for page readiness state
    ASYNC_SCRIPT_TIMEOUT: int = 60  # Default for execute_async_script
    LOGGED_IN_CHECK_TIMEOUT: int = 15  # Specific timeout for login status UI check
    MODAL_TIMEOUT: int = 10  # Timeout for modals (like consent)
    DNA_LIST_PAGE_TIMEOUT: int = 30  # Timeout specific to DNA list page elements
    NEW_TAB_TIMEOUT: int = 15  # Timeout for new tab operations
    TWO_FA_CODE_ENTRY_TIMEOUT: int = 300  # Long timeout for manual 2FA entry (5 mins)
    # --- API Timeout for 'requests' library ---
    API_TIMEOUT: int = 60  # Default timeout for requests library calls

    def __init__(self):
        """Initializes SeleniumConfig by loading relevant environment variables."""
        self._load_values()
        logger.debug("Selenium configuration loaded.")

    # end __init__

    def _load_values(self):
        """Load Selenium specific variables from .env or use defaults."""
        # Paths (use _get_path_env which handles Path objects and None)
        self.CHROME_DRIVER_PATH: Optional[Path] = self._get_path_env(
            "CHROME_DRIVER_PATH", None
        )
        self.CHROME_BROWSER_PATH: Optional[Path] = self._get_path_env(
            "CHROME_BROWSER_PATH", None
        )
        # Provide a sensible string default for CHROME_USER_DATA_DIR path
        default_user_data_str = str(Path.home() / ".ancestry_chrome_data")
        self.CHROME_USER_DATA_DIR: Optional[Path] = self._get_path_env(
            "CHROME_USER_DATA_DIR", default_user_data_str
        )
        # Ensure user data directory exists (only if path is resolved)
        if self.CHROME_USER_DATA_DIR:
            self.CHROME_USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Load other settings using defaults defined above
        self.HEADLESS_MODE = self._get_bool_env("HEADLESS_MODE", self.HEADLESS_MODE)
        self.PROFILE_DIR = self._get_string_env("PROFILE_DIR", self.PROFILE_DIR)
        self.DEBUG_PORT = self._get_int_env("DEBUG_PORT", self.DEBUG_PORT)
        self.CHROME_MAX_RETRIES = self._get_int_env(
            "CHROME_MAX_RETRIES", self.CHROME_MAX_RETRIES
        )
        self.CHROME_RETRY_DELAY = self._get_int_env(
            "CHROME_RETRY_DELAY", self.CHROME_RETRY_DELAY
        )
        # Load Explicit Wait Timeouts from Env or use Class Defaults
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
        # Load API_TIMEOUT from ENV or use Class Default
        self.API_TIMEOUT = self._get_int_env("API_TIMEOUT", self.API_TIMEOUT)

        # Log loaded paths and key settings
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

    # end _load values

    # --- WebDriverWait Factory Methods (providing convenience) ---
    def default_wait(self, driver, timeout: Optional[int] = None) -> WebDriverWait:
        """Returns a WebDriverWait instance with the default element timeout."""
        return WebDriverWait(
            driver, timeout if timeout is not None else self.ELEMENT_TIMEOUT
        )

    def page_load_wait(self, driver, timeout: Optional[int] = None) -> WebDriverWait:
        """Returns a WebDriverWait instance with the page load timeout."""
        return WebDriverWait(
            driver, timeout if timeout is not None else self.PAGE_TIMEOUT
        )

    def short_wait(self, driver, timeout: int = 5) -> WebDriverWait:
        """Returns a WebDriverWait instance with a short, fixed timeout."""
        return WebDriverWait(driver, timeout)

    def long_wait(self, driver, timeout: Optional[int] = None) -> WebDriverWait:
        """Returns a WebDriverWait instance with a long timeout (defaults to 2FA timeout)."""
        return WebDriverWait(
            driver, timeout if timeout is not None else self.TWO_FA_CODE_ENTRY_TIMEOUT
        )

    # --- Specific Wait Factories (for clarity in code) ---
    def logged_in_check_wait(self, driver) -> WebDriverWait:
        """Returns WebDriverWait for login UI checks."""
        return WebDriverWait(driver, self.LOGGED_IN_CHECK_TIMEOUT)

    def element_wait(self, driver) -> WebDriverWait:
        """Returns WebDriverWait using the default element timeout."""
        return WebDriverWait(driver, self.ELEMENT_TIMEOUT)

    def page_wait(self, driver) -> WebDriverWait:
        """Returns WebDriverWait using the default page load timeout."""
        return WebDriverWait(driver, self.PAGE_TIMEOUT)

    def modal_wait(self, driver) -> WebDriverWait:
        """Returns WebDriverWait for modal dialogs."""
        return WebDriverWait(driver, self.MODAL_TIMEOUT)

    def dna_list_page_wait(self, driver) -> WebDriverWait:
        """Returns WebDriverWait specifically for DNA list page elements."""
        return WebDriverWait(driver, self.DNA_LIST_PAGE_TIMEOUT)

    def new_tab_wait(self, driver) -> WebDriverWait:
        """Returns WebDriverWait for new tab operations."""
        return WebDriverWait(driver, self.NEW_TAB_TIMEOUT)


#
# End of SeleniumConfig class

# --- Singleton Instances ---
# These instances are created when the module is imported, making settings globally accessible.
config_instance = Config_Class()
selenium_config = SeleniumConfig()

# --- End of config.py ---
