#!/usr/bin/env python3

# config.py

import os
import logging
from typing import Optional, Dict, Any, Tuple
from dotenv import load_dotenv # Keep import here
from selenium.webdriver.support.ui import WebDriverWait
# Assuming logger is configured in logging_config.py, get it AFTER setup usually
# from logging_config import logger
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
        value_str = self._get_env_var(key) # Get value or None
        if value_str is None:
            logger.debug(f"Environment variable '{key}' not set, using default: {default}")
            return default
        try:
            return int(value_str)
        except (ValueError, TypeError):
            logger.warning(f"Invalid integer value '{value_str}' for '{key}'. Using default: {default}")
            return default
    # end _get_int_env

    def _get_float_env(self, key: str, default: float) -> float:
        """Gets a float environment variable, falling back to default."""
        value_str = self._get_env_var(key) # Get value or None
        if value_str is None:
            logger.debug(f"Environment variable '{key}' not set, using default: {default}")
            return default
        try:
            return float(value_str)
        except (ValueError, TypeError):
            logger.warning(f"Invalid float value '{value_str}' for '{key}'. Using default: {default}")
            return default
    # end _get_float_env

    def _get_string_env(self, key: str, default: str) -> str:
        """Gets a string environment variable, falling back to default."""
        value = self._get_env_var(key) # Get value or None
        if value is None:
             # Only log if the default itself isn't an empty string or similar 'falsy' default
             if default:
                 logger.debug(f"Environment variable '{key}' not set, using default: '{default}'")
             else:
                 logger.debug(f"Environment variable '{key}' not set, using default (empty string or similar).")
             return default
        return str(value) # Should already be string, but ensures type
    # end  _get_string_env

    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Gets a boolean environment variable, falling back to default."""
        value_str = self._get_env_var(key) # Get value or None
        if value_str is None:
            logger.debug(f"Environment variable '{key}' not set, using default: {default}")
            return default
        # Check against truthy values
        return value_str.lower() in ("true", "1", "t", "y", "yes")
    # end _get_bool_env

    def _get_json_env(self, key: str, default: Any) -> Any:
        """Gets a JSON environment variable, falling back to default."""
        value_str = self._get_env_var(key) # Get value or None
        if value_str: # If found in environment
            try:
                return json.loads(value_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON format for '{key}': {e}. Using default: {default}")
                return default
        else: # Not set in environment, use the provided default
            logger.debug(f"Environment variable '{key}' not set or empty, using default.")
            return default
    # end _get_json_env

    def _get_path_env(self, key: str, default: Optional[str]) -> Optional[Path]:
        """Gets a Path environment variable, falling back to default."""
        value_str = self._get_env_var(key) # Get value or None
        if value_str is None:
            if default is None:
                 logger.debug(f"Environment variable '{key}' not set and no default path provided.")
                 return None
            else:
                 logger.debug(f"Environment variable '{key}' not set, using default path: '{default}'")
                 value_str = default
        # Convert to Path if we have a string value
        try:
             # Use resolve() to get absolute path, handle potential errors
             path_obj = Path(value_str).resolve() if value_str else None
             return path_obj
        except TypeError:
             logger.warning(f"Could not create Path object from value '{value_str}' for key '{key}'. Returning None.")
             return None
        except Exception as e: # Catch potential resolution errors (e.g., invalid characters)
             logger.warning(f"Error resolving path '{value_str}' for key '{key}': {e}. Returning Path object without resolving.")
             try:
                  # Fallback to creating Path without resolving if resolve fails
                  return Path(value_str) if value_str else None
             except Exception:
                  logger.error(f"Could not create Path object even without resolving for '{value_str}'. Returning None.")
                  return None
    # end _get_path_env
#
# End of BaseConfig class

# config.py

import os
import logging
from typing import Optional, Dict, Any, Tuple
from dotenv import load_dotenv # Keep import here
from selenium.webdriver.support.ui import WebDriverWait
# Assuming logger is configured in logging_config.py, get it AFTER setup usually
# from logging_config import logger
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
        value_str = self._get_env_var(key) # Get value or None
        if value_str is None:
            logger.debug(f"Environment variable '{key}' not set, using default: {default}")
            return default
        try:
            return int(value_str)
        except (ValueError, TypeError):
            logger.warning(f"Invalid integer value '{value_str}' for '{key}'. Using default: {default}")
            return default
    # end _get_int_env

    def _get_float_env(self, key: str, default: float) -> float:
        """Gets a float environment variable, falling back to default."""
        value_str = self._get_env_var(key) # Get value or None
        if value_str is None:
            logger.debug(f"Environment variable '{key}' not set, using default: {default}")
            return default
        try:
            return float(value_str)
        except (ValueError, TypeError):
            logger.warning(f"Invalid float value '{value_str}' for '{key}'. Using default: {default}")
            return default
    # end _get_float_env

    def _get_string_env(self, key: str, default: str) -> str:
        """Gets a string environment variable, falling back to default."""
        value = self._get_env_var(key) # Get value or None
        if value is None:
             # Only log if the default itself isn't an empty string or similar 'falsy' default
             if default:
                 logger.debug(f"Environment variable '{key}' not set, using default: '{default}'")
             else:
                 logger.debug(f"Environment variable '{key}' not set, using default (empty string or similar).")
             return default
        return str(value) # Should already be string, but ensures type
    # end  _get_string_env

    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Gets a boolean environment variable, falling back to default."""
        value_str = self._get_env_var(key) # Get value or None
        if value_str is None:
            logger.debug(f"Environment variable '{key}' not set, using default: {default}")
            return default
        # Check against truthy values
        return value_str.lower() in ("true", "1", "t", "y", "yes")
    # end _get_bool_env

    def _get_json_env(self, key: str, default: Any) -> Any:
        """Gets a JSON environment variable, falling back to default."""
        value_str = self._get_env_var(key) # Get value or None
        if value_str: # If found in environment
            try:
                return json.loads(value_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON format for '{key}': {e}. Using default: {default}")
                return default
        else: # Not set in environment, use the provided default
            logger.debug(f"Environment variable '{key}' not set or empty, using default.")
            return default
    # end _get_json_env

    def _get_path_env(self, key: str, default: Optional[str]) -> Optional[Path]:
        """Gets a Path environment variable, falling back to default."""
        value_str = self._get_env_var(key) # Get value or None
        if value_str is None:
            if default is None:
                 logger.debug(f"Environment variable '{key}' not set and no default path provided.")
                 return None
            else:
                 logger.debug(f"Environment variable '{key}' not set, using default path: '{default}'")
                 value_str = default
        # Convert to Path if we have a string value
        try:
             # Use resolve() to get absolute path, handle potential errors
             path_obj = Path(value_str).resolve() if value_str else None
             return path_obj
        except TypeError:
             logger.warning(f"Could not create Path object from value '{value_str}' for key '{key}'. Returning None.")
             return None
        except Exception as e: # Catch potential resolution errors (e.g., invalid characters)
             logger.warning(f"Error resolving path '{value_str}' for key '{key}': {e}. Returning Path object without resolving.")
             try:
                  # Fallback to creating Path without resolving if resolve fails
                  return Path(value_str) if value_str else None
             except Exception:
                  logger.error(f"Could not create Path object even without resolving for '{value_str}'. Returning None.")
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
    BACKOFF_FACTOR: float = 2.0
    DECREASE_FACTOR: float = 0.95
    INITIAL_DELAY: float = 1.0
    MAX_DELAY: float = 90.0
    LOG_LEVEL: str = "DEBUG" # Default log level if not set otherwise
    RETRY_STATUS_CODES: Tuple[int, ...] = (429, 500, 502, 503, 504) # Use tuple for immutability
    DB_POOL_SIZE = 25

    # --- Feature Flags ---
    CHECK_JS_ERRORS_ACTN_6: bool = False

    # --- User Agents ---
    USER_AGENTS: list[str] = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ]

    def __init__(self):
        """Initializes Config_Class by loading values from environment."""
        self._load_values() # This now sets self.LOG_DIR, self.DATA_DIR, self.CACHE_DIR correctly

        # Construct derived URLs after base URL is loaded
        if hasattr(self, 'BASE_URL') and self.BASE_URL and \
           hasattr(self, 'API_BASE_URL_PATH'):
             self.API_BASE_URL = urljoin(self.BASE_URL, self.API_BASE_URL_PATH)
             logger.debug(f"Constructed API_BASE_URL: {self.API_BASE_URL}")
        else:
             logger.error("BASE_URL or API_BASE_URL_PATH not loaded correctly, cannot construct API_BASE_URL.")
             self.API_BASE_URL = None

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

        # === Paths & Files (Using Relative Paths from .env) ===
        # Get directory names from .env, defaulting to 'Logs', 'Data', 'Cache'
        log_dir_name = self._get_string_env("LOG_DIR", "Logs")
        data_dir_name = self._get_string_env("DATA_DIR", "Data")
        cache_dir_name = self._get_string_env("CACHE_DIR", "Cache")

        # Create Path objects relative to the project root
        self.LOG_DIR: Path = Path(log_dir_name)
        self.DATA_DIR: Path = Path(data_dir_name)
        self.CACHE_DIR: Path = Path(cache_dir_name) # Primary Path object for Cache

        # Ensure these base directories exist
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True) # Ensure cache dir exists
        logger.debug(f"Log directory set to: {self.LOG_DIR.resolve()}")
        logger.debug(f"Data directory set to: {self.DATA_DIR.resolve()}")
        logger.debug(f"Cache directory set to: {self.CACHE_DIR.resolve()}")

        # Define file paths based on the directories
        self.DATABASE_FILE: Path = self.DATA_DIR / self._get_string_env("DATABASE_FILE", "ancestry_data.db")
        self.GEDCOM_FILE_PATH: Optional[Path] = self._get_path_env("GEDCOM_FILE_PATH", None) # Remains the same

        # NOTE: CACHE_DIR_PATH might now be redundant if CACHE_DIR is used directly.
        # Kept for potential backward compatibility, points to the same resolved path.
        self.CACHE_DIR_PATH: Path = self.CACHE_DIR
        # Log the final paths being used
        logger.debug(f"Database file path: {self.DATABASE_FILE.resolve()}")
        if self.GEDCOM_FILE_PATH:
             logger.debug(f"Gedcom file path: {self.GEDCOM_FILE_PATH.resolve()}")
        else:
             logger.debug("Gedcom file path: Not set")

        # === URLs ===
        self.BASE_URL: str = self._get_string_env("BASE_URL", "https://www.ancestry.co.uk/")
        self.ALTERNATIVE_API_URL: Optional[str] = self._get_string_env("ALTERNATIVE_API_URL", "")
        self.API_BASE_URL_PATH: str = self._get_string_env("API_BASE_URL_PATH", "")

        # === Application Behavior ===
        self.APP_MODE: str = self._get_string_env("APP_MODE", "dry_run")
        self.MAX_PAGES: int = self._get_int_env("MAX_PAGES", 2) # Increased from 1 for testing
        self.MAX_RETRIES: int = self._get_int_env("MAX_RETRIES", 5) # Increased default retries
        self.MAX_INBOX: int = self._get_int_env("MAX_INBOX", 0)
        self.BATCH_SIZE: int = self._get_int_env("BATCH_SIZE", 5) # Reduced batch size

        # === Database ===
        self.DB_POOL_SIZE: int = self._get_int_env("DB_POOL_SIZE", 20)

        # === Caching ===
        self.CACHE_TIMEOUT: int = self._get_int_env("CACHE_TIMEOUT", 3600)

        # Load as JSON list/tuple, fall back to class default if not set or invalid
        retry_codes_from_env = self._get_json_env("RETRY_STATUS_CODES", self.RETRY_STATUS_CODES)
        if isinstance(retry_codes_from_env, (list, tuple)):
             # Validate that elements are integers
             if all(isinstance(code, int) for code in retry_codes_from_env):
                 self.RETRY_STATUS_CODES = tuple(retry_codes_from_env) # Store as tuple
                 logger.debug(f"Using RETRY_STATUS_CODES from environment: {self.RETRY_STATUS_CODES}")
             else:
                 logger.warning(f"Invalid non-integer value found in RETRY_STATUS_CODES environment variable. Using default: {self.RETRY_STATUS_CODES}")
                 # Keep the class default if validation fails
        else:
            logger.warning(f"RETRY_STATUS_CODES environment variable is not a valid JSON list/tuple. Using default: {self.RETRY_STATUS_CODES}")
            # Keep the class default if parsing fails or type is wrong

        # === API Headers ===
        # Define Origin header dynamically later if needed, or set base here
        parsed_base_url = urlparse(self.BASE_URL)
        origin_header_value = f"{parsed_base_url.scheme}://{parsed_base_url.netloc}"

        self.API_CONTEXTUAL_HEADERS: Dict[str, Dict[str, Optional[str]]] = {
            "CSRF Token API": {
                "Accept": "application/json",
                "Referer": urljoin(self.BASE_URL, "/discoveryui-matches/list/"), # Referer often matches list page
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
            },
            "Get my profile_id": {
                "Accept": "application/json, text/plain, */*",
                "ancestry-clientpath": "p13n-js",
                "Referer": self.BASE_URL, # Referer likely base URL
            },
            "Get UUID API": {
                "Accept": "application/json",
                "Referer": self.BASE_URL, # Base URL as Referer
            },
            "Header Trees API": {
                "Accept": "*/*", # Often accepts anything
                "Referer": self.BASE_URL,
            },
            "Tree Owner Name API": {
                "Accept": "application/json, text/plain, */*",
                "ancestry-clientpath": "Browser:meexp-uhome",
                "Referer": self.BASE_URL,
            },
            # --- Headers for Match List API ---
            "Match List API": {
                "Accept": "application/json", # Expecting JSON
                "Referer": urljoin(self.BASE_URL,"/discoveryui-matches/list/"), # Referer is the matches list page
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "cache-control": "no-cache", # Added based on curl
                "pragma": "no-cache",       # Added based on curl
                "priority": "u=1, i",       # Added based on curl
            },
            # --- Headers for In-Tree Status Check ---
            "In-Tree Status Check": {
                "Accept": "application/json",
                "Content-Type": "application/json", # Sending JSON body
                "Referer": urljoin(self.BASE_URL,"/discoveryui-matches/list/"),
                "Origin": origin_header_value,
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
            },
            # --- Headers for Match Probability API ---
            "Match Probability API": {
                "Accept": "application/json",
                "Content-Type": "application/json", # Sending JSON body (even if empty)
                "Referer": urljoin(self.BASE_URL,"/discoveryui-matches/list/"),
                "Origin": origin_header_value,
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
            },
            # --- Headers for Badge Details API ---
            "Badge Details API": {
                "Accept": "application/json",
                "Referer": urljoin(self.BASE_URL,"/discoveryui-matches/list/"), # Referer is likely matches list or compare page
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
            },
            # --- Headers for Get Ladder API ---
            "Get Ladder API": {
                "Accept": "*/*", # Expects JSONP, Accept *.* is safe
                # Referer added dynamically in _get_relShip
            },
            # --- Headers for API Login Verification ---
            "API Login Verification (header/dna)": {
                "Accept": "application/json",
                "Referer": self.BASE_URL,
            },
            # --- Headers for messaging APIs ---
            "Create Conversation API": {
                "Accept": "*/*", # Based on successful cURL
                "Content-Type": "application/json", # Correct Content-Type
                "ancestry-clientpath": "express-fe",
                # "ancestry-userid": handled dynamically in send_messages_to_matches
                "Origin": origin_header_value, # Use dynamically determined origin
                "Referer": urljoin(self.BASE_URL, "/messaging"), # Referer is messaging page
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "cache-control": "no-cache",
                "pragma": "no-cache",
                "priority": "u=1, i",
            },
            "Send Message API (Existing Conv)": {
                "Accept": "*/*",
                "Content-Type": "application/json", # Correct Content-Type
                "ancestry-clientpath": "express-fe",
                # "ancestry-userid": handled dynamically
                "Origin": origin_header_value,
                "Referer": urljoin(self.BASE_URL, "/messaging"),
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "cache-control": "no-cache",
                "pragma": "no-cache",
                "priority": "u=1, i",
            },
            "Fetch Conversation Messages API (action8 confirmation)": {
                "accept": "*/*",
                # "accept-language": "en-GB,en;q=0.9", # Optional
                "ancestry-clientpath": "express-fe",
                "cache-control": "no-cache",
                "pragma": "no-cache",
                "referer": urljoin(self.BASE_URL, "/messaging"),
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-origin",
                "priority": "u=1, i",
            },
            # --- Headers for Profile Details API (/details and profile APIs in _get_match_details_and_admin) ---
            "Match Details API": { # For /discoveryui-matchesservice/api/samples/.../details
                "Accept": "application/json",
                # Referer added dynamically based on compare page URL
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
            },
            "Profile Details API": { # For /app-api/express/v1/profiles/details
                    "accept": "application/json",
                    "ancestry-clientpath": "express-fe",
                    # "ancestry-userid": Added dynamically in _get_match_details_and_admin
                    "cache-control": "no-cache",
                    "pragma": "no-cache",
                    # Referer added dynamically based on compare page URL
                    "sec-fetch-dest": "empty",
                    "sec-fetch-mode": "cors",
                    "sec-fetch-site": "same-origin",
                    "priority": "u=1, i",
            }
        }

        # --- Log loaded critical values ---
        logger.info(f"Configuration Loaded: BASE_URL='{self.BASE_URL}', DB='{self.DATABASE_FILE}', TREE_NAME='{self.TREE_NAME or 'Not Set'}'")
        if not self.ANCESTRY_USERNAME or not self.ANCESTRY_PASSWORD:
             logger.warning("Ancestry username or password not found in configuration! Required for login.")
        logger.debug(f"API Base Path: '{self.API_BASE_URL_PATH or 'Not Set'}'")
        logger.debug(f"Alternative API URL: '{self.ALTERNATIVE_API_URL or 'Not Set'}'")
        logger.debug(f"Max Pages to Process: {self.MAX_PAGES}")
        logger.debug(f"Batch Size: {self.BATCH_SIZE}")

        logger.debug("Config settings loading complete.\n")
    # end _load_values
#
# End of Config_Class

class SeleniumConfig(BaseConfig):
    """
    Configuration specific to Selenium WebDriver settings.
    Accessed via the 'selenium_config' object.
    """

    # --- Define defaults directly as class attributes ---
    HEADLESS_MODE: bool = False
    PROFILE_DIR: str = "Default"
    DEBUG_PORT: int = 9516
    CHROME_MAX_RETRIES: int = 3
    CHROME_RETRY_DELAY: int = 5
    IMPLICIT_WAIT: int = 0
    ELEMENT_TIMEOUT: int = 20
    PAGE_TIMEOUT: int = 40
    ASYNC_SCRIPT_TIMEOUT: int = 60 # Default increased previously
    # --- INCREASED API_TIMEOUT DEFAULT ---
    API_TIMEOUT: int = 60 # Increased default for requests library
    # --- END INCREASE ---
    LOGGED_IN_CHECK_TIMEOUT: int = 15
    MODAL_TIMEOUT: int = 10
    DNA_LIST_PAGE_TIMEOUT: int = 30
    NEW_TAB_TIMEOUT: int = 15
    TWO_FA_CODE_ENTRY_TIMEOUT: int = 300

    def __init__(self):
        """Initializes SeleniumConfig by loading relevant environment variables."""
        self._load_values()
        logger.debug("Selenium configuration loaded.")
    # end __init__

    def _load_values(self):
        """Load Selenium specific variables from .env or use defaults."""
        # Paths (use _get_path_env which handles Path objects and None)
        self.CHROME_DRIVER_PATH: Optional[Path] = self._get_path_env("CHROME_DRIVER_PATH", None)
        self.CHROME_BROWSER_PATH: Optional[Path] = self._get_path_env("CHROME_BROWSER_PATH", None)
        # Provide a sensible string default for CHROME_USER_DATA_DIR path
        default_user_data_str = str(Path.home() / ".ancestry_chrome_data")
        self.CHROME_USER_DATA_DIR: Optional[Path] = self._get_path_env("CHROME_USER_DATA_DIR", default_user_data_str)
        # Ensure user data directory exists (only if path is resolved)
        if self.CHROME_USER_DATA_DIR:
             self.CHROME_USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Load other settings using defaults defined above
        self.HEADLESS_MODE = self._get_bool_env("HEADLESS_MODE", self.HEADLESS_MODE)
        self.PROFILE_DIR = self._get_string_env("PROFILE_DIR", self.PROFILE_DIR)
        self.DEBUG_PORT = self._get_int_env("DEBUG_PORT", self.DEBUG_PORT)
        self.CHROME_MAX_RETRIES = self._get_int_env("CHROME_MAX_RETRIES", self.CHROME_MAX_RETRIES)
        self.CHROME_RETRY_DELAY = self._get_int_env("CHROME_RETRY_DELAY", self.CHROME_RETRY_DELAY)
        self.ELEMENT_TIMEOUT = self._get_int_env("ELEMENT_TIMEOUT", self.ELEMENT_TIMEOUT)
        self.PAGE_TIMEOUT = self._get_int_env("PAGE_TIMEOUT", self.PAGE_TIMEOUT)
        self.ASYNC_SCRIPT_TIMEOUT = self._get_int_env("ASYNC_SCRIPT_TIMEOUT", self.ASYNC_SCRIPT_TIMEOUT)
        # --- LOAD API_TIMEOUT FROM ENV OR USE NEW DEFAULT ---
        self.API_TIMEOUT = self._get_int_env("API_TIMEOUT", self.API_TIMEOUT) # Timeout for requests library
        # --- END LOAD ---
        self.TWO_FA_CODE_ENTRY_TIMEOUT = self._get_int_env(
            "TWO_FA_CODE_ENTRY_TIMEOUT",
            self.TWO_FA_CODE_ENTRY_TIMEOUT)

        # Log loaded paths (check for None)
        logger.debug(f"ChromeDriver Path: {self.CHROME_DRIVER_PATH or 'Auto-detect'}")
        logger.debug(f"Chrome Browser Path: {self.CHROME_BROWSER_PATH or 'System default'}")
        logger.debug(f"Chrome User Data Dir: {self.CHROME_USER_DATA_DIR or 'Not Set'}")
        logger.debug(f"Chrome Profile Dir: {self.PROFILE_DIR}")
        logger.debug(f"Headless Mode: {self.HEADLESS_MODE}")
        logger.debug(f"Async Script Timeout: {self.ASYNC_SCRIPT_TIMEOUT}s")
        # --- ADDED LOG FOR API_TIMEOUT ---
        logger.info(f"API Request Timeout (requests lib): {self.API_TIMEOUT}s") # Use INFO level for this important value
        # --- END LOG ---
    # end _load values

    # --- WebDriverWait Factory Methods ---
    def default_wait(self, driver, timeout: Optional[int] = None) -> WebDriverWait:
        return WebDriverWait(driver, timeout if timeout is not None else self.ELEMENT_TIMEOUT)

    def page_load_wait(self, driver, timeout: Optional[int] = None) -> WebDriverWait:
        return WebDriverWait(driver, timeout if timeout is not None else self.PAGE_TIMEOUT)

    def short_wait(self, driver, timeout: int = 5) -> WebDriverWait:
        return WebDriverWait(driver, timeout)

    def long_wait(self, driver, timeout: Optional[int] = None) -> WebDriverWait:
        return WebDriverWait(driver, timeout if timeout is not None else self.TWO_FA_CODE_ENTRY_TIMEOUT)

    # --- Specific Wait Factories ---
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
#
# End of SeleniumConfig class

# --- Singleton Instances ---
config_instance = Config_Class()
selenium_config = SeleniumConfig()

# --- End of config.py ---