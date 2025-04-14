# config.py

#!/usr/bin/env python3

"""
config.py - Centralized Configuration Management

Loads application settings from environment variables (.env file) and defines
configuration classes (`Config_Class`, `SeleniumConfig`) with sensible defaults.
Provides typed access to settings like credentials, paths, URLs, API keys,
behavioral parameters, and Selenium options. Includes contextual headers for API calls.
"""

# --- Standard library imports ---
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

# --- Third-party imports ---
from dotenv import load_dotenv
from selenium.webdriver.support.wait import (
    WebDriverWait,
)  # Required for type hinting in SeleniumConfig

# --- Load .env file early ---
load_dotenv()

# --- Initialize logger (used internally within config loading) ---
# Use basicConfig for simplicity during config loading itself
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
logger = logging.getLogger("config_setup")  # Specific logger for config process


# --- Base Configuration Class ---
class BaseConfig:
    """Base class providing helper methods for retrieving typed environment variables."""

    def _get_env_var(self, key: str) -> Optional[str]:
        """Retrieves an environment variable's value."""
        # Step 1: Get value from environment
        value = os.getenv(key)
        # Step 2: Return the value (or None if not found)
        # Logging is handled by the typed getters below
        return value

    # End of _get_env_var

    def _get_int_env(self, key: str, default: int) -> int:
        """Retrieves an environment variable as an integer, with default."""
        # Step 1: Get raw environment variable string
        value_str = self._get_env_var(key)
        # Step 2: Return default if variable not set
        if value_str is None:
            # logger.debug(f"Env var '{key}' not set. Using default: {default}")
            return default
        # Step 3: Attempt conversion to integer
        try:
            return int(value_str)
        # Step 4: Handle conversion errors, return default
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid integer value '{value_str}' for env var '{key}'. Using default: {default}"
            )
            return default

    # End of _get_int_env

    def _get_float_env(self, key: str, default: float) -> float:
        """Retrieves an environment variable as a float, with default."""
        # Step 1: Get raw environment variable string
        value_str = self._get_env_var(key)
        # Step 2: Return default if variable not set
        if value_str is None:
            return default
        # Step 3: Attempt conversion to float
        try:
            return float(value_str)
        # Step 4: Handle conversion errors, return default
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid float value '{value_str}' for env var '{key}'. Using default: {default}"
            )
            return default

    # End of _get_float_env

    def _get_string_env(self, key: str, default: str) -> str:
        """Retrieves an environment variable as a string, with default."""
        # Step 1: Get raw environment variable string
        value = self._get_env_var(key)
        # Step 2: Return default if variable not set, otherwise return the string value
        return default if value is None else str(value)

    # End of _get_string_env

    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Retrieves an environment variable as a boolean, with default.
        Considers 'true', '1', 't', 'y', 'yes' (case-insensitive) as True."""
        # Step 1: Get raw environment variable string
        value_str = self._get_env_var(key)
        # Step 2: Return default if variable not set
        if value_str is None:
            return default
        # Step 3: Check if the lowercase string matches common True values
        return value_str.lower() in ("true", "1", "t", "y", "yes")

    # End of _get_bool_env

    def _get_json_env(self, key: str, default: Any) -> Any:
        """Retrieves an environment variable as a JSON object, with default."""
        # Step 1: Get raw environment variable string
        value_str = self._get_env_var(key)
        # Step 2: Attempt to parse JSON if variable is set
        if value_str:
            try:
                return json.loads(value_str)
            # Step 3: Handle JSON decoding errors, return default
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Invalid JSON in env var '{key}': {e}. Using default: {default}"
                )
                return default
        # Step 4: Return default if variable not set
        else:
            return default

    # End of _get_json_env

    def _get_path_env(self, key: str, default: Optional[str]) -> Optional[Path]:
        """Retrieves an environment variable as a resolved Path object, with default."""
        # Step 1: Get raw environment variable string or default
        value_str = self._get_env_var(key)
        if value_str is None:
            value_str = default  # Use default string path if env var not set

        # Step 2: Return None if no path string is available
        if value_str is None:
            return None

        # Step 3: Attempt to create and resolve the Path object
        try:
            # Create Path object and resolve to absolute path
            path_obj = Path(value_str).resolve()
            return path_obj
        except Exception as e:
            # Log error during path resolution but return unresolved Path cautiously
            logger.warning(
                f"Error resolving path '{value_str}' for env var '{key}': {e}. Returning unresolved path."
            )
            return Path(value_str)

    # End of _get_path_env


# End of BaseConfig class


# --- Main Application Configuration Class ---
class Config_Class(BaseConfig):
    """
    Loads and provides access to application-wide configuration settings,
    primarily from environment variables defined in the .env file. Includes
    constants, API settings, behavior controls, and AI configuration.
    """

    # --- Static Constants / Fixed Settings ---
    # Default testing profile ID (can be overridden by .env)
    TESTING_PROFILE_ID: Optional[str] = "08FA6E79-0006-0000-0000-000000000000"
    # Default rate limiting parameters (can be overridden by .env)
    INITIAL_DELAY: float = 0.5
    MAX_DELAY: float = 60.0
    BACKOFF_FACTOR: float = 1.8
    DECREASE_FACTOR: float = 0.98
    TOKEN_BUCKET_CAPACITY: float = 10.0
    TOKEN_BUCKET_FILL_RATE: float = 2.0
    # Default log level (can be overridden by .env)
    LOG_LEVEL: str = "INFO"
    # Default HTTP status codes that trigger retries in retry_api decorator
    RETRY_STATUS_CODES: Tuple[int, ...] = (429, 500, 502, 503, 504)
    # Default DB connection pool size (can be overridden by .env)
    DB_POOL_SIZE = 10
    # Max length for storing message content in DB
    MESSAGE_TRUNCATION_LENGTH: int = 300
    # Default User-Agent strings to choose from
    USER_AGENTS: List[str] = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",  # Added Linux UA
    ]
    # Default AI context limits (can be overridden by .env)
    AI_CONTEXT_MESSAGES_COUNT: int = 7
    AI_CONTEXT_MESSAGE_MAX_WORDS: int = 500
    # Default processing limits (0 = unlimited, can be overridden by .env)
    MAX_PAGES: int = 0
    MAX_INBOX: int = 0
    MAX_PRODUCTIVE_TO_PROCESS: int = 0
    TREE_SEARCH_METHOD: str = "GEDCOM" # Default search method

    # --- Initializer ---
    def __init__(self):
        """Initializes the Config_Class by loading values."""
        self._load_values()
        self._validate_critical_configs()
    # End of __init__

    def _load_values(self):
        """Loads configuration values from environment variables or defaults."""
        logger.debug("Loading application configuration settings...")

        # === Credentials & Identifiers ===
        self.ANCESTRY_USERNAME: str = self._get_string_env("ANCESTRY_USERNAME", "")
        self.ANCESTRY_PASSWORD: str = self._get_string_env("ANCESTRY_PASSWORD", "")
        self.TREE_NAME: str = self._get_string_env(
            "TREE_NAME", ""
        )  # Optional tree name for ID lookup
        self.MY_PROFILE_ID: Optional[str] = self._get_string_env(
            "MY_PROFILE_ID", ""
        )  # Optional pre-set profile ID
        self.MS_GRAPH_CLIENT_ID: str = self._get_string_env(
            "MS_GRAPH_CLIENT_ID", ""
        )  # Required for MS Graph
        self.MS_GRAPH_TENANT_ID: str = self._get_string_env(
            "MS_GRAPH_TENANT_ID", "common"
        )  # Default 'common'
        self.MS_TODO_LIST_NAME: str = self._get_string_env(
            "MS_TODO_LIST_NAME", "Tasks"
        )  # Default MS To-Do list

        # === Testing Specific Configuration ===
        loaded_test_id = self._get_string_env(
            "TESTING_PROFILE_ID", self.TESTING_PROFILE_ID or ""
        )
        if loaded_test_id != self.TESTING_PROFILE_ID and loaded_test_id:
            logger.info(
                f"Overriding default TESTING_PROFILE_ID with env var: '{loaded_test_id}'"
            )
            self.TESTING_PROFILE_ID = loaded_test_id.upper()  # Ensure uppercase
        elif self.TESTING_PROFILE_ID:
            logger.info(
                f"Using default TESTING_PROFILE_ID: '{self.TESTING_PROFILE_ID}' (set in .env to override)"
            )
            self.TESTING_PROFILE_ID = (
                self.TESTING_PROFILE_ID.upper()
            )  # Ensure uppercase
        else:
            logger.warning("TESTING_PROFILE_ID is not set in environment or defaults.")
            self.TESTING_PROFILE_ID = None  # Ensure it's None

        # === Paths & Files ===
        log_dir_name = self._get_string_env("LOG_DIR", "Logs")
        data_dir_name = self._get_string_env("DATA_DIR", "Data")
        cache_dir_name = self._get_string_env("CACHE_DIR", "Cache")
        # Resolve paths relative to the project root or use absolute paths from .env
        self.LOG_DIR: Path = Path(log_dir_name).resolve()
        self.DATA_DIR: Path = Path(data_dir_name).resolve()
        self.CACHE_DIR: Path = Path(cache_dir_name).resolve()
        # Create directories if they don't exist
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Database file location within DATA_DIR
        self.DATABASE_FILE: Path = self.DATA_DIR / self._get_string_env(
            "DATABASE_FILE", "ancestry_data.db"
        )
        # Optional GEDCOM file path
        self.GEDCOM_FILE_PATH: Optional[Path] = self._get_path_env(
            "GEDCOM_FILE_PATH", None
        )
        # Cache directory path (redundant but kept for clarity)
        self.CACHE_DIR_PATH: Path = self.CACHE_DIR

        # === URLs ===
        self.BASE_URL: str = self._get_string_env(
            "BASE_URL", "https://www.ancestry.co.uk/"
        )
        # Optional alternative API base (currently unused)
        self.ALTERNATIVE_API_URL: Optional[str] = self._get_env_var(
            "ALTERNATIVE_API_URL"
        )
        # Path for API V2 (relative to BASE_URL)
        self.API_BASE_URL_PATH: str = self._get_string_env(
            "API_BASE_URL_PATH", "/api/v2/"
        )
        # Construct full API base URL
        self.API_BASE_URL: Optional[str] = (
            urljoin(self.BASE_URL, self.API_BASE_URL_PATH)
            if self.BASE_URL and self.API_BASE_URL_PATH
            else None
        )

        # === Application Behavior ===
        self.APP_MODE: str = self._get_string_env(
            "APP_MODE", "dry_run"
        )  # Modes: dry_run, testing, production
        self.MAX_PAGES: int = self._get_int_env(
            "MAX_PAGES", self.MAX_PAGES
        )  # Limit Action 6 page processing
        self.MAX_RETRIES: int = self._get_int_env(
            "MAX_RETRIES", 5
        )  # Default retries for decorators
        self.MAX_INBOX: int = self._get_int_env(
            "MAX_INBOX", self.MAX_INBOX
        )  # Limit Action 7 inbox scan
        self.BATCH_SIZE: int = self._get_int_env(
            "BATCH_SIZE", 50
        )  # DB commit batch size
        self.MAX_PRODUCTIVE_TO_PROCESS: int = self._get_int_env(
            "MAX_PRODUCTIVE_TO_PROCESS", self.MAX_PRODUCTIVE_TO_PROCESS
        )  # Limit Action 9

        # === Database ===
        self.DB_POOL_SIZE: int = self._get_int_env(
            "DB_POOL_SIZE", self.DB_POOL_SIZE
        )  # SQLAlchemy pool size

        # === Caching ===
        self.CACHE_TIMEOUT: int = self._get_int_env(
            "CACHE_TIMEOUT", 3600
        )  # Default cache expiry (1 hour)

        # === Rate Limiting ===
        self.INITIAL_DELAY = self._get_float_env("INITIAL_DELAY", self.INITIAL_DELAY)
        self.MAX_DELAY = self._get_float_env("MAX_DELAY", self.MAX_DELAY)
        self.BACKOFF_FACTOR = self._get_float_env("BACKOFF_FACTOR", self.BACKOFF_FACTOR)
        self.DECREASE_FACTOR = self._get_float_env(
            "DECREASE_FACTOR", self.DECREASE_FACTOR
        )
        self.TOKEN_BUCKET_CAPACITY = self._get_float_env(
            "TOKEN_BUCKET_CAPACITY", self.TOKEN_BUCKET_CAPACITY
        )
        self.TOKEN_BUCKET_FILL_RATE = self._get_float_env(
            "TOKEN_BUCKET_FILL_RATE", self.TOKEN_BUCKET_FILL_RATE
        )
        # Log loaded rate limiter values
        logger.debug(
            f"Rate Limiter Loaded: InitialDelay={self.INITIAL_DELAY:.2f}s, Backoff={self.BACKOFF_FACTOR:.2f}, Decrease={self.DECREASE_FACTOR:.2f}"
        )
        logger.debug(
            f"Token Bucket Loaded: Capacity={self.TOKEN_BUCKET_CAPACITY:.1f}, FillRate={self.TOKEN_BUCKET_FILL_RATE:.1f}/sec"
        )

        # === Retry Status Codes ===
        retry_codes_env = self._get_json_env(
            "RETRY_STATUS_CODES", list(self.RETRY_STATUS_CODES)
        )  # Default to list for parsing
        if isinstance(retry_codes_env, (list, tuple)) and all(
            isinstance(code, int) for code in retry_codes_env
        ):
            self.RETRY_STATUS_CODES = tuple(retry_codes_env)  # Store as tuple
        else:
            logger.warning(
                f"RETRY_STATUS_CODES from env ('{retry_codes_env}') invalid. Using default: {self.RETRY_STATUS_CODES}"
            )
            # Ensure default is used if parsing failed
            self.RETRY_STATUS_CODES = tuple(
                self._get_json_env("RETRY_STATUS_CODES", self.RETRY_STATUS_CODES)
            )

        # === AI Configuration ===
        self.AI_PROVIDER: str = self._get_string_env(
            "AI_PROVIDER", ""
        ).lower()  # deepseek or gemini
        # DeepSeek specific
        self.DEEPSEEK_API_KEY: str = self._get_string_env("DEEPSEEK_API_KEY", "")
        self.DEEPSEEK_AI_MODEL: str = self._get_string_env(
            "DEEPSEEK_AI_MODEL", "deepseek-chat"
        )
        self.DEEPSEEK_AI_BASE_URL: Optional[str] = self._get_string_env(
            "DEEPSEEK_AI_BASE_URL", "https://api.deepseek.com"
        )
        # Google Gemini specific
        self.GOOGLE_API_KEY: str = self._get_string_env("GOOGLE_API_KEY", "")
        self.GOOGLE_AI_MODEL: str = self._get_string_env(
            "GOOGLE_AI_MODEL", "gemini-1.5-flash-latest"
        )
        # AI Context limits
        self.AI_CONTEXT_MESSAGES_COUNT = self._get_int_env(
            "AI_CONTEXT_MESSAGES_COUNT", self.AI_CONTEXT_MESSAGES_COUNT
        )
        self.AI_CONTEXT_MESSAGE_MAX_WORDS = self._get_int_env(
            "AI_CONTEXT_MESSAGE_MAX_WORDS", self.AI_CONTEXT_MESSAGE_MAX_WORDS
        )
        logger.debug(
            f"AI Context Limits: Last {self.AI_CONTEXT_MESSAGES_COUNT} msgs, Max {self.AI_CONTEXT_MESSAGE_MAX_WORDS} words/msg."
        )
        # Log AI provider info
        logger.info(f"AI Provider Configured: '{self.AI_PROVIDER or 'None'}'")
        if self.AI_PROVIDER == "deepseek":
            logger.info(
                f"  DeepSeek Settings: Model='{self.DEEPSEEK_AI_MODEL}', BaseURL='{self.DEEPSEEK_AI_BASE_URL}', KeySet={'Yes' if self.DEEPSEEK_API_KEY else 'No'}"
            )
        elif self.AI_PROVIDER == "gemini":
            logger.info(
                f"  Google Settings: Model='{self.GOOGLE_AI_MODEL}', KeySet={'Yes' if self.GOOGLE_API_KEY else 'No'}"
            )

        # === API Contextual Headers ===
        # Define default headers used by _api_req based on the 'api_description' parameter.
        # _api_req will merge these with other headers (User-Agent, CSRF, UBE, etc.) as needed.
        # Use None for values that should be dynamically added by _api_req (like ancestry-userid).
        parsed_base_url = urlparse(self.BASE_URL)
        origin_header_value = f"{parsed_base_url.scheme}://{parsed_base_url.netloc}"
        default_list_referer = urljoin(self.BASE_URL, "/discoveryui-matches/list/")

        self.API_CONTEXTUAL_HEADERS: Dict[str, Dict[str, Optional[str]]] = {
            # --- Headers for User Identifier APIs ---
            "Get my profile_id": {"ancestry-clientpath": "p13n-js"},
            "Tree Owner Name API": {"ancestry-clientpath": "Browser:meexp-uhome"},

            # --- Headers for Profile Details API ---
            "Profile Details API (Batch)": {"ancestry-clientpath": "express-fe"},
            "Profile Details API (Action 7)": {"ancestry-clientpath": "express-fe"},

            # --- Headers for Messaging APIs (Action 7/8/9) ---
            "Create Conversation API": {"ancestry-clientpath": "express-fe"},
            "Send Message API (Existing Conv)": {"ancestry-clientpath": "express-fe"},
            "Get Inbox Conversations": {"ancestry-clientpath": "express-fe"},
            "Fetch Conversation Context": {"ancestry-clientpath": "express-fe"},

            # --- APIs Where Context Might Still Be Useful (But Minimal) ---
            # These likely don't *strictly* need context anymore, but kept for potential minor differences
            "CSRF Token API": {},  # No special headers needed beyond defaults
            "Get UUID API": {},  # No special headers needed beyond defaults
            "Header Trees API": {},  # No special headers needed beyond defaults
            "Match Details API (Batch)": {},  # No special headers needed beyond defaults
            "Badge Details API (Batch)": {},  # No special headers needed beyond defaults
            "Get Ladder API (Batch)": {},  # No special headers needed beyond defaults
        }

        # === Tree Search Method ===
        loaded_search_method = self._get_string_env(
            "TREE_SEARCH_METHOD", self.TREE_SEARCH_METHOD
        ).upper()
        if loaded_search_method in ["GEDCOM", "API", "NONE"]:
            self.TREE_SEARCH_METHOD = loaded_search_method
            logger.info(f"Tree Search Method set to: {self.TREE_SEARCH_METHOD}")
        else:
            logger.warning(
                f"Invalid TREE_SEARCH_METHOD '{loaded_search_method}' in config. Defaulting to 'GEDCOM'."
            )
            self.TREE_SEARCH_METHOD = "GEDCOM"

        # === Final Logging of Key Config Values ===
        logger.info(
            f"Config Loaded: BASE_URL='{self.BASE_URL}', DB='{self.DATABASE_FILE.name}', TREE='{self.TREE_NAME or 'N/A'}'"
        )
        max_pages_log = (
            f"MaxPages={self.MAX_PAGES if self.MAX_PAGES > 0 else 'Unlimited'}"
        )
        max_inbox_log = (
            f"MaxInbox={self.MAX_INBOX if self.MAX_INBOX > 0 else 'Unlimited'}"
        )
        max_prod_log = f"MaxProductive={self.MAX_PRODUCTIVE_TO_PROCESS if self.MAX_PRODUCTIVE_TO_PROCESS > 0 else 'Unlimited'}"
        logger.info(
            f"Processing Limits: {max_pages_log}, {max_inbox_log}, {max_prod_log}"
        )
        if not self.ANCESTRY_USERNAME or not self.ANCESTRY_PASSWORD:
            logger.warning(
                "Ancestry credentials (USERNAME/PASSWORD) missing in config!"
            )
        logger.info("--- MS Graph Config ---")
        logger.info(f"  Client ID Loaded: {'Yes' if self.MS_GRAPH_CLIENT_ID else 'No'}")
        logger.info(f"  Tenant ID: {self.MS_GRAPH_TENANT_ID}")
        logger.info(f"  To-Do List Name: '{self.MS_TODO_LIST_NAME}'")
        logger.info("----------------------")
        logger.debug("Application config loading complete.\n")
    # End of _load_values

    def _validate_critical_configs(self):
        """Validates essential configuration values after loading."""
        logger.debug("Validating critical configuration settings...")
        errors_found = []

        # === Credentials ===
        if not self.ANCESTRY_USERNAME:
            errors_found.append("ANCESTRY_USERNAME is missing or empty in .env file.")
        if not self.ANCESTRY_PASSWORD:
            errors_found.append("ANCESTRY_PASSWORD is missing or empty in .env file.")

        # === AI Provider & Keys ===
        # Check AI provider validity first
        if self.AI_PROVIDER not in ["deepseek", "gemini", ""]:  # Allow empty provider
            logger.warning(
                f"AI_PROVIDER '{self.AI_PROVIDER}' is not recognized (expected 'deepseek', 'gemini', or empty). AI features may fail."
            )
            # Not making this fatal, as user might not intend to use AI

        # Check keys based on selected provider
        elif self.AI_PROVIDER == "deepseek":
            if not self.DEEPSEEK_API_KEY:
                errors_found.append(
                    "AI_PROVIDER is 'deepseek' but DEEPSEEK_API_KEY is missing in .env file."
                )
            if not self.DEEPSEEK_AI_BASE_URL:
                # Base URL is technically optional if using standard OpenAI endpoint via DeepSeek key,
                # but usually needed for DeepSeek. Add warning, not error.
                logger.warning(
                    "DEEPSEEK_AI_BASE_URL is not set in .env file. Ensure this is intended if using DeepSeek."
                )
        elif self.AI_PROVIDER == "gemini":
            if not self.GOOGLE_API_KEY:
                errors_found.append(
                    "AI_PROVIDER is 'gemini' but GOOGLE_API_KEY is missing in .env file."
                )
        elif not self.AI_PROVIDER:
            logger.info(
                "AI_PROVIDER is not set. AI-dependent features (Actions 7 & 9 sentiment/extraction) will be skipped or may fail if attempted."
            )

        # === MS Graph (Optional Feature - Action 9 Tasks) ===
        # Only warn if Client ID is missing, as MS Graph tasks are optional
        if not self.MS_GRAPH_CLIENT_ID:
            logger.warning(
                "MS_GRAPH_CLIENT_ID is missing in .env file. MS To-Do task creation (Action 9) will fail authentication."
            )
        # No need to validate tenant ID or list name as critically here

        # === Database Path ===
        # Check if the database file path seems valid (e.g., parent dir exists)
        try:
            db_parent_dir = self.DATABASE_FILE.parent
            if not db_parent_dir.exists():
                # Try creating it, maybe it's just missing
                try:
                    db_parent_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(
                        f"Created missing parent directory for database: {db_parent_dir}"
                    )
                except OSError as mkdir_err:
                    errors_found.append(
                        f"Parent directory for DATABASE_FILE ('{db_parent_dir}') does not exist and could not be created: {mkdir_err}"
                    )
            elif not db_parent_dir.is_dir():
                errors_found.append(
                    f"Parent path for DATABASE_FILE ('{db_parent_dir}') exists but is not a directory."
                )
            # We don't check if the file itself exists here, as SQLAlchemy creates it.
        except Exception as path_err:
            errors_found.append(
                f"Error checking DATABASE_FILE path ('{self.DATABASE_FILE}'): {path_err}"
            )

        # === Testing Profile ID (If mode is 'testing') ===
        if self.APP_MODE == "testing" and not self.TESTING_PROFILE_ID:
            errors_found.append(
                "APP_MODE is 'testing' but TESTING_PROFILE_ID is not set in .env file or defaults."
            )

        # === Report Errors or Success ===
        if errors_found:
            logger.critical("--- CRITICAL CONFIGURATION ERRORS ---")
            for error in errors_found:
                logger.critical(f" - {error}")
            logger.critical("---------------------------------------")
            # Raise a specific error to halt execution cleanly
            raise ValueError(
                "Critical configuration(s) missing or invalid. Please check .env file and logs, then restart."
            )
        else:
            logger.info("Critical configuration settings validated successfully.")

    # End of _validate_critical_configs


# End of Config_Class class


# --- Selenium Specific Configuration Class ---
class SeleniumConfig(BaseConfig):
    """
    Loads and provides access to Selenium WebDriver specific configuration settings,
    primarily from environment variables defined in the .env file. Includes paths,
    modes, timeouts, and factory methods for WebDriverWait.
    """

    # --- Default Selenium Settings (can be overridden by .env) ---
    HEADLESS_MODE: bool = False
    PROFILE_DIR: str = "Default"  # Chrome profile directory name
    DEBUG_PORT: int = 9516  # Default debug port (rarely needed)
    CHROME_MAX_RETRIES: int = 3  # Retries for driver initialization
    CHROME_RETRY_DELAY: int = 5  # Delay between init retries (seconds)
    # Timeouts (seconds)
    IMPLICIT_WAIT: int = 0  # Implicit wait (generally avoid, use explicit waits)
    ELEMENT_TIMEOUT: int = 20  # Default explicit wait for element presence/visibility
    PAGE_TIMEOUT: int = 40  # Default explicit wait for page load state
    ASYNC_SCRIPT_TIMEOUT: int = 60  # Timeout for execute_async_script
    LOGGED_IN_CHECK_TIMEOUT: int = 15  # Specific timeout for login status UI check
    MODAL_TIMEOUT: int = 10  # Timeout for waiting for modals/popups
    DNA_LIST_PAGE_TIMEOUT: int = 30  # Specific timeout for DNA list page load
    NEW_TAB_TIMEOUT: int = 15  # Timeout waiting for new tab handle
    TWO_FA_CODE_ENTRY_TIMEOUT: int = (
        300  # Max time to wait for manual 2FA entry (5 mins)
    )
    API_TIMEOUT: int = 60  # Default timeout for requests library calls via _api_req

    # --- Initializer ---
    def __init__(self):
        """Initializes the SeleniumConfig by loading values."""
        self._load_values()
        logger.debug("Selenium configuration loaded.")

    # End of __init__

    def _load_values(self):
        """Loads Selenium specific values from environment variables or defaults."""
        # Optional paths for driver and browser executables
        self.CHROME_DRIVER_PATH: Optional[Path] = self._get_path_env(
            "CHROME_DRIVER_PATH", None
        )
        self.CHROME_BROWSER_PATH: Optional[Path] = self._get_path_env(
            "CHROME_BROWSER_PATH", None
        )
        # Path for Chrome user data directory (profile storage)
        default_user_data_str = str(
            Path.home() / ".ancestry_chrome_data"
        )  # Default location in user home
        self.CHROME_USER_DATA_DIR: Optional[Path] = self._get_path_env(
            "CHROME_USER_DATA_DIR", default_user_data_str
        )
        # Ensure user data directory exists if specified
        if self.CHROME_USER_DATA_DIR:
            try:
                self.CHROME_USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(
                    f"Failed to create CHROME_USER_DATA_DIR: {self.CHROME_USER_DATA_DIR} - {e}"
                )

        # Load boolean/string/integer settings
        self.HEADLESS_MODE = self._get_bool_env("HEADLESS_MODE", self.HEADLESS_MODE)
        self.PROFILE_DIR = self._get_string_env("PROFILE_DIR", self.PROFILE_DIR)
        self.DEBUG_PORT = self._get_int_env("DEBUG_PORT", self.DEBUG_PORT)
        self.CHROME_MAX_RETRIES = self._get_int_env(
            "CHROME_MAX_RETRIES", self.CHROME_MAX_RETRIES
        )
        self.CHROME_RETRY_DELAY = self._get_int_env(
            "CHROME_RETRY_DELAY", self.CHROME_RETRY_DELAY
        )

        # Load timeout values
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

        # Log key Selenium config settings
        logger.debug(
            f" ChromeDriver Path: {self.CHROME_DRIVER_PATH.resolve() if self.CHROME_DRIVER_PATH else 'Managed by UC'}"
        )
        logger.debug(
            f" Chrome Browser Path: {self.CHROME_BROWSER_PATH.resolve() if self.CHROME_BROWSER_PATH else 'System Default'}"
        )
        logger.debug(
            f" Chrome User Data Dir: {self.CHROME_USER_DATA_DIR.resolve() if self.CHROME_USER_DATA_DIR else 'Temporary'}"
        )
        logger.debug(f" Chrome Profile Dir: {self.PROFILE_DIR}")
        logger.debug(f" Headless Mode: {self.HEADLESS_MODE}")
        logger.debug(f" Default Element Timeout: {self.ELEMENT_TIMEOUT}s")
        logger.debug(f" Default Page Load Timeout: {self.PAGE_TIMEOUT}s")
        logger.debug(f" API Request Timeout (requests lib): {self.API_TIMEOUT}s")

    # End of _load_values

    # --- WebDriverWait Factory Methods ---
    # These methods provide convenient ways to create WebDriverWait instances
    # with predefined timeouts based on the configuration.

    def default_wait(self, driver, timeout: Optional[int] = None) -> WebDriverWait:
        """Returns WebDriverWait with default element timeout."""
        return WebDriverWait(
            driver, timeout if timeout is not None else self.ELEMENT_TIMEOUT
        )

    # End of default_wait

    def page_load_wait(self, driver, timeout: Optional[int] = None) -> WebDriverWait:
        """Returns WebDriverWait with default page load timeout."""
        return WebDriverWait(
            driver, timeout if timeout is not None else self.PAGE_TIMEOUT
        )

    # End of page_load_wait

    def short_wait(self, driver, timeout: int = 5) -> WebDriverWait:
        """Returns WebDriverWait with a fixed short timeout (default 5s)."""
        return WebDriverWait(driver, timeout)

    # End of short_wait

    def long_wait(self, driver, timeout: Optional[int] = None) -> WebDriverWait:
        """Returns WebDriverWait with a long timeout (default: 2FA entry timeout)."""
        return WebDriverWait(
            driver, timeout if timeout is not None else self.TWO_FA_CODE_ENTRY_TIMEOUT
        )

    # End of long_wait

    def logged_in_check_wait(self, driver) -> WebDriverWait:
        """Returns WebDriverWait with timeout specific for login UI checks."""
        return WebDriverWait(driver, self.LOGGED_IN_CHECK_TIMEOUT)

    # End of logged_in_check_wait

    def element_wait(self, driver) -> WebDriverWait:
        """Returns WebDriverWait using the default ELEMENT_TIMEOUT."""
        return WebDriverWait(driver, self.ELEMENT_TIMEOUT)

    # End of element_wait

    def page_wait(self, driver) -> WebDriverWait:
        """Returns WebDriverWait using the default PAGE_TIMEOUT."""
        return WebDriverWait(driver, self.PAGE_TIMEOUT)

    # End of page_wait

    def modal_wait(self, driver) -> WebDriverWait:
        """Returns WebDriverWait using the default MODAL_TIMEOUT."""
        return WebDriverWait(driver, self.MODAL_TIMEOUT)

    # End of modal_wait

    def dna_list_page_wait(self, driver) -> WebDriverWait:
        """Returns WebDriverWait using the DNA_LIST_PAGE_TIMEOUT."""
        return WebDriverWait(driver, self.DNA_LIST_PAGE_TIMEOUT)

    # End of dna_list_page_wait

    def new_tab_wait(self, driver) -> WebDriverWait:
        """Returns WebDriverWait using the NEW_TAB_TIMEOUT."""
        return WebDriverWait(driver, self.NEW_TAB_TIMEOUT)

    # End of new_tab_wait


# End of SeleniumConfig class

# --- Create Singleton Instances ---
# These instances are created when the module is imported and can be accessed globally.
# Wrap this in a try...except to handle validation errors during standalone testing
try:
    config_instance = Config_Class()
    selenium_config = SeleniumConfig()
    _config_valid = True  # Flag to indicate successful load
except ValueError as config_err:
    # Log the specific validation error message raised by _validate_critical_configs
    logger.critical(f"CONFIG VALIDATION FAILED during initial load: {config_err}")
    # Set instances to None so the test block knows loading failed
    config_instance = None
    selenium_config = None
    _config_valid = False
except Exception as general_err:
    # Catch any other unexpected errors during instantiation
    logger.critical(
        f"UNEXPECTED ERROR during config instantiation: {general_err}", exc_info=True
    )
    config_instance = None
    selenium_config = None
    _config_valid = False


# --- Log Module Load ---
if _config_valid:
    logger.debug("config.py loaded and configuration instances created successfully.")
else:
    logger.error(
        "config.py loaded, but configuration instance creation FAILED validation."
    )

# --- Standalone Test Block ---
if __name__ == "__main__":
    print(f"\n--- Running {__file__} standalone test ---")

    # Check if config loading was successful before trying to print values
    if not _config_valid or config_instance is None or selenium_config is None:
        print("\nERROR: Configuration loading failed during module import.")
        print("Please check the log output above for critical configuration errors.")
        print("Standalone test cannot proceed.")
    else:
        # --- Proceed with printing config values if loading succeeded ---
        print("\n--- General Config (config_instance) ---")
        print(f"  APP_MODE: {config_instance.APP_MODE}")
        print(f"  LOG_LEVEL: {config_instance.LOG_LEVEL}")
        print(f"  BASE_URL: {config_instance.BASE_URL}")
        print(
            f"  USERNAME: {config_instance.ANCESTRY_USERNAME[:3]}***"
            if config_instance.ANCESTRY_USERNAME
            else "Not Set"
        )  # Mask username
        print(
            f"  PASSWORD: {'*' * len(config_instance.ANCESTRY_PASSWORD) if config_instance.ANCESTRY_PASSWORD else 'Not Set'}"
        )  # Mask password
        print(f"  DATABASE_FILE: {config_instance.DATABASE_FILE}")
        print(f"  LOG_DIR: {config_instance.LOG_DIR}")
        print(f"  DATA_DIR: {config_instance.DATA_DIR}")
        print(f"  CACHE_DIR: {config_instance.CACHE_DIR}")
        print(f"  GEDCOM_FILE_PATH: {config_instance.GEDCOM_FILE_PATH or 'Not Set'}")
        print(f"  TREE_NAME: {config_instance.TREE_NAME or 'Not Set'}")
        print(
            f"  MY_PROFILE_ID (Env): {os.getenv('MY_PROFILE_ID', 'Not Set')}"
        )  # Show if set in env
        print(
            f"  TESTING_PROFILE_ID: {config_instance.TESTING_PROFILE_ID or 'Not Set'}"
        )
        print(f"  MAX_PAGES: {config_instance.MAX_PAGES}")
        print(f"  MAX_INBOX: {config_instance.MAX_INBOX}")
        print(f"  MAX_PRODUCTIVE: {config_instance.MAX_PRODUCTIVE_TO_PROCESS}")
        print(f"  BATCH_SIZE: {config_instance.BATCH_SIZE}")
        print(f"  CACHE_TIMEOUT: {config_instance.CACHE_TIMEOUT}s")
        print(f"  RETRY_CODES: {config_instance.RETRY_STATUS_CODES}")
        print(f"  TREE_SEARCH_METHOD: {config_instance.TREE_SEARCH_METHOD}")

        print("\n--- AI Config ---")
        print(f"  AI_PROVIDER: {config_instance.AI_PROVIDER or 'Not Set'}")
        if config_instance.AI_PROVIDER == "deepseek":
            print(f"  DEEPSEEK_MODEL: {config_instance.DEEPSEEK_AI_MODEL}")
            print(f"  DEEPSEEK_BASE_URL: {config_instance.DEEPSEEK_AI_BASE_URL}")
            print(
                f"  DEEPSEEK_API_KEY: {'Set' if config_instance.DEEPSEEK_API_KEY else 'Not Set'}"
            )
        elif config_instance.AI_PROVIDER == "gemini":
            print(f"  GOOGLE_MODEL: {config_instance.GOOGLE_AI_MODEL}")
            print(
                f"  GOOGLE_API_KEY: {'Set' if config_instance.GOOGLE_API_KEY else 'Not Set'}"
            )
        print(f"  AI Context Msgs: {config_instance.AI_CONTEXT_MESSAGES_COUNT}")
        print(f"  AI Context Words: {config_instance.AI_CONTEXT_MESSAGE_MAX_WORDS}")

        print("\n--- MS Graph Config ---")
        print(
            f"  MS_GRAPH_CLIENT_ID: {'Set' if config_instance.MS_GRAPH_CLIENT_ID else 'Not Set'}"
        )
        print(f"  MS_GRAPH_TENANT_ID: {config_instance.MS_GRAPH_TENANT_ID}")
        print(f"  MS_TODO_LIST_NAME: {config_instance.MS_TODO_LIST_NAME}")

        print("\n--- Selenium Config (selenium_config) ---")
        print(f"  HEADLESS_MODE: {selenium_config.HEADLESS_MODE}")
        print(
            f"  CHROME_DRIVER_PATH: {selenium_config.CHROME_DRIVER_PATH or 'Managed by UC'}"
        )
        print(
            f"  CHROME_BROWSER_PATH: {selenium_config.CHROME_BROWSER_PATH or 'System Default'}"
        )
        print(f"  CHROME_USER_DATA_DIR: {selenium_config.CHROME_USER_DATA_DIR}")
        print(f"  PROFILE_DIR: {selenium_config.PROFILE_DIR}")
        print(f"  ELEMENT_TIMEOUT: {selenium_config.ELEMENT_TIMEOUT}s")
        print(f"  PAGE_TIMEOUT: {selenium_config.PAGE_TIMEOUT}s")
        print(f"  API_TIMEOUT: {selenium_config.API_TIMEOUT}s")
        print(f"  CHROME_MAX_RETRIES: {selenium_config.CHROME_MAX_RETRIES}")
        print(f"  CHROME_RETRY_DELAY: {selenium_config.CHROME_RETRY_DELAY}s")

        print(
            f"\n--- Standalone Test Complete ({'OK' if _config_valid else 'FAILED - Check Logs'}) ---"
        )
# End of config.py standalone test block



# --- End of config.py ---