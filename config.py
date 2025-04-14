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
            # --- Headers for CSRF Token Retrieval ---
            "CSRF Token API": {
                "Accept": "application/json",  # Expect JSON, though API returns text
                "Referer": default_list_referer,
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                # No Origin needed
            },
            "CSRF Token API Test": {  # For standalone test
                "Accept": "application/json",  # Expect JSON, though API returns text
                "Referer": default_list_referer,
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
            },
            # --- Headers for User Identifier APIs ---
            "Get my profile_id": {
                "Accept": "application/json, text/plain, */*",
                "ancestry-clientpath": "p13n-js",
                "Referer": self.BASE_URL,  # Referer is base URL
                # No Origin needed
            },
            "Get UUID API": {  # header/dna endpoint
                "Accept": "application/json",
                "Referer": self.BASE_URL,
                # No Origin needed
            },
            "API Login Verification (header/dna)": {  # Used by login_status
                "Accept": "application/json",
                "Referer": self.BASE_URL,
                # No Origin needed
            },
            # --- Headers for Tree Information APIs ---
            "Header Trees API": {  # Used for getting tree ID from name
                "Accept": "*/*",  # Accepts anything
                "Referer": self.BASE_URL,
                # No Origin needed
            },
            "Tree Owner Name API": {
                "Accept": "application/json, text/plain, */*",
                "ancestry-clientpath": "Browser:meexp-uhome",  # Specific client path
                "Referer": self.BASE_URL,
                # No Origin needed
            },
            # --- Headers for Match List & Details APIs (Action 6) ---
            "Match List API": {  # Specific endpoint with special handling in _api_req
                "Accept": "application/json",
                "Referer": default_list_referer,
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "cache-control": "no-cache",  # Explicitly disable cache
                "pragma": "no-cache",
                "priority": "u=1, i",  # Browser priority hint
                # Origin header removed by _api_req for this specific call
                # X-CSRF-Token added by _api_req based on specific cookie read in get_matches
            },
            "In-Tree Status Check": {  # POST request
                "Accept": "application/json",
                "Content-Type": "application/json",  # Specify JSON payload
                "Referer": default_list_referer,
                "Origin": origin_header_value,  # Requires Origin
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                # X-CSRF-Token added by _api_req based on specific cookie read in get_matches
            },
            "Match Details API (Batch)": {  # GET details for a single match
                "Accept": "application/json",
                "Referer": None,  # Referer set dynamically in calling function (_fetch_combined_details)
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                # No Origin needed
            },
            "Badge Details API (Batch)": {  # GET badge details for a single match
                "Accept": "application/json",
                "Referer": default_list_referer,
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                # No Origin needed
            },
            "Match Probability API (Cloudscraper)": {  # POST probability via Scraper
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Referer": default_list_referer,
                "Origin": origin_header_value,
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                # X-CSRF-Token added dynamically in _fetch_batch_relationship_prob
                # User-Agent added dynamically in _fetch_batch_relationship_prob
            },
            "Get Ladder API (Batch)": {  # Used by Action 6 to get relationship path
                "Accept": "*/*",  # Accepts anything (JSONP response)
                "Referer": None,  # Referer set dynamically in _fetch_batch_ladder
                # No Origin needed
            },
            # --- Headers for Profile Details API (Used by Action 7/9/utils) ---
            "Profile Details API (Batch)": {  # Used by Action 6 _fetch_combined_details
                "accept": "application/json",
                "ancestry-clientpath": "express-fe",  # Specific client path
                "cache-control": "no-cache",
                "pragma": "no-cache",
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-origin",
                "priority": "u=1, i",
                "Referer": None,  # Set dynamically based on context (e.g., compare page)
                "ancestry-userid": None,  # Added dynamically by _api_req using session_manager.my_profile_id
                # No Origin needed
            },
            "Profile Details API (Action 7)": {  # Used by Action 7 _fetch_profile_details_for_person
                "accept": "application/json",
                "ancestry-clientpath": "express-fe",
                "cache-control": "no-cache",
                "pragma": "no-cache",
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-origin",
                "priority": "u=1, i",
                "Referer": urljoin(
                    self.BASE_URL, "/messaging/"
                ),  # Referer is messaging page
                "ancestry-userid": None,  # Added dynamically by _api_req
                # No Origin needed
            },
            # --- Headers for Messaging APIs (Action 7/8/9) ---
            "Create Conversation API": {  # POST to create new thread
                "Accept": "*/*",
                "Content-Type": "application/json",
                "ancestry-clientpath": "express-fe",
                "Origin": origin_header_value,  # Requires Origin
                "Referer": urljoin(self.BASE_URL, "/messaging/"),
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "cache-control": "no-cache",
                "pragma": "no-cache",
                "priority": "u=1, i",
                "ancestry-userid": None,  # Added dynamically by _api_req
            },
            "Send Message API (Existing Conv)": {  # POST to existing thread
                "Accept": "*/*",
                "Content-Type": "application/json",
                "ancestry-clientpath": "express-fe",
                "Origin": origin_header_value,  # Requires Origin
                "Referer": urljoin(self.BASE_URL, "/messaging/"),
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "cache-control": "no-cache",
                "pragma": "no-cache",
                "priority": "u=1, i",
                "ancestry-userid": None,  # Added dynamically by _api_req
            },
            "Get Inbox Conversations": {  # GET conversation list overview
                "Accept": "*/*",
                "ancestry-clientpath": "express-fe",
                "Referer": urljoin(self.BASE_URL, "/messaging/"),
                "ancestry-userid": None,  # Added dynamically by _api_req
                # No Origin needed
            },
            "Fetch Conversation Context": {  # GET messages within a conversation
                "accept": "*/*",
                "ancestry-clientpath": "express-fe",
                "referer": urljoin(self.BASE_URL, "/messaging/"),
                "ancestry-userid": None,  # Added dynamically by _api_req
                # No Origin needed
            },
        }
        # --- End API Contextual Headers ---

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

        # Validate Ancestry Credentials
        if not self.ANCESTRY_USERNAME:
            errors_found.append("ANCESTRY_USERNAME is missing or empty.")
        if not self.ANCESTRY_PASSWORD:
            errors_found.append("ANCESTRY_PASSWORD is missing or empty.")

        # Validate AI Provider Configuration
        if self.AI_PROVIDER == "deepseek" and not self.DEEPSEEK_API_KEY:
            errors_found.append(
                "AI_PROVIDER is 'deepseek' but DEEPSEEK_API_KEY is missing."
            )
        elif self.AI_PROVIDER == "gemini" and not self.GOOGLE_API_KEY:
            errors_found.append(
                "AI_PROVIDER is 'gemini' but GOOGLE_API_KEY is missing."
            )
        elif not self.AI_PROVIDER:
            logger.warning(
                "AI_PROVIDER is not configured. AI features will be disabled."
            )
        elif self.AI_PROVIDER not in ["deepseek", "gemini"]:
            logger.warning(
                f"AI_PROVIDER '{self.AI_PROVIDER}' is not recognized. AI features may not work."
            )

        # Validate MS Graph Client ID (Needed for Action 9 task creation)
        # Only make it critical if Action 9 is likely to be used? For now, just check if empty.
        if not self.MS_GRAPH_CLIENT_ID:
            logger.warning(
                "MS_GRAPH_CLIENT_ID is missing. MS To-Do task creation (Action 9) will fail authentication."
            )
            # Decide if this should be fatal - maybe not if user doesn't intend to use Action 9?
            # errors_found.append("MS_GRAPH_CLIENT_ID is missing (required for Action 9 task creation).")

        # Add other critical checks here if needed (e.g., database path validity)

        if errors_found:
            for error in errors_found:
                logger.critical(f"CONFIG VALIDATION FAILED: {error}")
            # Option 1: Raise an exception to halt execution
            raise ValueError(
                "Critical configuration missing. Please check .env file and documentation."
            )
            # Option 2: Exit directly (less clean, but avoids further execution)
            # sys.exit("Critical configuration missing. Exiting.")
            # Option 3: Set an internal flag and let main.py check it (more complex)
            # self._is_valid = False
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
config_instance = Config_Class()
selenium_config = SeleniumConfig()

# --- Log Module Load ---
logger.debug("config.py loaded and configuration instances created.")

# --- End of config.py ---
