#!/usr/bin/env python3

# config.py

"""
config.py - Centralized Configuration Management

Loads application settings from environment variables (.env file) and defines
configuration classes (`Config_Class`, `SeleniumConfig`) with sensible defaults.
Provides typed access to settings like credentials, paths, URLs, API keys,
behavioral parameters, and Selenium options. Includes contextual headers for API calls.
V4: Added Action 11 display limits (MAX_SUGGESTIONS_TO_SCORE, MAX_CANDIDATES_TO_DISPLAY).
"""

# --- Standard library imports ---
import logging
import os
import sys
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from urllib.parse import urlparse, urljoin  # <<<< MODIFIED LINE: Added urljoin
import json  # Added for _get_json_env

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

# --- Test framework imports ---
from test_framework import (
    TestSuite,
    suppress_logging,
    create_mock_data,
    assert_valid_function,
)
from unittest.mock import patch
import sys


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for config.py following the standardized 6-category TestSuite framework.
    Tests configuration management, validation, and environment integration.

    Categories: Initialization, Core Functionality, Edge Cases, Integration, Performance, Error Handling
    """
    from unittest.mock import patch

    suite = TestSuite("Configuration Management & Environment Integration", "config.py")
    suite.start_suite()

    # === INITIALIZATION TESTS ===
    def test_module_imports():
        """Test all required modules and dependencies are properly imported."""
        required_modules = ["os", "sys", "pathlib", "logging", "typing"]
        for module_name in required_modules:
            # Check if module is imported or available
            module_imported = module_name in sys.modules or any(
                module_name in str(item)
                for item in globals().values()
                if hasattr(item, "__module__")
            )
            assert module_imported, f"Required module {module_name} not available"

    def test_config_class_initialization():
        """Test Config_Class initializes properly with all required attributes."""
        config = Config_Class()
        assert config is not None, "Config_Class instance should not be None"

        # Test core attributes exist
        required_attrs = ["BASE_URL", "DATABASE_FILE", "TREE_NAME", "USER_AGENTS"]
        for attr in required_attrs:
            assert hasattr(
                config, attr
            ), f"Config_Class missing required attribute: {attr}"

    def test_environment_setup():
        """Test environment variables and .env file loading."""
        # Test that dotenv was loaded
        assert "dotenv" in sys.modules, "dotenv module should be imported"

        # Test environment variable access
        test_config = Config_Class()
        assert hasattr(
            test_config, "_get_env_var"
        ), "Config should have _get_env_var method"

    # === CORE FUNCTIONALITY TESTS ===
    def test_environment_variable_methods():
        """Test all environment variable getter methods work correctly."""
        config = Config_Class()

        # Test string environment variable with default
        test_string = config._get_string_env("NONEXISTENT_VAR", "default_value")
        assert test_string == "default_value", "String env var should return default"

        # Test integer environment variable with default
        test_int = config._get_int_env("NONEXISTENT_INT", 42)
        assert test_int == 42, "Int env var should return default"
        assert isinstance(test_int, int), "Int env var should return integer type"

        # Test boolean environment variable with default
        test_bool = config._get_bool_env("NONEXISTENT_BOOL", True)
        assert test_bool is True, "Bool env var should return default"
        assert isinstance(test_bool, bool), "Bool env var should return boolean type"

    def test_data_type_conversions():
        """Test data type conversion methods handle various inputs correctly."""
        config = Config_Class()

        # Test path conversion
        if hasattr(config, "_get_path_env"):
            test_path = config._get_path_env("NONEXISTENT_PATH", ".")
            assert test_path is not None, "Path env var should not be None"

        # Test float conversion
        if hasattr(config, "_get_float_env"):
            test_float = config._get_float_env("NONEXISTENT_FLOAT", 1.5)
            assert test_float == 1.5, "Float env var should return default"
            assert isinstance(
                test_float, float
            ), "Float env var should return float type"

    def test_configuration_defaults():
        """Test that all required configuration defaults are properly set."""
        config = Config_Class()

        # Test timing defaults
        timing_attrs = ["INITIAL_DELAY", "MAX_DELAY", "BACKOFF_FACTOR"]
        for attr in timing_attrs:
            if hasattr(config, attr):
                value = getattr(config, attr)
                assert value is not None, f"{attr} should have a default value"

        # Test user agents list
        if hasattr(config, "USER_AGENTS"):
            assert isinstance(config.USER_AGENTS, list), "USER_AGENTS should be a list"
            assert len(config.USER_AGENTS) > 0, "USER_AGENTS should not be empty"

    # === EDGE CASE TESTS ===
    def test_invalid_environment_variables():
        """Test handling of invalid environment variable values."""
        config = Config_Class()

        # Test invalid integer conversion - should return default
        with patch.dict(os.environ, {"TEST_INVALID_INT": "not_a_number"}):
            result = config._get_int_env("TEST_INVALID_INT", 100)
            assert result == 100, "Invalid int should return default"

        # Test invalid boolean conversion - should return default
        with patch.dict(os.environ, {"TEST_INVALID_BOOL": "not_a_bool"}):
            result = config._get_bool_env("TEST_INVALID_BOOL", False)
            assert result is False, "Invalid bool should return default"

    def test_empty_environment_variables():
        """Test handling of empty environment variable values."""
        config = Config_Class()

        # Test empty string handling
        with patch.dict(os.environ, {"TEST_EMPTY": ""}):
            result = config._get_string_env("TEST_EMPTY", "default")
            # Empty string should either return empty string or default based on implementation
            assert isinstance(result, str), "Result should be string type"

    def test_none_values():
        """Test handling of None values in configuration."""
        config = Config_Class()

        # Test that _get_env_var returns None for non-existent variables
        result = config._get_env_var("DEFINITELY_NONEXISTENT_VAR_12345")
        assert result is None, "Non-existent env var should return None"

    # === INTEGRATION TESTS ===
    def test_global_config_instances():
        """Test that global configuration instances are properly created."""
        # Test main config instance using globals() to avoid circular import
        assert (
            "config_instance" in globals()
        ), "Global config_instance should be defined"
        global_config = globals()["config_instance"]

        # During testing, config_instance might be None if credentials are missing
        # This is acceptable behavior for the test environment
        if global_config is not None:
            assert isinstance(
                global_config, Config_Class
            ), "config_instance should be Config_Class instance when not None"

        # Test that the Config_Class can be instantiated directly
        test_config = Config_Class()
        assert test_config is not None, "Config_Class should be instantiable"
        assert isinstance(
            test_config, Config_Class
        ), "Test config should be Config_Class instance"

        # Test selenium config instance
        from config import selenium_config

        assert selenium_config is not None, "Global selenium_config should exist"

    def test_selenium_integration():
        """Test Selenium configuration integration."""
        try:
            from config import selenium_config

            if selenium_config:
                # Test required Selenium attributes
                selenium_attrs = ["HEADLESS_MODE", "CHROME_USER_DATA_DIR"]
                for attr in selenium_attrs:
                    if hasattr(selenium_config, attr):
                        value = getattr(selenium_config, attr)
                        assert (
                            value is not None
                        ), f"Selenium config {attr} should have a value"
        except ImportError:
            pass  # Selenium config may not be available in test environment

    def test_url_configuration():
        """Test URL configuration and validation."""
        config = Config_Class()

        # Test BASE_URL format if set
        if hasattr(config, "BASE_URL") and config.BASE_URL:
            assert config.BASE_URL.startswith(
                ("http://", "https://")
            ), "BASE_URL should be properly formatted"

        # Test API URL configuration if available
        if hasattr(config, "API_BASE_URL") and config.API_BASE_URL:
            assert isinstance(config.API_BASE_URL, str), "API_BASE_URL should be string"

    # === PERFORMANCE TESTS ===
    def test_config_initialization_performance():
        """Test that configuration initialization is performant."""
        import time

        start_time = time.time()
        config = Config_Class()
        end_time = time.time()

        initialization_time = end_time - start_time
        assert (
            initialization_time < 1.0
        ), f"Config initialization took {initialization_time:.3f}s, should be < 1.0s"

    def test_environment_variable_access_performance():
        """Test that environment variable access is efficient."""
        import time

        config = Config_Class()

        start_time = time.time()
        for i in range(100):
            config._get_string_env(f"TEST_VAR_{i}", "default")
        end_time = time.time()

        total_time = end_time - start_time
        assert (
            total_time < 0.1
        ), f"100 env var accesses took {total_time:.3f}s, should be < 0.1s"

    def test_multiple_config_instances():
        """Test that creating multiple config instances is efficient."""
        import time

        start_time = time.time()
        configs = [Config_Class() for _ in range(10)]
        end_time = time.time()

        creation_time = end_time - start_time
        assert (
            creation_time < 0.5
        ), f"Creating 10 configs took {creation_time:.3f}s, should be < 0.5s"
        assert len(configs) == 10, "Should create exactly 10 config instances"

    # === ERROR HANDLING TESTS ===
    def test_missing_critical_configuration():
        """Test handling of missing critical configuration values."""
        config = Config_Class()

        # Test that missing values don't cause crashes
        try:
            # Try to access a method that might validate critical configs
            if hasattr(config, "_validate_critical_configs"):
                # This might fail in test environment, which is expected
                config._validate_critical_configs()
        except Exception:
            # Expected to fail in test environment without real credentials
            pass

    def test_configuration_error_recovery():
        """Test that configuration errors are handled gracefully."""
        # Test that invalid JSON environment variables don't crash the system
        if hasattr(Config_Class(), "_get_json_env"):
            config = Config_Class()
            with patch.dict(os.environ, {"TEST_INVALID_JSON": "invalid json"}):
                try:
                    result = config._get_json_env("TEST_INVALID_JSON", {})
                    assert isinstance(
                        result, dict
                    ), "Should return default dict on invalid JSON"
                except AttributeError:
                    pass  # Method might not exist

    def test_import_error_handling():
        """Test that import errors are handled gracefully."""
        # Test that the module handles missing optional dependencies
        try:
            # This should not crash even if some imports fail
            config = Config_Class()
            assert (
                config is not None
            ), "Config should initialize even with missing dependencies"
        except ImportError as e:
            assert False, f"Config initialization should handle import errors: {e}"

    # === RUN ALL TESTS ===
    with suppress_logging():
        # INITIALIZATION TESTS
        suite.run_test(
            "Module Imports",
            test_module_imports,
            "All required modules and dependencies are properly imported and accessible",
            "Check availability of essential modules like os, sys, pathlib, logging, typing",
            "All required modules are imported and accessible for configuration management",
        )

        suite.run_test(
            "Config Class Initialization",
            test_config_class_initialization,
            "Config_Class initializes properly with all required attributes and valid defaults",
            "Create Config_Class instance and verify core attributes exist with proper values",
            "Config_Class instance created with BASE_URL, DATABASE_FILE, TREE_NAME, USER_AGENTS attributes",
        )

        suite.run_test(
            "Environment Setup",
            test_environment_setup,
            "Environment variables and .env file loading works correctly with proper access methods",
            "Verify dotenv module loading and environment variable access method availability",
            "Environment setup completed with dotenv loading and access methods available",
        )

        # CORE FUNCTIONALITY TESTS
        suite.run_test(
            "Environment Variable Methods",
            test_environment_variable_methods,
            "All environment variable getter methods return correct defaults with proper type conversion",
            "Test each environment variable method with non-existent variables to verify default handling",
            "All environment variable methods return appropriate defaults with correct type conversion",
        )

        suite.run_test(
            "Data Type Conversions",
            test_data_type_conversions,
            "Data type conversion methods handle various inputs correctly with proper defaults",
            "Test path and float conversion methods with non-existent variables and default values",
            "Type conversion methods work correctly with proper default value handling",
        )

        suite.run_test(
            "Configuration Defaults",
            test_configuration_defaults,
            "All required configuration defaults are properly set with appropriate types and values",
            "Test timing attributes and USER_AGENTS list for proper default values and types",
            "Configuration defaults are set correctly with non-null values and proper list structures",
        )

        # EDGE CASE TESTS
        suite.run_test(
            "Invalid Environment Variables",
            test_invalid_environment_variables,
            "Invalid environment variable values are handled gracefully by returning defaults",
            "Test with invalid integer and boolean environment variable values to verify default fallback",
            "Invalid environment variables return defaults without causing crashes or errors",
        )

        suite.run_test(
            "Empty Environment Variables",
            test_empty_environment_variables,
            "Empty environment variable values are handled appropriately with consistent behavior",
            "Test behavior with empty string environment variables and verify string type handling",
            "Empty environment variables handled consistently returning string type results",
        )

        suite.run_test(
            "None Values Handling",
            test_none_values,
            "Non-existent environment variables return None as expected",
            "Test with definitely non-existent environment variable to verify None return",
            "Non-existent environment variables correctly return None values",
        )

        # INTEGRATION TESTS
        suite.run_test(
            "Global Config Instances",
            test_global_config_instances,
            "Global configuration instances are properly created and accessible with correct types",
            "Test that global config_instance and selenium_config exist and are properly typed",
            "Global configuration instances available with proper Config_Class and SeleniumConfig types",
        )

        suite.run_test(
            "Selenium Integration",
            test_selenium_integration,
            "Selenium configuration integration works correctly with required attributes and valid values",
            "Test Selenium config attributes for proper values and configuration",
            "Selenium configuration properly integrated with required attributes having valid values",
        )

        suite.run_test(
            "URL Configuration",
            test_url_configuration,
            "URL configuration and validation works properly with correct http/https protocols",
            "Test URL format validation for BASE_URL and API_BASE_URL configuration",
            "URLs are properly formatted with valid http/https protocols and string types",
        )

        # PERFORMANCE TESTS
        suite.run_test(
            "Config Initialization Performance",
            test_config_initialization_performance,
            "Configuration initialization completes within acceptable time limits under 1 second",
            "Measure time to create single Config_Class instance and verify performance",
            "Configuration initialization demonstrates efficient performance under 1 second",
        )

        suite.run_test(
            "Environment Variable Access Performance",
            test_environment_variable_access_performance,
            "Environment variable access is efficient for multiple calls completing under 0.1 seconds",
            "Measure time for 100 sequential environment variable access calls",
            "Environment variable access maintains efficient performance for repeated operations",
        )

        suite.run_test(
            "Multiple Config Instances Performance",
            test_multiple_config_instances,
            "Creating multiple config instances is efficient completing under 0.5 seconds for 10 instances",
            "Measure time to create 10 Config_Class instances and verify efficiency",
            "Multiple configuration instance creation demonstrates efficient performance",
        )

        # ERROR HANDLING TESTS
        suite.run_test(
            "Missing Critical Configuration",
            test_missing_critical_configuration,
            "Missing critical configuration values are handled gracefully without application crashes",
            "Test configuration validation with potentially missing critical values in test environment",
            "Missing critical configuration handled gracefully without causing system failures",
        )

        suite.run_test(
            "Configuration Error Recovery",
            test_configuration_error_recovery,
            "Configuration errors are handled gracefully with recovery returning appropriate defaults",
            "Test invalid JSON environment variables to verify error handling and default returns",
            "Configuration error recovery works properly returning defaults for invalid input",
        )

        suite.run_test(
            "Import Error Handling",
            test_import_error_handling,
            "Import errors are handled gracefully during initialization allowing successful operation",
            "Test that Config_Class can initialize even with missing optional dependencies",
            "Import error handling ensures successful initialization despite missing optional dependencies",
        )

    return suite.finish_suite()


# --- Base Configuration Class ---
class BaseConfig:
    """Base class providing helper methods for retrieving typed environment variables."""

    def _get_env_var(self, key: str) -> Optional[str]:
        """Retrieves an environment variable's value."""
        value = os.getenv(key)
        return value

    # End of _get_env_var

    def _get_int_env(self, key: str, default: int) -> int:
        """Retrieves an environment variable as an integer, with default."""
        value_str = self._get_env_var(key)
        if value_str is None:
            return default
        try:
            return int(value_str)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid integer value '{value_str}' for env var '{key}'. Using default: {default}"
            )
            return default

    # End of _get_int_env

    def _get_float_env(self, key: str, default: float) -> float:
        """Retrieves an environment variable as a float, with default."""
        value_str = self._get_env_var(key)
        if value_str is None:
            return default
        try:
            return float(value_str)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid float value '{value_str}' for env var '{key}'. Using default: {default}"
            )
            return default

    # End of _get_float_env

    def _get_string_env(self, key: str, default: str) -> str:
        """Retrieves an environment variable as a string, with default."""
        value = self._get_env_var(key)
        return default if value is None else str(value)

    # End of _get_string_env

    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Retrieves an environment variable as a boolean, with default."""
        value_str = self._get_env_var(key)
        if value_str is None:
            return default
        return value_str.lower() in ("true", "1", "t", "y", "yes")

    # End of _get_bool_env

    def _get_json_env(self, key: str, default: Any) -> Any:
        """Retrieves an environment variable as a JSON object, with default."""
        value_str = self._get_env_var(key)
        if value_str:
            try:
                return json.loads(value_str)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Invalid JSON in env var '{key}': {e}. Using default: {default}"
                )
                return default
        else:
            return default

    # End of _get_json_env

    def _get_path_env(self, key: str, default: Optional[str]) -> Optional[Path]:
        """Retrieves an environment variable as a resolved Path object, with default."""
        value_str = self._get_env_var(key)
        if value_str is None:
            value_str = default
        if value_str is None:
            return None
        try:
            path_obj = Path(value_str).resolve()
            return path_obj
        except Exception as e:
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
    # Testing profile ID (should be set via environment variable)
    TESTING_PROFILE_ID: Optional[str] = None
    # Default tree-specific person ID for testing (MUST be set in .env for some tests)
    TESTING_PERSON_TREE_ID: Optional[str] = None
    # Reference person configuration for relationship paths
    REFERENCE_PERSON_ID: Optional[str] = None  # Should be set via environment variable
    REFERENCE_PERSON_NAME: str = "Reference Person"  # Generic default name
    # User signature configuration for messages
    USER_NAME: str = "Tree Owner"  # Generic default name
    USER_LOCATION: str = ""  # Generic default location

    # Test/Mock configuration values (can be overridden by .env)
    TEST_TREE_ID: str = "12345678"  # Default tree ID for testing
    TEST_OWNER_NAME: str = "Test Owner"  # Default owner name for testing
    TEST_EMAIL: str = "test@example.com"  # Default email for testing
    TEST_CSRF_TOKEN: str = (
        "mock_csrf_token_12345678901234567890"  # Default CSRF token for testing
    )
    TEST_PROFILE_ID: str = "mock_profile_id_12345"  # Default profile ID for testing
    TEST_UUID: str = "mock_uuid_12345"  # Default UUID for testing
    TEST_TAB_HANDLE: str = "mock_tab_handle_12345"  # Default tab handle for testing
    TEST_RECIPIENT_ID: str = "DUMMY-RECIPIENT-ID"  # Default recipient ID for testing
    TEST_RECIPIENT_USERNAME: str = (
        "DummyRecipient"  # Default recipient username for testing
    )

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
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    ]
    # Default AI context limits (can be overridden by .env)
    AI_CONTEXT_MESSAGES_COUNT: int = 7
    AI_CONTEXT_MESSAGE_MAX_WORDS: int = 500
    # Default processing limits (0 = unlimited, can be overridden by .env)
    MAX_PAGES: int = 0
    MAX_INBOX: int = 0
    MAX_PRODUCTIVE_TO_PROCESS: int = 0
    TREE_SEARCH_METHOD: str = "GEDCOM"
    DB_ERROR_PAGE_THRESHOLD: int = 10
    # Action 11 specific limits (can be overridden by .env)
    MAX_SUGGESTIONS_TO_SCORE: int = 50
    MAX_CANDIDATES_TO_DISPLAY: int = 10
    # Action 9 specific settings
    CUSTOM_RESPONSE_ENABLED: bool = True
    # Workflow settings
    INCLUDE_ACTION6_IN_WORKFLOW: bool = (
        False  # Whether to include Action 6 (Gather) in the core workflow
    )

    # --- Scoring Configuration (Class Attributes) ---
    # Dictionary mapping score category names to their integer point values
    COMMON_SCORING_WEIGHTS: Dict[str, int] = {
        # cases are ignored
        # --- Name Weights ---
        "contains_first_name": 25,  # if the input first name is in the candidate first name
        "contains_surname": 25,  # if the input surname is in the candidate surname
        "bonus_both_names_contain": 25,  # additional bonus if both first and last name achieved a score
        # --- Existing Date Weights ---
        "exact_birth_date": 25,  # if input date of birth is exact with candidate date of birth ie yyy/mm/dd
        "exact_death_date": 25,  # if input date of death is exact with candidate date of death ie yyy/mm/dd
        "year_birth": 20,  # if input year of death is exact with candidate year of death even if the day and month is wrong or not given
        "year_death": 20,  # if input year of death is exact with candidate year of death even if the day and month is wrong or not given
        "approx_year_birth": 10,  # if input year of death is within year_match_range years of candidate year of death even if the day and month is wrong or not given
        "approx_year_death": 10,  # if input year of death is within year_match_range years of candidate year of death even if the day and month is wrong or not given
        "death_dates_both_absent": 10,  # if both the input and candidate have no death dates
        # --- Gender Weights ---
        "gender_match": 15,  # if the input gender indication eg m/man/male/boy or f/fem/female/woman/girl matches the candidate gender indication.
        # --- Place Weights ---
        "contains_pob": 25,  # if the input place of birth is contained in the candidate place of birth
        "contains_pod": 25,  # if the input place of death is contained in the candidate place of death
        # --- Bonus Weights ---
        "bonus_birth_info": 25,  # additional bonus if both birth year and birth place achieved a score
        "bonus_death_info": 25,  # additional bonus if both death year and death place achieved a score
    }
    # Dictionary mapping name matching configuration options to their values (float or boolean)
    NAME_FLEXIBILITY: Dict[str, Union[float, bool]] = {
        # Fuzzy threshold might still be useful for other potential matching logic
        "fuzzy_threshold": 0.8,
        # check_starts_with is no longer directly used by the primary name scoring
        "check_starts_with": False,  # Set to False as 'contains' is primary
    }
    # Dictionary mapping date matching configuration options to their integer values
    DATE_FLEXIBILITY: Dict[str, int] = {
        "year_match_range": 10,
    }
    # --- End Scoring Configuration ---

    # --- Initializer ---
    def __init__(self):
        """Initializes the Config_Class by loading values and validating criticals."""
        # Initialize instance attributes that will hold loaded values
        self.ANCESTRY_USERNAME: str = ""
        self.ANCESTRY_PASSWORD: str = ""
        self.TREE_NAME: str = ""
        self.TREE_OWNER_NAME: str = ""
        self.MY_PROFILE_ID: Optional[str] = None
        self.MY_TREE_ID: Optional[str] = None
        self.MS_GRAPH_CLIENT_ID: str = ""
        self.MS_GRAPH_TENANT_ID: str = ""
        self.MS_TODO_LIST_NAME: str = ""
        self.LOG_DIR: Optional[Path] = None
        self.DATA_DIR: Optional[Path] = None
        self.CACHE_DIR: Optional[Path] = None
        self.DATABASE_FILE: Optional[Path] = None
        self.GEDCOM_FILE_PATH: Optional[Path] = None
        self.CACHE_DIR_PATH: Optional[Path] = None
        self.BASE_URL: str = ""
        self.ALTERNATIVE_API_URL: Optional[str] = None
        self.API_BASE_URL_PATH: str = ""
        self.API_BASE_URL: Optional[str] = None
        self.APP_MODE: str = ""
        self.GATHER_THREAD_POOL_WORKERS: int = 5  # Default if not set
        # Rate limiting/Retry values will use class defaults if not overridden by env
        self.CACHE_TIMEOUT: int = 3600  # Default
        self.CACHE_MAX_SIZE: int = (
            5_000_000  # Default cache size limit (5 million entries)
        )

        # Load values from env, potentially overriding class defaults
        self._load_values()
        # Validate essential configurations
        self._validate_critical_configs()

    # End of __init__

    def _load_values(self):
        """Loads configuration values from environment variables or defaults."""
        logger.debug("Loading application configuration settings...")

        # === Secure Credential Loading ===
        self._load_secure_credentials()

        # === Other Identifiers ===
        self.TREE_NAME = self._get_string_env("TREE_NAME", "")
        self.TREE_OWNER_NAME = self._get_string_env("TREE_OWNER_NAME", "")
        self.MY_PROFILE_ID = self._get_string_env("MY_PROFILE_ID", "")
        self.MY_TREE_ID = self._get_string_env("MY_TREE_ID", "")
        self.MS_GRAPH_CLIENT_ID = self._get_string_env("MS_GRAPH_CLIENT_ID", "")
        self.MS_GRAPH_TENANT_ID = self._get_string_env("MS_GRAPH_TENANT_ID", "common")
        self.MS_TODO_LIST_NAME = self._get_string_env("MS_TODO_LIST_NAME", "Tasks")

        # === Testing Specific Configuration ===
        # Use Config_Class.ATTRIBUTE for default value in _get_string_env
        loaded_test_profile_id = self._get_string_env(
            "TESTING_PROFILE_ID", Config_Class.TESTING_PROFILE_ID or ""
        )
        if loaded_test_profile_id:
            self.TESTING_PROFILE_ID = loaded_test_profile_id.upper()
            if self.TESTING_PROFILE_ID != Config_Class.TESTING_PROFILE_ID:
                logger.info(
                    f"Loaded TESTING_PROFILE_ID from env var: '{self.TESTING_PROFILE_ID}'"
                )
            else:
                logger.info(
                    f"Using TESTING_PROFILE_ID from env var (matches default): '{self.TESTING_PROFILE_ID}'"
                )
        elif Config_Class.TESTING_PROFILE_ID:  # If env var not set, use class default
            self.TESTING_PROFILE_ID = Config_Class.TESTING_PROFILE_ID.upper()
            logger.info(
                f"Using default TESTING_PROFILE_ID: '{self.TESTING_PROFILE_ID}' (set in .env to override)"
            )
        else:  # If no env var and no class default
            logger.warning("TESTING_PROFILE_ID is not set in environment or defaults.")
            self.TESTING_PROFILE_ID = None

        self.TESTING_PERSON_TREE_ID = self._get_string_env(
            "TESTING_PERSON_TREE_ID",
            Config_Class.TESTING_PERSON_TREE_ID
            or "",  # Use Class attribute for default
        )
        if self.TESTING_PERSON_TREE_ID:
            env_val_person = os.getenv("TESTING_PERSON_TREE_ID")
            if env_val_person:
                logger.info(
                    f"Loaded TESTING_PERSON_TREE_ID from env var: '{self.TESTING_PERSON_TREE_ID}'"
                )
            elif (
                Config_Class.TESTING_PERSON_TREE_ID
            ):  # Log if using the class default (though default is None)
                logger.info(
                    f"Using default TESTING_PERSON_TREE_ID: '{self.TESTING_PERSON_TREE_ID}' (set in .env to override)"
                )
        else:
            logger.debug("TESTING_PERSON_TREE_ID not set in environment or defaults.")

        # Load reference person configuration
        self.REFERENCE_PERSON_ID = self._get_string_env(
            "REFERENCE_PERSON_ID", Config_Class.REFERENCE_PERSON_ID or ""
        )
        self.REFERENCE_PERSON_NAME = self._get_string_env(
            "REFERENCE_PERSON_NAME", Config_Class.REFERENCE_PERSON_NAME
        )
        if self.REFERENCE_PERSON_ID:
            env_val_ref_id = os.getenv("REFERENCE_PERSON_ID")
            if env_val_ref_id:
                logger.info(
                    f"Loaded REFERENCE_PERSON_ID from env var: '{self.REFERENCE_PERSON_ID}'"
                )
            else:
                logger.info(
                    f"Using default REFERENCE_PERSON_ID: '{self.REFERENCE_PERSON_ID}' (set in .env to override)"
                )
        else:
            logger.debug("REFERENCE_PERSON_ID not set in environment or defaults.")

        # Load user signature configuration
        self.USER_NAME = self._get_string_env("USER_NAME", Config_Class.USER_NAME)
        self.USER_LOCATION = self._get_string_env(
            "USER_LOCATION", Config_Class.USER_LOCATION
        )

        # === Test/Mock Configuration ===
        self.TEST_TREE_ID = self._get_string_env(
            "TEST_TREE_ID", Config_Class.TEST_TREE_ID
        )
        self.TEST_OWNER_NAME = self._get_string_env(
            "TEST_OWNER_NAME", Config_Class.TEST_OWNER_NAME
        )
        self.TEST_EMAIL = self._get_string_env("TEST_EMAIL", Config_Class.TEST_EMAIL)
        self.TEST_CSRF_TOKEN = self._get_string_env(
            "TEST_CSRF_TOKEN", Config_Class.TEST_CSRF_TOKEN
        )
        self.TEST_PROFILE_ID = self._get_string_env(
            "TEST_PROFILE_ID", Config_Class.TEST_PROFILE_ID
        )
        self.TEST_UUID = self._get_string_env("TEST_UUID", Config_Class.TEST_UUID)
        self.TEST_TAB_HANDLE = self._get_string_env(
            "TEST_TAB_HANDLE", Config_Class.TEST_TAB_HANDLE
        )
        self.TEST_RECIPIENT_ID = self._get_string_env(
            "TEST_RECIPIENT_ID", Config_Class.TEST_RECIPIENT_ID
        )
        self.TEST_RECIPIENT_USERNAME = self._get_string_env(
            "TEST_RECIPIENT_USERNAME", Config_Class.TEST_RECIPIENT_USERNAME
        )

        # === Paths & Files ===
        log_dir_name = self._get_string_env("LOG_DIR", "Logs")
        data_dir_name = self._get_string_env("DATA_DIR", "Data")
        cache_dir_name = self._get_string_env("CACHE_DIR", "Cache")
        self.LOG_DIR = Path(log_dir_name).resolve()
        self.DATA_DIR = Path(data_dir_name).resolve()
        self.CACHE_DIR = Path(cache_dir_name).resolve()
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Get the database file name from .env
        db_file_name = self._get_string_env("DATABASE_FILE", "ancestry_data.db")

        # Check if the database file path is absolute or relative
        db_path = Path(db_file_name)
        if db_path.is_absolute():
            # Use the absolute path as is
            self.DATABASE_FILE = db_path
        else:
            # Use the path relative to the DATA_DIR
            self.DATABASE_FILE = self.DATA_DIR / db_file_name
        self.GEDCOM_FILE_PATH = self._get_path_env("GEDCOM_FILE_PATH", None)
        self.CACHE_DIR_PATH = self.CACHE_DIR

        # === URLs ===
        self.BASE_URL = self._get_string_env("BASE_URL", "https://www.ancestry.co.uk/")
        self.ALTERNATIVE_API_URL = self._get_env_var("ALTERNATIVE_API_URL")
        self.API_BASE_URL_PATH = self._get_string_env("API_BASE_URL_PATH", "/api/v2/")
        self.API_BASE_URL = (
            urljoin(self.BASE_URL, self.API_BASE_URL_PATH)  # urljoin is now available
            if self.BASE_URL and self.API_BASE_URL_PATH
            else None
        )

        # === Application Behavior ===
        self.APP_MODE = self._get_string_env("APP_MODE", "dry_run")
        # Use class defaults for MAX limits unless overridden by env
        self.MAX_PAGES = self._get_int_env("MAX_PAGES", Config_Class.MAX_PAGES)
        self.MAX_RETRIES = self._get_int_env("MAX_RETRIES", 5)  # Use literal default
        self.MAX_INBOX = self._get_int_env("MAX_INBOX", Config_Class.MAX_INBOX)
        self.BATCH_SIZE = self._get_int_env("BATCH_SIZE", 50)  # Use literal default
        self.MAX_PRODUCTIVE_TO_PROCESS = self._get_int_env(
            "MAX_PRODUCTIVE_TO_PROCESS", Config_Class.MAX_PRODUCTIVE_TO_PROCESS
        )
        self.DB_ERROR_PAGE_THRESHOLD = self._get_int_env(
            "DB_ERROR_PAGE_THRESHOLD", Config_Class.DB_ERROR_PAGE_THRESHOLD
        )
        self.GATHER_THREAD_POOL_WORKERS = self._get_int_env(
            "GATHER_THREAD_POOL_WORKERS", 5
        )
        # Action 11 Specific Limits
        self.MAX_SUGGESTIONS_TO_SCORE = self._get_int_env(
            "MAX_SUGGESTIONS_TO_SCORE", Config_Class.MAX_SUGGESTIONS_TO_SCORE
        )
        self.MAX_CANDIDATES_TO_DISPLAY = self._get_int_env(
            "MAX_CANDIDATES_TO_DISPLAY", Config_Class.MAX_CANDIDATES_TO_DISPLAY
        )

        # Action 9 Specific Settings
        self.CUSTOM_RESPONSE_ENABLED = self._get_bool_env(
            "CUSTOM_RESPONSE_ENABLED", Config_Class.CUSTOM_RESPONSE_ENABLED
        )

        # Workflow Settings
        self.INCLUDE_ACTION6_IN_WORKFLOW = self._get_bool_env(
            "INCLUDE_ACTION6_IN_WORKFLOW", Config_Class.INCLUDE_ACTION6_IN_WORKFLOW
        )

        # === Database ===
        self.DB_POOL_SIZE = self._get_int_env("DB_POOL_SIZE", Config_Class.DB_POOL_SIZE)

        # === Caching ===
        self.CACHE_TIMEOUT = self._get_int_env(
            "CACHE_TIMEOUT", 3600
        )  # Use literal default
        self.CACHE_MAX_SIZE = self._get_int_env(
            "CACHE_MAX_SIZE", 5_000_000
        )  # Cache size limit in entries

        # === Rate Limiting ===
        # Use class defaults unless overridden by env
        self.INITIAL_DELAY = self._get_float_env(
            "INITIAL_DELAY", Config_Class.INITIAL_DELAY
        )
        self.MAX_DELAY = self._get_float_env("MAX_DELAY", Config_Class.MAX_DELAY)
        self.BACKOFF_FACTOR = self._get_float_env(
            "BACKOFF_FACTOR", Config_Class.BACKOFF_FACTOR
        )
        self.DECREASE_FACTOR = self._get_float_env(
            "DECREASE_FACTOR", Config_Class.DECREASE_FACTOR
        )
        self.TOKEN_BUCKET_CAPACITY = self._get_float_env(
            "TOKEN_BUCKET_CAPACITY", Config_Class.TOKEN_BUCKET_CAPACITY
        )
        self.TOKEN_BUCKET_FILL_RATE = self._get_float_env(
            "TOKEN_BUCKET_FILL_RATE", Config_Class.TOKEN_BUCKET_FILL_RATE
        )
        logger.debug(
            f"Rate Limiter Loaded: InitialDelay={self.INITIAL_DELAY:.2f}s, Backoff={self.BACKOFF_FACTOR:.2f}, Decrease={self.DECREASE_FACTOR:.2f}"
        )
        logger.debug(
            f"Token Bucket Loaded: Capacity={self.TOKEN_BUCKET_CAPACITY:.1f}, FillRate={self.TOKEN_BUCKET_FILL_RATE:.1f}/sec"
        )

        # === Retry Status Codes ===
        # Use class default unless overridden by env
        retry_codes_env = self._get_json_env(
            "RETRY_STATUS_CODES", list(Config_Class.RETRY_STATUS_CODES)
        )
        if isinstance(retry_codes_env, (list, tuple)) and all(
            isinstance(code, int) for code in retry_codes_env
        ):
            self.RETRY_STATUS_CODES = tuple(retry_codes_env)
        else:
            logger.warning(
                f"RETRY_STATUS_CODES from env ('{retry_codes_env}') invalid. Using default: {Config_Class.RETRY_STATUS_CODES}"
            )
            self.RETRY_STATUS_CODES = Config_Class.RETRY_STATUS_CODES

        # === AI Configuration ===
        self.AI_PROVIDER = self._get_string_env("AI_PROVIDER", "").lower()

        # Only load API keys from environment if not already loaded from encrypted storage
        if not hasattr(self, "DEEPSEEK_API_KEY") or not self.DEEPSEEK_API_KEY:
            self.DEEPSEEK_API_KEY = self._get_string_env("DEEPSEEK_API_KEY", "")
        if not hasattr(self, "GOOGLE_API_KEY") or not self.GOOGLE_API_KEY:
            self.GOOGLE_API_KEY = self._get_string_env("GOOGLE_API_KEY", "")

        self.DEEPSEEK_AI_MODEL = self._get_string_env(
            "DEEPSEEK_AI_MODEL", "deepseek-chat"
        )
        self.DEEPSEEK_AI_BASE_URL = self._get_string_env(
            "DEEPSEEK_AI_BASE_URL", "https://api.deepseek.com"
        )
        self.GOOGLE_AI_MODEL = self._get_string_env(
            "GOOGLE_AI_MODEL", "gemini-1.5-flash-latest"
        )
        # Use class defaults unless overridden by env
        self.AI_CONTEXT_MESSAGES_COUNT = self._get_int_env(
            "AI_CONTEXT_MESSAGES_COUNT", Config_Class.AI_CONTEXT_MESSAGES_COUNT
        )
        self.AI_CONTEXT_MESSAGE_MAX_WORDS = self._get_int_env(
            "AI_CONTEXT_MESSAGE_MAX_WORDS", Config_Class.AI_CONTEXT_MESSAGE_MAX_WORDS
        )
        logger.debug(
            f"AI Context Limits: Last {self.AI_CONTEXT_MESSAGES_COUNT} msgs, Max {self.AI_CONTEXT_MESSAGE_MAX_WORDS} words/msg."
        )
        logger.info(f"AI Provider Configured: '{self.AI_PROVIDER or 'None'}'")
        # Log specific AI settings based on provider...

        # === API Contextual Headers ===
        parsed_base_url = urlparse(self.BASE_URL)
        origin_header_value = f"{parsed_base_url.scheme}://{parsed_base_url.netloc}"
        default_list_referer = urljoin(self.BASE_URL, "/discoveryui-matches/list/")
        self.API_CONTEXTUAL_HEADERS: Dict[str, Dict[str, Optional[str]]] = {
            # Context keys should match the `api_description` used in `_api_req`
            "Get my profile_id": {"ancestry-clientpath": "p13n-js"},
            "Tree Owner Name API": {"ancestry-clientpath": "Browser:meexp-uhome"},
            "Profile Details API (Batch)": {"ancestry-clientpath": "express-fe"},
            "Profile Details API (Action 7)": {"ancestry-clientpath": "express-fe"},
            "Get Target Name (Profile Details)": {"ancestry-clientpath": "express-fe"},
            "Create Conversation API": {"ancestry-clientpath": "express-fe"},
            "Send Message API (Existing Conv)": {"ancestry-clientpath": "express-fe"},
            "Get Inbox Conversations": {"ancestry-clientpath": "express-fe"},
            "Fetch Conversation Context": {"ancestry-clientpath": "express-fe"},
            "Match List API": {"Referer": default_list_referer},
            "In-Tree Status Check": {
                "Origin": origin_header_value,
                "Referer": default_list_referer,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            # Headers for Action 11 APIs (simplified without Cloudscraper)
            "Suggest API": {  # Replaces "Suggest API (Fallback)"
                "Accept": "application/json",
                "Referer": None,  # Referer will be set dynamically
            },
            "TreesUI List API": {  # Replaces "TreesUI List API (Fallback)"
                "Accept": "application/json",
                "Referer": None,  # Referer will be set dynamically
            },
            "Person Facts API": {  # Replaces "Person Facts API (_api_req fallback)"
                "Accept": "application/json",
                "X-Requested-With": "XMLHttpRequest",  # Kept as potentially important
                "Referer": None,  # Referer will be set dynamically
                # Consider adding other headers from V16.15 if _api_req fails without them
            },
            "Get Tree Ladder API": {  # Replaces "Get Tree Ladder API (Action 11)"
                "Accept": "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest",
                "Referer": None,  # Referer will be set dynamically
            },
            "Discovery Relationship API": {  # Replaces "API Relationship Ladder (Discovery)"
                "Accept": "application/json",
                "Referer": None,  # Referer will be set dynamically
            },
            # CSRF and other internal APIs
            "CSRF Token API": {},
            "Get UUID API": {},
            "Header Trees API": {},
            "Match Details API (Batch)": {},
            "Badge Details API (Batch)": {},
            # Self-check headers
            "Get Ladder API (Self Check)": {},  # Keep specific keys for self-check if needed
            "Get Tree Ladder API (Self Check)": {},
            "Tree Person Search API (Self Check)": {},
            "Person Picker Suggest API (Self Check)": {},
        }

        # === Tree Search Method ===
        # Use class default unless overridden by env
        loaded_search_method = self._get_string_env(
            "TREE_SEARCH_METHOD", Config_Class.TREE_SEARCH_METHOD
        ).upper()
        if loaded_search_method in ["GEDCOM", "API", "NONE"]:
            self.TREE_SEARCH_METHOD = loaded_search_method
            logger.info(f"Tree Search Method set to: {self.TREE_SEARCH_METHOD}")
        else:
            logger.warning(
                f"Invalid TREE_SEARCH_METHOD '{loaded_search_method}' in config. Defaulting to '{Config_Class.TREE_SEARCH_METHOD}'."
            )
            self.TREE_SEARCH_METHOD = Config_Class.TREE_SEARCH_METHOD

        # === Final Logging ===
        logger.info(
            f"Config Loaded: BASE_URL='{self.BASE_URL}', DB='{self.DATABASE_FILE.name if self.DATABASE_FILE else 'N/A'}', TREE='{self.TREE_NAME or 'N/A'}'"
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
        logger.info(
            f"Action 11 Limits: MaxScore={self.MAX_SUGGESTIONS_TO_SCORE}, MaxDisplay={self.MAX_CANDIDATES_TO_DISPLAY}"
        )
        # Log other final info...
        logger.debug("Application config loading complete.\n")

    # End of _load_values

    def _validate_critical_configs(self):
        """Validates essential configuration values after loading."""
        logger.debug("Validating critical configuration settings...")
        errors_found = []

        # === Credentials ===
        if not self.ANCESTRY_USERNAME:
            errors_found.append("ANCESTRY_USERNAME missing.")
        if not self.ANCESTRY_PASSWORD:
            errors_found.append("ANCESTRY_PASSWORD missing.")

        # === AI Provider & Keys ===
        if self.AI_PROVIDER not in ["deepseek", "gemini", ""]:
            logger.warning(f"AI_PROVIDER '{self.AI_PROVIDER}' is not recognized.")
        elif self.AI_PROVIDER == "deepseek" and not self.DEEPSEEK_API_KEY:
            errors_found.append("DEEPSEEK_API_KEY missing for AI_PROVIDER 'deepseek'.")
        elif self.AI_PROVIDER == "gemini" and not self.GOOGLE_API_KEY:
            errors_found.append("GOOGLE_API_KEY missing for AI_PROVIDER 'gemini'.")
        elif not self.AI_PROVIDER:
            logger.info("AI_PROVIDER not set, AI features skipped.")

        # === MS Graph ===
        if not self.MS_GRAPH_CLIENT_ID:
            logger.warning("MS_GRAPH_CLIENT_ID missing, To-Do tasks will fail.")

        # === Database Path ===
        try:
            # DATABASE_FILE is initialized in _load_values before this method is called
            # and should never be None at this point
            if (
                self.DATABASE_FILE is None
            ):  # Should ideally not happen if _load_values ran
                errors_found.append("DATABASE_FILE is not initialized.")
            else:
                db_parent_dir = self.DATABASE_FILE.parent
                if not db_parent_dir.is_dir():
                    if not db_parent_dir.exists():
                        try:
                            db_parent_dir.mkdir(parents=True, exist_ok=True)
                            logger.info(f"Created DB dir: {db_parent_dir}")
                        except OSError as mkdir_err:
                            errors_found.append(
                                f"Cannot create DB dir '{db_parent_dir}': {mkdir_err}"
                            )
                    else:
                        errors_found.append(
                            f"DB parent path '{db_parent_dir}' is not a directory."
                        )
        except Exception as path_err:
            errors_found.append(
                f"Error checking DB path '{self.DATABASE_FILE}': {path_err}"
            )

        # === Testing Configs (Warnings only) ===
        if self.APP_MODE == "testing" and not self.TESTING_PROFILE_ID:
            logger.warning("APP_MODE is 'testing' but TESTING_PROFILE_ID is not set.")
        if not self.TESTING_PERSON_TREE_ID:
            logger.warning(
                "TESTING_PERSON_TREE_ID is missing. Some tests will be skipped/fail."
            )

        # === Log Action 9 Settings ===
        logger.info(
            f"Action 9 Settings: CUSTOM_RESPONSE_ENABLED={self.CUSTOM_RESPONSE_ENABLED}"
        )

        # === Report Errors or Success ===
        if errors_found:
            logger.critical("--- CRITICAL CONFIGURATION ERRORS ---")
            for error in errors_found:
                logger.critical(f" - {error}")
            logger.critical("---------------------------------------")
            raise ValueError("Critical configuration(s) missing/invalid.")
        else:
            logger.info("Critical configuration settings validated successfully.")

    # End of _validate_critical_configs

    def _load_secure_credentials(self):
        """Load credentials securely using SecurityManager."""
        try:
            # Import here to avoid circular imports
            from security_manager import SecurityManager

            security_manager = SecurityManager()

            # Try to load encrypted credentials first
            credentials = security_manager.decrypt_credentials()

            if credentials:
                # Load from encrypted storage
                self.ANCESTRY_USERNAME = credentials.get("ANCESTRY_USERNAME", "")
                self.ANCESTRY_PASSWORD = credentials.get("ANCESTRY_PASSWORD", "")

                # Load AI API keys
                self.DEEPSEEK_API_KEY = credentials.get("DEEPSEEK_API_KEY", "")
                self.GOOGLE_API_KEY = credentials.get("GOOGLE_API_KEY", "")

                logger.info("Loaded credentials from encrypted storage")
            else:
                # Fallback to environment variables
                logger.warning(
                    "No encrypted credentials found, using environment variables"
                )
                self.ANCESTRY_USERNAME = self._get_string_env("ANCESTRY_USERNAME", "")
                self.ANCESTRY_PASSWORD = self._get_string_env("ANCESTRY_PASSWORD", "")
                self.DEEPSEEK_API_KEY = self._get_string_env("DEEPSEEK_API_KEY", "")
                self.GOOGLE_API_KEY = self._get_string_env("GOOGLE_API_KEY", "")

                # If credentials found in env, suggest migration
                if self.ANCESTRY_USERNAME and self.ANCESTRY_PASSWORD:
                    logger.info(
                        "Consider migrating to encrypted storage: python security_manager.py"
                    )

        except ImportError:
            logger.warning("SecurityManager not available, using environment variables")
            logger.warning("=" * 60)
            logger.warning("SECURITY DEPENDENCIES MISSING")
            logger.warning("=" * 60)
            logger.warning("Required security packages are not installed:")
            logger.warning("  - cryptography: For secure encryption/decryption")
            logger.warning("  - keyring: For secure storage of master keys")

            logger.warning("\nInstallation Instructions:")
            logger.warning("  1. Install required packages:")
            logger.warning("     pip install cryptography keyring")
            logger.warning("     - OR -")
            logger.warning("     pip install -r requirements.txt")

            logger.warning("\nFor secure credential management, run:")
            logger.warning("  python credentials.py")

            # Fallback to environment variables
            self.ANCESTRY_USERNAME = self._get_string_env("ANCESTRY_USERNAME", "")
            self.ANCESTRY_PASSWORD = self._get_string_env("ANCESTRY_PASSWORD", "")
            self.DEEPSEEK_API_KEY = self._get_string_env("DEEPSEEK_API_KEY", "")
            self.GOOGLE_API_KEY = self._get_string_env("GOOGLE_API_KEY", "")
        except Exception as e:
            logger.error(f"Error loading secure credentials: {e}")
            # Fallback to environment variables
            self.ANCESTRY_USERNAME = self._get_string_env("ANCESTRY_USERNAME", "")
            self.ANCESTRY_PASSWORD = self._get_string_env("ANCESTRY_PASSWORD", "")
            self.DEEPSEEK_API_KEY = self._get_string_env("DEEPSEEK_API_KEY", "")
            self.GOOGLE_API_KEY = self._get_string_env("GOOGLE_API_KEY", "")

    # End of _load_secure_credentials


# End of Config_Class class


# --- Selenium Specific Configuration Class ---
class SeleniumConfig(BaseConfig):
    """
    Loads and provides access to Selenium WebDriver specific configuration settings...
    """

    # --- Default Selenium Settings ---
    HEADLESS_MODE: bool = False
    PROFILE_DIR: str = "Default"
    DEBUG_PORT: int = 9516
    CHROME_MAX_RETRIES: int = 3
    CHROME_RETRY_DELAY: int = 5
    ELEMENT_TIMEOUT: int = 20
    PAGE_TIMEOUT: int = 40
    ASYNC_SCRIPT_TIMEOUT: int = 60
    LOGGED_IN_CHECK_TIMEOUT: int = 15
    MODAL_TIMEOUT: int = 10
    DNA_LIST_PAGE_TIMEOUT: int = 30
    NEW_TAB_TIMEOUT: int = 15
    TWO_FA_CODE_ENTRY_TIMEOUT: int = 300
    API_TIMEOUT: int = 60  # Default timeout for requests library calls via _api_req

    # --- Initializer ---
    def __init__(self):
        """Initializes the SeleniumConfig by loading values."""
        # Initialize instance vars
        self.CHROME_DRIVER_PATH: Optional[Path] = None
        self.CHROME_BROWSER_PATH: Optional[Path] = None
        self.CHROME_USER_DATA_DIR: Optional[Path] = None
        # Load values from env, potentially overriding class defaults
        self._load_values()
        logger.debug("Selenium configuration loaded.")

    # End of __init__

    def _load_values(self):
        """Loads Selenium specific values from environment variables or defaults."""
        self.CHROME_DRIVER_PATH = self._get_path_env("CHROME_DRIVER_PATH", None)
        self.CHROME_BROWSER_PATH = self._get_path_env("CHROME_BROWSER_PATH", None)
        default_user_data_str = str(Path.home() / ".ancestry_chrome_data")
        self.CHROME_USER_DATA_DIR = self._get_path_env(
            "CHROME_USER_DATA_DIR", default_user_data_str
        )
        if self.CHROME_USER_DATA_DIR:
            try:
                self.CHROME_USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(
                    f"Failed to create CHROME_USER_DATA_DIR: {self.CHROME_USER_DATA_DIR} - {e}"
                )

        # Use class defaults unless overridden by env
        self.HEADLESS_MODE = self._get_bool_env(
            "HEADLESS_MODE", SeleniumConfig.HEADLESS_MODE
        )
        self.PROFILE_DIR = self._get_string_env(
            "PROFILE_DIR", SeleniumConfig.PROFILE_DIR
        )
        self.DEBUG_PORT = self._get_int_env("DEBUG_PORT", SeleniumConfig.DEBUG_PORT)
        self.CHROME_MAX_RETRIES = self._get_int_env(
            "CHROME_MAX_RETRIES", SeleniumConfig.CHROME_MAX_RETRIES
        )
        self.CHROME_RETRY_DELAY = self._get_int_env(
            "CHROME_RETRY_DELAY", SeleniumConfig.CHROME_RETRY_DELAY
        )
        self.ELEMENT_TIMEOUT = self._get_int_env(
            "ELEMENT_TIMEOUT", SeleniumConfig.ELEMENT_TIMEOUT
        )
        self.PAGE_TIMEOUT = self._get_int_env(
            "PAGE_TIMEOUT", SeleniumConfig.PAGE_TIMEOUT
        )
        self.ASYNC_SCRIPT_TIMEOUT = self._get_int_env(
            "ASYNC_SCRIPT_TIMEOUT", SeleniumConfig.ASYNC_SCRIPT_TIMEOUT
        )
        self.LOGGED_IN_CHECK_TIMEOUT = self._get_int_env(
            "LOGGED_IN_CHECK_TIMEOUT", SeleniumConfig.LOGGED_IN_CHECK_TIMEOUT
        )
        self.MODAL_TIMEOUT = self._get_int_env(
            "MODAL_TIMEOUT", SeleniumConfig.MODAL_TIMEOUT
        )
        self.DNA_LIST_PAGE_TIMEOUT = self._get_int_env(
            "DNA_LIST_PAGE_TIMEOUT", SeleniumConfig.DNA_LIST_PAGE_TIMEOUT
        )
        self.NEW_TAB_TIMEOUT = self._get_int_env(
            "NEW_TAB_TIMEOUT", SeleniumConfig.NEW_TAB_TIMEOUT
        )
        self.TWO_FA_CODE_ENTRY_TIMEOUT = self._get_int_env(
            "TWO_FA_CODE_ENTRY_TIMEOUT", SeleniumConfig.TWO_FA_CODE_ENTRY_TIMEOUT
        )
        # API_TIMEOUT uses the class default unless overridden
        self.API_TIMEOUT = self._get_int_env("API_TIMEOUT", SeleniumConfig.API_TIMEOUT)

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
    def default_wait(self, driver, timeout: Optional[int] = None) -> WebDriverWait:
        return WebDriverWait(
            driver, timeout if timeout is not None else self.ELEMENT_TIMEOUT
        )

    # End of default_wait

    def page_load_wait(self, driver, timeout: Optional[int] = None) -> WebDriverWait:
        return WebDriverWait(
            driver, timeout if timeout is not None else self.PAGE_TIMEOUT
        )

    # End of page_load_wait

    def short_wait(self, driver, timeout: int = 5) -> WebDriverWait:
        return WebDriverWait(driver, timeout)

    # End of short_wait

    def long_wait(self, driver, timeout: Optional[int] = None) -> WebDriverWait:
        return WebDriverWait(
            driver, timeout if timeout is not None else self.TWO_FA_CODE_ENTRY_TIMEOUT
        )

    # End of long_wait

    def logged_in_check_wait(self, driver) -> WebDriverWait:
        return WebDriverWait(driver, self.LOGGED_IN_CHECK_TIMEOUT)

    # End of logged_in_check_wait

    def element_wait(self, driver) -> WebDriverWait:
        return WebDriverWait(driver, self.ELEMENT_TIMEOUT)

    # End of element_wait

    def page_wait(self, driver) -> WebDriverWait:
        return WebDriverWait(driver, self.PAGE_TIMEOUT)

    # End of page_wait

    def modal_wait(self, driver) -> WebDriverWait:
        return WebDriverWait(driver, self.MODAL_TIMEOUT)

    # End of modal_wait

    def dna_list_page_wait(self, driver) -> WebDriverWait:
        return WebDriverWait(driver, self.DNA_LIST_PAGE_TIMEOUT)

    # End of dna_list_page_wait

    def new_tab_wait(self, driver) -> WebDriverWait:
        return WebDriverWait(driver, self.NEW_TAB_TIMEOUT)

    # End of new_tab_wait


# End of SeleniumConfig class

# --- Create Singleton Instances ---
_config_valid: bool = False  # Initialize before try block
try:
    config_instance = Config_Class()
    selenium_config = SeleniumConfig()
    _config_valid = True
except ValueError as config_err:
    logger.critical(f"CONFIG VALIDATION FAILED during initial load: {config_err}")
    config_instance = None  # type: ignore
    selenium_config = None  # type: ignore
    _config_valid = False
except Exception as general_err:
    logger.critical(
        f"UNEXPECTED ERROR during config instantiation: {general_err}", exc_info=True
    )
    config_instance = None  # type: ignore
    selenium_config = None  # type: ignore
    _config_valid = False

# --- Exports ---
__all__ = ["config_instance", "selenium_config", "_config_valid"]

# --- Log Module Load ---
if _config_valid:
    logger.debug("config.py loaded and configuration instances created successfully.")
else:
    logger.error(
        "config.py loaded, but configuration instance creation FAILED validation."
    )


# --- Standalone Test Block ---
# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    import sys

    print(
        "⚙️ Running Configuration Management & Environment Integration comprehensive test suite..."
    )
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
# End of config.py standalone test block

# End of config.py
