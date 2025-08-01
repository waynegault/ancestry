#!/usr/bin/env python3

"""
Configuration Schema Definitions.

This module defines type-safe configuration schemas using dataclasses
with comprehensive validation, environment variable integration,
and schema versioning support.
"""

# === CORE INFRASTRUCTURE ===
import sys
import os

# Add parent directory to path for standard_imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
from error_handling import (
    retry_on_failure,
    circuit_breaker,
    timeout_protection,
    graceful_degradation,
    error_context,
    AncestryException,
    RetryableError,
    NetworkTimeoutError,
    AuthenticationExpiredError,
    APIRateLimitError,
    ErrorContext,
)

# === STANDARD LIBRARY IMPORTS ===
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable
import os
import re
from enum import Enum
from datetime import datetime


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


class EnvironmentType(Enum):
    """Supported environment types."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class ValidationRule:
    """Configuration validation rule."""

    field_name: str
    validator: Callable[[Any], bool]
    error_message: str
    required: bool = True


class ConfigValidator:
    """Advanced configuration validator with custom rules."""

    def __init__(self):
        self.rules: List[ValidationRule] = []
        self.environment_rules: Dict[EnvironmentType, List[ValidationRule]] = {}

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule."""
        self.rules.append(rule)

    def add_environment_rule(self, env: EnvironmentType, rule: ValidationRule) -> None:
        """Add an environment-specific validation rule."""
        if env not in self.environment_rules:
            self.environment_rules[env] = []
        self.environment_rules[env].append(rule)

    def validate(
        self, config: Any, environment: EnvironmentType = EnvironmentType.DEVELOPMENT
    ) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Apply general rules
        for rule in self.rules:
            if hasattr(config, rule.field_name):
                value = getattr(config, rule.field_name)
                if value is None and rule.required:
                    errors.append(f"Required field {rule.field_name} is missing")
                elif value is not None and not rule.validator(value):
                    errors.append(rule.error_message)
            elif rule.required:
                errors.append(f"Required field {rule.field_name} is missing")

        # Apply environment-specific rules
        if environment in self.environment_rules:
            for rule in self.environment_rules[environment]:
                if hasattr(config, rule.field_name):
                    value = getattr(config, rule.field_name)
                    if value is None and rule.required:
                        errors.append(
                            f"Environment-required field {rule.field_name} is missing for {environment.value}"
                        )
                    elif value is not None and not rule.validator(value):
                        errors.append(
                            f"{rule.error_message} (environment: {environment.value})"
                        )
                elif rule.required:
                    errors.append(
                        f"Environment-required field {rule.field_name} is missing for {environment.value}"
                    )

        return errors


def validate_path_exists(path: Union[str, Path]) -> bool:
    """Validate that a path exists."""
    return Path(path).exists() if path else False


def validate_file_extension(
    extensions: List[str],
) -> Callable[[Union[str, Path]], bool]:
    """Create validator for file extensions."""

    def validator(path: Union[str, Path]) -> bool:
        if not path:
            return True  # Allow None/empty values
        path_obj = Path(path)
        return path_obj.suffix.lower() in [ext.lower() for ext in extensions]

    return validator


def validate_port_range(port: int) -> bool:
    """Validate port is in valid range."""
    return 1024 <= port <= 65535


def validate_positive_integer(value: int) -> bool:
    """Validate value is a positive integer."""
    return isinstance(value, int) and value > 0


@dataclass
class DatabaseConfig:
    """Enhanced database configuration schema with validation."""

    database_file: Optional[Path] = None  # Database file path
    gedcom_file_path: Optional[Path] = None  # GEDCOM file path (loaded from .env)

    # Connection pool settings
    pool_size: int = 10
    max_overflow: int = 5
    pool_timeout: int = 30

    # SQLite-specific settings
    journal_mode: str = "WAL"
    foreign_keys: bool = True
    synchronous: str = "NORMAL"

    # Backup settings
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_backups: int = 7
    backup_compression: bool = True
    backup_encryption: bool = False

    # Performance settings
    cache_size_mb: int = 256
    page_size: int = 4096
    auto_vacuum: str = "INCREMENTAL"

    # Monitoring settings
    log_slow_queries: bool = True
    slow_query_threshold_ms: int = 1000
    enable_query_stats: bool = False

    # Field with default_factory must come last
    data_dir: Optional[Path] = field(default_factory=lambda: Path("Data"))

    def __post_init__(self):
        """Enhanced validation after initialization."""
        validator = self._get_validator()
        errors = validator.validate(self, self._get_environment())

        if errors:
            error_msg = f"Database configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            logger.error(error_msg)
            raise ConfigValidationError(error_msg)

        # Convert string paths to Path objects
        if isinstance(self.database_file, str):
            self.database_file = Path(self.database_file)
        if isinstance(self.gedcom_file_path, str):
            self.gedcom_file_path = Path(self.gedcom_file_path)
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)

        # Create directories if they don't exist
        self._ensure_directories()

        logger.debug(
            f"Database configuration validated successfully for {self._get_environment().value} environment"
        )

    def _get_validator(self) -> ConfigValidator:
        """Get configured validator for database settings."""
        validator = ConfigValidator()

        # General validation rules
        validator.add_rule(
            ValidationRule(
                "pool_size",
                validate_positive_integer,
                "pool_size must be a positive integer",
            )
        )

        validator.add_rule(
            ValidationRule(
                "max_overflow",
                lambda x: isinstance(x, int) and x >= 0,
                "max_overflow must be a non-negative integer",
            )
        )

        validator.add_rule(
            ValidationRule(
                "pool_timeout",
                lambda x: isinstance(x, int) and x > 0,
                "pool_timeout must be a positive integer",
            )
        )

        validator.add_rule(
            ValidationRule(
                "journal_mode",
                lambda x: x
                in ["DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"],
                "journal_mode must be one of: DELETE, TRUNCATE, PERSIST, MEMORY, WAL, OFF",
            )
        )

        validator.add_rule(
            ValidationRule(
                "synchronous",
                lambda x: x in ["OFF", "NORMAL", "FULL", "EXTRA"],
                "synchronous must be one of: OFF, NORMAL, FULL, EXTRA",
            )
        )

        validator.add_rule(
            ValidationRule(
                "backup_interval_hours",
                lambda x: isinstance(x, int) and 1 <= x <= 168,  # 1 hour to 1 week
                "backup_interval_hours must be between 1 and 168",
            )
        )

        validator.add_rule(
            ValidationRule(
                "max_backups",
                lambda x: isinstance(x, int) and 1 <= x <= 100,
                "max_backups must be between 1 and 100",
            )
        )

        validator.add_rule(
            ValidationRule(
                "cache_size_mb",
                lambda x: isinstance(x, int) and 16 <= x <= 4096,
                "cache_size_mb must be between 16 and 4096",
            )
        )

        # Production-specific rules
        validator.add_environment_rule(
            EnvironmentType.PRODUCTION,
            ValidationRule(
                "database_file",
                lambda x: x is not None,
                "database_file is required in production environment",
            ),
        )

        validator.add_environment_rule(
            EnvironmentType.PRODUCTION,
            ValidationRule(
                "backup_enabled",
                lambda x: x is True,
                "backups must be enabled in production environment",
            ),
        )

        return validator

    def _get_environment(self) -> EnvironmentType:
        """Determine current environment."""
        env_str = os.getenv("ENVIRONMENT", "development").lower()
        try:
            return EnvironmentType(env_str)
        except ValueError:
            logger.warning(
                f"Unknown environment '{env_str}', defaulting to development"
            )
            return EnvironmentType.DEVELOPMENT

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        if self.data_dir:
            self.data_dir.mkdir(parents=True, exist_ok=True)

        if self.database_file:
            self.database_file.parent.mkdir(parents=True, exist_ok=True)

    def get_connection_string(self) -> str:
        """Get SQLite connection string with optimizations."""
        if not self.database_file:
            raise ConfigValidationError("database_file not configured")

        params = []
        if self.journal_mode != "DELETE":
            params.append(f"journal_mode={self.journal_mode}")
        if self.foreign_keys:
            params.append("foreign_keys=ON")
        if self.synchronous != "NORMAL":
            params.append(f"synchronous={self.synchronous}")

        cache_size = -(self.cache_size_mb * 1024)  # Negative value means KB
        params.append(f"cache_size={cache_size}")
        params.append(f"page_size={self.page_size}")

        if params:
            return f"sqlite:///{self.database_file}?{'&'.join(params)}"
        else:
            return f"sqlite:///{self.database_file}"


@dataclass
class SeleniumConfig:
    """Selenium/WebDriver configuration schema."""

    # Chrome settings
    chrome_driver_path: Optional[Path] = None
    chrome_browser_path: Optional[Path] = None
    chrome_user_data_dir: Optional[Path] = None
    profile_dir: str = "Default"

    # Browser behavior
    headless_mode: bool = False
    debug_port: int = 9222
    window_size: str = "1920,1080"

    # Retry settings
    chrome_max_retries: int = 3
    chrome_retry_delay: int = 5  # Timeouts
    page_load_timeout: int = 30
    implicit_wait: int = 10
    explicit_wait: int = 20
    api_timeout: int = 30  # Add API timeout for requests
    two_fa_code_entry_timeout: int = 300  # 5 minutes for 2FA code entry
    two_fa_code_entry_timeout: int = 180  # Timeout for 2FA code entry

    # Features
    disable_images: bool = False
    disable_javascript: bool = False
    disable_plugins: bool = True
    disable_notifications: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.debug_port <= 0 or self.debug_port > 65535:
            raise ValueError("debug_port must be between 1 and 65535")
        if self.chrome_max_retries < 0:
            raise ValueError("chrome_max_retries must be non-negative")
        if self.chrome_retry_delay < 0:
            raise ValueError("chrome_retry_delay must be non-negative")
        if not self.window_size or "," not in self.window_size:
            raise ValueError("window_size must be in format 'width,height'")


@dataclass
class APIConfig:
    """API configuration schema."""

    # Base URLs
    base_url: str = "https://www.ancestry.com/"
    api_base_url: Optional[str] = None

    # Authentication
    username: str = ""
    password: str = ""

    # AI API Keys
    deepseek_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    deepseek_ai_model: str = "deepseek-chat"
    deepseek_ai_base_url: str = "https://api.deepseek.com"
    google_ai_model: str = "gemini-1.5-flash-latest"

    # Request settings
    request_timeout: int = 30
    max_retries: int = 3
    retry_backoff_factor: float = 0.5

    # Rate limiting
    rate_limit_enabled: bool = True
    requests_per_second: float = 2.0
    burst_limit: int = 10  # Headers
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    accept_language: str = "en-US,en;q=0.9"

    # Pagination settings
    max_pages: int = 0  # 0 means no limit    # Timing settings
    initial_delay: float = 0.5  # Initial delay between requests
    max_delay: float = 60.0  # Maximum delay for exponential backoff

    # Tree settings
    tree_name: Optional[str] = None
    tree_id: Optional[str] = None

    # Fields with default_factory must come last
    # User agents list for rotation
    user_agents: List[str] = field(
        default_factory=lambda: [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        ]
    )

    # Retry settings
    retry_status_codes: List[int] = field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )

    # API Headers
    api_contextual_headers: Dict[str, Dict[str, Optional[str]]] = field(
        default_factory=dict
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.base_url:
            raise ValueError("base_url is required")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_backoff_factor < 0:
            raise ValueError("retry_backoff_factor must be non-negative")
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")


@dataclass
class LoggingConfig:
    """Logging configuration schema."""

    # Log levels
    log_level: str = "INFO"
    console_log_level: str = "INFO"
    file_log_level: str = "DEBUG"

    # Log files
    log_file: Optional[Path] = None
    error_log_file: Optional[Path] = None
    max_log_size_mb: int = 10
    backup_count: int = 5

    # Formatting
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # Features
    enable_console_logging: bool = True
    enable_file_logging: bool = True
    enable_rotation: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            raise ValueError(f"log_level must be one of: {valid_levels}")
        if self.console_log_level not in valid_levels:
            raise ValueError(f"console_log_level must be one of: {valid_levels}")
        if self.file_log_level not in valid_levels:
            raise ValueError(f"file_log_level must be one of: {valid_levels}")
        if self.max_log_size_mb <= 0:
            raise ValueError("max_log_size_mb must be positive")
        if self.backup_count < 0:
            raise ValueError("backup_count must be non-negative")


@dataclass
class CacheConfig:
    """Cache configuration schema."""

    # Cache directories
    cache_dir: Optional[Path] = None
    temp_cache_dir: Optional[Path] = None

    # Memory cache settings
    memory_cache_size: int = 1000
    memory_cache_ttl: int = 3600  # seconds    # Disk cache settings
    disk_cache_enabled: bool = True
    disk_cache_size_mb: int = 100
    disk_cache_ttl: int = 86400  # seconds (24 hours)

    # Cleanup settings
    auto_cleanup_enabled: bool = True
    cleanup_interval_hours: int = 24
    max_cache_age_days: int = 30

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.memory_cache_size <= 0:
            raise ValueError("memory_cache_size must be positive")
        if self.memory_cache_ttl <= 0:
            raise ValueError("memory_cache_ttl must be positive")
        if self.disk_cache_size_mb <= 0:
            raise ValueError("disk_cache_size_mb must be positive")
        if self.disk_cache_ttl <= 0:
            raise ValueError("disk_cache_ttl must be positive")


@dataclass
class SecurityConfig:
    """Security configuration schema."""

    # Encryption
    encryption_enabled: bool = True
    encryption_key_file: Optional[Path] = None

    # Credential storage
    use_system_keyring: bool = True
    credential_file: Optional[Path] = None

    # Session security
    session_timeout_minutes: int = 120
    auto_logout_enabled: bool = True

    # Request security    verify_ssl: bool = True
    allow_redirects: bool = True
    max_redirects: int = 10

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.session_timeout_minutes <= 0:
            raise ValueError("session_timeout_minutes must be positive")
        if self.max_redirects < 0:
            raise ValueError("max_redirects must be non-negative")


@dataclass
class TestConfig:
    """Test configuration schema."""  # Test identifiers

    test_profile_id: str = "mock_profile_id"
    test_uuid: str = "mock_uuid"
    test_tree_id: str = "mock_tree_id"
    test_owner_name: str = "Mock Owner"
    test_tab_handle: str = "mock_tab_handle"
    test_csrf_token: str = "mock_csrf_token_12345678901234567890"


@dataclass
class ConfigSchema:
    """Main configuration schema that combines all sub-schemas."""

    # Environment
    environment: str = "development"
    debug_mode: bool = False

    # Application settings
    app_name: str = "Ancestry Automation"
    app_version: str = "1.0.0"

    # Workflow settings
    include_action6_in_workflow: bool = True

    # Action 11 (API Report) settings
    name_flexibility: float = 0.8
    date_flexibility: float = 5.0  # years
    max_suggestions_to_score: int = 50
    max_candidates_to_display: int = 20

    # Messaging settings
    message_truncation_length: int = 1000
    app_mode: str = "development"
    custom_response_enabled: bool = True

    # AI settings
    ai_provider: str = ""  # "deepseek", "gemini", or ""
    ai_context_messages_count: int = 5
    ai_context_message_max_words: int = 100

    # User settings
    user_name: str = "Tree Owner"
    user_location: str = ""

    # Batch processing settings
    batch_size: int = 100
    max_productive_to_process: int = 50
    max_inbox: int = 100

    # Tree search settings
    tree_search_method: str = "api"
    reference_person_name: str = "Reference Person"

    # Microsoft To-Do integration
    ms_todo_list_name: str = "Ancestry Tasks"

    # Scoring weights for action9
    score_weight_first_name: int = 25
    score_weight_surname: int = 25
    score_weight_gender: int = 10
    score_weight_birth_year: int = 20
    score_weight_birth_place: int = 15
    score_weight_death_year: int = 15
    score_weight_death_place: int = 10
    year_flexibility: int = 2
    exact_date_bonus: int = 25

    # Optional fields (must come after fields with default values)
    testing_profile_id: Optional[str] = None
    reference_person_id: Optional[str] = (
        None  # Fields with complex defaults (must come last)
    )
    common_scoring_weights: Dict[str, float] = field(
        default_factory=lambda: {
            # --- Name Weights ---
            "contains_first_name": 25.0,  # if the input first name is in the candidate first name
            "contains_surname": 25.0,  # if the input surname is in the candidate surname
            "bonus_both_names_contain": 25.0,  # additional bonus if both first and last name achieved a score
            # --- Existing Date Weights ---
            "exact_birth_date": 25.0,  # if input date of birth is exact with candidate date of birth
            "exact_death_date": 25.0,  # if input date of death is exact with candidate date of death
            "birth_year_match": 20.0,  # if input birth year matches candidate birth year (Action 11 key)
            "year_birth": 20.0,  # if input birth year matches candidate birth year (GEDCOM utils key)
            "death_year_match": 20.0,  # if input death year matches candidate death year (Action 11 key)
            "year_death": 20.0,  # if input death year matches candidate death year (GEDCOM utils key)
            "birth_year_close": 10.0,  # if input birth year is within range of candidate birth year
            "death_year_close": 10.0,  # if input death year is within range of candidate death year
            # --- Special Death Weights ---
            "death_dates_both_absent": 15.0,  # if both search and candidate have no death date (person is alive)
            # --- Place Weights ---
            "birth_place_match": 20.0,  # if input birth place matches candidate birth place
            "death_place_match": 20.0,  # if input death place matches candidate death place
            # --- Gender Weight ---
            "gender_match": 15.0,  # if input gender matches candidate gender
            # --- Bonus Weights ---
            "bonus_birth_date_and_place": 15.0,  # bonus if both birth date and place match
            "bonus_death_date_and_place": 15.0,  # bonus if both death date and place match
        }
    )

    # Sub-configurations (must come last due to default_factory)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    selenium: SeleniumConfig = field(default_factory=SeleniumConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    test: TestConfig = field(default_factory=TestConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_environments = ["development", "testing", "production"]
        if self.environment not in valid_environments:
            raise ValueError(f"environment must be one of: {valid_environments}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}

        # Convert each sub-config to dict
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if hasattr(value, "__dataclass_fields__"):
                # Convert dataclass to dict
                result[field_name] = {
                    sub_field: getattr(value, sub_field)
                    for sub_field in value.__dataclass_fields__
                }
            else:
                result[field_name] = value

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigSchema":
        """Create configuration from dictionary."""  # Extract sub-config data
        database_data = data.get("database", {})
        selenium_data = data.get("selenium", {})
        api_data = data.get("api", {})
        logging_data = data.get("logging", {})
        cache_data = data.get("cache", {})
        security_data = data.get("security", {})
        test_data = data.get("test", {})  # Create sub-configs
        database_config = DatabaseConfig(**database_data)
        selenium_config = SeleniumConfig(**selenium_data)
        api_config = APIConfig(**api_data)
        logging_config = LoggingConfig(**logging_data)
        cache_config = CacheConfig(**cache_data)
        security_config = SecurityConfig(**security_data)
        test_config = TestConfig(**test_data)

        # Extract main config data
        main_data = {
            k: v
            for k, v in data.items()
            if k
            not in [
                "database",
                "selenium",
                "api",
                "logging",
                "cache",
                "security",
                "test",
            ]
        }

        return cls(
            database=database_config,
            selenium=selenium_config,
            api=api_config,
            logging=logging_config,
            cache=cache_config,
            security=security_config,
            test=test_config,
            **main_data,
        )

    def validate(self) -> List[str]:
        """
        Validate the entire configuration.

        Returns:
            List of validation error messages
        """
        errors = []

        try:
            # Validate each sub-config by triggering __post_init__
            DatabaseConfig(**self.database.__dict__)
            SeleniumConfig(**self.selenium.__dict__)
            APIConfig(**self.api.__dict__)
            LoggingConfig(**self.logging.__dict__)
            CacheConfig(**self.cache.__dict__)
            SecurityConfig(**self.security.__dict__)

            # Validate main config
            self.__post_init__()

        except ValueError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"Unexpected validation error: {e}")

        return errors


def run_comprehensive_tests() -> bool:
    """
    Run comprehensive tests for the Config Schema classes.

    This function tests all major functionality of the configuration schemas
    to ensure proper validation and data handling.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    from test_framework import (
        TestSuite,
        suppress_logging,
        create_mock_data,
        assert_valid_function,
    )
    import tempfile
    from pathlib import Path

    # Initialize test suite
    suite = TestSuite("ConfigSchema", __name__)
    suite.start_suite()

    # Test 1: Database Config Creation and Validation
    def test_database_config():
        """Test DatabaseConfig creation and validation."""
        # Test default creation
        db_config = DatabaseConfig()
        assert db_config.pool_size == 10
        assert db_config.journal_mode == "WAL"
        assert db_config.backup_enabled is True

        # Test custom values
        custom_config = DatabaseConfig(
            pool_size=20, journal_mode="DELETE", backup_enabled=False
        )
        assert custom_config.pool_size == 20
        assert custom_config.journal_mode == "DELETE"  # Test validation errors
        try:
            DatabaseConfig(pool_size=-1)
            assert (
                False
            ), "Should have raised ConfigValidationError for negative pool_size"
        except ConfigValidationError:
            pass  # Expected

        try:
            DatabaseConfig(journal_mode="INVALID")
            assert (
                False
            ), "Should have raised ConfigValidationError for invalid journal_mode"
        except ConfigValidationError:
            pass  # Expected

    # Test 2: Selenium Config Creation and Validation
    def test_selenium_config():
        """Test SeleniumConfig creation and validation."""
        with suppress_logging():
            # Test default creation
            selenium_config = SeleniumConfig()
            assert selenium_config.headless_mode is False
            assert selenium_config.debug_port == 9222
            assert selenium_config.window_size == "1920,1080"

            # Test custom values
            custom_config = SeleniumConfig(
                headless_mode=True, debug_port=9223, window_size="1366,768"
            )
            assert custom_config.headless_mode is True
            assert custom_config.debug_port == 9223

            # Test validation errors
            try:
                SeleniumConfig(debug_port=-1)
                assert False, "Should have raised ValueError for invalid debug_port"
            except ValueError:
                pass  # Expected

            try:
                SeleniumConfig(window_size="invalid")
                assert False, "Should have raised ValueError for invalid window_size"
            except ValueError:
                pass  # Expected

    # Test 3: API Config Creation and Validation
    def test_api_config():
        """Test APIConfig creation and validation."""
        with suppress_logging():
            # Test default creation
            api_config = APIConfig()
            assert api_config.base_url == "https://www.ancestry.com/"
            assert api_config.request_timeout == 30
            assert api_config.rate_limit_enabled is True

            # Test custom values
            custom_config = APIConfig(
                base_url="https://example.com/",
                request_timeout=60,
                rate_limit_enabled=False,
            )
            assert custom_config.base_url == "https://example.com/"
            assert custom_config.request_timeout == 60

            # Test validation errors
            try:
                APIConfig(base_url="invalid-url")
                assert False, "Should have raised ValueError for invalid base_url"
            except ValueError:
                pass  # Expected

            try:
                APIConfig(request_timeout=-1)
                assert (
                    False
                ), "Should have raised ValueError for negative request_timeout"
            except ValueError:
                pass  # Expected

    # Test 4: Logging Config Creation and Validation
    def test_logging_config():
        """Test LoggingConfig creation and validation."""
        with suppress_logging():
            # Test default creation
            logging_config = LoggingConfig()
            assert logging_config.log_level == "INFO"
            assert logging_config.enable_console_logging is True
            assert logging_config.max_log_size_mb == 10

            # Test custom values
            custom_config = LoggingConfig(
                log_level="DEBUG", enable_console_logging=False, max_log_size_mb=20
            )
            assert custom_config.log_level == "DEBUG"
            assert custom_config.enable_console_logging is False

            # Test validation errors
            try:
                LoggingConfig(log_level="INVALID")
                assert False, "Should have raised ValueError for invalid log_level"
            except ValueError:
                pass  # Expected

            try:
                LoggingConfig(max_log_size_mb=-1)
                assert (
                    False
                ), "Should have raised ValueError for negative max_log_size_mb"
            except ValueError:
                pass  # Expected

    # Test 5: Cache Config Creation and Validation
    def test_cache_config():
        """Test CacheConfig creation and validation."""
        with suppress_logging():
            # Test default creation
            cache_config = CacheConfig()
            assert cache_config.memory_cache_size == 1000
            assert cache_config.disk_cache_enabled is True
            assert cache_config.auto_cleanup_enabled is True

            # Test custom values
            custom_config = CacheConfig(
                memory_cache_size=2000,
                disk_cache_enabled=False,
                auto_cleanup_enabled=False,
            )
            assert custom_config.memory_cache_size == 2000
            assert custom_config.disk_cache_enabled is False

            # Test validation errors
            try:
                CacheConfig(memory_cache_size=-1)
                assert (
                    False
                ), "Should have raised ValueError for negative memory_cache_size"
            except ValueError:
                pass  # Expected

    # Test 6: Security Config Creation and Validation
    def test_security_config():
        """Test SecurityConfig creation and validation."""
        with suppress_logging():
            # Test default creation
            security_config = SecurityConfig()
            assert security_config.encryption_enabled is True
            assert security_config.use_system_keyring is True
            assert security_config.session_timeout_minutes == 120  # Test custom values
            custom_config = SecurityConfig(
                encryption_enabled=False, session_timeout_minutes=60
            )
            assert custom_config.encryption_enabled is False
            assert custom_config.session_timeout_minutes == 60

            # Test validation errors
            try:
                SecurityConfig(session_timeout_minutes=-1)
                assert (
                    False
                ), "Should have raised ValueError for negative session_timeout_minutes"
            except ValueError:
                pass  # Expected

    # Test 7: Main Config Schema Creation
    def test_config_schema_creation():
        """Test ConfigSchema creation with default sub-configs."""
        with suppress_logging():
            # Test default creation
            config = ConfigSchema()
            assert config.environment == "development"
            assert config.debug_mode is False
            assert config.app_name == "Ancestry Automation"

            # Verify sub-configs are created
            assert isinstance(config.database, DatabaseConfig)
            assert isinstance(config.selenium, SeleniumConfig)
            assert isinstance(config.api, APIConfig)
            assert isinstance(config.logging, LoggingConfig)
            assert isinstance(config.cache, CacheConfig)
            assert isinstance(config.security, SecurityConfig)

            # Test custom environment
            custom_config = ConfigSchema(environment="production", debug_mode=True)
            assert custom_config.environment == "production"
            assert custom_config.debug_mode is True

    # Test 8: Config Schema to_dict Conversion
    def test_config_schema_to_dict():
        """Test ConfigSchema to_dict method."""
        with suppress_logging():
            config = ConfigSchema()
            config_dict = config.to_dict()

            # Verify structure
            assert isinstance(config_dict, dict)
            assert "database" in config_dict
            assert "selenium" in config_dict
            assert "api" in config_dict
            assert "logging" in config_dict
            assert "cache" in config_dict
            assert "security" in config_dict
            assert "environment" in config_dict

            # Verify sub-configs are dicts
            assert isinstance(config_dict["database"], dict)
            assert isinstance(config_dict["selenium"], dict)

            # Verify some values
            assert config_dict["environment"] == "development"
            assert config_dict["database"]["pool_size"] == 10

    # Test 9: Config Schema from_dict Creation
    def test_config_schema_from_dict():
        """Test ConfigSchema from_dict method."""
        with suppress_logging():
            # Create a test dictionary
            test_data = {
                "environment": "testing",
                "debug_mode": True,
                "database": {"pool_size": 15, "journal_mode": "DELETE"},
                "selenium": {"headless_mode": True, "debug_port": 9224},
                "api": {
                    "base_url": "https://test.ancestry.com/",
                    "request_timeout": 45,
                },
            }

            # Create config from dict
            config = ConfigSchema.from_dict(test_data)

            # Verify main config
            assert config.environment == "testing"
            assert config.debug_mode is True

            # Verify sub-configs
            assert config.database.pool_size == 15
            assert config.database.journal_mode == "DELETE"
            assert config.selenium.headless_mode is True
            assert config.selenium.debug_port == 9224
            assert config.api.base_url == "https://test.ancestry.com/"
            assert config.api.request_timeout == 45

    # Test 10: Config Schema Validation
    def test_config_schema_validation():
        """Test ConfigSchema validation method."""
        with suppress_logging():
            # Test valid configuration
            config = ConfigSchema()
            errors = config.validate()
            assert len(errors) == 0, f"Valid config should have no errors: {errors}"

            # Test configuration with invalid environment
            try:
                invalid_config = ConfigSchema(environment="invalid")
                assert False, "Should have raised ValueError for invalid environment"
            except ValueError:
                pass  # Expected

    # Test 11: Edge Cases and Error Handling
    def test_edge_cases():
        """Test edge cases and error scenarios."""
        with suppress_logging():
            # Test Path handling in configs
            temp_path = Path(tempfile.gettempdir()) / "test_config"

            db_config = DatabaseConfig(database_file=temp_path)
            assert db_config.database_file == temp_path

            # Test None values
            selenium_config = SeleniumConfig(chrome_driver_path=None)
            assert selenium_config.chrome_driver_path is None

            # Test empty strings in API config
            try:
                APIConfig(base_url="")
                assert False, "Should have raised ValueError for empty base_url"
            except ValueError:
                pass  # Expected

    # Test 12: Integration Testing
    def test_integration():
        """Test integration between different config components."""
        with suppress_logging():
            # Create a full configuration
            config = ConfigSchema(environment="production", debug_mode=False)

            # Modify sub-configs
            config.database.pool_size = 20
            config.selenium.headless_mode = True
            config.api.rate_limit_enabled = False

            # Convert to dict and back
            config_dict = config.to_dict()
            restored_config = ConfigSchema.from_dict(config_dict)

            # Verify restoration
            assert restored_config.environment == "production"
            assert restored_config.database.pool_size == 20
            assert restored_config.selenium.headless_mode is True
            assert restored_config.api.rate_limit_enabled is False

    # Test 13: Performance and Memory Usage
    def test_performance():
        """Test performance and memory efficiency."""
        with suppress_logging():
            import time

            start_time = time.time()

            # Create multiple configurations
            configs = []
            for i in range(100):
                config = ConfigSchema(environment="testing", debug_mode=i % 2 == 0)
                configs.append(config)

            creation_time = time.time() - start_time
            logger.info(f"Created 100 configs in {creation_time:.4f} seconds")

            # Test serialization performance
            start_time = time.time()
            for config in configs[:10]:  # Test a subset
                config_dict = config.to_dict()
                restored = ConfigSchema.from_dict(config_dict)

            serialization_time = time.time() - start_time
            logger.info(
                f"Serialized/deserialized 10 configs in {serialization_time:.4f} seconds"
            )

    # Test 14: Method Existence and Structure
    def test_function_structure():
        """Test that all expected methods and properties exist."""
        with suppress_logging():
            # Test ConfigSchema methods
            config = ConfigSchema()
            assert_valid_function(config.to_dict, "ConfigSchema.to_dict")
            assert_valid_function(config.validate, "ConfigSchema.validate")
            assert_valid_function(ConfigSchema.from_dict, "ConfigSchema.from_dict")

            # Test sub-config classes exist and have __post_init__
            for config_class in [
                DatabaseConfig,
                SeleniumConfig,
                APIConfig,
                LoggingConfig,
                CacheConfig,
                SecurityConfig,
            ]:
                instance = config_class()
                assert hasattr(instance, "__post_init__")
                assert_valid_function(
                    instance.__post_init__, f"{config_class.__name__}.__post_init__"
                )

    # Test 15: Type Definitions and Import Dependencies
    def test_import_dependencies():
        """Test that all required imports and dependencies are available."""
        with suppress_logging():
            # Test dataclass functionality
            from dataclasses import dataclass, field

            assert callable(dataclass)
            assert callable(field)

            # Test typing imports
            from typing import Optional, Dict, Any, List

            # Test pathlib
            from pathlib import Path

            test_path = Path("/test/path")
            assert isinstance(test_path, Path)

            # Test that all config classes are properly defined as dataclasses
            for config_class in [
                DatabaseConfig,
                SeleniumConfig,
                APIConfig,
                LoggingConfig,
                CacheConfig,
                SecurityConfig,
                ConfigSchema,
            ]:
                assert hasattr(config_class, "__dataclass_fields__")

    # Define all tests
    tests = [
        ("Database Config Validation", test_database_config),
        ("Selenium Config Validation", test_selenium_config),
        ("API Config Validation", test_api_config),
        ("Logging Config Validation", test_logging_config),
        ("Cache Config Validation", test_cache_config),
        ("Security Config Validation", test_security_config),
        ("Config Schema Creation", test_config_schema_creation),
        ("Config Schema to_dict", test_config_schema_to_dict),
        ("Config Schema from_dict", test_config_schema_from_dict),
        ("Config Schema Validation", test_config_schema_validation),
        ("Edge Cases", test_edge_cases),
        ("Integration Testing", test_integration),
        ("Performance Testing", test_performance),
        ("Function Structure", test_function_structure),
        ("Import Dependencies", test_import_dependencies),
    ]

    # Run each test using TestSuite
    for test_name, test_func in tests:
        suite.run_test(test_name, test_func, f"Test {test_name}")

    # Finish suite and return result
    return suite.finish_suite()


if __name__ == "__main__":
    run_comprehensive_tests()
