#!/usr/bin/env python3

"""
Configuration Schema Definitions.

This module defines type-safe configuration schemas using dataclasses
with comprehensive validation, environment variable integration,
and schema versioning support.
"""

# === CORE INFRASTRUCTURE ===
import os
import sys

# Add parent directory to path for standard_imports
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===

# === STANDARD LIBRARY IMPORTS ===
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union


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

    def __init__(self) -> None:
        self.rules: list[ValidationRule] = []
        self.environment_rules: dict[EnvironmentType, list[ValidationRule]] = {}

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule."""
        self.rules.append(rule)

    def add_environment_rule(self, env: EnvironmentType, rule: ValidationRule) -> None:
        """Add an environment-specific validation rule."""
        if env not in self.environment_rules:
            self.environment_rules[env] = []
        self.environment_rules[env].append(rule)

    def _validate_single_rule(self, config: Any, rule: ValidationRule) -> Optional[str]:
        """Validate a single rule against config. Returns error message or None."""
        if hasattr(config, rule.field_name):
            value = getattr(config, rule.field_name)
            if value is None and rule.required:
                return f"Required field {rule.field_name} is missing"
            if value is not None and not rule.validator(value):
                return rule.error_message
        elif rule.required:
            return f"Required field {rule.field_name} is missing"
        return None

    def _validate_environment_rule(
        self, config: Any, rule: ValidationRule, environment: EnvironmentType
    ) -> Optional[str]:
        """Validate an environment-specific rule. Returns error message or None."""
        if hasattr(config, rule.field_name):
            value = getattr(config, rule.field_name)
            if value is None and rule.required:
                return f"Environment-required field {rule.field_name} is missing for {environment.value}"
            if value is not None and not rule.validator(value):
                return f"{rule.error_message} (environment: {environment.value})"
        elif rule.required:
            return f"Environment-required field {rule.field_name} is missing for {environment.value}"
        return None

    def _apply_general_rules(self, config: Any) -> list[str]:
        """Apply general validation rules and return errors."""
        errors = []
        for rule in self.rules:
            error = self._validate_single_rule(config, rule)
            if error:
                errors.append(error)
        return errors

    def _apply_environment_rules(self, config: Any, environment: EnvironmentType) -> list[str]:
        """Apply environment-specific validation rules and return errors."""
        errors = []
        if environment in self.environment_rules:
            for rule in self.environment_rules[environment]:
                error = self._validate_environment_rule(config, rule, environment)
                if error:
                    errors.append(error)
        return errors

    def validate(
        self, config: Any, environment: EnvironmentType = EnvironmentType.DEVELOPMENT
    ) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Apply general rules
        errors.extend(self._apply_general_rules(config))

        # Apply environment-specific rules
        errors.extend(self._apply_environment_rules(config, environment))

        return errors


def validate_path_exists(path: Union[str, Path]) -> bool:
    """Validate that a path exists."""
    return Path(path).exists() if path else False


# Import centralized file extension validation utility


# Import centralized validation utilities
from test_utilities import validate_positive_integer


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

    def __post_init__(self) -> None:
        """Enhanced validation after initialization."""
        validator = self._get_validator()
        errors = validator.validate(self, self._get_environment())

        if errors:
            error_msg = "Database configuration validation failed:\n" + "\n".join(
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
    page_load_timeout: int = 45  # Increased from 30s to 45s for slower international domains (ancestry.co.uk)
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

    def __post_init__(self) -> None:
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
    request_timeout: int = 60  # Increased from 30 to 60 seconds for slower API responses during rate limiting
    max_retries: int = 5  # Increased from 3 to 5 for better resilience to transient rate limits
    retry_backoff_factor: float = 6.0  # Increased from 4.0 to 6.0 for much longer exponential backoff on 429 errors

    # Rate limiting (optimized for performance with circuit breaker protection)
    rate_limit_enabled: bool = True
    requests_per_second: float = 5.0  # OPTIMIZATION: 0.4 → 5.0 (12x faster) - safe with circuit breaker; Ancestry API typically allows 10-20 RPS
    burst_limit: int = 4  # Allows better burst efficiency while maintaining stability
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    accept_language: str = "en-US,en;q=0.9"

    # Concurrency - REMOVED: Parallel processing eliminated for API safety
    # Sequential processing only to prevent 429 rate limiting errors
    # Previous parallel processing with ThreadPoolExecutor caused burst requests
    # that triggered Ancestry API rate limits (72-second penalties per 429 error)
    max_concurrency: int = 1  # Sequential processing only
    thread_pool_workers: int = 2  # Number of worker threads for thread pool (loaded from THREAD_POOL_WORKERS env var)

    # Pagination settings
    max_pages: int = 0  # 0 means no limit

    # Data freshness settings
    person_refresh_days: int = 7  # Skip fetching person details if updated within N days (0=disabled, 7=default)

    # Timing settings - Adaptive rate limiting parameters
    initial_delay: float = 1.0  # Starting delay between requests (seconds) - optimized for 2 workers
    max_delay: float = 15.0  # Maximum delay on rate limiting (seconds) - reduced for faster recovery
    backoff_factor: float = 1.5  # Multiplier for increasing delay on errors
    decrease_factor: float = 0.95  # Multiplier for decreasing delay on success
    token_bucket_capacity: float = 10.0  # Token bucket capacity for burst handling
    token_bucket_fill_rate: float = 2.0  # Tokens added per second

    # Tree settings
    tree_name: Optional[str] = None
    tree_id: Optional[str] = None
    my_user_id: Optional[str] = None  # User profile ID for API calls

    # Fields with default_factory must come last
    # User agents list for rotation
    user_agents: list[str] = field(
        default_factory=lambda: [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        ]
    )

    # Retry settings
    retry_status_codes: list[int] = field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )

    # API Headers
    api_contextual_headers: dict[str, dict[str, Optional[str]]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
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

    # Timing/interval settings
    proactive_refresh_interval_seconds: int = 1320  # 22 minutes between proactive refresh checks
    # Timeout for Action 6 coord (seconds) — extend to avoid mid-run timeout/auto-retry
    action6_coord_timeout_seconds: int = 14400  # 4 hours

    # Action 11 (API Report) settings
    name_flexibility: float = 0.8
    date_flexibility: float = 5.0  # years
    max_suggestions_to_score: int = 50
    max_candidates_to_display: int = 20

    # Messaging settings
    message_truncation_length: int = 1000
    app_mode: str = "development"
    custom_response_enabled: bool = True
    enable_task_dedup: bool = False  # Guard flag for Phase 4.3 de-dup rollout (testing mode only initially)
    enable_task_enrichment: bool = False  # Guard flag for Phase 4.4 enriched task generation rollout
    enable_prompt_experiments: bool = False  # Guard flag for Phase 8.2 prompt A/B experimentation rollout

    # AI settings
    ai_provider: str = ""  # "deepseek", "gemini", or ""
    ai_context_messages_count: int = 5
    ai_context_message_max_words: int = 100
    ai_context_window_messages: int = 6  # Sliding window of recent msgs used to classify last USER message

    # Proactive refresh settings
    proactive_refresh_cooldown_seconds: int = 300  # Minimum seconds between proactive session refreshes to avoid per-page loops

    # User settings
    user_name: str = "Tree Owner"
    user_location: str = ""

    # Batch processing settings
    batch_size: int = 10  # Updated to match .env BATCH_SIZE=10
    matches_per_page: int = 20  # Number of matches displayed per page by Ancestry
    max_productive_to_process: int = 50
    max_inbox: int = 100
    person_refresh_days: int = 14  # Skip re-fetching person details if updated within this many days (0 = always fetch)
    conversation_refresh_hours: int = 24  # Skip re-processing conversations if processed within this many hours (0 = always process)
    parallel_workers: int = 1  # Number of parallel workers for Action 6 match detail fetching (1=sequential, 2-3=parallel)

    # Engagement-based messaging timing (Phase 4.1)
    engagement_high_threshold: int = 70  # High engagement score threshold (0-100)
    engagement_medium_threshold: int = 40  # Medium engagement score threshold (0-100)
    engagement_low_threshold: int = 20  # Low engagement score threshold (0-100)
    login_active_threshold: int = 7  # Active login threshold (days since last_logged_in)
    login_moderate_threshold: int = 30  # Moderate login threshold (days since last_logged_in)
    followup_high_engagement_days: int = 7  # Follow-up interval for high engagement
    followup_medium_engagement_days: int = 14  # Follow-up interval for medium engagement
    followup_low_engagement_days: int = 21  # Follow-up interval for low engagement
    followup_no_engagement_days: int = 30  # Follow-up interval for no engagement

    # Status change detection (Phase 4.2)
    status_change_recent_days: int = 7  # Days threshold for "recent" FamilyTree creation

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
    common_scoring_weights: dict[str, float] = field(
        default_factory=lambda: {
            # --- Name Weights ---
            "contains_first_name": 25.0,  # if the input first name is in the candidate first name
            "contains_surname": 25.0,  # if the input surname is in the candidate surname
            "bonus_both_names_contain": 25.0,  # additional bonus if both first and last name achieved a score
            # --- Date Weights ---
            "exact_birth_date": 25.0,  # if input date of birth is exact with candidate date of birth
            "exact_death_date": 25.0,  # if input date of death is exact with candidate date of death
            "birth_year_match": 20.0,  # if input birth year matches candidate birth year (Action 11 key)
            "year_birth": 20.0,  # if input birth year matches candidate birth year (GEDCOM utils key)
            "death_year_match": 20.0,  # if input death year matches candidate death year (Action 11 key)
            "year_death": 20.0,  # if input death year matches candidate death year (GEDCOM utils key)
            "birth_year_close": 10.0,  # if input birth year is within range of candidate birth year
            "death_year_close": 10.0,  # if input death year is within range of candidate death year
            # --- Special Death Weights ---
            "death_dates_both_absent": 25.0,  # if both search and candidate have no death date (person is alive) - UPDATED
            # --- Place Weights ---
            "birth_place_match": 25.0,  # if input birth place matches candidate birth place - UPDATED
            "death_place_match": 25.0,  # if input death place matches candidate death place - UPDATED
            # --- Gender Weight ---
            "gender_match": 15.0,  # if input gender matches candidate gender
            # --- Bonus Weights ---
            "bonus_birth_date_and_place": 25.0,  # bonus if both birth date and place match - UPDATED
            "bonus_death_date_and_place": 25.0,  # bonus if both death date and place match - UPDATED
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

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        valid_environments = ["development", "testing", "production"]
        if self.environment not in valid_environments:
            raise ValueError(f"environment must be one of: {valid_environments}")

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> "ConfigSchema":
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

    def validate(self) -> list[str]:
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


# === Module-level test functions for config_schema_module_tests ===

def _test_database_config() -> None:
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
        raise AssertionError("Should have raised ConfigValidationError for negative pool_size")
    except ConfigValidationError:
        pass  # Expected

    try:
        DatabaseConfig(journal_mode="INVALID")
        raise AssertionError("Should have raised ConfigValidationError for invalid journal_mode")
    except ConfigValidationError:
        pass  # Expected


def _test_selenium_config() -> None:
    """Test SeleniumConfig creation and validation."""
    from test_framework import suppress_logging

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
            raise AssertionError("Should have raised ValueError for invalid debug_port")
        except ValueError:
            pass  # Expected

        try:
            SeleniumConfig(window_size="invalid")
            raise AssertionError("Should have raised ValueError for invalid window_size")
        except ValueError:
            pass  # Expected


def _test_api_config() -> None:
    """Test APIConfig creation and validation."""
    from test_framework import suppress_logging

    with suppress_logging():
        # Test default creation
        api_config = APIConfig()
        assert api_config.base_url == "https://www.ancestry.com/"
        assert api_config.request_timeout == 60  # Updated to match our rate limiting fixes
        assert api_config.rate_limit_enabled is True

        # Test custom values
        custom_config = APIConfig(
            base_url="https://example.com/",
            request_timeout=90,  # Updated to use higher value for testing
            rate_limit_enabled=False,
        )
        assert custom_config.base_url == "https://example.com/"
        assert custom_config.request_timeout == 90  # Updated to match

        # Test validation errors
        try:
            APIConfig(base_url="invalid-url")
            raise AssertionError("Should have raised ValueError for invalid base_url")
        except ValueError:
            pass  # Expected

        try:
            APIConfig(request_timeout=-1)
            raise AssertionError("Should have raised ValueError for negative request_timeout")
        except ValueError:
            pass  # Expected


def _test_logging_config() -> None:
    """Test LoggingConfig creation and validation."""
    from test_framework import suppress_logging

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
            raise AssertionError("Should have raised ValueError for invalid log_level")
        except ValueError:
            pass  # Expected

        try:
            LoggingConfig(max_log_size_mb=-1)
            raise AssertionError("Should have raised ValueError for negative max_log_size_mb")
        except ValueError:
            pass  # Expected


def _test_cache_config() -> None:
    """Test CacheConfig creation and validation."""
    from test_framework import suppress_logging

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
            raise AssertionError("Should have raised ValueError for negative memory_cache_size")
        except ValueError:
            pass  # Expected


def _test_security_config() -> None:
    """Test SecurityConfig creation and validation."""
    from test_framework import suppress_logging

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
            raise AssertionError("Should have raised ValueError for negative session_timeout_minutes")
        except ValueError:
            pass  # Expected


def _test_config_schema_creation() -> None:
    """Test ConfigSchema creation with default sub-configs."""
    from test_framework import suppress_logging

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


def _test_config_schema_to_dict() -> None:
    """Test ConfigSchema to_dict method."""
    from test_framework import suppress_logging

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


def _test_config_schema_from_dict() -> None:
    """Test ConfigSchema from_dict method."""
    from test_framework import suppress_logging

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


def _test_config_schema_validation() -> None:
    """Test ConfigSchema validation method."""
    from test_framework import suppress_logging

    with suppress_logging():
        # Test valid configuration
        config = ConfigSchema()
        errors = config.validate()
        assert len(errors) == 0, f"Valid config should have no errors: {errors}"

        # Test configuration with invalid environment
        try:
            ConfigSchema(environment="invalid")
            raise AssertionError("Should have raised ValueError for invalid environment")
        except ValueError:
            pass  # Expected


def _test_edge_cases() -> None:
    """Test edge cases and error scenarios."""
    import tempfile
    from pathlib import Path

    from test_framework import suppress_logging

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
            raise AssertionError("Should have raised ValueError for empty base_url")
        except ValueError:
            pass  # Expected


def _test_integration() -> None:
    """Test integration between different config components."""
    from test_framework import suppress_logging

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


def _test_performance() -> None:
    """Test performance and memory efficiency."""
    import time

    from test_framework import suppress_logging

    with suppress_logging():
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
            ConfigSchema.from_dict(config_dict)

        serialization_time = time.time() - start_time
        logger.info(
            f"Serialized/deserialized 10 configs in {serialization_time:.4f} seconds"
        )


def _test_function_structure() -> None:
    """Test that all expected methods and properties exist."""
    from test_framework import assert_valid_function, suppress_logging

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


def _test_import_dependencies() -> None:
    """Test that all required imports and dependencies are available."""
    from test_framework import suppress_logging

    with suppress_logging():
        # Test dataclass functionality
        from dataclasses import dataclass, field

        assert callable(dataclass)
        assert callable(field)

        # Test typing imports

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


def _test_rate_limiting_configuration() -> None:
    """Test that rate limiting configuration values are conservative for API stability."""
    print("🚦 Testing Rate Limiting Configuration...")

    api_config = APIConfig()

    # Define validation rules as data structure to reduce complexity
    # Updated for Phase 2 optimization (October 2025)
    # RPS increased to 5.0 with circuit breaker protection
    validation_rules = [
        ("requests_per_second", lambda v: v > 10.0, "too high"),  # 5.0 RPS is safe with circuit breaker; Ancestry API allows 10-20 RPS
        ("max_concurrency", lambda v: v > 1, "too high"),  # Sequential processing only
        ("burst_limit", lambda v: v > 4, "too high"),
        ("max_retries", lambda v: v < 5, "too low"),
        ("retry_backoff_factor", lambda v: v < 6.0, "too low"),
        ("request_timeout", lambda v: v < 60, "too low"),
        ("max_delay", lambda v: v < 5.0, "too low"),  # Optimized: 15s is excellent, warn only if below 5s
    ]

    # Validate conservative settings for API rate limiting compliance
    issues = []
    for field_name, check_func, issue_type in validation_rules:
        field_value = getattr(api_config, field_name)
        if check_func(field_value):
            issues.append(f"{field_name} {issue_type}: {field_value}")

    if issues:
        print("   ❌ Configuration issues found:")
        for issue in issues:
            print(f"      - {issue}")
        raise AssertionError(f"Rate limiting configuration issues: {issues}")
    print("   ✅ All rate limiting settings are properly conservative")


def _test_max_pages_configuration() -> None:
    """Test MAX_PAGES configuration loading and validation."""
    print("📄 Testing MAX_PAGES Configuration...")

    api_config = APIConfig()
    max_pages = api_config.max_pages
    print(f"   MAX_PAGES default value: {max_pages}")

    # Validate that max_pages is properly configured
    assert isinstance(max_pages, int), f"MAX_PAGES should be integer, got {type(max_pages)}"
    assert max_pages >= 0, f"MAX_PAGES should be non-negative, got {max_pages}"

    if max_pages == 0:
        print("   ✅ MAX_PAGES=0 correctly configured for unlimited processing")
    else:
        print(f"   ⚠️  MAX_PAGES={max_pages} limits processing (not unlimited)")


def config_schema_module_tests() -> bool:
    """
    Run comprehensive tests for the Config Schema classes.

    This function tests all major functionality of the configuration schemas
    to ensure proper validation and data handling.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    from test_framework import TestSuite

    # Initialize test suite
    suite = TestSuite("ConfigSchema", __name__)
    suite.start_suite()

    # Define all tests
    tests = [
        ("Database Config Validation", _test_database_config),
        ("Selenium Config Validation", _test_selenium_config),
        ("API Config Validation", _test_api_config),
        ("Logging Config Validation", _test_logging_config),
        ("Cache Config Validation", _test_cache_config),
        ("Security Config Validation", _test_security_config),
        ("Config Schema Creation", _test_config_schema_creation),
        ("Config Schema to_dict", _test_config_schema_to_dict),
        ("Config Schema from_dict", _test_config_schema_from_dict),
        ("Config Schema Validation", _test_config_schema_validation),
        ("Rate Limiting Configuration", _test_rate_limiting_configuration),
        ("MAX_PAGES Configuration", _test_max_pages_configuration),
        ("Edge Cases", _test_edge_cases),
        ("Integration Testing", _test_integration),
        ("Performance Testing", _test_performance),
        ("Function Structure", _test_function_structure),
        ("Import Dependencies", _test_import_dependencies),
    ]

    # Run each test using TestSuite
    for test_name, test_func in tests:
        suite.run_test(test_name, test_func, f"Test {test_name}")

    # Finish suite and return result
    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(config_schema_module_tests)


if __name__ == "__main__":
    run_comprehensive_tests()
