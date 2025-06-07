"""
Configuration Schema Definitions.

This module defines type-safe configuration schemas using dataclasses
with validation and default values.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import os

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration schema."""

    # Database file path
    database_file: Optional[Path] = None

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

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if self.max_overflow < 0:
            raise ValueError("max_overflow must be non-negative")
        if self.pool_timeout <= 0:
            raise ValueError("pool_timeout must be positive")
        if self.journal_mode not in ["DELETE", "WAL", "MEMORY"]:
            raise ValueError("journal_mode must be one of: DELETE, WAL, MEMORY")
        if self.synchronous not in ["OFF", "NORMAL", "FULL"]:
            raise ValueError("synchronous must be one of: OFF, NORMAL, FULL")


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
    chrome_retry_delay: int = 5

    # Timeouts
    page_load_timeout: int = 30
    implicit_wait: int = 10
    explicit_wait: int = 20
    api_timeout: int = 30  # Add API timeout for requests

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

    # Request settings
    request_timeout: int = 30
    max_retries: int = 3
    retry_backoff_factor: float = 0.5

    # Rate limiting
    rate_limit_enabled: bool = True
    requests_per_second: float = 2.0
    burst_limit: int = 10

    # Headers
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    accept_language: str = "en-US,en;q=0.9"

    # Tree settings
    tree_name: Optional[str] = None
    tree_id: Optional[str] = None

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
    memory_cache_ttl: int = 3600  # seconds

    # Disk cache settings
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

    # Request security
    verify_ssl: bool = True
    allow_redirects: bool = True
    max_redirects: int = 10

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.session_timeout_minutes <= 0:
            raise ValueError("session_timeout_minutes must be positive")
        if self.max_redirects < 0:
            raise ValueError("max_redirects must be non-negative")


@dataclass
class ConfigSchema:
    """Main configuration schema that combines all sub-schemas."""

    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    selenium: SeleniumConfig = field(default_factory=SeleniumConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    # Environment
    environment: str = "development"
    debug_mode: bool = False

    # Application settings
    app_name: str = "Ancestry Automation"
    app_version: str = "1.0.0"

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
        """Create configuration from dictionary."""
        # Extract sub-config data
        database_data = data.get("database", {})
        selenium_data = data.get("selenium", {})
        api_data = data.get("api", {})
        logging_data = data.get("logging", {})
        cache_data = data.get("cache", {})
        security_data = data.get("security", {})

        # Create sub-configs
        database_config = DatabaseConfig(**database_data)
        selenium_config = SeleniumConfig(**selenium_data)
        api_config = APIConfig(**api_data)
        logging_config = LoggingConfig(**logging_data)
        cache_config = CacheConfig(**cache_data)
        security_config = SecurityConfig(**security_data)

        # Extract main config data
        main_data = {
            k: v
            for k, v in data.items()
            if k not in ["database", "selenium", "api", "logging", "cache", "security"]
        }

        return cls(
            database=database_config,
            selenium=selenium_config,
            api=api_config,
            logging=logging_config,
            cache=cache_config,
            security=security_config,
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


def run_comprehensive_tests():
    """
    Run comprehensive tests for the Config Schema classes.

    This function tests all major functionality of the configuration schemas
    to ensure proper validation and data handling.
    """
    import sys
    import traceback
    from typing import Dict, Any
    import tempfile
    from pathlib import Path

    # Test framework imports with fallback
    try:
        from test_framework import (
            TestSuite,
            suppress_logging,
            create_mock_data,
            assert_valid_function,
        )

        HAS_TEST_FRAMEWORK = True
    except ImportError:
        # Fallback implementations
        HAS_TEST_FRAMEWORK = False

        class TestSuite:
            def __init__(self, name, module):
                self.name = name
                self.tests_passed = 0
                self.tests_failed = 0

            def start_suite(self):
                print(f"Starting {self.name} tests...")

            def run_test(self, name, func, description):
                try:
                    func()
                    self.tests_passed += 1
                    print(f"✓ {name}")
                except Exception as e:
                    self.tests_failed += 1
                    print(f"✗ {name}: {e}")

            def finish_suite(self):
                print(f"Tests: {self.tests_passed} passed, {self.tests_failed} failed")
                return self.tests_failed == 0

        class suppress_logging:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        def create_mock_data():
            return {}

        def assert_valid_function(func, func_name):
            assert callable(func), f"{func_name} should be callable"

    logger.info("=" * 60)
    logger.info("CONFIG SCHEMA COMPREHENSIVE TESTS")
    logger.info("=" * 60)

    test_results = {"passed": 0, "failed": 0, "errors": []}

    def run_test(test_name: str, test_func) -> bool:
        """Helper to run individual tests with error handling."""
        try:
            logger.info(f"\n--- Running: {test_name} ---")
            test_func()
            test_results["passed"] += 1
            logger.info(f"✓ PASSED: {test_name}")
            return True
        except Exception as e:
            test_results["failed"] += 1
            error_msg = f"✗ FAILED: {test_name} - {str(e)}"
            test_results["errors"].append(error_msg)
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False

    # Test 1: Database Config Creation and Validation
    def test_database_config():
        """Test DatabaseConfig creation and validation."""
        with suppress_logging():
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
            assert custom_config.journal_mode == "DELETE"

            # Test validation errors
            try:
                DatabaseConfig(pool_size=-1)
                assert False, "Should have raised ValueError for negative pool_size"
            except ValueError:
                pass  # Expected

            try:
                DatabaseConfig(journal_mode="INVALID")
                assert False, "Should have raised ValueError for invalid journal_mode"
            except ValueError:
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
            assert security_config.session_timeout_minutes == 120

            # Test custom values
            custom_config = SecurityConfig(
                encryption_enabled=False, session_timeout_minutes=60, verify_ssl=False
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

    # Run each test
    for test_name, test_func in tests:
        run_test(test_name, test_func)

    # Print summary
    total_tests = len(tests)
    logger.info("\n" + "=" * 60)
    logger.info("CONFIG SCHEMA TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {test_results['passed']}")
    logger.info(f"Failed: {test_results['failed']}")
    if test_results["errors"]:
        logger.info("\nErrors:")
        for error in test_results["errors"]:
            logger.error(f"  {error}")

    success = test_results["failed"] == 0
    if success:
        logger.info("🎉 ALL CONFIG SCHEMA TESTS PASSED!")
    else:
        logger.warning("⚠️ Some Config Schema tests failed")
    return success


if __name__ == "__main__":
    run_comprehensive_tests()
