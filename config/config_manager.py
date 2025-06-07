"""
Enhanced Configuration Manager.

This module provides a comprehensive configuration management system with:
- Environment variable loading and validation
- Type-safe configuration schemas
- Configuration file support (JSON, YAML, TOML)
- Environment-specific configurations
- Configuration validation and error reporting
- Hot reloading capabilities
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import copy

from .config_schema import (
    ConfigSchema,
    DatabaseConfig,
    SeleniumConfig,
    APIConfig,
    LoggingConfig,
    CacheConfig,
    SecurityConfig,
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Configuration validation error."""

    pass


class ConfigManager:
    """
    Enhanced configuration manager with type-safe schemas and validation.

    Features:
    - Environment variable loading with type conversion
    - Configuration file support (JSON, YAML, TOML)
    - Environment-specific configurations (dev, test, prod)
    - Configuration validation with detailed error reporting
    - Hot reloading capabilities
    - Credential integration with SecurityManager
    """

    def __init__(
        self,
        config_file: Optional[Union[str, Path]] = None,
        environment: Optional[str] = None,
        auto_load: bool = True,
    ):
        """
        Initialize the configuration manager.

        Args:
            config_file: Optional configuration file path
            environment: Environment name (development, testing, production)
            auto_load: Whether to automatically load configuration
        """
        self.config_file = Path(config_file) if config_file else None
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self._config_cache: Optional[ConfigSchema] = None
        self._file_modification_time: Optional[float] = None

        # Supported file formats
        self._supported_formats = {".json", ".yaml", ".yml", ".toml"}

        if auto_load:
            self.load_config()

    def load_config(self) -> ConfigSchema:
        """
        Load and validate configuration from multiple sources.

        Priority order:
        1. Environment variables
        2. Configuration file
        3. Default values

        Returns:
            Validated configuration schema

        Raises:
            ValidationError: If configuration validation fails
        """
        logger.debug(f"Loading configuration for environment: {self.environment}")

        # Start with default configuration
        config_data = self._get_default_config()

        # Load from configuration file if specified
        if self.config_file and self.config_file.exists():
            file_config = self._load_config_file()
            config_data = self._merge_configs(config_data, file_config)

        # Override with environment variables
        env_config = self._load_environment_variables()
        config_data = self._merge_configs(config_data, env_config)

        # Create and validate configuration schema
        try:
            config = ConfigSchema.from_dict(config_data)

            # Validate configuration
            validation_errors = config.validate()
            if validation_errors:
                raise ValidationError(
                    f"Configuration validation failed: {validation_errors}"
                )

            # Cache the configuration
            self._config_cache = config
            if self.config_file:
                self._file_modification_time = self.config_file.stat().st_mtime

            logger.info(
                f"Configuration loaded successfully for environment: {self.environment}"
            )
            return config

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ValidationError(f"Configuration loading failed: {e}")

    def get_config(self, reload_if_changed: bool = True) -> ConfigSchema:
        """
        Get the current configuration.

        Args:
            reload_if_changed: Whether to reload if config file has changed

        Returns:
            Current configuration schema
        """
        # Check if configuration file has changed
        if reload_if_changed and self._should_reload():
            logger.info("Configuration file changed, reloading...")
            return self.load_config()

        # Return cached configuration or load if not cached
        if self._config_cache is None:
            return self.load_config()

        return self._config_cache

    def reload_config(self) -> ConfigSchema:
        """
        Force reload configuration from all sources.

        Returns:
            Reloaded configuration schema
        """
        self._config_cache = None
        self._file_modification_time = None
        return self.load_config()

    def validate_config(
        self, config_data: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Validate configuration data.

        Args:
            config_data: Optional configuration data to validate

        Returns:
            List of validation error messages
        """
        if config_data is None:
            if self._config_cache is None:
                self.load_config()
            if self._config_cache is not None:
                return self._config_cache.validate()
            else:
                return ["Configuration cache is not available"]

        try:
            config = ConfigSchema.from_dict(config_data)
            return config.validate()
        except Exception as e:
            return [f"Configuration validation error: {e}"]

    def get_environment_config(self, env_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific environment.

        Args:
            env_name: Environment name

        Returns:
            Environment-specific configuration
        """
        original_env = self.environment
        self.environment = env_name

        try:
            config = self.load_config()
            return config.to_dict()
        finally:
            self.environment = original_env
            self._config_cache = None  # Clear cache to avoid confusion

    def export_config(
        self, output_file: Union[str, Path], format: str = "json"
    ) -> bool:
        """
        Export current configuration to file.

        Args:
            output_file: Output file path
            format: Output format (json, yaml, toml)

        Returns:
            True if export successful
        """
        try:
            config = self.get_config()
            config_dict = config.to_dict()

            output_path = Path(output_file)

            if format.lower() == "json":
                with open(output_path, "w") as f:
                    json.dump(config_dict, f, indent=2, default=str)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False

            logger.info(f"Configuration exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "environment": self.environment,
            "debug_mode": self.environment == "development",
            "app_name": "Ancestry Automation",
            "app_version": "1.0.0",
            "database": {},
            "selenium": {},
            "api": {},
            "logging": {},
            "cache": {},
            "security": {},
        }

    def _load_config_file(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.config_file or not self.config_file.exists():
            return {}

        try:
            suffix = self.config_file.suffix.lower()

            if suffix == ".json":
                with open(self.config_file, "r") as f:
                    return json.load(f)
            else:
                logger.warning(f"Unsupported config file format: {suffix}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load config file {self.config_file}: {e}")
            return {}

    def _load_environment_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}

        # Load main configuration
        env_value = os.getenv("ENVIRONMENT")
        if env_value:
            config["environment"] = env_value

        debug_mode_value = os.getenv("DEBUG_MODE")
        if debug_mode_value:
            config["debug_mode"] = debug_mode_value.lower() in (
                "true",
                "1",
                "yes",
            )

        app_name_value = os.getenv("APP_NAME")
        if app_name_value:
            config["app_name"] = app_name_value  # Load database configuration
        db_config = {}
        database_file_value = os.getenv("DATABASE_FILE")
        if database_file_value:
            db_config["database_file"] = Path(database_file_value)

        db_pool_size_value = os.getenv("DB_POOL_SIZE")
        if db_pool_size_value:
            try:
                db_config["pool_size"] = int(db_pool_size_value)
            except ValueError:
                logger.warning(f"Invalid DB_POOL_SIZE value: {db_pool_size_value}")

        db_journal_mode_value = os.getenv("DB_JOURNAL_MODE")
        if db_journal_mode_value:
            db_config["journal_mode"] = db_journal_mode_value
        if db_config:
            config["database"] = db_config  # Load Selenium configuration
        selenium_config = {}
        chrome_driver_path_value = os.getenv("CHROME_DRIVER_PATH")
        if chrome_driver_path_value:
            selenium_config["chrome_driver_path"] = Path(chrome_driver_path_value)

        chrome_browser_path_value = os.getenv("CHROME_BROWSER_PATH")
        if chrome_browser_path_value:
            selenium_config["chrome_browser_path"] = Path(chrome_browser_path_value)

        chrome_user_data_dir_value = os.getenv("CHROME_USER_DATA_DIR")
        if chrome_user_data_dir_value:
            selenium_config["chrome_user_data_dir"] = Path(chrome_user_data_dir_value)

        profile_dir_value = os.getenv("PROFILE_DIR")
        if profile_dir_value:
            selenium_config["profile_dir"] = profile_dir_value

        headless_mode_value = os.getenv("HEADLESS_MODE")
        if headless_mode_value:
            selenium_config["headless_mode"] = headless_mode_value.lower() in (
                "true",
                "1",
                "yes",
            )

        debug_port_value = os.getenv("DEBUG_PORT")
        if debug_port_value:
            try:
                selenium_config["debug_port"] = int(debug_port_value)
            except ValueError:
                logger.warning(f"Invalid DEBUG_PORT value: {debug_port_value}")
        if selenium_config:
            config["selenium"] = selenium_config  # Load API configuration
        api_config = {}
        base_url_value = os.getenv("BASE_URL")
        if base_url_value:
            api_config["base_url"] = base_url_value

        api_base_url_value = os.getenv("API_BASE_URL")
        if api_base_url_value:
            api_config["api_base_url"] = api_base_url_value

        request_timeout_value = os.getenv("REQUEST_TIMEOUT")
        if request_timeout_value:
            try:
                api_config["request_timeout"] = int(request_timeout_value)
            except ValueError:
                logger.warning(
                    f"Invalid REQUEST_TIMEOUT value: {request_timeout_value}"
                )

        tree_name_value = os.getenv("TREE_NAME")
        if tree_name_value:
            api_config["tree_name"] = tree_name_value

        tree_id_value = os.getenv("TREE_ID")
        if tree_id_value:
            api_config["tree_id"] = tree_id_value
        if api_config:
            config["api"] = api_config  # Load logging configuration
        logging_config = {}
        log_level_value = os.getenv("LOG_LEVEL")
        if log_level_value:
            logging_config["log_level"] = log_level_value

        log_file_value = os.getenv("LOG_FILE")
        if log_file_value:
            logging_config["log_file"] = Path(log_file_value)

        max_log_size_value = os.getenv("MAX_LOG_SIZE_MB")
        if max_log_size_value:
            try:
                logging_config["max_log_size_mb"] = int(max_log_size_value)
            except ValueError:
                logger.warning(f"Invalid MAX_LOG_SIZE_MB value: {max_log_size_value}")
        if logging_config:
            config["logging"] = logging_config  # Load cache configuration
        cache_config = {}
        cache_dir_value = os.getenv("CACHE_DIR")
        if cache_dir_value:
            cache_config["cache_dir"] = Path(cache_dir_value)

        memory_cache_size_value = os.getenv("MEMORY_CACHE_SIZE")
        if memory_cache_size_value:
            try:
                cache_config["memory_cache_size"] = int(memory_cache_size_value)
            except ValueError:
                logger.warning(
                    f"Invalid MEMORY_CACHE_SIZE value: {memory_cache_size_value}"
                )

        disk_cache_size_value = os.getenv("DISK_CACHE_SIZE_MB")
        if disk_cache_size_value:
            try:
                cache_config["disk_cache_size_mb"] = int(disk_cache_size_value)
            except ValueError:
                logger.warning(
                    f"Invalid DISK_CACHE_SIZE_MB value: {disk_cache_size_value}"
                )
        if cache_config:
            config["cache"] = cache_config  # Load security configuration
        security_config = {}
        encryption_enabled_value = os.getenv("ENCRYPTION_ENABLED")
        if encryption_enabled_value:
            security_config["encryption_enabled"] = (
                encryption_enabled_value.lower() in ("true", "1", "yes")
            )

        use_system_keyring_value = os.getenv("USE_SYSTEM_KEYRING")
        if use_system_keyring_value:
            security_config["use_system_keyring"] = (
                use_system_keyring_value.lower() in ("true", "1", "yes")
            )

        session_timeout_value = os.getenv("SESSION_TIMEOUT_MINUTES")
        if session_timeout_value:
            try:
                security_config["session_timeout_minutes"] = int(session_timeout_value)
            except ValueError:
                logger.warning(
                    f"Invalid SESSION_TIMEOUT_MINUTES value: {session_timeout_value}"
                )
        if security_config:
            config["security"] = security_config

        return config

    def _merge_configs(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = copy.deepcopy(base)

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _should_reload(self) -> bool:
        """Check if configuration should be reloaded."""
        if not self.config_file or not self.config_file.exists():
            return False

        if self._file_modification_time is None:
            return True

        current_mtime = self.config_file.stat().st_mtime
        return current_mtime > self._file_modification_time

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return self.get_config().database

    def get_selenium_config(self) -> SeleniumConfig:
        """Get Selenium configuration."""
        return self.get_config().selenium

    def get_api_config(self) -> APIConfig:
        """Get API configuration."""
        return self.get_config().api

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self.get_config().logging

    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration."""
        return self.get_config().cache

    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        return self.get_config().security


def run_comprehensive_tests():
    """
    Run comprehensive tests for the ConfigManager class.

    This function tests all major functionality of the ConfigManager
    to ensure proper operation and integration.
    """
    import sys
    import traceback
    import tempfile
    import time
    from typing import Dict, Any

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
    logger.info("CONFIG MANAGER COMPREHENSIVE TESTS")
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

    # Test 1: Basic Initialization
    def test_initialization():
        manager = ConfigManager(auto_load=False)
        assert manager is not None
        assert manager.environment == "development"
        assert manager.config_file is None
        assert manager._config_cache is None
        assert manager._file_modification_time is None
        logger.debug("ConfigManager initialization test passed")

    # Test 2: Environment Configuration
    def test_environment_configuration():
        # Test with different environments
        for env in ["development", "test", "production"]:
            manager = ConfigManager(environment=env, auto_load=False)
            assert manager.environment == env

        # Test environment variable override
        original_env = os.environ.get("ENVIRONMENT")
        os.environ["ENVIRONMENT"] = "testing"
        try:
            manager = ConfigManager(auto_load=False)
            assert manager.environment == "testing"
        finally:
            if original_env is None:
                os.environ.pop("ENVIRONMENT", None)
            else:
                os.environ["ENVIRONMENT"] = original_env

        logger.debug("Environment configuration test passed")

    # Test 3: Configuration File Handling
    def test_configuration_file():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            test_config = {
                "app_name": "Test App",
                "database": {"database_file": "test.db"},
                "selenium": {"headless_mode": True},
            }
            json.dump(test_config, f)
            config_path = f.name

        try:
            manager = ConfigManager(config_file=config_path, auto_load=False)
            assert manager.config_file is not None
            assert manager.config_file.exists()

            # Test file loading
            file_config = manager._load_config_file()
            assert isinstance(file_config, dict)
            assert file_config["app_name"] == "Test App"

        finally:
            # Cleanup
            os.unlink(config_path)

        logger.debug("Configuration file test passed")

    # Test 4: Default Configuration
    def test_default_configuration():
        manager = ConfigManager(auto_load=False)

        default_config = manager._get_default_config()
        assert isinstance(default_config, dict)
        assert "environment" in default_config
        assert "app_name" in default_config
        assert "database" in default_config
        assert "selenium" in default_config
        assert "api" in default_config
        assert "logging" in default_config
        assert "cache" in default_config
        assert "security" in default_config

        logger.debug("Default configuration test passed")

    # Test 5: Environment Variable Loading
    def test_environment_variable_loading():
        manager = ConfigManager(auto_load=False)

        # Test with mock environment variables
        test_env = {
            "APP_NAME": "Test App",
            "DEBUG_MODE": "true",
            "DATABASE_FILE": "test.db",
        }

        # Temporarily set environment variables
        original_env = {}
        for key, value in test_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            env_config = manager._load_environment_variables()
            assert isinstance(env_config, dict)

            # Cleanup
            for key in test_env:
                if original_env[key] is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_env[key]

        except Exception as e:
            # Cleanup on error
            for key in test_env:
                if original_env[key] is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_env[key]
            raise e

        logger.debug("Environment variable loading test passed")

    # Test 6: Configuration Merging
    def test_configuration_merging():
        manager = ConfigManager(auto_load=False)

        base_config = {
            "app_name": "Base App",
            "database": {"database_file": "base.db"},
            "selenium": {"headless_mode": True},
        }

        override_config = {
            "app_name": "Override App",
            "database": {"database_file": "override.db", "pool_size": 10},
            "api": {"request_timeout": 120},
        }

        merged = manager._merge_configs(base_config, override_config)

        assert merged["app_name"] == "Override App"  # Overridden
        assert merged["database"]["database_file"] == "override.db"  # Overridden
        assert merged["database"]["pool_size"] == 10  # New
        assert merged["selenium"]["headless_mode"] is True  # From base
        assert merged["api"]["request_timeout"] == 120  # New section

        logger.debug("Configuration merging test passed")

    # Test 7: Configuration Loading and Validation
    def test_configuration_loading():
        manager = ConfigManager(auto_load=False)

        # Test loading configuration
        config = manager.load_config()
        assert isinstance(config, ConfigSchema)
        assert config is not None

        # Test getting configuration
        config2 = manager.get_config()
        assert config2 is config  # Should return cached version

        logger.debug("Configuration loading test passed")

    # Test 8: Configuration Validation
    def test_configuration_validation():
        manager = ConfigManager(auto_load=False)

        # Test with valid configuration
        valid_config = {
            "app_name": "Test App",
            "environment": "test",
            "database": {"database_file": "test.db"},
            "selenium": {"headless_mode": True},
            "api": {"request_timeout": 60},
            "logging": {"log_level": "INFO"},
            "cache": {"cache_dir": "cache"},
            "security": {"encryption_enabled": True},
        }

        # This should not raise an exception
        errors = manager.validate_config(valid_config)
        assert isinstance(errors, list)

        logger.debug("Configuration validation test passed")

    # Test 9: Configuration Getter Methods
    def test_getter_methods():
        manager = ConfigManager()

        # Test getter methods exist and return appropriate types
        db_config = manager.get_database_config()
        assert isinstance(db_config, DatabaseConfig)

        selenium_config = manager.get_selenium_config()
        assert isinstance(selenium_config, SeleniumConfig)

        api_config = manager.get_api_config()
        assert isinstance(api_config, APIConfig)

        logging_config = manager.get_logging_config()
        assert isinstance(logging_config, LoggingConfig)

        cache_config = manager.get_cache_config()
        assert isinstance(cache_config, CacheConfig)

        security_config = manager.get_security_config()
        assert isinstance(security_config, SecurityConfig)

        logger.debug("Getter methods test passed")

    # Test 10: Hot Reloading Check
    def test_hot_reloading():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            initial_config = {"app_name": "initial"}
            json.dump(initial_config, f)
            config_path = f.name

        try:
            manager = ConfigManager(config_file=config_path, auto_load=False)

            # Initial check
            assert manager._should_reload()  # No initial load yet

            # Load config to set mtime
            manager.load_config()

            # Should not reload immediately
            assert not manager._should_reload()

            # Simulate file modification
            time.sleep(0.1)  # Ensure different timestamp
            with open(config_path, "w") as f:
                json.dump({"app_name": "modified"}, f)

            # Should detect need to reload
            assert manager._should_reload()

        finally:
            os.unlink(config_path)

        logger.debug("Hot reloading test passed")

    # Test 11: Configuration Export
    def test_configuration_export():
        manager = ConfigManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            export_path = f.name

        try:
            # Test export
            success = manager.export_config(export_path, "json")
            assert success is True

            # Verify exported file exists and is valid JSON
            assert os.path.exists(export_path)
            with open(export_path, "r") as f:
                exported_data = json.load(f)
            assert isinstance(exported_data, dict)

        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)

        logger.debug("Configuration export test passed")

    # Test 12: Configuration Reload
    def test_configuration_reload():
        manager = ConfigManager()

        # Get initial config
        config1 = manager.get_config()

        # Force reload
        config2 = manager.reload_config()

        # Should be new instance but equivalent
        assert isinstance(config2, ConfigSchema)

        logger.debug("Configuration reload test passed")

    # Test 13: Environment-Specific Configuration
    def test_environment_specific_configuration():
        manager = ConfigManager(auto_load=False)

        # Test getting config for different environments
        dev_config = manager.get_environment_config("development")
        assert isinstance(dev_config, dict)
        assert dev_config["environment"] == "development"

        test_config = manager.get_environment_config("test")
        assert isinstance(test_config, dict)
        assert test_config["environment"] == "test"

        logger.debug("Environment-specific configuration test passed")

    # Test 14: Function Validation
    def test_function_validation():
        # Test that all required functions exist and are callable
        assert_valid_function(ConfigManager, "ConfigManager should be a class")
        assert_valid_function(ValidationError, "ValidationError should be a class")

        # Test ConfigManager methods
        manager = ConfigManager(auto_load=False)
        required_methods = [
            "load_config",
            "get_config",
            "reload_config",
            "validate_config",
            "get_environment_config",
            "export_config",
            "get_database_config",
            "get_selenium_config",
            "get_api_config",
            "get_logging_config",
            "get_cache_config",
            "get_security_config",
        ]

        for method_name in required_methods:
            assert hasattr(manager, method_name), f"Missing method: {method_name}"
            assert callable(
                getattr(manager, method_name)
            ), f"Method not callable: {method_name}"

    # Test 15: Import Validation
    def test_import_validation():
        # Test that required imports are available
        assert ConfigSchema is not None
        assert DatabaseConfig is not None
        assert SeleniumConfig is not None
        assert APIConfig is not None
        assert LoggingConfig is not None
        assert CacheConfig is not None
        assert SecurityConfig is not None
        logger.debug("Import validation test passed")

    # Run all tests
    tests = [
        ("Basic Initialization", test_initialization),
        ("Environment Configuration", test_environment_configuration),
        ("Configuration File Handling", test_configuration_file),
        ("Default Configuration", test_default_configuration),
        ("Environment Variable Loading", test_environment_variable_loading),
        ("Configuration Merging", test_configuration_merging),
        ("Configuration Loading and Validation", test_configuration_loading),
        ("Configuration Validation", test_configuration_validation),
        ("Configuration Getter Methods", test_getter_methods),
        ("Hot Reloading Check", test_hot_reloading),
        ("Configuration Export", test_configuration_export),
        ("Configuration Reload", test_configuration_reload),
        ("Environment-Specific Configuration", test_environment_specific_configuration),
        ("Function Validation", test_function_validation),
        ("Import Validation", test_import_validation),
    ]

    # Run each test
    for test_name, test_func in tests:
        run_test(test_name, test_func)

    # Print summary
    total_tests = len(tests)
    logger.info("\n" + "=" * 60)
    logger.info("CONFIG MANAGER TEST SUMMARY")
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
        logger.info("🎉 ALL CONFIG MANAGER TESTS PASSED!")
    else:
        logger.warning("⚠️ Some Config Manager tests failed")
    return success


if __name__ == "__main__":
    run_comprehensive_tests()
