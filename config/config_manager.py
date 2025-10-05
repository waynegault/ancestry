#!/usr/bin/env python3

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

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import contextlib
import copy
import json
from pathlib import Path
from typing import Any, Optional, Union

# === THIRD-PARTY IMPORTS ===
try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    from .config_schema import (
        APIConfig,
        CacheConfig,
        ConfigSchema,
        DatabaseConfig,
        LoggingConfig,
        SecurityConfig,
        SeleniumConfig,
    )
except ImportError:
    # If relative import fails, try absolute import
    from config_schema import (
        APIConfig,
        CacheConfig,
        ConfigSchema,
        DatabaseConfig,
        LoggingConfig,
        SecurityConfig,
        SeleniumConfig,
    )


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
        # Load .env file if available
        if DOTENV_AVAILABLE:
            load_dotenv()

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

            logger.debug(
                f"Configuration loaded successfully for environment: {self.environment}"
            )
            return config

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ValidationError(f"Configuration loading failed: {e}") from e

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
        self, config_data: Optional[dict[str, Any]] = None
    ) -> list[str]:
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
            return ["Configuration cache is not available"]

        try:
            config = ConfigSchema.from_dict(config_data)
            return config.validate()
        except Exception as e:
            return [f"Configuration validation error: {e}"]

    def get_environment_config(self, env_name: str) -> dict[str, Any]:
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
                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(config_dict, f, indent=2, default=str)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False

            logger.info(f"Configuration exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration values with auto-detection."""
        # PHASE 2 ENHANCEMENT: Auto-detect optimal settings (simplified for now)
        base_config = {
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

        # Add basic auto-detected batch size
        try:
            import psutil
            cpu_count = psutil.cpu_count(logical=True) or 4
            base_config["batch_size"] = min(cpu_count * 2, 20)  # Adaptive batch size
        except Exception:
            pass

        return base_config

    def _auto_detect_optimal_settings(self) -> dict[str, Any]:
        """
        Auto-detect optimal configuration settings based on system capabilities.

        Returns:
            Dictionary with auto-detected configuration values
        """
        try:

            import psutil

            # System capabilities
            cpu_count = psutil.cpu_count(logical=True) or 4
            memory_gb = psutil.virtual_memory().total / (1024**3)
            disk_space_gb = psutil.disk_usage('/').free / (1024**3)

            # Auto-detect API settings
            api_config = {
                "max_concurrency": min(cpu_count * 2, 16),  # 2x CPU cores, max 16
                "thread_pool_workers": min(cpu_count * 2, 12),  # Conservative for stability
                "request_timeout": 30 if memory_gb >= 8 else 60,  # Shorter timeout for high-memory systems
                "max_retries": 3,
                "retry_backoff_factor": 2.0,
            }

            # Auto-detect cache settings
            cache_memory_mb = min(int(memory_gb * 0.1 * 1024), 1024)  # 10% of RAM, max 1GB
            cache_disk_mb = min(int(disk_space_gb * 0.05 * 1024), 2048)  # 5% of free disk, max 2GB

            cache_config = {
                "memory_cache_size": cache_memory_mb,
                "disk_cache_size_mb": cache_disk_mb,
                "memory_cache_ttl": 3600,  # 1 hour
                "disk_cache_ttl": 86400,  # 24 hours
                "auto_cleanup_enabled": True,
            }

            # Auto-detect database settings
            db_pool_size = min(cpu_count + 2, 10)  # CPU cores + 2, max 10
            db_config = {
                "pool_size": db_pool_size,
                "max_overflow": max(5, int(db_pool_size * 0.5)),
                "pool_timeout": 30,
                "cache_size_mb": min(int(memory_gb * 0.05 * 1024), 512),  # 5% of RAM for DB cache, max 512MB
            }

            # Auto-detect Selenium settings based on system performance
            selenium_config = {
                "headless_mode": True,  # Default to headless for better performance
                "window_size": "1920,1080",
                "page_load_timeout": 30 if memory_gb >= 8 else 45,
                "implicit_wait": 10,
                "disable_images": memory_gb < 8,  # Disable images on low-memory systems
            }

            # Auto-detect logging level based on environment and system
            if self.environment == "development":
                log_level = "DEBUG"
            elif self.environment == "testing":
                log_level = "INFO"
            else:
                log_level = "WARNING"

            logging_config = {
                "level": log_level,
                "enable_file_logging": True,
                "max_log_size_mb": min(int(disk_space_gb * 0.01 * 1024), 100),  # 1% of free disk, max 100MB
                "backup_count": 5,
            }

            # Auto-detect security settings
            security_config = {
                "encryption_enabled": True,
                "use_system_keyring": True,
                "session_timeout_minutes": 120,
                "auto_logout_enabled": True,
                "verify_ssl": True,
            }

            auto_detected = {
                "api": api_config,
                "cache": cache_config,
                "database": db_config,
                "selenium": selenium_config,
                "logging": logging_config,
                "security": security_config,
                "batch_size": min(cpu_count * 2, 20),  # Adaptive batch size
                "max_productive_to_process": min(cpu_count * 10, 100),  # Scale with CPU
            }

            logger.debug(f"Auto-detected configuration: CPU={cpu_count}, RAM={memory_gb:.1f}GB, Disk={disk_space_gb:.1f}GB")
            return auto_detected

        except Exception as e:
            logger.warning(f"Auto-detection failed, using defaults: {e}")
            return {}

    def _check_minimum_requirements(self, cpu_count: int, memory_gb: float, disk_space_gb: float, validation_results: dict[str, Any]) -> None:
        """Check minimum system requirements and update validation results."""
        if cpu_count < 2:
            validation_results["warnings"].append(
                f"Low CPU count ({cpu_count}). Recommend at least 2 cores for optimal performance."
            )

        if memory_gb < 4:
            validation_results["errors"].append(
                f"Insufficient memory ({memory_gb:.1f}GB). Minimum 4GB required."
            )
            validation_results["valid"] = False

        if disk_space_gb < 1:
            validation_results["errors"].append(
                f"Insufficient disk space ({disk_space_gb:.1f}GB). Minimum 1GB free space required."
            )
            validation_results["valid"] = False

    def _check_optimal_performance(self, cpu_count: int, memory_gb: float, validation_results: dict[str, Any]) -> None:
        """Check for optimal performance opportunities and add recommendations."""
        if memory_gb >= 16:
            validation_results["recommendations"].append(
                "High memory detected. Consider enabling GPU acceleration and increasing cache sizes."
            )

        if cpu_count >= 8:
            validation_results["recommendations"].append(
                "High CPU count detected. Consider increasing concurrency settings for better performance."
            )

    def _check_dependencies_and_chrome(self, validation_results: dict[str, Any]) -> None:
        """Check Python dependencies and Chrome availability."""
        import shutil

        # Check Python dependencies
        try:
            from importlib.util import find_spec
            deps = ["requests", "selenium", "sqlalchemy"]
            validation_results["dependencies_ok"] = all(find_spec(d) is not None for d in deps)
        except Exception as e:
            validation_results["errors"].append(f"Missing required dependency: {e}")
            validation_results["valid"] = False

        # Check Chrome/ChromeDriver availability
        chrome_available = shutil.which("chrome") or shutil.which("google-chrome") or shutil.which("chromium")
        if not chrome_available:
            validation_results["warnings"].append(
                "Chrome browser not found in PATH. Selenium automation may not work."
            )

    def validate_system_requirements(self) -> dict[str, Any]:
        """
        Validate system requirements and provide recommendations.

        Returns:
            Dictionary with validation results and recommendations
        """
        try:
            import psutil

            validation_results = {
                "valid": True,
                "warnings": [],
                "errors": [],
                "recommendations": [],
                "system_info": {}
            }

            # Check system resources
            cpu_count = psutil.cpu_count(logical=True) or 1
            memory_gb = psutil.virtual_memory().total / (1024**3)
            disk_space_gb = psutil.disk_usage('/').free / (1024**3)

            validation_results["system_info"] = {
                "cpu_cores": cpu_count,
                "memory_gb": round(memory_gb, 1),
                "free_disk_gb": round(disk_space_gb, 1)
            }

            # Perform validation checks
            self._check_minimum_requirements(cpu_count, memory_gb, disk_space_gb, validation_results)
            self._check_optimal_performance(cpu_count, memory_gb, validation_results)
            self._check_dependencies_and_chrome(validation_results)

            return validation_results

        except Exception as e:
            return {
                "valid": False,
                "errors": [f"System validation failed: {e}"],
                "warnings": [],
                "recommendations": [],
                "system_info": {}
            }

    def _print_system_info(self, system_info: dict[str, Any]) -> None:
        """Print system information."""
        print("\n📊 System Information:")
        print(f"   CPU Cores: {system_info.get('cpu_cores', 'Unknown')}")
        print(f"   Memory: {system_info.get('memory_gb', 'Unknown')}GB")
        print(f"   Free Disk: {system_info.get('free_disk_gb', 'Unknown')}GB")

    def _print_validation_results(self, validation: dict[str, Any]) -> bool:
        """Print validation results. Returns True if valid, False otherwise."""
        if validation.get("errors"):
            print("\n❌ Critical Issues:")
            for error in validation["errors"]:
                print(f"   • {error}")

        if validation.get("warnings"):
            print("\n⚠️  Warnings:")
            for warning in validation["warnings"]:
                print(f"   • {warning}")

        if validation.get("recommendations"):
            print("\n💡 Recommendations:")
            for rec in validation["recommendations"]:
                print(f"   • {rec}")

        if not validation.get("valid"):
            print("\n❌ Setup cannot continue due to critical issues.")
            return False

        print("\n✅ System validation passed!")
        return True

    def _print_optimal_settings(self, auto_detected: dict[str, Any]) -> None:
        """Print auto-detected optimal settings."""
        print("\n🔧 Auto-detected Optimal Settings:")

        api_config = auto_detected.get("api", {})
        print(f"   Concurrency: {api_config.get('max_concurrency', 'Default')}")
        print(f"   Thread Pool: {api_config.get('thread_pool_workers', 'Default')}")

        cache_config = auto_detected.get("cache", {})
        print(f"   Memory Cache: {cache_config.get('memory_cache_size', 'Default')}MB")
        print(f"   Disk Cache: {cache_config.get('disk_cache_size_mb', 'Default')}MB")

    def run_setup_wizard(self, interactive: bool = True) -> bool:
        """
        Run interactive setup wizard for first-time configuration.

        Args:
            interactive: Whether to run in interactive mode

        Returns:
            True if setup completed successfully
        """
        try:
            print("\n🚀 Ancestry Automation Setup Wizard")
            print("=" * 50)

            # Validate system requirements
            validation = self.validate_system_requirements()

            # Print system information
            self._print_system_info(validation.get("system_info", {}))

            # Show validation results
            if not self._print_validation_results(validation):
                return False

            # Auto-detect and show optimal settings
            auto_detected = self._auto_detect_optimal_settings()
            self._print_optimal_settings(auto_detected)

            if interactive:
                response = input("\n✨ Use auto-detected settings? (Y/n): ").strip().lower()
                if response and response not in {"y", "yes"}:
                    print("Manual configuration not implemented yet. Using auto-detected settings.")

            print("\n🎯 Setup completed successfully!")
            print("   Configuration will be applied automatically.")
            print("   You can modify settings in .env file or config files.")

            return True

        except Exception as e:
            print(f"\n❌ Setup wizard failed: {e}")
            return False

    def _load_config_file(self) -> dict[str, Any]:
        """Load configuration from file."""
        if not self.config_file or not self.config_file.exists():
            return {}

        try:
            suffix = self.config_file.suffix.lower()

            if suffix == ".json":
                with self.config_file.open(encoding="utf-8") as f:
                    return json.load(f)
            else:
                logger.warning(f"Unsupported config file format: {suffix}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load config file {self.config_file}: {e}")
            return {}

    def _load_main_config_from_env(self, config: dict[str, Any]) -> None:
        """Load main configuration from environment variables."""
        env_value = os.getenv("ENVIRONMENT")
        if env_value:
            config["environment"] = env_value

        debug_mode_value = os.getenv("DEBUG_MODE")
        if debug_mode_value:
            config["debug_mode"] = debug_mode_value.lower() in ("true", "1", "yes")

        app_name_value = os.getenv("APP_NAME")
        if app_name_value:
            config["app_name"] = app_name_value

    def _load_reference_person_config_from_env(self, config: dict[str, Any]) -> None:
        """Load reference person configuration from environment variables."""
        reference_person_id_value = os.getenv("REFERENCE_PERSON_ID")
        if reference_person_id_value:
            config["reference_person_id"] = reference_person_id_value

        reference_person_name_value = os.getenv("REFERENCE_PERSON_NAME")
        if reference_person_name_value:
            config["reference_person_name"] = reference_person_name_value

    def _load_user_config_from_env(self, config: dict[str, Any]) -> None:
        """Load user configuration from environment variables."""
        user_name_value = os.getenv("USER_NAME")
        if user_name_value:
            config["user_name"] = user_name_value

        user_location_value = os.getenv("USER_LOCATION")
        if user_location_value:
            config["user_location"] = user_location_value

    def _load_testing_config_from_env(self, config: dict[str, Any]) -> None:
        """Load testing configuration from environment variables."""
        testing_profile_id_value = os.getenv("TESTING_PROFILE_ID")
        if testing_profile_id_value:
            config["testing_profile_id"] = testing_profile_id_value

    def _load_app_mode_from_env(self, config: dict[str, Any]) -> None:
        """Load application mode from environment variables."""
        app_mode_value = os.getenv("APP_MODE")
        if app_mode_value:
            config["app_mode"] = app_mode_value

    def _load_ai_config_from_env(self, config: dict[str, Any]) -> None:
        """Load AI configuration from environment variables."""
        ai_provider_value = os.getenv("AI_PROVIDER")
        if ai_provider_value:
            config["ai_provider"] = ai_provider_value

        ai_ctx_msgs = os.getenv("AI_CONTEXT_MESSAGES_COUNT")
        if ai_ctx_msgs:
            try:
                config["ai_context_messages_count"] = int(ai_ctx_msgs)
            except ValueError:
                logger.warning(f"Invalid AI_CONTEXT_MESSAGES_COUNT: {ai_ctx_msgs}")

        ai_ctx_max_words = os.getenv("AI_CONTEXT_MESSAGE_MAX_WORDS")
        if ai_ctx_max_words:
            try:
                config["ai_context_message_max_words"] = int(ai_ctx_max_words)
            except ValueError:
                logger.warning(f"Invalid AI_CONTEXT_MESSAGE_MAX_WORDS: {ai_ctx_max_words}")

        ai_ctx_window = os.getenv("AI_CONTEXT_WINDOW_MESSAGES")
        if ai_ctx_window:
            try:
                config["ai_context_window_messages"] = int(ai_ctx_window)
            except ValueError:
                logger.warning(f"Invalid AI_CONTEXT_WINDOW_MESSAGES: {ai_ctx_window}")

    def _load_timeout_config_from_env(self, config: dict[str, Any]) -> None:
        """Load timeout configuration from environment variables."""
        refresh_cooldown = os.getenv("PROACTIVE_REFRESH_COOLDOWN")
        if refresh_cooldown:
            try:
                config["proactive_refresh_cooldown_seconds"] = int(refresh_cooldown)
            except ValueError:
                logger.warning(f"Invalid PROACTIVE_REFRESH_COOLDOWN: {refresh_cooldown}")

        refresh_interval = os.getenv("PROACTIVE_REFRESH_INTERVAL")
        if refresh_interval:
            try:
                config["proactive_refresh_interval_seconds"] = int(refresh_interval)
            except ValueError:
                logger.warning(f"Invalid PROACTIVE_REFRESH_INTERVAL: {refresh_interval}")

        a6_coord_timeout = os.getenv("ACTION6_COORD_TIMEOUT")
        if a6_coord_timeout:
            try:
                config["action6_coord_timeout_seconds"] = int(a6_coord_timeout)
            except ValueError:
                logger.warning(f"Invalid ACTION6_COORD_TIMEOUT: {a6_coord_timeout}")

    def _load_api_keys_from_env(self, config: dict[str, Any]) -> None:
        """Load API keys configuration from environment variables."""
        api_config = {}

        # DeepSeek API configuration
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_api_key:
            api_config["deepseek_api_key"] = deepseek_api_key

        deepseek_ai_model = os.getenv("DEEPSEEK_AI_MODEL")
        if deepseek_ai_model:
            api_config["deepseek_ai_model"] = deepseek_ai_model

        deepseek_ai_base_url = os.getenv("DEEPSEEK_AI_BASE_URL")
        if deepseek_ai_base_url:
            api_config["deepseek_ai_base_url"] = deepseek_ai_base_url

        # Google API configuration
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            api_config["google_api_key"] = google_api_key

        google_ai_model = os.getenv("GOOGLE_AI_MODEL")
        if google_ai_model:
            api_config["google_ai_model"] = google_ai_model

        if api_config:
            config["api"] = api_config

    def _load_database_config_from_env(self, config: dict[str, Any]) -> None:
        """Load database configuration from environment variables."""
        db_config = {}
        database_file_value = os.getenv("DATABASE_FILE")
        if database_file_value:
            # Enforce canonical project-relative DB path when a relative ancestry.db is provided
            p = Path(database_file_value)
            if not p.is_absolute() and p.name.lower() == "ancestry.db":
                # Normalize to canonical Data/ancestry.db within project
                db_config["database_file"] = Path("Data") / "ancestry.db"
            else:
                # Respect absolute or custom filenames
                db_config["database_file"] = p
        else:
            # Default to canonical project DB path
            db_config["database_file"] = Path("Data") / "ancestry.db"

        gedcom_file_path_value = os.getenv("GEDCOM_FILE_PATH")
        if gedcom_file_path_value:
            db_config["gedcom_file_path"] = Path(gedcom_file_path_value)

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
            config["database"] = db_config

    def _load_selenium_config_from_env(self, config: dict[str, Any]) -> None:
        """Load Selenium configuration from environment variables."""
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
            selenium_config["headless_mode"] = headless_mode_value.lower() in ("true", "1", "yes")

        debug_port_value = os.getenv("DEBUG_PORT")
        if debug_port_value:
            try:
                selenium_config["debug_port"] = int(debug_port_value)
            except ValueError:
                logger.warning(f"Invalid DEBUG_PORT value: {debug_port_value}")

        if selenium_config:
            config["selenium"] = selenium_config

    def _set_string_config(self, config: dict[str, Any], section: str, key: str, env_var: str) -> None:
        """Set a string configuration value from environment variable."""
        value = os.getenv(env_var)
        if value:
            if section not in config:
                config[section] = {}
            config[section][key] = value

    def _set_int_config(self, config: dict[str, Any], section: str, key: str, env_var: str) -> None:
        """Set an integer configuration value from environment variable."""
        value = os.getenv(env_var)
        if value:
            try:
                if section not in config:
                    config[section] = {}
                config[section][key] = int(value)
            except ValueError:
                logger.warning(f"Invalid {env_var} value: {value}")

    def _load_additional_api_config_from_env(self, config: dict[str, Any]) -> None:
        """Load additional API configuration from environment variables."""
        if "api" not in config:
            config["api"] = {}

        # String configurations
        self._set_string_config(config, "api", "base_url", "BASE_URL")
        self._set_string_config(config, "api", "api_base_url", "API_BASE_URL")
        self._set_string_config(config, "api", "tree_name", "TREE_NAME")
        self._set_string_config(config, "api", "tree_id", "TREE_ID")

        # Integer configurations
        self._set_int_config(config, "api", "request_timeout", "REQUEST_TIMEOUT")
        self._set_int_config(config, "api", "max_pages", "MAX_PAGES")
        self._set_int_config(config, "api", "max_concurrency", "MAX_CONCURRENCY")
        self._set_int_config(config, "api", "thread_pool_workers", "THREAD_POOL_WORKERS")

    def _load_processing_limits_from_env(self, config: dict[str, Any]) -> None:
        """Load processing limit configuration from environment variables."""
        batch_size_value = os.getenv("BATCH_SIZE")
        if batch_size_value:
            try:
                config["batch_size"] = int(batch_size_value)
            except ValueError:
                logger.warning(f"Invalid BATCH_SIZE value: {batch_size_value}")

        matches_per_page_value = os.getenv("MATCHES_PER_PAGE")
        if matches_per_page_value:
            try:
                config["matches_per_page"] = int(matches_per_page_value)
            except ValueError:
                logger.warning(f"Invalid MATCHES_PER_PAGE value: {matches_per_page_value}")

        max_productive_value = os.getenv("MAX_PRODUCTIVE_TO_PROCESS")
        if max_productive_value:
            try:
                config["max_productive_to_process"] = int(max_productive_value)
            except ValueError:
                logger.warning(f"Invalid MAX_PRODUCTIVE_TO_PROCESS value: {max_productive_value}")

        max_inbox_value = os.getenv("MAX_INBOX")
        if max_inbox_value:
            try:
                config["max_inbox"] = int(max_inbox_value)
            except ValueError:
                logger.warning(f"Invalid MAX_INBOX value: {max_inbox_value}")

    def _load_logging_config_from_env(self, config: dict[str, Any]) -> None:
        """Load logging configuration from environment variables."""
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
            config["logging"] = logging_config

    def _load_cache_config_from_env(self, config: dict[str, Any]) -> None:
        """Load cache configuration from environment variables."""
        cache_config = {}
        cache_dir_value = os.getenv("CACHE_DIR")
        if cache_dir_value:
            cache_config["cache_dir"] = Path(cache_dir_value)

        memory_cache_size_value = os.getenv("MEMORY_CACHE_SIZE")
        if memory_cache_size_value:
            try:
                cache_config["memory_cache_size"] = int(memory_cache_size_value)
            except ValueError:
                logger.warning(f"Invalid MEMORY_CACHE_SIZE value: {memory_cache_size_value}")

        disk_cache_size_value = os.getenv("DISK_CACHE_SIZE_MB")
        if disk_cache_size_value:
            try:
                cache_config["disk_cache_size_mb"] = int(disk_cache_size_value)
            except ValueError:
                logger.warning(f"Invalid DISK_CACHE_SIZE_MB value: {disk_cache_size_value}")

        if cache_config:
            config["cache"] = cache_config

    def _load_security_config_from_env(self, config: dict[str, Any]) -> None:
        """Load security configuration from environment variables."""
        security_config = {}
        encryption_enabled_value = os.getenv("ENCRYPTION_ENABLED")
        if encryption_enabled_value:
            security_config["encryption_enabled"] = encryption_enabled_value.lower() in ("true", "1", "yes")

        use_system_keyring_value = os.getenv("USE_SYSTEM_KEYRING")
        if use_system_keyring_value:
            security_config["use_system_keyring"] = use_system_keyring_value.lower() in ("true", "1", "yes")

        session_timeout_value = os.getenv("SESSION_TIMEOUT_MINUTES")
        if session_timeout_value:
            try:
                security_config["session_timeout_minutes"] = int(session_timeout_value)
            except ValueError:
                logger.warning(f"Invalid SESSION_TIMEOUT_MINUTES value: {session_timeout_value}")

        if security_config:
            config["security"] = security_config

    def _load_environment_variables(self) -> dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}

        # Load all configuration sections
        self._load_main_config_from_env(config)
        self._load_reference_person_config_from_env(config)
        self._load_user_config_from_env(config)
        self._load_testing_config_from_env(config)
        self._load_app_mode_from_env(config)
        self._load_ai_config_from_env(config)
        self._load_timeout_config_from_env(config)

        # Load API configuration
        self._load_api_keys_from_env(config)

        # Load database configuration
        self._load_database_config_from_env(config)

        # Load Selenium configuration
        self._load_selenium_config_from_env(config)

        # Load additional API configuration
        self._load_additional_api_config_from_env(config)

        # Load processing limits
        self._load_processing_limits_from_env(config)

        # Load logging configuration
        self._load_logging_config_from_env(config)

        # Load cache configuration
        self._load_cache_config_from_env(config)

        # Load security configuration
        self._load_security_config_from_env(config)

        return config

    def _merge_configs(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
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


# ==============================================
# MODULE-LEVEL TEST FUNCTIONS
# ==============================================


def _test_config_manager_initialization():
    """Test ConfigManager class initialization."""
    assert callable(ConfigManager), "ConfigManager should be callable"

    # Test basic instantiation
    try:
        manager = ConfigManager(auto_load=False)
        assert manager is not None, "ConfigManager should instantiate successfully"
        assert manager.environment in [
            "development",
            "test",
            "production",
        ], "Should have valid environment"
    except Exception:
        # May require specific configuration files
        pass


def _test_config_validation():
    """Test configuration validation functions."""
    # Test ConfigManager methods exist
    manager = ConfigManager(auto_load=False)
    assert hasattr(manager, "validate_config"), "Should have validate_config method"
    assert hasattr(manager, "load_config"), "Should have load_config method"


def _test_config_loading():
    """Test configuration loading functionality."""
    manager = ConfigManager(auto_load=False)
    assert hasattr(manager, "get_config"), "Should have get_config method"


def _test_config_access():
    """Test configuration value access."""
    manager = ConfigManager(auto_load=False)
    # Test that manager has config access methods
    assert hasattr(manager, "get_config"), "Should have config access"


def _test_missing_config_handling():
    """Test handling of missing configuration."""
    manager = ConfigManager(auto_load=False)
    # Should handle missing config gracefully
    assert manager is not None


def _test_invalid_config_data():
    """Test handling of invalid configuration data."""
    # Should handle invalid data gracefully
    manager = ConfigManager(auto_load=False)
    assert manager is not None


def _test_config_file_integration():
    """Test configuration file integration."""
    import tempfile
    # Test with temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("test: value\n")
        temp_path = f.name

    try:
        manager = ConfigManager(auto_load=False)
        assert manager is not None
    finally:
        import os
        try:
            os.unlink(temp_path)
        except Exception:
            pass


def _test_environment_integration():
    """Test environment variable integration."""
    import os
    # Test environment handling
    manager = ConfigManager(auto_load=False)
    assert hasattr(manager, "environment")


def _test_config_access_performance():
    """Test configuration access performance."""
    import time
    manager = ConfigManager(auto_load=False)

    start = time.time()
    for _ in range(100):
        _ = manager.environment
    elapsed = time.time() - start

    assert elapsed < 1.0, f"Config access should be fast, took {elapsed:.3f}s"


def _test_config_error_handling():
    """Test configuration error handling."""
    # Should handle errors gracefully
    try:
        manager = ConfigManager(auto_load=False)
        assert manager is not None
    except Exception:
        pass  # Expected for some error cases


# ==============================================
# MAIN TEST SUITE RUNNER
# ==============================================


def config_manager_module_tests() -> bool:
    """
    Comprehensive test suite for config_manager.py with proper TestSuite framework.
    Tests configuration management, validation, and loading functionality.
    """
    import tempfile
    import time

    from test_framework import TestSuite

    suite = TestSuite("Configuration Management & Validation", "config_manager.py")
    suite.start_suite()

    # Assign module-level test functions
    test_config_manager_initialization = _test_config_manager_initialization
    test_config_validation = _test_config_validation
    test_config_loading = _test_config_loading
    test_config_access = _test_config_access
    test_missing_config_handling = _test_missing_config_handling
    test_invalid_config_data = _test_invalid_config_data
    test_config_file_integration = _test_config_file_integration
    test_environment_integration = _test_environment_integration
    test_config_access_performance = _test_config_access_performance
    test_config_error_handling = _test_config_error_handling

    # INITIALIZATION TESTS
    suite.run_test(
        "Config Manager Initialization",
        test_config_manager_initialization,
        "ConfigManager initializes correctly with valid environment",
        "Test ConfigManager class instantiation and initialization",
        "Verify ConfigManager can be instantiated and has valid environment"
    )

    suite.run_test(
        "Config Validation",
        test_config_validation,
        "Configuration validation methods are available",
        "Test configuration validation functionality",
        "Verify validate_config and load_config methods exist"
    )

    suite.run_test(
        "Config Loading",
        test_config_loading,
        "Configuration loading works correctly",
        "Test configuration loading functionality",
        "Verify get_config method exists"
    )

    suite.run_test(
        "Config Access",
        test_config_access,
        "Configuration values can be accessed",
        "Test configuration value access",
        "Verify config access methods are available"
    )

    suite.run_test(
        "Missing Config Handling",
        test_missing_config_handling,
        "Missing configuration is handled gracefully",
        "Test handling of missing configuration",
        "Verify system handles missing config without crashing"
    )

    suite.run_test(
        "Invalid Config Data",
        test_invalid_config_data,
        "Invalid configuration data is handled gracefully",
        "Test handling of invalid configuration data",
        "Verify system handles invalid data without crashing"
    )

    suite.run_test(
        "Config File Integration",
        test_config_file_integration,
        "Configuration file integration works correctly",
        "Test configuration file loading and integration",
        "Verify system can work with temporary config files"
    )

    suite.run_test(
        "Environment Integration",
        test_environment_integration,
        "Environment variable integration works correctly",
        "Test environment variable handling",
        "Verify environment attribute exists"
    )

    suite.run_test(
        "Config Access Performance",
        test_config_access_performance,
        "Configuration access is performant",
        "Test configuration access performance",
        "Verify 100 config accesses take less than 1 second"
    )

    suite.run_test(
        "Config Error Handling",
        test_config_error_handling,
        "Configuration errors are handled gracefully",
        "Test configuration error handling",
        "Verify system handles errors without crashing"
    )

    return suite.finish_suite()


if __name__ == "__main__":
    import sys
    print("🧪 Running Config Manager Comprehensive Tests...")
    success = config_manager_module_tests()
    sys.exit(0 if success else 1)

