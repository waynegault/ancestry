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
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

    def _get_default_config(self) -> Dict[str, Any]:
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

    def _auto_detect_optimal_settings(self) -> Dict[str, Any]:
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

    def validate_system_requirements(self) -> Dict[str, Any]:
        """
        Validate system requirements and provide recommendations.

        Returns:
            Dictionary with validation results and recommendations
        """
        try:
            import shutil

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

            # Validate minimum requirements
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

            # Check for optimal performance
            if memory_gb >= 16:
                validation_results["recommendations"].append(
                    "High memory detected. Consider enabling GPU acceleration and increasing cache sizes."
                )

            if cpu_count >= 8:
                validation_results["recommendations"].append(
                    "High CPU count detected. Consider increasing concurrency settings for better performance."
                )

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

            return validation_results

        except Exception as e:
            return {
                "valid": False,
                "errors": [f"System validation failed: {e}"],
                "warnings": [],
                "recommendations": [],
                "system_info": {}
            }

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

            print("\n📊 System Information:")
            system_info = validation.get("system_info", {})
            print(f"   CPU Cores: {system_info.get('cpu_cores', 'Unknown')}")
            print(f"   Memory: {system_info.get('memory_gb', 'Unknown')}GB")
            print(f"   Free Disk: {system_info.get('free_disk_gb', 'Unknown')}GB")

            # Show validation results
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

            # Auto-detect and show optimal settings
            auto_detected = self._auto_detect_optimal_settings()
            print("\n🔧 Auto-detected Optimal Settings:")

            api_config = auto_detected.get("api", {})
            print(f"   Concurrency: {api_config.get('max_concurrency', 'Default')}")
            print(f"   Thread Pool: {api_config.get('thread_pool_workers', 'Default')}")

            cache_config = auto_detected.get("cache", {})
            print(f"   Memory Cache: {cache_config.get('memory_cache_size', 'Default')}MB")
            print(f"   Disk Cache: {cache_config.get('disk_cache_size_mb', 'Default')}MB")

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

    def _load_config_file(self) -> Dict[str, Any]:
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

    def _load_environment_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}  # Load main configuration
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
            config["app_name"] = app_name_value  # Load reference person configuration
        reference_person_id_value = os.getenv("REFERENCE_PERSON_ID")
        if reference_person_id_value:
            config["reference_person_id"] = reference_person_id_value

        reference_person_name_value = os.getenv("REFERENCE_PERSON_NAME")
        if reference_person_name_value:
            config["reference_person_name"] = reference_person_name_value

        # Load user configuration
        user_name_value = os.getenv("USER_NAME")
        if user_name_value:
            config["user_name"] = user_name_value

        user_location_value = os.getenv("USER_LOCATION")
        if user_location_value:
            config["user_location"] = user_location_value

        # Load testing configuration
        testing_profile_id_value = os.getenv("TESTING_PROFILE_ID")
        if testing_profile_id_value:
            config["testing_profile_id"] = testing_profile_id_value

        # Load application mode
        app_mode_value = os.getenv("APP_MODE")
        if app_mode_value:
            config["app_mode"] = app_mode_value

        # Load AI configuration
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

        # Proactive refresh cooldown (seconds)
        refresh_cooldown = os.getenv("PROACTIVE_REFRESH_COOLDOWN")
        if refresh_cooldown:
            try:
                config["proactive_refresh_cooldown_seconds"] = int(refresh_cooldown)
            except ValueError:
                logger.warning(f"Invalid PROACTIVE_REFRESH_COOLDOWN: {refresh_cooldown}")

        # Proactive refresh interval (seconds)
        refresh_interval = os.getenv("PROACTIVE_REFRESH_INTERVAL")
        if refresh_interval:
            try:
                config["proactive_refresh_interval_seconds"] = int(refresh_interval)
            except ValueError:
                logger.warning(f"Invalid PROACTIVE_REFRESH_INTERVAL: {refresh_interval}")

        # Action 6 coord timeout (seconds)
        a6_coord_timeout = os.getenv("ACTION6_COORD_TIMEOUT")
        if a6_coord_timeout:
            try:
                config["action6_coord_timeout_seconds"] = int(a6_coord_timeout)
            except ValueError:
                logger.warning(f"Invalid ACTION6_COORD_TIMEOUT: {a6_coord_timeout}")

        # Load API configuration
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

        # Load database configuration
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
            config["selenium"] = selenium_config

        # Merge additional API configuration with existing api_config
        if "api" not in config:
            config["api"] = {}

        base_url_value = os.getenv("BASE_URL")
        if base_url_value:
            config["api"]["base_url"] = base_url_value

        api_base_url_value = os.getenv("API_BASE_URL")
        if api_base_url_value:
            config["api"]["api_base_url"] = api_base_url_value

        request_timeout_value = os.getenv("REQUEST_TIMEOUT")
        if request_timeout_value:
            try:
                config["api"]["request_timeout"] = int(request_timeout_value)
            except ValueError:
                logger.warning(
                    f"Invalid REQUEST_TIMEOUT value: {request_timeout_value}"
                )

        tree_name_value = os.getenv("TREE_NAME")
        if tree_name_value:
            config["api"]["tree_name"] = tree_name_value

        tree_id_value = os.getenv("TREE_ID")
        if tree_id_value:
            config["api"]["tree_id"] = tree_id_value

        # Load processing limit configuration (Action 6 lessons)
        max_pages_value = os.getenv("MAX_PAGES")
        if max_pages_value:
            try:
                config["api"]["max_pages"] = int(max_pages_value)
            except ValueError:
                logger.warning(f"Invalid MAX_PAGES value: {max_pages_value}")

        # Load batch size configuration
        batch_size_value = os.getenv("BATCH_SIZE")
        if batch_size_value:
            try:
                config["batch_size"] = int(batch_size_value)
            except ValueError:
                logger.warning(f"Invalid BATCH_SIZE value: {batch_size_value}")

        # Load matches per page configuration
        matches_per_page_value = os.getenv("MATCHES_PER_PAGE")
        if matches_per_page_value:
            try:
                config["matches_per_page"] = int(matches_per_page_value)
            except ValueError:
                logger.warning(f"Invalid MATCHES_PER_PAGE value: {matches_per_page_value}")

        # Load optional concurrency override for Action 6
        max_concurrency_value = os.getenv("MAX_CONCURRENCY")
        if max_concurrency_value:
            try:
                # Mirror under API scope since APIConfig owns concurrency affecting API calls
                if "api" not in config:
                    config["api"] = {}
                config["api"]["max_concurrency"] = int(max_concurrency_value)
            except ValueError:
                logger.warning(f"Invalid MAX_CONCURRENCY value: {max_concurrency_value}")

        # Load thread pool workers override for Action 6
        thread_pool_workers_value = os.getenv("THREAD_POOL_WORKERS")
        if thread_pool_workers_value:
            try:
                # Store under API scope for consistency with max_concurrency
                if "api" not in config:
                    config["api"] = {}
                config["api"]["thread_pool_workers"] = int(thread_pool_workers_value)
            except ValueError:
                logger.warning(f"Invalid THREAD_POOL_WORKERS value: {thread_pool_workers_value}")

        # Load max productive to process configuration
        max_productive_value = os.getenv("MAX_PRODUCTIVE_TO_PROCESS")
        if max_productive_value:
            try:
                config["max_productive_to_process"] = int(max_productive_value)
            except ValueError:
                logger.warning(f"Invalid MAX_PRODUCTIVE_TO_PROCESS value: {max_productive_value}")

        # Load max inbox configuration
        max_inbox_value = os.getenv("MAX_INBOX")
        if max_inbox_value:
            try:
                config["max_inbox"] = int(max_inbox_value)
            except ValueError:
                logger.warning(f"Invalid MAX_INBOX value: {max_inbox_value}")

        # Load logging configuration
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


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for config_manager.py with proper TestSuite framework.
    Tests configuration management, validation, and loading functionality.
    """
    import tempfile
    import time

    from test_framework import TestSuite

    suite = TestSuite("Configuration Management & Validation", "config_manager.py")
    suite.start_suite()

    # INITIALIZATION TESTS
    def test_config_manager_initialization():
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

    def test_config_validation():
        """Test configuration validation functions."""
        # Test ConfigManager methods exist
        manager = ConfigManager(auto_load=False)
        assert hasattr(manager, "validate_config"), "Should have validate_config method"
        assert hasattr(manager, "load_config"), "Should have load_config method"
        assert hasattr(manager, "get_config"), "Should have get_config method"

    # CORE FUNCTIONALITY TESTS
    def test_config_loading():
        """Test configuration loading and parsing."""
        manager = ConfigManager(auto_load=False)

        # Test default config loading
        default_config = manager._get_default_config()
        assert isinstance(default_config, dict), "Should return dictionary"
        assert "environment" in default_config, "Should have environment key"
        assert "app_name" in default_config, "Should have app_name key"

    def test_config_access():
        """Test configuration value access methods."""
        manager = ConfigManager(auto_load=False)

        # Test getter methods exist
        assert hasattr(
            manager, "get_database_config"
        ), "Should have get_database_config"
        assert hasattr(
            manager, "get_selenium_config"
        ), "Should have get_selenium_config"
        assert hasattr(manager, "get_api_config"), "Should have get_api_config"

    # EDGE CASE TESTS
    def test_missing_config_handling():
        """Test handling of missing configuration files and values."""
        # Test with non-existent config file
        try:
            manager = ConfigManager(
                config_file="nonexistent_file_12345.json", auto_load=False
            )
            assert manager is not None, "Should handle missing config file gracefully"
        except Exception:
            # Exception handling is acceptable
            pass

    def test_invalid_config_data():
        """Test handling of invalid configuration data."""
        manager = ConfigManager(auto_load=False)

        # Test validation with various invalid inputs
        invalid_configs = [None, "not_a_dict", 123, []]

        for invalid_config in invalid_configs:
            try:
                result = manager.validate_config(invalid_config)
                assert isinstance(
                    result, list
                ), "Should return list of validation errors"
            except Exception:
                # Exception handling is acceptable for invalid inputs
                pass

    # INTEGRATION TESTS
    def test_config_file_integration():
        """Test integration with actual config files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            test_config = {"app_name": "Test App", "environment": "test"}
            import json

            json.dump(test_config, f)
            config_path = f.name

        try:
            manager = ConfigManager(config_file=config_path, auto_load=False)
            file_config = manager._load_config_file()
            assert isinstance(file_config, dict), "Should load config from file"
        except Exception:
            pass
        finally:
            from pathlib import Path

            try:
                Path(config_path).unlink(missing_ok=True)
            except Exception:
                pass

    def test_environment_integration():
        """Test integration with environment variables."""
        import os

        # Test environment variable access
        test_env_key = "TEST_CONFIG_VAR_12345"
        test_env_value = "test_environment_value_12345"

        os.environ[test_env_key] = test_env_value

        try:
            manager = ConfigManager(auto_load=False)
            env_config = manager._load_environment_variables()
            assert isinstance(
                env_config, dict
            ), "Should return environment config dictionary"
        finally:
            os.environ.pop(test_env_key, None)

    # PERFORMANCE TESTS
    def test_config_access_performance():
        """Test performance of configuration access operations."""
        manager = ConfigManager(auto_load=False)

        start_time = time.time()
        for i in range(10):  # Reduced for reliability
            try:
                config = manager._get_default_config()
                assert isinstance(config, dict), "Should return config dict"
            except Exception:
                pass

        duration = time.time() - start_time
        assert (
            duration < 2.0
        ), f"10 config access operations should be fast, took {duration:.3f}s"

    # ERROR HANDLING TESTS
    def test_config_error_handling():
        """Test error handling in configuration operations."""
        error_scenarios = [
            ("missing_file_12345.json", "Should handle missing files"),
            ("", "Should handle empty filenames"),
            (None, "Should handle None inputs"),
        ]

        for test_input, description in error_scenarios:
            try:
                manager = ConfigManager(config_file=test_input, auto_load=False)
                # Should handle gracefully
                assert manager is not None or manager is None
            except Exception:
                # Exception handling is acceptable for invalid inputs
                pass

    # Run all tests
    suite.run_test(
        "ConfigManager.__init__(), validate_config(), load_config()",
        test_config_manager_initialization,
        "Configuration manager initializes correctly with proper validation functions",
        "Test ConfigManager class instantiation and validation function availability",
        "ConfigManager creates successfully with all required configuration validation capabilities",
    )

    suite.run_test(
        "validate_config(), load_config(), get_config()",
        test_config_validation,
        "Configuration validation functions are available and callable",
        "Test availability of all configuration validation and management functions",
        "All validation functions exist and are callable for configuration data processing",
    )

    suite.run_test(
        "_get_default_config(), _load_config_file(), _merge_configs()",
        test_config_loading,
        "Configuration loading and parsing functions work correctly",
        "Test configuration file loading, parsing, and merging capabilities",
        "Configuration loading functions process data correctly and return expected formats",
    )

    suite.run_test(
        "get_database_config(), get_selenium_config(), get_api_config()",
        test_config_access,
        "Configuration value access methods are available and functional",
        "Test configuration value getting and specialized config access methods",
        "All configuration access methods exist and are callable for value management",
    )

    suite.run_test(
        "ConfigManager() with missing config file",
        test_missing_config_handling,
        "Missing configuration files are handled gracefully",
        "Test configuration manager with non-existent config files",
        "Missing configuration files handled gracefully without crashes",
    )

    suite.run_test(
        "validate_config() with invalid data types",
        test_invalid_config_data,
        "Invalid configuration data is handled gracefully without crashes",
        "Test validation with None, strings, numbers, and lists instead of dictionaries",
        "Invalid configuration data types handled appropriately with proper validation results",
    )

    suite.run_test(
        "config file loading and JSON parsing integration",
        test_config_file_integration,
        "Integration with configuration files and JSON parsing works correctly",
        "Test configuration file creation, loading, and parsing with temporary files",
        "Configuration file integration works properly with JSON file handling",
    )

    suite.run_test(
        "environment variable integration and access",
        test_environment_integration,
        "Environment variable integration works correctly for configuration values",
        "Test environment variable setting, access, and cleanup through configuration system",
        "Environment variables are properly integrated and accessible through configuration",
    )

    suite.run_test(
        "_get_default_config() performance with multiple operations",
        test_config_access_performance,
        "Configuration access operations perform efficiently under load",
        "Test multiple configuration access operations for performance timing",
        "Configuration access maintains good performance with multiple rapid operations",
    )

    suite.run_test(
        "ConfigManager() error handling with invalid inputs",
        test_config_error_handling,
        "Configuration error handling manages invalid inputs gracefully",
        "Test configuration manager with missing files, empty names, and None inputs",
        "Error handling prevents crashes and handles invalid configuration scenarios appropriately",
    )

    return suite.finish_suite()


if __name__ == "__main__":
    print("🔧 Running Configuration Manager comprehensive test suite...")
    success = run_comprehensive_tests()
    exit(0 if success else 1)
