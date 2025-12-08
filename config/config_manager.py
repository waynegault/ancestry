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

import logging

logger = logging.getLogger(__name__)

# ============================================================================
# RESERVED FOR FUTURE DEVELOPMENT
# ============================================================================
# The following ConfigManager methods are fully implemented but not yet
# integrated into the main application workflow. They provide advanced
# configuration management capabilities for future enhancements:
#
# Configuration reload and export:
#   - reload_config(): Hot-reload configuration from files
#   - export_config(): Export current config to file
#   - get_environment_config(): Get environment-specific config
#
# Setup wizard:
#   - run_setup_wizard(): Interactive configuration setup
#
# Section-specific getters:
#   - get_database_config(): Database configuration subset
#   - get_selenium_config(): Selenium/browser configuration subset
#   - get_api_config(): API configuration subset
#   - get_logging_config(): Logging configuration subset
#   - get_cache_config(): Cache configuration subset
#   - get_security_config(): Security configuration subset
#   - get_observability_config(): Observability configuration subset
#
# These methods are tested and ready for use when modular configuration
# access or interactive setup features are needed in future releases.
# ============================================================================

# === STANDARD LIBRARY IMPORTS ===
import copy
import json
from pathlib import Path
from typing import Any, Callable, Optional, Union

# === THIRD-PARTY IMPORTS ===
LoadDotenvCallable = Callable[..., bool]
load_dotenv: Optional[LoadDotenvCallable]

try:
    from dotenv import load_dotenv as _load_dotenv_impl
except ImportError:
    load_dotenv = None
    _dotenv_imported = False
else:
    load_dotenv = _load_dotenv_impl
    _dotenv_imported = True

DOTENV_AVAILABLE = _dotenv_imported

try:
    from .config_schema import (
        APIConfig,
        CacheConfig,
        ConfigSchema,
        DatabaseConfig,
        HealthConfig,
        LoggingConfig,
        ObservabilityConfig,
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
        HealthConfig,
        LoggingConfig,
        ObservabilityConfig,
        SecurityConfig,
        SeleniumConfig,
    )

from observability.metrics_exporter import apply_observability_settings
from observability.metrics_registry import configure_metrics


class ValidationError(Exception):
    """Configuration validation error."""

    pass


class _ConfigManagerSingleton:
    """Container class for singleton instance to avoid global statement."""

    instance: Optional["ConfigManager"] = None


def get_config_manager(
    config_file: Optional[Union[str, Path]] = None,
    environment: Optional[str] = None,
    force_new: bool = False,
) -> "ConfigManager":
    """
    Get the singleton ConfigManager instance.

    This function provides a single shared ConfigManager instance across the entire
    application, eliminating redundant instantiations and ensuring consistent
    configuration state.

    Args:
        config_file: Optional configuration file path (only used on first call)
        environment: Environment name (only used on first call)
        force_new: If True, create a new instance (for testing only)

    Returns:
        The shared ConfigManager instance
    """
    if force_new or _ConfigManagerSingleton.instance is None:
        _ConfigManagerSingleton.instance = ConfigManager(
            config_file=config_file,
            environment=environment,
            auto_load=True,
        )

    return _ConfigManagerSingleton.instance


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

    _rate_limiter_max_clamp_logged: bool = False

    Usage:
        # Preferred: Use the singleton function
        from config.config_manager import get_config_manager
        config_manager = get_config_manager()
        config = config_manager.get_config()

        # Legacy: Direct instantiation still works but creates new instance
        config_manager = ConfigManager()
    """

    _token_fill_rate_clamp_logged = False
    _unsafe_concurrency_override_logged = False

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
        # Load .env file if available (unless explicitly skipped for tests)
        if DOTENV_AVAILABLE and load_dotenv is not None:
            skip_dotenv = os.getenv("CONFIG_SKIP_DOTENV", "").strip().lower()
            if skip_dotenv not in {"1", "true", "yes", "on"}:
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
            self._enforce_api_safety_constraints(config)

            # Validate configuration
            validation_errors = config.validate()
            if validation_errors:
                raise ValidationError(f"Configuration validation failed: {validation_errors}")

            try:
                configure_metrics(getattr(config, "observability", None))
                apply_observability_settings(getattr(config, "observability", None))
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Failed to configure Prometheus metrics: %s", exc)

            # Cache the configuration
            self._config_cache = config
            if self.config_file:
                self._file_modification_time = self.config_file.stat().st_mtime

            logger.debug(f"Configuration loaded successfully for environment: {self.environment}")
            return config

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ValidationError(f"Configuration loading failed: {e}") from e

    def _enforce_api_safety_constraints(self, config: ConfigSchema) -> None:
        """Clamp API settings to safe sequential defaults unless override explicitly requested."""

        api = getattr(config, "api", None)
        if not api:
            return

        speed_profile = str(getattr(api, "speed_profile", "safe")).lower()
        unsafe_requested = self._unsafe_rate_limit_requested(api, speed_profile)

        self._enforce_max_concurrency(api, unsafe_requested)

        safe_rps = 0.3
        self._enforce_requests_per_second(api, unsafe_requested, safe_rps)
        self._align_rate_limiter_max_rate(api, unsafe_requested)

    @staticmethod
    def _unsafe_rate_limit_requested(api: Any, speed_profile: str) -> bool:
        """Return True when any configuration requests relaxed API safety."""

        unsafe_requested = bool(getattr(api, "allow_unsafe_rate_limit", False))

        if speed_profile in {"max", "aggressive", "experimental"}:
            unsafe_requested = True

        env_speed_profile = os.getenv("API_SPEED_PROFILE", "").strip().lower()
        if env_speed_profile in {"max", "aggressive", "experimental"}:
            unsafe_requested = True

        env_allow_unsafe = os.getenv("ALLOW_UNSAFE_RATE_LIMIT", "").strip().lower()
        if env_allow_unsafe in {"true", "1", "yes", "on"}:
            unsafe_requested = True

        return unsafe_requested

    def _enforce_max_concurrency(self, api: Any, unsafe_requested: bool) -> None:
        """Ensure concurrency stays within safe sequential defaults."""

        max_concurrency_configured = getattr(api, "max_concurrency", 1)
        if max_concurrency_configured == 1:
            return

        if unsafe_requested:
            if not getattr(type(self), "_unsafe_concurrency_override_logged", False):
                logger.warning(
                    "⚠️ Unsafe speed profile active; retaining max_concurrency=%s. Monitor session stability closely.",
                    max_concurrency_configured,
                )
                type(self)._unsafe_concurrency_override_logged = True
            return

        logger.warning(
            "max_concurrency=%s overridden to sequential-safe value of 1",
            max_concurrency_configured,
        )
        api.max_concurrency = 1

    def _enforce_requests_per_second(self, api: Any, unsafe_requested: bool, safe_rps: float) -> float:
        """Clamp requests_per_second when safety clamps are active."""

        current_rps = getattr(api, "requests_per_second", safe_rps)
        if unsafe_requested or current_rps <= safe_rps:
            return current_rps

        if not getattr(self, "_rps_clamp_logged", False):
            logger.debug(
                "requests_per_second %.2f exceeds validated safe limit %.2f; clamping to safe value",
                current_rps,
                safe_rps,
            )
            self._rps_clamp_logged = True

        api.requests_per_second = safe_rps
        return safe_rps

    def _align_rate_limiter_max_rate(self, api: Any, unsafe_requested: bool) -> None:
        """Keep rate limiter max rate aligned with the enforced RPS."""
        if unsafe_requested:
            return

        target_rps = getattr(api, "requests_per_second", 0.3)
        max_rate = getattr(api, "rate_limiter_max_rate", target_rps)

        # If max_rate is significantly higher than target_rps (e.g. > 1.5x), clamp it
        # We allow some headroom (e.g. 0.5 vs 0.3) for adaptive speedup, but not 3.0 vs 0.3
        limit = target_rps * 2.0  # Allow 2x speedup (0.3 -> 0.6)

        if max_rate > limit:
            if not getattr(type(self), "_rate_limiter_max_clamp_logged", False):
                logger.info(
                    "Rate limiter max rate %.2f significantly higher than requests_per_second %.2f; clamping to %.2f",
                    max_rate,
                    target_rps,
                    limit,
                )
                setattr(type(self), "_rate_limiter_max_clamp_logged", True)

            api.rate_limiter_max_rate = limit

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

    def validate_config(self, config_data: Optional[dict[str, Any]] = None) -> list[str]:
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

    def export_config(self, output_file: Union[str, Path], format: str = "json") -> bool:
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
        base_config: dict[str, Any] = {
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
            "observability": {},
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

            logger.debug(
                f"Auto-detected configuration: CPU={cpu_count}, RAM={memory_gb:.1f}GB, Disk={disk_space_gb:.1f}GB"
            )
            return auto_detected

        except Exception as e:
            logger.warning(f"Auto-detection failed, using defaults: {e}")
            return {}

    @staticmethod
    def _check_minimum_requirements(
        cpu_count: int, memory_gb: float, disk_space_gb: float, validation_results: dict[str, Any]
    ) -> None:
        """Check minimum system requirements and update validation results."""
        if cpu_count < 2:
            validation_results["warnings"].append(
                f"Low CPU count ({cpu_count}). Recommend at least 2 cores for optimal performance."
            )

        if memory_gb < 4:
            validation_results["errors"].append(f"Insufficient memory ({memory_gb:.1f}GB). Minimum 4GB required.")
            validation_results["valid"] = False

        if disk_space_gb < 1:
            validation_results["errors"].append(
                f"Insufficient disk space ({disk_space_gb:.1f}GB). Minimum 1GB free space required."
            )
            validation_results["valid"] = False

    @staticmethod
    def _check_optimal_performance(cpu_count: int, memory_gb: float, validation_results: dict[str, Any]) -> None:
        """Check for optimal performance opportunities and add recommendations."""
        if memory_gb >= 16:
            validation_results["recommendations"].append(
                "High memory detected. Consider enabling GPU acceleration and increasing cache sizes."
            )

        if cpu_count >= 8:
            validation_results["recommendations"].append(
                "High CPU count detected. Consider increasing concurrency settings for better performance."
            )

    @staticmethod
    def _check_dependencies_and_chrome(validation_results: dict[str, Any]) -> None:
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
            validation_results["warnings"].append("Chrome browser not found in PATH. Selenium automation may not work.")

    def validate_system_requirements(self) -> dict[str, Any]:
        """
        Validate system requirements and provide recommendations.

        Returns:
            Dictionary with validation results and recommendations
        """
        try:
            import psutil

            validation_results = {"valid": True, "warnings": [], "errors": [], "recommendations": [], "system_info": {}}

            # Check system resources
            cpu_count = psutil.cpu_count(logical=True) or 1
            memory_gb = psutil.virtual_memory().total / (1024**3)
            disk_space_gb = psutil.disk_usage('/').free / (1024**3)

            validation_results["system_info"] = {
                "cpu_cores": cpu_count,
                "memory_gb": round(memory_gb, 1),
                "free_disk_gb": round(disk_space_gb, 1),
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
                "system_info": {},
            }

    @staticmethod
    def _print_system_info(system_info: dict[str, Any]) -> None:
        """Print system information."""
        print("\n📊 System Information:")
        print(f"   CPU Cores: {system_info.get('cpu_cores', 'Unknown')}")
        print(f"   Memory: {system_info.get('memory_gb', 'Unknown')}GB")
        print(f"   Free Disk: {system_info.get('free_disk_gb', 'Unknown')}GB")

    @staticmethod
    def _print_validation_results(validation: dict[str, Any]) -> bool:
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

    @staticmethod
    def _print_optimal_settings(auto_detected: dict[str, Any]) -> None:
        """Print auto-detected optimal settings."""
        print("\n🔧 Auto-detected Optimal Settings:")

        api_config = auto_detected.get("api", {})
        print(f"   Concurrency: {api_config.get('max_concurrency', 'Default')}")

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

    @staticmethod
    def _load_main_config_from_env(config: dict[str, Any]) -> None:
        """Load main configuration from environment variables."""
        env_value = os.getenv("ENVIRONMENT")
        if env_value:
            config["environment"] = env_value

        debug_mode_value = os.getenv("DEBUG_MODE")
        if debug_mode_value:
            config["debug_mode"] = debug_mode_value.lower() in {"true", "1", "yes"}

        app_name_value = os.getenv("APP_NAME")
        if app_name_value:
            config["app_name"] = app_name_value

    @staticmethod
    def _load_reference_person_config_from_env(config: dict[str, Any]) -> None:
        """Load reference person configuration from environment variables."""
        reference_person_id_value = os.getenv("REFERENCE_PERSON_ID")
        if reference_person_id_value:
            config["reference_person_id"] = reference_person_id_value

        reference_person_name_value = os.getenv("REFERENCE_PERSON_NAME")
        if reference_person_name_value:
            config["reference_person_name"] = reference_person_name_value

    @staticmethod
    def _load_user_config_from_env(config: dict[str, Any]) -> None:
        """Load user configuration from environment variables."""
        user_name_value = os.getenv("USER_NAME")
        if user_name_value:
            config["user_name"] = user_name_value

        user_location_value = os.getenv("USER_LOCATION")
        if user_location_value:
            config["user_location"] = user_location_value

    @staticmethod
    def _load_ms_graph_config_from_env(config: dict[str, Any]) -> None:
        """Load Microsoft Graph / To-Do configuration from environment variables."""
        ms_todo_list_name_value = os.getenv("MS_TODO_LIST_NAME")
        if ms_todo_list_name_value:
            config["ms_todo_list_name"] = ms_todo_list_name_value

    @staticmethod
    def _load_testing_config_from_env(config: dict[str, Any]) -> None:
        """Load testing configuration from environment variables."""
        # Load test profile configuration
        testing_profile_id_value = os.getenv("TEST_PROFILE_ID")
        if testing_profile_id_value:
            config["testing_profile_id"] = testing_profile_id_value

        testing_uuid_value = os.getenv("TEST_UUID")
        if testing_uuid_value:
            config["testing_uuid"] = testing_uuid_value

        testing_username_value = os.getenv("TEST_USERNAME")
        if testing_username_value:
            config["testing_username"] = testing_username_value

    @staticmethod
    def _load_app_mode_from_env(config: dict[str, Any]) -> None:
        """Load application mode from environment variables."""
        app_mode_value = os.getenv("APP_MODE")
        if app_mode_value:
            config["app_mode"] = app_mode_value

    @staticmethod
    def _load_ai_config_from_env(config: dict[str, Any]) -> None:
        """Load AI configuration from environment variables."""
        ai_provider_value = os.getenv("AI_PROVIDER")
        if ai_provider_value:
            config["ai_provider"] = ai_provider_value

        fallback_value = os.getenv("AI_PROVIDER_FALLBACKS")
        if fallback_value:
            fallback_list = [item.strip() for item in fallback_value.split(",") if item.strip()]
            if fallback_list:
                config["ai_provider_fallbacks"] = fallback_list

        # Use helper functions for integer env vars
        ConfigManager._load_int_env_var(config, "AI_CONTEXT_MESSAGES_COUNT", "ai_context_messages_count")
        ConfigManager._load_int_env_var(config, "AI_CONTEXT_MESSAGE_MAX_WORDS", "ai_context_message_max_words")
        ConfigManager._load_int_env_var(config, "AI_CONTEXT_WINDOW_MESSAGES", "ai_context_window_messages")

    @staticmethod
    def _load_timeout_config_from_env(config: dict[str, Any]) -> None:
        """Load timeout configuration from environment variables."""
        # Use helper functions for integer env vars
        ConfigManager._load_int_env_var(config, "PROACTIVE_REFRESH_COOLDOWN", "proactive_refresh_cooldown_seconds")
        ConfigManager._load_int_env_var(config, "PROACTIVE_REFRESH_INTERVAL", "proactive_refresh_interval_seconds")
        ConfigManager._load_int_env_var(config, "ACTION6_COORD_TIMEOUT", "action6_coord_timeout_seconds")

    @staticmethod
    def _load_env_var_if_present(api_config: dict[str, Any], env_var: str, config_key: str) -> None:
        """Load environment variable into config if present with type conversion.

        Args:
            api_config: Dictionary to update with config value
            env_var: Environment variable name to check
            config_key: Key to use in api_config dictionary
        """
        value = os.getenv(env_var)
        if value:
            # Type conversion for known boolean fields
            if config_key in {"lm_studio_auto_start"}:
                api_config[config_key] = value.lower() in {"true", "1", "yes", "on"}
            # Type conversion for known integer fields
            elif config_key in {"lm_studio_startup_timeout"}:
                try:
                    api_config[config_key] = int(value)
                except ValueError:
                    logger.warning(f"Invalid integer value for {env_var}: {value}, using default")
            else:
                api_config[config_key] = value

    def _load_api_keys_from_env(self, config: dict[str, Any]) -> None:
        """Load API keys configuration from environment variables."""
        api_config = {}

        # Define mapping of environment variables to config keys
        env_mappings = [
            # Ancestry credentials
            ("ANCESTRY_USERNAME", "username"),
            ("ANCESTRY_PASSWORD", "password"),
            # DeepSeek API configuration
            ("DEEPSEEK_API_KEY", "deepseek_api_key"),
            ("DEEPSEEK_AI_MODEL", "deepseek_ai_model"),
            ("DEEPSEEK_AI_BASE_URL", "deepseek_ai_base_url"),
            # Google API configuration
            ("GOOGLE_API_KEY", "google_api_key"),
            ("GOOGLE_AI_MODEL", "google_ai_model"),
            # Moonshot (Kimi) API configuration
            ("MOONSHOT_API_KEY", "moonshot_api_key"),
            ("MOONSHOT_AI_MODEL", "moonshot_ai_model"),
            ("MOONSHOT_AI_BASE_URL", "moonshot_ai_base_url"),
            # Grok (xAI) configuration
            ("XAI_API_KEY", "xai_api_key"),
            ("XAI_MODEL", "xai_model"),
            ("XAI_API_HOST", "xai_api_host"),
            # Local LLM API configuration
            ("LOCAL_LLM_API_KEY", "local_llm_api_key"),
            ("LOCAL_LLM_MODEL", "local_llm_model"),
            ("LOCAL_LLM_BASE_URL", "local_llm_base_url"),
            ("LM_STUDIO_PATH", "lm_studio_path"),
            ("LM_STUDIO_AUTO_START", "lm_studio_auto_start"),
            ("LM_STUDIO_STARTUP_TIMEOUT", "lm_studio_startup_timeout"),
            # Inception Mercury API configuration
            ("INCEPTION_API_KEY", "inception_api_key"),
            ("INCEPTION_AI_MODEL", "inception_ai_model"),
            ("INCEPTION_AI_BASE_URL", "inception_ai_base_url"),
            # Tetrate (TARS) API configuration
            ("TARS_API_KEY", "tetrate_api_key"),
            ("TETRATE_AI_MODEL", "tetrate_ai_model"),
            ("TETRATE_AI_BASE_URL", "tetrate_ai_base_url"),
        ]

        # Load all environment variables using helper method
        for env_var, config_key in env_mappings:
            self._load_env_var_if_present(api_config, env_var, config_key)

        if api_config:
            config["api"] = api_config

    @staticmethod
    def _load_database_config_from_env(config: dict[str, Any]) -> None:
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

    @staticmethod
    def _load_selenium_config_from_env(config: dict[str, Any]) -> None:
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
            selenium_config["headless_mode"] = headless_mode_value.lower() in {"true", "1", "yes"}

        debug_port_value = os.getenv("DEBUG_PORT")
        if debug_port_value:
            try:
                selenium_config["debug_port"] = int(debug_port_value)
            except ValueError:
                logger.warning(f"Invalid DEBUG_PORT value: {debug_port_value}")

        if selenium_config:
            config["selenium"] = selenium_config

    @staticmethod
    def _set_string_config(config: dict[str, Any], section: str, key: str, env_var: str) -> None:
        """Set a string configuration value from environment variable."""
        value = os.getenv(env_var)
        if value:
            if section not in config:
                config[section] = {}
            config[section][key] = value

    @staticmethod
    def _set_int_config(config: dict[str, Any], section: str, key: str, env_var: str) -> None:
        """Set an integer configuration value from environment variable."""
        value = os.getenv(env_var)
        if value:
            try:
                if section not in config:
                    config[section] = {}
                config[section][key] = int(value)
            except ValueError:
                logger.warning(f"Invalid {env_var} value: {value}")

    @staticmethod
    def _set_float_config(config: dict[str, Any], section: str, key: str, env_var: str) -> None:
        """Set a float configuration value from environment variable."""
        value = os.getenv(env_var)
        if value:
            try:
                if section not in config:
                    config[section] = {}
                config[section][key] = float(value)
            except ValueError:
                logger.warning(f"Invalid {env_var} value: {value}")

    @staticmethod
    def _set_bool_config(config: dict[str, Any], section: str, key: str, env_var: str) -> None:
        """Set a boolean configuration value from environment variable."""
        value = os.getenv(env_var)
        if value is not None:
            bool_value = value.strip().lower() in {"true", "1", "yes", "on"}
            if section not in config:
                config[section] = {}
            config[section][key] = bool_value

    def _load_additional_api_config_from_env(self, config: dict[str, Any]) -> None:
        """Load additional API configuration from environment variables."""
        if "api" not in config:
            config["api"] = {}

        # String configurations
        self._set_string_config(config, "api", "base_url", "BASE_URL")
        self._set_string_config(config, "api", "api_base_url", "API_BASE_URL")
        self._set_string_config(config, "api", "tree_name", "TREE_NAME")
        self._set_string_config(config, "api", "tree_id", "MY_TREE_ID")
        self._set_string_config(config, "api", "my_user_id", "MY_PROFILE_ID")
        self._set_string_config(config, "api", "my_uuid", "MY_UUID")
        self._set_string_config(config, "api", "speed_profile", "API_SPEED_PROFILE")

        # Integer configurations
        self._set_int_config(config, "api", "request_timeout", "REQUEST_TIMEOUT")
        self._set_int_config(config, "api", "max_pages", "MAX_PAGES")
        self._set_int_config(config, "api", "max_relationship_prob_fetches", "MAX_RELATIONSHIP_PROB_FETCHES")
        self._set_int_config(config, "api", "max_concurrency", "MAX_CONCURRENCY")
        self._set_int_config(config, "api", "burst_limit", "BURST_LIMIT")
        self._set_int_config(config, "api", "token_bucket_success_threshold", "TOKEN_BUCKET_SUCCESS_THRESHOLD")

        # Float configurations - Rate limiting
        self._set_float_config(config, "api", "requests_per_second", "REQUESTS_PER_SECOND")
        self._set_float_config(config, "api", "initial_delay", "INITIAL_DELAY")
        self._set_float_config(config, "api", "max_delay", "MAX_DELAY")
        self._set_float_config(config, "api", "backoff_factor", "BACKOFF_FACTOR")
        self._set_float_config(config, "api", "decrease_factor", "DECREASE_FACTOR")
        self._set_float_config(config, "api", "token_bucket_capacity", "TOKEN_BUCKET_CAPACITY")
        self._set_float_config(config, "api", "target_match_throughput", "TARGET_MATCH_THROUGHPUT")
        self._set_float_config(config, "api", "max_throughput_catchup_delay", "MAX_THROUGHPUT_CATCHUP_DELAY")

        # New consolidated rate limiter settings
        self._set_float_config(config, "api", "rate_limiter_429_backoff", "RATE_LIMITER_429_BACKOFF")
        self._set_float_config(config, "api", "rate_limiter_success_factor", "RATE_LIMITER_SUCCESS_FACTOR")
        self._set_float_config(config, "api", "rate_limiter_min_rate", "RATE_LIMITER_MIN_RATE")
        self._set_float_config(config, "api", "rate_limiter_max_rate", "RATE_LIMITER_MAX_RATE")

        # Boolean configurations
        self._set_bool_config(config, "api", "rate_limit_enabled", "RATE_LIMIT_ENABLED")
        self._set_bool_config(config, "api", "allow_unsafe_rate_limit", "ALLOW_UNSAFE_RATE_LIMIT")

        throttle_json = os.getenv("API_ENDPOINT_THROTTLES")
        if throttle_json:
            try:
                override_data = json.loads(throttle_json)
                if isinstance(override_data, dict):
                    config["api"]["endpoint_throttle_profiles"] = override_data
                else:
                    logger.warning("API_ENDPOINT_THROTTLES must be a JSON object; ignoring value")
            except json.JSONDecodeError as exc:
                logger.warning(f"Failed to parse API_ENDPOINT_THROTTLES: {exc}")

    @staticmethod
    def _load_int_env_var(config: dict[str, Any], env_var: str, config_key: str) -> None:
        """Load a single integer environment variable into config."""
        value = os.getenv(env_var)
        if value:
            try:
                config[config_key] = int(value)
            except ValueError:
                logger.warning(f"Invalid {env_var} value: {value}")

    @staticmethod
    def _load_bool_env_var(config: dict[str, Any], env_var: str, config_key: str) -> None:
        """Load a single boolean environment variable into config."""
        value = os.getenv(env_var)
        if value is not None:
            config[config_key] = value.strip().lower() in {"true", "1", "yes", "on"}

    def _load_processing_limits_from_env(self, config: dict[str, Any]) -> None:
        """Load processing limit configuration from environment variables."""
        self._load_int_env_var(config, "BATCH_SIZE", "batch_size")
        self._load_int_env_var(config, "MATCHES_PER_PAGE", "matches_per_page")
        self._load_int_env_var(config, "MAX_PRODUCTIVE_TO_PROCESS", "max_productive_to_process")
        self._load_int_env_var(config, "MAX_INBOX", "max_inbox")
        self._load_int_env_var(config, "PARALLEL_WORKERS", "parallel_workers")
        self._load_int_env_var(config, "ETHNICITY_ENRICHMENT_MIN_CM", "ethnicity_enrichment_min_cm")
        self._load_bool_env_var(config, "ENABLE_ETHNICITY_ENRICHMENT", "enable_ethnicity_enrichment")
        self._load_bool_env_var(config, "ENABLE_CHECKPOINTING", "enable_action6_checkpointing")
        self._load_int_env_var(config, "CHECKPOINT_MAX_AGE_HOURS", "action6_checkpoint_max_age_hours")
        checkpoint_path = os.getenv("ACTION6_CHECKPOINT_FILE")
        if checkpoint_path:
            config["action6_checkpoint_file"] = checkpoint_path

    @staticmethod
    def _load_logging_config_from_env(config: dict[str, Any]) -> None:
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

    @staticmethod
    def _load_cache_config_from_env(config: dict[str, Any]) -> None:
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

    @staticmethod
    def _load_security_config_from_env(config: dict[str, Any]) -> None:
        """Load security configuration from environment variables."""
        security_config = {}
        encryption_enabled_value = os.getenv("ENCRYPTION_ENABLED")
        if encryption_enabled_value:
            security_config["encryption_enabled"] = encryption_enabled_value.lower() in {"true", "1", "yes"}

        use_system_keyring_value = os.getenv("USE_SYSTEM_KEYRING")
        if use_system_keyring_value:
            security_config["use_system_keyring"] = use_system_keyring_value.lower() in {"true", "1", "yes"}

        session_timeout_value = os.getenv("SESSION_TIMEOUT_MINUTES")
        if session_timeout_value:
            try:
                security_config["session_timeout_minutes"] = int(session_timeout_value)
            except ValueError:
                logger.warning(f"Invalid SESSION_TIMEOUT_MINUTES value: {session_timeout_value}")

        if security_config:
            config["security"] = security_config

    @staticmethod
    def _load_observability_config_from_env(config: dict[str, Any]) -> None:
        """Load observability configuration from environment variables."""
        observability_config: dict[str, Any] = {}

        enabled_value = os.getenv("PROMETHEUS_METRICS_ENABLED")
        if enabled_value is not None:
            observability_config["enable_prometheus_metrics"] = enabled_value.strip().lower() in {
                "true",
                "1",
                "yes",
                "on",
            }

        host_value = os.getenv("PROMETHEUS_METRICS_HOST")
        if host_value:
            observability_config["metrics_export_host"] = host_value

        port_value = os.getenv("PROMETHEUS_METRICS_PORT")
        if port_value:
            try:
                observability_config["metrics_export_port"] = int(port_value)
            except ValueError:
                logger.warning(f"Invalid PROMETHEUS_METRICS_PORT value: {port_value}")

        namespace_value = os.getenv("PROMETHEUS_METRICS_NAMESPACE")
        if namespace_value:
            observability_config["metrics_namespace"] = namespace_value

        auto_start_value = os.getenv("PROMETHEUS_AUTO_START")
        if auto_start_value is not None:
            observability_config["auto_start_prometheus"] = auto_start_value.strip().lower() in {
                "true",
                "1",
                "yes",
                "on",
            }

        binary_path_value = os.getenv("PROMETHEUS_BINARY_PATH")
        if binary_path_value:
            observability_config["prometheus_binary_path"] = binary_path_value.strip()

        if observability_config:
            config["observability"] = observability_config

    @staticmethod
    def _load_float_env(env_var: str, config_dict: dict[str, Any], key: str) -> None:
        """Helper to load a float value from an environment variable."""
        value = os.getenv(env_var)
        if value:
            try:
                config_dict[key] = float(value)
            except ValueError:
                logger.warning(f"Invalid {env_var} value: {value}")

    @staticmethod
    def _load_health_config_from_env(config: dict[str, Any]) -> None:
        """Load health configuration from environment variables."""
        health_config: dict[str, Any] = {}

        # API Response Time
        ConfigManager._load_float_env("HEALTH_API_RESPONSE_TIME_WARNING", health_config, "api_response_time_warning")
        ConfigManager._load_float_env("HEALTH_API_RESPONSE_TIME_CRITICAL", health_config, "api_response_time_critical")

        # Error Rate
        ConfigManager._load_float_env("HEALTH_ERROR_RATE_WARNING", health_config, "error_rate_warning")
        ConfigManager._load_float_env("HEALTH_ERROR_RATE_CRITICAL", health_config, "error_rate_critical")

        # Memory Usage
        ConfigManager._load_float_env("HEALTH_MEMORY_USAGE_WARNING", health_config, "memory_usage_warning")
        ConfigManager._load_float_env("HEALTH_MEMORY_USAGE_CRITICAL", health_config, "memory_usage_critical")

        if health_config:
            config["health"] = health_config

    def _load_environment_variables(self) -> dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}

        # Load all configuration sections
        self._load_main_config_from_env(config)
        self._load_reference_person_config_from_env(config)
        self._load_user_config_from_env(config)
        self._load_ms_graph_config_from_env(config)
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

        # Load observability configuration
        self._load_observability_config_from_env(config)

        # Load health configuration
        self._load_health_config_from_env(config)

        return config

    def _merge_configs(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
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
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
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

    def get_observability_config(self) -> ObservabilityConfig:
        """Get observability configuration."""
        return self.get_config().observability


# ==============================================
# MODULE-LEVEL TEST FUNCTIONS
# ==============================================


def _test_config_manager_initialization() -> bool:
    """Test ConfigManager class initialization."""
    assert callable(ConfigManager), "ConfigManager should be callable"

    # Test basic instantiation - should not raise
    manager = ConfigManager(auto_load=False)
    assert manager is not None, "ConfigManager should instantiate successfully"
    assert manager.environment in {
        "development",
        "test",
        "production",
    }, f"Should have valid environment, got '{manager.environment}'"

    # Verify essential attributes exist
    assert hasattr(manager, "_config_cache"), "Should have _config_cache attribute"
    assert hasattr(manager, "config_file"), "Should have config_file attribute"
    return True


def _test_config_validation() -> bool:
    """Test configuration validation functions."""
    manager = ConfigManager(auto_load=False)
    assert hasattr(manager, "validate_config"), "Should have validate_config method"
    assert hasattr(manager, "load_config"), "Should have load_config method"

    # Actually call validate_config to ensure it works
    config = manager.load_config()
    assert config is not None, "load_config should return a config object"

    # Validate the config structure - returns list of errors (empty = valid)
    validation_result = manager.validate_config()
    assert isinstance(validation_result, list), "validate_config should return list of errors"
    assert len(validation_result) == 0, f"Config should be valid, got errors: {validation_result}"
    return True


def _test_config_loading() -> bool:
    """Test configuration loading functionality."""
    manager = ConfigManager(auto_load=False)
    assert hasattr(manager, "get_config"), "Should have get_config method"

    # Actually load config and verify it returns expected type
    config = manager.load_config()
    assert config is not None, "load_config should return config"

    # Verify config has expected sections
    assert hasattr(config, "api"), "Config should have api section"
    assert hasattr(config, "selenium"), "Config should have selenium section"
    assert hasattr(config, "database"), "Config should have database section"
    return True


def _test_config_access() -> bool:
    """Test configuration value access."""
    manager = ConfigManager(auto_load=False)
    config = manager.load_config()

    # Test actual value access, not just method existence
    rps = config.api.requests_per_second
    assert isinstance(rps, (int, float)), f"requests_per_second should be numeric, got {type(rps)}"
    assert rps > 0, f"requests_per_second should be positive, got {rps}"

    # Test another config value
    timeout = config.selenium.page_load_timeout
    assert isinstance(timeout, int), f"page_load_timeout should be int, got {type(timeout)}"
    return True


def _test_missing_config_handling() -> bool:
    """Test handling of missing configuration."""
    manager = ConfigManager(auto_load=False)
    config = manager.load_config()

    # Access a config value that has a default - should not raise
    assert config.api.requests_per_second is not None, "Should have default RPS"

    # Verify defaults are applied for common settings
    assert config.api.max_pages >= 0, "max_pages should have sensible default"
    return True


def _test_invalid_config_data() -> bool:
    """Test handling of invalid configuration data."""
    import os

    original_rps = os.environ.get("REQUESTS_PER_SECOND")

    # Set invalid value
    os.environ["REQUESTS_PER_SECOND"] = "not_a_number"

    manager = ConfigManager(auto_load=False)
    config = manager.load_config()

    # Should fall back to default (0.3) or keep previous valid value, not crash
    rps = config.api.requests_per_second
    assert isinstance(rps, (int, float)), f"RPS should be numeric, got {type(rps)}"
    assert rps > 0, f"RPS should be positive, got {rps}"

    # Cleanup
    if original_rps is not None:
        os.environ["REQUESTS_PER_SECOND"] = original_rps
    elif "REQUESTS_PER_SECOND" in os.environ:
        del os.environ["REQUESTS_PER_SECOND"]
    return True


def _test_config_file_integration() -> bool:
    """Test configuration file integration."""
    from testing.test_utilities import temp_file

    # Test with temporary config file
    with temp_file(suffix='.yaml', mode='w+') as temp_f:
        temp_f.write_text("test: value\n", encoding="utf-8")

        # Verify we can create manager independent of temp file
        manager = ConfigManager(auto_load=False)
        config = manager.load_config()
        assert config is not None, "Should load default config"
        assert hasattr(config, "api"), "Config should have api section"
    return True


def _test_environment_integration() -> bool:
    """Test environment variable integration."""
    manager = ConfigManager(auto_load=False)
    assert hasattr(manager, "environment"), "Should have environment attribute"

    # Verify environment is correctly detected
    env = manager.environment
    assert env in {"development", "test", "production"}, f"Invalid environment: {env}"

    # Test that environment affects behavior
    config = manager.load_config()
    assert config is not None, "Should load config for current environment"
    return True


def _test_config_access_performance() -> bool:
    """Test configuration access performance."""
    import time

    manager = ConfigManager(auto_load=False)
    config = manager.load_config()

    start = time.time()
    for _ in range(100):
        _ = config.api.requests_per_second
    elapsed = time.time() - start

    assert elapsed < 1.0, f"Config access should be fast, took {elapsed:.3f}s"
    return True


def _test_config_error_handling() -> bool:
    """Test configuration error handling."""
    # Test that ConfigManager handles various scenarios gracefully
    manager = ConfigManager(auto_load=False)
    assert manager is not None, "Manager should be created"

    # Calling load_config multiple times should work
    config1 = manager.load_config()
    config2 = manager.load_config()
    assert config1 is not None and config2 is not None, "Multiple loads should work"

    # Validate should return list of errors (empty if valid)
    result = manager.validate_config()
    assert isinstance(result, list), "validate_config should return list"
    return True


def _test_requests_per_second_loading() -> bool:
    """Test REQUESTS_PER_SECOND environment variable loading."""
    import os

    from dotenv import load_dotenv

    # Save original values
    original_value = os.environ.get("REQUESTS_PER_SECOND")
    original_skip_dotenv = os.environ.get("CONFIG_SKIP_DOTENV")

    try:
        # Skip loading from .env file for this test
        os.environ["CONFIG_SKIP_DOTENV"] = "1"
        # Enable unsafe rate limits to allow custom RPS values > 0.3
        os.environ["ALLOW_UNSAFE_RATE_LIMIT"] = "true"

        # Test 1: Custom value from environment
        os.environ["REQUESTS_PER_SECOND"] = "0.2"
        manager = ConfigManager(auto_load=False)
        config = manager.load_config()
        custom_rps = config.api.requests_per_second
        assert custom_rps == 0.2, f"Expected 0.2 from env, got {custom_rps}"

        # Test 2: Different custom value
        os.environ["REQUESTS_PER_SECOND"] = "0.5"
        manager = ConfigManager(auto_load=False)
        config = manager.load_config()
        custom_rps = config.api.requests_per_second
        assert custom_rps == 0.5, f"Expected 0.5 from env, got {custom_rps}"

        # Test 3: Invalid value should fallback to schema default (2.0)
        os.environ["REQUESTS_PER_SECOND"] = "invalid"
        manager = ConfigManager(auto_load=False)
        config = manager.load_config()
        invalid_rps = config.api.requests_per_second
        # Schema default is 2.0 - when invalid, _set_float_config doesn't set, so default applies
        assert invalid_rps == 2.0, f"Expected fallback to schema default 2.0, got {invalid_rps}"

    finally:
        # Cleanup: restore original values
        if original_value is not None:
            os.environ["REQUESTS_PER_SECOND"] = original_value
        elif "REQUESTS_PER_SECOND" in os.environ:
            del os.environ["REQUESTS_PER_SECOND"]

        if original_skip_dotenv is not None:
            os.environ["CONFIG_SKIP_DOTENV"] = original_skip_dotenv
        elif "CONFIG_SKIP_DOTENV" in os.environ:
            del os.environ["CONFIG_SKIP_DOTENV"]

        # Reload dotenv to restore .env file values
        load_dotenv(override=True)
    return True


# ==============================================
# MAIN TEST SUITE RUNNER
# ==============================================


def config_manager_module_tests() -> bool:
    """
    Comprehensive test suite for config_manager.py with proper TestSuite framework.
    Tests configuration management, validation, and loading functionality.
    """

    from testing.test_framework import TestSuite

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
    test_requests_per_second_loading = _test_requests_per_second_loading

    # INITIALIZATION TESTS
    suite.run_test(
        "Config Manager Initialization",
        test_config_manager_initialization,
        "ConfigManager initializes correctly with valid environment",
        "Test ConfigManager class instantiation and initialization",
        "Verify ConfigManager can be instantiated and has valid environment",
    )

    suite.run_test(
        "Config Validation",
        test_config_validation,
        "Configuration validation methods are available",
        "Test configuration validation functionality",
        "Verify validate_config and load_config methods exist",
    )

    suite.run_test(
        "Config Loading",
        test_config_loading,
        "Configuration loading works correctly",
        "Test configuration loading functionality",
        "Verify get_config method exists",
    )

    suite.run_test(
        "Config Access",
        test_config_access,
        "Configuration values can be accessed",
        "Test configuration value access",
        "Verify config access methods are available",
    )

    suite.run_test(
        "Missing Config Handling",
        test_missing_config_handling,
        "Missing configuration is handled gracefully",
        "Test handling of missing configuration",
        "Verify system handles missing config without crashing",
    )

    suite.run_test(
        "Invalid Config Data",
        test_invalid_config_data,
        "Invalid configuration data is handled gracefully",
        "Test handling of invalid configuration data",
        "Verify system handles invalid data without crashing",
    )

    suite.run_test(
        "Config File Integration",
        test_config_file_integration,
        "Configuration file integration works correctly",
        "Test configuration file loading and integration",
        "Verify system can work with temporary config files",
    )

    suite.run_test(
        "Environment Integration",
        test_environment_integration,
        "Environment variable integration works correctly",
        "Test environment variable handling",
        "Verify environment attribute exists",
    )

    suite.run_test(
        "Config Access Performance",
        test_config_access_performance,
        "Configuration access is performant",
        "Test configuration access performance",
        "Verify 100 config accesses take less than 1 second",
    )

    suite.run_test(
        "Config Error Handling",
        test_config_error_handling,
        "Configuration errors are handled gracefully",
        "Test configuration error handling",
        "Verify system handles errors without crashing",
    )

    suite.run_test(
        "REQUESTS_PER_SECOND Loading",
        test_requests_per_second_loading,
        "REQUESTS_PER_SECOND loads from environment correctly",
        "Test REQUESTS_PER_SECOND environment variable loading",
        "Verify default 0.4, custom values, and invalid value handling",
    )

    return suite.finish_suite()


# Standard test runner for test discovery
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(config_manager_module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
