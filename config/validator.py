#!/usr/bin/env python3
"""
Unified Configuration Validation Layer.

This module provides comprehensive startup validation for all configuration settings,
ensuring clear, actionable error messages and preventing runtime failures from
missing or invalid configuration.

Phase 1 Implementation (Nov 2025):
- Validates ALL required configuration at startup
- Returns clear, actionable error messages
- Provides health check functionality for main menu integration
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if __package__ in {None, ""}:
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from config.config_schema import ConfigSchema


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    message: str
    severity: str = "error"  # "error", "warning", "info"
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        icon = "✅" if self.passed else ("⚠️" if self.severity == "warning" else "❌")
        result = f"{icon} {self.name}: {self.message}"
        if not self.passed and self.suggestion:
            result += f"\n   💡 {self.suggestion}"
        return result


@dataclass
class ValidationReport:
    """Aggregated validation results."""

    results: list[ValidationResult] = field(default_factory=list)
    config_loaded: bool = False

    @property
    def passed(self) -> bool:
        """Return True if all error-severity checks passed."""
        return all(r.passed for r in self.results if r.severity == "error")

    @property
    def errors(self) -> list[ValidationResult]:
        """Return all failed error-severity checks."""
        return [r for r in self.results if not r.passed and r.severity == "error"]

    @property
    def warnings(self) -> list[ValidationResult]:
        """Return all failed warning-severity checks."""
        return [r for r in self.results if not r.passed and r.severity == "warning"]

    def add(self, result: ValidationResult) -> None:
        """Add a validation result to the report."""
        self.results.append(result)

    def print_summary(self) -> None:
        """Print a formatted summary of all validation results."""
        print("\n" + "=" * 60)
        print(" CONFIGURATION HEALTH CHECK ".center(60, "="))
        print("=" * 60)

        if not self.config_loaded:
            print("\n❌ Configuration failed to load - cannot validate settings")
            return

        # Group results by category
        categories: dict[str, list[ValidationResult]] = {}
        for result in self.results:
            category = result.name.split(":")[0] if ":" in result.name else "General"
            if category not in categories:
                categories[category] = []
            categories[category].append(result)

        # Print each category
        for category, cat_results in categories.items():
            print(f"\n📋 {category}")
            print("-" * 40)
            for result in cat_results:
                print(f"   {result}")

        # Summary
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        errors = len(self.errors)
        warnings = len(self.warnings)

        print("\n" + "=" * 60)
        if errors == 0:
            print(f"✅ Configuration valid: {passed}/{total} checks passed")
            if warnings > 0:
                print(f"⚠️  {warnings} warning(s) - review recommended")
        else:
            print(f"❌ Configuration invalid: {errors} error(s), {warnings} warning(s)")
            print("\n💡 Fix the errors above before running actions.")
        print("=" * 60 + "\n")


class ConfigurationValidator:
    """
    Unified configuration validator for startup and health checks.

    Validates all required configuration settings and provides
    clear, actionable error messages.
    """

    def __init__(self, config: Optional[ConfigSchema] = None):
        """Initialize validator with optional config."""
        self._config = config
        self._report = ValidationReport()

    def _load_config(self) -> Optional[ConfigSchema]:
        """Load configuration if not already loaded."""
        if self._config is not None:
            self._report.config_loaded = True
            return self._config

        try:
            from config import config_schema

            self._config = config_schema
            self._report.config_loaded = True
            return self._config
        except ImportError as e:
            logger.error(f"Failed to import config_schema: {e}")
            self._report.config_loaded = False
            return None
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self._report.config_loaded = False
            return None

    def validate_all(self) -> ValidationReport:
        """Run all validation checks and return a comprehensive report."""
        self._report = ValidationReport()
        config = self._load_config()

        if config is None:
            self._report.add(
                ValidationResult(
                    name="Config: Load",
                    passed=False,
                    message="Configuration failed to load",
                    severity="error",
                    suggestion="Check .env file exists and is properly formatted",
                )
            )
            return self._report

        # Run all validation categories
        self._validate_environment(config)
        self._validate_database(config)
        self._validate_api(config)
        self._validate_rate_limiting(config)
        self._validate_selenium(config)
        self._validate_ai_provider(config)
        self._validate_paths(config)
        self._validate_processing_limits(config)
        self._validate_observability(config)

        return self._report

    def _validate_environment(self, config: ConfigSchema) -> None:
        """Validate environment settings."""
        env = getattr(config, "environment", None)
        valid_envs = {"development", "testing", "production"}

        self._report.add(
            ValidationResult(
                name="Environment: Type",
                passed=env in valid_envs,
                message=f"Environment is '{env}'" if env else "Environment not set",
                severity="error" if env not in valid_envs else "info",
                suggestion=f"Set ENVIRONMENT to one of: {', '.join(valid_envs)}",
            )
        )

    def _validate_database(self, config: ConfigSchema) -> None:
        """Validate database configuration."""
        db = getattr(config, "database", None)
        if db is None:
            self._report.add(
                ValidationResult(
                    name="Database: Config",
                    passed=False,
                    message="Database configuration missing",
                    severity="error",
                    suggestion="Ensure database configuration is defined",
                )
            )
            return

        # Check database file path
        db_file = getattr(db, "database_file", None)
        if db_file:
            db_path = Path(db_file) if isinstance(db_file, str) else db_file
            parent_exists = db_path.parent.exists() if db_path else False
            self._report.add(
                ValidationResult(
                    name="Database: Path",
                    passed=parent_exists,
                    message=f"Database directory {'exists' if parent_exists else 'does not exist'}: {db_path.parent if db_path else 'N/A'}",
                    severity="error" if not parent_exists else "info",
                    suggestion="Create the Data directory or set DATABASE_FILE to a valid path",
                )
            )
        else:
            self._report.add(
                ValidationResult(
                    name="Database: Path",
                    passed=False,
                    message="Database file path not configured",
                    severity="warning",
                    suggestion="Set DATABASE_FILE in .env (default: Data/ancestry.db)",
                )
            )

        # Check pool settings
        pool_size = getattr(db, "pool_size", 0)
        self._report.add(
            ValidationResult(
                name="Database: Pool Size",
                passed=1 <= pool_size <= 50,
                message=f"Pool size is {pool_size}",
                severity="warning" if pool_size < 1 or pool_size > 50 else "info",
                suggestion="Set DB_POOL_SIZE between 1 and 50",
            )
        )

    def _validate_api(self, config: ConfigSchema) -> None:
        """Validate API configuration."""
        api = getattr(config, "api", None)
        if api is None:
            self._report.add(
                ValidationResult(
                    name="API: Config",
                    passed=False,
                    message="API configuration missing",
                    severity="error",
                    suggestion="Ensure API configuration is defined",
                )
            )
            return

        # Check base URL
        base_url = getattr(api, "base_url", "")
        self._report.add(
            ValidationResult(
                name="API: Base URL",
                passed=bool(base_url and base_url.startswith(("http://", "https://"))),
                message=f"Base URL: {base_url[:50]}..." if len(base_url) > 50 else f"Base URL: {base_url}",
                severity="error" if not base_url else "info",
                suggestion="Set BASE_URL to https://www.ancestry.com/",
            )
        )

        # Check credentials (existence only, not values)
        username = getattr(api, "username", "")
        password = getattr(api, "password", "")
        creds_set = bool(username and password)
        self._report.add(
            ValidationResult(
                name="API: Credentials",
                passed=creds_set,
                message="Credentials configured" if creds_set else "Credentials not set",
                severity="error" if not creds_set else "info",
                suggestion="Set ANCESTRY_USERNAME and ANCESTRY_PASSWORD in .env",
            )
        )

        # Check timeout
        timeout = getattr(api, "request_timeout", 0)
        self._report.add(
            ValidationResult(
                name="API: Timeout",
                passed=timeout > 0,
                message=f"Request timeout: {timeout}s",
                severity="warning" if timeout < 30 else "info",
                suggestion="Set REQUEST_TIMEOUT to at least 30 seconds",
            )
        )

    def _validate_observability(self, config: ConfigSchema) -> None:
        """Validate observability / Prometheus settings."""
        obs = getattr(config, "observability", None)
        if obs is None:
            self._add_obs_result(
                name="Observability: Config",
                passed=False,
                message="Observability configuration missing",
                severity="error",
                suggestion="Define observability settings in config_schema (.env) section",
            )
            return

        metrics_enabled = bool(getattr(obs, "enable_prometheus_metrics", False))
        auto_start = bool(getattr(obs, "auto_start_prometheus", False))

        self._add_obs_result(
            name="Observability: Prometheus Enabled",
            passed=True,
            message=f"enable_prometheus_metrics={metrics_enabled}",
            severity="info",
        )

        self._validate_obs_host(obs, metrics_enabled)
        self._validate_obs_port(obs, metrics_enabled)
        self._validate_obs_namespace(obs, metrics_enabled)
        self._validate_obs_binary(obs, metrics_enabled, auto_start)
        self._validate_obs_client(metrics_enabled)

    def _add_obs_result(
        self,
        *,
        name: str,
        passed: bool,
        message: str,
        severity: str,
        suggestion: Optional[str] = None,
    ) -> None:
        self._report.add(
            ValidationResult(
                name=name,
                passed=passed,
                message=message,
                severity=severity,
                suggestion=suggestion,
            )
        )

    def _validate_obs_host(self, obs: Any, metrics_enabled: bool) -> None:
        export_host = getattr(obs, "metrics_export_host", "")
        host_valid = isinstance(export_host, str) and len(export_host.strip()) > 0
        self._add_obs_result(
            name="Observability: Export Host",
            passed=host_valid,
            message=f"metrics_export_host={export_host or 'unset'}",
            severity="error" if not host_valid and metrics_enabled else "warning",
            suggestion="Set METRICS_EXPORT_HOST (e.g., 127.0.0.1 or 0.0.0.0)",
        )

    def _validate_obs_port(self, obs: Any, metrics_enabled: bool) -> None:
        export_port = getattr(obs, "metrics_export_port", 0)
        port_valid = isinstance(export_port, int) and 1 <= export_port <= 65535
        self._add_obs_result(
            name="Observability: Export Port",
            passed=port_valid,
            message=f"metrics_export_port={export_port}",
            severity="error" if not port_valid and metrics_enabled else "warning",
            suggestion="Set METRICS_EXPORT_PORT to 1-65535",
        )

    def _validate_obs_namespace(self, obs: Any, metrics_enabled: bool) -> None:
        namespace = getattr(obs, "metrics_namespace", "")
        namespace_valid = isinstance(namespace, str) and len(namespace.strip()) > 0
        self._add_obs_result(
            name="Observability: Namespace",
            passed=namespace_valid,
            message=f"metrics_namespace={namespace or 'unset'}",
            severity="error" if not namespace_valid and metrics_enabled else "warning",
            suggestion="Set METRICS_NAMESPACE (default: ancestry)",
        )

    def _validate_obs_binary(self, obs: Any, metrics_enabled: bool, auto_start: bool) -> None:
        binary_path = getattr(obs, "prometheus_binary_path", None)
        binary_msg = "not set"
        binary_ok = True
        if binary_path:
            binary_msg = str(binary_path)
            binary_ok = Path(binary_path).exists()
        severity = "warning" if metrics_enabled or auto_start else "info"
        self._add_obs_result(
            name="Observability: Prometheus Binary",
            passed=binary_ok,
            message=f"prometheus_binary_path={binary_msg}",
            severity=severity if not binary_ok else "info",
            suggestion="Point PROMETHEUS_BINARY_PATH to prometheus.exe when auto_start_prometheus is enabled",
        )

    def _validate_obs_client(self, metrics_enabled: bool) -> None:
        prom_available = False
        import_error: str | None = None
        try:
            from observability.metrics_registry import PROMETHEUS_AVAILABLE

            prom_available = PROMETHEUS_AVAILABLE
        except Exception as exc:  # pragma: no cover - defensive
            import_error = str(exc)

        self._add_obs_result(
            name="Observability: Client Available",
            passed=prom_available or not metrics_enabled,
            message=(
                "prometheus_client importable"
                if prom_available
                else f"prometheus_client unavailable: {import_error or 'missing package'}"
            ),
            severity="error" if metrics_enabled and not prom_available else "info",
            suggestion="Install prometheus_client and ensure it imports in this environment",
        )

    def _validate_rate_limiting(self, config: ConfigSchema) -> None:
        """Validate rate limiting configuration."""
        api = getattr(config, "api", None)
        if api is None:
            return

        # Check RPS
        rps = getattr(api, "requests_per_second", 0)
        rps_safe = 0.1 <= rps <= 2.0  # Safe range for Ancestry API
        self._report.add(
            ValidationResult(
                name="Rate Limit: RPS",
                passed=rps_safe,
                message=f"Requests per second: {rps}",
                severity="warning" if not rps_safe else "info",
                suggestion="Keep REQUESTS_PER_SECOND between 0.1 and 2.0 to avoid 429 errors",
            )
        )

        # Check concurrency
        concurrency = getattr(api, "max_concurrency", 1)
        concurrency_safe = concurrency <= 2  # Sequential or limited parallel
        self._report.add(
            ValidationResult(
                name="Rate Limit: Concurrency",
                passed=concurrency_safe,
                message=f"Max concurrency: {concurrency}",
                severity="warning" if not concurrency_safe else "info",
                suggestion="Keep MAX_CONCURRENCY at 1-2 for API stability",
            )
        )

        # Check max delay
        max_delay = getattr(api, "max_delay", 0)
        self._report.add(
            ValidationResult(
                name="Rate Limit: Max Delay",
                passed=max_delay >= 5.0,
                message=f"Max delay: {max_delay}s",
                severity="warning" if max_delay < 5.0 else "info",
                suggestion="Set MAX_DELAY to at least 5.0 seconds for error recovery",
            )
        )

    def _validate_selenium(self, config: ConfigSchema) -> None:
        """Validate Selenium/browser configuration."""
        selenium = getattr(config, "selenium", None)
        if selenium is None:
            self._report.add(
                ValidationResult(
                    name="Selenium: Config",
                    passed=False,
                    message="Selenium configuration missing",
                    severity="error",
                    suggestion="Ensure Selenium configuration is defined",
                )
            )
            return

        # Check page load timeout
        page_timeout = getattr(selenium, "page_load_timeout", 0)
        self._report.add(
            ValidationResult(
                name="Selenium: Page Timeout",
                passed=page_timeout >= 30,
                message=f"Page load timeout: {page_timeout}s",
                severity="warning" if page_timeout < 30 else "info",
                suggestion="Set PAGE_LOAD_TIMEOUT to at least 30 seconds",
            )
        )

        # Check 2FA timeout
        tfa_timeout = getattr(selenium, "two_fa_code_entry_timeout", 0)
        self._report.add(
            ValidationResult(
                name="Selenium: 2FA Timeout",
                passed=tfa_timeout >= 60,
                message=f"2FA entry timeout: {tfa_timeout}s",
                severity="warning" if tfa_timeout < 60 else "info",
                suggestion="Set TWO_FA_CODE_ENTRY_TIMEOUT to at least 60 seconds",
            )
        )

    def _validate_ai_provider(self, config: ConfigSchema) -> None:
        """Validate AI provider configuration."""
        ai_provider = getattr(config, "ai_provider", "")
        api = getattr(config, "api", None)

        if not ai_provider:
            self._report.add(
                ValidationResult(
                    name="AI: Provider",
                    passed=True,
                    message="No AI provider configured (AI features disabled)",
                    severity="info",
                    suggestion=None,
                )
            )
            return

        valid_providers = {"deepseek", "gemini", "moonshot", "local_llm", "grok", "inception", "tetrate"}
        provider_valid = ai_provider.lower() in valid_providers
        self._report.add(
            ValidationResult(
                name="AI: Provider",
                passed=provider_valid,
                message=f"AI provider: {ai_provider}",
                severity="error" if not provider_valid else "info",
                suggestion=f"Set AI_PROVIDER to one of: {', '.join(valid_providers)}",
            )
        )

        if api is None:
            return

        # Check API key for configured provider
        key_map = {
            "deepseek": "deepseek_api_key",
            "gemini": "google_api_key",
            "moonshot": "moonshot_api_key",
            "local_llm": "local_llm_api_key",
            "grok": "xai_api_key",
            "inception": "inception_api_key",
            "tetrate": "tetrate_api_key",
        }

        if ai_provider.lower() in key_map:
            key_attr = key_map[ai_provider.lower()]
            api_key = getattr(api, key_attr, "")
            key_set = bool(api_key)
            self._report.add(
                ValidationResult(
                    name="AI: API Key",
                    passed=key_set,
                    message=f"API key {'configured' if key_set else 'not set'} for {ai_provider}",
                    severity="error" if not key_set else "info",
                    suggestion=f"Set the API key for {ai_provider} in .env",
                )
            )

    def _validate_paths(self, _config: ConfigSchema) -> None:
        """Validate file and directory paths.

        Note: _config parameter kept for API consistency with other validators.
        """
        # Check Data directory
        data_dir = Path("Data")
        self._report.add(
            ValidationResult(
                name="Paths: Data Directory",
                passed=data_dir.exists(),
                message=f"Data directory {'exists' if data_dir.exists() else 'does not exist'}",
                severity="warning" if not data_dir.exists() else "info",
                suggestion="Create the Data directory for database storage",
            )
        )

        # Check Logs directory
        logs_dir = Path("Logs")
        self._report.add(
            ValidationResult(
                name="Paths: Logs Directory",
                passed=logs_dir.exists(),
                message=f"Logs directory {'exists' if logs_dir.exists() else 'does not exist'}",
                severity="warning" if not logs_dir.exists() else "info",
                suggestion="Create the Logs directory for log file storage",
            )
        )

        # Check Cache directory
        cache_dir = Path("Cache")
        self._report.add(
            ValidationResult(
                name="Paths: Cache Directory",
                passed=cache_dir.exists(),
                message=f"Cache directory {'exists' if cache_dir.exists() else 'does not exist'}",
                severity="warning" if not cache_dir.exists() else "info",
                suggestion="Create the Cache directory for caching",
            )
        )

        # Check .env file
        env_file = Path(".env")
        self._report.add(
            ValidationResult(
                name="Paths: .env File",
                passed=env_file.exists(),
                message=f".env file {'exists' if env_file.exists() else 'does not exist'}",
                severity="error" if not env_file.exists() else "info",
                suggestion="Copy .env.example to .env and configure your settings",
            )
        )

    def _validate_processing_limits(self, config: ConfigSchema) -> None:
        """Validate processing limit settings."""
        batch_size = getattr(config, "batch_size", 0)
        self._report.add(
            ValidationResult(
                name="Processing: Batch Size",
                passed=1 <= batch_size <= 100,
                message=f"Batch size: {batch_size}",
                severity="warning" if batch_size < 1 or batch_size > 100 else "info",
                suggestion="Set BATCH_SIZE between 1 and 100",
            )
        )

        max_pages = getattr(config, "api", None)
        if max_pages:
            max_pages = getattr(max_pages, "max_pages", 0)
            # max_pages=0 means unlimited, which is valid
            self._report.add(
                ValidationResult(
                    name="Processing: Max Pages",
                    passed=max_pages >= 0,
                    message=f"Max pages: {max_pages} ({'unlimited' if max_pages == 0 else 'limited'})",
                    severity="info",
                    suggestion=None,
                )
            )

        parallel_workers = getattr(config, "parallel_workers", 1)
        self._report.add(
            ValidationResult(
                name="Processing: Parallel Workers",
                passed=1 <= parallel_workers <= 4,
                message=f"Parallel workers: {parallel_workers}",
                severity="warning" if parallel_workers > 4 else "info",
                suggestion="Keep PARALLEL_WORKERS between 1 and 4 for stability",
            )
        )


def run_startup_validation() -> bool:
    """
    Run startup validation and return whether configuration is valid.

    This should be called early in main() before any actions.

    Returns:
        True if all critical checks passed, False otherwise
    """
    validator = ConfigurationValidator()
    report = validator.validate_all()

    if not report.passed:
        report.print_summary()
        return False

    # Log success without full report (verbose startup is configurable)
    logger.info("✅ Configuration validation passed")
    return True


def run_health_check() -> ValidationReport:
    """
    Run a full configuration health check.

    This is intended for the main menu health check option.

    Returns:
        ValidationReport with all check results
    """
    validator = ConfigurationValidator()
    report = validator.validate_all()
    report.print_summary()
    return report


# === Module Tests ===


def _test_validation_result_str() -> bool:
    """Test ValidationResult string formatting."""
    # Passed result
    passed = ValidationResult(name="Test: Check", passed=True, message="All good")
    assert "✅" in str(passed), "Passed result should have checkmark"
    assert "Test: Check" in str(passed), "Should include name"

    # Failed error result
    failed = ValidationResult(
        name="Test: Failed",
        passed=False,
        message="Something wrong",
        severity="error",
        suggestion="Fix it",
    )
    assert "❌" in str(failed), "Failed error should have X"
    assert "💡" in str(failed), "Should include suggestion"

    # Failed warning result
    warning = ValidationResult(name="Test: Warn", passed=False, message="Minor issue", severity="warning")
    assert "⚠️" in str(warning), "Warning should have warning icon"

    return True


def _test_validation_report_aggregation() -> bool:
    """Test ValidationReport aggregation methods."""
    report = ValidationReport()
    report.config_loaded = True

    report.add(ValidationResult(name="Check1", passed=True, message="OK"))
    report.add(ValidationResult(name="Check2", passed=False, message="Error", severity="error"))
    report.add(ValidationResult(name="Check3", passed=False, message="Warn", severity="warning"))

    assert report.passed is False, "Should be False with error"
    assert len(report.errors) == 1, "Should have 1 error"
    assert len(report.warnings) == 1, "Should have 1 warning"

    return True


def _test_validation_report_passed_with_warnings() -> bool:
    """Test that report passes with warnings but no errors."""
    report = ValidationReport()
    report.config_loaded = True

    report.add(ValidationResult(name="Check1", passed=True, message="OK"))
    report.add(ValidationResult(name="Check2", passed=False, message="Warn", severity="warning"))

    assert report.passed is True, "Should pass with only warnings"
    assert len(report.warnings) == 1, "Should have 1 warning"

    return True


def _test_configuration_validator_initialization() -> bool:
    """Test ConfigurationValidator initialization."""
    validator = ConfigurationValidator()
    assert validator is not None, "Should create validator"

    report = validator.validate_all()
    assert isinstance(report, ValidationReport), "Should return ValidationReport"

    return True


def _test_validate_paths_checks_directories() -> bool:
    """Test that path validation checks expected directories."""
    validator = ConfigurationValidator()
    report = validator.validate_all()

    # Check that path validations are included
    path_results = [r for r in report.results if r.name.startswith("Paths:")]
    assert len(path_results) >= 3, f"Should have at least 3 path checks, got {len(path_results)}"

    return True


def _test_run_startup_validation_returns_bool() -> bool:
    """Test that run_startup_validation returns a boolean."""
    from unittest.mock import patch

    # Mock print to avoid output during test
    with patch("builtins.print"):
        result = run_startup_validation()
        assert isinstance(result, bool), "Should return boolean"

    return True


def _test_run_health_check_returns_report() -> bool:
    """Test that run_health_check returns a ValidationReport."""
    from unittest.mock import patch

    with patch("builtins.print"):
        report = run_health_check()
        assert isinstance(report, ValidationReport), "Should return ValidationReport"

    return True


def module_tests() -> bool:
    """Run module tests for config.validator."""
    from testing.test_framework import TestSuite

    suite = TestSuite("config.validator", "config/validator.py")

    suite.run_test(
        "ValidationResult string formatting",
        _test_validation_result_str,
        "Ensures ValidationResult formats correctly with icons and suggestions.",
    )

    suite.run_test(
        "ValidationReport aggregation",
        _test_validation_report_aggregation,
        "Ensures ValidationReport correctly aggregates errors and warnings.",
    )

    suite.run_test(
        "ValidationReport passes with warnings",
        _test_validation_report_passed_with_warnings,
        "Ensures report passes when only warnings present.",
    )

    suite.run_test(
        "ConfigurationValidator initialization",
        _test_configuration_validator_initialization,
        "Ensures ConfigurationValidator initializes and validates.",
    )

    suite.run_test(
        "Path validation checks directories",
        _test_validate_paths_checks_directories,
        "Ensures path validation includes directory checks.",
    )

    suite.run_test(
        "run_startup_validation returns bool",
        _test_run_startup_validation_returns_bool,
        "Ensures startup validation returns boolean.",
    )

    suite.run_test(
        "run_health_check returns report",
        _test_run_health_check_returns_report,
        "Ensures health check returns ValidationReport.",
    )

    return suite.finish_suite()


if __name__ == "__main__":
    from testing.test_utilities import create_standard_test_runner

    run_comprehensive_tests = create_standard_test_runner(module_tests)
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
