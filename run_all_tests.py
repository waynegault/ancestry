#!/usr/bin/env python3

"""
Comprehensive Test Orchestration & Quality Assurance Engine

Advanced test execution platform providing systematic validation of the entire
genealogical automation system through comprehensive test suite orchestration,
intelligent quality assessment, and detailed performance analytics with
automated test discovery and professional reporting for reliable system validation.

Test Orchestration:
• Comprehensive test suite execution with intelligent module discovery and coordination
• Advanced test scheduling with dependency management and parallel execution capabilities
• Sophisticated test reporting with detailed analytics, quality metrics, and performance insights
• Intelligent test categorization with enhanced module descriptions and quality scoring
• Comprehensive error handling with detailed debugging information and failure analysis
• Integration with continuous integration systems for automated testing workflows

Quality Assessment:
• Advanced quality scoring with comprehensive code analysis and best practices validation
• Intelligent quality gate enforcement with configurable thresholds and automated reporting
• Comprehensive linting integration with automated code style and quality checks
• Performance monitoring with timing analysis, resource usage tracking, and optimization recommendations
• Automated regression detection with baseline comparison and deviation analysis
• Integration with quality assessment tools for comprehensive system validation

System Validation:
• Complete system health validation with comprehensive module testing and verification
• Advanced test analytics with success rate tracking, failure pattern analysis, and trend monitoring
• Intelligent test prioritization with risk-based testing and impact assessment strategies
• Comprehensive test coverage analysis with functional coverage metrics and gap identification
• Automated test maintenance with self-healing tests and adaptive testing strategies
• Professional reporting with detailed test results, quality insights, and actionable recommendations

Foundation Services:
Provides the essential test orchestration infrastructure that ensures reliable,
high-quality genealogical automation through systematic validation, comprehensive
quality assessment, and professional testing for production-ready research workflows.

Usage:
    python run_all_tests.py                # Run all tests with detailed reporting
    python run_all_tests.py --fast         # Run with parallel execution optimization
    python run_all_tests.py --benchmark    # Run with detailed performance benchmarking
    python run_all_tests.py --integration  # Run integration tests with live API access
    python run_all_tests.py --slow         # Include slow simulation tests
    python run_all_tests.py --skip-linter  # Skip linter checks
    python run_all_tests.py --analyze-logs # Analyze application logs for performance metrics
    python run_all_tests.py --fast --analyze-logs  # Run tests and then analyze logs

Modes:
    Default: Unit tests only, SKIP_LIVE_API_TESTS=true, SKIP_SLOW_TESTS=true
    --integration: Enables live browser/API tests with real authenticated sessions
    --slow: Enables 724-page workload simulation and other long-running tests

IMPORTANT: Always run tests in venv (virtual environment)
    Windows: .venv\\Scripts\activate
    Linux/Mac: source .venv/bin/activate
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

from core.feature_flags import FeatureFlags, bootstrap_feature_flags


def _ensure_venv() -> None:
    """Ensure running in venv, auto-restart if needed."""
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

    if in_venv:
        return

    # Check if .venv exists
    venv_python = Path('.venv') / 'Scripts' / 'python.exe'
    if not venv_python.exists():
        # Try Unix-style path
        venv_python = Path('.venv') / 'bin' / 'python'
        if not venv_python.exists():
            print("⚠️  WARNING: Not running in virtual environment and .venv not found")
            print("   Some tests may fail due to missing dependencies")
            return

    # Re-run with venv Python using os.execv to replace current process
    import os as _os

    print(f"🔄 Re-running tests with venv Python: {venv_python}")
    print()
    _os.execv(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]])


_ensure_venv()
import re

# sys already imported above for venv check
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Final, Optional, TypedDict

try:
    import psutil as _psutil
except ImportError:
    psutil: Optional[ModuleType] = None
    _psutil_available = False
else:
    psutil = _psutil
    _psutil_available = True

PSUTIL_AVAILABLE = _psutil_available

SEPARATOR_LINE: Final[str] = "=" * 70
SECTION_SEPARATOR: Final[str] = "\n" + SEPARATOR_LINE

# Import code quality checker
from testing.code_quality_checker import CodeQualityChecker, QualityMetrics


def _fix_trailing_whitespace() -> None:
    """Fix trailing whitespace in all Python files before running tests."""
    python_files = [
        "connection_resilience.py",
        # Add other files here if needed in the future
    ]

    for file_name in python_files:
        file_path = Path(file_name)
        if not file_path.exists():
            continue

        try:
            # Read the file
            with file_path.open(encoding='utf-8') as f:
                lines = f.readlines()

            # Remove trailing whitespace from each line
            fixed_lines = [line.rstrip() + '\n' if line.strip() else '\n' for line in lines]

            # Only write if changes were made
            if lines != fixed_lines:
                with file_path.open('w', encoding='utf-8') as f:
                    f.writelines(fixed_lines)
        except Exception:
            # Silently skip files that can't be processed
            pass


def _invoke_ruff(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Run Ruff with the provided arguments and return the completed process."""
    command = [sys.executable, "-m", "ruff", *args]
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True, cwd=Path.cwd(), timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"⚠️ LINTER: Ruff command timed out after {timeout}s: {' '.join(args)}")
        # Return a failed CompletedProcess to indicate timeout
        return subprocess.CompletedProcess(args=command, returncode=1, stdout="", stderr=f"Timeout after {timeout}s")


def _ruff_available() -> bool:
    """Return True if Ruff CLI is available in the environment."""
    return _invoke_ruff(["--version"]).returncode == 0


def _print_tail(output: str, *, limit: int = 40) -> None:
    """Print the final lines of command output to avoid overwhelming logs."""
    if not output:
        return
    tail_lines = [line for line in output.splitlines() if line.strip()][-limit:]
    for line in tail_lines:
        print(line)


def _print_nonempty_lines(output: str, *, limit: int = 25) -> None:
    """Print up to ``limit`` non-empty lines from a command's stdout."""
    if not output:
        return
    lines = [line for line in output.splitlines() if line.strip()]
    for line in lines[-limit:]:
        print(line)


@dataclass
class TestExecutionMetrics:
    """Performance metrics for test execution."""

    module_name: str
    duration: float
    success: bool
    test_count: int
    memory_usage_mb: float
    cpu_usage_percent: float
    start_time: str
    end_time: str
    error_message: Optional[str] = None
    quality_metrics: Optional[QualityMetrics] = None


@dataclass
class TestSuitePerformance:
    """Overall test suite performance metrics."""

    total_duration: float
    total_tests: int
    passed_modules: int
    failed_modules: int
    avg_memory_usage: float
    peak_memory_usage: float
    avg_cpu_usage: float
    peak_cpu_usage: float
    parallel_efficiency: float
    optimization_suggestions: list[str]


@dataclass
class TestExecutionConfig:
    """Configuration for test execution."""

    modules_with_descriptions: list[tuple[str, str]]
    discovered_modules: list[str]
    module_descriptions: dict[str, str]
    enable_fast_mode: bool
    enable_monitoring: bool
    enable_benchmark: bool
    enable_integration: bool


@dataclass
class PerformanceMetricsConfig:
    """Configuration for performance metrics printing."""

    all_metrics: list[TestExecutionMetrics]
    total_duration: float
    total_tests_run: int
    passed_count: int
    failed_count: int
    enable_fast_mode: bool
    enable_benchmark: bool


class LogTimingEntry(TypedDict):
    """Structured representation for timing entries in log analysis."""

    matches: int
    total_seconds: float
    avg_per_match: float


class LogAnalysisData(TypedDict):
    """Structured log analysis results."""

    timing: list[LogTimingEntry]
    errors: dict[str, int]
    warnings: int
    pages_processed: int
    cache_hits: int
    api_fetches: int
    highest_page: int


class LogAnalysisError(TypedDict):
    """Log analysis error payload."""

    error: str


# ==============================================
# Test Execution Optimization
# ==============================================


class TestResultCache:
    """Cache test results to skip unchanged modules.

    This class provides hash-based change detection to skip retesting
    unchanged modules, which can significantly speed up iterative
    development workflows.
    """

    CACHE_FILE = Path("Cache/test_results_cache.json")

    @classmethod
    def load_cache(cls) -> dict[str, dict[str, Any]]:
        """Load test result cache from disk."""
        if not cls.CACHE_FILE.exists():
            return {}
        try:
            with cls.CACHE_FILE.open(encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    @classmethod
    def save_cache(cls, cache: dict[str, dict[str, Any]]) -> None:
        """Save test result cache to disk."""
        try:
            cls.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with cls.CACHE_FILE.open("w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        except Exception:
            pass  # Silently fail - caching is optional

    @classmethod
    def get_module_hash(cls, module_name: str) -> Optional[str]:
        """Get hash of module file contents."""
        try:
            import hashlib

            module_path = Path(module_name)
            if not module_path.exists():
                # Try appending .py if it doesn't exist
                module_path = Path(f"{module_name}.py")
                if not module_path.exists():
                    return None

            content = module_path.read_bytes()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return None

    @classmethod
    def should_skip_module(cls, module_name: str, cache: dict[str, dict[str, Any]]) -> bool:
        """Determine if module can be skipped based on cache."""
        if module_name not in cache:
            return False

        cached_entry = cache[module_name]
        if not cached_entry.get("success", False):
            return False  # Always re-run failed tests

        # Ensure we have necessary data to report results
        if "test_count" not in cached_entry:
            return False

        # Check if file has changed
        current_hash = cls.get_module_hash(module_name)
        cached_hash = cached_entry.get("file_hash")
        return bool(current_hash and cached_hash and current_hash == cached_hash)


def optimize_test_order(modules: list[str]) -> list[str]:
    """
    Optimize test execution order for faster feedback.

    Strategy:
    1. Fast tests first (< 1s historical duration)
    2. Recently failed tests next (for quick failure detection)
    3. Slow tests last (can be parallelized)

    Args:
        modules: List of module names to test

    Returns:
        Optimized list of module names
    """
    # Load historical performance data
    metrics_file = Path("Logs/test_metrics_history.json")
    historical_data: dict[str, dict[str, Any]] = {}
    if metrics_file.exists():
        try:
            with metrics_file.open(encoding="utf-8") as f:
                historical_data = json.load(f)
        except Exception:
            pass

    # Categorize modules
    fast_modules: list[tuple[str, float]] = []
    recently_failed: list[tuple[str, float]] = []
    slow_modules: list[tuple[str, float]] = []
    unknown_modules: list[str] = []

    for module in modules:
        if module not in historical_data:
            unknown_modules.append(module)
            continue

        data = historical_data[module]
        duration = data.get("avg_duration", 999)
        last_success = data.get("last_success", True)

        if not last_success:
            recently_failed.append((module, duration))
        elif duration < 1.0:
            fast_modules.append((module, duration))
        else:
            slow_modules.append((module, duration))

    # Sort each category by duration (fastest first)
    fast_modules.sort(key=lambda x: x[1])
    recently_failed.sort(key=lambda x: x[1])
    slow_modules.sort(key=lambda x: x[1])

    # Build optimized order: fast → failed → slow → unknown
    return (
        [m for m, _ in fast_modules] + [m for m, _ in recently_failed] + [m for m, _ in slow_modules] + unknown_modules
    )


def update_test_history(module_name: str, duration: float, success: bool) -> None:
    """
    Update test execution history for smart ordering.

    Args:
        module_name: Name of test module
        duration: Test execution duration in seconds
        success: Whether test passed
    """
    metrics_file = Path("Logs/test_metrics_history.json")
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data
    historical_data: dict[str, dict[str, Any]] = {}
    if metrics_file.exists():
        try:
            with metrics_file.open(encoding="utf-8") as f:
                historical_data = json.load(f)
        except Exception:
            pass

    # Update or create entry
    if module_name not in historical_data:
        historical_data[module_name] = {
            "avg_duration": duration,
            "run_count": 1,
            "last_success": success,
            "last_run": datetime.now().isoformat(),
        }
    else:
        entry = historical_data[module_name]
        run_count = entry.get("run_count", 1)
        avg_duration = entry.get("avg_duration", duration)

        # Update running average
        new_avg = ((avg_duration * run_count) + duration) / (run_count + 1)

        entry["avg_duration"] = new_avg
        entry["run_count"] = run_count + 1
        entry["last_success"] = success
        entry["last_run"] = datetime.now().isoformat()

    # Save updated data
    try:
        with metrics_file.open("w", encoding="utf-8") as f:
            json.dump(historical_data, f, indent=2)
    except Exception:
        pass  # Silently fail


# ==============================================
# End Test Infrastructure Todo #18
# ==============================================


class PerformanceMonitor:
    """Monitor system performance during test execution."""

    def __init__(self) -> None:
        self.process: Any = psutil.Process() if PSUTIL_AVAILABLE and psutil else None
        self.monitoring: bool = False
        self.metrics: list[dict[str, float]] = []
        self.monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self) -> None:
        """Start performance monitoring in background thread."""
        if not self.process:
            return
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self) -> dict[str, float]:
        """Stop monitoring and return aggregated metrics."""
        if not self.process:
            return {"memory_mb": 0.0, "cpu_percent": 0.0}
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        if not self.metrics:
            return {"memory_mb": 0.0, "cpu_percent": 0.0}

        memory_values = [m["memory_mb"] for m in self.metrics]
        cpu_values = [m["cpu_percent"] for m in self.metrics]

        return {
            "memory_mb": sum(memory_values) / len(memory_values),
            "peak_memory_mb": max(memory_values),
            "cpu_percent": sum(cpu_values) / len(cpu_values),
            "peak_cpu_percent": max(cpu_values),
        }

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        if not self.process:
            return
        while self.monitoring:
            try:
                memory_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent()

                self.metrics.append(
                    {"memory_mb": memory_info.rss / (1024 * 1024), "cpu_percent": cpu_percent, "timestamp": time.time()}
                )

                time.sleep(0.1)  # Sample every 100ms
            except Exception:
                break


def run_linter() -> bool:
    """Run Ruff and enforce blocking rules before tests.

    Steps:
    1) Apply ALL safe auto-fixes automatically
    2) Enforce blocking rule set; fail the run if violations remain
    3) Print non-blocking repository statistics
    """
    try:
        # Check if ruff is available
        if not _ruff_available():
            print("🧹 LINTER: Ruff not available, skipping linting checks...")
            return True

        # Step 1: Auto-fix ALL fixable issues (with 60s timeout)
        print("🧹 LINTER: Auto-fixing all fixable linting issues...")
        fix_result = _invoke_ruff(["check", "--fix", "."], timeout=60)
        if fix_result.returncode == 0:
            print("   ✅ All fixable issues resolved")
        else:
            print("   ⚠️  Some issues auto-fixed, checking for critical errors...")

        # Step 2: blocking rule set (only critical errors, with 30s timeout)
        print("🧹 LINTER: Enforcing critical blocking rules (E722,F821,F811,F823)...")
        block_res = _invoke_ruff(
            [
                "check",
                "--select",
                "E722,F821,F811,F823",
                ".",
            ],
            timeout=30,
        )
        if block_res.returncode != 0:
            print("❌ LINTER FAILED (blocking): critical violations found")
            _print_tail(block_res.stdout or block_res.stderr or "")
            return False

        # Step 3: non-blocking diagnostics (excluding PLR2004 and PLC0415, with 30s timeout)
        print("🧹 LINTER: Repository diagnostics (non-blocking summary)...")
        diag_res = _invoke_ruff(
            [
                "check",
                "--statistics",
                "--exit-zero",
                "--ignore=PLR2004,PLC0415",
                ".",
            ],
            timeout=30,
        )
        _print_nonempty_lines(diag_res.stdout)
        return True
    except Exception as e:
        print(f"⚠️ LINTER step skipped due to error: {e}")
        return True


# ============================================================================
# RESERVED FOR FUTURE DEVELOPMENT: Standalone Quality Checks
# This function provides programmatic quality checking outside the test flow.
# Integration pending: Will be connected to CI/CD pipeline or CLI command.
# ============================================================================
def _get_quality_check_files(flags: FeatureFlags) -> list[str]:
    """Get list of files to check for quality."""
    key_files_env = os.getenv("QUALITY_CHECK_FILES")
    if key_files_env:
        return [token.strip() for token in key_files_env.split(",") if token.strip()]

    key_files = [
        "actions/action10.py",
        "utils.py",
        "main.py",
        "python_best_practices.py",
        "code_quality_checker.py",
    ]
    if flags.is_enabled("EXTENDED_QUALITY_SCOPE", default=False):
        key_files.extend(["run_all_tests.py", "core/feature_flags.py"])
    return key_files


def _get_quality_threshold(flags: FeatureFlags) -> float:
    """Get quality check threshold from config."""
    default_threshold = 70.0
    env_threshold = os.getenv("QUALITY_CHECK_THRESHOLD")
    try:
        threshold = float(env_threshold) if env_threshold else default_threshold
    except ValueError:
        threshold = default_threshold
    if flags.is_enabled("STRICT_QUALITY_CHECKS", default=False):
        threshold = max(threshold, 80.0)
    return threshold


def run_quality_checks(flags: FeatureFlags | None = None) -> tuple[bool, list[tuple[str, float]]]:
    """Run Python best practices quality checks."""
    try:
        from testing.code_quality_checker import CodeQualityChecker

        checker = CodeQualityChecker()
        current_dir = Path()
        flags = flags or bootstrap_feature_flags()

        if flags.is_enabled("DISABLE_QUALITY_CHECKS", default=False):
            return True, []

        key_files = _get_quality_check_files(flags)
        threshold = _get_quality_threshold(flags)
        quality_scores: list[tuple[str, float]] = []
        total_score = 0.0
        files_checked = 0

        for file_name in key_files:
            file_path = current_dir / file_name
            if file_path.exists():
                metrics = checker.check_file(file_path)
                total_score += metrics.quality_score
                files_checked += 1
                quality_scores.append((file_name, metrics.quality_score))
            else:
                quality_scores.append((file_name, 0.0))

        if files_checked > 0 and (total_score / files_checked) < threshold:
            return False, quality_scores
        return True, quality_scores

    except Exception:
        return True, []


def _should_skip_system_file(python_file: Path) -> bool:
    """Check if file should be skipped (system files, test runner, etc.)."""
    # Skip specific system files
    if python_file.name in {"run_all_tests.py", "main.py", "__main__.py"}:
        return True

    # Only skip root-level __init__.py, not package __init__.py files
    # Package __init__.py files (like core/__init__.py) may contain tests
    if python_file.name == "__init__.py":
        # Check if this is a root-level __init__.py (no parent directory except project root)
        # Allow package __init__.py files in subdirectories
        return python_file.parent == Path(__file__).parent

    return False


def _should_skip_cache_or_temp_file(python_file: Path) -> bool:
    """Check if file should be skipped (cache, backup, temp, venv, archive)."""
    file_path_str = str(python_file)

    # Check backup files
    if python_file.name.endswith("_backup.py") or "backup_before_migration" in file_path_str:
        return True

    # Check temp files
    if python_file.name.startswith("temp_") or python_file.name.endswith("_temp.py"):
        return True

    # Check legacy/old files
    if "_old" in python_file.name:
        return True

    # Check system directories
    system_dirs = ["__pycache__", ".venv", "site-packages", "Cache", "Logs", "archive"]
    return any(d in file_path_str for d in system_dirs)


def _should_skip_demo_file(python_file: Path) -> bool:
    """Check if file should be skipped (demo/prototype scripts)."""
    demo_markers = ["demo", "prototype", "experimental", "sandbox"]
    name_lower = python_file.name.lower()
    return any(marker in name_lower for marker in demo_markers)


def _should_skip_interactive_file(python_file: Path) -> bool:
    """Check if file should be skipped (interactive modules)."""
    return python_file.name in {"db_viewer.py", "test_program_executor.py"}


def _has_test_function(content: str) -> bool:
    """Check if file content has the standardized test function."""
    return (
        "def run_comprehensive_tests" in content or "run_comprehensive_tests = create_standard_test_runner" in content
    )


def discover_test_modules() -> list[str]:
    """
    Discover all Python modules that contain tests by scanning the project directory.

    Returns a list of module paths that contain the run_comprehensive_tests() function,
    which indicates they follow the standardized testing framework.
    """
    project_root = Path(__file__).parent
    test_modules: list[str] = []

    # Get all Python files in the project
    for python_file in project_root.rglob("*.py"):
        # Skip various file types
        if _should_skip_system_file(python_file):
            continue
        if _should_skip_cache_or_temp_file(python_file):
            continue
        if _should_skip_demo_file(python_file):
            continue
        if _should_skip_interactive_file(python_file):
            continue

        # Check if the file has the standardized test function
        try:
            with python_file.open(encoding="utf-8") as f:
                content = f.read()
                if _has_test_function(content):
                    # Convert to relative path from project root
                    relative_path = python_file.relative_to(project_root)
                    test_modules.append(str(relative_path))

        except (UnicodeDecodeError, PermissionError):
            # Skip files that can't be read
            continue

    return sorted(test_modules)


def _should_skip_line(stripped: str) -> bool:
    """Check if a line should be skipped when parsing docstrings."""
    return stripped.startswith('#') or not stripped


def _extract_docstring_start(stripped: str, docstring_lines: list[str]) -> bool:
    """Extract content after opening docstring quotes. Returns True if docstring started."""
    if '"""' not in stripped:
        return False

    # Extract content after opening quotes
    after_quotes = stripped.split('"""', 1)[1].strip()
    if after_quotes:
        docstring_lines.append(after_quotes)
    return True


def _extract_docstring_end(stripped: str, docstring_lines: list[str]) -> bool:
    """Extract content before closing docstring quotes. Returns True if docstring ended."""
    if '"""' not in stripped:
        return False

    # End of docstring - extract content before closing quotes
    before_quotes = stripped.split('"""', maxsplit=1)[0].strip()
    if before_quotes:
        docstring_lines.append(before_quotes)
    return True


def _parse_docstring_lines(lines: list[str]) -> list[str]:
    """Parse file lines to extract docstring content."""
    in_docstring = False
    docstring_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Skip shebang and comments at the top
        if _should_skip_line(stripped):
            continue

        # Look for start of docstring
        if not in_docstring:
            if _extract_docstring_start(stripped, docstring_lines):
                in_docstring = True
            continue

        # If we're in docstring, collect lines until closing quotes
        if _extract_docstring_end(stripped, docstring_lines):
            break

        # Regular docstring line
        if stripped:
            docstring_lines.append(stripped)

    return docstring_lines


def _is_valid_description_line(line: str) -> bool:
    """Check if a line is a valid description (not a separator, long enough)."""
    return bool(line) and not line.startswith('=') and len(line) > 10


def _clean_description_text(description: str, module_base: str) -> str:
    """Clean up description text by removing redundant patterns."""
    # Remove leading dashes and clean up
    if description.startswith('-'):
        description = description[1:].strip()
    if description.startswith('.py'):
        description = description[3:].strip()

    # Remove redundant module name patterns
    words_to_remove = [module_base.lower(), 'module', 'py']
    for word in words_to_remove:
        if description.lower().startswith(word):
            description = description[len(word) :].strip()
            if description.startswith('-'):
                description = description[1:].strip()

    return description


def _extract_first_meaningful_line(docstring_lines: list[str], module_path: str) -> str | None:
    """Extract the first meaningful line from docstring as description."""
    module_base = module_path.replace('.py', '').replace('/', '').replace('\\', '')

    for line in docstring_lines:
        if not _is_valid_description_line(line):
            continue

        # Clean up common patterns
        description = line.replace(module_base, '').strip()
        description = description.replace(' - ', ' - ').strip()
        return _clean_description_text(description, module_base)

    return None


def extract_module_description(module_path: str) -> str | None:
    """Extract the first line of a module's docstring for use as description."""
    try:
        # Read the file and look for the module docstring
        from pathlib import Path

        with Path(module_path).open(encoding='utf-8') as f:
            content = f.read()

        # Parse docstring from file content
        lines = content.split('\n')
        docstring_lines = _parse_docstring_lines(lines)

        # Return the first meaningful line as description
        if docstring_lines:
            return _extract_first_meaningful_line(docstring_lines, module_path)

        return None

    except Exception:
        return None


def _generate_module_description(module_name: str, description: str | None = None) -> str:
    """Generate a meaningful description for a module based on its name."""
    if description:
        return description

    # Create a meaningful description based on module name
    result = None

    if "core/" in module_name:
        component = module_name.replace("core/", "").replace(".py", "").replace("_", " ").title()
        result = f"Core {component} functionality"
    elif "config/" in module_name:
        component = module_name.replace("config/", "").replace(".py", "").replace("_", " ").title()
        result = f"Configuration {component} management"
    elif "action" in module_name:
        action_name = module_name.replace(".py", "").replace("_", " ").title()
        result = f"{action_name} automation"
    elif module_name.endswith("_utils.py"):
        util_type = module_name.replace("_utils.py", "").replace("_", " ").title()
        result = f"{util_type} utility functions"
    elif module_name.endswith("_manager.py"):
        manager_type = module_name.replace("_manager.py", "").replace("_", " ").title()
        result = f"{manager_type} management system"
    elif module_name.endswith("_cache.py"):
        cache_type = module_name.replace("_cache.py", "").replace("_", " ").title()
        result = f"{cache_type} caching system"
    else:
        # Generic fallback
        clean_name = module_name.replace(".py", "").replace("_", " ").title()
        result = f"{clean_name} module functionality"

    return result


def _extract_marker_count(line: str, marker: str) -> Optional[int]:
    """Return integer count following a marker like '✅ Passed:'."""
    try:
        return int(line.split(marker)[1].split()[0])
    except (ValueError, IndexError):
        return None


def _find_failed_count(stdout_lines: list[str]) -> int:
    """Search stdout for a failed-test count marker."""
    for candidate in stdout_lines:
        failed = _extract_marker_count(candidate, "❌ Failed:")
        if failed is not None:
            return failed
    return 0


def _try_pattern_passed_failed(stdout_lines: list[str]) -> str:
    """Pattern 1: Look for '✅ Passed: X' and '❌ Failed: Y'."""
    for line in stdout_lines:
        if "✅ Passed:" not in line:
            continue

        passed = _extract_marker_count(line, "✅ Passed:")
        if passed is None:
            continue

        failed = _extract_marker_count(line, "❌ Failed:")
        if failed is None:
            failed = _find_failed_count(stdout_lines)

        return f"{passed + failed} tests"

    return "Unknown"


def _try_pattern_tests_passed(stdout_lines: list[str]) -> str:
    """Pattern 2: Look for 'X/Y tests passed' or 'Results: X/Y'."""
    for line in stdout_lines:
        if "tests passed" in line and "/" in line:
            try:
                # Extract from "📊 Results: 3/3 tests passed"
                parts = line.split("/")
                if len(parts) >= 2:
                    total = parts[1].split()[0]
                    return f"{total} tests"
            except (ValueError, IndexError):
                continue
    return "Unknown"


def _try_pattern_passed_failed_ansi(stdout_lines: list[str]) -> str:
    """Pattern 3: Look for Passed/Failed format with ANSI cleanup."""
    passed_count = None
    failed_count = None
    for line in stdout_lines:
        # Remove ANSI color codes and whitespace
        clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()

        if "✅ Passed:" in clean_line:
            try:
                passed_count = int(clean_line.split("✅ Passed:")[1].strip())
            except (ValueError, IndexError):
                continue
        elif "❌ Failed:" in clean_line:
            try:
                failed_count = int(clean_line.split("❌ Failed:")[1].strip())
            except (ValueError, IndexError):
                continue

    if passed_count is not None and failed_count is not None:
        return f"{passed_count + failed_count} tests"
    if passed_count is not None:
        return f"{passed_count}+ tests"
    return "Unknown"


def _try_pattern_unittest_ran(stdout_lines: list[str]) -> str:
    """Pattern 4: Look for Python unittest format 'Ran X tests in Y.Zs'."""
    for line in stdout_lines:
        clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        if "Ran" in clean_line and "tests in" in clean_line:
            try:
                # Extract from "Ran 24 tests in 0.458s"
                parts = clean_line.split()
                ran_index = parts.index("Ran")
                if ran_index + 1 < len(parts):
                    count = int(parts[ran_index + 1])
                    return f"{count} tests"
            except (ValueError, IndexError):
                continue
    return "Unknown"


def _try_pattern_numbered_tests(stdout_lines: list[str]) -> str:
    """Pattern 5: Look for numbered test patterns like 'Test 1:', 'Test 2:', etc."""
    test_numbers: set[int] = set()
    for line in stdout_lines:
        clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        # Look for patterns like "📋 Test 1:", "Test 2:", "• Test 3:"
        match = re.search(
            r"(?:�|•|\*|-|\d+\.?)\s*Test\s+(\d+):",
            clean_line,
            re.IGNORECASE,
        )
        if match:
            test_numbers.add(int(match.group(1)))

    if test_numbers:
        return f"{len(test_numbers)} tests"
    return "Unknown"


def _try_pattern_number_followed_by_test(stdout_lines: list[str]) -> str:
    """Pattern 6: Look for any number followed by 'test' or 'tests'."""
    for line in stdout_lines:
        clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        # Look for patterns like "5 tests", "10 test cases", "3 test functions"
        match = re.search(
            r"(\d+)\s+tests?(?:\s+(?:cases?|functions?|passed|completed))?",
            clean_line,
            re.IGNORECASE,
        )
        if match:
            count = int(match.group(1))
            return f"{count} tests"
    return "Unknown"


def _try_pattern_all_tests_completed(stdout_lines: list[str]) -> str:
    """Pattern 7: Look for test completion messages with counts."""
    for line in stdout_lines:
        clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        # Look for patterns like "All X tests passed", "X operations completed"
        match = re.search(
            r"(?:All|Total)\s+(\d+)\s+(?:tests?|operations?|checks?)\s+(?:passed|completed|successful)",
            clean_line,
            re.IGNORECASE,
        )
        if match:
            count = int(match.group(1))
            return f"{count} tests"
    return "Unknown"


def _try_pattern_all_tests_passed_with_counts(stdout_lines: list[str]) -> str:
    """Pattern 8: Look for 'ALL TESTS PASSED' with counts."""
    for line in stdout_lines:
        if "ALL TESTS PASSED" in line or "Status: ALL TESTS PASSED" in line:
            # Look for nearby lines with test counts
            for other_line in stdout_lines:
                if "Passed:" in other_line and other_line.count(":") >= 1:
                    try:
                        count = int(other_line.split("Passed:")[1].split()[0])
                        return f"{count} tests"
                    except (ValueError, IndexError):
                        continue
            break
    return "Unknown"


def _extract_count_from_line(line: str, keyword: str) -> Optional[int]:
    """Extract count from line containing keyword.

    Args:
        line: Line to parse
        keyword: Keyword to search for (e.g., "Passed:", "Failed:")

    Returns:
        Extracted count or None if not found
    """
    clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
    if keyword not in clean_line:
        return None
    try:
        return int(clean_line.split(keyword)[1].split()[0])
    except (ValueError, IndexError):
        return None


def _find_passed_failed_counts(stdout_lines: list[str]) -> tuple[Optional[int], Optional[int]]:
    """Find passed and failed counts in output lines.

    Args:
        stdout_lines: List of output lines to search

    Returns:
        Tuple of (passed_count, failed_count) or (None, None) if not found
    """
    passed = None
    failed = None
    for line in stdout_lines:
        if passed is None:
            passed = _extract_count_from_line(line, "Passed:")
        if failed is None:
            failed = _extract_count_from_line(line, "Failed:")
        if passed is not None and failed is not None:
            break
    return passed, failed


def _try_pattern_passed_failed_counts_any(stdout_lines: list[str]) -> str:
    """Pattern 9: Fallback - sum any 'Passed:' and 'Failed:' counts anywhere in output."""
    passed, failed = _find_passed_failed_counts(stdout_lines)

    if passed is not None or failed is not None:
        total = (passed or 0) + (failed or 0)
        if total >= 0:
            return f"{total} tests"
    return "Unknown"


def _extract_test_count_from_output(all_output_lines: list[str]) -> str:
    """Extract test count from output using multiple patterns."""
    if not all_output_lines:
        return "Unknown"

    # Try each pattern in order
    patterns: list[Callable[[list[str]], str]] = [
        _try_pattern_passed_failed,
        _try_pattern_tests_passed,
        _try_pattern_passed_failed_ansi,
        _try_pattern_unittest_ran,
        _try_pattern_numbered_tests,
        _try_pattern_number_followed_by_test,
        _try_pattern_all_tests_completed,
        _try_pattern_all_tests_passed_with_counts,
        _try_pattern_passed_failed_counts_any,
    ]

    for pattern_func in patterns:
        result = pattern_func(all_output_lines)
        if result != "Unknown":
            return result

    return "Unknown"


def _check_for_failures_in_output(success: bool, stdout: str) -> bool:
    """Check output for failure indicators."""
    if not success or not stdout:
        return success

    failure_indicators = [
        "❌ FAILED",
        "Status: FAILED",
        "AssertionError:",
        "Exception occurred:",
        "Test failed:",
        "❌ Failed: ",
        "CRITICAL ERROR",
        "FATAL ERROR",
    ]

    # Only mark as failed if we find actual failure indicators
    # Exclude lines that are just showing "Failed: 0" (which means 0 failures)
    stdout_lines = stdout.split("\n")
    for line in stdout_lines:
        for indicator in failure_indicators:
            if indicator in line and not ("Failed: 0" in line or "❌ Failed: 0" in line):
                return False
    return success


def _format_quality_info(quality_metrics: Optional[QualityMetrics]) -> str:
    """Format quality metrics into a display string."""
    if not quality_metrics:
        return ""

    score = quality_metrics.quality_score
    violations_count = len(quality_metrics.violations) if quality_metrics.violations else 0
    if score < 70:
        return f" | Quality: {score:.1f}/100 ⚠️ ({violations_count} issues)"
    if score < 95:
        return f" | Quality: {score:.1f}/100 📊 ({violations_count} issues)"
    return f" | Quality: {score:.1f}/100 ✅"


def _categorize_violation(violation: str) -> str:
    """Categorize a violation by type."""
    if "too long" in violation:
        return "Length"
    if "too complex" in violation:
        return "Complexity"
    if "missing type hint" in violation:
        return "Type Hints"
    return "Other"


def _format_violation_message(violation: str) -> str:
    """Format a violation message for display."""
    if "Function '" in violation and "'" in violation:
        func_name = violation.split("Function '")[1].split("'")[0]
        issue_type = violation.split("' ")[1] if "' " in violation else violation
        return f"• {func_name}: {issue_type}"
    return f"• {violation}"


def _print_quality_violations(quality_metrics: Optional[QualityMetrics]) -> None:
    """Print quality violation details."""
    if not quality_metrics or quality_metrics.quality_score >= 95 or not quality_metrics.violations:
        return

    print("   🔍 Quality Issues:")
    # Group violations by type for better readability
    violation_types: dict[str, list[str]] = {}
    for violation in quality_metrics.violations[:5]:  # Show first 5
        vtype = _categorize_violation(violation)
        violation_types.setdefault(vtype, []).append(violation)

    for vtype, violations in violation_types.items():
        print(f"      {vtype}: {len(violations)} issue(s)")
        for violation in violations[:2]:  # Show first 2 of each type
            print(f"        {_format_violation_message(violation)}")

    if len(quality_metrics.violations) > 5:
        print(f"      ... and {len(quality_metrics.violations) - 5} more issues")


def _extract_numeric_test_count(test_count: str) -> int:
    """Extract numeric test count from string format."""
    if test_count == "Unknown":
        return 0

    try:
        match = re.search(r"(\d+)", test_count)
        if match:
            return int(match.group(1))
    except (ValueError, AttributeError):
        pass
    return 0


def _print_failure_details(result: subprocess.CompletedProcess[str], failure_indicators: list[str]) -> None:
    """Print failure details from test output."""
    print("   🚨 Failure Details:")
    if result.stderr:
        error_lines = result.stderr.strip().split("\n")
        # Print debug lines to help diagnose issues
        for line in error_lines:
            if line.startswith("DEBUG:") or "CRITICAL:" in line:
                print(f"      {line}")

        for line in error_lines[-5:]:  # Show last 5 error lines for better context
            print(f"      {line}")
    if result.stdout and any(indicator in result.stdout for indicator in failure_indicators):
        stdout_lines = result.stdout.strip().split("\n")
        failure_lines = [line for line in stdout_lines if any(indicator in line for indicator in failure_indicators)]
        for line in failure_lines[-2:]:  # Show last 2 failure lines
            print(f"      {line}")


def _build_test_command(
    module_name: str,
    coverage: bool,
    db_path: Optional[str] = None,
    cache_path: Optional[str] = None,
    log_path: Optional[str] = None,
    user_data_dir: Optional[str] = None,
) -> tuple[list[str], dict[str, str]]:
    """Build the command and environment for running tests."""
    cmd = [sys.executable]

    # Always pass environment to subprocess to ensure SKIP_LIVE_API_TESTS is inherited
    env: dict[str, str] = dict(os.environ)

    # Force modules to execute their embedded tests instead of interactive CLIs
    env["RUN_MODULE_TESTS"] = "1"

    # Set custom database path if provided (for isolation)
    if db_path:
        env["DATABASE_FILE"] = db_path

    # Set custom cache path if provided (for isolation)
    if cache_path:
        env["CACHE_DIR"] = cache_path
        # print(f"DEBUG: Setting CACHE_DIR={cache_path} for {module_name}")

    # Set custom log file if provided (for isolation)
    if log_path:
        env["LOG_FILE"] = log_path

    # Set custom user data dir if provided (for isolation)
    if user_data_dir:
        env["CHROME_USER_DATA_DIR"] = user_data_dir

    # For modules with internal test suite, set env var to trigger test output
    suite_env_modules = {
        "prompt_telemetry.py",
        "quality_regression_gate.py",
        "ui/review_server.py",
        "ui\\review_server.py",
    }
    if module_name in suite_env_modules or module_name.endswith("review_server.py"):
        env["RUN_INTERNAL_TESTS"] = "1"

    # Convert file path to module name for proper import resolution
    # e.g., "core\common_params.py" -> "core.common_params"
    module_import_name = module_name.replace("\\", ".").replace("/", ".").removesuffix(".py")

    if coverage:
        cmd += ["-m", "coverage", "run", "--append", "-m", module_import_name]
    else:
        # Use -m flag to run as a module, ensuring proper import resolution
        cmd += ["-m", module_import_name]

    return cmd, env


def _run_quality_analysis(module_name: str):
    """Run quality analysis on a module."""
    try:
        module_path = Path(module_name)
        if module_path.exists() and module_path.suffix == '.py':
            quality_checker = CodeQualityChecker()
            return quality_checker.check_file(module_path)
    except Exception:
        # Quality check failed, continue without it
        pass
    return None


def _create_test_metrics(
    module_name: str, test_result: dict[str, Any], quality_metrics: Optional[QualityMetrics] = None
) -> TestExecutionMetrics:
    """
    Create TestExecutionMetrics object from test result data.

    Args:
        module_name: Name of the module being tested
        test_result: Dict containing duration, success, test_count, perf_metrics,
                     result, start_time, end_time
        quality_metrics: Optional quality assessment metrics
    """
    return TestExecutionMetrics(
        module_name=module_name,
        duration=test_result["duration"],
        success=test_result["success"],
        test_count=test_result["test_count"],
        memory_usage_mb=test_result["perf_metrics"].get("memory_mb", 0.0),
        cpu_usage_percent=test_result["perf_metrics"].get("cpu_percent", 0.0),
        start_time=test_result["start_time"],
        end_time=test_result["end_time"],
        error_message=test_result["result"].stderr
        if not test_result["success"] and test_result["result"].stderr
        else None,
        quality_metrics=quality_metrics,
    )


def _create_error_metrics(module_name: str, error_message: str) -> TestExecutionMetrics:
    """Create TestExecutionMetrics for error case."""
    return TestExecutionMetrics(
        module_name=module_name,
        duration=0.0,
        success=False,
        test_count=0,
        memory_usage_mb=0.0,
        cpu_usage_percent=0.0,
        start_time=datetime.now().isoformat(),
        end_time=datetime.now().isoformat(),
        error_message=error_message,
    )


def _run_test_subprocess(
    module_name: str,
    coverage: bool,
    db_path: Optional[str] = None,
    cache_path: Optional[str] = None,
    log_path: Optional[str] = None,
    user_data_dir: Optional[str] = None,
) -> tuple[subprocess.CompletedProcess[str], float, str]:
    """Run the test subprocess and return result, duration, and timestamp."""
    start_time = time.time()
    start_datetime = datetime.now().isoformat()

    cmd, env = _build_test_command(module_name, coverage, db_path, cache_path, log_path, user_data_dir)

    # Set timeout for subprocess (120 seconds for modules with many tests)
    # This prevents tests from hanging indefinitely
    # Some modules like action8_messaging (47 tests) and gedcom_utils (17 tests) need more time
    # action8_messaging has 47 tests and needs extra time for comprehensive testing
    # actions.action10 can take ~120s, so we give it more time
    if module_name == "action8_messaging.py":
        timeout_seconds = 180
    elif module_name in {"actions.action10.py", "actions.action10"}:
        timeout_seconds = 240
    elif "session_manager.py" in module_name:
        timeout_seconds = 240  # Session manager tests can be slow (116s+)
    elif "review_server.py" in module_name:
        timeout_seconds = 240  # Review server tests can be slow
    else:
        timeout_seconds = 120

    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            cmd, check=False, capture_output=True, text=True, cwd=Path.cwd(), env=env, timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired:
        # Create a fake CompletedProcess to indicate timeout
        result = subprocess.CompletedProcess[str](
            args=cmd,
            returncode=124,  # Standard timeout exit code
            stdout="",
            stderr=f"Test subprocess timed out after {timeout_seconds}s",
        )

    duration = time.time() - start_time
    return result, duration, start_datetime


def _analyze_test_output(result: subprocess.CompletedProcess[str]) -> tuple[bool, str, int]:
    """Analyze test output and return success status, test count string, and numeric count."""
    # Collect all output lines
    all_output_lines: list[str] = []
    if result.stdout:
        all_output_lines.extend(result.stdout.split("\n"))
    if result.stderr:
        all_output_lines.extend(result.stderr.split("\n"))

    # Determine success and extract test count
    success = result.returncode == 0
    success = _check_for_failures_in_output(success, result.stdout)
    test_count = _extract_test_count_from_output(all_output_lines)
    numeric_test_count = _extract_numeric_test_count(test_count)

    return success, test_count, numeric_test_count


def _print_test_result(
    success: bool,
    duration: float,
    test_count: str,
    quality_metrics: Optional[QualityMetrics],
    result: subprocess.CompletedProcess[str],
) -> None:
    """Print test result summary with quality info and failure details."""
    status = "✅ PASSED" if success else "❌ FAILED"
    quality_info = _format_quality_info(quality_metrics)
    print(f"   {status} | Duration: {duration:.2f}s | {test_count}{quality_info}")

    _print_quality_violations(quality_metrics)

    if not success:
        failure_indicators = [
            "❌ FAILED",
            "Status: FAILED",
            "AssertionError:",
            "Exception occurred:",
            "Test failed:",
            "❌ Failed: ",
            "CRITICAL ERROR",
            "FATAL ERROR",
        ]
        _print_failure_details(result, failure_indicators)


def run_module_tests(
    module_name: str,
    description: str | None = None,
    enable_monitoring: bool = False,
    coverage: bool = False,
    db_path: Optional[str] = None,
    cache_path: Optional[str] = None,
    log_path: Optional[str] = None,
    user_data_dir: Optional[str] = None,
) -> tuple[bool, int, Optional[TestExecutionMetrics]]:
    """Run tests for a specific module with optional performance monitoring."""
    # Print description
    desc = _generate_module_description(module_name, description)
    print(f"   📝 {desc}")

    try:
        # Start performance monitoring
        monitor = PerformanceMonitor() if enable_monitoring and PSUTIL_AVAILABLE else None
        if monitor:
            monitor.start_monitoring()

        # Run the test subprocess
        result, duration, start_datetime = _run_test_subprocess(
            module_name, coverage, db_path, cache_path, log_path, user_data_dir
        )
        end_datetime = datetime.now().isoformat()

        # Collect performance and quality metrics
        perf_metrics = monitor.stop_monitoring() if monitor else {}
        quality_metrics = _run_quality_analysis(module_name)

        # Analyze output
        success, test_count, numeric_test_count = _analyze_test_output(result)

        # Print results
        _print_test_result(success, duration, test_count, quality_metrics, result)

        # Create metrics object (always include quality metrics for final summary)
        metrics: Optional[TestExecutionMetrics] = None
        test_result: dict[str, Any] = {
            "duration": duration,
            "success": success,
            "test_count": numeric_test_count,
            "perf_metrics": perf_metrics,
            "result": result,
            "start_time": start_datetime,
            "end_time": end_datetime,
        }
        metrics = _create_test_metrics(module_name, test_result, quality_metrics)

        return success, numeric_test_count, metrics

    except Exception as e:
        print(f"   ❌ FAILED | Error: {e}")
        error_metrics = _create_error_metrics(module_name, str(e)) if enable_monitoring else None
        return False, 0, error_metrics


def _monitor_and_collect_results(
    future_to_module: dict[
        concurrent.futures.Future[tuple[bool, int, Optional[TestExecutionMetrics]]], tuple[str, str]
    ],
) -> tuple[list[TestExecutionMetrics], int, int]:
    """Monitor running tests and collect results."""
    all_metrics: list[TestExecutionMetrics] = []
    passed_count = 0
    total_test_count = 0
    pending_futures = set(future_to_module.keys())

    while pending_futures:
        # Wait for the next future to complete, with a timeout to print status
        done, _ = concurrent.futures.wait(
            pending_futures,
            timeout=15.0,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )

        if not done:
            # No test finished in the last 15 seconds
            print(f"\n   ⏳ Still running ({len(pending_futures)} tests):")
            # Sort by module name for consistent output
            running_modules = sorted([future_to_module[f][0] for f in pending_futures])
            for mod in running_modules[:5]:
                print(f"      • {mod}")
            if len(running_modules) > 5:
                print(f"      ... and {len(running_modules) - 5} more")
            continue

        for future in done:
            pending_futures.remove(future)
            try:
                success, test_count, metrics = future.result()
                if success:
                    passed_count += 1
                total_test_count += test_count

                if metrics:
                    all_metrics.append(metrics)
            except Exception as e:
                module = future_to_module[future][0]
                print(f"   ❌ FAILED | {module} | Error: {e}")

    return all_metrics, passed_count, total_test_count


def run_tests_parallel(
    modules_with_descriptions: list[tuple[str, str]], enable_monitoring: bool = False, coverage: bool = False
) -> tuple[list[TestExecutionMetrics], int, int]:
    """Run tests in parallel for improved performance."""
    # Determine optimal number of workers (don't exceed CPU count)
    cpu_count = (psutil.cpu_count() if PSUTIL_AVAILABLE and psutil else os.cpu_count()) or 1
    max_workers = min(len(modules_with_descriptions), cpu_count)

    # Create a temporary directory for isolated databases
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"   📂 Created temporary directory for isolated databases: {temp_dir}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all test jobs
            future_to_module = {}
            for module, desc in modules_with_descriptions:
                # Create a unique database path for this module
                db_path = str(Path(temp_dir) / f"{module.replace('.', '_')}.db")

                # Create a unique cache path for this module
                cache_path = str(Path(temp_dir) / f"{module.replace('.', '_')}_cache")

                # Create a unique log path for this module
                log_path = str(Path(temp_dir) / f"{module.replace('.', '_')}.log")

                # Create a unique user data dir for this module
                user_data_dir = str(Path(temp_dir) / f"{module.replace('.', '_')}_chrome_profile")

                # Submit the job with the isolated database path
                future = executor.submit(
                    run_module_tests,
                    module,
                    desc,
                    enable_monitoring,
                    coverage,
                    db_path,
                    cache_path,
                    log_path,
                    user_data_dir,
                )
                future_to_module[future] = (module, desc)

            return _monitor_and_collect_results(future_to_module)


def save_performance_metrics(metrics: list[TestExecutionMetrics], suite_performance: TestSuitePerformance):
    """Save performance metrics to file for trend analysis."""
    try:
        metrics_file = Path("test_performance_metrics.json")

        # Load existing metrics if file exists
        existing_data: list[dict[str, Any]] = []
        if metrics_file.exists():
            try:
                with metrics_file.open(encoding="utf-8") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []

        # Add new metrics
        new_entry: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "suite_performance": asdict(suite_performance),
            "module_metrics": [asdict(m) for m in metrics],
        }
        existing_data.append(new_entry)

        # Keep only last 50 runs to prevent file from growing too large
        if len(existing_data) > 50:
            existing_data = existing_data[-50:]

        # Save updated metrics
        with metrics_file.open('w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2)

        print(f"📊 Performance metrics saved to {metrics_file}")

    except Exception as e:
        print(f"⚠️  Failed to save performance metrics: {e}")


def _check_slow_tests(metrics: list[TestExecutionMetrics], suggestions: list[str]) -> None:
    """Check for slow tests and add suggestion if found."""
    slow_tests = [m for m in metrics if m.duration > 10.0]
    if slow_tests:
        slow_tests.sort(key=lambda x: x.duration, reverse=True)
        msg = f"🐌 {len(slow_tests)} slow tests detected (>10s). Slowest: {slow_tests[0].module_name} ({slow_tests[0].duration:.1f}s)"
        suggestions.append(msg)


def _check_memory_usage(metrics: list[TestExecutionMetrics], suggestions: list[str]) -> None:
    """Check for memory-intensive tests and add suggestion if found."""
    high_memory_tests = [m for m in metrics if m.memory_usage_mb > 100.0]
    if high_memory_tests:
        high_memory_tests.sort(key=lambda x: x.memory_usage_mb, reverse=True)
        msg = f"🧠 {len(high_memory_tests)} memory-intensive tests detected (>100MB). Highest: {high_memory_tests[0].module_name} ({high_memory_tests[0].memory_usage_mb:.1f}MB)"
        suggestions.append(msg)


def _check_cpu_usage(metrics: list[TestExecutionMetrics], suggestions: list[str]) -> None:
    """Check for CPU-intensive tests and add suggestion if found."""
    high_cpu_tests = [m for m in metrics if m.cpu_usage_percent > 50.0]
    if high_cpu_tests:
        suggestions.append(f"⚡ {len(high_cpu_tests)} CPU-intensive tests detected (>50% CPU)")


def _check_parallel_execution(metrics: list[TestExecutionMetrics], suggestions: list[str]) -> None:
    """Check if parallel execution would help and add suggestion if found."""
    total_duration = sum(m.duration for m in metrics)
    if total_duration > 60.0:
        suggestions.append("🚀 Consider using --fast flag for parallel execution to reduce total runtime")


def analyze_performance_trends(metrics: list[TestExecutionMetrics]) -> list[str]:
    """Analyze performance metrics and provide optimization suggestions."""
    suggestions: list[str] = []

    if not metrics:
        return suggestions

    # Analyze different performance aspects
    _check_slow_tests(metrics, suggestions)
    _check_memory_usage(metrics, suggestions)
    _check_cpu_usage(metrics, suggestions)
    _check_parallel_execution(metrics, suggestions)

    return suggestions


@dataclass
class TestEnvironment:
    """Command-line environment toggles for test execution."""

    enable_fast_mode: bool
    enable_benchmark: bool
    enable_monitoring: bool
    enable_integration: bool
    enable_log_analysis: bool
    enable_quality_checks: bool


def _setup_test_environment() -> TestEnvironment:
    """
    Setup test environment and parse command line arguments.

    Returns:
        TestEnvironment dataclass with toggle values
    """
    # Parse command line arguments
    enable_fast_mode = "--fast" in sys.argv
    enable_benchmark = "--benchmark" in sys.argv
    enable_integration = "--integration" in sys.argv
    enable_log_analysis = "--analyze-logs" in sys.argv
    enable_quality_checks = "--quality" in sys.argv or "--quality-checks" in sys.argv
    enable_monitoring = enable_benchmark or enable_fast_mode

    if enable_monitoring and not PSUTIL_AVAILABLE:
        print("⚠️ PERFORMANCE: psutil not installed; disabling monitoring features.")
        enable_monitoring = False

    # Handle SKIP_LIVE_API_TESTS based on mode
    # Integration mode enables live API tests for full end-to-end validation
    if enable_integration:
        # Unset SKIP_LIVE_API_TESTS to allow live browser/API tests
        os.environ.pop("SKIP_LIVE_API_TESTS", None)
        os.environ["SKIP_LIVE_API_TESTS"] = "false"
        print("🌐 LIVE API MODE: Browser and API tests will execute with real sessions")
    else:
        # Set environment variable to skip live API tests that require browser/network
        # Note: Some modules (action8_messaging, gedcom_utils) have tests that work better with live sessions
        # but should still complete within timeout even when skipped
        os.environ["SKIP_LIVE_API_TESTS"] = "true"

    # Set environment variable to skip slow simulation tests (724-page workload, etc.)
    # unless --slow flag is provided
    if "--slow" not in sys.argv:
        os.environ["SKIP_SLOW_TESTS"] = "true"
    else:
        os.environ.pop("SKIP_SLOW_TESTS", None)
        print("🐢 SLOW TESTS: Including slow simulation tests")

    return TestEnvironment(
        enable_fast_mode=enable_fast_mode,
        enable_benchmark=enable_benchmark,
        enable_monitoring=enable_monitoring,
        enable_integration=enable_integration,
        enable_log_analysis=enable_log_analysis,
        enable_quality_checks=enable_quality_checks,
    )


def _print_test_header(enable_fast_mode: bool, enable_benchmark: bool, enable_integration: bool) -> None:
    """Print test suite header with mode information."""
    print("\nANCESTRY PROJECT - COMPREHENSIVE TEST SUITE")
    if enable_integration:
        print("🔗 INTEGRATION MODE: Running end-to-end workflow tests with live API access")
    if enable_fast_mode:
        print("🚀 FAST MODE: Parallel execution enabled")
    if enable_benchmark:
        print("📊 BENCHMARK MODE: Performance monitoring enabled")
    print("=" * 60)
    print()  # Blank line


def _run_pre_test_checks() -> bool:
    """
    Run linter checks before tests.

    Returns:
        True if checks pass or can continue, False if critical failure
    """
    # Skip linter if --skip-linter flag is provided
    if "--skip-linter" in sys.argv:
        print("⏭️  Skipping linter checks (--skip-linter flag provided)")
        return True

    # Run linter first for hygiene; fail fast only on safe subset
    return run_linter()


def _discover_and_prepare_modules(enable_integration: bool) -> tuple[list[str], dict[str, str], list[tuple[str, str]]]:
    """
    Discover test modules and extract their descriptions.

    Args:
        enable_integration: Whether to run integration tests instead of unit tests

    Returns:
        Tuple of (discovered_modules, module_descriptions, modules_with_descriptions)
    """
    # Auto-discover all test modules with the standardized test function
    discovered_modules = discover_test_modules()
    # De-duplicate while preserving order
    discovered_modules = list(dict.fromkeys(discovered_modules))

    # Filter modules based on mode
    integration_modules = ["test_integration_workflow.py"]

    if enable_integration:
        # In integration mode, ONLY run integration tests
        discovered_modules = [m for m in discovered_modules if m in integration_modules]
        if (
            not discovered_modules
            and "test_integration_workflow.py" not in discovered_modules
            and Path("test_integration_workflow.py").exists()
        ):
            # Fallback if discovery didn't pick it up (e.g. if I haven't saved the file yet or something)
            # But since I added run_comprehensive_tests, it should be discovered.
            # Just in case, let's check if it exists and add it if missing from discovery but present on disk
            discovered_modules = ["test_integration_workflow.py"]
    else:
        # In normal mode, EXCLUDE integration tests
        discovered_modules = [m for m in discovered_modules if m not in integration_modules]

    # Apply smart test ordering
    if "--fast" in sys.argv:
        print("🎯 Optimizing test order for faster feedback...")
        discovered_modules = optimize_test_order(discovered_modules)

    # Extract descriptions from module docstrings for enhanced reporting
    module_descriptions: dict[str, str] = {}
    enhanced_count = 0

    for module_name in discovered_modules:
        description = extract_module_description(module_name)
        if description:
            module_descriptions[module_name] = description
            enhanced_count += 1

    print(f"📊 Found {len(discovered_modules)} test modules ({enhanced_count} with enhanced descriptions)")

    print(f"\n{'=' * 60}")
    print("🧪 RUNNING TESTS")
    print(f"{'=' * 60}")

    # Prepare modules with descriptions
    modules_with_descriptions: list[tuple[str, str]] = [
        (module, module_descriptions.get(module, "")) for module in discovered_modules
    ]

    return discovered_modules, module_descriptions, modules_with_descriptions


def _execute_tests(
    config: TestExecutionConfig,
) -> tuple[list[tuple[str, str, bool]], list[TestExecutionMetrics], int, int]:
    """
    Execute tests in parallel or sequential mode.

    Args:
        config: TestExecutionConfig with all test execution parameters

    Returns:
        Tuple of (results, all_metrics, total_tests_run, passed_count)
    """
    if config.enable_fast_mode:
        print("🚀 Running tests in parallel...")
        all_metrics, passed_count, total_tests_run = run_tests_parallel(
            config.modules_with_descriptions, config.enable_monitoring
        )
        results = [(m.module_name, config.module_descriptions.get(m.module_name, ""), m.success) for m in all_metrics]
    else:
        results, all_metrics, total_tests_run, passed_count = _run_tests_sequentially(config)

    return results, all_metrics, total_tests_run, passed_count


def _process_cached_module(
    cached_entry: dict[str, Any],
    all_metrics: list[TestExecutionMetrics],
) -> int:
    """Process a cached module and return test count."""
    test_count = cached_entry.get("test_count", 0)
    metrics_data = cached_entry.get("metrics")

    if metrics_data:
        # Reconstruct QualityMetrics if present
        qm_data = metrics_data.get("quality_metrics")
        qm = QualityMetrics(**qm_data) if qm_data else None

        # Create metrics object
        metrics_args = metrics_data.copy()
        if "quality_metrics" in metrics_args:
            del metrics_args["quality_metrics"]

        metrics = TestExecutionMetrics(quality_metrics=qm, **metrics_args)
        all_metrics.append(metrics)

    return test_count


def _run_tests_sequentially(
    config: TestExecutionConfig,
) -> tuple[list[tuple[str, str, bool]], list[TestExecutionMetrics], int, int]:
    """Run tests sequentially with caching support."""
    print("🔄 Running tests sequentially...")
    sys.stdout.flush()
    results: list[tuple[str, str, bool]] = []
    all_metrics: list[TestExecutionMetrics] = []
    total_tests_run = 0
    passed_count = 0

    # Load test result cache
    test_cache = TestResultCache.load_cache()
    cache_hits = 0

    for i, (module_name, description) in enumerate(config.modules_with_descriptions, 1):
        # Check cache
        if TestResultCache.should_skip_module(module_name, test_cache):
            cached_entry = test_cache[module_name]
            print(f"⏩ [{i:2d}/{len(config.discovered_modules)}] Skipping: {module_name} (Unchanged)")
            sys.stdout.flush()

            # Use cached results
            test_count = _process_cached_module(cached_entry, all_metrics)

            total_tests_run += test_count
            passed_count += 1
            results.append((module_name, description or f"Tests for {module_name}", True))
            cache_hits += 1
            continue

        print(f"\n🧪 [{i:2d}/{len(config.discovered_modules)}] Testing: {module_name}")
        sys.stdout.flush()

        # Track test execution start time
        start_time = time.time()

        # Always collect metrics for quality summary (not just when monitoring enabled)
        success, test_count, metrics = run_module_tests(
            module_name, description, enable_monitoring=True, coverage=config.enable_benchmark
        )

        # Update test history for smart ordering
        duration = time.time() - start_time
        update_test_history(module_name, duration, success)

        # Update cache if successful and quality is 100%
        if success:
            should_cache = True
            if metrics and metrics.quality_metrics and metrics.quality_metrics.quality_score < 100.0:
                should_cache = False

            if should_cache:
                test_cache[module_name] = {
                    "file_hash": TestResultCache.get_module_hash(module_name),
                    "success": True,
                    "test_count": test_count,
                    "metrics": asdict(metrics) if metrics else None,
                }

        total_tests_run += test_count
        if success:
            passed_count += 1
        if metrics:
            all_metrics.append(metrics)
        results.append((module_name, description or f"Tests for {module_name}", success))

    # Save cache
    if cache_hits > 0 or passed_count > 0:
        TestResultCache.save_cache(test_cache)
        if cache_hits > 0:
            print(f"\n⚡ Skipped {cache_hits} unchanged modules using cache")

    return results, all_metrics, total_tests_run, passed_count


def _execute_tests_with_timing(
    config: TestExecutionConfig,
) -> tuple[list[tuple[str, str, bool]], list[TestExecutionMetrics], int, int, float]:
    """Execute tests and measure runtime."""
    start_time = time.time()
    results, all_metrics, total_tests_run, passed_count = _execute_tests(config)
    return results, all_metrics, total_tests_run, passed_count, time.time() - start_time


def _print_basic_summary(
    total_duration: float,
    total_tests_run: int,
    passed_count: int,
    failed_count: int,
    results: list[tuple[str, str, bool]],
) -> None:
    """Print basic test summary statistics."""
    success_rate = (passed_count / len(results)) * 100 if results else 0

    print(f"\n{'=' * 60}")
    print("📊 FINAL TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"⏰ Duration: {total_duration:.1f}s")
    print(f"🧪 Total Tests Run: {total_tests_run}")
    print(f"✅ Passed: {passed_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📈 Success Rate: {success_rate:.1f}%")


def _calculate_performance_stats(all_metrics: list[TestExecutionMetrics]) -> tuple[float, float, float, float]:
    """Calculate performance statistics from metrics."""
    if not all_metrics:
        return 0.0, 0.0, 0.0, 0.0

    avg_memory = sum(m.memory_usage_mb for m in all_metrics) / len(all_metrics)
    peak_memory = max(m.memory_usage_mb for m in all_metrics)
    avg_cpu = sum(m.cpu_usage_percent for m in all_metrics) / len(all_metrics)
    peak_cpu = max(m.cpu_usage_percent for m in all_metrics)

    return avg_memory, peak_memory, avg_cpu, peak_cpu


def _print_performance_metrics(config: PerformanceMetricsConfig) -> None:
    """
    Print performance metrics and analysis.

    Args:
        config: PerformanceMetricsConfig with all metrics and settings
    """
    if not config.all_metrics:
        return

    avg_memory, peak_memory, avg_cpu, peak_cpu = _calculate_performance_stats(config.all_metrics)

    # Calculate parallel efficiency
    sequential_time = sum(m.duration for m in config.all_metrics)
    parallel_efficiency = (sequential_time / config.total_duration) if config.total_duration > 0 else 1.0

    print("\n📊 PERFORMANCE METRICS:")
    print(f"   💾 Memory Usage: {avg_memory:.1f}MB avg, {peak_memory:.1f}MB peak")
    print(f"   ⚡ CPU Usage: {avg_cpu:.1f}% avg, {peak_cpu:.1f}% peak")
    if config.enable_fast_mode:
        print(f"   🚀 Parallel Efficiency: {parallel_efficiency:.1f}x speedup")

    # Create suite performance metrics
    suite_performance = TestSuitePerformance(
        total_duration=config.total_duration,
        total_tests=config.total_tests_run,
        passed_modules=config.passed_count,
        failed_modules=config.failed_count,
        avg_memory_usage=avg_memory,
        peak_memory_usage=peak_memory,
        avg_cpu_usage=avg_cpu,
        peak_cpu_usage=peak_cpu,
        parallel_efficiency=parallel_efficiency,
        optimization_suggestions=analyze_performance_trends(config.all_metrics),
    )

    # Show optimization suggestions
    if suite_performance.optimization_suggestions:
        print("\n💡 OPTIMIZATION SUGGESTIONS:")
        for suggestion in suite_performance.optimization_suggestions:
            print(f"   {suggestion}")

    # Save metrics for trend analysis
    if config.enable_benchmark:
        save_performance_metrics(config.all_metrics, suite_performance)


def _print_failed_modules(results: list[tuple[str, str, bool]]) -> None:
    """Print list of failed modules."""
    if any(not success for _, _, success in results):
        print("\n❌ FAILED MODULES:")
        for module_name, _, success in results:
            if not success:
                print(f"   • {module_name}")


def _count_enhanced_results(
    results: list[tuple[str, str, bool]], module_descriptions: dict[str, str]
) -> tuple[int, int]:
    """Count enhanced modules that passed and failed."""
    enhanced_passed = sum(1 for module_name, _, success in results if success and module_name in module_descriptions)
    enhanced_failed = sum(
        1 for module_name, _, success in results if not success and module_name in module_descriptions
    )
    return enhanced_passed, enhanced_failed


def _print_final_results(
    results: list[tuple[str, str, bool]],
    module_descriptions: dict[str, str],
    discovered_modules: list[str],
    passed_count: int,
    failed_count: int,
) -> None:
    """Print final results summary by category."""
    # Show failed modules first if any
    _print_failed_modules(results)

    # Calculate enhanced module stats
    enhanced_passed, enhanced_failed = _count_enhanced_results(results, module_descriptions)
    standard_passed = passed_count - enhanced_passed
    standard_failed = failed_count - enhanced_failed

    print("\n📋 RESULTS BY CATEGORY:")
    print(f"   Enhanced Modules: {enhanced_passed} passed, {enhanced_failed} failed")
    print(f"   Standard Modules: {standard_passed} passed, {standard_failed} failed")

    # Print final summary
    if failed_count == 0:
        print(f"\n🎉 ALL {len(discovered_modules)} MODULES PASSED!")
        print(f"   Professional testing framework with {len(discovered_modules)} standardized modules complete.\n")
    else:
        print(f"\n⚠️{failed_count} module(s) failed.")
        print("   Check individual test outputs above for details.\n")


def analyze_application_logs(log_path: str | None = None) -> LogAnalysisData | LogAnalysisError:
    """
    Analyze application logs for performance metrics and errors.
    Integrated from monitor_performance.py for log analysis.

    Args:
        log_path: Path to the log file to analyze

    Returns:
        dict: Analysis results including timing stats, error counts, and warnings
    """
    from os import getenv

    log_path = log_path or getenv("LOG_FILE", "Logs/app.log")
    log_file = Path(log_path)

    if not log_file.exists():
        return {"error": f"Log file not found: {log_path}"}

    results: LogAnalysisData = {
        "timing": [],
        "errors": {
            "429": 0,
            "ERROR": 0,
            "CRITICAL": 0,
            "Exception": 0,
        },
        "warnings": 0,
        "pages_processed": 0,
        "cache_hits": 0,
        "api_fetches": 0,
        "highest_page": 0,
    }

    with log_file.open(encoding='utf-8') as f:
        content = f.read()

    # Extract API fetch timing data
    api_pattern = r"API fetch complete: (\d+) matches in ([\d.]+)s \(avg: ([\d.]+)s/match\)"
    for match in re.finditer(api_pattern, content):
        matches_count = int(match.group(1))
        total_time = float(match.group(2))
        avg_time = float(match.group(3))
        timing_entry: LogTimingEntry = {
            "matches": matches_count,
            "total_seconds": total_time,
            "avg_per_match": avg_time,
        }
        results["timing"].append(timing_entry)
        results["api_fetches"] += 1

    # Count errors
    results["errors"]["429"] = len(re.findall(r"429", content))
    results["errors"]["ERROR"] = len(re.findall(r"\bERROR\b", content))
    results["errors"]["CRITICAL"] = len(re.findall(r"\bCRITICAL\b", content))
    results["errors"]["Exception"] = len(re.findall(r"Exception", content))

    # Count warnings
    results["warnings"] = len(re.findall(r"\bWAR\b", content))

    # Count pages processed
    page_pattern = r"Page (\d+):"
    page_numbers = [int(m.group(1)) for m in re.finditer(page_pattern, content)]
    results["pages_processed"] = len(page_numbers)
    results["highest_page"] = max(page_numbers) if page_numbers else 0

    # Count cache hits
    cache_pattern = r"✓ All \d+ matches are up-to-date"
    results["cache_hits"] = len(re.findall(cache_pattern, content))

    return results


def _format_error_summary(results: LogAnalysisData) -> list[str]:
    """Format error summary section."""
    lines = ["🚨 ERROR SUMMARY:"]
    total_errors = sum(results["errors"].values())
    if total_errors == 0:
        lines.append("  ✅ NO ERRORS DETECTED")
    else:
        for error_type, count in results["errors"].items():
            if count > 0:
                lines.append(f"  ❌ {error_type}: {count}")
    return lines


def _format_timing_analysis(results: LogAnalysisData) -> list[str]:
    """Format timing analysis section."""
    if not results["timing"]:
        return []

    lines = ["⏱️  TIMING ANALYSIS:"]
    avg_times = [t["avg_per_match"] for t in results["timing"]]
    total_times = [t["total_seconds"] for t in results["timing"]]

    lines.append(f"  Pages Analyzed: {len(results['timing'])}")
    lines.append(f"  Average per match: {sum(avg_times) / len(avg_times):.2f}s")
    lines.append(f"  Fastest: {min(avg_times):.2f}s")
    lines.append(f"  Slowest: {max(avg_times):.2f}s")
    lines.append(f"  Variance: {max(avg_times) - min(avg_times):.2f}s")
    lines.append(f"  Total API time: {sum(total_times):.1f}s")

    # Calculate throughput and estimate
    total_matches = sum(t["matches"] for t in results["timing"])
    total_seconds = sum(total_times)
    if total_seconds > 0:
        matches_per_hour = (total_matches / total_seconds) * 3600
        lines.append(f"  Throughput: {matches_per_hour:.0f} matches/hour")

        pages_remaining = 802 - results['highest_page']
        if pages_remaining > 0:
            avg_page_time = sum(total_times) / len(total_times)
            est_hours = (pages_remaining * avg_page_time) / 3600
            lines.append(f"  Estimated time remaining: {est_hours:.1f} hours ({pages_remaining} pages)")

    return lines


def format_log_analysis(results: LogAnalysisData) -> str:
    """Format log analysis results as a readable report."""
    structured_results = results

    report: list[str] = []
    report.append(SEPARATOR_LINE)
    report.append("📊 APPLICATION LOG PERFORMANCE ANALYSIS")
    report.append(SEPARATOR_LINE)

    # Error Summary
    report.extend(_format_error_summary(structured_results))

    # Warnings
    report.append(f"\n⚠️  WARNINGS: {structured_results['warnings']}")

    # Processing Stats
    report.append("\n📈 PROCESSING STATISTICS:")
    report.append(f"  Pages Processed: {structured_results['pages_processed']}")
    report.append(f"  Highest Page: {structured_results['highest_page']}")
    report.append(f"  Cache Hits (pages skipped): {structured_results['cache_hits']}")
    report.append(f"  API Fetches (pages with new data): {structured_results['api_fetches']}")

    # Timing Analysis
    timing_lines = _format_timing_analysis(structured_results)
    if timing_lines:
        report.append("\n" + "\n".join(timing_lines))

    report.append(SECTION_SEPARATOR)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(SEPARATOR_LINE)

    return "\n".join(report)


# ============================================================================
def print_log_analysis(log_path: str | None = None) -> None:
    """Analyze and print application log performance metrics.

    Provides detailed analysis of API timing, 429 errors, and throughput
    metrics from log files.
    """
    results = analyze_application_logs(log_path)
    if "error" in results:
        print(f"\n❌ {results['error']}")
        return

    print("\n" + format_log_analysis(results))


def _calculate_quality_distribution(quality_scores: list[float]) -> tuple[int, int, int]:
    """Calculate quality score distribution."""
    below_70 = sum(1 for score in quality_scores if score < 70)
    between_70_95 = sum(1 for score in quality_scores if 70 <= score < 95)
    above_95 = sum(1 for score in quality_scores if score >= 95)
    return below_70, between_70_95, above_95


def _print_final_quality_summary(all_metrics: list[TestExecutionMetrics]) -> None:
    """
    Print comprehensive quality summary at the end of test run.

    Args:
        all_metrics: List of all test execution metrics
    """
    if not all_metrics:
        return

    quality_scores = [
        m.quality_metrics.quality_score for m in all_metrics if hasattr(m, 'quality_metrics') and m.quality_metrics
    ]
    if not quality_scores:
        return

    avg_quality = sum(quality_scores) / len(quality_scores)
    min_quality = min(quality_scores)
    max_quality = max(quality_scores)
    below_70, between_70_95, above_95 = _calculate_quality_distribution(quality_scores)

    print("\n" + "=" * 80)
    print("📊 Code Quality Summary")
    print("=" * 80)
    print(f"Average: {avg_quality:.1f}/100 | Min: {min_quality:.1f}/100 | Max: {max_quality:.1f}/100")
    print(f"   ✅ Above 95%: {above_95} modules")
    print(f"   📊 70-95%: {between_70_95} modules")
    if below_70 > 0:
        print(f"   ⚠️  Below 70%: {below_70} modules (NEEDS ATTENTION)")

    # List modules with < 100% quality
    imperfect_modules = [
        (m.module_name, m.quality_metrics.quality_score)
        for m in all_metrics
        if hasattr(m, "quality_metrics") and m.quality_metrics and m.quality_metrics.quality_score < 100
    ]

    if imperfect_modules:
        print("\n   📉 Modules with < 100% Quality:")
        # Sort by score (ascending)
        imperfect_modules.sort(key=lambda x: x[1])
        for name, score in imperfect_modules:
            print(f"      • {name}: {score:.1f}/100")

    print("=" * 80)


def main() -> bool:
    """Comprehensive test runner with performance monitoring and optimization."""
    flags = bootstrap_feature_flags()
    try:
        # Fix trailing whitespace before running tests
        _fix_trailing_whitespace()

        # Setup environment and parse arguments
        env = _setup_test_environment()

        # Print header
        _print_test_header(env.enable_fast_mode, env.enable_benchmark, env.enable_integration)

        # Run pre-test checks
        if not _run_pre_test_checks():
            return False

        # Discover and prepare modules
        discovered_modules, module_descriptions, modules_with_descriptions = _discover_and_prepare_modules(
            env.enable_integration
        )

        # Execute tests
        results, all_metrics, total_tests_run, passed_count, total_duration = _execute_tests_with_timing(
            TestExecutionConfig(
                modules_with_descriptions=modules_with_descriptions,
                discovered_modules=discovered_modules,
                module_descriptions=module_descriptions,
                enable_fast_mode=env.enable_fast_mode,
                enable_monitoring=env.enable_monitoring,
                enable_benchmark=env.enable_benchmark,
                enable_integration=env.enable_integration,
            )
        )

        # Print final results
        _print_basic_summary(
            total_duration, total_tests_run, passed_count, len(discovered_modules) - passed_count, results
        )
        _print_final_results(
            results, module_descriptions, discovered_modules, passed_count, len(discovered_modules) - passed_count
        )
        _print_final_quality_summary(all_metrics)

        # Print performance metrics if enabled
        if env.enable_monitoring:
            _print_performance_metrics(
                PerformanceMetricsConfig(
                    all_metrics=all_metrics,
                    total_duration=total_duration,
                    total_tests_run=total_tests_run,
                    passed_count=passed_count,
                    failed_count=len(discovered_modules) - passed_count,
                    enable_fast_mode=env.enable_fast_mode,
                    enable_benchmark=env.enable_benchmark,
                )
            )

        # Print log analysis if enabled
        if env.enable_log_analysis:
            print_log_analysis()

        quality_ok = True
        quality_scores: list[tuple[str, float]] = []
        if env.enable_quality_checks:
            if flags.is_enabled("DISABLE_QUALITY_CHECKS", default=False):
                print("\n🧪 Quality checks skipped (disabled via feature flag).")
            else:
                print("\n🧪 Running standalone code quality checks (--quality)...")
                quality_ok, quality_scores = run_quality_checks(flags)
                if quality_scores:
                    for fname, score in quality_scores:
                        label = "(missing)" if score == 0 else f"{score:.1f}/100"
                        print(f"   • {fname}: {label}")
                if not quality_ok:
                    print("❌ Quality checks failed (average score below threshold).")

        return passed_count == len(discovered_modules) and quality_ok

    finally:
        # Clean up any browser session that was opened during tests
        _cleanup_browser_after_all_tests()


def _cleanup_browser_after_all_tests() -> None:
    """Close any browser session that was opened during test execution."""
    try:
        from core.session_utils import close_cached_session, get_session_manager

        sm = get_session_manager()
        if sm is not None and hasattr(sm, "browser_manager") and getattr(sm.browser_manager, "driver_live", False):
            print("\n🧹 Closing browser session after tests...")
            close_cached_session(keep_db=True)
    except Exception:
        pass  # Silently ignore cleanup errors


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test run interrupted by user!\n\n")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error in test runner: {e}\n\n")
        sys.exit(1)
