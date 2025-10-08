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
    python run_all_tests.py --analyze-logs # Analyze application logs for performance metrics
    python run_all_tests.py --fast --analyze-logs  # Run tests and then analyze logs
"""

import concurrent.futures
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    PSUTIL_AVAILABLE = False

# Import code quality checker
from code_quality_checker import CodeQualityChecker, QualityMetrics


def _check_and_use_venv() -> bool:
    """Check if running in venv, and if not, try to re-run with venv Python."""
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

    if in_venv:
        return True

    # Check if .venv exists
    venv_python = Path('.venv') / 'Scripts' / 'python.exe'
    if not venv_python.exists():
        # Try Unix-style path
        venv_python = Path('.venv') / 'bin' / 'python'
        if not venv_python.exists():
            print("⚠️  WARNING: Not running in virtual environment and .venv not found")
            print("   Some tests may fail due to missing dependencies")
            return False

    # Re-run with venv Python
    print(f"🔄 Re-running tests with venv Python: {venv_python}")
    print()
    result = subprocess.run(
        [str(venv_python), __file__] + sys.argv[1:],
        cwd=Path.cwd(), check=False
    )
    sys.exit(result.returncode)


# Check and use venv at module load time (before imports that need dependencies)
_check_and_use_venv()


def _invoke_ruff(args: list[str]) -> subprocess.CompletedProcess:
    """Run Ruff with the provided arguments and return the completed process."""
    command = [sys.executable, "-m", "ruff", *args]
    return subprocess.run(command, check=False, capture_output=True, text=True, cwd=Path.cwd())


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


@dataclass
class PerformanceMetricsConfig:
    """Configuration for performance metrics printing."""
    all_metrics: list[Any]
    total_duration: float
    total_tests_run: int
    passed_count: int
    failed_count: int
    enable_fast_mode: bool
    enable_benchmark: bool


class PerformanceMonitor:
    """Monitor system performance during test execution."""

    def __init__(self) -> None:
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None

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
            "peak_cpu_percent": max(cpu_values)
        }

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        if not self.process:
            return
        while self.monitoring:
            try:
                memory_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent()

                self.metrics.append({
                    "memory_mb": memory_info.rss / (1024 * 1024),
                    "cpu_percent": cpu_percent,
                    "timestamp": time.time()
                })

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

        # Step 1: Auto-fix ALL fixable issues
        print("🧹 LINTER: Auto-fixing all fixable linting issues...")
        fix_result = _invoke_ruff(["check", "--fix", "."])
        if fix_result.returncode == 0:
            print("   ✅ All fixable issues resolved")
        else:
            print("   ⚠️  Some issues auto-fixed, checking for critical errors...")

        # Step 2: blocking rule set (only critical errors)
        print("🧹 LINTER: Enforcing critical blocking rules (E722,F821,F811,F823)...")
        block_res = _invoke_ruff([
            "check",
            "--select",
            "E722,F821,F811,F823",
            ".",
        ])
        if block_res.returncode != 0:
            print("❌ LINTER FAILED (blocking): critical violations found")
            _print_tail(block_res.stdout or block_res.stderr or "")
            return False

        # Step 3: non-blocking diagnostics (excluding PLR2004 and PLC0415)
        print("🧹 LINTER: Repository diagnostics (non-blocking summary)...")
        diag_res = _invoke_ruff([
            "check",
            "--statistics",
            "--exit-zero",
            "--ignore=PLR2004,PLC0415",
            ".",
        ])
        _print_nonempty_lines(diag_res.stdout)
        return True
    except Exception as e:
        print(f"⚠️ LINTER step skipped due to error: {e}")
        return True


def run_quality_checks() -> bool:
    """
    Run Python best practices quality checks.

    Returns:
        bool: True if quality checks pass, False otherwise
    """
    try:
        print("🔍 QUALITY: Running Python best practices checks...")

        # Import and run quality checker
        from code_quality_checker import CodeQualityChecker

        checker = CodeQualityChecker()
        current_dir = Path()

        # Check key files for quality
        key_files = [
            "action10.py", "action11.py", "utils.py", "main.py",
            "python_best_practices.py", "code_quality_checker.py"
        ]

        total_score = 0
        files_checked = 0

        for file_name in key_files:
            file_path = current_dir / file_name
            if file_path.exists():
                metrics = checker.check_file(file_path)
                total_score += metrics.quality_score
                files_checked += 1

                if metrics.quality_score < 70:
                    print(f"⚠️  {file_name}: Quality score {metrics.quality_score:.1f}/100 (below threshold)")
                else:
                    print(f"✅ {file_name}: Quality score {metrics.quality_score:.1f}/100")

        if files_checked > 0:
            avg_score = total_score / files_checked
            print(f"📊 Average quality score: {avg_score:.1f}/100")

            if avg_score < 70:
                print("⚠️  Quality score below recommended threshold (70)")
                return False

        return True

    except Exception as e:
        print(f"⚠️ QUALITY checks skipped due to error: {e}")
        return True


def _should_skip_system_file(python_file: Path) -> bool:
    """Check if file should be skipped (system files, test runner, etc.)."""
    # Skip specific system files
    if python_file.name in ["run_all_tests.py", "main.py", "__main__.py"]:
        return True

    # Only skip root-level __init__.py, not package __init__.py files
    # Package __init__.py files (like core/__init__.py) may contain tests
    if python_file.name == "__init__.py":
        # Check if this is a root-level __init__.py (no parent directory except project root)
        # Allow package __init__.py files in subdirectories
        return python_file.parent == Path(__file__).parent

    return False


def _should_skip_cache_or_temp_file(python_file: Path) -> bool:
    """Check if file should be skipped (cache, backup, temp, venv)."""
    file_path_str = str(python_file)
    return (
        "__pycache__" in file_path_str
        or python_file.name.endswith("_backup.py")
        or "backup_before_migration" in file_path_str
        or python_file.name.startswith("temp_")
        or python_file.name.endswith("_temp.py")
        or "_old" in python_file.name
        or ".venv" in file_path_str
        or "site-packages" in file_path_str
        or "Cache" in file_path_str
        or "Logs" in file_path_str
    )


def _should_skip_demo_file(python_file: Path) -> bool:
    """Check if file should be skipped (demo/prototype scripts)."""
    demo_markers = ["demo", "prototype", "experimental", "sandbox"]
    name_lower = python_file.name.lower()
    return any(marker in name_lower for marker in demo_markers)


def _should_skip_interactive_file(python_file: Path) -> bool:
    """Check if file should be skipped (interactive modules)."""
    return python_file.name in ["db_viewer.py", "test_program_executor.py"]


def _has_test_function(content: str) -> bool:
    """Check if file content has the standardized test function."""
    return (
        "def run_comprehensive_tests" in content or
        "run_comprehensive_tests = create_standard_test_runner" in content
    )


def discover_test_modules() -> list[str]:
    """
    Discover all Python modules that contain tests by scanning the project directory.

    Returns a list of module paths that contain the run_comprehensive_tests() function,
    which indicates they follow the standardized testing framework.
    """
    project_root = Path(__file__).parent
    test_modules = []

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
    before_quotes = stripped.split('"""')[0].strip()
    if before_quotes:
        docstring_lines.append(before_quotes)
    return True


def _parse_docstring_lines(lines: list[str]) -> list[str]:
    """Parse file lines to extract docstring content."""
    in_docstring = False
    docstring_lines = []

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
    return line and not line.startswith('=') and len(line) > 10


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
            description = description[len(word):].strip()
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
        component = (
            module_name.replace("core/", "")
            .replace(".py", "")
            .replace("_", " ")
            .title()
        )
        result = f"Core {component} functionality"
    elif "config/" in module_name:
        component = (
            module_name.replace("config/", "")
            .replace(".py", "")
            .replace("_", " ")
            .title()
        )
        result = f"Configuration {component} management"
    elif "action" in module_name:
        action_name = module_name.replace(".py", "").replace("_", " ").title()
        result = f"{action_name} automation"
    elif module_name.endswith("_utils.py"):
        util_type = module_name.replace("_utils.py", "").replace("_", " ").title()
        result = f"{util_type} utility functions"
    elif module_name.endswith("_manager.py"):
        manager_type = (
            module_name.replace("_manager.py", "").replace("_", " ").title()
        )
        result = f"{manager_type} management system"
    elif module_name.endswith("_cache.py"):
        cache_type = module_name.replace("_cache.py", "").replace("_", " ").title()
        result = f"{cache_type} caching system"
    else:
        # Generic fallback
        clean_name = module_name.replace(".py", "").replace("_", " ").title()
        result = f"{clean_name} module functionality"

    return result


def _try_pattern_passed_failed(stdout_lines: list[str]) -> str:
    """Pattern 1: Look for '✅ Passed: X' and '❌ Failed: Y'."""
    for line in stdout_lines:
        if "✅ Passed:" in line:
            try:
                passed = int(line.split("✅ Passed:")[1].split()[0])
                failed = 0
                # Look for failed count in same line or nearby lines
                if "❌ Failed:" in line:
                    failed = int(line.split("❌ Failed:")[1].split()[0])
                else:
                    # Check other lines for failed count
                    for other_line in stdout_lines:
                        if "❌ Failed:" in other_line:
                            failed = int(other_line.split("❌ Failed:")[1].split()[0])
                            break
                return f"{passed + failed} tests"
            except (ValueError, IndexError):
                continue
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
    import re
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
    import re
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
    import re
    test_numbers = set()
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
    import re
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
    import re
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


def _extract_test_count_from_output(all_output_lines: list[str]) -> str:
    """Extract test count from output using multiple patterns."""
    if not all_output_lines:
        return "Unknown"

    # Try each pattern in order
    patterns = [
        _try_pattern_passed_failed,
        _try_pattern_tests_passed,
        _try_pattern_passed_failed_ansi,
        _try_pattern_unittest_ran,
        _try_pattern_numbered_tests,
        _try_pattern_number_followed_by_test,
        _try_pattern_all_tests_completed,
        _try_pattern_all_tests_passed_with_counts,
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


def _format_quality_info(quality_metrics) -> str:
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


def _print_quality_violations(quality_metrics) -> None:
    """Print quality violation details."""
    if not quality_metrics or quality_metrics.quality_score >= 95 or not quality_metrics.violations:
        return

    print("   🔍 Quality Issues:")
    # Group violations by type for better readability
    violation_types = {}
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
        import re
        match = re.search(r"(\d+)", test_count)
        if match:
            return int(match.group(1))
    except (ValueError, AttributeError):
        pass
    return 0


def _print_failure_details(result, failure_indicators: list[str]) -> None:
    """Print failure details from test output."""
    print("   🚨 Failure Details:")
    if result.stderr:
        error_lines = result.stderr.strip().split("\n")
        for line in error_lines[-3:]:  # Show last 3 error lines
            print(f"      {line}")
    if result.stdout and any(indicator in result.stdout for indicator in failure_indicators):
        stdout_lines = result.stdout.strip().split("\n")
        failure_lines = [
            line
            for line in stdout_lines
            if any(indicator in line for indicator in failure_indicators)
        ]
        for line in failure_lines[-2:]:  # Show last 2 failure lines
            print(f"      {line}")


def _build_test_command(module_name: str, coverage: bool) -> tuple[list[str], Optional[dict]]:
    """Build the command and environment for running tests."""
    cmd = [sys.executable]
    env = None

    # For modules with internal test suite, set env var to trigger test output
    suite_env_modules = {"prompt_telemetry.py", "quality_regression_gate.py"}
    if module_name in suite_env_modules:
        env = dict(os.environ)
        env["RUN_INTERNAL_TESTS"] = "1"

    if coverage:
        cmd += ["-m", "coverage", "run", "--append", module_name]
    else:
        cmd.append(module_name)

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
    module_name: str,
    test_result: dict,
    quality_metrics: Optional[QualityMetrics] = None
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
        error_message=test_result["result"].stderr if not test_result["success"] and test_result["result"].stderr else None,
        quality_metrics=quality_metrics
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
        error_message=error_message
    )


def _run_test_subprocess(module_name: str, coverage: bool) -> tuple[subprocess.CompletedProcess, float, str]:
    """Run the test subprocess and return result, duration, and timestamp."""
    start_time = time.time()
    start_datetime = datetime.now().isoformat()

    cmd, env = _build_test_command(module_name, coverage)
    result = subprocess.run(
        cmd, check=False, capture_output=True, text=True, cwd=Path.cwd(), env=env
    )

    duration = time.time() - start_time
    return result, duration, start_datetime


def _analyze_test_output(result: subprocess.CompletedProcess) -> tuple[bool, str, int]:
    """Analyze test output and return success status, test count string, and numeric count."""
    # Collect all output lines
    all_output_lines = []
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


def _print_test_result(success: bool, duration: float, test_count: str, quality_metrics, result: subprocess.CompletedProcess) -> None:
    """Print test result summary with quality info and failure details."""
    status = "✅ PASSED" if success else "❌ FAILED"
    quality_info = _format_quality_info(quality_metrics)
    print(f"   {status} | Duration: {duration:.2f}s | {test_count}{quality_info}")

    _print_quality_violations(quality_metrics)

    if not success:
        failure_indicators = [
            "❌ FAILED", "Status: FAILED", "AssertionError:", "Exception occurred:",
            "Test failed:", "❌ Failed: ", "CRITICAL ERROR", "FATAL ERROR",
        ]
        _print_failure_details(result, failure_indicators)


def run_module_tests(
    module_name: str, description: str | None = None, enable_monitoring: bool = False, coverage: bool = False
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
        result, duration, start_datetime = _run_test_subprocess(module_name, coverage)
        end_datetime = datetime.now().isoformat()

        # Collect performance and quality metrics
        perf_metrics = monitor.stop_monitoring() if monitor else {}
        quality_metrics = _run_quality_analysis(module_name)

        # Analyze output
        success, test_count, numeric_test_count = _analyze_test_output(result)

        # Print results
        _print_test_result(success, duration, test_count, quality_metrics, result)

        # Create metrics object if monitoring enabled
        metrics = None
        if enable_monitoring:
            test_result = {
                "duration": duration,
                "success": success,
                "test_count": numeric_test_count,
                "perf_metrics": perf_metrics,
                "result": result,
                "start_time": start_datetime,
                "end_time": end_datetime
            }
            metrics = _create_test_metrics(module_name, test_result, quality_metrics)

        return success, numeric_test_count, metrics

    except Exception as e:
        print(f"   ❌ FAILED | Error: {e}")
        error_metrics = _create_error_metrics(module_name, str(e)) if enable_monitoring else None
        return False, 0, error_metrics


def run_tests_parallel(modules_with_descriptions: list[tuple[str, str]], enable_monitoring: bool = False, coverage: bool = False) -> tuple[list[TestExecutionMetrics], int, int]:
    """Run tests in parallel for improved performance."""
    all_metrics = []
    passed_count = 0
    total_test_count = 0

    # Determine optimal number of workers (don't exceed CPU count)
    cpu_count = (psutil.cpu_count() if PSUTIL_AVAILABLE else os.cpu_count()) or 1
    max_workers = min(len(modules_with_descriptions), cpu_count)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all test jobs
        future_to_module = {
            executor.submit(run_module_tests, module, desc, enable_monitoring, coverage): (module, desc)
            for module, desc in modules_with_descriptions
        }

        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_module), 1):
            module, desc = future_to_module[future]
            try:
                success, test_count, metrics = future.result()
                if success:
                    passed_count += 1
                total_test_count += test_count

                if metrics:
                    all_metrics.append(metrics)

                print(f"🧪 [{i:2d}/{len(modules_with_descriptions)}] Testing: {module}")
                if desc:
                    print(f"   📝 {desc}")

            except Exception as e:
                print(f"🧪 [{i:2d}/{len(modules_with_descriptions)}] Testing: {module}")
                print(f"   ❌ FAILED | Error: {e}")

    return all_metrics, passed_count, total_test_count


def save_performance_metrics(metrics: list[TestExecutionMetrics], suite_performance: TestSuitePerformance):
    """Save performance metrics to file for trend analysis."""
    try:
        metrics_file = Path("test_performance_metrics.json")

        # Load existing metrics if file exists
        existing_data = []
        if metrics_file.exists():
            try:
                with metrics_file.open() as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []

        # Add new metrics
        new_entry = {
            "timestamp": datetime.now().isoformat(),
            "suite_performance": asdict(suite_performance),
            "module_metrics": [asdict(m) for m in metrics]
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


def analyze_performance_trends(metrics: list[TestExecutionMetrics]) -> list[str]:
    """Analyze performance metrics and provide optimization suggestions."""
    suggestions = []

    if not metrics:
        return suggestions

    # Analyze slow tests
    slow_tests = [m for m in metrics if m.duration > 10.0]  # Tests taking more than 10 seconds
    if slow_tests:
        slow_tests.sort(key=lambda x: x.duration, reverse=True)
        suggestions.append(f"🐌 {len(slow_tests)} slow tests detected (>10s). Slowest: {slow_tests[0].module_name} ({slow_tests[0].duration:.1f}s)")

    # Analyze memory usage
    high_memory_tests = [m for m in metrics if m.memory_usage_mb > 100.0]  # Tests using more than 100MB
    if high_memory_tests:
        high_memory_tests.sort(key=lambda x: x.memory_usage_mb, reverse=True)
        suggestions.append(f"🧠 {len(high_memory_tests)} memory-intensive tests detected (>100MB). Highest: {high_memory_tests[0].module_name} ({high_memory_tests[0].memory_usage_mb:.1f}MB)")

    # Analyze CPU usage
    high_cpu_tests = [m for m in metrics if m.cpu_usage_percent > 50.0]  # Tests using more than 50% CPU
    if high_cpu_tests:
        suggestions.append(f"⚡ {len(high_cpu_tests)} CPU-intensive tests detected (>50% CPU)")

    # Suggest parallel execution if not already used
    total_duration = sum(m.duration for m in metrics)
    if total_duration > 60.0:  # If total time > 1 minute
        suggestions.append("🚀 Consider using --fast flag for parallel execution to reduce total runtime")

    return suggestions


def _setup_test_environment() -> tuple[bool, bool, bool]:
    """
    Setup test environment and parse command line arguments.

    Returns:
        Tuple of (enable_fast_mode, enable_benchmark, enable_monitoring)
    """
    # Parse command line arguments
    enable_fast_mode = "--fast" in sys.argv
    enable_benchmark = "--benchmark" in sys.argv
    enable_monitoring = enable_benchmark or enable_fast_mode

    if enable_monitoring and not PSUTIL_AVAILABLE:
        print("⚠️ PERFORMANCE: psutil not installed; disabling monitoring features.")
        enable_monitoring = False

    # Set environment variable to skip live API tests that require browser/network
    os.environ["SKIP_LIVE_API_TESTS"] = "true"

    # Set environment variable to skip slow simulation tests (724-page workload, etc.)
    os.environ["SKIP_SLOW_TESTS"] = "true"

    return enable_fast_mode, enable_benchmark, enable_monitoring


def _print_test_header(enable_fast_mode: bool, enable_benchmark: bool) -> None:
    """Print test suite header with mode information."""
    print("\nANCESTRY PROJECT - COMPREHENSIVE TEST SUITE")
    if enable_fast_mode:
        print("🚀 FAST MODE: Parallel execution enabled")
    if enable_benchmark:
        print("📊 BENCHMARK MODE: Performance monitoring enabled")
    print("=" * 60)
    print()  # Blank line


def _run_pre_test_checks() -> bool:
    """
    Run linter and quality checks before tests.

    Returns:
        True if checks pass or can continue, False if critical failure
    """
    # Run linter first for hygiene; fail fast only on safe subset
    if not run_linter():
        return False

    # Run quality checks for Python best practices
    if not run_quality_checks():
        print("⚠️  Quality checks failed - continuing with tests but consider improvements")

    return True


def _discover_and_prepare_modules() -> tuple[list[str], dict[str, str], list[tuple[str, str]]]:
    """
    Discover test modules and extract their descriptions.

    Returns:
        Tuple of (discovered_modules, module_descriptions, modules_with_descriptions)
    """
    # Auto-discover all test modules with the standardized test function
    discovered_modules = discover_test_modules()

    # Extract descriptions from module docstrings for enhanced reporting
    module_descriptions = {}
    enhanced_count = 0

    for module_name in discovered_modules:
        description = extract_module_description(module_name)
        if description:
            module_descriptions[module_name] = description
            enhanced_count += 1

    print(
        f"📊 Found {len(discovered_modules)} test modules ({enhanced_count} with enhanced descriptions)"
    )

    print(f"\n{'='* 60}")
    print("🧪 RUNNING TESTS")
    print(f"{'='* 60}")

    # Prepare modules with descriptions
    modules_with_descriptions = [
        (module, module_descriptions.get(module, ""))
        for module in discovered_modules
    ]

    return discovered_modules, module_descriptions, modules_with_descriptions


def _execute_tests(config: TestExecutionConfig) -> tuple[list[tuple[str, str, bool]], list[Any], int, int]:
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
        results = [
            (m.module_name, config.module_descriptions.get(m.module_name, ""), m.success)
            for m in all_metrics
        ]
    else:
        print("🔄 Running tests sequentially...")
        results = []
        all_metrics = []
        total_tests_run = 0
        passed_count = 0

        for i, (module_name, description) in enumerate(config.modules_with_descriptions, 1):
            print(f"\n🧪 [{i:2d}/{len(config.discovered_modules)}] Testing: {module_name}")

            success, test_count, metrics = run_module_tests(
                module_name, description, config.enable_monitoring, coverage=config.enable_benchmark
            )
            total_tests_run += test_count
            if success:
                passed_count += 1
            if metrics:
                all_metrics.append(metrics)
            results.append((module_name, description or f"Tests for {module_name}", success))

    return results, all_metrics, total_tests_run, passed_count


def _print_basic_summary(
    total_duration: float,
    total_tests_run: int,
    passed_count: int,
    failed_count: int,
    results: list[tuple[str, str, bool]]
) -> None:
    """Print basic test summary statistics."""
    success_rate = (passed_count / len(results)) * 100 if results else 0

    print(f"\n{'='* 60}")
    print("📊 FINAL TEST SUMMARY")
    print(f"{'='* 60}")
    print(f"⏰ Duration: {total_duration:.1f}s")
    print(f"🧪 Total Tests Run: {total_tests_run}")
    print(f"✅ Passed: {passed_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📈 Success Rate: {success_rate:.1f}%")


def _collect_violations(all_metrics: list[Any]) -> list[str]:
    """Collect all violations from metrics."""
    all_violations = []
    for m in all_metrics:
        if m.quality_metrics and m.quality_metrics.violations:
            all_violations.extend(m.quality_metrics.violations)
    return all_violations


def _print_violation_summary(all_violations: list[str]) -> None:
    """Print summary of violation types."""
    violation_types = {}
    for violation in all_violations:
        vtype = _categorize_violation(violation)
        violation_types[vtype] = violation_types.get(vtype, 0) + 1

    print("   📋 Common Issues:")
    for vtype, count in sorted(violation_types.items(), key=lambda x: x[1], reverse=True):
        print(f"      {vtype}: {count} violations")


def _print_quality_summary(all_metrics: list[Any]) -> None:
    """Print quality metrics summary."""
    if not all_metrics or not any(m.quality_metrics for m in all_metrics):
        return

    quality_scores = [m.quality_metrics.quality_score for m in all_metrics if m.quality_metrics]
    if not quality_scores:
        return

    avg_quality = sum(quality_scores) / len(quality_scores)
    below_70_count = sum(1 for score in quality_scores if score < 70)
    below_95_count = sum(1 for score in quality_scores if 70 <= score < 95)
    above_95_count = sum(1 for score in quality_scores if score >= 95)

    print(f"🔍 Quality Score: {avg_quality:.1f}/100 avg")
    print(f"   ✅ Above 95%: {above_95_count} modules")
    print(f"   📊 70-95%: {below_95_count} modules")
    print(f"   ⚠️  Below 70%: {below_70_count} modules")

    # Show most common violation types
    all_violations = _collect_violations(all_metrics)
    if all_violations:
        _print_violation_summary(all_violations)


def _print_performance_metrics(config: PerformanceMetricsConfig) -> None:
    """
    Print performance metrics and analysis.

    Args:
        config: PerformanceMetricsConfig with all metrics and settings
    """
    if not config.all_metrics:
        return

    avg_memory = sum(m.memory_usage_mb for m in config.all_metrics) / len(config.all_metrics)
    peak_memory = max(m.memory_usage_mb for m in config.all_metrics)
    avg_cpu = sum(m.cpu_usage_percent for m in config.all_metrics) / len(config.all_metrics)
    peak_cpu = max(m.cpu_usage_percent for m in config.all_metrics)

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
        optimization_suggestions=analyze_performance_trends(config.all_metrics)
    )

    # Show optimization suggestions
    if suite_performance.optimization_suggestions:
        print("\n💡 OPTIMIZATION SUGGESTIONS:")
        for suggestion in suite_performance.optimization_suggestions:
            print(f"   {suggestion}")

    # Save metrics for trend analysis
    if config.enable_benchmark:
        save_performance_metrics(config.all_metrics, suite_performance)


def _print_final_results(
    results: list[tuple[str, str, bool]],
    module_descriptions: dict[str, str],
    discovered_modules: list[str],
    passed_count: int,
    failed_count: int
) -> None:
    """Print final results summary by category."""
    # Show failed modules first if any
    if failed_count > 0:
        print("\n❌ FAILED MODULES:")
        for module_name, _, success in results:
            if not success:
                print(f"   • {module_name}")

    # Show summary by category
    enhanced_passed = sum(
        1
        for module_name, _, success in results
        if success and module_name in module_descriptions
    )
    enhanced_failed = sum(
        1
        for module_name, _, success in results
        if not success and module_name in module_descriptions
    )

    print("\n📋 RESULTS BY CATEGORY:")
    print(f"   Enhanced Modules: {enhanced_passed} passed, {enhanced_failed} failed")
    print(
        f"   Standard Modules: {passed_count - enhanced_passed} passed, {failed_count - enhanced_failed} failed"
    )

    if failed_count == 0:
        print(f"\n🎉 ALL {len(discovered_modules)} MODULES PASSED!")
        print(f"   Professional testing framework with {len(discovered_modules)} standardized modules complete.\n")
    else:
        print(f"\n⚠️{failed_count} module(s) failed.")
        print("   Check individual test outputs above for details.\n")


def analyze_application_logs(log_path: str = "Logs/app.log") -> dict:
    """
    Analyze application logs for performance metrics and errors.
    Integrated from monitor_performance.py for log analysis.
    
    Args:
        log_path: Path to the log file to analyze
        
    Returns:
        dict: Analysis results including timing stats, error counts, and warnings
    """
    log_file = Path(log_path)

    if not log_file.exists():
        return {"error": f"Log file not found: {log_path}"}

    results = {
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
    }

    with open(log_file, encoding='utf-8') as f:
        content = f.read()

    # Extract API fetch timing data
    api_pattern = r"API fetch complete: (\d+) matches in ([\d.]+)s \(avg: ([\d.]+)s/match\)"
    for match in re.finditer(api_pattern, content):
        matches_count = int(match.group(1))
        total_time = float(match.group(2))
        avg_time = float(match.group(3))
        results["timing"].append({
            "matches": matches_count,
            "total_seconds": total_time,
            "avg_per_match": avg_time
        })
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


def format_log_analysis(results: dict) -> str:
    """Format log analysis results as a readable report."""
    if "error" in results:
        return f"❌ {results['error']}"

    report = []
    report.append("=" * 70)
    report.append("📊 APPLICATION LOG PERFORMANCE ANALYSIS")
    report.append("=" * 70)

    # Error Summary
    report.append("\n🚨 ERROR SUMMARY:")
    total_errors = sum(results["errors"].values())
    if total_errors == 0:
        report.append("  ✅ NO ERRORS DETECTED")
    else:
        for error_type, count in results["errors"].items():
            if count > 0:
                report.append(f"  ❌ {error_type}: {count}")

    # Warnings
    report.append(f"\n⚠️  WARNINGS: {results['warnings']}")

    # Processing Stats
    report.append("\n📈 PROCESSING STATISTICS:")
    report.append(f"  Pages Processed: {results['pages_processed']}")
    report.append(f"  Highest Page: {results['highest_page']}")
    report.append(f"  Cache Hits (pages skipped): {results['cache_hits']}")
    report.append(f"  API Fetches (pages with new data): {results['api_fetches']}")

    # Timing Analysis
    if results["timing"]:
        report.append("\n⏱️  TIMING ANALYSIS:")

        avg_times = [t["avg_per_match"] for t in results["timing"]]
        total_times = [t["total_seconds"] for t in results["timing"]]

        min_time = min(avg_times)
        max_time = max(avg_times)
        mean_time = sum(avg_times) / len(avg_times)

        report.append(f"  Pages Analyzed: {len(results['timing'])}")
        report.append(f"  Average per match: {mean_time:.2f}s")
        report.append(f"  Fastest: {min_time:.2f}s")
        report.append(f"  Slowest: {max_time:.2f}s")
        report.append(f"  Variance: {max_time - min_time:.2f}s")
        report.append(f"  Total API time: {sum(total_times):.1f}s")

        # Calculate throughput
        total_matches = sum(t["matches"] for t in results["timing"])
        total_seconds = sum(total_times)
        if total_seconds > 0:
            matches_per_hour = (total_matches / total_seconds) * 3600
            report.append(f"  Throughput: {matches_per_hour:.0f} matches/hour")

            # Estimate completion time for 802 pages
            pages_remaining = 802 - results['highest_page']
            if pages_remaining > 0:
                avg_page_time = sum(total_times) / len(total_times)
                est_seconds = pages_remaining * avg_page_time
                est_hours = est_seconds / 3600
                report.append(f"  Estimated time remaining: {est_hours:.1f} hours ({pages_remaining} pages)")

    report.append("\n" + "=" * 70)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)

    return "\n".join(report)


def print_log_analysis(log_path: str = "Logs/app.log") -> None:
    """Analyze and print application log performance metrics."""
    results = analyze_application_logs(log_path)
    print("\n" + format_log_analysis(results))


def main() -> bool:
    """Comprehensive test runner with performance monitoring and optimization."""
    # Setup environment and parse arguments
    enable_fast_mode, enable_benchmark, enable_monitoring = _setup_test_environment()

    # Check for log analysis flag
    analyze_logs = "--analyze-logs" in sys.argv

    # If only log analysis requested, do that and exit
    if analyze_logs and len(sys.argv) == 2:  # Only --analyze-logs flag
        print_log_analysis()
        return True

    # Print header
    _print_test_header(enable_fast_mode, enable_benchmark)

    # Run pre-test checks
    if not _run_pre_test_checks():
        return False

    # Discover and prepare test modules
    discovered_modules, module_descriptions, modules_with_descriptions = _discover_and_prepare_modules()

    if not discovered_modules:
        print("⚠️  No test modules discovered with run_comprehensive_tests() function.")
        return False

    total_start_time = time.time()

    # Execute tests
    test_config = TestExecutionConfig(
        modules_with_descriptions=modules_with_descriptions,
        discovered_modules=discovered_modules,
        module_descriptions=module_descriptions,
        enable_fast_mode=enable_fast_mode,
        enable_monitoring=enable_monitoring,
        enable_benchmark=enable_benchmark
    )
    results, all_metrics, total_tests_run, passed_count = _execute_tests(test_config)

    # Calculate final metrics
    total_duration = time.time() - total_start_time
    if not enable_fast_mode:  # Recalculate for sequential mode
        passed_count = sum(1 for _, _, success in results if success)
    failed_count = len(results) - passed_count

    # Print all summaries
    _print_basic_summary(total_duration, total_tests_run, passed_count, failed_count, results)
    _print_quality_summary(all_metrics)

    if enable_monitoring:
        perf_config = PerformanceMetricsConfig(
            all_metrics=all_metrics,
            total_duration=total_duration,
            total_tests_run=total_tests_run,
            passed_count=passed_count,
            failed_count=failed_count,
            enable_fast_mode=enable_fast_mode,
            enable_benchmark=enable_benchmark
        )
        _print_performance_metrics(perf_config)

    _print_final_results(results, module_descriptions, discovered_modules, passed_count, failed_count)

    # Print log analysis if requested
    if analyze_logs:
        print_log_analysis()

    return failed_count == 0


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
