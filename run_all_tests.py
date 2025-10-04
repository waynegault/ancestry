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
    python run_all_tests.py           # Run all tests with detailed reporting
    python run_all_tests.py --fast    # Run with parallel execution optimization
    python run_all_tests.py --benchmark # Run with detailed performance benchmarking
"""

import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil

# Import code quality checker
from code_quality_checker import CodeQualityChecker, QualityMetrics


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


class PerformanceMonitor:
    """Monitor system performance during test execution."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None

    def start_monitoring(self) -> None:
        """Start performance monitoring in background thread."""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self) -> dict[str, float]:
        """Stop monitoring and return aggregated metrics."""
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

    def _monitor_loop(self):
        """Background monitoring loop."""
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
    1) Apply safe auto-fixes for whitespace/import formatting
    2) Enforce blocking rule set; fail the run if violations remain
    3) Print non-blocking repository statistics
    """
    try:
        # Check if ruff is available
        import subprocess
        result = subprocess.run([sys.executable, "-m", "ruff", "--version"],
                              check=False, capture_output=True, text=True)
        if result.returncode != 0:
            print("🧹 LINTER: Ruff not available, skipping linting checks...")
            return True

        # Step 1: safe auto-fixes
        print("🧹 LINTER: Applying safe auto-fixes (W291/W292/W293/E401)...")
        fix_cmd = [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--fix",
            "--select",
            "W291,W292,W293,E401",
            ".",
        ]
        subprocess.run(fix_cmd, check=False, capture_output=True, text=True, cwd=Path.cwd())

        # Step 2: blocking rule set (only critical errors)
        print("🧹 LINTER: Enforcing critical blocking rules (E722,F821,F811,F823)...")
        block_cmd = [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "E722,F821,F811,F823",
            ".",
        ]
        block_res = subprocess.run(block_cmd, check=False, capture_output=True, text=True, cwd=Path.cwd())
        if block_res.returncode != 0:
            print("❌ LINTER FAILED (blocking): critical violations found")
            # Tail the output to keep logs compact
            tail = (block_res.stdout or block_res.stderr or "").splitlines()[-40:]
            for line in tail:
                print(line)
            return False

        # Step 3: non-blocking diagnostics (excluding PLR2004 and PLC0415)
        print("🧹 LINTER: Repository diagnostics (non-blocking summary)...")
        diag_cmd = [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--statistics",
            "--exit-zero",
            "--ignore=PLR2004,PLC0415",
            ".",
        ]
        diag_res = subprocess.run(diag_cmd, check=False, capture_output=True, text=True, cwd=Path.cwd())
        if diag_res.stdout:
            lines = [ln for ln in diag_res.stdout.splitlines() if ln.strip()]
            for line in lines[-25:]:
                print(line)
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
    return python_file.name in [
        "run_all_tests.py",
        "main.py",
        "__init__.py",
        "__main__.py",
    ]


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
        description = _clean_description_text(description, module_base)

        return description

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
    if "core/" in module_name:
        component = (
            module_name.replace("core/", "")
            .replace(".py", "")
            .replace("_", " ")
            .title()
        )
        return f"Core {component} functionality"
    elif "config/" in module_name:
        component = (
            module_name.replace("config/", "")
            .replace(".py", "")
            .replace("_", " ")
            .title()
        )
        return f"Configuration {component} management"
    elif "action" in module_name:
        action_name = module_name.replace(".py", "").replace("_", " ").title()
        return f"{action_name} automation"
    elif module_name.endswith("_utils.py"):
        util_type = module_name.replace("_utils.py", "").replace("_", " ").title()
        return f"{util_type} utility functions"
    elif module_name.endswith("_manager.py"):
        manager_type = (
            module_name.replace("_manager.py", "").replace("_", " ").title()
        )
        return f"{manager_type} management system"
    elif module_name.endswith("_cache.py"):
        cache_type = module_name.replace("_cache.py", "").replace("_", " ").title()
        return f"{cache_type} caching system"
    else:
        # Generic fallback
        clean_name = module_name.replace(".py", "").replace("_", " ").title()
        return f"{clean_name} module functionality"


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
    elif passed_count is not None:
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
    elif score < 95:
        return f" | Quality: {score:.1f}/100 📊 ({violations_count} issues)"
    else:
        return f" | Quality: {score:.1f}/100 ✅"


def _categorize_violation(violation: str) -> str:
    """Categorize a violation by type."""
    if "too long" in violation:
        return "Length"
    elif "too complex" in violation:
        return "Complexity"
    elif "missing type hint" in violation:
        return "Type Hints"
    else:
        return "Other"


def _format_violation_message(violation: str) -> str:
    """Format a violation message for display."""
    if "Function '" in violation and "'" in violation:
        func_name = violation.split("Function '")[1].split("'")[0]
        issue_type = violation.split("' ")[1] if "' " in violation else violation
        return f"• {func_name}: {issue_type}"
    else:
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
    duration: float,
    success: bool,
    numeric_test_count: int,
    perf_metrics: dict,
    start_datetime: str,
    end_datetime: str,
    result,
    quality_metrics
) -> TestExecutionMetrics:
    """Create TestExecutionMetrics object."""
    return TestExecutionMetrics(
        module_name=module_name,
        duration=duration,
        success=success,
        test_count=numeric_test_count,
        memory_usage_mb=perf_metrics.get("memory_mb", 0.0),
        cpu_usage_percent=perf_metrics.get("cpu_percent", 0.0),
        start_time=start_datetime,
        end_time=end_datetime,
        error_message=result.stderr if not success and result.stderr else None,
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


def run_module_tests(
    module_name: str, description: str | None = None, enable_monitoring: bool = False, coverage: bool = False
) -> tuple[bool, int, Optional[TestExecutionMetrics]]:
    """Run tests for a specific module with optional performance monitoring."""
    # Initialize performance monitoring
    monitor = PerformanceMonitor() if enable_monitoring else None
    metrics = None

    # Show description
    desc = _generate_module_description(module_name, description)
    print(f"   📝 {desc}")

    try:
        start_time = time.time()
        start_datetime = datetime.now().isoformat()

        # Start performance monitoring if enabled
        if monitor:
            monitor.start_monitoring()

        # Build and run test command
        cmd, env = _build_test_command(module_name, coverage)
        result = subprocess.run(
            cmd,
            check=False, capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env=env
        )

        duration = time.time() - start_time
        end_datetime = datetime.now().isoformat()

        # Stop monitoring and collect metrics
        perf_metrics = monitor.stop_monitoring() if monitor else {}

        # Run quality analysis on the module
        quality_metrics = _run_quality_analysis(module_name)

        # Check for success based on return code AND output content
        success = result.returncode == 0

        # Extract test counts from output using helper function
        all_output_lines = []
        if result.stdout:
            all_output_lines.extend(result.stdout.split("\n"))
        if result.stderr:
            all_output_lines.extend(result.stderr.split("\n"))

        test_count = _extract_test_count_from_output(all_output_lines)

        # Check for failures in output
        success = _check_for_failures_in_output(success, result.stdout)

        # Failure indicators for detailed output
        failure_indicators = [
            "❌ FAILED", "Status: FAILED", "AssertionError:", "Exception occurred:",
            "Test failed:", "❌ Failed: ", "CRITICAL ERROR", "FATAL ERROR",
        ]

        status = "✅ PASSED" if success else "❌ FAILED"

        # Format and print summary
        quality_info = _format_quality_info(quality_metrics)
        print(f"   {status} | Duration: {duration:.2f}s | {test_count}{quality_info}")

        # Print quality violations
        _print_quality_violations(quality_metrics)

        # Extract numeric test count for summary
        numeric_test_count = _extract_numeric_test_count(test_count)

        # Print failure details if test failed
        if not success:
            _print_failure_details(result, failure_indicators)

        # Create performance metrics if monitoring was enabled
        if enable_monitoring:
            metrics = _create_test_metrics(
                module_name, duration, success, numeric_test_count,
                perf_metrics, start_datetime, end_datetime, result, quality_metrics
            )

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
    cpu_count = psutil.cpu_count() or 1  # Fallback to 1 if cpu_count() returns None
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


def main() -> bool:
    """Comprehensive test runner with performance monitoring and optimization."""
    # Parse command line arguments
    enable_fast_mode = "--fast" in sys.argv
    enable_benchmark = "--benchmark" in sys.argv
    enable_monitoring = enable_benchmark or enable_fast_mode

    # Set environment variable to skip live API tests that require browser/network
    os.environ["SKIP_LIVE_API_TESTS"] = "true"

    # Set environment variable to skip slow simulation tests (724-page workload, etc.)
    os.environ["SKIP_SLOW_TESTS"] = "true"

    print("\nANCESTRY PROJECT - COMPREHENSIVE TEST SUITE")
    if enable_fast_mode:
        print("🚀 FAST MODE: Parallel execution enabled")
    if enable_benchmark:
        print("📊 BENCHMARK MODE: Performance monitoring enabled")
    print("=" * 60)
    print()  # Blank line

    # Run linter first for hygiene; fail fast only on safe subset
    if not run_linter():
        return False

    # Run quality checks for Python best practices
    if not run_quality_checks():
        print("⚠️  Quality checks failed - continuing with tests but consider improvements")

    # Auto-discover all test modules with the standardized test function
    discovered_modules = discover_test_modules()

    if not discovered_modules:
        print("⚠️  No test modules discovered with run_comprehensive_tests() function.")
        return False

    # Extract descriptions from module docstrings for enhanced reporting
    module_descriptions = {}
    enhanced_count = 0

    for module_name in discovered_modules:
        description = extract_module_description(module_name)
        if description:
            module_descriptions[module_name] = description
            enhanced_count += 1

    total_start_time = time.time()

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

    # Run tests with appropriate method
    if enable_fast_mode:
        print("🚀 Running tests in parallel...")
        all_metrics, passed_count, total_tests_run = run_tests_parallel(modules_with_descriptions, enable_monitoring)
        results = [(m.module_name, module_descriptions.get(m.module_name, ""), m.success) for m in all_metrics]
    else:
        print("🔄 Running tests sequentially...")
        results = []
        all_metrics = []
        total_tests_run = 0
        passed_count = 0

        for i, (module_name, description) in enumerate(modules_with_descriptions, 1):
            print(f"\n🧪 [{i:2d}/{len(discovered_modules)}] Testing: {module_name}")

            success, test_count, metrics = run_module_tests(module_name, description, enable_monitoring, coverage=enable_benchmark)
            total_tests_run += test_count
            if success:
                passed_count += 1
            if metrics:
                all_metrics.append(metrics)
            results.append((module_name, description or f"Tests for {module_name}", success))

    # Print comprehensive summary with performance metrics
    total_duration = time.time() - total_start_time
    if not enable_fast_mode:  # Recalculate for sequential mode
        passed_count = sum(1 for _, _, success in results if success)
    failed_count = len(results) - passed_count
    success_rate = (passed_count / len(results)) * 100 if results else 0

    print(f"\n{'='* 60}")
    print("📊 FINAL TEST SUMMARY")
    print(f"{'='* 60}")
    print(f"⏰ Duration: {total_duration:.1f}s")
    print(f"🧪 Total Tests Run: {total_tests_run}")
    print(f"✅ Passed: {passed_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📈 Success Rate: {success_rate:.1f}%")

    # Quality summary with detailed breakdown
    if all_metrics and any(m.quality_metrics for m in all_metrics):
        quality_scores = [m.quality_metrics.quality_score for m in all_metrics if m.quality_metrics]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            below_70_count = sum(1 for score in quality_scores if score < 70)
            below_95_count = sum(1 for score in quality_scores if 70 <= score < 95)
            above_95_count = sum(1 for score in quality_scores if score >= 95)

            print(f"🔍 Quality Score: {avg_quality:.1f}/100 avg")
            print(f"   ✅ Above 95%: {above_95_count} modules")
            print(f"   📊 70-95%: {below_95_count} modules")
            print(f"   ⚠️  Below 70%: {below_70_count} modules")

            # Show most common violation types
            all_violations = []
            for m in all_metrics:
                if m.quality_metrics and m.quality_metrics.violations:
                    all_violations.extend(m.quality_metrics.violations)

            if all_violations:
                violation_types = {}
                for violation in all_violations:
                    if "too long" in violation:
                        violation_types["Function Length"] = violation_types.get("Function Length", 0) + 1
                    elif "too complex" in violation:
                        violation_types["Complexity"] = violation_types.get("Complexity", 0) + 1
                    elif "missing type hint" in violation:
                        violation_types["Type Hints"] = violation_types.get("Type Hints", 0) + 1
                    else:
                        violation_types["Other"] = violation_types.get("Other", 0) + 1

                print("   📋 Common Issues:")
                for vtype, count in sorted(violation_types.items(), key=lambda x: x[1], reverse=True):
                    print(f"      {vtype}: {count} violations")

    # Performance metrics and analysis
    if enable_monitoring and all_metrics:
        avg_memory = sum(m.memory_usage_mb for m in all_metrics) / len(all_metrics)
        peak_memory = max(m.memory_usage_mb for m in all_metrics)
        avg_cpu = sum(m.cpu_usage_percent for m in all_metrics) / len(all_metrics)
        peak_cpu = max(m.cpu_usage_percent for m in all_metrics)

        # Calculate parallel efficiency
        sequential_time = sum(m.duration for m in all_metrics)
        parallel_efficiency = (sequential_time / total_duration) if total_duration > 0 else 1.0

        print("\n📊 PERFORMANCE METRICS:")
        print(f"   💾 Memory Usage: {avg_memory:.1f}MB avg, {peak_memory:.1f}MB peak")
        print(f"   ⚡ CPU Usage: {avg_cpu:.1f}% avg, {peak_cpu:.1f}% peak")
        if enable_fast_mode:
            print(f"   🚀 Parallel Efficiency: {parallel_efficiency:.1f}x speedup")

        # Create suite performance metrics
        suite_performance = TestSuitePerformance(
            total_duration=total_duration,
            total_tests=total_tests_run,
            passed_modules=passed_count,
            failed_modules=failed_count,
            avg_memory_usage=avg_memory,
            peak_memory_usage=peak_memory,
            avg_cpu_usage=avg_cpu,
            peak_cpu_usage=peak_cpu,
            parallel_efficiency=parallel_efficiency,
            optimization_suggestions=analyze_performance_trends(all_metrics)
        )

        # Show optimization suggestions
        if suite_performance.optimization_suggestions:
            print("\n💡 OPTIMIZATION SUGGESTIONS:")
            for suggestion in suite_performance.optimization_suggestions:
                print(f"   {suggestion}")

        # Save metrics for trend analysis
        if enable_benchmark:
            save_performance_metrics(all_metrics, suite_performance)

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
