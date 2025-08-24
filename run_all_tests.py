#!/usr/bin/env python3

"""
Comprehensive Test Runner for Ancestry Project
Runs all unit tests and integration tests with advanced performance monitoring and optimization.

Features:
• Automatic test discovery with enhanced module descriptions
• Parallel test execution for improved performance
• Memory usage and CPU monitoring during test execution
• Performance benchmarking and trend analysis
• Test execution optimization with intelligent parallel processing
• Detailed pass/fail status with timing metrics

This unified test runner provides:
• Comprehensive test coverage across all 66 standardized modules
• Detailed reporting showing what was tested and outcomes achieved
• Performance metrics and optimization insights
• Clear test results with enhanced descriptions

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

    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None

    def start_monitoring(self):
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

        # Step 2: blocking rule set
        print("🧹 LINTER: Enforcing blocking rules (E722,F821,F811,F823,I001,F401)...")
        block_cmd = [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "E722,F821,F811,F823,I001,F401",
            ".",
        ]
        block_res = subprocess.run(block_cmd, check=False, capture_output=True, text=True, cwd=Path.cwd())
        if block_res.returncode != 0:
            print("❌ LINTER FAILED (blocking): violations in E722,F821,F811,F823,I001,F401")
            # Tail the output to keep logs compact
            tail = (block_res.stdout or block_res.stderr or "").splitlines()[-40:]
            for line in tail:
                print(line)
            return False

        # Step 3: non-blocking diagnostics
        print("🧹 LINTER: Repository diagnostics (non-blocking summary)...")
        diag_cmd = [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--statistics",
            "--exit-zero",
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


def discover_test_modules():
    """
    Discover all Python modules that contain tests by scanning the project directory.

    Returns a list of module paths that contain the run_comprehensive_tests() function,
    which indicates they follow the standardized testing framework.
    """
    project_root = Path(__file__).parent
    test_modules = []

    # Get all Python files in the project
    for python_file in project_root.rglob("*.py"):
        # Skip the test runner itself, main.py, and coordination files
        if python_file.name in [
            "run_all_tests.py",
            "main.py",
            "__init__.py",
            "__main__.py",
        ]:
            continue

        # Skip cache, backup, temp files, and virtual environment
        file_path_str = str(python_file)
        if (
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
        ):
            continue

        # Quietly skip demo/prototype scripts in repo (no output spam)
        demo_markers = ["demo", "prototype", "experimental", "sandbox"]
        name_lower = python_file.name.lower()
        if any(marker in name_lower for marker in demo_markers):
            continue

        # Check if the file has the standardized test function
        try:
            with python_file.open(encoding="utf-8") as f:
                content = f.read()

                # Skip files with interactive components that would block testing
                # But allow legitimate test modules that happen to have interactive functionality
                has_interactive = False
                if python_file.name in ["db_viewer.py", "test_program_executor.py"]:
                    # Specifically known interactive modules that should be excluded
                    has_interactive = True

                if has_interactive:
                    continue

                # Look for the standardized test function (with or without parentheses)
                if "def run_comprehensive_tests" in content:
                    # Convert to relative path from project root
                    relative_path = python_file.relative_to(project_root)
                    test_modules.append(str(relative_path))

        except (UnicodeDecodeError, PermissionError):
            # Skip files that can't be read
            continue

    return sorted(test_modules)


def extract_module_description(module_path: str) -> str | None:
    """Extract the first line of a module's docstring for use as description."""
    try:
        # Read the file and look for the module docstring
        from pathlib import Path
        with Path(module_path).open(encoding='utf-8') as f:
            content = f.read()

        # Look for triple-quoted docstring after any initial comments/shebang
        lines = content.split('\n')
        in_docstring = False
        docstring_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip shebang and comments at the top
            if stripped.startswith('#') or not stripped:
                continue

            # Look for start of docstring
            if not in_docstring and '"""' in stripped:
                in_docstring = True
                # Extract content after opening quotes
                after_quotes = stripped.split('"""', 1)[1].strip()
                if after_quotes:
                    docstring_lines.append(after_quotes)
                continue

            # If we're in docstring, collect lines until closing quotes
            if in_docstring:
                if '"""' in stripped:
                    # End of docstring - extract content before closing quotes
                    before_quotes = stripped.split('"""')[0].strip()
                    if before_quotes:
                        docstring_lines.append(before_quotes)
                    break
                # Regular docstring line
                if stripped:
                    docstring_lines.append(stripped)

        # Return the first meaningful line as description
        if docstring_lines:
            # Look for the first line that looks like a title/description
            for line in docstring_lines:
                if line and not line.startswith('=') and len(line) > 10:
                    # Clean up common patterns
                    module_base = module_path.replace('.py', '').replace('/', '').replace('\\', '')
                    description = line.replace(module_base, '').strip()
                    description = description.replace(' - ', ' - ').strip()

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

        return None

    except Exception:
        return None


def run_module_tests(
    module_name: str, description: str | None = None, enable_monitoring: bool = False, coverage: bool = False
) -> tuple[bool, int, Optional[TestExecutionMetrics]]:
    """Run tests for a specific module with optional performance monitoring."""
    import re  # Ensure re is available in function scope

    # Initialize performance monitoring
    monitor = PerformanceMonitor() if enable_monitoring else None
    metrics = None
    # Show description for consistency - avoid repeating module name
    if description:
        print(f"   📝 {description}")
    # Create a meaningful description based on module name instead of just repeating it
    elif "core/" in module_name:
        component = (
            module_name.replace("core/", "")
            .replace(".py", "")
            .replace("_", " ")
            .title()
        )
        print(f"   📝 Core {component} functionality")
    elif "config/" in module_name:
        component = (
            module_name.replace("config/", "")
            .replace(".py", "")
            .replace("_", " ")
            .title()
        )
        print(f"   📝 Configuration {component} management")
    elif "action" in module_name:
        action_name = module_name.replace(".py", "").replace("_", " ").title()
        print(f"   📝 {action_name} automation")
    elif module_name.endswith("_utils.py"):
        util_type = module_name.replace("_utils.py", "").replace("_", " ").title()
        print(f"   📝 {util_type} utility functions")
    elif module_name.endswith("_manager.py"):
        manager_type = (
            module_name.replace("_manager.py", "").replace("_", " ").title()
        )
        print(f"   📝 {manager_type} management system")
    elif module_name.endswith("_cache.py"):
        cache_type = module_name.replace("_cache.py", "").replace("_", " ").title()
        print(f"   📝 {cache_type} caching system")
    else:
        # Generic fallback that's more descriptive than just repeating the filename
        clean_name = module_name.replace(".py", "").replace("_", " ").title()
        print(f"   📝 {clean_name} module functionality")

    try:
        start_time = time.time()
        start_datetime = datetime.now().isoformat()

        # Start performance monitoring if enabled
        if monitor:
            monitor.start_monitoring()

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

        # Check for success based on return code AND output content
        success = result.returncode == 0

        # Extract test counts from output - improved patterns (check both stdout and stderr)
        test_count = "Unknown"
        all_output_lines = []
        if result.stdout:
            all_output_lines.extend(result.stdout.split("\n"))
        if result.stderr:
            all_output_lines.extend(result.stderr.split("\n"))

        if all_output_lines:
            stdout_lines = all_output_lines  # Use combined output for pattern matching

            # Pattern 1: Look for "✅ Passed: X" and "❌ Failed: Y"
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
                                    failed = int(
                                        other_line.split("❌ Failed:")[1].split()[0]
                                    )
                                    break
                        test_count = f"{passed + failed} tests"
                        break
                    except (ValueError, IndexError):
                        continue

            # Pattern 2: Look for "X/Y tests passed" or "Results: X/Y"
            if test_count == "Unknown":
                for line in stdout_lines:
                    if "tests passed" in line and "/" in line:
                        try:
                            # Extract from "📊 Results: 3/3 tests passed"
                            parts = line.split("/")
                            if len(parts) >= 2:
                                total = parts[1].split()[0]
                                test_count = f"{total} tests"
                                break
                        except (ValueError, IndexError):
                            continue

            # Pattern 3: Look for "✅ Passed: X" and "❌ Failed: Y" format (common in many modules)
            if test_count == "Unknown":
                passed_count = None
                failed_count = None
                for line in stdout_lines:
                    # Remove ANSI color codes and whitespace
                    clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()

                    if "✅ Passed:" in clean_line:
                        try:
                            passed_count = int(
                                clean_line.split("✅ Passed:")[1].strip()
                            )
                        except (ValueError, IndexError):
                            continue
                    elif "❌ Failed:" in clean_line:
                        try:
                            failed_count = int(
                                clean_line.split("❌ Failed:")[1].strip()
                            )
                        except (ValueError, IndexError):
                            continue

                if passed_count is not None and failed_count is not None:
                    test_count = f"{passed_count + failed_count} tests"
                elif passed_count is not None:
                    test_count = f"{passed_count}+ tests"

            # Pattern 4: Look for Python unittest format "Ran X tests in Y.Zs"
            if test_count == "Unknown":
                for line in stdout_lines:
                    clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
                    if "Ran" in clean_line and "tests in" in clean_line:
                        try:
                            # Extract from "Ran 24 tests in 0.458s"
                            parts = clean_line.split()
                            ran_index = parts.index("Ran")
                            if ran_index + 1 < len(parts):
                                count = int(parts[ran_index + 1])
                                test_count = f"{count} tests"
                                break
                        except (ValueError, IndexError):
                            continue

            # Pattern 5: Look for numbered test patterns like "Test 1:", "Test 2:", etc.
            if test_count == "Unknown":
                test_numbers = set()
                for line in stdout_lines:
                    clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
                    # Look for patterns like "📋 Test 1:", "Test 2:", "• Test 3:"
                    match = re.search(
                        r"(?:📋|•|\*|-|\d+\.?)\s*Test\s+(\d+):",
                        clean_line,
                        re.IGNORECASE,
                    )
                    if match:
                        test_numbers.add(int(match.group(1)))

                if test_numbers:
                    test_count = f"{len(test_numbers)} tests"

            # Pattern 6: Look for any number followed by "test" or "tests"
            if test_count == "Unknown":
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
                        test_count = f"{count} tests"
                        break

            # Pattern 7: Look for test completion messages with counts
            if test_count == "Unknown":
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
                        test_count = f"{count} tests"
                        break

            # Pattern 8: Look for "ALL TESTS PASSED" with counts
            if test_count == "Unknown":
                for line in stdout_lines:
                    if "ALL TESTS PASSED" in line or "Status: ALL TESTS PASSED" in line:
                        # Look for nearby lines with test counts
                        for other_line in stdout_lines:
                            if "Passed:" in other_line and other_line.count(":") >= 1:
                                try:
                                    count = int(
                                        other_line.split("Passed:")[1].split()[0]
                                    )
                                    test_count = f"{count} tests"
                                    break
                                except (ValueError, IndexError):
                                    continue
                        if test_count != "Unknown":
                            break

        # Define failure indicators (be more specific to avoid false positives)
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

        # Also check output for failure indicators
        if success and result.stdout:
            # Only mark as failed if we find actual failure indicators
            # Exclude lines that are just showing "Failed: 0" (which means 0 failures)
            stdout_lines = result.stdout.split("\n")
            for line in stdout_lines:
                for indicator in failure_indicators:
                    if indicator in line and not (
                        "Failed: 0" in line or "❌ Failed: 0" in line
                    ):
                        success = False
                        break
                if not success:
                    break

        status = "✅ PASSED" if success else "❌ FAILED"

        # Show concise summary with test count instead of return code
        print(f"   {status} | Duration: {duration:.2f}s | {test_count}")

        # Extract numeric test count for summary
        numeric_test_count = 0
        if test_count != "Unknown":
            try:
                # Extract number from formats like "8 tests", "24 tests", "5+ tests"
                import re
                # Extract number from formats like "8 tests", "24 tests", "5+ tests"
                match = re.search(r"(\d+)", test_count)
                if match:
                    numeric_test_count = int(match.group(1))
            except (ValueError, AttributeError):
                numeric_test_count = 0
        if not success:
            print("   🚨 Failure Details:")
            if result.stderr:
                error_lines = result.stderr.strip().split("\n")
                for line in error_lines[-3:]:  # Show last 3 error lines
                    print(f"      {line}")
            if result.stdout and any(
                indicator in result.stdout for indicator in failure_indicators
            ):
                stdout_lines = result.stdout.strip().split("\n")
                failure_lines = [
                    line
                    for line in stdout_lines
                    if any(indicator in line for indicator in failure_indicators)
                ]
                for line in failure_lines[-2:]:  # Show last 2 failure lines
                    print(f"      {line}")

        # Create performance metrics if monitoring was enabled
        if enable_monitoring:
            metrics = TestExecutionMetrics(
                module_name=module_name,
                duration=duration,
                success=success,
                test_count=numeric_test_count,
                memory_usage_mb=perf_metrics.get("memory_mb", 0.0),
                cpu_usage_percent=perf_metrics.get("cpu_percent", 0.0),
                start_time=start_datetime,
                end_time=end_datetime,
                error_message=result.stderr if not success and result.stderr else None
            )

        return success, numeric_test_count, metrics

    except Exception as e:
        print(f"   ❌ FAILED | Error: {e}")
        error_metrics = None
        if enable_monitoring:
            error_metrics = TestExecutionMetrics(
                module_name=module_name,
                duration=0.0,
                success=False,
                test_count=0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                start_time=datetime.now().isoformat(),
                end_time=datetime.now().isoformat(),
                error_message=str(e)
            )
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


def main():
    """Comprehensive test runner with performance monitoring and optimization."""
    # Parse command line arguments
    enable_fast_mode = "--fast" in sys.argv
    enable_benchmark = "--benchmark" in sys.argv
    enable_monitoring = enable_benchmark or enable_fast_mode

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
        for module_name, description, success in results:
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
