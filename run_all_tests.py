#!/usr/bin/env python3

"""
Comprehensive Test Runner for Ancestry Project
Runs all unit tests and integration tests across the entire project with detailed reporting.

This unified test runner provides:
• Comprehensive scoring breakdowns showing what was tested
• Detailed outcomes achieved with specific results
• Conclusions drawn from test results
• Clear pass/fail status for each test

Usage:
    python run_all_tests.py           # Run all tests with enhanced detailed reporting
"""

import sys
import time
import subprocess
from pathlib import Path


def discover_test_modules():
    """Discover all Python modules that contain tests by scanning the project directory."""
    project_root = Path(__file__).parent
    test_modules = []

    # Get all Python files in the project
    for python_file in project_root.rglob("*.py"):
        # Skip the test runner itself, main.py, and other non-test files
        if python_file.name in [
            "run_all_tests.py",
            "main.py",
            "__init__.py",
            "__main__.py",
            "credentials.py",  # Interactive credential manager
            "core_imports.py",  # Import utility, not a test module
            "standard_imports.py",  # Import utility, not a test module
        ]:
            continue

        # Skip cache, backup, temp files, and anything in __pycache__
        file_path_str = str(python_file)
        if (
            "__pycache__" in file_path_str
            or python_file.name.endswith("_backup.py")
            or "backup_before_migration" in file_path_str
            or "temp" in python_file.name.lower()
            or "_old" in python_file.name
            or python_file.name.startswith("phase1_cleanup")
            or python_file.name.startswith("test_phase1")
            or python_file.name.startswith("cleanup_")
            or python_file.name.startswith("migration_")
            or python_file.name.startswith("fix_")
            or python_file.name.startswith("convert_")
            or ".venv" in file_path_str
            or "site-packages" in file_path_str
        ):
            continue

        # Check if the file has test functionality by looking for test patterns
        try:
            with open(python_file, "r", encoding="utf-8") as f:
                content = f.read()

                # Skip files with interactive components
                has_interactive = any(
                    pattern in content
                    for pattern in [
                        "input(",
                        "getpass.getpass",
                        "Enter choice:",
                        "Select option:",
                        "Press any key",
                        "while True:",  # Often indicates interactive loops
                    ]
                )

                if has_interactive:
                    continue

                # Look for test patterns that indicate this file has tests
                has_tests = any(
                    pattern in content
                    for pattern in [
                        "run_comprehensive_tests",
                        'if __name__ == "__main__"',
                        "TestSuite(",
                        "def test_",
                        "run_test(",
                    ]
                )

                if has_tests:
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
        with open(module_path, 'r', encoding='utf-8') as f:
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
                else:
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
    module_name: str, description: str | None = None
) -> tuple[bool, int]:
    """Run tests for a specific module and return success status with clean output"""
    # Show description for consistency - avoid repeating module name
    if description:
        print(f"   📝 {description}")
    else:
        # Create a meaningful description based on module name instead of just repeating it
        if "core/" in module_name:
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
        result = subprocess.run(
            [sys.executable, module_name],
            capture_output=True,  # Capture output to check for failures
            text=True,
            cwd=Path.cwd(),
        )
        duration = time.time() - start_time

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
                    import re

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

                match = re.search(r"(\d+)", test_count)
                if match:
                    numeric_test_count = int(match.group(1))
            except (ValueError, AttributeError):
                numeric_test_count = 0

        # If failed, show error details
        if not success:
            print(f"   🚨 Failure Details:")
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

        return success, numeric_test_count

    except Exception as e:
        print(f"   ❌ FAILED | Error: {e}")
        return False, 0


def main():
    """Main test runner with comprehensive reporting"""
    print("\nANCESTRY PROJECT - COMPREHENSIVE TEST SUITE")
    print("=" * 50)
    print()  # Blank line instead of subtitle

    # Auto-discover all test modules
    discovered_modules = discover_test_modules()

    # Ensure action10.py and action11.py are included
    must_have = ["action10.py", "action11.py"]
    for mod in must_have:
        if mod not in discovered_modules:
            discovered_modules.append(mod)

    if not discovered_modules:
        print("⚠️  No test modules discovered.")
        return False

    # Extract descriptions from module docstrings
    module_descriptions = {}
    enhanced_count = 0

    for module_name in discovered_modules:
        description = extract_module_description(module_name)
        if description:
            module_descriptions[module_name] = description
            enhanced_count += 1

    results = []
    total_start_time = time.time()

    print(
        f"📊 Found {len(discovered_modules)} test modules ({enhanced_count} with enhanced descriptions)"
    )

    print(f"\n{'='* 50}")
    print(f"🧪 RUNNING TESTS")
    print(f"{'='* 50}")

    total_tests_run = 0

    for i, module_name in enumerate(discovered_modules, 1):
        print(f"\n🧪 [{i:2d}/{len(discovered_modules)}] Testing: {module_name}")

        # Use extracted description if available, otherwise generate a basic one
        description = module_descriptions.get(module_name, None)

        success, test_count = run_module_tests(module_name, description)
        total_tests_run += test_count
        results.append(
            (module_name, description or f"Tests for {module_name}", success)
        )

    # Print comprehensive summary
    total_duration = time.time() - total_start_time
    passed_count = sum(1 for _, _, success in results if success)
    failed_count = len(results) - passed_count
    success_rate = (passed_count / len(results)) * 100 if results else 0

    print(f"\n{'='* 50}")
    print(f"📊 FINAL TEST SUMMARY")
    print(f"{'='* 50}")
    print(f"⏰ Duration: {total_duration:.1f}s")
    print(f"🧪 Total Tests Run: {total_tests_run}")
    print(f"✅ Passed: {passed_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📈 Success Rate: {success_rate:.1f}%")

    # Show failed modules first if any
    if failed_count > 0:
        print(f"\n❌ FAILED MODULES:")
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

    print(f"\n📋 RESULTS BY CATEGORY:")
    print(f"   Enhanced Modules: {enhanced_passed} passed, {enhanced_failed} failed")
    print(
        f"   Standard Modules: {passed_count - enhanced_passed} passed, {failed_count - enhanced_failed} failed"
    )

    if failed_count == 0:
        print(f"\n🎉 ALL {len(discovered_modules)} MODULES PASSED!")
        print("   Enhanced detailed reporting is working perfectly.\n\n")
    else:
        print(f"\n⚠️  {failed_count} module(s) failed.")
        print("   Check individual test outputs above for details.\n\n")

    return failed_count == 0


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test run interrupted by user!\n\n")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error in test runner: {e}\n\n")
        sys.exit(1)
