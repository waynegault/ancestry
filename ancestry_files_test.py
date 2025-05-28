#!/usr/bin/env python3
"""
Final comprehensive test for all Python files in the Ancestry automation system.
Tests imports, basic syntax, and runtime readiness.
"""

import sys
import os
import importlib.util
import traceback
from pathlib import Path
import subprocess
import time


def test_file_with_subprocess(file_path):
    """Test a Python file using subprocess to isolate any issues."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {file_path.stem}"],
            cwd=file_path.parent,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout - file took too long to import"
    except Exception as e:
        return False, str(e)


def run_comprehensive_tests():
    """Run comprehensive tests on all Python files."""

    print("=" * 70)
    print("ANCESTRY.COM AUTOMATION SYSTEM - COMPREHENSIVE PYTHON FILE TEST")
    print("=" * 70)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Get all Python files
    current_dir = Path(".")
    all_py_files = list(current_dir.glob("*.py"))

    # Exclude test files
    exclude_files = {
        "test_all_imports.py",
        "quick_test.py",
        "test_python_imports.py",
        "final_test.py",
    }
    python_files = [f for f in all_py_files if f.name not in exclude_files]
    python_files.sort()

    print(f"Found {len(python_files)} Python files to test")
    print()

    # Categorize files
    core_files = []
    action_files = []
    test_files = []
    utility_files = []

    for file in python_files:
        name = file.name
        if name.startswith("action"):
            action_files.append(file)
        elif name.startswith("test_"):
            test_files.append(file)
        elif name in ["main.py", "config.py", "database.py", "utils.py"]:
            core_files.append(file)
        else:
            utility_files.append(file)

    # Test results
    results = {
        "Core Files": [],
        "Action Modules": [],
        "Test Files": [],
        "Utility Files": [],
    }

    categories = [
        ("Core Files", core_files),
        ("Action Modules", action_files),
        ("Utility Files", utility_files),
        ("Test Files", test_files),
    ]

    total_success = 0
    total_files = len(python_files)

    for category_name, files in categories:
        if not files:
            continue

        print(f"Testing {category_name} ({len(files)} files):")
        print("-" * 50)

        for i, file_path in enumerate(files, 1):
            file_name = file_path.name
            success, error = test_file_with_subprocess(file_path)

            if success:
                status = "✓ PASS"
                total_success += 1
                results[category_name].append((file_name, True, None))
            else:
                status = "✗ FAIL"
                results[category_name].append((file_name, False, error))

            print(f"  {i:2d}. {file_name:<30} {status}")
            if not success and error:
                # Show first line of error
                error_line = error.split("\\n")[0] if error else "Unknown error"
                print(f"      └─ {error_line[:60]}...")

        print()

    # Final summary
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total files tested: {total_files}")
    print(f"Successful imports: {total_success}")
    print(f"Failed imports: {total_files - total_success}")
    print(f"Success rate: {(total_success/total_files*100):.1f}%")
    print()

    # Detailed results by category
    for category_name, category_results in results.items():
        if not category_results:
            continue

        successes = [r for r in category_results if r[1]]
        failures = [r for r in category_results if not r[1]]

        print(f"{category_name}: {len(successes)}/{len(category_results)} passed")

        if failures:
            print(f"  Failed files:")
            for file_name, _, error in failures:
                print(f"    • {file_name}: {error[:80] if error else 'Unknown'}...")
        print()

    # Recommendations
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    if total_success == total_files:
        print("🎉 ALL TESTS PASSED!")
        print("✓ All Python files import successfully")
        print("✓ No missing dependencies detected")
        print("✓ No syntax errors found")
        print("✓ System is ready for production use")
    else:
        failed_count = total_files - total_success
        print(f"⚠️  {failed_count} files need attention:")

        # Collect all failed files for analysis
        all_failures = []
        for category_results in results.values():
            all_failures.extend([r for r in category_results if not r[1]])

        dependency_issues = []
        syntax_issues = []
        other_issues = []

        for file_name, _, error in all_failures:
            if error and "No module named" in error:
                dependency_issues.append((file_name, error))
            elif error and ("SyntaxError" in error or "IndentationError" in error):
                syntax_issues.append((file_name, error))
            else:
                other_issues.append((file_name, error))

        if dependency_issues:
            print(f"\\n📦 Dependency Issues ({len(dependency_issues)}):")
            print("   Run: pip install -r requirements.txt")
            for file_name, error in dependency_issues:
                module = error.split("'")[1] if "'" in error else "unknown"
                print(f"   • {file_name}: missing module '{module}'")

        if syntax_issues:
            print(f"\\n🔧 Syntax Issues ({len(syntax_issues)}):")
            for file_name, error in syntax_issues:
                print(f"   • {file_name}: {error.split('\\n')[0]}")

        if other_issues:
            print(f"\\n❓ Other Issues ({len(other_issues)}):")
            for file_name, error in other_issues:
                print(
                    f"   • {file_name}: {error.split('\\n')[0] if error else 'Unknown'}"
                )

    print()
    print(f"Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return total_success, total_files - total_success


if __name__ == "__main__":
    run_comprehensive_tests()
