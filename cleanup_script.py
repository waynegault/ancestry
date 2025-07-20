#!/usr/bin/env python3
"""
Codebase Cleanup Script for Ancestry Project

This script addresses the major issues identified in the codebase review:
1. Incomplete import statements
2. Duplicate module registrations
3. Inconsistent import patterns
4. Function registry references
5. Logging import standardization

Usage: python cleanup_script.py
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict


def main():
    """Main cleanup function."""
    print("🧹 Starting Ancestry Project Codebase Cleanup")
    print("=" * 60)

    project_root = Path(__file__).parent
    python_files = list(project_root.rglob("*.py"))

    # Filter out this cleanup script and __pycache__
    python_files = [
        f
        for f in python_files
        if not any(
            part.startswith("__pycache__") or part == "cleanup_script.py"
            for part in f.parts
        )
    ]

    print(f"📁 Found {len(python_files)} Python files to process")

    issues_summary = {
        "incomplete_imports": [],
        "duplicate_registrations": [],
        "legacy_logging": [],
        "function_registry_refs": [],
        "files_processed": 0,
        "total_issues": 0,
    }

    for file_path in python_files:
        try:
            process_file(file_path, issues_summary)
            issues_summary["files_processed"] += 1
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    print_summary(issues_summary)
    generate_recommendations(issues_summary)


def process_file(file_path: Path, issues_summary: Dict) -> None:
    """Process a single Python file for cleanup issues."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return

    # Check for issues
    check_incomplete_imports(file_path, content, issues_summary)
    check_duplicate_registrations(file_path, content, issues_summary)
    check_legacy_logging(file_path, content, issues_summary)
    check_function_registry_refs(file_path, content, issues_summary)


def check_incomplete_imports(
    file_path: Path, content: str, issues_summary: Dict
) -> None:
    """Check for incomplete import statements."""
    patterns = [
        r"from core_imports import\s*$",
        r"from core_imports import\s*\(\s*$",
    ]

    for pattern in patterns:
        if re.search(pattern, content, re.MULTILINE):
            issues_summary["incomplete_imports"].append(str(file_path))
            issues_summary["total_issues"] += 1
            break


def check_duplicate_registrations(
    file_path: Path, content: str, issues_summary: Dict
) -> None:
    """Check for duplicate auto_register_module calls."""
    pattern = r"auto_register_module\(globals\(\), __name__\)"
    matches = re.findall(pattern, content)

    if len(matches) > 1:
        issues_summary["duplicate_registrations"].append((str(file_path), len(matches)))
        issues_summary["total_issues"] += 1


def check_legacy_logging(file_path: Path, content: str, issues_summary: Dict) -> None:
    """Check for direct logging imports instead of using unified system."""
    if re.search(r"^import logging", content, re.MULTILINE):
        # Skip if it's the logging_config.py file itself
        if "logging_config.py" not in str(file_path):
            issues_summary["legacy_logging"].append(str(file_path))
            issues_summary["total_issues"] += 1


def check_function_registry_refs(
    file_path: Path, content: str, issues_summary: Dict
) -> None:
    """Check for legacy function_registry references."""
    patterns = [
        r"function_registry\s*=",
        r"function_registry\.is_available",
        r"get_function_registry\(\)",
    ]

    for pattern in patterns:
        if re.search(pattern, content):
            issues_summary["function_registry_refs"].append(str(file_path))
            issues_summary["total_issues"] += 1
            break


def print_summary(issues_summary: Dict) -> None:
    """Print cleanup summary."""
    print("\n📊 CLEANUP SUMMARY")
    print("=" * 60)
    print(f"Files processed: {issues_summary['files_processed']}")
    print(f"Total issues found: {issues_summary['total_issues']}")
    print()

    if issues_summary["incomplete_imports"]:
        print(f"🚨 Incomplete Imports ({len(issues_summary['incomplete_imports'])}):")
        for file_path in issues_summary["incomplete_imports"][:5]:
            print(f"  - {Path(file_path).name}")
        if len(issues_summary["incomplete_imports"]) > 5:
            print(f"  ... and {len(issues_summary['incomplete_imports']) - 5} more")
        print()

    if issues_summary["duplicate_registrations"]:
        print(
            f"🔄 Duplicate Registrations ({len(issues_summary['duplicate_registrations'])}):"
        )
        for file_path, count in issues_summary["duplicate_registrations"][:5]:
            print(f"  - {Path(file_path).name}: {count} registrations")
        if len(issues_summary["duplicate_registrations"]) > 5:
            print(
                f"  ... and {len(issues_summary['duplicate_registrations']) - 5} more"
            )
        print()

    if issues_summary["legacy_logging"]:
        print(f"📝 Legacy Logging Imports ({len(issues_summary['legacy_logging'])}):")
        for file_path in issues_summary["legacy_logging"][:5]:
            print(f"  - {Path(file_path).name}")
        if len(issues_summary["legacy_logging"]) > 5:
            print(f"  ... and {len(issues_summary['legacy_logging']) - 5} more")
        print()

    if issues_summary["function_registry_refs"]:
        print(
            f"🏷️  Function Registry Refs ({len(issues_summary['function_registry_refs'])}):"
        )
        for file_path in issues_summary["function_registry_refs"][:5]:
            print(f"  - {Path(file_path).name}")
        if len(issues_summary["function_registry_refs"]) > 5:
            print(f"  ... and {len(issues_summary['function_registry_refs']) - 5} more")


def generate_recommendations(issues_summary: Dict) -> None:
    """Generate specific recommendations for cleanup."""
    print("\n🎯 RECOMMENDED ACTIONS")
    print("=" * 60)

    if issues_summary["incomplete_imports"]:
        print("1. Fix Incomplete Imports:")
        print("   Replace incomplete 'from core_imports import' statements with:")
        print("   ```python")
        print("   from core_imports import (")
        print("       standardize_module_imports,")
        print("       auto_register_module,")
        print("       get_logger,")
        print("       safe_execute,")
        print("   )")
        print("   ```")
        print()

    if issues_summary["duplicate_registrations"]:
        print("2. Remove Duplicate Registrations:")
        print(
            "   Keep only ONE auto_register_module() call per file, preferably near the top."
        )
        print()

    if issues_summary["legacy_logging"]:
        print("3. Standardize Logging:")
        print("   Replace 'import logging' with 'logger = get_logger(__name__)'")
        print("   from the core_imports system.")
        print()

    if issues_summary["function_registry_refs"]:
        print("4. Update Function Registry References:")
        print("   Replace function_registry calls with:")
        print(
            "   - is_function_available(name) instead of function_registry.is_available(name)"
        )
        print("   - get_function(name) instead of function_registry.get(name)")
        print()

    print("5. Consolidation Opportunities:")
    print("   - Centralize run_comprehensive_tests() implementations")
    print("   - Use unified import template for all files")
    print("   - Remove legacy fallback patterns")
    print()

    print("See STANDARD_IMPORT_TEMPLATE.py for the recommended import pattern.")


if __name__ == "__main__":
    main()
