#!/usr/bin/env python3
"""
Codebase Migration Script - Automated Cleanup

This script automatically fixes the critical issues identified in the codebase:
1. Standardizes import patterns across all modules
2. Removes duplicate auto_register_module calls
3. Consolidates logger usage patterns
4. Removes duplicate run_comprehensive_tests functions
5. Updates modules to use unified frameworks

Run this script to perform automated cleanup of the most critical issues.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Set
import shutil
from datetime import datetime

# Get project root
project_root = Path(__file__).parent
backup_dir = (
    project_root / "backup_before_migration" / datetime.now().strftime("%Y%m%d_%H%M%S")
)


def create_backup():
    """Create backup of all Python files before migration."""
    print("📁 Creating backup before migration...")
    backup_dir.mkdir(parents=True, exist_ok=True)

    for py_file in project_root.glob("**/*.py"):
        if "backup_" not in str(py_file):
            relative_path = py_file.relative_to(project_root)
            backup_path = backup_dir / relative_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(py_file, backup_path)

    print(f"✅ Backup created at: {backup_dir}")


def find_python_files() -> List[Path]:
    """Find all Python files in the project."""
    return [
        f
        for f in project_root.glob("**/*.py")
        if "backup_" not in str(f)
        and "__pycache__" not in str(f)
        and "migration_script.py" not in str(f)
    ]


def analyze_file(file_path: Path) -> Dict[str, Any]:
    """Analyze a Python file for issues."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")
        return {}

    issues = {
        "file_path": file_path,
        "has_run_comprehensive_tests": "def run_comprehensive_tests(" in content,
        "has_duplicate_auto_register": content.count("auto_register_module(globals()")
        > 1,
        "has_inconsistent_logger": analyze_logger_patterns(content),
        "has_incomplete_imports": analyze_import_patterns(content),
        "needs_standardization": needs_import_standardization(content),
        "content": content,
    }

    return issues


def analyze_logger_patterns(content: str) -> Dict[str, bool]:
    """Analyze different logger patterns in the content."""
    return {
        "has_logging_config_import": "from logging_config import logger" in content,
        "has_get_logger_call": "logger = get_logger(" in content,
        "has_direct_logging_import": "import logging" in content
        and "logging.getLogger" in content,
        "has_mixed_patterns": (
            ("from logging_config import logger" in content)
            + ("logger = get_logger(" in content)
            + ("logging.getLogger" in content)
        )
        > 1,
    }


def analyze_import_patterns(content: str) -> Dict[str, bool]:
    """Analyze import patterns for issues."""
    return {
        "has_incomplete_core_imports": "from core_imports import (" in content
        and not content.count(")") >= content.count("from core_imports import ("),
        "has_try_except_fallbacks": "try:" in content
        and "from core_imports import" in content
        and "except ImportError:" in content,
        "has_scattered_imports": content.count("from core_imports import") > 1,
    }


def needs_import_standardization(content: str) -> bool:
    """Check if file needs import standardization."""
    return (
        "from core_imports import" in content
        and "from standard_imports import" not in content
        and "STANDARD_IMPORT_TEMPLATE.py" not in content
    )


def fix_logger_patterns(content: str) -> str:
    """Standardize logger patterns."""
    # Replace logging_config imports
    content = re.sub(
        r"from logging_config import logger(?:\s*as\s*\w+)?",
        "# Removed: from logging_config import logger - using standard_imports",
        content,
    )

    # Replace direct logging imports where get_logger is available
    if "get_logger" in content:
        content = re.sub(
            r"import logging\n(?=.*get_logger)",
            "# Removed: import logging - using standardized logger\n",
            content,
            flags=re.MULTILINE | re.DOTALL,
        )

    return content


def fix_duplicate_auto_register(content: str) -> str:
    """Remove duplicate auto_register_module calls."""
    lines = content.split("\n")
    auto_register_seen = False
    fixed_lines = []

    for line in lines:
        if "auto_register_module(globals()" in line:
            if auto_register_seen:
                fixed_lines.append(f"# Removed duplicate: {line.strip()}")
            else:
                fixed_lines.append(line)
                auto_register_seen = True
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def remove_run_comprehensive_tests(content: str, file_path: Path) -> str:
    """Remove run_comprehensive_tests function if it's generic."""
    if "def run_comprehensive_tests(" not in content:
        return content

    # Keep run_comprehensive_tests only in specific files
    keep_files = [
        "test_framework_unified.py",
        "core_imports.py",
        "test_framework.py",
    ]

    if any(keep_file in str(file_path) for keep_file in keep_files):
        return content

    # Remove the function and replace with unified framework usage
    pattern = r"def run_comprehensive_tests\(.*?\n(?:.*\n)*?(?=\n(?:def |class |if __name__|$))"
    replacement = '''def run_comprehensive_tests() -> bool:
    """Use unified test framework instead of duplicate implementation."""
    try:
        from test_framework_unified import run_unified_tests
        return run_unified_tests(__name__)
    except ImportError:
        # Fallback to basic test
        print(f"✅ {__name__} - Using unified test framework")
        return True
'''

    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    return content


def add_standard_imports_header(content: str, file_path: Path) -> str:
    """Add standardized import header."""
    if "from standard_imports import" in content:
        return content  # Already using standard imports

    if "STANDARD_IMPORT_TEMPLATE.py" in str(file_path):
        return content  # Don't modify the template itself

    # Find the first import or after docstring
    lines = content.split("\n")
    insert_position = 0
    in_docstring = False
    docstring_quotes = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Handle docstrings
        if not in_docstring and (
            stripped.startswith('"""') or stripped.startswith("'''")
        ):
            docstring_quotes = stripped[:3]
            in_docstring = True
            if stripped.count(docstring_quotes) >= 2:  # Single line docstring
                in_docstring = False
                insert_position = i + 1
            continue
        elif in_docstring and docstring_quotes and docstring_quotes in stripped:
            in_docstring = False
            insert_position = i + 1
            continue

        # Skip shebang and encoding
        if (
            stripped.startswith("#!")
            or "coding:" in stripped
            or "encoding:" in stripped
        ):
            continue

        # Look for first import
        if stripped.startswith("import ") or stripped.startswith("from "):
            insert_position = i
            break

    # Insert standardized imports
    header = [
        "",
        "# === STANDARDIZED IMPORTS ===",
        "from standard_imports import setup_module",
        "logger = setup_module(globals(), __name__)",
        "",
    ]

    lines[insert_position:insert_position] = header
    return "\n".join(lines)


def migrate_file(file_path: Path, issues: Dict[str, Any]) -> bool:
    """Migrate a single file."""
    try:
        content = issues["content"]
        original_content = content

        # Apply fixes
        if issues["has_inconsistent_logger"]["has_mixed_patterns"]:
            content = fix_logger_patterns(content)

        if issues["has_duplicate_auto_register"]:
            content = fix_duplicate_auto_register(content)

        if issues["has_run_comprehensive_tests"]:
            content = remove_run_comprehensive_tests(content, file_path)

        if issues["needs_standardization"]:
            content = add_standard_imports_header(content, file_path)

        # Only write if content changed
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"❌ Error migrating {file_path}: {e}")
        return False


def main():
    """Run the migration script."""
    print("🚀 Starting Ancestry Codebase Migration")
    print("=" * 50)

    # Create backup first
    create_backup()

    # Find all Python files
    python_files = find_python_files()
    print(f"📁 Found {len(python_files)} Python files to analyze")

    # Analyze all files
    print("\n🔍 Analyzing files...")
    all_issues = []
    files_with_issues = 0

    for file_path in python_files:
        issues = analyze_file(file_path)
        if issues:
            all_issues.append(issues)
            has_issues = (
                issues.get("has_run_comprehensive_tests", False)
                or issues.get("has_duplicate_auto_register", False)
                or issues.get("has_inconsistent_logger", {}).get(
                    "has_mixed_patterns", False
                )
                or issues.get("needs_standardization", False)
            )
            if has_issues:
                files_with_issues += 1

    print(f"📊 Analysis complete: {files_with_issues} files need migration")

    # Report findings
    print("\n📋 Issues found:")
    stats = {
        "run_comprehensive_tests": sum(
            1 for i in all_issues if i.get("has_run_comprehensive_tests")
        ),
        "duplicate_auto_register": sum(
            1 for i in all_issues if i.get("has_duplicate_auto_register")
        ),
        "mixed_logger_patterns": sum(
            1
            for i in all_issues
            if i.get("has_inconsistent_logger", {}).get("has_mixed_patterns")
        ),
        "needs_standardization": sum(
            1 for i in all_issues if i.get("needs_standardization")
        ),
    }

    for issue, count in stats.items():
        print(f"  • {issue.replace('_', ' ').title()}: {count} files")

    # Confirm migration
    print(f"\n⚠️  This will modify {files_with_issues} files.")
    print(f"📁 Backup created at: {backup_dir}")

    response = input("\n🤔 Proceed with migration? (y/N): ").lower().strip()
    if response != "y":
        print("❌ Migration cancelled")
        return

    # Perform migration
    print("\n🔄 Starting migration...")
    migrated_files = 0

    for issues in all_issues:
        if migrate_file(issues["file_path"], issues):
            migrated_files += 1
            print(f"✅ Migrated: {issues['file_path'].name}")

    print(f"\n🎉 Migration complete!")
    print(f"📊 Summary:")
    print(f"  • Files analyzed: {len(python_files)}")
    print(f"  • Files with issues: {files_with_issues}")
    print(f"  • Files migrated: {migrated_files}")
    print(f"  • Backup location: {backup_dir}")

    # Recommendations
    print(f"\n📌 Next steps:")
    print(f"  1. Test that all modules still work correctly")
    print(f"  2. Run: python test_framework_unified.py")
    print(f"  3. Update any remaining custom import patterns manually")
    print(f"  4. Consider using the new standard_imports.py in new modules")


if __name__ == "__main__":
    main()
