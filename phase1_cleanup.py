#!/usr/bin/env python3
"""
Phase 1 Cleanup Script: Critical Fixes

This script implements Phase 1 of the cleanup plan:
1. Standardize import patterns across ALL modules
2. Remove duplicate auto_register_module calls
3. Fix inconsistent logger usage
4. Remove legacy fallback patterns

Applies changes to all 42+ modules systematically.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple


def get_all_python_files() -> List[Path]:
    """Get all Python files in the project (excluding test runner and this script)."""
    project_root = Path(__file__).parent
    python_files = []

    # Define the specific project files we want to process
    project_files = [
        "action10.py",
        "action11.py",
        "action6_gather.py",
        "action7_inbox.py",
        "action8_messaging.py",
        "action9_process_productive.py",
        "ai_interface.py",
        "ai_prompt_utils.py",
        "api_cache.py",
        "api_search_utils.py",
        "api_utils.py",
        "cache.py",
        "cache_manager.py",
        "chromedriver.py",
        "cleanup_obsolete.py",
        "cleanup_script.py",
        "credentials.py",
        "database.py",
        "debug_config.py",
        "error_handling.py",
        "gedcom_cache.py",
        "gedcom_search_utils.py",
        "gedcom_utils.py",
        "integration_test.py",
        "logging_config.py",
        "main.py",
        "migration_script.py",
        "ms_graph_utils.py",
        "my_selectors.py",
        "performance_monitor.py",
        "person_search.py",
        "relationship_utils.py",
        "security_manager.py",
        "selenium_utils.py",
        "STANDARD_IMPORT_TEMPLATE.py",
        "standard_imports.py",
        "test_framework.py",
        "test_framework_unified.py",
        "utils.py",
    ]

    # Add files from subdirectories
    subdirs = ["config", "core"]
    for subdir in subdirs:
        subdir_path = project_root / subdir
        if subdir_path.exists():
            for file_path in subdir_path.rglob("*.py"):
                if file_path.name != "__init__.py":
                    python_files.append(file_path)

    # Add root level files
    for file_name in project_files:
        file_path = project_root / file_name
        if file_path.exists():
            python_files.append(file_path)

    return sorted(python_files)


def standardize_imports(content: str, file_path: Path) -> str:
    """Apply standard import pattern to file content."""
    # Skip critical system files
    if file_path.name in ["core_imports.py", "logging_config.py"]:
        return content

    lines = content.split("\n")
    new_lines = []

    # Find existing import sections and track what we need to preserve
    shebang_and_docstring = []
    imports_section = []
    rest_of_file = []

    current_section = "header"
    i = 0

    # Parse the file into sections
    while i < len(lines):
        line = lines[i]

        if current_section == "header":
            if (
                line.startswith("#!")
                or line.strip().startswith('"""')
                or line.strip().startswith("'''")
            ):
                shebang_and_docstring.append(line)
                # Handle multi-line docstrings
                if line.strip().startswith('"""') or line.strip().startswith("'''"):
                    quote = '"""' if '"""' in line else "'''"
                    if line.count(quote) == 1:  # Opening quote only
                        i += 1
                        while i < len(lines) and quote not in lines[i]:
                            shebang_and_docstring.append(lines[i])
                            i += 1
                        if i < len(lines):
                            shebang_and_docstring.append(lines[i])  # Closing quote
            elif line.strip() == "" or line.startswith("#"):
                shebang_and_docstring.append(line)
            else:
                current_section = "imports"
                continue  # Don't increment i, process this line as imports

        elif current_section == "imports":
            # Detect import-related lines
            if (
                line.startswith("from ")
                or line.startswith("import ")
                or "auto_register_module" in line
                or "get_logger" in line
                or line.startswith("try:")
                or line.startswith("except ImportError:")
                or line.strip() in ["pass", ""]
                or line.strip().startswith("#")
            ):
                imports_section.append(line)
            else:
                current_section = "rest"
                continue  # Don't increment i, process this line as rest

        elif current_section == "rest":
            rest_of_file.append(line)

        i += 1

    # Build the new file
    result = []

    # Add header section
    result.extend(shebang_and_docstring)

    # Add standard imports (only if we found imports to replace)
    if imports_section:
        result.append("")
        result.append("# === UNIFIED IMPORTS (REQUIRED) ===")
        result.append("from core_imports import (")
        result.append("    auto_register_module,")
        result.append("    get_logger,")
        result.append("    standardize_module_imports,")
        result.append("    safe_execute,")
        result.append("    register_function,")
        result.append("    get_function,")
        result.append("    is_function_available,")
        result.append(")")
        result.append("")
        result.append("# Auto-register immediately")
        result.append("auto_register_module(globals(), __name__)")
        result.append("")
        result.append("# Get logger")
        result.append("logger = get_logger(__name__)")
        result.append("")

    # Add rest of file
    result.extend(rest_of_file)

    return "\n".join(result)


def remove_duplicate_calls(content: str) -> str:
    """Remove duplicate auto_register_module calls."""
    # Count occurrences
    pattern = r"auto_register_module\(globals\(\), __name__\)"
    matches = list(re.finditer(pattern, content))

    if len(matches) <= 1:
        return content  # No duplicates

    # Keep first occurrence, comment out others
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if pattern in line and i > 0:  # Keep first, remove others
            # Check if we've already seen one
            previous_content = "\n".join(lines[:i])
            if pattern in previous_content:
                lines[i] = f"# REMOVED DUPLICATE: {line.strip()}"

    return "\n".join(lines)


def fix_logger_patterns(content: str) -> str:
    """Fix inconsistent logger usage patterns."""
    # Replace various logger patterns with standard one
    replacements = [
        (r"from logging_config import logger", "# Fixed: using unified logger"),
        (r"import logging\s*\n", "# Fixed: using unified logging\n"),
        (r"logging\.getLogger\([^)]*\)", "get_logger(__name__)"),
        (
            r"logger = logging\.getLogger\([^)]*\)",
            "# Fixed: using unified logger setup",
        ),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    return content


def apply_phase1_fixes(file_path: Path) -> Tuple[bool, str]:
    """Apply all Phase 1 fixes to a single file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        # Skip if file is too small or doesn't have imports
        if len(original_content) < 100 or "import" not in original_content:
            return False, f"Skipped {file_path.name} - too small or no imports"

        # Apply fixes
        content = original_content
        content = standardize_imports(content, file_path)
        content = remove_duplicate_calls(content)
        content = fix_logger_patterns(content)

        # Only write if content changed significantly
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True, f"Fixed {file_path.name}"
        else:
            return False, f"No changes needed for {file_path.name}"

    except Exception as e:
        return False, f"Error processing {file_path.name}: {e}"


def main():
    """Run Phase 1 cleanup across all modules."""
    print("🔧 Phase 1 Cleanup: Standardizing Import Patterns")
    print("=" * 60)

    files = get_all_python_files()
    print(f"Found {len(files)} Python files to process...")

    results = []
    for file_path in files:
        success, message = apply_phase1_fixes(file_path)
        results.append((success, message))
        print(f"{'✅' if success else '⚪'} {message}")

    # Summary
    successful = sum(1 for success, _ in results if success)
    total = len(results)

    print("\n" + "=" * 60)
    print(f"📊 Phase 1 Results:")
    print(f"   • Files processed: {total}")
    print(f"   • Files modified: {successful}")
    print(f"   • Files unchanged: {total - successful}")
    print(f"   • Success rate: {successful/total*100:.1f}%")
    print("=" * 60)

    return successful > 0


if __name__ == "__main__":
    main()
