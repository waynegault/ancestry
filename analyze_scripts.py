#!/usr/bin/env python3
"""Analyze all Python scripts to identify which can be removed."""

import os
import re
from collections import defaultdict
from pathlib import Path

# Scripts that are definitely needed (core production code)
KEEP_ESSENTIAL = {
    'main.py',  # Entry point
    'utils.py',  # Core utilities
    'database.py',  # Database layer
    'action6_gather.py',  # Production action
    'action7_inbox.py',  # Production action
    'action8_messaging.py',  # Production action
    'action9_process_productive.py',  # Production action
    'action10.py',  # Production action
    'action11.py',  # Production action
    'run_all_tests.py',  # Test runner
    'test_framework.py',  # Test infrastructure
}

# Development/diagnostic tools that can be removed
REMOVE_CANDIDATES = {
    'performance_validation.py',  # Empty file (0 lines)
    'fix_pylance_issues.py',  # Development tool (51 lines)
    'refactor_test_functions.py',  # Development tool (132 lines)
    'automate_too_many_args.py',  # Development tool (218 lines)
    'test_phase2_improvements.py',  # Development tool (218 lines)
    'add_noqa_comments.py',  # Development tool (213 lines)
    'apply_automated_refactoring.py',  # Development tool (341 lines)
    'diagnose_chrome.py',  # Diagnostic tool (499 lines)
    'validate_rate_limiting.py',  # Validation tool (330 lines)
    'test_rate_limiting.py',  # Test harness (447 lines)
}

def get_imports_from_file(filepath):
    """Extract all imports from a Python file."""
    imports = set()
    try:
        with open(filepath, encoding='utf-8') as f:
            content = f.read()
            # Find all "from X import" and "import X" statements
            from_imports = re.findall(r'from\s+(\w+)', content)
            direct_imports = re.findall(r'import\s+(\w+)', content)
            imports.update(from_imports)
            imports.update(direct_imports)
    except Exception:
        pass
    return imports

def analyze_scripts():
    """Analyze all scripts and their dependencies."""
    scripts = sorted([f for f in os.listdir('.') if f.endswith('.py')])

    # Map module names to files
    module_to_file = {}
    for script in scripts:
        module_name = script.replace('.py', '')
        module_to_file[module_name] = script

    # Find which scripts import which modules
    imports_by_script = {}
    for script in scripts:
        imports_by_script[script] = get_imports_from_file(script)

    # Find which scripts are imported by others
    imported_scripts = defaultdict(set)
    for script, imports in imports_by_script.items():
        for imp in imports:
            if imp in module_to_file:
                imported_scripts[module_to_file[imp]].add(script)

    # Categorize scripts
    print("=" * 80)
    print("SCRIPT ANALYSIS REPORT")
    print("=" * 80)
    print()

    print("KEEP - ESSENTIAL PRODUCTION CODE:")
    print("-" * 80)
    for script in sorted(KEEP_ESSENTIAL):
        if script in scripts:
            lines = len(open(script).readlines())
            print(f"  ✅ {script:40} ({lines:5} lines)")
    print()

    print("REMOVE - DEVELOPMENT/DIAGNOSTIC TOOLS:")
    print("-" * 80)
    for script in sorted(REMOVE_CANDIDATES):
        if script in scripts:
            lines = len(open(script).readlines())
            imported_by = imported_scripts.get(script, set())
            if imported_by:
                print(f"  ⚠️  {script:40} ({lines:5} lines) - USED BY: {', '.join(sorted(imported_by))}")
            else:
                print(f"  ❌ {script:40} ({lines:5} lines) - CAN REMOVE")
    print()

    print("REVIEW - UTILITY/SUPPORT MODULES:")
    print("-" * 80)
    other_scripts = set(scripts) - KEEP_ESSENTIAL - REMOVE_CANDIDATES
    for script in sorted(other_scripts):
        lines = len(open(script).readlines())
        imported_by = imported_scripts.get(script, set())
        if imported_by:
            print(f"  ✅ {script:40} ({lines:5} lines) - USED BY: {len(imported_by)} scripts")
        else:
            print(f"  ⚠️  {script:40} ({lines:5} lines) - NOT IMPORTED")
    print()

if __name__ == "__main__":
    analyze_scripts()

