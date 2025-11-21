#!/usr/bin/env python3

"""
Test Runner Standardization Automation Script

Automates the conversion of test modules to use the standardized
create_standard_test_runner pattern from test_utilities.

This script:
1. Identifies modules with inline run_comprehensive_tests implementations
2. Extracts the test logic into a separate module_tests function
3. Replaces run_comprehensive_tests with standardized pattern
4. Creates backups before modification
5. Validates changes by attempting to import and run tests
"""

import re
import shutil
import sys
from pathlib import Path
from typing import Optional


class StandardizationError(Exception):
    """Raised when a module cannot be converted to the standard runner."""


def extract_test_function_body(content: str) -> Optional[tuple[str, str, int]]:
    """Extract run_comprehensive_tests body and return (body, docstring, line)."""
    # Find the function definition
    pattern = r'def run_comprehensive_tests\(\).*?:\s*(""".*?"""|\'\'\'.*?\'\'\')?'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        return None

    start_pos = match.end()

    # Extract docstring if present
    docstring_match = re.search(r'"""(.*?)"""', match.group(0), re.DOTALL)
    docstring = docstring_match.group(1).strip() if docstring_match else "Module tests"

    # Find the function body by tracking indentation
    lines = content[start_pos:].split('\n')
    body_lines: list[str] = []
    base_indent = None

    for line in lines:
        if not line.strip():  # Empty line
            if body_lines:  # Only add if we've started collecting
                body_lines.append(line)
            continue

        # Determine base indentation from first non-empty line
        if base_indent is None and line.strip():
            base_indent = len(line) - len(line.lstrip())

        # Check if we've exited the function
        current_indent = len(line) - len(line.lstrip())
        if base_indent is not None and current_indent < base_indent and line.strip():
            break

        body_lines.append(line)

    # Get start line number
    start_line = content[:match.start()].count('\n') + 1

    return '\n'.join(body_lines).rstrip(), docstring, start_line


def generate_module_test_name(filepath: str) -> str:
    """Generate appropriate test function name from module path."""
    path = Path(filepath)

    # Handle core/module.py -> core_module_tests
    if 'core/' in filepath:
        base = path.stem
        return f"core_{base}_module_tests"

    # Handle observability/module.py -> observability_module_tests
    if 'observability/' in filepath:
        base = path.stem
        return f"observability_{base}_module_tests"

    # Handle root module.py -> module_tests
    return f"{path.stem}_module_tests"


def standardize_test_runner(filepath: str, dry_run: bool = True) -> bool:
    """Standardize a single module so it uses create_standard_test_runner."""
    path = Path(filepath)

    if not path.exists():
        print(f"❌ File not found: {filepath}")
        return False

    # Read the file
    content = path.read_text(encoding='utf-8')

    # Check if already standardized
    if 'create_standard_test_runner' in content:
        print(f"✅ Already standardized: {filepath}")
        return True

    try:
        standardized = _prepare_standardized_content(content, filepath)
    except StandardizationError as exc:
        print(f"❌ {exc}")
        return False

    new_content, new_func_name, start_line, body_lines = standardized

    if dry_run:
        print(f"📝 Would standardize: {filepath}")
        print(f"   New function: {new_func_name}")
        print(f"   Lines: {start_line} - {start_line + body_lines}")
        return True

    _write_standardized_file(path, new_content, new_func_name, filepath)
    _validate_import(path, filepath)
    return True


def _prepare_standardized_content(content: str, filepath: str) -> tuple[str, str, int, int]:
    """Return the new file content plus metadata needed for logging."""
    if 'def run_comprehensive_tests' not in content:
        raise StandardizationError(f"No run_comprehensive_tests found: {filepath}")

    result = extract_test_function_body(content)
    if not result:
        raise StandardizationError(f"Failed to extract run_comprehensive_tests body: {filepath}")

    body, docstring, start_line = result
    new_func_name = generate_module_test_name(filepath)

    rct_pattern = r'def run_comprehensive_tests\(\).*?return suite\.finish_suite\(\)'
    rct_match = re.search(rct_pattern, content, re.DOTALL)
    if not rct_match:
        raise StandardizationError(f"Incomplete run_comprehensive_tests implementation: {filepath}")

    new_function = (
        f"def {new_func_name}() -> bool:\n"
        f"    \"\"\"{docstring or 'Module tests'}\"\"\"\n"
        f"{body}\n\n\n"
        "# Use centralized test runner utility from test_utilities\n"
        "from test_utilities import create_standard_test_runner\n"
        f"run_comprehensive_tests = create_standard_test_runner({new_func_name})"
    )

    updated_content = content[:rct_match.start()] + new_function + content[rct_match.end():]
    body_lines = body.count('\n')

    return updated_content, new_func_name, start_line, body_lines


def _write_standardized_file(path: Path, new_content: str, func_name: str, filepath: str) -> None:
    """Persist the updated module content, creating a backup first."""
    backup_path = path.with_suffix('.py.bak')
    shutil.copy(path, backup_path)
    print(f"💾 Backup created: {backup_path}")

    path.write_text(new_content, encoding='utf-8')
    print(f"✅ Standardized: {filepath}")
    print(f"   New function: {func_name}")


def _validate_import(path: Path, filepath: str) -> None:
    """Try importing the standardized module to catch syntax errors early."""
    try:
        parent = path.parent.resolve()
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))

        module_name = path.stem
        if 'core/' in filepath:
            module_name = f"core.{module_name}"
            ancestor = parent.parent.resolve()
            if str(ancestor) not in sys.path:
                sys.path.insert(0, str(ancestor))
        elif 'observability/' in filepath:
            module_name = f"observability.{module_name}"

        __import__(module_name)
        print("✅ Import validation successful")
    except Exception as exc:  # pragma: no cover - best-effort diagnostics
        print(f"⚠️  Import validation failed: {exc}")
        print("   (This may be expected if module has external dependencies)")


def main() -> int:
    """CLI entry point for bulk standardization runs."""
    import argparse

    parser = argparse.ArgumentParser(description='Standardize test runner patterns')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--all', action='store_true',
                       help='Process all non-standardized modules')
    parser.add_argument('files', nargs='*',
                       help='Specific files to standardize')

    args = parser.parse_args()

    # List of modules needing standardization
    modules_to_standardize = [
        './api_constants.py',
        './common_params.py',
        './connection_resilience.py',
        './core/registry_utils.py',
        './core/progress_indicators.py',
        './core/metrics_collector.py',
        './core/metrics_integration.py',
        './core/error_handling.py',
        './dna_utils.py',
        './grafana_checker.py',
        './observability/metrics_exporter.py',
        './observability/metrics_registry.py',
        './rate_limiter.py',
        # Note: core/browser_manager.py and core/session_manager.py are very large
        # and may need manual handling
    ]

    if args.all:
        files_to_process = modules_to_standardize
    elif args.files:
        files_to_process = args.files
    else:
        print("No files specified. Defaulting to '--dry-run --all' for reporting consistency.")
        files_to_process = modules_to_standardize
        args.dry_run = True

    banner_prefix = 'DRY RUN - ' if args.dry_run else ''
    print(f"{banner_prefix}Standardizing {len(files_to_process)} modules...")
    print("=" * 70)

    success_count = 0
    for filepath in files_to_process:
        if standardize_test_runner(filepath, dry_run=args.dry_run):
            success_count += 1
        print()

    print("=" * 70)
    print(f"Results: {success_count}/{len(files_to_process)} modules {'would be ' if args.dry_run else ''}standardized")

    return 0 if success_count == len(files_to_process) else 1


if __name__ == '__main__':
    sys.exit(main())
