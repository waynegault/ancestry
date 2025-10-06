#!/usr/bin/env python3
"""
Automated test function refactoring script.
Extracts nested test functions to module level to reduce complexity.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def find_nested_test_functions(lines: List[str]) -> List[Tuple[int, str, str]]:
    """Find all nested test functions (4-space indented def test_* or def _test_*)"""
    pattern = re.compile(r'^    def (test_\w+|_test_\w+)\(')
    matches = []
    for i, line in enumerate(lines, start=1):
        match = pattern.match(line)
        if match:
            func_name = match.group(1)
            matches.append((i, func_name, line))
    return matches


def extract_function_body(lines: List[str], start_line_idx: int) -> List[str]:
    """Extract the complete function body starting from start_line_idx"""
    function_lines = []
    base_indent = 4  # Nested functions have 4-space indent

    # Start from the function definition line
    i = start_line_idx
    while i < len(lines):
        line = lines[i]

        # Check if we've reached the next function or end of nested functions
        if i > start_line_idx:
            # If line starts with 4 spaces followed by 'def ', it's the next function
            if re.match(r'^    def \w+', line):
                break
            # If line starts with non-whitespace or less than 4 spaces (except blank lines), we're done
            if line.strip() and not line.startswith('    '):
                break

        function_lines.append(line)
        i += 1

    return function_lines


def create_module_level_function(func_name: str, func_body_lines: List[str]) -> Tuple[str, List[str]]:
    """Convert nested function to module-level function with _ prefix"""
    module_func_name = f"_{func_name}" if not func_name.startswith('_') else func_name

    # Remove 4-space indent from all lines
    module_lines = []
    for line in func_body_lines:
        if line.strip():  # Non-empty line
            if line.startswith('    '):
                module_lines.append(line[4:])  # Remove 4 spaces
            else:
                module_lines.append(line)
        else:
            module_lines.append(line)  # Keep blank lines

    # Update function name to have _ prefix
    if module_lines:
        module_lines[0] = module_lines[0].replace(f'def {func_name}(', f'def {module_func_name}(')

    return module_func_name, module_lines


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python refactor_test_functions.py <file_path>")
        print("\nThis script will:")
        print("1. Find all nested test functions in the file")
        print("2. Extract them to module level with _ prefix")
        print("3. Show you the changes (does not modify the file)")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Read the file
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()

    # Find nested test functions
    nested_functions = find_nested_test_functions(lines)

    if not nested_functions:
        print(f"No nested test functions found in {file_path}")
        sys.exit(0)

    print(f"\n{'='*70}")
    print(f"Found {len(nested_functions)} nested test functions in {file_path}")
    print(f"{'='*70}\n")

    # Extract and display each function
    module_level_functions = []
    for line_num, func_name, _ in nested_functions:
        print(f"  Line {line_num}: {func_name}")

        # Extract function body
        func_body = extract_function_body(lines, line_num - 1)

        # Convert to module-level function
        module_func_name, module_lines = create_module_level_function(func_name, func_body)
        module_level_functions.append((func_name, module_func_name, module_lines))

    print(f"\n{'='*70}")
    print("Module-level function assignments to add:")
    print(f"{'='*70}\n")

    for func_name, module_func_name, _ in module_level_functions:
        print(f"    {func_name} = {module_func_name}")

    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}")
    print(f"Total functions to extract: {len(module_level_functions)}")
    print(f"File: {file_path}")
    print(f"Total lines: {len(lines)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

