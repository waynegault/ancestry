#!/usr/bin/env python3
"""
Script to fix missing type hints in test functions.
"""

import re
import sys
from pathlib import Path


def fix_test_function_type_hints(file_path: Path) -> int:
    """Fix missing type hints in test functions."""
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return 0

    with open(file_path, encoding='utf-8') as f:
        content = f.read()

    # Pattern to match test functions without type hints
    pattern = r'(\s+def test_[^(]+\([^)]*\)):'

    def add_type_hint(match):
        func_def = match.group(1)
        if ' -> ' not in func_def:
            return func_def + ' -> None:'
        return match.group(0)

    # Replace all test functions
    new_content = re.sub(pattern, add_type_hint, content)

    # Count changes
    changes = content.count('def test_') - new_content.count('def test_')
    changes = len(re.findall(r'def test_[^(]+\([^)]*\) -> None:', new_content)) - len(re.findall(r'def test_[^(]+\([^)]*\) -> None:', content))

    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed type hints in {file_path}")
        return changes
    print(f"No changes needed in {file_path}")
    return 0

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python fix_type_hints.py <file1> [file2] ...")
        sys.exit(1)

    total_changes = 0
    for file_path in sys.argv[1:]:
        path = Path(file_path)
        changes = fix_test_function_type_hints(path)
        total_changes += changes

    print(f"Total changes made: {total_changes}")

if __name__ == "__main__":
    main()
