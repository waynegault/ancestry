#!/usr/bin/env python3
"""
Script to check for linter errors excluding length and complexity violations.
"""

import sys
from pathlib import Path

from code_quality_checker import CodeQualityChecker


def filter_violations(violations: list[str]) -> list[str]:
    """Filter out length and complexity violations."""
    filtered = []
    for violation in violations:
        # Skip length violations
        if "is too long" in violation:
            continue
        # Skip complexity violations
        if "is too complex" in violation:
            continue
        # Keep all other violations
        filtered.append(violation)
    return filtered

def main():
    """Main function to check files for non-length/complexity issues."""
    if len(sys.argv) < 2:
        print("Usage: python check_non_length_complexity_issues.py <file1> [file2] ...")
        sys.exit(1)

    checker = CodeQualityChecker()

    for file_path in sys.argv[1:]:
        path = Path(file_path)
        if not path.exists():
            print(f"File not found: {file_path}")
            continue

        print(f"\n=== {file_path} ===")
        result = checker.check_file(path)

        # Filter out length and complexity violations
        filtered_violations = filter_violations(result.violations)

        if not filtered_violations:
            print("✅ No linter errors (excluding length/complexity)")
        else:
            print(f"❌ {len(filtered_violations)} linter errors (excluding length/complexity):")
            for i, violation in enumerate(filtered_violations, 1):
                print(f"  {i}. {violation}")

if __name__ == "__main__":
    main()
