#!/usr/bin/env python3
"""Quick test coverage check for the Ancestry project."""

import os
from pathlib import Path


def main():
    test_count = 0
    total_count = 0
    modules_with_tests = []
    modules_without_tests = []

    for python_file in Path(".").glob("*.py"):
        if python_file.name in [
            "run_all_tests.py",
            "__init__.py",
            "main.py",
            "check_test_coverage.py",
        ]:
            continue
        total_count += 1

        try:
            with open(python_file, "r", encoding="utf-8") as f:
                content = f.read()
                has_main_block = 'if __name__ == "__main__"' in content
                has_test_content = any(
                    indicator in content.lower()
                    for indicator in [
                        "test_",
                        "def test",
                        "class test",
                        "testsuite",
                        "testing",
                        "# test",
                    ]
                )

                if has_main_block or has_test_content:
                    test_count += 1
                    modules_with_tests.append(python_file.name)
                else:
                    modules_without_tests.append(python_file.name)
        except Exception as e:
            print(f"Error reading {python_file}: {e}")

    print(f"🧪 Test Coverage Analysis")
    print(f"========================")
    print(f"Total modules with test frameworks: {test_count}")
    print(f"Total substantial modules: {total_count}")
    print(f"Test coverage: {test_count/total_count*100:.1f}%")
    print()

    if modules_without_tests:
        print(f"❌ Modules without tests ({len(modules_without_tests)}):")
        for module in sorted(modules_without_tests):
            print(f"   - {module}")
    else:
        print("✅ All modules have test frameworks!")


if __name__ == "__main__":
    main()
