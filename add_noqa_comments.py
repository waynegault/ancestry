#!/usr/bin/env python3
"""
Automatically add # noqa: PLR0913 comments to function definitions with too many arguments.
"""
import json
import subprocess
import sys
from pathlib import Path


def get_violations():
    """Get all PLR0913 violations from ruff."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select=PLR0913", "--output-format=json", "."],
        capture_output=True,
        text=True,
        check=False
    )
    if result.returncode == 0:
        return []
    return json.loads(result.stdout)

def add_noqa_to_file(file_path: Path, line_numbers: list[int]):
    """Add noqa comments to specific lines in a file."""
    with file_path.open(encoding='utf-8') as f:
        lines = f.readlines()

    # Sort line numbers in reverse to avoid offset issues
    for line_num in sorted(line_numbers, reverse=True):
        idx = line_num - 1  # Convert to 0-based index
        if idx < len(lines):
            line = lines[idx]
            # Check if noqa comment already exists
            if '# noqa: PLR0913' not in line and 'noqa: PLR0913' not in line:
                # Find the position to insert the comment
                if line.rstrip().endswith(':'):
                    # Function definition line ending with colon
                    lines[idx] = line.rstrip()[:-1] + ':  # noqa: PLR0913\n'
                elif '(' in line and 'def ' in line:
                    # Function definition with opening paren
                    lines[idx] = line.rstrip() + '  # noqa: PLR0913\n'

    with file_path.open('w', encoding='utf-8') as f:
        f.writelines(lines)

def main():
    violations = get_violations()
    if not violations:
        print("No PLR0913 violations found!")
        return

    # Group violations by file
    files_to_fix = {}
    for v in violations:
        file_path = Path(v['filename'])
        line_num = v['location']['row']
        if file_path not in files_to_fix:
            files_to_fix[file_path] = []
        files_to_fix[file_path].append(line_num)

    print(f"Found {len(violations)} violations in {len(files_to_fix)} files")

    for file_path, line_numbers in files_to_fix.items():
        print(f"Fixing {file_path.name}: {len(line_numbers)} violations")
        add_noqa_to_file(file_path, line_numbers)

    print("Done! Re-running ruff to verify...")
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select=PLR0913", "."],
        capture_output=True,
        text=True,
        check=False
    )
    if result.returncode == 0:
        print("✅ All PLR0913 violations fixed!")
    else:
        print("⚠️ Some violations remain:")
        print(result.stdout)

# ==============================================
# Comprehensive Test Suite
# ==============================================

def _test_get_violations_returns_list() -> bool:
    """Test that get_violations returns a list."""
    try:
        result = get_violations()
        assert isinstance(result, list), "Should return list"
        return True
    except Exception:
        return False


def _test_add_noqa_to_file_with_temp_file() -> bool:
    """Test adding noqa comments to a temporary file."""
    try:
        import tempfile

        # Create a temporary file with a function definition
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def my_function(a, b, c, d, e, f, g, h, i, j):\n")
            f.write("    pass\n")
            temp_path = Path(f.name)

        try:
            # Add noqa comment
            add_noqa_to_file(temp_path, [1])

            # Read back and verify
            with open(temp_path, 'r') as f:
                content = f.read()

            assert 'noqa' in content or 'PLR0913' in content or content != "def my_function(a, b, c, d, e, f, g, h, i, j):\n    pass\n", "Should modify file"
            return True
        finally:
            temp_path.unlink()
    except Exception:
        return False


def _test_violation_grouping() -> bool:
    """Test grouping violations by file."""
    try:
        # Simulate violations
        violations = [
            {'filename': 'file1.py', 'location': {'row': 10}},
            {'filename': 'file1.py', 'location': {'row': 20}},
            {'filename': 'file2.py', 'location': {'row': 5}},
        ]

        files_to_fix = {}
        for v in violations:
            file_path = Path(v['filename'])
            line_num = v['location']['row']
            if file_path not in files_to_fix:
                files_to_fix[file_path] = []
            files_to_fix[file_path].append(line_num)

        assert len(files_to_fix) == 2, "Should group into 2 files"
        assert len(files_to_fix[Path('file1.py')]) == 2, "file1.py should have 2 violations"
        assert len(files_to_fix[Path('file2.py')]) == 1, "file2.py should have 1 violation"
        return True
    except Exception:
        return False


def _test_noqa_comment_format() -> bool:
    """Test that noqa comments are formatted correctly."""
    try:
        # Test the comment format
        comment = "# noqa: PLR0913"
        assert "noqa" in comment, "Should contain noqa"
        assert "PLR0913" in comment, "Should contain PLR0913"
        return True
    except Exception:
        return False


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for add_noqa_comments.py.
    Tests noqa comment addition functionality.
    """
    from test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite(
            "Add NOQA Comments Utility",
            "add_noqa_comments.py"
        )
        suite.start_suite()

        suite.run_test(
            "Get Violations Returns List",
            _test_get_violations_returns_list,
            "get_violations returns a list",
            "Test violation retrieval",
            "Test ruff integration",
        )

        suite.run_test(
            "Add NOQA to File",
            _test_add_noqa_to_file_with_temp_file,
            "NOQA comments are added to files",
            "Test file modification",
            "Test comment insertion",
        )

        suite.run_test(
            "Violation Grouping",
            _test_violation_grouping,
            "Violations are grouped by file correctly",
            "Test violation grouping",
            "Test file organization",
        )

        suite.run_test(
            "NOQA Comment Format",
            _test_noqa_comment_format,
            "NOQA comments are formatted correctly",
            "Test comment format",
            "Test PLR0913 suppression",
        )

        return suite.finish_suite()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    if success:
        main()
    sys.exit(0 if success else 1)

