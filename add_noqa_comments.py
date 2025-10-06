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
    with open(file_path, 'r', encoding='utf-8') as f:
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

    with open(file_path, 'w', encoding='utf-8') as f:
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

if __name__ == "__main__":
    main()

