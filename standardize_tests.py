#!/usr/bin/env python3
"""
Script to standardize test formatting by removing numbered test comments
from all Python files in the project.
"""

import os
import re
import glob


def standardize_test_comments(directory):
    """Remove numbered test comments from all Python files."""
    pattern = re.compile(r"(\s*#\s*)Test \d+:\s*(.+)")

    # Get all Python files
    python_files = glob.glob(os.path.join(directory, "*.py"))

    modified_files = []

    for file_path in python_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Find and replace numbered test comments
            modified_content = pattern.sub(r"\1\2", content)

            if content != modified_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(modified_content)
                modified_files.append(os.path.basename(file_path))

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return modified_files


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    modified = standardize_test_comments(current_dir)

    if modified:
        print(f"Standardized test comments in {len(modified)} files:")
        for file in sorted(modified):
            print(f"  - {file}")
    else:
        print("No files needed standardization.")
