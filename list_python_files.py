import os
import sys
from pathlib import Path


def find_all_python_files(directory="."):
    """Find all Python files in the given directory and subdirectories."""
    python_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    return python_files


if __name__ == "__main__":
    # Get all Python files
    python_files = find_all_python_files()
    print(f"Found {len(python_files)} Python files")

    # Print the paths for easy copy-pasting into VS Code for error checking
    for file in python_files:
        print(f'"{os.path.abspath(file)}",')
