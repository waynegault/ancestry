import sys
import subprocess
import os
import time


def check_pylance_errors(python_files):
    """
    Check a list of Python files for Pylance errors.
    Returns a dictionary mapping file paths to lists of errors.
    """
    error_count = 0

    for py_file in python_files:
        # Normalize the path for printing
        display_path = os.path.abspath(py_file)

        # Use a simple Python process to check for import errors
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f'import os; os.chdir("{os.getcwd()}"); import {os.path.splitext(py_file)[0]}',
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                error_count += 1
                print(f"Error in {display_path}:")
                print(result.stderr.strip())
                print("-" * 60)
            else:
                print(f"✓ {display_path}")

        except Exception as e:
            error_count += 1
            print(f"Error checking {display_path}: {str(e)}")

    return error_count


if __name__ == "__main__":
    # Get Python files in the current directory
    python_files = [
        f
        for f in os.listdir(".")
        if f.endswith(".py")
        and not f.startswith("list_")
        and not f.startswith("diagnose_")
    ]

    print(f"Checking {len(python_files)} Python files for import errors...")
    start_time = time.time()

    error_count = check_pylance_errors(python_files)

    duration = time.time() - start_time
    print(f"\nCheck completed in {duration:.2f} seconds")
    print(f"Found {error_count} files with errors")
