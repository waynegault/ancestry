import os
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

with Path("debug_result.txt").open("w", encoding="utf-8") as f:
    try:
        f.write("Starting debug run...\n")
        from messaging.message_types import module_tests

        f.write("Successfully imported module_tests\n")

        from testing.test_framework import TestSuite

        f.write("Successfully imported TestSuite\n")

        f.write("Running module_tests()...\n")
        # Capture stdout/stderr from module_tests if possible, or just the result
        # Since module_tests prints to stdout, we might miss that unless we redirect stdout

        # Redirect stdout to the file
        sys.stdout = f
        sys.stderr = f

        result = module_tests()
        f.write(f"\nResult: {result}\n")

    except Exception:
        f.write("\nException occurred:\n")
        traceback.print_exc(file=f)
