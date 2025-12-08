# Allows running `python -m config` without error.
# You can add test or main logic here if needed.

import contextlib
import io
import sys
from collections.abc import Iterable
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def build_package_info() -> list[str]:
    """Return the informational banner printed when running python -m config."""

    return [
        "Configuration Package - Enhanced Config Management",
        "Version: 2.0.0",
        "Available modules: config_manager, config_schema",
        "Note: This is a package init file. Import individual modules as needed.",
        "",
        "Example usage:",
        "  from config.config_manager import ConfigManager",
    ]


def print_package_info(lines: Iterable[str]) -> None:
    """Print each line from the provided iterable."""

    for line in lines:
        print(line)


def main() -> None:
    """Primary entrypoint for `python -m config`."""

    try:
        print_package_info(build_package_info())
    except ImportError as e:
        print(f"Error importing config modules: {e}")
        print("Note: Run individual modules directly for testing.")


from testing.test_framework import TestSuite
from testing.test_utilities import create_standard_test_runner


def _test_build_package_info_contents() -> bool:
    lines = build_package_info()
    assert len(lines) >= 5, "Package info should contain multiple lines"
    assert lines[0].startswith("Configuration Package"), "First line should describe the package"
    assert "config_manager" in lines[2], "Available modules line should mention config_manager"
    assert lines[-1].strip().startswith("from config.config_manager"), "Example usage should be present"
    return True


def _test_print_package_info() -> bool:
    lines = ["line1", "line2"]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print_package_info(lines)
    captured = buffer.getvalue().splitlines()
    assert captured == lines, "print_package_info should emit each provided line"
    return True


def module_tests() -> bool:
    suite = TestSuite("config.__main__", "config/__main__.py")

    suite.run_test(
        "Package info contents",
        _test_build_package_info_contents,
        "Ensures build_package_info returns the expected messaging for python -m config.",
    )

    suite.run_test(
        "Print helper",
        _test_print_package_info,
        "Ensures print_package_info emits every provided line.",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    if "--run-tests" in sys.argv:
        success = module_tests()
        sys.exit(0 if success else 1)
    main()
