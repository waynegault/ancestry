"""
Test Infrastructure TODO #20: Test Diagnostics Enhancement

Provides enhanced error reporting, stack traces with context, input/output capture,
and failure reproduction steps for test failures.

Usage example:
    from test_diagnostics import capture_io, format_test_failure

    @capture_io
    def my_test():
        print("Debug info")
        assert some_condition()
        return True

    result, stdout, stderr, exc = my_test()
    if exc:
        print(format_test_failure("my_test", my_test, exc, stdout, stderr))
"""

import sys
import traceback
import inspect
from typing import Callable, Any, Optional, Tuple
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr


def capture_io(test_func: Callable[[], bool]) -> Callable[[], Tuple[bool, str, str, Optional[Exception]]]:
    """
    Decorator that captures stdout/stderr during test execution.

    Args:
        test_func: Test function to wrap

    Returns:
        Wrapped function that returns (result, stdout, stderr, exception)
    """
    def wrapper() -> Tuple[bool, str, str, Optional[Exception]]:
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        exception = None
        result = False

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                result = test_func()
        except Exception as e:
            exception = e

        return (
            result,
            stdout_capture.getvalue(),
            stderr_capture.getvalue(),
            exception
        )

    return wrapper


def format_test_failure(
    test_name: str,
    test_func: Callable,
    exception: Exception,
    stdout: str = "",
    stderr: str = ""
) -> str:
    """
    Generate detailed failure report for a failed test.

    Args:
        test_name: Name of the failed test
        test_func: The test function that failed
        exception: Exception raised during test
        stdout: Captured stdout output
        stderr: Captured stderr output

    Returns:
        Formatted failure report with diagnostics
    """
    report_lines = [
        "=" * 80,
        "TEST FAILURE DIAGNOSTIC REPORT",
        "=" * 80,
        f"Test: {test_name}",
        ""
    ]

    # Source location
    try:
        source_file = inspect.getsourcefile(test_func)
        source_lines = inspect.getsourcelines(test_func)
        source_line = source_lines[1] if source_lines else None
    except (TypeError, OSError):
        source_file = None
        source_line = None

    if source_file and source_line:
        report_lines.append(f"Source: {source_file}:{source_line}")
        report_lines.append("")

    # Exception details
    report_lines.append(f"Exception: {type(exception).__name__}: {exception}")
    report_lines.append("")

    # Stack trace with context
    report_lines.append("Stack Trace:")
    tb_lines = traceback.format_exception(type(exception), exception, exception.__traceback__)
    report_lines.extend(["  " + line.rstrip() for line in tb_lines])
    report_lines.append("")

    # Captured output
    if stdout:
        report_lines.append("Captured stdout:")
        report_lines.append("-" * 40)
        report_lines.extend(["  " + line for line in stdout.splitlines()])
        report_lines.append("")

    if stderr:
        report_lines.append("Captured stderr:")
        report_lines.append("-" * 40)
        report_lines.extend(["  " + line for line in stderr.splitlines()])
        report_lines.append("")

    # Reproduction steps
    report_lines.append("Reproduction Steps:")
    module_name = test_func.__module__ if hasattr(test_func, '__module__') else 'unknown'
    report_lines.append(f"1. Run: python -m {module_name}")
    report_lines.append(f"2. Look for test: '{test_name}'")
    if source_file and source_line:
        report_lines.append(f"3. Check source at: {source_file}:{source_line}")
    report_lines.append("")

    report_lines.append("=" * 80)

    return "\n".join(report_lines)


# ============================================================================
# TESTS
# ============================================================================


def _test_capture_io_basic() -> bool:
    """Test that capture_io decorator works for basic output."""
    @capture_io
    def test_with_output():
        print("stdout message")
        sys.stderr.write("stderr message\n")
        return True

    result, stdout, stderr, exc = test_with_output()

    has_stdout = "stdout message" in stdout
    has_stderr = "stderr message" in stderr
    no_exception = exc is None
    passed = result is True

    return has_stdout and has_stderr and no_exception and passed


def _test_capture_io_exception() -> bool:
    """Test that exceptions are captured with full details."""
    @capture_io
    def failing_test():
        print("Before exception")
        raise ValueError("Test failure reason")

    result, stdout, stderr, exc = failing_test()

    result_false = result is False
    has_stdout = "Before exception" in stdout
    has_exception = exc is not None and isinstance(exc, ValueError)
    correct_message = str(exc) == "Test failure reason"

    return result_false and has_stdout and has_exception and correct_message


def _test_format_failure_report_basic() -> bool:
    """Test that failure reports are generated correctly."""
    def test_func():
        raise AssertionError("Expected value mismatch")

    try:
        test_func()
    except AssertionError as e:
        report = format_test_failure("Test name", test_func, e)

        has_title = "TEST FAILURE DIAGNOSTIC REPORT" in report
        has_exception = "AssertionError" in report
        has_message = "Expected value mismatch" in report
        has_reproduction = "Reproduction Steps:" in report

        return has_title and has_exception and has_message and has_reproduction

    return False


def _test_format_failure_with_output() -> bool:
    """Test that failure reports include captured output."""
    def test_func():
        pass  # Dummy function

    exc = RuntimeError("Something broke")
    stdout = "Debug line 1\nDebug line 2"
    stderr = "Warning line"

    report = format_test_failure("Test with output", test_func, exc, stdout, stderr)

    has_stdout_section = "Captured stdout:" in report
    has_stdout_content = "Debug line 1" in report and "Debug line 2" in report
    has_stderr_section = "Captured stderr:" in report
    has_stderr_content = "Warning line" in report

    return has_stdout_section and has_stdout_content and has_stderr_section and has_stderr_content


def _test_capture_io_no_exception() -> bool:
    """Test that capture_io works when test passes."""
    @capture_io
    def passing_test():
        print("Test passed")
        return True

    result, stdout, stderr, exc = passing_test()

    passed = result is True
    has_output = "Test passed" in stdout
    no_exception = exc is None

    return passed and has_output and no_exception


def _test_capture_io_stderr() -> bool:
    """Test that stderr is captured separately from stdout."""
    @capture_io
    def test_with_stderr():
        print("stdout line")
        sys.stderr.write("stderr line\n")
        return True

    result, stdout, stderr, exc = test_with_stderr()

    stdout_correct = "stdout line" in stdout and "stderr line" not in stdout
    stderr_correct = "stderr line" in stderr and "stdout line" not in stderr

    return stdout_correct and stderr_correct


def _test_format_failure_source_location() -> bool:
    """Test that source file and line number are included in report."""
    def test_func():
        pass

    exc = Exception("Test")
    report = format_test_failure("Test", test_func, exc)

    # Should contain "Source:" line with file path
    has_source = "Source:" in report
    has_reproduction = "test_diagnostics.py" in report or "test_diagnostics" in report

    return has_source or has_reproduction  # At least one should be present


def _test_capture_io_complex_scenario() -> bool:
    """Test capture_io with complex output and exception."""
    @capture_io
    def complex_test():
        print("Step 1: Initialize")
        value = 42
        print(f"Step 2: Value is {value}")
        sys.stderr.write("Warning: edge case detected\n")

        if value != 100:
            raise AssertionError(f"Expected 100, got {value}")

        return True

    result, stdout, stderr, exc = complex_test()

    has_stdout = "Step 1: Initialize" in stdout and "Step 2: Value is 42" in stdout
    has_stderr = "Warning: edge case detected" in stderr
    has_exception = exc is not None and isinstance(exc, AssertionError)
    has_message = str(exc) == "Expected 100, got 42"

    return has_stdout and has_stderr and has_exception and has_message


def _test_format_failure_stack_trace() -> bool:
    """Test that stack traces are included in failure reports."""
    def test_func():
        raise ValueError("Test error")

    try:
        test_func()
    except ValueError as e:
        report = format_test_failure("Stack trace test", test_func, e)

        has_stack_trace_header = "Stack Trace:" in report
        has_traceback = "Traceback" in report
        has_value_error = "ValueError" in report

        return has_stack_trace_header and has_traceback and has_value_error

    return False


def _test_format_failure_empty_output() -> bool:
    """Test that reports handle empty stdout/stderr gracefully."""
    def test_func():
        pass

    exc = Exception("Test")
    report = format_test_failure("Test", test_func, exc, "", "")

    # Should not have output sections if empty
    has_title = "TEST FAILURE DIAGNOSTIC REPORT" in report
    has_exception = "Exception" in report
    no_empty_stdout_section = "Captured stdout:" not in report
    no_empty_stderr_section = "Captured stderr:" not in report

    return has_title and has_exception and no_empty_stdout_section and no_empty_stderr_section


def module_tests() -> bool:
    """Test suite for test diagnostics functionality."""
    # Import here to avoid circular dependencies
    from test_framework import TestSuite

    suite = TestSuite("Test Diagnostics", "test_diagnostics.py")

    suite.start_suite()

    suite.run_test("IO capture - basic output", _test_capture_io_basic)
    suite.run_test("IO capture - exception handling", _test_capture_io_exception)
    suite.run_test("IO capture - passing tests", _test_capture_io_no_exception)
    suite.run_test("IO capture - stderr separation", _test_capture_io_stderr)
    suite.run_test("IO capture - complex scenario", _test_capture_io_complex_scenario)
    suite.run_test("Failure report - basic generation", _test_format_failure_report_basic)
    suite.run_test("Failure report - with output", _test_format_failure_with_output)
    suite.run_test("Failure report - source location", _test_format_failure_source_location)
    suite.run_test("Failure report - stack traces", _test_format_failure_stack_trace)
    suite.run_test("Failure report - empty output", _test_format_failure_empty_output)

    return suite.finish_suite()


if __name__ == "__main__":
    # Import here to avoid circular dependencies
    from test_framework import create_standard_test_runner

    run_comprehensive_tests = create_standard_test_runner(module_tests)
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
