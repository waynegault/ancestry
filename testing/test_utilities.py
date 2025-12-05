#!/usr/bin/env python3

"""
Centralized Testing Utilities & DRY Implementation Engine

Comprehensive testing utility platform providing centralized test helper functions,
advanced validation factories, and reusable delegation patterns that implement
systematic DRY principles across the genealogical automation codebase with
sophisticated utility frameworks for consistent testing and code organization.

Test Utility Framework:
• Centralized test helper functions eliminating duplication across 30+ modules
• Advanced test function factories with parameterized test generation capabilities
• Comprehensive test data generation utilities with genealogical context awareness
• Standardized test runner patterns with consistent execution and reporting
• Reusable assertion helpers with detailed validation and error reporting
• Common test fixtures and setup utilities for consistent test environments
• Temporary file and directory helpers with atomic writes and automatic cleanup

Validation Infrastructure:
• Sophisticated validation factory functions with configurable validation criteria
• Advanced range validation with inclusive/exclusive boundary support
• Comprehensive type validation with optional None handling and complex type support
• Intelligent string validation with length, format, and content validation
• File extension validation with flexible format support and path handling
• Composite validation with multiple criteria combination and logical operations

DRY Implementation:
• Systematic elimination of code duplication through centralized utility functions
• Advanced delegation patterns with property and method delegation factories
• Generic factory patterns enabling rapid consolidation of similar functions
• Reusable base classes for common inheritance patterns and shared functionality
• Comprehensive utility frameworks supporting future consolidation opportunities
• Pattern establishment for consistent code organization and maintenance

Foundation Services:
Provides the essential utility infrastructure that enables systematic DRY
implementation, consistent testing practices, and maintainable code organization
for professional genealogical automation development and quality assurance.
"""

import contextlib
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import MagicMock

from sqlalchemy.orm import Session


class EmptyTestService:
    """
    Base class for test service classes that need no initialization.

    Consolidates the pattern of empty test service classes that just contain 'pass'.
    This implements DRY principles by providing a single base class instead of
    multiple identical empty class definitions.

    Usage:
        class ServiceA(EmptyTestService):
            pass  # Can be omitted entirely

        # Or simply:
        class ServiceA(EmptyTestService): ...
    """

    pass


# ==============================================================================
# Temporary File and Directory Helpers (Task 5)
# ==============================================================================


@contextlib.contextmanager
def atomic_write_file(target_path: Path, mode: str = "w", encoding: str = "utf-8") -> Iterator[Any]:
    """Context manager for atomic file writes using temp file + rename."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    with temp_path.open(mode, encoding=encoding if 'b' not in mode else None) as f:
        yield f
    temp_path.replace(target_path)


@contextlib.contextmanager
def temp_directory(prefix: str = "test-", cleanup: bool = True) -> Iterator[Path]:
    """Context manager for temporary directory creation with optional cleanup."""
    if cleanup:
        with tempfile.TemporaryDirectory(prefix=prefix) as temp_dir:
            yield Path(temp_dir)
    else:
        # Create but don't auto-cleanup
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        try:
            yield temp_dir
        finally:
            pass  # Caller responsible for cleanup


@contextlib.contextmanager
def temp_file(
    suffix: str = "", prefix: str = "test-", mode: str = "w+", encoding: str = "utf-8", delete: bool = True
) -> Iterator[Path]:
    """Context manager for temporary file creation with optional cleanup."""
    with tempfile.NamedTemporaryFile(
        mode=mode,
        suffix=suffix,
        prefix=prefix,
        delete=False,
        encoding=encoding if 'b' not in mode else None,
    ) as temp_file_handle:
        temp_path = Path(temp_file_handle.name)

    try:
        yield temp_path
    finally:
        if delete and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def mock_func() -> str:
    """
    Mock function that returns a predictable result.

    Used for testing function registration, retrieval, and execution systems.
    Consolidated from multiple modules where this exact function was duplicated.

    Returns:
        str: "test_result"
    """
    return "test_result"


def mock_func_with_param(x: int) -> int:
    """
    Mock function that takes a parameter and returns a computed result.

    Used for testing function registration with parameters.
    Consolidated from core_imports.py and other modules.

    Args:
        x: Integer input

    Returns:
        int: x * 2
    """
    return x * 2


def sample_function() -> str:
    """
    Sample function for testing function registration systems.

    Used in test suites to verify that functions can be registered and retrieved.
    Consolidated from standard_imports.py and other modules.

    Returns:
        str: "sample_result"
    """
    return "sample_result"


def temp_function() -> str:
    """
    Temporary function for testing cleanup and registration systems.

    Used to test function registration and cleanup mechanisms.
    Consolidated from standard_imports.py and other modules.

    Returns:
        str: "temp"
    """
    return "temp"


def safe_func() -> str:
    """
    Safe function for testing error handling and safe execution.

    Used to test safe execution patterns and error handling decorators.
    Consolidated from core_imports.py and other modules.

    Returns:
        str: "safe_result"
    """
    return "safe_result"


def decorated_safe_func() -> str:
    """
    Function designed to test decorator application and safe execution.

    Used to verify that decorators can be applied and function correctly.
    Consolidated from core_imports.py and other modules.

    Returns:
        str: "decorated_safe_result"
    """
    return "decorated_safe_result"


def create_test_function(return_value: Any = "test") -> Callable[[], Any]:
    """
    Factory function to create test functions with specific return values.

    This replaces the pattern of defining identical lambda functions or
    small test functions inline within test code.

    Args:
        return_value: The value the created function should return

    Returns:
        Callable: A function that returns the specified value
    """

    def generated_function() -> Any:
        return return_value

    return generated_function


def create_parameterized_test_function(operation: str = "multiply", factor: int = 2) -> Callable[[int], int]:
    """
    Factory function to create parameterized test functions.

    This replaces the pattern of defining similar mathematical test functions
    across different modules.

    Args:
        operation: The operation to perform ("multiply", "add", "subtract")
        factor: The factor to use in the operation

    Returns:
        Callable: A function that performs the specified operation
    """

    def generated_function(x: int) -> int:
        if operation == "multiply":
            return x * factor
        if operation == "add":
            return x + factor
        if operation == "subtract":
            return x - factor
        return x

    return generated_function


# Test function registry for common test patterns
TEST_FUNCTIONS = {
    "mock_func": mock_func,
    "mock_func_with_param": mock_func_with_param,
    "sample_function": sample_function,
    "temp_function": temp_function,
    "safe_func": safe_func,
    "decorated_safe_func": decorated_safe_func,
}


def get_test_function(name: str) -> Callable[..., Any]:
    """
    Get a test function by name from the registry.

    Args:
        name: Name of the test function

    Returns:
        Callable: The requested test function

    Raises:
        KeyError: If the test function is not found
    """
    if name not in TEST_FUNCTIONS:
        raise KeyError(f"Test function '{name}' not found. Available: {list(TEST_FUNCTIONS.keys())}")
    return TEST_FUNCTIONS[name]


def create_property_delegator(target_attr: str, property_name: str):
    """
    Create a property that delegates to an attribute of another object.

    Consolidates the common pattern of properties that just return attributes
    from composed objects. This implements DRY principles by eliminating
    repetitive delegation property definitions.

    Args:
        target_attr: The attribute name of the target object (e.g., 'api_manager')
        property_name: The property name on the target object (e.g., 'my_profile_id')

    Returns:
        property: A property descriptor that delegates to the target

    Example:
        class SessionManager:
            def __init__(self):
                self.api_manager = APIManager()

            # Instead of:
            # @property
            # def my_profile_id(self):
            #     return self.api_manager.my_profile_id

            # Use:
            my_profile_id = create_property_delegator('api_manager', 'my_profile_id')
    """

    def getter(self: Any) -> Any:
        target_obj = getattr(self, target_attr, None)
        if target_obj is None:
            return None
        return getattr(target_obj, property_name, None)

    return property(getter)


def create_method_delegator(target_attr: str, method_name: str):
    """
    Create a method that delegates to a method of another object.

    Consolidates the common pattern of methods that just call methods
    on composed objects. This implements DRY principles by eliminating
    repetitive delegation method definitions.

    Args:
        target_attr: The attribute name of the target object (e.g., 'browser_manager')
        method_name: The method name on the target object (e.g., 'close_driver')

    Returns:
        function: A method that delegates to the target

    Example:
        class SessionManager:
            def __init__(self):
                self.browser_manager = BrowserManager()

            # Instead of:
            # def close_driver(self):
            #     return self.browser_manager.close_driver()

            # Use:
            close_driver = create_method_delegator('browser_manager', 'close_driver')
    """

    def delegator(self: Any, *args: Any, **kwargs: Any) -> Any:
        target_obj = getattr(self, target_attr, None)
        if target_obj is None:
            raise AttributeError(f"Target object '{target_attr}' not found")
        method = getattr(target_obj, method_name, None)
        if method is None:
            raise AttributeError(f"Method '{method_name}' not found on {target_attr}")
        return method(*args, **kwargs)

    return delegator


# ===== VALIDATION UTILITIES =====
# Consolidates common validation patterns found across the codebase


def create_range_validator(min_val: Any, max_val: Any, inclusive: bool = True):
    """
    Create a range validation function.

    Consolidates the pattern of validation functions that check if a value
    is within a specific range. This implements DRY principles by providing
    a single factory for range validation instead of multiple similar functions.

    Args:
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        inclusive: Whether the range is inclusive (default: True)

    Returns:
        function: A validation function that checks if value is in range

    Example:
        # Instead of:
        # def _is_valid_year(year: int) -> bool:
        #     return 1000 <= year <= 2100

        # Use:
        is_valid_year = create_range_validator(1000, 2100)

        # Instead of:
        # def validate_port_range(port: int) -> bool:
        #     return 1024 <= port <= 65535

        # Use:
        validate_port_range = create_range_validator(1024, 65535)
    """

    def validator(value: Any) -> bool:
        try:
            if inclusive:
                return min_val <= value <= max_val
            return min_val < value < max_val
        except (TypeError, ValueError):
            return False

    return validator


def create_type_validator(expected_type: type, allow_none: bool = False):
    """
    Create a type validation function.

    Consolidates the pattern of validation functions that check if a value
    is of a specific type. This implements DRY principles by providing
    a single factory for type validation.

    Args:
        expected_type: The expected type
        allow_none: Whether None values are allowed (default: False)

    Returns:
        function: A validation function that checks type

    Example:
        # Instead of:
        # def validate_positive_integer(value: int) -> bool:
        #     return isinstance(value, int) and value > 0

        # Use:
        is_integer = create_type_validator(int)
        is_positive = create_range_validator(1, float('inf'))
        validate_positive_integer = lambda x: is_integer(x) and is_positive(x)
    """

    def validator(value: Any) -> bool:
        if value is None:
            return allow_none
        return isinstance(value, expected_type)

    return validator


def create_string_validator(
    min_length: int = 0, max_length: Optional[int] = None, allow_empty: bool = True, strip_whitespace: bool = True
):
    """
    Create a string validation function.

    Consolidates the pattern of string validation functions found across
    the codebase. This implements DRY principles by providing a single
    factory for string validation with common options.

    Args:
        min_length: Minimum string length (default: 0)
        max_length: Maximum string length (default: None for unlimited)
        allow_empty: Whether empty strings are allowed (default: True)
        strip_whitespace: Whether to strip whitespace before validation (default: True)

    Returns:
        function: A validation function that checks string criteria
    """

    def validator(value: Any) -> bool:
        if not isinstance(value, str):
            return False

        test_value = value.strip() if strip_whitespace else value

        if not allow_empty and not test_value:
            return False

        length = len(test_value)
        if length < min_length:
            return False

        return not (max_length is not None and length > max_length)

    return validator


def create_composite_validator(*validators: Callable[..., bool]) -> Callable[..., bool]:
    """
    Create a composite validation function that requires all validators to pass.

    Consolidates the pattern of validation functions that combine multiple
    validation criteria. This implements DRY principles by providing a way
    to compose validation functions.

    Args:
        *validators: Variable number of validation functions

    Returns:
        function: A validation function that checks all criteria

    Example:
        # Instead of:
        # def validate_positive_integer(value: int) -> bool:
        #     return isinstance(value, int) and value > 0

        # Use:
        validate_positive_integer = create_composite_validator(
            create_type_validator(int),
            create_range_validator(1, float('inf'))
        )
    """

    def validator(value: Any) -> bool:
        return all(v(value) for v in validators)

    return validator


# Pre-built common validators for immediate use
is_valid_year = create_range_validator(1000, 2100)
validate_port_range = create_range_validator(1024, 65535)
validate_positive_integer = create_composite_validator(
    create_type_validator(int), create_range_validator(1, float('inf'))
)
validate_non_empty_string = create_string_validator(min_length=1, allow_empty=False)
validate_optional_string = create_string_validator(allow_empty=True)


def create_file_extension_validator(extensions: list[str]):
    """
    Create a file extension validation function.

    Consolidates the pattern of validating file extensions found in config modules.
    This implements DRY principles by providing a single factory for file extension
    validation instead of multiple similar functions.

    Args:
        extensions: List of allowed file extensions (e.g., ['.py', '.txt'])

    Returns:
        function: A validation function that checks file extensions

    Example:
        # Instead of defining custom file extension validators
        validate_python_files = create_file_extension_validator(['.py', '.pyx'])
        validate_data_files = create_file_extension_validator(['.json', '.yaml', '.yml'])
    """
    from typing import Union

    def validator(path: Union[str, Path, None]) -> bool:
        if not path:
            return True  # Allow None/empty values
        try:
            path_obj = Path(path)
            return path_obj.suffix.lower() in [ext.lower() for ext in extensions]
        except (TypeError, ValueError):
            return False

    return validator


# Additional pre-built validators
validate_python_files = create_file_extension_validator(['.py', '.pyx'])
validate_config_files = create_file_extension_validator(['.json', '.yaml', '.yml', '.toml'])
validate_data_files = create_file_extension_validator(['.csv', '.json', '.xml', '.gedcom'])


def create_standard_test_runner(module_test_function: Callable[[], bool]) -> Callable[[], bool]:
    """
    Create a standardized test runner function.

    Consolidates the pattern of run_comprehensive_tests functions that just call
    their module-specific test function. This implements DRY principles by providing
    a single factory for test runner functions instead of 31+ identical implementations.

    Automatically closes any browser session after tests complete to prevent
    orphaned browser windows.

    Args:
        module_test_function: The module-specific test function to call

    Returns:
        function: A standardized run_comprehensive_tests function

    Example:
        # Instead of:
        # def run_comprehensive_tests() -> bool:
        #     return my_module_tests()

        # Use:
        run_comprehensive_tests = create_standard_test_runner(my_module_tests)
    """

    def run_comprehensive_tests() -> bool:
        """Run comprehensive tests using standardized test runner pattern."""
        try:
            return module_test_function()
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            return False
        finally:
            # Clean up any browser session that was opened during tests
            _cleanup_browser_after_tests()

    return run_comprehensive_tests


def _cleanup_browser_after_tests() -> None:
    """Close any browser session that was opened during test execution."""
    try:
        from core.session_utils import close_cached_session, get_session_manager

        sm = get_session_manager()
        if sm is not None and hasattr(sm, "browser_manager") and getattr(sm.browser_manager, "driver_live", False):
            close_cached_session(keep_db=True)
    except Exception:
        pass  # Silently ignore cleanup errors


# ==============================================
# Test Helper Functions
# ==============================================


def create_mock_session_manager() -> MagicMock:
    """
    Create a mock SessionManager for testing.

    Consolidates the pattern of creating mock session managers across test modules.

    Returns:
        MagicMock: A mock SessionManager with common attributes

    Example:
        sm = create_mock_session_manager()
        sm.get_db_conn.return_value = mock_db_session
    """
    sm = MagicMock()
    sm.session_ready = True
    sm.get_db_conn.return_value = MagicMock(spec=Session)
    sm.browser_manager.driver = MagicMock()
    sm.api_manager.requests_session = MagicMock()
    return sm


@dataclass
class LiveSessionHandle:
    """Container returned by live_session_fixture() for easy attribute access."""

    session_manager: Any
    session_uuid: str
    db_session: Session


@contextlib.contextmanager
def live_session_fixture(
    action_name: str = "Test Session",
    *,
    skip_csrf: bool = True,
) -> Iterator[LiveSessionHandle]:
    """Yield an authenticated SessionManager plus rollback-protected DB session.

    This helper ensures every integration test that touches Ancestry APIs uses the
    single global session (via session_utils.ensure_session_for_tests) and that any
    database writes are rolled back automatically when the test completes.

    Example:
        with live_session_fixture("Action 7 Tests") as live:
            assert live.session_manager.session_ready
            # Perform DB work - it will be rolled back automatically
            live.db_session.add(...)

    Args:
        action_name: Friendly label for logging inside ensure_session_for_tests.
        skip_csrf: Whether to skip CSRF validation (default mirrors existing tests).
    """

    from core.session_utils import ensure_session_for_tests  # Local import to avoid cycles
    from testing.test_framework import database_rollback_test

    session_manager, session_uuid = ensure_session_for_tests(action_name, skip_csrf)
    db_session = session_manager.db_manager.get_session()
    if db_session is None:
        raise RuntimeError("SessionManager did not return a database session; ensure DB is ready before running tests.")

    try:
        with database_rollback_test(db_session):
            yield LiveSessionHandle(
                session_manager=session_manager,
                session_uuid=session_uuid,
                db_session=db_session,
            )
    finally:
        with contextlib.suppress(Exception):
            session_manager.db_manager.return_session(db_session)


def create_test_database() -> Session:
    """
    Create an in-memory test database with schema.

    Consolidates the pattern of creating test databases across modules.

    Returns:
        Session: SQLAlchemy session connected to in-memory test database

    Example:
        session = create_test_database()
        # Use session for testing
        session.close()
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from core.database import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def load_test_gedcom(gedcom_path: Optional[str] = None) -> Any:
    """
    Load a GEDCOM file for testing.

    Consolidates the pattern of loading GEDCOM test files.

    Args:
        gedcom_path: Path to GEDCOM file (default: uses config)

    Returns:
        Parsed GEDCOM data object

    Example:
        gedcom_data = load_test_gedcom()
        individual = gedcom_data.get_individual("@I1@")
    """
    try:
        gedcom_module = import_module("gedcom")
    except ImportError:
        return None

    if gedcom_path is None:
        from config import config_schema

        gedcom_path = getattr(config_schema, "gedcom_file_path", None)

    if not gedcom_path or not Path(gedcom_path).exists():
        return None

    try:
        return gedcom_module.parse(gedcom_path)
    except Exception:
        return None


def create_test_person(
    person_id: int = 1,
    uuid: str = "TEST-UUID-1234",
    username: str = "Test User",
    cm_dna: Optional[int] = None,
    engagement_score: int = 50,
) -> MagicMock:
    """
    Create a mock Person object for testing.

    Consolidates the pattern of creating test Person objects.

    Args:
        person_id: Database ID
        uuid: Person UUID
        username: Display name
        cm_dna: Shared centimorgans (creates dna_match if provided)
        engagement_score: Current engagement score (0-100)

    Returns:
        MagicMock: Mock Person object with DNA match if cm_dna provided

    Example:
        person = create_test_person(cm_dna=100, engagement_score=75)
        assert person.dna_match.cm_dna == 100
    """
    from core.database import Person

    person = MagicMock(spec=Person)
    person.id = person_id
    person.uuid = uuid
    person.username = username
    person.current_engagement_score = engagement_score

    if cm_dna is not None:
        from core.database import DnaMatch

        person.dna_match = MagicMock(spec=DnaMatch)
        person.dna_match.cm_dna = cm_dna
    else:
        person.dna_match = None

    return person


def run_parameterized_tests(test_cases: list[tuple[str, Callable[..., Any], Any, str]], suite: Any) -> None:
    """
    Run a list of parameterized test cases.

    Consolidates the pattern of running multiple test cases with different parameters.
    This implements DRY principles by providing a single function for parameterized testing.

    Args:
        test_cases: List of tuples (test_name, test_func, expected_result, description)
        suite: TestSuite instance to run tests on

    Example:
        test_cases = [
            ("Test case 1", lambda: my_func(1), 2, "Tests with input 1"),
            ("Test case 2", lambda: my_func(2), 4, "Tests with input 2"),
        ]
        run_parameterized_tests(test_cases, suite)
    """
    for test_name, test_func, expected_result, description in test_cases:
        suite.run_test(test_name, test_func, expected_result, description, "parameterized")


def assert_function_behavior(
    func: Callable[..., Any], args: tuple[Any, ...], expected_result: Any, error_message: Optional[str] = None
) -> None:
    """
    Assert that a function behaves as expected with given arguments.

    Consolidates the pattern of testing function behavior with specific inputs.
    This implements DRY principles by providing a single assertion helper.

    Args:
        func: The function to test
        args: Tuple of arguments to pass to the function
        expected_result: The expected return value
        error_message: Optional custom error message

    Raises:
        AssertionError: If the function doesn't return the expected result

    Example:
        assert_function_behavior(my_func, (1, 2), 3, "my_func(1, 2) should return 3")
    """
    result = func(*args)
    if error_message is None:
        error_message = f"{func.__name__}{args} should return {expected_result}, got {result}"
    assert result == expected_result, error_message


def create_test_session() -> Session:
    """
    Create a test database session.

    Consolidates the pattern of creating database sessions for testing.
    This implements DRY principles by providing a single function for session creation.

    Returns:
        Session: A database session for testing

    Example:
        session = create_test_session()
        try:
            # Run tests with session
            pass
        finally:
            session.close()
    """
    from core.database_manager import DatabaseManager

    dm = DatabaseManager()
    session = dm.get_session()
    if session is None:
        raise RuntimeError("Failed to create a test database session")
    return session


def assert_database_state(
    session: Session, model: Any, filters: dict[str, Any], expected_count: int, error_message: Optional[str] = None
) -> None:
    """
    Assert that the database contains the expected number of records.

    Consolidates the pattern of checking database state in tests.
    This implements DRY principles by providing a single assertion helper.

    Args:
        session: Database session
        model: SQLAlchemy model class
        filters: Dictionary of filter criteria
        expected_count: Expected number of records
        error_message: Optional custom error message

    Raises:
        AssertionError: If the database doesn't contain the expected number of records

    Example:
        assert_database_state(session, Person, {"name": "John"}, 1, "Should have 1 John")
    """
    query = session.query(model)
    for key, value in filters.items():
        query = query.filter(getattr(model, key) == value)
    count = query.count()
    if error_message is None:
        error_message = f"Expected {expected_count} records, found {count}"
    assert count == expected_count, error_message


def mock_api_response(
    status_code: int = 200, json_data: Optional[dict[str, Any]] = None, text: Optional[str] = None
) -> MagicMock:
    """
    Create a mock API response object.

    Consolidates the pattern of creating mock API responses for testing.
    This implements DRY principles by providing a single function for mock responses.

    Args:
        status_code: HTTP status code (default: 200)
        json_data: JSON response data (default: None)
        text: Text response data (default: None)

    Returns:
        MagicMock: A mock response object

    Example:
        mock_response = mock_api_response(200, {"result": "success"})
        assert mock_response.status_code == 200
        assert mock_response.json() == {"result": "success"}
    """
    mock_response = MagicMock()
    mock_response.status_code = status_code
    if json_data is not None:
        mock_response.json.return_value = json_data
    if text is not None:
        mock_response.text = text
    return mock_response


# ==============================================
# Test Decorators (Todo §7: Test Utility Framework)
# ==============================================


def with_temp_database(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that provides a temporary in-memory database session.

    The decorated function receives a `db_session` keyword argument with
    an SQLAlchemy session connected to a fresh in-memory SQLite database.
    The session is automatically closed after the test completes.

    Args:
        func: Test function to wrap

    Returns:
        Wrapped function with database session injection

    Example:
        @with_temp_database
        def test_person_creation(db_session):
            person = Person(username="Test")
            db_session.add(person)
            db_session.commit()
            assert db_session.query(Person).count() == 1
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        session = create_test_database()
        try:
            kwargs["db_session"] = session
            return func(*args, **kwargs)
        finally:
            session.close()

    return wrapper


def with_mock_session(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that provides a mock SessionManager.

    The decorated function receives a `session_manager` keyword argument
    with a pre-configured MagicMock simulating SessionManager behavior.

    Args:
        func: Test function to wrap

    Returns:
        Wrapped function with mock session manager injection

    Example:
        @with_mock_session
        def test_api_call(session_manager):
            session_manager.api_manager.requests_session.get.return_value.status_code = 200
            # Test code using session_manager
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        mock_sm = create_mock_session_manager()
        kwargs["session_manager"] = mock_sm
        return func(*args, **kwargs)

    return wrapper


def with_test_config(overrides: Optional[dict[str, Any]] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator factory that provides test configuration with optional overrides.

    The decorated function receives a `test_config` keyword argument with
    a dictionary of configuration values. Original config is restored after test.

    Args:
        overrides: Dictionary of config values to override for the test

    Returns:
        Decorator function

    Example:
        @with_test_config({"max_pages": 5, "debug_mode": True})
        def test_with_custom_config(test_config):
            assert test_config["max_pages"] == 5
            # Test code using modified config
    """
    import functools

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build test config with defaults and overrides
            test_config: dict[str, Any] = {
                "max_pages": 1,
                "requests_per_second": 0.3,
                "debug_mode": False,
                "skip_live_api_tests": True,
                "max_concurrency": 1,
            }
            if overrides:
                test_config.update(overrides)
            kwargs["test_config"] = test_config
            return func(*args, **kwargs)

        return wrapper

    return decorator


def with_rollback(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that wraps test in a database transaction rollback.

    Requires `db_session` argument to be passed to the function.
    All database changes are rolled back after test completion.

    Args:
        func: Test function to wrap

    Returns:
        Wrapped function with automatic rollback

    Example:
        @with_rollback
        def test_data_modification(db_session):
            db_session.add(Person(username="Test"))
            # Changes will be rolled back after test
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        db_session = kwargs.get("db_session")
        if db_session is None:
            raise ValueError("with_rollback requires db_session argument")

        try:
            result = func(*args, **kwargs)
            db_session.rollback()  # Always rollback
            return result
        except Exception:
            db_session.rollback()
            raise

    return wrapper


# ==============================================
# Additional Fixture Factories (Todo §7)
# ==============================================


def create_test_match(
    match_id: int = 1,
    people_id: int = 1,
    cm_dna: int = 100,
    predicted_relationship: str = "3rd-4th Cousin",
    shared_segments: int = 5,
    longest_shared_segment: float = 25.0,
    compare_link: str = "https://ancestry.com/compare/test",
    from_my_fathers_side: bool = False,
    from_my_mothers_side: bool = False,
) -> MagicMock:
    """
    Create a mock DnaMatch object for testing.

    Provides a fully-configured mock DNA match for testing.

    Args:
        match_id: Database ID for the match
        people_id: Foreign key to Person
        cm_dna: Shared centimorgans
        predicted_relationship: Ancestry's relationship prediction
        shared_segments: Number of shared DNA segments
        longest_shared_segment: Length of longest segment in cM
        compare_link: URL to comparison page
        from_my_fathers_side: Paternal match flag
        from_my_mothers_side: Maternal match flag

    Returns:
        MagicMock: Mock DnaMatch object with all attributes

    Example:
        match = create_test_match(cm_dna=250, predicted_relationship="2nd Cousin")
        assert match.cm_dna == 250
        assert match.predicted_relationship == "2nd Cousin"
    """
    from core.database import DnaMatch

    match = MagicMock(spec=DnaMatch)
    match.id = match_id
    match.people_id = people_id
    match.cm_dna = cm_dna
    match.predicted_relationship = predicted_relationship
    match.shared_segments = shared_segments
    match.longest_shared_segment = longest_shared_segment
    match.compare_link = compare_link
    match.from_my_fathers_side = from_my_fathers_side
    match.from_my_mothers_side = from_my_mothers_side
    return match


def create_test_conversation(
    conversation_id: int = 1,
    person_id: int = 1,
    message_content: str = "Test message content",
    direction: str = "received",
    classification: Optional[str] = None,
) -> MagicMock:
    """
    Create a mock ConversationLog object for testing.

    Args:
        conversation_id: Database ID
        person_id: Foreign key to Person
        message_content: The message text
        direction: 'sent' or 'received'
        classification: Optional AI classification (PRODUCTIVE/DESIST/OTHER)

    Returns:
        MagicMock: Mock ConversationLog object

    Example:
        conv = create_test_conversation(classification="PRODUCTIVE")
        assert conv.classification == "PRODUCTIVE"
    """
    from core.database import ConversationLog

    conv = MagicMock(spec=ConversationLog)
    conv.id = conversation_id
    conv.people_id = person_id
    conv.message_content = message_content
    conv.direction = direction
    conv.classification = classification
    return conv


def create_test_person_with_match(
    person_id: int = 1,
    uuid: str = "TEST-UUID-1234",
    username: str = "Test User",
    cm_dna: int = 100,
    predicted_relationship: str = "3rd-4th Cousin",
    engagement_score: int = 50,
) -> MagicMock:
    """
    Create a mock Person with an associated DnaMatch for testing.

    Convenience function that creates both Person and DnaMatch mocks
    with proper linkage.

    Args:
        person_id: Database ID
        uuid: Person UUID
        username: Display name
        cm_dna: Shared centimorgans
        predicted_relationship: Ancestry's relationship prediction
        engagement_score: Current engagement score (0-100)

    Returns:
        MagicMock: Mock Person object with linked DnaMatch

    Example:
        person = create_test_person_with_match(cm_dna=200)
        assert person.dna_match.cm_dna == 200
        assert person.dna_match.people_id == person.id
    """
    person = create_test_person(
        person_id=person_id,
        uuid=uuid,
        username=username,
        cm_dna=cm_dna,
        engagement_score=engagement_score,
    )
    # Link the match properly
    person.dna_match.people_id = person_id
    person.dna_match.predicted_relationship = predicted_relationship
    return person


# ==============================================
# Module Tests
# ==============================================


def _test_basic_functions() -> None:
    """Test basic utility functions."""
    assert mock_func() == "test_result"
    assert mock_func_with_param(5) == 10
    assert sample_function() == "sample_result"
    assert temp_function() == "temp"
    assert safe_func() == "safe_result"
    assert decorated_safe_func() == "decorated_safe_result"


def _test_factory_functions() -> None:
    """Test factory functions."""
    custom_func = create_test_function("custom_result")
    assert custom_func() == "custom_result"

    multiply_func = create_parameterized_test_function("multiply", 3)
    assert multiply_func(4) == 12

    add_func = create_parameterized_test_function("add", 10)
    assert add_func(5) == 15


def _test_function_registry() -> None:
    """Test function registry."""
    retrieved_func = get_test_function("mock_func")
    assert retrieved_func() == "test_result"


def _test_runner_factory() -> None:
    """Test the test runner factory."""

    def dummy_test():
        return True

    test_runner = create_standard_test_runner(dummy_test)
    assert test_runner()


def _test_assert_function_behavior() -> None:
    """Test the assert_function_behavior helper."""
    # Test successful assertion
    assert_function_behavior(mock_func_with_param, (5,), 10)

    # Test with custom error message
    assert_function_behavior(mock_func, (), "test_result", "Custom error message")


def _test_mock_api_response() -> None:
    """Test the mock_api_response helper."""
    # Test with JSON data
    mock_response = mock_api_response(200, {"result": "success"})
    assert mock_response.status_code == 200
    assert mock_response.json() == {"result": "success"}

    # Test with text data
    mock_response = mock_api_response(404, text="Not found")
    assert mock_response.status_code == 404
    assert mock_response.text == "Not found"


def _test_create_mock_session_manager() -> None:
    """Test the create_mock_session_manager helper (Todo #17)."""
    sm = create_mock_session_manager()
    assert sm.session_ready is True
    assert sm.get_db_conn.return_value is not None
    assert sm.browser_manager.driver is not None
    assert sm.api_manager.requests_session is not None


def _test_create_test_database() -> None:
    """Test the create_test_database helper (Todo #17)."""
    session = create_test_database()
    assert session is not None
    # Test that we can query
    from core.database import Person

    result = session.query(Person).count()
    assert result == 0  # Empty database
    session.close()


def _test_create_test_person() -> None:
    """Test the create_test_person helper (Todo #17)."""
    # Test without DNA match
    person = create_test_person()
    assert person.id == 1
    assert person.uuid == "TEST-UUID-1234"
    assert person.username == "Test User"
    assert person.current_engagement_score == 50
    assert person.dna_match is None

    # Test with DNA match
    person_with_dna = create_test_person(cm_dna=150, engagement_score=80)
    assert person_with_dna.dna_match is not None
    assert person_with_dna.dna_match.cm_dna == 150
    assert person_with_dna.current_engagement_score == 80


def _test_temp_directory() -> None:
    """Test the temp_directory helper (Task 5)."""
    # Test with cleanup
    with temp_directory(prefix="test-task5-") as tmpdir:
        assert tmpdir.exists()
        assert tmpdir.is_dir()
        assert "test-task5-" in tmpdir.name

        # Create a test file
        test_file = tmpdir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()

    # Directory should be cleaned up
    assert not tmpdir.exists()


def _test_temp_file() -> None:
    """Test the temp_file helper (Task 5)."""
    # Test with automatic cleanup
    with temp_file(suffix=".json", prefix="test-") as tmp_path:
        assert tmp_path.exists()
        assert tmp_path.suffix == ".json"
        assert "test-" in tmp_path.name

        # Write and read content
        tmp_path.write_text('{"test": true}')
        content = tmp_path.read_text()
        assert "test" in content

    # File should be deleted
    assert not tmp_path.exists()


def _test_atomic_write_file() -> None:
    """Test the atomic_write_file helper (Task 5)."""
    import json

    with temp_directory() as tmpdir:
        target = tmpdir / "test.json"

        # Test successful write
        with atomic_write_file(target) as f:
            json.dump({"key": "value"}, f)

        assert target.exists()
        with target.open(encoding="utf-8") as f:
            data = json.load(f)
        assert data == {"key": "value"}

        # Test that file is atomically replaced
        with atomic_write_file(target) as f:
            json.dump({"updated": "data"}, f)

        with target.open(encoding="utf-8") as f:
            data = json.load(f)
        assert data == {"updated": "data"}


def _test_with_temp_database_decorator() -> None:
    """Test the @with_temp_database decorator (Todo §7)."""

    @with_temp_database
    def inner_test(db_session: Session) -> bool:
        from core.database import Person

        # Verify session works
        count = db_session.query(Person).count()
        assert count == 0, "Fresh database should be empty"
        return True

    result = inner_test()
    assert result is True


def _test_with_mock_session_decorator() -> None:
    """Test the @with_mock_session decorator (Todo §7)."""

    @with_mock_session
    def inner_test(session_manager: MagicMock) -> bool:
        # Verify mock is properly configured
        assert session_manager.session_ready is True
        assert session_manager.browser_manager.driver is not None
        assert session_manager.api_manager.requests_session is not None
        return True

    result = inner_test()
    assert result is True


def _test_with_test_config_decorator() -> None:
    """Test the @with_test_config decorator (Todo §7)."""

    @with_test_config({"max_pages": 10, "debug_mode": True})
    def inner_test(test_config: dict[str, Any]) -> bool:
        # Verify overrides applied
        assert test_config["max_pages"] == 10
        assert test_config["debug_mode"] is True
        # Verify defaults preserved
        assert test_config["requests_per_second"] == 0.3
        return True

    result = inner_test()
    assert result is True


def _test_create_test_match() -> None:
    """Test the create_test_match helper (Todo §7)."""
    # Test with defaults
    match = create_test_match()
    assert match.id == 1
    assert match.cm_dna == 100
    assert match.predicted_relationship == "3rd-4th Cousin"
    assert match.shared_segments == 5

    # Test with custom values
    custom_match = create_test_match(
        cm_dna=250,
        predicted_relationship="2nd Cousin",
        from_my_fathers_side=True,
    )
    assert custom_match.cm_dna == 250
    assert custom_match.predicted_relationship == "2nd Cousin"
    assert custom_match.from_my_fathers_side is True


def _test_create_test_conversation() -> None:
    """Test the create_test_conversation helper (Todo §7)."""
    # Test with defaults
    conv = create_test_conversation()
    assert conv.id == 1
    assert conv.direction == "received"
    assert conv.classification is None

    # Test with classification
    classified = create_test_conversation(
        classification="PRODUCTIVE",
        message_content="I have info about our ancestor",
    )
    assert classified.classification == "PRODUCTIVE"
    assert classified.message_content == "I have info about our ancestor"


def _test_create_test_person_with_match() -> None:
    """Test the create_test_person_with_match helper (Todo §7)."""
    person = create_test_person_with_match(
        person_id=5,
        cm_dna=200,
        predicted_relationship="2nd Cousin",
    )
    assert person.id == 5
    assert person.dna_match is not None
    assert person.dna_match.cm_dna == 200
    assert person.dna_match.people_id == 5
    assert person.dna_match.predicted_relationship == "2nd Cousin"


def test_utilities_module_tests() -> bool:
    """Test the test utilities module itself."""
    from testing.test_framework import TestSuite, suppress_logging

    suite = TestSuite("Test Utilities", "test_utilities.py")

    tests = [
        (
            "Basic utility functions",
            _test_basic_functions,
            "Test basic utilities",
            "direct",
            "Test core utility functions",
        ),
        ("Factory functions", _test_factory_functions, "Test function factories", "direct", "Test function factories"),
        ("Function registry", _test_function_registry, "Test function registry", "direct", "Test function registry"),
        ("Test runner factory", _test_runner_factory, "Test runner creation", "direct", "Test runner creation"),
        (
            "Assert function behavior",
            _test_assert_function_behavior,
            "Test assertion helper",
            "direct",
            "Test assertion helper",
        ),
        (
            "Mock API response",
            _test_mock_api_response,
            "Test mock response creation",
            "direct",
            "Test mock response creation",
        ),
        (
            "Create mock session manager (Todo #17)",
            _test_create_mock_session_manager,
            "Test session manager mock helper",
            "direct",
            "Test session manager mock helper",
        ),
        (
            "Create test database (Todo #17)",
            _test_create_test_database,
            "Test database creation helper",
            "direct",
            "Test database creation helper",
        ),
        (
            "Create test person (Todo #17)",
            _test_create_test_person,
            "Test person mock helper",
            "direct",
            "Test person mock helper",
        ),
        (
            "Temp directory helper (Task 5)",
            _test_temp_directory,
            "Test temp directory creation and cleanup",
            "direct",
            "Test temp directory helper",
        ),
        (
            "Temp file helper (Task 5)",
            _test_temp_file,
            "Test temp file creation and cleanup",
            "direct",
            "Test temp file helper",
        ),
        (
            "Atomic write file helper (Task 5)",
            _test_atomic_write_file,
            "Test atomic file writing",
            "direct",
            "Test atomic write helper",
        ),
        # New decorators and factories (Todo §7)
        (
            "@with_temp_database decorator (Todo §7)",
            _test_with_temp_database_decorator,
            "Test temp database injection",
            "direct",
            "Test @with_temp_database decorator",
        ),
        (
            "@with_mock_session decorator (Todo §7)",
            _test_with_mock_session_decorator,
            "Test mock session injection",
            "direct",
            "Test @with_mock_session decorator",
        ),
        (
            "@with_test_config decorator (Todo §7)",
            _test_with_test_config_decorator,
            "Test config override injection",
            "direct",
            "Test @with_test_config decorator",
        ),
        (
            "Create test match (Todo §7)",
            _test_create_test_match,
            "Test DNA match mock helper",
            "direct",
            "Test create_test_match factory",
        ),
        (
            "Create test conversation (Todo §7)",
            _test_create_test_conversation,
            "Test conversation mock helper",
            "direct",
            "Test create_test_conversation factory",
        ),
        (
            "Create test person with match (Todo §7)",
            _test_create_test_person_with_match,
            "Test person+match mock helper",
            "direct",
            "Test create_test_person_with_match factory",
        ),
    ]

    with suppress_logging():
        for test_name, test_func, expected_behavior, test_description, method_description in tests:
            suite.run_test(test_name, test_func, expected_behavior, test_description, method_description)

    return suite.finish_suite()


# Use centralized test runner utility (self-referential)
run_comprehensive_tests = create_standard_test_runner(test_utilities_module_tests)


if __name__ == "__main__":
    import sys

    success = test_utilities_module_tests()
    sys.exit(0 if success else 1)
