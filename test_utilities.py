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


def get_test_function(name: str) -> Callable:
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
    def getter(self) -> Any:
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
    def delegator(self, *args: Any, **kwargs: Any) -> Any:
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


def create_string_validator(min_length: int = 0, max_length: Optional[int] = None,
                          allow_empty: bool = True, strip_whitespace: bool = True):
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


def create_composite_validator(*validators: Callable) -> Callable:
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
    create_type_validator(int),
    create_range_validator(1, float('inf'))
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
    from pathlib import Path
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


def create_standard_test_runner(module_test_function):
    """
    Create a standardized test runner function.

    Consolidates the pattern of run_comprehensive_tests functions that just call
    their module-specific test function. This implements DRY principles by providing
    a single factory for test runner functions instead of 31+ identical implementations.

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

    return run_comprehensive_tests


# ==============================================
# Test Helper Functions
# ==============================================


def create_mock_session_manager() -> MagicMock:
    """
    Create a mock SessionManager for testing.

    Test Infrastructure Todo #17: Shared test helpers
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
    sm.driver = MagicMock()
    sm.requests_session = MagicMock()
    return sm


def create_test_database() -> Session:
    """
    Create an in-memory test database with schema.

    Test Infrastructure Todo #17: Shared test helpers
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
    from database import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def load_test_gedcom(gedcom_path: Optional[str] = None) -> Any:
    """
    Load a GEDCOM file for testing.

    Test Infrastructure Todo #17: Shared test helpers
    Consolidates the pattern of loading GEDCOM test files.

    Args:
        gedcom_path: Path to GEDCOM file (default: uses config)

    Returns:
        Parsed GEDCOM data object

    Example:
        gedcom_data = load_test_gedcom()
        individual = gedcom_data.get_individual("@I1@")
    """
    import sys
    from pathlib import Path

    try:
        import gedcom  # type: ignore
    except ImportError:
        return None

    if gedcom_path is None:
        from config import config_schema
        gedcom_path = getattr(config_schema, "gedcom_file_path", None)

    if not gedcom_path or not Path(gedcom_path).exists():
        return None

    try:
        gedcom_data = gedcom.parse(gedcom_path)
        return gedcom_data
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

    Test Infrastructure Todo #17: Shared test helpers
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
    from database import Person

    person = MagicMock(spec=Person)
    person.id = person_id
    person.uuid = uuid
    person.username = username
    person.current_engagement_score = engagement_score

    if cm_dna is not None:
        from database import DnaMatch
        person.dna_match = MagicMock(spec=DnaMatch)
        person.dna_match.cm_dna = cm_dna
    else:
        person.dna_match = None

    return person


def run_parameterized_tests(test_cases: list[tuple[str, Callable, Any, str]], suite: Any) -> None:
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


def assert_function_behavior(func: Callable, args: tuple, expected_result: Any,
                            error_message: Optional[str] = None) -> None:
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
    return dm.get_session()


def assert_database_state(session: Session, model: Any, filters: dict[str, Any],
                         expected_count: int, error_message: Optional[str] = None) -> None:
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


def mock_api_response(status_code: int = 200, json_data: Optional[dict] = None,
                     text: Optional[str] = None) -> MagicMock:
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
    assert sm.driver is not None
    assert sm.requests_session is not None


def _test_create_test_database() -> None:
    """Test the create_test_database helper (Todo #17)."""
    session = create_test_database()
    assert session is not None
    # Test that we can query
    from database import Person
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


def test_utilities_module_tests() -> bool:
    """Test the test utilities module itself."""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Test Utilities", "test_utilities.py")

    tests = [
        ("Basic utility functions", _test_basic_functions, True, "direct", "Test core utility functions"),
        ("Factory functions", _test_factory_functions, "Test function factories", "direct", "Test function factories"),
        ("Function registry", _test_function_registry, "Test function registry", "direct", "Test function registry"),
        ("Test runner factory", _test_runner_factory, "Test runner creation", "direct", "Test runner creation"),
        ("Assert function behavior", _test_assert_function_behavior, "Test assertion helper", "direct", "Test assertion helper"),
        ("Mock API response", _test_mock_api_response, "Test mock response creation", "direct", "Test mock response creation"),
        ("Create mock session manager (Todo #17)", _test_create_mock_session_manager, "Test session manager mock helper", "direct", "Test session manager mock helper"),
        ("Create test database (Todo #17)", _test_create_test_database, "Test database creation helper", "direct", "Test database creation helper"),
        ("Create test person (Todo #17)", _test_create_test_person, "Test person mock helper", "direct", "Test person mock helper"),
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
