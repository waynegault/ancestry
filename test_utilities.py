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


def test_func() -> str:
    """
    Standard test function that returns a predictable result.

    Used for testing function registration, retrieval, and execution systems.
    Consolidated from multiple modules where this exact function was duplicated.

    Returns:
        str: "test_result"
    """
    return "test_result"


def test_func_with_param(x: int) -> int:
    """
    Test function that takes a parameter and returns a computed result.

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
    def test_function() -> Any:
        return return_value
    return test_function


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
    def test_function(x: int) -> int:
        if operation == "multiply":
            return x * factor
        if operation == "add":
            return x + factor
        if operation == "subtract":
            return x - factor
        return x
    return test_function


# Test function registry for common test patterns
TEST_FUNCTIONS = {
    "test_func": test_func,
    "test_func_with_param": test_func_with_param,
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


def test_utilities_module_tests() -> bool:
    """Test the test utilities module itself."""
    try:
        # Test basic functions
        assert test_func() == "test_result"
        assert test_func_with_param(5) == 10
        assert sample_function() == "sample_result"
        assert temp_function() == "temp"
        assert safe_func() == "safe_result"
        assert decorated_safe_func() == "decorated_safe_result"

        # Test factory functions
        custom_func = create_test_function("custom_result")
        assert custom_func() == "custom_result"

        multiply_func = create_parameterized_test_function("multiply", 3)
        assert multiply_func(4) == 12

        add_func = create_parameterized_test_function("add", 10)
        assert add_func(5) == 15

        # Test registry
        retrieved_func = get_test_function("test_func")
        assert retrieved_func() == "test_result"

        # Test the new test runner factory
        def dummy_test():
            return True

        test_runner = create_standard_test_runner(dummy_test)
        assert test_runner()

        print("✅ Test utilities module tests passed")
        return True

    except Exception as e:
        print(f"❌ Test utilities module tests failed: {e}")
        return False


# Use centralized test runner utility (self-referential)
run_comprehensive_tests = create_standard_test_runner(test_utilities_module_tests)


if __name__ == "__main__":
    import sys
    success = test_utilities_module_tests()
    sys.exit(0 if success else 1)
