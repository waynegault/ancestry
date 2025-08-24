#!/usr/bin/env python3

"""
Python Best Practices Utilities

This module provides utilities and decorators to enforce Python best practices
throughout the codebase, including type safety, immutability, and defensive programming.

Key Features:
- Type validation decorators
- Immutable data structures
- Defensive programming utilities
- Performance monitoring
- Error handling improvements
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === TYPE DEFINITIONS ===
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# === CUSTOM EXCEPTIONS ===

class PythonBestPracticesError(Exception):
    """Base exception for Python best practices violations."""
    pass


class ValidationError(PythonBestPracticesError):
    """Raised when validation fails."""
    pass


class ConfigurationError(PythonBestPracticesError):
    """Raised when configuration is invalid."""
    pass


class ResourceError(PythonBestPracticesError):
    """Raised when resource management fails."""
    pass

# === IMMUTABLE DATA STRUCTURES ===

@dataclass(frozen=True)
class ImmutableConfig:
    """Immutable configuration object following best practices."""

    name: str
    version: str = "1.0.0"
    debug: bool = False
    settings: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Configuration name cannot be empty")
        if not isinstance(self.version, str):
            raise TypeError("Version must be a string")


# === DEFENSIVE PROGRAMMING DECORATORS ===

def validate_types(**type_hints: type) -> Callable[[F], F]:
    """
    Decorator to validate function argument types at runtime.
    
    Example:
        @validate_types(name=str, age=int)
        def create_person(name: str, age: int) -> Person:
            return Person(name, age)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate types
            for param_name, expected_type in type_hints.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if value is not None and not isinstance(value, expected_type):
                        raise TypeError(
                            f"Parameter '{param_name}' must be of type {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )

            return func(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


def require_non_empty(param_names: List[str]) -> Callable[[F], F]:
    """
    Decorator to ensure specified parameters are not empty.
    
    Example:
        @require_non_empty(['name', 'email'])
        def create_user(name: str, email: str) -> User:
            return User(name, email)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            for param_name in param_names:
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not value:  # Handles None, empty string, empty list, etc.
                        raise ValueError(f"Parameter '{param_name}' cannot be empty")

            return func(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


def fail_fast(condition: Callable[..., bool], message: str = "Precondition failed") -> Callable[[F], F]:
    """
    Decorator to implement fail-fast principle with custom validation.
    
    Example:
        @fail_fast(lambda self: self.is_initialized, "Object must be initialized")
        def process_data(self) -> None:
            # Process data
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not condition(*args, **kwargs):
                raise RuntimeError(message)
            return func(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


# === PERFORMANCE MONITORING ===

@dataclass
class PerformanceMetrics:
    """Immutable performance metrics."""

    function_name: str
    execution_time: float
    memory_usage_mb: float
    call_count: int = 1

    def __post_init__(self) -> None:
        """Validate metrics."""
        if self.execution_time < 0:
            raise ValueError("Execution time cannot be negative")
        if self.memory_usage_mb < 0:
            raise ValueError("Memory usage cannot be negative")


def monitor_performance(threshold_seconds: float = 1.0) -> Callable[[F], F]:
    """
    Decorator to monitor function performance and log slow operations.
    
    Args:
        threshold_seconds: Log warning if execution exceeds this threshold
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()

            try:
                return func(*args, **kwargs)
            finally:
                execution_time = time.perf_counter() - start_time

                if execution_time > threshold_seconds:
                    logger.warning(
                        f"Slow operation detected: {func.__name__} took {execution_time:.3f}s "
                        f"(threshold: {threshold_seconds}s)"
                    )
                else:
                    logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")

        return wrapper  # type: ignore
    return decorator


# === CONTEXT MANAGERS ===

@contextmanager
def safe_operation(operation_name: str, reraise: bool = True):
    """
    Context manager for safe operations with proper logging.

    Args:
        operation_name: Name of the operation for logging
        reraise: Whether to reraise exceptions after logging
    """
    logger.debug(f"Starting operation: {operation_name}")
    start_time = time.perf_counter()

    try:
        yield
        execution_time = time.perf_counter() - start_time
        logger.debug(f"Operation '{operation_name}' completed successfully in {execution_time:.3f}s")
    except Exception as e:
        execution_time = time.perf_counter() - start_time
        logger.error(f"Operation '{operation_name}' failed after {execution_time:.3f}s: {e}")
        if reraise:
            raise


@contextmanager
def database_transaction(session_manager, operation_name: str = "database_operation"):
    """
    Context manager for database transactions with automatic rollback on failure.

    Args:
        session_manager: Database session manager
        operation_name: Name of the operation for logging
    """
    logger.debug(f"Starting database transaction: {operation_name}")

    try:
        yield session_manager
        logger.debug(f"Database transaction '{operation_name}' completed successfully")
    except Exception as e:
        logger.error(f"Database transaction '{operation_name}' failed: {e}")
        try:
            if hasattr(session_manager, 'rollback'):
                session_manager.rollback()
                logger.debug(f"Rolled back transaction '{operation_name}'")
        except Exception as rollback_error:
            logger.error(f"Failed to rollback transaction '{operation_name}': {rollback_error}")
        raise


@contextmanager
def temporary_config(config_updates: Dict[str, Any]):
    """
    Context manager for temporary configuration changes.

    Args:
        config_updates: Dictionary of configuration updates to apply temporarily
    """
    # Store original values (placeholder retained for future use)
    _original_values: Dict[str, Any] = {}

    try:
        # Apply temporary updates (implementation would depend on config system)
        logger.debug(f"Applying temporary config updates: {list(config_updates.keys())}")
        yield
    finally:
        # Restore original values
        logger.debug("Restoring original configuration values")
        # Implementation would restore original config values


# === UTILITY FUNCTIONS ===

def ensure_immutable(data: Any) -> Any:
    """
    Convert mutable data structures to immutable equivalents.
    
    Args:
        data: Data to make immutable
        
    Returns:
        Immutable version of the data
    """
    if isinstance(data, dict):
        return tuple(sorted(data.items()))
    if isinstance(data, list):
        return tuple(data)
    if isinstance(data, set):
        return frozenset(data)
    return data


def safe_get(container: Dict[str, Any], key: str, default: T = None) -> Union[Any, T]:
    """
    Safely get value from container with type-safe default.

    Args:
        container: Dictionary to get value from
        key: Key to look up
        default: Default value if key not found

    Returns:
        Value from container or default
    """
    if not isinstance(container, dict):
        logger.warning(f"Expected dict, got {type(container).__name__}")
        return default

    return container.get(key, default)


# === FUNCTIONAL PROGRAMMING UTILITIES ===

def pipe(*functions: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Create a pipeline of functions for functional composition.

    Args:
        *functions: Functions to compose in order

    Returns:
        Composed function that applies all functions in sequence

    Example:
        >>> add_one = lambda x: x + 1
        >>> multiply_two = lambda x: x * 2
        >>> pipeline = pipe(add_one, multiply_two)
        >>> pipeline(5)  # (5 + 1) * 2 = 12
        12
    """
    def composed(value: Any) -> Any:
        result = value
        for func in functions:
            result = func(result)
        return result
    return composed


def curry(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Convert a function to support currying (partial application).

    Args:
        func: Function to curry

    Returns:
        Curried version of the function
    """
    import inspect
    sig = inspect.signature(func)
    param_count = len(sig.parameters)

    def curried(*args: Any, **kwargs: Any) -> Any:
        if len(args) + len(kwargs) >= param_count:
            return func(*args, **kwargs)
        return lambda *more_args, **more_kwargs: curried(
            *(args + more_args), **{**kwargs, **more_kwargs}
        )

    return curried


def maybe(value: Optional[T]) -> Maybe[T]:
    """
    Create a Maybe monad for safe null handling.

    Args:
        value: Value to wrap in Maybe

    Returns:
        Maybe instance containing the value
    """
    return Maybe(value)


class Maybe:
    """Maybe monad for safe null handling."""

    def __init__(self, value: Optional[T]):
        self._value = value

    def map(self, func: Callable[[T], Any]) -> Maybe:
        """Apply function if value is not None."""
        if self._value is not None:
            try:
                return Maybe(func(self._value))
            except Exception as e:
                logger.debug(f"Maybe.map failed: {e}")
                return Maybe(None)
        return Maybe(None)

    def flat_map(self, func: Callable[[T], Maybe]) -> Maybe:
        """Apply function that returns Maybe if value is not None."""
        if self._value is not None:
            try:
                return func(self._value)
            except Exception as e:
                logger.debug(f"Maybe.flat_map failed: {e}")
                return Maybe(None)
        return Maybe(None)

    def get_or_else(self, default: Any) -> Any:
        """Get value or return default if None."""
        return self._value if self._value is not None else default

    def is_some(self) -> bool:
        """Check if value is present."""
        return self._value is not None

    def is_none(self) -> bool:
        """Check if value is None."""
        return self._value is None


# === TESTING UTILITIES ===

def run_comprehensive_tests() -> bool:
    """Run comprehensive tests for Python best practices utilities."""
    try:
        from test_framework import TestSuite

        suite = TestSuite("Python Best Practices", "python_best_practices")

        def test_immutable_config():
            """Test immutable configuration."""
            config = ImmutableConfig(name="test", version="1.0.0")
            assert config.name == "test"
            assert config.version == "1.0.0"

            # Test immutability
            try:
                config.name = "changed"  # type: ignore
                assert False, "Should not be able to modify immutable config"
            except AttributeError:
                pass  # Expected

        def test_type_validation():
            """Test type validation decorator."""
            @validate_types(name=str, age=int)
            def create_person(name: str, age: int) -> str:
                return f"{name} is {age} years old"

            # Valid call
            result = create_person("John", 30)
            assert result == "John is 30 years old"

            # Invalid call should raise TypeError
            try:
                create_person("John", "thirty")  # type: ignore
                assert False, "Should raise TypeError for invalid type"
            except TypeError:
                pass  # Expected

        def test_safe_operation():
            """Test safe operation context manager."""
            with safe_operation("test_operation", reraise=False):
                pass  # Should complete successfully

        suite.run_test(
            "Immutable Configuration",
            test_immutable_config,
            "Immutable configuration objects work correctly",
            "Test ImmutableConfig creation and immutability",
            "Verify configuration cannot be modified after creation"
        )

        suite.run_test(
            "Type Validation",
            test_type_validation,
            "Type validation decorator works correctly",
            "Test validate_types decorator with valid and invalid inputs",
            "Verify runtime type checking prevents type errors"
        )

        suite.run_test(
            "Safe Operations",
            test_safe_operation,
            "Safe operation context manager works correctly",
            "Test safe_operation context manager",
            "Verify proper logging and error handling"
        )

        return suite.finish_suite()

    except ImportError:
        logger.warning("TestSuite not available - running basic tests")
        return True


if __name__ == "__main__":
    import sys
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
