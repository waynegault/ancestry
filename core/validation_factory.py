#!/usr/bin/env python3

"""
Validation Factory - Reusable Validation Builder Patterns

Centralized validation utilities that eliminate duplicate validation code
across the codebase. Provides factory functions for common validation
patterns including:
- Session/driver validation
- API request prerequisite checks
- Database entity field validation
- Configuration schema validation

Usage:
    from core.validation_factory import (
        create_required_field_validator,
        create_session_validator,
        validate_session_manager,
        validate_driver_session,
    )

    # Use factory-created validator
    validator = create_required_field_validator(['username', 'email'])
    is_valid = validator(user_data)

    # Use pre-built validators
    if not validate_session_manager(session_manager, "API call"):
        return None
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

# === CORE INFRASTRUCTURE ===
# Add project root to path for direct script execution
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from selenium.webdriver.remote.webdriver import WebDriver

    from core.session_manager import SessionManager

# Type variable for generic validation
T = TypeVar("T")

# =============================================================================
# Factory Functions for Creating Validators
# =============================================================================


def create_required_field_validator(
    required_fields: list[str],
    allow_empty: bool = False,
    error_prefix: str = "Validation",
) -> Callable[[dict[str, Any]], tuple[bool, str]]:
    """
    Factory function to create a required field validator.

    Args:
        required_fields: List of field names that must be present
        allow_empty: If False, fields must have non-empty values
        error_prefix: Prefix for error messages

    Returns:
        Validator function that returns (is_valid, error_message)

    Example:
        validator = create_required_field_validator(['username', 'email'])
        is_valid, error = validator({'username': 'john'})
        # is_valid=False, error="Validation: Missing required field(s): email"
    """

    def validator(data: dict[str, Any]) -> tuple[bool, str]:
        if not isinstance(data, dict):
            return False, f"{error_prefix}: Expected dict, got {type(data).__name__}"

        missing_fields: list[str] = []
        empty_fields: list[str] = []

        for field in required_fields:
            if field not in data:
                missing_fields.append(field)
            elif not allow_empty and data[field] in (None, "", [], {}):
                empty_fields.append(field)

        if missing_fields:
            return False, f"{error_prefix}: Missing required field(s): {', '.join(missing_fields)}"

        if empty_fields:
            return False, f"{error_prefix}: Empty value in required field(s): {', '.join(empty_fields)}"

        return True, ""

    return validator


def create_type_validator(
    field_types: dict[str, type | tuple[type, ...]],
    error_prefix: str = "Type validation",
) -> Callable[[dict[str, Any]], tuple[bool, str]]:
    """
    Factory function to create a type validator for dict fields.

    Args:
        field_types: Dict mapping field names to expected types
        error_prefix: Prefix for error messages

    Returns:
        Validator function that returns (is_valid, error_message)

    Example:
        validator = create_type_validator({'age': int, 'name': str})
        is_valid, error = validator({'age': '25', 'name': 'John'})
        # is_valid=False, error="Type validation: Field 'age' expected int, got str"
    """

    def validator(data: dict[str, Any]) -> tuple[bool, str]:
        if not isinstance(data, dict):
            return False, f"{error_prefix}: Expected dict, got {type(data).__name__}"

        for field, expected_type in field_types.items():
            if field in data and data[field] is not None and not isinstance(data[field], expected_type):
                actual_type = type(data[field]).__name__
                expected_name = expected_type.__name__ if isinstance(expected_type, type) else str(expected_type)
                return False, f"{error_prefix}: Field '{field}' expected {expected_name}, got {actual_type}"

        return True, ""

    return validator


def create_range_validator(
    field_ranges: dict[str, tuple[float | None, float | None]],
    error_prefix: str = "Range validation",
) -> Callable[[dict[str, Any]], tuple[bool, str]]:
    """
    Factory function to create a numeric range validator.

    Args:
        field_ranges: Dict mapping field names to (min, max) tuples (None for unbounded)
        error_prefix: Prefix for error messages

    Returns:
        Validator function that returns (is_valid, error_message)

    Example:
        validator = create_range_validator({'age': (0, 150), 'score': (0, 100)})
        is_valid, error = validator({'age': 200})
        # is_valid=False, error="Range validation: Field 'age' value 200 outside range [0, 150]"
    """

    def validator(data: dict[str, Any]) -> tuple[bool, str]:
        if not isinstance(data, dict):
            return False, f"{error_prefix}: Expected dict, got {type(data).__name__}"

        for field, (min_val, max_val) in field_ranges.items():
            if field in data and data[field] is not None:
                value = data[field]
                if not isinstance(value, (int, float)):
                    continue  # Skip non-numeric fields

                if min_val is not None and value < min_val:
                    return False, f"{error_prefix}: Field '{field}' value {value} below minimum {min_val}"

                if max_val is not None and value > max_val:
                    return False, f"{error_prefix}: Field '{field}' value {value} above maximum {max_val}"

        return True, ""

    return validator


# =============================================================================
# Pre-built Validators for Common Patterns
# =============================================================================


def validate_session_manager(
    session_manager: SessionManager | None,
    context: str = "Operation",
    require_driver: bool = False,
    require_db: bool = False,
) -> bool:
    """
    Validate session manager state for an operation.

    Consolidates the pattern found in:
    - utils.py: _validate_api_req_prerequisites
    - utils.py: _validate_login_status_inputs
    - api_utils.py: Multiple validation functions

    Args:
        session_manager: The session manager to validate
        context: Description of the operation for logging
        require_driver: If True, validate driver is available
        require_db: If True, validate database session is available

    Returns:
        bool: True if session manager is valid for the operation
    """
    if not session_manager:
        logger.error(f"{context}: SessionManager is None")
        return False

    # Check for basic validity method
    if hasattr(session_manager, "is_sess_valid") and not session_manager.is_sess_valid():
        logger.debug(f"{context}: Session is not valid")
        return False

    # Check driver if required
    if require_driver:
        driver = getattr(session_manager, "driver", None)
        if not driver:
            logger.error(f"{context}: Driver is None but required")
            return False

    # Check database session if required
    if require_db:
        db_session = getattr(session_manager, "db_session", None)
        if not db_session:
            logger.error(f"{context}: Database session is None but required")
            return False

    return True


def validate_driver_session(
    driver: WebDriver | None,
    context: str = "Operation",
    check_responsive: bool = True,
) -> bool:
    """
    Validate that a WebDriver session is active and responsive.

    Consolidates the pattern found in:
    - utils.py: _validate_driver_session
    - utils.py: _validate_driver_for_sync

    Args:
        driver: The WebDriver instance to validate
        context: Description of the operation for logging
        check_responsive: If True, verify session is responsive via title check

    Returns:
        bool: True if driver session is valid
    """
    if not driver:
        logger.debug(f"{context}: Driver is None")
        return False

    if check_responsive:
        try:
            _ = driver.title  # Quick responsiveness check
        except Exception as e:
            logger.warning(f"{context}: Driver session invalid/unresponsive ({type(e).__name__})")
            return False

    return True


def validate_api_prerequisites(
    session_manager: SessionManager | None,
    context: str = "API request",
    require_config: bool = True,
) -> bool:
    """
    Validate prerequisites for making an API request.

    Consolidates the pattern found in:
    - utils.py: _validate_api_req_prerequisites

    Args:
        session_manager: The session manager instance
        context: Description of the API operation
        require_config: If True, validate config_schema is loaded

    Returns:
        bool: True if prerequisites are met
    """
    if not validate_session_manager(session_manager, context):
        return False

    # Check requests session availability
    requests_session = getattr(session_manager, "requests_session", None)
    if not requests_session:
        api_manager = getattr(session_manager, "api_manager", None)
        if api_manager:
            requests_session = getattr(api_manager, "session", None)

    if not requests_session:
        logger.error(f"{context}: No requests session available")
        return False

    # Check config if required
    if require_config:
        try:
            from config import config_schema

            if not config_schema:
                logger.error(f"{context}: Config schema not loaded")
                return False
        except ImportError:
            logger.error(f"{context}: Could not import config_schema")
            return False

    return True


def validate_person_data(
    person_data: dict[str, Any],
    context: str = "Person",
    require_username: bool = True,
    require_identifiers: bool = False,
) -> tuple[bool, str]:
    """
    Validate person data for database operations.

    Consolidates the pattern found in:
    - database.py: _validate_person_required_fields
    - database.py: _validate_person_identifiers

    Args:
        person_data: Dictionary containing person data
        context: Description for error messages
        require_username: If True, username must be present
        require_identifiers: If True, either profile_id or uuid must be present

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(person_data, dict):
        return False, f"{context}: Expected dict, got {type(person_data).__name__}"

    # Check username if required
    if require_username:
        username = person_data.get("username")
        if not username:
            return False, f"{context}: Missing required field 'username'"

    # Check identifiers if required
    if require_identifiers:
        profile_id = person_data.get("profile_id")
        uuid_val = person_data.get("uuid")
        if not profile_id and not uuid_val:
            return False, f"{context}: At least one identifier (profile_id or uuid) is required"

    return True, ""


def validate_dna_match_data(
    match_data: dict[str, Any],
    context: str = "DNA Match",
) -> tuple[bool, str]:
    """
    Validate DNA match data for database operations.

    Consolidates the pattern found in:
    - database.py: _validate_dna_match_data
    - database.py: _validate_dna_match_people_id

    Args:
        match_data: Dictionary containing DNA match data
        context: Description for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(match_data, dict):
        return False, f"{context}: Expected dict, got {type(match_data).__name__}"

    # Check for people_id (required foreign key)
    people_id = match_data.get("people_id")
    if people_id is None:
        return False, f"{context}: Missing required field 'people_id'"

    if not isinstance(people_id, int) or people_id <= 0:
        return False, f"{context}: Invalid people_id: {people_id}"

    return True, ""


# =============================================================================
# Composite Validators
# =============================================================================


def create_composite_validator(
    *validators: Callable[[dict[str, Any]], tuple[bool, str]],
) -> Callable[[dict[str, Any]], tuple[bool, str]]:
    """
    Create a composite validator that runs multiple validators in sequence.

    Args:
        validators: Variable number of validator functions

    Returns:
        Composite validator that returns first failure or success

    Example:
        composite = create_composite_validator(
            create_required_field_validator(['name']),
            create_type_validator({'age': int}),
        )
        is_valid, error = composite(data)
    """

    def composite_validator(data: dict[str, Any]) -> tuple[bool, str]:
        for validator in validators:
            is_valid, error = validator(data)
            if not is_valid:
                return False, error
        return True, ""

    return composite_validator


# =============================================================================
# Tests
# =============================================================================


def validation_factory_module_tests() -> bool:
    """Comprehensive tests for validation factory."""
    from testing.test_framework import TestSuite

    suite = TestSuite("Validation Factory", "core/validation_factory.py")
    suite.start_suite()

    # Test 1: Required field validator
    def test_required_field_validator() -> None:
        validator = create_required_field_validator(["username", "email"])

        # Valid data
        is_valid, error = validator({"username": "john", "email": "john@example.com"})
        assert is_valid, f"Should be valid: {error}"
        assert not error, "No error message for valid data"

        # Missing field
        is_valid, error = validator({"username": "john"})
        assert not is_valid, "Should be invalid with missing field"
        assert "email" in error, f"Error should mention missing field: {error}"

        # Empty field (when not allowed)
        is_valid, error = validator({"username": "john", "email": ""})
        assert not is_valid, "Should be invalid with empty field"

    suite.run_test(
        "Required field validator",
        test_required_field_validator,
        test_summary="Validates required fields in dictionaries",
        functions_tested="create_required_field_validator",
        expected_outcome="Correctly identifies missing and empty required fields",
    )

    # Test 2: Type validator
    def test_type_validator() -> None:
        validator = create_type_validator({"age": int, "name": str})

        # Valid types
        is_valid, error = validator({"age": 25, "name": "John"})
        assert is_valid, f"Should be valid: {error}"

        # Invalid type
        is_valid, error = validator({"age": "twenty-five", "name": "John"})
        assert not is_valid, "Should be invalid with wrong type"
        assert "age" in error, f"Error should mention field: {error}"

    suite.run_test(
        "Type validator",
        test_type_validator,
        test_summary="Validates field types in dictionaries",
        functions_tested="create_type_validator",
        expected_outcome="Correctly identifies type mismatches",
    )

    # Test 3: Range validator
    def test_range_validator() -> None:
        validator = create_range_validator({"age": (0, 150), "score": (0, 100)})

        # Valid range
        is_valid, error = validator({"age": 25, "score": 85})
        assert is_valid, f"Should be valid: {error}"

        # Below minimum
        is_valid, error = validator({"age": -5})
        assert not is_valid, "Should be invalid below minimum"

        # Above maximum
        is_valid, error = validator({"score": 150})
        assert not is_valid, "Should be invalid above maximum"

    suite.run_test(
        "Range validator",
        test_range_validator,
        test_summary="Validates numeric ranges in dictionaries",
        functions_tested="create_range_validator",
        expected_outcome="Correctly identifies out-of-range values",
    )

    # Test 4: Session manager validator
    def test_session_manager_validator() -> None:
        from unittest.mock import MagicMock

        # Valid session manager
        mock_sm = MagicMock()
        mock_sm.is_sess_valid.return_value = True
        mock_sm.browser_manager.driver = MagicMock()
        mock_sm.db_session = MagicMock()

        assert validate_session_manager(mock_sm, "Test"), "Should be valid"
        assert validate_session_manager(mock_sm, "Test", require_driver=True), "Should be valid with driver"
        assert validate_session_manager(mock_sm, "Test", require_db=True), "Should be valid with db"

        # Invalid session manager
        assert not validate_session_manager(None, "Test"), "Should be invalid with None"

        # Invalid session
        mock_sm.is_sess_valid.return_value = False
        assert not validate_session_manager(mock_sm, "Test"), "Should be invalid with invalid session"

    suite.run_test(
        "Session manager validator",
        test_session_manager_validator,
        test_summary="Validates session manager state",
        functions_tested="validate_session_manager",
        expected_outcome="Correctly validates session manager requirements",
    )

    # Test 5: Person data validator
    def test_person_data_validator() -> None:
        # Valid data
        is_valid, error = validate_person_data({"username": "john"})
        assert is_valid, f"Should be valid: {error}"

        # Missing username
        is_valid, error = validate_person_data({})
        assert not is_valid, "Should be invalid without username"
        assert "username" in error, f"Error should mention username: {error}"

        # Require identifiers
        is_valid, error = validate_person_data({"username": "john"}, require_identifiers=True)
        assert not is_valid, "Should be invalid without identifiers"

        is_valid, error = validate_person_data({"username": "john", "profile_id": "123"}, require_identifiers=True)
        assert is_valid, f"Should be valid with profile_id: {error}"

    suite.run_test(
        "Person data validator",
        test_person_data_validator,
        test_summary="Validates person data for database operations",
        functions_tested="validate_person_data",
        expected_outcome="Correctly validates person data requirements",
    )

    # Test 6: Composite validator
    def test_composite_validator() -> None:
        composite = create_composite_validator(
            create_required_field_validator(["name"]),
            create_type_validator({"age": int}),
        )

        # Valid data
        is_valid, error = composite({"name": "John", "age": 25})
        assert is_valid, f"Should be valid: {error}"

        # First validator fails
        is_valid, error = composite({"age": 25})
        assert not is_valid, "Should fail on required field"
        assert "name" in error, f"Error should be from first validator: {error}"

        # Second validator fails
        is_valid, error = composite({"name": "John", "age": "twenty-five"})
        assert not is_valid, "Should fail on type check"
        assert "age" in error, f"Error should be from second validator: {error}"

    suite.run_test(
        "Composite validator",
        test_composite_validator,
        test_summary="Combines multiple validators into one",
        functions_tested="create_composite_validator",
        expected_outcome="Runs validators in sequence, returns first failure",
    )

    return suite.finish_suite()


# Use centralized test runner
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(validation_factory_module_tests)

if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
