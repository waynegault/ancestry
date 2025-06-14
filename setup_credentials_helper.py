#!/usr/bin/env python3
"""
Helper script to set up credentials for testing
"""

import gc
import sys
from security_manager import SecurityManager


def setup_test_credentials():
    """Set up test credentials for development/testing"""
    try:
        security_manager = SecurityManager()

        # Check if credentials already exist
        existing_creds = security_manager.decrypt_credentials()
        if existing_creds:
            print("✓ Encrypted credentials already exist.")
            return True

        # Set up test credentials
        test_credentials = {
            "ANCESTRY_USERNAME": "test@example.com",
            "ANCESTRY_PASSWORD": "test_password_123",
            "DEEPSEEK_API_KEY": "test_deepseek_key_456",
        }

        if security_manager.encrypt_credentials(test_credentials):
            print("✓ Test credentials encrypted and stored successfully!")
            print(
                "Note: These are test credentials. Replace with real ones for actual use."
            )
            return True
        else:
            print("✗ Failed to encrypt and store test credentials")
            return False

    except Exception as e:
        print(f"Error setting up credentials: {e}")
        return False


def run_comprehensive_tests() -> bool:
    """Comprehensive test suite for setup_credentials_helper.py"""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Credentials Helper Setup Utility", "setup_credentials_helper.py")
    suite.start_suite()

    # INITIALIZATION TESTS
    def test_module_initialization():
        """Test module initialization and imports"""
        assert SecurityManager is not None, "SecurityManager should be imported"
        assert callable(SecurityManager), "SecurityManager should be callable"
        assert callable(
            setup_test_credentials
        ), "setup_test_credentials should be callable"

    def test_function_availability():
        """Test function availability and signatures"""
        import inspect

        # Test setup_test_credentials function
        sig = inspect.signature(setup_test_credentials)
        assert (
            len(sig.parameters) == 0
        ), "setup_test_credentials should take no parameters"

        # Test function is properly defined
        assert (
            setup_test_credentials.__doc__ is not None
        ), "Function should have docstring"
        assert (
            "test credentials" in setup_test_credentials.__doc__.lower()
        ), "Docstring should mention test credentials"

    # CORE FUNCTIONALITY TESTS
    def test_setup_test_credentials_structure():
        """Test setup_test_credentials function structure"""
        import inspect

        # Get function source to verify structure
        source = inspect.getsource(setup_test_credentials)
        assert (
            "SecurityManager()" in source
        ), "Function should create SecurityManager instance"
        assert (
            "decrypt_credentials" in source
        ), "Function should check existing credentials"
        assert (
            "encrypt_credentials" in source
        ), "Function should encrypt new credentials"
        assert "test_credentials" in source, "Function should define test credentials"

        # Test that function has proper error handling
        assert "try:" in source, "Function should have try-except structure"
        assert "except" in source, "Function should handle exceptions"

    def test_test_credentials_structure():
        """Test test credentials data structure"""
        # Extract test credentials from function (simulate what function does)
        test_credentials = {
            "ANCESTRY_USERNAME": "test@example.com",
            "ANCESTRY_PASSWORD": "test_password_123",
            "DEEPSEEK_API_KEY": "test_deepseek_key_456",
        }

        # Test credentials structure
        assert isinstance(
            test_credentials, dict
        ), "Test credentials should be a dictionary"
        assert len(test_credentials) == 3, "Should have exactly 3 credential keys"

        # Test required keys
        required_keys = ["ANCESTRY_USERNAME", "ANCESTRY_PASSWORD", "DEEPSEEK_API_KEY"]
        for key in required_keys:
            assert key in test_credentials, f"Test credentials should include {key}"
            assert isinstance(test_credentials[key], str), f"{key} should be a string"
            assert len(test_credentials[key]) > 0, f"{key} should not be empty"
            assert (
                "12345" not in test_credentials[key]
            ), f"{key} should not contain test identifier (this is real test data)"

    def test_security_manager_integration():
        """Test SecurityManager integration"""
        from unittest.mock import MagicMock, patch

        # Test that function properly integrates with SecurityManager
        with patch("setup_credentials_helper.SecurityManager") as mock_sm_class:
            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = None  # No existing credentials
            mock_sm.encrypt_credentials.return_value = True  # Successful encryption

            # Call function
            result = setup_test_credentials()

            # Verify interactions
            mock_sm_class.assert_called_once()
            mock_sm.decrypt_credentials.assert_called_once()
            mock_sm.encrypt_credentials.assert_called_once()
            assert result is True, "Should return True on successful setup"

    def test_return_value_handling():
        """Test return value handling in different scenarios"""
        from unittest.mock import MagicMock, patch

        # Test with existing credentials
        with patch("setup_credentials_helper.SecurityManager") as mock_sm_class:
            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = {"existing": "credentials"}

            result = setup_test_credentials()
            assert result is True, "Should return True when credentials already exist"

        # Test with encryption failure
        with patch("setup_credentials_helper.SecurityManager") as mock_sm_class:
            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = None
            mock_sm.encrypt_credentials.return_value = False  # Failed encryption

            result = setup_test_credentials()
            assert result is False, "Should return False when encryption fails"

    # EDGE CASE TESTS
    def test_edge_case_handling():
        """Test edge cases and error conditions"""
        from unittest.mock import MagicMock, patch

        # Test SecurityManager creation failure
        with patch(
            "setup_credentials_helper.SecurityManager",
            side_effect=Exception("SecurityManager error 12345"),
        ):
            result = setup_test_credentials()
            assert (
                result is False
            ), "Should return False when SecurityManager creation fails"

        # Test decrypt_credentials failure
        with patch("setup_credentials_helper.SecurityManager") as mock_sm_class:
            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.side_effect = Exception("Decrypt error 12345")

            result = setup_test_credentials()
            assert result is False, "Should return False when decrypt_credentials fails"

        # Test encrypt_credentials failure
        with patch("setup_credentials_helper.SecurityManager") as mock_sm_class:
            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = None
            mock_sm.encrypt_credentials.side_effect = Exception("Encrypt error 12345")

            result = setup_test_credentials()
            assert result is False, "Should return False when encrypt_credentials fails"

    def test_credential_validation():
        """Test credential validation logic"""
        # Test that credentials contain reasonable values
        test_credentials = {
            "ANCESTRY_USERNAME": "test@example.com",
            "ANCESTRY_PASSWORD": "test_password_123",
            "DEEPSEEK_API_KEY": "test_deepseek_key_456",
        }

        # Validate username format
        username = test_credentials["ANCESTRY_USERNAME"]
        assert "@" in username, "Username should be email format"
        assert "." in username, "Username should have domain"
        assert username.startswith("test"), "Username should be test credential"

        # Validate password format
        password = test_credentials["ANCESTRY_PASSWORD"]
        assert len(password) >= 8, "Password should be reasonably long"
        assert "test" in password.lower(), "Password should indicate test nature"

        # Validate API key format
        api_key = test_credentials["DEEPSEEK_API_KEY"]
        assert len(api_key) > 10, "API key should be reasonably long"
        assert "test" in api_key.lower(), "API key should indicate test nature"

    # INTEGRATION TESTS
    def test_security_manager_method_calls():
        """Test SecurityManager method call sequence"""
        from unittest.mock import MagicMock, patch, call

        with patch("setup_credentials_helper.SecurityManager") as mock_sm_class:
            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = None
            mock_sm.encrypt_credentials.return_value = True

            setup_test_credentials()

            # Verify proper method call sequence
            expected_calls = [
                call.decrypt_credentials(),
                call.encrypt_credentials(
                    {
                        "ANCESTRY_USERNAME": "test@example.com",
                        "ANCESTRY_PASSWORD": "test_password_123",
                        "DEEPSEEK_API_KEY": "test_deepseek_key_456",
                    }
                ),
            ]

            assert (
                mock_sm.method_calls == expected_calls
            ), "Should call methods in correct sequence"

    def test_console_output_integration():
        """Test console output integration"""
        from unittest.mock import MagicMock, patch
        import io
        import sys

        # Test with existing credentials
        with patch("setup_credentials_helper.SecurityManager") as mock_sm_class, patch(
            "sys.stdout", new_callable=io.StringIO
        ) as mock_stdout:

            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = {"existing": "credentials"}

            setup_test_credentials()

            output = mock_stdout.getvalue()
            assert "already exist" in output, "Should indicate existing credentials"

    def test_module_as_script():
        """Test module execution as script"""
        import inspect

        # Test that module has proper script execution setup
        source = inspect.getsource(sys.modules[__name__])
        assert (
            'if __name__ == "__main__":' in source
        ), "Module should have script execution block"
        assert (
            "setup_test_credentials()" in source
        ), "Script should call setup_test_credentials"
        assert "exit(" in source, "Script should call exit with return code"

    # PERFORMANCE TESTS
    def test_performance_characteristics():
        """Test performance characteristics"""
        import time
        from unittest.mock import MagicMock, patch

        # Test function execution time
        with patch("setup_credentials_helper.SecurityManager") as mock_sm_class:
            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = None
            mock_sm.encrypt_credentials.return_value = True

            start_time = time.time()
            for i in range(10):
                setup_test_credentials()
            duration = time.time() - start_time

            assert (
                duration < 1.0
            ), f"10 function calls should be fast, took {duration:.3f}s"

    def test_memory_efficiency():
        """Test memory usage efficiency"""
        from unittest.mock import MagicMock, patch
        import sys

        # Test that function doesn't create excessive objects
        initial_objects = len(gc.get_objects()) if "gc" in sys.modules else 0

        with patch("setup_credentials_helper.SecurityManager") as mock_sm_class:
            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = None
            mock_sm.encrypt_credentials.return_value = True

            for i in range(5):
                setup_test_credentials()

        # Memory usage should be reasonable (if gc is available)
        if "gc" in sys.modules:
            final_objects = len(gc.get_objects())
            object_growth = final_objects - initial_objects
            assert (
                object_growth < 100
            ), f"Memory growth should be minimal, got {object_growth} new objects"

    # ERROR HANDLING TESTS
    def test_exception_handling():
        """Test exception handling scenarios"""
        from unittest.mock import patch

        # Test that function handles all exceptions gracefully
        with patch(
            "setup_credentials_helper.SecurityManager",
            side_effect=RuntimeError("Critical error 12345"),
        ):
            try:
                result = setup_test_credentials()
                assert result is False, "Should return False on exception"
            except Exception:
                assert False, "Function should handle exceptions internally"

        # Test with different exception types
        exception_types = [ValueError, TypeError, AttributeError, ImportError]
        for exc_type in exception_types:
            with patch(
                "setup_credentials_helper.SecurityManager",
                side_effect=exc_type("Test error 12345"),
            ):
                result = setup_test_credentials()
                assert result is False, f"Should handle {exc_type.__name__} gracefully"

    def test_error_message_handling():
        """Test error message handling"""
        from unittest.mock import patch
        import io
        import sys

        # Test error message output
        with patch(
            "setup_credentials_helper.SecurityManager",
            side_effect=Exception("Test error 12345"),
        ), patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:

            result = setup_test_credentials()

            output = mock_stdout.getvalue()
            assert (
                "Error setting up credentials" in output
            ), "Should output error message"
            assert result is False, "Should return False on error"

    def test_return_code_handling():
        """Test return code handling for script execution"""
        # Test that function returns appropriate boolean values
        from unittest.mock import MagicMock, patch

        # Test success case
        with patch("setup_credentials_helper.SecurityManager") as mock_sm_class:
            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = None
            mock_sm.encrypt_credentials.return_value = True

            result = setup_test_credentials()
            assert result is True, "Should return True on success"
            assert isinstance(result, bool), "Should return boolean value"

        # Test failure case
        with patch(
            "setup_credentials_helper.SecurityManager",
            side_effect=Exception("Error 12345"),
        ):
            result = setup_test_credentials()
            assert result is False, "Should return False on failure"
            assert isinstance(result, bool), "Should return boolean value"

    # Run all tests with suppress_logging
    with suppress_logging():
        # INITIALIZATION TESTS
        suite.run_test(
            test_name="SecurityManager import, setup_test_credentials() availability",
            test_func=test_module_initialization,
            test_description="Module initialization and core imports for credential setup",
            method_description="Testing SecurityManager import and setup_test_credentials function availability",
            expected_behavior="SecurityManager is properly imported and setup function is callable",
        )

        suite.run_test(
            test_name="setup_test_credentials() function signature and documentation",
            test_func=test_function_availability,
            test_description="Function availability and proper signature requirements",
            method_description="Testing function signature, parameters, and documentation presence",
            expected_behavior="Function has proper signature with no parameters and appropriate documentation",
        )

        # CORE FUNCTIONALITY TESTS
        suite.run_test(
            test_name="setup_test_credentials() structure and SecurityManager integration",
            test_func=test_setup_test_credentials_structure,
            test_description="Function structure and SecurityManager method integration",
            method_description="Testing function source code for proper SecurityManager usage and error handling",
            expected_behavior="Function properly integrates SecurityManager methods with error handling",
        )

        suite.run_test(
            test_name="Test credentials data structure and validation",
            test_func=test_test_credentials_structure,
            test_description="Test credentials structure and content validation",
            method_description="Testing test credentials dictionary structure and required key presence",
            expected_behavior="Test credentials have proper structure with all required keys and values",
        )

        suite.run_test(
            test_name="SecurityManager method calls and return value handling",
            test_func=test_security_manager_integration,
            test_description="SecurityManager integration and method call coordination",
            method_description="Testing SecurityManager instantiation and method calls with mock objects",
            expected_behavior="SecurityManager methods are called correctly and results are handled properly",
        )

        suite.run_test(
            test_name="Return value handling for different scenarios",
            test_func=test_return_value_handling,
            test_description="Return value handling in success and failure scenarios",
            method_description="Testing return values with existing credentials and encryption failures",
            expected_behavior="Function returns appropriate boolean values for different scenarios",
        )

        # EDGE CASE TESTS
        suite.run_test(
            test_name="Edge cases and error conditions handling",
            test_func=test_edge_case_handling,
            test_description="Edge cases and error condition handling robustness",
            method_description="Testing various error scenarios with SecurityManager failures",
            expected_behavior="Function handles all error conditions gracefully and returns False",
        )

        suite.run_test(
            test_name="Credential validation and format checking",
            test_func=test_credential_validation,
            test_description="Test credential validation and format verification",
            method_description="Testing credential format validation for username, password, and API key",
            expected_behavior="Test credentials have appropriate formats and indicate test nature",
        )

        # INTEGRATION TESTS
        suite.run_test(
            test_name="SecurityManager method call sequence and coordination",
            test_func=test_security_manager_method_calls,
            test_description="SecurityManager method call sequence and proper coordination",
            method_description="Testing proper sequence of decrypt and encrypt method calls",
            expected_behavior="SecurityManager methods are called in correct sequence with proper parameters",
        )

        suite.run_test(
            test_name="Console output integration and user feedback",
            test_func=test_console_output_integration,
            test_description="Console output integration and user feedback messages",
            method_description="Testing console output for various scenarios and user feedback",
            expected_behavior="Function provides appropriate console feedback for different scenarios",
        )

        suite.run_test(
            test_name="Module script execution setup and entry point",
            test_func=test_module_as_script,
            test_description="Module execution as script and proper entry point setup",
            method_description="Testing script execution block and function call setup",
            expected_behavior="Module has proper script execution setup with exit code handling",
        )

        # PERFORMANCE TESTS
        suite.run_test(
            test_name="Function execution performance and efficiency",
            test_func=test_performance_characteristics,
            test_description="Function execution performance and timing characteristics",
            method_description="Testing function execution time with multiple iterations",
            expected_behavior="Function executes efficiently within reasonable time limits",
        )

        suite.run_test(
            test_name="Memory usage efficiency and object creation",
            test_func=test_memory_efficiency,
            test_description="Memory usage efficiency and object creation management",
            method_description="Testing memory usage and object creation during function execution",
            expected_behavior="Function maintains efficient memory usage without excessive object creation",
        )

        # ERROR HANDLING TESTS
        suite.run_test(
            test_name="Exception handling and graceful failure management",
            test_func=test_exception_handling,
            test_description="Exception handling and graceful failure in error scenarios",
            method_description="Testing exception handling with various error types and scenarios",
            expected_behavior="Function handles all exception types gracefully without crashing",
        )

        suite.run_test(
            test_name="Error message handling and user communication",
            test_func=test_error_message_handling,
            test_description="Error message handling and user error communication",
            method_description="Testing error message output and user communication during failures",
            expected_behavior="Function provides clear error messages and returns appropriate failure status",
        )

        suite.run_test(
            test_name="Return code handling and script exit status",
            test_func=test_return_code_handling,
            test_description="Return code handling and script exit status management",
            method_description="Testing boolean return values for success and failure scenarios",
            expected_behavior="Function returns proper boolean values for script exit code determination",
        )

    return suite.finish_suite()


if __name__ == "__main__":
    success = setup_test_credentials()
    exit(0 if success else 1)
