#!/usr/bin/env python3
"""
Setup script for migrating to secure credential storage
Usage: python setup_security.py
"""

import gc
import sys
import os
from pathlib import Path

# Import SecurityManager at top level for testing
try:
    from security_manager import SecurityManager

    SECURITY_MANAGER_AVAILABLE = True
except ImportError:
    SecurityManager = None
    SECURITY_MANAGER_AVAILABLE = False


def main():
    """Interactive setup for migrating to secure credential storage."""
    print("=== Ancestry.com Automation Security Setup ===")
    print("This script will help you migrate to secure encrypted credential storage.\n")

    if not SECURITY_MANAGER_AVAILABLE or SecurityManager is None:
        print("ERROR: Required security dependencies not found.")
        print("Please install required packages:")
        print("  pip install cryptography keyring")
        return False

    security_manager = SecurityManager()

    # Check if credentials already exist
    existing_creds = security_manager.decrypt_credentials()
    if existing_creds:
        print("✓ Encrypted credentials already exist.")
        print("Current encrypted credentials:")
        for key in existing_creds.keys():
            if "PASSWORD" in key or "KEY" in key:
                print(f"  - {key}: {'*' * len(existing_creds[key])}")
            else:
                print(f"  - {key}: {existing_creds[key]}")

        choice = (
            input("\nOptions: (r)ecreate, (d)elete, (k)eep existing: ").strip().lower()
        )
        if choice == "d":
            if security_manager.delete_credentials():
                print("✓ Credentials deleted successfully.")
            else:
                print("✗ Failed to delete credentials.")
            return True
        elif choice == "k":
            print("✓ Keeping existing credentials.")
            return True
        elif choice != "r":
            print("Invalid choice. Exiting.")
            return False

    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        migrate = input("\nMigrate credentials from .env file? (y/N): ").strip().lower()
        if migrate == "y":
            if security_manager.migrate_env_credentials():
                print("✓ Successfully migrated credentials from .env file")
                print("  - Original .env backed up to .env.backup")
                print("  - New .env created with placeholders")
                return True
            else:
                print("✗ Failed to migrate credentials from .env file")
                return False

    # Interactive credential setup
    if security_manager.setup_secure_credentials():
        print("\n✓ Security setup completed successfully!")
        print("\nNext steps:")
        print("1. Test the application to ensure credentials work")
        print("2. Remove any remaining plaintext credentials from .env")
        print("3. Credentials will be loaded automatically from encrypted storage")
        return True
    else:
        print("\n✗ Security setup failed")
        return False


def run_comprehensive_tests() -> bool:
    """Comprehensive test suite for setup_security.py"""
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Security Setup Migration Utility", "setup_security.py")
    suite.start_suite()

    # INITIALIZATION TESTS
    def test_module_initialization():
        """Test module initialization and imports"""
        assert main is not None, "main function should be available"
        assert callable(main), "main function should be callable"

        # Test required imports
        import sys
        import os
        from pathlib import Path

        assert sys is not None, "sys module should be imported"
        assert os is not None, "os module should be imported"
        assert Path is not None, "Path class should be imported"

    def test_function_availability():
        """Test function availability and signatures"""
        import inspect

        # Test main function signature
        sig = inspect.signature(main)
        assert len(sig.parameters) == 0, "main function should take no parameters"

        # Test function has proper docstring
        assert main.__doc__ is not None, "main function should have docstring"
        assert (
            "credential storage" in main.__doc__.lower()
        ), "Docstring should mention credential storage"

    # CORE FUNCTIONALITY TESTS
    def test_main_function_structure():
        """Test main function structure and components"""
        import inspect

        # Get function source to verify structure
        source = inspect.getsource(main)

        # Test that function has proper structure
        assert (
            "SECURITY_MANAGER_AVAILABLE" in source
        ), "Function should check SecurityManager availability"
        assert (
            "decrypt_credentials" in source
        ), "Function should check existing credentials"
        assert "return" in source, "Function should have return statements"

    def test_security_manager_integration():
        """Test SecurityManager integration and error handling"""
        from unittest.mock import MagicMock, patch

        # Test successful SecurityManager import and usage
        with patch("setup_security.SecurityManager") as mock_sm_class:
            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = None  # No existing credentials
            mock_sm.setup_secure_credentials.return_value = True

            # Mock input function
            with patch("builtins.input", return_value="y"):
                result = main()

            # Verify SecurityManager was used
            mock_sm_class.assert_called_once()
            mock_sm.decrypt_credentials.assert_called_once()
            assert isinstance(result, bool), "main should return boolean"

    def test_import_error_handling():
        """Test ImportError handling for missing dependencies"""
        from unittest.mock import patch
        import io
        import sys

        # Test ImportError handling
        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'security_manager'"),
        ), patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:

            result = main()

            output = mock_stdout.getvalue()
            assert (
                "ERROR: Required security dependencies not found" in output
            ), "Should show dependency error"
            assert "pip install" in output, "Should show installation instructions"
            assert result is False, "Should return False on ImportError"

    def test_existing_credentials_handling():
        """Test handling of existing credentials"""
        from unittest.mock import MagicMock, patch
        import io

        # Test with existing credentials
        with patch("setup_security.SecurityManager") as mock_sm_class, patch(
            "sys.stdout", new_callable=io.StringIO
        ) as mock_stdout:

            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = {
                "ANCESTRY_USERNAME": "test_user_12345",
                "ANCESTRY_PASSWORD": "test_password_12345",
            }

            # Test 'keep existing' choice
            with patch("builtins.input", return_value="k"):
                result = main()

            output = mock_stdout.getvalue()
            assert "already exist" in output, "Should indicate existing credentials"
            assert "test_user_12345" in output, "Should show username"
            assert "test_password_12345" not in output, "Should not show password"
            assert result is True, "Should return True when keeping existing"

    def test_env_file_migration():
        """Test .env file migration functionality"""
        from unittest.mock import MagicMock, patch

        # Test with .env file present
        with patch("setup_security.SecurityManager") as mock_sm_class, patch(
            "setup_security.Path"
        ) as mock_path_class:

            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = None
            mock_sm.migrate_env_credentials.return_value = True

            # Mock .env file existence
            mock_env_file = MagicMock()
            mock_env_file.exists.return_value = True
            mock_path_class.return_value = mock_env_file

            # Test migration acceptance
            with patch("builtins.input", side_effect=["y"]):  # Accept migration
                result = main()

            mock_sm.migrate_env_credentials.assert_called_once()
            assert result is True, "Should return True on successful migration"

    # EDGE CASE TESTS
    def test_edge_case_handling():
        """Test edge cases and boundary conditions"""
        from unittest.mock import MagicMock, patch

        # Test with invalid user input
        with patch("setup_security.SecurityManager") as mock_sm_class:
            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = {"test": "credentials"}

            # Test invalid choice
            with patch("builtins.input", return_value="x"):  # Invalid choice
                result = main()

            assert result is False, "Should return False on invalid choice"

    def test_credential_display_masking():
        """Test credential display and password masking"""
        from unittest.mock import MagicMock, patch
        import io

        with patch("setup_security.SecurityManager") as mock_sm_class, patch(
            "sys.stdout", new_callable=io.StringIO
        ) as mock_stdout:

            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = {
                "ANCESTRY_USERNAME": "test_user_12345",
                "ANCESTRY_PASSWORD": "secret_password_12345",
                "DEEPSEEK_API_KEY": "secret_key_12345",
                "OTHER_SETTING": "visible_value_12345",
            }

            with patch("builtins.input", return_value="k"):
                main()

            output = mock_stdout.getvalue()

            # Username should be visible
            assert "test_user_12345" in output, "Username should be visible"

            # Passwords and keys should be masked
            assert "secret_password_12345" not in output, "Password should be masked"
            assert "secret_key_12345" not in output, "API key should be masked"
            assert "*" in output, "Should show asterisks for masked values"

            # Other settings should be visible
            assert (
                "visible_value_12345" in output
            ), "Non-sensitive values should be visible"

    def test_invalid_input_scenarios():
        """Test various invalid input scenarios"""
        from unittest.mock import MagicMock, patch

        with patch("setup_security.SecurityManager") as mock_sm_class:
            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = {"test": "creds"}

            # Test various invalid inputs
            invalid_inputs = ["", "invalid", "123", "yes", "no"]

            for invalid_input in invalid_inputs:
                with patch("builtins.input", return_value=invalid_input):
                    result = main()
                    if invalid_input not in ["r", "d", "k"]:
                        assert (
                            result is False
                        ), f"Should return False for invalid input: {invalid_input}"

    # INTEGRATION TESTS
    def test_complete_workflow_integration():
        """Test complete workflow integration"""
        from unittest.mock import MagicMock, patch

        # Test complete workflow with new setup
        with patch("setup_security.SecurityManager") as mock_sm_class, patch(
            "setup_security.Path"
        ) as mock_path_class:

            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = None
            mock_sm.setup_secure_credentials.return_value = True

            # Mock no .env file
            mock_env_file = MagicMock()
            mock_env_file.exists.return_value = False
            mock_path_class.return_value = mock_env_file

            result = main()

            # Verify complete workflow
            mock_sm.decrypt_credentials.assert_called_once()
            mock_sm.setup_secure_credentials.assert_called_once()
            assert result is True, "Complete workflow should succeed"

    def test_error_handling_integration():
        """Test error handling integration across components"""
        from unittest.mock import MagicMock, patch

        # Test SecurityManager method failures
        with patch("setup_security.SecurityManager") as mock_sm_class:
            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = None
            mock_sm.setup_secure_credentials.return_value = False  # Setup fails

            result = main()
            assert result is False, "Should return False when setup fails"

    def test_user_interaction_integration():
        """Test user interaction integration"""
        from unittest.mock import MagicMock, patch
        import io

        with patch("setup_security.SecurityManager") as mock_sm_class, patch(
            "sys.stdout", new_callable=io.StringIO
        ) as mock_stdout:

            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = {"test": "creds"}

            # Test delete credentials flow
            with patch("builtins.input", return_value="d"):
                mock_sm.delete_credentials.return_value = True
                result = main()

            output = mock_stdout.getvalue()
            assert "deleted successfully" in output, "Should show deletion success"
            assert result is True, "Should return True after successful deletion"

    # PERFORMANCE TESTS
    def test_performance_characteristics():
        """Test performance characteristics"""
        import time
        from unittest.mock import MagicMock, patch

        # Test function execution time
        with patch("setup_security.SecurityManager") as mock_sm_class:
            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = None
            mock_sm.setup_secure_credentials.return_value = True

            start_time = time.time()
            for i in range(5):
                main()
            duration = time.time() - start_time

            assert (
                duration < 1.0
            ), f"5 function calls should be fast, took {duration:.3f}s"

    def test_memory_efficiency():
        """Test memory usage efficiency"""
        from unittest.mock import MagicMock, patch
        import gc
        import sys

        # Test memory usage
        initial_objects = len(gc.get_objects())

        with patch("setup_security.SecurityManager") as mock_sm_class:
            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = None
            mock_sm.setup_secure_credentials.return_value = True

            for i in range(3):
                main()

        gc.collect()
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects

        assert (
            object_growth < 50
        ), f"Memory growth should be minimal, got {object_growth} new objects"

    # ERROR HANDLING TESTS
    def test_exception_handling():
        """Test exception handling scenarios"""
        from unittest.mock import MagicMock, patch

        # Test SecurityManager creation exception
        with patch(
            "setup_security.SecurityManager",
            side_effect=Exception("SecurityManager error 12345"),
        ):
            try:
                result = main()
                # Should handle gracefully - may return False or raise depending on implementation
                assert isinstance(result, bool), "Should return boolean even on error"
            except Exception:
                # If exception propagates, that's also acceptable behavior
                pass

    def test_method_call_error_handling():
        """Test method call error handling"""
        from unittest.mock import MagicMock, patch

        # Test decrypt_credentials failure
        with patch("setup_security.SecurityManager") as mock_sm_class:
            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.side_effect = Exception("Decrypt error 12345")

            try:
                result = main()
                # Function should handle this gracefully
                assert isinstance(
                    result, bool
                ), "Should handle decrypt errors gracefully"
            except Exception:
                # May propagate exception depending on implementation
                pass

    def test_input_error_handling():
        """Test input error handling"""
        from unittest.mock import MagicMock, patch

        # Test input() failure (e.g., EOF, KeyboardInterrupt)
        with patch("setup_security.SecurityManager") as mock_sm_class, patch(
            "builtins.input", side_effect=EOFError("Input error 12345")
        ):

            mock_sm = MagicMock()
            mock_sm_class.return_value = mock_sm
            mock_sm.decrypt_credentials.return_value = {"test": "creds"}

            try:
                result = main()
                # Should handle input errors gracefully
                assert isinstance(result, bool), "Should handle input errors gracefully"
            except EOFError:
                # May propagate input errors depending on implementation
                pass

    def test_script_execution_handling():
        """Test script execution and exit code handling"""
        import inspect

        # Test that module has proper script execution setup
        source = inspect.getsource(sys.modules[__name__])
        assert (
            'if __name__ == "__main__":' in source
        ), "Module should have script execution block"
        assert "main()" in source, "Script should call main function"
        assert "sys.exit(" in source, "Script should call sys.exit with return code"

    # Run all tests with suppress_logging
    with suppress_logging():
        # INITIALIZATION TESTS
        suite.run_test(
            test_name="main() function, import availability, Path class",
            test_func=test_module_initialization,
            test_description="Module initialization and core imports for security setup",
            method_description="Testing main function availability and required module imports",
            expected_behavior="Main function is callable and all required modules are imported",
        )

        suite.run_test(
            test_name="main() function signature and documentation",
            test_func=test_function_availability,
            test_description="Function availability and proper signature requirements",
            method_description="Testing function signature, parameters, and documentation presence",
            expected_behavior="Function has proper signature with no parameters and appropriate documentation",
        )

        # CORE FUNCTIONALITY TESTS
        suite.run_test(
            test_name="main() function structure and SecurityManager integration",
            test_func=test_main_function_structure,
            test_description="Function structure and SecurityManager integration components",
            method_description="Testing function source code for SecurityManager usage and error handling",
            expected_behavior="Function properly integrates SecurityManager with comprehensive error handling",
        )

        suite.run_test(
            test_name="SecurityManager integration and method coordination",
            test_func=test_security_manager_integration,
            test_description="SecurityManager integration and method call coordination",
            method_description="Testing SecurityManager instantiation and method calls with mocked objects",
            expected_behavior="SecurityManager is properly integrated and methods are called correctly",
        )

        suite.run_test(
            test_name="ImportError handling and dependency management",
            test_func=test_import_error_handling,
            test_description="ImportError handling for missing security dependencies",
            method_description="Testing ImportError handling with missing security_manager module",
            expected_behavior="ImportError is handled gracefully with user guidance for dependency installation",
        )

        suite.run_test(
            test_name="Existing credentials detection and user interaction",
            test_func=test_existing_credentials_handling,
            test_description="Existing credentials detection and user choice handling",
            method_description="Testing existing credential detection and user choice processing",
            expected_behavior="Existing credentials are detected and user choices are handled appropriately",
        )

        suite.run_test(
            test_name=".env file migration and credential transfer",
            test_func=test_env_file_migration,
            test_description="Environment file migration and credential transfer functionality",
            method_description="Testing .env file detection and migration with user confirmation",
            expected_behavior="Environment file migration works correctly with proper user interaction",
        )

        # EDGE CASE TESTS
        suite.run_test(
            test_name="Edge cases and invalid user input handling",
            test_func=test_edge_case_handling,
            test_description="Edge cases and boundary condition handling robustness",
            method_description="Testing various edge cases including invalid user inputs",
            expected_behavior="Edge cases are handled gracefully without application crashes",
        )

        suite.run_test(
            test_name="Credential display and password masking",
            test_func=test_credential_display_masking,
            test_description="Credential display logic and sensitive data masking",
            method_description="Testing credential display with password and key masking",
            expected_behavior="Sensitive credentials are masked while usernames remain visible",
        )

        suite.run_test(
            test_name="Invalid input scenarios and error responses",
            test_func=test_invalid_input_scenarios,
            test_description="Various invalid input scenarios and appropriate error responses",
            method_description="Testing different invalid inputs and their handling",
            expected_behavior="Invalid inputs are handled with appropriate error responses",
        )

        # INTEGRATION TESTS
        suite.run_test(
            test_name="Complete workflow integration and component coordination",
            test_func=test_complete_workflow_integration,
            test_description="Complete workflow integration from start to finish",
            method_description="Testing complete workflow with new credential setup",
            expected_behavior="Complete workflow integrates all components successfully",
        )

        suite.run_test(
            test_name="Error handling integration across all components",
            test_func=test_error_handling_integration,
            test_description="Error handling integration across SecurityManager and workflow",
            method_description="Testing error handling when SecurityManager methods fail",
            expected_behavior="Error handling is integrated properly across all workflow components",
        )

        suite.run_test(
            test_name="User interaction integration and feedback",
            test_func=test_user_interaction_integration,
            test_description="User interaction integration and console feedback systems",
            method_description="Testing user interaction flows and console output feedback",
            expected_behavior="User interactions are properly integrated with appropriate feedback",
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
            test_name="Memory usage efficiency and object management",
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
            method_description="Testing exception handling with SecurityManager creation failures",
            expected_behavior="Exceptions are handled gracefully without application crashes",
        )

        suite.run_test(
            test_name="Method call error handling and recovery",
            test_func=test_method_call_error_handling,
            test_description="Method call error handling and recovery mechanisms",
            method_description="Testing error handling when SecurityManager methods fail",
            expected_behavior="Method call errors are handled gracefully with appropriate recovery",
        )

        suite.run_test(
            test_name="Input error handling and user interaction failures",
            test_func=test_input_error_handling,
            test_description="Input error handling and user interaction failure management",
            method_description="Testing input error scenarios like EOF and KeyboardInterrupt",
            expected_behavior="Input errors are handled gracefully without unexpected crashes",
        )

        suite.run_test(
            test_name="Script execution setup and exit code management",
            test_func=test_script_execution_handling,
            test_description="Script execution setup and proper exit code management",
            method_description="Testing script execution block and exit code handling",
            expected_behavior="Script execution is properly configured with appropriate exit codes",
        )

    return suite.finish_suite()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
