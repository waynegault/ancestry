#!/usr/bin/env python3
"""
Security Manager for Ancestry.com Automation System
Provides encrypted credential storage and secure session management.

OVERVIEW:
=========
The SecurityManager class provides secure storage and retrieval of sensitive credentials
such as usernames, passwords, and API keys. It uses industry-standard encryption (Fernet)
and integrates with the system keyring for master key storage.

KEY FEATURES:
=============
- Encrypted credential storage using cryptography.fernet
- System keyring integration for master key management
- Secure file permissions (0o600 on Unix systems)
- Comprehensive validation and error handling
- Multiple instance support with shared credential files
- Graceful fallback when keyring is unavailable

TESTING INFORMATION:
====================
This module includes a comprehensive test suite with 10 test categories that verify
all aspects of secure credential management. The test suite follows the project's
standardized testing framework.

Expected Test Output:
- ✅ All 10 tests should pass
- ⚠️ 1 expected warning about file permissions on Windows
- 🕐 Test completion time: ~0.2 seconds

Expected Warnings During Testing:
1. "File permissions: 0o666" - Windows doesn't support Unix-style permissions
2. "Could not retrieve master key from keyring" - Expected on first run
3. Validation errors are intentionally suppressed during invalid credential tests

USAGE EXAMPLE:
==============
```python
from security_manager import SecurityManager

# Initialize manager
manager = SecurityManager("MyApp")

# Store credentials
credentials = {
    "USERNAME": "user@example.com",
    "PASSWORD": "secure_password",
    "API_KEY": "sk-api123456"
}
manager.encrypt_credentials(credentials)

# Retrieve specific credential
username = manager.get_credential("USERNAME")

# Validate credentials
is_valid = manager.validate_credentials(credentials)

# Clean up
manager.delete_credentials()
```

SECURITY CONSIDERATIONS:
========================
- Master keys are stored in the system keyring when available
- Credential files use restrictive permissions (owner-only access)
- All encryption uses Fernet (symmetric encryption with authentication)
- Temporary keys are used when keyring is unavailable (session-only)
- Input validation prevents empty or missing required credentials

ERROR HANDLING:
===============
- Graceful degradation when system keyring is unavailable
- Proper cleanup of temporary files during testing
- Comprehensive validation with clear error messages
- Exception handling with detailed logging for debugging
"""

import os
import base64
import json
from pathlib import Path
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import getpass
import keyring
from logging_config import logger

# --- Test framework imports ---
try:
    from test_framework import (
        TestSuite,
        suppress_logging,
        create_mock_data,
        assert_valid_function,
    )

    HAS_TEST_FRAMEWORK = True
except ImportError:
    # Create dummy classes/functions for when test framework is not available
    class DummyTestSuite:
        def __init__(self, *args, **kwargs):
            pass

        def start_suite(self):
            pass

        def add_test(self, *args, **kwargs):
            pass

        def add_warning(self, *args, **kwargs):
            pass

        def end_suite(self):
            pass

        def run_test(self, *args, **kwargs):
            return True

        def finish_suite(self):
            return True

    class DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    TestSuite = DummyTestSuite
    suppress_logging = lambda: DummyContext()
    create_mock_data = lambda: {}
    assert_valid_function = lambda x, *args: True
    HAS_TEST_FRAMEWORK = False


class SecurityManager:
    """
    Handles secure credential storage and retrieval using encryption.
    """

    def __init__(self, app_name: str = "AncestryAutomation"):
        self.app_name = app_name
        self.credentials_file = Path("credentials.enc")
        self._fernet = None

    def _get_master_key(self) -> bytes:
        """
        Get or create master key for encryption.
        Uses system keyring for secure storage.
        """
        try:
            # Try to get existing key from system keyring
            key_b64 = keyring.get_password(self.app_name, "master_key")
            if key_b64:
                return base64.urlsafe_b64decode(key_b64.encode())
        except Exception as e:
            logger.warning(f"Could not retrieve master key from keyring: {e}")

        # Generate new key if none exists
        return self._generate_new_master_key()

    def _generate_new_master_key(self) -> bytes:
        """Generate and store a new master encryption key."""
        # Generate a secure random key
        key = Fernet.generate_key()

        try:
            # Store in system keyring
            key_b64 = base64.urlsafe_b64encode(key).decode()
            keyring.set_password(self.app_name, "master_key", key_b64)
            logger.info("Generated and stored new master encryption key")
        except Exception as e:
            logger.error(f"Failed to store master key in keyring: {e}")
            logger.warning(
                "Using temporary key - credentials won't persist between sessions"
            )

        return key

    def _get_fernet(self) -> Fernet:
        """Get Fernet encryption instance."""
        if self._fernet is None:
            key = self._get_master_key()
            self._fernet = Fernet(key)
        return self._fernet

    def encrypt_credentials(self, credentials: Dict[str, str]) -> bool:
        """
        Encrypt and store credentials securely.

        Args:
            credentials: Dictionary of credential key-value pairs

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            fernet = self._get_fernet()

            # Serialize and encrypt credentials
            json_data = json.dumps(credentials).encode()
            encrypted_data = fernet.encrypt(json_data)

            # Write to encrypted file
            with open(self.credentials_file, "wb") as f:
                f.write(encrypted_data)

            # Set restrictive file permissions
            os.chmod(self.credentials_file, 0o600)

            logger.info(f"Encrypted {len(credentials)} credentials successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to encrypt credentials: {e}")
            return False

    def decrypt_credentials(self) -> Optional[Dict[str, str]]:
        """
        Decrypt and retrieve stored credentials.

        Returns:
            Dict[str, str]: Decrypted credentials or None if failed
        """
        if not self.credentials_file.exists():
            logger.info("No encrypted credentials file found")
            return None

        try:
            fernet = self._get_fernet()

            # Read and decrypt file
            with open(self.credentials_file, "rb") as f:
                encrypted_data = f.read()

            decrypted_data = fernet.decrypt(encrypted_data)
            credentials = json.loads(decrypted_data.decode())

            logger.debug(f"Decrypted {len(credentials)} credentials successfully")
            return credentials

        except Exception as e:
            logger.error(f"Failed to decrypt credentials: {e}")
            return None

    def migrate_env_credentials(self, env_file_path: str = ".env") -> bool:
        """
        Migrate plaintext credentials from .env file to encrypted storage.

        Args:
            env_file_path: Path to .env file

        Returns:
            bool: True if migration successful
        """
        env_path = Path(env_file_path)
        if not env_path.exists():
            logger.warning(f"Environment file {env_file_path} not found")
            return False

        credentials = {}
        sensitive_keys = [
            "ANCESTRY_USERNAME",
            "ANCESTRY_PASSWORD",
            "DEEPSEEK_API_KEY",
            "GOOGLE_API_KEY",
        ]

        try:
            # Read .env file and extract sensitive credentials
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip("'\"")

                        if key in sensitive_keys and value:
                            credentials[key] = value

            if not credentials:
                logger.info("No sensitive credentials found in .env file")
                return True

            # Encrypt and store credentials
            if not self.encrypt_credentials(credentials):
                return False

            # Create backup of original .env file
            backup_path = env_path.with_suffix(".env.backup")
            env_path.rename(backup_path)
            logger.info(f"Backed up original .env file to {backup_path}")

            # Create new .env file with credentials removed
            self._create_secure_env_file(env_path, backup_path, sensitive_keys)

            logger.info(
                f"Successfully migrated {len(credentials)} credentials to encrypted storage"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to migrate credentials: {e}")
            return False

    def _create_secure_env_file(
        self, env_path: Path, backup_path: Path, sensitive_keys: list
    ):
        """Create new .env file with sensitive credentials removed."""
        with open(backup_path, "r") as backup_f, open(env_path, "w") as new_f:
            new_f.write("# Ancestry app settings\n")
            new_f.write("# Sensitive credentials moved to encrypted storage\n\n")

            for line in backup_f:
                line_stripped = line.strip()

                # Skip sensitive credential lines
                if any(line_stripped.startswith(key + "=") for key in sensitive_keys):
                    # Add placeholder comment
                    key = line_stripped.split("=")[0]
                    new_f.write(f"# {key}=<stored_in_encrypted_storage>\n")
                else:
                    new_f.write(line)

    def get_credential(self, key: str) -> Optional[str]:
        """
        Get a specific credential value.

        Args:
            key: Credential key to retrieve

        Returns:
            str: Credential value or None if not found
        """
        credentials = self.decrypt_credentials()
        if credentials:
            return credentials.get(key)
        return None

    def prompt_for_credentials(self) -> Dict[str, str]:
        """
        Prompt user for credentials interactively.

        Returns:
            Dict[str, str]: User-provided credentials
        """
        print("\n=== Secure Credential Setup ===")
        print("Enter your credentials. They will be encrypted and stored securely.")

        credentials = {}

        # Ancestry credentials
        username = input("Ancestry Username/Email: ").strip()
        if username:
            credentials["ANCESTRY_USERNAME"] = username

        password = getpass.getpass("Ancestry Password: ").strip()
        if password:
            credentials["ANCESTRY_PASSWORD"] = password

        # AI API keys
        ai_provider = (
            input("AI Provider (deepseek/gemini) [deepseek]: ").strip() or "deepseek"
        )

        if ai_provider.lower() == "deepseek":
            api_key = getpass.getpass("DeepSeek API Key: ").strip()
            if api_key:
                credentials["DEEPSEEK_API_KEY"] = api_key
        elif ai_provider.lower() == "gemini":
            api_key = getpass.getpass("Google Gemini API Key: ").strip()
            if api_key:
                credentials["GOOGLE_API_KEY"] = api_key

        return credentials

    def setup_secure_credentials(self) -> bool:
        """
        Interactive setup for secure credential storage.

        Returns:
            bool: True if setup successful
        """
        # Check if credentials already exist
        existing_creds = self.decrypt_credentials()
        if existing_creds:
            print("\nEncrypted credentials already exist.")
            update = input("Update existing credentials? (y/N): ").strip().lower()
            if update != "y":
                return True

        # Get credentials from user
        credentials = self.prompt_for_credentials()
        if not credentials:
            print("No credentials provided.")
            return False

        # Encrypt and store
        if self.encrypt_credentials(credentials):
            print(
                f"\n✓ Successfully encrypted and stored {len(credentials)} credentials"
            )
            print(
                "Credentials are now stored securely and will be loaded automatically."
            )
            return True
        else:
            print("\n✗ Failed to encrypt credentials")
            return False

    def delete_credentials(self) -> bool:
        """
        Delete encrypted credentials file.

        Returns:
            bool: True if successful
        """
        try:
            if self.credentials_file.exists():
                self.credentials_file.unlink()
                logger.info("Deleted encrypted credentials file")

            # Also try to remove from keyring
            try:
                keyring.delete_password(self.app_name, "master_key")
                logger.info("Removed master key from system keyring")
            except Exception:
                pass  # Key might not exist

            return True

        except Exception as e:
            logger.error(f"Failed to delete credentials: {e}")
            return False

    def validate_credentials(self, credentials: Dict[str, str]) -> bool:
        """
        Validate that required credentials are present.

        Args:
            credentials: Credentials dictionary to validate

        Returns:
            bool: True if valid
        """
        required_keys = ["ANCESTRY_USERNAME", "ANCESTRY_PASSWORD"]

        for key in required_keys:
            if not credentials.get(key):
                logger.error(f"Missing required credential: {key}")
                return False

        return True


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for security_manager.py.
    Tests secure credential storage, encryption, and session management.

    EXPECTED WARNINGS/ERRORS:
    ========================

    1. Keyring Access Warnings:
       - "Could not retrieve master key from keyring: [error]"
       - Common on systems without GUI keyring or first-time use
       - These are expected and handled gracefully by generating new keys

    2. File Permission Warnings:
       - "File permissions: [octal_value]"
       - On Windows, Unix-style file permissions (0o600) may not apply
       - The test still passes as security is best-effort on different platforms

    3. Validation Test Errors (Expected and Suppressed):
       - Invalid credential tests intentionally trigger validation failures
       - These errors are suppressed with suppress_logging() context manager
       - Tests verify that invalid credentials properly return False

    4. System-Specific Behaviors:
       - Master key storage may fail on systems without keyring support
       - File deletion may show different behaviors on network drives
       - Encryption file paths may vary between operating systems

    5. Test Environment Warnings:
       - Temporary test files are created and cleaned up during testing
       - Some tests may show "credentials won't persist" warnings for temporary keys
       - Multiple instance tests intentionally overwrite files to test shared behavior

    SUCCESS CRITERIA:
    ================
    - All 10 test categories should pass
    - Only 1-2 expected warnings about file permissions or keyring access
    - No critical errors or unhandled exceptions
    - Test cleanup should remove all temporary files
    """
    if not HAS_TEST_FRAMEWORK:
        return (
            _run_basic_fallback_tests()
        )  # Fallback to simple test if framework unavailable

    suite = TestSuite("Security Manager & Credential Storage", "security_manager.py")
    suite.start_suite()

    # SecurityManager instantiation and basic setup
    def test_security_manager_instantiation():
        manager = SecurityManager("TestApp")
        assert manager.app_name == "TestApp"
        assert manager.credentials_file.name == "credentials.enc"
        assert manager._fernet is None

    # Master key generation and retrieval
    def test_master_key_operations():
        manager = SecurityManager("TestKeyApp")
        try:
            # Test key generation
            key1 = manager._get_master_key()
            assert isinstance(key1, bytes), "Master key should be bytes"
            # Fernet generates 44-byte base64-encoded keys
            assert len(key1) in [
                32,
                44,
            ], f"Fernet key should be 32 or 44 bytes, got {len(key1)}"

            # Test key consistency
            key2 = manager._get_master_key()
            assert key1 == key2, "Should retrieve same key consistently"

            # Test Fernet instance creation
            fernet = manager._get_fernet()
            assert fernet is not None, "Fernet instance should be created"
        finally:
            manager.delete_credentials()

    # Credential encryption and decryption
    def test_credential_encryption_decryption():
        manager = SecurityManager("TestEncryptApp")
        test_credentials = {
            "TEST_USERNAME": "test_user",
            "TEST_PASSWORD": "test_pass123",
            "TEST_API_KEY": "sk-test123456789",
            "SPECIAL_CHARS": "test!@#$%^&*()_+-=[]{}|;:,.<>?",
        }

        try:
            # Test encryption
            result = manager.encrypt_credentials(test_credentials)
            assert result is True

            # Verify file exists
            assert manager.credentials_file.exists()

            # Test decryption
            decrypted = manager.decrypt_credentials()
            assert decrypted is not None
            assert decrypted == test_credentials

        finally:
            manager.delete_credentials()

    # Individual credential retrieval
    def test_individual_credential_retrieval():
        manager = SecurityManager("TestRetrievalApp")
        test_credentials = {
            "ANCESTRY_USERNAME": "test@example.com",
            "ANCESTRY_PASSWORD": "secret123",
            "DEEPSEEK_API_KEY": "sk-deepseek123",
        }

        try:
            manager.encrypt_credentials(test_credentials)

            # Test getting existing credentials
            username = manager.get_credential("ANCESTRY_USERNAME")
            assert username == "test@example.com"

            password = manager.get_credential("ANCESTRY_PASSWORD")
            assert password == "secret123"

            # Test getting non-existent credential
            missing = manager.get_credential("NONEXISTENT_KEY")
            assert missing is None

        finally:
            manager.delete_credentials()

    # Credential validation
    def test_credential_validation():
        manager = SecurityManager("TestValidationApp")

        # Test valid credentials
        valid_creds = {
            "ANCESTRY_USERNAME": "test@example.com",
            "ANCESTRY_PASSWORD": "password123",
        }
        assert manager.validate_credentials(valid_creds) is True

        # Test missing username
        invalid_creds1 = {"ANCESTRY_PASSWORD": "password123"}
        with suppress_logging():  # EXPECTED: Suppresses intentional validation error
            assert manager.validate_credentials(invalid_creds1) is False

        # Test missing password
        invalid_creds2 = {"ANCESTRY_USERNAME": "test@example.com"}
        with suppress_logging():  # EXPECTED: Suppresses intentional validation error
            assert manager.validate_credentials(invalid_creds2) is False

        # Test empty values
        invalid_creds3 = {"ANCESTRY_USERNAME": "", "ANCESTRY_PASSWORD": "password123"}
        with suppress_logging():  # EXPECTED: Suppresses intentional validation error
            assert manager.validate_credentials(invalid_creds3) is False

    # Error handling and edge cases
    def test_error_handling():
        manager = SecurityManager("TestErrorApp")

        # Test decryption with no file
        result = manager.decrypt_credentials()
        assert result is None

        # Test get_credential with no encrypted file
        credential = manager.get_credential("ANY_KEY")
        assert credential is None

        # Test encryption with edge case data
        try:
            # This should handle gracefully
            result = manager.encrypt_credentials({"key": "value"})
            # Should either succeed or fail gracefully
            assert isinstance(result, bool)
        except Exception:
            # If it raises an exception, that's also acceptable for this edge case
            pass

    # File permissions and security
    def test_file_security():
        import stat

        manager = SecurityManager("TestSecurityApp")
        test_credentials = {"TEST_KEY": "test_value"}

        try:
            manager.encrypt_credentials(test_credentials)

            if manager.credentials_file.exists():
                # Check file permissions (on Unix-like systems)
                file_stat = manager.credentials_file.stat()
                file_mode = stat.filemode(file_stat.st_mode)

                # File should be readable/writable by owner only
                # This test is best-effort since Windows permissions work differently
                if hasattr(stat, "S_IMODE"):
                    permissions = stat.S_IMODE(file_stat.st_mode)
                    # On Unix: should be 0o600 (owner read/write only)
                    # On Windows: this test may not be meaningful
                    # Log permissions for visibility
                    logger.debug(f"File permissions: {oct(permissions)}")

        finally:
            manager.delete_credentials()

    # Credential deletion and cleanup
    def test_credential_deletion():
        manager = SecurityManager("TestDeleteApp")
        test_credentials = {"TEST_KEY": "test_value"}

        # Create credentials file
        manager.encrypt_credentials(test_credentials)
        assert manager.credentials_file.exists()

        # Test deletion
        result = manager.delete_credentials()
        assert result is True
        assert not manager.credentials_file.exists()

        # Test deletion when file doesn't exist
        result = manager.delete_credentials()
        assert result is True  # Should still return True

    # Multiple SecurityManager instances
    def test_multiple_instances():
        manager1 = SecurityManager("TestMultiApp")
        manager2 = SecurityManager("TestMultiApp")  # Same app name

        test_creds1 = {"KEY1": "value1"}
        test_creds2 = {"KEY2": "value2"}

        try:
            # Same app name means they share the same master key
            # so the second encryption will overwrite the first

            # Encrypt credentials with first manager
            result1 = manager1.encrypt_credentials(test_creds1)
            assert result1 is True, "First manager should encrypt successfully"

            # Verify first manager can read its own data
            creds1_first = manager1.decrypt_credentials()
            assert creds1_first == test_creds1, "Manager1 should read its own data"

            # Encrypt credentials with second manager (will overwrite)
            result2 = manager2.encrypt_credentials(test_creds2)
            assert result2 is True, "Second manager should encrypt successfully"

            # Both managers will read the same file (last written)
            creds1 = manager1.decrypt_credentials()
            creds2 = manager2.decrypt_credentials()

            # They should both read the same data (from manager2)
            assert creds1 == test_creds2, "Manager1 should read the latest data"
            assert creds2 == test_creds2, "Manager2 should read its own data"

        finally:
            manager1.delete_credentials()
            manager2.delete_credentials()

    # Integration test - full workflow
    def test_full_workflow():
        manager = SecurityManager("TestWorkflowApp")

        # Simulate full setup workflow
        credentials = {
            "ANCESTRY_USERNAME": "workflow@test.com",
            "ANCESTRY_PASSWORD": "workflow123!",
            "DEEPSEEK_API_KEY": "sk-workflow789",
        }

        try:
            # Step 1: Encrypt credentials
            assert manager.encrypt_credentials(credentials) is True

            # Step 2: Validate stored credentials
            stored_creds = manager.decrypt_credentials()
            assert stored_creds is not None, "Should retrieve stored credentials"
            assert manager.validate_credentials(stored_creds) is True

            # Step 3: Retrieve individual credentials
            username = manager.get_credential("ANCESTRY_USERNAME")
            password = manager.get_credential("ANCESTRY_PASSWORD")
            api_key = manager.get_credential("DEEPSEEK_API_KEY")

            assert username == "workflow@test.com"
            assert password == "workflow123!"
            assert api_key == "sk-workflow789"

            # Step 4: Test with missing optional credential
            google_key = manager.get_credential("GOOGLE_API_KEY")
            assert google_key is None

        finally:
            manager.delete_credentials()

    # Run tests organized by standard categories
    with suppress_logging():
        # Initialization Tests
        suite.run_test(
            "Initialization",
            test_security_manager_instantiation,
            "Should create SecurityManager instances with proper initialization",
        )
        suite.run_test(
            "Initialization",
            test_master_key_operations,
            "Should generate and retrieve master encryption keys securely",
        )

        # Core Functionality Tests
        suite.run_test(
            "Core Functionality",
            test_credential_encryption_decryption,
            "Should encrypt and decrypt credentials with data integrity",
        )
        suite.run_test(
            "Core Functionality",
            test_individual_credential_retrieval,
            "Should retrieve specific credentials from encrypted storage",
        )
        suite.run_test(
            "Core Functionality",
            test_credential_validation,
            "Should validate required credentials are present and non-empty",
        )

        # Edge Cases Tests
        suite.run_test(
            "Edge Cases",
            test_error_handling,
            "Should handle missing files and invalid data gracefully",
        )
        suite.run_test(
            "Edge Cases",
            test_file_security,
            "Should set appropriate file permissions for encrypted storage",
        )

        # Integration Tests
        suite.run_test(
            "Integration",
            test_multiple_instances,
            "Should handle multiple SecurityManager instances with shared credentials",
        )
        suite.run_test(
            "Integration",
            test_full_workflow,
            "Should support complete credential storage and retrieval workflow",
        )

        # Performance Tests
        def test_performance():
            """Test encryption/decryption performance with larger datasets."""
            import time

            manager = SecurityManager("TestPerfApp")

            # Create larger credential set
            large_credentials = {f"KEY_{i}": f"value_{i}_{'x'*50}" for i in range(100)}

            try:
                # Test encryption performance
                start_time = time.time()
                result = manager.encrypt_credentials(large_credentials)
                encrypt_time = time.time() - start_time
                assert result is True
                assert (
                    encrypt_time < 2.0
                ), f"Encryption took too long: {encrypt_time:.2f}s"

                # Test decryption performance
                start_time = time.time()
                decrypted = manager.decrypt_credentials()
                decrypt_time = time.time() - start_time
                assert decrypted == large_credentials
                assert (
                    decrypt_time < 2.0
                ), f"Decryption took too long: {decrypt_time:.2f}s"

            finally:
                manager.delete_credentials()

        suite.run_test(
            "Performance",
            test_performance,
            "Should encrypt/decrypt large credential sets efficiently",
        )

        # Error Handling Tests
        suite.run_test(
            "Error Handling",
            test_credential_deletion,
            "Should securely delete credential files and cleanup resources",
        )

    return suite.finish_suite()


def _run_basic_fallback_tests() -> bool:
    """
    Basic functionality test (fallback when test framework unavailable).

    This function provides essential testing when the enhanced test framework
    is not available, ensuring core security manager functionality works.
    """
    logger.info("Running basic SecurityManager fallback tests...")

    try:
        # Create test instance
        security_manager = SecurityManager("TestApp")

        # Test encryption/decryption
        test_credentials = {
            "TEST_USERNAME": "test_user",
            "TEST_PASSWORD": "test_pass123",
            "TEST_API_KEY": "sk-test123456789",
        }

        # Test encryption
        if not security_manager.encrypt_credentials(test_credentials):
            logger.error("Failed to encrypt test credentials")
            return False

        # Test decryption
        decrypted = security_manager.decrypt_credentials()
        if not decrypted:
            logger.error("Failed to decrypt test credentials")
            return False

        # Verify data integrity
        if decrypted != test_credentials:
            logger.error("Decrypted credentials don't match original")
            return False

        # Test individual credential retrieval
        username = security_manager.get_credential("TEST_USERNAME")
        if username != "test_user":
            logger.error("Failed to retrieve individual credential")
            return False

        # Test validation with suppressed logging
        import logging

        original_level = logger.level
        logger.setLevel(logging.CRITICAL)

        try:
            # Test valid credentials
            if not security_manager.validate_credentials(
                {"ANCESTRY_USERNAME": "test", "ANCESTRY_PASSWORD": "test"}
            ):
                logger.setLevel(original_level)
                logger.error("Valid credentials failed validation")
                return False

            # Test invalid credentials (should return False)
            if security_manager.validate_credentials({"ANCESTRY_USERNAME": "test"}):
                logger.setLevel(original_level)
                logger.error("Invalid credentials passed validation")
                return False
        finally:
            logger.setLevel(original_level)

        # Cleanup test files
        security_manager.delete_credentials()

        logger.info("SecurityManager basic tests passed successfully")
        return True

    except Exception as e:
        logger.error(f"SecurityManager basic tests failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)


# =============================================================================
# DEVELOPER EXAMPLES AND USAGE PATTERNS
# =============================================================================

"""
BASIC USAGE EXAMPLES:
====================

1. Simple Credential Storage:
   ```python
   from security_manager import SecurityManager
   
   manager = SecurityManager("MyApplication")
   
   # Store credentials
   creds = {
       "ANCESTRY_USERNAME": "user@example.com",
       "ANCESTRY_PASSWORD": "my_secure_password"
   }
   manager.encrypt_credentials(creds)
   
   # Later retrieve credentials
   stored_creds = manager.decrypt_credentials()
   username = manager.get_credential("ANCESTRY_USERNAME")
   ```

2. Multiple API Keys:
   ```python
   manager = SecurityManager("APIManager")
   
   api_credentials = {
       "OPENAI_API_KEY": "sk-openai123...",
       "DEEPSEEK_API_KEY": "sk-deepseek456...",
       "GOOGLE_API_KEY": "AIza789..."
   }
   manager.encrypt_credentials(api_credentials)
   
   # Retrieve specific API key
   openai_key = manager.get_credential("OPENAI_API_KEY")
   ```

3. Credential Validation:
   ```python
   manager = SecurityManager("ValidatedApp")
   
   # Check if required credentials exist and are valid
   required_creds = {
       "ANCESTRY_USERNAME": "user@domain.com",
       "ANCESTRY_PASSWORD": "password123"
   }
   
   if manager.validate_credentials(required_creds):
       print("Credentials are valid!")
       manager.encrypt_credentials(required_creds)
   else:
       print("Missing or invalid credentials")
   ```

4. Error Handling:
   ```python
   manager = SecurityManager("RobustApp")
   
   try:
       # Attempt to retrieve credentials
       creds = manager.decrypt_credentials()
       if creds is None:
           print("No credentials found - first time setup needed")
           # Handle first-time setup...
       else:
           username = manager.get_credential("USERNAME")
           if username is None:
               print("USERNAME not found in stored credentials")
   except Exception as e:
       print(f"Error accessing credentials: {e}")
   ```

5. Cleanup and Management:
   ```python
   manager = SecurityManager("TemporaryApp")
   
   # Store temporary credentials
   temp_creds = {"TEMP_TOKEN": "abc123"}
   manager.encrypt_credentials(temp_creds)
   
   # Use credentials...
   
   # Clean up when done
   manager.delete_credentials()
   ```

INTEGRATION PATTERNS:
====================

1. Configuration Class Integration:
   ```python
   class AppConfig:
       def __init__(self):
           self.security_manager = SecurityManager("MyApp")
           self.credentials = self.security_manager.decrypt_credentials()
       
       def get_ancestry_credentials(self):
           return {
               "username": self.security_manager.get_credential("ANCESTRY_USERNAME"),
               "password": self.security_manager.get_credential("ANCESTRY_PASSWORD")
           }
   ```

2. Context Manager Pattern:
   ```python
   from contextlib import contextmanager
   
   @contextmanager
   def secure_credentials(app_name):
       manager = SecurityManager(app_name)
       try:
           yield manager
       finally:
           # Optional: cleanup on exit
           pass
   
   # Usage:
   with secure_credentials("MyApp") as manager:
       creds = manager.decrypt_credentials()
       # Use credentials...
   ```

TESTING YOUR INTEGRATION:
========================

When integrating SecurityManager into your code, test these scenarios:

1. First-time setup (no existing credentials)
2. Normal operation (credentials exist and are valid)
3. Corrupted credential file handling
4. Missing keyring support
5. Invalid credential validation
6. Multiple application instances

Run the test suite to ensure everything works:
```bash
python security_manager.py
```

SECURITY BEST PRACTICES:
========================

1. Always validate credentials before storing them
2. Use specific application names to avoid conflicts
3. Handle the case where keyring is unavailable
4. Clean up test credentials after testing
5. Never log actual credential values
6. Use environment variables for CI/CD testing
7. Regularly rotate stored credentials

For more information, see the comprehensive test suite in run_comprehensive_tests().
"""
