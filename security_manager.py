#!/usr/bin/env python3
"""
Security Manager for Ancestry.com Automation System
Provides encrypted credential storage and secure session management.
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


def self_test() -> bool:
    """Test the SecurityManager functionality."""
    logger.info("Starting SecurityManager self-test...")

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
            return False  # Test validation - temporarily reduce log level for expected failures
        import logging

        original_level = logger.level
        logger.setLevel(
            logging.CRITICAL
        )  # Suppress expected error messages during testing

        try:
            # Test valid credentials
            if not security_manager.validate_credentials(
                {"ANCESTRY_USERNAME": "test", "ANCESTRY_PASSWORD": "test"}
            ):
                logger.setLevel(original_level)
                logger.error("Valid credentials failed validation")
                return False

            # Test invalid credentials (missing password) - this should return False
            if security_manager.validate_credentials({"ANCESTRY_USERNAME": "test"}):
                logger.setLevel(original_level)
                logger.error("Invalid credentials passed validation")
                return False
        finally:
            # Restore original log level
            logger.setLevel(original_level)

        # Cleanup test files
        security_manager.delete_credentials()

        logger.info("SecurityManager self-test passed successfully")
        return True

    except Exception as e:
        logger.error(f"SecurityManager self-test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = self_test()
    exit(0 if success else 1)
