#!/usr/bin/env python3
"""
Quick setup for test credentials to resolve configuration validation issues.
"""

from security_manager import SecurityManager


def setup_test_credentials():
    """Set up minimal test credentials for development/testing"""
    try:
        security_manager = SecurityManager()

        # Check if credentials already exist
        existing_creds = security_manager.decrypt_credentials()
        if existing_creds:
            print("✓ Encrypted credentials already exist.")
            print("Keys:", list(existing_creds.keys()))
            return True

        # Set up minimal test credentials to satisfy validation
        test_credentials = {
            "ANCESTRY_USERNAME": "test@example.com",
            "ANCESTRY_PASSWORD": "test_password_123",
            "DEEPSEEK_API_KEY": "test_deepseek_key_456",
        }

        if security_manager.encrypt_credentials(test_credentials):
            print("✓ Test credentials encrypted and stored successfully!")
            print("Note: These are test credentials for development only.")
            print("Replace with real credentials using: python credential_manager.py")
            return True
        else:
            print("✗ Failed to encrypt and store test credentials")
            return False

    except Exception as e:
        print(f"Error setting up credentials: {e}")
        return False


if __name__ == "__main__":
    success = setup_test_credentials()
    exit(0 if success else 1)
