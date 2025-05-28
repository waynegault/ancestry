#!/usr/bin/env python3
"""
Set up encrypted credentials using the actual values from .env file
"""

import os
from pathlib import Path


def setup_real_credentials():
    """Set up encrypted credentials using actual values from .env"""
    print("Setting up encrypted credentials from .env file...")

    # Read the current .env file to get the actual credentials
    env_file = Path(".env")
    if not env_file.exists():
        print("✗ .env file not found")
        return False

    # Parse the .env file
    credentials = {}
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                if key in [
                    "ANCESTRY_USERNAME",
                    "ANCESTRY_PASSWORD",
                    "DEEPSEEK_API_KEY",
                ]:
                    credentials[key] = value

    print(f"Found credentials for: {list(credentials.keys())}")

    # Now try to import SecurityManager and encrypt the credentials
    try:
        from security_manager import SecurityManager

        security_manager = SecurityManager()

        # Check if credentials already exist
        existing_creds = security_manager.decrypt_credentials()
        if existing_creds:
            print("✓ Encrypted credentials already exist.")
            print("Keys:", list(existing_creds.keys()))

            # Check if we need to update with missing keys
            missing_keys = set(credentials.keys()) - set(existing_creds.keys())
            if missing_keys:
                print(f"Adding missing keys: {missing_keys}")
                existing_creds.update({k: credentials[k] for k in missing_keys})
                if security_manager.encrypt_credentials(existing_creds):
                    print("✓ Updated encrypted credentials with missing keys")
                else:
                    print("✗ Failed to update encrypted credentials")
                    return False
            return True

        # Encrypt and store the credentials
        if security_manager.encrypt_credentials(credentials):
            print("✓ Real credentials encrypted and stored successfully!")
            print("✓ System will now use encrypted credentials instead of .env")
            return True
        else:
            print("✗ Failed to encrypt and store credentials")
            return False

    except Exception as e:
        print(f"Error setting up encrypted credentials: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = setup_real_credentials()
    if success:
        print("\n🎉 Encrypted credentials are now set up!")
        print("You can now test the availability flags.")
    else:
        print("\n❌ Failed to set up encrypted credentials")
    exit(0 if success else 1)
