#!/usr/bin/env python3
"""
Setup script for migrating to secure credential storage
Usage: python setup_security.py
"""

import sys
import os
from pathlib import Path


def main():
    """Interactive setup for migrating to secure credential storage."""
    print("=== Ancestry.com Automation Security Setup ===")
    print("This script will help you migrate to secure encrypted credential storage.\n")

    try:
        from security_manager import SecurityManager
    except ImportError:
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


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
