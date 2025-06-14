#!/usr/bin/env python3
"""
Credential Manager - Interactive interface for managing encrypted credentials

This script provides a user-friendly interface for managing encrypted credentials
used by the Ancestry application. All credentials are stored securely using
encryption and never stored in plain text.

Usage: python credential_manager.py
"""

import sys
from typing import Dict, Optional
from security_manager import SecurityManager


def display_menu():
    """Display the credential management menu."""
    print("\n" + "=" * 60)
    print("           ANCESTRY CREDENTIAL MANAGER")
    print("=" * 60)
    print("\n1. View stored credentials (masked)")
    print("2. Add/Update credentials")
    print("3. Remove specific credential")
    print("4. Export credentials to new installation")
    print("5. Delete all credentials")
    print("6. Exit")
    print("\n" + "=" * 60)


def view_credentials(sm: SecurityManager):
    """Display all stored credentials with masked values."""
    credentials = sm.decrypt_credentials()

    if not credentials:
        print("\n❌ No credentials are currently stored.")
        return

    print(f"\n✅ Found {len(credentials)} stored credentials:")
    print("-" * 50)

    for key, value in sorted(credentials.items()):
        masked_value = (
            value[:3] + "*" * (len(value) - 6) + value[-3:]
            if len(value) > 6
            else "*" * len(value)
        )
        print(f"  {key:<20}: {masked_value}")

    print("-" * 50)


def add_update_credentials(sm: SecurityManager):
    """Add or update credentials."""
    print("\n📝 Add/Update Credentials")
    print("Supported credential types:")
    print("  - ANCESTRY_USERNAME")
    print("  - ANCESTRY_PASSWORD")
    print("  - DEEPSEEK_API_KEY")
    print("  - GOOGLE_API_KEY")

    # Load existing credentials
    existing_creds = sm.decrypt_credentials() or {}
    updated_creds = dict(existing_creds)

    while True:
        print(f"\nCurrent credentials: {list(updated_creds.keys())}")
        key = input("\nEnter credential name (or 'done' to finish): ").strip()

        if key.lower() == "done":
            break

        if not key:
            print("❌ Please enter a credential name.")
            continue

        # Show current value if it exists
        if key in updated_creds:
            current = updated_creds[key]
            masked = (
                current[:3] + "*" * (len(current) - 6) + current[-3:]
                if len(current) > 6
                else "*" * len(current)
            )
            print(f"Current value: {masked}")

        value = input(f"Enter value for {key}: ").strip()

        if not value:
            print("❌ Value cannot be empty.")
            continue

        updated_creds[key] = value
        print(f"✅ {key} updated.")

    if updated_creds != existing_creds:
        if sm.encrypt_credentials(updated_creds):
            print(f"\n✅ Successfully saved {len(updated_creds)} credentials.")
        else:
            print("\n❌ Failed to save credentials.")
    else:
        print("\nℹ️  No changes made.")


def remove_credential(sm: SecurityManager):
    """Remove a specific credential."""
    credentials = sm.decrypt_credentials()

    if not credentials:
        print("\n❌ No credentials are currently stored.")
        return

    print(f"\nStored credentials: {list(credentials.keys())}")
    key = input("Enter credential name to remove: ").strip()

    if key not in credentials:
        print(f"❌ Credential '{key}' not found.")
        return

    confirm = (
        input(f"⚠️  Are you sure you want to remove '{key}'? (yes/no): ").strip().lower()
    )
    if confirm not in ["yes", "y"]:
        print("❌ Removal cancelled.")
        return

    del credentials[key]

    if sm.encrypt_credentials(credentials):
        print(f"✅ Successfully removed '{key}'.")
    else:
        print(f"❌ Failed to remove '{key}'.")


def export_credentials(sm: SecurityManager):
    """Export credentials for backup/transfer."""
    credentials = sm.decrypt_credentials()

    if not credentials:
        print("\n❌ No credentials are currently stored.")
        return

    print("\n📤 Export Credentials")
    print("⚠️  WARNING: This will display credentials in plain text!")
    confirm = input("Continue? (yes/no): ").strip().lower()

    if confirm not in ["yes", "y"]:
        print("❌ Export cancelled.")
        return

    print("\n" + "=" * 50)
    print("CREDENTIAL EXPORT")
    print("=" * 50)

    for key, value in sorted(credentials.items()):
        print(f"{key}={value}")

    print("=" * 50)
    print("Copy these values to your new installation's credential manager.")
    print("⚠️  Clear your terminal history after copying!")


def delete_all_credentials(sm: SecurityManager):
    """Delete all stored credentials."""
    credentials = sm.decrypt_credentials()

    if not credentials:
        print("\n❌ No credentials are currently stored.")
        return

    print(f"\n⚠️  WARNING: This will delete ALL {len(credentials)} stored credentials!")
    print("This action cannot be undone.")

    confirm = input("Type 'DELETE ALL' to confirm: ").strip()
    if confirm != "DELETE ALL":
        print("❌ Deletion cancelled.")
        return

    if sm.delete_credentials():
        print("✅ All credentials deleted successfully.")
    else:
        print("❌ Failed to delete credentials.")


def main():
    """Main credential manager interface."""
    print("Ancestry Credential Manager")
    print("Managing encrypted credentials for security...")

    sm = SecurityManager()

    while True:
        display_menu()

        try:
            choice = input("\nEnter choice (1-6): ").strip()

            if choice == "1":
                view_credentials(sm)
            elif choice == "2":
                add_update_credentials(sm)
            elif choice == "3":
                remove_credential(sm)
            elif choice == "4":
                export_credentials(sm)
            elif choice == "5":
                delete_all_credentials(sm)
            elif choice == "6":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-6.")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

        input("\nPress Enter to continue...")


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for setup_credentials_interactive.py.
    Tests credential management, interactive setup, and security operations.
    """
    from test_framework import TestSuite, suppress_logging
    from unittest.mock import MagicMock, patch

    suite = TestSuite(
        "Interactive Credential Setup & Management", "setup_credentials_interactive.py"
    )
    suite.start_suite()

    def test_module_initialization():
        """Test module initialization and function availability."""
        # Test that main function exists
        assert callable(main), "main function should be callable"

        # Test that display functions exist
        assert callable(display_menu), "display_menu function should be callable"
        assert callable(
            view_credentials
        ), "view_credentials function should be callable"
        assert callable(
            add_update_credentials
        ), "add_update_credentials function should be callable"

    def test_interactive_functionality():
        """Test interactive credential setup functionality."""
        # Test with mock inputs - need to include the "Press Enter" prompts
        with patch(
            "builtins.input", side_effect=["1", "", "6"]
        ):  # View credentials, press enter, then exit
            with patch("setup_credentials_interactive.SecurityManager") as mock_sm:
                mock_instance = MagicMock()
                mock_instance.decrypt_credentials.return_value = {}
                mock_sm.return_value = mock_instance

                try:
                    main()
                except SystemExit:
                    # Expected when choosing exit option
                    pass

    def test_menu_functions():
        """Test individual menu functions."""
        mock_sm = MagicMock()
        mock_sm.decrypt_credentials.return_value = {"test": "value"}

        # Test view_credentials function
        with patch("builtins.print"):
            view_credentials(mock_sm)
            mock_sm.decrypt_credentials.assert_called()

    def test_error_handling():
        """Test error handling in interactive setup."""
        # Test with invalid menu choices - need to include "Press Enter" prompts after each invalid choice
        with patch("builtins.input", side_effect=["invalid", "", "99", "", "6"]):
            with patch("setup_credentials_interactive.SecurityManager") as mock_sm:
                mock_instance = MagicMock()
                mock_sm.return_value = mock_instance
                try:
                    main()
                except SystemExit:
                    # Expected when choosing exit
                    pass

    with suppress_logging():
        suite.run_test(
            "Module initialization",
            test_module_initialization,
            "Module initializes with required functions for interactive credential setup",
            "Test main function and setup function availability",
            "Module provides necessary functions for credential management",
        )

        suite.run_test(
            "Interactive functionality",
            test_interactive_functionality,
            "Interactive setup process works with user input simulation",
            "Test interactive credential setup with mocked user inputs",
            "Setup process handles user interaction correctly",
        )

        suite.run_test(
            "Menu functions",
            test_menu_functions,
            "Menu functions work correctly with SecurityManager",
            "Test view_credentials and other menu functions",
            "Menu functions handle SecurityManager interaction properly",
        )

        suite.run_test(
            "Error handling",
            test_error_handling,
            "Error handling works correctly with invalid inputs",
            "Test error handling with invalid menu choices",
            "Error handling prevents crashes and provides feedback",
        )

    return suite.finish_suite()

    with suppress_logging():
        suite.run_test(
            "Module initialization",
            test_module_initialization,
            "Module initializes with required functions for interactive credential setup",
            "Test main function and setup function availability",
            "Module provides necessary functions for credential management",
        )

        suite.run_test(
            "Interactive functionality",
            test_interactive_functionality,
            "Interactive setup process works with user input simulation",
            "Test interactive credential setup with mocked user inputs",
            "Setup process handles user interaction correctly",
        )

        suite.run_test(
            "Menu functions",
            test_menu_functions,
            "Menu functions work correctly with SecurityManager",
            "Test view_credentials and other menu functions",
            "Menu functions handle SecurityManager interaction properly",
        )

        suite.run_test(
            "Error handling",
            test_error_handling,
            "Error handling works correctly with invalid inputs",
            "Test error handling with invalid menu choices",
            "Error handling prevents crashes and provides feedback",
        )

    return suite.finish_suite()


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("🔐 Running Interactive Credential Setup comprehensive test suite...")
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    else:
        main()
