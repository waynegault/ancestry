#!/usr/bin/env python3
"""
Unified Credential Manager for Ancestry Project
Replaces: setup_security.py, setup_credentials_interactive.py, setup_credentials_helper.py

This is the single entry point for all credential management tasks.
Usage: python credentials.py
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import (
    setup_module,
    register_function,
    get_function,
    is_function_available,
)

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
from error_handling import (
    retry_on_failure,
    circuit_breaker,
    timeout_protection,
    graceful_degradation,
    error_context,
    AncestryException,
    RetryableError,
    NetworkTimeoutError,
    AuthenticationExpiredError,
    APIRateLimitError,
    ErrorContext,
)

# === STANDARD LIBRARY IMPORTS ===
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# === LOCAL IMPORTS ===
# Import SecurityManager
try:
    from security_manager import SecurityManager

    SECURITY_AVAILABLE = True
except ImportError as e:
    print(f"❌ Security dependencies not available: {e}")
    print("\n" + "=" * 60)
    print("           SECURITY DEPENDENCIES MISSING")
    print("=" * 60)
    print("\nRequired security packages:")
    print("  - cryptography: For secure encryption/decryption of credentials")
    print("  - keyring: For secure storage of master encryption keys")
    print("\n📋 Installation Instructions:")
    print("1. Install required packages using pip:")
    print("   pip install cryptography keyring")
    print("   - OR -")
    print("   pip install -r requirements.txt")
    print("\n💻 Platform-Specific Information:")
    print("• Windows: No additional configuration needed")
    print("• Linux/macOS: You may need an alternative keyring backend:")
    print("  pip install keyrings.alt")
    print("  (Some Linux distros may require: sudo apt-get install python3-dbus)")
    print("\n🔍 Troubleshooting:")
    print(
        "• If you have permission issues: Use 'pip install --user cryptography keyring'"
    )
    print("• If you have build errors with cryptography:")
    print("  - Windows: Ensure Visual C++ Build Tools are installed")
    print("  - Linux: Install 'python3-dev' and 'libffi-dev' packages")
    print("• For more information, see SECURITY_STREAMLINED.md")
    print("\n💡 Quick Fix:")
    print(
        "Run the credential manager again and select 'y' when prompted to install dependencies"
    )
    print("Or type: python credentials.py")

    SECURITY_AVAILABLE = False


class UnifiedCredentialManager:
    """Unified interface for all credential management operations."""

    def __init__(self):
        if not SECURITY_AVAILABLE:
            raise ImportError("Security dependencies not available")
        self.security_manager = SecurityManager()

    @staticmethod
    def check_and_install_dependencies():
        """Check for security dependencies and offer to install them if missing."""
        if SECURITY_AVAILABLE:
            return True

        print("\n" + "=" * 60)
        print("           SECURITY DEPENDENCY CHECK")
        print("=" * 60)
        print("\nThe following security dependencies are required:")
        print("  - cryptography: For secure encryption/decryption")
        print("  - keyring: For secure storage of master keys")

        choice = (
            input("\nWould you like to install these dependencies now? (y/N): ")
            .strip()
            .lower()
        )
        if choice != "y":
            print("\nOperation cancelled. Please install dependencies manually:")
            print("  pip install cryptography keyring")
            print("  - OR -")
            print("  pip install -r requirements.txt")

            print("\n💡 After installation, run: python credentials.py")
            return False

        try:
            import subprocess

            print("\nInstalling required dependencies...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "cryptography", "keyring"]
            )

            print("\n✅ Core dependencies installed successfully!")

            # For Linux systems, also install keyrings.alt
            if os.name != "nt":  # Not Windows
                print("\n⚠️ Linux/macOS detected - additional backend may be required")
                choice = (
                    input(
                        "Install alternative keyring backend for Linux/macOS? (Y/n): "
                    )
                    .strip()
                    .lower()
                )
                if choice != "n":
                    try:
                        print("Installing keyrings.alt...")
                        subprocess.check_call(
                            [sys.executable, "-m", "pip", "install", "keyrings.alt"]
                        )
                        print("✅ Alternative keyring backend installed!")
                    except Exception as e:
                        print(f"⚠️ Could not install alternative backend: {e}")
                        print(
                            "You may need to manually install system dependencies first:"
                        )
                        print("  Ubuntu/Debian: sudo apt-get install python3-dbus")
                        print("  Fedora: sudo dnf install python3-dbus")
                        print("Then try: pip install keyrings.alt")

            print("\n" + "=" * 60)
            print("✅ Dependencies installed successfully!")
            print("Please restart the credential manager: python credentials.py")
            print("=" * 60)
            return True

        except Exception as e:
            print(f"\n❌ Failed to install dependencies: {e}")
            print("\nPlease install manually:")
            print("  pip install cryptography keyring")
            print("  - OR -")
            print("  pip install -r requirements.txt")

            if os.name != "nt":
                print("\nFor Linux/macOS users, you may also need:")
                print("  pip install keyrings.alt")

            print("\nIf you're encountering permissions errors:")
            print("  pip install --user cryptography keyring")

            print("\nIf you're encountering build errors with cryptography:")
            print("  - Windows: Ensure Visual C++ Build Tools are installed")
            print("  - Linux: Install python3-dev and libffi-dev packages")

            return False

    def display_main_menu(self):
        """Display the main credential management menu."""
        print("\n" + "=" * 60)
        print("           ANCESTRY CREDENTIAL MANAGER")
        print("=" * 60)
        print("\n📋 Credential Operations:")
        print("  1. View stored credentials (masked)")
        print("  2. Setup/Update credentials")
        print("  3. Remove specific credential")
        print("  4. Delete all credentials")
        print("\n🔧 Utility Operations:")
        print("  5. Setup test credentials (for development)")
        print("  6. Import credentials from .env file")
        print("  7. Export credentials for backup")
        print("  8. Check credential status")
        print("\n  9. Edit credential types configuration")
        print("\n  0. Exit")
        print("\n" + "=" * 60)

    def view_credentials(self):
        """Display all stored credentials with masked values."""
        # Check if encrypted file exists but can't be decrypted
        if self.security_manager.credentials_file.exists():
            credentials = self.security_manager.decrypt_credentials()
            if credentials is None:
                print("\n❌ Failed to decrypt credentials.")
                print("This usually indicates a master key mismatch or corrupted data.")
                print("\n💡 Possible solutions:")
                print("  - Use option 6 to import from .env file (will fix key issues)")
                print("  - Use option 4 to delete all credentials and start fresh")
                print("  - Use option 2 to setup new credentials")
                return
        else:
            credentials = self.security_manager.decrypt_credentials()

        if not credentials:
            print("\n💡 No credentials are currently stored.")
            print(
                "Use option 2 to setup credentials or option 6 to import from .env file."
            )
            return

        print(f"\n✅ Found {len(credentials)} stored credentials:")
        print("-" * 50)

        for key, value in sorted(credentials.items()):
            # Better masking for different credential types
            if len(value) <= 4:
                masked_value = "*" * len(value)
            elif len(value) <= 8:
                masked_value = value[:2] + "*" * (len(value) - 2)
            else:
                masked_value = value[:3] + "*" * (len(value) - 6) + value[-3:]

            print(f"  {key:<25}: {masked_value}")

        print("-" * 50)

    def setup_credentials(self):
        """Interactive credential setup."""
        print("\n" + "=" * 50)
        print("         CREDENTIAL SETUP")
        print("=" * 50)

        # Check existing credentials
        existing_creds = self.security_manager.decrypt_credentials()
        if existing_creds:
            print(f"✓ Found {len(existing_creds)} existing credentials")
            print("Options:")
            print("  a) Add/update individual credentials")
            print("  r) Replace all credentials")
            print("  c) Cancel")

            choice = input("\nChoice (a/r/c): ").strip().lower()
            if choice == "c":
                return
            elif choice == "r":
                if not self._confirm_action("replace ALL credentials"):
                    return
                existing_creds = {}
        else:
            existing_creds = {}
            print("No existing credentials found. Setting up new credentials...")

        # Load credential types from configuration file
        required_creds, optional_creds = self._load_credential_types()

        new_creds = existing_creds.copy()

        # Setup required credentials
        print(f"\n🔸 Required Credentials:")
        for key, description in required_creds.items():
            current_value = existing_creds.get(key, "")
            if current_value:
                print(f"  {key}: ***existing*** (press Enter to keep)")

            value = input(f"  {description}: ").strip()
            if value or not current_value:
                if not value and not current_value:
                    print(f"    ⚠️ Warning: {key} is required for basic functionality")
                if value:
                    new_creds[key] = value

        # Setup optional credentials
        print(f"\n🔹 Optional Credentials (press Enter to skip):")
        for key, description in optional_creds.items():
            current_value = existing_creds.get(key, "")
            if current_value:
                print(f"  {key}: ***existing*** (press Enter to keep)")

            value = input(f"  {description}: ").strip()
            if value:
                new_creds[key] = value

        # Save credentials
        if self.security_manager.encrypt_credentials(new_creds):
            print(f"\n✅ Successfully saved {len(new_creds)} credentials!")
        else:
            print("\n❌ Failed to save credentials")

    def remove_credential(self):
        """Remove a specific credential."""
        credentials = self.security_manager.decrypt_credentials()

        if not credentials:
            print("\n❌ No credentials to remove.")
            return

        print(f"\nAvailable credentials:")
        for i, key in enumerate(sorted(credentials.keys()), 1):
            print(f"  {i}. {key}")

        try:
            choice = input(f"\nEnter number to remove (1-{len(credentials)}): ").strip()
            index = int(choice) - 1
            keys = sorted(credentials.keys())

            if 0 <= index < len(keys):
                key_to_remove = keys[index]
                if self._confirm_action(f"remove '{key_to_remove}'"):
                    del credentials[key_to_remove]

                    if self.security_manager.encrypt_credentials(credentials):
                        print(f"✅ Removed '{key_to_remove}' successfully!")
                    else:
                        print(f"❌ Failed to remove '{key_to_remove}'")

            else:
                print("❌ Invalid selection")

        except ValueError:
            print("❌ Invalid input")

    def delete_all_credentials(self):
        """Delete all stored credentials."""
        credentials = self.security_manager.decrypt_credentials()

        if not credentials:
            print("\n❌ No credentials to delete.")
            return

        print(f"\n⚠️ This will delete ALL {len(credentials)} stored credentials!")
        if self._confirm_action("delete ALL credentials"):
            if self.security_manager.delete_credentials():
                print("✅ All credentials deleted successfully!")
            else:
                print("❌ Failed to delete credentials")

    def setup_test_credentials(self):
        """Setup test credentials for development."""
        print("\n" + "=" * 50)
        print("       TEST CREDENTIAL SETUP")
        print("=" * 50)

        existing_creds = self.security_manager.decrypt_credentials()
        if existing_creds:
            print("⚠️ Warning: You already have credentials stored.")
            if not self._confirm_action(
                "add test credentials (this will overwrite existing ones)"
            ):
                return

        test_credentials = {
            "ANCESTRY_USERNAME": "test@example.com",
            "ANCESTRY_PASSWORD": "test_password_123",
            "DEEPSEEK_API_KEY": "test_deepseek_key_456",
        }

        if self.security_manager.encrypt_credentials(test_credentials):
            print("✅ Test credentials saved successfully!")
            print("⚠️ Note: These are dummy credentials for testing only.")
            print("   Replace with real credentials before production use.")
        else:
            print("❌ Failed to save test credentials")

    def import_from_env(self):
        """Import credentials from a .env file."""
        print("\n" + "=" * 50)
        print("      IMPORT FROM .ENV FILE")
        print("=" * 50)

        # Ask for .env file path when in interactive mode
        env_file = input("Enter path to .env file (default: .env): ").strip()
        if not env_file:
            env_file = ".env"

        # Try multiple locations for the .env file
        potential_paths = []

        # 1. Check user-provided path (absolute or relative)
        if os.path.isabs(env_file):
            potential_paths.append(env_file)
        else:
            # 2. Check relative to current directory
            potential_paths.append(os.path.join(os.getcwd(), env_file))

            # 3. Check relative to script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            potential_paths.append(os.path.join(script_dir, env_file))

            # 4. Check parent directory (project root)
            parent_dir = os.path.dirname(script_dir)
            potential_paths.append(os.path.join(parent_dir, env_file))

        # Find the first existing file from potential paths
        env_file_path = None
        for path in potential_paths:
            if os.path.exists(path):
                env_file_path = path
                break

        if not env_file_path:
            print(f"❌ File not found: {env_file}")
            print("💡 Make sure the .env file exists in one of these locations:")
            for path in potential_paths:
                print(f"  - {path}")
            return

        try:
            print(f"📂 Using .env file: {env_file_path}")
            # Read and parse .env file
            env_credentials = {}
            with open(env_file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue

                    # Parse KEY=VALUE format
                    if "=" not in line:
                        print(f"⚠️ Warning: Skipping invalid line {line_num}: {line}")
                        continue

                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or (
                        value.startswith("'") and value.endswith("'")
                    ):
                        value = value[1:-1]

                    env_credentials[key] = value

            if not env_credentials:
                print("❌ No valid credentials found in .env file.")
                return

            print(f"📋 Found {len(env_credentials)} credentials in .env file:")
            for key in sorted(env_credentials.keys()):
                # Show masked preview
                masked_value = (
                    env_credentials[key][:3]
                    + "*" * max(0, len(env_credentials[key]) - 6)
                    + env_credentials[key][-3:]
                    if len(env_credentials[key]) > 6
                    else "*" * len(env_credentials[key])
                )
                print(f"  {key}: {masked_value}")

            # Check for existing credentials
            existing_creds = self.security_manager.decrypt_credentials()
            if existing_creds:
                print(f"\n⚠️ You already have {len(existing_creds)} stored credentials.")
                print("Import options:")
                print("  m) Merge (add new, keep existing)")
                print("  o) Overwrite (replace existing with .env values)")
                print("  r) Replace all (delete existing, use only .env)")
                print("  c) Cancel")

                choice = input("Choice (m/o/r/c): ").strip().lower()
                if choice == "c":
                    return
                elif choice == "m":
                    # Merge: existing credentials take precedence
                    final_creds = env_credentials.copy()
                    final_creds.update(existing_creds)  # existing wins on conflicts
                    action_desc = "merged with existing"
                elif choice == "o":
                    # Overwrite: .env credentials take precedence
                    final_creds = existing_creds.copy()
                    final_creds.update(env_credentials)  # .env wins on conflicts
                    action_desc = "merged (overwriting existing)"
                elif choice == "r":
                    # Replace: only use .env credentials
                    if not self._confirm_action("replace ALL existing credentials"):
                        return
                    final_creds = env_credentials
                    action_desc = "replaced all existing"
                else:
                    print("❌ Invalid choice.")
                    return
            else:
                final_creds = env_credentials
                action_desc = "imported"

            # Save the credentials
            if self.security_manager.encrypt_credentials(final_creds):
                print(
                    f"✅ Successfully {action_desc} {len(env_credentials)} credentials from .env file!"
                )

                # Show summary
                required = ["ANCESTRY_USERNAME", "ANCESTRY_PASSWORD"]
                found_required = [cred for cred in required if cred in final_creds]
                if found_required:
                    print(f"✅ Required credentials found: {', '.join(found_required)}")

                missing_required = [
                    cred for cred in required if cred not in final_creds
                ]
                if missing_required:
                    print(
                        f"⚠️ Missing required credentials: {', '.join(missing_required)}"
                    )
                    print("   Use option 2 to add missing credentials.")

            else:
                print("❌ Failed to save imported credentials.")

        except Exception as e:
            print(f"❌ Error reading .env file: {e}")
            print("💡 Make sure the file is readable and in KEY=VALUE format.")

    def export_credentials(self):
        """Export credentials for backup."""
        print("\n" + "=" * 50)
        print("         CREDENTIAL EXPORT")
        print("=" * 50)

        credentials = self.security_manager.decrypt_credentials()

        if not credentials:
            print("\n❌ No credentials to export.")
            return

        print(f"\n📦 Export Options:")
        print(f"  1. Display credentials (for manual copy)")
        print(f"  2. Export to file (encrypted)")
        print(f"  3. Cancel")

        choice = input("Choice (1/2/3): ").strip()

        if choice == "1":
            print(f"\n📋 Current Credentials (for manual backup):")
            print("-" * 50)
            for key, value in sorted(credentials.items()):
                print(f"{key}={value}")
            print("-" * 50)
            print("⚠️ Store this information securely!")

        elif choice == "2":
            filename = input(
                "Export filename (default: credentials_backup.json): "
            ).strip()
            if not filename:
                filename = "credentials_backup.json"

            try:
                import json

                with open(filename, "w") as f:
                    json.dump(credentials, f, indent=2)
                print(f"✅ Credentials exported to '{filename}'")
                print("⚠️ This file contains unencrypted credentials - handle securely!")
            except Exception as e:
                print(f"❌ Export failed: {e}")

    def check_status(self):
        """Check credential and security status.

        Returns:
            bool: True if all security checks pass, False otherwise
        """
        print("\n" + "=" * 50)
        print("         SECURITY STATUS")
        print("=" * 50)  # Check dependencies

        security_ok = True

        try:
            import cryptography

            crypto_version = cryptography.__version__
            crypto_status = f"✅ Installed (v{crypto_version})"
        except ImportError:
            crypto_status = "❌ Not installed (required for encryption)"
            security_ok = False

        try:
            import keyring

            keyring_status = f"✅ Installed"
            try:
                # Try to get version if available
                keyring_version = getattr(keyring, "__version__", "unknown")
                if keyring_version != "unknown":
                    keyring_status = f"✅ Installed (v{keyring_version})"
            except:
                pass
        except ImportError:
            keyring_status = "❌ Not installed (required for secure key storage)"
            security_ok = False

        print(f"🔐 Security Dependencies:")
        print(f"  - cryptography: {crypto_status}")
        print(f"  - keyring: {keyring_status}")

        # Check if alt keyring is needed (for Linux/macOS)
        if os.name != "nt" and "✅" in keyring_status:
            try:
                import keyrings.alt

                alt_status = "✅ Installed"
                try:
                    alt_version = getattr(keyrings.alt, "__version__", "unknown")
                    if alt_version != "unknown":
                        alt_status = f"✅ Installed (v{alt_version})"
                except:
                    pass
            except ImportError:
                alt_status = "⚠️ Not installed (recommended for Linux/macOS)"
            print(f"  - keyrings.alt: {alt_status}")

        # Show installation instructions if any dependencies are missing
        if "❌" in crypto_status or "❌" in keyring_status:
            print("\n📋 Installation Instructions:")
            print("  Run: pip install cryptography keyring")
            print("  - OR -")
            print("  Run: pip install -r requirements.txt")

            if os.name != "nt":
                print("\n  For Linux/macOS users:")
                print("  Run: pip install keyrings.alt")
                print(
                    "  Some Linux distros may require: sudo apt-get install python3-dbus"
                )

        # Check SecurityManager status
        print(
            f"🔐 Security Manager: {'✅ Available' if SECURITY_AVAILABLE else '❌ Not Available'}"
        )

        if not SECURITY_AVAILABLE:
            security_ok = False

        # Check credentials
        credentials = self.security_manager.decrypt_credentials()
        if credentials:
            print(f"🗝️  Stored Credentials: ✅ {len(credentials)} credentials found")

            # Get credential types from configuration
            required_creds, optional_creds = self._load_credential_types()
            all_configured_creds = set(required_creds.keys()) | set(
                optional_creds.keys()
            )

            # Check for required credentials
            missing = [
                cred for cred in required_creds.keys() if cred not in credentials
            ]

            if missing:
                print(f"⚠️  Missing Required: {', '.join(missing)}")
                security_ok = False
            else:
                print(f"✅ Required Credentials: All present")

            # Check for optional credentials
            present_optional = [
                cred for cred in optional_creds.keys() if cred in credentials
            ]
            if present_optional:
                print(f"🔹 Optional Credentials: {', '.join(present_optional)}")

            # Check for credentials not in configuration
            unknown_creds = [
                cred for cred in credentials.keys() if cred not in all_configured_creds
            ]
            if unknown_creds:
                print(f"ℹ️  Additional Credentials: {', '.join(unknown_creds)}")
        else:
            print(f"🗝️  Stored Credentials: ❌ None found")
            security_ok = False

        # Check encryption status
        try:
            test_data = {"test": "value"}
            encrypted = self.security_manager.encrypt_credentials(test_data)
            self.security_manager.delete_credentials()  # Clean up test
            print(f"🔒 Encryption Test: {'✅ Working' if encrypted else '❌ Failed'}")
            if not encrypted:
                security_ok = False
        except Exception as e:
            print(f"🔒 Encryption Test: ❌ Error - {e}")
            security_ok = False

        return security_ok

    def edit_credential_types(self):
        """Edit credential types configuration."""
        print("\n" + "=" * 50)
        print("    CREDENTIAL TYPES CONFIGURATION")
        print("=" * 50)

        cred_types_file = Path(__file__).parent / "credential_types.json"

        # Load current configuration
        if cred_types_file.exists():
            try:
                with open(cred_types_file, "r") as f:
                    cred_types = json.load(f)
                required_creds = cred_types.get("required_credentials", {})
                optional_creds = cred_types.get("optional_credentials", {})
                print(f"✅ Loaded credential types from configuration file")
            except Exception as e:
                print(f"⚠️ Error loading credential types: {e}, using defaults")
                required_creds, optional_creds = self._load_credential_types()
        else:
            print(f"⚠️ Configuration file not found, creating new one")
            required_creds, optional_creds = self._load_credential_types()

        print("\nCurrent Configuration:")
        print("Required Credentials:")
        for key, desc in required_creds.items():
            print(f"  {key}: {desc}")

        print("\nOptional Credentials:")
        for key, desc in optional_creds.items():
            print(f"  {key}: {desc}")

        print("\nOptions:")
        print("  1. Add/Edit Required Credential")
        print("  2. Add/Edit Optional Credential")
        print("  3. Move Credential (Required ↔ Optional)")
        print("  4. Remove Credential")
        print("  5. Save and Return")

        while True:
            choice = input("\nChoice (1-5): ").strip()

            if choice == "1":
                # Add/Edit Required Credential
                key = input("Credential Key (e.g., API_KEY_NAME): ").strip().upper()
                if not key:
                    print("❌ Key cannot be empty")
                    continue

                desc = input(f"Description for {key}: ").strip()
                if not desc:
                    desc = f"{key} (required)"

                required_creds[key] = desc
                print(f"✅ Added/Updated required credential: {key}")

            elif choice == "2":
                # Add/Edit Optional Credential
                key = input("Credential Key (e.g., API_KEY_NAME): ").strip().upper()
                if not key:
                    print("❌ Key cannot be empty")
                    continue

                desc = input(f"Description for {key}: ").strip()
                if not desc:
                    desc = f"{key} (optional)"

                optional_creds[key] = desc
                print(f"✅ Added/Updated optional credential: {key}")

            elif choice == "3":
                # Move Credential
                print("\nAvailable Credentials:")
                print("Required:")
                for i, (key, _) in enumerate(required_creds.items(), 1):
                    print(f"  {i}. {key}")

                print("Optional:")
                for i, (key, _) in enumerate(
                    optional_creds.items(), len(required_creds) + 1
                ):
                    print(f"  {i}. {key}")

                try:
                    idx = int(input("\nSelect credential number to move: ").strip())
                    if 1 <= idx <= len(required_creds):
                        # Move from required to optional
                        key = list(required_creds.keys())[idx - 1]
                        desc = required_creds.pop(key)
                        optional_creds[key] = desc
                        print(f"✅ Moved {key} from Required to Optional")
                    elif (
                        len(required_creds)
                        < idx
                        <= len(required_creds) + len(optional_creds)
                    ):
                        # Move from optional to required
                        key = list(optional_creds.keys())[idx - len(required_creds) - 1]
                        desc = optional_creds.pop(key)
                        required_creds[key] = desc
                        print(f"✅ Moved {key} from Optional to Required")
                    else:
                        print("❌ Invalid selection")
                except (ValueError, IndexError):
                    print("❌ Invalid selection")

            elif choice == "4":
                # Remove Credential
                print("\nAvailable Credentials:")
                all_creds = []
                print("Required:")
                for i, (key, _) in enumerate(required_creds.items()):
                    all_creds.append(("required", key))
                    print(f"  {i+1}. {key}")

                print("Optional:")
                for i, (key, _) in enumerate(optional_creds.items()):
                    all_creds.append(("optional", key))
                    print(f"  {i+len(required_creds)+1}. {key}")

                try:
                    idx = int(input("\nSelect credential number to remove: ").strip())
                    if 1 <= idx <= len(all_creds):
                        cred_type, key = all_creds[idx - 1]
                        if cred_type == "required":
                            del required_creds[key]
                        else:
                            del optional_creds[key]
                        print(f"✅ Removed {key}")
                    else:
                        print("❌ Invalid selection")
                except (ValueError, IndexError):
                    print("❌ Invalid selection")

            elif choice == "5":
                # Save and return
                try:
                    # Create updated configuration
                    updated_config = {
                        "required_credentials": required_creds,
                        "optional_credentials": optional_creds,
                    }

                    # Save to file
                    with open(cred_types_file, "w") as f:
                        json.dump(updated_config, f, indent=4)

                    print(
                        f"✅ Saved credential types configuration to {cred_types_file}"
                    )
                    break
                except Exception as e:
                    print(f"❌ Error saving configuration: {e}")
            else:
                print("❌ Invalid choice, please select 1-5")

    def _confirm_action(self, action: str) -> bool:
        """Helper to confirm dangerous actions."""
        response = (
            input(f"Are you sure you want to {action}? (yes/no): ").strip().lower()
        )
        return response in ["yes", "y"]

    def run(self):
        """Main menu loop."""
        if not SECURITY_AVAILABLE:
            return False

        print("🔐 Ancestry Project - Unified Credential Manager")

        while True:
            try:
                self.display_main_menu()
                choice = input("Enter choice: ").strip()

                if choice == "0":
                    print("👋 Goodbye!")
                    break
                elif choice == "1":
                    self.view_credentials()
                elif choice == "2":
                    self.setup_credentials()
                elif choice == "3":
                    self.remove_credential()
                elif choice == "4":
                    self.delete_all_credentials()
                elif choice == "5":
                    self.setup_test_credentials()
                elif choice == "6":
                    self.import_from_env()
                elif choice == "7":
                    self.export_credentials()
                elif choice == "8":
                    self.check_status()
                elif choice == "9":
                    self.edit_credential_types()
                else:
                    print("❌ Invalid choice. Please try again.")
                if choice != "0":
                    input("\nPress Enter to continue...")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("Press Enter to continue...")

        return True

    def _load_credential_types(self) -> tuple[Dict[str, str], Dict[str, str]]:
        """Load credential types from configuration file.

        Returns:
            Tuple of (required_creds, optional_creds)
        """
        cred_types_file = Path(__file__).parent / "credential_types.json"
        try:
            if cred_types_file.exists():
                with open(cred_types_file, "r") as f:
                    cred_types = json.load(f)
                required_creds = cred_types.get("required_credentials", {})
                optional_creds = cred_types.get("optional_credentials", {})
                return required_creds, optional_creds
            else:
                # Fallback to default credential types if file doesn't exist
                print(
                    f"⚠️ Credential types configuration file not found, using defaults"
                )
                return (
                    {
                        "ANCESTRY_USERNAME": "Ancestry.com username/email",
                        "ANCESTRY_PASSWORD": "Ancestry.com password",
                    },
                    {
                        "DEEPSEEK_API_KEY": "DeepSeek AI API key (optional)",
                        "OPENAI_API_KEY": "OpenAI API key (optional)",
                    },
                )
        except Exception as e:
            print(f"⚠️ Error loading credential types: {e}, using defaults")
            # Fallback to default credential types if there's an error
            return (
                {
                    "ANCESTRY_USERNAME": "Ancestry.com username/email",
                    "ANCESTRY_PASSWORD": "Ancestry.com password",
                },
                {
                    "DEEPSEEK_API_KEY": "DeepSeek AI API key (optional)",
                    "OPENAI_API_KEY": "OpenAI API key (optional)",
                },
            )

    def _save_credential(self, key: str, value: str, description: str) -> bool:
        """Save a credential to the secure store.

        Args:
            key: Credential key
            value: Credential value
            description: Human-readable description

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            if not key or not value:
                return False

            # Get existing credentials
            credentials = self.security_manager.decrypt_credentials() or {}

            # Add or update credential
            credentials[key] = value

            # Save back to secure store
            return self.security_manager.encrypt_credentials(credentials)
        except Exception as e:
            print(f"❌ Error saving credential: {e}")
            return False

    def _save_credential_types(
        self, required_creds: Dict[str, str], optional_creds: Dict[str, str]
    ) -> bool:
        """Save credential types to configuration file.

        Args:
            required_creds: Dictionary of required credentials
            optional_creds: Dictionary of optional credentials

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            cred_types_file = Path(__file__).parent / "credential_types.json"
            cred_types = {
                "required_credentials": required_creds,
                "optional_credentials": optional_creds,
            }

            with open(cred_types_file, "w") as f:
                json.dump(cred_types, f, indent=4)
            return True
        except Exception as e:
            print(f"❌ Error saving credential types: {e}")
            return False


def credentials_module_tests() -> bool:
    """Comprehensive test suite for credentials.py"""
    import tempfile
    import json
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Unified Credential Manager", "credentials.py")
    suite.start_suite()

    # Create temporary directory for test files
    test_dir = tempfile.TemporaryDirectory()
    test_dir_path = Path(test_dir.name)

    # Helper functions for test setup
    def create_test_credential_file(valid=True):
        """Create a temporary credential types file for testing"""
        file_path = test_dir_path / "test_credential_types.json"

        if valid:
            content = {
                "required_credentials": {"TEST_REQUIRED": "Test required credential"},
                "optional_credentials": {"TEST_OPTIONAL": "Test optional credential"},
            }
        else:
            # Invalid JSON structure
            content = {"invalid_structure": {"TEST_BAD": "Test bad credential"}}

        with open(file_path, "w") as f:
            json.dump(content, f)
        return file_path

    def test_security_availability():
        """Test that security components are available."""
        assert SECURITY_AVAILABLE, "Security dependencies should be available"
        assert SecurityManager is not None, "SecurityManager should be importable"

    def test_manager_initialization():
        """Test credential manager initialization."""
        if SECURITY_AVAILABLE:
            manager = UnifiedCredentialManager()
            assert (
                manager.security_manager is not None
            ), "SecurityManager should be initialized"

    def test_manager_initialization_with_security_unavailable():
        """Test that initialization fails properly when security is unavailable."""
        # Temporarily set SECURITY_AVAILABLE to False
        global SECURITY_AVAILABLE
        original_value = SECURITY_AVAILABLE
        SECURITY_AVAILABLE = False

        try:
            # Should raise ImportError
            raised = False
            try:
                UnifiedCredentialManager()
            except ImportError:
                raised = True
            assert raised, "Should raise ImportError when security is unavailable"
        finally:
            # Restore original value
            SECURITY_AVAILABLE = original_value

    def test_menu_methods():
        """Test that all menu methods exist."""
        if SECURITY_AVAILABLE:
            manager = UnifiedCredentialManager()
            methods = [
                "view_credentials",
                "setup_credentials",
                "remove_credential",
                "delete_all_credentials",
                "setup_test_credentials",
                "export_credentials",
                "check_status",
                "edit_credential_types",
            ]

            for method in methods:
                assert hasattr(manager, method), f"Manager should have {method} method"
                assert callable(
                    getattr(manager, method)
                ), f"{method} should be callable"

    def test_load_credential_types_with_valid_file():
        """Test loading credential types from a valid file."""
        if SECURITY_AVAILABLE:
            manager = UnifiedCredentialManager()

            # Create a valid test credential file
            test_file = create_test_credential_file(valid=True)

            # Mock Path's parent/__truediv__ combination that's used in _load_credential_types
            with patch.object(Path, "__truediv__", return_value=test_file):
                # Mock exists to return True
                with patch.object(Path, "exists", return_value=True):
                    # Load credential types
                    required, optional = manager._load_credential_types()

                    # Verify loaded credentials
                    assert (
                        "TEST_REQUIRED" in required
                    ), "Should load required credentials"
                    assert (
                        "TEST_OPTIONAL" in optional
                    ), "Should load optional credentials"

    def test_load_credential_types_with_missing_file():
        """Test loading credential types when the file is missing."""
        if SECURITY_AVAILABLE:
            manager = UnifiedCredentialManager()

            # Mock file operations
            with patch("pathlib.Path.exists", return_value=False):

                # Load credential types
                required, optional = manager._load_credential_types()

                # Verify default credentials are used
                assert len(required) > 0, "Should use default required credentials"
                assert len(optional) > 0, "Should use default optional credentials"
                assert (
                    "ANCESTRY_USERNAME" in required
                ), "Default credentials should include ANCESTRY_USERNAME"

    def test_load_credential_types_with_invalid_json():
        """Test loading credential types when the file has invalid JSON."""
        if SECURITY_AVAILABLE:
            manager = UnifiedCredentialManager()

            # Create an invalid test credential file
            test_file = test_dir_path / "invalid.json"
            with open(test_file, "w") as f:
                f.write("{invalid json")

            # Mock file operations
            with patch(
                "pathlib.Path.open",
                side_effect=lambda *args, **kwargs: open(
                    test_file, *args[1:], **kwargs
                ),
            ), patch("pathlib.Path.exists", return_value=True):

                # Load credential types
                required, optional = manager._load_credential_types()

                # Verify default credentials are used
                assert (
                    len(required) > 0
                ), "Should use default required credentials on JSON error"
                assert (
                    len(optional) > 0
                ), "Should use default optional credentials on JSON error"

    def test_load_credential_types_with_invalid_structure():
        """Test loading credential types when the file has invalid structure."""
        if SECURITY_AVAILABLE:
            manager = UnifiedCredentialManager()

            # Create a test credential file with invalid structure
            test_file = create_test_credential_file(valid=False)

            # Mock file operations
            with patch(
                "pathlib.Path.open",
                side_effect=lambda *args, **kwargs: open(
                    test_file, *args[1:], **kwargs
                ),
            ), patch("pathlib.Path.exists", return_value=True):

                # Load credential types
                required, optional = manager._load_credential_types()
                # Verify default credentials are used when structure is invalid
                assert (
                    "ANCESTRY_USERNAME" in required
                ), "Should use default required credentials on structure error"
                assert (
                    "OPENAI_API_KEY" in optional
                ), "Should use default optional credentials on structure error"

    def test_edit_credential_types_error_handling():
        """Test error handling in edit_credential_types."""
        if SECURITY_AVAILABLE:
            manager = UnifiedCredentialManager()

            # Mock open to simulate file access error
            with patch(
                "builtins.open", side_effect=PermissionError("Permission denied")
            ):

                # This should handle the error gracefully without crashing
                success = manager._save_credential_types({}, {})
                assert not success, "Should return False when file cannot be written"

    def test_check_status_with_missing_credentials():
        """Test check_status when credentials are missing."""
        if SECURITY_AVAILABLE:
            manager = UnifiedCredentialManager()

            # Mock the decrypt_credentials method
            with patch.object(
                manager.security_manager, "decrypt_credentials", return_value=None
            ):

                # Call check_status and verify it doesn't crash
                status = manager.check_status()
                assert status is not None, "check_status should not return None"
                assert (
                    not status
                ), "Status should be False when required credentials are missing"

    def test_setup_credentials_permission_error():
        """Test setup_credentials handling of permission errors."""
        if SECURITY_AVAILABLE:
            manager = UnifiedCredentialManager()

            # Mock encrypt_credentials to simulate error
            with patch.object(
                manager.security_manager,
                "encrypt_credentials",
                side_effect=PermissionError("Test permission error"),
            ):

                # This should handle the error gracefully
                success = manager._save_credential(
                    "TEST_CRED", "test_value", "Test credential"
                )
                assert not success, "Should return False when permission error occurs"

    with suppress_logging():
        suite.run_test(
            "SECURITY_AVAILABLE, SecurityManager import",
            test_security_availability,
            "Security dependencies are properly available and importable",
            "Test security manager import and availability flag",
            "Security components are available for credential management",
        )

        suite.run_test(
            "UnifiedCredentialManager initialization",
            test_manager_initialization,
            "Credential manager initializes with working SecurityManager instance",
            "Test UnifiedCredentialManager constructor and SecurityManager setup",
            "Manager initializes successfully with all required components",
        )

        suite.run_test(
            "UnifiedCredentialManager initialization with security unavailable",
            test_manager_initialization_with_security_unavailable,
            "Credential manager fails properly when security is unavailable",
            "Test UnifiedCredentialManager handling of missing security dependencies",
            "Manager properly raises ImportError when security is unavailable",
        )

        suite.run_test(
            "Menu method availability and callability",
            test_menu_methods,
            "All credential management methods are available and callable",
            "Test existence and callability of all manager methods",
            "All required methods exist and are properly callable",
        )

        suite.run_test(
            "Load credential types from valid file",
            test_load_credential_types_with_valid_file,
            "Credential types are properly loaded from a valid file",
            "Test loading credential types from a valid JSON file",
            "Credential types are correctly parsed from a valid configuration file",
        )

        suite.run_test(
            "Load credential types with missing file",
            test_load_credential_types_with_missing_file,
            "Default credential types are used when file is missing",
            "Test fallback to defaults when credential types file is missing",
            "System gracefully falls back to defaults when configuration file is not found",
        )

        suite.run_test(
            "Load credential types with invalid JSON",
            test_load_credential_types_with_invalid_json,
            "Default credential types are used when JSON is invalid",
            "Test handling of invalid JSON in credential types file",
            "System gracefully falls back to defaults when JSON cannot be parsed",
        )

        suite.run_test(
            "Load credential types with invalid structure",
            test_load_credential_types_with_invalid_structure,
            "Default credential types are used when structure is invalid",
            "Test handling of invalid structure in credential types file",
            "System gracefully falls back to defaults when JSON structure is incorrect",
        )

        suite.run_test(
            "Edit credential types error handling",
            test_edit_credential_types_error_handling,
            "Errors during credential type editing are properly handled",
            "Test handling of permission errors when saving credential types",
            "System gracefully handles file permission errors during configuration updates",
        )

        suite.run_test(
            "Check status with missing credentials",
            test_check_status_with_missing_credentials,
            "Status check properly identifies missing credentials",
            "Test credential status checking with missing credentials",
            "System correctly reports when required credentials are missing",
        )

        suite.run_test(
            "Setup credentials permission error handling",
            test_setup_credentials_permission_error,
            "Permission errors during credential setup are properly handled",
            "Test handling of permission errors when saving credentials",
            "System gracefully handles permission errors during credential updates",
        )

    # Clean up temporary directory
    test_dir.cleanup()

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive credentials tests using standardized TestSuite format."""
    return credentials_module_tests()


def main():
    """Main entry point."""
    # Support non-interactive .env import
    if len(sys.argv) > 1 and sys.argv[1] == "--import-env":
        if not SECURITY_AVAILABLE:
            print("\n❌ Security dependencies are required for credential management.")
            UnifiedCredentialManager.check_and_install_dependencies()
            return False
        try:
            manager = UnifiedCredentialManager()
            # Use default .env path, non-interactive
            env_file = ".env"
            print("\n🔐 Importing credentials from .env (non-interactive)...")
            manager.import_from_env = manager.import_from_env.__func__.__get__(manager)
            # Patch input to always use default .env and merge
            import builtins

            orig_input = builtins.input

            def fake_input(prompt=""):
                if "path to .env" in prompt:
                    return ""
                if "Choice (m/o/r/c):" in prompt:
                    return "o"  # Overwrite existing with .env values
                if "Are you sure you want to" in prompt:
                    return "yes"
                return ""

            builtins.input = fake_input
            try:
                manager.import_from_env()
            finally:
                builtins.input = orig_input
            print("\n✅ .env import complete.")
            return True
        except Exception as e:
            print(f"❌ Error during .env import: {e}")
            return False

    # Auto-detect if being run as part of the test harness
    if os.environ.get("RUNNING_ANCESTRY_TESTS") == "1":
        print("🔐 Auto-detected test execution from test harness...")
        return run_comprehensive_tests()

    # Time-based auto-test detection (add a short timeout to avoid hanging in test suites)
    if time.time() > 0 and os.path.basename(sys.argv[0]) == "credentials.py":
        if not sys.stdin.isatty() or os.environ.get("CI") == "true":
            print("🔐 Auto-detected non-interactive environment, running tests...")
            return run_comprehensive_tests()

    if not SECURITY_AVAILABLE:
        print("\n" + "=" * 60)
        print("        SECURITY DEPENDENCIES MISSING")
        print("=" * 60)
        print("\n❌ Security dependencies are required for credential management.")
        print("Would you like to install them now?")

        if UnifiedCredentialManager.check_and_install_dependencies():
            print("\n✅ Dependencies installed successfully!")
            print("Please restart the program: python credentials.py")
        else:
            print("\n⚠️ Dependencies are still missing. Please install them manually:")
            print("  pip install cryptography keyring")
            print("  - OR -")
            print("  pip install -r requirements.txt")
            print("\nFor more information, see SECURITY_STREAMLINED.md")

        return False

    try:
        manager = UnifiedCredentialManager()
        return manager.run()
    except Exception as e:
        print(f"❌ Failed to start credential manager: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
