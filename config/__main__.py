# Allows running `python -m config` without error.
# You can add test or main logic here if needed.

import sys
import os

# Add parent directory to path for core_imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

if __name__ == "__main__":
    try:
        # Import the config package modules
        from config.config_manager import ConfigManager
        from config.credential_manager import CredentialManager
        from config.config_schema import ConfigSchema

        print("Configuration Package - Enhanced Credential and Config Management")
        print("Version: 2.0.0")
        print("Available modules: config_manager, credential_manager, config_schema")
        print("Note: This is a package init file. Import individual modules as needed.")
        print("\nExample usage:")
        print("  from config.credential_manager import CredentialManager")
        print("  from config.config_manager import ConfigManager")

    except ImportError as e:
        print(f"Error importing config modules: {e}")
        print("Note: Run individual modules directly for testing.")

    # Optionally, call your test runner or main logic here
    pass
