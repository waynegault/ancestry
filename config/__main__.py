# Allows running `python -m config` without error.
# You can add test or main logic here if needed.

import sys

# Add parent directory to path for core_imports
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

if __name__ == "__main__":
    try:
        # Import the config package modules
        # Import the config package modules (for availability check)
        from config.config_manager import ConfigManager  # noqa: F401
        from config.config_schema import ConfigSchema  # noqa: F401
        from config.credential_manager import CredentialManager  # noqa: F401

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
