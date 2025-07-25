# Allows running `python -m core` without error.
# You can add test or main logic here if needed.

import sys
import os

# Add parent directory to path for core_imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

if __name__ == "__main__":
    try:
        # Import the core package modules
        from core.session_manager import SessionManager
        from core.database_manager import DatabaseManager
        from core.browser_manager import BrowserManager
        from core.api_manager import APIManager
        from core.error_handling import ErrorHandler
        from core.dependency_injection import DIContainer

        print("Core Package - Modular Session Management Architecture")
        print("Version: 2.0.0")
        print("Available modules: session_manager, database_manager, browser_manager,")
        print("                  api_manager, error_handling, dependency_injection,")
        print("                  registry_utils, session_validator")
        print("Note: This is a package init file. Import individual modules as needed.")
        print("\nExample usage:")
        print("  from core.session_manager import SessionManager")
        print("  from core.database_manager import DatabaseManager")
        print("  from core.browser_manager import BrowserManager")

    except ImportError as e:
        print(f"Error importing core modules: {e}")
        print("Note: Run individual modules directly for testing.")

    # Optionally, call your test runner or main logic here
    pass
