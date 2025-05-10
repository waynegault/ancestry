#!/usr/bin/env python3

"""
run_action7_from_main.py - Script to run Action 7 (Inbox Processor) directly from main.py

This script imports the main module and calls the action7 function directly.
"""

import sys
import time
from typing import Optional

def run_action7_from_main():
    """
    Runs Action 7 (Inbox Processor) directly from main.py.
    
    Returns:
        bool: True if the action completed successfully, False otherwise.
    """
    try:
        # Import main module
        import main
        
        # Call action7 function directly
        print("\n=== Running Action 7 from main.py ===\n")
        result = main.action7()
        
        if result:
            print("\n✅ Action 7 completed successfully.")
        else:
            print("\n❌ Action 7 failed. Check the logs for details.")
        
        return result
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return False
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        return False

if __name__ == "__main__":
    # Run Action 7
    success = run_action7_from_main()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)
