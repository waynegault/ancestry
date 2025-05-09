#!/usr/bin/env python3

# action6.py

"""
action6.py - Compatibility module for action6_gather.py

This module imports and re-exports the necessary functions from action6_gather.py
to maintain backward compatibility with code that imports from action6.py.
"""

# Import the main function from action6_gather.py
from action6_gather import coord  # Main function that orchestrates DNA match gathering

# Import self_test for the __main__ block
from action6_gather import self_test

# Make sure coord is exported from this module
__all__ = ["coord", "self_test"]

# If this file is run directly, run the self-test from action6_gather
if __name__ == "__main__":
    print("Running Action 6 (Gather DNA Matches) self-test via compatibility module...")
    success = self_test()
    import sys

    sys.exit(0 if success else 1)

# End of action6.py
