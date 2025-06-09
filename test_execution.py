#!/usr/bin/env python3
"""Simple test script to verify Python execution"""

import sys
import os

print("Test script started")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Script path: {__file__}")
print("Test script completed successfully")

# Exit with code 0 for success
sys.exit(0)
