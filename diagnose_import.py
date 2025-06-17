import sys
import os

# Print current working directory
print(f"Current directory: {os.getcwd()}")

# Print Python path
print(f"Python path: {sys.path}")

# Try to import test_framework
try:
    import test_framework

    print("Successfully imported test_framework")
    print(f"Module location: {test_framework.__file__}")
except ImportError as e:
    print(f"Failed to import test_framework: {e}")
