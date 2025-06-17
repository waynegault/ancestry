import os
import glob

# Find Python files in the main project directory only
python_files = glob.glob("*.py")
print(f"Found {len(python_files)} Python files in main directory:")
for file in sorted(python_files):
    print(f'"{file}",')
