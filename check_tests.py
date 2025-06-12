#!/usr/bin/env python3

import os
import re

# Directory to search
base_dir = r"c:\Users\wayne\GitHub\Python\Projects\Ancestry"

# Pattern to find files with run_comprehensive_tests
pattern = r"def run_comprehensive_tests\(\) -> bool:"

# Files to check
files_to_check = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".py") and not file.startswith("check_tests"):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    if re.search(pattern, content):
                        files_to_check.append(filepath)
            except:
                pass

print(f"Found {len(files_to_check)} files with run_comprehensive_tests:")
for file in sorted(files_to_check):
    print(f"  {os.path.basename(file)}")
