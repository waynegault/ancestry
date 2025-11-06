#!/usr/bin/env python
"""Remove all progress bar references from action6_gather.py"""
import re
from pathlib import Path

file_path = Path("action6_gather.py")
content = file_path.read_text(encoding="utf-8")

# Remove all progress_bar function parameters and references
patterns = [
    # Remove progress_bar parameters from function signatures
    (r',\s*progress_bar:\s*Optional\[tqdm\](?:\s*=\s*None)?', ''),
    (r'progress_bar:\s*Optional\[tqdm\](?:\s*=\s*None)?,?\s*', ''),
    (r',\s*progress_bar(?=[,\)])', ''),
    (r'progress_bar,\s*', ''),
    (r'\bprogress_bar\s*,', ''),

    # Remove progress_bar arguments in function calls
    (r',\s*progress_bar\s*=\s*\w+', ''),
    (r'progress_bar\s*=\s*\w+,?\s*', ''),

    # Remove conditional progress_bar checks and updates
    (r'\n\s*if\s+progress_bar[^\n]*\n(?:\s+[^\n]+\n)*?\s+pass', ''),
    (r'\n\s*if\s+(?:not\s+)?progress_bar[^\n]*\n(?:\s+[^\n]+\n)+', '\n'),

    # Remove progress_bar assignments
    (r'progress_bar\s*=\s*tqdm\([^)]*\)\s*\n', ''),
    (r'with\s+logging_redirect_tqdm\(\):\s*\n', ''),

    # Remove progress_bar method calls
    (r'\s*progress_bar\.(update|set_postfix|set_description|refresh|close)\([^)]*\)\s*\n?', ''),

    # Remove try/except blocks that only handle progress_bar
    (r'\n\s*try:\s*\n(?:\s+[^\n]+progress_bar[^\n]+\n)+\s+except[^\n]+pbar[^\n]+:\s*\n\s+pass', ''),

    # Remove remaining orphaned commas
    (r',\s*,', ','),
    (r'\(\s*,', '('),
    (r',\s*\)', ')'),
]

for pattern, replacement in patterns:
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

# Write back
file_path.write_text(content, encoding="utf-8")
print(f"Cleaned up progress_bar references from {file_path}")
