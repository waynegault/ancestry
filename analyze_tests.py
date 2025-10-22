"""Analyze test coverage across all Python files."""
import re
from pathlib import Path

files_with_tests = []
for filepath in Path('.').rglob('*.py'):
    if '.venv' in str(filepath) or 'archive' in str(filepath):
        continue
    try:
        content = filepath.read_text(encoding='utf-8')
        test_funcs = re.findall(r'def (_test_\w+|test_\w+)\(', content)
        if test_funcs:
            files_with_tests.append((str(filepath), len(test_funcs), test_funcs))
    except Exception:
        pass

# Sort by test count descending
files_with_tests.sort(key=lambda x: x[1], reverse=True)

print('Files with tests (sorted by test count):')
print('=' * 100)
for filepath, count, test_names in files_with_tests:
    print(f'{count:3d} tests | {filepath}')
    # Show first 3 test names as sample
    for test_name in test_names[:3]:
        print(f'          - {test_name}')
    if len(test_names) > 3:
        print(f'          ... and {len(test_names) - 3} more')
print('=' * 100)
print(f'Total files with tests: {len(files_with_tests)}')
print(f'Total test functions: {sum(count for _, count, _ in files_with_tests)}')

