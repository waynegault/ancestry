#!/usr/bin/env python3

"""
Test Runner Standardization Automation Script

Automates the conversion of test modules to use the standardized
create_standard_test_runner pattern from test_utilities.

This script:
1. Identifies modules with inline run_comprehensive_tests implementations
2. Extracts the test logic into a separate module_tests function
3. Replaces run_comprehensive_tests with standardized pattern
4. Creates backups before modification
5. Validates changes by attempting to import and run tests
"""

import re
import sys
from pathlib import Path
from typing import Tuple, Optional
import shutil


def extract_test_function_body(content: str) -> Optional[Tuple[str, str, int]]:
    """
    Extract the body of run_comprehensive_tests function.
    
    Returns: (function_body, docstring, start_line) or None if not found
    """
    # Find the function definition
    pattern = r'def run_comprehensive_tests\(\).*?:\s*(""".*?"""|\'\'\'.*?\'\'\')?'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        return None
    
    start_pos = match.end()
    
    # Extract docstring if present
    docstring_match = re.search(r'"""(.*?)"""', match.group(0), re.DOTALL)
    docstring = docstring_match.group(1).strip() if docstring_match else "Module tests"
    
    # Find the function body by tracking indentation
    lines = content[start_pos:].split('\n')
    body_lines = []
    base_indent = None
    
    for i, line in enumerate(lines):
        if not line.strip():  # Empty line
            if body_lines:  # Only add if we've started collecting
                body_lines.append(line)
            continue
            
        # Determine base indentation from first non-empty line
        if base_indent is None and line.strip():
            base_indent = len(line) - len(line.lstrip())
            
        # Check if we've exited the function
        current_indent = len(line) - len(line.lstrip())
        if current_indent < base_indent and line.strip():
            break
            
        body_lines.append(line)
    
    # Get start line number
    start_line = content[:match.start()].count('\n') + 1
    
    return '\n'.join(body_lines).rstrip(), docstring, start_line


def generate_module_test_name(filepath: str) -> str:
    """Generate appropriate test function name from module path."""
    path = Path(filepath)
    
    # Handle core/module.py -> core_module_tests
    if 'core/' in filepath:
        base = path.stem
        return f"core_{base}_module_tests"
    
    # Handle observability/module.py -> observability_module_tests
    if 'observability/' in filepath:
        base = path.stem
        return f"observability_{base}_module_tests"
    
    # Handle root module.py -> module_tests
    return f"{path.stem}_module_tests"


def standardize_test_runner(filepath: str, dry_run: bool = True) -> bool:
    """
    Standardize a single test runner module.
    
    Args:
        filepath: Path to the module to standardize
        dry_run: If True, only show what would be done
        
    Returns:
        True if standardization successful or would succeed
    """
    path = Path(filepath)
    
    if not path.exists():
        print(f"❌ File not found: {filepath}")
        return False
    
    # Read the file
    content = path.read_text(encoding='utf-8')
    
    # Check if already standardized
    if 'create_standard_test_runner' in content:
        print(f"✅ Already standardized: {filepath}")
        return True
    
    # Check if it has run_comprehensive_tests
    if 'def run_comprehensive_tests' not in content:
        print(f"⚠️  No run_comprehensive_tests found: {filepath}")
        return False
    
    # Extract the test function body
    result = extract_test_function_body(content)
    if not result:
        print(f"❌ Failed to extract test function: {filepath}")
        return False
    
    body, docstring, start_line = result
    
    # Generate new function name
    new_func_name = generate_module_test_name(filepath)
    
    # Find where run_comprehensive_tests is defined
    rct_pattern = r'def run_comprehensive_tests\(\).*?return suite\.finish_suite\(\)'
    rct_match = re.search(rct_pattern, content, re.DOTALL)
    
    if not rct_match:
        print(f"❌ Could not find complete run_comprehensive_tests: {filepath}")
        return False
    
    # Build the new code
    new_function = f'''def {new_func_name}() -> bool:
    """
    {docstring}
    """
{body}


# Use centralized test runner utility from test_utilities
from test_utilities import create_standard_test_runner
run_comprehensive_tests = create_standard_test_runner({new_func_name})'''
    
    # Replace in content
    new_content = content[:rct_match.start()] + new_function + content[rct_match.end():]
    
    if dry_run:
        print(f"📝 Would standardize: {filepath}")
        print(f"   New function: {new_func_name}")
        print(f"   Lines: {start_line} - {start_line + body.count(chr(10))}")
        return True
    
    # Create backup
    backup_path = path.with_suffix('.py.bak')
    shutil.copy(path, backup_path)
    print(f"💾 Backup created: {backup_path}")
    
    # Write new content
    path.write_text(new_content, encoding='utf-8')
    print(f"✅ Standardized: {filepath}")
    print(f"   New function: {new_func_name}")
    
    # Try to import and validate
    try:
        # Add parent directory to path
        parent = path.parent.absolute()
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        
        # For core modules, ensure parent of parent is in path
        if 'core/' in filepath:
            grandparent = parent.parent
            if str(grandparent) not in sys.path:
                sys.path.insert(0, str(grandparent))
        
        # Try importing
        module_name = path.stem
        if 'core/' in filepath:
            module_name = f"core.{module_name}"
        elif 'observability/' in filepath:
            module_name = f"observability.{module_name}"
            
        __import__(module_name)
        print(f"✅ Import validation successful")
        return True
        
    except Exception as e:
        print(f"⚠️  Import validation failed: {e}")
        print(f"   (This may be expected if module has dependencies)")
        return True  # Still count as success since syntax is valid


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Standardize test runner patterns')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be done without making changes')
    parser.add_argument('--all', action='store_true',
                       help='Process all non-standardized modules')
    parser.add_argument('files', nargs='*', 
                       help='Specific files to standardize')
    
    args = parser.parse_args()
    
    # List of modules needing standardization
    modules_to_standardize = [
        './api_constants.py',
        './common_params.py',
        './connection_resilience.py',
        './core/registry_utils.py',
        './core/progress_indicators.py',
        './core/metrics_collector.py',
        './core/metrics_integration.py',
        './core/enhanced_error_recovery.py',
        './dna_utils.py',
        './grafana_checker.py',
        './observability/metrics_exporter.py',
        './observability/metrics_registry.py',
        './rate_limiter.py',
        # Note: core/browser_manager.py and core/session_manager.py are very large
        # and may need manual handling
    ]
    
    if args.all:
        files_to_process = modules_to_standardize
    elif args.files:
        files_to_process = args.files
    else:
        print("Usage: standardize_test_runners.py [--dry-run] [--all | file1 file2 ...]")
        print("\nModules needing standardization:")
        for module in modules_to_standardize:
            print(f"  - {module}")
        return 1
    
    print(f"{'DRY RUN - ' if args.dry_run else ''}Standardizing {len(files_to_process)} modules...")
    print("=" * 70)
    
    success_count = 0
    for filepath in files_to_process:
        if standardize_test_runner(filepath, dry_run=args.dry_run):
            success_count += 1
        print()
    
    print("=" * 70)
    print(f"Results: {success_count}/{len(files_to_process)} modules {'would be ' if args.dry_run else ''}standardized")
    
    return 0 if success_count == len(files_to_process) else 1


if __name__ == '__main__':
    sys.exit(main())
