#!/usr/bin/env python3
"""
Script to standardize run_comprehensive_tests implementations across the codebase.
Converts manual function implementations to use create_standard_test_runner pattern.
"""

import re
from pathlib import Path
from typing import Optional

# Root directory
ROOT = Path(__file__).parent.parent

# Modules that need conversion (from analysis)
MODULES_TO_CONVERT = [
    # Root modules
    ('utils.py', 'utils_module_tests'),
    ('rate_limiter.py', None),  # Inline implementation
    ('main.py', 'main_module_tests'),
    ('dna_utils.py', None),  # Inline implementation
    ('connection_resilience.py', None),  # Inline implementation
    ('common_params.py', None),  # Inline implementation
    ('api_search_utils.py', 'api_search_utils_module_tests'),
    ('api_constants.py', None),  # Inline implementation
    ('ai_prompt_utils.py', 'ai_prompt_utils_module_tests'),
    ('ai_interface.py', 'ai_interface_module_tests'),
    ('action9_process_productive.py', 'action9_process_productive_module_tests'),
    ('action6_gather.py', 'action6_gather_module_tests'),

    # Core modules
    ('core/__main__.py', None),  # Inline implementation
    ('core/session_manager.py', None),  # Inline implementation
    ('core/registry_utils.py', None),  # Inline implementation
    ('core/progress_indicators.py', None),  # Inline implementation
    ('core/metrics_integration.py', None),  # Inline implementation
    ('core/metrics_collector.py', None),  # Inline implementation
    ('core/enhanced_error_recovery.py', None),  # Inline implementation
    ('core/cancellation.py', None),  # Inline implementation
    ('core/browser_manager.py', None),  # Inline implementation

    # Observability modules
    ('observability/metrics_registry.py', None),  # Inline implementation
    ('observability/metrics_exporter.py', None),  # Inline implementation
]


def find_module_test_function(content: str) -> Optional[str]:
    """Find the name of the module test function."""
    # Pattern 1: Simple wrapper - return module_tests()
    pattern1 = r'def run_comprehensive_tests\(\) -> bool:\s+"""[^"]*"""\s+return (\w+)\(\)'
    match = re.search(pattern1, content)
    if match:
        return match.group(1)

    # Pattern 2: Look for module_tests() function defined before run_comprehensive_tests
    pattern2 = r'def (\w*_module_tests)\(\) -> bool:'
    matches = list(re.finditer(pattern2, content))
    if matches:
        return matches[-1].group(1)  # Return last match

    return None


def check_has_import(content: str) -> bool:
    """Check if module already imports create_standard_test_runner."""
    return 'from test_utilities import' in content and 'create_standard_test_runner' in content


def add_import_to_module(content: str, module_path: Path) -> str:
    """Add import for create_standard_test_runner."""
    # Check if already has the import
    if check_has_import(content):
        return content

    # Find the right place to add import
    # Strategy: Add after test_framework import if it exists, otherwise after standard_imports

    if 'from test_framework import' in content:
        # Add after test_framework import
        content = re.sub(
            r'(from test_framework import[^\n]+)',
            r'\1\nfrom test_utilities import create_standard_test_runner',
            content,
            count=1
        )
    elif 'from standard_imports import' in content:
        # Add after standard_imports
        content = re.sub(
            r'(from standard_imports import[^\n]+)',
            r'\1\nfrom test_utilities import create_standard_test_runner',
            content,
            count=1
        )
    elif 'logger = setup_module' in content:
        # Add after logger setup
        content = re.sub(
            r'(logger = setup_module\(globals\(\), __name__\))',
            r'\1\n\n# Test utilities\nfrom test_utilities import create_standard_test_runner',
            content,
            count=1
        )
    else:
        print(f"  ⚠️  Could not find appropriate place to add import in {module_path.name}")
        return content

    return content


def convert_simple_wrapper(content: str, test_func_name: str) -> tuple[str, bool]:
    """Convert simple wrapper pattern: def run_comprehensive_tests() -> bool: return module_tests()"""
    pattern = r'def run_comprehensive_tests\(\) -> bool:\s+"""[^"]*"""\s+return ' + test_func_name + r'\(\)'

    replacement = f'# Use centralized test runner utility from test_utilities\nrun_comprehensive_tests = create_standard_test_runner({test_func_name})'

    new_content, count = re.subn(pattern, replacement, content)
    return new_content, count > 0


def convert_module(module_path: Path, test_func_name: Optional[str] = None) -> bool:
    """Convert a single module to use create_standard_test_runner."""
    if not module_path.exists():
        print(f"  ❌ File not found: {module_path}")
        return False

    print(f"\n📄 Processing: {module_path.relative_to(ROOT)}")

    # Read content
    content = module_path.read_text(encoding='utf-8')

    # Check if already converted
    if 'run_comprehensive_tests = create_standard_test_runner(' in content:
        print("  ✅ Already uses create_standard_test_runner")
        return True

    # Find test function name if not provided
    if test_func_name is None:
        test_func_name = find_module_test_function(content)
        if test_func_name:
            print(f"  🔍 Found test function: {test_func_name}")
        else:
            print("  ⚠️  Could not find test function - has inline implementation")
            return False

    # Add import if needed
    if not check_has_import(content):
        print("  📦 Adding import for create_standard_test_runner")
        content = add_import_to_module(content, module_path)
        if not check_has_import(content):
            print("  ❌ Failed to add import")
            return False

    # Convert the function
    print(f"  🔄 Converting to use create_standard_test_runner({test_func_name})")
    new_content, converted = convert_simple_wrapper(content, test_func_name)

    if not converted:
        print("  ❌ Failed to convert (pattern didn't match)")
        return False

    # Write back
    module_path.write_text(new_content, encoding='utf-8')
    print("  ✅ Successfully converted!")
    return True


def main() -> None:
    """Main conversion process."""
    print("🚀 Starting test runner standardization...")
    print("=" * 80)

    converted_count = 0
    skipped_count = 0
    failed_count = 0

    for module_file, test_func_name in MODULES_TO_CONVERT:
        module_path = ROOT / module_file

        try:
            if convert_module(module_path, test_func_name):
                converted_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed_count += 1

    print("\n" + "=" * 80)
    print("📊 Summary:")
    print(f"   ✅ Converted: {converted_count}")
    print(f"   ⏭️  Skipped: {skipped_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   📁 Total: {len(MODULES_TO_CONVERT)}")

    if failed_count == 0:
        print("\n🎉 All conversions completed successfully!")
    else:
        print(f"\n⚠️  {failed_count} modules need manual attention")


if __name__ == '__main__':
    main()
