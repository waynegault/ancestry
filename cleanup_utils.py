#!/usr/bin/env python3
"""
Cleanup script to remove unreachable code from utils.py
"""
import re

def clean_utils():
    """Clean the utils.py file by removing unreachable code after return statement."""
    
    with open('utils.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the return statement in run_comprehensive_tests
    return_pattern = r'return run_unified_tests\(__name__, utils_module_tests\)'
    return_match = re.search(return_pattern, content)
    
    if not return_match:
        print("Return statement not found!")
        return False
    
    # Find the position after the return statement
    return_end = return_match.end()
    
    # Find the next legitimate section (auto_register_module)
    remaining_content = content[return_end:]
    
    # Look for the auto_register_module call
    auto_reg_pattern = r'auto_register_module\(globals\(\), __name__\)'
    auto_reg_match = re.search(auto_reg_pattern, remaining_content)
    
    if not auto_reg_match:
        print("auto_register_module not found!")
        return False
    
    # Find the standalone test block
    test_block_pattern = r'# ==============================================\n# Standalone Test Block\n# =============================================='
    test_block_match = re.search(test_block_pattern, remaining_content)
    
    if not test_block_match:
        print("Test block not found!")
        return False
    
    # Get the content from the test block onwards
    test_block_start = return_end + test_block_match.start()
    
    # Construct clean content: everything up to return + module registration + test block
    clean_content = (
        content[:return_end] +
        "\n\n\n# ==============================================\n" +
        "# Module Registration\n" +
        "# ==============================================\n\n" +
        "# Auto-register module functions for optimized access\n" +
        "auto_register_module(globals(), __name__)\n\n\n" +
        content[test_block_start:]
    )
    
    # Write the cleaned content back
    with open('utils.py', 'w', encoding='utf-8') as f:
        f.write(clean_content)
    
    print(f"✅ Cleaned utils.py - removed {len(content) - len(clean_content)} characters of unreachable code")
    return True

if __name__ == "__main__":
    clean_utils()
