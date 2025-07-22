#!/usr/bin/env python3
"""
Reconstruct utils.py by copying everything up to the return statement,
then adding proper module registration and standalone test block.
"""

def reconstruct_utils():
    """Reconstruct utils.py with correct structure."""
    
    with open('utils_backup.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the line with the return statement
    return_line_idx = None
    for i, line in enumerate(lines):
        if 'return run_unified_tests(__name__, utils_module_tests)' in line:
            return_line_idx = i
            break
    
    if return_line_idx is None:
        print("Return statement not found!")
        return False
    
    # Keep everything up to and including the return statement
    good_lines = lines[:return_line_idx + 1]
    
    # Add the proper module registration and standalone test block
    clean_ending = [
        "\n\n\n",
        "# ==============================================\n",
        "# Module Registration\n", 
        "# ==============================================\n",
        "\n",
        "# Auto-register module functions for optimized access\n",
        "auto_register_module(globals(), __name__)\n",
        "\n\n",
        "# ==============================================\n",
        "# Standalone Test Block\n",
        "# ==============================================\n",
        "if __name__ == \"__main__\":\n",
        "    print(\"🛠️ Running Core Utilities & Session Management comprehensive test suite...\")\n",
        "    success = run_comprehensive_tests()\n",
        "    sys.exit(0 if success else 1)\n",
        "\n\n",
        "# End of utils.py\n"
    ]
    
    # Combine the good parts with the clean ending
    clean_content = good_lines + clean_ending
    
    # Write the reconstructed file
    with open('utils.py', 'w', encoding='utf-8') as f:
        f.writelines(clean_content)
    
    print(f"✅ Reconstructed utils.py with {len(clean_content)} lines (was {len(lines)} lines)")
    print(f"✅ Removed {len(lines) - len(clean_content)} lines of duplicate/unreachable code")
    return True

if __name__ == "__main__":
    reconstruct_utils()
