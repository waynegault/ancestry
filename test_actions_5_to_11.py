#!/usr/bin/env python3
"""
Quick test script for Actions 5-11
Tests that each action module can be imported and has required functions.
"""

import sys
import importlib
from pathlib import Path

def test_action_module(action_num, module_name, required_functions):
    """Test that an action module can be imported and has required functions."""
    print(f"\n{'='*60}")
    print(f"Testing Action {action_num}: {module_name}")
    print('='*60)
    
    try:
        # Import the module
        module = importlib.import_module(module_name)
        print(f"✅ Module '{module_name}' imported successfully")
        
        # Check for required functions
        missing_functions = []
        for func_name in required_functions:
            if hasattr(module, func_name):
                print(f"  ✅ Function '{func_name}' found")
            else:
                print(f"  ❌ Function '{func_name}' NOT FOUND")
                missing_functions.append(func_name)
        
        if missing_functions:
            print(f"\n⚠️  Missing functions: {', '.join(missing_functions)}")
            return False
        else:
            print(f"\n✅ All required functions present in {module_name}")
            return True
            
    except Exception as e:
        print(f"❌ Error importing {module_name}: {e}")
        return False

def main():
    """Run tests for actions 5-11."""
    print("\n" + "="*60)
    print("TESTING ACTIONS 5-11")
    print("="*60)
    
    # Define actions to test
    actions = [
        # Action 5 is in main.py
        (5, "main", ["check_login_actn"]),
        # Action 6
        (6, "action6_gather", ["coord"]),
        # Action 7
        (7, "action7_inbox", ["InboxProcessor"]),
        # Action 8
        (8, "action8_messaging", ["send_messages_to_matches"]),
        # Action 9
        (9, "action9_process_productive", ["process_productive_messages"]),
        # Action 10
        (10, "action10", ["main", "load_gedcom_data", "filter_and_score_individuals"]),
        # Action 11
        (11, "action11", ["run_action11", "main"]),
    ]
    
    results = []
    for action_num, module_name, required_functions in actions:
        result = test_action_module(action_num, module_name, required_functions)
        results.append((action_num, result))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for action_num, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"Action {action_num}: {status}")
    
    print(f"\nTotal: {passed}/{total} actions passed")
    
    if passed == total:
        print("\n🎉 All actions are ready for testing!")
        return True
    else:
        print(f"\n⚠️  {total - passed} action(s) failed validation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

