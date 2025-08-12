#!/usr/bin/env python3

"""
Test utils.py specifically to demonstrate numbered test reporting.
"""

# Import and run utils test suite
if __name__ == "__main__":
    print("🧪 Testing utils.py with numbered test reporting...")
    
    try:
        from utils import utils_module_tests
        success = utils_module_tests()
        print(f"\n📊 Utils tests {'✅ PASSED' if success else '❌ FAILED'}")
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Test execution error: {e}")
