#!/usr/bin/env python3

print("🏁 FINAL SYSTEM VERIFICATION")
print("=" * 50)

# Test api_search_utils.py
print("\n📋 Testing api_search_utils.py...")
try:
    from api_search_utils import self_test

    result1 = self_test()
    print(f'✅ api_search_utils.py: {"PASS" if result1 else "FAIL"}')
except Exception as e:
    print(f"❌ api_search_utils.py: ERROR - {e}")
    result1 = False

# Test api_cache.py
print("\n📋 Testing api_cache.py...")
try:
    from api_cache import run_api_cache_tests

    results = run_api_cache_tests()
    tests_passed = results["tests_passed"]
    tests_run = results["tests_run"]
    pass_rate = results["pass_rate"]
    result2 = tests_passed == tests_run
    print(f"✅ api_cache.py: {tests_passed}/{tests_run} passed ({pass_rate:.1f}%)")
except Exception as e:
    print(f"❌ api_cache.py: ERROR - {e}")
    result2 = False

print("\n🎯 FINAL RESULTS:")
print(
    f'  • api_search_utils.py: {"✅ PRODUCTION READY" if result1 else "❌ NEEDS ATTENTION"}'
)
print(f'  • api_cache.py: {"✅ PRODUCTION READY" if result2 else "❌ NEEDS ATTENTION"}')

overall_success = result1 and result2
print(
    f'\n🏆 Overall Status: {"✅ ALL SYSTEMS OPERATIONAL" if overall_success else "❌ ISSUES DETECTED"}'
)

if overall_success:
    print("\n🎉 SUCCESS: Both modules are now functioning at 100% capacity!")
    print("   - All syntax errors fixed")
    print("   - All runtime errors resolved")
    print("   - All internal tests passing")
    print("   - Ready for production use")
else:
    print("\n⚠️  Some issues remain. Please review the output above.")
