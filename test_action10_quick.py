#!/usr/bin/env python3
"""
Quick test to verify action10.py TestSuite framework works
"""


def test_action10_import():
    """Test that action10 can be imported and run_comprehensive_tests exists"""
    try:
        # First test the test framework alone
        from test_framework import TestSuite, suppress_logging

        print("✅ TestSuite imported successfully")

        # Test basic functionality
        suite = TestSuite("Test Suite", "test.py")
        suite.start_suite()

        def test_basic():
            assert True, "Basic test"

        with suppress_logging():
            suite.run_test(
                "Basic Test",
                test_basic,
                "Should pass",
                "Basic assertion",
                "Should return True",
            )

        result = suite.finish_suite()
        print(f"✅ Test framework working, result: {result}")

        # Now test if we can import action10 and get its test function
        print("Testing action10 import...")
        try:
            import action10

            if hasattr(action10, "run_comprehensive_tests"):
                print("✅ action10.run_comprehensive_tests found")
                return True
            else:
                print("❌ action10.run_comprehensive_tests not found")
                return False
        except Exception as import_error:
            print(f"❌ Error importing action10: {import_error}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_action10_import()
    exit(0 if success else 1)
