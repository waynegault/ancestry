# Testing Directory

## Note: Tests Are Embedded in Module Files

This project follows an **embedded testing pattern** where tests are included directly in the source modules rather than in a separate `tests/` directory.

## How Testing Works

Each module contains its own tests using the `TestSuite` framework from `testing/test_framework.py`:

```python
# Example from any module
def module_tests() -> bool:
    from testing.test_framework import TestSuite

    suite = TestSuite("Module Name", "module_file.py")
    suite.start_suite()

    suite.run_test(
        "Test name",
        test_function,
        "Test summary",
        "Functions tested",
        "Method description",
    )

    return suite.finish_suite()

# Standard runner at bottom of file
if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
```

## Running Tests

### Run All Tests
```powershell
python run_all_tests.py              # Sequential (all modules)
python run_all_tests.py --fast       # Parallel execution
python run_all_tests.py --analyze-logs  # With performance analysis
```

### Run Single Module Tests
```powershell
python -m core.approval_queue        # Run approval_queue tests
python -m actions.action6_gather     # Run action6 tests
python -m testing.protocol_mocks     # Run protocol mocks tests
```

## Testing Utilities

- `testing/test_framework.py` - TestSuite class, assertions, utilities
- `testing/test_utilities.py` - Shared helpers (create_test_database, etc.)
- `testing/protocol_mocks.py` - Protocol-based mock implementations

## Why Embedded Tests?

1. **Proximity**: Tests live next to the code they test
2. **Discoverability**: No need to search separate directory
3. **Module Validation**: Run `python -m module` to validate any module
4. **58+ Test Modules**: All validated by `run_all_tests.py`
