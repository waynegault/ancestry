# Test Examples & Best Practices

This directory contains examples and documentation for writing tests in the Ancestry Research Automation Platform.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Test Patterns](#test-patterns)
3. [Best Practices](#best-practices)
4. [Common Pitfalls](#common-pitfalls)
5. [Examples](#examples)

---

## Quick Start

### Basic Test Structure

Every test module follows this pattern:

```python
"""
Module docstring describing what is being tested.
"""

import sys
from test_framework import TestSuite, create_standard_test_runner


def _test_something() -> bool:
    """Test description."""
    # Your test logic here
    result = some_function()
    return result == expected_value


def module_tests() -> bool:
    """Test suite for this module."""
    suite = TestSuite("Suite Name", "module_name.py")

    suite.start_suite()
    suite.run_test("Test description", _test_something)

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
```

### Running Tests

```bash
# Run single test module
python -m module_name

# Run all tests
python run_all_tests.py

# Run all tests in parallel (faster)
python run_all_tests.py --fast

# Run with performance analysis
python run_all_tests.py --analyze-logs
```

---

## Test Patterns

### Pattern 1: Simple Assertion Test

```python
def _test_math_operations() -> bool:
    """Test basic math operations."""
    result = 2 + 2
    return result == 4
```

**Use when:** Testing simple functions with clear expected outputs.

### Pattern 2: Exception Testing

```python
def _test_error_handling() -> bool:
    """Test that invalid input raises ValueError."""
    try:
        some_function(invalid_input)
        return False  # Should have raised exception
    except ValueError:
        return True  # Expected exception
    except Exception:
        return False  # Wrong exception type
```

**Use when:** Validating error handling and exception types.

### Pattern 3: State Validation

```python
def _test_state_changes() -> bool:
    """Test that object state changes correctly."""
    obj = MyClass()
    initial_state = obj.state

    obj.do_something()

    state_changed = obj.state != initial_state
    correct_value = obj.state == expected_state

    return state_changed and correct_value
```

**Use when:** Testing stateful objects or side effects.

### Pattern 4: Multiple Assertions

```python
def _test_complex_behavior() -> bool:
    """Test multiple aspects of a function."""
    result = complex_function(input_data)

    has_required_field = 'field' in result
    correct_type = isinstance(result['field'], str)
    correct_value = result['field'] == expected

    return has_required_field and correct_type and correct_value
```

**Use when:** Testing functions with multiple outputs or properties.

### Pattern 5: I/O Capture

```python
from test_diagnostics import capture_io

def _test_with_output() -> bool:
    """Test function that produces output."""
    @capture_io
    def test_func():
        print("Debug info")
        return some_function()

    result, stdout, stderr, exc = test_func()

    has_output = "Debug info" in stdout
    no_errors = exc is None
    correct_result = result == expected

    return has_output and no_errors and correct_result
```

**Use when:** Testing functions that print to stdout/stderr.

---

## Best Practices

### ✅ DO

1. **Name tests descriptively**
   ```python
   def _test_uuid_stored_uppercase() -> bool:
       """Test that UUIDs are stored in uppercase format."""
   ```

2. **Test one concept per function**
   - Each test function should verify one specific behavior
   - Multiple assertions are OK if they're part of the same concept

3. **Use helper functions for setup**
   ```python
   def _create_test_person():
       """Helper to create test Person object."""
       return Person(uuid="TEST-UUID", name="Test User")

   def _test_person_creation() -> bool:
       person = _create_test_person()
       return person.uuid == "TEST-UUID"
   ```

4. **Return early on failure**
   ```python
   def _test_complex_flow() -> bool:
       result1 = step1()
       if not result1:
           return False  # Fail fast

       result2 = step2()
       return result2
   ```

5. **Use suppress_logging for cleaner output**
   ```python
   from test_utilities import suppress_logging

   @suppress_logging
   def _test_noisy_function() -> bool:
       """Test function that logs a lot."""
       return noisy_function() == expected
   ```

### ❌ DON'T

1. **Don't use hardcoded credentials**
   ```python
   # BAD
   password = "my_secret_password"

   # GOOD
   password = os.getenv("TEST_PASSWORD", "test_dummy")
   ```

2. **Don't leave tests that always pass**
   ```python
   # BAD
   def _test_placeholder() -> bool:
       return True  # TODO: implement actual test
   ```

3. **Don't modify production data in tests**
   ```python
   # BAD - modifies real database
   session.query(Person).delete()

   # GOOD - use test fixtures or mock data
   test_session.query(TestPerson).delete()
   ```

4. **Don't ignore exceptions silently**
   ```python
   # BAD
   try:
       result = risky_function()
   except:
       pass

   # GOOD
   try:
       result = risky_function()
   except SpecificException as e:
       return False  # Test fails if exception occurs
   ```

5. **Don't create tests with external dependencies**
   ```python
   # BAD - requires internet
   def _test_api_call() -> bool:
       response = requests.get("https://example.com")
       return response.status_code == 200

   # GOOD - mock or skip if unavailable
   def _test_api_call() -> bool:
       if not is_network_available():
           return True  # Skip test
       # ... test with mock
   ```

---

## Common Pitfalls

### Pitfall 1: SQLAlchemy Session Caching

**Problem:** After bulk insert, immediate lookup returns None.

```python
# WRONG
session.bulk_insert_mappings(Person, person_dicts)
person = session.query(Person).filter_by(uuid=test_uuid).first()  # None!
```

**Solution:** Call `session.expire_all()` after bulk operations.

```python
# CORRECT
session.bulk_insert_mappings(Person, person_dicts)
session.expire_all()  # Force refresh from DB
person = session.query(Person).filter_by(uuid=test_uuid).first()  # Works!
```

### Pitfall 2: UUID Case Sensitivity

**Problem:** UUID lookups fail because of case mismatch.

```python
# WRONG
uuid = "abc123"  # lowercase
person = session.query(Person).filter(Person.uuid == uuid).first()  # None!
```

**Solution:** Always use `.upper()` for UUID operations.

```python
# CORRECT
uuid = "abc123".upper()  # UPPERCASE
person = session.query(Person).filter(Person.uuid == uuid).first()  # Works!
```

### Pitfall 3: Rate Limiter Not Initialized

**Problem:** Test hangs or fails because rate limiter isn't ready.

```python
# WRONG
api_response = make_api_call()  # Hangs waiting for rate limiter
```

**Solution:** Ensure SessionManager is initialized or mock rate limiter.

```python
# CORRECT
session_manager = SessionManager()
session_manager.ensure_session_ready()
api_response = session_manager.api_manager.get(url)
```

### Pitfall 4: Browser Session Not Ready

**Problem:** Browser operations fail with "session not ready" error.

```python
# WRONG
driver.get(url)  # Error: session not ready
```

**Solution:** Use `exec_actn()` wrapper or manually ensure session ready.

```python
# CORRECT
session_manager.ensure_session_ready()
driver = session_manager.browser_manager.driver
driver.get(url)
```

---

## Examples

See the example files in this directory:

- **[example_basic_test.py](./example_basic_test.py)** - Simple function testing
- **[example_database_test.py](./example_database_test.py)** - Database model testing
- **[example_error_handling_test.py](./example_error_handling_test.py)** - Exception and error testing
- **[example_integration_test.py](./example_integration_test.py)** - Multi-component testing
- **[example_mock_test.py](./example_mock_test.py)** - Mocking external dependencies

---

## Test Coverage Guidelines

### Minimum Coverage

Every module should test:

1. **Happy path** - Normal, expected usage
2. **Edge cases** - Boundary conditions (empty input, max values, etc.)
3. **Error cases** - Invalid input, missing data
4. **Integration points** - How it works with other components

### Example Coverage Checklist

For a function like `process_dna_match(uuid: str, shared_cm: float)`:

- ✅ Valid UUID with normal shared_cm value
- ✅ Valid UUID with edge case shared_cm (0, very large number)
- ✅ Invalid UUID (None, empty string, wrong format)
- ✅ Invalid shared_cm (negative, None, string)
- ✅ Database integration (data is saved correctly)
- ✅ Error handling (exceptions raised as expected)

---

## Test Data Management

### Use Test Fixtures

Create reusable test data in `test_data/`:

```python
from test_data.fixtures import create_test_person, create_test_match

def _test_with_fixture() -> bool:
    person = create_test_person()
    match = create_test_match(person)
    return match.person_id == person.id
```

### Clean Up After Tests

```python
def _test_with_cleanup() -> bool:
    # Create test data
    test_person = create_test_person()

    try:
        # Run test
        result = do_something(test_person)
        return result

    finally:
        # Clean up
        session.delete(test_person)
        session.commit()
```

---

## Advanced Topics

### Testing Async Code

```python
import asyncio

def _test_async_function() -> bool:
    """Test async function."""
    async def run_test():
        result = await async_function()
        return result == expected

    return asyncio.run(run_test())
```

### Testing with Timeouts

```python
import signal

def _test_with_timeout() -> bool:
    """Test that completes within timeout."""
    def timeout_handler(signum, frame):
        raise TimeoutError("Test took too long")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)  # 5 second timeout

    try:
        result = slow_function()
        signal.alarm(0)  # Cancel alarm
        return result == expected
    except TimeoutError:
        return False
```

### Parametrized Tests

```python
def _test_multiple_inputs() -> bool:
    """Test function with multiple inputs."""
    test_cases = [
        (input1, expected1),
        (input2, expected2),
        (input3, expected3),
    ]

    for input_val, expected in test_cases:
        result = function(input_val)
        if result != expected:
            return False  # Fail on first mismatch

    return True  # All passed
```

---

## Resources

- **test_framework.py** - Core testing infrastructure
- **test_diagnostics.py** - Enhanced error reporting
- **test_utilities.py** - Helper functions and decorators
- **run_all_tests.py** - Test runner with parallel execution
- **comprehensive_auth_tests.py** - Example of comprehensive testing
- **end_to_end_tests.py** - Example of integration testing

---

## Getting Help

If you encounter issues or have questions:

1. Check existing test modules for similar patterns
2. Review error messages in test output
3. Use `test_diagnostics.capture_io` for debugging
4. Run tests with `--analyze-logs` flag for performance insights
5. Check `Logs/app.log` for detailed execution logs

---

**Last Updated:** November 2025
**Test Framework Version:** 1.0
**Total Test Modules:** 58+
