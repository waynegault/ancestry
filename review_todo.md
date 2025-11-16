# Review Todos

## Test-Review Todos

1. **Standardize Entry Points**: Ensure that all `run_comprehensive_tests` entrypoints are standardized across the test suites to maintain consistency.

2. **Centralize Test Utilities**: Move all test utilities into `test_utilities.py` to streamline test function access and improve maintainability.

3. **Strengthen Assertions**: Review and enhance assertions in `gedcom_intelligence.py` and `message_personalization.py` to increase the reliability of tests.

4. **Separate Unit vs Integration Tests**: Organize tests into unit tests and integration tests, using shared live-session helpers to improve clarity and purpose.

5. **Consolidate Temp File and Dir Helpers**: Create a centralized helper for temporary files and directories to reduce duplication and improve reliability.

6. **Enforce Test Quality**: Make `analyze_test_quality.py` a gatekeeper for smoke tests, ensuring that all tests are of sufficient quality before being executed.

7. **Tighten Enforcement**: Ensure that Ruff and Pyright configurations require 100% quality without suppressions to maintain code integrity.

---

_Last updated on 2025-11-16 00:20:25 by waynegault_