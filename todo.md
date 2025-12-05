# Production Readiness Actions

## 1. Code Consolidation & Streamlining
- [x] **Audit `main.py`**: Remove re-exports (lines 84-102) if they are legacy aliases and not used by external scripts.
- [x] **Refactor `utils.py`**: This file is over 4000 lines. Analyze for logical splits (e.g., `utils_browser.py`, `utils_string.py`) or consolidate redundant helpers into `core/` modules.
- [x] **Refactor `database.py`**: This file is over 4000 lines. Review "Priority 1 Todo" comments (e.g., lines 112, 253) and implement or remove.
- [ ] **Remove Stubs**: Search for `pass` or `...` bodies that are not abstract methods and remove/implement them.
- [ ] **Remove Aliases**: Search for `X = Y` patterns at module level and replace usages of `X` with `Y`.

## 2. Test Regimen Improvement
- [x] **Verify Test Discovery**: Ensure `run_all_tests.py` correctly discovers and runs tests in `actions/` and `core/`.
- [x] **Inline Tests**: Ensure *every* file has a `if __name__ == "__main__":` block that runs its tests.
- [ ] **Real Tests**: Review existing tests to ensure they are not using `unittest.mock` for core logic where a live session could be used.
- [ ] **Live Session Access**: Ensure tests that need it can access `SessionManager` with a valid session.

## 3. Linting & Quality (100% Target)
- [ ] **Fix Type Errors**: Run `pyright` and fix all errors. Remove `type: ignore` comments.
- [x] **Fix Lint Errors**: Run `ruff` and fix all warnings.
- [ ] **Strict Mode**: Enable strict type checking in `pyrightconfig.json` if not already enabled.

## 4. Documentation & Comments
- [ ] **Update Comments**: Remove "Phase 4.1", "Priority 1 Todo", "Phase 5.1" etc. if they are just historical markers.
- [ ] **Clean Docstrings**: Ensure docstrings are accurate and up-to-date with current function signatures.
