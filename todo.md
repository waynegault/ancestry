# TODO List

## Operational Enablement
- [x] Deploy Grafana dashboards (`python scripts/deploy_dashboards.py`) - ✅ 4 dashboards deployed

## Code Quality & Maintenance
- [x] Run dead code scan (`python testing/dead_code_scan.py`) - ✅ 146 candidates in Cache/dead_code_candidates.json
- [x] Review import audit (`python testing/import_audit.py`) - ✅ 3/3 tests passed
- [x] Check for type ignore directives (`python testing/check_type_ignores.py`) - ✅ Zero found
- [x] Review dead code candidates - ✅ 146 analyzed, ~120 false positives (Flask/TypedDict/mixin), 4 genuine items commented out

## Documentation
- [x] Update README.md with current capabilities - ✅ Updated mission status, test counts
- [x] Review operator_manual.md for accuracy - ✅ No outdated content
- [x] Generate code_graph.json updates - ✅ 6998 nodes, 6835 links

## Production Readiness
- [x] Run production guard check (`python scripts/check_production_guard.py`) - ✅ APP_MODE=dry_run (safe)
- [x] Review rate limiting configuration - ✅ REQUESTS_PER_SECOND=0.3
- [x] Validate session management - ✅ SessionManager, db_manager, api_manager initialized

---

## Technical Debt (Commented Out Dead Code)

The following functions have been commented out as dead code (2025-12-18). They are preserved in case they become useful in the future.

| File | Function | Reason |
|------|----------|--------|
| `ai/ai_prompt_utils.py` | `quick_test()` | Replaced by `run_comprehensive_tests()`, never called |
| `actions/action10.py` | `detailed_scoring_breakdown()` | Debug helper function, never called in production |
| `actions/action10.py` | `get_user_criteria()` | Interactive user input helper, unused in production |
| `api/api_utils.py` | `print_group()` | Debug printing helper, never called |

### Removal Candidates (False Positives in Dead Code Scan)

The dead code scanner flagged 146 items, but most are **false positives**:

1. **Flask routes** (`@api_bp.route`) - 10 items - Called by HTTP, not Python
2. **Mixin methods** (`TreeHealthChecksMixin`, etc.) - 23 items - Called via inheritance
3. **TypedDicts/Dataclasses** - 20+ items - Used for type hints
4. **Abstract/Protocol methods** - 8 items - Implemented by subclasses
5. **Test functions** (`module_tests()`) - Called by test runner
6. **`__init__` methods** - Called implicitly by Python

These should NOT be removed. The scanner doesn't understand decorators, inheritance, or type hints.

