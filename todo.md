# TODO List

### Remaining Future Enhancements (Not Blocking Production)

1. Enable auto-approval after building sufficient human review baseline
2. Activate TreeUpdateService in production after validation
3. Deploy Grafana dashboards via `python scripts/deploy_dashboards.py`
4. Real-time alerting for safety events

---

## Technical Debt (Commented Out Dead Code)

The following functions have been commented out as dead code (2025-12-18). They are preserved in case they become useful in the future.

| File | Function | Reason |
| ---- | -------- | ------ |
| `ai/ai_prompt_utils.py` | `quick_test()` | Replaced by `run_comprehensive_tests()`, never called |
| `actions/action10.py` | `detailed_scoring_breakdown()` | Debug helper function, never called in production |
| `actions/action10.py` | `get_user_criteria()` | Interactive user input helper, unused in production |
| `api/api_utils.py` | `print_group()` | Debug printing helper, never called |
| `actions/action7_inbox.py` | `_check_browser_health()` | Method defined but never called |
| `actions/action7_inbox.py` | `_calculate_api_limit()` | Method defined but never called |
| `actions/action7_inbox.py` | `_classify_message_with_ai()` | Method defined but never called |
| `actions/action7_inbox.py` | `_build_follow_up_context()` | Method defined but never called |
| `actions/action8_messaging.py` | `print_template_effectiveness_report()` | Report function never called |
| `actions/action8_messaging.py` | `_inject_research_suggestions()` | Helper function never called |
| `browser/selenium_utils.py` | `wait_for_element()` | Duplicate of `core/utils._wait_for_element` |
| `api/api_search_utils.py` | `get_api_relationship_path()` | Function defined but never called |

### Removal Candidates (False Positives in Dead Code Scan)

The dead code scanner flagged 146 items, but most are **false positives**:

1. **Flask routes** (`@api_bp.route`) - 10 items - Called by HTTP, not Python
2. **Mixin methods** (`TreeHealthChecksMixin`, etc.) - 23 items - Called via inheritance
3. **TypedDicts/Dataclasses** - 20+ items - Used for type hints
4. **Abstract/Protocol methods** - 8 items - Implemented by subclasses
5. **Test functions** (`module_tests()`) - Called by test runner
6. **`__init__` methods** - Called implicitly by Python

These should NOT be removed. The scanner doesn't understand decorators, inheritance, or type hints.

