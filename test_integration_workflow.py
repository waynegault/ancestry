from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.append(str(Path.cwd()))

from core.action_runner import exec_actn
from core.workflow_actions import gather_dna_matches, process_productive_messages_action, srch_inbox_actn
from session_utils import get_session_manager
from test_framework import TestSuite
from test_utilities import LiveSessionHandle, create_standard_test_runner, live_session_fixture


def _should_skip_live_api_tests() -> bool:
    """Return True when live API integration tests are disabled via environment."""

    return os.environ.get("SKIP_LIVE_API_TESTS", "").lower() == "true"


def _log_skip(action_label: str, reason: str) -> None:
    """Print a consistent skip message for visibility in test output."""

    print(f"SKIP {action_label}: {reason}")


def _run_with_live_session(action_label: str, executor: Callable[[LiveSessionHandle], None]) -> bool:
    """Execute a live integration helper if prerequisites are satisfied."""

    if _should_skip_live_api_tests():
        _log_skip(action_label, "SKIP_LIVE_API_TESTS=true")
        return True

    if get_session_manager() is None:
        _log_skip(action_label, "global session not initialized (run main.py to authenticate before tests)")
        return True

    with live_session_fixture(action_label) as live_handle:
        executor(live_handle)

    return True


def _test_action6_gather_live() -> bool:
    """Run Action 6 end to end using the authenticated browser/API session."""

    def _execute(live_handle: LiveSessionHandle) -> None:
        success = exec_actn(gather_dna_matches, live_handle.session_manager, "6 1")
        assert success, "Action 6 gather should complete successfully using the live session"

    return _run_with_live_session("Action 6 Integration", _execute)


def _test_action7_inbox_live() -> bool:
    """Validate Action 7 inbox processing through the live session stack."""

    def _execute(live_handle: LiveSessionHandle) -> None:
        success = exec_actn(srch_inbox_actn, live_handle.session_manager, "7")
        assert success, "Action 7 inbox search should complete successfully using the live session"

    return _run_with_live_session("Action 7 Integration", _execute)


def _test_action9_productive_live() -> bool:
    """Exercise Action 9 productive processing via the authenticated session."""

    def _execute(live_handle: LiveSessionHandle) -> None:
        success = exec_actn(process_productive_messages_action, live_handle.session_manager, "9")
        assert success, "Action 9 processing should complete successfully using the live session"

    return _run_with_live_session("Action 9 Integration", _execute)


def module_tests() -> bool:
    suite = TestSuite("Integration Workflow", "test_integration_workflow.py")

    suite.run_test(
        "Action 6 live gather",
        _test_action6_gather_live,
        "Runs the Action 6 gather workflow end to end via the shared live session.",
    )

    suite.run_test(
        "Action 7 live inbox",
        _test_action7_inbox_live,
        "Runs the Action 7 inbox workflow using the shared live session.",
    )

    suite.run_test(
        "Action 9 live productive processing",
        _test_action9_productive_live,
        "Runs the Action 9 productive workflow using the shared live session.",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
