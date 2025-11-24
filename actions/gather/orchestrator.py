from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast
from unittest import mock

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from standard_imports import setup_module
from test_framework import TestSuite, create_standard_test_runner

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from core.session_manager import SessionManager
else:  # Runtime fallback keeps module importable without pulling heavy dependency.

    class SessionManager:  # pragma: no cover
        """Minimal stub, only used so tests can run without heavy imports."""

        pass


logger = setup_module(globals(), __name__)


def _delegate_to_legacy_coord(session_manager: SessionManager, start: Optional[int]) -> bool:
    """Call the legacy coord() implementation.

    This helper lives in a dedicated function so tests can patch it
    without importing the enormous `action6_gather` module.
    """

    from action6_gather import coord as legacy_coord

    return legacy_coord(session_manager, start)


@dataclass(frozen=True)
class GatherOrchestrator:
    """Thin adapter that forwards coord() calls to the legacy module.

    During Phase 3 of the refactor the implementation will migrate
    into this class, but for now we keep a lightweight wrapper so the
    rest of the codebase can begin referencing `actions.gather`.
    """

    session_manager: SessionManager

    def coord(self, start: Optional[int] = None) -> bool:
        """Execute the Action 6 gather flow using the legacy function."""

        logger.debug("Delegating gather coord() call to legacy implementation", extra={"start_page": start})
        return _delegate_to_legacy_coord(self.session_manager, start)


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------


def _test_orchestrator_delegates_to_legacy() -> bool:
    class FakeSessionManager:  # Minimal stand-in for SessionManager
        pass

    fake_session_manager = cast(SessionManager, FakeSessionManager())
    orchestrator = GatherOrchestrator(session_manager=fake_session_manager)

    with mock.patch(__name__ + "._delegate_to_legacy_coord", return_value=True) as delegate_mock:
        assert orchestrator.coord(start=42) is True
        delegate_mock.assert_called_once_with(fake_session_manager, 42)
    return True


def module_tests() -> bool:
    suite = TestSuite("actions.gather.orchestrator", "actions/gather/orchestrator.py")
    suite.run_test(
        "Delegates coord() calls",
        _test_orchestrator_delegates_to_legacy,
        "Ensures the orchestrator forwards work to the legacy function for now.",
    )
    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = module_tests()
    raise SystemExit(0 if success else 1)
