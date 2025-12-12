"""Workflow action wrappers for Actions 6-9 and core workflow.

This module uses lazy imports to avoid circular dependencies with action modules.
"""

import sys
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

import logging

from browser.css_selectors import WAIT_FOR_PAGE_SELECTOR
from core.session_guards import ensure_navigation_ready, require_interactive_session
from core.session_manager import SessionManager
from core.utils import nav_to_page

logger = logging.getLogger(__name__)

# Initialize config
from config.config_manager import get_config_manager

config_manager = get_config_manager()
config = config_manager.get_config()


def _get_coord() -> Any:
    """Lazy import of coord from action6_gather to avoid circular imports."""
    from actions.action6_gather import coord

    return coord


def _get_inbox_processor() -> Any:
    """Lazy import of InboxProcessor from action7_inbox to avoid circular imports."""
    from actions.action7_inbox import InboxProcessor

    return InboxProcessor


def _get_send_messages() -> Any:
    """Lazy import of send_messages_to_matches from action8_messaging."""
    from actions.action8_messaging import send_messages_to_matches

    return send_messages_to_matches


def _get_send_approved_drafts() -> Any:
    """Lazy import of send_approved_drafts from Action 11."""
    from actions.action11_send_approved_drafts import send_approved_drafts

    return send_approved_drafts


def _get_process_productive() -> Any:
    """Lazy import of process_productive_messages from action9_process_productive."""
    from actions.action9_process_productive import process_productive_messages

    return process_productive_messages


# --- Action Wrappers (7, 8, 9) ---


@require_interactive_session("search inbox")
def srch_inbox_actn(session_manager: Any, *_: Any) -> bool:
    """Action to search the inbox. Relies on exec_actn ensuring session is ready."""
    logger.debug("Starting inbox search...")
    try:
        InboxProcessor = _get_inbox_processor()
        processor = InboxProcessor(session_manager=session_manager)
        result = processor.search_inbox()
        if result is False:
            logger.error("Inbox search reported failure.")
            return False
        print("")
        logger.info("Inbox search OK.")
        return True
    except Exception as e:
        logger.error(f"Error during inbox search: {e}", exc_info=True)
        return False


@require_interactive_session("send messages")
def send_messages_action(session_manager: Any, *_: Any) -> bool:
    """Action to send messages. Relies on exec_actn ensuring session is ready."""
    logger.debug("Starting message sending...")
    try:
        # Navigate to Base URL first (good practice before starting message loops)
        logger.debug("Navigating to Base URL before sending...")
        if not nav_to_page(
            session_manager.browser_manager.driver,
            config.api.base_url,
            WAIT_FOR_PAGE_SELECTOR,
            session_manager,
        ):
            logger.error("Failed nav to base URL. Aborting message sending.")
            return False
        logger.debug("Navigation OK. Proceeding to send messages...")

        # Call the actual sending function
        send_messages_to_matches = _get_send_messages()
        result = send_messages_to_matches(session_manager)
        if result is False:
            logger.error("Message sending reported failure.")
            return False
        print("")
        logger.info("Messages sent OK.")
        return True
    except Exception as e:
        logger.error(f"Error during message sending: {e}", exc_info=True)
        return False


@require_interactive_session("send approved drafts")
def send_approved_drafts_action(session_manager: Any, *_: Any) -> bool:
    """Action to send only approved drafts. Relies on exec_actn ensuring session is ready."""
    logger.debug("Starting approved draft send...")
    try:
        logger.debug("Navigating to Base URL before sending approved drafts...")
        if not nav_to_page(
            session_manager.browser_manager.driver,
            config.api.base_url,
            WAIT_FOR_PAGE_SELECTOR,
            session_manager,
        ):
            logger.error("Failed nav to base URL. Aborting approved draft send.")
            return False

        send_approved_drafts = _get_send_approved_drafts()
        result = send_approved_drafts(session_manager)
        if result is False:
            logger.error("Approved draft send reported failure.")
            return False
        print("")
        logger.info("Approved drafts sent OK.")
        return True
    except Exception as e:
        logger.error(f"Error during approved draft send: {e}", exc_info=True)
        return False


@require_interactive_session("process productive messages")
def process_productive_messages_action(session_manager: Any, *_: Any) -> bool:
    """Action to process productive messages. Relies on exec_actn ensuring session is ready."""
    logger.debug("Starting productive message processing...")
    try:
        # Call the actual processing function
        process_productive_messages = _get_process_productive()
        result = process_productive_messages(session_manager)
        if result is False:
            logger.error("Productive message processing reported failure.")
            return False
        logger.info("Productive message processing OK.")
        return True
    except Exception as e:
        logger.error(f"Error during productive message processing: {e}", exc_info=True)
        return False


# --- Action 6 Wrapper ---


def gather_dna_matches(
    session_manager: SessionManager,
    config_schema: Any = None,
    start: int | None = None,
) -> bool:
    """
    Action wrapper for gathering matches (coord function from action6).
    Relies on exec_actn ensuring session is ready before calling.
    """
    # Use global config if config_schema not provided
    if config_schema is None:
        config_schema = config

    # Guard clause now checks session_ready
    if not session_manager or not session_manager.session_ready:
        logger.error("Cannot gather matches: Session not ready.")
        return False

    try:
        # Call the imported function from action6 (lazy import)
        coord = _get_coord()
        result = coord(session_manager, start=start)
        if result is False:
            logger.error("⚠️  WARNING: Match gathering incomplete or failed. Check logs for details.")
            return False
        print("")
        logger.info("✓ Match gathering completed successfully.")

        return True
    except Exception as e:
        logger.error(f"Error during gather_dna_matches: {e}", exc_info=True)
        return False


# --- Action 1 (Core Workflow) Helpers ---


def _run_action6_gather(session_manager: SessionManager) -> bool:
    """Run Action 6: Gather Matches."""
    logger.info("--- Running Action 6: Gather Matches (Always from page 1) ---")
    gather_result = gather_dna_matches(session_manager, config, start=1)
    if gather_result is False:
        logger.error("Action 6 FAILED.")
        print("ERROR: Match gathering failed. Check logs for details.")
        return False
    logger.info("Action 6 OK.")
    print("✓ Match gathering completed successfully.")
    return True


def _run_action7_inbox(session_manager: SessionManager) -> bool:
    """Run Action 7: Search Inbox."""
    logger.info("--- Running Action 7: Search Inbox ---")
    inbox_url = urljoin(config.api.base_url, "/messaging/")
    logger.debug(f"Navigating to Inbox ({inbox_url}) for Action 7...")

    try:
        if not ensure_navigation_ready(
            session_manager,
            action_label="Action 7",
            target_url=inbox_url,
            wait_selector="div.messaging-container",
            failure_reason="Could not navigate to inbox page.",
        ):
            return False

        logger.debug("Running inbox search...")
        InboxProcessor = _get_inbox_processor()
        inbox_processor = InboxProcessor(session_manager=session_manager)
        search_result = inbox_processor.search_inbox()

        if search_result is False:
            logger.error("Action 7 FAILED - Inbox search returned failure.")
            print("ERROR: Inbox search failed. Check logs for details.")
            return False

        logger.info("Action 7 OK.")
        print("✓ Inbox search completed successfully.")
        return True

    except Exception as inbox_error:
        logger.error(f"Action 7 FAILED with exception: {inbox_error}", exc_info=True)
        print(f"ERROR during inbox search: {inbox_error}")
        return False


def _run_action9_process_productive(session_manager: SessionManager) -> bool:
    """Run Action 9: Process Productive Messages."""
    logger.info("--- Running Action 9: Process Productive Messages ---")
    logger.debug("Navigating to Base URL for Action 9...")

    try:
        if not ensure_navigation_ready(
            session_manager,
            action_label="Action 9",
            target_url=config.api.base_url,
            wait_selector=WAIT_FOR_PAGE_SELECTOR,
            failure_reason="Could not navigate to base URL.",
        ):
            return False

        logger.debug("Processing productive messages after navigation guard passed...")

        process_productive_messages = _get_process_productive()
        process_result = process_productive_messages(session_manager)

        if process_result is False:
            logger.error("Action 9 FAILED - Productive message processing returned failure.")
            print("ERROR: Productive message processing failed. Check logs for details.")
            return False

        logger.info("Action 9 OK.")
        print("✓ Productive message processing completed successfully.")
        return True

    except Exception as process_error:
        logger.error(f"Action 9 FAILED with exception: {process_error}", exc_info=True)
        print(f"ERROR during productive message processing: {process_error}")
        return False


def _run_action8_send_messages(session_manager: SessionManager) -> bool:
    """Run Action 8: Send Messages."""
    logger.info("--- Running Action 8: Send Messages ---")
    logger.debug("Navigating to Base URL for Action 8...")

    try:
        if not ensure_navigation_ready(
            session_manager,
            action_label="Action 8",
            target_url=config.api.base_url,
            wait_selector=WAIT_FOR_PAGE_SELECTOR,
            failure_reason="Could not navigate to base URL.",
        ):
            return False

        logger.debug("Navigation guard passed. Sending messages...")

        send_messages_to_matches = _get_send_messages()
        send_result = send_messages_to_matches(session_manager)

        if send_result is False:
            logger.error("Action 8 FAILED - Message sending returned failure.")
            print("ERROR: Message sending failed. Check logs for details.")
            return False

        logger.info("Action 8 OK.")
        print("✓ Message sending completed successfully.")
        return True

    except Exception as message_error:
        logger.error(f"Action 8 FAILED with exception: {message_error}", exc_info=True)
        print(f"ERROR during message sending: {message_error}")
        return False


# --- Action 1 (Core Workflow) ---


def run_core_workflow_action(session_manager: SessionManager, *_: Any) -> bool:
    """
    Action to run the core workflow sequence: Action 7 (Inbox) → Action 9 (Process Productive) → Action 8 (Send Messages).
    Optionally runs Action 6 (Gather) first if configured.
    Relies on exec_actn ensuring session is ready beforehand.
    """
    result = False

    if not session_manager or not session_manager.session_ready:
        logger.error("Cannot run core workflow: Session not ready.")
        return result

    try:
        # Run Action 6 if configured
        run_action6 = config.include_action6_in_workflow
        if run_action6 and not _run_action6_gather(session_manager):
            return result

        # Run Action 7, 9, and 8 in sequence
        if (
            _run_action7_inbox(session_manager)
            and _run_action9_process_productive(session_manager)
            and _run_action8_send_messages(session_manager)
        ):
            # Build success message
            action_sequence: list[str] = []
            if run_action6:
                action_sequence.append("6")
            action_sequence.extend(["7", "9", "8"])
            action_sequence_str = "-".join(action_sequence)

            logger.info(f"Core Workflow (Actions {action_sequence_str}) finished successfully.")
            print(f"\n✓ Core Workflow (Actions {action_sequence_str}) completed successfully.")
            result = True

    except Exception as e:
        logger.error(f"Critical error during core workflow: {e}", exc_info=True)
        print(f"CRITICAL ERROR during core workflow: {e}")

    return result


def run_daily_review_first_loop_action(session_manager: SessionManager, *_: Any) -> bool:
    """Run a review-first daily operator loop: Action 7 → Review Queue → (confirm) Action 11."""
    if not session_manager or not session_manager.session_ready:
        logger.error("Cannot run daily review-first loop: Session not ready.")
        return False

    logger.info("--- Running Daily Review-First Loop: Action 7 → Review Queue → Action 11 ---")

    if not _run_action7_inbox(session_manager):
        return False

    try:
        from cli.maintenance import MainCLIHelpers

        helpers = MainCLIHelpers(logger=logger)
        helpers.show_review_queue(session_manager=session_manager)
    except Exception as exc:
        logger.error("Failed to open review queue: %s", exc, exc_info=True)
        print(f"ERROR: Failed to open review queue: {exc}")
        return False

    answer = input("\nSend APPROVED drafts now (Action 11)? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        logger.info("Operator declined Action 11 send step.")
        print("\nSkipped sending approved drafts.")
        return True

    return send_approved_drafts_action(session_manager)


# === TESTS ===


def _test_gather_dna_matches_requires_ready_session() -> bool:
    """Test that gather_dna_matches fails when session is not ready."""
    from types import SimpleNamespace
    from typing import cast

    # Test with None session (cast to bypass type checking)
    none_session: Any = None
    result = gather_dna_matches(cast(SessionManager, none_session))
    assert result is False, "Should fail with None session"

    # Test with session_ready=False
    mock_session = SimpleNamespace(session_ready=False)
    result = gather_dna_matches(cast(SessionManager, mock_session))
    assert result is False, "Should fail with session_ready=False"

    return True


def _test_run_core_workflow_requires_ready_session() -> bool:
    """Test that run_core_workflow_action fails when session is not ready."""
    from types import SimpleNamespace
    from typing import cast

    # Test with None session (cast to bypass type checking)
    none_session: Any = None
    result = run_core_workflow_action(cast(SessionManager, none_session))
    assert result is False, "Should fail with None session"

    # Test with session_ready=False
    mock_session = SimpleNamespace(session_ready=False)
    result = run_core_workflow_action(cast(SessionManager, mock_session))
    assert result is False, "Should fail with session_ready=False"

    return True


def _test_daily_review_first_loop_requires_ready_session() -> bool:
    """Test that daily loop fails when session is not ready."""
    from types import SimpleNamespace
    from typing import cast

    none_session: Any = None
    result = run_daily_review_first_loop_action(cast(SessionManager, none_session))
    assert result is False, "Should fail with None session"

    mock_session = SimpleNamespace(session_ready=False)
    result = run_daily_review_first_loop_action(cast(SessionManager, mock_session))
    assert result is False, "Should fail with session_ready=False"

    return True


def _test_daily_review_first_loop_respects_send_confirmation() -> bool:
    """Test that Action 11 is only invoked when operator confirms."""
    from types import SimpleNamespace
    from typing import cast
    from unittest.mock import patch

    mock_session = SimpleNamespace(session_ready=True)

    module_ref = sys.modules[__name__]
    with (
        patch.object(module_ref, "_run_action7_inbox", return_value=True),
        patch("cli.maintenance.MainCLIHelpers.show_review_queue", return_value=None),
        patch("builtins.input", return_value="n"),
        patch.object(module_ref, "send_approved_drafts_action", return_value=True) as mock_send,
    ):
        result = run_daily_review_first_loop_action(cast(SessionManager, mock_session))
        assert result is True
        mock_send.assert_not_called()

    with (
        patch.object(module_ref, "_run_action7_inbox", return_value=True),
        patch("cli.maintenance.MainCLIHelpers.show_review_queue", return_value=None),
        patch("builtins.input", return_value="y"),
        patch.object(module_ref, "send_approved_drafts_action", return_value=True) as mock_send,
    ):
        result = run_daily_review_first_loop_action(cast(SessionManager, mock_session))
        assert result is True
        mock_send.assert_called_once()

    return True


def _test_srch_inbox_actn_decorator_applied() -> bool:
    """Test that srch_inbox_actn has the require_interactive_session decorator."""
    # The decorator wraps the function - check it blocks on None session
    result = srch_inbox_actn(None)
    assert result is False, "Decorator should block when session manager is None"
    return True


def _test_send_messages_action_decorator_applied() -> bool:
    """Test that send_messages_action has the require_interactive_session decorator."""
    result = send_messages_action(None)
    assert result is False, "Decorator should block when session manager is None"
    return True


def _test_process_productive_messages_action_decorator_applied() -> bool:
    """Test that process_productive_messages_action has the require_interactive_session decorator."""
    result = process_productive_messages_action(None)
    assert result is False, "Decorator should block when session manager is None"
    return True


def module_tests() -> bool:
    """Run module tests for workflow_actions."""
    from testing.test_framework import TestSuite

    suite = TestSuite("core.workflow_actions", "core/workflow_actions.py")

    suite.run_test(
        "gather_dna_matches session guard",
        _test_gather_dna_matches_requires_ready_session,
        "Ensures gather_dna_matches fails cleanly when session is not ready.",
    )

    suite.run_test(
        "run_core_workflow session guard",
        _test_run_core_workflow_requires_ready_session,
        "Ensures run_core_workflow_action fails cleanly when session is not ready.",
    )

    suite.run_test(
        "daily review-first loop session guard",
        _test_daily_review_first_loop_requires_ready_session,
        "Ensures daily review-first loop fails cleanly when session is not ready.",
    )

    suite.run_test(
        "daily review-first loop send confirmation",
        _test_daily_review_first_loop_respects_send_confirmation,
        "Ensures Action 11 runs only when operator confirms.",
    )

    suite.run_test(
        "srch_inbox_actn decorator",
        _test_srch_inbox_actn_decorator_applied,
        "Ensures srch_inbox_actn blocks when session manager is None.",
    )

    suite.run_test(
        "send_messages_action decorator",
        _test_send_messages_action_decorator_applied,
        "Ensures send_messages_action blocks when session manager is None.",
    )

    suite.run_test(
        "process_productive_messages_action decorator",
        _test_process_productive_messages_action_decorator_applied,
        "Ensures process_productive_messages_action blocks when session manager is None.",
    )

    return suite.finish_suite()


if __name__ == "__main__":
    import sys

    from testing.test_framework import create_standard_test_runner

    run_comprehensive_tests = create_standard_test_runner(module_tests)
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
