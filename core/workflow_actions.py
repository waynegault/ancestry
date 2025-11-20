import logging
from typing import Any, Callable
from urllib.parse import urljoin

from action6_gather import coord
from action7_inbox import InboxProcessor
from action8_messaging import send_messages_to_matches
from action9_process_productive import process_productive_messages
from config.config_manager import ConfigManager
from core.action_registry import get_action_registry
from core.session_guards import ensure_navigation_ready, require_interactive_session
from core.session_manager import SessionManager
from my_selectors import WAIT_FOR_PAGE_SELECTOR
from standard_imports import setup_module
from utils import nav_to_page

logger = setup_module(globals(), __name__)

# Initialize config
config_manager = ConfigManager()
config = config_manager.get_config()


# --- Action Wrappers (7, 8, 9) ---

@require_interactive_session("search inbox")
def srch_inbox_actn(session_manager: Any, *_: Any) -> bool:
    """Action to search the inbox. Relies on exec_actn ensuring session is ready."""
    logger.debug("Starting inbox search...")
    try:
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
            session_manager.driver,
            config.api.base_url,
            WAIT_FOR_PAGE_SELECTOR,
            session_manager,
        ):
            logger.error("Failed nav to base URL. Aborting message sending.")
            return False
        logger.debug("Navigation OK. Proceeding to send messages...")

        # Call the actual sending function
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


@require_interactive_session("process productive messages")
def process_productive_messages_action(session_manager: Any, *_: Any) -> bool:
    """Action to process productive messages. Relies on exec_actn ensuring session is ready."""
    logger.debug("Starting productive message processing...")
    try:
        # Call the actual processing function
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
        # Call the imported function from action6
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
        if (_run_action7_inbox(session_manager) and
            _run_action9_process_productive(session_manager) and
            _run_action8_send_messages(session_manager)):

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
