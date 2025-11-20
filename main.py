#!/usr/bin/env python3

"""
main.py - Ancestry Research Automation Main Entry Point

Provides the main application entry point with menu-driven interface for
all automation workflows including DNA match gathering, inbox processing,
messaging, and genealogical research tools.
"""

# === SUPPRESS CONFIG WARNINGS FOR PRODUCTION ===
import os

os.environ["SUPPRESS_CONFIG_WARNINGS"] = "1"

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

# === MODULE SETUP ===
logger = setup_module(globals(), __name__)

# === TEST UTILITIES ===
# === STANDARD LIBRARY IMPORTS ===
import gc
import logging

# os already imported at top for SUPPRESS_CONFIG_WARNINGS
import shutil
import sys
import threading
import time
import webbrowser
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

# === LOCAL IMPORTS ===
# Action modules
from importlib import import_module
from logging import StreamHandler
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, TextIO, cast
from urllib.parse import urljoin

# === THIRD-PARTY IMPORTS ===
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as SASession

# === ACTION MODULES ===
from action6_gather import coord  # Import the main DNA match gathering function
from action7_inbox import InboxProcessor
from action9_process_productive import process_productive_messages
from core.action_registry import (
    ActionMetadata,
    get_action_registry,
)
from core.action_runner import (
    DatabaseManagerProtocol,
    configure_action_runner,
    exec_actn,
    get_action_metadata as _get_action_metadata,
    get_api_manager as _get_api_manager,
    get_browser_manager as _get_browser_manager,
    get_database_manager as _get_database_manager,
    parse_menu_choice as _parse_menu_choice,
)

SendMessagesAction = Callable[["SessionManager"], bool]
MatchRecord = dict[str, Any]
MatchList = list[MatchRecord]
SearchAPIFunc = Callable[["SessionManager", dict[str, Any], int], MatchList]
RowBuilder = Callable[[MatchRecord], list[str]]
MatchAnalyzer = Callable[[Any, MatchRecord, Optional[str], str], None]
SupplementaryHandler = Callable[[MatchRecord, "SessionManager"], None]
IDNormalizer = Callable[[Optional[str]], Optional[str]]
AnalyticsExtrasSetter = Callable[[dict[str, Any]], None]


def _import_send_messages_action() -> SendMessagesAction:
    """Import and return the messaging action with a precise type."""

    module = import_module("action8_messaging")
    send_messages_attr = getattr(module, "send_messages_to_matches")
    return cast(SendMessagesAction, send_messages_attr)


send_messages_to_matches: SendMessagesAction = _import_send_messages_action()


class ConfigManagerProtocol(Protocol):
    """Protocol describing the ConfigManager behavior used here."""

    def get_config(self) -> Any: ...


class GrafanaCheckerProtocol(Protocol):
    """Protocol for optional grafana_checker helpers."""

    def ensure_dashboards_imported(self) -> None: ...

    def check_grafana_status(self) -> Mapping[str, Any]: ...

    def ensure_grafana_ready(self, *, auto_setup: bool = False, silent: bool = True) -> None: ...


_config_manager_factory: type[ConfigManagerProtocol] | None = None
_config_manager_error: Exception | None = None

try:
    _config_module = import_module("config.config_manager")
except Exception as exc:
    _config_manager_error = exc
else:
    _config_candidate = getattr(_config_module, "ConfigManager", None)
    if isinstance(_config_candidate, type):
        _config_manager_factory = cast(type[ConfigManagerProtocol], _config_candidate)
    else:
        _config_manager_error = RuntimeError("ConfigManager class missing from config.config_manager")


_grafana_checker: GrafanaCheckerProtocol | None = None
try:
    _grafana_checker_module = import_module("grafana_checker")
except Exception:
    _grafana_checker = None
else:
    _grafana_checker = cast(GrafanaCheckerProtocol, _grafana_checker_module)

grafana_checker: GrafanaCheckerProtocol | None = _grafana_checker


_metrics_factory: Callable[[], Any] | None = None
_metrics_import_error: Exception | None = None

try:
    _metrics_module = import_module("observability.metrics_registry")
except Exception as exc:
    _metrics_import_error = exc
else:
    _metrics_candidate = getattr(_metrics_module, "metrics", None)
    if callable(_metrics_candidate):
        _metrics_factory = cast(Callable[[], Any], _metrics_candidate)
    else:
        _metrics_import_error = RuntimeError("metrics() not found in observability.metrics_registry")


def _create_config_manager() -> Optional[ConfigManagerProtocol]:
    """Instantiate ConfigManager if available."""

    if _config_manager_factory is None:
        if _config_manager_error is not None:
            logger.debug("ConfigManager unavailable: %s", _config_manager_error)
        return None

    try:
        return _config_manager_factory()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to instantiate ConfigManager: %s", exc, exc_info=True)
        return None


def _get_metrics_bundle() -> Optional[Any]:
    """Return a metrics bundle if the observability module is available."""

    if _metrics_factory is None:
        return None

    try:
        return _metrics_factory()
    except Exception:  # pragma: no cover - metrics are optional
        logger.debug("Metrics bundle request failed", exc_info=True)
        return None


def _load_result_row_builders() -> tuple[RowBuilder, RowBuilder]:
    """Load row builder helpers from GEDCOM and API modules with type safety."""

    action10_module = import_module("action10")
    api_search_module = import_module("api_search_core")
    gedcom_builder = cast(RowBuilder, getattr(action10_module, "_create_table_row"))
    api_builder = cast(RowBuilder, getattr(api_search_module, "_create_table_row_for_candidate"))
    return gedcom_builder, api_builder


def _load_match_analysis_helpers() -> tuple[IDNormalizer, MatchAnalyzer, SupplementaryHandler]:
    """Load helper functions used when rendering match details."""

    action10_module = import_module("action10")
    api_search_module = import_module("api_search_core")
    normalize_id = cast(IDNormalizer, getattr(action10_module, "_normalize_id"))
    analyze_top_match = cast(MatchAnalyzer, getattr(action10_module, "analyze_top_match"))
    handle_supplementary = cast(
        SupplementaryHandler,
        getattr(api_search_module, "_handle_supplementary_info_phase"),
    )
    return normalize_id, analyze_top_match, handle_supplementary


def _set_comparison_mode_analytics(gedcom_count: int, api_count: int) -> None:
    """Record analytics metadata when comparison mode runs successfully."""

    try:
        analytics_module = import_module("analytics")
    except Exception:
        return

    setter = getattr(analytics_module, "set_transient_extras", None)
    if not callable(setter):
        return

    extras = {
        "comparison_mode": True,
        "gedcom_candidates": gedcom_count,
        "api_candidates": api_count,
    }
    try:
        cast(AnalyticsExtrasSetter, setter)(extras)
    except Exception:
        logger.debug("Analytics extras setter failed", exc_info=True)


def _initialize_db_manager_engine(db_manager: DatabaseManagerProtocol) -> tuple[Any, Any]:
    """Ensure the database manager has an initialized engine and session factory."""

    initializer = getattr(db_manager, "_initialize_engine_and_session", None)
    if callable(initializer):
        initializer()

    engine = getattr(db_manager, "engine", None)
    session_factory = getattr(db_manager, "Session", None)
    if engine is None or session_factory is None:
        raise SQLAlchemyError("Database manager missing engine or session factory")
    return engine, session_factory


def _load_and_validate_config_schema() -> Optional[Any]:
    """Load and validate configuration schema."""
    try:
        from config import config_schema
        logger.debug("Configuration loaded successfully")
        return cast(Any, config_schema)
    except ImportError as e:
        logger.error(f"Could not import config_schema from config package: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        return None


def _check_processing_limits(config: Any) -> None:
    """Check essential processing limits and log warnings."""
    # Note: MAX_PAGES=0 is valid (means unlimited), so no warning needed
    if config.batch_size <= 0:
        logger.warning("BATCH_SIZE not set or invalid - actions may use large batches")
    # Note: MAX_PRODUCTIVE_TO_PROCESS=0 and MAX_INBOX=0 are valid (means unlimited)


def _check_rate_limiting_settings(config: Any) -> None:
    """Check rate limiting settings and log warnings."""
    # Note: Rate limiting settings are user preferences - no warnings needed
    # Users can adjust REQUESTS_PER_SECOND, INITIAL_DELAY, and BACKOFF_FACTOR as needed
    _ = config  # Parameter kept for API compatibility but not currently used


def _log_basic_configuration_values(config: Any) -> None:
    """Log the primary configuration values shown at startup."""

    logger.info(f"  MAX_PAGES: {config.api.max_pages}")
    logger.info(f"  BATCH_SIZE: {config.batch_size}")
    logger.info(f"  MAX_PRODUCTIVE_TO_PROCESS: {config.max_productive_to_process}")
    logger.info(f"  MAX_INBOX: {config.max_inbox}")
    logger.info(f"  PARALLEL_WORKERS: {config.parallel_workers}")
    logger.info(
        f"  Rate Limiting - RPS: {config.api.requests_per_second}, Delay: {config.api.initial_delay}s"
    )

    match_throughput = getattr(config.api, "target_match_throughput", 0.0)
    if match_throughput > 0:
        logger.info("  Match Throughput Target: %.2f match/s", match_throughput)
    else:
        logger.info("  Match Throughput Target: disabled")

    logger.info(
        "  Max Pacing Delay/Page: %.2fs",
        getattr(config.api, "max_throughput_catchup_delay", 0.0),
    )


def _should_suppress_config_warnings() -> bool:
    """Return True when runtime context indicates configuration warnings should be muted."""

    if os.environ.get("SUPPRESS_CONFIG_WARNINGS") == "1":
        return True
    if os.environ.get("PYTEST_CURRENT_TEST") is not None:
        return True
    return any("test" in arg.lower() for arg in sys.argv)


def _warn_if_unsafe_profile(speed_profile: str, allow_unsafe: bool, suppress_warnings: bool) -> None:
    """Emit warning when unsafe API profiles are active."""

    if suppress_warnings:
        return

    if not (allow_unsafe or speed_profile in {"max", "aggressive", "experimental"}):
        return

    profile_label = speed_profile or "custom"
    logger.warning(
        "  Unsafe API speed profile '%s' active; safety clamps relaxed. Monitor for 429 errors.",
        profile_label,
    )


def _log_persisted_rate_state(persisted_state: dict[str, Any]) -> None:
    """Log persisted rate limiter metadata from previous runs."""

    saved_rate = persisted_state.get("fill_rate")
    saved_requests = persisted_state.get("total_requests", "n/a")
    timestamp_value = persisted_state.get("timestamp")
    if isinstance(timestamp_value, (int, float)):
        timestamp_str = datetime.fromtimestamp(timestamp_value).strftime("%Y-%m-%d %H:%M:%S")
    else:
        timestamp_str = "unknown"

    if isinstance(saved_rate, (int, float)):
        logger.info(
            "    Last run: %.3f req/s | saved at %s | total_requests=%s",
            float(saved_rate),
            timestamp_str,
            saved_requests,
        )


def _log_rate_limiter_summary(config: Any, allow_unsafe: bool, speed_profile: str) -> None:
    """Log the adaptive rate limiter plan without instantiating it early."""

    try:
        from rate_limiter import get_persisted_rate_state
    except ImportError:
        logger.debug("Rate limiter module unavailable during configuration summary")
        return

    persisted_state = get_persisted_rate_state()
    batch_threshold = max(getattr(config, "batch_size", 50) or 50, 1)
    configured_threshold = getattr(config.api, "token_bucket_success_threshold", None)
    if isinstance(configured_threshold, int) and configured_threshold > 0:
        success_threshold = configured_threshold
    else:
        success_threshold = max(batch_threshold, 10)
    safe_rps = getattr(config.api, "requests_per_second", 0.3) or 0.3
    desired_rate = getattr(config.api, "token_bucket_fill_rate", None) or safe_rps
    allow_aggressive = allow_unsafe or speed_profile in {"max", "aggressive", "experimental"}
    min_fill_rate = max(0.05, safe_rps * 0.25)
    max_fill_rate = desired_rate if allow_aggressive else safe_rps
    max_fill_rate = max(max_fill_rate, min_fill_rate)
    bucket_capacity = getattr(config.api, "token_bucket_capacity", 10.0)

    logger.info(
        "  Rate Limiter (planned): target=%.3f req/s | success_threshold=%d | bounds=%.3f-%.3f | capacity=%.1f",
        desired_rate,
        success_threshold,
        min_fill_rate,
        max_fill_rate,
        bucket_capacity,
    )

    if persisted_state:
        _log_persisted_rate_state(persisted_state)


def _log_configuration_summary(config: Any) -> None:
    """Log current configuration for transparency."""
    # Clear screen at startup (temporarily disabled for debugging global session complaints)
    # import os
    # os.system('cls' if os.name == 'nt' else 'clear')

    print(" CONFIG ".center(80, "="))
    speed_profile = str(getattr(config.api, "speed_profile", "safe")).lower()
    allow_unsafe = bool(getattr(config.api, "allow_unsafe_rate_limit", False))
    _log_basic_configuration_values(config)
    suppress_warnings = _should_suppress_config_warnings()
    _warn_if_unsafe_profile(speed_profile, allow_unsafe, suppress_warnings)
    _log_rate_limiter_summary(config, allow_unsafe, speed_profile)
    print("")  # Blank line after configuration


# Configuration validation
def validate_action_config() -> bool:
    """
    Validate that all actions respect .env configuration limits.
    Prevents Action 6-style failures by ensuring conservative settings are applied.
    """
    try:
        # Load and validate configuration
        config = _load_and_validate_config_schema()
        if config is None:
            return False

        # Check processing limits
        _check_processing_limits(config)

        # Check rate limiting settings
        _check_rate_limiting_settings(config)

        # Log configuration summary
        _log_configuration_summary(config)

        return True

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


# Core modules
from core.caching_bootstrap import ensure_caching_initialized
from core.database_manager import backup_database, db_transn
from core.session_manager import SessionManager
from database import (
    Base,
    ConversationLog,
    DnaMatch,
    FamilyTree,
    MessageTemplate,
    Person,
)
from logging_config import setup_logging
from my_selectors import WAIT_FOR_PAGE_SELECTOR
from ui.menu import render_main_menu
from utils import (
    log_in,
    login_status,
    nav_to_page,
)

config_manager = _create_config_manager()
config: Any = config_manager.get_config() if config_manager is not None else None

configure_action_runner(config=config, metrics_provider=_get_metrics_bundle)


def menu() -> str:
    """Display the main menu and return the normalized choice."""

    return render_main_menu(logger, config, get_action_registry())


# End of menu


def clear_log_file() -> tuple[bool, Optional[str]]:
    """Finds the FileHandler, closes it, clears the log file, and returns a success flag and the log file path."""
    cleared = False
    log_file_handler: Optional[logging.FileHandler] = None
    log_file_path: Optional[str] = None
    try:
        # Step 1: Find the FileHandler in the logger's handlers
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file_handler = handler
                log_file_path = handler.baseFilename
                break
        if log_file_handler is not None and log_file_path is not None:
            # Step 2: Flush the handler (ensuring all previous writes are persisted to disk)
            log_file_handler.flush()
            # Step 3: Close the handler (releases resources)
            log_file_handler.close()
            # Step 4: Clear the log file contents
            with Path(log_file_path).open("w", encoding="utf-8"):
                pass
            cleared = True
    except PermissionError as permission_error:
        # Handle permission errors when attempting to open the log file
        logger.warning(
            f"Permission denied clearing log '{log_file_path}': {permission_error}"
        )
    except OSError as io_error:
        # Handle I/O errors when attempting to open the log file
        logger.warning(f"IOError clearing log '{log_file_path}': {io_error}")
    except Exception as error:
        # Handle any other exceptions during the log clearing process
        logger.warning(f"Error clearing log '{log_file_path}': {error}", exc_info=True)
    return cleared, log_file_path


# End of clear_log_file


# --- Action Functions


# Action 1 (all_but_first_actn)
def _delete_table_records(
    sess: SASession,
    table_class: type[Any],
    filter_condition: Any,
    table_name: str,
    person_id_to_keep: int,
) -> int:
    """Delete records from a table based on filter condition."""
    logger.debug(f"Deleting from {table_name} where people_id != {person_id_to_keep}...")
    result = sess.query(table_class).filter(filter_condition).delete(synchronize_session=False)
    count = int(result or 0)
    logger.info(f"Deleted {count} {table_name} records.")
    return count


def _perform_deletions(sess: SASession, person_id_to_keep: int) -> dict[str, int]:
    """Perform all deletion operations and return counts."""
    deleted_counts: dict[str, int] = {
        "conversation_log": _delete_table_records(
            sess, ConversationLog, ConversationLog.people_id != person_id_to_keep,
            "conversation_log", person_id_to_keep
        ),
        "dna_match": _delete_table_records(
            sess, DnaMatch, DnaMatch.people_id != person_id_to_keep,
            "dna_match", person_id_to_keep
        ),
        "family_tree": _delete_table_records(
            sess, FamilyTree, FamilyTree.people_id != person_id_to_keep,
            "family_tree", person_id_to_keep
        ),
        "people": _delete_table_records(
            sess, Person, Person.id != person_id_to_keep,
            "people", person_id_to_keep
        ),
    }

    total_deleted = int(sum(deleted_counts.values()))
    if total_deleted == 0:
        logger.info(f"No records found to delete besides Person ID {person_id_to_keep}.")

    return deleted_counts


def all_but_first_actn(session_manager: SessionManager, *_extra: Any) -> bool:
    """
    V1.5: Delete all records except for the test profile (Frances Milne).
    Uses TEST_PROFILE_ID from .env to identify which profile to keep.
    Browserless database-only action.
    Closes the provided main session pool FIRST.
    Creates a temporary SessionManager for the delete operation.
    """
    # Get profile ID from config (TEST_PROFILE_ID from .env)
    profile_id_to_keep = config.testing_profile_id if config else None

    if not profile_id_to_keep:
        logger.error(
            "Profile ID not available from config. Cannot determine which profile to keep.\n"
            "Please ensure TEST_PROFILE_ID is set in .env file."
        )
        return False

    profile_id_to_keep = profile_id_to_keep.upper()
    logger.info(f"Deleting all records except test profile: {profile_id_to_keep}")

    temp_manager = None  # Initialize
    session = None
    success = False
    try:
        # --- Close main pool FIRST ---
        if session_manager:
            logger.debug(
                f"Closing main DB connections before deleting data (except {profile_id_to_keep})..."
            )
            session_manager.cls_db_conn(keep_db=False)
            logger.debug("Main DB pool closed.")
        # --- End closing main pool ---

        logger.debug(
            f"Deleting data for all people except Profile ID: {profile_id_to_keep}..."
        )
        # Create a temporary SessionManager for this specific operation
        temp_manager = SessionManager()
        session = temp_manager.get_db_conn()
        if session is None:
            raise Exception("Failed to get DB session via temporary manager.")

        with db_transn(session) as sess:
            # Check if database is empty
            total_people = sess.query(Person).filter(Person.deleted_at.is_(None)).count()

            if total_people == 0:
                print("\n" + "=" * 60)
                print("INFO: DATABASE IS EMPTY")
                print("=" * 60)
                print("\nThe database contains no records.")
                print("Please run Action 2 (Reset Database) first to initialize")
                print("the database with test data.")
                print("\nAction 1 is used to delete all records EXCEPT the test")
                print("profile, but there are currently no records to delete.")
                print("=" * 60 + "\n")
                logger.info("Action 1: Database is empty. No records to delete.")
                success = True  # Not an error - just nothing to do
                return True  # Return to menu without closing session

            # 1. Find the person to keep by profile_id
            person_to_keep = (
                sess.query(Person.id, Person.username, Person.first_name, Person.profile_id)
                .filter(
                    Person.profile_id == profile_id_to_keep, Person.deleted_at.is_(None)
                )
                .first()
            )

            if not person_to_keep:
                print("\n" + "=" * 60)
                print("⚠️  TEST PROFILE NOT FOUND")
                print("=" * 60)
                print("\nThe database does not contain the test profile:")
                print(f"  Profile ID: {profile_id_to_keep}")
                print("\nThis could mean:")
                print("  1. The database is empty or doesn't have this profile")
                print("  2. TEST_PROFILE_ID in .env doesn't match any person")
                print("\nPlease run Action 2 (Reset Database) to initialize")
                print("the database with the test profile.")
                print("=" * 60 + "\n")
                logger.info("Action 1 aborted: Test profile not found in database.")
                success = True  # Don't treat as error - just inform user
                return True  # Return True to avoid closing session

            person_id_to_keep = person_to_keep.id
            logger.debug(
                f"Keeping test profile: ID={person_id_to_keep}, "
                f"Username='{person_to_keep.username}', "
                f"First Name='{person_to_keep.first_name}', "
                f"Profile ID='{person_to_keep.profile_id}'"
            )

            # --- Perform Deletions ---
            _perform_deletions(sess, person_id_to_keep)

        success = True  # Mark success if transaction completes

    except Exception as e:
        logger.error(
            f"Error during deletion (except {profile_id_to_keep}): {e}", exc_info=True
        )
        success = False  # Explicitly mark failure
    finally:
        # Clean up the temporary session manager and its resources
        if temp_manager:
            if session:
                temp_manager.return_session(session)
            temp_manager.cls_db_conn(keep_db=False)  # Close the temp pool
        logger.debug(f"Delete action (except {profile_id_to_keep}) finished.")
    return success


# end of action 1 (all_but_first_actn)


# Helper functions for run_core_workflow_action

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


def _ensure_navigation_ready(
    session_manager: SessionManager,
    *,
    action_label: str,
    target_url: str,
    wait_selector: str,
    failure_reason: str,
) -> bool:
    """Shared guard that ensures driver availability and page navigation."""

    driver = session_manager.driver
    if driver is None:
        logger.error(f"Driver not available for {action_label} navigation")
        print("ERROR: Browser session not available. Please rerun login (Action 5).")
        return False

    if not nav_to_page(driver, target_url, wait_selector, session_manager):
        logger.error(f"{action_label} nav FAILED - {failure_reason}")
        print(f"ERROR: {failure_reason} Check network connection.")
        return False

    logger.debug(
        f"Navigation to {target_url} successful for {action_label}. Waiting briefly before continuing..."
    )
    time.sleep(2)
    return True


def _run_action7_inbox(session_manager: SessionManager) -> bool:
    """Run Action 7: Search Inbox."""
    logger.info("--- Running Action 7: Search Inbox ---")
    inbox_url = urljoin(config.api.base_url, "/messaging/")
    logger.debug(f"Navigating to Inbox ({inbox_url}) for Action 7...")

    try:
        if not _ensure_navigation_ready(
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
        if not _ensure_navigation_ready(
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
        if not _ensure_navigation_ready(
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


# Action 1
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


# End Action 1


# Action 2 (reset_db_actn)
def _truncate_all_tables(temp_manager: SessionManager) -> bool:
    """Truncate all tables in the database."""
    logger.debug("Truncating all tables...")
    truncate_session = temp_manager.get_db_conn()
    if not truncate_session:
        logger.critical("Failed to get session for truncating tables. Reset aborted.")
        return False

    try:
        with db_transn(truncate_session) as sess:
            # Delete all records from tables in reverse order of dependencies
            sess.query(ConversationLog).delete(synchronize_session=False)
            sess.query(DnaMatch).delete(synchronize_session=False)
            sess.query(FamilyTree).delete(synchronize_session=False)
            sess.query(Person).delete(synchronize_session=False)
            # Keep MessageType table intact
        temp_manager.return_session(truncate_session)
        logger.debug("All tables truncated successfully.")
        return True
    except Exception as e:
        logger.error(f"Error truncating tables: {e}", exc_info=True)
        temp_manager.return_session(truncate_session)
        return False


def _reinitialize_database_schema(temp_manager: SessionManager) -> bool:
    """Re-initialize database schema by dropping and recreating all tables."""
    logger.debug("Re-initializing database schema...")
    try:
        db_manager = _get_database_manager(temp_manager)
        if db_manager is None:
            logger.error("Database manager unavailable for schema reinitialization")
            return False

        # This will create a new engine and session factory pointing to the file path
        engine = getattr(db_manager, "engine", None)
        session_factory = getattr(db_manager, "Session", None)
        if engine is None or session_factory is None:
            engine, session_factory = _initialize_db_manager_engine(db_manager)

        # Drop all existing tables first to ensure clean schema
        logger.debug("Dropping all existing tables...")
        Base.metadata.drop_all(engine)
        logger.debug("All tables dropped successfully.")

        # Recreate all tables with current schema definitions
        logger.debug("Creating tables with current schema...")
        Base.metadata.create_all(engine)
        logger.debug("Database schema recreated successfully.")
        return True
    except Exception as e:
        logger.error(f"Error reinitializing database schema: {e}", exc_info=True)
        return False


def _seed_message_templates(recreation_session: Any) -> bool:
    """Seed message templates from database defaults (single source of truth)."""
    logger.debug("Seeding message_templates table from database defaults...")
    try:
        # Import inside function to avoid circular imports at module import time
        database_module = import_module("database")

        get_default_templates = getattr(database_module, "_get_default_message_templates", None)
        if get_default_templates is None:
            logger.error("Default message template helper not available.")
            return False

        with db_transn(recreation_session) as sess:
            existing_count = sess.query(func.count(MessageTemplate.id)).scalar() or 0

            if existing_count > 0:
                logger.debug(
                    f"Found {existing_count} existing message templates. Skipping seeding."
                )
            else:
                templates_data: list[dict[str, Any]] = get_default_templates()
                if not templates_data:
                    logger.warning("Default message templates list is empty. Nothing to seed.")
                else:
                    for template in templates_data:
                        # Template dict from database helper already contains all fields
                        sess.add(MessageTemplate(**template))
                    logger.debug(f"Added {len(templates_data)} message templates from defaults.")

        count = recreation_session.query(func.count(MessageTemplate.id)).scalar() or 0
        logger.debug(
            f"MessageTemplate seeding complete. Total templates in DB: {count}"
        )
        return True
    except Exception as e:
        logger.error(f"Error seeding message templates: {e}", exc_info=True)
        return False


def _initialize_ethnicity_columns_from_metadata(db_manager: SessionManager) -> bool:
    """
    Initialize ethnicity columns in dna_match table using saved metadata file.
    This is browserless - it only adds columns if ethnicity_regions.json exists.
    If the file doesn't exist, columns will be added during first Action 6 run.

    Args:
        db_manager: SessionManager with active database connection

    Returns:
        True if columns added successfully, False if metadata file doesn't exist
    """
    try:
        from dna_ethnicity_utils import initialize_ethnicity_columns_from_metadata

        logger.debug("Checking for saved ethnicity metadata...")
        return initialize_ethnicity_columns_from_metadata(db_manager)

    except Exception as e:
        logger.error(f"Error adding ethnicity columns from metadata: {e}", exc_info=True)
        return False


def _close_main_pool_for_reset(session_manager: SessionManager) -> None:
    """Close main pool and force garbage collection."""
    if session_manager:
        logger.debug("Closing main DB connections before database deletion...")
        session_manager.cls_db_conn(keep_db=False)  # Ensure pool is closed
        logger.debug("Main DB pool closed.")

    # Force garbage collection to release any file handles
    logger.debug("Running garbage collection to release file handles...")
    gc.collect()
    time.sleep(1.0)
    gc.collect()


def _perform_database_reset_steps(temp_manager: SessionManager) -> tuple[bool, Any]:
    """Perform database reset steps and return success status and session.

    Returns:
        Tuple of (success, recreation_session)
    """
    # Step 1: Truncate all tables
    if not _truncate_all_tables(temp_manager):
        return False, None

    # Step 2: Re-initialize database schema
    if not _reinitialize_database_schema(temp_manager):
        return False, None

    # Step 3: Seed MessageType Table
    recreation_session = temp_manager.get_db_conn()
    if not recreation_session:
        raise SQLAlchemyError("Failed to get session for seeding MessageTypes!")

    _seed_message_templates(recreation_session)

    # Step 4: Add ethnicity columns from saved metadata (if available)
    logger.debug("Adding ethnicity columns from saved metadata...")
    if _initialize_ethnicity_columns_from_metadata(temp_manager):
        logger.info("✅ Ethnicity columns added from saved metadata")
    else:
        logger.info("INFO: No ethnicity metadata found - columns will be added during first Action 6 run")

    # Step 5: Commit all changes to ensure they're flushed to disk
    logger.debug("Committing database changes...")
    recreation_session.commit()
    logger.debug("Database changes committed successfully.")

    return True, recreation_session


def reset_db_actn(session_manager: SessionManager, *_extra: Any) -> bool:
    """
    Action to COMPLETELY reset the database by deleting the file. Browserless.
    - Closes main pool.
    - Deletes the .db file.
    - Recreates schema from scratch.
    - Seeds the MessageType table.
    """
    db_path = config.database.database_file
    reset_successful = False
    temp_manager = None
    recreation_session = None

    try:
        # Step 1: Close main pool
        _close_main_pool_for_reset(session_manager)

        # Step 2: Validate database path
        if db_path is None:
            logger.critical("DATABASE_FILE is not configured. Reset aborted.")
            return False

        logger.debug(f"Attempting to delete database file: {db_path}...")

        try:
            # Create temporary session manager
            logger.debug("Creating temporary session manager for database reset...")
            temp_manager = SessionManager()

            # Perform reset steps
            reset_successful, recreation_session = _perform_database_reset_steps(temp_manager)

            if reset_successful:
                logger.info("✅ Database reset completed successfully.")

        except Exception as recreate_err:
            logger.error(f"Error during DB recreation/seeding: {recreate_err}", exc_info=True)
            reset_successful = False
        finally:
            # Clean up the temporary manager and its session/engine
            logger.debug("Cleaning up temporary resource manager for reset...")
            if temp_manager and recreation_session:
                temp_manager.return_session(recreation_session)
            logger.debug("Temporary resource manager cleanup finished.")

    except Exception as e:
        logger.error(f"Outer error during DB reset action: {e}", exc_info=True)
        reset_successful = False

    finally:
        logger.debug("Reset DB action finished.")

    return reset_successful


# end of Action 2 (reset_db_actn)


# Action 3 (backup_db_actn)
def backup_db_actn(*_: Any) -> bool:
    """Action to backup the database. Browserless."""
    try:
        logger.debug("Starting DB backup...")
        # _session_manager isn't used but needed for exec_actn compatibility
        result = backup_database()
        if result:
            logger.info("DB backup OK.")
            return True
        logger.error("DB backup failed.")
        return False
    except Exception as e:
        logger.error(f"Error during DB backup: {e}", exc_info=True)
        return False


# end of Action 3


# Action 4 (restore_db_actn)
def restore_db_actn(session_manager: SessionManager, *_extra: Any) -> bool:  # Added session_manager back
    """
    Action to restore the database. Browserless.
    Closes the provided main session pool FIRST.
    """
    backup_dir = config.database.data_dir
    db_path = config.database.database_file
    success = False

    # Validate paths
    if backup_dir is None:
        logger.error("Cannot restore database: DATA_DIR is not configured.")
        return False

    if db_path is None:
        logger.error("Cannot restore database: DATABASE_FILE is not configured.")
        return False

    backup_path = backup_dir / "ancestry_backup.db"

    try:
        # --- Close main pool FIRST ---
        if session_manager:
            logger.debug("Closing main DB connections before restore...")
            session_manager.cls_db_conn(keep_db=False)
            logger.debug("Main DB pool closed.")
        # --- End closing main pool ---

        logger.debug(f"Restoring DB from: {backup_path}")
        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_path}.")
            return False

        logger.debug("Running GC before restore...")
        gc.collect()
        time.sleep(0.5)
        gc.collect()

        shutil.copy2(backup_path, db_path)
        logger.info("Db restored from backup OK.")

        # Display table statistics
        _display_table_statistics()

        success = True
    except FileNotFoundError:
        logger.error(f"Backup not found during copy: {backup_path}")
    except (OSError, shutil.Error) as e:
        logger.error(f"Error restoring DB: {e}", exc_info=True)
    except Exception as e:
        logger.critical(f"Unexpected restore error: {e}", exc_info=True)
    finally:
        logger.debug("DB restore action finished.")
    return success


def _display_table_statistics() -> None:
    """Display statistics for all tables in the database."""
    from sqlalchemy import create_engine, inspect, text

    db_path = config.database.database_file
    if not db_path:
        logger.warning("Cannot display table statistics: DATABASE_FILE not configured")
        return

    try:
        engine = create_engine(f"sqlite:///{db_path}")
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        if not table_names:
            logger.info("Database contains no tables")
            return

        logger.info("")
        logger.info("Database Table Statistics:")
        logger.info("-" * 60)

        with engine.connect() as conn:
            for table_name in sorted(table_names):
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                count = result.scalar()
                logger.info(f"  {table_name:30s} {count:>10,} records")

        logger.info("-" * 60)

    except Exception as e:
        logger.warning(f"Could not display table statistics: {e}")


# end of Action 4


def _display_session_info(session_manager: SessionManager) -> None:
    """Display session information and all key identifiers from the global session."""
    # Display all identifiers (should already be available from global session authentication)
    if session_manager.tree_owner_name:
        print(f"Account Name:    {session_manager.tree_owner_name}")

    if session_manager.my_profile_id:
        print(f"Profile ID:      {session_manager.my_profile_id}")

    if session_manager.my_uuid:
        print(f"UUID:            {session_manager.my_uuid}")

    if session_manager.my_tree_id:
        print(f"Tree ID:         {session_manager.my_tree_id}")

    api_manager = _get_api_manager(session_manager)
    if api_manager and api_manager.csrf_token:
        token = api_manager.csrf_token
        csrf_preview = token[:20] + "..." if len(token) > 20 else token
        print(f"CSRF Token:      {csrf_preview}")


def _handle_logged_in_status(session_manager: SessionManager) -> bool:
    """Handle the case when user is already logged in."""
    print("\n✓ You are currently logged in to Ancestry.\n")
    _display_session_info(session_manager)
    return True


def _verify_login_success(session_manager: SessionManager) -> bool:
    """Verify login was successful and display session info."""
    final_status = login_status(session_manager, disable_ui_fallback=False)
    if final_status is True:
        print("✓ Login verification confirmed.")
        _display_session_info(session_manager)
        return True
    print("⚠️  Login appeared successful but verification failed.")
    return False


def _attempt_login(session_manager: SessionManager) -> bool:
    """Attempt to log in with stored credentials."""
    print("\n✗ You are NOT currently logged in to Ancestry.")
    print("  Attempting to log in with stored credentials...")

    try:
        login_result = log_in(session_manager)

        if login_result:
            print("✓ Login successful!")
            return _verify_login_success(session_manager)

        print("✗ Login failed. Please check your credentials in .env file.")
        return False

    except Exception as login_e:
        logger.error(f"Exception during login attempt: {login_e}", exc_info=True)
        print(f"✗ Login failed with error: {login_e}")
        print("  Please check your credentials in .env file.")
        return False


# Action 5 (check_login_actn)
def check_login_actn(session_manager: SessionManager, *_extra: Any) -> bool:
    """
    REVISED V13: Checks login status, attempts login if needed, and displays all identifiers.
    This action starts a browser session and checks login status.
    If not logged in, it attempts to log in using stored credentials.
    Displays all key identifiers: Profile ID, UUID, Tree ID, CSRF Token.
    Provides clear user feedback about the final login state.
    """
    # Phase 1 (Driver Start) is handled by exec_actn if needed.
    # We only need to check if driver is live before proceeding.
    if not session_manager.driver_live:
        logger.error("Driver not live. Cannot check login status.")
        print("ERROR: Browser not started. Cannot check login status.")
        print(
            "       Select any browser-required action (1, 6-9) to start the browser."
        )
        return False

    print("Checking login status...")

    # Call login_status directly to check initial status
    try:
        status = login_status(
            session_manager, disable_ui_fallback=False
        )  # Use UI fallback for reliability

        if status is True:
            return _handle_logged_in_status(session_manager)
        if status is False:
            return _attempt_login(session_manager)
        # Status is None
        print("\n? Unable to determine login status due to a technical error.")
        print("  This may indicate a browser or network issue.")
        logger.warning("Login status check returned None (ambiguous result).")
        return False

    except Exception as e:
        logger.error(f"Exception during login status check: {e}", exc_info=True)
        print(f"\n! Error checking login status: {e}")
        print("  This may indicate a browser or network issue.")
        return False


# End Action 5


# Action 6 (gather_dna_matches wrapper)
def gather_dna_matches(
    session_manager: SessionManager,
    config_schema: Optional[Any] = None,
    start: Optional[int] = None,
) -> bool:
    """
    Action wrapper for gathering matches (coord function from action6).
    Relies on exec_actn ensuring session is ready before calling.

    Args:
        session_manager: The SessionManager instance.
        config_schema: The configuration schema (optional, uses global config if None).
        start: Optional page number override. When None, resume from checkpoint if available.
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
# End of gather_dna_matches


def _ensure_interactive_session_ready(session_manager: Any, action_label: str) -> bool:
    """Ensure session_manager exists and session_ready flag is true for an action."""
    if not session_manager:
        logger.error(f"Cannot {action_label}: SessionManager is None.")
        return False

    session_ready = getattr(session_manager, "session_ready", None)
    if session_ready is None:
        driver_live = getattr(session_manager, "driver_live", False)
        if driver_live:
            logger.warning("session_ready not set, initializing based on driver_live")
            session_manager.session_ready = True
            session_ready = True
        else:
            logger.warning("session_ready and driver_live not set, initializing to False")
            session_manager.session_ready = False
            session_ready = False

    if not session_ready:
        logger.error(f"Cannot {action_label}: Session not ready.")
        return False

    return True


def require_interactive_session(action_label: str) -> Callable[[Callable[..., Any]], Callable[..., bool]]:
    """Decorator that enforces session readiness before running an action."""

    def decorator(func: Callable[..., Any]) -> Callable[..., bool]:
        @wraps(func)
        def wrapper(session_manager: Any, *args: Any, **kwargs: Any) -> bool:
            if not _ensure_interactive_session_ready(session_manager, action_label):
                return False
            result = func(session_manager, *args, **kwargs)
            return bool(result)

        return wrapper

    return decorator


# Action 7 (srch_inbox_actn)
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
        return True  # Use INFO
    except Exception as e:
        logger.error(f"Error during inbox search: {e}", exc_info=True)
        return False


# End of srch_inbox_actn


# Action 8 (send_messages_action)
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
        return True  # Use INFO
    except Exception as e:
        logger.error(f"Error during message sending: {e}", exc_info=True)
        return False


# End of send_messages_action


# Action 9 (process_productive_messages_action)
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


# End of process_productive_messages_action


@dataclass
class _ComparisonConfig:
    """Configuration inputs needed for the GEDCOM/API comparison run."""

    gedcom_path: Optional[Path]
    reference_person_id_raw: Optional[str]
    reference_person_name: Optional[str]
    date_flex: Optional[dict[str, Any]]
    scoring_weights: dict[str, Any]
    max_display_results: int


@dataclass
class _ComparisonResults:
    """Container for search results spanning GEDCOM and API sources."""

    gedcom_data: Any
    gedcom_matches: MatchList
    api_matches: MatchList


def _perform_gedcom_search(
    gedcom_path: Optional[Path],
    criteria: dict[str, Any],
    scoring_weights: dict[str, Any],
    date_flex: Optional[dict[str, Any]],
) -> tuple[Any, MatchList]:
    """Perform GEDCOM search and return data and matches."""

    action10_module = import_module("action10")
    load_gedcom_data_fn = cast(Callable[[Path], Any], getattr(action10_module, "load_gedcom_data"))
    build_filter_criteria = cast(Callable[[dict[str, Any]], Any], getattr(action10_module, "_build_filter_criteria"))
    filter_and_score = cast(Callable[..., MatchList], getattr(action10_module, "filter_and_score_individuals"))

    gedcom_data: Any = None
    gedcom_matches: MatchList = []

    if gedcom_path is not None:
        try:
            gedcom_data = load_gedcom_data_fn(gedcom_path)
            filter_criteria = build_filter_criteria(criteria)
            normalized_date_flex: dict[str, Any] = date_flex or {}
            gedcom_matches = filter_and_score(
                gedcom_data,
                filter_criteria,
                criteria,
                scoring_weights,
                normalized_date_flex,
            )
        except Exception as exc:
            logger.error(f"GEDCOM search failed: {exc}")

    return gedcom_data, gedcom_matches


def _perform_api_search_fallback(
    session_manager: SessionManager,
    criteria: dict[str, Any],
    max_results: int,
) -> MatchList:
    """Perform API search as fallback when GEDCOM has no matches."""

    api_module = import_module("api_search_core")
    search_api = cast(
        SearchAPIFunc,
        getattr(api_module, "search_ancestry_api_for_person"),
    )

    try:
        session_ok = session_manager.ensure_session_ready(
            action_name="GEDCOM/API Search - API Fallback", skip_csrf=False
        )
        if not session_ok:
            logger.error("Could not establish browser session for API search")
            return []

        api_manager = _get_api_manager(session_manager)
        browser_manager = _get_browser_manager(session_manager)
        if api_manager is None or browser_manager is None:
            logger.error("Browser/API managers unavailable for API fallback search")
            return []

        try:
            synced = api_manager.sync_cookies_from_browser(
                browser_manager,
                session_manager=session_manager,
            )
            if not synced:
                logger.warning("Cookie sync from browser failed, but attempting API search anyway")
        except Exception as sync_err:
            logger.warning(f"Cookie sync error: {sync_err}, but attempting API search anyway")

        return search_api(session_manager, criteria, max_results)
    except Exception as exc:
        logger.error(f"API search failed: {exc}", exc_info=True)
        return []


def _format_table_row(row: Sequence[str], widths: Sequence[int]) -> str:
    """Return padded string for display rows."""

    return " | ".join(col.ljust(width) for col, width in zip(row, widths))


def _compute_table_widths(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> list[int]:
    """Return column widths based on headers and row content."""

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    return widths


def _log_result_table(label: str, rows: list[list[str]], total: int, headers: list[str]) -> None:
    """Log table output for GEDCOM/API matches when debug logging is enabled."""
    widths = _compute_table_widths(rows, headers)
    header_row = _format_table_row(headers, widths)

    logger.debug("")
    logger.debug(f"=== {label} Results (Top {len(rows)} of {total}) ===")
    logger.debug(header_row)
    logger.debug("-" * len(header_row))

    if rows:
        for row in rows:
            logger.debug(_format_table_row(row, widths))
    else:
        logger.debug("(no matches)")


def _display_search_results(gedcom_matches: MatchList, api_matches: MatchList, max_to_show: int) -> None:
    """Display GEDCOM and API search results in tables."""
    try:
        create_row_gedcom, create_row_api = _load_result_row_builders()
    except Exception as exc:
        logger.error(f"Unable to load search-result row builders: {exc}", exc_info=True)
        print("Unable to display search results (internal helper unavailable).")
        return

    headers = ["ID", "Name", "Birth", "Birth Place", "Death", "Death Place", "Total"]
    left_rows: list[list[str]] = [create_row_gedcom(m) for m in gedcom_matches[:max_to_show]]
    right_rows: list[list[str]] = [create_row_api(m) for m in api_matches[:max_to_show]]

    if not left_rows and not right_rows:
        print("")
        print("No matches found.")
        return

    if logger.isEnabledFor(logging.DEBUG):
        _log_result_table("GEDCOM", left_rows, len(gedcom_matches), headers)
        _log_result_table("API", right_rows, len(api_matches), headers)
        logger.debug("")
        logger.debug(
            f"Summary: GEDCOM — showing top {len(left_rows)} of {len(gedcom_matches)} total | "
            f"API — showing top {len(right_rows)} of {len(api_matches)} total"
        )


def _display_detailed_match_info(
    gedcom_matches: MatchList,
    api_matches: MatchList,
    gedcom_data: Any,
    _reference_person_id_raw: Optional[str],
    _reference_person_name: Optional[str],
    session_manager: SessionManager,
) -> None:
    """Display detailed information for top match."""
    normalize_id: Optional[IDNormalizer] = None
    analyze_top_match_fn: Optional[MatchAnalyzer] = None
    supplementary_handler: Optional[SupplementaryHandler] = None

    try:
        normalize_id, analyze_top_match_fn, supplementary_handler = _load_match_analysis_helpers()
    except Exception as exc:
        logger.error(f"Unable to load match analysis helpers: {exc}", exc_info=True)

    try:
        if gedcom_matches and gedcom_data is not None and analyze_top_match_fn is not None:
            if _reference_person_id_raw and normalize_id is not None:
                ref_norm = normalize_id(_reference_person_id_raw)
            else:
                ref_norm = _reference_person_id_raw
            analyze_top_match_fn(
                gedcom_data,
                gedcom_matches[0],
                ref_norm,
                _reference_person_name or "Reference Person",
            )
    except Exception as e:
        logger.error(f"GEDCOM family/relationship display failed: {e}")

    try:
        if api_matches and not gedcom_matches and supplementary_handler is not None:
            supplementary_handler(api_matches[0], session_manager)
    except Exception as e:
        logger.error(f"API family/relationship display failed: {e}")


def _collect_comparison_inputs() -> Optional[tuple[_ComparisonConfig, dict[str, Any]]]:
    """Load search criteria plus configuration needed for comparison mode."""

    try:
        from action10 import validate_config
        from search_criteria_utils import get_unified_search_criteria
    except Exception as exc:
        logger.error(f"Side-by-side setup failed: {exc}", exc_info=True)
        return None

    criteria = get_unified_search_criteria()
    if not criteria:
        return None

    (
        gedcom_path,
        reference_person_id_raw,
        reference_person_name,
        date_flex,
        scoring_weights,
        max_display_results,
    ) = validate_config()

    config = _ComparisonConfig(
        gedcom_path=gedcom_path,
        reference_person_id_raw=reference_person_id_raw,
        reference_person_name=reference_person_name,
        date_flex=date_flex,
        scoring_weights=scoring_weights,
        max_display_results=max_display_results,
    )
    return config, criteria


def _execute_comparison_search(
    session_manager: SessionManager,
    *,
    comparison_config: _ComparisonConfig,
    criteria: dict[str, Any],
) -> _ComparisonResults:
    """Run GEDCOM search followed by API fallback when needed."""

    gedcom_data, gedcom_matches = _perform_gedcom_search(
        comparison_config.gedcom_path,
        criteria,
        comparison_config.scoring_weights,
        comparison_config.date_flex,
    )

    api_matches: MatchList = []
    if not gedcom_matches:
        api_matches = _perform_api_search_fallback(
            session_manager,
            criteria,
            comparison_config.max_display_results,
        )
    else:
        logger.debug("Skipping API search because GEDCOM returned matches.")

    return _ComparisonResults(
        gedcom_data=gedcom_data,
        gedcom_matches=gedcom_matches,
        api_matches=api_matches,
    )


def _render_comparison_results(
    session_manager: SessionManager,
    *,
    comparison_config: _ComparisonConfig,
    comparison_results: _ComparisonResults,
) -> None:
    """Display summary tables, detail view, and analytics for comparison mode."""

    _display_search_results(
        comparison_results.gedcom_matches,
        comparison_results.api_matches,
        max_to_show=1,
    )

    _display_detailed_match_info(
        comparison_results.gedcom_matches,
        comparison_results.api_matches,
        comparison_results.gedcom_data,
        comparison_config.reference_person_id_raw,
        comparison_config.reference_person_name,
        session_manager,
    )

    _set_comparison_mode_analytics(
        len(comparison_results.gedcom_matches),
        len(comparison_results.api_matches),
    )


def run_gedcom_then_api_fallback(session_manager: SessionManager, *_: Any) -> bool:
    """Action 10: GEDCOM-first search with API fallback; unified presentation (header → family → relationship)."""
    collected = _collect_comparison_inputs()
    if not collected:
        return False

    comparison_config, criteria = collected
    comparison_results = _execute_comparison_search(
        session_manager,
        comparison_config=comparison_config,
        criteria=criteria,
    )

    _render_comparison_results(
        session_manager,
        comparison_config=comparison_config,
        comparison_results=comparison_results,
    )

    return bool(
        comparison_results.gedcom_matches or comparison_results.api_matches
    )


# End of run_action11_wrapper


def _get_windows_console_handles() -> tuple[Optional[Any], Optional[Any]]:
    """Return kernel32/user32 handles when available on Windows."""

    if os.name != "nt":
        return None, None

    try:
        import ctypes
    except Exception:
        return None, None

    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return None, None

    kernel32 = getattr(windll, "kernel32", None)
    user32 = getattr(windll, "user32", None)
    return kernel32, user32


def _set_windows_console_focus() -> None:
    """Ensure terminal window has focus on Windows."""

    kernel32, user32 = _get_windows_console_handles()
    if kernel32 is None or user32 is None:
        return

    try:
        console_window = kernel32.GetConsoleWindow()
    except Exception:
        return

    if not console_window:
        return

    try:
        user32.SetForegroundWindow(console_window)
        user32.ShowWindow(console_window, 9)  # SW_RESTORE
    except Exception:
        logger.debug("Unable to focus Windows console window", exc_info=True)


def _print_config_error_message() -> None:
    """Print detailed configuration error message and exit."""
    logger.critical("Configuration validation failed - unable to proceed")
    print("\n❌ CONFIGURATION ERROR:")
    print("   Critical configuration validation failed.")
    print("   This usually means missing credentials or configuration files.")
    print("")
    print("💡 SOLUTION:")
    print("   1. Copy .env.example to .env and add your credentials")
    print("   2. Ensure all required environment variables are set")

    print("\n📚 For detailed instructions:")
    print("   See ENV_IMPORT_GUIDE.md or readme.md")

    print("\nExiting application...")
    sys.exit(1)


def _check_action_confirmation(choice: str) -> bool:
    """
    Check if action requires confirmation and get user confirmation.

    Returns:
        True if action should proceed, False if cancelled

    """
    action_id, _ = _parse_menu_choice(choice)
    metadata = _get_action_metadata(action_id)

    if metadata and metadata.requires_confirmation:
        action_desc = metadata.confirmation_message or metadata.name
        confirm = (
            input(
                f"Are you sure you want to {action_desc}? ⚠️  This cannot be undone. (yes/no): "
            )
            .strip()
            .lower()
        )
        if confirm not in {"yes", "y"}:
            print("Action cancelled.\n")
            return False
        print(" ")  # Newline after confirmation

    return True


def _run_main_tests() -> None:
    """Run Main.py Internal Tests."""
    try:
        from tests.main_module_suite import run_comprehensive_tests as run_main_suite

        print("\n" + "=" * 60)
        print("RUNNING MAIN.PY INTERNAL TESTS")
        print("=" * 60)
        result = run_main_suite()
        if result:
            print("\n🎉 All main.py tests completed successfully!")
        else:
            print("\n⚠️ Some main.py tests failed. Check output above.")
    except Exception as e:
        logger.error(f"Error running main.py tests: {e}")
        print(f"Error running main.py tests: {e}")
    print("\nReturning to main menu...")
    input("Press Enter to continue...")


def _run_all_tests() -> None:
    """Run All Module Tests."""
    try:
        import subprocess

        print("\n" + "=" * 60)
        print("RUNNING ALL MODULE TESTS")
        print("=" * 60)
        result = subprocess.run(
            [sys.executable, "run_all_tests.py"],
            check=False, capture_output=False,
            text=True,
        )
        if result.returncode == 0:
            print("\n🎉 All module tests completed successfully!")
        else:
            print(f"\n⚠️ Some tests failed (exit code: {result.returncode})")
    except FileNotFoundError:
        print("Error: run_all_tests.py not found in current directory.")
    except Exception as e:
        logger.error(f"Error running all tests: {e}")
        print(f"Error running all tests: {e}")
    print("\nReturning to main menu...")
    input("Press Enter to continue...")


def _open_graph_visualization() -> None:
    """Launch local web server and open the code graph visualization."""
    server: Optional[ThreadingHTTPServer] = None
    try:
        root_dir = Path(__file__).resolve().parent
        preferred_port = 8765

        class GraphRequestHandler(SimpleHTTPRequestHandler):
            """Custom handler serving project files quietly."""

            directory = str(root_dir)

            def log_message(self, format: str, *args: Any) -> None:
                client_host, client_port = getattr(self, "client_address", ("?", "?"))
                logger.debug(
                    "Graph server %s:%s - %s",
                    client_host,
                    client_port,
                    format % args,
                )

        for candidate_port in range(preferred_port, preferred_port + 20):
            try:
                server = ThreadingHTTPServer(("127.0.0.1", candidate_port), GraphRequestHandler)
                preferred_port = candidate_port
                break
            except OSError:
                continue

        if server is None:
            print("❌ Unable to start the graph visualization server (no open ports).")
            logger.error("Graph visualization server failed to start: no open ports available")
            input("\nPress Enter to continue...")
            return

        server_thread = threading.Thread(
            target=server.serve_forever,
            name="GraphVisualizationServer",
            daemon=True,
        )
        server_thread.start()

        url = f"http://127.0.0.1:{preferred_port}/visualize_code_graph.html"
        print("\n" + "=" * 70)
        print("CODE GRAPH VISUALIZATION")
        print("=" * 70)
        print(f"Serving from: {root_dir}")
        print(f"URL: {url}")
        print("\nPress Enter when you are finished exploring the visualization.")
        print("=" * 70)

        try:
            webbrowser.open(url, new=1)
        except webbrowser.Error as browser_err:
            logger.warning(f"Unable to open browser automatically: {browser_err}")
            print("⚠️  Please open the URL manually in your browser.")

        input("\nPress Enter to stop the visualization server and return to the menu...")

    except Exception as graph_error:
        logger.error(f"Error running graph visualization: {graph_error}", exc_info=True)
        print(f"Error running graph visualization: {graph_error}")
        input("\nPress Enter to continue...")
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
            logger.info("Graph visualization server stopped")


def _show_analytics_dashboard() -> None:
    """Display conversation analytics dashboard."""
    try:
        from conversation_analytics import print_analytics_dashboard
        from core.session_manager import SessionManager

        print("\n" + "=" * 80)
        print("LOADING ANALYTICS DASHBOARD")
        print("=" * 80)

        # Get database session
        sm = SessionManager()
        db_session = sm.get_db_conn()

        if not db_session:
            print("✗ Failed to get database session")
            logger.error("Failed to get database session for analytics")
            return

        # Display analytics dashboard
        print_analytics_dashboard(db_session)

    except Exception as e:
        logger.error(f"Error displaying analytics dashboard: {e}", exc_info=True)
        print(f"Error displaying analytics dashboard: {e}")

    print("\nReturning to main menu...")
    input("Press Enter to continue...")


_CACHE_KIND_ICONS = {
    "disk": "📁",
    "memory": "🧠",
    "session": "🔐",
    "system": "⚙️",
    "gedcom": "🌳",
    "performance": "📊",
    "database": "🗄️",
    "retention": "🧹",
}


def _format_cache_stat_value(value: Any) -> str:
    """Format cache stat values for console display."""

    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, (list, tuple, set)):
        return f"{len(value)} items"
    if isinstance(value, dict):
        preview_items = list(value.items())[:3]
        preview = ", ".join(f"{k}={v}" for k, v in preview_items)
        if len(value) > 3:
            preview += ", ..."
        return f"{{{preview}}}"
    return str(value)


def _render_retention_targets(targets: Any) -> bool:
    if not (isinstance(targets, list) and targets and isinstance(targets[0], dict)):
        return False

    print("  Targets:")
    now_ts = time.time()
    for target in targets:
        name = target.get("name", "?")
        files = target.get("files_remaining", target.get("files_scanned", "?"))
        size_bytes = target.get("total_size_bytes", 0)
        size_mb = (size_bytes / (1024 * 1024)) if isinstance(size_bytes, (int, float)) else 0.0
        deleted = target.get("files_deleted", 0)
        run_ts = target.get("run_timestamp")
        if isinstance(run_ts, (int, float)) and run_ts:
            age_minutes = max(0.0, (now_ts - run_ts) / 60)
            age_str = f"{age_minutes:.1f}m ago"
        else:
            age_str = "n/a"
        print(
            f"    - {name}: {files} files, {size_mb:.2f} MB, removed {deleted} ({age_str})"
        )
    return True


def _render_stat_fields(stats: dict[str, Any]) -> bool:
    shown_any = False
    for key in sorted(stats.keys()):
        if key in {"name", "kind", "health", "targets"}:
            continue
        value = stats[key]
        if value in (None, "", [], {}):
            continue
        print(f"  {key.replace('_', ' ').title()}: {_format_cache_stat_value(value)}")
        shown_any = True
    return shown_any


def _render_health_stats(health: Any) -> bool:
    if not (isinstance(health, dict) and health):
        return False
    score = health.get("overall_score")
    score_str = f"{score:.1f}" if isinstance(score, (int, float)) else str(score)
    print(f"  Health Score: {score_str}")
    recommendations = health.get("recommendations")
    if recommendations:
        print(f"  Recommendations: {len(recommendations)}")
    return True


def _print_cache_component(component_name: str, stats: dict[str, Any]) -> None:
    icon = _CACHE_KIND_ICONS.get(stats.get("kind", ""), "🗃️")
    display_name = stats.get("name", component_name).upper()
    kind = stats.get("kind", "unknown")
    print(f"{icon} {display_name} [{kind}]")
    print("-" * 70)

    had_output = False
    had_output |= _render_retention_targets(stats.get("targets"))
    had_output |= _render_stat_fields(stats)
    had_output |= _render_health_stats(stats.get("health"))

    if not had_output:
        print("  No statistics available for this component.")
    print()


def _show_cache_registry_stats() -> bool:
    """Display consolidated cache stats through CacheRegistry."""

    try:
        from core.cache_registry import get_cache_registry

        registry = get_cache_registry()
        summary = registry.summary()
        component_names = summary.get("registry", {}).get("names", [])
        if not component_names:
            return False

        for component_name in component_names:
            stats = summary.get(component_name, {})
            _print_cache_component(component_name, stats)

        registry_info = summary.get("registry", {})
        print("REGISTRY OVERVIEW")
        print("-" * 70)
        print(f"  Components: {registry_info.get('components', len(component_names))}")
        print(f"  Registered: {', '.join(component_names)}")
        print()
        return True
    except Exception as exc:
        logger.error("Failed to display cache registry stats: %s", exc, exc_info=True)
        return False


def _show_base_cache_stats() -> bool:
    """Show base disk cache statistics.

    Returns:
        True if stats were displayed, False otherwise
    """
    try:
        from cache import get_cache_stats
        base_stats = get_cache_stats()
        if base_stats:
            print("📁 DISK CACHE (Base System)")
            print("-" * 70)
            print(f"  Hits: {base_stats.get('hits', 0):,}")
            print(f"  Misses: {base_stats.get('misses', 0):,}")
            print(f"  Hit Rate: {base_stats.get('hit_rate', 0):.1f}%")
            print(f"  Entries: {base_stats.get('entries', 0):,} / {base_stats.get('max_entries', 'N/A')}")
            print(f"  Volume: {base_stats.get('volume', 0):,} bytes")
            print(f"  Cache Dir: {base_stats.get('cache_dir', 'N/A')}")
            print()
            return True
    except Exception as e:
        logger.debug(f"Could not get base cache stats: {e}")
    return False


def _show_unified_cache_stats() -> bool:
    """Show unified cache manager statistics.

    Returns:
        True if any stats were displayed, False otherwise
    """
    try:
        from cache_manager import get_unified_cache_manager
        unified_mgr = get_unified_cache_manager()
        comprehensive_stats = unified_mgr.get_comprehensive_stats()
        stats_shown = False

        # Session cache
        session_stats = comprehensive_stats.get('session_cache', {})
        if session_stats:
            print("🔐 SESSION CACHE")
            print("-" * 70)
            print(f"  Active Sessions: {session_stats.get('active_sessions', 0)}")
            print(f"  Tracked Sessions: {session_stats.get('tracked_sessions', 0)}")
            print(f"  Component TTL: {session_stats.get('component_ttl', 0)}s")
            print(f"  Session TTL: {session_stats.get('session_ttl', 0)}s")
            print()
            stats_shown = True

        # API cache
        api_stats = comprehensive_stats.get('api_cache', {})
        if api_stats:
            print("🌐 API CACHE")
            print("-" * 70)
            print(f"  Active Sessions: {api_stats.get('active_sessions', 0)}")
            print(f"  Cache Available: {api_stats.get('cache_available', False)}")
            print()
            stats_shown = True

        # System cache
        system_stats = comprehensive_stats.get('system_cache', {})
        if system_stats:
            print("⚙️  SYSTEM CACHE")
            print("-" * 70)
            print(f"  GC Collections: {system_stats.get('gc_collections', 0)}")
            print(f"  Memory Freed: {system_stats.get('memory_freed_mb', 0):.2f} MB")
            print(f"  Peak Memory: {system_stats.get('peak_memory_mb', 0):.2f} MB")
            print(f"  Current Memory: {system_stats.get('current_memory_mb', 0):.2f} MB")
            print()
            stats_shown = True

        return stats_shown
    except Exception as e:
        logger.debug(f"Could not get unified cache stats: {e}")
    return False


def _show_performance_cache_stats() -> bool:
    """Show performance cache (GEDCOM) statistics.

    Returns:
        True if stats were displayed, False otherwise
    """
    try:
        from performance_cache import get_cache_stats as get_perf_stats
        perf_stats = get_perf_stats()
        if perf_stats:
            print("📊 PERFORMANCE CACHE (GEDCOM)")
            print("-" * 70)
            print(f"  Memory Entries: {perf_stats.get('memory_entries', 0)}")
            print(f"  Memory Usage: {perf_stats.get('memory_usage_mb', 0):.2f} MB")
            print(f"  Memory Pressure: {perf_stats.get('memory_pressure', 0):.1f}%")
            print(f"  Disk Cache Dir: {perf_stats.get('disk_cache_dir', 'N/A')}")
            print()
            return True
    except Exception as e:
        logger.debug(f"Could not get performance cache stats: {e}")
    return False


def _show_cache_statistics() -> None:
    """Show comprehensive cache statistics from all cache subsystems."""
    try:
        os.system("cls" if os.name == "nt" else "clear")
        print("\n" + "=" * 70)
        print("CACHE STATISTICS")
        print("=" * 70 + "\n")

        stats_collected = _show_cache_registry_stats()

        # Fallback to legacy collectors while registry adoption continues
        if not stats_collected:
            stats_collected = any([
                _show_base_cache_stats(),
                _show_unified_cache_stats(),
                _show_performance_cache_stats(),
            ])

        if not stats_collected:
            print("No cache statistics available.")
            print("Caches may not be initialized yet.")

        logger.debug("Cache statistics displayed")
        print("=" * 70)

    except Exception as e:
        logger.error(f"Error displaying cache statistics: {e}", exc_info=True)
        print("Error displaying cache statistics. Check logs for details.")

    input("\nPress Enter to continue...")


def _run_schema_migrations_action() -> None:
    """Apply pending schema migrations and report the current version state."""

    print("\n" + "=" * 70)
    print("SCHEMA MIGRATIONS")
    print("=" * 70)

    db_manager: Optional[DatabaseManagerProtocol] = None
    try:
        from core import schema_migrator
        from core.database_manager import DatabaseManager

        db_manager = DatabaseManager()
        if db_manager is None or not db_manager.ensure_ready():
            print("Unable to initialize database engine. See logs for details.")
            return

        engine = getattr(db_manager, "engine", None)
        if engine is None:
            print("Unable to access database engine instance.")
            return

        registered_migrations = schema_migrator.get_registered_migrations()
        print(f"Registered migrations: {len(registered_migrations)}")

        applied_versions = schema_migrator.apply_pending_migrations(engine)
        installed_versions = schema_migrator.get_applied_versions(engine)

        if applied_versions:
            print(f"\nApplied migrations: {', '.join(applied_versions)}")
        else:
            print("\nNo pending migrations; schema already current.")

        if installed_versions:
            print(
                f"Installed versions ({len(installed_versions)}): "
                f"{', '.join(installed_versions)}"
            )
        else:
            print("Installed versions: none recorded.")

        pending_versions = [
            migration.version for migration in registered_migrations if migration.version not in installed_versions
        ]
        if pending_versions:
            print(f"Pending migrations ({len(pending_versions)}): {', '.join(pending_versions)}")
        else:
            print("All registered migrations have been applied.")
    except Exception as exc:
        logger.error("Failed to run schema migrations: %s", exc, exc_info=True)
        print(f"Error applying migrations: {exc}")
    finally:
        if db_manager is not None:
            try:
                db_manager.close_connections(dispose_engine=True)
            except Exception:
                logger.debug("Failed to close temporary database manager", exc_info=True)
        input("\nPress Enter to continue...")


def _toggle_log_level() -> None:
    """Toggle console log level between DEBUG and INFO."""
    os.system("cls" if os.name == "nt" else "clear")
    if logger and logger.handlers:
        console_handler: Optional[StreamHandler[TextIO]] = None
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler_typed = cast(StreamHandler[TextIO], handler)
                if handler_typed.stream == sys.stderr:
                    console_handler = handler_typed
                    break
        if console_handler:
            current_level = console_handler.level
            new_level = (
                logging.DEBUG
                if current_level > logging.DEBUG
                else logging.INFO
            )
            new_level_name = logging.getLevelName(new_level)
            # Re-call setup_logging to potentially update filters etc. too
            setup_logging(log_level=new_level_name, allow_env_override=False)
            logger.info(f"Console log level toggled to: {new_level_name}")
        else:
            logger.warning(
                "Could not find console handler to toggle level."
            )
    else:
        print(
            "WARNING: Logger not ready or has no handlers.", file=sys.stderr
        )


def _show_metrics_report() -> None:
    """Open Grafana dashboard in browser."""
    try:
        import urllib.request
        import webbrowser

        from observability.metrics_registry import is_metrics_enabled

        print("\n" + "=" * 70)
        print("📊 GRAFANA METRICS DASHBOARD")
        print("=" * 70)

        if not is_metrics_enabled():
            print("\n⚠️  Metrics collection is DISABLED")
            print("\nTo enable metrics:")
            print("  1. Add to .env: PROMETHEUS_METRICS_ENABLED=true")
            print("  2. Optionally configure: PROMETHEUS_METRICS_PORT=9000")
            print("  3. Restart the application")
            print("\n" + "=" * 70 + "\n")
            return

        # Check if Grafana is running
        grafana_base = "http://localhost:3000"
        try:
            urllib.request.urlopen(grafana_base, timeout=1)
            grafana_running = True
        except Exception:
            grafana_running = False

        if not grafana_running:
            print("\n⚠️  Grafana is NOT running on http://localhost:3000")
            print("\n💡 Setup Instructions:")
            print("   1. Install Grafana: https://grafana.com/grafana/download")
            print("   2. Start Grafana service")
            print("   3. Login at http://localhost:3000 (default: admin/admin)")
            print("   4. Add Prometheus data source → http://localhost:9000")
            print("   5. Import dashboard: docs/grafana/ancestry_overview.json")
            print("\n📊 For now, opening raw metrics at http://localhost:9000/metrics")
            print("\n" + "=" * 70 + "\n")
            webbrowser.open("http://localhost:9000/metrics")
            return

        # Grafana is running - check and import dashboards if needed
        print("\n✅ Grafana is running!")
        print("🔍 Checking dashboards...")

        # Try to import dashboards automatically
        if grafana_checker:
            try:
                # This will attempt to import missing dashboards
                grafana_checker.ensure_dashboards_imported()
            except Exception as import_err:
                logger.debug(f"Dashboard auto-import check: {import_err}")

        system_perf_url = f"{grafana_base}/d/ancestry-performance"
        genealogy_url = f"{grafana_base}/d/ancestry-genealogy"
        code_quality_url = f"{grafana_base}/d/ancestry-code-quality"

        print("🌐 Opening dashboards:")
        print(f"   1. System Performance & Health: {system_perf_url}")
        print(f"   2. Genealogy Research Insights: {genealogy_url}")
        print(f"   3. Code Quality & Architecture: {code_quality_url}")
        print("\n💡 If dashboards show 'Not found', run: setup-grafana")
        print("\n" + "=" * 70 + "\n")

        webbrowser.open(system_perf_url)
        time.sleep(0.5)  # Small delay between opening tabs
        webbrowser.open(genealogy_url)
        time.sleep(0.5)
        webbrowser.open(code_quality_url)

    except Exception as e:
        logger.error(f"Error opening Grafana: {e}", exc_info=True)
        print(f"\n⚠️  Error: {e}")
        print("\n" + "=" * 70 + "\n")


def _run_grafana_setup() -> None:
    """Run Grafana setup if available."""

    if grafana_checker:
        status = grafana_checker.check_grafana_status()
        if status["ready"]:
            print("\n✅ Grafana is already fully configured and running!")
            print("   Dashboard URL: http://localhost:3000")
            print("   Default credentials: admin / ancestry")
            print("\n📊 Checking dashboards...")
            grafana_checker.ensure_dashboards_imported()
            print("\n✅ Dashboard check complete!")
            print("\n📊 Available Dashboards:")
            print("   • Overview:    http://localhost:3000/d/ancestry-overview")
            print("   • Performance: http://localhost:3000/d/ancestry-performance")
            print("   • Genealogy:   http://localhost:3000/d/ancestry-genealogy")
            print("   • Code Quality: http://localhost:3000/d/ancestry-code-quality")
            print("\n💡 If dashboards are empty, configure data sources:")
            print("   Run: .\\docs\\grafana\\configure_datasources.ps1\n")
        else:
            grafana_checker.ensure_grafana_ready(auto_setup=False, silent=False)
    else:
        print("\n⚠️  Grafana checker module not available")
        print("Ensure grafana_checker.py is in the project root directory\n")


def _clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _exit_application() -> bool:
    _clear_screen()
    print("Exiting.")
    return False


def _execute_meta_action(metadata: ActionMetadata) -> Optional[bool]:
    """Execute a meta action and return loop control signal."""

    action_func = metadata.function
    if not callable(action_func):
        logger.error("Meta action '%s' is not wired to a function", metadata.id)
        print("Action not available.\n")
        return True

    result = action_func()
    if isinstance(result, bool):
        return result
    return True


def _execute_test_action(metadata: ActionMetadata) -> None:
    """Execute a registered test action."""

    action_func = metadata.function
    if not callable(action_func):
        logger.error("Test action '%s' is not wired to a function", metadata.id)
        print("Test action not available.\n")
        return
    action_func()


def _validate_action_args(metadata: ActionMetadata, arg_tokens: list[str]) -> bool:
    """Validate CLI arguments against action metadata."""

    if not arg_tokens:
        return True
    if metadata.max_args == 0:
        print(f"Action '{metadata.name}' does not accept additional arguments.\n")
        return False
    if len(arg_tokens) > metadata.max_args:
        print(f"Action '{metadata.name}' accepts at most {metadata.max_args} argument(s).\n")
        return False
    return True


def _parse_start_page_argument(arg_tokens: list[str]) -> Optional[int]:
    """Parse optional start page for Action 6."""

    if not arg_tokens:
        return None

    start_token = arg_tokens[0]
    try:
        start_arg = int(start_token)
        if start_arg > 0:
            return start_arg
        logger.warning("Invalid start page '%s'. Using checkpoint resume if available.", start_token)
        print(f"Invalid start page '{start_token}'. Defaulting to checkpoint resume.")
    except ValueError:
        logger.warning("Invalid start page '%s'. Using checkpoint resume if available.", start_token)
        print(f"Invalid start page '{start_token}'. Defaulting to checkpoint resume.")
    return None


def _execute_primary_action(
    metadata: ActionMetadata,
    session_manager: SessionManager,
    config_obj: Any,
    arg_tokens: list[str],
) -> bool:
    """Execute a primary action (non-meta/test) using exec_actn."""

    action_func = metadata.function
    if not callable(action_func):
        logger.error("Action '%s' is not wired to an executable function", metadata.id)
        print("Action not implemented yet.\n")
        return True

    if not _validate_action_args(metadata, arg_tokens):
        return True

    extra_args: list[Any] = []
    if metadata.inject_config:
        extra_args.append(config_obj)

    if metadata.id == "6":
        extra_args.append(_parse_start_page_argument(arg_tokens))
    elif arg_tokens:
        logger.debug("Ignoring unused arguments %s for action %s", arg_tokens, metadata.id)

    if metadata.enable_caching:
        ensure_caching_initialized()

    exec_actn(action_func, session_manager, metadata.id, metadata.close_session_after, *extra_args)
    return True


def _dispatch_menu_action(choice: str, session_manager: SessionManager, config: Any) -> bool:
    """
    Dispatch menu action based on user choice.

    Returns:
        True to continue menu loop, False to exit
    """
    action_id, arg_tokens = _parse_menu_choice(choice)
    metadata = _get_action_metadata(action_id)

    if metadata is None:
        print("Invalid choice.\n")
        return True

    if metadata.is_meta_action:
        if arg_tokens:
            print("This option does not accept arguments.\n")
            return True
        result = _execute_meta_action(metadata)
        return True if result is None else bool(result)

    if metadata.is_test_action:
        if arg_tokens:
            print("This option does not accept arguments.\n")
            return True
        _execute_test_action(metadata)
        return True

    return _execute_primary_action(metadata, session_manager, config, arg_tokens)


def _assign_action_registry_functions() -> None:
    """Attach callable implementations to action registry metadata."""

    registry = get_action_registry()
    registry.set_action_function("1", all_but_first_actn)
    registry.set_action_function("2", reset_db_actn)
    registry.set_action_function("3", backup_db_actn)
    registry.set_action_function("4", restore_db_actn)
    registry.set_action_function("5", check_login_actn)
    registry.set_action_function("6", gather_dna_matches)
    registry.set_action_function("7", srch_inbox_actn)
    registry.set_action_function("8", send_messages_action)
    registry.set_action_function("9", process_productive_messages_action)
    registry.set_action_function("10", run_gedcom_then_api_fallback)

    registry.set_action_function("analytics", _show_analytics_dashboard)
    registry.set_action_function("metrics", _show_metrics_report)
    registry.set_action_function("setup-grafana", _run_grafana_setup)
    registry.set_action_function("graph", _open_graph_visualization)
    registry.set_action_function("s", _show_cache_statistics)
    registry.set_action_function("migrate-db", _run_schema_migrations_action)
    registry.set_action_function("t", _toggle_log_level)
    registry.set_action_function("c", _clear_screen)
    registry.set_action_function("q", _exit_application)

    registry.set_action_function("test", _run_main_tests)
    registry.set_action_function("testall", _run_all_tests)


_assign_action_registry_functions()


def _check_startup_status(session_manager: SessionManager) -> None:
    """
    Check and display database connection status at startup.
    Authentication status is already displayed by the authentication steps above.
    """
    # Check database connection
    try:
        db_manager = _get_database_manager(session_manager)
        if db_manager and db_manager.ensure_ready():
            logger.info("✅ Database connection OK")
        else:
            logger.warning("⚠️ Database connection not available")
    except Exception as e:
        logger.warning(f"⚠️ Database connection check failed: {e}")

    # CRITICAL: Proactive cookie sync during warmup
    # Ensures fresh cookies for ALL actions before menu display
    # Prevents 303 redirects from stale cookies across all actions
    try:
        api_manager = _get_api_manager(session_manager)
        browser_manager = _get_browser_manager(session_manager)
        if session_manager.is_sess_valid() and api_manager and browser_manager:
            logger.debug("Syncing browser cookies to API session during warmup...")
            synced = api_manager.sync_cookies_from_browser(
                browser_manager, session_manager=session_manager
            )
            if synced:
                logger.info("✅ Cookies refreshed and OK")
            else:
                logger.debug("Cookie sync returned False (may be normal)")
    except Exception as cookie_err:
        logger.debug(f"Cookie sync during warmup failed (non-fatal): {cookie_err}")


def _check_lm_studio_running() -> bool:
    """Check if LM Studio process is running.

    Returns:
        True if LM Studio is running, False otherwise
    """
    import psutil

    for proc in psutil.process_iter(['name']):
        try:
            proc_name = proc.info['name'].lower()
            if 'lm studio' in proc_name or 'lmstudio' in proc_name:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def _validate_local_llm_config(config_schema: Any) -> bool:
    """Validate local LLM configuration with auto-start support.

    Returns:
        True if validation passed, False otherwise
    """
    from openai import OpenAI

    api_key = config_schema.api.local_llm_api_key
    model_name = config_schema.api.local_llm_model
    base_url = config_schema.api.local_llm_base_url

    if not all([api_key, model_name, base_url]):
        logger.warning("⚠️ Local LLM configuration incomplete - AI features may not work")
        return False

    if not _check_lm_studio_running():
        logger.debug("LM Studio process not detected; manager will attempt startup if needed")

    # Auto-start LM Studio if not running (or verify it's ready)
    try:
        lm_module = import_module("lm_studio_manager")
        manager_factory = cast(
            Callable[[Any], Any],
            getattr(lm_module, "create_manager_from_config"),
        )
        lm_manager = manager_factory(config_schema)
        success, error_msg = lm_manager.ensure_ready()

        if not success:
            logger.warning(f"⚠️ LM Studio not ready: {error_msg}")
            return False
        logger.info(f"✅ LM Studio ready; verifying model '{model_name}' is loaded")

    except Exception as e:
        logger.warning(f"⚠️ Failed to start/verify LM Studio: {e}")
        logger.warning("   Please start LM Studio manually and load a model")
        return False

    # Check if model is loaded
    try:
        client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        ai_module = import_module("ai_interface")
        validate_llm_loaded = cast(
            Callable[[Any, str], tuple[Optional[str], Optional[str]]],
            getattr(ai_module, "_validate_local_llm_model_loaded"),
        )
        actual_model_name, error_msg = validate_llm_loaded(client, model_name)

        if error_msg:
            logger.warning(f"⚠️ {error_msg}")
            logger.warning("   Please load the model in LM Studio before using AI features")
            return False
        logger.info(f"✅ Local LLM {actual_model_name} OK")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Could not validate Local LLM: {e}")
        logger.warning("   AI features may not work until model is loaded in LM Studio")
        return False


def _validate_cloud_provider(provider_name: str, api_key: Any, model: Any) -> bool:
    """Validate cloud AI provider configuration.

    Args:
        provider_name: Name of the provider (DeepSeek, Gemini, etc.)
        api_key: API key for the provider
        model: Model name

    Returns:
        True if configured, False otherwise
    """
    if api_key and model:
        logger.info(f"✅ {provider_name} configured: {model}")
        return True
    logger.warning(f"⚠️ {provider_name} configuration incomplete")
    return False


def _validate_ai_provider_on_startup() -> None:
    """
    Validate AI provider configuration on startup.

    For local_llm: Checks if model is loaded in LM Studio
    For cloud providers: Just logs the configuration
    """
    from config import config_schema

    ai_provider = config_schema.ai_provider.lower() if config_schema.ai_provider else ""

    if not ai_provider:
        logger.debug("No AI provider configured - AI features will be disabled")
        return

    logger.debug(f"Validating AI provider: {ai_provider}")

    if ai_provider == "local_llm":
        _validate_local_llm_config(config_schema)
    elif ai_provider == "deepseek":
        _validate_cloud_provider("DeepSeek", config_schema.api.deepseek_api_key,
                                config_schema.api.deepseek_ai_model)
    elif ai_provider == "gemini":
        _validate_cloud_provider("Gemini", config_schema.api.google_api_key,
                                config_schema.api.google_ai_model)
    elif ai_provider == "moonshot":
        _validate_cloud_provider("Moonshot", config_schema.api.moonshot_api_key,
                                config_schema.api.moonshot_ai_model)
    elif ai_provider == "inception":
        _validate_cloud_provider("Inception", config_schema.api.inception_api_key,
                                config_schema.api.inception_ai_model)
    else:
        logger.warning(f"⚠️ Unknown AI provider: {ai_provider}")


def _display_tree_owner(session_manager: SessionManager) -> None:
    """Display tree owner name at the end of startup checks."""
    try:
        # Get tree owner if available - this will log it via session_manager
        api_manager = _get_api_manager(session_manager) if session_manager else None
        if api_manager:
            owner_name = api_manager.tree_owner_name
            if owner_name:
                print("")
                logger.info(f"Tree owner name: {owner_name}\n")
    except Exception:
        pass  # Silently ignore - not critical for startup


def _initialize_application() -> tuple["SessionManager", Any]:
    """Initialize application logging, configuration, and sleep prevention."""
    print("")

    # Clear log file before initializing logging (uses .env settings)
    try:
        log_dir = Path(os.getenv("LOG_DIR", "Logs"))
        if not log_dir.is_absolute():
            log_dir = (Path(__file__).parent / log_dir).resolve()
        log_file = os.getenv("LOG_FILE", "app.log")
        log_path = log_dir / log_file
        if log_path.exists():
            log_path.write_text("", encoding="utf-8")
    except Exception:
        pass  # Silently ignore if can't clear

    setup_logging()
    validate_action_config()

    sleep_state: Any = None
    try:
        from utils import prevent_system_sleep

        sleep_state = prevent_system_sleep()
    except Exception as sleep_err:
        logger.warning(f"⚠️ System sleep prevention unavailable: {sleep_err}")

    print(" Checks ".center(80, "="))
    if sleep_state is not None:
        logger.info("✅ System sleep prevention active")
    else:
        logger.info("⚠️ System sleep prevention inactive")

    logger.info("✅ Action registry initialized (%d actions)", len(get_action_registry().get_all_actions()))

    # Run Chrome/ChromeDriver diagnostics before any browser automation (silent mode)
    try:
        from diagnose_chrome import run_silent_diagnostic
        success, message = run_silent_diagnostic()
        if success:
            logger.info("✅ Chrome/ChromeDriver OK")
        else:
            logger.warning(f"⚠️  Chrome diagnostic issue: {message}")
    except Exception as diag_error:
        logger.warning(f"Chrome diagnostics failed to run: {diag_error}")

    # Check Grafana installation status
    if grafana_checker:
        try:
            grafana_status = grafana_checker.check_grafana_status()
            if grafana_status["ready"]:
                logger.info("✅ Grafana ready (http://localhost:3000)")
            elif grafana_status["installed"]:
                logger.info("⚠️  Grafana installed but not fully configured (run 'setup-grafana' from menu)")
            else:
                logger.info("💡 Grafana not installed (run 'setup-grafana' from menu for automated setup)")
        except Exception as grafana_error:
            logger.debug(f"Grafana check skipped: {grafana_error}")

    if config is None:
        _print_config_error_message()

    session_manager = SessionManager()

    from session_utils import set_global_session
    set_global_session(session_manager)
    logger.debug("✅ SessionManager registered as global session")

    return session_manager, sleep_state


def _pre_authenticate_session() -> None:
    """
    Pre-authenticate the global session with proper browser startup.
    CRITICAL: This ensures the global session is fully authenticated and ready
    before the menu displays, preventing authentication delays during actions.
    """
    try:
        from session_utils import get_authenticated_session

        # Authenticate session - this will start browser if needed
        logger.debug("Pre-authenticating global session for immediate availability...")
        session_manager, uuid = get_authenticated_session(
            action_name="Main Menu Initialization",
            skip_csrf=False
        )

        # Verify session is actually ready
        if session_manager and session_manager.session_ready:
            logger.info("✅ Global session authenticated and ready")
            logger.debug(f"   UUID: {uuid}")
        else:
            logger.warning("⚠️ Session authentication incomplete - will retry during action")

    except Exception as e:
        logger.warning(f"⚠️ Session pre-authentication failed: {e}")
        logger.warning("   Session will be authenticated when first action requires it")


def _pre_authenticate_ms_graph() -> None:
    """Pre-authenticate MS Graph for MS To-Do integration."""
    try:
        from ms_graph_utils import acquire_token_device_flow
        logger.debug("Attempting MS Graph authentication at startup...")
        ms_token = acquire_token_device_flow()
        if ms_token:
            logger.info("✅ MS Graph authenticated OK")
        else:
            logger.debug("MS Graph authentication skipped or failed (will retry during Action 9 if needed)")
    except Exception as e:
        logger.debug(f"MS Graph authentication failed at startup: {e}")


def _cleanup_session_manager(session_manager: Optional[Any]) -> None:
    """Clean up session manager on shutdown with proper resource ordering.

    Args:
        session_manager: SessionManager instance to clean up
    """

    if session_manager is not None:
        try:
            import contextlib
            import gc
            import io

            # Suppress stderr during cleanup to hide undetected_chromedriver errors
            with contextlib.redirect_stderr(io.StringIO()):
                session_manager.close_sess(keep_db=False)

            # Clear reference immediately to prevent delayed destruction
            session_manager = None

            # Force garbage collection while process is still active
            # This ensures Chrome cleanup happens before Windows handles are invalidated
            gc.collect()

            logger.debug("✅ Session manager closed cleanly")
        except Exception as final_close_e:
            logger.debug(f"⚠️  Cleanup warning (non-critical): {final_close_e}")

    print("\nExit.")


def main() -> None:
    session_manager = None
    sleep_state = None  # Track sleep prevention state
    _set_windows_console_focus()

    try:
        # Initialize application (handles sleep prevention and diagnostics)
        session_manager, sleep_state = _initialize_application()

        # Pre-authenticate services
        _pre_authenticate_session()
        _pre_authenticate_ms_graph()

        # Check startup status and validate AI provider
        _check_startup_status(session_manager)
        _validate_ai_provider_on_startup()

        # Display tree owner at the end of startup checks
        _display_tree_owner(session_manager)

        # Main menu loop
        while True:
            choice = menu()
            print("")

            if not _check_action_confirmation(choice):
                continue

            if not _dispatch_menu_action(choice, session_manager, config):
                break  # Exit requested

    except KeyboardInterrupt:
        os.system("cls" if os.name == "nt" else "clear")
        print("\nCTRL+C detected. Exiting.")
    except Exception as e:
        logger.critical(f"Critical error in main: {e}", exc_info=True)
    finally:
        # Restore system sleep settings
        if sleep_state is not None:
            from utils import restore_system_sleep
            restore_system_sleep(sleep_state)
            logger.info("🔓 System sleep prevention deactivated")

        import contextlib
        import io
        # Suppress all stderr output during cleanup to hide undetected_chromedriver errors
        with contextlib.redirect_stderr(io.StringIO()):
            _cleanup_session_manager(session_manager)
            # Small delay to allow cleanup to complete before exit
            time.sleep(0.2)


# end main


# --- Entry Point ---

if __name__ == "__main__":
    import io
    import sys

    # Run main program
    main()

    # Suppress stderr during final garbage collection to hide undetected_chromedriver cleanup errors
    sys.stderr = io.StringIO()


# end of main.py
