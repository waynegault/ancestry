import contextlib
import gc
import io
import os
import shutil
import sys
import time
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
import logging
from unittest import mock

from sqlalchemy import create_engine, func, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as SASession

from actions.gather.checkpoint import clear_checkpoint
from caching.cache import clear_cache
from core.action_runner import DatabaseManagerProtocol
from core.database import (
    Base,
    ConversationLog,
    DnaMatch,
    FamilyTree,
    MessageTemplate,
    Person,
)
from core.database_manager import backup_database, db_transn
from core.session_manager import SessionManager
from core.utils import log_in, login_status
from testing.test_framework import TestSuite, create_standard_test_runner

logger = logging.getLogger(__name__)

# Initialize config
from config.config_manager import get_config_manager

config_manager = get_config_manager()
config = config_manager.get_config()


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
            sess, ConversationLog, ConversationLog.people_id != person_id_to_keep, "conversation_log", person_id_to_keep
        ),
        "dna_match": _delete_table_records(
            sess, DnaMatch, DnaMatch.people_id != person_id_to_keep, "dna_match", person_id_to_keep
        ),
        "family_tree": _delete_table_records(
            sess, FamilyTree, FamilyTree.people_id != person_id_to_keep, "family_tree", person_id_to_keep
        ),
        "people": _delete_table_records(sess, Person, Person.id != person_id_to_keep, "people", person_id_to_keep),
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
            logger.debug(f"Closing main DB connections before deleting data (except {profile_id_to_keep})...")
            session_manager.db_manager.close_connections(dispose_engine=True)
            logger.debug("Main DB pool closed.")
        # --- End closing main pool ---

        logger.debug(f"Deleting data for all people except Profile ID: {profile_id_to_keep}...")
        # Create a temporary SessionManager for this specific operation
        temp_manager = SessionManager()
        session = temp_manager.db_manager.get_session()
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
                .filter(Person.profile_id == profile_id_to_keep, Person.deleted_at.is_(None))
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
        logger.error(f"Error during deletion (except {profile_id_to_keep}): {e}", exc_info=True)
        success = False  # Explicitly mark failure
    finally:
        # Clean up the temporary session manager and its resources
        if temp_manager:
            if session:
                temp_manager.db_manager.return_session(session)
            temp_manager.db_manager.close_connections(dispose_engine=True)  # Close the temp pool
        logger.debug(f"Delete action (except {profile_id_to_keep}) finished.")
    return success


# Action 2 (reset_db_actn)
def _truncate_all_tables(temp_manager: SessionManager) -> bool:
    """Truncate all tables in the database."""
    logger.debug("Truncating all tables...")
    truncate_session = temp_manager.db_manager.get_session()
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
        temp_manager.db_manager.return_session(truncate_session)
        logger.debug("All tables truncated successfully.")
        return True
    except Exception as e:
        logger.error(f"Error truncating tables: {e}", exc_info=True)
        temp_manager.db_manager.return_session(truncate_session)
        return False


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


def _reinitialize_database_schema(temp_manager: SessionManager) -> bool:
    """Re-initialize database schema by dropping and recreating all tables."""
    logger.debug("Re-initializing database schema...")
    try:
        # We need to access the database manager from the session manager
        # Since SessionManager doesn't expose it directly, we might need to use a protected member or helper
        # In main.py it used _get_database_manager from core.action_runner
        # Here we can try to access it if it's available or import the helper
        from core.action_runner import get_database_manager as _get_database_manager

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
        database_module = import_module("core.database")

        get_default_templates = getattr(database_module, "_get_default_message_templates", None)
        if get_default_templates is None:
            logger.error("Default message template helper not available.")
            return False

        with db_transn(recreation_session) as sess:
            existing_count = sess.query(func.count(MessageTemplate.id)).scalar() or 0

            if existing_count > 0:
                logger.debug(f"Found {existing_count} existing message templates. Skipping seeding.")
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
        logger.debug(f"MessageTemplate seeding complete. Total templates in DB: {count}")
        return True
    except Exception as e:
        logger.error(f"Error seeding message templates: {e}", exc_info=True)
        return False


def _initialize_ethnicity_columns_from_metadata(db_manager: SessionManager) -> bool:
    """
    Initialize ethnicity columns in dna_match table using saved metadata file.
    This is browserless - it only adds columns if Data/ethnicity_regions.json exists.
    If the file doesn't exist, columns will be added during first Action 6 run.

    Args:
        db_manager: SessionManager with active database connection

    Returns:
        True if columns added successfully, False if metadata file doesn't exist
    """
    try:
        from genealogy.dna.dna_ethnicity_utils import initialize_ethnicity_columns_from_metadata

        logger.debug("Checking for saved ethnicity metadata...")
        return initialize_ethnicity_columns_from_metadata(db_manager)

    except Exception as e:
        logger.error(f"Error adding ethnicity columns from metadata: {e}", exc_info=True)
        return False


def _close_main_pool_for_reset(session_manager: SessionManager) -> None:
    """Close main pool and force garbage collection."""
    if session_manager:
        logger.debug("Closing main DB connections before database deletion...")
        session_manager.db_manager.close_connections(dispose_engine=True)  # Ensure pool is closed
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
    recreation_session = temp_manager.db_manager.get_session()
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
                # Clear Action 6 checkpoint to ensure fresh start
                try:
                    clear_checkpoint()
                    logger.info("✅ Action 6 checkpoint cleared.")
                except Exception as e:
                    logger.warning(f"Failed to clear Action 6 checkpoint: {e}")

                # Clear general cache (may fail if cache.db is locked - this is OK)
                try:
                    if clear_cache():
                        logger.info("✅ General cache cleared.")
                    # Note: False return is expected if cache.db is locked during reset
                except Exception as e:
                    logger.debug(f"Cache clear skipped: {e}")

                logger.info("✅ Database reset completed successfully.")

        except Exception as recreate_err:
            logger.error(f"Error during DB recreation/seeding: {recreate_err}", exc_info=True)
            reset_successful = False
        finally:
            # Clean up the temporary manager and its session/engine
            logger.debug("Cleaning up temporary resource manager for reset...")
            if temp_manager and recreation_session:
                temp_manager.db_manager.return_session(recreation_session)
            logger.debug("Temporary resource manager cleanup finished.")

    except Exception as e:
        logger.error(f"Outer error during DB reset action: {e}", exc_info=True)
        reset_successful = False

    finally:
        logger.debug("Reset DB action finished.")

    return reset_successful


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


# Action 4 (restore_db_actn)
def _display_table_statistics() -> None:
    """Display statistics for all tables in the database."""

    db_path = config.database.database_file
    if not db_path:
        logger.warning("Cannot display table statistics: DATABASE_FILE not configured")
        return

    try:
        engine = create_engine(f"sqlite:///{db_path}")
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        if not table_names:
            print("Database contains no tables")
            return

        print("Database Table Statistics:")
        print("-" * 60)

        with engine.connect() as conn:
            for table_name in sorted(table_names):
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                count = result.scalar()
                print(f"  {table_name:30s} {count:>10,} records")

        print("-" * 60)

    except Exception as e:
        logger.warning(f"Could not display table statistics: {e}")


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
            session_manager.db_manager.close_connections(dispose_engine=True)
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
        print()  # Blank line after restore confirmation

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


# Action 5 (check_login_actn)
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

    # We need api_manager to get csrf_token
    # In main.py it used _get_api_manager from core.action_runner
    from core.action_runner import get_api_manager as _get_api_manager

    api_manager = _get_api_manager(session_manager)
    if api_manager and api_manager.csrf_token:
        token = api_manager.csrf_token
        csrf_preview = token[:20] + "..." if len(token) > 20 else token
        print(f"CSRF Token:      {csrf_preview}")


def _handle_logged_in_status(session_manager: SessionManager) -> bool:
    """Handle the case when user is already logged in."""
    logger.info("You are currently logged in to Ancestry.")
    logger.info("")  # Blank line after login confirmation
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
    if not session_manager.browser_manager.driver_live:
        logger.error("Driver not live. Cannot check login status.")
        print("ERROR: Browser not started. Cannot check login status.")
        print("       Select any browser-required action (1, 6-9) to start the browser.")
        return False

    logger.info("Checking login status...")

    # Call login_status directly to check initial status
    try:
        status = login_status(session_manager, disable_ui_fallback=False)  # Use UI fallback for reliability

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


# ------------------------------------------------------------------
# Embedded regression tests
# ------------------------------------------------------------------


class _FakeQuery:
    def __init__(self, delete_result: int, scalar_values: list[int] | None = None) -> None:
        self.delete_result = delete_result
        self.scalar_values = scalar_values or []
        self.filter_args: list[Any] = []
        self.delete_kwargs: list[Any] = []

    def filter(self, condition: Any) -> "_FakeQuery":
        self.filter_args.append(condition)
        return self

    def delete(self, *, synchronize_session: Any) -> int:
        self.delete_kwargs.append(synchronize_session)
        return self.delete_result

    def scalar(self) -> int:
        if self.scalar_values:
            return self.scalar_values.pop(0)
        return 0


class _FakeSession:
    def __init__(self, delete_result: int = 1, scalar_values: list[int] | None = None) -> None:
        self.query_calls: list[Any] = []
        self.added: list[Any] = []
        self.query_instance = _FakeQuery(delete_result, scalar_values)

    def query(self, _table: Any) -> _FakeQuery:
        self.query_calls.append(_table)
        return self.query_instance

    def add(self, obj: Any) -> None:
        self.added.append(obj)


def _capture_stdout(func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func(*args, **kwargs)
    return buffer.getvalue()


def _test_delete_table_records_uses_session_query() -> bool:
    fake_session = _FakeSession(delete_result=2)
    session_for_call = cast(SASession, fake_session)
    count = _delete_table_records(session_for_call, Person, "condition", "people", 42)
    assert count == 2
    assert fake_session.query_calls == [Person]
    assert fake_session.query_instance.filter_args == ["condition"]
    assert fake_session.query_instance.delete_kwargs == [False]
    return True


def _test_perform_deletions_aggregates_counts() -> bool:
    with mock.patch(
        f"{__name__}._delete_table_records",
        side_effect=[1, 2, 3, 4],
    ) as patched:
        fake_session = SimpleNamespace()
        counts = _perform_deletions(cast(SASession, fake_session), 7)

    assert counts == {
        "conversation_log": 1,
        "dna_match": 2,
        "family_tree": 3,
        "people": 4,
    }
    assert patched.call_count == 4
    return True


def _test_initialize_db_manager_engine_invokes_initializer() -> bool:
    class FakeManager:
        def __init__(self) -> None:
            self.engine = None
            self.Session = None
            self.ready_checks = 0
            self.close_invocations: list[bool] = []

        def _initialize_engine_and_session(self) -> None:
            self.engine = "engine"
            self.Session = "session"

        def ensure_ready(self) -> bool:
            self.ready_checks += 1
            return True

        def close_connections(self, dispose_engine: bool = False) -> None:
            self.close_invocations.append(dispose_engine)

        def get_session(self) -> Any:
            return self.Session

    fake_manager = FakeManager()
    engine, session = _initialize_db_manager_engine(fake_manager)
    assert engine == "engine"
    assert session == "session"
    return True


def _test_initialize_ethnicity_columns_from_metadata_imports_module() -> bool:
    sentinel = object()
    fake_module = SimpleNamespace(initialize_ethnicity_columns_from_metadata=lambda _mgr: sentinel)
    with mock.patch.dict(sys.modules, {"genealogy.dna.dna_ethnicity_utils": fake_module}):
        session_manager = SimpleNamespace()
        result = _initialize_ethnicity_columns_from_metadata(cast(SessionManager, session_manager))
    assert result is sentinel
    return True


def _test_close_main_pool_for_reset_invokes_gc() -> bool:
    fake_manager = SimpleNamespace(db_manager=mock.Mock())
    with mock.patch("gc.collect") as gc_collect, mock.patch("time.sleep") as sleep:
        manager_for_call = cast(SessionManager, fake_manager)
        _close_main_pool_for_reset(manager_for_call)

    fake_manager.db_manager.close_connections.assert_called_once_with(dispose_engine=True)
    assert gc_collect.call_count >= 2
    sleep.assert_called_once_with(1.0)
    return True


def _test_handle_logged_in_status_outputs_message() -> bool:
    session_manager = SimpleNamespace()
    with (
        mock.patch(f"{__name__}._display_session_info") as display_info,
        mock.patch(f"{__name__}.logger") as mock_logger,
    ):
        result = _handle_logged_in_status(cast(SessionManager, session_manager))

    # Verify logger.info was called with login message
    calls = [str(call) for call in mock_logger.info.call_args_list]
    assert any("logged in" in call.lower() for call in calls), f"Expected 'logged in' in {calls}"
    display_info.assert_called_once_with(session_manager)
    assert result is True
    return True


def _test_verify_login_success_reports_status() -> bool:
    session_manager = SimpleNamespace()
    with (
        mock.patch(f"{__name__}.login_status", return_value=True),
        mock.patch(f"{__name__}._display_session_info") as display_info,
    ):
        assert _verify_login_success(cast(SessionManager, session_manager)) is True
        display_info.assert_called_once_with(session_manager)

    with mock.patch(f"{__name__}.login_status", return_value=False):
        output = _capture_stdout(_verify_login_success, cast(SessionManager, session_manager))
        assert "verification failed" in output
        assert _verify_login_success(cast(SessionManager, session_manager)) is False
    return True


def _test_attempt_login_success_and_failure_paths() -> bool:
    session_manager = SimpleNamespace()
    with (
        mock.patch(f"{__name__}.log_in", return_value=True),
        mock.patch(f"{__name__}._verify_login_success", return_value=True) as verify_login,
    ):
        assert _attempt_login(cast(SessionManager, session_manager)) is True
        verify_login.assert_called_once_with(session_manager)

    with mock.patch(f"{__name__}.log_in", return_value=False):
        assert _attempt_login(cast(SessionManager, session_manager)) is False

    with mock.patch(f"{__name__}.log_in", side_effect=RuntimeError("boom")):
        assert _attempt_login(cast(SessionManager, session_manager)) is False
    return True


def module_tests() -> bool:
    suite = TestSuite("core.maintenance_actions", "core/maintenance_actions.py")

    suite.run_test(
        "Delete table records",
        _test_delete_table_records_uses_session_query,
        "Ensures delete helper issues a query and returns count.",
    )

    suite.run_test(
        "Aggregate deletion counts",
        _test_perform_deletions_aggregates_counts,
        "Ensures perform deletions aggregates each table count.",
    )

    suite.run_test(
        "Initialize DB manager engine",
        _test_initialize_db_manager_engine_invokes_initializer,
        "Ensures DB manager initializer runs when engine missing.",
    )

    suite.run_test(
        "Initialize ethnicity columns",
        _test_initialize_ethnicity_columns_from_metadata_imports_module,
        "Ensures ethnicity initializer delegates to helper module.",
    )

    suite.run_test(
        "Close pool for reset",
        _test_close_main_pool_for_reset_invokes_gc,
        "Ensures reset helper closes pool and frees resources.",
    )

    suite.run_test(
        "Handle logged in status",
        _test_handle_logged_in_status_outputs_message,
        "Ensures logged-in handler prints message and session info.",
    )

    suite.run_test(
        "Verify login success",
        _test_verify_login_success_reports_status,
        "Ensures login verification reports success and failure states.",
    )

    suite.run_test(
        "Attempt login paths",
        _test_attempt_login_success_and_failure_paths,
        "Ensures login attempts handle success, failure, and exceptions.",
    )

    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


def _should_run_module_tests() -> bool:
    return os.environ.get("RUN_MODULE_TESTS") == "1"


def _print_module_usage() -> int:
    print("core.maintenance_actions exposes helpers for main.py and has no standalone CLI entry.")
    print("Set RUN_MODULE_TESTS=1 before execution to run embedded regression tests.")
    return 0


if __name__ == "__main__":
    if _should_run_module_tests():
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    sys.exit(_print_module_usage())
