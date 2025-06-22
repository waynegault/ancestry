"""
Database Manager - Handles all database-related operations.

This module extracts database management functionality from the monolithic
SessionManager class to provide a clean separation of concerns.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional, Generator
import contextlib
import sys
import os

from sqlalchemy import create_engine, event, pool as sqlalchemy_pool, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from config.config_manager import ConfigManager

# Initialize config
config_manager = ConfigManager()
config_schema = config_manager.get_config()

logger = logging.getLogger(__name__)

# Use centralized path management
from path_manager import ensure_imports

ensure_imports()

try:
    from database import Base

    HAS_DATABASE_BASE = True
except ImportError:
    HAS_DATABASE_BASE = False
    logger.warning("Could not import Base from database module")


class DatabaseManager:
    """
    Manages database connections, session pools, and database operations.

    This class handles all SQLAlchemy-related functionality including:
    - Engine creation and configuration
    - Connection pooling
    - Session management
    - Database initialization
    - Transaction management
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the DatabaseManager.

        Args:
            db_path: Path to the database file. If None, uses config default."""  # Database configuration
        if db_path:
            self.db_path = db_path
        else:
            if config_schema and config_schema.database.database_file:
                self.db_path = str(config_schema.database.database_file.resolve())
            else:
                self.db_path = "ancestry.db"  # Default fallback

        # SQLAlchemy components
        self.engine = None
        self.Session: Optional[sessionmaker] = None
        self._db_init_attempted: bool = False
        self._db_ready: bool = False

        logger.debug(f"DatabaseManager initialized with path: {self.db_path}")

    def ensure_ready(self) -> bool:
        """
        Ensures the database connection is ready.

        Returns:
            bool: True if database is ready, False otherwise
        """
        logger.debug("Ensuring database is ready...")

        if not self.engine or not self.Session:
            try:
                self._initialize_engine_and_session()
                self._db_ready = True
                logger.debug("Database initialized successfully.")
                return True
            except Exception as db_init_e:
                logger.critical(f"DB Initialization failed: {db_init_e}")
                self._db_ready = False
                return False
        else:
            self._db_ready = True
            logger.debug("Database already initialized.")
            return True

    def _initialize_engine_and_session(self):
        """Initialize SQLAlchemy engine and session factory."""
        # Prevent re-initialization if already done
        if self.engine and self.Session:
            logger.debug("DB Engine/Session already initialized. Skipping.")
            return

        logger.debug("Initializing SQLAlchemy Engine/Session...")
        self._db_init_attempted = True

        # Dispose existing engine if somehow present but Session is not
        if self.engine:
            logger.warning("Disposing existing engine before re-initializing.")
            try:
                self.engine.dispose()
            except Exception as dispose_e:
                logger.error(f"Error disposing existing engine: {dispose_e}")
            self.engine = None
            self.Session = None

        try:
            logger.debug(f"DB Path: {self.db_path}")  # Pool configuration
            pool_size = (
                config_schema.database.pool_size
                if config_schema and config_schema.database
                else 10
            )
            if not isinstance(pool_size, int) or pool_size <= 0:
                logger.warning(f"Invalid DB_POOL_SIZE '{pool_size}'. Using default 10.")
                pool_size = 10

            pool_size = min(pool_size, 100)  # Cap pool size
            max_overflow = max(5, int(pool_size * 0.2))
            pool_timeout = 30
            pool_class = sqlalchemy_pool.QueuePool

            logger.debug(
                f"DB Pool Config: Size={pool_size}, MaxOverflow={max_overflow}, Timeout={pool_timeout}"
            )

            # Create Engine
            self.engine = create_engine(
                f"sqlite:///{self.db_path}",
                echo=False,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                poolclass=pool_class,
                connect_args={"check_same_thread": False},
            )

            logger.debug(f"Created SQLAlchemy engine: ID={id(self.engine)}")

            # Attach event listener for PRAGMA settings
            @event.listens_for(self.engine, "connect")
            def enable_sqlite_settings(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                try:
                    cursor.execute("PRAGMA journal_mode=WAL;")
                    cursor.execute("PRAGMA foreign_keys=ON;")
                    cursor.execute("PRAGMA synchronous=NORMAL;")
                    logger.debug(
                        "SQLite PRAGMA settings applied (WAL, Foreign Keys, Sync Normal)."
                    )
                except sqlite3.Error as pragma_e:
                    logger.error(f"Failed setting PRAGMA: {pragma_e}")
                finally:
                    cursor.close()

            # Create Session Factory
            self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
            logger.debug(f"Created Session factory for Engine ID={id(self.engine)}")

            # Ensure tables are created
            self._ensure_tables_created()

        except SQLAlchemyError as sql_e:
            logger.critical(f"FAILED to initialize SQLAlchemy: {sql_e}", exc_info=True)
            if self.engine:
                self.engine.dispose()
            self.engine = None
            self.Session = None
            self._db_init_attempted = False
            raise sql_e
        except Exception as e:
            logger.critical(
                f"UNEXPECTED error initializing SQLAlchemy: {e}", exc_info=True
            )
            if self.engine:
                self.engine.dispose()
            self.engine = None
            self.Session = None
            self._db_init_attempted = False
            raise e

    def _ensure_tables_created(self):
        """Ensure database tables are created if needed."""
        if not self.engine:
            logger.error("Cannot check/create tables: Engine is None")
            return

        try:
            # Check if the database file exists and has tables
            inspector = inspect(self.engine)
            existing_tables = inspector.get_table_names()

            if existing_tables:
                logger.debug(f"Database already exists with tables: {existing_tables}")
            else:
                # Create tables only if the database is empty
                if HAS_DATABASE_BASE:
                    Base.metadata.create_all(self.engine)
                    logger.debug("DB tables created successfully.")
                else:
                    logger.warning(
                        "Cannot create tables - Base not available from database module"
                    )
        except SQLAlchemyError as table_create_e:
            logger.warning(
                f"Non-critical error during DB table check/creation: {table_create_e}"
            )
            # Don't raise the error, just log it and continue

    def get_session(self) -> Optional[Session]:
        """
        Get a database session from the pool.

        Returns:
            Session or None if unable to create session
        """
        engine_id_str = id(self.engine) if self.engine else "None"
        logger.debug(f"get_session called. Current Engine ID: {engine_id_str}")

        # Initialize DB if needed
        if not self._db_init_attempted or not self.engine or not self.Session:
            logger.debug(
                "Engine/Session factory not ready. Triggering initialization..."
            )
            try:
                self._initialize_engine_and_session()
                if not self.Session:
                    logger.error("Initialization failed, cannot get DB connection.")
                    return None
            except Exception as init_e:
                logger.error(f"Exception during lazy initialization: {init_e}")
                return None

        # Get session from factory
        try:
            new_session: Session = self.Session()
            logger.debug(
                f"Obtained DB session {id(new_session)} from Engine ID={id(self.engine)}"
            )
            return new_session
        except Exception as e:
            logger.error(f"Error getting DB session from factory: {e}", exc_info=True)
            # Attempt to recover by disposing engine and resetting flags
            if self.engine:
                try:
                    self.engine.dispose()
                except Exception:
                    pass
            self.engine = None
            self.Session = None
            self._db_init_attempted = False
            return None

    def return_session(self, session: Session):
        """
        Return a session to the pool (close it).

        Args:
            session: The session to return
        """
        if session:
            session_id = id(session)
            try:
                session.close()
                logger.debug(f"DB session {session_id} closed and returned to pool.")
            except Exception as e:
                logger.error(
                    f"Error closing DB session {session_id}: {e}", exc_info=True
                )
        else:
            logger.warning("Attempted to return a None DB session.")

    @contextlib.contextmanager
    def get_session_context(self) -> Generator[Optional[Session], None, None]:
        """
        Context manager for database sessions with automatic transaction handling.

        Yields:
            Session or None if session creation failed
        """
        session: Optional[Session] = None
        session_id_for_log = "N/A"

        try:
            session = self.get_session()
            if session:
                session_id_for_log = str(id(session))
                logger.debug(
                    f"DB Context Manager: Acquired session {session_id_for_log}."
                )
                yield session

                # After the 'with' block finishes:
                if session.is_active:
                    try:
                        session.commit()
                        logger.debug(
                            f"DB Context Manager: Commit successful for session {session_id_for_log}."
                        )
                    except SQLAlchemyError as commit_err:
                        logger.error(
                            f"DB Context Manager: Commit failed for session {session_id_for_log}: {commit_err}. Rolling back."
                        )
                        session.rollback()
                        raise
                else:
                    logger.warning(
                        f"DB Context Manager: Session {session_id_for_log} inactive after yield, skipping commit."
                    )
            else:
                logger.error("DB Context Manager: Failed to obtain DB session.")
                yield None

        except SQLAlchemyError as sql_e:
            logger.error(
                f"DB Context Manager: SQLAlchemyError ({type(sql_e).__name__}). Rolling back session {session_id_for_log}.",
                exc_info=True,
            )
            if session and session.is_active:
                try:
                    session.rollback()
                    logger.warning(
                        f"DB Context Manager: Rollback successful for session {session_id_for_log}."
                    )
                except Exception as rb_err:
                    logger.error(
                        f"DB Context Manager: Error during rollback for session {session_id_for_log}: {rb_err}"
                    )
            raise sql_e
        except Exception as e:
            logger.error(
                f"DB Context Manager: Unexpected Exception ({type(e).__name__}). Rolling back session {session_id_for_log}.",
                exc_info=True,
            )
            if session and session.is_active:
                try:
                    session.rollback()
                    logger.warning(
                        f"DB Context Manager: Rollback successful for session {session_id_for_log}."
                    )
                except Exception as rb_err:
                    logger.error(
                        f"DB Context Manager: Error during rollback for session {session_id_for_log}: {rb_err}"
                    )
            raise e
        finally:
            # Always return the session to the pool
            if session:
                self.return_session(session)
            else:
                logger.debug("DB Context Manager: No valid session to return.")

    def close_connections(self, dispose_engine: bool = False):
        """
        Close database connections.

        Args:
            dispose_engine: If True, disposes the engine completely
        """
        if dispose_engine and self.engine:
            engine_id = id(self.engine)
            logger.debug(f"Disposing Engine ID: {engine_id}")
            try:
                self.engine.dispose()
                logger.debug(f"Engine ID={engine_id} disposed successfully.")
            except Exception as e:
                logger.error(
                    f"Error disposing SQLAlchemy engine ID={engine_id}: {e}",
                    exc_info=True,
                )
            finally:
                self.engine = None
                self.Session = None
                self._db_init_attempted = False
        else:
            logger.debug("close_connections called without engine disposal.")

    @property
    def is_ready(self) -> bool:
        """Check if the database manager is ready."""
        return self._db_ready and self.engine is not None and self.Session is not None

    def get_db_path(self) -> str:
        """Get the database path."""
        return self.db_path


# ==============================================
# Test Suite Implementation
# ==============================================


import unittest
from unittest.mock import MagicMock, patch


class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        self.db_manager = DatabaseManager(db_path=":memory:")

    def test_initialization(self):
        self.assertEqual(self.db_manager.db_path, ":memory:")
        self.assertIsNone(self.db_manager.engine)
        self.assertIsNone(self.db_manager.Session)
        self.assertFalse(self.db_manager._db_ready)


def run_comprehensive_tests():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestDatabaseManager))
    runner = unittest.TextTestRunner()
    runner.run(suite)


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    from path_manager import import_context

    # Use clean import context for testing
    with import_context():
        print("🗄️ Running Database Manager comprehensive test suite...")
        run_comprehensive_tests()
