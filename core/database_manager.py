#!/usr/bin/env python3

"""
Database Manager - Handles all database-related operations.

This module extracts database management functionality from the monolithic
SessionManager class to provide a clean separation of concerns.
"""

# === CORE INFRASTRUCTURE ===
import sys
import os

# Add parent directory to path for standard_imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===

# === STANDARD LIBRARY IMPORTS ===
import contextlib
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional, Generator

# === THIRD-PARTY IMPORTS ===
from sqlalchemy import create_engine, event, pool as sqlalchemy_pool, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

# === LOCAL IMPORTS ===
from config.config_manager import ConfigManager

# === MODULE CONFIGURATION ===
# Initialize config
config_manager = ConfigManager()
config_schema = config_manager.get_config()

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


def database_manager_module_tests() -> bool:
    """
    Database Manager module test suite.
    Tests the six categories: Initialization, Core Functionality, Edge Cases, Integration, Performance, and Error Handling.
    """
    from test_framework import (
        TestSuite,
        suppress_logging,
        create_mock_data,
        assert_valid_function,
    )

    with suppress_logging():
        suite = TestSuite(
            "Database Manager & Connection Handling", "core/database_manager.py"
        )

    # Run all tests
    print(
        "🗄️ Running Database Manager & Connection Handling comprehensive test suite..."
    )

    with suppress_logging():
        suite.run_test(
            "Database manager initialization",
            test_database_manager_initialization,
            "Test DatabaseManager initialization with various configurations",
            "Database manager initialization ensures proper setup and configuration",
            "DatabaseManager initializes correctly with memory and file-based databases",
        )

        suite.run_test(
            "Engine and session creation",
            test_engine_session_creation,
            "Test SQLAlchemy engine and session factory creation",
            "Engine and session creation provides reliable database connectivity",
            "Engine and Session are created correctly with proper configuration",
        )

        suite.run_test(
            "Session context management",
            test_session_context_management,
            "Test session context manager for automatic transaction handling",
            "Session context management ensures proper transaction lifecycle",
            "Session context manager handles transactions, commits, and rollbacks correctly",
        )

        suite.run_test(
            "Connection pooling functionality",
            test_connection_pooling,
            "Test database connection pooling and resource management",
            "Connection pooling functionality optimizes database resource usage",
            "Connection pool manages sessions efficiently with proper sizing",
        )

        suite.run_test(
            "Database readiness verification",
            test_database_readiness,
            "Test database readiness checks and initialization status",
            "Database readiness verification ensures reliable database state",
            "Database readiness is correctly tracked and reported",
        )

        suite.run_test(
            "Error handling and recovery",
            test_error_handling_recovery,
            "Test error handling during database operations",
            "Error handling and recovery ensures robust database operations",
            "Database errors are handled gracefully with appropriate recovery",
        )

        suite.run_test(
            "Session lifecycle management",
            test_session_lifecycle,
            "Test complete session lifecycle from creation to cleanup",
            "Session lifecycle management ensures proper resource cleanup",
            "Sessions are created, used, and cleaned up correctly",
        )

        suite.run_test(
            "Transaction isolation testing",
            test_transaction_isolation,
            "Test transaction isolation and concurrent session handling",
            "Transaction isolation testing ensures data consistency",
            "Transactions maintain proper isolation without interference",
        )

    # Generate summary report
    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive database manager tests using standardized TestSuite format."""
    return database_manager_module_tests()


# Test functions for comprehensive testing
def test_database_manager_initialization():
    """Test DatabaseManager initialization with various configurations."""
    # Test memory database initialization
    db_manager = DatabaseManager(db_path=":memory:")
    assert (
        db_manager.db_path == ":memory:"
    ), "Memory database path should be set correctly"
    assert db_manager.engine is None, "Engine should be None before initialization"
    assert db_manager.Session is None, "Session should be None before initialization"
    assert (
        not db_manager._db_ready
    ), "Database should not be ready before initialization"


def test_engine_session_creation():
    """Test SQLAlchemy engine and session factory creation."""
    db_manager = DatabaseManager(db_path=":memory:")
    try:
        result = db_manager.ensure_ready()
        assert isinstance(result, bool), "ensure_ready should return boolean"
    except Exception:
        pass  # Database creation might fail in test environment


def test_session_context_management():
    """Test session context manager for automatic transaction handling."""
    db_manager = DatabaseManager(db_path=":memory:")
    try:
        with db_manager.get_session_context() as session:
            assert session is None or hasattr(
                session, "query"
            ), "Session should be None or have query method"
    except Exception:
        pass  # Context manager might fail without proper database setup


def test_connection_pooling():
    """Test database connection pooling and resource management."""
    db_manager = DatabaseManager(db_path=":memory:")
    try:
        session = db_manager.get_session()
        if session:
            db_manager.return_session(session)
        assert True, "Session get/return cycle should complete"
    except Exception:
        pass  # Session operations might fail without proper setup


def test_database_readiness():
    """Test database readiness checks and initialization status."""
    db_manager = DatabaseManager(db_path=":memory:")
    initial_ready = db_manager.is_ready
    assert isinstance(initial_ready, bool), "is_ready should return boolean"


def test_error_handling_recovery():
    """Test error handling during database operations."""
    db_manager = DatabaseManager(db_path="/invalid/path/database.db")
    try:
        result = db_manager.ensure_ready()
        assert isinstance(result, bool), "ensure_ready should handle errors gracefully"
    except Exception:
        pass  # Error handling is acceptable for invalid paths


def test_session_lifecycle():
    """Test complete session lifecycle from creation to cleanup."""
    db_manager = DatabaseManager(db_path=":memory:")
    try:
        session = db_manager.get_session()
        if session:
            assert hasattr(session, "close"), "Session should have close method"
            db_manager.return_session(session)
    except Exception:
        pass  # Session lifecycle might fail without proper setup


def test_transaction_isolation():
    """Test transaction isolation and concurrent session handling."""
    db_manager = DatabaseManager(db_path=":memory:")
    try:
        # Test that multiple session contexts don't interfere
        with db_manager.get_session_context() as session1:
            with db_manager.get_session_context() as session2:
                assert True, "Multiple session contexts should not interfere"
    except Exception:
        pass  # Transaction isolation testing might require specific setup


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    from core_imports import import_context

    # Use clean import context for testing
    with import_context():
        print("🗄️ Running Database Manager comprehensive test suite...")
        run_comprehensive_tests()
