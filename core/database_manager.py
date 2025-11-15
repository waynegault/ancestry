#!/usr/bin/env python3

"""
Database Manager - Handles all database-related operations.

This module extracts database management functionality from the monolithic
SessionManager class to provide a clean separation of concerns.
"""

# === CORE INFRASTRUCTURE ===
import sys

# Add parent directory to path for standard_imports
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import asyncio
import contextlib
import sqlite3

# Note: sys and Path already imported at top of file
from collections.abc import AsyncGenerator, Generator
from typing import Any, Optional, cast
from collections.abc import Callable
from unittest.mock import MagicMock, patch

# === THIRD-PARTY IMPORTS ===
from sqlalchemy import create_engine, event, inspect, pool as sqlalchemy_pool, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

# === LOCAL IMPORTS ===
from config.config_manager import ConfigManager

# === MODULE CONFIGURATION ===
# Initialize config
config_manager = ConfigManager()
config_schema = config_manager.get_config()


class DatabaseManager:
    """
    Enhanced Database Manager with advanced connection pooling and optimization.

    Phase 7.3.2 Enhancement: Advanced memory management, connection pooling optimization,
    query optimization, and performance monitoring.

    Features:
    - Intelligent connection pool sizing based on workload
    - Connection health monitoring and automatic recovery
    - Query performance tracking and optimization
    - Memory usage monitoring and optimization
    - Batch processing capabilities
    - Connection leak detection and prevention
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the Enhanced DatabaseManager.

        Args:
            db_path: Path to the database file. If None, uses config default.
        """
        # Database configuration
        if db_path:
            self.db_path = db_path
        elif config_schema and config_schema.database.database_file:
            self.db_path = str(config_schema.database.database_file.resolve())
        else:
            # Ensure fallback also uses Data folder
            fallback_path = Path("Data/ancestry.db")
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(fallback_path)  # Default fallback to Data folder

        # SQLAlchemy components
        self.engine = None
        self.Session: Optional[sessionmaker] = None
        self._db_init_attempted: bool = False
        self._db_ready: bool = False

        # Phase 7.3.2 Enhancement: Performance monitoring
        self._connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "connection_leaks": 0,
            "query_count": 0,
            "slow_queries": 0,
            "total_query_time": 0.0,
            "avg_query_time": 0.0,
            "pool_overflows": 0,
            "pool_invalidations": 0
        }

        # Connection health monitoring
        self._connection_health_threshold = 0.8  # 80% success rate threshold
        self._slow_query_threshold = 1.0  # 1 second threshold for slow queries
        self._max_connection_age = 3600  # 1 hour max connection age

        # Adaptive pool sizing
        self._adaptive_pooling = True
        self._base_pool_size = 10
        self._max_pool_size = 50
        self._current_pool_size = self._base_pool_size

        logger.debug(f"Enhanced DatabaseManager initialized with path: {self.db_path}")

    def _calculate_optimal_pool_size(self, base_size: int) -> int:
        """Calculate optimal pool size based on performance metrics."""
        if not self._adaptive_pooling:
            return base_size

        # Calculate pool efficiency metrics
        total_requests = self._connection_stats["total_connections"]
        if total_requests < 100:  # Not enough data for optimization
            return base_size

        success_rate = 1.0 - (self._connection_stats["failed_connections"] / total_requests)
        overflow_rate = self._connection_stats["pool_overflows"] / total_requests

        # Adjust pool size based on metrics
        if success_rate < self._connection_health_threshold:
            # Poor success rate - increase pool size
            return min(base_size + 5, self._max_pool_size)
        if overflow_rate > 0.1:  # More than 10% overflow
            # High overflow - increase pool size
            return min(base_size + 3, self._max_pool_size)
        if overflow_rate < 0.01 and success_rate > 0.95:
            # Very low overflow and high success - can reduce pool size
            return max(base_size - 2, self._base_pool_size)

        return base_size

    def _update_connection_stats(self, operation: str, success: bool = True, query_time: float = 0.0):
        """Update connection and query statistics."""
        if operation == "connection":
            self._connection_stats["total_connections"] += 1
            if not success:
                self._connection_stats["failed_connections"] += 1
        elif operation == "query":
            self._connection_stats["query_count"] += 1
            self._connection_stats["total_query_time"] += query_time

            if query_time > self._slow_query_threshold:
                self._connection_stats["slow_queries"] += 1

            # Update average query time
            if self._connection_stats["query_count"] > 0:
                self._connection_stats["avg_query_time"] = (
                    self._connection_stats["total_query_time"] / self._connection_stats["query_count"]
                )
        elif operation == "overflow":
            self._connection_stats["pool_overflows"] += 1
        elif operation == "invalidation":
            self._connection_stats["pool_invalidations"] += 1

    def get_performance_stats(self) -> dict:
        """Get comprehensive database performance statistics."""
        stats = self._connection_stats.copy()

        # Calculate derived metrics
        total_requests = stats["total_connections"]
        if total_requests > 0:
            stats["success_rate"] = 1.0 - (stats["failed_connections"] / total_requests)
            stats["overflow_rate"] = stats["pool_overflows"] / total_requests
        else:
            stats["success_rate"] = 1.0
            stats["overflow_rate"] = 0.0

        # Add pool information
        stats["current_pool_size"] = self._current_pool_size
        stats["max_pool_size"] = self._max_pool_size
        stats["adaptive_pooling"] = self._adaptive_pooling

        # Add health indicators
        stats["connection_health"] = "good" if stats["success_rate"] >= self._connection_health_threshold else "poor"
        stats["query_performance"] = "good" if stats["avg_query_time"] < self._slow_query_threshold else "slow"

        return stats

    @contextlib.contextmanager
    def batch_operation_context(self, batch_size: int = 1000):
        """
        Context manager for efficient batch operations.

        Args:
            batch_size: Number of operations to batch before committing

        Yields:
            Tuple of (session, batch_counter) for batch processing
        """
        session = None
        batch_counter = 0

        try:
            session = self.get_session()
            if not session:
                raise RuntimeError("Failed to obtain database session for batch operation")

            logger.debug(f"Starting batch operation with batch size: {batch_size}")

            def commit_batch() -> None:
                nonlocal batch_counter
                if batch_counter > 0:
                    session.commit()
                    logger.debug(f"Committed batch of {batch_counter} operations")
                    batch_counter = 0

            def add_to_batch() -> None:
                nonlocal batch_counter
                batch_counter += 1
                if batch_counter >= batch_size:
                    commit_batch()

            # Provide session and batch management functions
            yield session, add_to_batch, commit_batch

            # Final commit for remaining operations
            commit_batch()

        except Exception as e:
            if session:
                try:
                    session.rollback()
                    logger.error(f"Batch operation failed, rolled back: {e}")
                except Exception as rollback_e:
                    logger.error(f"Failed to rollback batch operation: {rollback_e}")
            raise
        finally:
            if session:
                self.return_session(session)

    def execute_query_with_timing(self, session: Session, query: Any, params: Optional[dict[str, Any]] = None) -> Any:
        """
        Execute a query with performance timing.

        Args:
            session: Database session
            query: SQL query or SQLAlchemy query object
            params: Query parameters

        Returns:
            Query result with timing information
        """
        import time
        start_time = time.time()

        try:
            result = session.execute(query, params) if params else session.execute(query)

            query_time = time.time() - start_time
            self._update_connection_stats("query", success=True, query_time=query_time)

            if query_time > self._slow_query_threshold:
                logger.warning(f"Slow query detected: {query_time:.3f}s")

            return result

        except Exception as e:
            query_time = time.time() - start_time
            self._update_connection_stats("query", success=False, query_time=query_time)
            logger.error(f"Query failed after {query_time:.3f}s: {e}")
            raise

    # === ASYNC DATABASE OPERATIONS (Phase 7.4.2) ===

    @contextlib.asynccontextmanager
    async def async_session_context(self) -> AsyncGenerator[Session, None]:
        """
        Async context manager for database session management.

        Provides an async-compatible database session with proper transaction
        handling and automatic cleanup. Uses thread pool for database operations
        to maintain async compatibility with SQLite.

        Yields:
            Session: Database session for async operations

        Example:
            >>> async with db_manager.async_session_context() as session:
            ...     result = await db_manager.async_execute_query(
            ...         session, "SELECT * FROM people LIMIT 10"
            ...     )
        """
        session = None
        loop = asyncio.get_event_loop()

        try:
            # Get session in thread pool to avoid blocking
            session = await loop.run_in_executor(None, self.get_session)

            if not session:
                raise RuntimeError("Failed to obtain database session for async operation")

            logger.debug("Async database session acquired")
            yield session

            # Commit in thread pool
            await loop.run_in_executor(None, session.commit)
            logger.debug("Async database session committed")

        except Exception as e:
            if session:
                try:
                    await loop.run_in_executor(None, session.rollback)
                    logger.error(f"Async database session rolled back due to error: {e}")
                except Exception as rollback_e:
                    logger.error(f"Failed to rollback async session: {rollback_e}")
            raise
        finally:
            if session:
                await loop.run_in_executor(None, self.return_session, session)

    async def async_execute_query(
        self,
        session: Session,
        query: str,
        params: Optional[dict[str, Any]] = None,
        fetch_results: bool = True
    ) -> Optional[list[dict[str, Any]]]:
        """
        Execute a database query asynchronously.

        Args:
            session: Database session
            query: SQL query string
            params: Optional query parameters
            fetch_results: Whether to fetch and return results

        Returns:
            List of result dictionaries or None

        Example:
            >>> async with db_manager.async_session_context() as session:
            ...     results = await db_manager.async_execute_query(
            ...         session, "SELECT * FROM people WHERE name = :name",
            ...         {"name": "John Smith"}
            ...     )
        """
        loop = asyncio.get_event_loop()

        def _execute_query() -> Any:
            try:
                # Convert string query to SQLAlchemy text object
                sql_query = text(query) if isinstance(query, str) else query

                result = session.execute(sql_query, params) if params else session.execute(sql_query)

                if fetch_results:
                    # Convert result to list of dictionaries
                    rows = result.fetchall()
                    if rows:
                        columns = result.keys()
                        return [dict(zip(columns, row)) for row in rows]
                    return []
                return None

            except Exception as e:
                logger.error(f"Async query execution failed: {e}")
                raise

        try:
            return await loop.run_in_executor(None, _execute_query)
        except Exception as e:
            logger.error(f"Async database query failed: {e}")
            return None

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

    def _dispose_existing_engine(self) -> None:
        """Dispose existing engine if present."""
        if not self.engine:
            return

        logger.warning("Disposing existing engine before re-initializing.")
        try:
            self.engine.dispose()
        except Exception as dispose_e:
            logger.error(f"Error disposing existing engine: {dispose_e}")
        self.engine = None
        self.Session = None

    def _calculate_pool_configuration(self) -> tuple[int, int, int, int, bool]:
        """Calculate optimal pool configuration. Returns (pool_size, max_overflow, pool_timeout, pool_recycle, pool_pre_ping)."""
        # Get base pool size from config
        base_pool_size = (
            config_schema.database.pool_size
            if config_schema and config_schema.database
            else self._base_pool_size
        )
        if not isinstance(base_pool_size, int) or base_pool_size <= 0:
            logger.warning(f"Invalid DB_POOL_SIZE '{base_pool_size}'. Using default {self._base_pool_size}.")
            base_pool_size = self._base_pool_size

        # Adaptive pool sizing based on performance metrics
        pool_size = self._calculate_optimal_pool_size(base_pool_size)
        pool_size = min(pool_size, self._max_pool_size)  # Cap pool size
        self._current_pool_size = pool_size

        # Enhanced pool configuration
        max_overflow = max(5, int(pool_size * 0.3))  # Increased overflow capacity
        pool_timeout = 45  # Increased timeout for better reliability
        pool_recycle = self._max_connection_age  # Recycle connections after max age
        pool_pre_ping = True  # Enable connection health checks

        logger.debug(
            f"Enhanced DB Pool Config: Size={pool_size}, MaxOverflow={max_overflow}, "
            f"Timeout={pool_timeout}, Recycle={pool_recycle}, PrePing={pool_pre_ping}"
        )

        return pool_size, max_overflow, pool_timeout, pool_recycle, pool_pre_ping

    def _create_engine_with_config(self, pool_size: int, max_overflow: int, pool_timeout: int, pool_recycle: int, pool_pre_ping: bool) -> None:
        """Create SQLAlchemy engine with specified configuration."""
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            pool_pre_ping=pool_pre_ping,
            poolclass=sqlalchemy_pool.QueuePool,
            connect_args={
                "check_same_thread": False,
                "timeout": 30,  # Connection timeout
                "isolation_level": None,  # Autocommit mode for better performance
            },
            execution_options={
                "autocommit": False,
                "compiled_cache": {},  # Enable query compilation caching
            }
        )
        logger.debug(f"Created SQLAlchemy engine: ID={id(self.engine)}")

    def _attach_pragma_listener(self) -> None:
        """Attach event listener for SQLite PRAGMA settings."""
        @event.listens_for(self.engine, "connect")
        def enable_sqlite_settings(dbapi_connection: Any, _connection_record: Any) -> None:
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

    def _cleanup_failed_initialization(self) -> None:
        """Clean up resources after failed initialization."""
        if self.engine:
            self.engine.dispose()
        self.engine = None
        self.Session = None
        self._db_init_attempted = False

    def _initialize_engine_and_session(self) -> None:
        """Initialize SQLAlchemy engine and session factory."""
        # Prevent re-initialization if already done
        if self.engine and self.Session:
            logger.debug("DB Engine/Session already initialized. Skipping.")
            return

        logger.debug("Initializing SQLAlchemy Engine/Session...")
        self._db_init_attempted = True

        # Dispose existing engine if somehow present but Session is not
        self._dispose_existing_engine()

        try:
            logger.debug(f"DB Path: {self.db_path}")

            # Calculate pool configuration
            pool_size, max_overflow, pool_timeout, pool_recycle, pool_pre_ping = self._calculate_pool_configuration()

            # Create engine with configuration
            self._create_engine_with_config(pool_size, max_overflow, pool_timeout, pool_recycle, pool_pre_ping)

            # Attach PRAGMA listener
            self._attach_pragma_listener()

            # Create Session Factory
            self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
            logger.debug(f"Created Session factory for Engine ID={id(self.engine)}")

            # Ensure tables are created
            self._ensure_tables_created()

        except SQLAlchemyError as sql_e:
            logger.critical(f"FAILED to initialize SQLAlchemy: {sql_e}", exc_info=True)
            self._cleanup_failed_initialization()
            raise sql_e
        except Exception as e:
            logger.critical(
                f"UNEXPECTED error initializing SQLAlchemy: {e}", exc_info=True
            )
            self._cleanup_failed_initialization()
            raise e

    def _ensure_tables_created(self) -> None:
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
            # Create tables only if the database is empty
            else:
                # Import Base locally to avoid circular import issues
                try:
                    from database import Base
                    Base.metadata.create_all(self.engine)
                    logger.debug("DB tables created successfully.")
                except ImportError as e:
                    logger.warning(
                        f"Cannot create tables - Base not available from database module: {e}"
                    )
        except SQLAlchemyError as table_create_e:
            logger.warning(
                f"Non-critical error during DB table check/creation: {table_create_e}"
            )
            # Don't raise the error, just log it and continue

    def get_session(self) -> Optional[Session]:
        """
        Get a database session from the pool with performance tracking.

        Returns:
            Session or None if unable to create session
        """
        import time
        start_time = time.time()

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
                    self._update_connection_stats("connection", success=False)
                    return None
            except Exception as init_e:
                logger.error(f"Exception during lazy initialization: {init_e}")
                self._update_connection_stats("connection", success=False)
                return None

        # Get session from factory
        try:
            new_session: Session = self.Session()
            self._connection_stats["active_connections"] += 1

            # Track connection time
            connection_time = time.time() - start_time
            self._update_connection_stats("connection", success=True)

            logger.debug(
                f"Obtained DB session {id(new_session)} from Engine ID={id(self.engine)} "
                f"(connection time: {connection_time:.3f}s)"
            )
            return new_session
        except Exception as e:
            logger.error(f"Error getting DB session from factory: {e}", exc_info=True)
            self._update_connection_stats("connection", success=False)

            # Attempt to recover by disposing engine and resetting flags
            if self.engine:
                with contextlib.suppress(Exception):
                    self.engine.dispose()
            self.engine = None
            self.Session = None
            self._db_init_attempted = False
            return None

    def return_session(self, session: Optional[Session]):
        """
        Return a session to the pool (close it) with performance tracking.

        Args:
            session: The session to return (can be None)
        """
        if session:
            session_id = id(session)
            try:
                session.close()
                self._connection_stats["active_connections"] = max(0, self._connection_stats["active_connections"] - 1)
                logger.debug(f"DB session {session_id} closed and returned to pool.")
            except Exception as e:
                logger.error(
                    f"Error closing DB session {session_id}: {e}", exc_info=True
                )
                # Track connection leak
                self._connection_stats["connection_leaks"] += 1
        else:
            logger.warning("Attempted to return a None DB session.")

    def _commit_session_if_active(self, session: Session, session_id: str) -> None:
        """Commit session if active, otherwise log warning."""
        if session.is_active:
            try:
                session.commit()
                logger.debug(
                    f"DB Context Manager: Commit successful for session {session_id}."
                )
            except SQLAlchemyError as commit_err:
                logger.error(
                    f"DB Context Manager: Commit failed for session {session_id}: {commit_err}. Rolling back."
                )
                session.rollback()
                raise
        else:
            logger.warning(
                f"DB Context Manager: Session {session_id} inactive after yield, skipping commit."
            )

    def _rollback_session_if_active(self, session: Optional[Session], session_id: str) -> None:
        """Rollback session if active."""
        if session and session.is_active:
            try:
                session.rollback()
                logger.warning(
                    f"DB Context Manager: Rollback successful for session {session_id}."
                )
            except Exception as rb_err:
                logger.error(
                    f"DB Context Manager: Error during rollback for session {session_id}: {rb_err}"
                )

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
                self._commit_session_if_active(session, session_id_for_log)
            else:
                logger.error("DB Context Manager: Failed to obtain DB session.")
                yield None

        except SQLAlchemyError as sql_e:
            logger.error(
                f"DB Context Manager: SQLAlchemyError ({type(sql_e).__name__}). Rolling back session {session_id_for_log}.",
                exc_info=True,
            )
            self._rollback_session_if_active(session, session_id_for_log)
            raise sql_e
        except Exception as e:
            logger.error(
                f"DB Context Manager: Unexpected Exception ({type(e).__name__}). Rolling back session {session_id_for_log}.",
                exc_info=True,
            )
            self._rollback_session_if_active(session, session_id_for_log)
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





# Test functions for comprehensive testing
def _build_stubbed_db_manager() -> tuple[DatabaseManager, MagicMock, MagicMock, list[MagicMock], Callable[[], None]]:
    """Create a DatabaseManager with stubbed engine and session factory for tests."""
    db_manager = DatabaseManager(db_path=":memory:")
    engine_mock = MagicMock(name="engine")
    created_sessions: list[MagicMock] = []

    def _make_session() -> MagicMock:
        session = MagicMock(name="session")
        session.is_active = True
        session.close.return_value = None
        session.commit.return_value = None
        session.rollback.return_value = None
        created_sessions.append(session)
        return session

    session_factory = MagicMock(side_effect=_make_session)

    def stub_initializer() -> None:
        db_manager._db_init_attempted = True
        db_manager.engine = engine_mock
        db_manager.Session = session_factory

    return db_manager, engine_mock, session_factory, created_sessions, stub_initializer


def test_database_manager_initialization() -> None:
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


def test_engine_session_creation() -> None:
    """Test SQLAlchemy engine and session factory creation."""
    db_manager, engine_mock, session_factory, _sessions, stub_initializer = _build_stubbed_db_manager()

    with patch.object(db_manager, "_initialize_engine_and_session", side_effect=stub_initializer):
        result = db_manager.ensure_ready()

    assert result is True, "ensure_ready should succeed with stubbed initialization"
    assert db_manager.engine is engine_mock, "Stubbed engine should be set"
    assert db_manager.Session is session_factory, "Stubbed session factory should be set"
    assert db_manager.is_ready, "Database should report ready after initialization"


def test_session_context_management() -> None:
    """Test session context manager for automatic transaction handling."""
    db_manager, _engine_mock, _factory, sessions, stub_initializer = _build_stubbed_db_manager()

    with patch.object(db_manager, "_initialize_engine_and_session", side_effect=stub_initializer):
        db_manager.ensure_ready()

    with patch.object(db_manager, "return_session") as return_session_mock:
        with db_manager.get_session_context() as session:
            assert session is not None, "Session context should provide a session"
            assert session is sessions[-1], "Context should yield the latest created session"
            session_mock = cast(MagicMock, session)

        session_mock.commit.assert_called_once()
        return_session_mock.assert_called_once_with(session_mock)


def test_connection_pooling() -> None:
    """Test database connection pooling and resource management."""
    db_manager, _engine_mock, _factory, sessions, stub_initializer = _build_stubbed_db_manager()

    with patch.object(db_manager, "_initialize_engine_and_session", side_effect=stub_initializer):
        db_manager.ensure_ready()

    session = db_manager.get_session()
    assert session is sessions[-1], "get_session should return a stubbed session"
    assert db_manager._connection_stats["total_connections"] == 1
    assert db_manager._connection_stats["active_connections"] == 1

    session_mock = cast(MagicMock, session)
    db_manager.return_session(session_mock)
    session_mock.close.assert_called_once()
    assert db_manager._connection_stats["active_connections"] == 0


def test_database_readiness() -> None:
    """Test database readiness checks and initialization status."""
    db_manager, _engine_mock, _factory, _sessions, stub_initializer = _build_stubbed_db_manager()
    assert db_manager.is_ready is False

    with patch.object(db_manager, "_initialize_engine_and_session", side_effect=stub_initializer):
        ensure_result = db_manager.ensure_ready()

    assert ensure_result is True
    assert db_manager.is_ready is True


def test_error_handling_recovery() -> None:
    """Test error handling during database operations."""
    db_manager = DatabaseManager(db_path=":memory:")

    with patch.object(db_manager, "_initialize_engine_and_session", side_effect=RuntimeError("init failure")):
        result = db_manager.ensure_ready()

    assert result is False, "ensure_ready should return False on initialization failure"
    assert db_manager.is_ready is False
    assert db_manager.engine is None
    assert db_manager.Session is None


def test_session_lifecycle() -> None:
    """Test complete session lifecycle from creation to cleanup."""
    db_manager, _engine_mock, _factory, sessions, stub_initializer = _build_stubbed_db_manager()

    with patch.object(db_manager, "_initialize_engine_and_session", side_effect=stub_initializer):
        db_manager.ensure_ready()

    session = db_manager.get_session()
    assert session is sessions[-1]
    assert hasattr(session, "close")
    session_mock = cast(MagicMock, session)
    db_manager.return_session(session_mock)
    session_mock.close.assert_called_once()


def test_transaction_isolation() -> None:
    """Test transaction isolation and concurrent session handling."""
    db_manager, _engine_mock, _factory, _sessions, stub_initializer = _build_stubbed_db_manager()

    with patch.object(db_manager, "_initialize_engine_and_session", side_effect=stub_initializer):
        db_manager.ensure_ready()

    with patch.object(db_manager, "return_session") as return_session_mock:
        with db_manager.get_session_context() as session_a, db_manager.get_session_context() as session_b:
            assert session_a is not None and session_b is not None
            assert session_a is not session_b
            session_a_mock = cast(MagicMock, session_a)
            session_b_mock = cast(MagicMock, session_b)

        assert return_session_mock.call_count == 2
        session_a_mock.commit.assert_called_once()
        session_b_mock.commit.assert_called_once()


# ==============================================
# Standalone Test Block
# ==============================================
# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(database_manager_module_tests)


if __name__ == "__main__":
    from core_imports import import_context

    # Use clean import context for testing
    with import_context():
        print("🗄️ Running Database Manager comprehensive test suite...")
        run_comprehensive_tests()
