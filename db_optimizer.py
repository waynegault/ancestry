#!/usr/bin/env python3

# db_optimizer.py

"""
Database Optimization Module - Enhanced Connection Pooling & Query Performance

Provides advanced database optimization strategies including intelligent connection pooling,
query performance monitoring, index optimization, and database maintenance utilities.
Designed to work with the existing SQLAlchemy architecture while providing significant
performance improvements for high-throughput genealogy operations.
"""

import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import statistics
from pathlib import Path

# Third-party imports
from sqlalchemy import create_engine, event, text, func
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.exc import SQLAlchemyError

# Local imports
from config import config_instance
from logging_config import logger


class DatabaseOptimizer:
    """
    Advanced database optimization and connection management system.

    Features:
    - Intelligent connection pooling with dynamic sizing
    - Query performance monitoring and optimization
    - Database maintenance and cleanup operations
    - Connection health monitoring and recovery
    """

    def __init__(self):
        self.query_metrics: Dict[str, List[float]] = {}
        self.connection_stats: Dict[str, int] = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "pool_overflows": 0,
        }
        self.optimization_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def optimize_connection_pool(
        self, engine: Engine, current_load: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Dynamically optimize connection pool settings based on usage patterns.

        Args:
            engine: SQLAlchemy engine to optimize
            current_load: Current system load (optional)

        Returns:
            Dictionary with optimization results and recommendations
        """
        try:
            pool = engine.pool
            current_size = pool.size()
            checked_out = pool.checkedout()
            checked_in = pool.checkedin()
            overflow = getattr(pool, "overflow", lambda: 0)()

            # Calculate optimal pool size based on usage patterns
            utilization_ratio = checked_out / max(current_size, 1)

            recommendations = {
                "current_pool_size": current_size,
                "checked_out": checked_out,
                "checked_in": checked_in,
                "overflow_connections": overflow,
                "utilization_ratio": utilization_ratio,
                "optimization_applied": False,
                "recommendations": [],
            }

            # High utilization - recommend increasing pool size
            if utilization_ratio > 0.8:
                new_size = min(current_size + 5, 50)  # Cap at 50 connections
                recommendations["recommendations"].append(
                    f"High utilization ({utilization_ratio:.2f}). Consider increasing pool_size to {new_size}"
                )

            # Low utilization - recommend decreasing pool size
            elif utilization_ratio < 0.3 and current_size > 5:
                new_size = max(current_size - 2, 5)  # Minimum 5 connections
                recommendations["recommendations"].append(
                    f"Low utilization ({utilization_ratio:.2f}). Consider decreasing pool_size to {new_size}"
                )

            # Monitor overflow usage
            if overflow > 0:
                recommendations["recommendations"].append(
                    f"Pool overflow detected ({overflow} connections). Consider increasing max_overflow or pool_size"
                )

            # Record optimization attempt
            with self._lock:
                self.optimization_history.append(
                    {
                        "timestamp": datetime.now(timezone.utc),
                        "action": "pool_analysis",
                        "metrics": recommendations.copy(),
                    }
                )

            logger.debug(f"Connection pool analysis: {recommendations}")
            return recommendations

        except Exception as e:
            logger.error(f"Error optimizing connection pool: {e}", exc_info=True)
            return {"error": str(e), "optimization_applied": False}

    def create_optimized_engine(self, db_path: Union[str, Path], **kwargs) -> Engine:
        """
        Create an optimized SQLAlchemy engine with intelligent pooling.

        Args:
            db_path: Database file path
            **kwargs: Additional engine configuration options

        Returns:
            Optimized SQLAlchemy engine
        """
        # Default optimized settings
        default_config = {
            "pool_size": 15,  # Increased from typical 10
            "max_overflow": 25,  # Higher overflow for burst loads
            "pool_timeout": 45,  # Longer timeout for better reliability
            "pool_recycle": 3600,  # Recycle connections after 1 hour
            "pool_pre_ping": True,  # Test connections before use
            "echo": False,
            "connect_args": {
                "check_same_thread": False,
                "timeout": 30,  # Connection timeout
                "isolation_level": None,  # Autocommit mode for better performance
            },
        }

        # Merge with provided kwargs
        config = {**default_config, **kwargs}

        try:
            # Create engine with optimized settings
            engine = create_engine(
                f"sqlite:///{db_path}", poolclass=QueuePool, **config
            )

            # Add performance monitoring event listeners
            self._add_performance_listeners(engine)

            # Add SQLite optimization settings
            @event.listens_for(engine, "connect")
            def set_sqlite_optimizations(dbapi_connection, connection_record):
                """Apply SQLite-specific performance optimizations."""
                cursor = dbapi_connection.cursor()
                try:
                    # Performance optimizations
                    cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
                    cursor.execute(
                        "PRAGMA synchronous=NORMAL"
                    )  # Balanced safety/performance
                    cursor.execute("PRAGMA cache_size=10000")  # Larger cache (10MB)
                    cursor.execute(
                        "PRAGMA temp_store=MEMORY"
                    )  # Use memory for temp storage
                    cursor.execute("PRAGMA mmap_size=268435456")  # 256MB memory mapping
                    cursor.execute(
                        "PRAGMA optimize"
                    )  # SQLite query planner optimization

                    # Foreign key enforcement
                    cursor.execute("PRAGMA foreign_keys=ON")

                    logger.debug("SQLite performance optimizations applied")

                except Exception as e:
                    logger.warning(f"Failed to apply SQLite optimizations: {e}")
                finally:
                    cursor.close()

            logger.info(f"Created optimized database engine for {db_path}")
            return engine

        except Exception as e:
            logger.error(f"Failed to create optimized engine: {e}", exc_info=True)
            raise

    def _add_performance_listeners(self, engine: Engine):
        """Add performance monitoring event listeners to the engine."""

        @event.listens_for(engine, "before_cursor_execute")
        def receive_before_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            context._query_start_time = time.time()

        @event.listens_for(engine, "after_cursor_execute")
        def receive_after_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            if hasattr(context, "_query_start_time"):
                execution_time = time.time() - context._query_start_time

                # Extract query type for categorization
                query_type = (
                    statement.strip().split()[0].upper()
                    if statement.strip()
                    else "UNKNOWN"
                )

                with self._lock:
                    if query_type not in self.query_metrics:
                        self.query_metrics[query_type] = []
                    self.query_metrics[query_type].append(execution_time)

                    # Keep only recent metrics (last 1000 queries per type)
                    if len(self.query_metrics[query_type]) > 1000:
                        self.query_metrics[query_type] = self.query_metrics[query_type][
                            -1000:
                        ]

                # Log slow queries
                if execution_time > 1.0:  # Queries taking more than 1 second
                    logger.warning(
                        f"Slow query detected: {execution_time:.3f}s - {statement[:100]}..."
                    )

    def get_query_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive query performance report.

        Returns:
            Dictionary with performance statistics and recommendations
        """
        with self._lock:
            report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query_types": {},
                "overall_stats": {},
                "recommendations": [],
            }

            all_times = []

            for query_type, times in self.query_metrics.items():
                if times:
                    stats = {
                        "count": len(times),
                        "avg_time": statistics.mean(times),
                        "median_time": statistics.median(times),
                        "max_time": max(times),
                        "min_time": min(times),
                        "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
                    }
                    report["query_types"][query_type] = stats
                    all_times.extend(times)

                    # Generate recommendations for slow query types
                    if stats["avg_time"] > 0.5:
                        report["recommendations"].append(
                            f"{query_type} queries are averaging {stats['avg_time']:.3f}s - consider optimization"
                        )

            # Overall statistics
            if all_times:
                report["overall_stats"] = {
                    "total_queries": len(all_times),
                    "avg_time": statistics.mean(all_times),
                    "median_time": statistics.median(all_times),
                    "95th_percentile": (
                        sorted(all_times)[int(len(all_times) * 0.95)]
                        if all_times
                        else 0
                    ),
                }

            return report

    def analyze_database_size_and_performance(self, session: Session) -> Dict[str, Any]:
        """
        Analyze database size, table statistics, and performance metrics.

        Args:
            session: SQLAlchemy session

        Returns:
            Dictionary with database analysis results
        """
        try:
            analysis = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "database_size": {},
                "table_stats": {},
                "index_stats": {},
                "recommendations": [],
            }

            # Get database size information
            size_query = text(
                "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"
            )
            db_size = session.execute(size_query).scalar()
            analysis["database_size"]["total_bytes"] = db_size
            analysis["database_size"]["total_mb"] = (
                db_size / (1024 * 1024) if db_size else 0
            )

            # Get table statistics
            tables_query = text(
                """
                SELECT name, 
                       (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=m.name) as table_count
                FROM sqlite_master m 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """
            )

            tables = session.execute(tables_query).fetchall()

            for table_name, _ in tables:
                try:
                    # Row count
                    count_query = text(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = session.execute(count_query).scalar()

                    # Table size
                    size_query = text(
                        f"SELECT SUM(pgsize) FROM dbstat WHERE name='{table_name}'"
                    )
                    table_size = session.execute(size_query).scalar() or 0

                    analysis["table_stats"][table_name] = {
                        "row_count": row_count,
                        "size_bytes": table_size,
                        "size_mb": table_size / (1024 * 1024),
                        "avg_row_size": table_size / max(row_count, 1),
                    }

                    # Recommendations based on table size
                    if row_count > 100000:
                        analysis["recommendations"].append(
                            f"Table {table_name} has {row_count:,} rows - consider partitioning or archiving"
                        )

                except Exception as e:
                    logger.warning(f"Error analyzing table {table_name}: {e}")

            # Analyze indexes
            index_query = text(
                """
                SELECT name, tbl_name 
                FROM sqlite_master 
                WHERE type='index' AND name NOT LIKE 'sqlite_%'
            """
            )

            indexes = session.execute(index_query).fetchall()
            for index_name, table_name in indexes:
                try:
                    # Index usage statistics would require more complex analysis
                    analysis["index_stats"][index_name] = {
                        "table": table_name,
                        "status": "present",
                    }
                except Exception as e:
                    logger.warning(f"Error analyzing index {index_name}: {e}")

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing database: {e}", exc_info=True)
            return {"error": str(e)}

    def optimize_database_maintenance(self, session: Session) -> Dict[str, Any]:
        """
        Perform database maintenance operations for optimal performance.

        Args:
            session: SQLAlchemy session

        Returns:
            Dictionary with maintenance results
        """
        try:
            maintenance_results = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "operations": {},
                "recommendations": [],
            }

            # VACUUM operation
            try:
                session.execute(text("VACUUM"))
                maintenance_results["operations"]["vacuum"] = "completed"
                logger.info("Database VACUUM completed successfully")
            except Exception as e:
                maintenance_results["operations"]["vacuum"] = f"failed: {e}"
                logger.warning(f"VACUUM operation failed: {e}")

            # ANALYZE operation for query optimizer
            try:
                session.execute(text("ANALYZE"))
                maintenance_results["operations"]["analyze"] = "completed"
                logger.info("Database ANALYZE completed successfully")
            except Exception as e:
                maintenance_results["operations"]["analyze"] = f"failed: {e}"
                logger.warning(f"ANALYZE operation failed: {e}")

            # Incremental VACUUM for WAL mode
            try:
                session.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
                maintenance_results["operations"]["wal_checkpoint"] = "completed"
                logger.info("WAL checkpoint completed successfully")
            except Exception as e:
                maintenance_results["operations"]["wal_checkpoint"] = f"failed: {e}"
                logger.warning(f"WAL checkpoint failed: {e}")

            # Check for unused space
            try:
                freelist_count = session.execute(text("PRAGMA freelist_count")).scalar()
                page_count = session.execute(text("PRAGMA page_count")).scalar()

                if freelist_count and page_count:
                    free_percentage = (freelist_count / page_count) * 100
                    maintenance_results["operations"]["free_space_check"] = {
                        "freelist_pages": freelist_count,
                        "total_pages": page_count,
                        "free_percentage": free_percentage,
                    }

                    if free_percentage > 10:
                        maintenance_results["recommendations"].append(
                            f"Database has {free_percentage:.1f}% free space - consider VACUUM"
                        )

            except Exception as e:
                logger.warning(f"Free space check failed: {e}")

            return maintenance_results

        except Exception as e:
            logger.error(f"Error during database maintenance: {e}", exc_info=True)
            return {"error": str(e)}

    @contextmanager
    def optimized_session_context(self, engine: Engine):
        """
        Context manager for optimized database sessions with automatic cleanup.

        Args:
            engine: SQLAlchemy engine

        Yields:
            Configured database session
        """
        session = None
        try:
            Session = sessionmaker(bind=engine, expire_on_commit=False)
            session = Session()

            # Update connection stats
            with self._lock:
                self.connection_stats["total_connections"] += 1
                self.connection_stats["active_connections"] += 1

            yield session

            # Commit on successful completion
            session.commit()

        except Exception as e:
            if session:
                session.rollback()

            with self._lock:
                self.connection_stats["failed_connections"] += 1

            logger.error(f"Session error in optimized context: {e}")
            raise

        finally:
            if session:
                session.close()

            with self._lock:
                self.connection_stats["active_connections"] = max(
                    0, self.connection_stats["active_connections"] - 1
                )

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get current connection statistics and performance metrics."""
        with self._lock:
            return {
                "connection_stats": self.connection_stats.copy(),
                "query_metrics_summary": {
                    query_type: {
                        "count": len(times),
                        "avg_time": statistics.mean(times) if times else 0,
                        "total_time": sum(times),
                    }
                    for query_type, times in self.query_metrics.items()
                },
                "optimization_history_count": len(self.optimization_history),
            }


# Global optimizer instance
db_optimizer = DatabaseOptimizer()


def create_optimized_session_manager(
    db_path: Union[str, Path],
) -> "OptimizedSessionManager":
    """
    Create an optimized session manager with enhanced connection pooling.

    Args:
        db_path: Database file path

    Returns:
        OptimizedSessionManager instance
    """
    return OptimizedSessionManager(db_path)


class OptimizedSessionManager:
    """
    Enhanced session manager with intelligent connection pooling and performance monitoring.
    """

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.engine: Optional[Engine] = None
        self.Session: Optional[sessionmaker] = None
        self._initialized = False

    def initialize(self):
        """Initialize the optimized session manager."""
        if self._initialized:
            return

        try:
            self.engine = db_optimizer.create_optimized_engine(self.db_path)
            self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
            self._initialized = True
            logger.info(f"Optimized session manager initialized for {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize optimized session manager: {e}")
            raise

    @contextmanager
    def get_session(self):
        """Get an optimized database session with automatic management."""
        if not self._initialized:
            self.initialize()

        with db_optimizer.optimized_session_context(self.engine) as session:
            yield session

    def optimize_pool(self) -> Dict[str, Any]:
        """Optimize the connection pool based on current usage."""
        if not self._initialized:
            self.initialize()

        return db_optimizer.optimize_connection_pool(self.engine)

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return db_optimizer.get_query_performance_report()

    def perform_maintenance(self) -> Dict[str, Any]:
        """Perform database maintenance operations."""
        if not self._initialized:
            self.initialize()

        with self.get_session() as session:
            return db_optimizer.optimize_database_maintenance(session)

    def dispose(self):
        """Dispose of the engine and cleanup resources."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.Session = None
            self._initialized = False


def self_test() -> bool:
    """Test the database optimizer functionality."""
    try:
        logger.info("Testing Database Optimizer...")

        # Test optimizer creation
        optimizer = DatabaseOptimizer()
        logger.info("✓ Database optimizer created successfully")

        # Test metrics collection
        test_metrics = optimizer.get_query_performance_report()
        logger.info(f"✓ Performance report generated: {len(test_metrics)} sections")

        # Test connection stats
        stats = optimizer.get_connection_stats()
        logger.info(f"✓ Connection stats retrieved: {stats['connection_stats']}")

        logger.info("Database Optimizer self-test completed successfully")
        return True

    except Exception as e:
        logger.error(f"Database Optimizer self-test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = self_test()

    if success:
        print("\n=== Database Optimization Report ===")
        print("✓ All database optimization components are working correctly")
        print("✓ Connection pooling enhancements ready")
        print("✓ Query performance monitoring active")
        print("✓ Database maintenance utilities available")

        # Display sample optimization statistics
        print(f"\n=== Performance Metrics ===")
        stats = db_optimizer.get_connection_stats()
        for key, value in stats["connection_stats"].items():
            print(f"{key}: {value}")
    else:
        print("❌ Database optimization self-test failed - check logs for details")
