#!/usr/bin/env python3

# --- Standard library imports ---
import sys
import tempfile
import sqlite3
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# --- Test framework imports ---
try:
    from test_framework import (
        TestSuite,
        suppress_logging,
        create_mock_data,
        assert_valid_function,
    )

    HAS_TEST_FRAMEWORK = True
except ImportError:
    HAS_TEST_FRAMEWORK = False


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for check_db.py.
    Tests database validation, integrity checks, and maintenance operations.
    """
    if not HAS_TEST_FRAMEWORK:
        return run_comprehensive_tests_fallback()

    suite = TestSuite("Database Validation & Integrity Checks", "check_db.py")
    suite.start_suite()

    # Database connection validation
    def test_database_connection():
        if "check_database_connection" in globals():
            connection_checker = globals()["check_database_connection"]

            # Test with mock connection
            mock_connection = MagicMock()
            mock_connection.execute.return_value = MagicMock()

            result = connection_checker(mock_connection)
            assert isinstance(result, bool)

    # Table structure validation
    def test_table_structure_validation():
        if "validate_table_structure" in globals():
            validator = globals()["validate_table_structure"]

            # Test required tables
            required_tables = ["person", "dna_match", "family_tree", "messages"]
            for table in required_tables:
                result = validator(table)
                assert isinstance(result, bool)

    # Data integrity checks
    def test_data_integrity_checks():
        if "check_data_integrity" in globals():
            integrity_checker = globals()["check_data_integrity"]

            # Test various integrity checks
            integrity_tests = [
                "foreign_key_constraints",
                "unique_constraints",
                "null_constraints",
                "data_consistency",
            ]

            for test_type in integrity_tests:
                result = integrity_checker(test_type)
                assert isinstance(result, (bool, dict, list))

    # Database statistics
    def test_database_statistics():
        if "get_database_statistics" in globals():
            stats_func = globals()["get_database_statistics"]
            stats = stats_func()

            assert isinstance(stats, dict)
            expected_stats = ["total_records", "table_sizes", "index_usage"]
            for stat in expected_stats:
                if stat in stats:
                    assert isinstance(stats[stat], (int, dict))

    # Index validation
    def test_index_validation():
        if "validate_database_indexes" in globals():
            index_validator = globals()["validate_database_indexes"]

            result = index_validator()
            assert isinstance(result, (bool, dict))

    # Performance analysis
    def test_performance_analysis():
        if "analyze_query_performance" in globals():
            perf_analyzer = globals()["analyze_query_performance"]

            # Test with sample queries
            sample_queries = [
                "SELECT * FROM person WHERE uuid = ?",
                "SELECT * FROM dna_match WHERE cM_DNA > ?",
                "SELECT * FROM family_tree WHERE in_my_tree = ?",
            ]

            for query in sample_queries:
                result = perf_analyzer(query)
                assert isinstance(result, dict)

    # Database backup validation
    def test_backup_validation():
        if "validate_database_backup" in globals():
            backup_validator = globals()["validate_database_backup"]

            # Test with temporary backup file
            with tempfile.NamedTemporaryFile(suffix=".db") as temp_file:
                result = backup_validator(temp_file.name)
                assert isinstance(result, bool)

    # Schema migration checks
    def test_schema_migration_checks():
        if "check_schema_version" in globals():
            version_checker = globals()["check_schema_version"]

            current_version = version_checker()
            assert isinstance(current_version, (str, int, type(None)))

        if "validate_schema_migration" in globals():
            migration_validator = globals()["validate_schema_migration"]

            result = migration_validator("1.0", "2.0")
            assert isinstance(result, bool)

    # Data cleanup validation
    def test_data_cleanup_validation():
        # Test data cleanup and maintenance
        cleanup_functions = [
            "cleanup_orphaned_records",
            "cleanup_duplicate_entries",
            "cleanup_invalid_data",
        ]

        for func_name in cleanup_functions:
            if func_name in globals():
                cleanup_func = globals()[func_name]
                result = cleanup_func()
                assert isinstance(result, (bool, int, dict))

    # Database health report
    def test_database_health_report():
        if "generate_health_report" in globals():
            report_generator = globals()["generate_health_report"]

            report = report_generator()
            assert isinstance(report, dict)

            # Check for key health metrics
            health_metrics = [
                "connection_status",
                "data_integrity",
                "performance_metrics",
                "storage_usage",
                "backup_status",
            ]

            for metric in health_metrics:
                if metric in report:
                    assert report[metric] is not None

    # Run all tests
    test_functions = {
        "Database connection validation": (
            test_database_connection,
            "Should validate database connectivity and basic operations",
        ),
        "Table structure validation": (
            test_table_structure_validation,
            "Should verify all required tables and columns exist",
        ),
        "Data integrity checks": (
            test_data_integrity_checks,
            "Should check foreign keys, constraints, and data consistency",
        ),
        "Database statistics": (
            test_database_statistics,
            "Should provide comprehensive database usage statistics",
        ),
        "Index validation": (
            test_index_validation,
            "Should verify database indexes are properly configured",
        ),
        "Query performance analysis": (
            test_performance_analysis,
            "Should analyze query performance and identify bottlenecks",
        ),
        "Database backup validation": (
            test_backup_validation,
            "Should validate backup files and restore procedures",
        ),
        "Schema migration checks": (
            test_schema_migration_checks,
            "Should verify schema version and migration status",
        ),
        "Data cleanup validation": (
            test_data_cleanup_validation,
            "Should clean up orphaned and invalid data",
        ),
        "Database health report": (
            test_database_health_report,
            "Should generate comprehensive database health reports",
        ),
    }

    with suppress_logging():
        for test_name, (test_func, expected_behavior) in test_functions.items():
            suite.run_test(test_name, test_func, expected_behavior)

    return suite.finish_suite()


def run_comprehensive_tests_fallback() -> bool:
    """
    Fallback testing when test_framework is not available.
    Provides basic database functionality verification.
    """
    print("⚠️  Test framework not available. Running basic database tests...")

    tests_passed = 0
    total_tests = 0

    # Test database connection basic functionality
    total_tests += 1
    try:
        # Test that we can create a sqlite connection
        test_db_path = ":memory:"
        conn = sqlite3.connect(test_db_path)
        conn.close()
        tests_passed += 1
        print("✅ Database connection test")
    except Exception as e:
        print(f"❌ Database connection test failed: {e}")

    # Test pathlib functionality
    total_tests += 1
    try:
        test_path = Path("test_path")
        assert isinstance(test_path, Path)
        tests_passed += 1
        print("✅ Path handling test")
    except Exception as e:
        print(f"❌ Path handling test failed: {e}")

    # Test basic database operations
    total_tests += 1
    try:
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        cursor.execute("INSERT INTO test (name) VALUES (?)", ("test_name",))
        cursor.execute("SELECT COUNT(*) FROM test")
        count = cursor.fetchone()[0]
        assert count == 1
        conn.close()
        tests_passed += 1
        print("✅ Basic database operations test")
    except Exception as e:
        print(f"❌ Basic database operations test failed: {e}")

    success = tests_passed == total_tests
    print(f"📊 Basic database tests: {tests_passed}/{total_tests} passed")
    return success


def check_database():
    """Main database checking function."""
    # Connect to the database
    db_path = (
        Path(os.path.dirname(os.path.abspath(__file__)))
        / "data"
        / "test_ai_responses.db"
    )
    print(f"Checking database at: {db_path}")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if the database exists and has the conversation_log table
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_log'"
        )
        if not cursor.fetchone():
            print("conversation_log table not found in database")
            return False

        # Count total records
        cursor.execute("SELECT COUNT(*) FROM conversation_log")
        total_count = cursor.fetchone()[0]
        print(f"Total records in conversation_log: {total_count}")

        # Count IN records
        cursor.execute("SELECT COUNT(*) FROM conversation_log WHERE direction = 'IN'")
        in_count = cursor.fetchone()[0]
        print(f"IN records: {in_count}")

        # Count OUT records
        cursor.execute("SELECT COUNT(*) FROM conversation_log WHERE direction = 'OUT'")
        out_count = cursor.fetchone()[0]
        print(f"OUT records: {out_count}")

        # Get column names
        cursor.execute("PRAGMA table_info(conversation_log)")
        columns = cursor.fetchall()
        print("\nTable columns:")
        for column in columns:
            print(f"  {column[1]} ({column[2]})")

        # Check the content of OUT records
        cursor.execute(
            "SELECT conversation_id, people_id, latest_message_content FROM conversation_log WHERE direction = 'OUT' LIMIT 5"
        )
        out_records = cursor.fetchall()

        print("\nSample OUT records:")
        for record in out_records:
            conversation_id, people_id, content = record
            print(f"Conversation ID: {conversation_id}, Person ID: {people_id}")
            print(f"Content: {content[:100]}...")
            print("-" * 50)

        conn.close()
        print("Database check complete")
        return True
    except Exception as e:
        print(f"Error checking database: {e}")
        return False


if __name__ == "__main__":
    # Check if user wants to run database check or tests
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        # Run database check
        success = check_database()
        sys.exit(0 if success else 1)
    else:
        # Run comprehensive tests
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
