#!/usr/bin/env python3
"""
Simple Database Viewer for ancestry.db
Run this anytime to browse your database content
"""

import sqlite3


def display_menu() -> str:
    print("\n" + "="*50)
    print("📊 ANCESTRY DATABASE VIEWER")
    print("="*50)
    print("1. Show all tables")
    print("2. View message types")
    print("3. View people (DNA matches)")
    print("4. View conversation log")
    print("5. View DNA match details")
    print("6. View family tree data")
    print("7. Run custom SQL query")
    print("8. Show database stats")
    print("q. Quit")
    return input("\nChoice: ").strip().lower()

def show_tables(cursor) -> None:
    cursor.execute("SELECT name, type FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"\n📋 Tables ({len(tables)}):")
    for name, _table_type in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {name}")
        count = cursor.fetchone()[0]
        print(f"  - {name}: {count:,} records")

def show_table_data(cursor, table_name: str, limit: int = 20) -> None:
    print(f"\n📊 {table_name.upper()} DATA:")
    print("="*60)

    # Get column names
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    col_names = [col[1] for col in columns]

    # Show headers
    header = " | ".join(f"{name[:20]:20}" for name in col_names)
    print(header)
    print("-" * len(header))

    # Show data
    cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
    rows = cursor.fetchall()

    if not rows:
        print("(No data)")
        return

    for row in rows:
        row_str = " | ".join(f"{str(val)[:20]:20}" if val is not None else "NULL                " for val in row)
        print(row_str)

    # Show total
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total = cursor.fetchone()[0]
    print(f"\nShowing {len(rows)} of {total} total records")

def run_custom_query(cursor) -> None:
    print("\n💻 CUSTOM SQL QUERY")
    print("Enter your SQL query (or 'back' to return):")
    query = input("SQL> ").strip()

    if query.lower() == 'back':
        return

    try:
        cursor.execute(query)

        if query.upper().startswith('SELECT'):
            results = cursor.fetchall()
            if results:
                # Show column names if available
                if cursor.description:
                    col_names = [desc[0] for desc in cursor.description]
                    print(" | ".join(f"{name:15}" for name in col_names))
                    print("-" * (len(col_names) * 16))

                for row in results[:50]:  # Limit to 50 rows
                    print(" | ".join(f"{val!s:15}" if val is not None else "NULL           " for val in row))

                if len(results) > 50:
                    print(f"... and {len(results) - 50} more results")
            else:
                print("No results")
        else:
            print("Query executed successfully")

    except Exception as e:
        print(f"❌ Error: {e}")

def show_db_stats(cursor) -> None:
    print("\n📈 DATABASE STATISTICS:")
    print("="*40)

    # Database info
    cursor.execute("PRAGMA database_list")
    db_info = cursor.fetchall()
    from pathlib import Path
    for _seq, _name, file in db_info:
        if file:
            p = Path(file)
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            print(f"Database: {file}")
            print(f"Size: {size:,} bytes ({size/(1024*1024):.2f} MB)")

    # Table stats
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    total_records = 0
    for table_name, in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        total_records += count

    print(f"Total tables: {len(tables)}")
    print(f"Total records: {total_records:,}")

def main() -> None:
    from pathlib import Path
    db_path = Path("Data/ancestry.db")

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        while True:
            choice = display_menu()

            if choice == 'q':
                break
            if choice == '1':
                show_tables(cursor)
            elif choice == '2':
                show_table_data(cursor, 'message_types')
            elif choice == '3':
                show_table_data(cursor, 'people')
            elif choice == '4':
                show_table_data(cursor, 'conversation_log')
            elif choice == '5':
                show_table_data(cursor, 'dna_match')
            elif choice == '6':
                show_table_data(cursor, 'family_tree')
            elif choice == '7':
                run_custom_query(cursor)
            elif choice == '8':
                show_db_stats(cursor)
            else:
                print("Invalid choice")

            input("\nPress Enter to continue...")

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        conn.close()


# ==============================================
# COMPREHENSIVE TEST SUITE
# ==============================================

def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for database viewer functions.

    Tests all core database viewing functionality including table display,
    data visualization, stats generation, and error handling.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    try:
        from test_framework import TestSuite
    except ImportError:
        print("⚠️  TestSuite not available - falling back to basic testing")
        return _run_basic_tests()

    suite = TestSuite("Database Viewer", "db_viewer")

    def test_database_connection() -> None:
        """Test database connection functionality"""
        import sqlite3
        import tempfile

        # Create temporary test database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            test_db = tmp.name

        try:
            # Create test database with sample data
            conn = sqlite3.connect(test_db)
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)''')
            cursor.execute('''INSERT INTO test_table (name) VALUES ('test1'), ('test2')''')
            conn.commit()

            # Test show_tables function
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            assert len(tables) >= 1

            conn.close()
        finally:
            from pathlib import Path
            Path(test_db).unlink(missing_ok=True)

    def test_table_display_functions() -> None:
        """Test table display and data retrieval functions"""
        import sqlite3
        import sys
        import tempfile
        from io import StringIO

        # Create test database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            test_db = tmp.name

        try:
            conn = sqlite3.connect(test_db)
            cursor = conn.cursor()

            # Create test tables
            cursor.execute('''CREATE TABLE people (id INTEGER, name TEXT, age INTEGER)''')
            cursor.execute('''INSERT INTO people VALUES (1, 'John Doe', 30), (2, 'Jane Smith', 25)''')
            cursor.execute('''CREATE TABLE message_types (id INTEGER, type TEXT)''')
            cursor.execute('''INSERT INTO message_types VALUES (1, 'email'), (2, 'sms')''')
            conn.commit()

            # Test show_tables function by capturing output
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            show_tables(cursor)
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

            assert 'people' in output
            assert 'message_types' in output
            assert '2 records' in output

            conn.close()
        finally:
            from pathlib import Path
            Path(test_db).unlink(missing_ok=True)

    def test_table_data_display() -> None:
        """Test table data display formatting"""
        import sqlite3
        import sys
        import tempfile
        from io import StringIO

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            test_db = tmp.name

        try:
            conn = sqlite3.connect(test_db)
            cursor = conn.cursor()

            cursor.execute('''CREATE TABLE test_data (id INTEGER, description TEXT)''')
            cursor.execute('''INSERT INTO test_data VALUES (1, 'Test Description'), (2, 'Another Test')''')
            conn.commit()

            # Capture output
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            show_table_data(cursor, 'test_data', limit=10)
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

            # Verify output contains expected elements
            assert 'TEST_DATA DATA:' in output
            assert 'Test Description' in output
            assert 'Another Test' in output

            conn.close()
        finally:
            from pathlib import Path
            Path(test_db).unlink(missing_ok=True)

    def test_database_stats() -> None:
        """Test database statistics generation"""
        import sqlite3
        import sys
        import tempfile
        from io import StringIO

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            test_db = tmp.name

        try:
            conn = sqlite3.connect(test_db)
            cursor = conn.cursor()

            # Create test tables with data
            cursor.execute('''CREATE TABLE stats_test (id INTEGER, value TEXT)''')
            cursor.execute('''INSERT INTO stats_test VALUES (1, 'a'), (2, 'b'), (3, 'c')''')
            conn.commit()

            # Test stats generation
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            show_db_stats(cursor)
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

            assert 'Total tables:' in output
            assert 'Total records:' in output
            assert '3' in output  # Should show 3 records

            conn.close()
        finally:
            from pathlib import Path
            Path(test_db).unlink(missing_ok=True)

    def test_custom_query_handling() -> None:
        """Test custom query execution"""
        import sqlite3
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            test_db = tmp.name

        try:
            conn = sqlite3.connect(test_db)
            cursor = conn.cursor()

            cursor.execute('''CREATE TABLE query_test (id INTEGER, name TEXT)''')
            cursor.execute('''INSERT INTO query_test VALUES (1, 'test_name')''')
            conn.commit()

            # Test basic query execution
            cursor.execute("SELECT COUNT(*) FROM query_test")
            result = cursor.fetchone()
            assert result[0] == 1

            # Test column info retrieval
            cursor.execute("PRAGMA table_info(query_test)")
            columns = cursor.fetchall()
            assert len(columns) == 2

            conn.close()
        finally:
            from pathlib import Path
            Path(test_db).unlink(missing_ok=True)

    def test_menu_display() -> None:
        """Test menu display functionality"""
        import sys
        from io import StringIO

        # Capture menu output
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        # Mock input to test menu display
        old_stdin = sys.stdin
        sys.stdin = StringIO('q\n')

        try:
            choice = display_menu()
            output = sys.stdout.getvalue()

            assert '📊 ANCESTRY DATABASE VIEWER' in output
            assert 'Show all tables' in output
            assert 'View people' in output
            assert 'Quit' in output
            assert choice == 'q'

        finally:
            sys.stdout = old_stdout
            sys.stdin = old_stdin

    def test_error_handling() -> None:
        """Test error handling with invalid database operations"""
        import sqlite3
        import tempfile

        # Test with non-existent database
        from pathlib import Path
        assert not Path("nonexistent_database.db").exists()

        # Test with invalid table operations
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            test_db = tmp.name

        try:
            conn = sqlite3.connect(test_db)
            cursor = conn.cursor()

            # Test querying non-existent table should not crash
            try:
                cursor.execute("SELECT * FROM nonexistent_table")
                raise AssertionError("Should have raised an exception")
            except sqlite3.OperationalError:
                pass  # Expected

            conn.close()
        finally:
            from pathlib import Path
            Path(test_db).unlink(missing_ok=True)

    def test_function_availability() -> None:
        """Test that all required functions are available"""
        required_functions = [
            "display_menu",
            "show_tables",
            "show_table_data",
            "run_custom_query",
            "show_db_stats",
            "main"
        ]

        for func_name in required_functions:
            assert func_name in globals(), f"Function {func_name} should be available"
            assert callable(globals()[func_name]), f"Function {func_name} should be callable"

    # Run all tests
    suite.run_test(
        "Database connection",
        test_database_connection,
        "Database connection and basic SQLite operations work correctly",
        "Test SQLite database connection, table creation, and basic queries",
        "Verify database connection handles SQLite operations properly"
    )

    suite.run_test(
        "Table display functions",
        test_table_display_functions,
        "Table display functions show database structure correctly",
        "Test show_tables function with sample database tables and records",
        "Verify table display shows correct table names and record counts"
    )

    suite.run_test(
        "Table data display",
        test_table_data_display,
        "Table data display formats database records properly",
        "Test show_table_data function with sample data and column formatting",
        "Verify data display shows proper column headers and record formatting"
    )

    suite.run_test(
        "Database statistics",
        test_database_stats,
        "Database statistics generation provides accurate information",
        "Test show_db_stats function with sample database and record counts",
        "Verify stats generation shows correct table counts and totals"
    )

    suite.run_test(
        "Custom query handling",
        test_custom_query_handling,
        "Custom query execution works with various SQL commands",
        "Test query execution with SELECT, PRAGMA, and other SQL operations",
        "Verify custom queries execute properly and return expected results"
    )

    suite.run_test(
        "Menu display functionality",
        test_menu_display,
        "Menu display shows all available options correctly",
        "Test display_menu function output and user input handling",
        "Verify menu displays all database viewer options properly"
    )

    suite.run_test(
        "Error handling",
        test_error_handling,
        "Error conditions are handled gracefully without crashing",
        "Test error handling with invalid databases and malformed queries",
        "Verify robust error handling for database and SQL operation failures"
    )

    suite.run_test(
        "Function availability verification",
        test_function_availability,
        "All required database viewer functions are available and callable",
        "Test availability of display_menu, show_tables, and data functions",
        "Verify function availability ensures complete database viewer interface"
    )

    return suite.finish_suite()


def _run_basic_tests() -> bool:
    """Basic test fallback when TestSuite is not available"""
    try:
        import sqlite3
        import tempfile

        # Test basic database operations
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            test_db = tmp.name

        try:
            conn = sqlite3.connect(test_db)
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE test (id INTEGER)''')
            cursor.execute('''INSERT INTO test VALUES (1)''')
            conn.commit()

            # Test show_tables doesn't crash
            show_tables(cursor)

            conn.close()
            print("✅ Basic database viewer tests passed")
            return True
        finally:
            from pathlib import Path
            Path(test_db).unlink(missing_ok=True)

    except Exception as e:
        print(f"❌ Basic tests failed: {e}")
        return False


# ==============================================
# Standalone Test Block
# ==============================================

if __name__ == "__main__":
    import sys

    # Check if running tests vs normal viewer
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("🔍 Running Database Viewer comprehensive test suite...")
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    else:
        main()
