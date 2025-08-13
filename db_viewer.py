#!/usr/bin/env python3
"""
Simple Database Viewer for ancestry.db
Run this anytime to browse your database content
"""

import sqlite3
import os

def display_menu():
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

def show_tables(cursor):
    cursor.execute("SELECT name, type FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"\n📋 Tables ({len(tables)}):")
    for name, table_type in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {name}")
        count = cursor.fetchone()[0]
        print(f"  - {name}: {count:,} records")

def show_table_data(cursor, table_name, limit=20):
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

def run_custom_query(cursor):
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
                    print(" | ".join(f"{str(val):15}" if val is not None else "NULL           " for val in row))
                
                if len(results) > 50:
                    print(f"... and {len(results) - 50} more results")
            else:
                print("No results")
        else:
            print("Query executed successfully")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def show_db_stats(cursor):
    print("\n📈 DATABASE STATISTICS:")
    print("="*40)
    
    # Database info
    cursor.execute("PRAGMA database_list")
    db_info = cursor.fetchall()
    for seq, name, file in db_info:
        if file:
            size = os.path.getsize(file)
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

def main():
    db_path = "Data/ancestry.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        while True:
            choice = display_menu()
            
            if choice == 'q':
                break
            elif choice == '1':
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

if __name__ == "__main__":
    main()
