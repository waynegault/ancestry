import os
import pathlib
import sqlite3
import sys

db_path = "Data/ancestry.db"

if not pathlib.Path(db_path).exists():
    print(f"Database not found at {db_path}")
    sys.exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]

print("Tables found:", tables)

required_tables = ["conversation_state", "suggested_facts"]
missing_tables = [t for t in required_tables if t not in tables]

if missing_tables:
    print(f"Missing tables: {missing_tables}")
else:
    print("All required tables present.")

# Check columns in conversation_state
if "conversation_state" in tables:
    cursor.execute("PRAGMA table_info(conversation_state);")
    columns = [row[1] for row in cursor.fetchall()]
    print("Columns in conversation_state:", columns)

    # Check for specific columns added recently
    required_columns = ["status", "safety_flag", "last_intent"]
    missing_columns = [c for c in required_columns if c not in columns]
    if missing_columns:
        print(f"Missing columns in conversation_state: {missing_columns}")
    else:
        print("All required columns present in conversation_state.")

conn.close()
