#!/usr/bin/env python3
"""Debug config validation issue."""

from config.config_schema import DatabaseConfig, validate_positive_integer

print("Testing validate_positive_integer with 10:", validate_positive_integer(10))
print("Testing DatabaseConfig creation...")
try:
    db = DatabaseConfig()
    print("DatabaseConfig created successfully")
    print("pool_size value:", db.pool_size)
    print("pool_size type:", type(db.pool_size))
except Exception as e:
    print("Error:", e)
