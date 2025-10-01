"""Test if credentials are being loaded correctly."""

import sys
sys.path.insert(0, '.')

from config import config_schema

print("=" * 70)
print("CREDENTIAL LOADING TEST")
print("=" * 70)

print("\n1. Checking config_schema.api...")
print(f"   config_schema exists: {config_schema is not None}")
print(f"   config_schema.api exists: {hasattr(config_schema, 'api')}")

if hasattr(config_schema, 'api'):
    print(f"\n2. Checking username...")
    username = config_schema.api.username
    print(f"   Username: {username}")
    print(f"   Username type: {type(username)}")
    print(f"   Username length: {len(username) if username else 0}")
    print(f"   Username is empty: {not username}")
    
    print(f"\n3. Checking password...")
    password = config_schema.api.password
    print(f"   Password: {'*' * len(password) if password else '(empty)'}")
    print(f"   Password type: {type(password)}")
    print(f"   Password length: {len(password) if password else 0}")
    print(f"   Password is empty: {not password}")
    
    print(f"\n4. Summary:")
    if username and password:
        print(f"   ✅ Both credentials loaded successfully!")
        print(f"   Username: {username}")
        print(f"   Password: {'*' * len(password)}")
    else:
        print(f"   ❌ Credentials NOT loaded properly!")
        if not username:
            print(f"      - Username is missing or empty")
        if not password:
            print(f"      - Password is missing or empty")
else:
    print("   ❌ config_schema.api does not exist!")

print("\n" + "=" * 70)

