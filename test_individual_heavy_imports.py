# Test individual heavy imports from action9
import sys
import time

print("Testing individual heavy imports from action9...")


def test_import_with_timing(module_name, import_statement):
    print(f"\nTesting: {import_statement}")
    start_time = time.time()
    try:
        exec(import_statement)
        elapsed = time.time() - start_time
        print(f"   ✓ Success in {elapsed:.2f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   ✗ Failed in {elapsed:.2f}s: {e}")
        return False


# Test each import individually
imports_to_test = [
    ("config", "from config import config_instance"),
    (
        "database",
        "from database import ConversationLog, MessageDirectionEnum, MessageType, Person, PersonStatusEnum, commit_bulk_data",
    ),
    ("logging_config", "from logging_config import logger"),
    ("ms_graph_utils", "import ms_graph_utils"),
    ("utils", "from utils import SessionManager, format_name"),
    ("ai_interface", "from ai_interface import extract_and_suggest_tasks"),
    ("cache", "from cache import cache_result"),
    ("api_utils", "from api_utils import call_send_message_api"),
]

results = {}
for module_name, import_statement in imports_to_test:
    results[module_name] = test_import_with_timing(module_name, import_statement)

print(f"\n=== SUMMARY ===")
for module_name, success in results.items():
    status = "✓" if success else "✗"
    print(f"{status} {module_name}")

print("\nTesting the problematic import combinations...")
print("Testing all together:")
start_time = time.time()
try:
    # Import all at once
    from config import config_instance
    from database import (
        ConversationLog,
        MessageDirectionEnum,
        MessageType,
        Person,
        PersonStatusEnum,
        commit_bulk_data,
    )
    from logging_config import logger
    import ms_graph_utils
    from utils import SessionManager, format_name
    from ai_interface import extract_and_suggest_tasks
    from cache import cache_result
    from api_utils import call_send_message_api

    elapsed = time.time() - start_time
    print(f"   ✓ All imports successful in {elapsed:.2f}s")
except Exception as e:
    elapsed = time.time() - start_time
    print(f"   ✗ Combined imports failed in {elapsed:.2f}s: {e}")

print("Test completed!")
