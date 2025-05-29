import sys

print("Testing import...")
try:
    from ai_interface import extract_genealogical_entities

    print("SUCCESS: Import worked!")
except Exception as e:
    print(f"ERROR: Import failed - {e}")
    sys.exit(1)

print("Testing simple call...")
try:
    result = extract_genealogical_entities("John Smith was born in 1850.")
    print(f"SUCCESS: Function call worked! Result type: {type(result)}")
    print(f"Result: {result}")
except Exception as e:
    print(f"ERROR: Function call failed - {e}")
    import traceback

    traceback.print_exc()
