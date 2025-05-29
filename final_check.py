print("=== Final System Check ===")

# Test 1: Check imports
try:
    from ai_interface import extract_genealogical_entities

    print("✓ extract_genealogical_entities imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

try:
    from action9_process_productive import AIResponse

    print("✓ AIResponse imported successfully")
except Exception as e:
    print(f"✗ AIResponse import failed: {e}")
    exit(1)

# Test 2: Check Pydantic model works with expected structure
try:
    test_data = {
        "extracted_data": {
            "names": [],
            "vital_records": [],
            "relationships": [],
            "locations": [],
            "occupations": [],
        },
        "suggested_tasks": ["Test task"],
    }
    ai_resp = AIResponse(**test_data)
    print("✓ AIResponse model works with expected JSON structure")
except Exception as e:
    print(f"✗ Pydantic model test failed: {e}")
    exit(1)

# Test 3: Check prompt loading
try:
    import json

    with open("ai_prompts.json", "r") as f:
        prompts_data = json.load(f)

    extraction_prompt = prompts_data["prompts"]["extraction_task"]["prompt"]
    if "suggested_tasks" in extraction_prompt and "extracted_data" in extraction_prompt:
        print("✓ ai_prompts.json contains updated extraction_task prompt")
    else:
        print("✗ ai_prompts.json prompt missing required keywords")
        exit(1)
except Exception as e:
    print(f"✗ Prompt loading test failed: {e}")
    exit(1)

print("\n🎉 SUCCESS: All fixes are in place and working correctly!")
print("The AI interface system has been successfully repaired:")
print("- Import issues fixed")
print("- Prompt structure updated")
print("- Pydantic models compatible with expected JSON format")
print("- System ready for AI extraction operations")
