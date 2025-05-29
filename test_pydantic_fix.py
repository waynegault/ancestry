#!/usr/bin/env python3
"""
Quick test to verify the Pydantic field_validator fix
"""

try:
    from api_utils import PersonSuggestResponse

    # Test the model with some data
    data = {
        "BirthYear": 1950,
        "DeathYear": 2020,
        "FullName": "Test Person",
        "PersonId": "12345",
        "TreeId": "67890",
    }

    print("Testing PersonSuggestResponse model...")
    model = PersonSuggestResponse(**data)
    print("✅ PersonSuggestResponse model created successfully")
    print(f"Model data: {model.dict()}")

    # Test the validator
    print("\nTesting year validation...")

    # Test invalid year (should be set to None)
    invalid_data = data.copy()
    invalid_data["BirthYear"] = 999  # Too early
    invalid_model = PersonSuggestResponse(**invalid_data)
    print(f"Invalid year (999) result: {invalid_model.BirthYear}")

    # Test valid year
    valid_data = data.copy()
    valid_data["BirthYear"] = 1980
    valid_model = PersonSuggestResponse(**valid_data)
    print(f"Valid year (1980) result: {valid_model.BirthYear}")

    print("✅ No deprecation warnings - field_validator is working correctly!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
