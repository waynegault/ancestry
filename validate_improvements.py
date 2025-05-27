#!/usr/bin/env python3
"""
Validation script for the comprehensive genealogy system improvements.

This script validates:
1. AI prompts are properly formatted and accessible
2. Data models are correctly defined
3. Intent categories are updated
4. Integration points are working
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List


def validate_ai_prompts() -> bool:
    """Validate that AI prompts are properly formatted."""
    print("🔍 Validating AI prompts...")

    try:
        # Check if ai_prompts.json exists and is valid
        prompts_file = Path("ai_prompts.json")
        if not prompts_file.exists():
            print("❌ ai_prompts.json file not found")
            return False

        with open(prompts_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        # Check for required prompts in the prompts section
        prompts_section = prompts.get("prompts", {})
        if not prompts_section:
            print("❌ Missing 'prompts' section in ai_prompts.json")
            return False

        required_prompts = [
            "intent_classification",
            "extraction_task",
            "genealogical_reply",
        ]
        for prompt_name in required_prompts:
            if prompt_name not in prompts_section:
                print(f"❌ Missing {prompt_name} in prompts section")
                return False

        # Check for enhanced intent categories in prompts
        intent_prompt = prompts_section["intent_classification"]["prompt"]
        enhanced_categories = [
            "ENTHUSIASTIC",
            "CAUTIOUSLY_INTERESTED",
            "CONFUSED",
            "PRODUCTIVE",
        ]

        for category in enhanced_categories:
            if category not in intent_prompt:
                print(
                    f"⚠️  Warning: {category} not found in intent classification prompt"
                )

        print("✅ AI prompts validation passed")
        return True

    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in ai_prompts.json: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating AI prompts: {e}")
        return False


def validate_data_models() -> bool:
    """Validate that enhanced data models are properly defined."""
    print("🔍 Validating enhanced data models...")

    try:
        # Try to import the enhanced data models
        sys.path.insert(0, str(Path(__file__).parent))

        from action9_process_productive import (
            ExtractedData,
            NameData,
            VitalRecord,
            Relationship,
            Location,
            Occupation,
        )

        # Test basic model creation
        name_data = NameData(
            full_name="Test Name",
            nicknames=[],
            maiden_name=None,
            generational_suffix=None,
        )

        vital_record = VitalRecord(
            person="Test Person",
            event_type="birth",
            date="1850",
            place="Test Place",
            certainty="probable",
        )

        extracted_data = ExtractedData(
            mentioned_names=["Test Name"],
            mentioned_locations=["Test Location"],
            mentioned_dates=["1850"],
            potential_relationships=["father"],
            key_facts=["Test fact"],
        )

        # Test get_all_names method
        all_names = extracted_data.get_all_names()
        if "Test Name" not in all_names:
            print("❌ get_all_names method not working correctly")
            return False

        print("✅ Enhanced data models validation passed")
        return True

    except ImportError as e:
        print(f"❌ Could not import enhanced data models: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating data models: {e}")
        return False


def validate_intent_categories() -> bool:
    """Validate that intent categories are properly updated."""
    print("🔍 Validating intent categories...")

    try:
        sys.path.insert(0, str(Path(__file__).parent))

        from ai_interface import EXPECTED_INTENT_CATEGORIES

        expected_categories = {
            "ENTHUSIASTIC",
            "CAUTIOUSLY_INTERESTED",
            "UNINTERESTED",
            "CONFUSED",
            "PRODUCTIVE",
            "OTHER",
        }

        if EXPECTED_INTENT_CATEGORIES != expected_categories:
            print(
                f"❌ Intent categories mismatch. Expected: {expected_categories}, Got: {EXPECTED_INTENT_CATEGORIES}"
            )
            return False

        print("✅ Intent categories validation passed")
        return True

    except ImportError as e:
        print(f"❌ Could not import intent categories: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating intent categories: {e}")
        return False


def validate_action7_updates() -> bool:
    """Validate that Action 7 updates are properly implemented."""
    print("🔍 Validating Action 7 updates...")

    try:
        sys.path.insert(0, str(Path(__file__).parent))

        # Check if action7_inbox.py exists
        action7_file = Path("action7_inbox.py")
        if not action7_file.exists():
            print("❌ action7_inbox.py file not found")
            return False

        # Read the file and check for PRODUCTIVE handling
        with open(action7_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for enhanced PRODUCTIVE message handling
        if 'ai_sentiment_result == "PRODUCTIVE"' not in content:
            print(
                "❌ Enhanced PRODUCTIVE message handling not found in action7_inbox.py"
            )
            return False

        print("✅ Action 7 updates validation passed")
        return True

    except Exception as e:
        print(f"❌ Error validating Action 7 updates: {e}")
        return False


def validate_action9_enhancements() -> bool:
    """Validate that Action 9 enhancements are properly implemented."""
    print("🔍 Validating Action 9 enhancements...")

    try:
        sys.path.insert(0, str(Path(__file__).parent))

        from action9_process_productive import _search_ancestry_tree, ExtractedData

        # Check if the function signature accepts ExtractedData
        import inspect

        sig = inspect.signature(_search_ancestry_tree)
        params = list(sig.parameters.keys())

        if "extracted_data" not in params:
            print("❌ _search_ancestry_tree does not accept extracted_data parameter")
            return False

        print("✅ Action 9 enhancements validation passed")
        return True

    except ImportError as e:
        print(f"❌ Could not import Action 9 modules: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating Action 9 enhancements: {e}")
        return False


def validate_file_structure() -> bool:
    """Validate that all required files exist."""
    print("🔍 Validating file structure...")

    required_files = [
        "ai_prompts.json",
        "ai_interface.py",
        "action7_inbox.py",
        "action9_process_productive.py",
        "config.py",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False

    print("✅ File structure validation passed")
    return True


def run_validation() -> bool:
    """Run all validation checks."""
    print("=" * 60)
    print("GENEALOGY SYSTEM IMPROVEMENTS VALIDATION")
    print("=" * 60)

    validations = [
        ("File Structure", validate_file_structure),
        ("AI Prompts", validate_ai_prompts),
        ("Data Models", validate_data_models),
        ("Intent Categories", validate_intent_categories),
        ("Action 7 Updates", validate_action7_updates),
        ("Action 9 Enhancements", validate_action9_enhancements),
    ]

    results = []
    for name, validation_func in validations:
        print(f"\n📋 {name}")
        print("-" * 40)
        try:
            result = validation_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Validation failed with exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<25} {status}")

    print(f"\nOverall: {passed}/{total} validations passed")

    if passed == total:
        print("🎉 All validations passed! The improvements are ready to use.")
        return True
    else:
        print("⚠️  Some validations failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
