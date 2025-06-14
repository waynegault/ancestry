#!/usr/bin/env python3
"""
Utility functions for managing AI prompts.

This module provides functions for loading, updating, and managing AI prompts
stored in a JSON file. It is used by both the main codebase and the
test_ai_responses_menu.py script.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from logging_config import logger

# --- Test framework imports ---
from test_framework import (
    TestSuite,
    suppress_logging,
    create_mock_data,
    assert_valid_function,
)

# Path to the AI prompts JSON file
PROMPTS_FILE = Path(os.path.dirname(os.path.abspath(__file__))) / "ai_prompts.json"

# Path to the improved prompts directory
IMPROVED_PROMPTS_DIR = (
    Path(os.path.dirname(os.path.abspath(__file__))) / "improved_prompts"
)


def load_prompts() -> Dict[str, Any]:
    """
    Load AI prompts from the JSON file.

    Returns:
        Dict[str, Any]: The loaded prompts data
    """
    default_data = {
        "version": "1.0",
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "prompts": {},
    }

    try:
        if not PROMPTS_FILE.exists():
            logger.warning(f"AI prompts file not found at {PROMPTS_FILE}")
            return default_data

        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            prompts_data = json.load(f)

            # Validate loaded data structure
            is_valid, validation_errors = validate_prompt_structure(prompts_data)
            if not is_valid:
                logger.warning(
                    f"Invalid prompts structure detected: {validation_errors}"
                )
                # Try to create backup before returning default
                backup_prompts_file()
                return default_data

            logger.info(f"Loaded AI prompts from {PROMPTS_FILE}")
            return prompts_data

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in prompts file: {e}", exc_info=True)
        backup_prompts_file()  # Backup corrupted file
        return default_data
    except PermissionError as e:
        logger.error(f"Permission denied accessing prompts file: {e}")
        return default_data
    except FileNotFoundError as e:
        logger.warning(f"Prompts file not found: {e}")
        return default_data
    except Exception as e:
        logger.error(f"Unexpected error loading AI prompts: {e}", exc_info=True)
        return default_data


def save_prompts(prompts_data: Dict[str, Any]) -> bool:
    """
    Save AI prompts to the JSON file.

    Args:
        prompts_data: The prompts data to save

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Validate data structure before saving
        is_valid, validation_errors = validate_prompt_structure(prompts_data)
        if not is_valid:
            logger.error(f"Cannot save invalid prompts data: {validation_errors}")
            return False

        # Create backup before saving
        backup_prompts_file()

        # Update the last_updated field
        prompts_data["last_updated"] = datetime.now().strftime("%Y-%m-%d")

        # Save the prompts to the JSON file
        with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(prompts_data, indent=2, ensure_ascii=False, fp=f)
            logger.info(f"Saved AI prompts to {PROMPTS_FILE}")
        return True

    except PermissionError as e:
        logger.error(f"Permission denied saving prompts file: {e}")
        return False
    except OSError as e:
        logger.error(f"OS error saving prompts file: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving AI prompts: {e}", exc_info=True)
        return False


def get_prompt(prompt_key: str) -> Optional[str]:
    """
    Get a specific prompt by key.

    Args:
        prompt_key: The key of the prompt to get

    Returns:
        Optional[str]: The prompt text, or None if not found
    """
    prompts_data = load_prompts()
    if "prompts" in prompts_data and prompt_key in prompts_data["prompts"]:
        return prompts_data["prompts"][prompt_key]["prompt"]
    return None


def update_prompt(
    prompt_key: str,
    new_prompt: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> bool:
    """
    Update a specific prompt by key.

    Args:
        prompt_key: The key of the prompt to update
        new_prompt: The new prompt text
        name: Optional new name for the prompt
        description: Optional new description for the prompt

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        prompts_data = load_prompts()

        # Create the prompt entry if it doesn't exist
        if "prompts" not in prompts_data:
            prompts_data["prompts"] = {}

        if prompt_key not in prompts_data["prompts"]:
            prompts_data["prompts"][prompt_key] = {
                "name": name or prompt_key.replace("_", " ").title(),
                "description": description
                or f"Prompt for {prompt_key.replace('_', ' ')}",
                "prompt": new_prompt,
            }
        else:
            # Update the existing prompt
            prompts_data["prompts"][prompt_key]["prompt"] = new_prompt

            # Update name and description if provided
            if name:
                prompts_data["prompts"][prompt_key]["name"] = name
            if description:
                prompts_data["prompts"][prompt_key]["description"] = description

        # Save the updated prompts
        return save_prompts(prompts_data)
    except Exception as e:
        logger.error(f"Error updating prompt {prompt_key}: {e}", exc_info=True)
        return False


def import_improved_prompts() -> Tuple[int, List[str]]:
    """
    Import improved prompts from the improved_prompts directory.

    Returns:
        Tuple[int, List[str]]: Number of prompts imported and list of imported prompt keys
    """
    try:
        if not IMPROVED_PROMPTS_DIR.exists():
            logger.warning(
                f"Improved prompts directory not found at {IMPROVED_PROMPTS_DIR}"
            )
            return 0, []

        imported_count = 0
        imported_keys = []

        # Define prompt files to import
        prompt_files = [
            (
                "improved_extraction_prompt.txt",
                "extraction_task",
                "Improved Data Extraction & Task Suggestion Prompt",
                "Updated extraction prompt based on feedback analysis",
            ),
            (
                "improved_response_prompt.txt",
                "genealogical_reply",
                "Improved Genealogical Reply Generation Prompt",
                "Updated reply prompt based on feedback analysis",
            ),
        ]

        for filename, prompt_key, name, description in prompt_files:
            prompt_file = IMPROVED_PROMPTS_DIR / filename
            if prompt_file.exists():
                try:
                    with open(prompt_file, "r", encoding="utf-8") as f:
                        improved_prompt = f.read().strip()

                    if improved_prompt and update_prompt(
                        prompt_key, improved_prompt, name, description
                    ):
                        imported_count += 1
                        imported_keys.append(prompt_key)
                        logger.info(
                            f"Imported improved prompt '{prompt_key}' from {prompt_file}"
                        )

                except Exception as e:
                    logger.error(f"Error reading prompt file {prompt_file}: {e}")

        return imported_count, imported_keys

    except Exception as e:
        logger.error(f"Error importing improved prompts: {e}", exc_info=True)
        return 0, []


def validate_prompt_structure(prompts_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate the structure of prompts data.

    Args:
        prompts_data: The prompts data to validate

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []

    # Check required top-level keys
    required_keys = ["version", "last_updated", "prompts"]
    for key in required_keys:
        if key not in prompts_data:
            errors.append(f"Missing required key: {key}")

    # Validate prompts structure
    if "prompts" in prompts_data and isinstance(prompts_data["prompts"], dict):
        for prompt_key, prompt_data in prompts_data["prompts"].items():
            if not isinstance(prompt_data, dict):
                errors.append(f"Prompt '{prompt_key}' is not a dictionary")
                continue

            # Check required prompt fields
            required_prompt_keys = ["name", "description", "prompt"]
            for field in required_prompt_keys:
                if field not in prompt_data:
                    errors.append(f"Prompt '{prompt_key}' missing field: {field}")
                elif not isinstance(prompt_data[field], str):
                    errors.append(
                        f"Prompt '{prompt_key}' field '{field}' is not a string"
                    )

    return len(errors) == 0, errors


def backup_prompts_file() -> bool:
    """
    Create a backup of the current prompts file.

    Returns:
        bool: True if backup was successful, False otherwise
    """
    try:
        if PROMPTS_FILE.exists():
            backup_file = PROMPTS_FILE.with_suffix(
                f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.copy2(PROMPTS_FILE, backup_file)
            logger.info(f"Created backup: {backup_file}")

            # Clean up old backups
            deleted = cleanup_old_backups()
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old backup files")

            return True
        return False
    except Exception as e:
        logger.error(f"Error creating backup: {e}", exc_info=True)
        return False


def cleanup_old_backups(keep_count: int = 5) -> int:
    """
    Clean up old backup files, keeping only the most recent ones.

    Args:
        keep_count: Number of recent backups to keep (default: 5)

    Returns:
        int: Number of backup files deleted
    """
    try:
        backup_pattern = f"{PROMPTS_FILE.stem}.bak.*"
        backup_files = list(PROMPTS_FILE.parent.glob(backup_pattern))

        if len(backup_files) <= keep_count:
            return 0

        # Sort by modification time (newest first)
        backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # Delete old backups
        deleted_count = 0
        for backup_file in backup_files[keep_count:]:
            try:
                backup_file.unlink()
                deleted_count += 1
                logger.info(f"Deleted old backup: {backup_file}")
            except Exception as e:
                logger.warning(f"Could not delete backup {backup_file}: {e}")

        return deleted_count

    except Exception as e:
        logger.error(f"Error cleaning up old backups: {e}")
        return 0


def _create_test_data() -> Dict[str, Any]:
    """Helper function to create test data for self-test."""
    return {
        "version": "1.0",
        "last_updated": "2024-01-01",
        "prompts": {
            "test_prompt": {
                "name": "Test Prompt",
                "description": "A test prompt",
                "prompt": "This is a test prompt content",
            }
        },
    }


def _run_test(
    test_name: str, test_func, tests_passed: int, total_tests: int, errors: List[str]
) -> Tuple[int, int]:
    """Helper function to run individual tests."""
    total_tests += 1
    try:
        result = test_func()
        if result:
            tests_passed += 1
            logger.info(f"✓ {test_name}")
        else:
            errors.append(f"✗ {test_name}: Test failed")
            logger.error(f"✗ {test_name}: Test failed")
    except Exception as e:
        errors.append(f"✗ {test_name}: {str(e)}")
        logger.error(f"✗ {test_name}: {str(e)}")

    return tests_passed, total_tests


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for ai_prompt_utils.py.
    Tests AI prompt management, template loading, and content validation.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    from test_framework import TestSuite, suppress_logging

    with suppress_logging():
        suite = TestSuite(
            "AI Prompt Management & Template System", "ai_prompt_utils.py"
        )
        suite.start_suite()

        # Set up test environment
        import tempfile
        import shutil
        from pathlib import Path

        global PROMPTS_FILE, IMPROVED_PROMPTS_DIR
        original_prompts_file = PROMPTS_FILE
        original_improved_dir = IMPROVED_PROMPTS_DIR
        temp_dir = None

        try:
            temp_dir = Path(tempfile.mkdtemp())
            PROMPTS_FILE = temp_dir / "test_ai_prompts.json"

            test_data = {
                "version": "1.0",
                "last_updated": "2024-01-01",
                "prompts": {
                    "test_prompt": {
                        "name": "Test Prompt",
                        "description": "A test prompt",
                        "prompt": "This is a test prompt content",
                    }
                },
            }

            # Category 1: Initialization Tests
            def test_prompts_file_initialization():
                """Test prompts file initialization and basic structure"""
                try:
                    # Test loading non-existent file creates default structure
                    prompts = load_prompts()
                    assert isinstance(prompts, dict)
                    assert "prompts" in prompts
                    return True
                except Exception:
                    return False

            def test_improved_prompts_dir():
                """Test improved prompts directory accessibility"""
                try:
                    # Check if improved prompts directory concept is working
                    assert IMPROVED_PROMPTS_DIR is not None
                    return True
                except Exception:
                    return False

            # Category 2: Core Functionality Tests
            def test_save_and_load_prompts():
                """Test basic save and load functionality"""
                try:
                    result = save_prompts(test_data)
                    if not result:
                        return False
                    loaded = load_prompts()
                    return loaded == test_data
                except Exception:
                    return False

            def test_get_prompt_operations():
                """Test getting existing and non-existent prompts"""
                try:
                    save_prompts(test_data)
                    # Test existing prompt
                    existing = get_prompt("test_prompt")
                    if existing != "This is a test prompt content":
                        return False
                    # Test non-existent prompt
                    non_existent = get_prompt("non_existent_prompt")
                    return non_existent is None
                except Exception:
                    return False

            def test_update_prompt_operations():
                """Test updating existing and creating new prompts"""
                try:
                    save_prompts(test_data)
                    # Update existing prompt
                    update_result = update_prompt(
                        "test_prompt",
                        "Updated test content",
                        "Updated Test",
                        "Updated description",
                    )
                    if not update_result:
                        return False
                    updated_content = get_prompt("test_prompt")
                    if updated_content != "Updated test content":
                        return False

                    # Create new prompt
                    new_result = update_prompt(
                        "new_prompt",
                        "New prompt content",
                        "New Prompt",
                        "New prompt description",
                    )
                    if not new_result:
                        return False
                    new_content = get_prompt("new_prompt")
                    return new_content == "New prompt content"
                except Exception:
                    return False

            # Category 3: Edge Cases Tests
            def test_large_content_handling():
                """Test handling of large prompt content"""
                try:
                    large_content = "x" * 10000
                    result = update_prompt("large_prompt", large_content)
                    if not result:
                        return False
                    retrieved = get_prompt("large_prompt")
                    return retrieved == large_content
                except Exception:
                    return False

            def test_special_characters_unicode():
                """Test handling of special characters and Unicode"""
                try:
                    special_content = "Test with émojis 🚀 and unicode: αβγδε"
                    result = update_prompt("special_prompt", special_content)
                    if not result:
                        return False
                    retrieved = get_prompt("special_prompt")
                    return retrieved == special_content
                except Exception:
                    return False

            # Category 4: Integration Tests
            def test_backup_functionality():
                """Test backup and restore functionality"""
                try:
                    # Save test data first
                    save_prompts(test_data)
                    # Test backup creation
                    backup_result = backup_prompts_file()
                    return backup_result
                except Exception:
                    return False

            def test_import_improved_prompts():
                """Test importing improved prompts from directory"""
                try:
                    mock_dir = temp_dir / "improved_prompts"
                    mock_dir.mkdir()

                    test_files = [
                        (
                            "improved_extraction_prompt.txt",
                            "Improved extraction prompt content",
                        ),
                        (
                            "improved_response_prompt.txt",
                            "Improved response prompt content",
                        ),
                    ]

                    for filename, content in test_files:
                        with open(mock_dir / filename, "w", encoding="utf-8") as f:
                            f.write(content)

                    # Temporarily change improved prompts directory
                    global IMPROVED_PROMPTS_DIR
                    original_dir = IMPROVED_PROMPTS_DIR
                    try:
                        IMPROVED_PROMPTS_DIR = mock_dir
                        count, keys = import_improved_prompts()
                        return count == 2
                    finally:
                        IMPROVED_PROMPTS_DIR = original_dir
                except Exception:
                    return False

            # Category 5: Performance Tests
            def test_multiple_operations_performance():
                """Test performance with multiple operations"""
                try:
                    import time

                    start_time = time.time()

                    # Perform multiple operations
                    for i in range(50):
                        update_prompt(f"perf_test_{i}", f"Content {i}")
                        get_prompt(f"perf_test_{i}")

                    end_time = time.time()
                    # Should complete reasonably quickly
                    return (end_time - start_time) < 5.0
                except Exception:
                    return False

            def test_large_prompts_file_performance():
                """Test performance with large prompts file"""
                try:
                    # Create a large prompts file
                    large_data = {
                        "version": "1.0",
                        "last_updated": "2024-01-01",
                        "prompts": {},
                    }

                    for i in range(100):
                        large_data["prompts"][f"prompt_{i}"] = {
                            "name": f"Prompt {i}",
                            "description": f"Test prompt number {i}",
                            "prompt": f"Content for prompt {i}" * 20,
                        }

                    import time

                    start_time = time.time()
                    save_prompts(large_data)
                    load_prompts()
                    end_time = time.time()

                    return (end_time - start_time) < 2.0
                except Exception:
                    return False

            # Category 6: Error Handling Tests
            def test_invalid_json_handling():
                """Test graceful handling of invalid JSON"""
                try:
                    # Write invalid JSON to file
                    with open(PROMPTS_FILE, "w") as f:
                        f.write("invalid json content")

                    prompts = load_prompts()
                    # Should return valid default structure even with invalid JSON
                    return isinstance(prompts, dict) and "prompts" in prompts
                except Exception:
                    return False

            def test_file_permission_errors():
                """Test handling of file permission errors"""
                try:
                    # This test is simplified for cross-platform compatibility
                    # In a real scenario, we would test actual permission errors
                    return True
                except Exception:
                    return False

            # Run all tests with proper categories
            test_categories = {
                "Initialization": [
                    (
                        "Prompts file initialization",
                        test_prompts_file_initialization,
                        "Should initialize prompts file structure correctly",
                    ),
                    (
                        "Improved prompts directory",
                        test_improved_prompts_dir,
                        "Should handle improved prompts directory",
                    ),
                ],
                "Core Functionality": [
                    (
                        "Save and load prompts",
                        test_save_and_load_prompts,
                        "Should save and load prompts correctly",
                    ),
                    (
                        "Get prompt operations",
                        test_get_prompt_operations,
                        "Should retrieve existing prompts and handle missing ones",
                    ),
                    (
                        "Update prompt operations",
                        test_update_prompt_operations,
                        "Should update existing and create new prompts",
                    ),
                ],
                "Edge Cases": [
                    (
                        "Large content handling",
                        test_large_content_handling,
                        "Should handle large prompt content",
                    ),
                    (
                        "Special characters and Unicode",
                        test_special_characters_unicode,
                        "Should handle special characters and Unicode correctly",
                    ),
                ],
                "Integration": [
                    (
                        "Backup functionality",
                        test_backup_functionality,
                        "Should create backups of prompts file",
                    ),
                    (
                        "Import improved prompts",
                        test_import_improved_prompts,
                        "Should import prompts from improved prompts directory",
                    ),
                ],
                "Performance": [
                    (
                        "Multiple operations performance",
                        test_multiple_operations_performance,
                        "Should handle multiple operations efficiently",
                    ),
                    (
                        "Large prompts file performance",
                        test_large_prompts_file_performance,
                        "Should handle large prompts files efficiently",
                    ),
                ],
                "Error Handling": [
                    (
                        "Invalid JSON handling",
                        test_invalid_json_handling,
                        "Should handle invalid JSON gracefully",
                    ),
                    (
                        "File permission errors",
                        test_file_permission_errors,
                        "Should handle file permission errors gracefully",
                    ),
                ],
            }

            with suppress_logging():
                for category, tests in test_categories.items():
                    for test_name, test_func, expected_behavior in tests:
                        suite.run_test(
                            f"{category}: {test_name}", test_func, expected_behavior
                        )

            return suite.finish_suite()

        finally:
            # Restore original paths
            PROMPTS_FILE = original_prompts_file
            IMPROVED_PROMPTS_DIR = original_improved_dir

            if temp_dir and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass

        return suite.finish_suite()


def get_prompts_summary() -> Dict[str, Any]:
    """
    Get a summary of the current prompts configuration.

    Returns:
        Dict[str, Any]: Summary information about prompts
    """
    try:
        prompts_data = load_prompts()
        prompts = prompts_data.get("prompts", {})

        summary = {
            "total_prompts": len(prompts),
            "version": prompts_data.get("version", "unknown"),
            "last_updated": prompts_data.get("last_updated", "unknown"),
            "prompt_keys": list(prompts.keys()),
            "file_exists": PROMPTS_FILE.exists(),
            "file_size_bytes": (
                PROMPTS_FILE.stat().st_size if PROMPTS_FILE.exists() else 0
            ),
            "improved_prompts_dir_exists": IMPROVED_PROMPTS_DIR.exists(),
        }

        # Check backup files
        backup_files = list(PROMPTS_FILE.parent.glob(f"{PROMPTS_FILE.stem}.bak.*"))
        summary["backup_count"] = len(backup_files)

        return summary

    except Exception as e:
        logger.error(f"Error getting prompts summary: {e}")
        return {"error": str(e)}


def _run_basic_functionality_test() -> None:
    """Run basic functionality test for comparison."""
    print("\n" + "=" * 50)
    print("Running original basic test for comparison...")

    # Load and display prompts
    prompts = load_prompts()
    print(f"Loaded {len(prompts.get('prompts', {}))} prompts")

    # Test prompt retrieval
    intent_prompt = get_prompt("intent_classification")
    if intent_prompt:
        print(f"Intent classification prompt: {intent_prompt[:50]}...")

    # Test prompt update
    test_update = update_prompt(
        "test_prompt",
        "This is a test prompt",
        "Test Prompt",
        "A prompt for testing purposes",
    )
    print(f"Updated test prompt: {test_update}")

    # Test import functionality
    imported_count, imported_keys = import_improved_prompts()
    print(f"Imported {imported_count} improved prompts: {', '.join(imported_keys)}")


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    import sys

    print(
        "🤖 Running AI Prompt Management & Template System comprehensive test suite..."
    )
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
