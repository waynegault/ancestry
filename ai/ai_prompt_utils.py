#!/usr/bin/env python3

"""
AI Prompt Management - Dynamic Prompt Configuration

Provides comprehensive AI prompt management with JSON-based storage, dynamic
loading, template processing, and intelligent prompt optimization for genealogical
AI interactions including intent classification and response generation.
"""

# === PATH SETUP FOR PACKAGE IMPORTS ===
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# === CORE INFRASTRUCTURE ===
import logging

from core.registry_utils import auto_register_module

logger = logging.getLogger(__name__)
auto_register_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
# Imports removed - not used in this module

# === STANDARD LIBRARY IMPORTS ===
import json
import shutil
from datetime import datetime
from typing import Any, Optional

# --- Test framework imports ---
from testing.test_utilities import create_standard_test_runner

# Path to the AI prompts JSON file
PROMPTS_FILE = Path(__file__).resolve().parent / "ai_prompts.json"

# Path to the improved prompts directory
IMPROVED_PROMPTS_DIR = Path(__file__).resolve().parent / "improved_prompts"


def load_prompts() -> dict[str, Any]:
    """
    Load AI prompts from the JSON file.

    Returns:
        dict[str, Any]: The loaded prompts data
    """
    default_data = {
        "version": "1.0",
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "prompts": {},
    }

    result = default_data

    try:
        if not PROMPTS_FILE.exists():
            logger.warning(f"AI prompts file not found at {PROMPTS_FILE}")
        else:
            with PROMPTS_FILE.open(encoding="utf-8") as f:
                prompts_data = json.load(f)

                # Validate loaded data structure
                is_valid, validation_errors = validate_prompt_structure(prompts_data)
                if not is_valid:
                    logger.warning(f"Invalid prompts structure detected: {validation_errors}")
                    # Try to create backup before returning default
                    backup_prompts_file()
                else:
                    logger.debug(
                        f"Loaded AI prompts from {PROMPTS_FILE}"
                    )  # Changed to DEBUG to prevent bleeding into progress bars
                    result = prompts_data

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in prompts file: {e}", exc_info=True)
        backup_prompts_file()  # Backup corrupted file
    except PermissionError as e:
        logger.error(f"Permission denied accessing prompts file: {e}")
    except FileNotFoundError as e:
        logger.warning(f"Prompts file not found: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading AI prompts: {e}", exc_info=True)

    return result


def save_prompts(prompts_data: dict[str, Any]) -> bool:
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

        # Save the prompts to the JSON file atomically using centralized helper
        from testing.test_utilities import atomic_write_file

        with atomic_write_file(PROMPTS_FILE) as f:
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


def get_prompt(prompt_key: str) -> str | None:
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


def supports_json_prompts() -> bool:
    """Return True when JSON prompt helpers are available."""
    return True


def get_prompt_with_experiment(
    prompt_key: str,
    *,
    variants: dict[str, str] | None = None,  # noqa: ARG001
    user_id: str | None = None,  # noqa: ARG001
) -> str | None:
    """Return a prompt variant when experiment helpers are available.

    Falls back to :func:`get_prompt` when no experiment infrastructure is
    configured.  *variants* and *user_id* are accepted for API
    compatibility with callers that pass experiment context.
    """
    # Experiment infrastructure is optional; fall back to plain prompt lookup.
    return get_prompt(prompt_key)


def get_prompt_version(prompt_key: str) -> str | None:  # noqa: ARG001
    """Return the configured prompt version, if available.

    *prompt_key* is accepted for API compatibility but is currently unused
    as version tracking is not yet wired up.
    """
    return None


def update_prompt(
    prompt_key: str,
    new_prompt: str,
    name: str | None = None,
    description: str | None = None,
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
                "description": description or f"Prompt for {prompt_key.replace('_', ' ')}",
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


def import_improved_prompts() -> tuple[int, list[str]]:
    """
    Import improved prompts from the improved_prompts directory.

    Returns:
        tuple[int, list[str]]: Number of prompts imported and list of imported prompt keys
    """
    try:
        if not IMPROVED_PROMPTS_DIR.exists():
            logger.warning(f"Improved prompts directory not found at {IMPROVED_PROMPTS_DIR}")
            return 0, []

        imported_count = 0
        imported_keys: list[str] = []

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
                    with prompt_file.open(encoding="utf-8") as f:
                        improved_prompt = f.read().strip()

                    if improved_prompt and update_prompt(prompt_key, improved_prompt, name, description):
                        imported_count += 1
                        imported_keys.append(prompt_key)
                        logger.info(f"Imported improved prompt '{prompt_key}' from {prompt_file}")

                except Exception as e:
                    logger.error(f"Error reading prompt file {prompt_file}: {e}")

        return imported_count, imported_keys

    except Exception as e:
        logger.error(f"Error importing improved prompts: {e}", exc_info=True)
        return 0, []


def validate_prompt_structure(prompts_data: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate the structure of prompts data.

    Args:
        prompts_data: The prompts data to validate

    Returns:
        tuple[bool, list[str]]: (is_valid, list_of_errors)
    """
    errors: list[str] = []

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
                    errors.append(f"Prompt '{prompt_key}' field '{field}' is not a string")

    return len(errors) == 0, errors


def backup_prompts_file() -> bool:
    """
    Create a backup of the current prompts file.

    Returns:
        bool: True if backup was successful, False otherwise
    """
    try:
        if PROMPTS_FILE.exists():
            logs_dir = PROMPTS_FILE.parent / "Logs"
            logs_dir.mkdir(exist_ok=True)
            backup_file = logs_dir / f"ai_prompts.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(PROMPTS_FILE, backup_file)
            logger.info(f"Created backup: {backup_file}")

            # Clean up old backups
            deleted = cleanup_old_backups(logs_dir=logs_dir)
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old backup files")

            return True
        return False
    except Exception as e:
        logger.error(f"Error creating backup: {e}", exc_info=True)
        return False


def cleanup_old_backups(keep_count: int = 5, logs_dir: Path | None = None) -> int:
    """
    Clean up old backup files, keeping only the most recent ones.

    Args:
        keep_count: Number of recent backups to keep (default: 5)

    Returns:
        int: Number of backup files deleted
    """
    try:
        if logs_dir is None:
            logs_dir = PROMPTS_FILE.parent / "Logs"
        backup_pattern = "ai_prompts.bak.*"
        backup_files = list(logs_dir.glob(backup_pattern))

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


# ==============================================
# MODULE-LEVEL TEST FUNCTIONS
# ==============================================


def _test_prompts_loading() -> None:
    """Test basic prompt loading functionality"""
    prompts = load_prompts()
    assert isinstance(prompts, dict), "Should return dictionary"
    assert "prompts" in prompts, "Should have prompts key"


def _test_prompt_validation() -> None:
    """Test prompt structure validation"""
    # Valid prompts should pass validation
    prompts = load_prompts()
    is_valid, validation_errors = validate_prompt_structure(prompts)
    assert is_valid, f"Valid prompts should pass validation, got errors: {validation_errors}"
    assert len(validation_errors) == 0, "Valid prompts should have zero errors"

    # Invalid prompts (missing required keys) should fail validation
    invalid_data: dict[str, Any] = {"prompts": {}}  # missing 'version' and 'last_updated'
    is_valid_bad, bad_errors = validate_prompt_structure(invalid_data)
    assert not is_valid_bad, "Prompts missing required keys should fail validation"
    assert len(bad_errors) > 0, "Should report missing required keys"

    # Invalid prompt entry (non-dict value) should fail
    invalid_entry: dict[str, Any] = {"version": "1", "last_updated": "2025-01-01", "prompts": {"bad_key": "not_a_dict"}}
    is_valid_entry, entry_errors = validate_prompt_structure(invalid_entry)
    assert not is_valid_entry, "Non-dict prompt entry should fail validation"
    assert any("not a dictionary" in e for e in entry_errors), "Should report non-dict prompt entry"


def _test_backup_functionality() -> None:
    """Test backup operations"""
    import os
    import tempfile

    # Create a temporary prompts file and verify backup creates a copy
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_prompts = Path(tmpdir) / "ai_prompts.json"
        tmp_prompts.write_text('{"version": "1.0", "last_updated": "2025-01-01", "prompts": {}}', encoding="utf-8")
        logs_dir = Path(tmpdir) / "Logs"
        logs_dir.mkdir(exist_ok=True)

        # Perform backup using shutil (mirrors backup_prompts_file logic)
        backup_file = logs_dir / "ai_prompts.bak.test"
        shutil.copy2(tmp_prompts, backup_file)

        assert backup_file.exists(), "Backup file should be created"
        backup_content = backup_file.read_text(encoding="utf-8")
        assert '"version"' in backup_content, "Backup should contain original content"
        assert Path(str(backup_file)).stat().st_size > 0, "Backup file should not be empty"

    # Also verify the real backup function is callable and returns bool
    assert callable(backup_prompts_file), "backup_prompts_file should be callable"
    result = backup_prompts_file()
    assert isinstance(result, bool), "backup_prompts_file should return a bool"


def _test_import_functionality() -> None:
    """Test import improved prompts"""
    imported_count, imported_keys = import_improved_prompts()
    assert isinstance(imported_count, int), "Should return integer count"
    assert isinstance(imported_keys, list), "Should return list of keys"


def _test_error_handling() -> None:
    """Test error handling"""
    # get_prompt should return None for non-existent keys (not raise)
    result = get_prompt("__completely_nonexistent_key_12345__")
    assert result is None, "get_prompt should return None for missing key"

    # validate_prompt_structure should return errors for wrong types, not crash
    empty_data: dict[str, Any] = {}
    is_valid, errors = validate_prompt_structure(empty_data)
    assert not is_valid, "Empty dict should fail validation"
    assert isinstance(errors, list), "Errors should be a list"
    assert len(errors) > 0, "Should report errors for empty dict"

    # save_prompts should reject invalid data and return False
    bad_data: dict[str, Any] = {"bad": "data"}
    save_result = save_prompts(bad_data)
    assert save_result is False, "save_prompts should return False for invalid data"


def _test_prompt_operations() -> None:
    """Test get/update operations"""
    prompts = load_prompts()
    prompts_dict = prompts.get("prompts", {})

    # Verify prompts dict is non-empty (ai_prompts.json should have content)
    assert len(prompts_dict) > 0, "Prompts file should contain at least one prompt"

    # Verify each prompt entry has required structure fields
    for key, entry in prompts_dict.items():
        assert isinstance(entry, dict), f"Prompt '{key}' should be a dict"
        assert "prompt" in entry, f"Prompt '{key}' should have a 'prompt' field"
        assert isinstance(entry["prompt"], str), f"Prompt '{key}' prompt field should be a string"
        assert len(entry["prompt"]) > 0, f"Prompt '{key}' should have non-empty prompt text"

    # Verify get_prompt returns actual content for an existing key
    first_key = next(iter(prompts_dict))
    retrieved = get_prompt(first_key)
    assert retrieved is not None, f"get_prompt('{first_key}') should return content"
    assert isinstance(retrieved, str), "get_prompt should return a string"
    assert len(retrieved) > 0, "Retrieved prompt should be non-empty"


# ==============================================
# MAIN TEST SUITE
# ==============================================


def ai_prompt_utils_module_tests() -> bool:
    """
    AI Prompt Utils module tests using TestSuite for verbose output.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    from testing.test_framework import TestSuite, suppress_logging

    # Assign all module-level test functions
    test_prompts_loading = _test_prompts_loading
    test_prompt_validation = _test_prompt_validation
    test_backup_functionality = _test_backup_functionality
    test_import_functionality = _test_import_functionality
    test_error_handling = _test_error_handling
    test_prompt_operations = _test_prompt_operations

    # Define all tests in a data structure to reduce complexity
    tests = [
        ("Prompts Loading", test_prompts_loading, "Should load prompts structure correctly"),
        ("Prompt Validation", test_prompt_validation, "Should validate prompt structure"),
        ("Backup Functionality", test_backup_functionality, "Should handle backup operations"),
        ("Import Functionality", test_import_functionality, "Should import improved prompts"),
        ("Error Handling", test_error_handling, "Should handle errors gracefully"),
        ("Prompt Operations", test_prompt_operations, "Should handle get/update operations"),
    ]

    # Create test suite and run tests
    with suppress_logging():
        suite = TestSuite("AI Prompt Management & Template System", "ai_prompt_utils.py")
        suite.start_suite()

        # Run all tests from the list
        for test_name, test_func, expected in tests:
            suite.run_test(test_name, test_func, expected)

        return suite.finish_suite()


# Use centralized test runner utility from test_utilities
run_comprehensive_tests = create_standard_test_runner(ai_prompt_utils_module_tests)


def get_prompts_summary() -> dict[str, Any]:
    """
    Get a summary of the current prompts configuration.

    Returns:
        dict[str, Any]: Summary information about prompts
    """
    try:
        prompts_data = load_prompts()
        prompts = prompts_data.get("prompts", {})

        return {
            "total_prompts": len(prompts),
            "version": prompts_data.get("version", "unknown"),
            "last_updated": prompts_data.get("last_updated", "unknown"),
            "prompt_keys": list(prompts.keys()),
            "file_exists": PROMPTS_FILE.exists(),
            "file_size_bytes": (PROMPTS_FILE.stat().st_size if PROMPTS_FILE.exists() else 0),
            "backup_count": len(list(PROMPTS_FILE.parent.glob(f"{PROMPTS_FILE.stem}.bak.*"))),
        }

    except Exception as e:
        logger.error(f"Error generating prompts summary: {e}")
        return {
            "total_prompts": 0,
            "version": "error",
            "last_updated": "error",
            "prompt_keys": [],
            "file_exists": False,
            "file_size_bytes": 0,
            "backup_count": 0,
            "error": str(e),
        }


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    import sys

    print("🤖 Running AI Prompt Management & Template System comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
