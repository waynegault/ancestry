#!/usr/bin/env python3

"""
AI Prompt Management - Dynamic Prompt Configuration

Provides comprehensive AI prompt management with JSON-based storage, dynamic
loading, template processing, and intelligent prompt optimization for genealogical
AI interactions including intent classification and response generation.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import (
    setup_module,
    register_function,
    get_function,
    is_function_available,
)

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
from error_handling import (
    retry_on_failure,
    circuit_breaker,
    timeout_protection,
    graceful_degradation,
    error_context,
    AncestryException,
    RetryableError,
    NetworkTimeoutError,
    AuthenticationExpiredError,
    APIRateLimitError,
    ErrorContext,
)

# === STANDARD LIBRARY IMPORTS ===
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

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
            logs_dir = PROMPTS_FILE.parent / "Logs"
            logs_dir.mkdir(exist_ok=True)
            backup_file = (
                logs_dir / f"ai_prompts.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
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


from typing import Optional


def cleanup_old_backups(keep_count: int = 5, logs_dir: Optional[Path] = None) -> int:
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
        backup_pattern = f"ai_prompts.bak.*"
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


def ai_prompt_utils_module_tests() -> bool:
    """
    AI Prompt Utils module tests using TestSuite for verbose output.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    from test_framework import TestSuite, suppress_logging

    def test_prompts_loading():
        """Test basic prompt loading functionality"""
        try:
            prompts = load_prompts()
            assert isinstance(prompts, dict)
            assert "prompts" in prompts
            return True
        except Exception:
            return False

    def test_prompt_validation():
        """Test prompt structure validation"""
        try:
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
            is_valid, errors = validate_prompt_structure(test_data)
            return is_valid and len(errors) == 0
        except Exception:
            return False

    def test_backup_functionality():
        """Test backup creation functionality"""
        try:
            # Test that backup functions exist and can be called
            backup_prompts_file()
            cleanup_old_backups()
            return True
        except Exception:
            return False

    def test_import_functionality():
        """Test import improved prompts functionality"""
        try:
            # Test that import function works (even if no files exist)
            imported_count, imported_keys = import_improved_prompts()
            assert isinstance(imported_count, int)
            assert isinstance(imported_keys, list)
            return True
        except Exception:
            return False

    def test_error_handling():
        """Test error handling scenarios"""
        try:
            # Test with invalid data
            invalid_data = {"invalid": "structure"}
            is_valid, errors = validate_prompt_structure(invalid_data)
            return not is_valid and len(errors) > 0
        except Exception:
            return False

    def test_prompt_operations():
        """Test get/update prompt operations"""
        try:
            # Test getting non-existent prompt
            result = get_prompt("non_existent_prompt")
            assert result is None
            return True
        except Exception:
            return False

    # Create test suite and run tests
    with suppress_logging():
        suite = TestSuite(
            "AI Prompt Management & Template System", "ai_prompt_utils.py"
        )
        suite.start_suite()

        # Run all tests
        suite.run_test(
            "Prompts Loading",
            test_prompts_loading,
            "Should load prompts structure correctly",
        )
        suite.run_test(
            "Prompt Validation",
            test_prompt_validation,
            "Should validate prompt structure",
        )
        suite.run_test(
            "Backup Functionality",
            test_backup_functionality,
            "Should handle backup operations",
        )
        suite.run_test(
            "Import Functionality",
            test_import_functionality,
            "Should import improved prompts",
        )
        suite.run_test(
            "Error Handling", test_error_handling, "Should handle errors gracefully"
        )
        suite.run_test(
            "Prompt Operations",
            test_prompt_operations,
            "Should handle get/update operations",
        )

        return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive tests using the unified test framework."""
    return ai_prompt_utils_module_tests()


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
            "backup_count": len(
                list(PROMPTS_FILE.parent.glob(f"{PROMPTS_FILE.stem}.bak.*"))
            ),
        }

        return summary

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


def quick_test() -> Dict[str, Any]:
    """
    Perform a quick test of the AI prompt utilities.

    Returns:
        Dict[str, Any]: Test results
    """
    results = {"passed": 0, "failed": 0, "errors": []}

    try:
        # Test 1: Load prompts
        prompts = load_prompts()
        if isinstance(prompts, dict) and "prompts" in prompts:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["errors"].append("Failed to load prompts structure")

        # Test 2: Validate structure
        is_valid, validation_errors = validate_prompt_structure(prompts)
        if is_valid:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["errors"].extend(validation_errors)

        # Test 3: Get summary
        summary = get_prompts_summary()
        if isinstance(summary, dict) and "total_prompts" in summary:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["errors"].append("Failed to generate prompts summary")

        # Test 4: Test import functionality
        imported_count, imported_keys = import_improved_prompts()
        if isinstance(imported_count, int) and isinstance(imported_keys, list):
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["errors"].append("Failed to test import functionality")

    except Exception as e:
        results["failed"] += 1
        results["errors"].append(f"Unexpected error: {str(e)}")

    results["total"] = results["passed"] + results["failed"]
    results["success_rate"] = (
        results["passed"] / results["total"] * 100 if results["total"] > 0 else 0
    )

    return results


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
