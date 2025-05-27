"""
Utility functions for managing AI prompts.

This module provides functions for loading, updating, and managing AI prompts
stored in a JSON file. It is used by both the main codebase and the
test_ai_responses_menu.py script.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from logging_config import logger

# Path to the AI prompts JSON file
PROMPTS_FILE = Path(os.path.dirname(os.path.abspath(__file__))) / "ai_prompts.json"

# Path to the improved prompts directory
IMPROVED_PROMPTS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "improved_prompts"


def load_prompts() -> Dict[str, Any]:
    """
    Load AI prompts from the JSON file.

    Returns:
        Dict[str, Any]: The loaded prompts data
    """
    try:
        if not PROMPTS_FILE.exists():
            logger.warning(f"AI prompts file not found at {PROMPTS_FILE}")
            return {"version": "1.0", "last_updated": datetime.now().strftime("%Y-%m-%d"), "prompts": {}}

        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            prompts_data = json.load(f)
            logger.info(f"Loaded AI prompts from {PROMPTS_FILE}")
            return prompts_data
    except Exception as e:
        logger.error(f"Error loading AI prompts: {e}", exc_info=True)
        return {"version": "1.0", "last_updated": datetime.now().strftime("%Y-%m-%d"), "prompts": {}}


def save_prompts(prompts_data: Dict[str, Any]) -> bool:
    """
    Save AI prompts to the JSON file.

    Args:
        prompts_data: The prompts data to save

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Update the last_updated field
        prompts_data["last_updated"] = datetime.now().strftime("%Y-%m-%d")

        # Save the prompts to the JSON file
        with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
            json.dump(prompts_data, indent=2, ensure_ascii=False, fp=f)
            logger.info(f"Saved AI prompts to {PROMPTS_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving AI prompts: {e}", exc_info=True)
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


def update_prompt(prompt_key: str, new_prompt: str, name: Optional[str] = None, description: Optional[str] = None) -> bool:
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
                "prompt": new_prompt
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
            logger.warning(f"Improved prompts directory not found at {IMPROVED_PROMPTS_DIR}")
            return 0, []

        imported_count = 0
        imported_keys = []

        # Check for improved extraction prompt
        extraction_prompt_file = IMPROVED_PROMPTS_DIR / "improved_extraction_prompt.txt"
        if extraction_prompt_file.exists():
            with open(extraction_prompt_file, "r", encoding="utf-8") as f:
                improved_extraction_prompt = f.read()
                if update_prompt(
                    "extraction_task",
                    improved_extraction_prompt,
                    "Improved Data Extraction & Task Suggestion Prompt",
                    "Updated extraction prompt based on feedback analysis"
                ):
                    imported_count += 1
                    imported_keys.append("extraction_task")
                    logger.info(f"Imported improved extraction prompt from {extraction_prompt_file}")

        # Check for improved response prompt
        response_prompt_file = IMPROVED_PROMPTS_DIR / "improved_response_prompt.txt"
        if response_prompt_file.exists():
            with open(response_prompt_file, "r", encoding="utf-8") as f:
                improved_response_prompt = f.read()
                if update_prompt(
                    "genealogical_reply",
                    improved_response_prompt,
                    "Improved Genealogical Reply Generation Prompt",
                    "Updated reply prompt based on feedback analysis"
                ):
                    imported_count += 1
                    imported_keys.append("genealogical_reply")
                    logger.info(f"Imported improved response prompt from {response_prompt_file}")

        return imported_count, imported_keys
    except Exception as e:
        logger.error(f"Error importing improved prompts: {e}", exc_info=True)
        return 0, []


if __name__ == "__main__":
    # Test the module
    print("Testing AI prompt utilities...")
    
    # Load prompts
    prompts = load_prompts()
    print(f"Loaded {len(prompts.get('prompts', {}))} prompts")
    
    # Get a specific prompt
    intent_prompt = get_prompt("intent_classification")
    if intent_prompt:
        print(f"Intent classification prompt: {intent_prompt[:50]}...")
    
    # Update a prompt
    test_update = update_prompt(
        "test_prompt",
        "This is a test prompt",
        "Test Prompt",
        "A prompt for testing purposes"
    )
    print(f"Updated test prompt: {test_update}")
    
    # Import improved prompts
    imported_count, imported_keys = import_improved_prompts()
    print(f"Imported {imported_count} improved prompts: {', '.join(imported_keys)}")
