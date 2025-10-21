"""
Demonstration: Enhanced MS To-Do Task Creation

Phase 5.3: Enhanced MS To-Do Task Creation
Demonstrates the new priority-based task creation with due dates and categories.

This script shows how tasks are created with intelligent priority and due dates
based on relationship closeness to DNA matches.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from standard_imports import *

# Configure logging for demonstration
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


def demonstrate_priority_calculation():
    """
    Demonstrate priority calculation for different relationship types.

    Shows how priority, due dates, and categories are assigned based on
    relationship closeness.
    """
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION: Enhanced MS To-Do Task Creation - Priority Calculation")
    logger.info("="*80)

    # Test cases with different relationship types
    test_cases = [
        {
            "name": "John Smith",
            "relationship": "1st cousin",
            "tree_status": "in_tree",
            "shared_dna": "850 cM",
            "expected_priority": "high",
            "expected_days": 7
        },
        {
            "name": "Mary Johnson",
            "relationship": "2nd cousin",
            "tree_status": "out_tree",
            "shared_dna": "212 cM",
            "expected_priority": "high",
            "expected_days": 7
        },
        {
            "name": "Robert Brown",
            "relationship": "3rd cousin",
            "tree_status": "in_tree",
            "shared_dna": "90 cM",
            "expected_priority": "normal",
            "expected_days": 14
        },
        {
            "name": "Sarah Davis",
            "relationship": "4th cousin",
            "tree_status": "out_tree",
            "shared_dna": "35 cM",
            "expected_priority": "normal",
            "expected_days": 14
        },
        {
            "name": "James Wilson",
            "relationship": "5th cousin",
            "tree_status": "in_tree",
            "shared_dna": "15 cM",
            "expected_priority": "low",
            "expected_days": 30
        },
        {
            "name": "Patricia Martinez",
            "relationship": "Distant cousin",
            "tree_status": "out_tree",
            "shared_dna": "8 cM",
            "expected_priority": "low",
            "expected_days": 30
        },
    ]

    logger.info("\n📊 Testing Priority Calculation for Different Relationships:\n")

    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"Test Case {i}: {test_case['name']}")
        logger.info(f"  Relationship: {test_case['relationship']}")
        logger.info(f"  Tree Status: {test_case['tree_status']}")
        logger.info(f"  Shared DNA: {test_case['shared_dna']}")

        # Calculate priority based on relationship
        importance, due_date, categories = _calculate_priority_for_demo(
            test_case['relationship'],
            test_case['tree_status']
        )

        logger.info(f"  → Priority: {importance.upper()}")
        logger.info(f"  → Due Date: {due_date} ({test_case['expected_days']} days from now)")
        logger.info(f"  → Categories: {', '.join(categories)}")

        # Verify expected results
        assert importance == test_case['expected_priority'], \
            f"Expected priority {test_case['expected_priority']}, got {importance}"

        logger.info("  ✓ Priority calculation correct!\n")

    logger.info("="*80)
    logger.info("✅ All priority calculations validated successfully!")
    logger.info("="*80)


def _calculate_priority_for_demo(relationship: str, tree_status: str) -> tuple[str, str, list[str]]:
    """
    Calculate task priority and due date for demonstration.

    This mirrors the logic in action9_process_productive.py
    """
    # Default values
    importance = "normal"
    due_date = None
    categories = ["Ancestry Research"]

    # Calculate priority based on relationship closeness
    if relationship:
        rel_lower = relationship.lower()

        # High priority: Close relatives (1st-2nd cousins, immediate family)
        if any(term in rel_lower for term in ["1st", "2nd", "parent", "sibling", "child", "grandparent", "grandchild"]):
            importance = "high"
            due_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")  # 1 week
            categories.append("Close Relative")

        # Normal priority: 3rd-4th cousins
        elif any(term in rel_lower for term in ["3rd", "4th"]):
            importance = "normal"
            due_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")  # 2 weeks
            categories.append("Distant Relative")

        # Low priority: 5th+ cousins
        elif any(term in rel_lower for term in ["5th", "6th", "7th", "8th", "distant"]):
            importance = "low"
            due_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")  # 1 month
            categories.append("Distant Relative")

    # Adjust based on tree status
    if tree_status == "in_tree":
        categories.append("In Tree")
    elif tree_status == "out_tree":
        categories.append("Out of Tree")

    return importance, due_date, categories


def demonstrate_task_body_formatting():
    """
    Demonstrate enhanced task body formatting with context.

    Shows how task bodies include relationship context, DNA info, and tree status.
    """
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION: Enhanced Task Body Formatting")
    logger.info("="*80)

    # Example task
    task_desc = "Review shared ancestors and identify common family lines"
    person_name = "John Smith"
    person_id = 12345
    profile_id = "abc123xyz"
    relationship = "1st cousin"
    shared_dna = "850 cM"
    tree_status = "in_tree"

    logger.info(f"\n📝 Creating Task for: {person_name}\n")

    # Build task title
    task_title = f"Ancestry Follow-up: {person_name} (#{person_id})"
    logger.info(f"Task Title: {task_title}")

    # Calculate priority
    importance, due_date, categories = _calculate_priority_for_demo(relationship, tree_status)

    # Build task body
    task_body_parts = [
        f"AI Suggested Task: {task_desc}",
        "",
        f"Match: {person_name} (#{person_id})",
        f"Profile: {profile_id}",
        f"Relationship: {relationship}",
        f"Shared DNA: {shared_dna}",
        "Tree Status: In Tree",
    ]

    task_body = "\n".join(task_body_parts)

    logger.info(f"\nTask Body:\n{'-'*40}")
    logger.info(task_body)
    logger.info(f"{'-'*40}")

    logger.info("\nTask Metadata:")
    logger.info(f"  Priority: {importance.upper()}")
    logger.info(f"  Due Date: {due_date}")
    logger.info(f"  Categories: {', '.join(categories)}")

    logger.info("\n" + "="*80)
    logger.info("✅ Task formatting demonstration complete!")
    logger.info("="*80)


def demonstrate_ms_graph_api_payload():
    """
    Demonstrate MS Graph API payload structure for enhanced tasks.

    Shows the JSON payload sent to Microsoft Graph API.
    """
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION: MS Graph API Payload Structure")
    logger.info("="*80)

    import json

    # Example task data
    task_title = "Ancestry Follow-up: John Smith (#12345)"
    task_body = "AI Suggested Task: Review shared ancestors\n\nMatch: John Smith (#12345)\nProfile: abc123xyz\nRelationship: 1st cousin\nShared DNA: 850 cM\nTree Status: In Tree"
    importance = "high"
    due_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    categories = ["Ancestry Research", "Close Relative", "In Tree"]

    # Construct MS Graph API payload
    task_data = {
        "title": task_title,
        "body": {
            "content": task_body,
            "contentType": "text"
        },
        "importance": importance,
        "dueDateTime": {
            "dateTime": f"{due_date}T00:00:00",
            "timeZone": "UTC"
        },
        "categories": categories
    }

    logger.info("\n📤 MS Graph API Payload:\n")
    logger.info(json.dumps(task_data, indent=2))

    logger.info("\n" + "="*80)
    logger.info("✅ API payload demonstration complete!")
    logger.info("="*80)


if __name__ == "__main__":
    """Run all demonstrations."""
    logger.info("\n" + "🎯 "*40)
    logger.info("ENHANCED MS TO-DO TASK CREATION DEMONSTRATION")
    logger.info("Phase 5.3: Priority-Based Task Management")
    logger.info("🎯 "*40)

    try:
        # Run demonstrations
        demonstrate_priority_calculation()
        demonstrate_task_body_formatting()
        demonstrate_ms_graph_api_payload()

        logger.info("\n" + "🎉 "*40)
        logger.info("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        logger.info("🎉 "*40 + "\n")

    except Exception as e:
        logger.error(f"\n❌ Demonstration failed: {e}", exc_info=True)
        sys.exit(1)

