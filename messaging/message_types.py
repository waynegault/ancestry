"""
Message type constants and state machine for messaging workflow.

Centralizes message type definitions and transition logic used by Actions 7-9.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure parent directory is on path for test imports
if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Message Type Constants
# ------------------------------------------------------------------------------

MESSAGE_TYPES: dict[str, str] = {
    "In_Tree-Initial": "In_Tree-Initial",
    "In_Tree-Follow_Up": "In_Tree-Follow_Up",
    "In_Tree-Final_Reminder": "In_Tree-Final_Reminder",
    "Out_Tree-Initial": "Out_Tree-Initial",
    "Out_Tree-Follow_Up": "Out_Tree-Follow_Up",
    "Out_Tree-Final_Reminder": "Out_Tree-Final_Reminder",
    "In_Tree-Initial_for_was_Out_Tree": "In_Tree-Initial_for_was_Out_Tree",
    "User_Requested_Desist": "User_Requested_Desist",
    "In_Tree-Initial_Short": "In_Tree-Initial_Short",
    "Out_Tree-Initial_Short": "Out_Tree-Initial_Short",
    "In_Tree-Initial_Confident": "In_Tree-Initial_Confident",
    "Out_Tree-Initial_Exploratory": "Out_Tree-Initial_Exploratory",
}

# Backwards compatibility alias
MESSAGE_TYPES_ACTION8 = MESSAGE_TYPES


# Core required template keys for validation
CORE_REQUIRED_TEMPLATE_KEYS = frozenset({
    "In_Tree-Initial",
    "In_Tree-Follow_Up",
    "In_Tree-Final_Reminder",
    "Out_Tree-Initial",
    "Out_Tree-Follow_Up",
    "Out_Tree-Final_Reminder",
    "In_Tree-Initial_for_was_Out_Tree",
    "User_Requested_Desist",
    "Productive_Reply_Acknowledgement",
})


# ------------------------------------------------------------------------------
# Message Transition State Machine
# ------------------------------------------------------------------------------

# Maps (current_message_type, is_in_family_tree) to next_message_type
MESSAGE_TRANSITION_TABLE: dict[tuple[Optional[str], bool], Optional[str]] = {
    # Initial messages (no previous message)
    (None, True): "In_Tree-Initial",
    (None, False): "Out_Tree-Initial",
    # In-Tree sequences
    ("In_Tree-Initial", True): "In_Tree-Follow_Up",
    ("In_Tree-Initial_for_was_Out_Tree", True): "In_Tree-Follow_Up",
    ("In_Tree-Initial_Confident", True): "In_Tree-Follow_Up",
    ("In_Tree-Initial_Short", True): "In_Tree-Follow_Up",
    ("In_Tree-Follow_Up", True): "In_Tree-Final_Reminder",
    ("In_Tree-Final_Reminder", True): None,
    # Out-Tree sequences
    ("Out_Tree-Initial", False): "Out_Tree-Follow_Up",
    ("Out_Tree-Initial_Short", False): "Out_Tree-Follow_Up",
    ("Out_Tree-Initial_Exploratory", False): "Out_Tree-Follow_Up",
    ("Out_Tree-Follow_Up", False): "Out_Tree-Final_Reminder",
    ("Out_Tree-Final_Reminder", False): None,
    # Tree status changes (Out->In)
    ("Out_Tree-Initial", True): "In_Tree-Initial_for_was_Out_Tree",
    ("Out_Tree-Follow_Up", True): "In_Tree-Initial_for_was_Out_Tree",
    ("Out_Tree-Final_Reminder", True): "In_Tree-Initial_for_was_Out_Tree",
    ("Out_Tree-Initial_Short", True): "In_Tree-Initial_for_was_Out_Tree",
    ("Out_Tree-Initial_Exploratory", True): "In_Tree-Initial_for_was_Out_Tree",
    # Tree status changes (In->Out)
    ("In_Tree-Initial", False): None,
    ("In_Tree-Follow_Up", False): None,
    ("In_Tree-Final_Reminder", False): None,
    ("In_Tree-Initial_Confident", False): None,
    ("In_Tree-Initial_Short", False): None,
    ("In_Tree-Initial_for_was_Out_Tree", False): "Out_Tree-Initial",
    # Desist ends sequence
    ("User_Requested_Desist", True): None,
    ("User_Requested_Desist", False): None,
    # Fallback for unknown types
    ("Unknown", True): "In_Tree-Initial",
    ("Unknown", False): "Out_Tree-Initial",
}


def determine_next_message_type(
    last_message_details: Optional[tuple[Optional[str], datetime, str]],
    is_in_family_tree: bool,
) -> Optional[str]:
    """
    Determine next message type based on last message and tree status.

    Uses state machine with transition table mapping (current_type, is_in_tree) to next_type.

    Args:
        last_message_details: Tuple of (message_type, timestamp, direction) or None
        is_in_family_tree: Whether the person is currently in the family tree

    Returns:
        Next message type string, or None if sequence is complete
    """
    last_message_type: Optional[str] = None
    if last_message_details:
        last_message_type, _, _ = last_message_details

    transition_key = (last_message_type, is_in_family_tree)

    if transition_key in MESSAGE_TRANSITION_TABLE:
        next_type = MESSAGE_TRANSITION_TABLE[transition_key]
    elif last_message_type:
        # Recover from unknown types by treating as initial
        logger.warning(f"Unknown message type '{last_message_type}', treating as initial")
        next_type = "In_Tree-Initial" if is_in_family_tree else "Out_Tree-Initial"
    else:
        # Fallback for initial message
        next_type = "In_Tree-Initial" if is_in_family_tree else "Out_Tree-Initial"

    if next_type:
        next_type = MESSAGE_TYPES.get(next_type, next_type)

    return next_type


def is_terminal_message_type(message_type: Optional[str]) -> bool:
    """
    Check if a message type is terminal (no further messages in sequence).

    Args:
        message_type: The message type to check

    Returns:
        True if this is a terminal message type
    """
    if message_type is None:
        return False

    terminal_types = {
        "In_Tree-Final_Reminder",
        "Out_Tree-Final_Reminder",
        "User_Requested_Desist",
    }
    return message_type in terminal_types


def get_message_type_category(message_type: Optional[str]) -> Optional[str]:
    """
    Get the category of a message type (In_Tree, Out_Tree, or special).

    Args:
        message_type: The message type to categorize

    Returns:
        Category string: "In_Tree", "Out_Tree", "Desist", or None
    """
    if message_type is None:
        return None

    if message_type.startswith("In_Tree"):
        return "In_Tree"
    if message_type.startswith("Out_Tree"):
        return "Out_Tree"
    if message_type == "User_Requested_Desist":
        return "Desist"

    return None


# ------------------------------------------------------------------------------
# Module Tests
# ------------------------------------------------------------------------------


def _test_message_types_constant() -> bool:
    """Test MESSAGE_TYPES has expected entries."""
    assert len(MESSAGE_TYPES) == 12, f"Expected 12 entries, got {len(MESSAGE_TYPES)}"
    assert "In_Tree-Initial" in MESSAGE_TYPES
    assert "Out_Tree-Final_Reminder" in MESSAGE_TYPES
    assert MESSAGE_TYPES_ACTION8 is MESSAGE_TYPES, "Alias should reference same dict"
    return True


def _test_core_required_template_keys() -> bool:
    """Test CORE_REQUIRED_TEMPLATE_KEYS has expected entries."""
    assert len(CORE_REQUIRED_TEMPLATE_KEYS) == 9, f"Expected 9 entries, got {len(CORE_REQUIRED_TEMPLATE_KEYS)}"
    assert "User_Requested_Desist" in CORE_REQUIRED_TEMPLATE_KEYS
    assert "Productive_Reply_Acknowledgement" in CORE_REQUIRED_TEMPLATE_KEYS
    return True


def _test_determine_next_message_type_initial() -> bool:
    """Test initial message type determination."""
    assert determine_next_message_type(None, True) == "In_Tree-Initial"
    assert determine_next_message_type(None, False) == "Out_Tree-Initial"
    return True


def _test_determine_next_message_type_sequence() -> bool:
    """Test message sequence progression."""
    now = datetime.now()
    assert determine_next_message_type(("In_Tree-Initial", now, "OUT"), True) == "In_Tree-Follow_Up"
    assert determine_next_message_type(("In_Tree-Follow_Up", now, "OUT"), True) == "In_Tree-Final_Reminder"
    assert determine_next_message_type(("In_Tree-Final_Reminder", now, "OUT"), True) is None
    return True


def _test_determine_next_message_type_tree_change() -> bool:
    """Test tree status change handling."""
    now = datetime.now()
    result = determine_next_message_type(("Out_Tree-Initial", now, "OUT"), True)
    assert result == "In_Tree-Initial_for_was_Out_Tree", f"Expected In_Tree-Initial_for_was_Out_Tree, got {result}"
    return True


def _test_is_terminal_message_type() -> bool:
    """Test terminal message type detection."""
    assert is_terminal_message_type("In_Tree-Final_Reminder") is True
    assert is_terminal_message_type("Out_Tree-Final_Reminder") is True
    assert is_terminal_message_type("User_Requested_Desist") is True
    assert is_terminal_message_type("In_Tree-Initial") is False
    assert is_terminal_message_type(None) is False
    return True


def _test_get_message_type_category() -> bool:
    """Test message type categorization."""
    assert get_message_type_category("In_Tree-Initial") == "In_Tree"
    assert get_message_type_category("In_Tree-Follow_Up") == "In_Tree"
    assert get_message_type_category("Out_Tree-Follow_Up") == "Out_Tree"
    assert get_message_type_category("User_Requested_Desist") == "Desist"
    assert get_message_type_category(None) is None
    return True


def module_tests() -> bool:
    """Test message type functionality."""
    from test_framework import TestSuite

    suite = TestSuite("Message Types Module", "messaging/message_types.py")

    suite.run_test(
        "MESSAGE_TYPES constant",
        _test_message_types_constant,
        "Validates MESSAGE_TYPES has 12 entries and alias works"
    )

    suite.run_test(
        "CORE_REQUIRED_TEMPLATE_KEYS",
        _test_core_required_template_keys,
        "Validates required template keys set has 9 entries"
    )

    suite.run_test(
        "Initial message type determination",
        _test_determine_next_message_type_initial,
        "Tests first message in sequence for tree/non-tree"
    )

    suite.run_test(
        "Message sequence progression",
        _test_determine_next_message_type_sequence,
        "Tests Initial -> Follow_Up -> Final_Reminder -> None"
    )

    suite.run_test(
        "Tree status change handling",
        _test_determine_next_message_type_tree_change,
        "Tests Out_Tree person becoming In_Tree"
    )

    suite.run_test(
        "Terminal message type detection",
        _test_is_terminal_message_type,
        "Tests Final_Reminder and Desist are terminal"
    )

    suite.run_test(
        "Message type categorization",
        _test_get_message_type_category,
        "Tests In_Tree/Out_Tree/Desist categorization"
    )

    return suite.finish_suite()


# Standard test runner pattern
from test_framework import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(module_tests)

if __name__ == "__main__":
    success = run_comprehensive_tests()
    raise SystemExit(0 if success else 1)
