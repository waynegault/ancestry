"""
Unit tests for core/action_registry.py.

Tests the ActionRegistry implementation including:
- Registration and lookups
- Menu generation
- Workflow composition
- Consistency validation
- Backward compatibility
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.action_registry import (
    ActionMetadata,
    ActionRegistry,
    get_action_metadata,
    get_action_registry,
    get_browser_requirement,
    get_menu_items,
    get_required_state,
)
from test_framework import TestSuite


def test_registry_creation() -> bool:
    """Test that global registry is created correctly."""
    registry = get_action_registry()
    assert registry is not None, "Registry should not be None"
    assert len(registry) == 11, f"Expected 11 actions, got {len(registry)}"
    return True


def test_lookup_by_choice() -> bool:
    """Test O(1) lookup by choice string."""
    registry = get_action_registry()

    # Test valid choices
    for choice in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
        metadata = registry.get_by_choice(choice)
        assert metadata is not None, f"Action {choice} should exist"
        assert metadata.choice == choice, f"Choice should match: {choice}"

    # Test invalid choice
    metadata = registry.get_by_choice("999")
    assert metadata is None, "Invalid choice should return None"

    return True


def test_lookup_by_name() -> bool:
    """Test lookup by function name."""
    registry = get_action_registry()

    names = [
        "all_but_first_actn",
        "run_core_workflow_action",
        "reset_db_actn",
        "backup_db_actn",
        "restore_db_actn",
        "check_login_actn",
        "gather_dna_matches",
        "srch_inbox_actn",
        "send_messages_action",
        "process_productive_messages_action",
        "run_gedcom_then_api_fallback",
    ]

    for name in names:
        metadata = registry.get_by_name(name)
        assert metadata is not None, f"Action {name} should exist"
        assert metadata.name == name, f"Name should match: {name}"

    # Test invalid name
    metadata = registry.get_by_name("nonexistent_action")
    assert metadata is None, "Invalid name should return None"

    return True


def test_browser_requirement_consistency() -> bool:
    """Test that browser requirement is correctly set for all actions."""
    registry = get_action_registry()

    # Database-only actions (no browser needed)
    db_only = ["0", "2", "3", "4", "10"]
    for choice in db_only:
        metadata = registry.get_by_choice(choice)
        assert metadata is not None, f"Action {choice} should exist"
        assert not metadata.requires_browser, f"Action {choice} should not require browser"

    # Browser-required actions
    browser_required = ["1", "5", "6", "7", "8", "9"]
    for choice in browser_required:
        metadata = registry.get_by_choice(choice)
        assert metadata is not None, f"Action {choice} should exist"
        assert metadata.requires_browser, f"Action {choice} should require browser"

    return True


def test_required_state_consistency() -> bool:
    """Test that required state is correctly set."""
    registry = get_action_registry()

    # DB-only actions should have "db_only" or "any"
    db_only_choices = ["0", "2", "3", "4", "10"]
    for choice in db_only_choices:
        metadata = registry.get_by_choice(choice)
        assert metadata is not None, f"Action {choice} should exist"
        assert metadata.required_state in ["db_only", "any"], \
            f"Action {choice} should have db_only or any state"

    # Browser-required actions should have "ready" or "any"
    browser_choices = ["1", "5", "6", "7", "8", "9"]
    for choice in browser_choices:
        metadata = registry.get_by_choice(choice)
        assert metadata is not None, f"Action {choice} should exist"
        assert metadata.required_state in ["ready", "any"], \
            f"Action {choice} should have ready state"

    return True


def test_workflow_actions() -> bool:
    """Test that workflow actions are correctly marked."""
    registry = get_action_registry()
    workflow = registry.get_workflow_actions()

    # Should have at least 3 actions (7, 8, 9 are in workflow)
    assert len(workflow) >= 3, f"Expected at least 3 workflow actions, got {len(workflow)}"

    # Check specific workflow actions
    workflow_choices = {m.choice for m in workflow}
    assert "7" in workflow_choices, "Action 7 should be in workflow"
    assert "8" in workflow_choices, "Action 8 should be in workflow"
    assert "9" in workflow_choices, "Action 9 should be in workflow"
    assert "1" in workflow_choices, "Action 1 should be in workflow"

    return True


def test_menu_generation() -> bool:
    """Test menu item generation."""
    registry = get_action_registry()
    menu_items = registry.get_menu_items()

    # Should have 11 items
    assert len(menu_items) == 11, f"Expected 11 menu items, got {len(menu_items)}"

    # Check structure: each item should be (choice, description) tuple
    for choice, description in menu_items:
        assert isinstance(choice, str), "Choice should be string"
        assert isinstance(description, str), "Description should be string"
        assert len(description) > 0, "Description should not be empty"

    # Verify all choices are present
    choices = [choice for choice, _ in menu_items]
    for i in range(11):
        assert str(i) in choices, f"Choice {i} should be in menu"

    return True


def test_consistency_validation() -> bool:
    """Test registry consistency validation."""
    registry = get_action_registry()
    is_valid, errors = registry.validate_consistency()

    assert is_valid, f"Registry validation should pass: {errors}"
    assert len(errors) == 0, f"Should have no errors: {errors}"

    return True


def test_duplicate_prevention() -> bool:
    """Test that registry prevents duplicate registration."""
    registry = ActionRegistry()

    # Register first action
    metadata1 = ActionMetadata(
        choice="X",
        name="test_action_1",
        description="Test Action 1",
        help_text="Help",
        requires_browser=False,
        required_state="db_only",
        category="test",
    )
    registry.register(metadata1)

    # Try to register duplicate choice - should raise ValueError
    metadata2 = ActionMetadata(
        choice="X",  # Duplicate choice
        name="test_action_2",
        description="Test Action 2",
        help_text="Help",
        requires_browser=False,
        required_state="db_only",
        category="test",
    )

    try:
        registry.register(metadata2)
        raise AssertionError("Should have raised ValueError for duplicate choice")
    except ValueError as e:
        assert "Duplicate" in str(e)

    # Try to register duplicate name - should raise ValueError
    metadata3 = ActionMetadata(
        choice="Y",
        name="test_action_1",  # Duplicate name
        description="Test Action 3",
        help_text="Help",
        requires_browser=False,
        required_state="db_only",
        category="test",
    )

    try:
        registry.register(metadata3)
        raise AssertionError("Should have raised ValueError for duplicate name")
    except ValueError as e:
        assert "Duplicate" in str(e)

    return True


def test_backward_compatibility_helpers() -> bool:
    """Test helper functions exported for main.py backward compatibility."""
    # Test get_browser_requirement (replaces _determine_browser_requirement)
    assert not get_browser_requirement("0"), "Action 0 should not require browser"
    assert get_browser_requirement("6"), "Action 6 should require browser"

    # Test get_required_state (replaces _determine_required_state)
    assert get_required_state("0") in ["db_only", "any"], "Action 0 should have db_only state"
    assert get_required_state("6") == "ready", "Action 6 should have ready state"

    # Test get_menu_items (replaces hardcoded print statements)
    menu = get_menu_items()
    assert len(menu) == 11, "Menu should have 11 items"
    assert menu[0][0] == "0", "First item should be choice 0"

    # Test get_action_metadata (new utility)
    metadata = get_action_metadata("6")
    assert metadata is not None, "Action 6 metadata should exist"
    assert metadata.name == "gather_dna_matches", "Name should match"

    return True


def test_all_actions_have_descriptions() -> bool:
    """Test that all actions have proper descriptions."""
    registry = get_action_registry()

    for choice in range(11):
        metadata = registry.get_by_choice(str(choice))
        assert metadata is not None, f"Action {choice} should exist"
        assert len(metadata.description) > 0, f"Action {choice} should have description"
        assert len(metadata.help_text) > 0, f"Action {choice} should have help text"
        assert metadata.category, f"Action {choice} should have category"

    return True


def test_singleton_pattern() -> bool:
    """Test that global registry is a singleton."""
    registry1 = get_action_registry()
    registry2 = get_action_registry()

    # Should be the same object (identity, not just equality)
    assert registry1 is registry2, "Registry should be singleton"

    return True


def test_estimated_durations_reasonable() -> bool:
    """Test that estimated durations are reasonable."""
    registry = get_action_registry()

    for choice in range(11):
        metadata = registry.get_by_choice(str(choice))
        assert metadata is not None, f"Action {choice} should exist"
        assert metadata.estimated_duration_sec >= 0, \
            f"Action {choice} duration should be non-negative"
        # Reasonable upper bound: 2 hours
        assert metadata.estimated_duration_sec <= 7200, \
            f"Action {choice} duration seems unreasonable: {metadata.estimated_duration_sec}s"

    # Action 6 (gathering) should take a while
    action_6 = registry.get_by_choice("6")
    assert action_6 is not None, "Action 6 should exist"
    assert action_6.estimated_duration_sec >= 600, \
        "Action 6 should take at least 10 minutes"

    return True


def module_tests() -> bool:
    """Main test orchestrator."""
    suite = TestSuite("ActionRegistry", "core/action_registry.py")
    suite.start_suite()

    suite.run_test("Registry creation and population", test_registry_creation)
    suite.run_test("O(1) lookup by choice", test_lookup_by_choice)
    suite.run_test("Lookup by function name", test_lookup_by_name)
    suite.run_test("Browser requirement consistency", test_browser_requirement_consistency)
    suite.run_test("Required state consistency", test_required_state_consistency)
    suite.run_test("Workflow action identification", test_workflow_actions)
    suite.run_test("Menu item generation", test_menu_generation)
    suite.run_test("Registry consistency validation", test_consistency_validation)
    suite.run_test("Duplicate registration prevention", test_duplicate_prevention)
    suite.run_test("Backward compatibility helpers", test_backward_compatibility_helpers)
    suite.run_test("All actions documented", test_all_actions_have_descriptions)
    suite.run_test("Singleton pattern enforcement", test_singleton_pattern)
    suite.run_test("Reasonable duration estimates", test_estimated_durations_reasonable)

    return suite.finish_suite()


# Standard test runner pattern
def run_comprehensive_tests():
    return module_tests()


if __name__ == "__main__":
    success = module_tests()
    sys.exit(0 if success else 1)
