"""
Unit tests for core/action_registry.py.

Tests the ActionRegistry implementation including:
- Registration and lookups
- Menu generation
- Consistency validation
- Backward compatibility
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.action_registry import (
    ActionCategory,
    ActionMetadata,
    ActionRegistry,
    ActionRequirement,
    get_action,
    get_action_registry,
    get_browserless_actions,
    get_menu_actions,
    is_browserless_action,
    requires_browser_session,
)
from test_framework import TestSuite


def test_registry_creation() -> bool:
    """Test that global registry is created correctly."""
    registry = get_action_registry()
    assert registry is not None, "Registry should not be None"
    actions = registry.get_all_actions()
    assert len(actions) > 0, f"Expected actions, got {len(actions)}"
    return True


def test_action_retrieval_by_id() -> bool:
    """Test action retrieval by ID."""
    registry = get_action_registry()

    # Test valid IDs (0-10, test, testall, etc.)
    for action_id in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
        metadata = registry.get_action(action_id)
        assert metadata is not None, f"Action {action_id} should exist"
        assert metadata.id == action_id, f"ID should match: {action_id}"

    # Test invalid ID
    metadata = registry.get_action("999")
    assert metadata is None, "Invalid ID should return None"

    return True


def test_browser_requirement_consistency() -> bool:
    """Test that browser requirement is correctly set for all actions."""
    registry = get_action_registry()

    # Database-only actions (no browser needed)
    db_only = ["0", "2", "3", "4"]
    for action_id in db_only:
        metadata = registry.get_action(action_id)
        assert metadata is not None, f"Action {action_id} should exist"
        assert metadata.browser_requirement == ActionRequirement.NONE, \
            f"Action {action_id} should not require browser"
        assert registry.is_browserless_action(action_id), \
            f"Action {action_id} should be browserless"

    # Browser-required actions
    browser_required = ["1", "5", "6", "7", "8", "9", "10"]
    for action_id in browser_required:
        metadata = registry.get_action(action_id)
        assert metadata is not None, f"Action {action_id} should exist"
        assert metadata.browser_requirement != ActionRequirement.NONE, \
            f"Action {action_id} should require browser"
        assert not registry.is_browserless_action(action_id), \
            f"Action {action_id} should require browser"

    return True


def test_category_grouping() -> bool:
    """Test that actions are properly categorized."""
    registry = get_action_registry()

    # Get actions by category
    db_actions = registry.get_actions_by_category(ActionCategory.DATABASE)
    assert len(db_actions) > 0, "Should have database actions"
    assert all(a.category == ActionCategory.DATABASE for a in db_actions), \
        "All database actions should have DATABASE category"

    workflow_actions = registry.get_actions_by_category(ActionCategory.WORKFLOW)
    assert len(workflow_actions) > 0, "Should have workflow actions"
    assert all(a.category == ActionCategory.WORKFLOW for a in workflow_actions), \
        "All workflow actions should have WORKFLOW category"

    browser_actions = registry.get_actions_by_category(ActionCategory.BROWSER)
    assert len(browser_actions) > 0, "Should have browser actions"
    assert all(a.category == ActionCategory.BROWSER for a in browser_actions), \
        "All browser actions should have BROWSER category"

    return True


def test_menu_actions() -> bool:
    """Test menu action generation."""
    registry = get_action_registry()
    menu_actions = registry.get_menu_actions()

    # Should have menu actions (non-test, non-meta)
    assert len(menu_actions) > 0, "Should have menu actions"

    # All should be in sorted order by menu_order
    orders = [a.menu_order for a in menu_actions]
    assert orders == sorted(orders), "Menu actions should be sorted by menu_order"

    # Test convenience function
    menu_actions_2 = get_menu_actions()
    assert len(menu_actions) == len(menu_actions_2), "Convenience function should return same count"

    return True


def test_browserless_actions() -> bool:
    """Test browserless action grouping."""
    registry = get_action_registry()

    # Get browserless actions
    browserless = registry.get_actions_by_browser_requirement(ActionRequirement.NONE)
    assert len(browserless) > 0, "Should have browserless actions"

    # All should have NONE requirement
    assert all(a.browser_requirement == ActionRequirement.NONE for a in browserless), \
        "All browserless actions should have NONE requirement"

    # Test convenience function
    browserless_2 = get_browserless_actions()
    assert len(browserless) == len(browserless_2), "Convenience function should return same count"

    return True


def test_requires_browser_session() -> bool:
    """Test full session requirement checking."""
    # Action 6 should require full session
    assert requires_browser_session("6"), "Action 6 should require full session"

    # Action 0 should not
    assert not requires_browser_session("0"), "Action 0 should not require session"

    return True


def test_test_and_meta_actions() -> bool:
    """Test identification of test and meta actions."""
    registry = get_action_registry()

    # Get test actions
    test_actions = registry.get_test_actions()
    assert len(test_actions) > 0, "Should have test actions"
    assert all(a.is_test_action for a in test_actions), \
        "All test actions should have is_test_action=True"

    # Get meta actions
    meta_actions = registry.get_meta_actions()
    assert len(meta_actions) > 0, "Should have meta actions"
    assert all(a.is_meta_action for a in meta_actions), \
        "All meta actions should have is_meta_action=True"

    return True


def test_confirmation_messages() -> bool:
    """Test confirmation message handling."""
    registry = get_action_registry()

    # Some actions should require confirmation
    action_0 = registry.get_action("0")
    assert action_0 is not None, "Action 0 should exist"
    assert action_0.requires_confirmation, "Action 0 should require confirmation"
    assert registry.requires_confirmation("0"), "Confirmation check should work"
    confirmation = registry.get_confirmation_message("0")
    assert confirmation is not None, "Should have confirmation message"
    assert len(confirmation) > 0, "Confirmation message should not be empty"

    # Some should not require confirmation
    action_5 = registry.get_action("5")
    assert action_5 is not None, "Action 5 should exist"
    assert not action_5.requires_confirmation, "Action 5 should not require confirmation"
    assert not registry.requires_confirmation("5"), "Confirmation check should return False"

    return True


def test_duplicate_prevention() -> bool:
    """Test that registry prevents duplicate registration."""
    registry = ActionRegistry()

    # Register first action
    metadata1 = ActionMetadata(
        id="test_1",
        name="test_action_1",
        description="Test Action 1",
        category=ActionCategory.UTILITY,
        browser_requirement=ActionRequirement.NONE,
    )
    registry.register(metadata1)

    # Try to register duplicate ID - should raise ValueError
    metadata2 = ActionMetadata(
        id="test_1",  # Duplicate ID
        name="test_action_2",
        description="Test Action 2",
        category=ActionCategory.UTILITY,
        browser_requirement=ActionRequirement.NONE,
    )

    try:
        registry.register(metadata2)
        raise AssertionError("Should have raised ValueError for duplicate ID")
    except ValueError as e:
        assert "already registered" in str(e).lower()

    return True


def test_backward_compatibility_helpers() -> bool:
    """Test helper functions for backward compatibility."""
    # Test is_browserless_action (convenience function)
    assert is_browserless_action("0"), "Action 0 should be browserless"
    assert not is_browserless_action("6"), "Action 6 should require browser"

    # Test requires_browser_session (convenience function)
    assert requires_browser_session("6"), "Action 6 should require full session"
    assert not requires_browser_session("0"), "Action 0 should not require session"

    # Test get_action (convenience function)
    action = get_action("6")
    assert action is not None, "Should get action 6"
    assert action.name == "Gather DNA Matches", "Name should match"

    return True


def test_all_actions_have_proper_metadata() -> bool:
    """Test that all actions have proper metadata."""
    registry = get_action_registry()

    for action_id, metadata in registry.get_all_actions().items():
        assert metadata.id == action_id, f"ID mismatch: {action_id}"
        assert len(metadata.name) > 0, f"Action {action_id} should have name"
        assert len(metadata.description) > 0, f"Action {action_id} should have description"
        assert metadata.category is not None, f"Action {action_id} should have category"
        assert metadata.browser_requirement is not None, \
            f"Action {action_id} should have browser_requirement"
        assert metadata.menu_order >= 0, f"Action {action_id} should have non-negative menu_order"

    return True


def test_singleton_pattern() -> bool:
    """Test that global registry is a singleton."""
    registry1 = get_action_registry()
    registry2 = get_action_registry()

    # Should be the same object (identity, not just equality)
    assert registry1 is registry2, "Registry should be singleton"

    return True


def test_action_action_attributes() -> bool:
    """Test specific action attributes."""
    registry = get_action_registry()

    # Action 0: Delete all except first
    action_0 = registry.get_action("0")
    assert action_0 is not None, "Action 0 should exist"
    assert action_0.category == ActionCategory.DATABASE
    assert action_0.browser_requirement == ActionRequirement.NONE
    assert action_0.requires_confirmation
    assert action_0.menu_order == 0, "Action 0 should be first in menu"

    # Action 6: Gather DNA Matches
    action_6 = registry.get_action("6")
    assert action_6 is not None, "Action 6 should exist"
    assert action_6.category == ActionCategory.BROWSER
    assert action_6.browser_requirement == ActionRequirement.FULL_SESSION
    assert action_6.max_args == 1, "Action 6 should allow 1 argument"

    # Action 1: Full workflow
    action_1 = registry.get_action("1")
    assert action_1 is not None, "Action 1 should exist"
    assert action_1.category == ActionCategory.WORKFLOW
    assert action_1.close_session_after
    assert action_1.enable_caching

    return True


def module_tests() -> bool:
    """Main test orchestrator."""
    suite = TestSuite("ActionRegistry", "core/action_registry.py")
    suite.start_suite()

    suite.run_test("Registry creation and population", test_registry_creation)
    suite.run_test("Action retrieval by ID", test_action_retrieval_by_id)
    suite.run_test("Browser requirement consistency", test_browser_requirement_consistency)
    suite.run_test("Category grouping", test_category_grouping)
    suite.run_test("Menu action generation", test_menu_actions)
    suite.run_test("Browserless action grouping", test_browserless_actions)
    suite.run_test("Full session requirement checking", test_requires_browser_session)
    suite.run_test("Test and meta action identification", test_test_and_meta_actions)
    suite.run_test("Confirmation message handling", test_confirmation_messages)
    suite.run_test("Duplicate registration prevention", test_duplicate_prevention)
    suite.run_test("Backward compatibility helpers", test_backward_compatibility_helpers)
    suite.run_test("All actions have proper metadata", test_all_actions_have_proper_metadata)
    suite.run_test("Singleton pattern enforcement", test_singleton_pattern)
    suite.run_test("Specific action attributes", test_action_action_attributes)

    return suite.finish_suite()


# Standard test runner pattern
def run_comprehensive_tests():
    return module_tests()


if __name__ == "__main__":
    success = module_tests()
    sys.exit(0 if success else 1)
