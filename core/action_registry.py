"""
core/action_registry.py - Centralized Action Metadata Registry

Provides a unified registry for all application actions with typed metadata,
eliminating scattered action lists and providing a single source of truth
for action management throughout the application.

Phase 5 Implementation - Centralize Action Metadata (Opportunity #1)
"""

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from standard_imports import setup_module

logger = setup_module(globals(), __name__)


class ActionCategory(Enum):
    """Categories of actions based on their primary function."""
    DATABASE = "database"
    WORKFLOW = "workflow"
    BROWSER = "browser"
    ANALYTICS = "analytics"
    UTILITY = "utility"


class ActionRequirement(Enum):
    """Browser requirements for actions."""
    NONE = "none"           # No browser needed
    DRIVER_ONLY = "driver"  # Browser driver needed, no session
    FULL_SESSION = "session"  # Full browser session needed


@dataclass
class ActionMetadata:
    """
    Comprehensive metadata for application actions.

    This dataclass provides a single source of truth for action properties,
    replacing scattered helper lists throughout the codebase.
    """
    id: str
    name: str
    description: str
    category: ActionCategory
    browser_requirement: ActionRequirement
    function: Optional[Callable[..., object]] = None
    requires_confirmation: bool = False
    confirmation_message: Optional[str] = None
    close_session_after: bool = False
    enable_caching: bool = False
    max_args: int = 0
    menu_order: int = 999  # Lower numbers appear first in menu
    is_test_action: bool = False
    is_meta_action: bool = False


class ActionRegistry:
    """
    Centralized registry for all application actions.

    Provides unified access to action metadata and eliminates the need for
    scattered action lists throughout the codebase.
    """

    def __init__(self) -> None:
        self._actions: dict[str, ActionMetadata] = {}
        self._menu_actions: list[ActionMetadata] = []
        self._test_actions: list[ActionMetadata] = []
        self._meta_actions: list[ActionMetadata] = []
        self._initialized = False

    def register(self, action: ActionMetadata) -> None:
        """Register a new action with the registry."""
        if action.id in self._actions:
            raise ValueError(f"Action with ID '{action.id}' already registered")

        self._actions[action.id] = action

        # Categorize for efficient access
        if action.is_test_action:
            self._test_actions.append(action)
        elif action.is_meta_action:
            self._meta_actions.append(action)
        else:
            self._menu_actions.append(action)

            # Sort by menu_order for consistent display
            self._menu_actions.sort(key=lambda x: x.menu_order)

    def get_action(self, action_id: str) -> Optional[ActionMetadata]:
        """Get action metadata by ID."""
        return self._actions.get(action_id)

    def get_all_actions(self) -> dict[str, ActionMetadata]:
        """Get all registered actions."""
        return self._actions.copy()

    def get_menu_actions(self) -> list[ActionMetadata]:
        """Get all menu actions in display order."""
        return self._menu_actions.copy()

    def get_test_actions(self) -> list[ActionMetadata]:
        """Get all test actions."""
        return self._test_actions.copy()

    def get_meta_actions(self) -> list[ActionMetadata]:
        """Get all meta actions."""
        return self._meta_actions.copy()

    def get_actions_by_category(self, category: ActionCategory) -> list[ActionMetadata]:
        """Get all actions in a specific category."""
        return [action for action in self._actions.values() if action.category == category]

    def get_actions_by_browser_requirement(self, requirement: ActionRequirement) -> list[ActionMetadata]:
        """Get all actions with specific browser requirement."""
        return [action for action in self._actions.values() if action.browser_requirement == requirement]

    def requires_confirmation(self, action_id: str) -> bool:
        """Check if action requires user confirmation."""
        action = self.get_action(action_id)
        return action.requires_confirmation if action else False

    def get_confirmation_message(self, action_id: str) -> Optional[str]:
        """Get confirmation message for action."""
        action = self.get_action(action_id)
        return action.confirmation_message if action else None

    def should_close_session_after(self, action_id: str) -> bool:
        """Check if session should be closed after action completion."""
        action = self.get_action(action_id)
        return action.close_session_after if action else False

    def get_browser_requirement(self, action_id: str) -> ActionRequirement:
        """Get browser requirement for action."""
        action = self.get_action(action_id)
        return action.browser_requirement if action else ActionRequirement.NONE

    def is_browserless_action(self, action_id: str) -> bool:
        """Check if action is browserless (no browser required)."""
        return self.get_browser_requirement(action_id) == ActionRequirement.NONE

    def initialize_default_actions(self) -> None:
        """Initialize the registry with default application actions."""
        if self._initialized:
            logger.warning("Action registry already initialized")
            return

        logger.info("Initializing default action registry...")

        # Import action functions - these will be registered lazily to avoid circular imports
        # The actual function references will be set during registration

        # Database Actions (Browserless)
        self.register(ActionMetadata(
            id="0",
            name="Delete All Except First Person",
            description="Delete all people except the test profile (database-only operation)",
            function=None,  # Will be set during registration
            category=ActionCategory.DATABASE,
            browser_requirement=ActionRequirement.NONE,
            requires_confirmation=True,
            confirmation_message="delete all people except first person (test profile)",
            menu_order=0,
        ))

        self.register(ActionMetadata(
            id="2",
            name="Reset Database",
            description="Completely reset the database by deleting and recreating all tables",
            function=None,
            category=ActionCategory.DATABASE,
            browser_requirement=ActionRequirement.NONE,
            requires_confirmation=True,
            confirmation_message="COMPLETELY reset the database (deletes data)",
            menu_order=2,
        ))

        self.register(ActionMetadata(
            id="3",
            name="Backup Database",
            description="Create a backup of the current database",
            function=None,
            category=ActionCategory.DATABASE,
            browser_requirement=ActionRequirement.NONE,
            menu_order=3,
        ))

        self.register(ActionMetadata(
            id="4",
            name="Restore Database",
            description="Restore database from backup (overwrites current data)",
            function=None,
            category=ActionCategory.DATABASE,
            browser_requirement=ActionRequirement.NONE,
            requires_confirmation=True,
            confirmation_message="restore database from backup (overwrites data)",
            menu_order=4,
        ))

        # Workflow Actions
        self.register(ActionMetadata(
            id="1",
            name="Run Full Workflow",
            description="Run complete workflow: Action 7 → Action 9 → Action 8",
            function=None,
            category=ActionCategory.WORKFLOW,
            browser_requirement=ActionRequirement.FULL_SESSION,
            close_session_after=True,
            enable_caching=True,
            menu_order=1,
        ))

        # Browser Actions
        self.register(ActionMetadata(
            id="5",
            name="Check Login Status",
            description="Check current login status and display all identifiers",
            function=None,
            category=ActionCategory.BROWSER,
            browser_requirement=ActionRequirement.DRIVER_ONLY,
            menu_order=5,
        ))

        self.register(ActionMetadata(
            id="6",
            name="Gather DNA Matches",
            description="Automated DNA match harvesting from Ancestry.com",
            function=None,
            category=ActionCategory.BROWSER,
            browser_requirement=ActionRequirement.FULL_SESSION,
            max_args=1,  # Optional start page
            menu_order=6,
        ))

        self.register(ActionMetadata(
            id="7",
            name="Search Inbox",
            description="Process and analyze inbox messages with AI classification",
            function=None,
            category=ActionCategory.BROWSER,
            browser_requirement=ActionRequirement.FULL_SESSION,
            menu_order=7,
        ))

        self.register(ActionMetadata(
            id="8",
            name="Send Messages",
            description="Send AI-powered messages to DNA matches with context awareness",
            function=None,
            category=ActionCategory.BROWSER,
            browser_requirement=ActionRequirement.FULL_SESSION,
            menu_order=8,
        ))

        self.register(ActionMetadata(
            id="9",
            name="Process Productive Messages",
            description="Manage ongoing productive conversations with automated follow-ups",
            function=None,
            category=ActionCategory.BROWSER,
            browser_requirement=ActionRequirement.FULL_SESSION,
            enable_caching=True,
            menu_order=9,
        ))

        self.register(ActionMetadata(
            id="10",
            name="Compare: GEDCOM vs API",
            description="Side-by-side comparison of GEDCOM and API search results",
            function=None,
            category=ActionCategory.BROWSER,
            browser_requirement=ActionRequirement.FULL_SESSION,
            enable_caching=True,
            menu_order=10,
        ))

        # Test Actions
        self.register(ActionMetadata(
            id="test",
            name="Run Main.py Internal Tests",
            description="Run comprehensive tests for main.py functionality",
            function=None,
            category=ActionCategory.UTILITY,
            browser_requirement=ActionRequirement.NONE,
            is_test_action=True,
        ))

        self.register(ActionMetadata(
            id="testall",
            name="Run All Module Tests",
            description="Run all module tests across the entire codebase",
            function=None,
            category=ActionCategory.UTILITY,
            browser_requirement=ActionRequirement.NONE,
            is_test_action=True,
        ))

        # Meta Actions
        self.register(ActionMetadata(
            id="analytics",
            name="View Conversation Analytics Dashboard",
            description="Display conversation analytics dashboard with insights",
            function=None,
            category=ActionCategory.ANALYTICS,
            browser_requirement=ActionRequirement.NONE,
            is_meta_action=True,
        ))

        self.register(ActionMetadata(
            id="s",
            name="Show Cache Statistics",
            description="Display comprehensive cache statistics from all subsystems",
            function=None,
            category=ActionCategory.ANALYTICS,
            browser_requirement=ActionRequirement.NONE,
            is_meta_action=True,
        ))

        self.register(ActionMetadata(
            id="t",
            name="Toggle Console Log Level",
            description="Toggle between INFO and DEBUG log levels",
            function=None,
            category=ActionCategory.UTILITY,
            browser_requirement=ActionRequirement.NONE,
            is_meta_action=True,
        ))

        self.register(ActionMetadata(
            id="c",
            name="Clear Screen",
            description="Clear the console screen",
            function=None,
            category=ActionCategory.UTILITY,
            browser_requirement=ActionRequirement.NONE,
            is_meta_action=True,
        ))

        self.register(ActionMetadata(
            id="q",
            name="Exit",
            description="Exit the application",
            function=None,
            category=ActionCategory.UTILITY,
            browser_requirement=ActionRequirement.NONE,
            is_meta_action=True,
        ))

        self._initialized = True
        logger.info(f"Action registry initialized with {len(self._actions)} actions")


# Global registry instance
_action_registry: Optional[ActionRegistry] = None


def get_action_registry() -> ActionRegistry:
    """Get the global action registry instance."""
    # Avoid global statement by using globals() dict
    if globals()["_action_registry"] is None:
        globals()["_action_registry"] = ActionRegistry()
        globals()["_action_registry"].initialize_default_actions()

    return globals()["_action_registry"]


def register_action(action: ActionMetadata) -> None:
    """Register a new action with the global registry."""
    registry = get_action_registry()
    registry.register(action)


def get_action(action_id: str) -> Optional[ActionMetadata]:
    """Get action metadata by ID from global registry."""
    registry = get_action_registry()
    return registry.get_action(action_id)


# Convenience functions for common operations
def is_browserless_action(action_id: str) -> bool:
    """Check if action is browserless."""
    registry = get_action_registry()
    return registry.is_browserless_action(action_id)


def requires_browser_session(action_id: str) -> bool:
    """Check if action requires full browser session."""
    registry = get_action_registry()
    requirement = registry.get_browser_requirement(action_id)
    return requirement == ActionRequirement.FULL_SESSION


def get_menu_actions() -> list[ActionMetadata]:
    """Get all menu actions in display order."""
    registry = get_action_registry()
    return registry.get_menu_actions()


def get_browserless_actions() -> list[ActionMetadata]:
    """Get all browserless actions."""
    registry = get_action_registry()
    return registry.get_actions_by_browser_requirement(ActionRequirement.NONE)


# ==============================================
# Module Tests
# ==============================================


def _test_registry_creation() -> bool:
    """Test that global registry is created correctly."""
    registry = get_action_registry()
    assert registry is not None, "Registry should not be None"
    actions = registry.get_all_actions()
    assert actions, "Expected actions to be registered"
    return True


def _test_action_retrieval_by_id() -> bool:
    """Test action retrieval by ID."""
    registry = get_action_registry()

    for action_id in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
        metadata = registry.get_action(action_id)
        assert metadata is not None, f"Action {action_id} should exist"
        assert metadata.id == action_id, f"ID should match: {action_id}"

    assert registry.get_action("999") is None, "Invalid ID should return None"
    return True


def _test_browser_requirement_consistency() -> bool:
    """Test that browser requirement is correctly set for all actions."""
    registry = get_action_registry()

    db_only = ["0", "2", "3", "4"]
    for action_id in db_only:
        metadata = registry.get_action(action_id)
        assert metadata is not None, f"Action {action_id} should exist"
        assert metadata.browser_requirement == ActionRequirement.NONE, "Database actions should be browserless"
        assert registry.is_browserless_action(action_id), "Database actions should report browserless"

    browser_required = ["1", "5", "6", "7", "8", "9", "10"]
    for action_id in browser_required:
        metadata = registry.get_action(action_id)
        assert metadata is not None, f"Action {action_id} should exist"
        assert metadata.browser_requirement != ActionRequirement.NONE, "Should require browser"
        assert not registry.is_browserless_action(action_id), "Should not report browserless"

    return True


def _test_category_grouping() -> bool:
    """Test that actions are properly categorized."""
    registry = get_action_registry()

    db_actions = registry.get_actions_by_category(ActionCategory.DATABASE)
    assert db_actions and all(a.category is ActionCategory.DATABASE for a in db_actions)

    workflow_actions = registry.get_actions_by_category(ActionCategory.WORKFLOW)
    assert workflow_actions and all(a.category is ActionCategory.WORKFLOW for a in workflow_actions)

    browser_actions = registry.get_actions_by_category(ActionCategory.BROWSER)
    assert browser_actions and all(a.category is ActionCategory.BROWSER for a in browser_actions)

    return True


def _test_menu_actions() -> bool:
    """Test menu action generation order."""
    registry = get_action_registry()
    menu_actions = registry.get_menu_actions()
    assert menu_actions, "Menu actions should not be empty"
    orders = [a.menu_order for a in menu_actions]
    assert orders == sorted(orders), "Menu order should be sorted"

    assert len(menu_actions) == len(get_menu_actions()), "Convenience function should match registry"
    return True


def _test_browserless_actions() -> bool:
    """Test browserless action grouping."""
    registry = get_action_registry()
    browserless = registry.get_actions_by_browser_requirement(ActionRequirement.NONE)
    assert browserless, "Expected browserless actions"
    assert all(a.browser_requirement == ActionRequirement.NONE for a in browserless)
    assert len(browserless) == len(get_browserless_actions()), "Convenience function should match registry"
    return True


def _test_requires_browser_session() -> bool:
    """Test full session requirement helper."""
    assert requires_browser_session("6"), "Action 6 should require full session"
    assert not requires_browser_session("0"), "Action 0 should not require session"
    return True


def _test_test_and_meta_actions() -> bool:
    """Test identification of test and meta actions."""
    registry = get_action_registry()

    test_actions = registry.get_test_actions()
    assert test_actions and all(a.is_test_action for a in test_actions)

    meta_actions = registry.get_meta_actions()
    assert meta_actions and all(a.is_meta_action for a in meta_actions)
    return True


def _test_confirmation_messages() -> bool:
    """Test confirmation message handling."""
    registry = get_action_registry()

    action_0 = registry.get_action("0")
    assert action_0 is not None and action_0.requires_confirmation
    assert registry.requires_confirmation("0")
    message = registry.get_confirmation_message("0")
    assert message, "Expected confirmation message"

    action_5 = registry.get_action("5")
    assert action_5 is not None and not action_5.requires_confirmation
    assert not registry.requires_confirmation("5")
    return True


def _test_duplicate_prevention() -> bool:
    """Test that duplicate registrations are blocked."""
    registry = ActionRegistry()
    metadata1 = ActionMetadata(
        id="test_1",
        name="test_action_1",
        description="Test Action 1",
        category=ActionCategory.UTILITY,
        browser_requirement=ActionRequirement.NONE,
    )
    registry.register(metadata1)

    metadata2 = ActionMetadata(
        id="test_1",
        name="test_action_2",
        description="Test Action 2",
        category=ActionCategory.UTILITY,
        browser_requirement=ActionRequirement.NONE,
    )

    try:
        registry.register(metadata2)
    except ValueError as exc:
        assert "already registered" in str(exc).lower()
        return True

    raise AssertionError("Expected ValueError for duplicate action ID")


def _test_backward_compatibility_helpers() -> bool:
    """Test convenience helper consistency."""
    assert is_browserless_action("0")
    assert not is_browserless_action("6")
    assert requires_browser_session("6")
    assert not requires_browser_session("0")
    action = get_action("6")
    assert action is not None and action.name == "Gather DNA Matches"
    return True


def _test_action_metadata_integrity() -> bool:
    """Test that all actions expose consistent metadata."""
    registry = get_action_registry()
    for action_id, metadata in registry.get_all_actions().items():
        assert metadata.id == action_id, f"ID mismatch for {action_id}"
        assert metadata.name, f"Action {action_id} missing name"
        assert metadata.description, f"Action {action_id} missing description"
        assert metadata.category is not None, f"Action {action_id} missing category"
        assert metadata.browser_requirement is not None, f"Action {action_id} missing requirement"
        assert metadata.menu_order >= 0, f"Action {action_id} menu order should be non-negative"
    return True


def _test_singleton_pattern() -> bool:
    """Test global registry singleton behavior."""
    registry1 = get_action_registry()
    registry2 = get_action_registry()
    assert registry1 is registry2, "Expected singleton instance"
    return True


def _test_specific_action_attributes() -> bool:
    """Spot-check key action metadata entries."""
    registry = get_action_registry()

    action_0 = registry.get_action("0")
    assert action_0 is not None
    assert action_0.category is ActionCategory.DATABASE
    assert action_0.browser_requirement is ActionRequirement.NONE
    assert action_0.requires_confirmation
    assert action_0.menu_order == 0

    action_6 = registry.get_action("6")
    assert action_6 is not None
    assert action_6.category is ActionCategory.BROWSER
    assert action_6.browser_requirement is ActionRequirement.FULL_SESSION
    assert action_6.max_args == 1

    action_1 = registry.get_action("1")
    assert action_1 is not None
    assert action_1.category is ActionCategory.WORKFLOW
    assert action_1.close_session_after
    assert action_1.enable_caching

    return True


def action_registry_module_tests() -> bool:
    """Run the action registry test suite."""
    from test_framework import TestSuite

    suite = TestSuite("ActionRegistry", "core/action_registry.py")
    suite.start_suite()

    suite.run_test("Registry creation and population", _test_registry_creation)
    suite.run_test("Action retrieval by ID", _test_action_retrieval_by_id)
    suite.run_test("Browser requirement consistency", _test_browser_requirement_consistency)
    suite.run_test("Category grouping", _test_category_grouping)
    suite.run_test("Menu action generation", _test_menu_actions)
    suite.run_test("Browserless action grouping", _test_browserless_actions)
    suite.run_test("Full session requirement checking", _test_requires_browser_session)
    suite.run_test("Test and meta action identification", _test_test_and_meta_actions)
    suite.run_test("Confirmation message handling", _test_confirmation_messages)
    suite.run_test("Duplicate registration prevention", _test_duplicate_prevention)
    suite.run_test("Backward compatibility helpers", _test_backward_compatibility_helpers)
    suite.run_test("Action metadata integrity", _test_action_metadata_integrity)
    suite.run_test("Singleton pattern enforcement", _test_singleton_pattern)
    suite.run_test("Specific action attributes", _test_specific_action_attributes)

    return suite.finish_suite()


try:
    from test_utilities import create_standard_test_runner
    run_comprehensive_tests = create_standard_test_runner(action_registry_module_tests)
except ImportError:  # pragma: no cover - fallback for minimal test environments
    def run_comprehensive_tests() -> bool:
        return action_registry_module_tests()


if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
