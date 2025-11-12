"""
core/action_registry.py - Centralized Action Metadata Registry

Provides a unified registry for all application actions with typed metadata,
eliminating scattered action lists and providing a single source of truth
for action management throughout the application.

Phase 5 Implementation - Centralize Action Metadata (Opportunity #1)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Protocol

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
    function: Callable
    category: ActionCategory
    browser_requirement: ActionRequirement
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

    def __init__(self):
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


# Module test function
def test_action_registry() -> bool:
    """Test the action registry functionality."""
    try:
        logger.info("Testing action registry...")

        # Get registry
        registry = get_action_registry()

        # Test basic functionality
        assert len(registry.get_all_actions()) > 0, "Registry should have actions"

        # Test action retrieval
        action_6 = registry.get_action("6")
        assert action_6 is not None, "Should be able to retrieve action 6"
        assert action_6.name == "Gather DNA Matches", "Action 6 should have correct name"
        assert action_6.category == ActionCategory.BROWSER, "Action 6 should be browser category"

        # Test browser requirement detection
        assert registry.is_browserless_action("0"), "Action 0 should be browserless"
        assert not registry.is_browserless_action("6"), "Action 6 should require browser"

        # Test menu actions
        menu_actions = registry.get_menu_actions()
        assert len(menu_actions) > 0, "Should have menu actions"

        # Test browserless actions
        browserless_actions = registry.get_browserless_actions()
        assert len(browserless_actions) > 0, "Should have browserless actions"

        logger.info("✅ Action registry tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ Action registry tests failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    success = test_action_registry()
    sys.exit(0 if success else 1)
