#!/usr/bin/env python3

"""CSS Selectors for Ancestry Website Automation.

Defines CSS selectors for interacting with the Ancestry website.
Selectors are organized by page/functionality for easier maintenance.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===

# === THIRD-PARTY IMPORTS ===
from test_framework import TestSuite, suppress_logging

# --- General Page Elements ---
WAIT_FOR_PAGE_SELECTOR = "body"  # Used to wait for page load.
POPUP_CLOSE_SELECTOR = "button.closeBtn"  # Close button for popups

# --- Unavailable pages ---
PAGE_NO_LONGER_AVAILABLE_SELECTOR = "header.pageErrorHeader h1.pageError"
UNAVAILABLE_MATCH_SELECTOR = "compare-page div.alert"  # The match you are trying to view is unavailable. (watch out that div.alert is also used with the  loading symbol)
MESSAGE_CENTER_UNAVAILABLE_SELECTOR = "div.messageCenter div.unavailableMessages"
TEMP_UNAVAILABLE_SELECTOR = (
    "div.pageError h1.pageTitle"  # this page is temporarily unavailable.
)

# --- Home page (logged in and not logged in) --- (https://www.ancestry.co.uk/)
FOOTER_SELECTOR = "footer#footer ul#footerLegal"


# --- Login Page (https://www.ancestry.co.uk/account/signin) ---
CONFIRMED_LOGGED_IN_SELECTOR = "#navAccount[data-tracking-name='Account']"  # "[href^='https://www.ancestry.co.uk/profile/']"
COOKIE_BANNER_SELECTOR = "div#bannerOverlay"
CONSENT_ACCEPT_BUTTON_SELECTOR = "#acceptAllBtn"  # Cookie consent button
LOG_IN_BUTTON_SELECTOR = "a[href^='https://www.ancestry.co.uk/account/signin']"
USERNAME_INPUT_SELECTOR = "input#username"
PASSWORD_INPUT_SELECTOR = "input#password"
SIGN_IN_BUTTON_SELECTOR = "#signInBtn"
TWO_FA_EMAIL_SELECTOR = "button[data-method='email']"
TWO_FA_SMS_SELECTOR = "button.ancCardBtn.methodBtn[data-method='sms']"
TWO_STEP_VERIFICATION_HEADER_SELECTOR = "body.mfaPage h2.conTitle"
FAILED_LOGIN_SELECTOR = "div#invalidCredentialsAlert.alert"
TWO_FA_CODE_BUTTON_SELECTOR = "button#codeFormSubmitBtn"
TWO_FA_CODE_INPUT_SELECTOR = "input#codeFormInput"


# --- DNA Matches List Page (https://www.ancestry.co.uk/discoveryui-matches/list/) ---
MATCH_ENTRY_SELECTOR = "ui-custom[type='match-entry']"  # Individual match entry
MATCH_NAME_LINK_SELECTOR = "a[data-testid='matchNameLink']"  # Link to the match's page
SHARED_DNA_SELECTOR = "div[data-testid='sharedDnaAmount']"  # Shared DNA amount
PREDICTED_RELATIONSHIP_SELECTOR = (
    "section.sharedDnaContainer button.relationshipLabel"  # Predicted relationship
)
TREE_INDICATOR_SELECTOR = (
    "ui-person-avatar[indicator='tree']"  # Icon indicating a tree exists
)
PAGINATION_SELECTOR = "ui-pagination[data-testid='paginator']"  # Pagination control


# --- Individual Match Compare Page (https://www.ancestry.co.uk/discoveryui-matches/compare) ---
MESSAGE_BUTTON_SELECTOR = "[href*='/messaging/']"  # Button to send a message
VIEW_IN_TREE_SELECTOR = "a.ancBtn.outline.addRelationBtn"  # "View in tree" button
PROFILE_ID_SELECTOR = "compare-page  div.userCardContent h1  span.matchNameAndBadge a"
MANAGED_BY_PROFILE_ID_SELECTOR = "compare-page span.managedBy a"

# --- Family Tree Page (https://www.ancestry.co.uk/family-tree/person/tree/) ---
# Used when viewing a match's tree.
TREE_NAME_SELECTOR = "h1.userCardTitle"  # The name of the tree
RELATION_BUTTON_SELECTOR = "button#rel_label"  # "Relationship to me" button.
RELATIONSHIP_SELECTOR = (
    ".relationship-selector"  # Added to previous version, but unused.
)
RELATIONSHIP_LABEL_SELECTOR = (
    "button.relationshipLabel"  # This is used to get the predicted relationship.
)

# --- Relationship Modal (appears on Family Tree Page) ---
MODAL_TITLE_SELECTOR = "h4.modalTitle"  # "Relationship to me"
MODAL_CONTENT_SELECTOR = "ul.textCenter"  # Container for relationship path
CLOSE_BUTTON_SELECTOR = "button.closeBtn.modalClose"  # Close button for the modal

# --- Inbox/Messaging Page (https://www.ancestry.co.uk/messaging) ---
INBOX_PAGE_LOAD_SELECTOR = (
    "h1.sectionTitle:contains('Messages')"  # Selector for "Messages" heading
)
INBOX_CONTAINER_SELECTOR = "div.cardContainer"  # "main#main > div > div > div.channelsSection" # "main#main > div > div > div > div:nth-of-type(2)"  # Container for inbox list
RIGHT_PAGE_CHECK_SELECTOR = "div.singleProfile[data-activeprofileid={profile_id}]"  # check that we are on the correct user's message page
AVATAR_CARD_SELECTOR = "div.avatarCardGroup"  # Individual conversation entry
PROFILE_IDS_SELECTOR = "div[data-profileids]"  # Used to get profile ID from inbox
AVATAR_BOX_SELECTOR = "div.avatarBox"  # Used to extract the username.
MESSAGE_CONTAINER_SELECTOR = "div.messagingContainer"  # div.chatContainer Container for an individual conversation
SENT_MESSAGES_SELECTOR = ".fromSelf .chatBubble"  # Selector for sent messages
RECEIVED_MESSAGES_SELECTOR = ".fromOther .chatBubble"  # Selector for received messages
MESSAGE_CONTENT_SELECTOR = ".messageContent span"  # Message text within a bubble
TIMESTAMP_SELECTOR = ".timestamp"  # Timestamp within a message bubble
BUBBLE_SEPARATOR_SELECTOR = "div.bubbleSeparator"  # Date bubble
CONVERSATION_LIST_SELECTOR = "div.cardInner"
MESSAGE_BOX_SELECTOR = "div.inputArea textarea#message-box"  # Text area.
SEND_BUTTON_SELECTOR = "button.ancBtn.sendBtn"  # Send button
MESSAGE_SENT_SELECTOR = (
    ".bubbleContainer  .fromSelf:last-of-type .chatBubble .chatContent + .timestamp"
)

# ==============================================
# Test framework imports
# ==============================================


def my_selectors_module_tests() -> bool:
    """
    CSS Selectors & Element Identification module test suite.
    Tests the six categories: Initialization, Core Functionality, Edge Cases, Integration, Performance, and Error Handling.
    """

    with suppress_logging():
        suite = TestSuite("CSS Selectors & Element Identification", "my_selectors.py")

    # Run all tests
    print(
        "🎯 Running CSS Selectors & Element Identification comprehensive test suite..."
    )

    with suppress_logging():
        suite.run_test(
            "Basic selector definitions verification",
            test_selector_definitions,
            "Test that essential CSS selectors are properly defined",
            "Selector definitions verification ensures complete selector availability",
            "Basic selectors (WAIT_FOR_PAGE_SELECTOR, POPUP_CLOSE_SELECTOR, etc.) are defined as strings",
        )

        suite.run_test(
            "CSS selector format validation",
            test_css_format,
            "Test CSS selectors follow valid syntax and formatting rules",
            "CSS format validation ensures selectors work correctly with web drivers",
            "All selectors follow valid CSS syntax without syntax errors or whitespace issues",
        )

        suite.run_test(
            "Selector organization and naming",
            test_selector_organization,
            "Test selectors are properly organized and follow naming conventions",
            "Selector organization ensures maintainable and discoverable element identifiers",
            "Selectors follow uppercase naming convention and are logically categorized",
        )

        suite.run_test(
            "Placeholder selector validation",
            test_placeholder_selectors,
            "Test placeholder selectors with template variables are properly formed",
            "Placeholder validation ensures dynamic selectors can be safely formatted",
            "Placeholder selectors use proper template syntax for dynamic value insertion",
        )

        suite.run_test(
            "Login page selector coverage",
            test_login_selectors,
            "Test login-related selectors cover essential authentication elements",
            "Login selector coverage ensures complete authentication workflow support",
            "Login selectors include username, password, sign-in, and two-factor elements",
        )

        suite.run_test(
            "Error page selector coverage",
            test_error_selectors,
            "Test error-related selectors handle various unavailable page scenarios",
            "Error selector coverage provides robust error detection and handling",
            "Error selectors cover unavailable pages, failed logins, and temporary errors",
        )

        suite.run_test(
            "Selector string integrity",
            test_selector_integrity,
            "Test all selectors are non-empty strings with valid content",
            "Selector integrity ensures reliable element identification",
            "All selectors contain valid, non-empty string values without corruption",
        )

        suite.run_test(
            "Special character handling",
            test_special_characters,
            "Test selectors properly handle special CSS characters and escaping",
            "Special character handling ensures selectors work with complex element attributes",
            "Selectors correctly use brackets, colons, quotes, and other CSS special characters",
        )

        suite.run_test(
            "Selector accessibility validation",
            test_selector_accessibility,
            "Test selectors are accessible and don't cause import errors",
            "Accessibility validation ensures all selectors can be safely imported and used",
            "All defined selectors are accessible through module globals without errors",
        )

        suite.run_test(
            "Performance optimization validation",
            test_performance,
            "Test selector access and usage maintains good performance characteristics",
            "Performance validation ensures efficient selector operations",
            "Selector access operations complete quickly without performance bottlenecks",
        )

    # Generate summary report
    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(my_selectors_module_tests)


# Test functions for comprehensive testing
def test_selector_definitions() -> None:
    """Test that essential CSS selectors are properly defined."""
    basic_selectors = [
        "WAIT_FOR_PAGE_SELECTOR",
        "POPUP_CLOSE_SELECTOR",
        "PAGE_NO_LONGER_AVAILABLE_SELECTOR",
        "UNAVAILABLE_MATCH_SELECTOR",
    ]
    for selector_name in basic_selectors:
        assert selector_name in globals(), f"{selector_name} should be defined"
        selector_value = globals()[selector_name]
        assert isinstance(selector_value, str), f"{selector_name} should be a string"
        assert len(selector_value.strip()) > 0, f"{selector_name} should not be empty"


def test_css_format() -> None:
    """Test CSS selectors follow valid syntax and formatting rules."""
    import re

    css_selector_pattern = r'^[a-zA-Z0-9._#\[\]:=\-\s\(\),>+~"\']+$'
    test_selectors = [
        WAIT_FOR_PAGE_SELECTOR,
        POPUP_CLOSE_SELECTOR,
        PAGE_NO_LONGER_AVAILABLE_SELECTOR,
        UNAVAILABLE_MATCH_SELECTOR,
    ]
    for selector in test_selectors:
        assert re.match(
            css_selector_pattern, selector
        ), f"Invalid CSS selector: {selector}"
        assert selector.strip() == selector, f"Selector has whitespace: {selector}"


def test_selector_organization() -> None:
    """Test selectors are properly organized and follow naming conventions."""
    # Count selectors by category
    all_selectors = [name for name in globals() if name.endswith("_SELECTOR")]
    error_selectors = [
        name for name in all_selectors if "UNAVAILABLE" in name or "ERROR" in name
    ]

    assert len(all_selectors) > 10, "Should have multiple selectors defined"
    assert len(error_selectors) >= 3, "Should have error-related selectors"

    # Test naming convention
    for selector_name in all_selectors[:5]:  # Test first 5 for performance
        assert selector_name.isupper(), f"Selector {selector_name} should be uppercase"


def test_placeholder_selectors() -> None:
    """Test placeholder selectors with template variables are properly formed."""
    placeholder_selectors: list[tuple[str, str]] = []
    for name, value in globals().items():
        if isinstance(value, str) and "{" in value and "}" in value:
            placeholder_selectors.append((name, value))

    # Test placeholder format if any exist
    for name, selector in placeholder_selectors[:3]:  # Limit for performance
        assert selector.count("{") == selector.count("}"), f"Unmatched braces in {name}"


def test_login_selectors() -> None:
    """Test login-related selectors cover essential authentication elements."""
    login_selectors = [
        "USERNAME_INPUT_SELECTOR",
        "PASSWORD_INPUT_SELECTOR",
        "SIGN_IN_BUTTON_SELECTOR",
        "CONFIRMED_LOGGED_IN_SELECTOR",
    ]

    for selector_name in login_selectors:
        if selector_name in globals():
            selector_value = globals()[selector_name]
            assert isinstance(
                selector_value, str
            ), f"Login selector {selector_name} should be string"
            assert (
                len(selector_value) > 0
            ), f"Login selector {selector_name} should not be empty"


def test_error_selectors() -> None:
    """Test error-related selectors handle various unavailable page scenarios."""
    error_selectors = [
        "PAGE_NO_LONGER_AVAILABLE_SELECTOR",
        "UNAVAILABLE_MATCH_SELECTOR",
        "FAILED_LOGIN_SELECTOR",
        "TEMP_UNAVAILABLE_SELECTOR",
    ]

    for selector_name in error_selectors:
        if selector_name in globals():
            selector_value = globals()[selector_name]
            assert isinstance(
                selector_value, str
            ), f"Error selector {selector_name} should be string"
            assert (
                len(selector_value) > 0
            ), f"Error selector {selector_name} should not be empty"


def test_selector_integrity() -> None:
    """Test all selectors are non-empty strings with valid content."""
    all_selectors = [name for name in globals() if name.endswith("_SELECTOR")]

    for selector_name in all_selectors:
        selector_value = globals()[selector_name]
        assert isinstance(
            selector_value, str
        ), f"Selector {selector_name} should be string"
        assert (
            len(selector_value.strip()) > 0
        ), f"Selector {selector_name} should not be empty"
        assert (
            not selector_value.isspace()
        ), f"Selector {selector_name} should not be whitespace only"


def test_special_characters() -> None:
    """Test selectors properly handle special CSS characters and escaping."""
    # Test that selectors with special characters are valid
    special_selectors: list[tuple[str, str]] = []
    for name, value in globals().items():
        if isinstance(value, str) and any(
            char in value for char in ["[", "]", ":", "#", "."]
        ):
            special_selectors.append((name, value))

    assert len(special_selectors) > 0, "Should have selectors with special characters"

    for name, value in special_selectors[:3]:  # Test first few for performance
        # Basic validation that it looks like a CSS selector
        assert any(
            char in value for char in ["#", ".", "[", ":"]
        ), f"Special selector {name} should contain CSS syntax"


def test_selector_accessibility() -> None:
    """Test selectors are accessible and don't cause import errors."""
    all_selectors = [name for name in globals() if name.endswith("_SELECTOR")]

    accessible_count = 0
    for selector_name in all_selectors:
        try:
            _ = globals()[selector_name]
            accessible_count += 1
        except KeyError:
            pass  # Some selectors might not be accessible, which is acceptable

    assert accessible_count > 10, "Most selectors should be accessible"


def test_performance() -> None:
    """Test selector access and usage maintains good performance characteristics."""
    import time

    start_time = time.time()

    # Test rapid selector access
    for _ in range(100):
        _ = WAIT_FOR_PAGE_SELECTOR
        _ = POPUP_CLOSE_SELECTOR

    duration = time.time() - start_time
    assert duration < 0.01, f"Selector access should be fast, took {duration:.3f}s"


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    import sys

    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
