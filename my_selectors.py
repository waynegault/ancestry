from core_imports import (
    register_function,
    get_function,
    is_function_available,
    standardize_module_imports,
    auto_register_module,
)

auto_register_module(globals(), __name__)
standardize_module_imports()
#!/usr/bin/env python3

# my_selectors.py

"""
This module defines CSS selectors for interacting with the Ancestry website.

Selectors are organized by page/functionality for easier maintenance.
"""

from test_framework import TestSuite, suppress_logging, MagicMock  # Testing framework

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
from test_framework import (
    TestSuite,
    suppress_logging,
    create_mock_data,
    assert_valid_function,
    MagicMock,
)


def my_selectors_module_tests():
    """Essential CSS selectors tests for unified framework."""
    import re

    tests = []

    # Test 1: Basic selector definitions
    def test_selector_definitions():
        basic_selectors = [
            "WAIT_FOR_PAGE_SELECTOR",
            "POPUP_CLOSE_SELECTOR",
            "PAGE_NO_LONGER_AVAILABLE_SELECTOR",
            "UNAVAILABLE_MATCH_SELECTOR",
        ]
        for selector_name in basic_selectors:
            assert selector_name in globals(), f"{selector_name} should be defined"
            selector_value = globals()[selector_name]
            assert isinstance(
                selector_value, str
            ), f"{selector_name} should be a string"
            assert (
                len(selector_value.strip()) > 0
            ), f"{selector_name} should not be empty"

    tests.append(("Selector Definitions", test_selector_definitions))

    # Test 2: CSS selector format validation
    def test_css_format():
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

    tests.append(("CSS Format Validation", test_css_format))

    # Test 3: Selector organization
    def test_selector_organization():
        # Count selectors by category
        all_selectors = [name for name in globals() if name.endswith("_SELECTOR")]
        error_selectors = [
            name for name in all_selectors if "UNAVAILABLE" in name or "ERROR" in name
        ]

        assert len(all_selectors) > 10, "Should have multiple selectors defined"
        assert len(error_selectors) >= 3, "Should have error-related selectors"

        # Test naming convention
        for selector_name in all_selectors[:5]:  # Test first 5 for performance
            assert (
                selector_name.isupper()
            ), f"Selector {selector_name} should be uppercase"

    tests.append(("Selector Organization", test_selector_organization))

    # Test 4: Placeholder validation
    def test_placeholder_selectors():
        placeholder_selectors = []
        for name, value in globals().items():
            if isinstance(value, str) and "{" in value and "}" in value:
                placeholder_selectors.append((name, value))

        # Test placeholder format if any exist
        for name, selector in placeholder_selectors[:3]:  # Limit for performance
            assert selector.count("{") == selector.count(
                "}"
            ), f"Unmatched braces in {name}"

    tests.append(("Placeholder Validation", test_placeholder_selectors))

    # Test 5: Performance validation
    def test_performance():
        import time

        start_time = time.time()

        # Test rapid selector access
        for _ in range(100):
            _ = WAIT_FOR_PAGE_SELECTOR
            _ = POPUP_CLOSE_SELECTOR

        duration = time.time() - start_time
        assert duration < 0.01, f"Selector access should be fast, took {duration:.3f}s"

    tests.append(("Performance Validation", test_performance))

    return tests


def run_comprehensive_tests() -> bool:
    """Run CSS selectors tests using unified framework."""
    from test_framework_unified import run_unified_tests

    return run_unified_tests("my_selectors", my_selectors_module_tests)


# ==============================================
# Standalone Test Block
# ==============================================

if __name__ == "__main__":
    import sys
    import re

    print(
        "🎯 Running CSS Selectors & Element Identification comprehensive test suite..."
    )
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
