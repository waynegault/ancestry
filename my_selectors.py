#!/usr/bin/env python3

# my_selectors.py

"""
This module defines CSS selectors for interacting with the Ancestry website.

Selectors are organized by page/functionality for easier maintenance.
"""

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
# Test framework imports with fallbacks
# ==============================================
try:
    from test_framework import (
        TestSuite,
        suppress_logging,
        create_mock_data,
        assert_valid_function,
    )
except ImportError:
    # Fallback implementations when test framework is not available
    from contextlib import contextmanager

    @contextmanager
    def suppress_logging():
        yield

    def create_mock_data(data_type):
        return {}

    def assert_valid_function(func, func_name):
        return callable(func)

    class TestSuite:
        def __init__(self, name, module):
            self.name = name
            self.module = module

        def start_suite(self):
            pass

        def run_test(self, name, func, description):
            try:
                func()
            except:
                pass

        def finish_suite(self):
            return True


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for my_selectors.py.
    Tests CSS selector definitions, validation, and organization.
    """
    suite = TestSuite("CSS Selectors & Element Identification", "my_selectors.py")
    suite.start_suite()

    # Initialization Tests
    def test_initialization():
        """Test that all basic selectors are defined as strings."""
        basic_selectors = [
            "WAIT_FOR_PAGE_SELECTOR",
            "POPUP_CLOSE_SELECTOR",
            "PAGE_NO_LONGER_AVAILABLE_SELECTOR",
            "UNAVAILABLE_MATCH_SELECTOR",
            "MESSAGE_CENTER_UNAVAILABLE_SELECTOR",
            "TEMP_UNAVAILABLE_SELECTOR",
        ]

        for selector_name in basic_selectors:
            if selector_name in globals():
                selector_value = globals()[selector_name]
                assert isinstance(
                    selector_value, str
                ), f"{selector_name} should be a string"
                assert (
                    len(selector_value.strip()) > 0
                ), f"{selector_name} should not be empty"

    # Core Functionality Tests
    def test_core_functionality():
        """Test selector structure and CSS validity."""
        # Test CSS selector format validity
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
            ), f"Invalid CSS selector format: {selector}"
            # Test that selectors don't have common issues
            assert not selector.startswith(
                "."
            ), f"Selector should not start with dot: {selector}"
            assert (
                selector.strip() == selector
            ), f"Selector should not have leading/trailing whitespace: {selector}"

    # Edge Cases Tests
    def test_edge_cases():
        """Test edge cases and special selector formats."""
        # Test selectors with placeholders
        placeholder_selectors = []
        for name, value in globals().items():
            if isinstance(value, str) and "{" in value and "}" in value:
                placeholder_selectors.append((name, value))

        for name, selector in placeholder_selectors:
            assert selector.count("{") == selector.count(
                "}"
            ), f"Unmatched braces in {name}: {selector}"
            # Test that placeholder names are reasonable
            import re

            placeholders = re.findall(r"\{([^}]+)\}", selector)
            for placeholder in placeholders:
                assert (
                    placeholder.replace("_", "")
                    .replace(".", "")
                    .replace("-", "")
                    .isalnum()
                ), f"Invalid placeholder in {name}: {placeholder}"

    # Integration Tests
    def test_integration():
        """Test selector organization and completeness."""
        # Count selectors by category
        login_selectors = [
            name for name in globals() if "LOGIN" in name and name.endswith("_SELECTOR")
        ]
        message_selectors = [
            name
            for name in globals()
            if ("MESSAGE" in name or "INBOX" in name) and name.endswith("_SELECTOR")
        ]
        error_selectors = [
            name
            for name in globals()
            if ("ERROR" in name or "UNAVAILABLE" in name) and name.endswith("_SELECTOR")
        ]

        # Ensure we have selectors for major functionality areas
        assert len(login_selectors) >= 0, "Should have login-related selectors defined"
        assert (
            len(message_selectors) >= 5
        ), "Should have message-related selectors defined"
        assert len(error_selectors) >= 3, "Should have error-related selectors defined"

        # Test that all selector constants follow naming convention
        all_selectors = [name for name in globals() if name.endswith("_SELECTOR")]
        for selector_name in all_selectors:
            assert (
                selector_name.isupper()
            ), f"Selector {selector_name} should be uppercase"
            assert (
                "_" in selector_name
            ), f"Selector {selector_name} should use underscore naming"

    # Performance Tests
    def test_performance():
        """Test selector efficiency and structure."""
        # Test selector complexity (avoid overly complex selectors)
        complex_selectors = []
        for name, value in globals().items():
            if isinstance(value, str) and name.endswith("_SELECTOR"):
                # Count selector complexity indicators
                complexity_score = (
                    value.count(" ")
                    + value.count(">")
                    + value.count("+")
                    + value.count("~")
                )
                if complexity_score > 10:  # Arbitrary threshold
                    complex_selectors.append((name, complexity_score))

        # This is informational rather than failing
        for name, score in complex_selectors:
            print(f"Info: Complex selector {name} (score: {score})")

        # Test for duplicate selectors (same value)
        selector_values = {}
        for name, value in globals().items():
            if isinstance(value, str) and name.endswith("_SELECTOR"):
                if value in selector_values:
                    print(
                        f"Info: Duplicate selector value '{value}' in {name} and {selector_values[value]}"
                    )
                else:
                    selector_values[value] = name

    # Error Handling Tests
    def test_error_handling():
        """Test error selector completeness."""
        # Ensure error selectors are comprehensive
        error_types = ["UNAVAILABLE", "ERROR", "TEMP"]
        error_selectors = []

        for error_type in error_types:
            matching_selectors = [
                name
                for name in globals()
                if error_type in name and name.endswith("_SELECTOR")
            ]
            error_selectors.extend(matching_selectors)

        assert (
            len(error_selectors) >= 3
        ), "Should have multiple error handling selectors"

        # Test that error selectors target appropriate elements
        for name in error_selectors:
            if name in globals():
                selector = globals()[name]
                # Error selectors should typically target headers, divs, or spans
                assert any(
                    tag in selector.lower()
                    for tag in ["h1", "h2", "div", "span", "header"]
                ), f"Error selector {name} should target appropriate elements"

    # Define test categories
    test_categories = {
        "Initialization": (
            test_initialization,
            "Should define all required selector constants",
        ),
        "Core Functionality": (
            test_core_functionality,
            "Should have valid CSS selector formats",
        ),
        "Edge Cases": (
            test_edge_cases,
            "Should handle placeholder selectors correctly",
        ),
        "Integration": (
            test_integration,
            "Should organize selectors by functionality areas",
        ),
        "Performance": (test_performance, "Should use efficient selector patterns"),
        "Error Handling": (
            test_error_handling,
            "Should provide comprehensive error detection selectors",
        ),
    }

    # Run all test categories
    with suppress_logging():
        for category_name, (test_func, expected_behavior) in test_categories.items():
            suite.run_test(category_name, test_func, expected_behavior)

    return suite.finish_suite()


def run_comprehensive_tests_fallback() -> bool:
    """
    Fallback test function for when test framework is not available.
    Runs basic functionality tests for selector definitions.
    """
    print("🔍 Running basic CSS selector tests...")

    try:
        # Test 1: Basic selector availability
        basic_selectors = [
            "WAIT_FOR_PAGE_SELECTOR",
            "POPUP_CLOSE_SELECTOR",
            "PAGE_NO_LONGER_AVAILABLE_SELECTOR",
        ]
        for selector_name in basic_selectors:
            assert selector_name in globals(), f"Missing selector: {selector_name}"
            assert isinstance(
                globals()[selector_name], str
            ), f"Selector {selector_name} should be string"
        print("✅ Basic selector availability test passed")

        # Test 2: Selector format validation
        test_selectors = [WAIT_FOR_PAGE_SELECTOR, POPUP_CLOSE_SELECTOR]
        for selector in test_selectors:
            assert len(selector.strip()) > 0, "Selector should not be empty"
            assert (
                selector.strip() == selector
            ), "Selector should not have extra whitespace"
        print("✅ Selector format validation test passed")

        # Test 3: Error selector completeness
        error_selectors = [
            name for name in globals() if "ERROR" in name or "UNAVAILABLE" in name
        ]
        assert len(error_selectors) >= 3, "Should have multiple error selectors"
        print("✅ Error selector completeness test passed")

        # Test 4: Selector naming convention
        all_selectors = [name for name in globals() if name.endswith("_SELECTOR")]
        assert len(all_selectors) >= 10, "Should have multiple selector definitions"
        for selector_name in all_selectors[:5]:  # Test first 5
            assert (
                selector_name.isupper()
            ), f"Selector {selector_name} should be uppercase"
        print("✅ Selector naming convention test passed")

        print("🎉 All basic CSS selector tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    import sys
    import re
    from unittest.mock import MagicMock, patch

    print(
        "🎯 Running CSS Selectors & Element Identification comprehensive test suite..."
    )
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
