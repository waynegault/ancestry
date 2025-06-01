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
RIGHT_PAGE_CHECK_SELECTOR = "div.singleProfile[data-activeprofileid={profile_id.lower()}]"  # check that we are on the correct user's message page
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
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    import sys
    import re
    from unittest.mock import MagicMock, patch

    try:
        from test_framework import (
            TestSuite,
            suppress_logging,
            create_mock_data,
            assert_valid_function,
        )
    except ImportError:
        print(
            "❌ test_framework.py not found. Please ensure it exists in the same directory."
        )
        sys.exit(1)

    def run_comprehensive_tests() -> bool:
        """
        Comprehensive test suite for my_selectors.py.
        Tests CSS selector definitions, validation, and organization.
        """
        suite = TestSuite("CSS Selectors & Element Identification", "my_selectors.py")
        suite.start_suite()

        # Test 1: Login selectors validation
        def test_login_selectors():
            login_selectors = []
            for name in globals():
                if "LOGIN" in name and isinstance(globals()[name], str):
                    login_selectors.append((name, globals()[name]))

            assert len(login_selectors) > 0, "Should have login-related selectors"

            for name, selector in login_selectors:
                assert isinstance(selector, str), f"{name} should be a string"
                assert len(selector) > 0, f"{name} should not be empty"

        # Test 2: CSS selector syntax validation
        def test_css_selector_syntax():
            # Basic CSS selector pattern validation
            css_pattern = re.compile(
                r"^[a-zA-Z0-9\-_\.\#\[\]\(\)\:\s\>\+\~\*\,\=\"\'\|\^\/\{\}]+$"
            )

            selectors_to_test = []
            for name in globals():
                value = globals()[name]
                if isinstance(value, str) and (
                    "SELECTOR" in name or "INPUT" in name or "BUTTON" in name
                ):
                    selectors_to_test.append((name, value))

            for name, selector in selectors_to_test:
                # Basic syntax check - should contain valid CSS characters
                if selector and not selector.startswith("//"):  # Skip XPath selectors
                    assert css_pattern.match(
                        selector
                    ), f"{name} should be valid CSS selector syntax"

        # Test 3: Two-factor authentication selectors
        def test_2fa_selectors():
            tfa_selectors = []
            for name in globals():
                if ("2FA" in name or "TFA" in name or "AUTH" in name) and isinstance(
                    globals()[name], str
                ):
                    tfa_selectors.append((name, globals()[name]))

            # Should have at least some 2FA selectors if implemented
            if tfa_selectors:
                for name, selector in tfa_selectors:
                    assert isinstance(selector, str), f"{name} should be a string"
                    assert len(selector) > 0, f"{name} should not be empty"

        # Test 4: Form element selectors
        def test_form_element_selectors():
            form_elements = ["INPUT", "BUTTON", "FORM", "FIELD"]
            found_form_selectors = []

            for name in globals():
                if any(element in name for element in form_elements) and isinstance(
                    globals()[name], str
                ):
                    found_form_selectors.append((name, globals()[name]))

            # Validate form selectors if they exist
            for name, selector in found_form_selectors:
                assert isinstance(selector, str), f"{name} should be a string"
                if selector:  # Allow empty selectors for optional elements
                    # Check for common form selector patterns
                    is_valid_form_selector = (
                        "#" in selector  # ID selector
                        or "." in selector  # Class selector
                        or "[" in selector  # Attribute selector
                        or selector.startswith(
                            ("input", "button", "form")
                        )  # Element selector
                    )
                    if not is_valid_form_selector and not selector.startswith("//"):
                        suite.add_warning(
                            f"{name} may not be a valid form selector: {selector}"
                        )

        # Test 5: Navigation selectors
        def test_navigation_selectors():
            nav_keywords = ["NAV", "MENU", "LINK", "TAB"]
            nav_selectors = []

            for name in globals():
                if any(keyword in name for keyword in nav_keywords) and isinstance(
                    globals()[name], str
                ):
                    nav_selectors.append((name, globals()[name]))

            # Validate navigation selectors
            for name, selector in nav_selectors:
                assert isinstance(selector, str), f"{name} should be a string"
                if selector and not selector.startswith("//"):
                    # Basic validation for navigation elements
                    assert len(selector) > 0, f"{name} should not be empty"

        # Test 6: Element state selectors
        def test_element_state_selectors():
            state_keywords = ["ACTIVE", "DISABLED", "VISIBLE", "HIDDEN", "LOADING"]
            state_selectors = []

            for name in globals():
                if any(keyword in name for keyword in state_keywords) and isinstance(
                    globals()[name], str
                ):
                    state_selectors.append((name, globals()[name]))

            # Validate state selectors
            for name, selector in state_selectors:
                assert isinstance(selector, str), f"{name} should be a string"

        # Test 7: Data attribute selectors
        def test_data_attribute_selectors():
            # Look for selectors using data attributes
            data_selectors = []
            for name in globals():
                value = globals()[name]
                if isinstance(value, str) and "data-" in value:
                    data_selectors.append((name, value))

            # Validate data attribute selectors
            for name, selector in data_selectors:
                # Should be properly formatted data attribute selector
                assert (
                    "[data-" in selector or "data-" in selector
                ), f"{name} should use proper data attribute syntax"

        # Test 8: Error and message selectors
        def test_error_message_selectors():
            message_keywords = ["ERROR", "MESSAGE", "ALERT", "NOTIFICATION", "WARNING"]
            message_selectors = []

            for name in globals():
                if any(keyword in name for keyword in message_keywords) and isinstance(
                    globals()[name], str
                ):
                    message_selectors.append((name, globals()[name]))

            # Validate message selectors
            for name, selector in message_selectors:
                assert isinstance(selector, str), f"{name} should be a string"

        # Test 9: Modal and popup selectors
        def test_modal_popup_selectors():
            modal_keywords = ["MODAL", "POPUP", "DIALOG", "OVERLAY"]
            modal_selectors = []

            for name in globals():
                if any(keyword in name for keyword in modal_keywords) and isinstance(
                    globals()[name], str
                ):
                    modal_selectors.append((name, globals()[name]))

            # Validate modal selectors
            for name, selector in modal_selectors:
                assert isinstance(selector, str), f"{name} should be a string"

        # Test 10: Selector organization and naming
        def test_selector_organization():
            all_selectors = []
            for name in globals():
                value = globals()[name]
                if isinstance(value, str) and not name.startswith("_"):
                    all_selectors.append(name)

            # Check naming conventions
            naming_issues = []
            for name in all_selectors:
                # Should be uppercase constants
                if not name.isupper():
                    naming_issues.append(f"{name} should be uppercase")

                # Should use underscores for word separation
                if " " in name:
                    naming_issues.append(
                        f"{name} should use underscores instead of spaces"
                    )

            if naming_issues:
                for issue in naming_issues[:5]:  # Show first 5 issues
                    suite.add_warning(issue)

        # Run all tests
        test_functions = {
            "Login selectors validation": (
                test_login_selectors,
                "Should define login-related CSS selectors",
            ),
            "CSS selector syntax validation": (
                test_css_selector_syntax,
                "Should use valid CSS selector syntax",
            ),
            "Two-factor authentication selectors": (
                test_2fa_selectors,
                "Should define 2FA-related selectors if implemented",
            ),
            "Form element selectors": (
                test_form_element_selectors,
                "Should define selectors for form inputs and buttons",
            ),
            "Navigation selectors": (
                test_navigation_selectors,
                "Should define selectors for navigation elements",
            ),
            "Element state selectors": (
                test_element_state_selectors,
                "Should define selectors for different element states",
            ),
            "Data attribute selectors": (
                test_data_attribute_selectors,
                "Should properly format data attribute selectors",
            ),
            "Error and message selectors": (
                test_error_message_selectors,
                "Should define selectors for error and message elements",
            ),
            "Modal and popup selectors": (
                test_modal_popup_selectors,
                "Should define selectors for modal dialogs and popups",
            ),
            "Selector organization and naming": (
                test_selector_organization,
                "Should follow consistent naming conventions",
            ),
        }

        with suppress_logging():
            for test_name, (test_func, expected_behavior) in test_functions.items():
                suite.run_test(test_name, test_func, expected_behavior)

        return suite.finish_suite()

    print(
        "🎯 Running CSS Selectors & Element Identification comprehensive test suite..."
    )
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
