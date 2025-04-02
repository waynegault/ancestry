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
TEMP_UNAVAILABLE_SELECTOR =     "div.pageError h1.pageTitle"  # this page is temporarily unavailable.

# --- Home page (logged in and not logged in) --- (https://www.ancestry.co.uk/)
FOOTER_SELECTOR = "footer#footer ul#footerLegal"


# --- Login Page (https://www.ancestry.co.uk/account/signin) ---
CONFIRMED_LOGGED_IN_SELECTOR = "#navAccount[data-tracking-name='Account']"  # "[href^='https://www.ancestry.co.uk/profile/']"
COOKIE_BANNER_SELECTOR = "div#bannerOverlay"
consent_ACCEPT_BUTTON_SELECTOR = "#acceptAllBtn"  # Cookie consent button
LOG_IN_BUTTON_SELECTOR =  "[href^='https://www.ancestry.co.uk/account/signin']"
USERNAME_INPUT_SELECTOR = "input#username"  
PASSWORD_INPUT_SELECTOR = "input#password"  
SIGN_IN_BUTTON_SELECTOR = "#signInBtn"
TWO_FA_EMAIL_SELECTOR = "button[data-method='email']"
TWO_FA_SMS_SELECTOR = "button.ancCardBtn.methodBtn[data-method='sms']"
TWO_STEP_VERIFICATION_HEADER_SELECTOR = "body.mfaPage h2.conTitle"
FAILED_LOGIN_SELECTOR = "div#invalidCredentialsAlert.alert" 
TWO_FA_CODE_BUTTON_SELECTOR =  "button#codeFormSubmitBtn"
TWO_FA_CODE_INPUT_SELECTOR = "button.ancCardBtn.methodBtn[data-method='sms']"


# --- DNA Matches List Page (https://www.ancestry.co.uk/discoveryui-matches/list/) ---
MATCH_ENTRY_SELECTOR = "ui-custom[type='match-entry']"  # Individual match entry
MATCH_NAME_LINK_SELECTOR = "a[data-testid='matchNameLink']"  # Link to the match's page
SHARED_DNA_SELECTOR = "div[data-testid='sharedDnaAmount']"  # Shared DNA amount
PREDICTED_RELATIONSHIP_SELECTOR =  "section.sharedDnaContainer button.relationshipLabel"  # Predicted relationship
TREE_INDICATOR_SELECTOR = "ui-person-avatar[indicator='tree']"  # Icon indicating a tree exists
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
RELATIONSHIP_SELECTOR =    ".relationship-selector"  # Added to previous version, but unused.
RELATIONSHIP_LABEL_SELECTOR = "button.relationshipLabel"  # This is used to get the predicted relationship.

# --- Relationship Modal (appears on Family Tree Page) ---
MODAL_TITLE_SELECTOR = "h4.modalTitle"  # "Relationship to me"
MODAL_CONTENT_SELECTOR = "ul.textCenter"  # Container for relationship path
CLOSE_BUTTON_SELECTOR = "button.closeBtn.modalClose"  # Close button for the modal

# --- Inbox/Messaging Page (https://www.ancestry.co.uk/messaging) ---
INBOX_PAGE_LOAD_SELECTOR = "h1.sectionTitle:contains('Messages')" # Selector for "Messages" heading
INBOX_CONTAINER_SELECTOR =  "div.cardContainer" # "main#main > div > div > div.channelsSection" # "main#main > div > div > div > div:nth-of-type(2)"  # Container for inbox list
RIGHT_PAGE_CHECK_SELECTOR = "div.singleProfile[data-activeprofileid={profile_id.lower()}]"  # check that we are on the correct user's message page
AVATAR_CARD_SELECTOR = "div.avatarCardGroup"  # Individual conversation entry
PROFILE_IDS_SELECTOR = "div[data-profileids]"  # Used to get profile ID from inbox
AVATAR_BOX_SELECTOR = "div.avatarBox"  # Used to extract the username.
MESSAGE_CONTAINER_SELECTOR = "div.messagingContainer" # div.chatContainer Container for an individual conversation
SENT_MESSAGES_SELECTOR = ".fromSelf .chatBubble"  # Selector for sent messages
RECEIVED_MESSAGES_SELECTOR = ".fromOther .chatBubble"  # Selector for received messages
MESSAGE_CONTENT_SELECTOR = ".messageContent span"  # Message text within a bubble
TIMESTAMP_SELECTOR = ".timestamp"  # Timestamp within a message bubble
BUBBLE_SEPARATOR_SELECTOR = "div.bubbleSeparator"  # Date bubble
CONVERSATION_LIST_SELECTOR = "div.cardInner"
MESSAGE_BOX_SELECTOR = "div.inputArea textarea#message-box"  # Text area.
SEND_BUTTON_SELECTOR = "button.ancBtn.sendBtn"  # Send button
MESSAGE_SENT_SELECTOR = ".bubbleContainer  .fromSelf:last-of-type .chatBubble .chatContent + .timestamp"