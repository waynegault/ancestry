# Ancestry.com Automation Project

## Overview

This project automates interactions with Ancestry.com, leveraging a hybrid approach of web browser automation and direct API calls. It aims to gather genealogical data, manage communications, and streamline family history research using Python, Selenium, SQLAlchemy, and requests.

## Purpose

The primary goals of this project are to:

*   **Gather DNA Match Information:** Automatically collect data on DNA matches, including relationships, shared DNA, and tree information, primarily through Ancestry's internal APIs.
*   **Inbox Management:** Automate the process of searching and processing messages within an Ancestry.com inbox via API calls.
*   **Automated Messaging:** Send personalized messages to DNA matches to facilitate collaboration and expand genealogical knowledge.

## How It Works

The project operates through a series of automated actions orchestrated through a Python-based menu system (`main.py`). Key components include:

1.  **Session Management & Authentication (Selenium & `utils.py`):**
    *   Selenium WebDriver (via `chromedriver.py`) controls a Chrome browser instance, primarily used to establish an authenticated user session, handle logins, and obtain necessary cookies (like `ANCSESSIONID`) and security tokens (like CSRF tokens).
    *   The `SessionManager` in `utils.py` manages the browser session, extracts critical identifiers (`my_profile_id`, `my_uuid`, `my_tree_id`), retrieves necessary cookies and headers, and prepares a `requests` session for subsequent API calls.
    *   It handles initial navigation to ensure the correct session context.

2.  **API Interaction (`utils._api_req`, `action*.py`):**
    *   Most data retrieval (DNA matches, inbox messages, relationship details, etc.) is performed by making direct calls to Ancestry's internal web APIs.
    *   The `_api_req` helper function in `utils.py` centralizes the logic for making these API calls, injecting necessary dynamic headers (like `ancestry-context-ube`, `newrelic`, `traceparent`) and cookies obtained during session initialization. It uses Selenium's `execute_script` to leverage the authenticated browser context for making fetch requests.
    *   Specific action modules (`action6_gather.py`, `action7_inbox.py`, `action8_messaging.py`) define the sequence of API calls needed for each task.

3.  **Database Management (SQLAlchemy & SQLite):**
    *   SQLAlchemy ORM (`database.py`) defines database models for storing information about people, DNA matches, family trees, and inbox status.
    *   A local SQLite database (`ancestry.db` specified in `.env`) stores the collected data persistently.
    *   A connection pool (`database.ConnectionPool`) manages database connections efficiently.

4.  **Configuration (`config.py`, `.env`):**
    *   Core settings (login credentials, database paths, API base URLs, processing limits) are loaded from a `.env` file via `config.py`.

5.  **Logging (`logging_config.py`):**
    *   Comprehensive logging tracks script execution, API calls, errors, and debugging information, saved to `logs/ancestry.log`.

6.  **Utilities (`utils.py`):**
    *   Includes functions for robust session startup, navigation helpers (used less now but still available), retry logic (`@retry`), dynamic rate limiting (`DynamicRateLimiter`), and header generation helpers.

7.  **Caching (`cache.py`):**
    *   The `cache_result` decorator can cache function results (e.g., API responses for static data) using `diskcache` to improve performance and reduce redundant calls.

## Technical Background

*   **Python:** The project is written in Python 3.
*   **Selenium:** Used primarily for browser session initialization, authentication, and as a helper for executing authenticated API calls via JavaScript injection (`execute_script`).
*   **Requests (Implicit):** While not directly imported everywhere, the `_api_req` function essentially uses the browser's `fetch` API (similar to `requests`) executed via Selenium.
*   **SQLAlchemy:** Python SQL toolkit and ORM for database interaction.
*   **SQLite:** Lightweight, file-based database engine.
*   **dotenv:** Python package for loading environment variables from `.env`.
*   **Logging:** Standard Python module for logging events.
*   **Diskcache:** Python library for persistent function result caching.
*   **Undetected Chromedriver:** Used to make the controlled browser appear more like a regular user browser.
*   **PSUtil:** Used for monitoring memory usage during actions.

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.7+
    *   Chrome browser
    *   ChromeDriver (compatible with your Chrome version, path specified in `.env`)

2.  **Install Dependencies:**

    ```bash
    # It's recommended to use a virtual environment
    # python -m venv venv
    # source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows

    pip install selenium sqlalchemy python-dotenv undetected-chromedriver diskcache psutil requests beautifulsoup4 urllib3
    # Note: requests and beautifulsoup4 might be needed for specific API response handling or HTML parsing within API results.
    ```
    *(Consider creating a `requirements.txt` file)*

3.  **Configuration:**

    *   Create a `.env` file in the project root directory.
    *   Add the following variables, replacing placeholder values:

        ```dotenv
        # .env file

        # Ancestry credentials
        ANCESTRY_USERNAME=your_ancestry_username
        ANCESTRY_PASSWORD=your_ancestry_password

        # Names & Paths
        TREE_NAME='Your Tree Name' # Match exactly as it appears on Ancestry
        DATABASE_FILE="Data/ancestry.db" # Ensure 'Data' directory exists or adjust path
        LOG_DIR="Logs" # Ensure 'Logs' directory exists or adjust path
        # GEDCOM_FILE_PATH=Data/YourGedcom.ged # Optional: If needed for other features

        # URLs
        BASE_URL=https://www.ancestry.co.uk/

        # Local Selenium/Chrome settings
        CHROME_DRIVER_PATH=C:\path\to\your\chromedriver.exe # Adjust path
        CHROME_USER_DATA_DIR=C:\Users\YourUser\AppData\Local\Google\Chrome\User Data # Adjust path if using specific profile
        # CHROME_BROWSER_PATH=/path/to/chrome # Optional: Specify if not in default location

        # Application settings
        MAX_PAGES=1 # 0 = process all pages
        MAX_INBOX=10 # 0 = process all inbox items
        BATCH_SIZE=5 # Number of matches/messages to process per API call/DB transaction
        MAX_RETRIES=3 # Max retries for API calls/actions
        DB_POOL_SIZE=5 # Number of connections in the DB pool

        # Application Mode: 'dry_run' (logs actions, minimal writes), 'testing' (more writes, limited scope), 'production' (full operation)
        APP_MODE="testing"

        # Delays (in seconds)
        INITIAL_DELAY=1.0 # Starting delay for rate limiter
        MIN_DELAY=0.5     # Minimum delay
        MAX_DELAY=5.0     # Maximum delay
        # Add other delay settings as needed

        ```

4.  **Run the Project:**

    ```bash
    python main.py
    ```

## Project Structure

```text
ancestry/
├── Data/                 # Directory for database, GEDCOMs etc. (needs creation)
│   └── ancestry.db       # (Created by the script)
├── Logs/                 # Directory for log files (needs creation)
│   └── ancestry.log      # (Created by the script)
├── action6_gather.py     # Gather DNA match data (API-based)
├── action7_inbox.py      # Process inbox messages (API-based)
├── action8_messaging.py  # Send messages to matches
├── cache.py              # Caching module
├── chromedriver.py       # ChromeDriver management
├── config.py             # Configuration loading
├── database.py           # Database models, connection pool, basic operations
├── logging_config.py     # Logging setup
├── main.py               # Main entry point and menu
├── my_selectors.py       # CSS selectors (primarily for login/initial setup)
├── utils.py              # SessionManager, API request helper, utilities
├── messages.json         # Message templates (if used by Action 8)
├── README.md             # This file
├── .env                  # Environment variables (create this manually, DO NOT COMMIT)
└── requirements.txt      # Optional: List of dependencies
Use code with caution.
Markdown
Usage
Ensure Python, Chrome, and the correct ChromeDriver are installed.

Create the .env file with your configuration.

Create the Data and Logs directories if they don't exist.

Install dependencies (e.g., pip install -r requirements.txt if you create one).

Run python main.py.

Select options from the menu to perform actions.

Refer to log files (Logs/ancestry.log) for detailed output and debugging information.

Maintenance
Keep Dependencies Up-to-Date: Regularly update Python packages.

Monitor Ancestry.com API Changes: Internal APIs can change without notice. Be prepared to adapt the API endpoints, request parameters, headers, or response parsing. Use browser developer tools to monitor network requests during manual use of Ancestry to detect changes.

Review and Refactor Code: Continuously improve code quality, error handling, and maintainability.

Disclaimer
This project interacts with internal Ancestry.com APIs which are not officially documented or supported for third-party use. Use this project responsibly, ethically, and at your own risk. Be mindful of Ancestry's Terms of Service. Excessive requests could potentially lead to account restrictions. The author assumes no liability for the use or misuse of this software.

Technical Appendix: API Interaction Details
This section documents the key technical aspects of how the project interacts with Ancestry's APIs, based on current understanding.

1. Session Initialization (utils.start_sess)
Establishing a valid, authenticated session is the prerequisite for all API calls. start_sess performs these steps:

Launch Browser: Starts undetected-chromedriver.

Login Check: Navigates to Ancestry and verifies login status using UI elements (e.g., account menu) or cookies. Performs login if necessary.

Context Navigation: Ensures the browser is on a suitable page (like the main dashboard) to establish the correct context for cookies and subsequent API calls.

Cookie Retrieval: Extracts essential cookies from the browser session (e.g., ANCSESSIONID, potentially consent cookies).

CSRF Token: Retrieves the x-csrf-token via an API call or by extracting it from cookies/page source.

Identifier Extraction: Uses API calls (like /api/uhome/secure/rest/header/trees and potentially others inferred from network traffic) to get the user's unique identifiers:

my_profile_id (e.g., 07BDD45E-0006-0000-0000-000000000000)

my_uuid (DNA Test GUID, e.g., FB609BA5-5A0D-46EE-BF18-C300D8DE5AB7)

my_tree_id (Active tree ID, e.g., 175946702)

Header Preparation: Stores necessary components (like CSRF token, ANCSESSIONID) to construct required headers for subsequent API calls via _api_req.

2. Key Identifiers
uuid (DNA Test GUID / sampleId): Unique identifier for a specific DNA test kit. Primary key for linking DNA data. Found in match lists and compare links.

profile_id: Unique identifier for a user account profile. Used in messaging links (/messaging/?p={profile_id}) and potentially profile URLs. Note: A user might designate another profile_id to manage their messages. Used as the primary lookup key in the people table when processing inbox messages.

cfpid (Contextual Family Person ID): Identifier for a person within the context of a specific family tree. Used in tree navigation links (/family-tree/.../person/{cfpid}/...). Not globally unique across all trees.

tree_id: Unique identifier for a specific family tree.

3. Core API Workflows & Endpoints (Examples)
Action 6 (Gather Matches):

Navigate: Go to /discoveryui-matches/list/{my_uuid} (primarily for browser context).

Get Total Pages/Initial List: GET /discoveryui-matches/parents/list/api/matchList/{my_uuid}?currentPage=1

Get Match List (Paginated): GET /discoveryui-matches/parents/list/api/matchList/{my_uuid}?currentPage={page_num} (Response contains basic match data including sampleId (their UUID), displayName, userId (their profile_id), shared cM/segments).

Get In-Tree Status: POST /discoveryui-matches/parents/list/api/badges/matchesInTree/{my_uuid} (Body: {"sampleIds": ["uuid1", "uuid2", ...]}). Returns list of sampleIds that are in the user's tree.

Get Predicted Relationship: POST /discoveryui-matches/parents/list/api/matchProbabilityData/{my_uuid}/{their_uuid} (Body: {}). Response contains relationship probabilities and labels.

Get Tree/CFPID Details (if In-Tree): GET /discoveryui-matchesservice/api/samples/{my_uuid}/matches/{their_uuid}/badgedetails (Response contains personBadged.personId (their CFPID)).

Get Relationship Path (if In-Tree): GET /family-tree/person/tree/{my_tree_id}/person/{their_cfpid}/getladder?callback=jQuery (Requires parsing JSONP response and HTML content to get actual_relationship and relationship_path).

Construct Links: compare_link, message_link, view_in_tree_link, facts_link are constructed locally using the retrieved identifiers.

Action 7 (Search Inbox):

Get Conversations (Paginated): GET /app-api/express/v2/conversations?q=user:{my_profile_id}&limit={batch_size}&cursor={cursor_token} (Response contains conversation list, participant details (user_id, display_name), last message snippet, timestamp, author, and paging.forward_cursor).

4. Authentication & Headers (_api_req)
Making successful API calls requires specific headers, managed by _api_req:

Cookie Header: Includes all necessary cookies captured by Selenium (ANCSESSIONID, Cloudflare cookies, etc.), formatted as a single string.

x-csrf-token: The CSRF token obtained during session initialization. Must match the _dnamatches-matchlistui-x-csrf-token cookie value for certain endpoints.

ancestry-context-ube: Dynamically generated Base64-encoded JSON header. Crucial for many API calls. Its structure includes:

eventId: Static UUID (000...).

correlatedScreenViewedId: New UUID per request.

correlatedSessionId: Value of the ANCSESSIONID cookie.

userConsent: Crucial but problematic. Reflects user's cookie consent choices (e.g., necessary|preference|performance|...). Currently hardcoded or requires reliable dynamic retrieval. See Known Issues.

vendors, vendorConfigurations: Often related to analytics/marketing consent.

newrelic, traceparent, tracestate: Headers related to New Relic performance monitoring and W3C Trace Context standard. Values are dynamic and likely generated by Ancestry's frontend JavaScript. _api_req likely needs helper functions (make_newrelic, make_traceparent, etc.) to generate syntactically valid (if not functionally perfect) versions of these.

Standard Headers: Accept: application/json, Content-Type: application/json, User-Agent, Referer.

5. Known Issues & Challenges
userConsent in ancestry-context-ube: Reliably determining the correct, dynamic userConsent string based on the user's actual settings is the biggest challenge. Hardcoding is likely to fail. Potential solutions involve inspecting cookies set by the consent manager (OneTrust) or finding JavaScript variables where this is stored.

Cloudflare Protection: Ancestry uses Cloudflare, which may involve checks (__cf_bm, _cfuvid, cf_clearance cookies) to prevent bot activity. undetected-chromedriver helps, but aggressive scraping could still trigger challenges.

Dynamic Headers (newrelic, etc.): While we can generate headers with the correct format, ensuring the values are meaningful to Ancestry's backend might be difficult without perfectly replicating their frontend logic. However, often just providing a correctly formatted header is sufficient.

API Instability: Internal APIs can change structure, endpoints, or required parameters without warning.

6. URL Structures (Reference)
DNA Match List: /discoveryui-matches/list/{my_uuid}

Compare Page: /discoveryui-matches/compare/{my_uuid}/with/{their_uuid}

Messaging: /messaging/?p={profile_id} (or with &testguid1=...&testguid2=...)

Tree View: /family-tree/tree/{tree_id}/family?cfpid={cfpid}

Facts View: /family-tree/person/tree/{tree_id}/person/{cfpid}/facts

(See self-help notes for API endpoint structures)

7. Error Handling & Robustness
Retries: The @retry decorator handles transient network errors or temporary API failures.

Rate Limiting: DynamicRateLimiter adjusts delays between requests to avoid overwhelming the server. Configurable via .env.

Specific Errors: Code should attempt to catch specific exceptions (e.g., TimeoutException, SQLAlchemyError, RequestException) for tailored handling.

Intermittent Pages: Logic should ideally detect "temporarily unavailable" messages (as noted in self-help notes) and potentially trigger retries or skip operations gracefully, rather than crashing. This might involve checking page content or specific status codes if returned by APIs.

Wayne Gault<br>
(Date Updated: 2025-03-29)



Action 8 details:

Core Purpose:

To automatically send personalized messages to DNA matches found in the database via the Ancestry API, following a defined sequence and respecting specific rules to avoid spamming or messaging inappropriately.

Candidate Selection:

The primary pool of candidates consists of all individuals present in the DnaMatch table who also have an active Person record (status == "active") and a valid profile_id.

Messaging Rules & Logic:

Conversation Requirement: Sending a message via the known POST API requires an existing conversation_id. This ID is typically obtained by Action 7 (InboxProcessor) when it finds existing conversations. Therefore, Action 8 can currently only send messages (initial or follow-up) to individuals for whom Action 7 has successfully run and populated an InboxStatus record with a conversation_id. It cannot initiate a brand new conversation via the API with its current known endpoint.

"Reply Received" Rule: Before sending any message (initial or follow-up), the script checks:

The InboxStatus table for the timestamp of the last message received from the match (last_received_message_timestamp).

The MessageHistory table for the timestamp of the last message sent by the script (last_sent_at).

If a message has been received (last_received_message_timestamp exists) AND it is newer than the last message the script sent (last_received_message_timestamp > last_sent_at OR if last_sent_at is None), then NO automated message is sent. The script assumes manual intervention is needed. This rule ensures the script doesn't interrupt an ongoing exchange initiated by the match.

Anti-Spam Timing Rule: If the "Reply Received" rule doesn't apply, the script checks the time elapsed since the last message sent by the script (last_sent_at). If this time is less than the configured MIN_MESSAGE_INTERVAL (which varies based on APP_MODE), NO message is sent.

Message Sequencing: If neither of the above rules prevents sending, the script determines the next message type based on:

The type_name of the last message sent (from MessageHistory).

The current in_my_tree status of the match (from the Person table).

It follows distinct sequences:

In Tree: In_Tree-Initial -> In_Tree-Follow_Up -> In_Tree-Final_Reminder -> (Stop)

Out of Tree: Out_Tree-Initial -> Out_Tree-Follow_Up -> Out_Tree-Final_Reminder -> (Stop)

Transition: If the match was previously Out_Tree but is now In_Tree, it sends In_Tree-Initial_for_was_Out_Tree and then follows the "In Tree" sequence.

If the last sent message was the final reminder in a sequence, no further messages are sent.

(Dry Run Consideration): The logic now treats a previous "typed (dry_run)" history entry as if no message was sent when determining if an initial message should be sent, allowing repeated dry runs of initial messages. However, the timing rule still applies based on the timestamp of that dry run entry.

Limit: An overall limit (MAX_INBOX) can be set to restrict the total number of messages sent or typed (dry run) in a single execution of Action 8.

Message Formatting & Sending:

Template Selection: Chooses the correct template from messages.json based on the next_message_type_key.

Name Selection: Selects the name for the {name} placeholder with the priority: FamilyTree.person_name_in_tree (if available & In Tree) > Person.first_name > Person.username > "Valued Relative".

Formatting: Populates the chosen template with available data (name, predicted_relationship, actual_relationship, relationship_path, total_rows). It skips sending if required data for the template is missing.

Sending (API):

Constructs the correct API endpoint (/conversations/{id}).

Constructs the JSON payload (content, author).

Calls _api_req to POST the message.

Dry Run: If APP_MODE is "dry_run", it logs the intent to send and the formatted message but skips the API POST call.

Database Updates:

On Send/Type: If a message is successfully sent via API (based on the initial API response) or typed in dry run:

A new record is added to MessageHistory detailing the message type, text, status ("sent (api_ok)", "sent_confirmed", "send_error (...)", or "typed (dry_run)"), and timestamp.

The corresponding InboxStatus record is updated (or created if missing but a conversation_id was somehow available - though this path is now less likely), setting my_role to AUTHOR and updating last_message and last_message_timestamp. The conversation_id is also updated if it was newly obtained (which isn't happening with the current POST logic).

Commit: Database changes are committed after each successful message send/type attempt where DB updates were staged. (Correction: Commit happens once at the end of the with block if no errors occurred).

In summary, Action 8 aims to be a rules-based messaging assistant, identifying DNA matches, checking if/when/what to message them next based on history and interaction status, formatting a relevant message, and attempting to send it via the known "add message to conversation" API endpoint, finally recording the action. Its major limitation currently is the inability to initiate new conversations via API due to the lack of the required endpoint information.