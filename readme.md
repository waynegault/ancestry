# Ancestry.com Automation Project

## 1. Overview

This project automates interactions with Ancestry.com, employing a hybrid strategy that combines web browser automation (via Selenium and Undetected Chromedriver) for session establishment and direct API calls for most data operations. It streamlines genealogical research by automating tasks such as gathering DNA match data, managing inbox messages, sending templated communications, processing productive replies with AI assistance, and generating reports from both local GEDCOM files and Ancestry's online data.

The system is built in Python and leverages libraries like Selenium for browser control, SQLAlchemy for database interaction (SQLite), Requests for API communication (often indirectly via Selenium's JavaScript execution or direct HTTP calls with synced cookies), and external AI models (DeepSeek/Gemini) for intelligent message processing. It features a modular design with distinct "actions" for different functionalities, robust session management, dynamic API header generation, and comprehensive logging.

## 2. Purpose

The primary objectives of this project are to:

*   **Automate Data Collection:** Systematically gather comprehensive information on DNA matches, including shared DNA amounts, predicted relationships, family tree linkages, and profile details.
*   **Efficient Inbox Management:** Automate the retrieval and processing of messages from the Ancestry inbox, identify new communications, and classify their intent using AI to prioritize follow-ups.
*   **Streamlined Communication:** Send personalized, templated messages to DNA matches to initiate contact, follow up on previous communications, or acknowledge productive replies, adhering to defined rules and sequences to avoid over-messaging.
*   **AI-Powered Research Assistance:** Utilize AI to extract key genealogical entities (names, dates, locations, relationships, key facts) from productive user messages and suggest actionable research tasks to further the investigation.
*   **Local Data Persistence:** Store all collected data (person profiles, DNA match details, tree links, conversation logs, AI analysis) in a local SQLite database for offline analysis, custom querying, historical tracking, and to inform future automated actions.
*   **Genealogical Reporting:** Provide tools to generate reports by finding matches within local GEDCOM files (Action 10) and searching for individuals using Ancestry's various internal APIs (Action 11).
*   **Task Management Integration:** Create tasks in Microsoft To-Do based on AI-suggested follow-ups from productive messages, integrating research directly into a task management workflow.
*   **Reduce Manual Effort:** Significantly minimize the time and manual effort required for common, repetitive, and data-intensive tasks associated with online genealogy research on Ancestry.com.

## 3. Key Features

*   **Hybrid Automation Strategy:** Utilizes Selenium primarily for robust session initialization, handling complex login flows (including 2FA), and obtaining essential authentication tokens/cookies. Most subsequent operations leverage direct API calls for efficiency and speed.
*   **Modular Action System:** Functionality is clearly divided into distinct "action" modules (e.g., `action6_gather.py` for DNA matches, `action7_inbox.py` for inbox processing), making the system extensible and easier to maintain.
*   **Sophisticated Session Management:** The `SessionManager` class in `utils.py` is central to the project, managing the browser lifecycle, login processes, cookie and CSRF token extraction/synchronization, and maintaining the authenticated context for API calls. It also manages database connections.
*   **Dynamic API Interaction:** The core `_api_req` utility (in `utils.py`) dynamically constructs necessary HTTP headers (including complex ones like `ancestry-context-ube`, `newrelic`, `traceparent`) for authenticated API calls, often using JavaScript execution via Selenium or synced cookies with the `requests` library.
*   **AI Integration (`ai_interface.py`):**
    *   Classifies incoming message intent (PRODUCTIVE, UNINTERESTED, DESIST, OTHER).
    *   Extracts key genealogical entities (names, dates, locations, relationships, facts) from messages.
    *   Suggests actionable follow-up research tasks based on message content.
    *   Supports multiple AI providers (DeepSeek, Google Gemini) configurable via `.env`.
*   **Database Storage (`database.py`):** Employs SQLAlchemy ORM with an SQLite backend. Defines models for Person, DnaMatch, FamilyTree, ConversationLog, and MessageType. Includes transaction management and schema creation.
*   **Templated Messaging (`action8_messaging.py`, `messages.json`):** Sends personalized messages using predefined templates stored in `messages.json`, with placeholders for dynamic content. Follows a rule-based sequencing logic.
*   **Rate Limiting & Retries:** Implements a `DynamicRateLimiter` and `@retry_api` decorator in `utils.py` for API calls to handle transient errors and avoid overwhelming the server.
*   **Configuration Management (`config.py`, `.env`):** All critical settings (credentials, paths, API URLs, behavioral parameters, AI keys) are loaded from a `.env` file and managed by the `Config_Class` and `SeleniumConfig` classes.
*   **Comprehensive Logging (`logging_config.py`):** Detailed, multi-level logging for debugging, tracking script execution, API calls, and errors. Supports both console and file output with custom formatting.
*   **Caching (`cache.py`):** Utilizes `diskcache` for persistent caching of function results (e.g., static API data, message templates) to improve performance and reduce redundant API calls.
*   **GEDCOM Utilities (`gedcom_utils.py`, `relationship_utils.py`):** Supports loading, parsing, and querying local GEDCOM files, including calculating match scores and finding relationship paths.
*   **Microsoft Graph Integration (`ms_graph_utils.py`):** Authenticates with MS Graph API and creates tasks in Microsoft To-Do based on AI suggestions.
*   **Standalone Action Runners & Self-Tests:** Many modules include `if __name__ == "__main__":` blocks with self-test routines or dedicated `run_action*.py` scripts for testing individual actions in isolation.

## 4. Architecture

### 4.1 Core Components

*   **`main.py`:** The main entry point, providing a command-line menu to trigger various actions. Orchestrates calls to action modules.
*   **`config.py` & `.env`:** Centralized configuration. `Config_Class` and `SeleniumConfig` load settings from the `.env` file.
*   **`logging_config.py`:** Sets up application-wide logging.
*   **`database.py`:** Defines SQLAlchemy ORM models, database engine/session setup, utility functions (backup, restore, schema creation), and transaction management (`db_transn`).
*   **`utils.py`:**
    *   **`SessionManager`:** The heart of session and state management. Handles:
        *   Browser initialization (`init_webdvr` from `chromedriver.py`).
        *   Login process (`log_in`, `handle_twoFA`).
        *   Session validation (`is_sess_valid`, `login_status`).
        *   Extraction of user identifiers (`my_profile_id`, `my_uuid`, `my_tree_id`).
        *   CSRF token management.
        *   Cookie synchronization between Selenium and `requests.Session`.
        *   Database connection pooling.
    *   **`_api_req`:** The core function for making authenticated API calls. Constructs dynamic headers, handles retries (via `@retry_api`), and processes responses.
    *   **`DynamicRateLimiter`:** Manages request rates to avoid API abuse.
    *   General utility functions (formatting, decorators).
*   **`chromedriver.py`:** Manages the `undetected_chromedriver` lifecycle, including Chrome options, preference file setup, and process cleanup.
*   **`selenium_utils.py`:** Selenium-specific helper functions (element interaction, cookie export).
*   **`api_utils.py`:** Contains wrapper functions for specific Ancestry API endpoints, abstracting the direct call logic from action modules.
*   **`ai_interface.py`:** Handles all interactions with external AI models for message classification and data extraction.
*   **`ms_graph_utils.py`:** Manages authentication and task creation with the Microsoft Graph API for To-Do integration.
*   **`cache.py`:** Provides the `@cache_result` decorator and management functions for disk-based caching.
*   **`gedcom_utils.py` & `relationship_utils.py`:** Provide tools for parsing local GEDCOM files, calculating match scores against GEDCOM data, and determining/formatting relationship paths.
*   **Action Modules (`action*.py`):** Each module encapsulates a major piece of functionality (e.g., `action6_gather.py` for DNA matches, `action7_inbox.py` for inbox processing). They utilize `SessionManager`, API helpers, database functions, and AI services as needed.
*   **`my_selectors.py`:** Stores CSS selectors used by Selenium, primarily for login and initial UI interactions.
*   **`messages.json`:** Contains templates for automated messages sent by Action 8.

### 4.2 Data Flow & Execution Logic

1.  **Initialization (`main.py` -> `config.py`, `logging_config.py`, `utils.SessionManager`):**
    *   Configuration is loaded.
    *   Logging is set up.
    *   A `SessionManager` instance is created. This initializes its internal `requests.Session` and `DynamicRateLimiter`. Database connections are prepared via `ensure_db_ready`.

2.  **User Action Selection (`main.py`):**
    *   The user selects an action from the menu.

3.  **Session Preparation (`main.exec_actn` -> `utils.SessionManager`):**
    *   The `exec_actn` function determines if the chosen action requires a browser.
    *   If a browser is needed:
        *   `SessionManager.ensure_driver_live()` is called, which may trigger `SessionManager.start_browser()`.
        *   `start_browser()` calls `chromedriver.init_webdvr()` to launch/attach to a Chrome instance.
        *   `SessionManager.ensure_session_ready()` then handles:
            *   Login (if not already logged in) using `utils.log_in()`, which uses UI interaction via Selenium and selectors from `my_selectors.py`. This includes 2FA handling.
            *   Navigation to appropriate pages to establish session context.
            *   Extraction of essential cookies (`ANCSESSIONID`, `SecureATT`, etc.).
            *   Synchronization of these cookies to the `SessionManager`'s internal `requests.Session` and `cloudscraper` instance.
            *   Fetching the CSRF token (often via an API call like `/api/csrfToken`).
            *   Retrieving user identifiers (`my_profile_id`, `my_uuid`, `my_tree_id`) via API calls.
            *   Fetching the tree owner's name.
    *   If only database access is needed:
        *   `SessionManager.ensure_db_ready()` ensures the database engine and session factory are initialized.

4.  **Action Execution (e.g., `action6_gather.coord`):**
    *   The specific action function is called, receiving the prepared `SessionManager` and `config_instance`.
    *   **API Calls:**
        *   Actions make API calls primarily through `utils._api_req` or wrappers in `api_utils.py`.
        *   `_api_req` uses the `SessionManager`'s `requests.Session` (which has synced cookies) and dynamically generated headers (CSRF, UBE, NewRelic, Traceparent, User-Agent, Referer).
        *   The `DynamicRateLimiter` controls request frequency.
        *   The `@retry_api` decorator handles transient network errors or specific HTTP status codes.
    *   **AI Interaction (e.g., `action7_inbox`, `action9_process_productive`):**
        *   Relevant data (e.g., message history) is passed to functions in `ai_interface.py`.
        *   These functions call the configured AI provider (DeepSeek or Gemini) with appropriate prompts.
        *   Responses (intent classification, extracted data, task suggestions) are processed.
    *   **Database Operations (`database.py`):**
        *   Actions use SQLAlchemy sessions obtained from `SessionManager.get_db_conn()`.
        *   Operations are typically wrapped in `db_transn` for atomic commits/rollbacks.
        *   Bulk operations (`commit_bulk_data`) are used for efficiency.
    *   **GEDCOM Processing (Action 10):**
        *   `gedcom_utils.GedcomData` loads and processes the local GEDCOM file.
        *   Functions from `gedcom_utils` and `relationship_utils` are used for searching and pathfinding.

5.  **Session Teardown/Continuation (`main.exec_actn`):**
    *   If an action fails or is configured to close the session, `SessionManager.close_sess()` is called, which may quit the browser and/or dispose of the database engine.
    *   Otherwise, the session (browser and/or database connection pool) can be kept alive for subsequent actions.

### 4.5 Hybrid Automation Rationale

*   **Robust Authentication:** Selenium, especially with `undetected_chromedriver`, is more resilient in handling complex login flows, JavaScript challenges, and 2FA, which can be difficult or impossible with direct HTTP requests alone.
*   **Efficient Data Operations:** Once authenticated and essential tokens/cookies are obtained, direct API calls using a `requests.Session` (or similar via Selenium's JS execution context for `_api_req`) are significantly faster and less resource-intensive than full page loads and HTML parsing for data retrieval.
*   **Access to Internal APIs:** The project targets Ancestry's internal APIs, which often provide structured JSON data that is easier to work with than scraping HTML.
*   **Dynamic Header Requirements:** Ancestry APIs require several dynamically generated or session-specific headers. The hybrid approach allows capturing these from an active browser session or constructing them with necessary session data.

## 5. Technical Details

### 5.1 API Authentication & Headers

Establishing a valid, authenticated session and constructing the correct HTTP headers are critical for successful API interaction.

*   **Session Initialization:**
    *   `SessionManager.start_sess()` and `SessionManager.ensure_session_ready()` orchestrate this.
    *   Selenium (`chromedriver.init_webdvr`) launches Chrome.
    *   `utils.log_in()` handles UI-based login and 2FA.
    *   Essential cookies like `ANCSESSIONID`, `SecureATT`, and consent-related cookies are extracted from the browser.
*   **Key Identifiers (Retrieved via API after login):**
    *   `my_profile_id` (UCDMID): User's global profile identifier.
    *   `my_uuid` (DNA Test GUID / `sampleId`): Identifier for the user's DNA test.
    *   `my_tree_id`: Identifier for the user's active family tree.
    *   `tree_owner_name`: Display name of the tree owner.
*   **CSRF Token (`X-CSRF-Token`):**
    *   Retrieved via an API call (e.g., to `/discoveryui-matches/parents/api/csrfToken`) or from cookies (e.g., `_dnamatches-matchlistui-x-csrf-token`).
    *   Stored in `SessionManager.csrf_token` and included in headers for state-changing requests (POST, PUT, DELETE).
*   **`ancestry-context-ube` Header:**
    *   A Base64-encoded JSON string containing contextual information. Generated by `utils.make_ube()`.
    *   Structure includes:
        *   `eventId`: Often a zero GUID.
        *   `correlatedScreenViewedId`: A new UUID per request.
        *   `correlatedSessionId`: The value of the `ANCSESSIONID` cookie.
        *   `screenNameStandard`, `screenNameLegacy`: Identifiers for the "current page" context.
        *   `userConsent`: A pipe-delimited string representing cookie consent status (e.g., "necessary|preference|performance|..."). Reliably obtaining the correct dynamic value for this is crucial and can be challenging.
        *   `vendors`, `vendorConfigurations`: Related to analytics/marketing consent.
*   **Tracking Headers (`newrelic`, `traceparent`, `tracestate`):**
    *   Related to New Relic performance monitoring and W3C Trace Context.
    *   Generated by `utils.make_newrelic()`, `utils.make_traceparent()`, `utils.make_tracestate()`. These functions create syntactically valid headers, though their functional impact on Ancestry's backend might vary.
*   **`_api_req` Function (`utils.py`):**
    *   This is the primary helper for making API calls.
    *   It takes the target URL, method, data, and other parameters.
    *   It uses the `SessionManager`'s internal `requests.Session` which has cookies synced from the Selenium browser session.
    *   It calls the `make_*` helper functions to construct the dynamic headers.
    *   It applies rate limiting and retry logic.

### 5.2 Database Schema (`database.py`)

The project uses an SQLite database managed by SQLAlchemy ORM. Key models include:

*   **`Person`**: Central table for individuals (DNA matches). Stores profile ID, UUID (DNA Sample ID), username, status (ACTIVE, DESIST, ARCHIVE, etc.), last login, and links to other tables. Supports soft deletion via `deleted_at`.
*   **`DnaMatch`**: Stores DNA-specific details for a Person, such as shared centimorgans (cM), number of segments, longest segment, and Ancestry's predicted relationship. Linked one-to-one with `Person`.
*   **`FamilyTree`**: Stores details if a DNA match is found and linked within the user's family tree, including their CFPID (Ancestry's internal ID within that tree), name in the tree, links to their "Facts" page, and the determined actual relationship path. Linked one-to-one with `Person`.
*   **`ConversationLog`**: Logs the latest INCOMING and OUTGOING messages for each conversation. Uses a composite primary key (`conversation_id`, `direction`). Stores message content (truncated), timestamp, AI-derived sentiment (for IN messages), and script message status (for OUT messages). Linked many-to-one with `Person`.
*   **`MessageType`**: A lookup table for predefined message templates used by Action 8 (e.g., "In\_Tree-Initial", "Productive\_Reply\_Acknowledgement").

A view named `messages` is created to join `ConversationLog`, `MessageType`, and `Person` for easier querying of message history.

### 5.3 AI Integration (`ai_interface.py`)

*   **Providers:** Supports "deepseek" (OpenAI-compatible API) and "gemini" (Google Gemini Pro). Configured via `AI_PROVIDER` and respective API keys in `.env`.
*   **Intent Classification (Action 7):**
    *   Uses `SYSTEM_PROMPT_INTENT` to instruct the AI.
    *   Analyzes conversation history (SCRIPT vs. USER messages).
    *   Classifies the *last user message* into: DESIST, UNINTERESTED, PRODUCTIVE, or OTHER.
*   **Data Extraction & Task Suggestion (Action 9):**
    *   Uses `EXTRACTION_TASK_SYSTEM_PROMPT`.
    *   Focuses on information shared by the USER.
    *   Extracts genealogical entities: names, locations, dates, relationships, key facts.
    *   Suggests 2-4 actionable research tasks based *only* on the conversation.
    *   Expects a structured JSON response: `{"extracted_data": {...}, "suggested_tasks": [...]}`.
    *   The `_process_ai_response` function in `action9_process_productive.py` uses Pydantic models (`ExtractedData`, `AIResponse`) for robust validation and parsing of this JSON.
*   **Microsoft To-Do Integration (`ms_graph_utils.py`, Action 9):**
    *   AI-suggested tasks can be automatically created in a specified Microsoft To-Do list.
    *   Uses MSAL for OAuth2 device code flow authentication with Microsoft Graph API.
    *   Persistent token cache (`ms_graph_cache.bin`) minimizes re-authentication.

### 5.4 Technology Stack

*   **Language:** Python 3 (as per `requirements.txt`, likely 3.7+ compatible, but 3.9+ recommended for newer features like `Path` methods).
*   **Web Automation & Interaction:**
    *   Selenium (`selenium`): For browser control and session initialization.
    *   Undetected Chromedriver (`undetected-chromedriver`): To make the controlled browser appear more like a regular user browser, helping to bypass some bot detection measures.
    *   Requests (`requests`): For making direct HTTP API calls (used by `SessionManager` and `cloudscraper`).
    *   Cloudscraper (`cloudscraper`): A specialized library built on `requests` designed to bypass Cloudflare's anti-bot protection. Used for specific API calls that might be more heavily protected (e.g., relationship probability).
*   **Database:**
    *   SQLAlchemy (`SQLAlchemy`): ORM for database interaction.
    *   SQLite: The file-based database engine used by default (via SQLAlchemy's `sqlite:///` connection string).
*   **Configuration:**
    *   python-dotenv (`python-dotenv`): For loading environment variables from a `.env` file.
*   **AI Integration:**
    *   OpenAI Client (`openai`): For interacting with OpenAI-compatible APIs like DeepSeek.
    *   Google Generative AI (`google-generativeai`): For Google's Gemini models.
    *   Pydantic (`pydantic`): For data validation and settings management, particularly for AI response parsing.
*   **Microsoft Graph API:**
    *   MSAL (`msal`): Microsoft Authentication Library for Python, used for OAuth2 authentication.
*   **GEDCOM Processing:**
    *   ged4py (`python-gedcom`): A library for parsing GEDCOM files.
*   **Data Handling & Utilities:**
    *   Diskcache (`diskcache`): For persistent, disk-based caching of function results.
    *   DateParser (`dateparser`): For flexible parsing of date strings from GEDCOM or API data.
    *   Tabulate (`tabulate`): For formatting data into clean, readable tables in the console output (used in Action 10 & 11).
    *   PSUtil (`psutil`): For system and process utilities, used here for monitoring memory usage.
    *   BeautifulSoup4 (`beautifulsoup4`): For parsing HTML, particularly useful if API responses contain HTML snippets (e.g., relationship ladder).
*   **Logging:** Standard Python `logging` module, configured by `logging_config.py`.
*   **Concurrency (Limited):** `concurrent.futures.ThreadPoolExecutor` is used in `action6_gather.py` for parallel API prefetches.

## 6. Project Structure
Use code with caution.
Markdown
.
├── Data/ # Stores database file, GEDCOMs, cache (user-created or script-created)
│ └── ancestry.db # SQLite database (created by script)
│ └── your_tree.ged # Example: User-provided GEDCOM file
│ └── ms_graph_cache.bin # MSAL token cache (created by ms_graph_utils.py)
├── Cache/ # Default directory for diskcache (created by cache.py)
├── Logs/ # Stores log files (user-created or script-created)
│ └── (various .log files) # Log files for main app, actions, utils
├── action6_gather.py # Logic for gathering DNA matches
├── action7_inbox.py # Logic for processing Ancestry inbox messages
├── action8_messaging.py # Logic for sending automated messages
├── action9_process_productive.py # Logic for processing AI-classified productive messages
├── action10.py # Logic for local GEDCOM file matching and reporting
├── action11.py # Logic for API-based person search and reporting
├── ai_interface.py # Interface for interacting with AI models
├── api_utils.py # Utilities for specific Ancestry API calls and response parsing
├── cache.py # Disk-based caching utilities and decorator
├── chromedriver.py # ChromeDriver management, Chrome options, preference reset
├── config.py # Configuration loading (Config_Class, SeleniumConfig)
├── database.py # SQLAlchemy models, database utilities, schema setup, transaction manager
├── gedcom_utils.py # Utilities for GEDCOM file parsing, data extraction, and scoring
├── logging_config.py # Centralized logging setup and custom formatters/filters
├── main.py # Main application entry point, menu system, action dispatcher
├── messages.json # Templates for automated messages (used by Action 8)
├── ms_graph_utils.py # Utilities for MS Graph API (Authentication, To-Do task creation)
├── my_selectors.py # CSS selectors for Selenium UI interaction (login, popups)
├── relationship_utils.py # Utilities for finding and formatting relationship paths (GEDCOM & API)
├── requirements.txt # Python package dependencies
├── selenium_utils.py # Selenium-specific helper functions (element interaction, cookie export)
├── .env # Environment variables (user-created, DO NOT COMMIT SENSITIVE DATA)
└── README.md # This file
## 7. Usage Guide

### 7.1 Setup

1.  **Prerequisites:**
    *   Python 3.9+ (recommended, see `requirements.txt` for specific library compatibility).
    *   Google Chrome Browser installed.
    *   An active Ancestry.com account.

2.  **Installation:**
    ```bash
    # Clone the repository (if applicable)
    # git clone <repository_url>
    # cd <project_directory>

    # Create and activate a virtual environment (recommended)
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On Linux/macOS:
    source venv/bin/activate

    # Install dependencies
    pip install -r requirements.txt
    ```

3.  **Configuration (`.env` file):**
    *   Create a file named `.env` in the project's root directory.
    *   Copy the contents from a provided `.env.example` (if available) or populate it manually.
    *   **Crucial settings include:**
        *   `ANCESTRY_USERNAME` and `ANCESTRY_PASSWORD`.
        *   `DATABASE_FILE` (e.g., `Data/ancestry.db`).
        *   `LOG_DIR` (e.g., `Logs`).
        *   `CACHE_DIR` (e.g., `Cache`).
        *   `CHROME_USER_DATA_DIR`: Path to a Chrome user data directory. This is important for `undetected-chromedriver` to potentially reuse profiles or operate with a dedicated profile. Example: `Data/ChromeProfile`.
        *   `CHROME_DRIVER_PATH`: Path to your `chromedriver.exe` (if you want to force a specific version, otherwise `undetected-chromedriver` will attempt to manage it).
        *   `BASE_URL`: (e.g., `https://www.ancestry.co.uk/`).
        *   `AI_PROVIDER`, `DEEPSEEK_API_KEY`/`GOOGLE_API_KEY`, and AI model names if using AI features.
        *   `MS_GRAPH_CLIENT_ID`, `MS_GRAPH_TENANT_ID`, `MS_TODO_LIST_NAME` if using Microsoft To-Do integration.
        *   Optional: `TREE_NAME`, `MY_PROFILE_ID`, `MY_TREE_ID`, `TESTING_PROFILE_ID`, `REFERENCE_PERSON_ID`, processing limits (`MAX_PAGES`, `MAX_INBOX`, etc.). See section 11 for more.
    *   Ensure the directories specified for `DATABASE_FILE`, `LOG_DIR`, `CACHE_DIR`, and `CHROME_USER_DATA_DIR` exist or can be created by the script. It's good practice to create `Data`, `Logs`, and `Cache` directories manually in the project root.

4.  **ChromeDriver:**
    *   `undetected-chromedriver` attempts to download and manage a compatible ChromeDriver automatically.
    *   If you encounter issues, you can manually download a ChromeDriver compatible with your Chrome browser version and specify its path in `CHROME_DRIVER_PATH` in the `.env` file.

### 7.2 Running the Application

1.  **Activate the virtual environment** (if you created one).
2.  **Run the main script:**
    ```bash
    python main.py
    ```
3.  **Menu System:**
    *   The script will display a command-line menu with various options.
    *   Enter the number corresponding to the desired action.
    *   Follow any on-screen prompts (e.g., for Microsoft Graph device code authentication if using To-Do integration for the first time).

### 7.3 Key Actions Explained

*   **Action 0 (Delete all but first):** A utility action for development/testing. Deletes most data from the database, keeping only a specific "sentinel" person record (identified by `08FA6E79-0006-0000-0000-000000000000`). *Use with extreme caution.*
*   **Action 1 (Run Actions 6, 7, 8 Sequentially):** Executes a common workflow: Gather DNA matches, process the inbox, and then send out initial/follow-up messages.
*   **Action 2 (Reset Database):** **Deletes all data** from the application's tables (except `message_types`) and re-initializes the schema. *Use with extreme caution.*
*   **Action 3 (Backup Database):** Creates a backup copy of the SQLite database file (`ancestry_backup.db`) in the `Data` directory.
*   **Action 4 (Restore Database):** Restores the database from `ancestry_backup.db`, overwriting the current database. *Use with caution.*
*   **Action 5 (Check Login Status):** Verifies if the current session is authenticated with Ancestry.com.
*   **Action 6 (Gather Matches):** (`action6_gather.coord`)
    *   Fetches your DNA match list page by page from Ancestry using APIs.
    *   Extracts relevant details for each match (shared cM, segments, tree status, profile info).
    *   Compares with existing database records.
    *   For new or significantly changed matches, fetches additional details via other APIs (e.g., relationship probability, tree linkage specifics).
    *   Performs bulk updates/inserts into the local database.
    *   Can be started from a specific page number (e.g., `6 10` to start from page 10).
*   **Action 7 (Search Inbox):** (`action7_inbox.InboxProcessor.search_inbox`)
    *   Fetches conversations from your Ancestry inbox via API.
    *   Identifies new incoming messages.
    *   Uses AI (`ai_interface.py`) to classify the intent of new messages (PRODUCTIVE, DESIST, UNINTERESTED, OTHER).
    *   Updates the `ConversationLog` and `Person` status (e.g., to DESIST) in the database.
*   **Action 8 (Send Messages):** (`action8_messaging.send_messages_to_matches`)
    *   Identifies DNA matches eligible for messaging based on their status, communication history, and tree linkage.
    *   Uses templates from `messages.json` to format personalized messages.
    *   Respects configured time intervals between follow-ups.
    *   Sends messages via Ancestry's messaging API.
    *   Logs sent messages in `ConversationLog`.
*   **Action 9 (Process Productive Messages):** (`action9_process_productive.process_productive_messages`)
    *   Processes conversations where the latest user message was classified as "PRODUCTIVE" by Action 7.
    *   Uses AI (`ai_interface.py`) to extract genealogical entities (names, dates, locations, etc.) and suggest follow-up research tasks from the conversation.
    *   Optionally creates tasks in a specified Microsoft To-Do list via MS Graph API (`ms_graph_utils.py`).
    *   Sends an acknowledgement message to the match.
    *   Updates the Person's status to ARCHIVE in the database.
*   **Action 10 (GEDCOM Report):** (`action10.run_action10`)
    *   Prompts the user for search criteria (name, birth year, etc.).
    *   Searches a local GEDCOM file (specified by `GEDCOM_FILE_PATH` in `.env`).
    *   Scores potential matches based on the criteria.
    *   Displays the top matches and, for the best match, their immediate relatives and relationship path to a configured reference person.
*   **Action 11 (API Report):** (`action11.run_action11`)
    *   Prompts the user for search criteria.
    *   Searches Ancestry's online database using various internal APIs to find matching individuals in the user's tree or public trees.
    *   Scores and ranks the suggestions.
    *   For the top candidate, fetches and displays detailed information, family members, and (if possible) the relationship path to the tree owner.
*   **t (Toggle Log Level):** Switches the console logging verbosity between INFO and DEBUG.
*   **c (Clear Screen):** Clears the console.
*   **q (Exit):** Terminates the application.

## 8. Maintenance Guide

### 8.1 API Changes & Monitoring

Ancestry.com's internal APIs are not officially documented for third-party use and **can change without notice**. This is the most significant maintenance challenge.

*   **Monitoring:**
    *   Regularly run the script's core actions (especially 6, 7, 11) to check for functionality.
    *   When errors occur, use your browser's Developer Tools (Network tab) while manually performing the failing action on Ancestry.com. Compare the requests made by your browser with those made by the script.
    *   Look for changes in:
        *   **URL Endpoints:** API paths might change. Constants are defined in `utils.py` and `api_utils.py`.
        *   **Request Parameters:** Query parameters or JSON body structures might be altered.
        *   **Required Headers:** Pay close attention to `ancestry-context-ube`, `X-CSRF-Token`, `newrelic`, `traceparent`, `User-Agent`, and `Referer`. The `userConsent` string within the UBE header is particularly sensitive to changes in Ancestry's consent management.
        *   **Response Formats:** The structure of JSON responses can change, requiring updates to parsing logic in action modules or `api_utils.py`.
*   **Adaptation:**
    *   Update URL constants and header generation logic in `utils.py` (for `_api_req` and `make_*` functions) and `api_utils.py` (for specific API wrappers).
    *   Modify JSON parsing in the relevant action modules or `api_utils.py` if response structures change.
    *   Adjust selectors in `my_selectors.py` if UI elements used for login/initial setup are modified.

### 8.2 AI Provider & Prompt Engineering

*   **Provider Updates:** If you switch AI providers (e.g., from DeepSeek to a new Gemini model or vice-versa) or if a provider updates its API:
    *   Update API keys and model names in your `.env` file and `config.py` defaults.
    *   Modify the corresponding API call logic in `ai_interface.py`.
    *   Test thoroughly using the `ai_interface.py` self-check or by running Actions 7 and 9.
*   **Prompt Effectiveness:** The effectiveness of AI classification and extraction depends heavily on the system prompts in `ai_interface.py`.
    *   If AI performance degrades, review and refine these prompts.
    *   Test changes by directly calling functions in `ai_interface.py` with example conversation contexts.
    *   Be mindful of token limits and JSON output requirements for the extraction prompt.

### 8.3 Database Schema & Migrations

*   If the database schema (`database.py` models) needs changes:
    *   Modify the SQLAlchemy model definitions.
    *   **For existing databases with data:** You will need to implement a schema migration strategy. Tools like Alembic can be integrated for this, or manual SQL `ALTER TABLE` scripts can be used for simpler changes. *Directly changing models without migrating an existing database can lead to errors or data loss.*
    *   After schema changes, run `python database.py` standalone to ensure `Base.metadata.create_all(engine)` correctly reflects the new schema (for new databases).
    *   Backup your database before making schema changes.

### 8.4 Adding New Actions or Features

1.  **Create a New Module:** Typically, `actionN_your_feature.py`.
2.  **Define Core Functionality:** Implement the main logic, accepting `SessionManager` and `config_instance` as parameters if needed.
3.  **API Helpers:** If new API endpoints are required, add corresponding wrapper functions to `api_utils.py` to keep API logic centralized.
4.  **Database Interaction:** Use `SessionManager.get_db_conn()` for database sessions and leverage existing models or add new ones to `database.py` (see schema migration note above).
5.  **Menu Integration:** Add the new action to the `menu()` function and the main dispatching logic in `main.py`.
6.  **Configuration:** Add any new required settings to `config.py` (with defaults) and document them for the `.env` file.
7.  **Standalone Runner (Optional):** Create a `run_actionN.py` script for isolated testing.

### 8.5 Dependencies

*   Keep `requirements.txt` up to date.
*   Periodically update dependencies: `pip install --upgrade -r requirements.txt` (test thoroughly after updates).

## 9. Troubleshooting

### 9.1 Common Issues & Solutions

*   **Login Failures / 2FA Loops:**
    *   **Cause:** Ancestry UI changes, incorrect credentials, outdated ChromeDriver, network issues, overly aggressive bot detection.
    *   **Solution:**
        *   Verify credentials in `.env`.
        *   Ensure `CHROME_USER_DATA_DIR` in `.env` points to a valid and writable directory. Consider using a dedicated, clean profile for the script.
        *   Let `undetected-chromedriver` manage the driver version. If issues persist, try specifying `CHROME_DRIVER_PATH` with a manually downloaded compatible version.
        *   Check selectors in `my_selectors.py` against Ancestry's current login page structure.
        *   Increase `TWO_FA_CODE_ENTRY_TIMEOUT` in `config.py` (SeleniumConfig) if manual 2FA entry is too slow.
        *   Temporarily disable headless mode (`HEADLESS_MODE=False` in `.env`) to observe the login process.
*   **API Calls Failing (401/403 Unauthorized, 429 Rate Limited, other errors):**
    *   **Cause:** Invalid/expired session cookies or CSRF token, incorrect API endpoint/parameters, malformed dynamic headers (UBE, NewRelic), aggressive rate limiting by Ancestry.
    *   **Solution:**
        *   Run Action 5 (Check Login Status) to verify session.
        *   Restart the script to establish a fresh session.
        *   Enable DEBUG logging to inspect headers sent by `_api_req` and compare with browser's network requests.
        *   Verify the `userConsent` string logic in `utils.make_ube()` if UBE-related errors occur. This is a common point of failure.
        *   Increase rate limiting delays in `.env` (`INITIAL_DELAY`, `MAX_DELAY`).
        *   Reduce `BATCH_SIZE` in `.env`.
*   **`WebDriverException` (e.g., "disconnected", "target crashed"):**
    *   **Cause:** Browser crashed, ChromeDriver lost connection, network interruption.
    *   **Solution:** The script's retry mechanisms and session validation should handle some of these. Ensure Chrome and ChromeDriver are stable. Check system resources.
*   **AI Calls Failing or Returning Unexpected Results:**
    *   **Cause:** Invalid API key, incorrect model name, AI provider API changes, poorly performing prompts, network issues to AI provider.
    *   **Solution:**
        *   Verify API keys and model names in `.env` and `config.py`.
        *   Test AI provider connectivity independently.
        *   Review and refine system prompts in `ai_interface.py`.
        *   Check `ai_interface.py` self-test.
*   **Database Errors (SQLAlchemyError, IntegrityError):**
    *   **Cause:** Schema mismatch (if models changed without DB migration), data violating constraints (e.g., duplicate unique keys), SQLite file corruption.
    *   **Solution:**
        *   Backup database.
        *   If schema changed, ensure migration or reset database (Action 2 - **data loss!**).
        *   Examine error messages for specific constraint violations.
*   **Module Not Found / Import Errors:**
    *   **Cause:** Dependencies not installed, virtual environment not activated, incorrect Python interpreter.
    *   **Solution:** Ensure `pip install -r requirements.txt` was successful in the correct environment. Activate virtual environment.

### 9.2 Effective Logging for Debugging

*   **Set Log Level:** Use the 't' option in the `main.py` menu to toggle console logging between `INFO` (default) and `DEBUG`. `DEBUG` provides much more detail. The log file level is also set in `logging_config.py` (via `setup_logging`) and can be configured.
*   **Log File Location:** Logs are typically stored in the directory specified by `LOG_DIR` in `.env` (default: `Logs/`). The main log file is often named based on the database file (e.g., `ancestry.log` if `DATABASE_FILE` is `ancestry.db`). Action-specific runners might create their own log files (e.g., `action11.log`).
*   **Key Log Messages to Look For:**
    *   `SessionManager` state changes (starting, ready, closing).
    *   `_api_req` entries showing request details (URL, method, key headers) and response status.
    *   Dynamic header generation messages from `make_ube`, `make_newrelic`, etc.
    *   Error messages from API calls, AI interactions, or database operations.
    *   `DEBUG` level often shows values being processed, selectors used, etc.

### 9.3 Debugging Tools & Techniques

*   **Browser Developer Tools:**
    *   **Network Tab:** Crucial for observing the API requests your browser makes when you manually perform an action on Ancestry.com. Compare these requests (URL, method, headers, payload, response) with what the script is attempting via `_api_req`. This is the primary way to diagnose API changes.
    *   **Console Tab:** Look for JavaScript errors on Ancestry's pages that might interfere with Selenium.
    *   **Elements Tab:** Verify CSS selectors used in `my_selectors.py` or for Selenium interactions.
*   **Database Inspection Tools:**
    *   Use an SQLite browser (e.g., "DB Browser for SQLite", DBeaver with SQLite driver) to open the `.db` file (`Data/ancestry.db`).
    *   Inspect table contents, check for data integrity, verify schema.
*   **Python Debugger (`pdb` or IDE Debugger):**
    *   Set breakpoints in the code to inspect variables and step through execution.
    *   Particularly useful for understanding data transformations and control flow within complex functions like `_api_req` or action modules.
*   **Module Self-Tests:**
    *   Many modules have self-test functionality that can be run directly (e.g., `python action7_inbox.py`) to test individual actions in isolation, simplifying debugging.
    *   Run `python <module_name>.py` for modules that have `if __name__ == "__main__":` self-test blocks (e.g., `utils.py`, `ai_interface.py`, `ms_graph_utils.py`, `selenium_utils.py`, `api_utils.py`, `gedcom_utils.py`).

## 10. Future Development Ideas

*   **Enhanced API Resilience:**
    *   Implement a more structured way to define API endpoints and their expected request/response schemas, possibly using Pydantic models. This could facilitate automated detection of some API changes.
    *   Develop a small suite of "API health check" tests that verify critical endpoints are behaving as expected.
*   **User Interface:**
    *   Develop a simple web interface (e.g., using Flask or Streamlit) for easier configuration, triggering actions, and viewing results/logs, instead of the command-line menu.
    *   Add a dashboard to visualize data collection progress, match statistics, etc.
*   **Advanced Genealogical Analysis:**
    *   Implement more sophisticated DNA match clustering algorithms (e.g., based on shared matches, "Leeds Method").
    *   Develop tools for automatically suggesting or identifying common ancestors based on tree data and DNA match information.
    *   Add features for visualizing relationship networks.
*   **AI Capabilities Expansion:**
    *   Use AI to summarize long conversation threads.
    *   Train a custom model (if feasible) for more accurate genealogical entity extraction or relationship inference.
    *   Implement AI-powered validation of tree data consistency.
    *   Explore natural language querying of the local database.
*   **Multi-Account Management:**
    *   Add functionality to manage and automate tasks for multiple Ancestry.com accounts.
*   **Improved Error Reporting:**
    *   More specific error messages to the user for common API failure scenarios.
    *   Option to automatically report certain types of errors (anonymously, if desired by user) to a central point for tracking common API breakages.
*   **Plugin System for Actions:**
    *   Refactor the action system to be more pluggable, making it easier to add new automation modules without modifying `main.py` extensively.

## 11. Configuration Reference (`.env` file)

This section details key configuration variables set in the `.env` file.

### General Settings

*   `ANCESTRY_USERNAME`: Your Ancestry.com login email.
*   `ANCESTRY_PASSWORD`: Your Ancestry.com login password.
*   `DATABASE_FILE`: Path to the SQLite database file (e.g., `Data/ancestry.db`).
*   `LOG_DIR`: Directory to store log files (e.g., `Logs`).
*   `CACHE_DIR`: Directory for `diskcache` (e.g., `Cache`).
*   `BASE_URL`: Base URL for Ancestry (e.g., `https://www.ancestry.co.uk/`).
*   `APP_MODE`: Application operational mode.
    *   `dry_run`: Logs actions, makes API calls for data retrieval, but messaging/DB writes are simulated or minimal. Good for testing API calls without side effects.
    *   `testing`: Allows more database writes and limited real actions, often with specific target profiles (`TESTING_PROFILE_ID`).
    *   `production`: Full operational mode. **Use with caution.**
*   `LOG_LEVEL`: Default logging level for console/file (e.g., `INFO`, `DEBUG`).

### Paths & Files

*   `GEDCOM_FILE_PATH`: Absolute or relative path to your GEDCOM file (used by Action 10).
*   `CHROME_USER_DATA_DIR`: Path to a Chrome user data directory. `undetected-chromedriver` uses this. It's recommended to point this to a dedicated directory (e.g., `Data/ChromeProfile`) to keep the automation browser profile separate from your main Chrome profile.
*   `PROFILE_DIR`: Name of the Chrome profile directory within `CHROME_USER_DATA_DIR` (default: `Default`).
*   `CHROME_DRIVER_PATH`: (Optional) Absolute path to `chromedriver.exe`. If not set, `undetected-chromedriver` attempts to manage it automatically.
*   `CHROME_BROWSER_PATH`: (Optional) Absolute path to `chrome.exe`. If not set, the system default is used.

### Tree & User Identifiers (Optional - script attempts to fetch these)

*   `TREE_NAME`: The exact name of your primary family tree on Ancestry. Used to fetch `MY_TREE_ID`.
*   `TREE_OWNER_NAME`: Your display name on Ancestry (used in messages).
*   `MY_PROFILE_ID`: Your Ancestry User Profile ID (UCDMID). The script attempts to fetch this.
*   `MY_TREE_ID`: The ID of your primary tree. The script attempts to fetch this if `TREE_NAME` is set.
*   `MY_UUID`: Your DNA Test Sample ID. The script attempts to fetch this.

### Testing & Reference Configuration

*   `TESTING_PROFILE_ID`: A specific Ancestry profile ID to target during `testing` mode (e.g., for sending test messages).
*   `TESTING_PERSON_TREE_ID`: A specific person's ID *within a tree* (CFPID) used for certain tests (e.g., Action 11 relationship ladder).
*   `REFERENCE_PERSON_ID`: The GEDCOM ID of the reference person (usually yourself) for relationship path calculations in Action 10.
*   `REFERENCE_PERSON_NAME`: The display name for the reference person.

### Processing Limits & Behavior

*   `MAX_PAGES`: Max DNA match pages to process in Action 6 (0 = all).
*   `MAX_INBOX`: Max inbox conversations to process in Action 7 (0 = all).
*   `MAX_PRODUCTIVE_TO_PROCESS`: Max "PRODUCTIVE" messages to process in Action 9 (0 = all).
*   `BATCH_SIZE`: Number of items (matches, messages) to process per API call batch or DB transaction.
*   `CACHE_TIMEOUT`: Default expiry for cached items in seconds (e.g., 3600 for 1 hour).
*   `TREE_SEARCH_METHOD`: Method for Action 9 tree search: `GEDCOM` (local file), `API` (Ancestry search - experimental), or `NONE`.
*   `MAX_SUGGESTIONS_TO_SCORE`: (Action 11) Max API search suggestions to score.
*   `MAX_CANDIDATES_TO_DISPLAY`: (Action 11) Max scored candidates to display in results.

### Rate Limiting & Retries

*   `MAX_RETRIES`: Default max retries for API calls.
*   `INITIAL_DELAY`: Initial delay (seconds) for `DynamicRateLimiter` and `@retry_api`.
*   `MAX_DELAY`: Maximum delay (seconds) for `DynamicRateLimiter` and `@retry_api`.
*   `BACKOFF_FACTOR`: Multiplier for increasing delay on retries/throttling.
*   `DECREASE_FACTOR`: Multiplier for decreasing delay after successful calls.
*   `TOKEN_BUCKET_CAPACITY`: Capacity of the token bucket for rate limiting.
*   `TOKEN_BUCKET_FILL_RATE`: Tokens added per second to the bucket.
*   `RETRY_STATUS_CODES`: JSON array of HTTP status codes that trigger a retry (e.g., `[429, 500, 502, 503, 504]`).

### AI Provider Configuration

*   `AI_PROVIDER`: Specifies the AI service to use.
    *   `deepseek`: For DeepSeek or other OpenAI-compatible APIs.
    *   `gemini`: For Google Gemini Pro.
    *   (blank or not set): AI features will be disabled.
*   **DeepSeek (if `AI_PROVIDER=deepseek`):**
    *   `DEEPSEEK_API_KEY`: Your API key for DeepSeek.
    *   `DEEPSEEK_AI_MODEL`: The model name (e.g., `deepseek-chat`).
    *   `DEEPSEEK_AI_BASE_URL`: The API base URL (e.g., `https://api.deepseek.com`).
*   **Google Gemini (if `AI_PROVIDER=gemini`):**
    *   `GOOGLE_API_KEY`: Your API key for Google AI Studio / Gemini.
    *   `GOOGLE_AI_MODEL`: The model name (e.g., `gemini-1.5-flash-latest`).
*   `AI_CONTEXT_MESSAGES_COUNT`: Number of recent messages to provide to AI for context.
*   `AI_CONTEXT_MESSAGE_MAX_WORDS`: Max words per message when constructing AI context string.

### Microsoft Graph API (for To-Do Integration - Action 9)

*   `MS_GRAPH_CLIENT_ID`: The Application (client) ID of your Azure AD registered application.
*   `MS_GRAPH_TENANT_ID`: The Directory (tenant) ID. For personal Microsoft accounts, often `consumers`. For organizational accounts, it's your specific tenant ID.
*   `MS_TODO_LIST_NAME`: The exact display name of the Microsoft To-Do list where tasks should be created (e.g., "Ancestry Follow-ups").

### Selenium WebDriver Configuration

*   `HEADLESS_MODE`: `True` to run Chrome headlessly, `False` for visible browser.
*   `DEBUG_PORT`: Debugging port for Chrome (used by `undetected-chromedriver`).
*   `CHROME_MAX_RETRIES`: Max attempts to initialize WebDriver.
*   `CHROME_RETRY_DELAY`: Delay (seconds) between WebDriver initialization retries.
*   `ELEMENT_TIMEOUT`, `PAGE_TIMEOUT`, `API_TIMEOUT`, etc.: Various timeout settings for Selenium waits and `requests` calls via `_api_req`.

## 12. License

[Specify license information here - e.g., MIT License, GPL, or "Proprietary - All Rights Reserved"]

*(If no license is specified, it typically defaults to "All Rights Reserved" by the author.)*

## 13. Disclaimer

This project interacts with internal Ancestry.com APIs which are not officially documented or supported for third-party use. Use this project responsibly, ethically, and at your own risk. Be mindful of Ancestry's Terms of Service. Excessive requests could potentially lead to account restrictions or other actions by Ancestry.com. The author(s) of this project assume no liability for its use or misuse. This software is provided "AS IS", without warranty of any kind, express or implied.
