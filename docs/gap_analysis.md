# Gap Analysis: Ancestry Research Automation Platform

## 1. "True" RAG Retrieval Pipeline

### Current State

The current system relies on structured data processing and keyword-based searching.

- **`actions/action10.py`**: Performs advanced GEDCOM analysis using graph traversal and structured relationship pathfinding.
- **`api/api_search_core.py`**: Executes searches against the Ancestry API using specific field filters (name, birth year, location).
- **`genealogy/gedcom/`**: Parsers handle standard GEDCOM tags but treat notes and stories as raw text strings.

### Missing Capabilities

- **Semantic Search**: There is no mechanism to search for concepts (e.g., "ancestors who were mariners" or "stories about migration") unless specific keywords match exactly.
- **Vector Embeddings**: The system does not generate or store vector embeddings for conversation history, notes, or biographical details.
- **Context Retrieval**: When the AI answers a question, it relies on the immediate conversation context and structured lookups, missing relevant unstructured context from other parts of the tree or past conversations.

### Recommended Implementation

1. **Vector Database**: Integrate a local vector store (e.g., **ChromaDB** or **FAISS**) to index:
   - GEDCOM notes and biographical sketches.
   - Past conversation logs (`ConversationLog`).
   - Extracted facts and stories.
2. **Embedding Pipeline**: Create a service in `ai/rag_service.py` to generate embeddings (using OpenAI or a local model like `sentence-transformers`) whenever data is imported or updated.
3. **Retrieval Action**: Implement a new retrieval step in `ai_interface.py` that queries the vector store for relevant context before generating a response, injecting the retrieved text into the prompt.

---

## 2. Formal "Conflict Detection"

### Current State

- **`actions/action9_process_productive.py`**: Uses the `extraction_task` prompt to parse user messages into structured JSON.
- **`ai_prompts.json`**: Includes a `family_tree_verification` prompt that asks the AI to identify conflicts textually.
- **`database.py`**: Stores `Person` and `FamilyTree` records but lacks specific structures for tracking data discrepancies.

### Missing Capabilities

- **Automated Comparison**: There is no code logic that systematically compares the *extracted* values (from Action 9) against the *stored* values in the database.
- **Conflict Persistence**: When the AI identifies a conflict (e.g., "User claims birth 1880, Tree says 1885"), this insight is likely lost in the logs or just printed, rather than stored as a structured "Conflict" record.
- **Validation Logic**: No formal validation layer prevents contradictory data from being automatically merged.

### Recommended Implementation

1. **Conflict Model**: Add a `DataConflict` model to `database.py` with fields for `entity_id`, `field_name`, `existing_value`, `new_value`, `source`, and `status` (Open, Resolved, Ignored).
2. **Comparison Service**: Create `research/conflict_detector.py`. After extraction in Action 9, pass the new data to this service to compare against the `Person` table.
3. **Alerting**: If a high-confidence conflict is found, create a `DataConflict` record instead of automatically updating the person, and flag it for the Review Queue.

---

## 3. Dedicated "Review Queue"

### Current State

- **`main.py`**: Provides a CLI menu for triggering automated actions (Gather, Inbox, Messaging).
- **`database.py`**: `PersonStatusEnum` exists but is primarily used for research status (e.g., "Researching", "Complete"), not for a data entry workflow.
- **Workflow**: Data extracted from conversations is often converted directly into tasks or updates without a dedicated "holding area" for human approval.

### Missing Capabilities

- **Staging Area**: No UI or CLI state where extracted data sits pending approval.
- **Approval Workflow**: No mechanism to "Accept", "Reject", or "Edit" an AI-proposed update before it commits to the database.
- **Visual Diff**: The user cannot see a side-by-side comparison of "Current vs. Proposed" data.

### Recommended Implementation

1. **Review Status**: Add a `review_status` column to `Person` or a separate `StagedUpdate` table in `database.py`.
2. **CLI Interface**: Add a "Review Queue" option to `ui/menu.py` and `main.py`.
3. **Review Action**: Create `actions/action_review.py` that:
   - Fetches all pending `DataConflict` or `StagedUpdate` records.
   - Displays a diff to the user.
   - Accepts input to Apply (merge data) or Reject (discard).

---

## 4. A/B Testing Framework

### Current State

- **`ai/ai_interface.py`**: The `call_ai` function accepts a `variant` parameter, allowing manual selection of prompt versions.
- **`ai/prompt_telemetry.py`**: Logs the `variant_label` and performance metrics (parse success, quality score).
- **`ai/ai_prompts.json`**: Stores prompt text and versions.

### Missing Capabilities

- **Traffic Splitting**: No logic to automatically route a percentage of requests (e.g., 20%) to a "Challenger" prompt variant.
- **Experiment Configuration**: No file or config to define active experiments (e.g., "Experiment: Greeting V2, Split: 50/50").
- **Success Correlation**: While telemetry tracks *extraction* quality, it doesn't easily correlate a prompt variant with downstream *business* success (e.g., "Did this prompt lead to a reply from the user?").

### Recommended Implementation

1. **Experiment Config**: Create `config/experiments.json` to define active experiments (e.g., `{"intent_classification": {"control": "v1", "challenger": "v2", "split": 0.5}}`).
2. **Experiment Manager**: Implement `ai/experiment_manager.py` to read the config and randomly select variants in `call_ai`.
3. **Outcome Tracking**: Enhance `prompt_telemetry.py` to log a `session_id` or `conversation_id` so that later events (like a user reply) can be joined back to the prompt variant that generated the previous message.

---

## 5. Technical Debt

### 1. Logic Duplication

**Findings:**
- **GEDCOM Loading**: Logic to load and cache GEDCOM data is repeated in `actions/action10.py`, `genealogy/gedcom/gedcom_utils.py`, and `genealogy/gedcom/gedcom_cache.py`. While `gedcom_cache.py` attempts to centralize this, other modules still occasionally bypass it or re-implement loading checks.
- **Person Search**: `actions/action9_process_productive.py` contains inline logic for searching the database that mirrors functionality in `research/person_lookup_utils.py`.
- **Data Models**: Pydantic models for genealogical concepts (e.g., `ExtractedPerson`, `ExtractedEvent`) are defined inside `actions/action9_process_productive.py` but represent core domain entities that should be shared.

**Recommendation:**
- **Centralize GEDCOM Access**: Enforce a single entry point (`GedcomService`) for all GEDCOM operations.
- **Unified Search Service**: Refactor `action9` to use `research/person_lookup_utils.py` instead of ad-hoc queries.
- **Shared Domain Models**: Move Pydantic models to a new `core/models/` package to be shared across actions.

### 2. Error Handling

**Findings:**
- **Bare Except Blocks**: Several modules contain `except:` or `except Exception:` blocks that catch all errors without specific handling or re-raising.
  - `actions/action6_gather.py`: Multiple instances in the main loop to prevent crashing, but this can mask underlying logic errors.
  - `actions/action7_inbox.py`: Generic error catching in message parsing loops.
- **AI Parsing Fragility**: While `ai_interface.py` has retry logic, the downstream consumers (like `action9`) often assume the returned JSON is semantically valid even if it passes syntax checks.

**Recommendation:**
- **Specific Exceptions**: Replace bare excepts with specific exception handling (`NoSuchElementException`, `TimeoutException`, `JSONDecodeError`).
- **Validation Layer**: Implement a "Semantic Validator" for AI responses that checks for logical consistency (e.g., death date > birth date) before processing.

### 3. Code Structure ("God Classes")

**Findings:**
- **`actions/action6_gather.py`**: This file is over 4,000 lines long. It handles UI navigation, API requests, database persistence, and error recovery.
- **`core/session_manager.py`**: While central, it has grown to handle too many disparate responsibilities (browser, DB, API, config).

**Recommendation:**
- **Refactor Action 6**: Split into `gather/navigator.py`, `gather/processor.py`, and `gather/persistence.py`.
- **Decompose SessionManager**: Delegate specific responsibilities to `BrowserService`, `DatabaseService`, and `APIService`, keeping `SessionManager` as a thin coordinator.


### Logic Duplication

- **GEDCOM Loading & Caching**:
  - **Issue**: Logic to load and cache GEDCOM data is triplicated across `actions/action9_process_productive.py`, `actions/action10.py`, and `genealogy/research_service.py`.
  - **Impact**: Inconsistent caching behavior, increased memory usage, and maintenance overhead.
  - **Recommendation**: Centralize all GEDCOM operations in `genealogy/research_service.py`. Refactor Action 9 and Action 10 to inject and use this service.

- **Person Search Logic**:
  - **Issue**: `actions/action9_process_productive.py` imports `filter_and_score_individuals` directly from `actions/action10.py` inside a method, creating tight coupling. `genealogy/research_service.py` implements a similar `search_people` method.
  - **Impact**: Circular dependency risks and fragmented search logic.
  - **Recommendation**: Consolidate person search logic into `ResearchService.search_people` and have all actions use this single entry point.

- **Data Models**:
  - **Issue**: Pydantic models (`ExtractedData`, `VitalRecord`, etc.) are defined inside `actions/action9_process_productive.py` but represent core domain concepts needed by other modules.
  - **Impact**: Prevents reuse of validation logic and data structures.
  - **Recommendation**: Move these models to a shared location like `genealogy/models.py` or `core/models.py`.

### Error Handling

- **Bare Except Blocks**:
  - **Issue**: Generic `except Exception:` blocks found in `actions/action7_inbox.py` (lines 890, 1545), `actions/action9_process_productive.py` (line 2942), and multiple locations in `utils.py`.
  - **Impact**: Masks specific errors (like `KeyboardInterrupt` or `SystemExit`), makes debugging difficult, and can hide critical failures.
  - **Recommendation**: Replace with specific exception handling (e.g., `WebDriverException`, `SQLAlchemyError`, `ValidationError`). Log the full traceback when catching generic exceptions is absolutely necessary.

- **AI Parsing Robustness**:
  - **Issue**: AI response parsing in `action9` relies on `_process_ai_response` without explicit `try...except ValidationError` blocks visible in the main flow.
  - **Impact**: Malformed JSON from the AI could crash the action or result in generic error logging without actionable details.
  - **Recommendation**: Wrap Pydantic model validation in explicit try/except blocks to handle `ValidationError` gracefully, potentially triggering a retry with a "correction" prompt.

### Code Structure

- **God Classes**:
  - **Issue**: Several files are excessively large and handle too many responsibilities:
    - `actions/action7_inbox.py` (~4100 lines)
    - `actions/action9_process_productive.py` (~4000 lines)
    - `actions/action10.py` (~3450 lines)
    - `utils.py` (~4300 lines)
  - **Impact**: High cognitive load for developers, difficult to test, and prone to merge conflicts.
  - **Recommendation**: Refactor these into packages (e.g., `actions/inbox/`, `actions/productive/`) and split logic into smaller, focused modules (e.g., `inbox_api.py`, `inbox_processor.py`, `message_classifier.py`).

- **Hardcoded Configuration**:
  - **Issue**: Configuration values and magic strings (e.g., `PRODUCTIVE_SENTIMENT`, `EXCLUSION_KEYWORDS`, `FAMILY_INFO_KEYWORDS`) are hardcoded in action files.
  - **Impact**: Changing business rules requires code changes.
  - **Recommendation**: Move these values to `config/` or `ai_prompts.json` (for text-based rules) to separate configuration from logic.
