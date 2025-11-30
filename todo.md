# Project Todo List - Intelligent Conversation Management

## Phase 1: Capability Audit & Gap Analysis (Completed)
- [x] Review actions/action10.py, ai/ai_interface.py, genealogy/research_service.py.
- [x] Review messaging/inbound.py, messaging/safety.py.
- [x] Identify missing RAG, Harvester, and Reply Generation logic in InboundOrchestrator.

## Phase 2: Database Schema (Completed)
- [x] ConversationState table exists.
- [x] SuggestedFact table exists.
- [x] ConversationMetrics table exists.

## Phase 3: Inbound Orchestrator Enhancement (The 'Reply Engine') (Completed)
- [x] **RAG Integration**:
  - Update messaging/inbound.py to use extract_genealogical_entities to identify search subjects.
  - Integrate ResearchService to search for these subjects in the GEDCOM/Tree.
  - If found, calculate relationship path.
  - Call generate_genealogical_reply with the found context.
- [x] **Harvester Implementation**:
  - In messaging/inbound.py, if extract_genealogical_entities returns new facts, create SuggestedFact records in the DB.
- [x] **Reply Generation**:
  - Store the generated reply in ConversationLog (as a draft or sent message).
  - Return the reply in the process_message result.

## Phase 4: Metrics & Observability (Completed)
- [x] **Update Metrics**:
  - In messaging/inbound.py, update ConversationMetrics (msg counts, engagement score).
  - Create EngagementTracking events for 'message_received', 'reply_generated', 'fact_extracted'.

## Phase 5: Blue Sky Innovation - Triangulation Hypothesis Generator (Completed)
- [x] Create genealogy/triangulation.py.
- [x] Implement logic to find shared matches who might be related through a specific ancestor.
- [x] Generate hypothesis messages.

## Phase 6: Testing (Completed)
- [x] Update messaging/test_inbound.py to test RAG, Harvester, and Reply flows.
- [x] Ensure 100% pass rate.
- [x] ensure messaging\inbound.py follows the same pattern and format  of tests as the rest of teh codebase

## Phase 7: Future Enhancements
- [x] **Draft Persistence**:
  - Add a `draft_reply` column to `conversation_logs` or a dedicated table to store them for human review before sending.
- [x] **Triangulation Action**:
  - Create a new Action 12 to run triangulation analysis on DNA matches using the `TriangulationService`.

## Phase 8: Triangulation Implementation

- [x] **Database Schema**:
  - Add `SharedMatch` table to `database.py`.
  - Update `Person` model with relationship.
- [x] **Shared Match Retrieval**:
  - Implement `_get_shared_matches` in `genealogy/triangulation.py` to query the new table.
  - (Optional) Create a script/action to populate this table from Ancestry API (out of scope for now, but the table is needed).
- [x] **Triangulation Logic**:
  - Update `find_triangulation_opportunities` to use real data.
- [x] **Test Infrastructure**:
  - Address "Test Infrastructure Todo #17" (Shared test helpers).
  - Address "Test Infrastructure Todo #18" (Smart ordering).

## Phase 9: Shared Match Collection & Advanced Analysis

- [x] **Shared Match Scraper**:
  - Update `action6_gather.py` (or create new action) to fetch shared matches for specific DNA matches (e.g., > 20cM).
  - Store shared matches in `SharedMatch` table.
- [x] **Data Enrichment**:
  - Ensure tree data is fetched for shared matches to enable common ancestor identification.
- [x] **Reporting**:
  - Add CSV/HTML export to Action 12 for triangulation results.
- [x] **Advanced Filtering**:
  - Filter triangulation opportunities by cM range and confidence.
