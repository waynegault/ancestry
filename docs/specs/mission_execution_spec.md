# Mission Execution Specification

**Last Updated:** December 17, 2025

## 1. Purpose

Deliver an end-to-end, automated but safety-conscious workflow to engage DNA matches, manage replies, and strengthen the family tree through validated facts and relationship evidence.

### 1.1 Mission Statement

> Strengthen the accuracy of my family tree by drawing on the insights of ~16,000 DNA matches whilst minimizing manual effort. The more DNA matches placed in the tree, the more evidence to corroborate genealogical connections. Goal: 100% automated communication except human-escalation cases.

### 1.2 Mission Requirements (from user)

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Acknowledge contact & respect opt-out; never contact again | ✅ Implemented (SafetyGuard, OptOutDetector, Person.automation_enabled) |
| 2 | Answer questions from tree (GEDCOM/API); explain relationship paths | ✅ Implemented (SemanticSearchService, TreeQueryService, response_generation prompt) |
| 3 | Extract & validate genealogical data; raise as MS To-Do tasks | ✅ Implemented (FactValidator, Action 9, DataConflict, SuggestedFact) |
| 4 | Suggest research areas based on ethnicity/clusters | ✅ Implemented (ContextBuilder._build_research_insights, triangulation, gap detection) |
| 5 | 100% automated unless escalation (visit requests, self-harm, threats) | ✅ Implemented (ApprovalQueueService.is_auto_approve_ready, gradual rollout, config toggles) |
| 6 | Performance metrics: engagement quality, code effectiveness | ✅ Implemented (4 Grafana dashboards, Prometheus metrics, EngagementTracking, ConversationMetrics) |
| 7 | High-quality, non-generic, personalized messages | ✅ Implemented (ContextBuilder, research suggestions, relationship context) |
| 8 | Extract insights for tree incorporation | ✅ Implemented (Phase 8: TreeUpdateService, SuggestedFact→GEDCOM, test_tree_update_integration.py) |

## 2. Current Capabilities (baseline)

### 2.1 Data Gathering (PRODUCTION READY)
- **Action 6:** Scrapes DNA matches with checkpointing, ethnicity comparison, tree data extraction
- **Action 12:** Shared matches for high-cM matches
- **Action 13:** DNA triangulation and cluster analysis

### 2.2 Inbox Processing (PRODUCTION READY)
- **Action 7:**
  - SafetyGuard critical-alert detection (regex + AI) runs BEFORE any AI work
  - Intent classification: PRODUCTIVE, DESIST, ENTHUSIASTIC, CASUAL_CHAT, OTHER
  - Entity extraction: names, dates, places, relationships
  - SuggestedFact harvest from inbound messages
  - ConversationState updates (status + safety_flag)

### 2.3 Productive Message Processing (PRODUCTION READY)
- **Action 9:**
  - Converts entities to `ExtractedFact` objects
  - Validates via `FactValidator` with conflict detection
  - Stages `SuggestedFact` and `DataConflict` records
  - Creates MS To-Do tasks from productive conversations
  - GEDCOM/API lookups for mentioned people

### 2.4 Response Generation (IMPLEMENTED, NEEDS INTEGRATION)
- **ContextBuilder:** Assembles rich context (identity, genetics, history, genealogy)
- **Action 8:**
  - Generates contextual drafts
  - Queues via `DraftReply`/ApprovalQueueService
  - Respects safety blocks (OPT_OUT, HUMAN_REVIEW, PAUSED)
  - Auto-send is DISABLED by default

### 2.5 Genealogy Search (PRODUCTION READY)
- **Action 10:** GEDCOM/API comparison with unified scoring
- **TreeQueryService:** Person lookup, relationship path calculation
- **SemanticSearchService:** Question detection, evidence-backed candidate retrieval

### 2.6 Safety & State (PRODUCTION READY)
- **SafetyGuard:** Critical alerts block automation immediately
- **OptOutDetector:** Multi-layer pattern matching for opt-out detection
- **ConversationState:** Status tracking (ACTIVE, OPT_OUT, HUMAN_REVIEW, PAUSED)
- **DraftReply queue:** All messages require approval before send

## 3. Gaps vs Mission

All core mission requirements are now implemented. Remaining enhancements for future consideration:

1. **Auto-approval activation**: Infrastructure complete (`ApprovalQueueService.is_auto_approve_ready()`), but requires 100+ human-reviewed drafts with 95%+ acceptance rate before enabling `AUTO_APPROVE_ENABLED=true`.
2. **Tree update activation**: `TreeUpdateService` implemented and tested; requires production validation before enabling automated GEDCOM writes.
3. **Research suggestion visibility**: Ethnicity/cluster insights threaded into ContextBuilder; consider surfacing in UI for operator awareness.
4. **Dashboard deployment**: 4 Grafana dashboards exist in `docs/grafana/`; deploy via `python scripts/deploy_dashboards.py`.
5. **Conversation state propagation**: Action 8 updates conversation_state on send; consider adding more granular state transitions.

## 4. Target Architecture

- **State machine:** Use `conversation_state` as source of truth. Action 7 sets state on inbound (Initial/AwaitingReply/Productive/Desist/HumanReview). Action 9 advances to ExtractionPending/ExtractionComplete. Action 8 consumes state to decide send/queue.
- **Safety gates:** SafetyGuard blocks automation on Critical Alert; DESIST/opt-out flips `automation_enabled` (Person) and conversation_state.status to OPT_OUT.
- **Contexted reply generation:** Action 8 creates replies using ContextBuilder + Action10 relationship context; ai_interface prompt `response_generation` (json return) produces draft messages with confidence.
- **Approval & send queue:** Draft replies stored in `MessageApproval` queue with confidence/priority; auto-approve path allowed for high-confidence, non-sensitive cases; outbound send obeys SystemControl and person-level automation flags.
- **Fact validation loop:** Action 9 converts AI facts → FactValidator → SuggestedFact (approved/pending) + DataConflict; pending items appear in review CLI.
- **Metrics:** Log engagement events (opened/replied/blocked), alert counts, fact validation outcomes, approval decisions; expose to Prometheus/Grafana hooks already present.

## 5. Scope for Next Increment

All items from the original scope have been implemented:

| Item | Status | Implementation |
|------|--------|----------------|
| Review queue and controls | ✅ Done | `ApprovalQueueService`, `cli/review_queue.py`, `cli/facts_queue.py` |
| Person-level automation toggle | ✅ Done | `Person.automation_enabled` column, honored in Action 8 |
| Inbound fact validation | ✅ Done | `FactValidator` integrated in `messaging/inbound.py`, `DataConflict` on conflicts |
| Observability metrics | ✅ Done | Prometheus metrics, 4 Grafana dashboards, `EngagementTracking` model |
| Research suggestions | ✅ Done | `ContextBuilder._build_research_insights()` with ethnicity/cluster/triangulation |

**Future enhancements** (optional):
- Enable auto-approval after sufficient human review baseline
- Activate tree updates in production after validation
- Add real-time dashboard alerts for safety events

## 6. Acceptance Criteria (increment)

All acceptance criteria have been met:

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Action 9 produces SuggestedFact with conflict metadata | ✅ Met | `FactValidator` integration, `DataConflict` creation |
| Action 8 uses ContextBuilder, respects safety | ✅ Met | `automation_enabled` check, safety blocks honored |
| CLI surfaces pending facts/drafts | ✅ Met | `cli/review_queue.py`, `cli/facts_queue.py` with list/approve/reject |
| Documentation updated | ✅ Met | This spec, README, copilot-instructions.md |

## 7. Risks & Mitigations

- **Rate limiting:** All new API calls must go through SessionManager/api_manager and respect 0.3 RPS; avoid new parallelism.
- **Data integrity:** No direct Person/Match mutation from AI facts; only SuggestedFact/DataConflict with review.
- **Safety:** Critical alerts must block outbound; ensure Action 8 checks conversation_state.status/automation flags before send.

## 8. Out-of-scope (future)

Items moved from out-of-scope to implemented:
- ~~Full automated tree updates from validated facts~~ → ✅ Implemented (Phase 8: TreeUpdateService)
- ~~Ethnicity-driven research suggestions auto-inserted into messaging~~ → ✅ Implemented (ContextBuilder research insights)

Remaining future enhancements:
- Real-time Grafana alerting for safety events
- A/B testing framework for message templates
- Multi-tree support (currently single GEDCOM focus)
