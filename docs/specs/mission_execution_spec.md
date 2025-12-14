# Mission Execution Specification

**Last Updated:** December 14, 2025

## 1. Purpose

Deliver an end-to-end, automated but safety-conscious workflow to engage DNA matches, manage replies, and strengthen the family tree through validated facts and relationship evidence.

### 1.1 Mission Statement

> Strengthen the accuracy of my family tree by drawing on the insights of ~16,000 DNA matches whilst minimizing manual effort. The more DNA matches placed in the tree, the more evidence to corroborate genealogical connections. Goal: 100% automated communication except human-escalation cases.

### 1.2 Mission Requirements (from user)

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Acknowledge contact & respect opt-out; never contact again | ✅ Implemented (SafetyGuard, OptOutDetector) |
| 2 | Answer questions from tree (GEDCOM/API); explain relationship paths | ⚠️ Partial (SemanticSearch + TreeQuery scaffolded) |
| 3 | Extract & validate genealogical data; raise as MS To-Do tasks | ✅ Implemented (FactValidator, Action 9) |
| 4 | Suggest research areas based on ethnicity/clusters | ⚠️ Not integrated into messaging |
| 5 | 100% automated unless escalation (visit requests, self-harm, threats) | ⚠️ Safety detection works; auto-approval not enabled |
| 6 | Performance metrics: engagement quality, code effectiveness | ⚠️ Minimal (ConversationMetrics exists, dashboards empty) |
| 7 | High-quality, non-generic, personalized messages | ✅ Implemented (ContextBuilder, templates) |
| 8 | Extract insights for tree incorporation | ⚠️ Extraction works; tree write not implemented |

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

1. **Review/approval system** still uses `DraftReply` only; MessageApproval/SystemControl tables + CLI wiring are not implemented.
2. **Automation guard rails**: person-level automation toggle is missing; outbound still relies on Person.status + opt-out detector only.
3. **Inbound fact validation** uses a basic SuggestedFact harvest in Action 7; FactValidator/DataConflict are not applied to inbound extractions or surfaced in review queues.
4. **Reply lifecycle orchestration**: conversation_state updates are not propagated from Action 8 sends, and queue/approval decisions are not tied back to conversation_state.
5. **Engagement insights**: metrics on reply quality/engagement and content-to-outcome correlation are minimal.
6. **Research suggestions** based on ethnicity/clusters are not surfaced back into messaging.

## 4. Target Architecture

- **State machine:** Use `conversation_state` as source of truth. Action 7 sets state on inbound (Initial/AwaitingReply/Productive/Desist/HumanReview). Action 9 advances to ExtractionPending/ExtractionComplete. Action 8 consumes state to decide send/queue.
- **Safety gates:** SafetyGuard blocks automation on Critical Alert; DESIST/opt-out flips `automation_enabled` (Person) and conversation_state.status to OPT_OUT.
- **Contexted reply generation:** Action 8 creates replies using ContextBuilder + Action10 relationship context; ai_interface prompt `response_generation` (json return) produces draft messages with confidence.
- **Approval & send queue:** Draft replies stored in `MessageApproval` queue with confidence/priority; auto-approve path allowed for high-confidence, non-sensitive cases; outbound send obeys SystemControl and person-level automation flags.
- **Fact validation loop:** Action 9 converts AI facts → FactValidator → SuggestedFact (approved/pending) + DataConflict; pending items appear in review CLI.
- **Metrics:** Log engagement events (opened/replied/blocked), alert counts, fact validation outcomes, approval decisions; expose to Prometheus/Grafana hooks already present.

## 5. Scope for Next Increment

- Align review queue and controls: implement MessageApproval/SystemControl tables (or extend DraftReply) with priority/auto-approve, expose minimal CLI commands, and route Action 8 contextual drafts through it.
- Enforce automation guard rails: add a person-level automation toggle and honor it in Action 8 alongside conversation_state + opt-out.
- Extend inbound validation loop: run FactValidator/DataConflict on Action 7 harvests and surface pending items in the review queue.
- Observability: emit engagement/reply metrics (sent/queued/blocked, opt-outs, validation conflicts) to existing analytics tables and Prometheus hooks.
- Research suggestions: thread ethnicity/cluster insights into ContextBuilder/Action 8 replies (still review-first).

## 6. Acceptance Criteria (increment)

- Running Action 9 on PRODUCTIVE messages produces SuggestedFact rows with conflict metadata and logs validation counts.
- Reply drafting in Action 8 uses ContextBuilder and refuses to send when safety/opt-out engaged; drafts saved (not auto-sent) pending approval.
- CLI command surfaces pending SuggestedFacts (and draft replies if queued) with approve/reject stub actions (no-ops allowed if send blocked).
- Documentation (README, code_graph metadata, todo.md) describes capabilities, gaps, and plan.

## 7. Risks & Mitigations

- **Rate limiting:** All new API calls must go through SessionManager/api_manager and respect 0.3 RPS; avoid new parallelism.
- **Data integrity:** No direct Person/Match mutation from AI facts; only SuggestedFact/DataConflict with review.
- **Safety:** Critical alerts must block outbound; ensure Action 8 checks conversation_state.status/automation flags before send.

## 8. Out-of-scope (future)

- Full automated tree updates from validated facts.
- Ethnicity-driven research suggestions auto-inserted into messaging.
- Full SystemControl/MessageApproval persistent models (unless implemented in this increment).
