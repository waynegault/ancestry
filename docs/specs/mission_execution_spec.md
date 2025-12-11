# Mission Execution Specification

## 1. Purpose

Deliver an end-to-end, automated but safety-conscious workflow to engage DNA matches, manage replies, and strengthen the family tree through validated facts and relationship evidence.

## 2. Current Capabilities (baseline)

- **Data gathering:** Action 6 scrapes DNA matches with checkpointing; Action 12 shared matches; Action 13 triangulation.
- **Inbox ingestion:** Action 7 pulls conversations, classifies intent via AI, runs SafetyGuard critical-alert detection (regex + AI) before classification, and updates `conversation_state` (status + safety_flag) plus SuggestedFact harvest.
- **Productive message processing:** Action 9 extracts entities, converts them to `ExtractedFact` objects, validates via `FactValidator`, stages `SuggestedFact`/`DataConflict`, and creates MS To-Do tasks; GEDCOM/API lookups run for mentioned people.
- **Response assets:** `ai/context_builder.py` builds rich context (identity, genetics, history, genealogy) and Action 8 can generate contextual drafts (queued via `DraftReply`/ApprovalQueueService; auto-send optional).
- **Genealogy search:** Action 10 GEDCOM/API comparison utilities and TreeQueryService support relationship explanations.
- **Safety & state:** SafetyGuard hooks in Action 7; outbound messaging in Action 8 blocks when `conversation_state.status` is OPT_OUT/HUMAN_REVIEW/PAUSED or `safety_flag` is set; DESIST handling via sentiment classification and OptOutDetector.

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
