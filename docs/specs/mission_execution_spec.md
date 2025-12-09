# Mission Execution Specification

## 1. Purpose

Deliver an end-to-end, automated but safety-conscious workflow to engage DNA matches, manage replies, and strengthen the family tree through validated facts and relationship evidence.

## 2. Current Capabilities (baseline)

- **Data gathering:** Action 6 scrapes DNA matches with checkpointing; Action 12 shared matches; Action 13 triangulation.
- **Inbox ingestion:** Action 7 pulls conversations, classifies intent via AI, and includes SafetyGuard critical-alert detection (regex + AI) before classification.
- **Productive message processing:** Action 9 extracts entities and creates MS To-Do tasks; GEDCOM/API lookups run for mentioned people; conversation_state table and workflow helpers exist.
- **Response assets:** `ai/context_builder.py` builds rich context (identity, genetics, history, genealogy) but is not wired into outbound messaging; `genealogy/fact_validator.py` implements conflict-aware fact validation but is unused.
- **Genealogy search:** Action 10 GEDCOM/API comparison utilities and TreeQueryService support relationship explanations.
- **Safety & state:** SafetyGuard hooks in Action 7; conversation_state model + helpers exist; DESIST handling via sentiment classification.

## 3. Gaps vs Mission

1. **Reply lifecycle orchestration** is not integrated across Actions 7/8/9; conversation_state not updated end-to-end.
2. **Automated replies** are not generated or queued from Action 9 outcomes; Action 8 lacks inbound-driven reply mode using ContextBuilder.
3. **Fact validation** results are not persisted/reviewed; FactValidator unused; no review queue CLI.
4. **Opt-out & safety enforcement** not enforced in outbound send path; per-person automation toggle missing in send checks.
5. **Engagement insights**: metrics on reply quality/engagement and content-to-outcome correlation are minimal.
6. **Research suggestions** based on ethnicity/clusters are not surfaced back into messaging.
7. **Human-in-the-loop/approval** tables and commands not implemented (MessageApproval, SystemControl from spec).

## 4. Target Architecture

- **State machine:** Use `conversation_state` as source of truth. Action 7 sets state on inbound (Initial/AwaitingReply/Productive/Desist/HumanReview). Action 9 advances to ExtractionPending/ExtractionComplete. Action 8 consumes state to decide send/queue.
- **Safety gates:** SafetyGuard blocks automation on Critical Alert; DESIST/opt-out flips `automation_enabled` (Person) and conversation_state.status to OPT_OUT.
- **Contexted reply generation:** Action 8 creates replies using ContextBuilder + Action10 relationship context; ai_interface prompt `response_generation` (json return) produces draft messages with confidence.
- **Approval & send queue:** Draft replies stored in `MessageApproval` queue with confidence/priority; auto-approve path allowed for high-confidence, non-sensitive cases; outbound send obeys SystemControl and person-level automation flags.
- **Fact validation loop:** Action 9 converts AI facts → FactValidator → SuggestedFact (approved/pending) + DataConflict; pending items appear in review CLI.
- **Metrics:** Log engagement events (opened/replied/blocked), alert counts, fact validation outcomes, approval decisions; expose to Prometheus/Grafana hooks already present.

## 5. Scope for Next Increment

- Wire FactValidator into Action 9 to persist SuggestedFacts/DataConflicts and log validation outcomes.
- Add ContextBuilder + response_generation prompt path to Action 8 for reply drafting; gate by SafetyGuard/opt-out.
- Add person-level automation checks in Action 8 (reuse messaging.workflow_helpers.can_send_message pattern once implemented).
- Create lightweight review queue CLI scaffolding to list pending SuggestedFacts and (if added) draft replies.
- Update README and docs to reflect current state and roadmap.

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
