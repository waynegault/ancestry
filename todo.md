# Implementation Roadmap (Reply Management & Tree Accuracy)

Date: 2025-12-12

This file is the working implementation plan for completing the mission:

- strengthen accuracy of the family tree
- maximize verified relationship paths to DNA matches
- automate communications at scale **safely** (no harassment, no unwanted contact)
- handle replies with high quality, evidence-backed answers, and a strict human-review/safety posture


> Note on DB changes: per operator preference, this roadmap avoids schema migrations in the current iteration.
> If/when the ORM schema is updated, the expectation is to **recreate/recollect** data rather than migrate.

---

## 0) Current Capabilities (What The Codebase Can Do Today)

### Data Collection & Analysis

- Action 6: gather DNA matches (checkpoint resume, rate limiting, caching)
- Action 12: shared match scraping
- Action 13: triangulation / cluster analysis
- Action 10: GEDCOM vs Ancestry API comparison, TreeQueryService utilities

### Reply Intake + AI Triage

- Action 7: inbox ingestion via Ancestry API
- SafetyGuard:
  - opt-out detection
  - critical alert detection (self-harm, legal/privacy, threats)
- InboundOrchestrator:
  - intent classification
  - entity extraction
  - fact validation/staging to SuggestedFact + DataConflict
  - research lookups (ResearchService + relationship path)
  - draft reply generation (stored for review; not auto-sent)

### Human-in-the-loop (HITL)

- Draft message review queue based on DraftReply table
- CLI review queue UI in `cli/maintenance.py` (draft approve/reject + fact approve/reject)
- Action 8: outbound messaging with strong guardrails:
  - respects `APP_MODE` (`dry_run`, `testing`, `production`)
  - requires `DRY_RUN_VERIFIED=true` before production
  - blocks on `EMERGENCY_STOP`
  - blocks auto-approval in production unless explicitly permitted
  - respects `Person.automation_enabled` and conversation_state status/safety
- Action 11: send approved drafts (review queue -> send) with strict guardrails

### Observability

- Conversation metrics tables and helper functions
- Prometheus/Grafana hooks available

---

## 1) Gaps vs Mission (What’s Not Yet Complete)

### A) Fully automated reply handling (100% automation except true escalation)

- Draft replies exist, but the end-to-end “approve -> send reply -> mark state” workflow needs a single, explicit operator loop.
- Clarify which replies are safe to auto-approve (likely very few) and keep default review-first.

### B) Evidence-backed Q&A over your tree (RAG)

- There is tree search (GEDCOM + some API lookup), but a robust Q&A layer needs:
  - explicit citations/evidence fields in generated replies
  - consistent identity resolution (who is “ROOT”, who is the match, who is the queried ancestor)
  - confidence scoring and refusal rules

### C) Better fact extraction quality and validation

- FactValidator exists and is used, but:
  - needs consistent provenance (which message, which claim)
  - needs clearer review UX for conflicts vs suggestions
  - needs “do not mutate tree automatically” guardrails documented and enforced

### D) Engagement optimization / performance metrics

- Metrics exist but mission-oriented KPIs are not yet standardized:
  - response rate by template
  - desist rate
  - time-to-first-response
  - “useful info gained” count per 100 conversations

### E) Research suggestions in replies (ethnicity/cluster-guided prompts)

- Some tools exist (research suggestions, ethnicity analysis), but are not consistently surfaced in reply drafts.

---

## 2) Specification (Next Increment)

### 2.1 Safety & Compliance (Non-Negotiable)

- Any inbound message matching critical alert patterns triggers **HUMAN_REVIEW** and disables outbound automation for that person.
- Any opt-out request triggers **OPT_OUT** and disables outbound automation permanently for that person.
- Outbound sending must always honor:
  - `ConversationState.status in {OPT_OUT, HUMAN_REVIEW, PAUSED}`
  - `ConversationState.safety_flag`
  - `Person.status in {DESIST, BLOCKED, ARCHIVE}`
  - `Person.automation_enabled == False`
  - production guard flags (`DRY_RUN_VERIFIED`, `EMERGENCY_STOP`, auto-approve lockouts)

### 2.2 Reply Draft Generation

- Inbound-generated replies must always be stored as drafts (review-first).
- Draft creation must be idempotent per person+conversation so repeated inbox scans do not create duplicates.

### 2.3 Fact Handling

- Extracted facts are staged only as `SuggestedFact` / `DataConflict` (no automatic tree mutation).
- Suggested facts should include:
  - source message id / conversation id
  - extracted structured value
  - confidence
  - conflict classification when applicable

### 2.4 Operator Workflow

- Single daily loop:
  1) run Action 7 (ingest)
  2) review queue: approve/reject drafts, approve/reject facts
  3) run Action 11 to send only approved drafts (start in `dry_run`, then limited `production` once ready)
  4) (optional) run Action 8 for proactive outbound messaging when configured

---

## 3) Implementation Plan (No Schema Migrations)

### Completed (this iteration)

- [x] De-duplicate draft creation in `ApprovalQueueService.queue_for_review` (update-in-place if already pending)
- [x] Route Action 7 draft creation through `ApprovalQueueService` (consistent policy + dedupe)
- [x] Enforce opt-out as OPT_OUT (not HUMAN_REVIEW) and disable automation on opt-out
- [x] Run critical-alert checks before AI work in inbound processing
- [x] Add Action 11 “Send Approved Drafts” runner (send approved drafts only; mark SENT; update logs/metrics/state)

### Next (high priority)

- [x] Add a small “Reply Send Runner” that sends only **approved** drafts and marks them as SENT, with strict guardrails
- [x] Ensure conversation_state / metrics are updated when a draft is sent (or rejected)
- [x] Add refusal rules for tree Q&A (e.g., no speculation; ask clarifying questions instead)

### Next (medium priority)

- [x] Add evidence fields to reply drafts (which GEDCOM person(s) matched, why, and relationship path)
- [x] Add standardized metrics output summary for each run (counts: processed, opted out, human review, drafts created)
- [x] Improve research suggestions injection (ethnicity/cluster-based) into drafts in a review-first manner
- [x] Fix ContextBuilder genealogy: resolve GEDCOM person id via match name (avoid using DNA GUID as GEDCOM id)
- [x] Include relationship path/confidence evidence in AI context (bounded/truncated)
- [x] Include Ancestry tree relationship/path (from FamilyTree) in AI context

### Later (requires schema decisions)

- [x] Consolidate/fix ORM model duplication in `core/database.py` (clean mapping, reduce risk)
- [x] Decide whether to rebuild DB (recollect) after schema cleanup

Decision note:
- No rebuild is required for the duplication fix alone (no schema changes were introduced).
- Plan a rebuild if/when you remove/rename columns or tables, or if you do a larger `core/database.py` cleanup that intentionally changes the ORM schema.
- Safe rebuild workflow: backup DB (main menu option), delete `Data/ancestry.db`, run `python -m core.database` to recreate tables/seed templates, then re-run Actions 6/12/13 to recollect.

---

## 4) Run Safety Checklist (Go/No-Go Inputs)

Before any production sends:

- `APP_MODE=production`
- `DRY_RUN_VERIFIED=true`
- `AUTO_APPROVE_ENABLED=false` (keep off until proven)
- `ALLOW_PRODUCTION_AUTO_APPROVE=false` (keep off)
- `EMERGENCY_STOP=false`
- `REQUESTS_PER_SECOND=0.3`
- Start with `MAX_INBOX=20-50` and `MAX_SEND_PER_RUN=20-50`

