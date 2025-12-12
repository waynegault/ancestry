# Implementation Roadmap (Dec 2025)

## Completed

- [x] Block outbound messaging when conversation_state status is OPT_OUT/HUMAN_REVIEW/PAUSED or safety_flag is set.
- [x] Contextual reply drafts remain review-first unless `contextual_reply_auto_send` is explicitly enabled.
- [x] Regenerate `docs/code_graph.json` via `scripts/update_code_graph.py` and keep repo-wide tests green.

## Next Actions (Planned In Order)

### 1) Person-Level Automation Toggle (Outbound Guard Rail)

- [ ] Add `Person.automation_enabled` boolean column (default True for backward compatibility).
  - Files: `core/database.py`, `core/database_manager.py`
- [ ] Enforce the toggle in Action 8 send gating (skip sends when disabled; still allow draft generation/queue).
  - Files: `actions/action8_messaging.py`
- [ ] Add a focused unit test proving disabled automation prevents sends.
  - Files: `actions/action8_messaging.py` (embedded `TestSuite`)

#### Acceptance Criteria (Automation Toggle)

- When `person.automation_enabled == False`, Action 8 produces `skipped (automation_disabled)` and does not call outbound send.
- Draft-only behavior remains unchanged (contextual drafts still queue when auto-send is off).
- `python run_all_tests.py --fast` passes.

### 2) Semantic Search (Tree-Aware Q&A) — Spec → Implementation

- [ ] Finalize spec wording and keep link in README.
  - Files: `docs/specs/semantic_search.md`, `readme.md`
- [ ] Implement a reusable service layer for tree-aware Q&A:
  - Parse query → intent/entities
  - Tree-first candidate retrieval (via `TreeQueryService`)
  - Evidence assembly + safe answer drafting
  - Output a structured result object (JSON-serializable)
  - Suggested location: `genealogy/semantic_search.py` or `research/semantic_search.py`
- [ ] Wire Action 7 to call semantic search for inbound questions (PRODUCTIVE only), storing results for review.
  - Files: `actions/action7_inbox.py`, `messaging/inbound.py` (or wherever InboundOrchestrator lives)
- [ ] Add tests for:
  - parse failure → clarification
  - ambiguity → clarification questions
  - candidate retrieval ranking stability for synthetic input

#### Acceptance Criteria (Semantic Search)

- Semantic search never invents facts; low evidence produces clarification questions.
- Tree-first retrieval is used; API search is optional/supplemental and still rate-limited.
- No direct writes to core person/tree fields from AI output.

### 3) Review/Approval Queue (HITL) Hardening

- [ ] Decide: implement `MessageApproval/SystemControl` tables vs extending `DraftReply`.
- [ ] Route Action 8 contextual drafts through the chosen approval queue abstraction.
- [ ] Provide minimal CLI surface to list/approve/reject queued drafts.
  - Files: `cli/maintenance.py` or `core/maintenance_actions.py` (depending on existing patterns)

#### Acceptance Criteria (Approval Queue)

- Drafts are reviewable with basic approve/reject lifecycle.
- Auto-send remains opt-in and guarded.

### 4) Inbound Fact Validation Consistency

- [ ] Apply `FactValidator`/`DataConflict` staging to Action 7 inbound fact harvests (not just Action 9).
- [ ] Surface inbound conflicts/pending facts in the review workflow.

#### Acceptance Criteria (Inbound Validation)

- Action 7 produces `SuggestedFact`/`DataConflict` consistently for extracted facts.

### 5) Observability (Minimal, No External Hard Dependencies in Tests)

- [ ] Emit counters for: drafts queued, sends blocked (opt-out/safety/automation), validation conflicts.
- [ ] Ensure automated tests do not require live Prometheus/Grafana.

#### Acceptance Criteria (Observability)

- Metrics hooks do not break offline test runs.

## Later

- [ ] Research suggestions: surface ethnicity/cluster insights in ContextBuilder + Action 8 replies (review-first) to nudge next-best questions.

## Testing

- [ ] Run `python run_all_tests.py --fast` after each completed section above.
- [ ] Validate messaging dry-runs with `contextual_reply_auto_send=false` to confirm drafts queue without sends.
- [ ] For production runs: ensure `DRY_RUN_VERIFIED=true` and keep `contextual_reply_auto_send=false` until the approval queue is operational.
