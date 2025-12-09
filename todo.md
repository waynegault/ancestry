# Implementation Plan

## Phase 1 – Documentation & Planning

- [x] Capture current capabilities and gaps versus mission.
- [x] Publish mission execution spec (`docs/specs/mission_execution_spec.md`).
- [ ] Update `docs/code_graph.json` metadata to reflect latest documentation refresh.
- [ ] Keep README aligned with current state and roadmap.

## Phase 2 – Reply Lifecycle Foundations

- [ ] Integrate `genealogy.fact_validator.FactValidator` into Action 9 to persist `SuggestedFact`/`DataConflict` results and log validation metrics.
- [ ] Ensure Action 7 updates `conversation_state` consistently for PRODUCTIVE/DESIST/HUMAN_REVIEW (reuse workflow_helpers), including safety flag propagation.
- [ ] Add person-level automation/opt-out checks before any send path (Action 8), blocking on DESIST/HUMAN_REVIEW/safety.

## Phase 3 – Contexted Reply Drafting

- [ ] Wire `ai.context_builder.ContextBuilder` into Action 8 to build match context and call a `response_generation` prompt in `ai_prompts.json`.
- [ ] Produce draft replies (not auto-send) with confidence and reasoning; save drafts for review queue consumption.
- [ ] Guard replies with SafetyGuard results and opt-out flags; fall back to no-op when safety trips.

## Phase 4 – Review & HITL Controls

- [ ] Implement lightweight review CLI to list/approve/reject pending `SuggestedFact` items (and reply drafts if queued).
- [ ] Add persistence for message approvals (minimal schema if not present) or stub queue logging for now.
- [ ] Expose metrics: counts of critical alerts, opt-outs, validated facts, approvals, sends.

## Phase 5 – Research Guidance & Insights

- [ ] Surface ethnicity/cluster-based research suggestions into reply drafts where data exists.
- [ ] Correlate engagement metrics with prompt variants and content (A/B hooks already in `ai/ab_testing.py`).

## Phase 6 – Testing & Validation

- [ ] Add TestSuite-covered unit tests for FactValidator integration, reply drafting guards, and review CLI entry points.
- [ ] Run targeted test modules affected by changes and update docs with results.
