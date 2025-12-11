# Implementation Roadmap (Dec 2025)

## Completed

- [x] Block outbound messaging when conversation_state status is OPT_OUT/HUMAN_REVIEW/PAUSED or safety_flag is set.
- [x] Contextual reply drafts remain review-first unless `contextual_reply_auto_send` is explicitly enabled.

## Next Actions

- [ ] Review/approval system: implement MessageApproval/SystemControl tables (or extend DraftReply with priority/auto-approve), wire ApprovalQueueService + CLI, and route Action 8 contextual drafts through it.
- [ ] Automation guard rails: add a person-level automation toggle and enforce it in Action 8 alongside conversation_state + opt-out checks (set by Action 7/9 outcomes).
- [ ] Inbound fact validation: run FactValidator/DataConflict on Action 7 harvests; persist SuggestedFacts/DataConflicts and surface pending items in the review queue.
- [ ] Observability: emit reply/queue/blocked metrics plus opt-out and validation outcomes to analytics/Prometheus hooks for engagement insight.
- [ ] Research suggestions: surface ethnicity/cluster insights in ContextBuilder + Action 8 replies (review-first) to nudge next-best questions.

## Testing

- [ ] Run `python run_all_tests.py --fast` after implementing the above tasks.
- [ ] Validate messaging dry-runs with `contextual_reply_auto_send=false` to confirm drafts queue without sends.
