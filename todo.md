# Ancestry Research Platform - Remaining Tasks

## Project Status: Phase 4 - Testing & Deployment

All core development sprints (1-4) are **COMPLETE**. Recent integration work (Dec 2025) added:
- ✅ Review Queue CLI integration (`review` command in main menu)
- ✅ Dry-Run Validation CLI (`validate` command in main menu)
- ✅ Metrics collection with quality tracking
- ✅ Real AI integration for validation drafts

---





## High Priority - Pre-Production Validation

### 1. Dry-Run Validation (Blocking Production)

**Status**: Ready to Execute

The system is now ready for validation against real historical data.

**CLI Commands**:
- Main menu: Type `validate` to run with default 50 conversations
- Direct: `python scripts/dry_run_validation.py --limit 50`
- Export results: `python scripts/dry_run_validation.py --export results.json`

**Remaining Tasks**:
- [ ] Run against 50+ historical PRODUCTIVE conversations
- [ ] Manual audit comparing AI-generated drafts vs actual human replies
- [ ] Document edge cases and failure modes
- [ ] Measure extraction quality scores (target: median >70)
- [ ] Verify opt-out detection catches all DESIST patterns

**Acceptance Criteria**:
- 90%+ parse success rate
- Median quality score >70
- Zero false-negative opt-out detections
- All generated drafts pass human review

---

## Medium Priority - Documentation & Polish

### 2. README.md Architecture Update

**Status**: ✅ FIXED (Jan 2025)

The README is comprehensive but could benefit from:

- [x] Add "Getting Started" flowchart for new developers
- [x] Document the full message lifecycle (Inbox → Classification → Extraction → Review → Send)
- [x] Add troubleshooting section for common AI provider issues

### 3. Operator Training Documentation

**Status**: Complete

The operator manual exists at `docs/operator_manual.md` and covers:
- Review Queue CLI commands
- Approval workflow
- A/B testing management
- Opt-out management
- Emergency controls
- Monitoring & metrics

---

## Low Priority - Future Enhancements

### 4. Innovation Features Integration

**Status**: ✅ INTEGRATED (Jan 2025)

These modules are fully implemented and tested and have been integrated into the CLI:

| Feature | Module | Status | Next Step |
|---------|--------|--------|-----------|
| Triangulation Intelligence | `research/triangulation_intelligence.py` | ✅ Integrated | Added CLI command (Action 14) |
| Predictive Gap Detection | `research/predictive_gaps.py` | ✅ Integrated | Added CLI command (Action 14) |
| Sentiment Adaptation | `ai/sentiment_adaptation.py` | ✅ Integrated | Added CLI command (Action 14) |
| Conflict Detection | `research/conflict_detector.py` | ✅ Integrated | Added CLI command (Action 14) |

### 5. Performance Optimizations

- [ ] Enable `TestResultCache` for faster test runs (in `run_all_tests.py`)
- [ ] Connect `--analyze-logs` CLI flag to `print_log_analysis()`
- [ ] Integrate `health_monitor.py` with SessionManager for auto-recovery

### 6. CI/CD Integration

- [ ] Add `quality_regression_gate.py` to GitHub Actions
- [ ] Set up automated test runs on PR
- [ ] Configure Grafana dashboard deployment

---



## Quick Reference: Key Documentation

| Document | Purpose |
|----------|---------|
| `docs/specs/codebase_assessment.md` | Detailed audit of core modules |
| `docs/specs/data_flow_map.md` | Visual data flow diagrams |
| `docs/specs/gap_analysis.md` | Missing features analysis |
| `docs/specs/operator_manual.md` | Review queue operations guide |
| `docs/specs/tech_stack.md` | Dependencies and infrastructure |
| `docs/specs/data_validation_pipeline.md` | Fact validation technical spec |
| `docs/specs/engagement_optimization.md` | A/B testing schema |
| `docs/specs/human_in_the_loop.md` | Approval workflow spec |
| `docs/specs/reply_management.md` | Conversation state machine |
| `docs/specs/response_engine.md` | Context builder architecture |

---

## Next Actions (Prioritized)

1. **Dry-run validation** - Run `scripts/dry_run_validation.py` against 50 historical conversations
2. **Manual audit** - Review AI-generated drafts vs actual replies
3. **Document findings** - Update README with validation results
4. **Production deployment** - Enable review queue for live messages

## Refactoring & Technical Debt (Q1 2026)

### Complexity Management

- [ ] **Refactor `_process_single_page` Orchestration**: The `actions/gather/orchestrator.py` logic relies on complex state dictionaries and hooks. Refactor into a proper `PageProcessor` class with typed state.
- [ ] **Simplify `nav_to_page` Logic**: `utils.py` navigation logic is split across multiple helpers. Consolidate and simplify the retry/backoff mechanism.
- [ ] **Refine Exception Handling**: Replace broad `except Exception` blocks in `action6_gather.py` and `orchestrator.py` with specific exception handling (e.g., `SQLAlchemyError`, `WebDriverException`) to prevent masking root causes.

### Technical Debt Reduction

- [ ] **Modularize Large Actions**: Split `action6_gather.py` (6400+ lines) and `action7_inbox.py` (4200+ lines) into smaller, domain-specific sub-modules (e.g., `actions/action6/database.py`, `actions/action6/api.py`).
- [ ] **Centralize Menu Logic**: Refactor `main.py` to use a data-driven registry for menu actions instead of manual `set_action_function` calls, reducing metadata duplication.
- [ ] **Async/Sync Bridging**: Review `DatabaseManager` usage to minimize thread-pool switching overhead and evaluate native async driver support for future scaling.
- [ ] **Centralize GEDCOM Access**: Enforce a single entry point (`GedcomService`) for all GEDCOM operations, removing duplication in `action10.py` and `gedcom_utils.py`.
- [ ] **Shared Domain Models**: Move Pydantic models from `action9` to `core/models/` to allow reuse across actions.

### Security Enhancements

- [ ] **PII Redaction**: Implement `logging.Filter` to mask emails/phones in `app.log`.
- [ ] **Telemetry Privacy**: Hash `scoring_inputs` in `prompt_telemetry.py` to prevent PII leakage in `Logs/prompt_experiments.jsonl`.
- [ ] **Opt-Out Hardening**: Introduce `force_stop` flag for immediate blocking, independent of acknowledgment logic.
