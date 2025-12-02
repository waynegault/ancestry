# Ancestry Research Platform - Remaining Tasks

## Project Status: Phase 4 - Testing & Deployment

All core development sprints (1-4) are **COMPLETE**. Recent integration work (Dec 2025) added:
- ✅ Review Queue CLI integration (`review` command in main menu)
- ✅ Dry-Run Validation CLI (`validate` command in main menu)
- ✅ Metrics collection with quality tracking
- ✅ Real AI integration for validation drafts

---

## 🔴 Critical - Issues Discovered in Testing (Jan 2025)

### Database Connection Pool Exhaustion

**Status**: ✅ FIXED (Dec 2025)

**Symptoms**:
```
TimeoutError: QueuePool limit of size 10 overflow 5 reached, connection timed out, timeout 45.00
```

**Root Cause**: Connection pool exhausted during high-concurrency inbox processing

**Fix Applied**:
- [x] Increased base pool size from 10 to 20 in `core/database_manager.py`
- [x] Increased max_overflow from 30% to 50% of pool size
- [x] Increased pool_timeout from 45s to 60s

**Files Modified**:
- `core/database_manager.py` - Pool configuration (lines 149-151, 488-492)

---

### Cache Module Import Errors

**Status**: ✅ FIXED (Dec 2025)

**Symptoms**:
```
ModuleNotFoundError: No module named 'disk_cache'
ModuleNotFoundError: No module named 'gedcom_cache'
ModuleNotFoundError: No module named 'tree_stats_cache'
ModuleNotFoundError: No module named 'performance_cache'
ModuleNotFoundError: No module named 'cache_retention'
```

**Root Cause**: Import paths not using correct module prefix

**Fix Applied**:
- [x] `disk_cache` → `caching.cache`
- [x] `gedcom_cache` → `genealogy.gedcom.gedcom_cache`
- [x] `tree_stats_cache` → `genealogy.tree_stats_utils`
- [x] `performance_cache` → `performance.performance_cache`
- [x] `cache_retention` → `caching.cache_retention`

**Files Modified**:
- `core/cache_registry.py` - All `_lazy_call` paths corrected

---

### AI Classification Missing Intent

**Status**: ✅ FIXED (Dec 2025)

**Symptoms**:
```
WARNING: AI returned unexpected classification: 'SOCIAL'. Defaulting to OTHER.
```

**Root Cause**: AI model returns 'SOCIAL' label not defined in validation set

**Fix Applied**:
- [x] Added 'SOCIAL' to `EXPECTED_INTENT_CATEGORIES` in `ai/ai_interface.py`

**Files Modified**:
- `ai/ai_interface.py` - Added SOCIAL to valid intent categories (line 215)

---

### Session Cookie Recovery

**Status**: ✅ PARTIALLY FIXED (Dec 2025)

**Symptoms**:
```
ERROR: Essential cookies not found: ['OptanonConsent', 'trees']
ERROR: Cannot fetch shared matches: session not ready
```

**Root Cause**: Actions 12, 13 require browser session but cookies expire or aren't available

**Fix Applied**:
- [x] Added Action 13 (`fetch_shared_matches`, `shared_match`) to cookie check skip patterns
- [x] Action 12 already uses `browser_requirement=NONE` (database-only operation)

**Remaining**:
- [ ] Add graceful degradation: return "session required" instead of crash
- [ ] Implement cookie refresh flow when essential cookies missing

**Files Modified**:
- `core/session_validator.py` - Added Action 13 to skip patterns
- `core/action_registry.py` - Action 12 already correctly configured

---

### Safety Check Logging

**Status**: ✅ FIXED (Dec 2025)

**Symptoms**:
```
INFO: Safety check failed for message: User requested opt-out
```

**Analysis**: This is CORRECT behavior - opt-out detection is working

**Fix Applied**:
- [x] Changed opt-out detection log to INFO level with clearer message
- [x] Differentiated between opt-out (expected) and other safety flags (warning)
- [x] New message: "Opt-out detected for {sender_id}: skipping automated response"

**Files Modified**:
- `messaging/inbound.py` - Improved safety check logging clarity

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

**Status**: Mostly Complete

The README is comprehensive but could benefit from:

- [ ] Add "Getting Started" flowchart for new developers
- [ ] Document the full message lifecycle (Inbox → Classification → Extraction → Review → Send)
- [ ] Add troubleshooting section for common AI provider issues

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

**Status**: Implemented, Not Integrated into Main Workflow

These modules are fully implemented and tested but need UI/CLI integration:

| Feature | Module | Status | Next Step |
|---------|--------|--------|-----------|
| Triangulation Intelligence | `research/triangulation_intelligence.py` | ✅ Implemented | Add CLI command or Action 12 integration |
| Predictive Gap Detection | `research/predictive_gaps.py` | ✅ Implemented | Add CLI command for gap reports |
| Sentiment Adaptation | `ai/sentiment_adaptation.py` | ✅ Implemented | Integrate with Action 8 messaging |
| Conflict Detection | `research/conflict_detector.py` | ✅ Implemented | Add conflict review workflow |

### 5. Performance Optimizations

- [ ] Enable `TestResultCache` for faster test runs (in `run_all_tests.py`)
- [ ] Connect `--analyze-logs` CLI flag to `print_log_analysis()`
- [ ] Integrate `health_monitor.py` with SessionManager for auto-recovery

### 6. CI/CD Integration

- [ ] Add `quality_regression_gate.py` to GitHub Actions
- [ ] Set up automated test runs on PR
- [ ] Configure Grafana dashboard deployment

---

## Completed Milestones

### Sprint 1: Core Intelligence & Retrieval ✅
- TreeQueryService with fuzzy matching
- Context Builder for rich AI prompts
- GEDCOM data caching and querying

### Sprint 2: Reply Processing & Classification ✅
- Enhanced intent classification with guardrails
- Fact Extraction 2.0 with standardized objects
- Critical alert detection (regex + AI)

### Sprint 3: Response Generation & Validation ✅
- RAG Response Generator connecting to TreeQueryService
- Data Validation Pipeline with conflict detection
- SuggestedFact persistence with review status

### Sprint 4: Engagement & Safeguards ✅
- Human-in-the-Loop Review Queue
- A/B Testing Framework
- Multi-layer Opt-out Detection
- End-to-end integration tests

### Phase 1: Codebase Assessment ✅
- Module audits (action7, action9, action10)
- Data flow mapping
- Tech stack catalog
- Gap analysis

### Phase 2: Technical Specification ✅
- Reply Management System design
- Automated Response Engine architecture
- Data Validation Pipeline spec
- Engagement Optimization framework
- Human-in-the-Loop safeguards

---

## Quick Reference: Key Documentation

| Document | Purpose |
|----------|---------|
| `docs/codebase_assessment.md` | Detailed audit of core modules |
| `docs/data_flow_map.md` | Visual data flow diagrams |
| `docs/gap_analysis.md` | Missing features analysis |
| `docs/operator_manual.md` | Review queue operations guide |
| `docs/tech_stack.md` | Dependencies and infrastructure |
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
