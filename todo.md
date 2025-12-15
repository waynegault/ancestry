# Ancestry Automation Platform - Implementation Roadmap

**Last Updated:** December 15, 2025 (Session 6: venv activation, code graph fixes)
**Status:** Active Development
**Mission:** Strengthen family tree accuracy through automated DNA match engagement with 100% AI-driven communication (except human-escalation cases)
**Review Status:** ~103,000+ lines reviewed across 55+ modules

---

## Immediate Tasks

### Code Graph Visualization (Priority: HIGH)
- [ ] **Fix visualize_code_graph.html** - Graph view not rendering properly
  - Currently using vis-network library but graph doesn't display
  - Need to debug: check if vis-network CDN is loading, verify node/link data structure
  - Consider fallback to D3.js or simpler canvas-based rendering
  - Test with smaller dataset first (filter by type or area)

---

## Executive Summary

This roadmap aligns the codebase with the mission of maximizing DNA match engagement while minimizing manual effort. The system currently excels at data gathering, safety detection, and draft generation but requires work on reply management, automated Q&A, and engagement analytics to achieve fully autonomous operation.

### Current Maturity Assessment

| Capability | Status | Confidence |
|------------|--------|------------|
| DNA Match Gathering | ✅ Production | High |
| Inbox Processing + Safety | ✅ Production | High |
| Entity Extraction | ✅ Production | Medium |
| GEDCOM/API Tree Search | ✅ Production | High |
| Context-Aware Draft Generation | ✅ Implemented | Medium |
| Fact Validation Pipeline | ✅ Implemented | Medium |
| Semantic Search (Q&A) | ✅ Implemented | Low |
| Approval Queue + Web UI | ✅ Implemented | Medium |
| Self-Message Prevention | ✅ Implemented | High |
| Message Personalization | ✅ Production | High |
| Rate Limiting | ✅ Production | High |
| Circuit Breakers | ✅ Production | High |
| Opt-Out Detection | ✅ Production | High |
| Health Checks | ✅ Implemented | Medium |
| Automated Reply Loop | ⚠️ Partial | Low |
| Tree-Aware Answer Generation | ⚠️ Partial | Low |
| Engagement Analytics | ⚠️ Scaffolded | Low |
| Auto-Approval Path | ⚠️ Scaffolded | Low |
| Scheduled Jobs | ⚠️ Not Implemented | Low |
| Research Integration | ⚠️ Scaffolded Only | Low |
| Feature Flags | ⚠️ Underutilized | Low |
| A/B Testing | ⚠️ Disconnected | Low |
| PII Redaction | ✅ Wired (Env Config) | Low |
| Dependency Injection | ⚠️ Underutilized | Low |
| Async Database Ops | ⚠️ Available | Low |
| GEDCOM Intelligence | ⚠️ Not Integrated | Low |
| DNA-GEDCOM Cross-Ref | ⚠️ Not Integrated | Low |

---

## Phase 1: Reply Management Foundation (Priority: CRITICAL)

**Goal:** Establish end-to-end reply lifecycle from inbound message to sent response

### 1.1 Conversation State Machine Hardening
- [ ] Verify state transitions per `reply_management.md` spec
- [x] Add `automation_enabled` column to Person table (person-level toggle) ✅ IMPLEMENTED
- [x] Ensure Action 8 respects `automation_enabled` + `conversation_state.status` ✅ IMPLEMENTED
- [ ] Add state transition logging for auditability

### 1.2 Review Queue Integration
- [x] Wire DraftReply queue to Action 7 inbound processing ✅ IMPLEMENTED (InboundOrchestrator)
- [ ] Implement idempotent draft creation (update existing pending draft per person/conversation)
- [ ] Add confidence scoring to drafts from ContextBuilder output
- [x] Create CLI commands: `list`, `review <id>`, `approve`, `reject`, `edit` ✅ PARTIAL (ApprovalQueueService exists)
- [x] Create Web UI for draft review ✅ IMPLEMENTED (review_server.py on localhost:5000)

### 1.3 Send Loop Completion
- [x] Action 11: Mark DraftReply as SENT after successful send ✅ IMPLEMENTED
- [ ] Update ConversationState.status after send (ACTIVE → AwaitingReply)
- [ ] Log engagement event for analytics

### 1.4 Opt-Out Enforcement
- [x] On DESIST detection: set Person.automation_enabled=False, ConversationState.status=OPT_OUT ✅ IMPLEMENTED
- [x] Block all future outbound for OPT_OUT persons ✅ IMPLEMENTED
- [ ] Add opt-out acknowledgment message generation (one final polite closure)

---

## Phase 1.5: Draft Quality Guards (Priority: CRITICAL)

**Goal:** Prevent AI from generating obviously incorrect or inappropriate drafts

### 1.5.1 Self-Messaging Prevention
- [x] Add pre-draft check: block if recipient.profile_id == owner_profile_id ✅ IMPLEMENTED
- [x] Add pre-draft check: block if recipient.uuid == owner_uuid ✅ IMPLEMENTED (via profile_id check)
- [x] Log and alert when self-message attempt detected ✅ IMPLEMENTED
- [ ] Add to SafetyGuard as SELF_MESSAGE category

### 1.5.2 Context Accuracy Validation
- [ ] Pre-draft validation: verify mentioned ancestors exist in OUR tree, not theirs
- [ ] Cross-reference extracted names against TreeQueryService before drafting
- [ ] Flag drafts where AI mentions facts already known to recipient
- [ ] Add AI prompt instruction: "Do not explain relationships the recipient already knows"

### 1.5.3 AI-Powered Draft Review
- [ ] Create `draft_quality_check` prompt in ai_prompts.json with:
  - Self-reference detection ("Am I the sender?")
  - Context inversion detection ("Is this explaining their own ancestor to them?")
  - Obvious error patterns (wrong relationship direction, deceased person as living)
- [ ] Run quality check as post-generation validation before queuing
- [ ] Auto-reject drafts that fail quality check with reason logged

### 1.5.4 Auto-Correction Pipeline
- [ ] When quality check fails, attempt one AI regeneration with explicit correction
- [ ] If regeneration also fails, route to HUMAN_REVIEW with error context
- [ ] Track correction success rate for prompt improvement

### 1.5.5 Discovered Issues Log (December 2025)
| Issue | Example | Root Cause | Fix Priority |
|-------|---------|------------|--------------|
| Self-message | Draft #3 sent to Wayne | No owner identity check | ✅ FIXED |
| Relationship inversion | Draft #1 explained ancestor already in their tree | AI context confusion | HIGH |

---

## Phase 1.6: Send Workflow Hardening (Priority: HIGH)

**Goal:** Ensure reliable, auditable message delivery with proper state management

### 1.6.1 Transaction Safety ✅ IMPLEMENTED
- [x] Wrap `_send_single_approved_draft` in explicit transaction (currently commits mid-loop)
- [x] Add rollback on partial failure (e.g., log created but metrics update failed)
- [x] Ensure draft status stays APPROVED if send API fails
- **Implementation:** Core operations wrapped in try/except with `db_session.rollback()` on failure; non-critical operations (engagement event, metrics) isolated with separate error handling

### 1.6.2 Draft Expiration ✅ IMPLEMENTED
- [x] Implement draft expiration logic (expires_at field calculated but unused)
- [x] Add scheduled job to mark PENDING → EXPIRED after 72 hours
- [x] Surface expired drafts in review queue with different styling
- **Implementation:** Added `expires_at` column to DraftReply model; `ApprovalQueueService.queue_for_review()` now stores expiration timestamp; `expire_old_drafts()` enhanced to use expires_at with fallback to created_at for legacy drafts

### 1.6.3 Duplicate Send Prevention ✅ IMPLEMENTED
- [x] Add guard: skip if draft already SENT (prevent re-processing on retry)
- [x] Check for recent OUT log before sending (idempotency window)
- [x] Log duplicate attempt instead of sending again
- **Implementation:** New `_check_duplicate_send()` function checks draft.status=="SENT" and queries ConversationLog for recent OUT messages within 5-minute idempotency window

### 1.6.4 ConversationState Synchronization ✅ IMPLEMENTED
- [x] Update ConversationState.status after successful send (ACTIVE → AwaitingReply)
- [x] Set ConversationState.last_outbound_at timestamp
- [x] Verify state machine transitions per reply_management.md spec
- **Implementation:** Enhanced `_touch_conversation_state_after_send()` to update `conversation_phase="awaiting_reply"`, refresh `updated_at`, respect hard-stop states (OPT_OUT, HUMAN_REVIEW, PAUSED, COMPLETED), with debug logging

---

## Phase 2: Tree-Aware Q&A System (Priority: HIGH)

**Goal:** Answer genealogical questions from matches using GEDCOM/API evidence

### 2.1 Semantic Search Enhancement
- [ ] Integrate `SemanticSearchService.search()` into InboundOrchestrator for PRODUCTIVE messages with questions
- [ ] Store `SemanticSearchResult` in ConversationState.ai_summary (or dedicated field)
- [ ] Pass search results to ContextBuilder for draft generation

### 2.2 Evidence-Backed Answers
- [ ] Extend `TreeQueryService.find_person()` with fuzzy birth-year tolerance (±5 years)
- [ ] Add `get_family_members()` method: parents, siblings, spouses, children
- [ ] Implement `explain_relationship()` for DNA match → common ancestor path
- [ ] Format evidence blocks for inclusion in AI prompts

### 2.3 Answer Generation
- [ ] Create `response_generation` prompt in ai_prompts.json with:
  - Evidence citation requirements
  - Uncertainty disclosure rules
  - Follow-up question generation
- [ ] Return structured JSON: draft_message, confidence, missing_information, suggested_facts
- [ ] Route low-confidence answers to HUMAN_REVIEW

### 2.4 Relationship Explanation
- [ ] Use ThruLines data (if scraped) or GEDCOM path calculation
- [ ] Generate natural-language explanations: "We both descend from [Ancestor] through [Path]"
- [ ] Include relationship labels: "3rd cousin twice removed via the Smith line"

---

## Phase 3: Fact Extraction & Validation (Priority: HIGH)

**Goal:** Extract, validate, and stage genealogical facts for tree improvement

### 3.1 Fact Extraction 2.0
- [ ] Standardize all entity extraction to output `ExtractedFact` objects
- [ ] Add `from_conversation()` factory for common patterns
- [ ] Include date normalization (circa, before, after qualifiers)
- [ ] Track extraction confidence per fact type

### 3.2 Conflict Detection
- [ ] Run `FactValidator.validate_fact()` on Action 9 extracted entities
- [ ] Stage conflicts in `DataConflict` table with severity classification
- [ ] Generate SuggestedFact rows for non-conflicting high-confidence facts

### 3.3 Review Queue for Facts
- [ ] Add CLI commands: `facts list`, `facts review <id>`, `facts approve/reject`
- [ ] Surface pending SuggestedFacts in main review queue
- [ ] Track approval rates for quality metrics

### 3.4 MS To-Do Integration
- [ ] Create tasks for MAJOR_CONFLICT items requiring research
- [ ] Include conflict details in task description
- [ ] Tag tasks with fact type and person name

---

## Phase 4: Engagement Analytics (Priority: MEDIUM)

**Goal:** Measure and optimize communication effectiveness

### 4.1 Metrics Collection
- [ ] Track per-conversation: messages_sent, messages_received, response_rate
- [ ] Calculate time_to_first_response
- [ ] Record opt-out_count, productive_count, human_review_count

### 4.2 Quality Scoring
- [ ] Score draft quality: 0-100 based on personalization, evidence, specificity
- [ ] Track acceptance_rate for drafts (approved vs rejected)
- [ ] Compare quality scores to engagement outcomes

### 4.3 Dashboard Integration
- [ ] Emit metrics to Prometheus hooks (already scaffolded)
- [ ] Create Grafana dashboard panels:
  - Response funnel (Sent → Replied → Productive → Fact Extracted)
  - Opt-out trends over time
  - Draft quality distribution
  - Review queue size and aging

### 4.4 Content-to-Outcome Correlation
- [ ] A/B test message templates (formal vs friendly)
- [ ] Track which prompt variants produce higher response rates
- [ ] Log experiment results for offline analysis

---

## Phase 5: Research Suggestions (Priority: MEDIUM)

**Goal:** Proactively suggest research areas based on match characteristics

### 5.1 Ethnicity-Based Suggestions
- [ ] Identify shared ethnicity regions between owner and match
- [ ] Surface region-specific research suggestions in drafts
- [ ] Link to relevant surname clusters

### 5.2 Gap-Based Suggestions
- [ ] Run PredictiveGapDetector on match tree (if available)
- [ ] Include gap-filling suggestions in follow-up messages
- [ ] Prioritize suggestions by research impact

### 5.3 Cluster Analysis
- [ ] Group matches by shared match patterns
- [ ] Identify cluster "anchors" with confirmed tree placement
- [ ] Suggest cluster-wide research hypotheses

---

## Phase 6: Human Escalation Handling (Priority: HIGH)

**Goal:** Correctly identify and route cases requiring personal attention

### 6.1 Escalation Categories
- [ ] VISIT_REQUEST: "I'd like to visit you" → Immediate human review
- [ ] SELF_HARM: Suicide/distress signals → CRITICAL alert, no response
- [ ] LEGAL_THREAT: Attorney mentions → Log and pause
- [ ] ARTIFACT_DISCOVERY: Family Bible/photos → Priority notification

### 6.2 Notification System
- [ ] Add email/SMS alert option for CRITICAL alerts
- [ ] Daily digest of HUMAN_REVIEW items
- [ ] Emergency stop: global pause switch

### 6.3 Response Guidelines
- [ ] Draft empathetic responses for escalation-path cases (for human editing)
- [ ] Never auto-send escalation responses
- [ ] Log all escalation decisions

---

## Phase 7: Auto-Approval Path (Priority: MEDIUM)

**Goal:** Enable safe auto-sending for high-confidence, routine messages

### 7.1 Auto-Approval Criteria
- [ ] quality_score >= 85
- [ ] opt_out_score >= 95 (safety)
- [ ] No aggressive sentiment
- [ ] Person.automation_enabled == True
- [ ] ConversationState.status == ACTIVE
- [ ] Not flagged for manual review

### 7.2 Gradual Rollout
- [ ] Start with DRY_RUN mode (save but don't send)
- [ ] Manual review first 100 auto-approved drafts
- [ ] Enable live sending only after 95%+ approval rate

### 7.3 Safety Rails
- [ ] Daily send limit per person (default: 1)
- [ ] Cooldown period between messages (default: 7 days)
- [ ] Emergency pause if opt-out rate > 5%

---

## Phase 8: Tree Update Automation (Priority: LOW - FUTURE)

**Goal:** Automatically incorporate validated facts into the tree

### 8.1 Prerequisites
- [ ] Ancestry API write access (may require authentication refresh)
- [ ] GEDCOM write utilities
- [ ] Full conflict resolution workflow

### 8.2 Implementation
- [ ] Route APPROVED SuggestedFacts to tree update queue
- [ ] Create audit log for all tree modifications
- [ ] Implement rollback capability

### 8.3 Validation
- [ ] Post-update verification against API
- [ ] Detect and alert on update failures

---

## Phase 9: Observability & Monitoring (Priority: MEDIUM)

**Goal:** Production-grade visibility into system health and engagement outcomes

### 9.1 Prometheus Metrics Integration ✅ IMPLEMENTED
- [x] Emit `drafts_queued_total` counter (by priority, confidence bucket)
- [x] Emit `drafts_sent_total` counter (by outcome: sent/skipped/error)
- [x] Emit `review_queue_depth` gauge (by status: PENDING/APPROVED/EXPIRED)
- [x] Add `response_rate` histogram (time from sent to reply received)
- **Implementation:** Added proxy classes and Prometheus metrics in metrics_registry.py; wired into approval_queue.py (queue_for_review, get_queue_stats) and action11 (send loop)

### 9.2 Grafana Dashboard Panels
- [ ] Response funnel: Sent → Replied → Productive → Fact Extracted
- [ ] Opt-out trend over time (daily/weekly)
- [ ] Draft quality score distribution
- [ ] Review queue age histogram (hours since creation)

### 9.3 Alerting Rules
- [ ] Alert if opt-out rate exceeds 5% in 24h window
- [ ] Alert if review queue depth > 50 for > 24 hours
- [ ] Alert on circuit breaker trips (API/session failures)
- [ ] Alert on emergency_stop_enabled activation

---

## Phase 10: Scheduled Jobs & Background Automation (Priority: MEDIUM)

**Goal:** Enable hands-off operation with periodic maintenance tasks

### 10.1 Draft Lifecycle Management ✅ IMPLEMENTED
- [x] Create `expire_old_drafts` scheduled job (call `ApprovalQueueService.expire_old_drafts()`)
- [x] Run every 6 hours or on application startup
- [x] Log expired draft count to metrics
- **Implementation:** Added `run_startup_maintenance_tasks()` to core/lifecycle.py; called from main.py after check_startup_status(); runs expire_old_drafts() on every app startup with count logging

### 10.2 Inbox Polling
- [ ] Add periodic inbox check (every 15-30 minutes when app is running)
- [ ] Process new inbound messages through InboundOrchestrator
- [ ] Queue drafts for review automatically

### 10.3 Session Maintenance
- [ ] Implement session keepalive for long-running operations

---

## Phase 11: Research Module Integration (Priority: MEDIUM)

**Goal:** Leverage existing research infrastructure to enhance draft quality

### Current State Assessment
The following modules are **fully implemented** but **not integrated** into the draft generation pipeline:

| Module | Lines | Status | Gap |
|--------|-------|--------|-----|
| triangulation_intelligence.py | 724 | ✅ Implemented | Not called from draft generation |
| conflict_detector.py | 547 | ✅ Implemented | Not called from fact review |
| predictive_gaps.py | 827 | ✅ Implemented | Not surfaced in drafts |
| message_personalization.py | 1987 | ✅ Implemented | 30+ personalization functions available |

### 11.1 Triangulation Integration ✅ IMPLEMENTED
- [x] Call `TriangulationIntelligence.generate_hypothesis()` during ContextBuilder assembly
- [x] Include confidence scores in draft context
- [x] Surface triangulation opportunities in follow-up message suggestions
- [ ] Add `MatchCluster` detection to group related matches
- **Implementation:** Added `_build_triangulation_hypothesis()` to ContextBuilder; hypothesis includes proposed_relationship, common_ancestor, confidence_score, evidence_count; formatted in `_format_research()` for AI prompts

### 11.2 Conflict Detection Integration ✅ IMPLEMENTED
- [x] Wire `ConflictDetector.detect_conflicts()` to Action 9 fact extraction
- [x] Route HIGH/CRITICAL severity conflicts to review queue
- [ ] Add conflict resolution workflow to operator manual
- [ ] Surface resolved conflicts as tree improvement candidates
- **Implementation:** Added `ConflictSeverityEnum` to database.py; added `severity` column to DataConflict model; added `_map_conflict_severity()` and enhanced `_stage_conflict_if_needed()` in action9; added `get_critical_conflicts()` query method to ConflictDetector; HIGH/CRITICAL conflicts logged with warning for visibility

### 11.3 Predictive Gap Integration ✅ IMPLEMENTED
- [x] Call `PredictiveGapDetector.analyze_gaps()` for match tree analysis
- [x] Include `ResearchGap` suggestions in draft personalization
- [ ] Prioritize brick wall research in message content
- [ ] Track gap resolution through conversation outcomes
- **Implementation:** Added `_build_predictive_gaps()` to ContextBuilder; surfaces top research gap with type, description, suggested actions; formatted in `_format_research()` for AI prompts

### 11.4 Personalization Enhancement ✅ IMPLEMENTED
- [x] Enable full personalization function registry in MessagePersonalizer
- [x] Add A/B testing for personalization strategies
- [x] Track effectiveness metrics per personalization type
- [x] Optimize based on response rates
- **Implementation:** Integrated ExperimentManager with MessagePersonalizer for A/B testing of personalization strategies (DNA-focused vs standard, research-heavy vs brief); added 4 Prometheus metrics (personalization_usage, personalization_effectiveness, personalization_ab_assignment, personalization_ab_outcome); enhanced `track_message_response()` to record effectiveness per function; added `get_experiment_summary()` for visibility; auto-optimization via `_get_ab_test_insights()` in recommendations

---

## Phase 12: GEDCOM/DNA Intelligence Integration (Priority: MEDIUM)

**Goal:** Leverage advanced GEDCOM analysis and DNA cross-referencing for enriched research

### Current State Assessment
The following modules are **fully implemented** but **not integrated** into the main workflow:

| Module | Lines | Status | Gap |
|--------|-------|--------|-----|
| gedcom_intelligence.py | 951 | ✅ Implemented | AI analysis not called from draft generation |
| dna_gedcom_crossref.py | 807 | ✅ Implemented | Cross-reference not wired to personalization |
| gedcom_cache.py | ~400 | ✅ Implemented | Caching available but not widely used |
| dna_utils.py | ~300 | ✅ Implemented | Utility functions available |

### 12.1 GEDCOM Intelligence Integration ✅ IMPLEMENTED
- [x] Call `GedcomIntelligenceAnalyzer.analyze_gaps()` during ContextBuilder assembly
- [x] Surface `GedcomGap` findings in draft personalization
- [x] Include `GedcomConflict` warnings in human review notes
- [ ] Route `ResearchOpportunity` items to MS To-Do tasks
- **Implementation:** Added `_build_gedcom_intelligence()` to ContextBuilder; surfaces top gap, conflict (with severity), and research opportunity; formatted in `_format_research()` for AI prompts with ⚠️ warning for critical/major conflicts

### 12.2 DNA-GEDCOM Cross-Reference Integration ✅ IMPLEMENTED
- [x] Call `CrossReferenceService.find_matches()` during match processing
- [x] Include relationship path confidence in draft context
- [x] Surface cross-reference validation in draft quality scoring
- [ ] Track cross-reference success rate for analytics
- **Implementation:** Added `_build_dna_gedcom_crossref()` to ContextBuilder; uses DNAGedcomCrossReferencer to validate DNA match against GEDCOM tree; surfaces top match with confidence, conflicts, and verification opportunities; formatted in `_format_research()` with ✓ for high-confidence validated matches

### 12.3 GEDCOM Cache Optimization ✅ IMPLEMENTED
- [x] Enable GEDCOM caching for repeated tree access
- [x] Add cache warming during startup for owner's tree
- [x] Implement cache invalidation on tree updates
- **Implementation:** Added GEDCOM cache warming to `run_startup_maintenance_tasks()` in lifecycle.py via `preload_gedcom_cache()`; added `invalidate_gedcom_cache_on_update()` for explicit invalidation; cache also auto-invalidates based on file mtime hash; tests: ALL PASSED (13/13 gedcom_cache, 6/6 lifecycle)

---

## Phase 13: Infrastructure Modernization (Priority: LOW)

**Goal:** Leverage advanced infrastructure patterns for better maintainability

### 13.1 Dependency Injection Expansion ✅ IMPLEMENTED
- [x] Expand DI container usage beyond session_utils.py (currently only 2 imports)
- [x] Register AIProviderManager in DI container
- [x] Register ApprovalQueueService in DI container
- [ ] Add DI integration guide to developer documentation
- **Implementation:** Added AIProviderManager as singleton and ApprovalQueueService as factory (requires DB session) to `configure_dependencies()` in dependency_injection.py; tests: ALL PASSED (19/19)

### 13.2 Async Database Operations ✅ IMPLEMENTED
- [x] Utilize `async_session_context()` for I/O-bound operations
- [x] Add async support to ApprovalQueueService batch operations
- [ ] Benchmark async vs sync for gather operations
- [ ] Document when to use async patterns
- **Implementation:** Added 3 async methods to ApprovalQueueService: `async_get_queue_stats()`, `async_get_pending_queue()`, `async_expire_old_drafts()`; uses run_in_executor for thread-pool execution; tests: 12/12 PASSED

### 13.3 Protocol-Based Testing ✅ IMPLEMENTED
- [x] Create mock implementations based on core/protocols.py
- [x] Add protocol-based dependency injection for testing
- [x] Reduce concrete type dependencies in tests
- **Implementation:** Created `testing/protocol_mocks.py` with 5 mock implementations (MockRateLimiter, MockDatabaseSession, MockSessionManager, MockCache, MockLogger) satisfying RateLimiterProtocol, DatabaseSessionProtocol, SessionManagerProtocol, CacheProtocol, LoggerProtocol; includes verify_protocol_compliance() validation; tests: 6/6 PASSED

---

## Technical Debt & Quality Tasks

### Code Quality
- [ ] Add missing type hints to action modules
- [x] Resolve remaining # type: ignore comments (see check_type_ignores.py) ✅ FIXED: 1 occurrence in message_personalization.py
- [x] Update dead_code_candidates.json and clean stale code ✅ UPDATED: 136 candidates across 199 files
- [x] Wire unused recovery decorators in error_handling.py (ancestry_session_recovery, ancestry_api_recovery) ✅ IMPLEMENTED
- [ ] Remove TODO/FIXME comments by completing referenced tasks (83 found across codebase)
- [x] Address placeholder implementations in triangulation_intelligence.py line 390 ✅ IMPLEMENTED: Now queries SharedMatch table properly
- [x] Clean up stub classes in relationship_utils.py (StubTag, StubIndi - test-only code in production file) ✅ REVIEWED: These are local dataclasses inside _test_gedcom_path_conversion() function, properly scoped; follows standard pattern of embedded tests

### Testing
- [ ] Add integration tests for full inbound→reply flow
- [ ] Add tests for SemanticSearchService end-to-end
- [ ] Add tests for FactValidator conflict detection
- [x] Populate empty `tests/` directory or remove it ✅ ADDED: tests/README.md explaining embedded test pattern
- [ ] Add Action 11 transaction failure recovery tests
- [ ] Add tests for triangulation_intelligence.py hypothesis scoring
- [ ] Add tests for conflict_detector.py field comparison logic
- [ ] Add tests for predictive_gaps.py gap detection heuristics

### Documentation
- [ ] Update copilot-instructions.md with Phase 2 patterns
- [x] Add operator manual for review queue ✅ EXISTS (docs/specs/operator_manual.md - 512 lines)
- [ ] Create architecture diagram for reply flow
- [x] Document Web UI review interface (review_server.py) ✅ EXISTS (operator_manual.md covers CLI and Web UI)
- [x] Add Web UI section to operator_manual.md (localhost:5000 workflow) ✅ CREATED: Section 9 with full usage guide
- [ ] Document MS Graph integration setup (integrations/ms_graph_utils.py - 813 lines)
- [ ] Add troubleshooting guide for common errors

### Error Handling
- [x] Implement recovery strategies referenced in error_handling.py header comments ✅ WIRED: action_runner._ensure_required_state() uses ancestry_session_recovery, ancestry_api_recovery, ancestry_database_recovery on failure
- [x] Add circuit breaker integration to Action 11 send loop ✅ IMPLEMENTED: SessionCircuitBreaker with threshold=5, 5min recovery
- [x] Create error categorization for send failures (network vs auth vs rate limit) ✅ IMPLEMENTED: SendErrorCategory enum with categorize_send_error() function

### Research Module Integration
- [x] Wire TriangulationIntelligence into draft generation (724 lines ready but not called) ✅ ALREADY WIRED: ContextBuilder._build_triangulation_hypothesis() uses TriangulationIntelligence.analyze_match()
- [x] Wire ConflictDetector into fact review workflow (547 lines ready but not called) ✅ ALREADY WIRED: Action 9 uses ConflictDetector for severity mapping; GEDCOM intelligence surfaces conflicts
- [x] Wire PredictiveGapDetector suggestions into message personalization (827 lines ready) ✅ ALREADY WIRED: ContextBuilder._build_predictive_gaps() uses PredictiveGapDetector
- [x] Add research module status to ContextBuilder output ✅ ALREADY IMPLEMENTED: _build_research_insights() includes triangulation, research_gaps, gedcom_intelligence, dna_gedcom_crossref

### Core Infrastructure Gaps
- [x] Wire FeatureFlags (594 lines) into action modules for gradual rollout ✅ IMPLEMENTED: Action 11 now uses ACTION11_SEND_ENABLED flag; bootstrap registers 3 default flags
- [x] Enable PII redaction filter (523 lines) in production logging ✅ WIRED: logging_config.py adds PIIRedactionFilter to file handler; enabled via PII_REDACTION_ENABLED=true
- [x] Integrate HealthCheckRunner into startup validation (currently menu action only) ✅ WIRED: lifecycle.py calls run_startup_health_checks() in initialize_application()
- [x] Wire ConversationAnalytics (892 lines) events into InboundOrchestrator ✅ ALREADY WIRED: _update_metrics() tracks EngagementTracking events (message_received, reply_generated, facts_extracted)
- [x] Connect A/B testing framework (612 lines) to prompt selection ✅ ALREADY WIRED: ai_interface.py uses get_prompt_with_experiment(); MessagePersonalizer uses ExperimentManager for strategy A/B tests

### CLI Enhancement
- [x] Add `cli/review_queue.py` module (referenced in operator_manual.md but uses approval_queue.py instead) ✅ CREATED: CLI with list, view, approve, reject, stats commands
- [ ] Consolidate ResearchToolsCLI (1210 lines) - many lazy-load patterns but good coverage
- [x] Add progress indicators for long-running CLI operations ✅ ADDED: tqdm progress for ethnicity batch analysis in cli/research_tools.py

### CI/CD Enhancements
- [x] GitHub Actions for tests ✅ EXISTS (.github/workflows/tests.yml)
- [x] GitHub Actions for quality gate ✅ EXISTS (.github/workflows/quality-gate.yml)
- [x] GitHub Actions for lint/typecheck ✅ EXISTS (.github/workflows/lint-typecheck.yml)
- [x] Add Docker workflow for containerized testing ✅ CREATED: .github/workflows/docker-test.yml
- [x] Add production checklist validation to CI (scripts/check_production_guard.py) ✅ CREATED: .github/workflows/production-guard.yml

---

## Go/No-Go Assessment for Live Messaging

### Prerequisites for SAFE Operation

| # | Prerequisite | Status |
|---|-------------|--------|
| 1 | Safety Detection: Critical alerts block automation | ✅ IMPLEMENTED (SafetyGuard) |
| 2 | Opt-Out Respect: DESIST detection disables automation | ✅ IMPLEMENTED (OptOutDetector) |
| 3 | Draft-First Posture: All messages require approval | ✅ IMPLEMENTED (ApprovalQueueService) |
| 4 | Person-Level Toggle: `automation_enabled` column | ✅ IMPLEMENTED (Person model) |
| 5 | Review Queue UI: Operational review interface | ✅ IMPLEMENTED (Web UI at localhost:5000, CLI commands) |
| 6 | Send Confirmation: Post-send state updates | ⚠️ PARTIAL (Action 11 marks SENT, ConversationState update missing) |
| 7 | Idempotent Draft Creation: Prevent duplicate drafts | ✅ IMPLEMENTED (queue_for_review deduplicates) |
| 8 | Auto-Approve Guards: First-message check, confidence threshold | ✅ IMPLEMENTED |
| 9 | Self-Message Prevention: Block drafts to tree owner | ✅ IMPLEMENTED (Dec 2025) |

### Current Recommendation: **CONDITIONAL GO for Review-First Mode**

The system is **SAFE** for:
- ✅ DNA match gathering (Action 6) - Production ready
- ✅ Inbox processing with safety checks (Action 7) - Production ready
- ✅ Draft generation with human review (Action 8 dry-run) - Production ready
- ✅ Task creation from productive messages (Action 9) - Production ready
- ✅ Tree search (Action 10) - Production ready
- ✅ Sending human-approved drafts (Action 11) - Production ready
- ✅ Shared match collection (Action 12) - Production ready
- ✅ Triangulation analysis (Action 13) - Production ready

**NOT YET SAFE for:**
- ❌ Auto-sending without human review (auto_approve_enabled=True)
- ❌ Fully automated reply loops

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Sending inappropriate message | HIGH | Draft-first posture, human approval required |
| Contacting opted-out person | HIGH | SafetyGuard + OptOutDetector block outbound |
| Rate limiting violation | MEDIUM | 0.3 RPS enforced, validated over 800+ pages |
| AI hallucination in reply | MEDIUM | Review queue catches before send |
| Session expiry mid-operation | LOW | Circuit breaker + proactive refresh |

---

## Appendix: Mission Alignment Matrix

| Mission Requirement | Implementation Status | Gap |
|---------------------|----------------------|-----|
| 1. Acknowledge & respect opt-out | ✅ SafetyGuard + OptOutDetector | Need opt-out acknowledgment message |
| 2. Answer questions from tree | ⚠️ SemanticSearch + TreeQuery scaffolded | Need end-to-end integration |
| 3. Extract & validate facts | ✅ FactValidator + DataConflict | ConflictDetector not wired (Phase 11.2) |
| 4. Suggest research areas | ⚠️ PredictiveGaps (827 lines) available | Not surfaced in drafts (Phase 11.3) |
| 5. 100% automated unless escalation | ⚠️ Safety escalation works | Auto-approval not enabled |
| 6. Performance metrics | ⚠️ ConversationMetrics exists | Dashboard not populated (Phase 9.2) |
| 7. High-quality, personalized messages | ✅ ContextBuilder + MessagePersonalizer (1987 lines) | 30+ functions available but underutilized |
| 8. Extract insights for tree | ⚠️ Fact extraction works | Tree update not implemented (Phase 8) |
| 9. Triangulation intelligence | ✅ TriangulationIntelligence (724 lines) ready | Not called from pipeline (Phase 11.1) |

---

## Appendix: Module Inventory (Reviewed Dec 2025)

### AI Prompts (ai/ai_prompts.json)
| Prompt Key | Version | Purpose |
|------------|---------|---------|
| intent_classification | v1.3.4 | Classify PRODUCTIVE/SOCIAL/ENTHUSIASTIC/etc |
| extraction_task | v1.2.0 | Extract genealogical entities from messages |
| dna_match_analysis | v1.1.0 | Analyze DNA evidence and relationships |
| family_tree_verification | v1.0.0 | Detect conflicts with tree data |
| record_research_guidance | v1.0.0 | Suggest research strategies |
| genealogical_reply | v2.0.1 | RAG-integrated reply generation |
| genealogical_dialogue_response | v1.0.1 | Phase 3 dialogue continuation |
| engagement_assessment | v1.0.0 | Analyze conversation engagement |
| intent_clarification | v1.0.0 | Handle ambiguous intent |

### Observability Infrastructure (observability/)
- **metrics_registry.py** (875 lines): Prometheus Counter/Gauge/Histogram wrappers with fallbacks
- **metrics_exporter.py** (432 lines): HTTP server for metrics scraping
- **apm.py** (673 lines): Application performance monitoring - PRODUCTION READY
  - Span tracking with context propagation
  - Decorator-based instrumentation (@trace)
  - Performance metrics: duration, memory, CPU
  - JSON export for external APM tools (Sentry, Datadog)
  - Configurable sampling for high-throughput
- **analytics.py** (516 lines): Lightweight monitoring engine - PRODUCTION READY
  - Logs to Logs/analytics.jsonl
  - Transient extras for action-specific context
  - Weekly summary generation and reporting
  - Zero external dependencies
- **conversation_analytics.py** (892 lines): Engagement analytics - NOT WIRED to InboundOrchestrator
- **prometheus/prometheus.yml**: Prometheus scraping configuration

### Integration Infrastructure (integrations/)
- **ms_graph_utils.py** (813 lines): Office 365 To-Do integration via MSAL
  - Token caching, atexit save, task creation
  - Used by Action 9 for genealogical task creation

### Core Infrastructure (core/)
- **feature_flags.py** (594 lines): Runtime feature toggle framework with rollout percentages - UNDERUTILIZED
- **pii_redaction.py** (523 lines): Log security with email/UUID/phone masking - NOT ENABLED by default
- **opt_out_detection.py** (599 lines): Multi-layer opt-out safeguards - PRODUCTION READY
- **rate_limiter.py** (1915 lines): Unified adaptive rate limiting - PRODUCTION READY
- **circuit_breaker.py** (793 lines): Standardized circuit breaker pattern - PRODUCTION READY
- **health_check.py** (830 lines): Startup health checks and runtime monitoring - MENU ACTION only

### A/B Testing Infrastructure (ai/)
- **ab_testing.py** (612 lines): Experiment framework with variant assignment and statistical analysis
- **ExperimentManager**: Manages experiments, assigns variants, tracks results
- **Status**: Implemented but not connected to main prompt selection flow

### CLI Infrastructure (cli/)
- **research_tools.py** (1210 lines): Interactive menu for innovation features
- **maintenance.py** (1895 lines): Log maintenance, analytics views, Grafana setup
- **Status**: Comprehensive CLI coverage, ResearchToolsCLI provides full research access

### Genealogy Modules (genealogy/)
- **gedcom_intelligence.py** (951 lines): AI-powered GEDCOM analysis with gap/conflict detection - NOT INTEGRATED
  - GedcomGap, GedcomConflict, ResearchOpportunity dataclasses
  - GedcomIntelligenceAnalyzer class for AI analysis
- **dna_gedcom_crossref.py** (807 lines): DNA-GEDCOM cross-referencing - NOT INTEGRATED
  - DNAMatch, GedcomPerson, CrossReferenceMatch dataclasses
  - CrossReferenceService for relationship validation
- **gedcom_cache.py** (~400 lines): GEDCOM tree caching
- **gedcom_utils.py** (~350 lines): GEDCOM file parsing utilities
- **dna_utils.py** (~300 lines): DNA match utility functions
- **dna_ethnicity_utils.py** (~200 lines): Ethnicity region handling

### Gather Infrastructure (actions/gather/)
- **orchestrator.py** (1060 lines): Gather action orchestration with metrics - PRODUCTION READY
  - GatherConfiguration dataclass with callbacks
  - initialize_gather_state() function
  - Circuit breaker integration
- **checkpoint.py** (414 lines): Checkpoint/resume for long-running gathers - PRODUCTION READY
  - GatherCheckpointService class
  - GatherCheckpointPlan dataclass
- **fetch.py** (~400 lines): API fetching with retry logic
- **persistence.py** (~350 lines): Database persistence for gather results
- **metrics.py** (~250 lines): Gather operation metrics

### Caching Infrastructure (caching/)
- **cache_manager.py** (908 lines): Centralized cache coordination - PRODUCTION READY
  - CacheCoordinator: Orchestrates all cache types
  - SessionComponentCache: Session state caching
  - APICacheManager: API response caching with TTL
  - SystemCacheManager: System-wide configuration cache
- **cache.py** (~500 lines): High-performance disk caching
- **cache_retention.py** (~300 lines): Cache retention policies

### Performance Infrastructure (performance/)
- **performance_orchestrator.py** (972 lines): System optimization engine
  - SmartQueryOptimizer: Database query analysis
  - MemoryPressureMonitor: Proactive memory management
  - APIBatchCoordinator: Intelligent request batching
- **connection_resilience.py** (~400 lines): Network connection handling
- **memory_utils.py** (~250 lines): Memory monitoring utilities
- **health_monitor.py** (~300 lines): System health monitoring

### Core Infrastructure - DI & Protocols (core/)
- **dependency_injection.py** (828 lines): Full DI container framework - UNDERUTILIZED
  - DIContainer class with singleton/transient/factory patterns
  - Thread-safe registration and resolution
  - Lifecycle management
  - **Gap**: Only 2 imports in entire codebase (both in session_utils.py)
- **protocols.py** (504 lines): Protocol definitions for duck typing - PRODUCTION READY
  - RateLimiterProtocol, DatabaseSessionProtocol, SessionHealthMonitor
  - @runtime_checkable decorators for isinstance checks
- **database_manager.py** (1428 lines): Database management with async support
  - async_session_context(): Async context manager
  - async_execute_query(): Async query execution
  - Uses asyncio.run_in_executor for thread pool

### Scripts (scripts/)
- **dry_run_validation.py** (602 lines): Test pipeline against historical data
  - DryRunProcessor class for end-to-end testing
  - Generates drafts and compares against actual responses
- **check_production_guard.py** (~200 lines): Production readiness validation
- **maintain_code_graph.py** (~300 lines): Code graph maintenance

### Messaging Infrastructure (messaging/)
- **inbound.py** (761 lines): InboundOrchestrator for message processing - PRODUCTION READY
  - Safety checks → Intent classification → Entity extraction → Draft generation
  - Integrates SafetyGuard, AI classification, ResearchService
- **safety.py** (407 lines): SafetyGuard for threat/opt-out detection - PRODUCTION READY
  - SafetyStatus enum: SAFE, UNSAFE, OPT_OUT, NEEDS_REVIEW, CRITICAL_ALERT, HIGH_VALUE
  - CriticalAlertCategory: THREATS_HOSTILITY, SELF_HARM, LEGAL_PRIVACY, HIGH_VALUE_DISCOVERY
- **message_personalization.py** (1987 lines): 30+ personalization functions - UNDERUTILIZED
- **message_types.py** (~200 lines): Message type definitions

### Research Modules (research/)
- **relationship_utils.py** (2219 lines): Core relationship processing - PRODUCTION READY
  - Relationship calculation, path finding, family tree traversal
  - fast_bidirectional_bfs() for GEDCOM path resolution
  - Relationship description formatting
- **research_prioritization.py** (1207 lines): AI-powered research task prioritization
  - ResearchPriority, FamilyLineStatus, LocationResearchCluster dataclasses
  - IntelligentResearchPrioritizer class
- **person_lookup_utils.py** (539 lines): Person lookup structures - PRODUCTION READY
  - PersonLookupResult, PersonMention, LookupContext dataclasses
  - Used by inbound message entity resolution
- **research_suggestions.py** (551 lines): Location/time-based research suggestions
  - ANCESTRY_COLLECTIONS by region (Scotland, England, Ireland, Canada, USA)
  - TIME_PERIOD_COLLECTIONS for 1800s/1900s
- **record_sharing.py** (539 lines): Record reference formatting for messages
  - format_record_reference(), format_multiple_records() functions
- **relationship_diagram.py** (468 lines): ASCII relationship diagrams
  - Vertical, horizontal, compact diagram styles
- **triangulation_intelligence.py** (724 lines): Hypothesis generation - NOT INTEGRATED
- **conflict_detector.py** (547 lines): Field comparison logic - NOT INTEGRATED
- **predictive_gaps.py** (827 lines): Gap detection heuristics - NOT INTEGRATED
- **search_criteria_utils.py** (~300 lines): Search criteria construction
- **research_guidance_prompts.py** (~200 lines): AI prompts for research suggestions

### Testing Infrastructure (testing/)
- **test_framework.py** (1115 lines): Comprehensive test suite infrastructure
  - TestSuite class, Colors, Icons for terminal output
  - MagicMock, patch exports for mocking
  - suppress_logging, mock_logger_context utilities
- **test_utilities.py** (~400 lines): Shared test helper functions
- **test_integration_e2e.py** (292 lines): End-to-end flow tests

### Configuration (config/)
- **config_schema.py** (1649 lines): Type-safe configuration schemas - PRODUCTION READY
  - ConfigValidator with environment-specific rules
  - ValidationRule dataclass for custom validation
  - EnvironmentType enum: DEVELOPMENT, TESTING, PRODUCTION
- **config_manager.py** (~500 lines): Configuration loading and caching

### Core Session Management (core/)
- **session_manager.py** (3127 lines): Central session orchestration - PRODUCTION READY
  - SessionLifecycleState enum: UNINITIALIZED, RECOVERING, READY, DEGRADED
  - SessionManager class with SessionIdentifierMixin, SessionHealthMixin
  - Integrates DatabaseManager, BrowserManager, APIManager
- **database.py** (4356 lines): SQLAlchemy ORM models and utilities - PRODUCTION READY
  - 15+ enums: MessageDirectionEnum, ConversationStatusEnum, PersonStatusEnum, etc.
  - 12+ tables: Person, DnaMatch, ConversationLog, DraftReply, SharedMatch, etc.
  - db_transn() context manager, backup_database() utility
- **utils.py** (4037 lines): Core utilities and API functions - PRODUCTION READY
  - RateLimiter, decorators, cookie persistence
  - API request helpers, login verification
- **error_handling.py** (2065 lines): Comprehensive error framework - PARTIALLY USED
  - RetryConfig dataclass for retry configuration
  - ErrorHandler ABC with DatabaseErrorHandler, NetworkErrorHandler, BrowserErrorHandler
  - ErrorHandlerRegistry and ErrorRecoveryManager
  - **Gap**: ancestry_session_recovery, ancestry_api_recovery not wired (documented in header)
- **approval_queue.py** (879 lines): Human-in-the-loop review queue - PRODUCTION READY
  - ApprovalStatus enum: PENDING, APPROVED, REJECTED, AUTO_APPROVED, EXPIRED, SENT
  - ReviewPriority enum: LOW, NORMAL, HIGH, CRITICAL
  - QueuedDraft, ReviewDecision, QueueStats dataclasses
  - ApprovalQueueService class

### API Layer (api/)
- **api_utils.py** (3814 lines): API intelligence and request orchestration - PRODUCTION READY
  - Unified API request handling with intelligent routing
  - Advanced authentication management with credential rotation
  - Rate limiting with dynamic throttling and circuit breakers
  - Request batching, compression, connection pooling
  - Comprehensive error handling with retry logic and exponential backoff
- **api_search_core.py** (1266 lines): TreesUI search with caching - PRODUCTION READY
  - 7-day search result caching (Priority 1 Todo #10)
  - GEDCOM enrichment integration
  - Cache statistics tracking
- **api_constants.py** (290 lines): Centralized API endpoint constants - PRODUCTION READY
  - Single source of truth for all Ancestry API paths
  - Protected by regression guard tests
  - Endpoints: CSRF, messaging, trees, DNA matches

### Genealogy Services (genealogy/)
- **tree_query_service.py** (603 lines): Real-time genealogical queries - PRODUCTION READY
  - PersonSearchResult, RelationshipResult dataclasses
  - TreeQueryService: find_person, explain_relationship, get_ancestors, get_descendants
- **fact_validator.py** (1004 lines): Fact validation pipeline - PRODUCTION READY
  - ExtractedFact dataclass for runtime fact representation
  - FactValidator service with conflict detection
  - ConflictType enum: EXACT_MATCH, COMPATIBLE, MINOR_CONFLICT, MAJOR_CONFLICT, NO_EXISTING
- **semantic_search.py** (652 lines): Tree-aware semantic search - PRODUCTION READY
  - SemanticSearchIntent enum: PERSON_LOOKUP, RELATIONSHIP_EXPLANATION, etc.
  - EvidenceBlock, CandidatePerson, SemanticSearchResult dataclasses
  - Tree-first retrieval, fail closed to clarification
- **research_service.py** (360 lines): GEDCOM research operations - PRODUCTION READY
  - ResearchService class with search_people, load_gedcom
  - Universal scoring with configurable weights
  - Relationship pathfinding via bidirectional BFS
- **triangulation.py** (142 lines): Triangulation opportunity detection - PRODUCTION READY
  - TriangulationService with find_triangulation_opportunities
  - Hypothesis generation based on shared matches and common ancestors

### Performance Infrastructure (performance/)
- **performance_orchestrator.py** (972 lines): System optimization engine - PRODUCTION READY
  - SmartQueryOptimizer: Database query analysis and slow query tracking
  - MemoryPressureMonitor: Proactive memory management with GC triggers
  - APIBatchCoordinator: Intelligent request batching
  - ModuleLoadOptimizer: Import and loading time optimization
- **connection_resilience.py** (395 lines): Connection protection framework - PRODUCTION READY
  - ConnectionResilienceManager class with sleep prevention
  - Automatic browser health monitoring
  - Graceful recovery from connection loss with exponential backoff
  - Cross-platform (Windows/macOS/Linux) sleep prevention

### Action Modules (actions/)
- **action6_gather.py** (5308 lines): DNA match gathering - PRODUCTION READY
  - coord() entry point with checkpoint/resume
  - 800+ pages tested, zero 429 errors with 0.3 RPS
  - Integrates circuit breaker, health monitoring, API caching
- **action7_inbox.py** (4540 lines): Inbox processing - PRODUCTION READY
  - AI-powered classification (PRODUCTIVE, DESIST, OTHER)
  - Sentiment analysis and engagement tracking
  - Database synchronization with conflict resolution
- **action8_messaging.py** (6853 lines): Intelligent messaging engine - PRODUCTION READY
  - Dynamic template selection, engagement prediction
  - Sentiment adaptation integration
  - Batch processing with rate limiting, circuit breaker patterns
- **action9_process_productive.py** (4576 lines): Productive match processing - PRODUCTION READY
  - GEDCOM integration, relationship analysis
  - Pydantic validation for extracted entities
  - MS To-Do task generation
- **action10.py** (3509 lines): GEDCOM analysis engine - PRODUCTION READY
  - Relationship pathfinding using bidirectional BFS
  - Match scoring with configurable weights
  - Research gap identification
- **action11_send_approved_drafts.py** (474 lines): Draft sending - PRODUCTION READY
  - Re-checks outbound guardrails before send
  - Marks drafts SENT on success, writes audit log
  - Updates ConversationMetrics
- **action12_shared_matches.py** (302 lines): Shared match scraping - PRODUCTION READY
  - Fetches shared matches for DNA matches > 9cM
  - Stores in SharedMatch table
- **action13_triangulation.py** (337 lines): Triangulation analysis - PRODUCTION READY
  - Shared match analysis, hypothesis generation
  - CSV/HTML export capability
- **action14_research_tools.py** (82 lines): Research tools launcher - PRODUCTION READY
  - Launches ResearchToolsCLI interactive menu

### AI Provider Infrastructure (ai/providers/)
- **base.py** (221 lines): Provider protocol definitions - PRODUCTION READY
  - ProviderRequest, ProviderResponse dataclasses
  - ProviderAdapter protocol with is_available(), call()
  - BaseProvider with ensure_available() helper
- **gemini.py** (320 lines): Google Gemini adapter - PRODUCTION READY
  - Model validation, client initialization
  - JSON response format support
- **deepseek.py** (230 lines): DeepSeek adapter (OpenAI-compatible) - PRODUCTION READY
  - OpenAI SDK-based implementation
  - Fallback provider option
- **local_llm.py** (~200 lines): Local LLM adapter - AVAILABLE
- **moonshot.py** (~150 lines): Moonshot adapter - AVAILABLE

### AI Core Modules (ai/)
- **context_builder.py** (931 lines): MatchContext assembly - PRODUCTION READY
  - Aggregates Database, GEDCOM, Conversation History
  - to_dict(), to_json(), to_prompt_string() methods
- **sentiment_adaptation.py** (733 lines): Tone adaptation - PRODUCTION READY
  - Sentiment enum: VERY_POSITIVE to VERY_NEGATIVE
  - MessageTone enum: FORMAL, FRIENDLY, ENTHUSIASTIC, etc.
  - SentimentAdapter class with ToneRecommendation output

### Documentation Specs (docs/specs/)
- **reply_management.md** (135 lines): Conversation state machine spec
  - 8 states: Initial, AwaitingReply, ProductiveEngagement, ExtractionPending, etc.
  - Critical alert detection patterns
  - Mermaid state diagram
- **human_in_the_loop.md** (495 lines): HITL safeguards spec
  - MessageApproval, SystemControl table schemas
  - Tiered approval workflow
  - Emergency stop controls
- **mission_execution_spec.md** (112 lines): Mission requirements tracking
  - 8 mission requirements with implementation status
  - Baseline capabilities assessment
  - Links mission statement to implementation
- **operator_manual.md** (512 lines): Review queue operations guide
  - CLI commands for draft review
  - Approval workflow procedures
  - Emergency controls and monitoring
- **tech_stack.md** (~100 lines): Technology and infrastructure spec
  - Core dependencies with versions
  - Rate limiting algorithm documentation
  - Session management strategy

### Production Documentation (docs/)
- **production_messaging_checklist.md** (~60 lines): Pre-send safety checklist
  - Configuration requirements (APP_MODE, AUTO_APPROVE, MAX_INBOX)
  - Dry run and manual review procedures
  - Controlled rollout guidance
  - Emergency stop procedures

### UI Components (ui/)
- **review_server.py** (432 lines): Flask-based web UI - PRODUCTION READY
  - Draft review and approval interface at localhost:5000
  - API endpoints for approve/reject/edit
- **menu.py** (305 lines): Terminal menu interface - PRODUCTION READY
  - Dynamic action rendering from registry
  - Keyboard navigation
- **terminal_test_agent.py** (110 lines): Menu testing agent

---

## Next Actions (Immediate)

1. **✅ DONE: Add self-message prevention** - Block drafts where recipient == owner (Draft #3 issue)
2. **⏸️ DEFERRED: Context inversion detection** - Detect when AI explains recipient's own ancestor to them (Draft #1 issue) - see Phase 1.5.2/1.5.3
3. **Send 4 approved drafts** - Run Option 11 to deliver reviewed messages
4. **✅ DONE: Add transaction safety to send loop** - Wrap in try/except with rollback (Phase 1.6.1)
5. **✅ DONE: Add duplicate send prevention** - Skip already-sent drafts, 5-min idempotency window (Phase 1.6.3)
6. **✅ DONE: ConversationState sync after send** - Update conversation_phase to "awaiting_reply" (Phase 1.6.4)
7. **✅ DONE: Add draft expiration job** - expires_at field + expire_old_drafts() method (Phase 1.6.2)
8. **✅ DONE: Emit Prometheus metrics** - drafts_queued, drafts_sent, review_queue_depth (Phase 9.1)
9. **Run full inbox → reply dry-run test** - Validate end-to-end flow before any live sends
10. **✅ DONE: Wire TriangulationIntelligence into ContextBuilder** - Hypothesis in draft context (Phase 11.1)
11. **✅ DONE: Wire PredictiveGapDetector into draft personalization** - Research gaps in context (Phase 11.3)
12. **✅ DONE: Add startup maintenance task** - Expire old drafts on app startup (Phase 10.1)
13. **✅ DONE: Wire ConflictDetector severity** - HIGH/CRITICAL conflicts flagged for review (Phase 11.2)

---

## Session Notes

### December 14, 2025 - Session 3 (Extended Review)

**Areas Reviewed:**
- AI prompts structure (ai_prompts.json) - 9 versioned prompts
- ContextBuilder (931 lines) - MatchContext assembly
- Research modules (3000+ lines total):
  - triangulation_intelligence.py (724 lines) - hypothesis scoring
  - conflict_detector.py (547 lines) - field comparison
  - predictive_gaps.py (827 lines) - gap detection
- Message personalization (1987 lines) - 30+ personalization functions
- Observability infrastructure (1300+ lines) - Prometheus/Grafana scaffolded
- MS Graph integration (813 lines) - Office 365 To-Do
- CI/CD workflows (3 GitHub Actions)
- Documentation (operator_manual.md exists, 512 lines)

**Key Findings:**
1. **Research modules fully implemented but not integrated** - Added Phase 11
2. **Observability scaffolded but not emitting** - Phase 9 gaps confirmed
3. **Operator manual exists** - Updated todo.md to reflect
4. **CI/CD in place** - tests.yml, quality-gate.yml, lint-typecheck.yml
5. **83 TODO/FIXME comments** remain in codebase
6. **Placeholder code in production** - triangulation_intelligence.py:390, relationship_utils.py stubs

### December 14, 2025 - Session 4 (Deep Dive)

**Additional Areas Reviewed:**
- CLI infrastructure (cli/research_tools.py 1210 lines, cli/maintenance.py 1895 lines)
- UI infrastructure (menu.py 305 lines, terminal_test_agent.py)
- API layer (api_search_core.py 1266 lines - Priority 1 Todo #10 caching)
- Core infrastructure:
  - feature_flags.py (594 lines) - Runtime toggles with rollout percentages
  - pii_redaction.py (523 lines) - Log security filters
  - opt_out_detection.py (599 lines) - Multi-layer safeguards
  - rate_limiter.py (1915 lines) - Unified adaptive rate limiting
  - circuit_breaker.py (793 lines) - Session circuit breaker pattern
  - health_check.py (830 lines) - Startup health checks
- A/B testing (ab_testing.py 612 lines) - Experiment management
- Integration tests (test_integration_e2e.py 292 lines) - E2E flow tests
- Conversation analytics (892 lines) - Engagement tracking

**Key Findings:**
1. **Feature flags underutilized** - Full rollout framework exists but not wired to actions ✅ FIXED: Action 11 uses ACTION11_SEND_ENABLED flag
2. **PII redaction not enabled** - 523 lines of redaction code but not active in production logging ✅ FIXED: Wired to logging_config.py, enabled via PII_REDACTION_ENABLED=true
3. **A/B testing disconnected** - ExperimentManager exists but not connected to prompt selection ✅ FIXED: ai_interface uses get_prompt_with_experiment(); MessagePersonalizer uses ExperimentManager
4. **Conversation analytics not wired** - 892 lines of engagement tracking not called from InboundOrchestrator ✅ FIXED: _update_metrics() already wires EngagementTracking
5. **Health checks menu-only** - Should run on startup, currently only via 'health' menu action ✅ FIXED: lifecycle.py calls run_startup_health_checks()
6. **Empty tests/ directory** - All tests are in testing/ directory or embedded in modules ✅ FIXED: Added tests/README.md explaining embedded pattern
7. **ResearchToolsCLI comprehensive** - Good lazy-loading patterns, full research feature access

**Total Codebase Lines Reviewed (Session 3+4):**
| Category | Lines |
|----------|-------|
| Research Modules | ~3,000 |
| Core Infrastructure | ~7,000 |
| CLI/UI | ~3,500 |
| Observability | ~2,200 |
| AI/Prompts | ~2,500 |
| Testing Infrastructure | ~3,000 |
| **Total** | **~21,000+** |
### December 14, 2025 - Session 5 (Infrastructure Deep Dive)

**Areas Reviewed:**
- Dependency Injection framework (core/dependency_injection.py 828 lines)
- Protocol definitions (core/protocols.py 504 lines)
- Async database operations (core/database_manager.py async methods)
- GEDCOM intelligence (genealogy/gedcom/gedcom_intelligence.py 951 lines)
- DNA-GEDCOM cross-reference (genealogy/dna/dna_gedcom_crossref.py 807 lines)
- Gather orchestration (actions/gather/orchestrator.py 1060 lines)
- Checkpoint/resume system (actions/gather/checkpoint.py 414 lines)
- Cache management (caching/cache_manager.py 908 lines)
- Performance orchestrator (performance/performance_orchestrator.py 972 lines)
- Selenium utilities (browser/selenium_utils.py 375 lines)
- MS Graph integration (integrations/ms_graph_utils.py 813 lines)
- Metrics exporter (observability/metrics_exporter.py 432 lines)
- Action registry (core/action_registry.py 923 lines)
- Dry-run validation (scripts/dry_run_validation.py 602 lines)
- Menu UI (ui/menu.py 305 lines)

**Key Findings:**
1. **DI container severely underutilized** - 828 lines of robust framework but only 2 imports in codebase
2. **Async DB operations available** - async_session_context, async_execute_query ready but not widely used
3. **GEDCOM intelligence not integrated** - 951 lines of AI-powered analysis ready but not called
4. **DNA-GEDCOM cross-ref not integrated** - 807 lines of cross-referencing available
5. **Gather system production-ready** - orchestrator + checkpoint = robust resume capability
6. **Cache infrastructure comprehensive** - CacheCoordinator, SessionComponentCache, APICacheManager
7. **Performance optimization ready** - SmartQueryOptimizer, memory monitoring available
8. **Protocol-based typing strong** - 504 lines of well-designed protocols for duck typing
9. **No real-time notification infrastructure** - No WebSocket/SSE for push notifications
10. **Dry-run validation script available** - 602 lines for testing against historical data
11. **Action modules extremely comprehensive** - 26,000+ lines across 9 action modules
12. **SessionManager is the largest module** - 3127 lines, central to all operations
13. **Error handling framework extensive but partially wired** - 2065 lines, recovery decorators not connected
14. **AI provider abstraction clean** - 4 providers with common protocol (base.py 221 lines)
15. **Sentiment adaptation production-ready** - 733 lines with tone recommendation system
16. **Documentation specs thorough** - reply_management.md + human_in_the_loop.md define state machines

**Added to Roadmap:**
- Phase 12: GEDCOM/DNA Intelligence Integration (3 subsections)
- Phase 13: Infrastructure Modernization (DI, async, protocols)
- Updated maturity assessment with DI, async, GEDCOM, DNA-GEDCOM status
- Comprehensive module inventory (40+ module entries)

**Session 5 Full Review List:**
- Core Infrastructure:
  - dependency_injection.py (828 lines), protocols.py (504 lines)
  - database_manager.py async methods, session_manager.py (3127 lines)
  - error_handling.py (2065 lines), database.py (4356 lines)
  - utils.py (4037 lines), workflow_actions.py (569 lines)
  - maintenance_actions.py (900 lines), action_registry.py (923 lines)
- Genealogy Services:
  - gedcom_intelligence.py (951 lines), dna_gedcom_crossref.py (807 lines)
  - research_service.py (360 lines), semantic_search.py (652 lines)
  - triangulation.py (142 lines), tree_query_service.py (603 lines)
  - fact_validator.py (1004 lines), gedcom_cache.py, gedcom_utils.py
- Gather System: orchestrator.py (1060 lines), checkpoint.py (414 lines)
- Caching: cache_manager.py (908 lines), cache.py, cache_retention.py
- Performance: performance_orchestrator.py (972 lines), connection_resilience.py (395 lines)
- Browser/UI: selenium_utils.py (375 lines), menu.py (305 lines), review_server.py (432 lines)
- Integrations: ms_graph_utils.py (813 lines)
- Observability: metrics_exporter.py (432 lines), apm.py (673 lines), analytics.py (516 lines)
- Scripts: dry_run_validation.py (602 lines), check_production_guard.py (~200 lines)
- Messaging: inbound.py (761 lines), safety.py (407 lines)
- Action Modules (9 total):
  - action6 (5308), action7 (4540), action8 (6853), action9 (4576)
  - action10 (3509), action11 (474), action12 (302), action13 (337), action14 (82)
- AI Infrastructure:
  - base.py (221 lines), gemini.py (320 lines), deepseek.py (230 lines)
  - context_builder.py (931 lines), sentiment_adaptation.py (733 lines)
- Documentation Specs:
  - reply_management.md (135 lines), human_in_the_loop.md (495 lines)
  - mission_execution_spec.md (112 lines), operator_manual.md (512 lines)
  - tech_stack.md (~100 lines), production_messaging_checklist.md (~60 lines)

**Session 5 Lines Reviewed:**
| Category | Lines |
|----------|-------|
| Core Infrastructure (DI, protocols, async, session_manager, error_handling, database, utils) | ~18,500 |
| Genealogy Modules (GEDCOM, DNA, services, triangulation, semantic_search) | ~5,600 |
| Gather System (orchestrator, checkpoint) | ~1,500 |
| Caching/Performance (cache_manager, performance_orchestrator, connection_resilience) | ~2,300 |
| Browser/UI (menu, review_server, terminal_test_agent) | ~1,100 |
| Scripts/Validation | ~800 |
| Messaging (inbound, safety) | ~1,200 |
| Research (relationship_utils, prioritization, suggestions, diagrams, etc.) | ~8,100 |
| Testing/Config | ~2,800 |
| API Layer (api_utils, api_search_core, api_constants) | ~5,400 |
| Action Modules (6, 7, 8, 9, 10, 11, 12, 13, 14) | ~26,000 |
| AI Infrastructure (providers, context, sentiment) | ~2,600 |
| Observability (metrics, APM, analytics) | ~3,600 |
| Documentation Specs | ~1,400 |
| Main Entry Point | ~850 |
| **Session 5 Total** | **~82,000** |

**Cumulative Lines Reviewed (Sessions 3-5):**
| Session | Lines |
|---------|-------|
| Session 3 | ~8,000 |
| Session 4 | ~13,000 |
| Session 5 | ~82,000 |
| **Grand Total** | **~103,000+** |