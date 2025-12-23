# TODO List

## 🔴 Priority: Unified Message Sending Service Refactoring

**Goal:** Consolidate 4 distinct message sending mechanisms (Action 8, Action 9, Action 11, DESIST acknowledgements) into a single `MessageSendOrchestrator` that handles all outbound messages through one decision-capable pipeline.

**Context:** The codebase currently has:
1. **Action 9** - Custom AI replies to productive conversations
2. **Action 8** - Generic 3-message sequence (Initial → Follow-Up → Final Reminder) + DESIST acknowledgements
3. **Action 11** - Human-approved draft replies

All routes ultimately call `call_send_message_api()` in [api_utils.py](api/api_utils.py), but safety checks, database updates, and business logic are duplicated across ~11,000 lines in 3+ files.

---

### Phase 1: Create Message Send Orchestrator Foundation

#### 1.1 Create Core Data Structures
- [x] **1.1.1** Create `messaging/send_orchestrator.py` with module docstring and imports ✅ (2025-12-23)
- [x] **1.1.2** Define `SendTrigger` enum with values: ✅ (2025-12-23)
  - `AUTOMATED_SEQUENCE` (Action 8 template messages)
  - `REPLY_RECEIVED` (Action 9 custom replies)
  - `OPT_OUT` (DESIST acknowledgements)
  - `HUMAN_APPROVED` (Action 11 draft approvals)
- [x] **1.1.3** Define `MessageSendContext` dataclass with fields: ✅ (2025-12-23)
  - `person: Person`
  - `send_trigger: SendTrigger`
  - `conversation_logs: list[ConversationLog]`
  - `conversation_state: Optional[ConversationState]`
  - `additional_data: dict[str, Any]` (AI context, draft content, template data)
- [x] **1.1.4** Define `SendDecision` dataclass for decision engine output: ✅ (2025-12-23)
  - `should_send: bool`
  - `block_reason: Optional[str]`
  - `message_type: Optional[MessageType]`
  - `content_source: str` (template/ai/draft/desist_ack)
- [x] **1.1.5** Define `SendResult` dataclass: ✅ (2025-12-23)
  - `success: bool`
  - `message_id: Optional[str]`
  - `error: Optional[str]`
  - `database_updates: list[str]` (audit trail of what was updated)

#### 1.2 Implement Safety Check Layer (Priority 1)
- [x] **1.2.1** Create `_check_opt_out_status()` method ✅ (2025-12-23)
  - Integrate existing detector from [opt_out_detection.py](messaging/opt_out_detection.py) lines 355-375
  - Return `(blocked: bool, reason: str)`
- [x] **1.2.2** Create `_check_app_mode_policy()` method ✅ (2025-12-23)
  - Integrate existing check from [app_mode_policy.py](config/app_mode_policy.py)
  - Return `(blocked: bool, reason: str)`
- [x] **1.2.3** Create `_check_conversation_hard_stops()` method ✅ (2025-12-23)
  - Check for DESIST, ARCHIVE, BLOCKED states
  - Reference existing logic in [action11_send_approved_drafts.py](actions/action11_send_approved_drafts.py) lines 119-135
  - Return `(blocked: bool, reason: str)`
- [x] **1.2.4** Create `_check_duplicate_prevention()` method ✅ (2025-12-23)
  - Centralize logic from [action11_send_approved_drafts.py](actions/action11_send_approved_drafts.py) lines 242-286
  - Check recent sends within configurable window (default 24h)
  - Return `(blocked: bool, reason: str)`
- [x] **1.2.5** Create `run_safety_checks()` method that combines all 4 checks ✅ (2025-12-23)
  - Short-circuit on first failure
  - Log all check results for audit trail

#### 1.3 Implement Decision Engine (Priority 2)
- [x] **1.3.1** Create `determine_message_strategy()` method with priority logic: ✅ (2025-12-23)
  1. **DESIST Acknowledgement** - Person status is DESIST + acknowledgement not sent
  2. **Human-Approved Draft** - Approved draft exists + not yet sent
  3. **Custom Reply** - Recent productive inbound + no custom reply sent
  4. **Generic Sequence** - Default to state machine from [message_types.py](messaging/message_types.py) lines 96-131
- [x] **1.3.2** Extract DESIST detection logic from [action8_messaging.py](actions/action8_messaging.py) lines 2284-2301 ✅ (2025-12-23)
- [x] **1.3.3** Extract draft approval check logic from [action11_send_approved_drafts.py](actions/action11_send_approved_drafts.py) lines 209-213 ✅ (2025-12-23)
- [x] **1.3.4** Extract custom reply detection from [action9_process_productive.py](actions/action9_process_productive.py) lines 672-730 ✅ (2025-12-23)
- [x] **1.3.5** Integrate state machine invocation from [message_types.py](messaging/message_types.py) ✅ (2025-12-23)

#### 1.4 Implement Content Generation Layer
- [x] **1.4.1** Create `generate_message_content()` method with branching: ✅ (2025-12-23)
  - Template messages → existing template selection logic
  - Custom replies → AI generation pipeline from Action 9
  - DESIST acks → opt-out acknowledgement generator
  - Approved drafts → strip internal metadata, use draft content
- [x] **1.4.2** Create `_generate_template_content()` helper ✅ (2025-12-23)
  - Reference existing template selection in Action 8
- [x] **1.4.3** Create `_generate_ai_reply_content()` helper ✅ (2025-12-23)
  - Reference AI generation in [action9_process_productive.py](actions/action9_process_productive.py) lines 2796-2805
- [x] **1.4.4** Create `_generate_desist_acknowledgement()` helper ✅ (2025-12-23)
  - Reference [opt_out_detection.py](messaging/opt_out_detection.py) lines 444-462
- [x] **1.4.5** Create `_extract_approved_draft_content()` helper ✅ (2025-12-23)
  - Strip internal metadata and formatting

#### 1.5 Implement Send Execution
- [x] **1.5.1** Create `execute_send()` method: ✅ (2025-12-23)
  - Final validation (re-run safety checks)
  - Call canonical `call_send_message_api()` from [api_utils.py](api/api_utils.py) line 188
  - Handle API response and errors
- [x] **1.5.2** Create `_update_database_records()` helper: ✅ (2025-12-23)
  - Update ConversationLog (new outbound entry)
  - Update ConversationState (advance state machine)
  - Update Person status (if applicable)
  - Update Draft status (mark as sent for Action 11)
- [x] **1.5.3** Create `_record_engagement_event()` helper: ✅ (2025-12-23)
  - Log to engagement_events table
  - Record metrics for observability
- [x] **1.5.4** Create `_log_audit_trail()` helper: ✅ (2025-12-23)
  - Log decision path, content source, database updates
  - Include all safety check results

#### 1.6 Main Orchestrator Entry Point
- [x] **1.6.1** Create `MessageSendOrchestrator` class with constructor: ✅ (2025-12-23)
  - Accept `session_manager: SessionManager`
  - Initialize logger, metrics collector
- [x] **1.6.2** Create `send()` method as main entry point: ✅ (2025-12-23)
  - Accept `context: MessageSendContext`
  - Orchestrate: safety checks → decision → content generation → execute
  - Return `SendResult`
- [x] **1.6.3** Add feature flag check at entry point ✅ (2025-12-23)
  - `ENABLE_UNIFIED_SEND_ORCHESTRATOR` in `.env`
  - Default to `False` during rollout

---

### Phase 2: Extract Supporting Modules

#### 2.1 Person Eligibility Module
- [ ] **2.1.1** Create `messaging/person_eligibility.py`
- [ ] **2.1.2** Extract person filtering logic from Action 8
  - In-tree vs out-tree classification
  - Contact eligibility rules
  - Rate limiting per-person
- [ ] **2.1.3** Create `PersonEligibilityChecker` class
- [ ] **2.1.4** Add unit tests for eligibility rules

#### 2.2 Template Selector Module
- [ ] **2.2.1** Create `messaging/template_selector.py`
- [ ] **2.2.2** Extract template selection logic from Action 8
- [ ] **2.2.3** Create `TemplateSelector` class with methods:
  - `select_initial_template(person, context) -> Template`
  - `select_followup_template(person, context) -> Template`
  - `select_final_reminder_template(person, context) -> Template`
- [ ] **2.2.4** Add unit tests for template selection

---

### Phase 3: Refactor Action Modules

#### 3.1 Refactor Action 8 (Generic Sequences)
- [ ] **3.1.1** Add feature flag check at Action 8 entry point
- [ ] **3.1.2** Create parallel code path using `MessageSendOrchestrator`
- [ ] **3.1.3** Replace direct `_send_message()` calls with orchestrator
- [ ] **3.1.4** Keep batch processing and rate limiting logic
- [ ] **3.1.5** Remove duplicated safety check code (now in orchestrator)
- [ ] **3.1.6** Remove duplicated database update code
- [ ] **3.1.7** Update metrics collection to use orchestrator metrics
- [ ] **3.1.8** Add shadow mode logging (compare old vs new decisions)

#### 3.2 Refactor Action 9 (Custom Replies)
- [ ] **3.2.1** Add feature flag check at Action 9 send points
- [ ] **3.2.2** Replace `_send_message()` with orchestrator call
  - Use `SendTrigger.REPLY_RECEIVED`
  - Pass AI-generated content via `additional_data`
- [ ] **3.2.3** Keep AI extraction and processing logic unchanged
- [ ] **3.2.4** Keep conversation context building unchanged
- [ ] **3.2.5** Remove duplicated database update code
- [ ] **3.2.6** Add shadow mode logging

#### 3.3 Refactor Action 11 (Approved Drafts)
- [ ] **3.3.1** Add feature flag check at Action 11 entry
- [ ] **3.3.2** Keep draft fetching and approval filtering logic
- [ ] **3.3.3** Keep circuit breaker pattern from lines 443-462
- [ ] **3.3.4** Replace `_send_single_approved_draft()` with orchestrator
  - Use `SendTrigger.HUMAN_APPROVED`
  - Pass draft content via `additional_data`
- [ ] **3.3.5** Move duplicate prevention to orchestrator (already done in 1.2.4)
- [ ] **3.3.6** Remove duplicated safety check code
- [ ] **3.3.7** Add shadow mode logging

#### 3.4 Refactor DESIST Handling
- [ ] **3.4.1** Ensure DESIST detection remains in orchestrator decision engine
- [ ] **3.4.2** Remove DESIST-specific send code from Action 8
- [ ] **3.4.3** Verify DESIST acknowledgements are sent via orchestrator
- [ ] **3.4.4** Test DESIST priority over other message types

---

### Phase 4: Testing Strategy

#### 4.1 Unit Tests for Orchestrator
- [ ] **4.1.1** Create `messaging/test_send_orchestrator.py`
- [ ] **4.1.2** Test safety check combinations:
  - All checks pass → should_send=True
  - Opt-out → should_send=False
  - App mode policy blocks → should_send=False
  - Conversation hard stop → should_send=False
  - Duplicate prevention → should_send=False
- [ ] **4.1.3** Test decision engine priority:
  - DESIST takes precedence over approved draft
  - Approved draft takes precedence over custom reply
  - Custom reply takes precedence over generic sequence
- [ ] **4.1.4** Test content generation for each path
- [ ] **4.1.5** Test database update consistency
- [ ] **4.1.6** Test error handling and rollback

#### 4.2 Integration Tests
- [ ] **4.2.1** Create `tests/test_send_integration.py`
- [ ] **4.2.2** Test full flow for AUTOMATED_SEQUENCE trigger
- [ ] **4.2.3** Test full flow for REPLY_RECEIVED trigger
- [ ] **4.2.4** Test full flow for OPT_OUT trigger
- [ ] **4.2.5** Test full flow for HUMAN_APPROVED trigger
- [ ] **4.2.6** Test mixed scenarios (e.g., approved draft exists + DESIST status)
- [ ] **4.2.7** Verify database consistency after each flow

#### 4.3 Shadow Mode Validation
- [ ] **4.3.1** Create `messaging/shadow_mode_analyzer.py`
- [ ] **4.3.2** Implement decision comparison logging
- [ ] **4.3.3** Create analysis report for decision discrepancies
- [ ] **4.3.4** Run shadow mode for minimum 1 week before cutover
- [ ] **4.3.5** Document and resolve all discrepancies

---

### Phase 5: Rollout Strategy

#### 5.1 Shadow Mode (Week 1-2)
- [ ] **5.1.1** Deploy with `ENABLE_UNIFIED_SEND_ORCHESTRATOR=false`
- [ ] **5.1.2** Enable shadow logging to compare decisions
- [ ] **5.1.3** Monitor logs for decision discrepancies
- [ ] **5.1.4** Fix any discrepancies found
- [ ] **5.1.5** Document baseline metrics for comparison

#### 5.2 Feature Flag Rollout (Week 3-4)
- [ ] **5.2.1** Create per-action feature flags:
  - `ORCHESTRATOR_ACTION8=true/false`
  - `ORCHESTRATOR_ACTION9=true/false`
  - `ORCHESTRATOR_ACTION11=true/false`
- [ ] **5.2.2** Enable for Action 11 first (lowest volume)
- [ ] **5.2.3** Monitor for 3+ days, verify no regressions
- [ ] **5.2.4** Enable for Action 9 (medium volume)
- [ ] **5.2.5** Monitor for 3+ days, verify no regressions
- [ ] **5.2.6** Enable for Action 8 (highest volume)
- [ ] **5.2.7** Monitor for 7+ days, verify no regressions

#### 5.3 Full Cutover (Week 5+)
- [ ] **5.3.1** Set `ENABLE_UNIFIED_SEND_ORCHESTRATOR=true` as default
- [ ] **5.3.2** Remove per-action feature flags
- [ ] **5.3.3** Keep old code paths as dead code (do not delete yet)
- [ ] **5.3.4** Monitor for 2 weeks post-cutover
- [ ] **5.3.5** Create backup/rollback procedure documentation

#### 5.4 Cleanup (Week 7+)
- [ ] **5.4.1** Remove shadow mode logging code
- [ ] **5.4.2** Remove old send code from Action 8
- [ ] **5.4.3** Remove old send code from Action 9
- [ ] **5.4.4** Remove old send code from Action 11
- [ ] **5.4.5** Update documentation and README
- [ ] **5.4.6** Update `.github/copilot-instructions.md` with new architecture

---

### Phase 6: Metrics & Observability

#### 6.1 Unified Metrics
- [ ] **6.1.1** Create `messaging/send_metrics.py`
- [ ] **6.1.2** Track sends by trigger type (AUTOMATED/REPLY/OPT_OUT/APPROVED)
- [ ] **6.1.3** Track safety check block rates by check type
- [ ] **6.1.4** Track content generation time by source
- [ ] **6.1.5** Track API success/failure rates
- [ ] **6.1.6** Export metrics to Prometheus/Grafana

#### 6.2 Audit Trail
- [ ] **6.2.1** Create `send_audit_log` table or JSON log
- [ ] **6.2.2** Log every send decision with full context
- [ ] **6.2.3** Include decision path, safety check results, content source
- [ ] **6.2.4** Enable query by person_id, trigger_type, date range

---

### Estimated Impact

| Metric | Before | After |
|--------|--------|-------|
| Lines of send logic | ~11,000 across 3+ files | ~800 in orchestrator + 200 in helpers |
| Safety check locations | 4 separate implementations | 1 centralized |
| Database update locations | 4 separate implementations | 1 centralized |
| Time to add new send type | ~4 hours | ~30 minutes |
| Audit trail coverage | Partial/inconsistent | Complete/consistent |

---

### Dependencies & Prerequisites

- [ ] Ensure `call_send_message_api()` in [api_utils.py](api/api_utils.py) is stable (no changes needed)
- [ ] Ensure [message_types.py](messaging/message_types.py) state machine is stable
- [ ] Ensure [opt_out_detection.py](messaging/opt_out_detection.py) detector is stable
- [ ] Create feature flag infrastructure if not present
- [ ] Backup database before shadow mode deployment

---

### Remaining Future Enhancements (Not Blocking Production)

1. Enable auto-approval after building sufficient human review baseline
2. Activate TreeUpdateService in production after validation
3. Deploy Grafana dashboards via `python scripts/deploy_dashboards.py`
4. Real-time alerting for safety events

---

## Technical Debt (Commented Out Dead Code)

The following functions have been commented out as dead code (2025-12-18). They are preserved in case they become useful in the future.

| File | Function | Reason |
| ---- | -------- | ------ |
| `ai/ai_prompt_utils.py` | `quick_test()` | Replaced by `run_comprehensive_tests()`, never called |
| `actions/action10.py` | `detailed_scoring_breakdown()` | Debug helper function, never called in production |
| `actions/action10.py` | `get_user_criteria()` | Interactive user input helper, unused in production |
| `api/api_utils.py` | `print_group()` | Debug printing helper, never called |
| `actions/action7_inbox.py` | `_check_browser_health()` | Method defined but never called |
| `actions/action7_inbox.py` | `_calculate_api_limit()` | Method defined but never called |
| `actions/action7_inbox.py` | `_classify_message_with_ai()` | Method defined but never called |
| `actions/action7_inbox.py` | `_build_follow_up_context()` | Method defined but never called |
| `actions/action8_messaging.py` | `print_template_effectiveness_report()` | Report function never called |
| `actions/action8_messaging.py` | `_inject_research_suggestions()` | Helper function never called |
| `browser/selenium_utils.py` | `wait_for_element()` | Duplicate of `core/utils._wait_for_element` |
| `api/api_search_utils.py` | `get_api_relationship_path()` | Function defined but never called |

### Removal Candidates (False Positives in Dead Code Scan)

The dead code scanner flagged 146 items, but most are **false positives**:

1. **Flask routes** (`@api_bp.route`) - 10 items - Called by HTTP, not Python
2. **Mixin methods** (`TreeHealthChecksMixin`, etc.) - 23 items - Called via inheritance
3. **TypedDicts/Dataclasses** - 20+ items - Used for type hints
4. **Abstract/Protocol methods** - 8 items - Implemented by subclasses
5. **Test functions** (`module_tests()`) - Called by test runner
6. **`__init__` methods** - Called implicitly by Python

These should NOT be removed. The scanner doesn't understand decorators, inheritance, or type hints.

