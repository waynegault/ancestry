# Ancestry Research Platform - Implementation Roadmap

## 1. Pre-Production Validation (Low Priority)

**Objective**: Validate system against real historical data before full production deployment.

- [ ] **Execute Dry-Run Validation**
  - Run `validate` command against 50+ historical PRODUCTIVE conversations.
  - Target: 90%+ parse success rate.
- [ ] **Quality Audit**
  - Manual audit comparing AI-generated drafts vs actual human replies.
  - Document edge cases and failure modes.
  - Measure extraction quality scores (target: median >70).
- [ ] **Safety Verification**
  - Verify opt-out detection catches all DESIST patterns (zero false negatives).

## 2. Performance & Reliability (Medium Priority)

**Objective**: Optimize system performance and ensure robust error recovery.

- [x] **Test Performance**
  - Enable `TestResultCache` in `run_all_tests.py` for faster test runs.
- [x] **Log Analysis**
  - Connect `--analyze-logs` CLI flag to `print_log_analysis()` function.
- [x] **Health Monitoring**
  - Integrate `health_monitor.py` with `SessionManager` for auto-recovery of stale sessions.

## 3. CI/CD & Operations (Low Priority)

**Objective**: Automate testing and deployment workflows.

- [ ] **GitHub Actions**
  - Add `quality_regression_gate.py` to CI pipeline.
  - Set up automated test runs on Pull Requests.
- [ ] **Deployment**
  - Configure automated Grafana dashboard deployment.

## 4. Refactoring & Technical Debt

**Objective**: Improve maintainability and reduce complexity (Medium Priority).

- [ ] **Modularize Large Actions**: Split `action6_gather.py` and `action7_inbox.py` into smaller sub-modules.
- [ ] **Refactor Orchestration**: Simplify `_process_single_page` in `actions/gather/orchestrator.py`.
- [x] **Centralize Menu Logic**: Use data-driven registry for menu actions.
- [ ] **Security**: Implement PII redaction in logs.
