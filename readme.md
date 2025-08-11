<img alt="Ancestry Research Automation" width="0" height="0"/>

# Ancestry Research Automation


## USER GUIDE (What It Is & What You Can Do)

### Purpose
An intelligent assistant that automates large portions of your genealogical research workflow on Ancestry.com: collecting DNA match data, analyzing inbox conversations, sending personalized messages, and generating specific research tasks (Microsoft To‑Do) — all while protecting credentials and adapting performance automatically.

### Core Outcomes
- Centralized, always-fresh DNA match intelligence
- Higher quality and higher response-rate messaging to matches
- Actionable, prioritized research task lists instead of vague TODOs
- Structured extraction of genealogical entities (names, locations, relationships, records, questions, DNA info)
- Continuous quality scoring & regression protection

### Key Capabilities
1. DNA Match Management
   - Collect & refresh all matches with change tracking
   - Detect in-tree status and relationship paths (where available)
2. Intelligent Inbox Processing
   - AI classifies conversations (productive / desist / low value)
   - Extracts genealogical context for personalization & tasks
3. Personalized Messaging
   - 6+ enhanced templates with 20+ dynamic placeholder functions
   - Avoids duplicates, supports dry-run, resilient fallbacks
4. Research Task Generation
   - 8 specialized genealogical task template categories
   - Prioritized, specific, evidence-seeking task descriptions
5. GEDCOM & Tree Intelligence (Selected Modules)
   - Gap analysis, prioritization, and DNA cross-referencing helpers
6. Performance & Adaptation
   - Adaptive rate limiting (0.1–2.0 RPS) & smart batching
   - Performance dashboard & optimization recommendations
7. Quality Scoring & Safeguards
   - Unified extraction quality_score (0–100)
   - Baseline & regression detection + optional CI gate

### Typical User Flows
| Goal | Run These Steps |
|------|-----------------|
| First-time setup | Configure credentials → action6_gather → action7_inbox |
| Full daily cycle | action6_gather → action7_inbox → action9_process_productive → action8_messaging |
| Just send new messages | action8_messaging |
| Generate research tasks only | action9_process_productive |
| Monitor performance | performance_dashboard.py |
| Guard quality | prompt_telemetry.py (baseline / regression) |

### Quick Start
```bash
git clone https://github.com/waynegault/ancestry.git
cd ancestry
pip install -r requirements.txt
python run_all_tests.py          # verify environment
python action6_gather.py         # gather DNA matches
python action7_inbox.py          # process inbox with AI
python action9_process_productive.py  # generate tasks
python action8_messaging.py      # send personalized messages
```

### Quality & Safety
- 100% passing test baseline (hundreds of tests across modules)
- Credential encryption with Fernet + system keyring
- Graceful degradation & retry logic; circuit breaker safeguards
- Optional quality regression gate: `python quality_regression_gate.py`

### What You’ll See Produced
- SQLite DB (ancestry.db) tracking people, matches, conversations
- Logs/ directory with telemetry, alerts, quality baselines
- Microsoft To‑Do tasks (when properly configured) with actionable descriptions

### High-Level Feature Summary
| Feature | Benefit |
|---------|---------|
| AI Entity Extraction | Higher specificity for tasks & messages |
| Message Personalization | Increased match response probability |
| Task Templates | Faster movement from discussion to evidence search |
| Adaptive Rate Limiting | Stable long-running sessions w/o throttling |
| Quality Scoring & Alerts | Early detection of silent degradation |
| Regression Gate | Protects median extraction quality over time |

---

## DEVELOPER GUIDE (How It Works)

### Architectural Layers
1. Action Scripts (workflow entrypoints): action6–11
2. Core Infrastructure (core/): session, database, browser, api, error handling
3. AI & Personalization: ai_interface.py, ai_prompts.json, message_personalization.py
4. Task Generation: genealogical_task_templates.py + action9 integration
5. Quality & Telemetry: extraction_quality.py, prompt_telemetry.py, quality_regression_gate.py
6. Performance & Adaptation: adaptive_rate_limiter.py, performance_dashboard.py
7. Security & Config: security_manager.py, config/ package

### Data Extraction & Quality Scoring
compute_extraction_quality combines:
- Entity richness (names, vitals, relationships, locations, etc.) up to 70 pts (penalty: -10 if no names)
- Task specificity (compute_task_quality) up to 30 pts (verbs, year, record terms, specificity tokens, healthy length, filler penalties)
- Bonus for 3–8 well‑formed tasks; penalty if zero tasks
Telemetry captures quality_score for each extraction event.

Baseline & Regression:
```bash
python prompt_telemetry.py --build-baseline --variant control --window 300 --min-events 8
python prompt_telemetry.py --check-regression --variant control --window 120 --drop-threshold 15
python quality_regression_gate.py  # exit 1 on regression
```

### Telemetry & Experimentation
- JSONL appends (Logs/prompt_experiments.jsonl)
- Fields: variant_label, parse_success, counts, tasks, raw size, quality_score
- Analysis heuristics (control vs alt) with auto-alerting (Logs/prompt_experiment_alerts.jsonl)
- Baseline JSON: Logs/prompt_quality_baseline.json

### AI Integration
- Prompt library (ai_prompts.json) aligned with structured output contract
- ai_interface.py centralizes provider calls, variant labeling, and response normalization
- Supports specialized prompts (DNA analysis, record research, reply generation)

### Messaging Personalization
- Templates use placeholders resolved by message_personalization.py (20+ functions)
- Robust fallback chain ensures message always sends even with sparse data
- Productivity classification informs template selection & task generation

### Research Task Generation
- genealogical_task_templates.py defines domain templates (vital, census, immigration, DNA, etc.)
- action9_process_productive maps extracted entities + conversation signals → prioritized tasks
- Optional enrichment & de-dup feature flags (enable_task_enrichment, enable_task_dedup)

### Performance & Adaptation
AdaptiveRateLimiter:
- Monitors success & 429 rates; adjusts effective RPS (0.1–2.0)
Smart batching selects batch size optimizing target cycle time
Performance dashboard aggregates per-session metrics & recommendations.

### Database & Persistence
- SQLite schema (people, dna_matches, conversation_logs, tasks) via DatabaseManager
- Caching layers for GEDCOM parsing & API responses
- Backup & restore operations (action menu options)

### Security Model
- Encrypted credentials (Fernet) + system keyring master key
- Minimal scope storage; local-only persistence
- CSRF token retrieval & cookie transfer from Selenium to requests session

### Testing Strategy
- run_all_tests.py orchestrates discovery, optional parallelism, performance metrics
- Categories: unit, integration, performance, error handling, personalization, extraction quality
- Quality gate script provides CI enforcement layer without modifying core tests

### Configuration & Flags
Environment & config manager unify defaults; feature toggles:
- enable_task_dedup
- enable_task_enrichment
- enable_prompt_experiments
Supports multiple AI providers via AI_PROVIDER env.

### Extensibility Guidelines
- Add new prompt variant: update ai_prompts.json + version label → capture telemetry automatically
- Add new task template: extend genealogical_task_templates.py & integrate selection logic in action9
- Introduce new quality dimension: extend extraction_quality.compute_extraction_quality (preserve backward compatibility by additive fields)

### Planned Enhancements
- Statistical significance (bootstrap / Mann-Whitney) for variant quality deltas
- Separate task_quality_score telemetry field
- Baseline rotation policy (age / sample volume)
- Cost efficiency metrics (quality per 1K chars/tokens)

### Troubleshooting (Developer Focus)
| Symptom | Check | Likely Cause |
|---------|-------|--------------|
| quality_regression_gate exits 1 | baseline & latest medians | Real score drop or stale baseline |
| Low success_rate in telemetry | parse_success flags | Prompt drift or extraction schema change |
| Frequent 429s | adaptive_rate_limiter stats | Too aggressive manual overrides |
| Missing tasks | action9 logs & feature flags | Enrichment flag disabled or empty extraction |

### Project Structure (Condensed)
```
action*.py            # Workflow drivers
adaptive_rate_limiter.py
ai_interface.py / ai_prompts.json
extraction_quality.py
genealogical_task_templates.py
message_personalization.py
prompt_telemetry.py / quality_regression_gate.py
performance_dashboard.py
core/  config/  utils/  (infrastructure & shared services)
Logs/ (telemetry, alerts, baseline)
```

### Contribution Checklist
- Add/update tests (maintain green suite)
- Run quality gate (if baseline exists)
- Update README Developer Guide if public surface changes
- Avoid breaking telemetry schema (additive changes preferred)

---

## LICENSE
MIT License – see LICENSE file.

## ACKNOWLEDGMENTS
Design emphasizes: robustness, observability, reversible low-risk incremental improvements, and genealogy-specific domain modeling.

Last Updated: 2025-08-11

