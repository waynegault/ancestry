# Architecture Notes Snapshot (2025-11-11)

- **Session & Resource Orchestration**
  - `core/session_manager.py` is the sole authority for browser, database, and API lifecycle; all actions must invoke work via `exec_actn()` in `main.py` so SessionManager can prepare the necessary resources.
  - No module may instantiate WebDriver, database engines, or API clients directly; helpers obtain them via the SessionManager and `session_utils` global helpers.
  - Browser sessions synchronize API cookies; proactive health checks refresh sessions every ~25 minutes during long runs.

- **Rate Limiting & Concurrency**
  - Token-bucket `RateLimiter` (utils.py) enforces sequential API access with adaptive backoff; configured at `REQUESTS_PER_SECOND=0.3`.
  - Only SessionManager constructs the limiter; API calls flow through `SessionManager.api_manager` to guarantee serialization.
  - Parallel execution for API work is explicitly forbidden after prior 429 incidents; validation requires `validate_rate_limiting.py` plus multi-page Action 6 runs and log review.

- **Database & Persistence**
  - SQLAlchemy ORM in `database.py` with uppercase UUID requirement (`Person.uuid` stored and queried with `.upper()`); soft deletes via `deleted_at`.
  - Checkpointing (Action 6) writes JSON under `Cache/action6_checkpoint.json` for resume support; uses atomic writes and age-based expiration.
  - Cookie persistence flows: `BrowserManager._load_saved_cookies` delegates to `utils._load_login_cookies`; ancestry cookies cached in `ancestry_cookies.json`.

- **Action Modules Pattern**
  - Actions 6–10 follow `exec_actn()` resource gating; each returns `bool` success and is invoked through menu definitions in `main.py`.
  - Action 6 gathers DNA matches sequentially with caching, circuit breaker, proactive session refresh, and rate-limit enforced API calls.
  - Action 7 processes inbox conversations and invokes AI prompt variants to classify intent and extract entities.
  - Action 9 generates tasks from productive conversations, referencing `genealogical_task_templates.py` and quality scoring logic.
  - Action 10 prioritizes GEDCOM data then API fallback; expects authenticated cookies from shared session state.

- **Testing & Quality Gates**
  - Embedded `TestSuite` harness (`test_framework.py`) used in-module; `run_all_tests.py` orchestrates 58 modules with quality scoring.
  - `quality_regression_gate.py` enforces AI prompt performance by checking telemetry medians; `prompt_telemetry.py` maintains JSONL logs and baselines.
  - Linting via Ruff; Pyright configured for type checking; all new functions require annotations.

- **AI Integration**
  - `ai_interface.py` abstracts providers (Google Gemini primary, DeepSeek fallback, local LM support) and logs telemetry for every call.
  - Prompt templates live in `ai_prompts.json`; variants support A/B testing; quality scoring via `extraction_quality.py`.
  - Local LLM flow can auto-start LM Studio based on `.env` settings; `demo_lm_studio_autostart.py` validates setup.

- **Error Handling & Resilience**
  - `core/error_handling.py` defines `AncestryException` hierarchy, differentiating retryable vs fatal errors; decorators (`retry_on_failure`, `graceful_degradation`, `error_context`) encapsulate recovery logic.
  - Circuit breaker pattern prevents repeated failures after five consecutive errors; used heavily in Action 6 to guard API loops.
  - Session and API utilities include exponential backoff calculators and watchdog timers to cap request duration.

- **Configuration Governance**
  - `config/config_schema.py` hosts dataclasses for database, Selenium, API, logging, cache, security, and aggregate schema validation; `config_manager.py` loads `.env` and ensures type safety.
  - Environment variables (e.g., `MAX_PAGES`, `SKIP_LIVE_API_TESTS`, AI provider keys) control runtime behavior; instructions stress not altering rate limiting without validation.

- **Documentation & Artifacts**
  - Knowledge graph maintained in `docs/code_graph.json`; progress tracked in `docs/review_todo.md` with phased checklist.
  - Repository inventory now captured in `docs/repo_inventory.md`; README due for overhaul per Phase 5 plan.
