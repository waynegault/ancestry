# Repository Inventory (Baseline Snapshot)

_Last updated: 2025-11-11_

## Top-Level Layout

- action modules (`action6_gather.py` … `action10.py`)
- core infrastructure (`core/`, `config/`, `database.py`, `session_utils.py`)
- utilities (`utils.py`, `standard_imports.py`, `memory_utils.py`, etc.)
- AI integration (`ai_interface.py`, `ai_prompt_utils.py`, `ai_prompts.json`)
- Scripts & diagnostics (`diagnose_*`, `demo_lm_studio_autostart.py`, `analyze_test_quality.py`)
- Documentation (`docs/`, `readme.md`)
- Data & cache (`Data/`, `Cache/`, `Logs/`, `ancestry_cookies.json`)
- Tests (`test_framework.py`, `test_diagnostics.py`, inline module tests)

## Key Directories

- `core/`: SessionManager, BrowserManager, APIManager, DatabaseManager, enhanced recovery, dependency injection.
- `config/`: Config schema and loader.
- `Cache/`: Rate limiter state, session checkpoints, performance snapshots.
- `Data/`: SQLite database (`ancestry.db`) and related assets.
- `Logs/`: Application and telemetry logs.
- `docs/`: Review artifacts (code graph, progress tracking, inventory).
- `test_data/`, `test_examples/`: Fixtures for testing and demos.

## Principal Files (Highlight)

- `main.py`: CLI entry point orchestrating actions via SessionManager.
- `utils.py`: Monolithic utility hub (login, navigation, rate limiting, formatting, UBE, testing harness).
- `action6_gather.py`: DNA match gathering pipeline with checkpoints, rate limiting, and performance metrics.
- `action7_inbox.py`: Inbox scraping and AI classification of conversations.
- `action8_messaging.py`: Automated messaging with personalization templates.
- `action9_process_productive.py`: Task generation for productive conversations.
- `action10.py`: Placeholder/next action module.
- `run_all_tests.py`: Orchestrates 58 embedded module tests (sequential/parallel modes).
- `quality_regression_gate.py`: Ensures AI extraction quality thresholds.
- `prompt_telemetry.py`: AI telemetry aggregation and baselining.
- `rate_limiter.py`: Adaptive rate limiter state persistence.
- `ai_interface.py`: Abstraction over Gemini/DeepSeek providers.

## Test Suites

- `run_all_tests.py` for full test pass (sequential, fast, analyze logs).
- Inline `TestSuite` usage within modules (`utils.py`, `memory_utils.py`, etc.).
- Specialized diagnostics (`test_diagnostics.py`, `test_utilities.py`).

## Supporting Assets

- `.env` / `.env.example`: Environment configuration.
- `requirements.txt`: Python dependencies.
- `pyrightconfig.json`, `ruff.toml`, `.pre-commit-config.yaml`: Quality tooling configs.
- `research_guidance_prompts.py`, `research_suggestions.py`: Prompt libraries.
- `lm_studio_manager.py`, `demo_lm_studio_autostart.py`: Local model orchestration helpers.

## Notes

- Inventory focuses on repos’ current structure; detailed module summaries tracked in `docs/code_graph.json`.
- Future updates: add size/date metadata or categorize tests/actions more granularly if useful.
