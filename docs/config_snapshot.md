# Configuration Snapshot (2025-11-11)

- **Environment Variables (.env expectations)**
  - Credentials: `ANCESTRY_USERNAME`, `ANCESTRY_PASSWORD` (required for login automation).
  - Rate limiting: `REQUESTS_PER_SECOND=0.3` (validated ceiling; sequential API processing only).
  - Processing limits: `MAX_PAGES`, `MAX_INBOX`, `MAX_PRODUCTIVE_TO_PROCESS`, `MATCHES_PER_PAGE`.
  - Session health: `HEALTH_CHECK_INTERVAL_PAGES`, `SESSION_REFRESH_THRESHOLD_MIN`.
  - Modes/flags: `APP_MODE`, `DEBUG_MODE`, `SKIP_LIVE_API_TESTS`.
  - AI providers: `AI_PROVIDER`, `MOONSHOT_API_KEY`, `DEEPSEEK_API_KEY`, `GOOGLE_API_KEY`, `LOCAL_LLM_*` (base URL, API key, model, auto-start controls).
  - Microsoft Graph integration: `MS_CLIENT_ID`, `MS_CLIENT_SECRET`, `MS_TENANT_ID`.
  - LM Studio automation: `LM_STUDIO_PATH`, `LM_STUDIO_AUTO_START`, `LM_STUDIO_STARTUP_TIMEOUT`.

- **Configuration Dataclasses (`config/config_schema.py`)**
  - `DatabaseConfig`: Paths, connection pooling, SQLite pragmas, backup cadence, monitoring toggles; enforces uppercase UUIDs and ensures directories exist.
  - `SeleniumConfig`: Chrome driver/browser paths, headless flag, retry/timeouts, window size, 2FA timeout, feature toggles (images, plugins, notifications).
  - `APIConfig`: Base URLs, credential fields, AI keys, retry/backoff parameters, rate-limit defaults, health check intervals.
  - `LoggingConfig`: Log file path, rotation policy, log levels (console vs file), structured logging switch.
  - `CacheConfig`: Cache directory, TTL defaults, eviction strategy, memory caps.
  - `SecurityConfig`: Feature flags for encryption, credential storage policy, two-factor enforcement, audit logging switch.
  - `TestConfig`: Controls test dataset paths, mock toggles, performance thresholds.
  - `ConfigSchema`: Aggregates all sub-configs, registers validation rules, exposes `load()` helpers for `config_manager`.

- **Loader Pipeline (`config/config_manager.py`)**
  - Reads `.env` via `python-dotenv`, applies type casting, and instantiates `ConfigSchema`.
  - Exposes `config_schema` singleton for global use; logs validation results and raises on failure.
  - Provides helper methods to refresh configuration at runtime if `.env` changes.

- **Operational Scripts Influencing Configuration**
  - `validate_rate_limiting.py`: Validates rate limiter settings against live runs; required prior to changing `REQUESTS_PER_SECOND`.
  - `run_all_tests.py`: Orchestrates module tests, depends on environment flags like `SKIP_LIVE_API_TESTS`.
  - `quality_regression_gate.py`: Uses telemetry configuration to guard AI prompt regressions.
  - `prompt_telemetry.py`: Generates baselines, stats, and alert logs; respects telemetry directories configured in `.env`.
  - `demo_lm_studio_autostart.py`: Exercises local LLM auto-start settings.
  - `main.py`: Entry point reading `config_schema`, sets up logging, starts SessionManager with configured rate limiter and resource managers.

- **Persistent Artifacts & Paths**
  - Database: `Data/ancestry.db` (SQLite) with backups controlled by `DatabaseConfig`.
  - Cache: `Cache/` subdirectories for rate limiter state, performance metrics, and session checkpoints.
  - Logs: `Logs/app.log` default target; telemetry JSONL files under `Logs/` (prompt experiments, alerts).

- **Governance Notes**
  - Production environment requires explicit database file path and backups enabled (enforced by `DatabaseConfig` rules).
  - Any change to rate limiting or session-related `.env` values must be documented and validated using the prescribed commands in README and `.github/copilot-instructions.md`.
