# Tech Stack & Infrastructure - Phase 1

## Core Dependencies

### Web Automation
*   **Selenium**: `4.40.0+` (Browser automation)
*   **WebDriver Manager**: (Driver management)
*   **CloudScraper**: `1.2.71` (Anti-bot protection bypass)

### Database
*   **SQLAlchemy**: `2.0.46+` (ORM)
*   **DiskCache**: `5.6.3` (Persistent caching)
*   **SQLite**: (Embedded database, inferred from `Data/ancestry.db` in instructions)

### AI / LLM
*   **Google Generative AI**: `0.8.6+` (Primary provider - Gemini)
*   **OpenAI**: `2.17.0+` (Shared client for DeepSeek/Moonshot)
*   **xai-sdk**: `1.4.0` (Grok AI)
*   **Providers via `OpenAICompatibleProvider`**: DeepSeek, Moonshot/Kimi, Local LLM (LM Studio), Grok, Inception, Tetrate

### Testing
*   **Pytest**: `9.0.2+` (Test runner)
*   **Pytest-Cov**: `7.0.0+` (Coverage reporting)
*   **Coverage**: `7.13.4+` (Code coverage)

### Utilities
*   **Requests**: `2.32.5+` (HTTP client)
*   **BeautifulSoup4**: `4.14.3+` (HTML parsing)
*   **Pydantic**: `2.12.5+` (Data validation)
*   **Pandas**: `3.0.0+` (Data manipulation)
*   **Tqdm**: `4.67.3+` (Progress bars)
*   **Psutil**: (System monitoring)
*   **RapidFuzz**: `3.14.3` (Fuzzy matching)
*   **Flask**: `3.1.2` (Web UI)
*   **prometheus_client**: `0.24.1` (Observability)
*   **tabulate**: `0.9.0` (Table formatting)
*   **tenacity**: `9.1.4` (Retry logic)

## Infrastructure Stability

### Rate Limiting
*   **Implementation**: `core/rate_limiter.py`
*   **Algorithm**: Adaptive Token Bucket
*   **Key Features**:
    *   **Single Source of Truth**: `fill_rate` controls request flow.
    *   **Adaptive Learning**: Adjusts `fill_rate` based on API feedback (429 errors vs. successes).
    *   **Balanced Speedup**: Requires 50 successes before increasing rate.
    *   **Gentle Slowdown**: Decreases rate by ~12% (0.85 factor) on 429 errors.
    *   **Thread Safety**: Uses `threading.Lock()` for concurrent access safety.
    *   **Persistence**: State is persisted to maintain rate limits across restarts.

### Session Management
*   **Implementation**: `core/session_manager.py`
*   **Strategy**: Central Orchestrator Pattern
    *   Coordinates `DatabaseManager`, `BrowserManager`, and `APIManager`.
    *   Acts as the single entry point for all resource-heavy operations.
*   **Lifecycle Management**:
    *   States: `UNINITIALIZED`, `RECOVERING`, `READY`, `DEGRADED`.
    *   Transitions are guarded and logged.
*   **Stability Features**:
    *   **Circuit Breaker**: `SessionCircuitBreaker` prevents cascading failures.
    *   **Proactive Refresh**: Logic exists to refresh sessions before expiry (e.g., cookie sync).
    *   **Error Handling**: Uses decorators like `@retry_on_failure`, `@graceful_degradation`, and `@timeout_protection`.
    *   **Performance Caching**: Uses `core.session_cache` to optimize manager initialization.
