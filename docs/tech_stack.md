# Tech Stack & Infrastructure - Phase 1

## Core Dependencies

### Web Automation
*   **Selenium**: `4.31.0+` (Browser automation)
*   **WebDriver Manager**: (Driver management)
*   **CloudScraper**: `1.2.71` (Anti-bot protection bypass)

### Database
*   **SQLAlchemy**: `2.0.40+` (ORM)
*   **DiskCache**: `5.6.3` (Persistent caching)
*   **SQLite**: (Embedded database, inferred from `Data/ancestry.db` in instructions)

### AI / LLM
*   **Google Generative AI**: `0.8.4+` (Primary provider - Gemini)
*   **OpenAI**: `1.82.0` (Shared client for DeepSeek/Moonshot)

### Testing
*   **Pytest**: `8.3.5` (Test runner)
*   **Pytest-Cov**: `6.1.1` (Coverage reporting)
*   **Coverage**: `7.8.0` (Code coverage)

### Utilities
*   **Requests**: `2.32.3+` (HTTP client)
*   **BeautifulSoup4**: `4.13.3+` (HTML parsing)
*   **Pydantic**: `2.11.3` (Data validation)
*   **Pandas**: `2.2.3` (Data manipulation)
*   **Tqdm**: `4.67.1+` (Progress bars)
*   **Psutil**: (System monitoring)

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
