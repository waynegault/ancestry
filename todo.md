# Production Readiness: Remaining Action Items

This document contains actionable items for the Ancestry Research Automation Platform codebase.

## Priority Legend
- 🔴 **HIGH** - Should be addressed before production deployment
- 🟠 **MEDIUM** - Should be addressed for code quality, can be done post-launch
- 🟢 **LOW** - Nice-to-have improvements, can be deferred

---

## 1. Large File Opportunities

### 🟢 LOW: File Size Reduction

| File | Lines | Suggestion |
|------|-------|------------|
| `utils.py` | ~4700 | Consider extracting login/consent functions to `auth_utils.py` |
| `core/session_manager.py` | ~3600 | Consider extracting cookie management to separate module |
| `core/error_handling.py` | ~2029 | Split into: `exceptions.py`, `retry.py`, `circuit_breaker.py`, `decorators.py` |

Only split if maintainability becomes an issue.

---

## 2. Architecture Improvements

### 🟢 LOW: Extract Business Logic from UI/Orchestration

**Problem:** Action files mix orchestration, business logic, API calls, and UI concerns.

**Suggested Approach:**
Create `actions/gather/` package structure:
- `coordinator.py` - high-level orchestration
- `api_client.py` - API interactions
- `data_processor.py` - business logic
- `persistence.py` - database operations

### 🟢 LOW: Unified Cache Interface with Clear Invalidation

**Problem:** Multiple cache systems with inconsistent interfaces.

**Suggested Approach:**
1. Create `core/cache/interface.py` with `Cache` protocol
2. Implement adapters: `DiskCache`, `MemoryCache`, `DatabaseCache`
3. Add explicit TTL and cache versioning

### 🟢 LOW: Browser Automation Service Layer

**Problem:** Selenium code mixed with business logic.

**Suggested Approach:**
1. Create `services/browser_service.py` with high-level operations
2. Implement page object pattern
3. Create mock browser service for testing

### ✅ ~~LOW: Runtime Feature Toggle Framework~~ COMPLETED

**Status:** COMPLETED

Created `core/feature_flags.py` with:
- `FeatureFlags` singleton class with thread-safe access
- Environment variable overrides (`FEATURE_FLAG_<NAME>=true/false`)
- Percentage-based rollout with consistent user bucketing (MD5 hashing)
- Runtime overrides via `set_override()`/`clear_override()`
- JSON config file loading/saving
- 9 comprehensive tests

### ✅ ~~LOW: Database Schema Evolution System~~ COMPLETED

**Status:** COMPLETED

Enhanced `core/schema_migrator.py` with:
- `Migration.downgrade` function for rollback capability
- `Migration.depends_on` tuple for migration dependencies
- `apply_pending_migrations(dry_run=True)` for preview mode
- `rollback_migration()` for single migration rollback
- `rollback_to_version()` for rolling back to a target version
- `MigrationRegistry.validate_dependencies()` for dependency validation
- CLI options: `--dry-run`, `--rollback VERSION`, `--rollback-to VERSION`, `--validate`
- Custom exceptions: `DependencyError`, `RollbackError`
- 10 comprehensive tests (up from 2)

---

## 3. Observability & Monitoring

### ✅ ~~LOW: Production Performance Monitoring~~ COMPLETED

**Status:** COMPLETED

Created `observability/apm.py` with OpenTelemetry-style tracing:
- `Span` class with attributes, events, status tracking
- `SpanContext` for trace/span ID propagation
- `Tracer` singleton for creating and managing spans
- `@trace` decorator for automatic function instrumentation
- `APMConfig` for sampling rate, buffer size, export interval
- `SpanExporter` protocol with `ConsoleSpanExporter` and `JSONFileSpanExporter`
- Automatic exception recording in spans
- Background export with configurable buffer
- 10 comprehensive tests

---

## 4. Future Enhancements

### ⚠️ LOW (v2.0): Async/Await for I/O Operations

**Problem:** Sequential API calls waste time; Action 6 takes hours for large datasets.

**Suggested Approach:**
Migrate to aiohttp and implement async database operations.

**Note:** Large change, consider for v2.0 after current architecture stabilizes.

---

## Summary

| Category | Count | Priority Breakdown |
|----------|-------|-------------------|
| Large Files | 1 item | 1 LOW |
| Architecture | 5 items | 2 LOW, 3 COMPLETED |
| Observability | 1 item | COMPLETED |
| Future | 1 item | v2.0 |

**Total Remaining Items:** 5 actionable LOW priority items
**Completed This Session:** APM module, Enhanced Schema Migrator
**Critical Issues:** 0 (All HIGH and MEDIUM priority items completed!)
