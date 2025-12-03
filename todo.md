# Ancestry Research Platform - Implementation Roadmap

## Production Readiness Audit - Remaining Tasks

This document tracks the remaining improvements required for production readiness.

---

## 1. Code Consolidation & Cleanup

### 1.1 Cookie Synchronization Duplication

**Pattern**: Cookie syncing logic exists in multiple locations (`core/session_manager.py` and `core/api_manager.py`).

**Action**:

- [x] Consolidate into `SessionManager` as single `sync_browser_cookies()` method
- [x] Have `APIManager` delegate to SessionManager for cookie operations

### 1.2 ConfigManager Singleton Migration

**Status**: Singleton pattern implemented, but migration is gradual.

**Action**:

- [ ] Gradually migrate remaining files to use `get_config_manager()` instead of direct instantiation

### 1.3 sys.path.insert() Cleanup

**Problem**: 96 files contain redundant `sys.path.insert()` calls.

**Action**:

- [ ] Create script to remove sys.path patterns from all 96 files
- [ ] Test all 148 modules after removal
- [ ] Verify standalone script execution still works

### 1.4 Legacy Code Review

**Action**:

- [ ] `genealogy/genealogical_normalization.py`: Verify if AI still returns legacy format; if not, remove `LEGACY_TO_STRUCTURED_MAP` and `_promote_legacy_fields()`
- [ ] `testing/verify_opt_out.py`: Review legacy `check_message` method usage

---

## 2. Test Quality Improvements

### 2.1 Minimal Assertion Tests

**Status**: Low priority, but can be improved.

**Action**:

- [ ] `ui/__init__.py`: Improve menu render existence check
- [ ] `ui/menu.py`: Improve menu render existence check

---

## 3. Pre-Production Validation

**Status**: Ready for manual validation
**Priority**: Low (requires production data access)

These tasks require manual testing with real historical data and cannot be automated:

- [ ] **Execute Dry-Run Validation**
  - Run `validate` command against 50+ historical PRODUCTIVE conversations
  - Target: 90%+ parse success rate

- [ ] **Quality Audit**
  - Manual audit comparing AI-generated drafts vs actual human replies
  - Document edge cases and failure modes
  - Measure extraction quality scores (target: median >70)
