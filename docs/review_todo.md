# Codebase Review Master Todo *(Updated 2025-11-18)*

All review-driven work is tracked here. Completed items remain documented with implementation notes until the next review cycle so future auditors can see what changed and why.

## CHeck that these have actually been done

- [x] **Pylance Diagnostics Sweep** (Nov 18)
  - `npx pyright` now runs clean with 0 errors / 0 warnings across the full workspace, establishing a post-refactor baseline after relocating `ai_api_test.py`.
  - Added developer documentation so anyone can rerun the sweep (`npx pyright`) and matching lint pass (`ruff check`) before landing changes that touch typed modules or AI tooling.

- [x] **Linting Configuration Hardening** (Low Priority, High Impact)
  - Pyright now runs in `standard` mode with `reportReturnType`, `reportUnusedVariable`, and `reportDuplicateImport` elevated to errors, closing out Phase 1 from the original plan.
  - Ruff picks up PLR cyclomatic-complexity (`PLR0912`) and excessive-argument (`PLR0913`) checks with tuned thresholds (`max-branches=25`, `max-args=10`) so we get actionable signals without blocking critical flows.
  - Documentation in `readme.md` explains the stricter defaults plus how to request per-file exemptions when legacy modules cannot yet comply.

- [x] **Memory/Test Guardrails + Rate-Limiter Messaging** (Nov 18)
  - `_test_memory_efficiency()` now filters out imported modules, classes, and functions before counting globals so it measures only true stateful variables (limit tightened to <80) while still keeping the module-size check intact.
  - Adaptive rate limiter info log now reads `✅ Endpoint cap enforced: clamped fill rate ...` to make the throttle reason obvious during ops reviews and match the UX request for a leading check mark.

- [x] **Ruff 0.14 Upgrade for Python 3.12.8** (Nov 18)
  - Upgraded the toolchain to `ruff==0.14.5`, added `target-version = "py39"`, and enabled preview rules so `UP045` ignores remain valid on the newer binary.
  - Cleaned up `utils.py` (spacing, literal placeholders, over-indented blocks, long membership lists) so the stricter linter passes there; the rest of the repo now surfaces ~500 real lint issues (mostly `PLR`/`PLC` suggestions) that will need phased follow-up.

## Backlog

- [ ] **Type Ignore Eradication (Nov 19)**
  - Cycle 1 in progress: `core/system_cache.py` now uses a typed stub for `BaseCacheModule`, eliminating the three `# type: ignore[misc]` suppressions without sacrificing runtime behavior.
  - Cycle 2 (Nov 19): `action6_gather.py` now keeps the default failure threshold constant (`CRITICAL_API_FAILURE_THRESHOLD_DEFAULT`) separate from the mutable runtime value, so the global reassignment no longer needs a suppression and downstream logging still reports both numbers.
  - Cycle 3 (Nov 19): `action6_gather.py` bulk operations now pass concrete SQLAlchemy `__mapper__` objects, use typed insert/update collections, and rely on `select()`-powered lookups for DnaMatch existence checks, clearing the remaining ignores around person/DNA/FamilyTree inserts, updates, and ID mapping helpers.
  - `_lookup_existing_persons` and `_get_person_id_mapping` have been rewritten to use SQLAlchemy's `select()`/`scalars()` APIs with typed dictionaries, removing the `.filter(... )` and `return-value` ignores while retaining eager loading and recovery queries.
  - The relationship-probability limiter now builds a typed `set[str]` for medium-priority UUIDs, ensuring the tuple return annotation matches without a `# type: ignore[return-value]`.
  - `action6_gather.py` constants are now dynamically derived via lowercase working vars, removing all `# type: ignore[misc]` fallbacks and cleaning up untyped third-party import suppressions (`cloudscraper`) plus unused `ENOVAL` baggage.
  - `action7_inbox.py` switched to `TYPE_CHECKING` aliases for `BrowserError`/`APIError`/`AuthenticationError`, so the assignment suppressions are gone while subclasses still inherit the concrete error types.
  - `cache.py` now imports DiskCache directly with lowercase staging for `CACHE_DIR`, letting us drop the legacy import/path suppressions.
  - `core/session_cache.py` and `observability/metrics_exporter.py` no longer rely on `# type: ignore` for logger setup, test-runner wiring, or fallbacks; both modules now expose wrapper functions with the exact signatures Pyright expects, and the repository’s Pyright run is back to 0 errors / 0 warnings.
  - `gedcom_intelligence.py` test helpers now subclass `GedcomIntelligenceAnalyzer` instead of monkey-patching, removing three ignores while keeping deterministic fixture data for gap detection coverage.
  - `test_utilities.py` loads the optional `gedcom` dependency via `import_module` so the function signature stays typed and the `# type: ignore` is no longer needed even when the package is absent.
  - `config/config_manager.py` declares a typed optional `load_dotenv` callable so the ImportError fallback no longer needs an assignment suppression.
  - `cache.py` relies on `Sized`/`Iterable` casts and safe attribute access to satisfy Pyright for `len(cache)`/iteration paths and the legacy `module_name` property, clearing the remaining ignores in that module.
  - `action6_gather.py` now validates ethnicity metadata via an `Any` staging variable and relies on the corrected Optional percentage typing so the helper no longer needs suppressions for metadata guards or percentage normalization.
  - `dna_ethnicity_utils.py` normalizes comparison payloads to `Optional[int]` values (with defensive conversions) so downstream code can reason about missing data without masking type errors.
  - Remaining work: ~180 additional `# type: ignore[...]` sites across action modules, ORM helpers, and GEDCOM tooling; need staged sweeps (imports first, then attr/arg issues) with Pyright checks and regression tests after each tranche.
