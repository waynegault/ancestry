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

- [ ] *No outstanding review tasks — add the next item here when identified.*
