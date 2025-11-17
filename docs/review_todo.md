# Codebase Review Master Todo *(Updated 2025-11-17)*

All open work is captured in the single checklist below. Address items in priority order unless a dependency is noted.

- [x] **Strengthen Assertions** (Est. 2h)
  Deepen coverage in `gedcom_intelligence.py` and `message_personalization.py` with edge-case fixtures, explicit assertion messaging, and negative-path tests.

  **Completed Enhancements:**

  `gedcom_intelligence.py` (4 → 10 tests, 0.078s duration):
  - [x] Added `test_invalid_gedcom_data()` - Tests None, invalid objects, empty indi_index
  - [x] Added `test_edge_case_dates()` - Tests malformed date strings without crashes
  - [x] Added `test_special_characters_in_names()` - Tests José, Müller, O'Brien, MacLeòid, Władysław, empty, None
  - [x] Added `test_circular_relationships()` - Tests I1→I2→I1 parent loops
  - [x] Added `test_empty_analysis_result_structure()` - Validates all required keys present
  - [x] Added `test_large_dataset_performance()` - Tests 1000 individuals in <5s
  - [x] All assertions now include explicit error messages
  - [x] All 10 tests passing

  `message_personalization.py` (11 → 18 tests, 0.523s duration):
  - [x] Added `test_null_and_none_inputs()` - Tests None person_data/extracted_data/base_format_data
  - [x] Added `test_malformed_extracted_data()` - Tests invalid structured_names, vital_records, locations types
  - [x] Added `test_unicode_and_special_characters()` - Tests José María, Müller, Владимир, 李明, O'Connor-Władysław
  - [x] Added `test_extremely_long_inputs()` - Tests 500-char names, 1000-char places, 50+ records
  - [x] Added `test_missing_template_keys()` - Tests templates with missing placeholder keys
  - [x] Added `test_zero_and_negative_numbers()` - Tests shared_dna_cm=0.0, birth_year=-100, total_rows=-5
  - [x] Added `test_format_single_vital_record_edge_cases()` - Tests None, non-dict, empty dict, missing fields
  - [x] All assertions now include explicit error messages
  - [x] All 18 tests passing- [ ] **Tighten Lint & Type Enforcement** (Est. 3h)
   Gradually remove Ruff ignores (start with F821/E722), then raise Pyright severities from warnings to errors once violations are cleared.

- [ ] **Comment & Docstring Spot Check** (Est. 1h per sprint)
   After each merge, scan new comments/docstrings for tone/verbosity alignment and adjust immediately.

- [ ] **Schema Versioning & Lightweight Migrations** (Est. 2h)
   Introduce SQLite schema version stamps plus a minimal migration runner to unblock future columns without manual SQL.

- [ ] **Data Integrity Checker** (Est. 3h)
   Schedule periodic audits for soft deletes, UUID uniqueness, and cross-table consistency; emit alerts to Logs and Grafana.

- [ ] **Regression Guardrails** (Ongoing)
   Enforce the policy: run `python run_all_tests.py --fast` and `ruff check .` for every behavioral change before pushing.

- [ ] **Knowledge Graph & README Export** (Est. 1h)
   regenerate the code graph/README snapshots and commit the artifacts so downstream consumers have the latest state.

- [ ] **Maintainer Handoff Brief** (Est. 1h)
   Summarize outcomes, unresolved risks, and recommended sequencing for the next maintainer before project pause/transition.

- [ ] **Workflow Replay Capability** (Est. 4h)
   Capture anonymized session traces (API inputs/outputs, key browser steps) and provide a replay harness that stubs external calls for offline debugging. Success = `replay/` toolkit plus documentation on capturing runs.
