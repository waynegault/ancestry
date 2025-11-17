# Codebase Review Master Todo *(Updated 2025-11-17)*

All open work is captured in the single checklist below. Address items in priority order unless a dependency is noted.

- [x] **Comment & Docstring Spot Check** (Est. 1h per sprint)
  scan  comments/docstrings for tone/verbosity alignment and adjust immediately.

  **Completed:**
  - [x] Reviewed module docstrings in gedcom_intelligence.py and message_personalization.py
  - [x] Replaced verbose corporate jargon with concise, professional descriptions
  - [x] Reduced gedcom_intelligence.py docstring from 47 lines to 15 lines
  - [x] Reduced message_personalization.py docstring from 43 lines to 13 lines
  - [x] Maintained all essential information while improving readability
  - [x] All tests passing after docstring updates

- [ ] **Schema Versioning & Lightweight Migrations** (Est. 2h)
  Introduce SQLite schema version stamps plus a minimal migration runner to unblock future columns without manual SQL.

- [ ] **Data Integrity Checker** (Est. 3h)
  Schedule periodic audits for soft deletes, UUID uniqueness, and cross-table consistency; emit alerts to Logs and Grafana.

- [ ] **Knowledge Graph & README Export** (Est. 1h)
  regenerate the code graph/README snapshots and commit the artifacts so downstream consumers have the latest state.

- [ ] **Maintainer Handoff Brief** (Est. 1h)
  Summarize outcomes, unresolved risks, and recommended sequencing for the next maintainer before project pause/transition.
