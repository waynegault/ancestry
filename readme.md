# Ancestry Genealogical Research Automation

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/waynegault/ancestry)

Comprehensive Python automation system for Ancestry.com genealogical research, featuring intelligent messaging, DNA match analysis, and family tree management.

## Overview

This project automates genealogical research workflows on Ancestry.com, including:
- **Action 6**: Automated page gathering and data collection
1. **Start LM Studio** (GUI)
   - Open LM Studio
   - Load your chosen instruct model (e.g., qwen3-4b-2507 or Llama/Mistral)
   - Click "Start Server"; ensure it shows Running at http://localhost:1234/v1

2. **Validate with tests**
   ```bash
   python test_local_llm.py
   ```
   - Test 1 (Direct connection) should PASS in <5s typical
   - Test 2 (Configuration) should PASS with AI_PROVIDER=local_llm
   - Test 3 (Genealogical prompt) should PASS and mention at least 3 of: census, birth, marriage, death, record, search, scotland

3. **Use in normal actions**
   - Keep LM Studio running
   - Run Action 8 or 9 normally; AI calls will route to the local model

### Testing Auto-Start

Run the demo script to test auto-start functionality:
```bash
python demo_lm_studio_autostart.py
```

### Troubleshooting

**LM Studio executable not found**:
- Update `LM_STUDIO_PATH` in `.env` to match your installation location

**API timeout after 60s**:
- Increase `LM_STUDIO_STARTUP_TIMEOUT` in `.env`
- Check firewall isn't blocking localhost:1234
- Verify nothing else is using port 1234: `netstat -ano | findstr :1234`

**Model not loaded**:
- LM Studio starts automatically, but you must manually load a model in the UI
- Open LM Studio → "My Models" → Load your configured model

**Disable auto-start**:
- Set `LM_STUDIO_AUTO_START=false` in `.env` to require manual startup

## Future Developer Ideas

Longer-term concepts are tracked in `development_plan.md`. Review that file for the prioritized backlog.

## Actions

### Action 6: DNA Match Gathering
Automates DNA match harvesting from Ancestry.com with resilient session management and configurable pagination.

- **Config-driven pagination**: `MATCHES_PER_PAGE` controls the API `itemsPerPage` value (default 30). Change the environment variable if Ancestry throughput rules shift.
- **Sequential rate limiting**: All API calls funnel through the shared `RateLimiter`; there is no parallel worker pool to guarantee zero 429 responses.
- **Session health monitoring**: Proactive checks keep the session refreshed before the 40-minute expiry window.
- **Checkpoint resume**: Automatic state saves allow interrupted runs to resume from the last completed page.
- **Ethnicity enrichment**: Optional fetches pull DNA ethnicity percentages when available.
- **Relationship ladder throttling**: Use `MAX_RELATIONSHIP_PROB_FETCHES` to cap expensive relationship probability lookups per page.

**API Endpoint:**
```
GET /discoveryui-matches/parents/list/api/matchList/{test_guid}?itemsPerPage={MATCHES_PER_PAGE}&currentPage={page}
```

**Configuration (.env):**
```bash
# Core controls
REQUESTS_PER_SECOND=0.3
MATCHES_PER_PAGE=30
MAX_RELATIONSHIP_PROB_FETCHES=0

# Optional health overrides
HEALTH_CHECK_INTERVAL_PAGES=5
SESSION_REFRESH_THRESHOLD_MIN=25
```

```bash
python action6_gather.py
```

### Action 7: Inbox Processing
Process and analyze inbox messages with AI classification.

```bash
python action7_inbox.py
```

### Action 8: Intelligent Messaging
Send AI-powered messages to DNA matches with context awareness.

```bash
python action8_messaging.py
```

### Action 9: Productive Conversation Management
Manage ongoing productive conversations with automated follow-ups.

```bash
python action9_process_productive.py
```

### Action 10: GEDCOM Analysis
Analyze GEDCOM files and score potential matches.

#### Current display and filtering policy
- Display: Show only the highest-scoring result. We search GEDCOM first; only if GEDCOM returns no matches do we call the API. Immediately after the top row is printed for the chosen source, the system displays that person's family members and the relationship path to the tree owner. The detailed results tables and the summary line are shown only when logging is set to DEBUG.
- Name containment (mandatory when provided):
  - If first_name is provided, candidate must contain it (case-insensitive contains)
  - If surname is provided, candidate must contain it (case-insensitive contains)
  - If both are provided, both must be contained
  - If neither name is provided, a broader OR filter is used on birth_year, birth_place, and alive-state
- Gender: Removed as a search and scoring criterion and removed from result displays; it is no longer collected as input and does not influence filtering or scoring.
- Alive-mode policy: When no death criteria are provided, candidates with death information receive a small penalty; missing death info is neutral.

#### API search sources and parsing notes

- Primary family endpoint: /family-tree/person/addedit/user/{owner_profile_id}/tree/{tree_id}/person/{person_id}/editrelationships
  - Response shape: { cssBundleUrl, jsBundleUrl, data }, where data is a JSON STRING that must be json.loads(...) into { userId, treeId, personId, person, urls, res }
  - The family arrays live under parsed_data.person: fathers[], mothers[], spouses[], children[] (children may be nested arrays per spouse)
  - res contains UI/localization strings, not family data
- Relationship ladder endpoint: /family-tree/person/card/user/{user_id}/tree/{tree_id}/person/{person_id}/kinship/relationladderwithlabels
- Design decisions:
  - Siblings are not displayed in the API path (per requirements); parents, spouses, and children are displayed
  - Session authentication occurs once via session_utils.get_authenticated_session; no redundant re-login or cookie syncs

```bash
python action10.py
```

## Testing

```bash
# Run all tests (57 modules, 457 tests)
python run_all_tests.py

# Run with parallel execution
python run_all_tests.py --fast

# Run with log analysis
python run_all_tests.py --analyze-logs

# Run specific module tests
python -m action6_gather
```

## Pylance Configuration

The project uses **basic** type checking mode with strict exclusions to ensure stable, accurate error reporting.

### Configuration Files

**pyrightconfig.json**: Primary configuration
- Type checking mode: `basic` (balances error detection with usability)
- **Includes**: Only `*.py` and `**/*.py` files (explicit Python files only)
- **Excludes**: `.git/**`, `__pycache__/**`, `.venv/**`, `Cache/**`, `Logs/**`, `Data/**`
- **Ignore**: `.git` and `**/.git` (prevents analyzing git internals)
- Silences false positives (unused variables, unreachable code, type inference limitations)
- Keeps critical errors visible (undefined variables, import errors, etc.)

**.vscode/settings.json**: Workspace settings
- **CRITICAL**: `.git` is excluded (`"**/.git": true`) to prevent Pylance from analyzing git internals
- File watching and search exclusions for performance
- Markdown linting configuration

**Note**: When `pyrightconfig.json` exists, `.vscode/settings.json` Python analysis settings are ignored.

### Reloading Configuration

**You MUST reload VS Code window for changes to take effect:**
1. Press `Ctrl+Shift+P`
2. Type "Developer: Reload Window"
3. Press Enter

**Expected result**: Stable error count showing only real issues (typically 10-20 errors)

### Troubleshooting Pylance

If you see thousands of errors or errors from `.git` files:
1. Check that `.vscode/settings.json` has `"**/.git": true` (not `false`)
2. Reload VS Code window (Ctrl+Shift+P → Developer: Reload Window)
3. If errors persist, restart VS Code completely
4. Clear Pylance cache: Delete `.vscode/.ropeproject` if it exists and reload

## Development Guidelines

### Code Quality
- Follow DRY (Don't Repeat Yourself) principles
- Use type hints for function signatures
- Add docstrings to all public functions
- Keep functions under 50 lines when possible

### Testing
- Write tests for all new functionality
- Tests should fail when functionality fails (no fake passes)
- Use `.env` test data for automated tests
- Maintain 100% test pass rate

### Rate Limiting
- Never bypass the rate limiter
- All API calls must go through `session_manager.rate_limiter`
- Monitor logs for 429 errors
- Validate changes with 50+ page runs

### Caching Strategy

The project uses **UnifiedCacheManager** (singleton, thread-safe) to reduce API load by caching responses with configurable TTL per endpoint.

**Current Achievement**: ~40-50% cache hit rate on DNA match operations, saving **15-25K API calls per 800-page run**.

**Cache-Enabled Endpoints** (in action6_gather.py):
- `combined_details` - Combined person/family tree data (TTL: 24 hours)
- `relationship_prob` - Relationship probability scores (TTL: 7 days)
- `ethnicity_regions` - DNA ethnicity percentages (TTL: 30 days)
- `badge_details` - Profile badge data (TTL: 7 days)
- `ladder_details` - Relationship ladder info (TTL: 7 days)
- `tree_search` - Family tree search results (TTL: 1 hour)

**Configuration (.env)**:
```bash
# Cache settings
CACHE_ENABLED=true
CACHE_MAX_SIZE=10000          # LRU eviction at 10K entries
CACHE_DEFAULT_TTL_SECONDS=86400  # 24 hours default
```

**Monitoring & Debugging**:

View cache statistics:
```python
from core.unified_cache_manager import get_unified_cache_manager
cache = get_unified_cache_manager()
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Entries: {stats['total_entries']}")
print(f"Memory: {stats['estimated_memory_mb']:.1f}MB")
```

Check per-endpoint hit rates:
```python
for service_name, endpoints in cache.get_service_stats().items():
    for endpoint_name, ep_stats in endpoints.items():
        hit_rate = ep_stats['hit_rate']
        print(f"{service_name}/{endpoint_name}: {hit_rate:.1%} hit rate")
```

Clear cache (if needed for testing):
```python
cache.clear()  # Full clear
cache.invalidate_by_endpoint("combined_details")  # Endpoint-specific
cache.invalidate_by_key("person_12345")  # Specific entry
```

Monitor in logs:
```bash
# Watch cache performance in real-time
Get-Content Logs/app.log -Wait | Select-String "cache hit|cache miss"

# Review integration test results
python -m test_action6_cache_integration
```

### Prometheus Monitoring

- Enable `PROMETHEUS_METRICS_ENABLED=true` in your `.env` file to expose a `/metrics` endpoint on the configured host and port (default `127.0.0.1:9000`).
- When you need a quick exporter without running the main menu, execute `python observability/metrics_exporter.py --serve`.
- Import the starter Grafana dashboard at `docs/grafana/ancestry_overview.json` and point panels to your Prometheus data source.
- Detailed setup steps live in `docs/monitoring.md`.

### Git Workflow
- Commit at each phase of implementation
- Write descriptive commit messages
- Test before committing
- Revert if tests fail

## Troubleshooting

### Common Issues

**429 Rate Limit Errors:**
- Check `THREAD_POOL_WORKERS=1` in `.env`
- Check `REQUESTS_PER_SECOND=0.4` in `.env`
- Monitor: `Select-String -Path Logs\app.log -Pattern "429 error"`

**Import Errors:**
- Verify Python 3.12+ is installed
- Run `pip install -r requirements.txt`
- Check virtual environment activation

**Browser Issues:**
- Update Chrome to latest version
- Run `python -c "from core.browser_manager import BrowserManager; BrowserManager()"`
- Check ChromeDriver compatibility

**Database Errors:**
- Backup database: `python main.py` → Option 3
- Check file permissions on `Data/ancestry.db`
- Verify SQLite installation

### Logs

All logs are written to the file defined by `LOG_FILE` in your `.env` (default: `Logs/app.log`):

```bash
# View recent errors
TAIL_TARGET="${LOG_FILE:-Logs/app.log}"
tail -100 "$TAIL_TARGET" | grep ERROR

# Monitor real-time
tail -f "$TAIL_TARGET"

# Check rate limiter initialization
grep "Thread-safe RateLimiter" "$TAIL_TARGET" | tail -1
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run `python run_all_tests.py`
5. Submit pull request

## License

This project is for personal genealogical research use.

## Support

For issues or questions:
- GitHub Issues: https://github.com/waynegault/ancestry/issues
- Email: 39455578+waynegault@users.noreply.github.com

---

## Appendices

### Appendix A: Chronology of Changes

2025-11-03
- Action 6 Browser Death Fix (commit 4c3277a): Reverted action6_gather.py and core/session_manager.py to stable commit 793d948 (last known-good version that processed 15,000+ matches flawlessly) and cleanly re-added ethnicity enrichment
  - Root cause: Massive refactoring after 793d948 introduced complex browser recovery mechanisms (_atomic_browser_replacement, attempt_browser_recovery, death cascade detection) that were failing with "atomic replacement failed" errors causing browser session death
  - Solution: Surgical revert to proven stable base from 793d948, removed all complex recovery mechanisms, adapted for global session_manager pattern from main.py, cleanly re-integrated ethnicity enrichment without recovery complexity
  - Changes made:
    - Reverted action6_gather.py from 2,088 lines (current) to 6,824 lines (793d948 base)
    - Reverted core/session_manager.py from 3,807 lines (current) to 2,023 lines (793d948 base)
    - Removed _config_schema_arg parameter from coord() function (now uses global config_schema)
    - Added ethnicity imports: fetch_ethnicity_comparison, extract_match_ethnicity_percentages, load_ethnicity_metadata
    - Added helper functions: _get_ethnicity_config(),_build_ethnicity_payload(), _needs_ethnicity_refresh()
    - Integrated ethnicity enrichment into _prepare_dna_match_operation_data() function
    - Ethnicity data fetched and added to DNA match records when creating new records or refreshing existing ones
  - Preserved database.py ethnicity persistence fix from commit 8649380 (CREATE-path raw SQL UPDATE)
  - Backup files created: action6_gather_current_backup.py, core/session_manager_current_backup.py for reference
  - Simple, proven approach from 793d948 without elaborate recovery mechanisms that were causing failures

2025-11-02
- Login Regression Recovery: Reverted to commit 2c8956d (Oct 31, 18:31) after discovering login had been broken since at least Oct 31 due to Ancestry's cookie consent banner blocking form interactions
- Key Lessons Learned:
  - Always test at known-good commits before attempting fixes - spent significant time trying to fix login at commits 7c57bd9 and 7c1cd81, only to discover the consent banner problem existed at both
  - Git history is invaluable for finding working versions - reverting to 2c8956d immediately restored working login with proper 2FA support
  - Consent banner behavior can be inconsistent/A-B tested by Ancestry - banner dynamically re-injects itself after form interactions (Next button click)
  - Incremental improvements on working baseline are safer than major refactors on broken code
- Startup Status Checks: Added non-blocking _check_startup_status() function to display database, Ancestry session cookie, and MS Graph token availability at startup without triggering authentication flows
  - Added _check_token_cache() helper function to ms_graph_utils.py
  - Provides user visibility into system readiness without delays
  - All checks are non-blocking (don't trigger authentication flows)
- Action 2 Browserless: Made Action 2 (Reset Database) browserless by loading ethnicity columns from saved ethnicity_regions.json metadata file
  - Execution time reduced from 60+ seconds to ~2 seconds
  - Fixed transaction handling in ethnicity column creation (use engine.begin() instead of manual commit() calls)
  - Removed manual commit() calls that caused 'closed transaction' errors
  - Added initialize_ethnicity_columns_from_metadata() function to dna_ethnicity_utils.py
  - If no metadata file exists, columns will be added during first Action 6 run
  - Addresses requirement that ethnicity columns are NOT optional while avoiding cookie consent/login issues
- All 71 modules pass tests with 100% success rate after improvements

- Action 6 Ethnicity Persistence Fix (commit 8649380): Persist ethnicity columns on CREATE via parameterized raw SQL UPDATE after initial INSERT; resolves NULL ethnicity on first-run creates when ORM model lacks dynamic columns. Verified with a clean reset + Action 6: New 20, Updated 0; 20/20 rows have non-NULL ethnicity values.
  - Root cause: ORM ignored dynamically added columns during INSERT; UPDATE-path raw SQL already worked
  - Solution: After inserting core fields and flush, run a single UPDATE by people_id to set all ethnicity columns; keep UPDATE-path raw SQL
  - Regression guards: Added endpoint-constant tests to api_utils.py and core/api_manager.py; added literal-presence guards to dna_ethnicity_utils.py and core/session_manager.py

2025-10-28
- Unified presenter: fixed header spacing ("=== Name (years) ==="), ensured empty sections print "None recorded", and normalized relationship header text
- GEDCOM/API: birth/death years now shown when available; GEDCOM path falls back to parsing years from display name if missing
- Owner name: now resolved from REFERENCE_PERSON_NAME, then USER_NAME, then stable fallback; eliminates "Unknown" in relationship header
- API utils: lowered family-relationship fetch log to DEBUG to avoid INFO-level noise
- API search: fixed urlencode type issue by removing custom quote_via; cleaned unused import
- Tests: Fast suite passed locally after changes; no new failing tests introduced
- GEDCOM header years: now also sourced from GedcomData.processed_data_cache when candidate.raw_data lacks years; ensures headers like "=== Name (YYYY-YYYY) ===" render when data exists
- API header years: Added robust fallback in action11 to derive birth/death years from parsed_suggestion and normalized date strings; fixes missing years like "=== Peter Fraser ===" without (YYYY-YYYY)

- Action 10: API fallback now self-initializes session by auto-attempting browser-based cookie sync when browserless cookies are missing/invalid; no need to run Action 5 first.
2025-10-29
- Fully removed action11.py. All API search, display, and post-selection logic is now provided by api_search_core.py and existing shared modules (api_search_utils, relationship_utils, genealogy_presenter, universal_scoring).
- Updated main.py and action9_process_productive.py to import from api_search_core.
- Session: ensure_api_ready_with_browser_fallback refactored to reduce returns and collapse nested conditionals; now a single return path with a success flag (fixes PLR0911 and SIM102).
- Linter cleanup:
  - Removed useless import aliases (PLC0414) by replacing the shim with a concrete api_search_core implementation.
  - Collapsed nested ifs in core/session_manager.py (SIM102).
  - Removed unused function argument warning in API presenter by prefixing with underscore (ARG001) and using parameter naming convention.
- Documentation: Updated Overview, Developer Instructions, Actions, Testing, and Appendix B to reflect api_search_core ownership of API endpoints and the retirement of action11.py.

- Hardened Action 10 API fallback: now requires real API login verification (profile ID retrieval) rather than accepting .env IDs; if browserless fails, it auto-launches the browser, attempts re-login, syncs cookies, re-verifies, and only then proceeds. This restores prior Action 11 behavior without requiring Action 5 first.

- Cleanup: Removed final references to Action 11 across code; consolidated API search into api_search_core and updated main.py import expectations
- Linter: Resolved all E702 (multiple statements on one line) in api_search_core; repository E702 count now 0
- Pylance: Fixed import path in api_search_core (from config import config_schema); removed broken **all** block and restored proper exports
- Complexity: Reduced action10.analyze_top_match complexity from 15 to under 11 by extracting helpers (_derive_display_fields,_build_family_data_dict,_compute_unified_path_if_possible)
- Tests: Added in-module tests for api_search_core and genealogy_presenter following the standard pattern; run_all_tests.py --fast now 100% pass (610 tests)

- API Relationship Path: Replaced getladder HTML parsing with relationladderwithlabels JSON API endpoint for clean, reliable relationship path data
- API Family Data: Replaced Edit Relationships API with New Family View API which includes siblings and complete family structure
- Complexity Reduction: Refactored get_api_family_details (29→<11),_parse_person_from_newfamilyview (15→<11),_extract_birth_event (<11), _extract_death_event (<11) by extracting helper functions
- Linter: Fixed global-statement warnings by using unittest.mock.patch instead of modifying globals
- Logging: Fixed inconsistent logging in api_search_core.py and search_criteria_utils.py by using centralized logger from logging_config
- Test Infrastructure: Added missing **main** blocks to api_search_core.py, search_criteria_utils.py, and updated core/session_cache.py and core/system_cache.py to use standard test runner pattern
- Import Order: Fixed E402 errors in action10.py by moving standard library imports before setup_module call
- Test Quality: Removed 5 redundant smoke tests from gedcom_utils.py (test_individual_detection, test_name_extraction, test_date_parsing, test_event_extraction, test_life_dates_formatting) as they only checked function existence/types; function availability already verified by test_function_availability()
- Final Results: All 74 modules pass with 100.0/100 quality scores, 643 tests passing at 100% success rate

2025-10-29
- API Fallback Fix: Updated api_search_core._resolve_base_and_tree to fall back to config.api.tree_id when session_manager.my_tree_id is None (browserless mode); fixes 404 errors in Action 10 API search
- Main.py Tests: Updated _test_edge_case_handling and _test_import_error_handling to remove assertions for api_search_core module (imported lazily inside run_gedcom_then_api_fallback)
- Type Hints: Added type annotations to fake_list_api test stub in api_search_core (resolved quality issue)
- Complexity Reductions:
  - api_search_core._handle_supplementary_info_phase: 14→<11 by extracting _extract_year_from_candidate and _get_relationship_paths helpers
  - action10._derive_display_fields: 13→<11 by extracting_extract_years_from_name_if_missing and_supplement_years_from_gedcom helpers
  - genealogy_presenter.display_family_members: 11→<11 by extracting_deduplicate_members and_filter_valid_members helpers
- Quality: All 72 modules now pass with 100.0/100 quality scores (610 tests, 100% success rate)
- Action 10 API Fallback: Changed from browserless-first to browser-required approach; browserless mode with cookie files consistently fails with 404 errors due to missing authentication state that can only be obtained through active browser login (matches original Action 11 behavior)
- TreesUI List API URL Fix: Corrected API_PATH_TREESUI_LIST from "trees/{tree_id}/persons" to "api/treesui-list/trees/{tree_id}/persons"; updated_build_treesui_url to use correct parameters: name (combined first+last), limit=100, fields=EVENTS,GENDERS,NAMES, isGetFullPersonObject=true (matches expected Ancestry API format)
- Search Criteria Mapping: Fixed _build_treesui_url to accept both first_name/surname (from get_unified_search_criteria) and first_name_raw/surname_raw (legacy) for compatibility
- TreesUI Response Parsing: Added _parse_treesui_list_response() function to api_utils.py to convert raw API response (Names, Events, gid fields) into standardized format (FullName, GivenName, Surname, PersonId, etc.) - this parsing function was lost when Action 11 was removed; refactored into helper functions (_extract_gid_parts, _extract_name_parts,_extract_birth_event, _extract_death_event) to reduce complexity from 31 to <11; updated to handle both old format (Events[].t="B"/"D", d={y,m,d}, p={n}) and new format (Events[].t="Birth"/"Death", d="formatted string", nd="YYYY-MM-DD", p="place string") returned by isGetFullPersonObject=true parameter; further refactored_extract_birth_event and _extract_death_event into smaller helpers (_extract_year_from_normalized_date, _extract_date_string_from_dict) to reduce complexity from 12 to <11
- Action 10 Tests: Added test_api_search_peter_fraser() test to validate API search functionality with real person data (Peter Fraser b. 1893 in Fyvie) - verifies URL building, API call, response parsing, and scoring all work correctly
- API Search Debug Logging: Added debug logging to _process_and_score_suggestions() to show scoring details for each result and top 3 matches - helps diagnose scoring/ranking issues
- Code Quality: Fixed 2 PLW0603 global-statement linter warnings in api_search_core.py by replacing global statement with unittest.mock.patch for test mocking
- New Family View API: Replaced get_api_family_details() to use newfamilyview API (api/treeviewer/tree/newfamilyview/{tree_id}) instead of editrelationships API; new API returns complete family data including siblings in cleaner JSON format with Persons array and Family relationships; added call_newfamilyview_api() to api_utils.py and_parse_person_from_newfamilyview() to api_search_utils.py; siblings now properly extracted by finding parents and getting their children (excluding target person); refactored get_api_family_details() into helper functions (_find_target_person_in_list,_create_persons_lookup, _extract_direct_family,_extract_siblings) to reduce complexity from 29 to <11; refactored_parse_person_from_newfamilyview() into helper functions (_extract_person_id_from_gid, _extract_full_name_from_names,_extract_year_from_event_type) to reduce complexity from 15 to <11
- Relation Ladder With Labels API: Replaced getladder HTML parsing with relationladderwithlabels API (family-tree/person/card/user/{user_id}/tree/{tree_id}/person/{person_id}/kinship/relationladderwithlabels) for relationship paths; new API returns clean JSON with kinshipPersons array containing name, lifeSpan, and relationship for each person in path; added call_relation_ladder_with_labels_api() to api_utils.py and_format_kinship_persons_path() to api_search_core.py; relationship paths now display proper names and dates instead of "Unknown"
- genealogy_presenter.py: Added **main** block to run internal tests when module is executed directly (python genealogy_presenter.py)
- Quality: All 72 modules now pass with 100.0/100 quality scores (611 tests, 100% success rate) - no complexity issues remaining

2025-10-28
- Main menu: Removed Action 11; Action 10 now runs a side-by-side comparison (GEDCOM vs API)
- Scoring: Added alive-mode penalty when no death criteria are provided and candidate has death info; no reward for missing death fields
- Output: Removed the “Scoring Policy” line from results; behavior remains unchanged
- Display: Always show top 5 results for each (GEDCOM and API) while tuning; summary line updated to “Summary: GEDCOM — showing top N of M total | API — showing top K of L total”
- Scoring: Increased birth/death year match weights to 25 (exact year); approximate year weights unchanged
- Scoring: Increased bonuses to 50 for: both names matched, both birth info matched (year+place), both death info matched (year/date + place)
- Display: Show only the top result. GEDCOM is preferred; API is called only when GEDCOM has no matches. Zero-results message simplified to "No matches found."; API no-results log demoted to DEBUG
- Display: For the chosen source's top result, show family members and a relationship path to the tree owner immediately after the (debug-only) result table

- Filtering: Name containment is now mandatory when provided (first and/or surname)
- Policy: Gender removed as a search and scoring criterion and removed from result displays
- UI: Removed Gender column from GEDCOM and API results; removed Gender input prompt; updated prompts to "Death Year (YYYY):" and "Death Place Contains:" (removed [Optional])
- Filtering: Enforced mandatory name containment when provided (case-insensitive). If both first and surname are provided, both must match. Fixed case normalization to prevent false non-matches

- Linter: Fixed SIM103 and SIM108 across relevant modules
- Docs: Updated Overview and Action 10 policy section to reflect these changes
- Tests: run_all_tests.py passed (72/72 modules)

- Filtering: Enforced mandatory place matching only when non-empty search values are provided; fixed a bug where empty birth/death place keys inadvertently excluded GEDCOM candidates.
- Action 10: Reduced complexity of _evaluate_filter_criteria using early returns and any/all helpers; module now at 100/100 quality (no complexity warnings).
- Family Display: De-duplicated family member lists in display_family_members() by name + year tuple; resolves duplicate Children lines from API family data.
- Behavior: API search remains gated to run only when GEDCOM returns zero matches (now correctly triggered when place criteria are unmet in GEDCOM).

- Logging: Converted DEBUG-only result tables and summary in main.py to logger.debug() (no prints). INFO level now shows only criteria, top match header, family, and relationship path
- Linter: Resolved 4 SIM102 (collapsible-if) occurrences in api_search_utils.py and gedcom_search_utils.py

- Search Criteria: Summary now prints only provided fields (omits empty Birth/Death fields) for a cleaner UI
- Spacing: Added a blank line between Children and Relationship sections for readability
- Relationship: Header standardized to "Relationship to {owner_name}:" (no emoji)
- API Layout: Top API match header now prints before family details, matching GEDCOM format
- Family headers: Removed emojis; sections are now "Parents", "Siblings", "Spouses", "Children"

- Consolidation: Action 11 wrappers removed in favor of shared helpers; Action 11 now calls api_search_utils.get_api_family_details and search_criteria_utils.display_family_members directly

- Test Runner: Parallel output synchronized (no out-of-order numbering), duplicate module headings removed, and discovered modules de-duplicated; improved test-count extraction significantly reduces prior "Unknown" counts (one remaining outlier to fix).
- Search Criteria UX: Added a blank line between the action header and the first input prompt; extracted summary/log helpers to reduce complexity and keep INFO output clean.
- Relationship Header: Verified fully dynamic header uses tree owner’s name everywhere (no hard-coded fallback); ensured consistent header in both Action 10 and 11 paths.

- Consolidation: Phase 3–4 complete — introduced a unified post-selection presenter (present_post_selection) in search_criteria_utils; Action 10 (GEDCOM) and Action 11 (API) now both call this to render header → family → relationship. Eliminated duplicated display code in action11 and refactored action10’s analyze_top_match to use the presenter. Outputs are now identical across sources and spacing/order is consistent; no hard-coded owner names.
- Wrapper rename: main wrapper renamed to run_gedcom_then_api_fallback (was run_side_by_side_search_wrapper); logs now reflect the new, accurate purpose
- API fallback display: removed legacy header print before presenter, fixing duplicate/Unknown headers; presenter now owns header → family → relationship exclusively
- Family sections: when no data, sections show "   - None recorded" (instead of a bare dash)
- Linter: removed 4 unused variables (F841) across action11 and relationship_utils; repo diagnostics now clean
- Complexity: simplified action11._handle_supplementary_info_phase by extracting logic and removing nested branches; quality back to 100/100

- API browserless: Added SessionManager.ensure_api_ready_browserless and switched Action 10 API fallback to use it; prevents unintended browser startup and fixes the minimize-window crash
- APIManager: Added load_cookies_from_file() to hydrate requests.Session cookies from ancestry_cookies.json for browserless operation
- Main: Switched imports to api_search_core shim with fallback to action11 for IDE/path resilience during migration
- Pylance: Removed unreachable checks and driver-coupling in api_search_utils; prefer identifier readiness; resolved unresolved-import warning for api_search_core via guarded import
- Refactor: Introduced api_search_core shim to re-export Action 11 helpers; prepares full retirement of action11.py name
- APIManager: Changed verify_api_login_status() to require profile ID retrieval (auth-only); UUID from .env is not treated as proof of login, preventing false positives and later 401s
- Presenter: Created genealogy_presenter.py with present_post_selection and display_family_members used identically by GEDCOM and API paths
- Main: Action 10 API fallback now uses browserless readiness; unchanged UX, identical output format across GEDCOM/API
- Pylance: Fixed unused-arg warning in ensure_api_ready_browserless by consuming action_name for debug; removed unreachable session-manager check in API search; general dead-import cleanup
- Migration (Phase 2 prep): api_search_core is now the public import point; action11 retains implementation for now; next step reverses the dependency with lazy wrappers

2025-10-24
- Eliminated module-level SessionManager creation in action11.py; switched to global session usage only
- Consolidated Action 11 authentication to a single path via session_utils.get_authenticated_session()
- Removed redundant login/cookie helper functions from action11.py
- Updated action8_messaging.py to load templates and get session via session_utils.get_global_session()
- Updated scripts/test_editrelationships_shape.py to require/use the global session
- Deferred Action 8 template loading to runtime (lazy initialization); eliminated import-time CRITICALs
- Added de-duplication for gender inference DEBUG logs in relationship_utils
- Reduced INFO banner noise in session_utils.get_authenticated_session; cache-hit now DEBUG
- Enhanced API call logging with durations in api_utils.call_enhanced_api
- Reworded cookie skip message in core/session_validator for clarity
- Temporarily disabled startup clear-screen in main.py to expose early errors
- Tests: run_all_tests.py passed (72/72 modules) after refactor

2025-10-26
- INFO logging tidy-up: removed "UUID: Not yet set" banner; if MY_UUID exists in .env we log it as "UUID (from .env) — will verify"; otherwise keep at DEBUG only
- Global session banner shows once on first auth; subsequent get_authenticated_session() calls reuse cache without re-printing banners
- Cookie sync logging clarified: "initial sync" first time, else "age Xs; threshold Ys"; removed misleading "cache expired, last sync ..." phrasing
- API Request Cookies logging compressed to count only (no full key dump)
- Edit Relationships API response logging summarized (type + top keys) instead of full structure dump
- Cookie-check message clarified: "Skipping essential cookies check (expected): …"
- Action 11: added success/error summary with duration at completion
- Action 8: lazy-load templates at runtime; no import-time CRITICALs
- Relationship logs: de-duplicated repeated gender inference debug lines
- Tests: run_all_tests.py passed (72/72 modules) in fast mode
- Re-enabled startup clear-screen in main.py (now that early error review is complete)
- Restored VISION_INTELLIGENT_DNA_MESSAGING.md to repository root from commit 2aee7d6 (temporary for implementation reference; will consolidate back into readme.md at completion per single-doc rule)

2025-10-27
- Complexity reduction: session_utils.get_authenticated_session split into `_assert_global_session_exists` and `_pre_auth_logging`; type-safety improvements
- Complexity reduction: api_search_utils.get_api_family_details decomposed into helpers for structure logging, section extraction, and per-relationship parsing
- Complexity reduction: action7_inbox extracted `_assess_and_upsert`; reduced analytics method branching
- Complexity reduction: action9_process_productive extracted `_formatting_fallback` to simplify exception path
- Documentation: Consolidated vision into readme.md; added Local LLM integration steps; cleaned markdown lint issues
- Policy: Removed messages.json earlier; VISION_INTELLIGENT_DNA_MESSAGING.md will be removed after verification (single-file docs policy)
- Complexity reduction: action9_process_productive._build_enrichment_lines refactored into helpers; module now 100/100 quality
- Documentation: Removed VISION_INTELLIGENT_DNA_MESSAGING.md (content consolidated here per single-doc policy)
- Phase 7 Local LLM: Executed real tests via test_local_llm.py; configuration test passed; direct connection and genealogical prompt failed due to LM Studio not loading a model (404 model_not_found). Environment steps documented in Appendix B

2025-10-23
- Switched Action 11 family extraction to the Edit Relationships endpoint and parsed nested data['person'] correctly
- Suppressed verbose raw path logging in Action 11; kept concise debug metrics

### Appendix B: Technical Specifications

- Monitoring & Analytics (Phase 6)
  - Each action run writes a JSON line to Logs/analytics.jsonl with fields: ts, action_name, choice, success, duration_sec, mem_used_mb, extras
  - Merged Actions 10/11 set extras.merged_10_11_branch to 'gedcom' or 'api_fallback' and include candidate counts
  - Weekly summary generator: from analytics import print_weekly_summary; print_weekly_summary(7)
  - Non-fatal by design: analytics never blocks action execution

- Action 0 (Delete all rows except test profile)
  - Set the .env variable TEST_PROFILE_ID to the Profile ID you want to keep (e.g., your mother’s ucdmid)
  - Safety: If TEST_PROFILE_ID is missing or equals MOCK_PROFILE_ID, the action aborts to prevent unintended deletion
  - If the configured keeper is not found in the database, the action will abort with instructions to correct TEST_PROFILE_ID
  - Requires explicit yes/no confirmation in the menu before executing

- Session Architecture
  - Exactly one SessionManager instance created by main.py
  - Registered globally via session_utils.set_global_session()
  - Consumers must call session_utils.get_authenticated_session(action_name=...) before API usage

- API Endpoints (do not change these - they work!)

  **User Identity Endpoints** (used by session_manager)
  - Profile ID: `app-api/cdp-p13n/api/v1/users/me?attributes=ucdmid`
    - Response: `{"data": {"ucdmid": "07bdd45e-0006-0000-0000-000000000000"}, "message": "OK", "status": 200}`
    - Returns: User's profile ID (ucdmid field)

  - UUID (DNA Test ID): `api/navheaderdata/v1/header/data/dna`
    - Response: `{"results": {"menuitems": [...]}, "testId": "FB609BA5-5A0D-46EE-BF18-C300D8DE5AB7", "testComplete": true, ...}`
    - Returns: User's DNA test UUID (testId field at ROOT level, not inside results dict)

  - Tree List: `api/treesui-list/trees?rights=own`
    - Response: `{"trees": [{"id": "175946702", "name": "Gault Family", "ownerUserId": "...", ...}], "count": 2}`
    - Returns: List of user's trees; match TREE_NAME from .env to get tree ID

  - Tree Owner Info: `api/uhome/secure/rest/user/tree-info?tree_id={tree_id}`
    - Response: `{"id": 175946702, "owner": {"userId": "...", "displayName": "Wayne Gault"}, "mePersonId": 102281560836, ...}`
    - Returns: Tree owner display name and mePersonId

  **Genealogical Data Endpoints** (used by api_search_core)
  - Edit Relationships: `/family-tree/person/addedit/user/{owner_profile_id}/tree/{tree_id}/person/{person_id}/editrelationships`
    - Response: `{ cssBundleUrl, jsBundleUrl, data }` where data is a JSON string; parse with json.loads

  - New Family View (preferred): `api/treeviewer/tree/newfamilyview/{tree_id}`
    - Returns: Persons array with people and Family relationships structure; includes siblings via parents' children
    - Used by: api_utils.call_newfamilyview_api and api_search_core._parse_person_from_newfamilyview

  - TreesUI List: `api/treesui-list/trees/{tree_id}/persons`
    - Purpose: Person suggestions for a tree; used for API search candidates
    - Typical params: name, limit=100, fields=EVENTS,GENDERS,NAMES, isGetFullPersonObject=true

  - Person Facts (User): `family-tree/person/facts/user/{owner_profile_id}/tree/{tree_id}/person/{person_id}`
  - GetLadder (legacy): `family-tree/person/tree/{tree_id}/person/{person_id}/getladder`

  **Ethnicity Endpoints** (used by Action 6)
  - Tree Owner Ethnicity: `dna/origins/secure/tests/{guid}/v2/ethnicity`
    - Purpose: Retrieves owner’s ethnicity regions and percentages; seeds dynamic columns
  - Ethnicity Comparison: `discoveryui-matchesservice/api/compare/{owner_guid}/with/{match_guid}/ethnicity`
    - Purpose: Retrieves match percentages aligned to owner’s regions (use rightSum values)
  - Ethnicity Region Names: `dna/origins/public/ethnicity/2025/names?locale=en-GB`
    - Purpose: Map region keys to friendly names for display and column naming

  **Messaging Endpoints** (used by Action 8)
  - Send New Message: `app-api/express/v2/conversations/message`
  - Send Existing Conversation: `app-api/express/v2/conversations/{conv_id}`
  - Profile Details: `/app-api/express/v1/profiles/details`

  Note: Values above are covered by regression guard tests in-module (api_utils, core/api_manager, dna_ethnicity_utils, core/session_manager). If any literal changes, tests fail to prevent accidental drift.

  - Family arrays: `parsed['person']` → fathers[], mothers[], spouses[], children[]

  - Relationship Ladder: `/family-tree/person/card/user/{user_id}/tree/{tree_id}/person/{person_id}/kinship/relationladderwithlabels`

- Display Rules
  - Parents, spouses, children shown; siblings intentionally omitted in API path

- AI Providers and Local LLM
  - ai_provider: one of ["moonshot", "deepseek", "gemini", "local_llm"]
  - Active providers: Moonshot (Kimi), DeepSeek, Google Gemini, Local LLM (LM Studio)
  - LOCAL_LLM_* when ai_provider=local_llm: LOCAL_LLM_API_KEY, LOCAL_LLM_MODEL, LOCAL_LLM_BASE_URL
  - Default base URL: http://localhost:1234/v1 (LM Studio)

- LM Studio quick-start checklist (real use)
  1) Install LM Studio and open it
  2) Load an instruct model (e.g., qwen3-4b-2507)
  3) Start the local server (Developer tab) and ensure it shows Running at http://localhost:1234/v1
  4) In .env set:
     - AI_PROVIDER=local_llm
     - LOCAL_LLM_BASE_URL=http://localhost:1234/v1
     - LOCAL_LLM_API_KEY=lm-studio
     - LOCAL_LLM_MODEL=qwen3-4b-2507
  5) Optional: enable JIT loading in LM Studio so the first inference auto-loads the model
  6) Run: python test_local_llm.py

- Programmatically triggering model load (Python)
  - LM Studio follows the OpenAI-compatible API; the model is selected by the `model` field in your request.
  - If JIT loading is enabled, the first request with that model name will load it automatically.

### Appendix C: Test Review Summary (condensed)

- Date: 2025-10-23; Reviewer: Augment Agent
- Coverage and quality highlights:
  - 82 modules analyzed; 80 with tests (97.6% coverage)
  - 1,048 public functions; 1,033 test functions
  - 100% test pass rate; average quality 100/100
  - Tests use live sessions (no smoke tests) and are co-located with code
- Notable improvements:
  - Reduced complexity in database.py and conversation_analytics.py below thresholds
  - Reduced returns in core/session_manager.py to ≤6
  - Fixed deprecated imports and unused args; remaining globals in session_utils.py are intentional for caching

### Appendix D: Test Quality Analysis (condensed)

- Strengths: excellent coverage, genuine assertions, error-path tests, DRY utilities
- AI components covered: ai_interface, ai_prompt_utils, universal_scoring
- Utilities covered: logging_config, error_handling, test_framework
- Modules without tests: `config/__init__.py` and `config/__main__.py` (acceptable)
- Code quality metrics: now 100/100 across modules; 0 complexity warnings

### Appendix E: Test Coverage Report (how to regenerate)

- To regenerate full coverage tables in your environment:
```bash
python run_all_tests.py --emit-coverage
```
- Summary from last run:
  - Total Modules: 82; With Tests: 80; Without: 2
  - Total Public Functions: 1,048; Total Test Functions: 1,033
- The detailed per-module table previously in test_coverage_report.md has been consolidated into this readme per single-file policy. Re-run the command above to produce a fresh, complete table locally.

  - Example minimal call using requests:

```python
import os, requests
base = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
api_key = os.getenv("LOCAL_LLM_API_KEY", "lm-studio")
model = os.getenv("LOCAL_LLM_MODEL", "qwen3-4b-2507")

r = requests.post(
    f"{base}/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello and report your model name."},
        ],
        "temperature": 0.2,
        "max_tokens": 64,
    },
    timeout=60,
)
print(r.status_code)
print(r.json())
```

  If the model isn’t loaded and JIT is disabled, start it in the LM Studio UI or via the `lms` CLI.

- Conversation State Model (Phase 3)
  - Fields: ai_summary (text), engagement_score (int 0-100), last_topic (text), pending_questions (text/json), conversation_phase (enum)
  - Updated via Action 7 analytics pipeline and assess_engagement()

- Enrichment Policy (Phase 5)
  - Flag: enable_task_enrichment (bool)
  - When enabled, Action 9 appends enrichment lines (to-do links, research prompts, record sharing hints)

- Logging
  - Single header/footer, info-level friendly
  - No raw HTML dumps; log length/count metrics instead

- Testing
  - run_all_tests.py is the canonical runner; tests should fail on genuine failures
  - API-dependent tests assume live authentication through the global session
