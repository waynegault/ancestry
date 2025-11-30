# Project Todo List

## Intelligent Conversation Management (Phases 1-8)

- [x] **Phase 1: Capability Audit**
  - Review existing code (Action 10, AI Interface, Relationship Utils) to understand current capabilities and gaps.
- [x] **Phase 2: Gap Analysis**
  - Identify missing features for inbound traffic handling: Reply Engine, Conversation State, Harvester, Safety Layer.
- [x] **Phase 3: Specification**
  - Define technical specs for Inbound Message Processor, Safety Layer, Genealogical RAG, and Harvester.
- [x] **Phase 4: Database Schema Updates**
  - Add `ConversationState` and `SuggestedFact` tables to `database.py`. Ensure SQLAlchemy models are correctly defined.
- [x] **Phase 5: Safety & Ethics Module**
  - Implement `SafetyGuard` class in `messaging/safety.py` to detect red flags (self-harm, hostility) and opt-outs.
- [x] **Phase 6: Refactor Action 10 (Research Service)**
  - Extract search and pathfinding logic from `actions/action10.py` into a reusable `genealogy/research_service.py`.
- [x] **Phase 7: Inbound Orchestrator**
  - Implement `InboundOrchestrator` in `messaging/inbound.py` to coordinate safety checks, intent classification, and RAG.
- [x] **Phase 8: Testing & Validation**
  - Create unit and integration tests for the new modules using the `TestSuite` pattern.

## Maintenance & Refactoring

- [x] **Database Migration**: Ensure all db files in `/Data` are migrated to the new schema. (Verified `ancestry.db` schema).
- [x] **Architecture Review**: Review whether we need separate `/core/cache` and `/caching` directories.
  - *Finding*: `caching/` contains the concrete implementation (`cache.py`), while `core/cache/` contains protocols/interfaces. `core/cache_backend.py` is also heavily used. Recommendation: Consolidate into `caching/` directory, moving interfaces to `caching/interfaces.py` and backend logic to `caching/backend.py`.
- [x] **File Organization**: Ensure `ethnicity_regions.json` is saved/created/read from `/Data` not root. Move `ethnicity_regions.json` to `/Data`.
