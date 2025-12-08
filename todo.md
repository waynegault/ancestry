# Development Ideas & Roadmap

## New Features

- [x] **Action 14 Enhancement**: Add capacity to choose one of 'ethnicity regions' and run the analysis against everyone who also shares that ethnicity region.

## UI/UX Improvements

- [x] **Menu Reordering**: Reorder actions so what is currently Action 13 (Shared Match Scraper) comes before Action 12 (Triangulation Analysis). This reflects the logical workflow where shared matches must be fetched before triangulation can be performed. (Already implemented: Action 12 is Scraper, Action 13 is Triangulation)

## Codebase Analysis

- [x] **Dead Code Identification**: Identify code that is written but not yet implemented/active in the production workflow. (See `dead_code_report.md`)
  - [x] `research/record_sharing.py`: Record sharing capabilities (Phase 5.5)
  - [x] `research/relationship_diagram.py`: Relationship diagram generation
  - [x] `research/research_prioritization.py`: Research prioritization system (Phase 12.3)
  - [x] `research/research_suggestions.py`: Research suggestion generation (Phase 5.2)

## Integration Tasks (From Dead Code Analysis)

- [x] **Integrate Relationship Diagrams**: Add `research/relationship_diagram.py` to Action 14 (Research Tools) to allow visualizing relationships.
- [x] **Integrate Record Sharing**: Add `research/record_sharing.py` to Action 8 (Messaging) to allow sharing records in messages.
- [x] **Integrate Research Suggestions**: Add `research/research_suggestions.py` to Action 9 (Productive Messages) to suggest next research steps.
- [x] **Integrate Research Prioritization**: Add `research/research_prioritization.py` to Action 14 (Research Tools) to prioritize research tasks.
