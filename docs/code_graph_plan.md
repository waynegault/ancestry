# Code Knowledge Graph Plan

This document tracks the structure and conventions used for the repository-wide knowledge graph.

## 1. Storage Format
- Primary artifact: `docs/code_graph.json`
- Encoding: UTF-8, plain JSON
- Top-level keys:
  - `nodes`: array of node objects (one per file/function/class/module)
  - `edges`: array of edge objects (relationships between nodes)
  - `metadata`: creation timestamp, schema version, notes

## 2. Node Schema
Each node object contains:
- `id`: unique string (e.g., `file:main.py`, `function:session_manager.exec_actn`)
- `type`: `file`, `module`, `class`, `function`, `workflow`, `config`, `asset`
- `name`: human-readable label
- `path`: relative path for file-backed entities
- `summary`: concise purpose statement (1-3 sentences)
- `mechanism`: key implementation approach or algorithms used
- `quality`: status flags (`stable`, `monitor`, `needs-review`, `dead-code`)
- `concerns`: list of reservations (performance, duplication, tech debt)
- `opportunities`: improvement or extension ideas
- `tests`: known tests covering the node or `null`
- `notes`: free-form observations (including TODO references)

### Node Types (controlled vocabulary)
- `file`: Top-level source file. Summary should describe the module’s primary responsibility and orchestration role.
- `module`: Logical namespace inside a file (e.g., grouped utilities). Use when the file contains multiple conceptual areas that merit separate tracking.
- `class`: Python class definitions. Mechanism field should capture inheritance or mixin behavior.
- `function`: Top-level or nested function. Record algorithmic approach and side effects.
- `workflow`: Cross-file procedure or CLI action that strings multiple functions together (e.g., Action 6 coordinator). Useful for documenting end-to-end flows.
- `config`: Configuration data structures, loaders, or environment-driven components.
- `asset`: Non-code artifacts (templates, prompts, SQL files) referenced by code paths.

### Quality Flags (single value)
- `stable`: Implementation reviewed with no outstanding risks.
- `monitor`: Works today but depends on volatile integrations or warrants telemetry review.
- `needs-review`: Unassessed or suspected of issues; prioritize during walkthrough.
- `dead-code`: Candidate for removal; confirm no legitimate references before cleanup.

Use the `concerns` and `opportunities` arrays to capture nuanced commentary; keep the `quality` flag high-level so dashboards can pivot on it.

## 3. Edge Schema
Each edge object contains:
- `source`: node id
- `target`: node id
- `type`: relationship type (`depends-on`, `calls`, `extends`, `uses-config`, `persists-to`, etc.)
- `description`: brief explanation of the link
- `quality`: optional indicator if relationship is risky (e.g., brittle workflow)

### Edge Types (recommended set)
- `calls`: Function invokes another function/class method.
- `depends-on`: File or module requires another component at runtime (imports, global state).
- `extends`: Class inheritance relationships.
- `uses-config`: Component reads configuration or environment values.
- `persists-to`: Writes to database tables, files, or caches.
- `reads-from`: Reads persisted data without mutating it.
- `publishes-event`: Emits telemetry, logs, or notifications consumed downstream.
- `guards`: Control-flow wrapper that enforces preconditions (e.g., rate limiter, circuit breaker).

Additional edge types can be introduced when necessary; document the rationale in `metadata.notes` before broad adoption.

## 4. Review Workflow
1. Select file according to traversal order.
2. Read file completely; summarize responsibilities.
3. Update inline comments/docstrings where clarity lacking.
4. Add/update node entries for file and nested constructs.
5. Capture outgoing edges (dependencies, API calls, DB interactions).
6. Record concerns, reservations, duplication notes, and improvement ideas.
7. Tick off corresponding task in `docs/review_todo.md` when file complete.

## 5. Versioning
- Increment `metadata.schemaVersion` if structure changes.
- Use ISO 8601 timestamps for `metadata.generatedAt`.
- Maintain changelog in `notes` for major revisions.

## 6. Automation Hooks
- Potential to generate visualizations from `code_graph.json` (future iteration).
- Consider CLI helper to validate node/edge structure.

## 7. Per-File Review Template
Capture the following when processing each module (adjust to file type as needed):

1. **Context**: File purpose, key entry points, and how it is invoked.
2. **Key Constructs**: Enumerate classes/functions added to the graph with summaries and mechanisms.
3. **Interactions**: List external modules, subsystems, or services touched (API, DB, AI, caches).
4. **Quality Notes**: Flag risks, duplication, performance concerns, or testing gaps; assign `quality` flag.
5. **Opportunities**: Record potential refactors, documentation improvements, or monitoring additions.
6. **Tests**: Note existing coverage (embedded TestSuite, external scripts) or required additions.
7. **Artifacts Updated**: Confirm `code_graph.json` nodes/edges added, `docs/review_todo.md` progress ticked, and any supplementary notes committed.

Store these notes briefly in `notes` fields or, if lengthy, in supplemental docs referenced by node ids.

## 8. Storage & Validation Workflow
- **Primary artifact**: Continue using `docs/code_graph.json`. Keep formatting stable (indentation, sorted keys where practical) to minimize diffs.
- **Edit protocol**: Use structured editors or scripted helpers to append nodes/edges. When editing manually, validate JSON with a quick `python -m json.tool docs/code_graph.json` check before committing.
- **Backups**: For major overhauls, take a timestamped copy under `Docs/archive/` (future enhancement) or rely on git branches.
- **Schema validation**: Before bumping `schemaVersion`, confirm new fields are present across representative nodes and update this plan.
- **Integration**: When new node types or edge categories emerge, extend the controlled vocab sections above and reference them in commit messages for easy diff tracking.
