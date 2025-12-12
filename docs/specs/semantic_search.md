# Technical Specification: Semantic Search (Tree-Aware Q&A)

## 1. Overview

**Semantic Search** is the tree-aware retrieval and answer synthesis capability that turns a natural-language question (from an inbound message, operator prompt, or workflow step) into:

- structured intent + extracted entities
- candidate resolution against the user's tree (GEDCOM) and Ancestry API search utilities
- evidence-backed, non-hallucinated answers
- follow-up questions when ambiguity exists
- optional staging of extracted facts through the existing validation + human-review loop

This spec is intentionally designed to build on existing primitives:

- `TreeQueryService` and Action 10 genealogy/search utilities
- `ContextBuilder` for match-centric context assembly
- Action 7/9 entity extraction patterns and `FactValidator`/`DataConflict` pipeline
- approval queue / draft-first posture for outbound messaging

**Non-goals (this increment):**

- Auto-sending replies without human review
- Introducing a vector database
- Writing directly into `Person`/tree records from AI output


---

## 2. User Stories

1. **Inbound Q&A (match asks about a person):**
   - “Do you have *Mary Ellen Smith* in your tree? She was in *Ohio around 1900*.”
   - The system extracts entities, finds candidate people, and replies with an evidence-backed answer or asks clarifying questions.

2. **Relationship explanation:**
   - “How do you think we’re related?”
   - The system attempts to explain the best-known relationship path (or the limits of current evidence), grounding claims in tree data.

3. **Operator research question (CLI / internal workflow):**
   - “Find candidates for John H. Miller born 1878 Indiana; list top 3 with sources.”

---

## 3. Inputs, Outputs, and Invariants

### 3.1 Inputs

- `query: str` — natural-language question
- `match_uuid: Optional[str]` — when available, used to anchor context to a specific DNA match
- `conversation_id: Optional[str]` — to associate logs and queued items

### 3.2 Output (SemanticSearchResult)

The core output should be a structured object (stored/logged as JSON) that supports both:

- generating a draft reply
- driving a human review UI/CLI

Recommended fields:

- `intent`: one of `PERSON_LOOKUP`, `RELATIONSHIP_EXPLANATION`, `RECORD_SUGGESTION`, `GENERAL_GENEALOGY_QA`, `CLARIFICATION_NEEDED`
- `entities`:
  - `people`: normalized names + optional constraints (birth/death year ranges)
  - `places`: normalized locations
  - `dates`: extracted years/ranges
  - `relationships`: keywords (mother, grandfather, etc.)
- `candidates`: for each person entity, up to N ranked candidates
- `evidence`: structured citations (what was used, from where)
- `answer_draft`: a safe, evidence-backed draft answer (or an explicit clarification request)
- `confidence`: 0–100
- `missing_information`: list of questions needed to disambiguate or validate
- `suggested_facts`: optional extracted facts for staging via `FactValidator` (never direct writes)

### 3.3 Invariants

- **No invented facts.** If evidence is insufficient, the answer must say so and ask for specifics.
- **Tree-first retrieval.** Prefer GEDCOM/tree sources; use Ancestry API search utilities as supplemental candidates.
- **Validation-first persistence.** Any “new” facts go through `FactValidator` → `SuggestedFact`/`DataConflict` before any downstream updates.

---

## 4. Architecture

### 4.1 Placement

Introduce a service layer that is reusable by Action 7 (inbound), Action 8 (draft generation), Action 9 (task generation), and operator tooling.

Proposed module:

- `genealogy/semantic_search.py` (or `research/semantic_search.py` if that better matches existing structure)

### 4.2 Dependencies

- Retrieval:
  - `TreeQueryService` (Action 10 utilities)
  - existing API search utilities under `api/` (name/date flexibility scoring)
- Context anchoring:
  - `ai/context_builder.py` (when `match_uuid` provided)
  - `ConversationLog` for recent thread context
- Extraction + validation:
  - AI prompts (existing patterns in `ai/ai_prompts.json`)
  - `FactValidator` and `DataConflict` staging
- HITL:
  - Approval queue / draft reply queueing

---

## 5. Pipeline

### 5.1 Query Understanding

**Goal:** parse the query into structured intent + entities.

- Use an LLM prompt (JSON output) to extract:
  - intent
  - people names (+ constraints)
  - places
  - dates
  - relationships
- Reuse the existing prompt/telemetry patterns:
  - parse success tracking
  - quality scoring

**Failure mode:** if parsing fails, return `CLARIFICATION_NEEDED` with a safe generic follow-up question.

### 5.2 Candidate Retrieval

For each extracted person entity:

1. **Tree query:** `TreeQueryService` name search (with fuzzing rules consistent with current `api_search_*` scoring).
2. **API candidate expansion (optional):** use Ancestry API search utilities if tree results are empty/weak.
3. **Rank + trim:** return top N candidates with:
   - name
   - birth/death (if known)
   - location snippets
   - source pointers (GEDCOM id, API record id, etc.)

### 5.3 Disambiguation

If candidates are ambiguous (close scores or conflicting evidence), produce:

- a short list of top candidates
- 1–3 clarifying questions (e.g., spouse name, exact birth year, county)

### 5.4 Evidence Assembly

Normalize evidence into a small set of “evidence blocks”:

- `source_type`: `GEDCOM`, `API_SEARCH`, `CONVERSATION_LOG`, `USER_NOTE`
- `source_id`: GEDCOM person id, API id, log id
- `summary`: short factual snippet
- `confidence`: 0–100

### 5.5 Answer Synthesis

Generate an answer draft from:

- the original question
- evidence blocks
- match-anchored context (if available)

**Answer constraints:**

- state uncertainty explicitly
- do not claim a specific relationship without evidence
- request missing facts when needed

### 5.6 Validation + Persistence (Optional)

If the pipeline extracts candidate facts that would improve the tree (names, dates, places, relationships), stage them via:

- `FactValidator` → `SuggestedFact` and `DataConflict`

Route uncertain/conflicting items into the review queue.

---

## 6. Integration Points

### 6.1 Action 7 (Inbox / InboundOrchestrator)

When an inbound message is classified as `PRODUCTIVE` and includes a question:

- run semantic search to assemble evidence and follow-up questions
- store semantic search result (log or new table) for transparency
- optionally stage extracted facts for Action 9 validation

### 6.2 Action 8 (Messaging)

Use semantic search output to:

- enrich contextual draft generation (answer + follow-up)
- remain **draft-first** unless explicitly enabled and guarded

### 6.3 Action 9 (Task Generation)

Semantic search evidence can seed task templates (“search obituary”, “verify census”) when uncertainty remains.

---

## 7. Acceptance Criteria

- Given a question containing a person name + year, the service returns:
  - extracted entity + top candidate list (<= 3)
  - at least one evidence block
  - either an answer draft or clarification questions
- No direct DB writes to core person/tree fields occur from AI output.
- Any extracted “facts” persist only as `SuggestedFact`/`DataConflict` (when enabled).
- Unit tests cover:
  - parse failure → safe clarification
  - ambiguity → clarification questions
  - candidate retrieval ranking stability for simple synthetic inputs

---

## 8. Risks & Mitigations

- **Hallucinated answers:** enforce evidence-only prompts; fail closed to clarification.
- **Over-querying APIs:** keep tree-first; use existing rate limiter and avoid new parallelism.
- **Unbounded context:** cap evidence blocks and conversation excerpts (reuse existing message window configs).
