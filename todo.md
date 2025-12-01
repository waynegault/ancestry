# Ancestry DNA Match Communication System - Implementation Plan

## Executive Summary

This plan outlines the roadmap for transforming the current Ancestry automation tools into a fully automated, intelligent system for DNA match engagement and family tree validation. The system will leverage existing modules (`action7`, `action8`, `action9`, `action10`) while introducing new capabilities for robust data validation, "RAG-style" knowledge retrieval, and human-in-the-loop safeguards.

## Phase 1: Codebase Assessment

### Current Capabilities

- [x] **Task 1: Document existing functionality**
  - [x] Audit `action7_inbox.py` for current classification accuracy and limitations.
  - [x] Audit `action9_process_productive.py` for data extraction quality.
  - [x] Review `action10.py` for relationship pathfinding performance.
  - [x] Document `ai_prompts.json` versioning and usage.
- [x] **Task 2: Map data flows**
  - [x] Trace flow from Inbox -> Classification -> Extraction -> Database.
  - [x] Map GEDCOM data ingestion -> `action10` analysis -> Context generation.
- [ ] **Task 3: Catalog tech stack**
  - [ ] Confirm dependencies (Selenium, SQLAlchemy, LLM providers).
  - [ ] Verify rate limiting and session management stability (`core/session_manager.py`).

### Gap Analysis

- [ ] **Task 1: Identify missing features**
  - [ ] "True" RAG retrieval pipeline (currently relies on structured lookups).
  - [ ] Formal "Conflict Detection" between conversation data and Tree data.
  - [ ] Dedicated "Review Queue" UI or CLI for human approval.
  - [ ] A/B testing framework for message templates.
- [ ] **Task 2: Document technical debt**
  - [ ] Consolidate duplicate logic between `action9` and `action10` (if any).
  - [ ] Review error handling for edge cases in conversation parsing.
- [ ] **Task 3: Note security considerations**
  - [ ] Audit PII handling in logs and AI prompts.
  - [ ] Verify "Opt-out" persistence across system restarts.

## Phase 2: Technical Specification

- [ ] **Reply Management System design**
  - [ ] Define state machine for conversation flow (Initial -> Productive -> Data Extraction -> Conclusion).
  - [ ] Spec for "Critical Alert" detection (regex + AI).
- [ ] **Automated Response Engine architecture**
  - [ ] Design "Context Builder" service: Aggregates Tree Data + DNA Match Info + Conversation History.
  - [ ] Define "RAG" retrieval logic: `PersonLookup` -> `RelationshipPath` -> `FactVerification`.
- [ ] **Data Validation Pipeline spec**
  - [ ] Define `Fact` data structure (Source, Confidence, Value).
  - [ ] Design "Conflict Resolution" logic (New Fact vs Existing Tree Fact).
- [ ] **Engagement Optimization framework**
  - [ ] Design A/B testing schema for `MessageTemplate`.
  - [ ] Define metrics for "Engagement Success" (Reply rate, Data extracted count).
- [ ] **Human-in-the-Loop safeguards**
  - [ ] Design "Approval Queue" database schema.
  - [ ] Define "Stop Buttons" and manual override controls.

## Phase 3: Implementation Roadmap

### Sprint 1: Core Intelligence & Retrieval (The "Brain")

*Focus: Enhancing the ability to answer questions and retrieve tree data.*

- [ ] **Refactor `action10` for Real-time Querying** (P0)
  - [ ] Create `TreeQueryService` class to wrap `action10` logic.
  - [ ] Implement `find_person(name, approx_date, location)` with fuzzy matching.
  - [ ] Implement `explain_relationship(person_a, person_b)` returning natural language text.
- [ ] **Implement "Context Builder"** (P1)
  - [ ] Create module to assemble prompt context from `TreeQueryService` results.
  - [ ] **Acceptance**: Can generate a rich context string for any DNA match in the DB.

### Sprint 2: Reply Processing & Classification (The "Ears")

*Focus: Understanding what matches are saying.*

- [ ] **Enhance `action7` Classification** (P1)
  - [ ] Implement "Critical Alert" regex filter (Self-harm, threats).
  - [ ] Refine `intent_classification` prompt for "Social" vs "Genealogical" distinction.
- [ ] **Implement "Fact Extraction" 2.0** (P1)
  - [ ] Upgrade `action9` extraction to output standardized `Fact` objects.
  - [ ] **Acceptance**: 95% accuracy in extracting names/dates from test set.

### Sprint 3: Response Generation & Validation (The "Voice")

*Focus: Generating accurate, safe responses.*

- [ ] **Implement RAG Response Generator** (P0)
  - [ ] Connect `action9` to `TreeQueryService`.
  - [ ] Update `genealogical_reply` prompt to use retrieved Tree facts.
  - [ ] **Acceptance**: System answers "Who is your grandfather?" correctly using Tree data.
- [ ] **Implement Data Validation Logic** (P2)
  - [ ] Create `FactValidator` service.
  - [ ] Compare extracted `Fact` objects against DB.
  - [ ] Flag conflicts (e.g., "User says died 1950, Tree says 1945").

### Sprint 4: Engagement & Safeguards (The "Safety Net")

*Focus: Optimization and control.*

- [ ] **Implement Review Queue** (P1)
  - [ ] Create CLI or simple Web UI to view "Pending Actions".
  - [ ] **Acceptance**: User can Approve/Reject suggested replies and data updates.
- [ ] **Implement A/B Testing** (P2)
  - [ ] Add `variant_id` to `MessageTemplate`.
  - [ ] Track reply rates by variant.
- [ ] **Opt-out Hardening** (P0)
  - [ ] Ensure "DESIST" status is immutable by automated processes.

## Phase 4: Testing & Deployment

- [ ] **Unit tests**
  - [ ] Test `TreeQueryService` with mock GEDCOM data.
  - [ ] Test `FactValidator` with conflicting data scenarios.
- [ ] **Integration tests**
  - [ ] End-to-end flow: Inbound Message -> Classification -> Context Build -> Draft Reply.
- [ ] **Dry-run validation**
  - [ ] Run against 50 historical conversations.
  - [ ] Manual audit of generated drafts vs actual human replies.
- [ ] **Documentation**
  - [ ] Update `README.md` with new architecture.
  - [ ] Create "Operator Manual" for the Review Queue.

## Innovation: AI-Driven Enhancements

- [ ] **Triangulation Inference**: Use `action12_triangulation.py` to infer relationships for matches who haven't replied, then use that inference to personalize the *first* message.
- [ ] **Predictive Gaps**: Analyze the tree to find "Missing Great-Grandparents" and specifically target DNA matches who might descend from that line.
- [ ] **Sentiment Adaptation**: Adjust response tone (Formal vs Casual) based on the user's writing style analysis.
