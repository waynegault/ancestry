# Reply Flow Architecture

## Overview

This document describes the message processing and reply generation flow in the Ancestry platform.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INBOUND MESSAGE FLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌───────────────┐     ┌─────────────────┐
  │   Ancestry   │     │  Action 7:    │     │   SafetyGuard   │
  │    Inbox     │────▶│  InboxScraper │────▶│ + OptOutDetect  │
  └──────────────┘     └───────────────┘     └────────┬────────┘
                                                      │
                        ┌─────────────────────────────▼──────────────────────┐
                        │                  InboundOrchestrator                │
                        │  (messaging/inbound.py)                             │
                        └─────────────────────────────┬──────────────────────┘
                                                      │
              ┌───────────────────────────────────────┼───────────────────┐
              │                                       │                   │
              ▼                                       ▼                   ▼
    ┌──────────────────┐               ┌──────────────────┐    ┌──────────────┐
    │ Intent           │               │ Entity           │    │ Safety       │
    │ Classification   │               │ Extraction       │    │ Escalation   │
    │ (AI)             │               │ (AI)             │    │              │
    └────────┬─────────┘               └────────┬─────────┘    └──────┬───────┘
             │                                  │                     │
             ▼                                  ▼                     ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                        Message Classification                         │
    │  PRODUCTIVE | ENTHUSIASTIC | SOCIAL | DESIST | CRITICAL | OTHER      │
    └──────────────────────────────────────────────────────────────────────┘
                                      │
                ┌─────────────────────┼─────────────────────┐
                │                     │                     │
                ▼                     ▼                     ▼
        ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
        │   PRODUCTIVE  │    │    DESIST     │    │   CRITICAL    │
        │  → Research   │    │  → Opt-Out    │    │  → Escalate   │
        │    Flow       │    │    Flow       │    │    Flow       │
        └───────┬───────┘    └───────────────┘    └───────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 2: RESEARCH FLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐
    │ SemanticSearch   │────▶│  TreeQueryService │────▶│  ContextBuilder  │
    │ Service          │     │  (GEDCOM lookup)  │     │                  │
    └──────────────────┘     └───────────────────┘     └────────┬─────────┘
                                                                │
                                                                ▼
                                                    ┌───────────────────────┐
                                                    │  generate_structured  │
                                                    │  _reply() (AI)        │
                                                    └───────────┬───────────┘
                                                                │
                        ┌───────────────────────────────────────┼──────────────┐
                        │                                       │              │
                        ▼                                       ▼              ▼
              ┌──────────────────┐               ┌─────────────────┐  ┌──────────────┐
              │  High Confidence │               │ Low Confidence  │  │   Error/     │
              │  (≥50)           │               │ (<50)           │  │   Failure    │
              └────────┬─────────┘               └────────┬────────┘  └──────┬───────┘
                       │                                  │                  │
                       ▼                                  ▼                  ▼
              ┌───────────────────────────────────────────────────────────────────┐
              │                    ApprovalQueueService                            │
              │                 (DraftReply → Review Queue)                        │
              └───────────────────────────────┬───────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          REVIEW & SEND FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐
    │  Review Queue    │     │  Human Operator   │     │  Action 11:      │
    │  (Web UI/CLI)    │────▶│  Approve/Reject   │────▶│  Send Approved   │
    └──────────────────┘     └───────────────────┘     └────────┬─────────┘
                                                                │
                        ┌───────────────────────────────────────┴──────────────┐
                        │                                                      │
                        ▼                                                      ▼
              ┌──────────────────┐                                  ┌──────────────────┐
              │  Pre-Send Checks │                                  │  Send Message    │
              │  - Duplicate     │                                  │  via Ancestry    │
              │  - Rate Limit    │                                  │  API             │
              │  - Circuit Break │                                  └────────┬─────────┘
              └──────────────────┘                                           │
                                                                             ▼
                                                              ┌───────────────────────┐
                                                              │  Update State:        │
                                                              │  - DraftReply → SENT  │
                                                              │  - ConversationState  │
                                                              │  - ConversationLog    │
                                                              └───────────────────────┘
```

## Key Components

### 1. Message Ingestion (Action 7)
| Component | File | Purpose |
|-----------|------|---------|
| InboxProcessor | `actions/action7_inbox.py` | Scrapes inbox HTML |
| SafetyGuard | `core/safety_guard.py` | Detects critical alerts |
| OptOutDetector | `core/opt_out_detection.py` | Detects opt-out requests |
| InboundOrchestrator | `messaging/inbound.py` | Coordinates all processing |

### 2. Research Flow (Phase 2)
| Component | File | Purpose |
|-----------|------|---------|
| SemanticSearchService | `genealogy/semantic_search.py` | GEDCOM-based Q&A |
| TreeQueryService | `genealogy/tree_query_service.py` | Structured tree queries |
| ContextBuilder | `ai/context_builder.py` | Assembles AI context |
| generate_structured_reply | `ai/ai_interface.py` | AI response generation |

### 3. Review & Send (Action 11)
| Component | File | Purpose |
|-----------|------|---------|
| ApprovalQueueService | `core/approval_queue.py` | Draft management |
| ReviewServer | `ui/review_server.py` | Web UI (localhost:5000) |
| SendApprovedDrafts | `actions/action11_send_approved_drafts.py` | Message sending |

## Data Flow

```
ConversationLog (IN)
        │
        ▼
┌───────────────────┐
│ Intent + Entities │──────▶ SuggestedFact (for tree)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Research Evidence │──────▶ ContextBuilder.research_insights
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Draft Reply     │──────▶ DraftReply (status: PENDING)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Human Approval    │──────▶ DraftReply (status: APPROVED)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Send Message     │──────▶ ConversationLog (OUT)
└───────────────────┘──────▶ DraftReply (status: SENT)
```

## Error Handling

| Component | Pattern | Recovery |
|-----------|---------|----------|
| Session | Circuit Breaker | 5 failures → trip |
| API Calls | Rate Limiter | 0.3 RPS with backoff |
| Database | Transaction Wrapper | Rollback on failure |
| AI Calls | Retry with Fallback | 3 attempts, then skip |

## Confidence Routing

| Confidence | Action |
|------------|--------|
| ≥85 | High confidence, minimal review |
| 50-84 | Standard review queue |
| <50 | Auto-route to HUMAN_REVIEW |
| Error | Log and skip |
