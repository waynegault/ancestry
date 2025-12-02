# Operator Manual: Review Queue & Human-in-the-Loop Controls

## Overview

This manual describes how to operate the Human-in-the-Loop (HITL) system for reviewing AI-generated messages before they are sent to DNA matches. The system ensures quality control and prevents unwanted communications.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Review Queue CLI](#review-queue-cli)
3. [Approval Workflow](#approval-workflow)
4. [A/B Testing](#ab-testing)
5. [Opt-Out Management](#opt-out-management)
6. [Emergency Controls](#emergency-controls)
7. [Monitoring & Metrics](#monitoring--metrics)

---

## Quick Start

### Prerequisites

- Access to the Ancestry automation system
- Database connection configured
- Python environment activated

### Basic Commands

```bash
# View pending drafts
python -m cli.review_queue list

# Approve a specific draft
python -m cli.review_queue approve --id 123

# Reject a draft with reason
python -m cli.review_queue reject --id 123 --reason "Inappropriate tone"

# View queue statistics
python -m cli.review_queue stats
```

---

## Review Queue CLI

### Listing Pending Drafts

```bash
# List all pending drafts (default: 50)
python -m cli.review_queue list

# List with custom limit
python -m cli.review_queue list --limit 100

# Filter by priority
python -m cli.review_queue list --priority high
python -m cli.review_queue list --priority critical
```

**Output Example:**
```
┌──────┬────────────────┬──────────────────┬────────────┬──────────┐
│  ID  │ Person         │ Conversation     │ Confidence │ Priority │
├──────┼────────────────┼──────────────────┼────────────┼──────────┤
│  123 │ John Smith     │ conv_abc123      │ 85%        │ normal   │
│  124 │ Mary Johnson   │ conv_def456      │ 72%        │ high     │
│  125 │ Robert Wilson  │ conv_ghi789      │ 45%        │ critical │
└──────┴────────────────┴──────────────────┴────────────┴──────────┘
```

### Viewing Draft Details

```bash
# View full draft content
python -m cli.review_queue view --id 123
```

**Output Example:**
```
═══════════════════════════════════════════════════════════════
DRAFT #123 - John Smith
═══════════════════════════════════════════════════════════════

📅 Created: 2025-12-01 10:30:00 UTC
📊 AI Confidence: 85%
🎯 Priority: normal
📝 Status: PENDING

─────────────────────────────────────────────────────────────────
CONTEXT:
─────────────────────────────────────────────────────────────────
Last message from John:
"Hi! I noticed we share DNA. Do you know anything about the
Smith family from Scotland?"

─────────────────────────────────────────────────────────────────
GENERATED DRAFT:
─────────────────────────────────────────────────────────────────
Hello John! Thank you for reaching out. I'm excited to explore
our potential connection through the Smith family.

Based on my family tree, I have several Smith ancestors from
Scotland, including James Smith (1845-1920) from Aberdeen.

Do you have any specific ancestors you'd like to compare?

═══════════════════════════════════════════════════════════════
```

### Approving Drafts

```bash
# Simple approval
python -m cli.review_queue approve --id 123

# Approval with edited content
python -m cli.review_queue approve --id 123 --edit

# This opens an editor to modify the draft before approving
```

### Rejecting Drafts

```bash
# Reject with reason
python -m cli.review_queue reject --id 123 --reason "Tone too formal"

# Reject multiple drafts
python -m cli.review_queue reject --ids 123,124,125 --reason "Batch rejection"
```

### Batch Operations

```bash
# Approve all drafts above 90% confidence
python -m cli.review_queue batch-approve --min-confidence 90

# Expire old pending drafts (default: 72 hours)
python -m cli.review_queue expire --hours 72

# Clear all rejected drafts
python -m cli.review_queue clear-rejected
```

---

## Approval Workflow

### Priority Levels

| Priority | Confidence | Description | Action Required |
|----------|------------|-------------|-----------------|
| **LOW** | ≥90% | High confidence, routine message | Auto-approve eligible |
| **NORMAL** | 70-89% | Standard confidence | Manual review |
| **HIGH** | <70% | Low confidence | Priority review |
| **CRITICAL** | Any | Sensitive history | Immediate review |

### Auto-Approval Rules

Auto-approval is enabled by default for drafts that meet ALL criteria:

1. AI confidence ≥ 90%
2. Not the first message to this person
3. Person has no DESIST history
4. Priority is LOW or NORMAL

**To disable auto-approval:**
```bash
python -m cli.review_queue config --auto-approve off
```

**To re-enable:**
```bash
python -m cli.review_queue config --auto-approve on
```

### Review Checklist

Before approving a draft, verify:

- [ ] **Accuracy**: Facts mentioned match your tree
- [ ] **Tone**: Appropriate for the conversation context
- [ ] **Personalization**: Uses correct names and details
- [ ] **No PII Leaks**: Doesn't expose sensitive information
- [ ] **Opt-Out Respected**: Person hasn't requested no contact

---

## A/B Testing

### Viewing Active Experiments

```bash
python -m cli.experiments list
```

### Experiment Results

```bash
# View summary statistics
python -m cli.experiments results --id experiment_1

# Export detailed results
python -m cli.experiments export --id experiment_1 --output results.json
```

**Output Example:**
```
═══════════════════════════════════════════════════════════════
EXPERIMENT: prompt_tone_test
═══════════════════════════════════════════════════════════════

Status: RUNNING
Start: 2025-11-15 00:00:00 UTC
Total Trials: 247

─────────────────────────────────────────────────────────────────
VARIANT STATISTICS:
─────────────────────────────────────────────────────────────────
┌──────────────┬─────────┬──────────────┬───────────────┐
│ Variant      │ Samples │ Avg Quality  │ Success Rate  │
├──────────────┼─────────┼──────────────┼───────────────┤
│ control      │ 125     │ 78.3         │ 82.4%         │
│ formal_tone  │ 122     │ 81.7         │ 87.5%         │
└──────────────┴─────────┴──────────────┴───────────────┘

🏆 Leading Variant: formal_tone (+3.4 quality points)
📊 Confidence: 72% (need more samples for significance)

═══════════════════════════════════════════════════════════════
```

### Stopping an Experiment

```bash
# Stop and declare winner
python -m cli.experiments stop --id experiment_1 --winner formal_tone

# Stop without winner (inconclusive)
python -m cli.experiments stop --id experiment_1
```

---

## Opt-Out Management

### Checking Opt-Out Status

```bash
# Check specific person
python -m cli.optout check --person-id 123

# Search by name
python -m cli.optout search --name "John Smith"
```

### Manual Opt-Out

```bash
# Mark person as DESIST
python -m cli.optout add --person-id 123 --reason "Requested via email"

# Remove opt-out (rare, requires confirmation)
python -m cli.optout remove --person-id 123 --confirm
```

### Opt-Out Audit

```bash
# View recent opt-out detections
python -m cli.optout audit --days 7

# Export opt-out log
python -m cli.optout export --output optout_log.csv
```

**Output Example:**
```
═══════════════════════════════════════════════════════════════
OPT-OUT AUDIT (Last 7 Days)
═══════════════════════════════════════════════════════════════

Total Detected: 12
Manual Overrides: 2
False Positives: 1

─────────────────────────────────────────────────────────────────
RECENT DETECTIONS:
─────────────────────────────────────────────────────────────────
2025-12-01 08:15:00 | Person #456 | "Please stop contacting me"
  → Confidence: 95% | Action: BLOCKED | Status: CONFIRMED

2025-11-30 14:22:00 | Person #789 | "I'm not interested"
  → Confidence: 70% | Action: FLAGGED | Status: REVIEWED (false positive)

═══════════════════════════════════════════════════════════════
```

---

## Emergency Controls

### Stop All Processing

```bash
# Immediately halt all message sending
python -m cli.emergency stop-all

# This will:
# - Disable auto-approval
# - Block all pending sends
# - Log emergency stop event
```

### Resume Processing

```bash
# Resume after emergency stop
python -m cli.emergency resume --confirm
```

### Emergency Blocklist

```bash
# Add person to emergency blocklist (immediate effect)
python -m cli.emergency block --person-id 123

# View emergency blocklist
python -m cli.emergency blocklist
```

---

## Monitoring & Metrics

### Queue Statistics

```bash
python -m cli.review_queue stats
```

**Output Example:**
```
═══════════════════════════════════════════════════════════════
QUEUE STATISTICS
═══════════════════════════════════════════════════════════════

📊 Current Queue:
   Pending: 23
   Auto-Approved (today): 145
   Manually Approved (today): 12
   Rejected (today): 3
   Expired: 5

📈 By Priority:
   Critical: 2
   High: 5
   Normal: 14
   Low: 2

⏱️ Processing Time:
   Average: 1.2s per draft
   Total Today: 2.5 minutes

═══════════════════════════════════════════════════════════════
```

### Performance Metrics

```bash
# View AI performance metrics
python -m cli.metrics ai

# View opt-out detection accuracy
python -m cli.metrics optout

# Export metrics to file
python -m cli.metrics export --output metrics.json
```

### Log Analysis

```bash
# View recent review activity
tail -f Logs/review_queue.log

# Search for specific events
grep "REJECTED" Logs/review_queue.log | tail -20
```

---

## Troubleshooting

### Common Issues

#### "No pending drafts found"

- Check if auto-approval is consuming all drafts
- Verify database connection
- Run `python -m cli.review_queue stats` to see queue status

#### "Draft already processed"

- Draft may have been auto-approved or expired
- Check the draft status with `python -m cli.review_queue view --id <id>`

#### "Cannot approve - person opted out"

- Person has DESIST status in database
- This is expected behavior - opt-outs are protected
- If this is an error, use `python -m cli.optout check --person-id <id>` to investigate

### Getting Help

```bash
# View all CLI commands
python -m cli.review_queue --help

# Get help for specific command
python -m cli.review_queue approve --help
```

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REVIEW_QUEUE_AUTO_APPROVE` | `true` | Enable/disable auto-approval |
| `REVIEW_QUEUE_EXPIRE_HOURS` | `72` | Hours before drafts expire |
| `REVIEW_QUEUE_MIN_CONFIDENCE` | `90` | Minimum confidence for auto-approve |

### Configuration File

Location: `config/review_queue.json`

```json
{
  "auto_approve": {
    "enabled": true,
    "min_confidence": 90,
    "exclude_first_message": true
  },
  "expiration": {
    "hours": 72,
    "auto_expire": true
  },
  "notifications": {
    "critical_alerts": true,
    "daily_summary": true
  }
}
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-01 | Initial release with Sprint 4 features |

---

*Last Updated: December 1, 2025*
