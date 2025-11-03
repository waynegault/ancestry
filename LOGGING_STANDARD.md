# Standardized Logging Format for Actions 6-9

## Overview

This document defines the standardized logging format that all Actions (6, 7, 8, 9) must follow for consistent user experience and debugging.

## Logging Levels

- **INFO**: User-facing output showing progress, summaries, and results
- **DEBUG**: Detailed technical information for debugging (only shown when DEBUG_MODE=true)
- **WARNING**: Non-fatal issues that should be noted
- **ERROR**: Fatal issues that prevent operation

## The 12-Step Logging Standard

### Step 1: Configuration Summary

**When**: At the very start of the action, immediately after header from main.py
**Format**: Single line with all relevant configuration settings
**Example**:
```
15:30:00 INF [action6_ coord    1234] Configuration: APP_MODE=dry_run, START_PAGE=1, MAX_PAGES=2, BATCH_SIZE=10, RATE_LIMIT_DELAY=2.50s, AI_PROVIDER=local_llm
```

**Required Settings by Action**:
- Action 6: APP_MODE, START_PAGE, MAX_PAGES, BATCH_SIZE, RATE_LIMIT_DELAY
- Action 7: APP_MODE, MAX_INBOX, AI_PROVIDER, RATE_LIMIT_DELAY
- Action 8: APP_MODE, MAX_MESSAGES, BATCH_SIZE, RATE_LIMIT_DELAY, AI_PROVIDER
- Action 9: APP_MODE, MAX_PRODUCTIVE_TO_PROCESS, BATCH_SIZE, RATE_LIMIT_DELAY, AI_PROVIDER

### Step 2: Starting Position Summary

**When**: Immediately after configuration
**Format**: One or more lines describing what will be processed
**Example**:
```
15:30:00 INF [action6_ coord    1235] Starting from page 1, will process up to 2 pages
15:30:00 INF [action6_ coord    1236] Estimated matches to process: ~40 (20 per page)
```

### Step 3: Initial Cumulative Counts (Starting at Zero)

**When**: Before any processing begins
**Format**: Single line showing all counters at zero
**Example**:
```
15:30:01 INF [action6_ coord    1237] Cumulative: Pages=0, Batches=0, New=0, Updated=0, Skipped=0, Errors=0
```

**Required Counters by Action**:
- Action 6: Pages, Batches, New, Updated, Skipped, Errors
- Action 7: Conversations, Processed, AI_Classifications, Updates, Errors
- Action 8: Messages_Sent, Skipped, Errors
- Action 9: Conversations, Processed, Messages_Sent, Skipped, Errors

### Step 4: Batch/Page Indicator (Before Processing)

**When**: Immediately before processing each batch or page
**Format**: Clear indicator of what is about to be processed
**Example**:
```
15:30:02 INF [action6_ _process 1238] 
15:30:02 INF [action6_ _process 1239] Processing page 1 of 2 pages
15:30:02 INF [action6_ _process 1240] Batch 1/2 (matches 1-10 of 20)
```

**Note**: Blank line before page separator for visual clarity

### Step 5: Temporary Progress Bar (During Processing)

**When**: During batch/page processing
**Format**: Single tqdm progress bar with specific metrics
**Example**:
```
Processing: 45%|████████████          | 9/20 [00:15<00:18, 0.60 match/s]
```

**Required Metrics**:
- Percentage complete
- Progress bar visual
- Current/Total items
- Elapsed time
- Estimated remaining time
- Processing rate (items/second)

**Implementation**:
```python
from tqdm import tqdm
with tqdm(total=total_items, desc="Processing", unit=" match", 
          bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
    for item in items:
        # Process item
        pbar.update(1)
```

### Step 6: Updated Cumulative Counts (After Each Batch)

**When**: Immediately after each batch completes
**Format**: Same format as Step 3, but with updated counts
**Example**:
```
15:30:20 INF [action6_ _process 1241] Cumulative: Pages=0, Batches=1, New=8, Updated=2, Skipped=0, Errors=0
```

### Step 7: Blank Line Between Batches

**When**: After batch summary, before next batch indicator
**Format**: Single blank line
**Example**:
```
15:30:20 INF [action6_ _process 1242] 
```

### Step 8: Page Complete Summary (After All Batches on Page)

**When**: After all batches on a page are complete
**Format**: Summary line with page-specific counts
**Example**:
```
15:30:45 INF [action6_ _process 1243] Page 1 complete: Batches=2, New=15, Updated=5, Skipped=0, Errors=0
15:30:45 INF [action6_ _process 1244] Cumulative: Pages=1, Batches=2, New=15, Updated=5, Skipped=0, Errors=0
```

### Step 9: Final Summary (After All Processing)

**When**: After all pages/batches complete
**Format**: Separator lines with comprehensive summary
**Example**:
```
15:31:00 INF [action6_ coord    1245] 
15:31:00 INF [action6_ coord    1246] ================================================================================
15:31:00 INF [action6_ coord    1247] FINAL SUMMARY
15:31:00 INF [action6_ coord    1248] ================================================================================
15:31:00 INF [action6_ coord    1249] Pages Scanned:           2
15:31:00 INF [action6_ coord    1250] Batches Processed:       4
15:31:00 INF [action6_ coord    1251] New Matches:             30
15:31:00 INF [action6_ coord    1252] Updated Matches:         10
15:31:00 INF [action6_ coord    1253] Skipped Matches:         0
15:31:00 INF [action6_ coord    1254] Errors:                  0
15:31:00 INF [action6_ coord    1255] Total Run Time:          0 hr 1 min 30.45 sec
15:31:00 INF [action6_ coord    1256] ================================================================================
15:31:00 INF [action6_ coord    1257] 
```

**Required Fields**:
- Separator line (80 equals signs)
- "FINAL SUMMARY" header
- Separator line
- All cumulative counts with aligned labels
- Total run time in "X hr Y min Z.ZZ sec" format
- Separator line
- Blank line after

### Step 10: Performance Statistics (Rate Limiter + Circuit Breaker)

**When**: Immediately after final summary
**Format**: Call session_manager.rate_limiter.print_metrics_summary()
**Example**:
```
15:31:00 INF [utils    print_me 1471] Rate Limiter Metrics:
15:31:00 INF [utils    print_me 1472] Total Requests:        45
15:31:00 INF [utils    print_me 1473] Total Wait Time:       22.5s
15:31:00 INF [utils    print_me 1474] Average Wait:          0.50s
15:31:00 INF [utils    print_me 1475] Circuit Breaker Metrics:
15:31:00 INF [utils    print_me 1476] Circuit Opens:         0
15:31:00 INF [utils    print_me 1477] Circuit Closes:        0
15:31:00 INF [utils    print_me 1478] Half-Open Successes:   0
15:31:00 INF [utils    print_me 1479] Half-Open Failures:    0
15:31:00 INF [utils    print_me 1480] Uptime:                90.5s
```

**Implementation**:
```python
if hasattr(session_manager, 'rate_limiter') and session_manager.rate_limiter:
    session_manager.rate_limiter.print_metrics_summary()
```

### Step 11: Final Status Statement

**When**: After performance statistics
**Format**: Single line with checkmark or X indicating success/failure
**Example**:
```
15:31:00 INF [main     gather_d 1289] ✓ Match gathering completed successfully.
```
or
```
15:31:00 ERR [main     gather_d 1289] ✗ Match gathering failed: Session expired
```

### Step 12: Footer from main.py

**When**: Automatically added by main.py after action returns
**Format**: Separator line from main.py
**Example**:
```
15:31:00 INF [main     main     1290] ================================================================================
```

## Helper Functions to Implement in utils.py

### log_action_configuration()
```python
def log_action_configuration(action_name: str, config_dict: Dict[str, Any]) -> None:
    """Log action configuration in standardized format."""
    config_str = ", ".join([f"{k}={v}" for k, v in config_dict.items()])
    logger.info(f"Configuration: {config_str}")
```

### log_starting_position()
```python
def log_starting_position(description: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Log starting position summary."""
    logger.info(description)
    if details:
        for key, value in details.items():
            logger.info(f"{key}: {value}")
```

### log_cumulative_counts()
```python
def log_cumulative_counts(counts: Dict[str, int], prefix: str = "Cumulative") -> None:
    """Log cumulative counts in standardized format."""
    count_str = ", ".join([f"{k}={v}" for k, v in counts.items()])
    logger.info(f"{prefix}: {count_str}")
```

### log_batch_indicator()
```python
def log_batch_indicator(batch_num: int, total_batches: int, 
                       item_range: Optional[Tuple[int, int]] = None,
                       page_num: Optional[int] = None, 
                       total_pages: Optional[int] = None) -> None:
    """Log batch/page indicator before processing."""
    logger.info("")  # Blank line before
    if page_num and total_pages:
        logger.info(f"Processing page {page_num} of {total_pages} pages")
    if item_range:
        logger.info(f"Batch {batch_num}/{total_batches} (items {item_range[0]}-{item_range[1]})")
    else:
        logger.info(f"Batch {batch_num}/{total_batches}")
```

### create_standard_progress_bar()
```python
def create_standard_progress_bar(total: int, desc: str = "Processing", 
                                 unit: str = " item") -> tqdm:
    """Create standardized progress bar."""
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
```

### log_page_complete()
```python
def log_page_complete(page_num: int, page_counts: Dict[str, int], 
                     cumulative_counts: Dict[str, int]) -> None:
    """Log page completion summary."""
    page_str = ", ".join([f"{k}={v}" for k, v in page_counts.items()])
    logger.info(f"Page {page_num} complete: {page_str}")
    log_cumulative_counts(cumulative_counts)
```

### log_final_summary()
```python
def log_final_summary(summary_dict: Dict[str, Any], run_time_seconds: float) -> None:
    """Log final summary with separators."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    
    # Log all summary items with aligned labels
    max_label_len = max(len(k) for k in summary_dict.keys())
    for label, value in summary_dict.items():
        logger.info(f"{label + ':':<{max_label_len + 1}} {value}")
    
    # Log run time
    hours = int(run_time_seconds // 3600)
    minutes = int((run_time_seconds % 3600) // 60)
    seconds = run_time_seconds % 60
    logger.info(f"Total Run Time: {hours} hr {minutes} min {seconds:.2f} sec")
    
    logger.info("=" * 80)
    logger.info("")
```

### log_action_status()
```python
def log_action_status(action_name: str, success: bool, error_msg: Optional[str] = None) -> None:
    """Log final action status."""
    if success:
        logger.info(f"✓ {action_name} completed successfully.")
    else:
        logger.error(f"✗ {action_name} failed: {error_msg or 'Unknown error'}")
```

## Implementation Checklist

For each action (6, 7, 8, 9):

- [ ] Step 1: Add configuration logging at start
- [ ] Step 2: Add starting position summary
- [ ] Step 3: Initialize and log cumulative counts at zero
- [ ] Step 4: Add batch/page indicators before processing
- [ ] Step 5: Use single standardized progress bar during processing
- [ ] Step 6: Update and log cumulative counts after each batch
- [ ] Step 7: Add blank line between batches
- [ ] Step 8: Add page complete summary after all batches
- [ ] Step 9: Add final summary with separators
- [ ] Step 10: Call rate_limiter.print_metrics_summary()
- [ ] Step 11: Add final status statement
- [ ] Step 12: Verify main.py adds footer

## Testing Validation

After implementing, verify:

1. **No duplicate progress bars** - Only one progress indication visible at a time
2. **Cumulative counts accurate** - Counts match actual processing
3. **Consistent formatting** - All actions use same format
4. **Proper spacing** - Blank lines in correct places
5. **Performance stats** - Rate limiter and circuit breaker metrics appear
6. **Final status** - Clear success/failure indication
7. **No debug bleeding** - DEBUG logs only appear when DEBUG_MODE=true
8. **Contained output** - All output between header and footer from main.py

