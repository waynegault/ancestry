#!/usr/bin/env python3

# log_analyzer.py - COMPLETE UPDATED SCRIPT (v7 - Removed _create_or_update_* timings)

import re
import logging
import statistics
from collections import defaultdict


def analyze_log(log_filepath):
    """
    Analyzes a log file with the updated format, extracts key metrics, and prints a summary report.
    _create_or_update_person and _create_or_update_dna_match timings have been removed.
    """

    error_warning_counts = defaultdict(int)
    wait_durations = defaultdict(list)
    process_match_durations = []
    skipped_updated_new_counts = defaultdict(int)
    dynamic_delay_counts = defaultdict(int)
    nav_to_page_durations = []  # Store nav_to_page durations separately
    scrape2_durations = []  # Store scrape2 durations

    try:
        with open(log_filepath, "r", encoding="utf-8") as f:
            for line in f:
                # --- Error/Warning Counts ---
                if (
                    "WARNING - [utils check_for_javascript_errors" in line
                    and "429 ()" in line
                ):
                    error_warning_counts["429 Errors"] += 1
                elif (
                    "WARNING - [utils nav_to_page line" in line
                    and "Timeout with no specific unavailability reason" in line
                ):
                    error_warning_counts["Navigation Timeouts"] += 1
                elif (
                    "ERROR - [action6_gather scrape3 line" in line
                    and "Timeout while extracting relationship path" in line
                ):
                    error_warning_counts["scrape3 Timeouts"] += 1
                elif (
                    "WARNING - [action6_gather _extract_tree_name line" in line
                    and "Tree name element not found." in line
                ):
                    error_warning_counts["_extract_tree_name Failures"] += 1
                elif (
                    "WARNING - [action6_gather _process_match_details line" in line
                    and "RETRYING _process_match_details." in line
                ):
                    error_warning_counts["_process_match_details Retries"] += 1
                elif (
                    "ERROR - [action6_gather _process_match_details line" in line
                    and "SKIPPING MATCH." in line
                ):
                    error_warning_counts["Action 6 Errors"] += 1

                # --- Dynamic Delay Counts ---
                elif (
                    "INFO - [action6_gather coord line" in line
                    and "Decreasing delay to" in line
                ):
                    dynamic_delay_counts["Delay Decreases"] += 1
                elif (
                    "INFO - [action6_gather coord line" in line
                    and "Maintaining delay" in line
                ):
                    dynamic_delay_counts["Delay Maintained (throttled)"] += 1

                # --- WebDriverWait Timings ---
                elif (
                    "DEBUG - [utils wrapper line" in line
                    and "Wait '" in line
                    and "duration" in line
                ):
                    match = re.search(
                        r"Wait '([^']+) duration' completed in (\d+\.\d+) seconds", line
                    )
                    if match:
                        wait_description = match.group(1)
                        duration = float(match.group(2))
                        wait_durations[wait_description].append(duration)

                # --- _process_match_details Timings ---
                elif (
                    "DEBUG - [utils wrapper line" in line
                    and "_process_match_details duration" in line
                ):
                    match = re.search(
                        r"_process_match_details duration (\d+\.\d+) s", line
                    )
                    if match:
                        duration = float(match.group(1))
                        process_match_durations.append(duration)

                # --- nav_to_page Timings ---
                elif (
                    "DEBUG - [utils wrapper line" in line
                    and "nav_to_page" in line
                    and "s." in line
                ):
                    match = re.search(r"nav_to_page (\d+\.\d+) s\.", line)
                    if match:
                        duration = float(match.group(1))
                        nav_to_page_durations.append(duration)

                # --- scrape2 Timings ---
                elif (
                    "DEBUG - [utils wrapper line" in line and "scrape2 duration" in line
                ):
                    match = re.search(r"scrape2 duration (\d+\.\d+) s", line)
                    if match:
                        duration = float(match.group(1))
                        scrape2_durations.append(duration)

                # --- Skipped/Updated/New Counts ---  (Updated Regex)
                elif (
                    "DEBUG - [action6_gather coord line" in line
                    and "Page" in line
                    and "batch" in line
                ):
                    match = re.search(
                        r"Page \d+ batch \d+: New (\d+), Update (\d+), Skip (\d+), Error (\d+)",
                        line,
                    )
                    if match:
                        skipped_updated_new_counts["New"] += int(match.group(1))
                        skipped_updated_new_counts["Updated"] += int(match.group(2))
                        skipped_updated_new_counts["Skipped"] += int(match.group(3))
                        error_warning_counts["Action 6 Errors"] += int(
                            match.group(4)
                        )  # Count errors from summary line

                # --- NEW:  Sub-total Counts (Page Level) ---
                elif (
                    "INFO - [action6_gather coord line" in line
                    and "Page" in line
                    and "sub-total" in line
                ):
                    match = re.search(
                        r"Page \d+ sub-total: New (\d+), Update (\d+), Skip (\d+), Error (\d+)",
                        line,
                    )
                    # NO IF NEEDED.  Directly add values to counts.  Values will accumulate.
                    if (
                        match
                    ):  # KEEP THIS IF, AS IT CHECKS THE REGEX MATCHED BEFORE PROCEEDING
                        skipped_updated_new_counts["New"] += int(match.group(1))
                        skipped_updated_new_counts["Updated"] += int(match.group(2))
                        skipped_updated_new_counts["Skipped"] += int(match.group(3))
                        error_warning_counts["Action 6 Errors"] += int(
                            match.group(4)
                        )  # Accumulate errors.

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_filepath}")
        return None
    except Exception as e:
        print(f"Error during log analysis: {e}")
        return None

    # --- Prepare Summary Report ---
    report = "\nLog Analysis Summary:\n\n"

    report += "Error/Warning Counts:\n"
    report += "---------------------\n"
    for error_type, count in error_warning_counts.items():
        report += f"{error_type:<30} {count}\n"
    report += "\n"

    report += "Dynamic Rate Limiting:\n"
    report += "-----------------------\n"
    if dynamic_delay_counts:
        for delay_type, count in dynamic_delay_counts.items():
            report += f"{delay_type:<30} {count}\n"
    else:
        report += "No dynamic delay adjustments recorded.\n"
    report += "\n"

    report += "Action Counts (Process Match Details):\n"
    report += "-------------------------------------\n"
    for result_type, count in skipped_updated_new_counts.items():
        report += f"{result_type:<30} {count}\n"
    # Don't print Action 6 Errors again here, as it is part of error_warning_counts
    report += "\n"

    report += "WebDriverWait Timing (seconds):\n"
    report += "-------------------------------\n"
    for wait_type, durations in wait_durations.items():
        if durations:
            report += format_statistics(wait_type, durations)
        else:
            report += f"Wait Type: {wait_type} - No timings recorded\n"
    report += "\n"

    report += "_process_match_details Timing (seconds):\n"
    report += "----------------------------------------\n"
    if process_match_durations:
        report += format_statistics("_process_match_details", process_match_durations)
    else:
        report += "    No _process_match_details timings recorded\n"
    report += "\n"

    report += "nav_to_page Timing (seconds):\n"
    report += "------------------------------\n"
    if nav_to_page_durations:
        report += format_statistics("nav_to_page", nav_to_page_durations)
    else:
        report += "    No nav_to_page timings recorded\n"
    report += "\n"

    report += "scrape2 Timing (seconds):\n"
    report += "--------------------------\n"
    if scrape2_durations:
        report += format_statistics("scrape2", scrape2_durations)
    else:
        report += "   No scrape2 timings recorded\n"
    report += "\n"

    return report


def format_statistics(name, durations):
    """Formats and returns timing statistics as a string."""
    if not durations:
        return f"    No {name} timings recorded\n"

    report = f"Wait Type: {name}\n"  # Use the provided name
    report += f"    Average:    {statistics.mean(durations):.3f}\n"
    report += f"    Min:        {min(durations):.3f}\n"
    report += f"    Max:        {max(durations):.3f}\n"
    report += f"    50th Percentile: {statistics.median(durations):.3f}\n"

    if len(durations) >= 10:
        report += (
            f"    90th Percentile: {statistics.quantiles(durations, n=10)[8]:.3f}\n"
        )
    else:
        report += "    90th Percentile: N/A (Data points < 10)\n"

    if len(durations) >= 100:
        report += (
            f"    99th Percentile: {statistics.quantiles(durations, n=100)[98]:.3f}\n"
        )
    else:
        report += "    99th Percentile: N/A (Data points < 100)\n"
    return report


if __name__ == "__main__":
    log_file = "logs/ancestry.log"  # Replace with your log file path
    summary = analyze_log(log_file)
    if summary:
        print(summary)

# End of log_analyzer.py - COMPLETE UPDATED SCRIPT (v7 - Removed _create_or_update_* timings)
