#!/usr/bin/env python3
"""
Verification script for Opt-Out / DESIST pattern detection.
Ensures that SafetyGuard catches all expected variations of opt-out requests.
"""

import logging
import sys

from messaging.safety import SafetyGuard, SafetyStatus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def verify_opt_out_patterns() -> bool:
    guard = SafetyGuard()

    # List of phrases that MUST be caught as either OPT_OUT, UNSAFE, or CRITICAL_ALERT
    # These represent various ways a user might say "stop"
    desist_phrases = [
        # Direct Opt-Outs
        "stop",
        "STOP",
        "unsubscribe",
        "remove me",
        "remove me from your list",
        "not interested",
        "I am not interested",
        "don't message me",
        "do not message me",
        "please stop messaging",
        "leave me alone",
        "take me off your list",
        # Hostile / Spam Accusations (should also stop automation)
        "spam",
        "this is spam",
        "harassment",
        "stop harassing me",
        "spammer",
        # Legal / Threats (should stop automation)
        "cease and desist",
        "do not contact me",
        "do not contact me again",
        "I will call the police",
        "lawyer",
        "sue you",
        "report you",
        # Variations
        "Pls stop",
        "Not interested.",
        "Remove me.",
    ]

    failures: list[str] = []
    passed_count = 0

    print(f"Testing {len(desist_phrases)} DESIST patterns...")
    print("-" * 60)
    print(f"{'PHRASE':<40} | {'STATUS':<15} | {'RESULT'}")
    print("-" * 60)

    for phrase in desist_phrases:
        # Check legacy check_message
        result_legacy = guard.check_message(phrase)

        # Check Phase 2 check_critical_alerts
        result_critical = guard.check_critical_alerts(phrase)

        # Determine if caught
        caught = False
        status_str = "MISSED"

        # Logic: It is caught if legacy returns OPT_OUT or UNSAFE
        # OR if critical returns CRITICAL_ALERT

        legacy_caught = result_legacy.status in {SafetyStatus.OPT_OUT, SafetyStatus.UNSAFE}
        critical_caught = result_critical.status == SafetyStatus.CRITICAL_ALERT

        if legacy_caught and critical_caught:
            caught = True
            status_str = "BOTH"
        elif legacy_caught:
            caught = True
            status_str = "LEGACY_ONLY"
        elif critical_caught:
            caught = True
            cat_val = result_critical.category.value if result_critical.category else "UNKNOWN"
            status_str = f"CRITICAL({cat_val})"

        if caught:
            print(f"{phrase:<40} | {status_str:<15} | ✅ CAUGHT")
            passed_count += 1
        else:
            print(f"{phrase:<40} | {status_str:<15} | ❌ FAILED")
            failures.append(phrase)

    print("-" * 60)
    print(f"Results: {passed_count}/{len(desist_phrases)} passed.")

    if failures:
        print("\n❌ FAILURES (False Negatives):")
        for f in failures:
            print(f"  - '{f}'")
        return False

    print("\n✅ SUCCESS: All DESIST patterns detected (Zero False Negatives).")
    return True


if __name__ == "__main__":
    success = verify_opt_out_patterns()
    sys.exit(0 if success else 1)

# -----------------------------------------------------------------------------
# Standard Test Runner
# -----------------------------------------------------------------------------
from testing.test_utilities import create_standard_test_runner


def _test_module_integrity() -> bool:
    "Test that module can be imported and definitions are valid."
    return True


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
