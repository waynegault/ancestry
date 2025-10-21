#!/usr/bin/env python3

"""
Phase 4.1: Adaptive Timing Demonstration

Demonstrates the engagement-based follow-up timing system that considers:
1. Engagement score from conversation_state (0-100)
2. Last login activity from people table
3. Configurable thresholds from .env

Run: python test_adaptive_timing_demo.py
"""

from datetime import datetime, timedelta, timezone

from action8_messaging import calculate_adaptive_interval
from config import config_schema


def print_header(title: str) -> None:
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_scenario(
    scenario_num: int,
    description: str,
    engagement_score: int,
    days_since_login: int | None,
    expected_tier: str,
    expected_days: int,
) -> None:
    """Print and test a timing scenario."""
    print(f"\n📊 Scenario {scenario_num}: {description}")
    print(f"   Engagement Score: {engagement_score}/100")

    if days_since_login is None:
        print("   Last Login: Never")
        last_logged_in = None
    else:
        print(f"   Last Login: {days_since_login} days ago")
        last_logged_in = datetime.now(timezone.utc) - timedelta(days=days_since_login)

    # Calculate interval
    interval = calculate_adaptive_interval(engagement_score, last_logged_in, f"Scenario {scenario_num}")

    # Verify result
    if interval.days == expected_days:
        print(f"   ✅ Result: {interval.days} days (Tier: {expected_tier})")
    else:
        print(f"   ❌ Result: {interval.days} days (Expected: {expected_days}, Tier: {expected_tier})")


def main() -> None:
    """Run adaptive timing demonstration."""
    print_header("PHASE 4.1: ADAPTIVE TIMING DEMONSTRATION")

    print("\n📋 Configuration from .env:")
    print(f"   High Engagement Threshold: {config_schema.engagement_high_threshold}")
    print(f"   Medium Engagement Threshold: {config_schema.engagement_medium_threshold}")
    print(f"   Low Engagement Threshold: {config_schema.engagement_low_threshold}")
    print(f"   Active Login Threshold: {config_schema.login_active_threshold} days")
    print(f"   Moderate Login Threshold: {config_schema.login_moderate_threshold} days")
    print()
    print(f"   High Engagement Follow-up: {config_schema.followup_high_engagement_days} days")
    print(f"   Medium Engagement Follow-up: {config_schema.followup_medium_engagement_days} days")
    print(f"   Low Engagement Follow-up: {config_schema.followup_low_engagement_days} days")
    print(f"   No Engagement Follow-up: {config_schema.followup_no_engagement_days} days")

    print_header("TIMING SCENARIOS")

    # Scenario 1: High engagement + active login
    print_scenario(
        1,
        "High Engagement + Active Login",
        engagement_score=85,
        days_since_login=3,
        expected_tier="High",
        expected_days=7,
    )

    # Scenario 2: High engagement but inactive login
    print_scenario(
        2,
        "High Engagement + Inactive Login",
        engagement_score=85,
        days_since_login=45,
        expected_tier="Medium",
        expected_days=14,
    )

    # Scenario 3: Medium engagement
    print_scenario(
        3,
        "Medium Engagement",
        engagement_score=55,
        days_since_login=20,
        expected_tier="Medium",
        expected_days=14,
    )

    # Scenario 4: Low engagement + moderate login
    print_scenario(
        4,
        "Low Engagement + Moderate Login",
        engagement_score=25,
        days_since_login=15,
        expected_tier="Medium (login activity)",
        expected_days=14,
    )

    # Scenario 5: Low engagement + inactive login
    print_scenario(
        5,
        "Low Engagement + Inactive Login",
        engagement_score=25,
        days_since_login=60,
        expected_tier="Low",
        expected_days=21,
    )

    # Scenario 6: No engagement
    print_scenario(
        6,
        "No Engagement",
        engagement_score=10,
        days_since_login=100,
        expected_tier="None",
        expected_days=30,
    )

    # Scenario 7: Never logged in
    print_scenario(
        7,
        "Never Logged In",
        engagement_score=15,
        days_since_login=None,
        expected_tier="None",
        expected_days=30,
    )

    # Scenario 8: Edge case - exactly at threshold
    print_scenario(
        8,
        "Exactly at High Threshold",
        engagement_score=70,
        days_since_login=5,
        expected_tier="High",
        expected_days=7,
    )

    print_header("SUMMARY")
    print("\n✅ Adaptive Timing System:")
    print("   • Considers BOTH engagement score AND login activity")
    print("   • Prioritizes active, engaged users (7-day follow-up)")
    print("   • Respects low engagement (21-30 day follow-up)")
    print("   • Prevents spam to uninterested matches")
    print("   • Configurable via .env settings")
    print()
    print("📝 Note: These intervals are ADDED to the minimum message interval:")
    print("   • Production: 8 weeks minimum + adaptive interval")
    print("   • Dry Run: 30 seconds minimum + adaptive interval")
    print("   • Testing: 10 seconds minimum + adaptive interval")
    print()
    print("🎯 Result: Intelligent, engagement-aware messaging that respects user interest!")
    print()


if __name__ == "__main__":
    main()

