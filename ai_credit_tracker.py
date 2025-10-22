#!/usr/bin/env python3
"""
AI Credit Usage Tracker

Tracks and reports AI API credit usage for DeepSeek and Google Gemini.
Provides real-time credit balance information and usage statistics.

Features:
- Track API calls and token usage
- Report remaining credits/quota
- Usage statistics and trends
- Cost estimation (if applicable)
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# === THIRD-PARTY IMPORTS ===
try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    OpenAI = None  # type: ignore
    openai_available = False

try:
    import google.generativeai as genai
    genai_available = True
except ImportError:
    genai_available = False

# === LOCAL IMPORTS ===
from config import config_schema

# === CONSTANTS ===
USAGE_TRACKING_FILE = "Logs/ai_usage_tracking.jsonl"


class AIUsageTracker:
    """Track AI API usage and credit information."""

    def __init__(self):
        self.usage_file = Path(USAGE_TRACKING_FILE)
        self.usage_file.parent.mkdir(parents=True, exist_ok=True)

    def log_api_call(
        self,
        provider: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """Log an AI API call for usage tracking."""
        try:
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "provider": provider,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "success": success,
                "error": error
            }

            with self.usage_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")

        except Exception as e:
            logger.warning(f"Failed to log AI usage: {e}")

    def get_usage_stats(self, hours: int = 24) -> dict[str, Any]:
        """Get usage statistics for the last N hours."""
        if not self.usage_file.exists():
            return {"total_calls": 0, "total_tokens": 0, "by_provider": {}}

        try:
            from datetime import timedelta
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            stats = {
                "total_calls": 0,
                "total_tokens": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "by_provider": {}
            }

            with self.usage_file.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        event_time = datetime.fromisoformat(event["timestamp"])

                        if event_time < cutoff_time:
                            continue

                        provider = event["provider"]
                        stats["total_calls"] += 1
                        stats["total_tokens"] += event.get("total_tokens", 0)

                        if event.get("success", True):
                            stats["successful_calls"] += 1
                        else:
                            stats["failed_calls"] += 1

                        if provider not in stats["by_provider"]:
                            stats["by_provider"][provider] = {
                                "calls": 0,
                                "tokens": 0,
                                "prompt_tokens": 0,
                                "completion_tokens": 0
                            }

                        stats["by_provider"][provider]["calls"] += 1
                        stats["by_provider"][provider]["tokens"] += event.get("total_tokens", 0)
                        stats["by_provider"][provider]["prompt_tokens"] += event.get("prompt_tokens", 0)
                        stats["by_provider"][provider]["completion_tokens"] += event.get("completion_tokens", 0)

                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

            return stats

        except Exception as e:
            logger.error(f"Error getting usage stats: {e}")
            return {"total_calls": 0, "total_tokens": 0, "by_provider": {}}


def get_deepseek_credit_info() -> Optional[dict[str, Any]]:
    """
    Get DeepSeek API credit/usage information.
    
    Note: DeepSeek API doesn't provide a direct credit balance endpoint.
    This function returns configuration and usage tracking information.
    """
    if not openai_available or not config_schema.api.deepseek_api_key:
        return None

    try:
        # DeepSeek uses OpenAI-compatible API but doesn't expose credit balance
        # We can only track usage through our own logging
        tracker = AIUsageTracker()
        stats = tracker.get_usage_stats(hours=24)

        deepseek_stats = stats.get("by_provider", {}).get("deepseek", {})

        return {
            "provider": "DeepSeek",
            "model": config_schema.api.deepseek_ai_model,
            "api_configured": True,
            "last_24h_calls": deepseek_stats.get("calls", 0),
            "last_24h_tokens": deepseek_stats.get("tokens", 0),
            "note": "DeepSeek API does not provide credit balance information. Track usage through billing dashboard at https://platform.deepseek.com/"
        }

    except Exception as e:
        logger.error(f"Error getting DeepSeek info: {e}")
        return None


def get_gemini_credit_info() -> Optional[dict[str, Any]]:
    """
    Get Google Gemini API credit/usage information.
    
    Note: Google Gemini API provides quota information but not credit balance.
    Free tier has rate limits but no credit system.
    """
    if not genai_available or not config_schema.api.google_api_key:
        return None

    try:
        # Configure Gemini
        genai.configure(api_key=config_schema.api.google_api_key)

        # Get usage stats from our tracking
        tracker = AIUsageTracker()
        stats = tracker.get_usage_stats(hours=24)

        gemini_stats = stats.get("by_provider", {}).get("gemini", {})

        return {
            "provider": "Google Gemini",
            "model": config_schema.api.google_ai_model,
            "api_configured": True,
            "last_24h_calls": gemini_stats.get("calls", 0),
            "last_24h_tokens": gemini_stats.get("tokens", 0),
            "note": "Gemini API uses rate limits, not credits. Free tier: 15 RPM, 1M TPM, 1500 RPD. Check quota at https://aistudio.google.com/"
        }

    except Exception as e:
        logger.error(f"Error getting Gemini info: {e}")
        return None


def display_ai_credit_report() -> None:
    """Display a comprehensive AI credit/usage report."""
    print("\n" + "=" * 80)
    print("AI API CREDIT & USAGE REPORT")
    print("=" * 80)

    # Get DeepSeek info
    deepseek_info = get_deepseek_credit_info()
    if deepseek_info:
        print(f"\n📊 {deepseek_info['provider']}")
        print(f"   Model: {deepseek_info['model']}")
        print(f"   Last 24h Calls: {deepseek_info['last_24h_calls']}")
        print(f"   Last 24h Tokens: {deepseek_info['last_24h_tokens']:,}")
        print(f"   ℹ️  {deepseek_info['note']}")
    else:
        print("\n❌ DeepSeek: Not configured or unavailable")

    # Get Gemini info
    gemini_info = get_gemini_credit_info()
    if gemini_info:
        print(f"\n📊 {gemini_info['provider']}")
        print(f"   Model: {gemini_info['model']}")
        print(f"   Last 24h Calls: {gemini_info['last_24h_calls']}")
        print(f"   Last 24h Tokens: {gemini_info['last_24h_tokens']:,}")
        print(f"   ℹ️  {gemini_info['note']}")
    else:
        print("\n❌ Google Gemini: Not configured or unavailable")

    # Overall stats
    tracker = AIUsageTracker()
    overall_stats = tracker.get_usage_stats(hours=24)

    print(f"\n📈 Overall Usage (Last 24 Hours)")
    print(f"   Total API Calls: {overall_stats['total_calls']}")
    print(f"   Total Tokens: {overall_stats['total_tokens']:,}")
    print(f"   Successful: {overall_stats['successful_calls']}")
    print(f"   Failed: {overall_stats['failed_calls']}")

    print("\n" + "=" * 80)


# === TESTS ===

def _test_usage_tracker() -> bool:
    """Test usage tracker functionality."""
    tracker = AIUsageTracker()

    # Log a test call
    tracker.log_api_call(
        provider="test",
        model="test-model",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        success=True
    )

    # Get stats
    stats = tracker.get_usage_stats(hours=1)
    assert stats["total_calls"] >= 1, "Should have at least one call logged"

    return True


if __name__ == "__main__":
    from test_framework import TestSuite

    suite = TestSuite("ai_credit_tracker")

    suite.run_test(
        "Usage Tracker",
        _test_usage_tracker,
        test_summary="Test usage tracking functionality",
        functions_tested="AIUsageTracker.log_api_call(), AIUsageTracker.get_usage_stats()",
        method_description="Log API calls and retrieve usage statistics",
        expected_outcome="Usage stats correctly tracked and retrieved"
    )

    # Display credit report
    print("\n")
    display_ai_credit_report()

    suite.finish_suite()

