#!/usr/bin/env python3
"""
Quick validation script to verify the optimized rate limiting configuration.
Tests that all new parameters are loaded correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).resolve().parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import config_schema


def validate_rate_limiting_config():
    """Validate rate limiting configuration."""
    print("🔍 Validating Rate Limiting Configuration...\n")

    # Expected values
    expected = {
        'thread_pool_workers': 2,
        'requests_per_second': 0.8,
        'initial_delay': 1.0,
        'max_delay': 15.0,
        'backoff_factor': 1.5,
        'decrease_factor': 0.95,
        'token_bucket_capacity': 10.0,
        'token_bucket_fill_rate': 2.0,
    }

    errors = []
    warnings = []

    print("📊 Configuration Values:")
    print("-" * 60)

    for key, expected_value in expected.items():
        actual_value = getattr(config_schema.api, key, None)
        status = "✅" if actual_value == expected_value else "⚠️"

        print(f"{status} {key:.<40} {actual_value}")

        if actual_value != expected_value:
            if actual_value is None:
                errors.append(f"Missing config: {key}")
            else:
                warnings.append(f"Unexpected value for {key}: {actual_value} (expected: {expected_value})")

    print("-" * 60)

    # Performance calculations
    print("\n📈 Expected Performance:")
    print("-" * 60)

    workers = config_schema.api.thread_pool_workers
    rps = config_schema.api.requests_per_second
    per_worker_rps = rps / workers if workers > 0 else 0
    delay_between_requests = 1 / per_worker_rps if per_worker_rps > 0 else 0

    print(f"Total Workers:            {workers}")
    print(f"Total RPS:                {rps:.2f} requests/second")
    print(f"Per-Worker RPS:           {per_worker_rps:.2f} requests/second")
    print(f"Delay Between Requests:   {delay_between_requests:.2f} seconds")
    print(f"Expected Time (20 items): {20 / rps:.1f} seconds")
    print(f"Expected Time (40 items): {40 / rps:.1f} seconds")

    print("-" * 60)

    # Safety checks
    print("\n🛡️ Safety Checks:")
    print("-" * 60)

    checks_passed = True

    # Check 1: RPS per worker should be <= 0.5 for safety
    if per_worker_rps > 0.5:
        print("⚠️  Warning: Per-worker RPS > 0.5 may risk rate limiting")
        warnings.append(f"High per-worker RPS: {per_worker_rps:.2f}")
        checks_passed = False
    else:
        print(f"✅ Per-worker RPS ({per_worker_rps:.2f}) is safe (≤ 0.5)")

    # Check 2: Initial delay should be reasonable
    if config_schema.api.initial_delay < 0.5:
        print(f"⚠️  Warning: Initial delay ({config_schema.api.initial_delay}s) may be too aggressive")
        warnings.append(f"Low initial delay: {config_schema.api.initial_delay}s")
    else:
        print(f"✅ Initial delay ({config_schema.api.initial_delay}s) is reasonable")

    # Check 3: Max delay should allow recovery
    if config_schema.api.max_delay < 10:
        print(f"⚠️  Warning: Max delay ({config_schema.api.max_delay}s) may be too short")
        warnings.append(f"Low max delay: {config_schema.api.max_delay}s")
    else:
        print(f"✅ Max delay ({config_schema.api.max_delay}s) allows proper recovery")

    # Check 4: Token bucket should be configured
    if config_schema.api.token_bucket_capacity <= 0:
        print("❌ Error: Token bucket capacity must be positive")
        errors.append("Invalid token bucket capacity")
        checks_passed = False
    else:
        print(f"✅ Token bucket capacity ({config_schema.api.token_bucket_capacity}) is configured")

    print("-" * 60)

    # Summary
    print("\n📝 Validation Summary:")
    print("=" * 60)

    if errors:
        print(f"\n❌ {len(errors)} Critical Error(s):")
        for error in errors:
            print(f"   • {error}")

    if warnings:
        print(f"\n⚠️  {len(warnings)} Warning(s):")
        for warning in warnings:
            print(f"   • {warning}")

    if not errors and not warnings:
        print("✅ All validations passed! Configuration is optimal.")
        print("\n🚀 Ready to run with 2 workers and improved rate limiting!")
        return True
    if not errors:
        print("\n✅ Configuration loaded successfully with minor warnings.")
        print("   Review warnings above, but system should run properly.")
        return True
    print("\n❌ Configuration has critical errors. Please fix before running.")
    return False


if __name__ == "__main__":
    print("=" * 60)
    print("RATE LIMITING CONFIGURATION VALIDATOR")
    print("=" * 60)
    print()

    try:
        success = validate_rate_limiting_config()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
