#!/usr/bin/env python3

# cache_manager.py

"""
cache_manager.py - Centralized Cache Management System

Provides centralized management for all caching systems including GEDCOM caching,
API response caching, database query caching, and cache warming strategies.
This module orchestrates aggressive caching across the entire application.
"""

# --- Standard library imports ---
import time
from typing import Dict, Any, List, Optional

# --- Local application imports ---
from cache import get_cache_stats, clear_cache, invalidate_cache_pattern
from config import config_instance
from logging_config import logger


class CacheManager:
    """
    Centralized cache management system for aggressive caching strategies.
    """

    def __init__(self):
        """Initialize the cache manager."""
        self.initialization_time = time.time()
        self.cache_stats_history: List[Dict[str, Any]] = []
        self.last_stats_time = 0

    def initialize_all_caches(self) -> Dict[str, bool]:
        """
        Initialize and warm all cache systems.

        Returns:
            Dictionary indicating success/failure of each cache system
        """
        logger.info("Initializing aggressive caching systems...")
        start_time = time.time()

        results = {
            "gedcom_cache": False,
            "api_cache": False,
            "db_cache": False,
            "message_templates": False,
        }

        # Initialize GEDCOM cache
        try:
            from gedcom_cache import preload_gedcom_cache

            results["gedcom_cache"] = preload_gedcom_cache()
            logger.info(
                f"GEDCOM cache initialization: {'SUCCESS' if results['gedcom_cache'] else 'FAILED'}"
            )
        except ImportError:
            logger.debug("GEDCOM cache module not available")
        except Exception as e:
            logger.error(f"Error initializing GEDCOM cache: {e}")

        # Initialize message templates cache
        try:
            from action8_messaging import load_message_templates

            templates = load_message_templates()
            results["message_templates"] = bool(templates)
            logger.info(
                f"Message templates cache: {'SUCCESS' if results['message_templates'] else 'FAILED'}"
            )
        except Exception as e:
            logger.error(f"Error loading message templates: {e}")

        # Initialize API cache (basic setup)
        try:
            from api_cache import get_api_cache_stats

            api_stats = get_api_cache_stats()
            results["api_cache"] = bool(api_stats)
            logger.info(
                f"API cache system: {'READY' if results['api_cache'] else 'FAILED'}"
            )
        except Exception as e:
            logger.error(f"Error initializing API cache: {e}")

        # Database cache is ready by default (uses decorators)
        results["db_cache"] = True

        initialization_time = time.time() - start_time
        successful_systems = sum(results.values())
        total_systems = len(results)

        logger.info(f"Cache initialization completed in {initialization_time:.2f}s")
        logger.info(
            f"Successfully initialized {successful_systems}/{total_systems} cache systems"
        )

        return results

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics from all cache systems.

        Returns:
            Dictionary with detailed cache statistics
        """
        stats = {
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self.initialization_time,
            "base_cache": {},
            "gedcom_cache": {},
            "api_cache": {},
        }

        # Get base cache statistics
        try:
            stats["base_cache"] = get_cache_stats()
        except Exception as e:
            logger.debug(f"Error getting base cache stats: {e}")

        # Get GEDCOM cache statistics
        try:
            from gedcom_cache import get_gedcom_cache_info

            stats["gedcom_cache"] = get_gedcom_cache_info()
        except ImportError:
            stats["gedcom_cache"] = {"status": "not_available"}
        except Exception as e:
            logger.debug(f"Error getting GEDCOM cache stats: {e}")
            stats["gedcom_cache"] = {"error": str(e)}

        # Get API cache statistics
        try:
            from api_cache import get_api_cache_stats

            stats["api_cache"] = get_api_cache_stats()
        except ImportError:
            stats["api_cache"] = {"status": "not_available"}
        except Exception as e:
            logger.debug(f"Error getting API cache stats: {e}")
            stats["api_cache"] = {"error": str(e)}

        # Store stats history (keep last 10 entries)
        self.cache_stats_history.append(stats)
        if len(self.cache_stats_history) > 10:
            self.cache_stats_history.pop(0)

        self.last_stats_time = time.time()
        return stats

    def log_cache_performance(self) -> None:
        """Log cache performance statistics."""
        stats = self.get_comprehensive_stats()

        logger.info("=== Cache Performance Report ===")

        # Base cache stats
        base_stats = stats.get("base_cache", {})
        if base_stats:
            hits = base_stats.get("hits", 0)
            misses = base_stats.get("misses", 0)
            total_requests = hits + misses
            hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0

            logger.info(
                f"Base Cache: {hits} hits, {misses} misses, {hit_rate:.1f}% hit rate"
            )
            logger.info(f"Cache size: {base_stats.get('size', 0)} entries")
            logger.info(f"Cache volume: {base_stats.get('volume', 0)} bytes")

        # GEDCOM cache stats
        gedcom_stats = stats.get("gedcom_cache", {})
        if gedcom_stats and "status" not in gedcom_stats:
            memory_entries = gedcom_stats.get("memory_cache_entries", 0)
            logger.info(f"GEDCOM Cache: {memory_entries} memory entries")

            if "gedcom_file_size_mb" in gedcom_stats:
                file_size = gedcom_stats["gedcom_file_size_mb"]
                logger.info(f"GEDCOM file size: {file_size:.1f} MB")

        # API cache stats
        api_stats = stats.get("api_cache", {})
        if api_stats and "status" not in api_stats:
            api_entries = api_stats.get("api_entries", 0)
            ai_entries = api_stats.get("ai_entries", 0)
            db_entries = api_stats.get("db_entries", 0)

            logger.info(
                f"API Cache: {api_entries} API, {ai_entries} AI, {db_entries} DB entries"
            )

        logger.info("=== End Cache Report ===")

    def clear_all_caches(self) -> Dict[str, bool]:
        """
        Clear all cache systems.

        Returns:
            Dictionary indicating success/failure of clearing each cache system
        """
        logger.info("Clearing all cache systems...")

        results = {
            "base_cache": False,
            "memory_cache": False,
        }

        # Clear base disk cache
        try:
            results["base_cache"] = clear_cache()
            logger.info(
                f"Base cache clear: {'SUCCESS' if results['base_cache'] else 'FAILED'}"
            )
        except Exception as e:
            logger.error(f"Error clearing base cache: {e}")

        # Clear GEDCOM memory cache
        try:
            from gedcom_cache import clear_memory_cache

            cleared_count = clear_memory_cache()
            results["memory_cache"] = True
            logger.info(f"Memory cache cleared: {cleared_count} entries")
        except ImportError:
            logger.debug("GEDCOM cache module not available for clearing")
        except Exception as e:
            logger.error(f"Error clearing memory cache: {e}")

        return results

    def invalidate_stale_caches(self) -> int:
        """
        Invalidate potentially stale cache entries.

        Returns:
            Number of cache entries invalidated
        """
        logger.info("Invalidating potentially stale cache entries...")

        total_invalidated = 0

        # Invalidate old API responses (older than 2 hours)
        patterns_to_invalidate = [
            "api_ancestry_profile",  # Profile details might change
            "api_ancestry_suggest",  # Suggestions might change
        ]

        for pattern in patterns_to_invalidate:
            try:
                count = invalidate_cache_pattern(pattern)
                total_invalidated += count
                logger.debug(f"Invalidated {count} entries matching pattern: {pattern}")
            except Exception as e:
                logger.debug(f"Error invalidating pattern {pattern}: {e}")

        logger.info(f"Total cache entries invalidated: {total_invalidated}")
        return total_invalidated

    def optimize_cache_performance(self) -> Dict[str, Any]:
        """
        Analyze and optimize cache performance.

        Returns:
            Dictionary with optimization results and recommendations
        """
        stats = self.get_comprehensive_stats()

        recommendations = []
        optimizations_applied = []

        # Analyze hit rates
        base_stats = stats.get("base_cache", {})
        if base_stats:
            hits = base_stats.get("hits", 0)
            misses = base_stats.get("misses", 0)
            total_requests = hits + misses

            if total_requests > 100:  # Only analyze if we have enough data
                hit_rate = hits / total_requests

                if hit_rate < 0.5:  # Less than 50% hit rate
                    recommendations.append("Consider increasing cache expiration times")
                    recommendations.append(
                        "Review cache key generation for consistency"
                    )

                if hit_rate > 0.9:  # Very high hit rate
                    recommendations.append("Cache is performing excellently")
                    recommendations.append(
                        "Consider expanding cache to cover more operations"
                    )

        # Check cache size
        cache_size = base_stats.get("size", 0)
        if cache_size > 10000:  # Large number of entries
            recommendations.append("Consider implementing cache size limits")
            recommendations.append("Review cache eviction policies")

        return {
            "stats": stats,
            "recommendations": recommendations,
            "optimizations_applied": optimizations_applied,
            "analysis_time": time.time(),
        }


# Global cache manager instance
cache_manager = CacheManager()


def initialize_aggressive_caching() -> bool:
    """
    Initialize all aggressive caching systems.

    Returns:
        True if initialization successful, False otherwise
    """
    try:
        results = cache_manager.initialize_all_caches()
        success_count = sum(results.values())
        total_count = len(results)

        logger.info(
            f"Aggressive caching initialization: {success_count}/{total_count} systems ready"
        )
        return success_count > 0  # At least one system should be working
    except Exception as e:
        logger.error(f"Error during aggressive caching initialization: {e}")
        return False


def get_cache_performance_report() -> Dict[str, Any]:
    """
    Get a comprehensive cache performance report.

    Returns:
        Dictionary with cache performance data
    """
    return cache_manager.get_comprehensive_stats()


def log_cache_status() -> None:
    """Log current cache status and performance."""
    cache_manager.log_cache_performance()


# --- Self-Testing Functions ---


def run_cache_manager_tests() -> bool:
    """
    Run comprehensive tests for the cache manager system.

    Returns:
        True if all tests pass, False otherwise
    """
    print("=" * 70)
    print("CACHE MANAGER - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    all_tests_passed = True
    test_count = 0
    passed_count = 0

    def test_result(test_name: str, success: bool, details: str = ""):
        nonlocal test_count, passed_count, all_tests_passed
        test_count += 1
        if success:
            passed_count += 1
            print(f"✅ {test_name}: PASSED {details}")
        else:
            all_tests_passed = False
            print(f"❌ {test_name}: FAILED {details}")

    print(f"\n🧪 Starting {test_count} Cache Manager Tests...")

    # Test 1: Cache Manager Initialization
    try:
        test_manager = CacheManager()
        test_result(
            "Cache Manager Initialization",
            hasattr(test_manager, "initialization_time"),
            f"(uptime: {time.time() - test_manager.initialization_time:.2f}s)",
        )
    except Exception as e:
        test_result("Cache Manager Initialization", False, f"(error: {e})")

    # Test 2: Initialize All Caches
    try:
        results = cache_manager.initialize_all_caches()
        success_count = sum(results.values())
        total_count = len(results)
        test_result(
            "Cache System Initialization",
            success_count > 0,
            f"({success_count}/{total_count} systems)",
        )
    except Exception as e:
        test_result("Cache System Initialization", False, f"(error: {e})")

    # Test 3: Get Comprehensive Stats
    try:
        stats = cache_manager.get_comprehensive_stats()
        required_keys = [
            "timestamp",
            "uptime_seconds",
            "base_cache",
            "gedcom_cache",
            "api_cache",
        ]
        has_all_keys = all(key in stats for key in required_keys)
        test_result(
            "Comprehensive Statistics", has_all_keys, f"(keys: {list(stats.keys())})"
        )
    except Exception as e:
        test_result("Comprehensive Statistics", False, f"(error: {e})")

    # Test 4: Cache Performance Logging
    try:
        # Capture log output (this will log to configured logger)
        cache_manager.log_cache_performance()
        test_result("Cache Performance Logging", True, "(logged successfully)")
    except Exception as e:
        test_result("Cache Performance Logging", False, f"(error: {e})")

    # Test 5: Cache Invalidation
    try:
        invalidated_count = cache_manager.invalidate_stale_caches()
        test_result(
            "Cache Invalidation",
            invalidated_count >= 0,
            f"({invalidated_count} entries invalidated)",
        )
    except Exception as e:
        test_result("Cache Invalidation", False, f"(error: {e})")

    # Test 6: Cache Performance Optimization
    try:
        optimization_results = cache_manager.optimize_cache_performance()
        required_keys = [
            "stats",
            "recommendations",
            "optimizations_applied",
            "analysis_time",
        ]
        has_all_keys = all(key in optimization_results for key in required_keys)
        recommendations_count = len(optimization_results.get("recommendations", []))
        test_result(
            "Performance Optimization",
            has_all_keys,
            f"({recommendations_count} recommendations)",
        )
    except Exception as e:
        test_result("Performance Optimization", False, f"(error: {e})")

    # Test 7: Global Function Tests
    try:
        success = initialize_aggressive_caching()
        test_result(
            "Global Initialize Function",
            isinstance(success, bool),
            f"(result: {success})",
        )
    except Exception as e:
        test_result("Global Initialize Function", False, f"(error: {e})")

    # Test 8: Global Performance Report
    try:
        report = get_cache_performance_report()
        test_result(
            "Global Performance Report",
            isinstance(report, dict) and len(report) > 0,
            f"(keys: {len(report)})",
        )
    except Exception as e:
        test_result("Global Performance Report", False, f"(error: {e})")

    # Test 9: Global Cache Status Logging
    try:
        log_cache_status()
        test_result("Global Cache Status Logging", True, "(logged successfully)")
    except Exception as e:
        test_result("Global Cache Status Logging", False, f"(error: {e})")

    # Test 10: Cache History Tracking
    try:
        # Generate multiple stats to test history tracking
        for i in range(3):
            cache_manager.get_comprehensive_stats()
            time.sleep(0.1)

        history_length = len(cache_manager.cache_stats_history)
        test_result(
            "Cache History Tracking", history_length >= 3, f"({history_length} entries)"
        )
    except Exception as e:
        test_result("Cache History Tracking", False, f"(error: {e})")

    # Test 11: Cache Clear Operations
    try:
        clear_results = cache_manager.clear_all_caches()
        test_result(
            "Cache Clear Operations",
            isinstance(clear_results, dict),
            f"(systems: {list(clear_results.keys())})",
        )
    except Exception as e:
        test_result("Cache Clear Operations", False, f"(error: {e})")

    # Final Results
    print(f"\n🏁 Cache Manager Test Results:")
    print(f"   Tests Run: {test_count}")
    print(f"   Passed: {passed_count}")
    print(f"   Failed: {test_count - passed_count}")
    print(f"   Success Rate: {(passed_count/test_count)*100:.1f}%")

    if all_tests_passed:
        print("\n✅ ALL CACHE MANAGER TESTS PASSED!")
    else:
        print(f"\n⚠️  {test_count - passed_count} CACHE MANAGER TESTS FAILED!")

    return all_tests_passed


def demonstrate_cache_manager_usage():
    """
    Demonstrates practical usage of the cache manager system.
    """
    print("\n" + "=" * 70)
    print("CACHE MANAGER - USAGE DEMONSTRATION")
    print("=" * 70)

    print("\n--- Cache Manager Overview ---")
    print(f"Cache Manager Instance: {cache_manager}")
    print(f"Uptime: {time.time() - cache_manager.initialization_time:.2f} seconds")
    print(f"Stats History Entries: {len(cache_manager.cache_stats_history)}")

    print("\n--- Initializing All Cache Systems ---")
    start_time = time.time()
    results = cache_manager.initialize_all_caches()
    init_time = time.time() - start_time

    print(f"Initialization completed in {init_time:.3f} seconds")
    for system, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {system}: {status}")

    print("\n--- Comprehensive Cache Statistics ---")
    stats = cache_manager.get_comprehensive_stats()

    # Base cache stats
    base_stats = stats.get("base_cache", {})
    if base_stats:
        hits = base_stats.get("hits", 0)
        misses = base_stats.get("misses", 0)
        total = hits + misses
        hit_rate = (hits / total * 100) if total > 0 else 0
        print(f"Base Cache: {hits} hits, {misses} misses ({hit_rate:.1f}% hit rate)")
        print(f"Cache Size: {base_stats.get('size', 0)} entries")
        print(f"Cache Volume: {base_stats.get('volume', 0):,} bytes")

    # GEDCOM cache stats
    gedcom_stats = stats.get("gedcom_cache", {})
    if gedcom_stats and "status" not in gedcom_stats:
        memory_entries = gedcom_stats.get("memory_cache_entries", 0)
        print(f"GEDCOM Cache: {memory_entries} memory entries")
    elif "status" in gedcom_stats:
        print(f"GEDCOM Cache: {gedcom_stats['status']}")

    # API cache stats
    api_stats = stats.get("api_cache", {})
    if api_stats and "status" not in api_stats:
        api_entries = api_stats.get("api_entries", 0)
        ai_entries = api_stats.get("ai_entries", 0)
        db_entries = api_stats.get("db_entries", 0)
        print(f"API Cache: {api_entries} API, {ai_entries} AI, {db_entries} DB entries")
    elif "status" in api_stats:
        print(f"API Cache: {api_stats['status']}")

    print("\n--- Performance Optimization Analysis ---")
    optimization = cache_manager.optimize_cache_performance()
    recommendations = optimization.get("recommendations", [])

    if recommendations:
        print(f"Found {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("No specific recommendations at this time.")

    print("\n--- Cache Invalidation Test ---")
    invalidated = cache_manager.invalidate_stale_caches()
    print(f"Invalidated {invalidated} potentially stale cache entries")

    print("\n--- Cache Manager Demo Complete ---")


# --- Main Execution Block ---

if __name__ == "__main__":
    print("Starting Cache Manager Test and Demonstration Suite...")
    print(f"Cache Manager Module: {__file__}")

    # Run comprehensive tests
    print("\n" + "🧪 " + "RUNNING COMPREHENSIVE TESTS" + " 🧪")
    success = run_cache_manager_tests()

    print("\n" + "💡 " + "RUNNING USAGE DEMONSTRATIONS" + " 💡")
    demonstrate_cache_manager_usage()

    print("\n" + "=" * 70)
    print("CACHE MANAGER TEST AND DEMO COMPLETE!")
    print("=" * 70)

    if success:
        print("✅ All tests passed! Cache Manager system is fully operational.")
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        exit(1)

# End of cache_manager.py
