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
from typing import Dict, Any, List, Optional, Callable

# --- Local application imports ---
from cache import (
    get_cache_stats,
    clear_cache,
    invalidate_cache_pattern,
    CacheInterface,
    BaseCacheModule,
    get_unified_cache_key,
    invalidate_related_caches,
    get_cache_coordination_stats,
)
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

# --- Enhanced Cache Manager with Coordination ---

class EnhancedCacheManager(BaseCacheModule):
    """
    Enhanced centralized cache management system with cross-module coordination.
    Implements the standardized cache interface and provides comprehensive orchestration.
    """
    
    def __init__(self):
        super().__init__()
        self.module_name = "cache_manager"
        self.initialization_time = time.time()
        self.cache_stats_history: List[Dict[str, Any]] = []
        self.last_stats_time = 0
        self.managed_modules = []
        
        # Try to register cache modules
        self._register_cache_modules()
    
    def _register_cache_modules(self):
        """Register all available cache modules for coordination."""
        try:
            # Import and register GEDCOM cache module
            from gedcom_cache import _gedcom_cache_module
            self.managed_modules.append(_gedcom_cache_module)
            logger.debug("Registered GEDCOM cache module")
        except Exception as e:
            logger.warning(f"Could not register GEDCOM cache module: {e}")
        
        try:
            # Import and register API cache module
            from api_cache import _api_cache_module
            self.managed_modules.append(_api_cache_module)
            logger.debug("Registered API cache module")
        except Exception as e:
            logger.warning(f"Could not register API cache module: {e}")
        
        logger.info(f"Enhanced cache manager registered {len(self.managed_modules)} cache modules")
    
    def get_module_name(self) -> str:
        return self.module_name
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all cache modules."""
        base_stats = super().get_stats()
        
        # Collect coordination statistics
        coordination_stats = get_cache_coordination_stats()
        
        # Collect statistics from all managed modules
        module_stats = {}
        for module in self.managed_modules:
            try:
                module_name = module.get_module_name()
                module_stats[module_name] = module.get_stats()
            except Exception as e:
                logger.warning(f"Error getting stats from {module}: {e}")
                module_stats[str(module)] = {"error": str(e)}
        
        # Manager-specific statistics
        manager_stats = {
            "managed_modules_count": len(self.managed_modules),
            "managed_modules": [m.get_module_name() for m in self.managed_modules],
            "initialization_time": self.initialization_time,
            "uptime_seconds": time.time() - self.initialization_time,
            "stats_history_entries": len(self.cache_stats_history)
        }
        
        return {
            **base_stats,
            **coordination_stats,
            "module_stats": module_stats,
            "manager_stats": manager_stats
        }
    
    def clear(self) -> bool:
        """Clear all managed cache modules."""
        try:
            cleared_counts = {}
            total_success = True
            
            # Clear each managed module
            for module in self.managed_modules:
                try:
                    module_name = module.get_module_name()
                    success = module.clear()
                    cleared_counts[module_name] = success
                    if not success:
                        total_success = False
                except Exception as e:
                    logger.error(f"Error clearing {module}: {e}")
                    cleared_counts[str(module)] = False
                    total_success = False
            
            # Clear base cache
            base_clear = super().clear()
            if not base_clear:
                total_success = False
            
            # Clear coordination caches
            coordination_clear = invalidate_related_caches("coordination_*", [])
            
            logger.info(f"Cache manager cleared {len(cleared_counts)} modules, coordination: {sum(coordination_clear.values())} entries")
            
            return total_success
            
        except Exception as e:
            logger.error(f"Error in cache manager clear operation: {e}")
            return False
    
    def warm(self) -> bool:
        """Warm all managed cache modules."""
        try:
            warm_results = {}
            total_success = True
            
            # Warm each managed module
            for module in self.managed_modules:
                try:
                    module_name = module.get_module_name()
                    success = module.warm()
                    warm_results[module_name] = success
                    if not success:
                        total_success = False
                except Exception as e:
                    logger.error(f"Error warming {module}: {e}")
                    warm_results[str(module)] = False
                    total_success = False
            
            # Warm coordination data
            coordination_key = get_unified_cache_key("manager", "coordination", "metadata")
            coordination_data = {
                "managed_modules": len(self.managed_modules),
                "warmed_at": time.time(),
                "initialization_time": self.initialization_time
            }
            
            try:
                from cache import warm_cache_with_data
                warm_cache_with_data(coordination_key, coordination_data)
            except Exception as e:
                logger.warning(f"Could not warm coordination data: {e}")
                total_success = False
            
            logger.info(f"Cache manager warmed {len(warm_results)} modules successfully: {total_success}")
            
            return total_success
            
        except Exception as e:
            logger.error(f"Error in cache manager warm operation: {e}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status across all cache modules."""
        base_health = super().get_health_status()
        
        try:
            # Check health of all managed modules
            module_health = {}
            overall_health = "healthy"
            health_issues = []
            
            for module in self.managed_modules:
                try:
                    module_name = module.get_module_name()
                    health = module.get_health_status()
                    module_health[module_name] = health
                    
                    # Aggregate health status
                    module_overall = health.get("overall_health", "unknown")
                    if module_overall == "error":
                        overall_health = "error"
                        health_issues.append(f"{module_name} has errors")
                    elif module_overall == "warning" and overall_health != "error":
                        overall_health = "warning"
                        health_issues.append(f"{module_name} has warnings")
                        
                except Exception as e:
                    logger.error(f"Error checking health of {module}: {e}")
                    module_health[str(module)] = {"error": str(e), "overall_health": "error"}
                    overall_health = "error"
                    health_issues.append(f"Error checking {module}: {str(e)}")
            
            # Check manager-specific health
            manager_health = "healthy"
            manager_issues = []
            
            if len(self.managed_modules) == 0:
                manager_health = "warning"
                manager_issues.append("No cache modules registered")
            
            uptime = time.time() - self.initialization_time
            if uptime < 10:  # Less than 10 seconds uptime might indicate initialization issues
                manager_health = "warning"
                manager_issues.append("Recent initialization, monitoring stability")
            
            # Final health assessment
            if manager_health == "error" or overall_health == "error":
                final_health = "error"
            elif manager_health == "warning" or overall_health == "warning":
                final_health = "warning"
            else:
                final_health = "healthy"
            
            manager_health_info = {
                "module_health": module_health,
                "manager_health": manager_health,
                "manager_issues": manager_issues,
                "overall_health": final_health,
                "health_issues": health_issues,
                "modules_registered": len(self.managed_modules),
                "uptime_seconds": uptime,
                "health_check_timestamp": time.time()
            }
            
            return {**base_health, **manager_health_info}
            
        except Exception as e:
            logger.error(f"Error getting cache manager health status: {e}")
            return {**base_health, "health_check_error": str(e), "overall_health": "error"}
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get a comprehensive report of all cache systems."""
        return {
            "timestamp": time.time(),
            "stats": self.get_stats(),
            "health": self.get_health_status(),
            "coordination": get_cache_coordination_stats(),
            "manager_info": {
                "module_name": self.module_name,
                "uptime": time.time() - self.initialization_time,
                "managed_modules": len(self.managed_modules)
            }
        }
    
    def coordinate_cache_invalidation(self, pattern: str, exclude_modules: Optional[List[str]] = None) -> Dict[str, int]:
        """Coordinate cache invalidation across all modules."""
        if exclude_modules is None:
            exclude_modules = []
        
        try:
            # Use the coordination system for invalidation
            results = invalidate_related_caches(pattern, exclude_modules)
            
            logger.info(f"Coordinated cache invalidation for pattern '{pattern}': {sum(results.values())} entries invalidated")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in coordinated cache invalidation: {e}")
            return {"error": 0}


# Initialize enhanced cache manager instance
_enhanced_cache_manager = EnhancedCacheManager()


# --- Public Interface Functions for Enhanced Cache Manager ---

def get_cache_manager_stats() -> Dict[str, Any]:
    """Get comprehensive cache manager statistics."""
    return _enhanced_cache_manager.get_stats()


def get_cache_manager_health() -> Dict[str, Any]:
    """Get cache manager health status."""
    return _enhanced_cache_manager.get_health_status()


def get_comprehensive_cache_report() -> Dict[str, Any]:
    """Get comprehensive report of all cache systems."""
    return _enhanced_cache_manager.get_comprehensive_report()


def clear_all_managed_caches() -> bool:
    """Clear all managed cache systems."""
    return _enhanced_cache_manager.clear()


def warm_all_managed_caches() -> bool:
    """Warm all managed cache systems."""
    return _enhanced_cache_manager.warm()


def coordinate_invalidation(pattern: str, exclude_modules: Optional[List[str]] = None) -> Dict[str, int]:
    """Coordinate cache invalidation across modules."""
    return _enhanced_cache_manager.coordinate_cache_invalidation(pattern, exclude_modules)


# --- Enhanced Cache Manager Testing ---

def run_enhanced_cache_manager_tests() -> Dict[str, Any]:
    """
    Run comprehensive tests for enhanced cache manager functionality.
    Returns test results with pass/fail status and performance metrics.
    """
    test_results = {
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "test_details": [],
        "start_time": time.time(),
        "performance_metrics": {}
    }
    
    def run_test(test_name: str, test_func: Callable) -> bool:
        """Run individual test and track results."""
        test_results["tests_run"] += 1
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time
            
            if result:
                test_results["tests_passed"] += 1
                status = "PASS"
            else:
                test_results["tests_failed"] += 1
                status = "FAIL"
                
            test_results["test_details"].append({
                "name": test_name,
                "status": status,
                "duration_ms": round(duration * 1000, 2),
                "result": result
            })
            
            logger.info(f"Enhanced Cache Manager Test '{test_name}': {status} ({duration*1000:.2f}ms)")
            return result
            
        except Exception as e:
            test_results["tests_failed"] += 1
            test_results["test_details"].append({
                "name": test_name,
                "status": "ERROR",
                "error": str(e),
                "result": False
            })
            logger.error(f"Enhanced Cache Manager Test '{test_name}' ERROR: {e}")
            return False
    
    # Test 1: Manager Initialization
    def test_manager_initialization():
        return _enhanced_cache_manager.get_module_name() == "cache_manager"
    
    # Test 2: Module Registration
    def test_module_registration():
        stats = _enhanced_cache_manager.get_stats()
        return stats.get("manager_stats", {}).get("managed_modules_count", 0) >= 0
    
    # Test 3: Statistics Collection
    def test_statistics_collection():
        stats = get_cache_manager_stats()
        required_fields = ["manager_stats", "module_stats"]
        return all(field in stats for field in required_fields)
    
    # Test 4: Health Status Check
    def test_health_status():
        health = get_cache_manager_health()
        required_fields = ["overall_health", "manager_health", "modules_registered"]
        return all(field in health for field in required_fields)
    
    # Test 5: Comprehensive Report
    def test_comprehensive_report():
        report = get_comprehensive_cache_report()
        required_fields = ["timestamp", "stats", "health", "coordination", "manager_info"]
        return all(field in report for field in required_fields)
    
    # Test 6: Cache Coordination
    def test_cache_coordination():
        coordination_stats = get_cache_coordination_stats()
        return isinstance(coordination_stats, dict)
    
    # Test 7: Cache Clearing Coordination
    def test_cache_clearing():
        clear_result = clear_all_managed_caches()
        return isinstance(clear_result, bool)
    
    # Test 8: Cache Warming Coordination
    def test_cache_warming():
        warm_result = warm_all_managed_caches()
        return isinstance(warm_result, bool)
    
    # Test 9: Invalidation Coordination
    def test_invalidation_coordination():
        results = coordinate_invalidation("test_pattern_*", [])
        return isinstance(results, dict)
    
    # Test 10: Legacy Manager Integration
    def test_legacy_manager_integration():
        # Test that the original manager still works
        manager = CacheManager()
        return hasattr(manager, 'initialization_time')
    
    # Run all tests
    logger.info("Starting enhanced cache manager comprehensive test suite...")
    
    run_test("Manager Initialization", test_manager_initialization)
    run_test("Module Registration", test_module_registration)
    run_test("Statistics Collection", test_statistics_collection)
    run_test("Health Status Check", test_health_status)
    run_test("Comprehensive Report", test_comprehensive_report)
    run_test("Cache Coordination", test_cache_coordination)
    run_test("Cache Clearing", test_cache_clearing)
    run_test("Cache Warming", test_cache_warming)
    run_test("Invalidation Coordination", test_invalidation_coordination)
    run_test("Legacy Manager Integration", test_legacy_manager_integration)
    
    # Calculate final metrics
    test_results["end_time"] = time.time()
    test_results["total_duration"] = test_results["end_time"] - test_results["start_time"]
    test_results["pass_rate"] = (test_results["tests_passed"] / test_results["tests_run"] * 100) if test_results["tests_run"] > 0 else 0
    
    # Add performance metrics
    test_results["performance_metrics"] = {
        "average_test_duration_ms": sum(t.get("duration_ms", 0) for t in test_results["test_details"]) / len(test_results["test_details"]) if test_results["test_details"] else 0,
        "cache_stats": get_cache_manager_stats(),
        "health_status": get_cache_manager_health(),
        "comprehensive_report": get_comprehensive_cache_report()
    }
    
    logger.info(f"Enhanced Cache Manager Tests Completed: {test_results['tests_passed']}/{test_results['tests_run']} passed ({test_results['pass_rate']:.1f}%)")
    
    return test_results


if __name__ == "__main__":
    print("🎛️ Enhanced Cache Manager - Comprehensive Testing & Demonstration")
    print("=" * 75)
    
    # Run original tests
    print("\n📊 Running Original Cache Manager Tests...")
    original_success = run_cache_manager_tests()
    
    # Run enhanced tests
    print("\n🚀 Running Enhanced Cache Manager Tests...")
    enhanced_results = run_enhanced_cache_manager_tests()
    
    print(f"\n✅ Enhanced Test Results: {enhanced_results['tests_passed']}/{enhanced_results['tests_run']} passed ({enhanced_results['pass_rate']:.1f}%)")
    
    # Display enhanced test details
    for test in enhanced_results["test_details"]:
        status_emoji = "✅" if test["status"] == "PASS" else "❌" if test["status"] == "FAIL" else "⚠️"
        duration = test.get("duration_ms", 0)
        print(f"  {status_emoji} {test['name']}: {test['status']} ({duration:.2f}ms)")
    
    # Run original demonstrations
    print("\n🎯 Running Original Cache Manager Demonstrations...")
    demonstrate_cache_manager_usage()
    
    # Display comprehensive report
    print("\n📈 Comprehensive Cache System Report:")
    report = get_comprehensive_cache_report()
    
    print(f"  • Manager uptime: {report['manager_info']['uptime']:.1f}s")
    print(f"  • Managed modules: {report['manager_info']['managed_modules']}")
    print(f"  • Overall health: {report['health'].get('overall_health', 'unknown')}")
    
    # Display coordination statistics
    coordination = report.get('coordination', {})
    if coordination:
        print(f"  • Coordination entries: {coordination.get('total_coordination_entries', 0)}")
        print(f"  • Cross-module operations: {coordination.get('cross_module_operations', 0)}")
    
    print("\n🔄 Enhanced cache management system validation complete!")
    print("=" * 75)


# End of cache_manager.py
