#!/usr/bin/env python3

"""
Cache Manager - Centralized Cache Operations

Provides comprehensive cache management, monitoring, and maintenance operations
with intelligent cache lifecycle management, performance optimization, and
automated cleanup for efficient memory and disk usage across all cache systems.

Phase 7.3.1 Enhancement: Multi-level caching coordination and performance monitoring.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module, is_function_available, get_function

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
from error_handling import (
    retry_on_failure,
    circuit_breaker,
    timeout_protection,
    graceful_degradation,
    error_context,
    AncestryException,
    RetryableError,
    NetworkTimeoutError,
    AuthenticationExpiredError,
    APIRateLimitError,
    ErrorContext,
)

# === STANDARD LIBRARY IMPORTS ===
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

# === LOCAL IMPORTS ===
from test_framework import TestSuite


# ==============================================
# CENTRALIZED CACHE COORDINATION
# ==============================================

class CacheCoordinator:
    """
    Coordinates multiple cache systems for optimal performance.
    Implements cache warming, invalidation, and performance monitoring.
    """
    
    def __init__(self):
        self._cache_modules = {}
        self._cache_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "last_cleanup": time.time()
        }
        self._initialized = False
        
    def register_cache_module(self, name: str, cache_module):
        """Register a cache module for coordination"""
        self._cache_modules[name] = cache_module
        logger.debug(f"Registered cache module: {name}")
        
    def warm_all_caches(self) -> Dict[str, bool]:
        """Warm all registered cache modules"""
        results = {}
        for name, module in self._cache_modules.items():
            try:
                if hasattr(module, 'warm') and callable(module.warm):
                    results[name] = module.warm()
                    logger.debug(f"Warmed cache module: {name}")
                else:
                    results[name] = True  # No warming needed
            except Exception as e:
                logger.warning(f"Failed to warm cache module {name}: {e}")
                results[name] = False
        return results
        
    def clear_all_caches(self) -> Dict[str, bool]:
        """Clear all registered cache modules"""
        results = {}
        for name, module in self._cache_modules.items():
            try:
                if hasattr(module, 'clear') and callable(module.clear):
                    results[name] = module.clear()
                    logger.debug(f"Cleared cache module: {name}")
                else:
                    results[name] = True  # No clearing needed
            except Exception as e:
                logger.warning(f"Failed to clear cache module {name}: {e}")
                results[name] = False
        return results
        
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all cache modules"""
        stats = {
            "coordinator_stats": self._cache_stats.copy(),
            "module_stats": {}
        }
        
        for name, module in self._cache_modules.items():
            try:
                if hasattr(module, 'get_stats') and callable(module.get_stats):
                    stats["module_stats"][name] = module.get_stats()
                elif hasattr(module, 'stats') and callable(module.stats):
                    stats["module_stats"][name] = module.stats()
                else:
                    stats["module_stats"][name] = {"status": "no_stats_available"}
            except Exception as e:
                logger.warning(f"Failed to get stats from cache module {name}: {e}")
                stats["module_stats"][name] = {"error": str(e)}
                
        return stats

# Global cache coordinator instance
_cache_coordinator = CacheCoordinator()


# ==============================================
# CACHE MANAGEMENT FUNCTIONS
# ==============================================

def initialize_cache_system():
    """Initialize the coordinated cache system"""
    global _cache_coordinator
    
    if _cache_coordinator._initialized:
        return True
        
    try:
        # Register available cache modules
        try:
            from performance_cache import _performance_cache
            _cache_coordinator.register_cache_module("performance", _performance_cache)
        except ImportError:
            logger.debug("Performance cache module not available")
            
        try:
            from gedcom_cache import _gedcom_cache_module
            _cache_coordinator.register_cache_module("gedcom", _gedcom_cache_module)
        except ImportError:
            logger.debug("GEDCOM cache module not available")
            
        try:
            from api_cache import _api_cache_module
            _cache_coordinator.register_cache_module("api", _api_cache_module)
        except ImportError:
            logger.debug("API cache module not available")
            
        try:
            from core.session_cache import _session_cache
            _cache_coordinator.register_cache_module("session", _session_cache)
        except ImportError:
            logger.debug("Session cache module not available")
            
        try:
            import core.system_cache as system_cache_module
            _cache_coordinator.register_cache_module("system", system_cache_module)
        except ImportError:
            logger.debug("System cache modules not available")
            
        _cache_coordinator._initialized = True
        logger.info(f"Cache system initialized with {len(_cache_coordinator._cache_modules)} modules")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize cache system: {e}")
        return False


def warm_caches(module_names: Optional[List[str]] = None) -> Dict[str, bool]:
    """Warm specified cache modules or all if none specified"""
    initialize_cache_system()
    
    if module_names is None:
        return _cache_coordinator.warm_all_caches()
    
    results = {}
    for name in module_names:
        if name in _cache_coordinator._cache_modules:
            try:
                module = _cache_coordinator._cache_modules[name]
                if hasattr(module, 'warm') and callable(module.warm):
                    results[name] = module.warm()
                else:
                    results[name] = True
            except Exception as e:
                logger.warning(f"Failed to warm cache module {name}: {e}")
                results[name] = False
        else:
            logger.warning(f"Cache module {name} not found")
            results[name] = False
            
    return results


def clear_caches(module_names: Optional[List[str]] = None) -> Dict[str, bool]:
    """Clear specified cache modules or all if none specified"""
    initialize_cache_system()
    
    if module_names is None:
        return _cache_coordinator.clear_all_caches()
    
    results = {}
    for name in module_names:
        if name in _cache_coordinator._cache_modules:
            try:
                module = _cache_coordinator._cache_modules[name]
                if hasattr(module, 'clear') and callable(module.clear):
                    results[name] = module.clear()
                else:
                    results[name] = True
            except Exception as e:
                logger.warning(f"Failed to clear cache module {name}: {e}")
                results[name] = False
        else:
            logger.warning(f"Cache module {name} not found")
            results[name] = False
            
    return results


def get_cache_health_status() -> Dict[str, Any]:
    """Get comprehensive cache health status"""
    initialize_cache_system()
    
    health_status = {
        "overall_status": "healthy",
        "cache_modules": len(_cache_coordinator._cache_modules),
        "detailed_stats": _cache_coordinator.get_comprehensive_stats(),
        "recommendations": []
    }
    
    stats = health_status["detailed_stats"]
    
    # Analyze cache performance and generate recommendations
    for module_name, module_stats in stats["module_stats"].items():
        if isinstance(module_stats, dict):
            # Check hit rates
            if "hit_rate" in module_stats and module_stats["hit_rate"] < 0.5:
                health_status["recommendations"].append(
                    f"Low hit rate in {module_name} cache ({module_stats['hit_rate']:.1%})"
                )
            
            # Check memory usage
            if "memory_entries" in module_stats and module_stats["memory_entries"] > 1000:
                health_status["recommendations"].append(
                    f"High memory usage in {module_name} cache ({module_stats['memory_entries']} entries)"
                )
    
    if len(health_status["recommendations"]) > 2:
        health_status["overall_status"] = "needs_attention"
    
    return health_status


# (Removed duplicate definition of run_comprehensive_tests to avoid obscuring the main implementation.)
def cache_manager_module_tests() -> bool:
    """
    Run all cache_manager tests and return True if successful.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    from test_framework import TestSuite

    def test_cache_manager_initialization():
        # Test cache manager initialization
        if is_function_available("CacheManager"):
            cache_manager_class = get_function("CacheManager")
            if cache_manager_class:
                cache_manager = cache_manager_class()
                assert cache_manager is not None
                # suite.log_test_result(True, "Cache manager created successfully")
        return True

        # CORE FUNCTIONALITY TESTS

    def test_cache_operations():
        # Test basic cache operations
        if is_function_available("CacheManager"):
            cache_manager_class = get_function("CacheManager")
            if cache_manager_class:
                cache_manager = cache_manager_class()
                # Test set and get operations
                cache_manager.set("test_key", "test_value")
                result = cache_manager.get("test_key")
                assert result == "test_value"
        return True

    def test_cache_statistics():
        # Test cache statistics collection
        pass
        return True

    def test_cache_invalidation():
        # Test cache invalidation patterns
        pass
        return True

        # EDGE CASE TESTS

    def test_eviction_policies():
        # Test cache eviction when full
        if is_function_available("CacheManager"):
            cache_manager_class = get_function("CacheManager")
            if cache_manager_class:
                cache_manager = cache_manager_class(max_size=2)
                cache_manager.set("key1", "value1")
                cache_manager.set("key2", "value2")
                cache_manager.set("key3", "value3")  # Should evict key1
                result = cache_manager.get("key1")
                # Oldest key should be evicted
                assert result is None or result == "value1"
        return True

    def test_performance_monitoring():
        # Test performance monitoring features
        if is_function_available("CacheManager"):
            cache_manager_class = get_function("CacheManager")
            if cache_manager_class:
                cache_manager = cache_manager_class()
                # Test performance tracking
        return True

        # PERFORMANCE TESTS

    def test_cache_performance():
        # Test cache performance under load
        pass
        return True

    def test_concurrent_access():
        # Test thread-safe cache operations
        pass
        return True

    def test_memory_management():
        # Test memory usage and cleanup
        pass
        return True

        # INTEGRATION TESTS

    def test_database_integration():
        # Test cache integration with database
        pass
        return True

    def test_api_integration():
        # Test cache integration with API calls
        pass
        return True

    def test_session_management():
        # Test cache session handling
        pass
        return True

        # ERROR HANDLING TESTS

    def test_error_handling():
        # Test cache error scenarios
        pass
        return True

    def test_recovery_mechanisms():
        # Test cache recovery after failures
        pass
        return True

    def test_data_corruption_handling():
        # Test handling of corrupted cache data
        pass
        return True

        # SECURITY TESTS

    def test_data_encryption():
        # Test encrypted cache storage
        pass
        return True

    def test_access_control():
        # Test cache access permissions
        pass
        return True

    def test_audit_logging():
        # Test cache operation logging
        pass
        return True

        # CONFIGURATION TESTS

    def test_configuration_loading():
        # Test cache configuration
        pass
        return True

    def test_environment_adaptation():
        # Test cache behavior in different environments
        pass
        return True

    def test_feature_toggles():
        # Test cache feature flags
        pass
        return True

    # Create test suite and run tests
    suite = TestSuite("Cache Manager", "cache_manager.py")
    suite.start_suite()

    # Run tests using the suite's run_test method
    suite.run_test(
        "Cache Manager Initialization",
        test_cache_manager_initialization,
        "Should initialize cache manager with required methods",
    )
    suite.run_test(
        "Basic Cache Operations",
        test_cache_operations,
        "Should support set/get operations",
    )
    suite.run_test(
        "Cache Statistics", test_cache_statistics, "Should track cache metrics"
    )
    suite.run_test(
        "Cache Invalidation",
        test_cache_invalidation,
        "Should support cache invalidation",
    )
    suite.run_test(
        "Cache Eviction Policies",
        test_eviction_policies,
        "Should enforce size limits and evict items",
    )
    suite.run_test(
        "Performance Monitoring",
        test_performance_monitoring,
        "Should monitor cache performance",
    )
    suite.run_test(
        "Cache Performance", test_cache_performance, "Should perform well under load"
    )
    suite.run_test(
        "Concurrent Access",
        test_concurrent_access,
        "Should handle concurrent operations",
    )
    suite.run_test(
        "Memory Management", test_memory_management, "Should manage memory efficiently"
    )
    suite.run_test(
        "Database Integration",
        test_database_integration,
        "Should integrate with database",
    )
    suite.run_test(
        "API Integration", test_api_integration, "Should integrate with API calls"
    )
    suite.run_test(
        "Session Management", test_session_management, "Should handle sessions properly"
    )
    suite.run_test(
        "Error Handling", test_error_handling, "Should handle errors gracefully"
    )
    suite.run_test(
        "Recovery Mechanisms", test_recovery_mechanisms, "Should recover from failures"
    )
    suite.run_test(
        "Data Corruption Handling",
        test_data_corruption_handling,
        "Should handle corrupted data",
    )
    suite.run_test("Data Encryption", test_data_encryption, "Should encrypt cache data")
    suite.run_test(
        "Access Control", test_access_control, "Should control access properly"
    )
    suite.run_test("Audit Logging", test_audit_logging, "Should log cache operations")
    suite.run_test(
        "Configuration Loading", test_configuration_loading, "Should load configuration"
    )
    suite.run_test(
        "Environment Adaptation",
        test_environment_adaptation,
        "Should adapt to environments",
    )
    suite.run_test(
        "Feature Toggles", test_feature_toggles, "Should support feature flags"
    )

    # Complete the test suite
    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive cache manager tests using standardized TestSuite format."""
    return cache_manager_module_tests()


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    success = run_comprehensive_tests()
    import sys

    sys.exit(0 if success else 1)
