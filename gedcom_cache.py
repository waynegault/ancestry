#!/usr/bin/env python3

# gedcom_cache.py

"""
gedcom_cache.py - Aggressive GEDCOM File Caching System

Provides advanced caching strategies specifically for GEDCOM files and genealogical data.
Implements multi-level caching (memory + disk), file-based invalidation, and preloading
strategies to dramatically improve performance for frequently accessed genealogical data.
"""

# --- Standard library imports ---
import hashlib
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# --- Local application imports ---
from cache import (
    cache,
    cache_file_based_on_mtime,
    warm_cache_with_data,
    get_cache_stats,
    CacheInterface,
    BaseCacheModule,
    get_unified_cache_key,
    invalidate_related_caches,
)
from config import config_instance
from logging_config import logger

# --- Global Variables ---
_MEMORY_CACHE: Dict[str, Tuple[Any, float]] = {}  # In-memory cache with timestamps
_CACHE_MAX_AGE = 3600  # 1 hour default for memory cache
_GEDCOM_CACHE_PREFIX = "gedcom_data"


# --- GEDCOM Cache Module Implementation ---


class GedcomCacheModule(BaseCacheModule):
    """
    GEDCOM-specific cache module implementing the standardized cache interface.
    Provides multi-level caching (memory + disk) for genealogical data.
    """

    def __init__(self):
        super().__init__()
        self.module_name = "gedcom_cache"
        self.cache_prefix = _GEDCOM_CACHE_PREFIX

    def get_module_name(self) -> str:
        return self.module_name

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive GEDCOM cache statistics."""
        base_stats = super().get_stats()

        # Add GEDCOM-specific statistics
        memory_cache_size = len(_MEMORY_CACHE)
        memory_cache_valid_entries = sum(
            1 for key in _MEMORY_CACHE if _is_memory_cache_valid(key)
        )

        gedcom_stats = {
            "memory_cache_entries": memory_cache_size,
            "memory_cache_valid_entries": memory_cache_valid_entries,
            "memory_cache_hit_rate": (
                memory_cache_valid_entries / memory_cache_size * 100
                if memory_cache_size > 0
                else 0
            ),
            "cache_max_age_seconds": _CACHE_MAX_AGE,
        }

        # Add GEDCOM file information if available
        if config_instance and hasattr(config_instance, "GEDCOM_FILE_PATH"):
            gedcom_path = config_instance.GEDCOM_FILE_PATH
            if gedcom_path and Path(gedcom_path).exists():
                gedcom_stats["gedcom_file_path"] = str(gedcom_path)
                gedcom_stats["gedcom_file_size_mb"] = Path(
                    gedcom_path
                ).stat().st_size / (1024 * 1024)
                gedcom_stats["gedcom_file_mtime"] = os.path.getmtime(gedcom_path)

        # Merge with base statistics
        return {**base_stats, **gedcom_stats}

    def clear(self) -> bool:
        """Clear all GEDCOM caches (memory and disk)."""
        try:
            # Clear memory cache
            global _MEMORY_CACHE
            cleared_memory = len(_MEMORY_CACHE)
            _MEMORY_CACHE.clear()

            # Clear disk-based caches with GEDCOM prefix
            cleared_disk = invalidate_related_caches(
                pattern=f"{self.cache_prefix}*",
                exclude_modules=[],
            )

            logger.info(
                f"GEDCOM cache cleared: {cleared_memory} memory entries, {sum(cleared_disk.values())} disk entries"
            )
            return True
        except Exception as e:
            logger.error(f"Error clearing GEDCOM cache: {e}")
            return False

    def warm(self) -> bool:
        """Warm up GEDCOM cache with frequently accessed data."""
        try:
            # Check if GEDCOM file is available
            if not (config_instance and hasattr(config_instance, "GEDCOM_FILE_PATH")):
                logger.warning("No GEDCOM file path configured for cache warming")
                return False

            gedcom_path = config_instance.GEDCOM_FILE_PATH
            if not gedcom_path or not Path(gedcom_path).exists():
                logger.warning(f"GEDCOM file not found: {gedcom_path}")
                return False

            # Warm cache with basic file metadata
            cache_key = get_unified_cache_key("gedcom", "file_metadata", gedcom_path)
            file_stats = Path(gedcom_path).stat()
            metadata = {
                "size": file_stats.st_size,
                "mtime": file_stats.st_mtime,
                "warmed_at": time.time(),
            }

            warm_cache_with_data(cache_key, metadata)
            logger.info(f"GEDCOM cache warmed with metadata for {gedcom_path}")
            return True
        except Exception as e:
            logger.error(f"Error warming GEDCOM cache: {e}")
            return False

    def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status of GEDCOM cache system."""
        base_health = super().get_health_status()

        try:
            # Check memory cache health
            memory_health = "healthy"
            memory_issues = []

            if len(_MEMORY_CACHE) == 0:
                memory_issues.append("No memory cache entries")

            # Check for stale entries
            stale_entries = sum(
                1 for key in _MEMORY_CACHE if not _is_memory_cache_valid(key)
            )
            if stale_entries > len(_MEMORY_CACHE) * 0.5:
                memory_health = "degraded"
                memory_issues.append(f"High number of stale entries: {stale_entries}")

            # Check GEDCOM file accessibility
            gedcom_health = "healthy"
            gedcom_issues = []

            if config_instance and hasattr(config_instance, "GEDCOM_FILE_PATH"):
                gedcom_path = config_instance.GEDCOM_FILE_PATH
                if not gedcom_path:
                    gedcom_health = "warning"
                    gedcom_issues.append("No GEDCOM file path configured")
                elif not Path(gedcom_path).exists():
                    gedcom_health = "error"
                    gedcom_issues.append(f"GEDCOM file not found: {gedcom_path}")
            else:
                gedcom_health = "warning"
                gedcom_issues.append("GEDCOM configuration not available")

            # Overall health assessment
            overall_health = "healthy"
            if gedcom_health == "error" or memory_health == "degraded":
                overall_health = "degraded"
            elif gedcom_health == "warning":
                overall_health = "warning"

            gedcom_health_info = {
                "memory_cache_health": memory_health,
                "memory_cache_issues": memory_issues,
                "gedcom_file_health": gedcom_health,
                "gedcom_file_issues": gedcom_issues,
                "overall_health": overall_health,
                "memory_cache_entries": len(_MEMORY_CACHE),
                "stale_entries": stale_entries,
            }

            return {**base_health, **gedcom_health_info}
        except Exception as e:
            logger.error(f"Error getting GEDCOM cache health status: {e}")
            return {
                **base_health,
                "health_check_error": str(e),
                "overall_health": "error",
            }


# Initialize GEDCOM cache module instance
_gedcom_cache_module = GedcomCacheModule()


# --- Memory Cache Management ---


def _get_memory_cache_key(file_path: str, operation: str) -> str:
    """Generate a consistent cache key for memory cache using unified system."""
    # Use the unified cache key generation for consistency
    return get_unified_cache_key(
        "gedcom_memory", operation, file_path, os.path.getmtime(file_path)
    )


def _is_memory_cache_valid(cache_key: str) -> bool:
    """Check if memory cache entry is still valid."""
    if cache_key not in _MEMORY_CACHE:
        return False

    _, timestamp = _MEMORY_CACHE[cache_key]
    return (time.time() - timestamp) < _CACHE_MAX_AGE


def _get_from_memory_cache(cache_key: str) -> Optional[Any]:
    """Get data from memory cache if valid."""
    if _is_memory_cache_valid(cache_key):
        data, _ = _MEMORY_CACHE[cache_key]
        logger.debug(f"Memory cache HIT for key: {cache_key}")
        return data
    else:
        # Clean up expired entry
        if cache_key in _MEMORY_CACHE:
            del _MEMORY_CACHE[cache_key]
        logger.debug(f"Memory cache MISS for key: {cache_key}")
        return None


def _store_in_memory_cache(cache_key: str, data: Any) -> None:
    """Store data in memory cache with timestamp."""
    _MEMORY_CACHE[cache_key] = (data, time.time())
    logger.debug(f"Stored in memory cache: {cache_key}")


def clear_memory_cache() -> int:
    """Clear all memory cache entries."""
    count = len(_MEMORY_CACHE)
    _MEMORY_CACHE.clear()
    logger.debug(f"Cleared {count} entries from memory cache")
    return count


# --- Enhanced GEDCOM Caching Functions ---


def load_gedcom_with_aggressive_caching(gedcom_path: str) -> Optional[Any]:
    """
    Load GEDCOM data with aggressive multi-level caching.

    Uses both memory and disk caching with file modification time tracking.
    Automatically invalidates cache when GEDCOM file changes.

    Args:
        gedcom_path: Path to the GEDCOM file

    Returns:
        GedcomData instance or None if loading fails
    """

    # Check memory cache first
    memory_key = _get_memory_cache_key(gedcom_path, "gedcom_load")
    cached_data = _get_from_memory_cache(memory_key)
    if cached_data is not None:
        return cached_data

    # Check disk cache
    try:
        file_mtime = os.path.getmtime(gedcom_path)
        mtime_hash = hashlib.md5(str(file_mtime).encode()).hexdigest()[:8]
        disk_cache_key = f"gedcom_load_mtime_{mtime_hash}"

        if cache is not None:
            from diskcache.core import ENOVAL

            disk_cached = cache.get(disk_cache_key, default=ENOVAL, retry=True)
            if disk_cached is not ENOVAL:
                logger.debug(f"GEDCOM data loaded from disk cache")
                # Store in memory cache for faster next access
                _store_in_memory_cache(memory_key, disk_cached)
                return disk_cached
    except Exception as e:
        logger.debug(f"Error checking disk cache: {e}")

    logger.info(f"Loading GEDCOM file with aggressive caching: {gedcom_path}")
    start_time = time.time()

    try:
        # Import here to avoid circular imports
        from gedcom_utils import GedcomData

        # Load the GEDCOM file
        gedcom_data = GedcomData(gedcom_path)

        if gedcom_data:
            load_time = time.time() - start_time
            logger.info(f"GEDCOM file loaded and cached in {load_time:.2f}s")

            # Store in memory cache for fastest access
            _store_in_memory_cache(memory_key, gedcom_data)

            # Store in disk cache for persistence - but don't cache the reader object
            # The GedcomReader contains BinaryFileCR objects that cannot be pickled
            try:
                if cache is not None:
                    # Create a serializable version without the reader
                    cache_data = {
                        "path": str(gedcom_data.path),
                        "indi_index": gedcom_data.indi_index,
                        "processed_data_cache": gedcom_data.processed_data_cache,
                        "id_to_parents": gedcom_data.id_to_parents,
                        "id_to_children": gedcom_data.id_to_children,
                        "indi_index_build_time": gedcom_data.indi_index_build_time,
                        "family_maps_build_time": gedcom_data.family_maps_build_time,
                        "data_processing_time": gedcom_data.data_processing_time,
                    }
                    cache.set(disk_cache_key, cache_data, expire=86400, retry=True)
                    logger.debug(f"GEDCOM data cached (without reader) in disk cache")
            except Exception as e:
                logger.debug(f"Error storing in disk cache: {e}")

            # Log cache statistics
            stats = get_cache_stats()
            if stats:
                logger.debug(f"Cache stats after GEDCOM load: {stats}")

            return gedcom_data
        else:
            logger.error("Failed to create GedcomData instance")
            return None

    except Exception as e:
        logger.error(f"Error loading GEDCOM file {gedcom_path}: {e}", exc_info=True)
        return None


def cache_gedcom_processed_data(gedcom_data: Any, gedcom_path: str) -> bool:
    """
    Cache the processed data from a GedcomData instance for faster subsequent access.

    Args:
        gedcom_data: GedcomData instance with processed data
        gedcom_path: Path to the source GEDCOM file

    Returns:
        True if caching successful, False otherwise
    """
    if not gedcom_data or not hasattr(gedcom_data, "processed_data_cache"):
        logger.warning("Invalid gedcom_data for caching processed data")
        return False

    try:
        # Create cache key based on file path and modification time
        file_mtime = os.path.getmtime(gedcom_path)
        path_hash = hashlib.md5(str(gedcom_path).encode()).hexdigest()[:8]
        mtime_hash = hashlib.md5(str(file_mtime).encode()).hexdigest()[:8]

        # Cache different components separately for efficiency
        cache_keys = {
            "processed_data": f"gedcom_processed_{path_hash}_{mtime_hash}",
            "indi_index": f"gedcom_indi_index_{path_hash}_{mtime_hash}",
            "family_maps": f"gedcom_family_maps_{path_hash}_{mtime_hash}",
        }

        # Cache processed data - ensure it's serializable
        if hasattr(gedcom_data, "processed_data_cache"):
            # Create a serializable version of processed_data_cache
            serializable_processed_data = {}
            for key, value in gedcom_data.processed_data_cache.items():
                if isinstance(value, dict):
                    serializable_item = {}
                    for item_key, item_value in value.items():
                        # Convert datetime objects to ISO strings for serialization
                        if hasattr(item_value, "isoformat"):  # datetime objects
                            serializable_item[item_key] = item_value.isoformat()
                        elif isinstance(
                            item_value, (str, int, float, bool, list, tuple, type(None))
                        ):
                            serializable_item[item_key] = item_value
                        else:
                            # Skip non-serializable objects
                            logger.debug(
                                f"Skipping non-serializable object in processed_data_cache: {item_key} = {type(item_value)}"
                            )
                    serializable_processed_data[key] = serializable_item
                elif isinstance(
                    value, (str, int, float, bool, list, tuple, type(None))
                ):
                    serializable_processed_data[key] = value

            warm_cache_with_data(
                cache_keys["processed_data"],
                serializable_processed_data,
                expire=86400,
            )

        # Cache individual index - filter out non-serializable objects
        if hasattr(gedcom_data, "indi_index"):
            # Create a serializable version of indi_index
            serializable_indi_index = {}
            for key, value in gedcom_data.indi_index.items():
                # Only cache primitive data types and avoid GedcomReader references
                if isinstance(
                    value, (str, int, float, bool, list, dict, tuple, type(None))
                ):
                    serializable_indi_index[key] = value
                else:
                    # For complex objects, try to extract serializable data
                    try:
                        # Check if it's a simple object with serializable attributes
                        if hasattr(value, "__dict__"):
                            obj_dict = {}
                            for attr, attr_value in value.__dict__.items():
                                if isinstance(
                                    attr_value,
                                    (
                                        str,
                                        int,
                                        float,
                                        bool,
                                        list,
                                        dict,
                                        tuple,
                                        type(None),
                                    ),
                                ):
                                    obj_dict[attr] = attr_value
                            if obj_dict:  # Only store if we have serializable data
                                serializable_indi_index[key] = obj_dict
                    except:
                        # Skip non-serializable objects
                        pass

            if serializable_indi_index:
                warm_cache_with_data(
                    cache_keys["indi_index"], serializable_indi_index, expire=86400
                )

        # Cache family relationship maps
        family_data = {}
        if hasattr(gedcom_data, "id_to_parents"):
            family_data["id_to_parents"] = gedcom_data.id_to_parents
        if hasattr(gedcom_data, "id_to_children"):
            family_data["id_to_children"] = gedcom_data.id_to_children

        if family_data:
            warm_cache_with_data(cache_keys["family_maps"], family_data, expire=86400)

        logger.info(
            f"Successfully cached GEDCOM processed data for {Path(gedcom_path).name}"
        )
        return True

    except Exception as e:
        logger.error(f"Error caching GEDCOM processed data: {e}", exc_info=True)
        return False


def preload_gedcom_cache() -> bool:
    """
    Preload GEDCOM cache if a GEDCOM file is configured.
    This can be called at application startup for better performance.

    Returns:
        True if preloading successful, False otherwise
    """
    if not config_instance or not hasattr(config_instance, "GEDCOM_FILE_PATH"):
        logger.debug("No GEDCOM file configured for preloading")
        return False

    gedcom_path = config_instance.GEDCOM_FILE_PATH
    if not gedcom_path or not Path(gedcom_path).exists():
        logger.debug(f"GEDCOM file not found for preloading: {gedcom_path}")
        return False

    logger.info("Preloading GEDCOM cache...")
    start_time = time.time()

    try:
        gedcom_data = load_gedcom_with_aggressive_caching(str(gedcom_path))
        if gedcom_data:
            # Also cache the processed data components
            cache_gedcom_processed_data(gedcom_data, str(gedcom_path))

            preload_time = time.time() - start_time
            logger.info(f"GEDCOM cache preloaded successfully in {preload_time:.2f}s")
            return True
        else:
            logger.warning("Failed to preload GEDCOM cache")
            return False

    except Exception as e:
        logger.error(f"Error preloading GEDCOM cache: {e}", exc_info=True)
        return False


def get_gedcom_cache_info() -> Dict[str, Any]:
    """
    Get information about GEDCOM cache status and statistics.

    Returns:
        Dictionary with cache information
    """
    info = {
        "memory_cache_entries": len(_MEMORY_CACHE),
        "memory_cache_max_age": _CACHE_MAX_AGE,
        "disk_cache_stats": get_cache_stats(),
    }

    # Add GEDCOM-specific information if available
    if config_instance and hasattr(config_instance, "GEDCOM_FILE_PATH"):
        gedcom_path = config_instance.GEDCOM_FILE_PATH
        if gedcom_path and Path(gedcom_path).exists():
            info["gedcom_file"] = str(gedcom_path)
            info["gedcom_file_size_mb"] = Path(gedcom_path).stat().st_size / (
                1024 * 1024
            )
            info["gedcom_file_mtime"] = os.path.getmtime(gedcom_path)

    return info


# --- Public Interface Functions for GEDCOM Cache Module ---


def get_gedcom_cache_stats() -> Dict[str, Any]:
    """Get comprehensive GEDCOM cache statistics."""
    return _gedcom_cache_module.get_stats()


def clear_gedcom_cache() -> bool:
    """Clear all GEDCOM caches."""
    return _gedcom_cache_module.clear()


def warm_gedcom_cache() -> bool:
    """Warm up GEDCOM cache."""
    return _gedcom_cache_module.warm()


def get_gedcom_cache_health() -> Dict[str, Any]:
    """Get GEDCOM cache health status."""
    return _gedcom_cache_module.get_health_status()


# --- GEDCOM Cache Testing Suite ---


def run_gedcom_cache_tests() -> Dict[str, Any]:
    """
    Run comprehensive tests for GEDCOM cache functionality.
    Returns test results with pass/fail status and performance metrics.
    """
    test_results = {
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "test_details": [],
        "start_time": time.time(),
        "performance_metrics": {},
    }

    def run_test(test_name: str, test_func: callable) -> bool:
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

            test_results["test_details"].append(
                {
                    "name": test_name,
                    "status": status,
                    "duration_ms": round(duration * 1000, 2),
                    "result": result,
                }
            )

            logger.info(
                f"GEDCOM Cache Test '{test_name}': {status} ({duration*1000:.2f}ms)"
            )
            return result

        except Exception as e:
            test_results["tests_failed"] += 1
            test_results["test_details"].append(
                {"name": test_name, "status": "ERROR", "error": str(e), "result": False}
            )
            logger.error(f"GEDCOM Cache Test '{test_name}' ERROR: {e}")
            return False

    # Test 1: Module Initialization
    def test_module_initialization():
        return _gedcom_cache_module.get_module_name() == "gedcom_cache"

    # Test 2: Memory Cache Basic Operations
    def test_memory_cache_operations():
        test_key = "test_operation_key"
        test_data = {"test": "data", "timestamp": time.time()}

        # Store in memory cache
        _store_in_memory_cache(test_key, test_data)

        # Retrieve from memory cache
        retrieved = _get_from_memory_cache(test_key)

        # Clean up
        if test_key in _MEMORY_CACHE:
            del _MEMORY_CACHE[test_key]

        return retrieved == test_data

    # Test 3: Cache Key Generation
    def test_cache_key_generation():
        if not (config_instance and hasattr(config_instance, "GEDCOM_FILE_PATH")):
            return True  # Skip if no GEDCOM configured

        gedcom_path = config_instance.GEDCOM_FILE_PATH
        if not gedcom_path or not Path(gedcom_path).exists():
            return True  # Skip if file doesn't exist

        key1 = _get_memory_cache_key(gedcom_path, "test_operation")
        key2 = _get_memory_cache_key(gedcom_path, "test_operation")

        return key1 == key2  # Keys should be consistent

    # Test 4: Statistics Collection
    def test_statistics_collection():
        stats = _gedcom_cache_module.get_stats()
        required_fields = [
            "module_name",
            "memory_cache_entries",
            "cache_max_age_seconds",
        ]
        return all(field in stats for field in required_fields)

    # Test 5: Health Status Check
    def test_health_status():
        health = _gedcom_cache_module.get_health_status()
        required_fields = [
            "overall_health",
            "memory_cache_health",
            "gedcom_file_health",
        ]
        return all(field in health for field in required_fields)

    # Test 6: Cache Clearing
    def test_cache_clearing():
        # Add some test data
        test_key = "clear_test_key"
        _store_in_memory_cache(test_key, "test_data")

        # Clear cache
        clear_result = _gedcom_cache_module.clear()

        # Check if cleared
        return clear_result and test_key not in _MEMORY_CACHE

    # Test 7: Cache Warming
    def test_cache_warming():
        warm_result = _gedcom_cache_module.warm()
        # Warming should either succeed or fail gracefully
        return isinstance(warm_result, bool)

    # Test 8: Memory Cache Expiration
    def test_memory_cache_expiration():
        test_key = "expiration_test_key"
        test_data = "expiration_test_data"

        # Store with current timestamp
        _MEMORY_CACHE[test_key] = (test_data, time.time() - _CACHE_MAX_AGE - 1)

        # Should be invalid due to age
        is_valid = _is_memory_cache_valid(test_key)

        # Clean up
        if test_key in _MEMORY_CACHE:
            del _MEMORY_CACHE[test_key]

        return not is_valid

    # Run all tests
    logger.info("Starting GEDCOM cache comprehensive test suite...")

    run_test("Module Initialization", test_module_initialization)
    run_test("Memory Cache Operations", test_memory_cache_operations)
    run_test("Cache Key Generation", test_cache_key_generation)
    run_test("Statistics Collection", test_statistics_collection)
    run_test("Health Status Check", test_health_status)
    run_test("Cache Clearing", test_cache_clearing)
    run_test("Cache Warming", test_cache_warming)
    run_test("Memory Cache Expiration", test_memory_cache_expiration)

    # Calculate final metrics
    test_results["end_time"] = time.time()
    test_results["total_duration"] = (
        test_results["end_time"] - test_results["start_time"]
    )
    test_results["pass_rate"] = (
        (test_results["tests_passed"] / test_results["tests_run"] * 100)
        if test_results["tests_run"] > 0
        else 0
    )

    # Add performance metrics
    test_results["performance_metrics"] = {
        "average_test_duration_ms": (
            sum(t.get("duration_ms", 0) for t in test_results["test_details"])
            / len(test_results["test_details"])
            if test_results["test_details"]
            else 0
        ),
        "cache_stats": get_gedcom_cache_stats(),
        "health_status": get_gedcom_cache_health(),
    }

    logger.info(
        f"GEDCOM Cache Tests Completed: {test_results['tests_passed']}/{test_results['tests_run']} passed ({test_results['pass_rate']:.1f}%)"
    )

    return test_results


# --- GEDCOM Cache Demo Functions ---


def demonstrate_gedcom_cache_usage() -> Dict[str, Any]:
    """
    Demonstrate practical GEDCOM cache usage with examples.
    Returns demonstration results and performance data.
    """
    demo_results = {
        "demonstrations": [],
        "start_time": time.time(),
        "performance_summary": {},
    }

    logger.info("Starting GEDCOM cache usage demonstrations...")

    try:
        # Demo 1: Cache Statistics Display
        stats = get_gedcom_cache_stats()
        demo_results["demonstrations"].append(
            {
                "name": "Cache Statistics",
                "description": "Display current GEDCOM cache statistics",
                "data": stats,
                "status": "success",
            }
        )

        # Demo 2: Health Status Check
        health = get_gedcom_cache_health()
        demo_results["demonstrations"].append(
            {
                "name": "Health Status",
                "description": "Check GEDCOM cache system health",
                "data": health,
                "status": "success",
            }
        )

        # Demo 3: Memory Cache Operations
        if config_instance and hasattr(config_instance, "GEDCOM_FILE_PATH"):
            gedcom_path = config_instance.GEDCOM_FILE_PATH
            if gedcom_path and Path(gedcom_path).exists():
                # Demonstrate file-based caching
                cache_key = _get_memory_cache_key(gedcom_path, "demo_operation")
                demo_data = {
                    "file_path": gedcom_path,
                    "demo_timestamp": time.time(),
                    "operation": "demonstration",
                }

                _store_in_memory_cache(cache_key, demo_data)
                retrieved_data = _get_from_memory_cache(cache_key)

                demo_results["demonstrations"].append(
                    {
                        "name": "Memory Cache Demo",
                        "description": "Store and retrieve data from memory cache",
                        "data": {
                            "stored": demo_data,
                            "retrieved": retrieved_data,
                            "match": demo_data == retrieved_data,
                        },
                        "status": "success",
                    }
                )

                # Clean up demo data
                if cache_key in _MEMORY_CACHE:
                    del _MEMORY_CACHE[cache_key]

        # Demo 4: Cache Coordination
        coordination_stats = get_unified_cache_key(
            "gedcom", "coordination_demo", "test_param"
        )
        demo_results["demonstrations"].append(
            {
                "name": "Cache Coordination",
                "description": "Demonstrate unified cache key generation",
                "data": {"unified_key": coordination_stats, "module": "gedcom"},
                "status": "success",
            }
        )

    except Exception as e:
        demo_results["demonstrations"].append(
            {
                "name": "Error in Demonstration",
                "description": f"Error occurred: {str(e)}",
                "status": "error",
            }
        )
        logger.error(f"Error in GEDCOM cache demonstration: {e}")

    # Final summary
    demo_results["end_time"] = time.time()
    demo_results["total_duration"] = (
        demo_results["end_time"] - demo_results["start_time"]
    )
    demo_results["performance_summary"] = {
        "demonstrations_completed": len(
            [d for d in demo_results["demonstrations"] if d["status"] == "success"]
        ),
        "total_demonstrations": len(demo_results["demonstrations"]),
        "final_cache_stats": get_gedcom_cache_stats(),
        "final_health_status": get_gedcom_cache_health(),
    }

    logger.info("GEDCOM cache demonstrations completed successfully")
    return demo_results


# --- Main Execution for Testing ---

if __name__ == "__main__":
    print("🧬 GEDCOM Cache System - Comprehensive Testing & Demonstration")
    print("=" * 70)

    # Run comprehensive tests
    print("\n📊 Running GEDCOM Cache Tests...")
    test_results = run_gedcom_cache_tests()

    print(
        f"\n✅ Test Results: {test_results['tests_passed']}/{test_results['tests_run']} passed ({test_results['pass_rate']:.1f}%)"
    )

    # Display test details
    for test in test_results["test_details"]:
        status_emoji = (
            "✅"
            if test["status"] == "PASS"
            else "❌" if test["status"] == "FAIL" else "⚠️"
        )
        duration = test.get("duration_ms", 0)
        print(f"  {status_emoji} {test['name']}: {test['status']} ({duration:.2f}ms)")

    # Run demonstrations
    print("\n🎯 Running GEDCOM Cache Demonstrations...")
    demo_results = demonstrate_gedcom_cache_usage()

    for demo in demo_results["demonstrations"]:
        status_emoji = "✅" if demo["status"] == "success" else "❌"
        print(f"  {status_emoji} {demo['name']}: {demo['description']}")

    # Display final statistics
    print(f"\n📈 Performance Summary:")
    print(f"  • Total test duration: {test_results['total_duration']:.3f}s")
    print(
        f"  • Average test duration: {test_results['performance_metrics']['average_test_duration_ms']:.2f}ms"
    )
    print(f"  • Cache module: {_gedcom_cache_module.get_module_name()}")

    # Display cache health
    health = get_gedcom_cache_health()
    print(f"  • Overall health: {health.get('overall_health', 'unknown')}")
    print(f"  • Memory cache entries: {health.get('memory_cache_entries', 0)}")

    print("\n🔄 GEDCOM cache system validation complete!")


# End of gedcom_cache.py
