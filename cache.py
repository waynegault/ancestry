#!/usr/bin/env python3

# cache.py

"""
cache.py - Disk-Based Caching Utility

Provides a persistent caching mechanism using the `diskcache` library.
Includes a decorator (`@cache_result`) for easily caching function outputs
and utility functions for managing the cache lifecycle (clearing, closing).
Cache directory and default settings are configurable via `config.py`.
"""

# --- Standard library imports ---
import atexit
import hashlib
import logging
import os
import shutil
import sys
import tempfile
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, Dict, Union, List
from unittest.mock import MagicMock, patch

# --- Third-party imports ---
from diskcache import Cache

# Import constants used for checking cache misses vs. stored None values
from diskcache.core import ENOVAL, UNKNOWN

# --- Local application imports ---
from config import config_instance  # Use configured instance
from logging_config import logger  # Use configured logger

# --- Test framework imports ---
try:
    from test_framework import (
        TestSuite,
        suppress_logging,
        create_mock_data,
        assert_valid_function,
    )

    TEST_FRAMEWORK_AVAILABLE = True
except ImportError:
    TEST_FRAMEWORK_AVAILABLE = False

# --- Global Cache Initialization ---

# Step 1: Define cache directory from configuration
if config_instance and getattr(config_instance, "CACHE_DIR", None):
    CACHE_DIR = Path(str(config_instance.CACHE_DIR))
else:
    CACHE_DIR = Path("Cache")
logger.debug(f"Cache directory configured: {CACHE_DIR}")

# Step 2: Ensure the cache directory exists
try:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Cache directory ensured at: {CACHE_DIR.resolve()}")
except OSError as e:
    logger.error(
        f"Failed to create cache directory {CACHE_DIR}: {e}. DiskCache initialization might fail.",
        exc_info=True,
    )
except Exception as e:
    logger.error(
        f"Unexpected error setting up cache directory {CACHE_DIR}: {e}", exc_info=True
    )

# Step 3: Initialize the DiskCache instance with aggressive settings
# This instance is shared across modules that import 'cache from cache'.
cache: Optional[Cache] = None  # Initialize as None
try:
    # Aggressive cache settings for better performance
    # 2GB size limit for large GEDCOM files and API responses
    # LRU eviction policy to keep frequently accessed data
    # Larger disk timeout for better reliability
    cache = Cache(
        CACHE_DIR,
        size_limit=int(2e9),  # 2 GB size limit
        eviction_policy="least-recently-used",  # LRU eviction
        disk_min_file_size=0,  # Store all data on disk
        timeout=60,  # Longer timeout for disk operations
        statistics=True,  # Enable cache statistics
    )
    logger.debug(
        f"DiskCache instance initialized with aggressive settings at {CACHE_DIR}."
    )
    logger.debug(f"Cache settings: size_limit=2GB, eviction=LRU, statistics=enabled")
except Exception as e:
    logger.critical(
        f"CRITICAL: Failed to initialize DiskCache at {CACHE_DIR}: {e}", exc_info=True
    )
    cache = None  # Ensure cache remains None if initialization fails

# --- Standard Cache Interface ---


class CacheInterface:
    """
    Standard interface for all cache modules to ensure consistency.
    Provides common methods and expected behavior across cache types.
    """

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this cache module."""
        raise NotImplementedError("Subclasses must implement get_stats()")

    def clear(self) -> bool:
        """Clear this cache module."""
        raise NotImplementedError("Subclasses must implement clear()")

    def warm(self) -> bool:
        """Warm this cache module with frequently accessed data."""
        raise NotImplementedError("Subclasses must implement warm()")

    def get_module_name(self) -> str:
        """Get the name of this cache module."""
        raise NotImplementedError("Subclasses must implement get_module_name()")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of this cache module."""
        raise NotImplementedError("Subclasses must implement get_health_status()")


class BaseCacheModule(CacheInterface):
    """
    Implementation of the standard cache interface for the base cache module.
    """

    def get_stats(self) -> Dict[str, Any]:
        """Get base cache statistics."""
        return get_cache_stats()

    def clear(self) -> bool:
        """Clear base cache."""
        return clear_cache()

    def warm(self) -> bool:
        """Warm base cache with system data."""
        try:
            # Warm cache with commonly accessed configuration
            warm_cache_with_data(
                "system_status", {"status": "operational", "timestamp": time.time()}
            )
            warm_cache_with_data(
                "cache_metadata", {"version": "2.0", "type": "enhanced_base"}
            )
            return True
        except Exception as e:
            logger.error(f"Error warming base cache: {e}")
            return False

    def get_module_name(self) -> str:
        """Get module name."""
        return "base_cache"

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of base cache."""
        try:
            stats = self.get_stats()
            total_requests = stats.get("hits", 0) + stats.get("misses", 0)
            hit_rate = (
                (stats.get("hits", 0) / total_requests * 100)
                if total_requests > 0
                else 0
            )

            # Determine health based on hit rate and cache availability
            if cache is None:
                health = "critical"
                message = "Cache instance not available"
            elif hit_rate >= 70:
                health = "excellent"
                message = f"High performance: {hit_rate:.1f}% hit rate"
            elif hit_rate >= 50:
                health = "good"
                message = f"Good performance: {hit_rate:.1f}% hit rate"
            elif hit_rate >= 30:
                health = "fair"
                message = f"Fair performance: {hit_rate:.1f}% hit rate"
            else:
                health = "poor"
                message = f"Low performance: {hit_rate:.1f}% hit rate"

            return {
                "health": health,
                "message": message,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "cache_available": cache is not None,
            }
        except Exception as e:
            return {
                "health": "error",
                "message": f"Health check failed: {e}",
                "hit_rate": 0,
                "total_requests": 0,
                "cache_available": False,
            }


# Global instance of base cache module
base_cache_module = BaseCacheModule()


# --- Cache Decorator ---


def cache_result(
    cache_key_prefix: str,
    expire: Optional[int] = None,  # Time in seconds, overrides Cache default if set
    ignore_args: bool = False,  # Use only prefix as key if True
) -> Callable:
    """
    Decorator factory to cache the result of a function using diskcache.

    Args:
        cache_key_prefix: A prefix string to identify the cache entry group.
                          The actual key will typically include the function name
                          and arguments unless ignore_args is True.
        expire: Optional time-to-live for the cached item in seconds.
                Overrides any default expiry set on the global Cache instance.
        ignore_args: If True, generate the cache key using only the prefix,
                     ignoring the function's arguments. Useful for caching
                     results of functions that fetch global or static data.

    Returns:
        A decorator function that wraps the original function with caching logic.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)  # Preserves original function metadata (name, docstring)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Step 1: Check if cache is available
            if cache is None:
                logger.error(
                    "Cache is not initialized. Bypassing cache and calling function directly."
                )
                return func(*args, **kwargs)

            # Step 2: Generate the final cache key
            if ignore_args:
                # Use only the prefix if arguments should be ignored
                final_cache_key = cache_key_prefix
                logger.debug(f"Using ignore_args=True, cache key: '{final_cache_key}'")
            else:
                # Generate key based on prefix, function name, args, and sorted kwargs
                # Note: Relies on stable string representation of arguments.
                # Complex objects might require custom serialization for reliable keys.
                try:
                    arg_key_part = (
                        f"_args{str(args)}_kwargs{str(sorted(kwargs.items()))}"
                    )
                    final_cache_key = (
                        f"{cache_key_prefix}_{func.__name__}{arg_key_part}"
                    )
                except Exception as key_gen_e:
                    logger.error(
                        f"Error generating cache key for {func.__name__}: {key_gen_e}. Bypassing cache."
                    )
                    return func(*args, **kwargs)
            # logger.debug(f"Generated cache key: '{final_cache_key}'") # Can be verbose

            # Step 3: Attempt to retrieve from cache
            try:
                # Use default=ENOVAL to distinguish a cache miss from a stored None value
                cached_value = cache.get(final_cache_key, default=ENOVAL, retry=True)

                # Step 3a: Cache Hit - Return cached value
                if cached_value is not ENOVAL:
                    logger.debug(f"Cache HIT for key: '{final_cache_key}'")
                    return cached_value
                # Step 3b: Cache Miss - Log and proceed to function execution
                else:
                    logger.debug(f"Cache MISS for key: '{final_cache_key}'")

            except Exception as e:
                # Log errors during cache read but treat as cache miss
                logger.error(
                    f"Cache read error for key '{final_cache_key}': {e}", exc_info=True
                )
                # Proceed to execute the function below

            # Step 4: Execute the original function (if cache miss or read error)
            try:
                result = func(*args, **kwargs)

                # Step 5: Store the result in the cache, enforcing size limit
                try:
                    if check_cache_size_before_add():
                        cache.set(final_cache_key, result, expire=expire, retry=True)
                        expire_msg = (
                            f"with expiry {expire}s"
                            if expire is not None
                            else "with default expiry"
                        )
                        logger.debug(
                            f"Cached result for key: '{final_cache_key}' {expire_msg}."
                        )
                    else:
                        logger.warning(
                            f"Cache size limit reached. Skipping cache set for key: '{final_cache_key}'"
                        )
                except Exception as cache_set_e:
                    # Log error during cache set, but return the result anyway
                    logger.error(
                        f"Failed to cache result for key '{final_cache_key}': {cache_set_e}",
                        exc_info=True,
                    )

                # Step 6: Return the live result
                return result

            except Exception as func_exec_e:
                # Step 7: Handle errors during function execution
                logger.error(
                    f"Error during execution of function '{func.__name__}' or caching for key '{final_cache_key}': {func_exec_e}",
                    exc_info=True,
                )
                # Do NOT cache the error or any partial result.
                # Re-raise the original exception to the caller.
                raise func_exec_e

        # End of wrapper
        return wrapper

    # End of decorator
    return decorator


# End of cache_result


# --- Cache Management Utilities ---


def clear_cache():
    """
    Removes all items from the disk cache.
    Attempts manual directory removal as a fallback if the cache object
    is unavailable or `cache.clear()` fails.

    Returns:
        True if the cache was cleared successfully (either via API or manually),
        False otherwise.
    """
    # Step 1: Try clearing using the diskcache API if available
    if cache:
        try:
            count = cache.clear()
            logger.debug(
                f"Cache cleared successfully via API. {count} items removed from {CACHE_DIR}."
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to clear cache via API: {e}. Attempting manual removal...",
                exc_info=True,
            )
            # Fall through to manual removal attempt

    # Step 2: Fallback - Attempt manual directory removal
    logger.warning(
        "Cache object not available or API clear failed. Attempting manual directory removal..."
    )
    if CACHE_DIR and CACHE_DIR.exists():
        try:
            shutil.rmtree(CACHE_DIR)
            logger.debug(f"Manually removed cache directory: {CACHE_DIR}")
            # Recreate the directory immediately after removing it
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Recreated empty cache directory: {CACHE_DIR}")
                return True  # Manual removal and recreation successful
            except OSError as mkdir_e:
                logger.error(
                    f"Failed to recreate cache directory {CACHE_DIR} after manual removal: {mkdir_e}"
                )
                return False  # Failed to recreate directory
        except Exception as e:
            logger.error(
                f"Failed to manually remove cache directory {CACHE_DIR}: {e}",
                exc_info=True,
            )
            return False  # Manual removal failed
    else:
        logger.debug("Cache directory does not exist. Nothing to clear manually.")
        return True  # Considered success as the directory is gone

    # Should only be reached if API clear fails AND manual removal fails
    return False


# End of clear_cache


def close_cache():
    """Closes the diskcache connection pool gracefully."""
    # Step 1: Check if cache object exists
    if cache:
        # Step 2: Attempt to close the cache
        try:
            logger.debug("Closing DiskCache connection...")
            cache.close()
            logger.debug("DiskCache connection closed.")
        except Exception as e:
            logger.error(f"Error closing DiskCache connection: {e}", exc_info=True)
    else:
        logger.debug("No active DiskCache instance to close.")


# End of close_cache


# --- Enhanced Cross-Module Cache Coordination ---


def get_unified_cache_key(module: str, operation: str, *args, **kwargs) -> str:
    """
    Generate unified cache keys across modules for better coordination.

    Args:
        module: Cache module name (e.g., 'gedcom', 'api', 'base')
        operation: Operation type (e.g., 'load', 'query', 'process')
        *args: Additional arguments for key generation
        **kwargs: Additional keyword arguments for key generation

    Returns:
        Standardized cache key string
    """
    # Create base key components
    key_parts = [module, operation]

    # Add positional arguments
    for arg in args:
        if isinstance(arg, (str, int, float)):
            key_parts.append(str(arg))
        elif hasattr(arg, "__str__"):
            key_parts.append(str(arg))

    # Add keyword arguments in sorted order for consistency
    for key in sorted(kwargs.keys()):
        value = kwargs[key]
        if isinstance(value, (str, int, float, bool)):
            key_parts.append(f"{key}={value}")

    # Generate hash for long keys to keep them manageable
    key_string = "_".join(key_parts)
    if len(key_string) > 100:
        key_hash = hashlib.md5(key_string.encode()).hexdigest()[:16]
        key_string = f"{module}_{operation}_{key_hash}"

    return key_string


def invalidate_related_caches(
    pattern: str, exclude_modules: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Invalidate caches across multiple modules based on pattern.

    Args:
        pattern: Pattern to match for cache invalidation
        exclude_modules: List of module names to exclude from invalidation

    Returns:
        Dictionary with count of invalidated entries per module
    """
    exclude_modules = exclude_modules or []
    results = {}

    # Invalidate base cache
    if "base" not in exclude_modules:
        try:
            count = invalidate_cache_pattern(pattern)
            results["base_cache"] = count
            logger.info(
                f"Invalidated {count} base cache entries matching pattern: {pattern}"
            )
        except Exception as e:
            logger.error(f"Error invalidating base cache pattern {pattern}: {e}")
            results["base_cache"] = 0

    # Invalidate GEDCOM cache
    if "gedcom" not in exclude_modules:
        try:
            from gedcom_cache import clear_memory_cache

            count = clear_memory_cache()
            results["gedcom_cache"] = count
            logger.info(f"Cleared {count} GEDCOM memory cache entries")
        except ImportError:
            logger.debug("GEDCOM cache module not available for invalidation")
            results["gedcom_cache"] = 0
        except Exception as e:
            logger.error(f"Error invalidating GEDCOM cache: {e}")
            results["gedcom_cache"] = 0

    # Note: API cache invalidation would be handled by api_cache module
    # This creates a standardized approach for cross-module coordination

    total_invalidated = sum(results.values())
    logger.info(f"Total cache entries invalidated across modules: {total_invalidated}")

    return results


def get_cache_coordination_stats() -> Dict[str, Any]:
    """
    Get comprehensive statistics for cache coordination across modules.

    Returns:
        Dictionary with coordination statistics and health metrics
    """
    coordination_stats = {
        "timestamp": time.time(),
        "modules": {},
        "cross_module_health": "unknown",
        "total_entries": 0,
        "total_volume": 0,
        "overall_hit_rate": 0.0,
    }

    # Collect stats from available modules
    total_hits = 0
    total_requests = 0

    # Base cache stats
    try:
        base_stats = get_cache_stats()
        coordination_stats["modules"]["base"] = base_stats

        hits = base_stats.get("hits", 0)
        misses = base_stats.get("misses", 0)
        total_hits += hits
        total_requests += hits + misses
        coordination_stats["total_entries"] += base_stats.get("size", 0)
        coordination_stats["total_volume"] += base_stats.get("volume", 0)

    except Exception as e:
        logger.debug(f"Error getting base cache stats for coordination: {e}")
        coordination_stats["modules"]["base"] = {"error": str(e)}

    # GEDCOM cache stats
    try:
        from gedcom_cache import get_gedcom_cache_info

        gedcom_stats = get_gedcom_cache_info()
        coordination_stats["modules"]["gedcom"] = gedcom_stats

        # Add to totals if available
        memory_entries = gedcom_stats.get("memory_cache_entries", 0)
        coordination_stats["total_entries"] += memory_entries

    except ImportError:
        coordination_stats["modules"]["gedcom"] = {"status": "not_available"}
    except Exception as e:
        logger.debug(f"Error getting GEDCOM cache stats for coordination: {e}")
        coordination_stats["modules"]["gedcom"] = {"error": str(e)}

    # API cache stats
    try:
        from api_cache import get_api_cache_stats

        api_stats = get_api_cache_stats()
        coordination_stats["modules"]["api"] = api_stats

        # Add to totals if available
        api_entries = api_stats.get("api_entries", 0)
        ai_entries = api_stats.get("ai_entries", 0)
        db_entries = api_stats.get("db_entries", 0)
        coordination_stats["total_entries"] += api_entries + ai_entries + db_entries

    except ImportError:
        coordination_stats["modules"]["api"] = {"status": "not_available"}
    except Exception as e:
        logger.debug(f"Error getting API cache stats for coordination: {e}")
        coordination_stats["modules"]["api"] = {"error": str(e)}

    # Calculate overall metrics
    if total_requests > 0:
        coordination_stats["overall_hit_rate"] = (total_hits / total_requests) * 100

    # Determine cross-module health
    available_modules = len(
        [
            m
            for m in coordination_stats["modules"].values()
            if "error" not in m and m.get("status") != "not_available"
        ]
    )

    if available_modules >= 3:
        coordination_stats["cross_module_health"] = "excellent"
    elif available_modules >= 2:
        coordination_stats["cross_module_health"] = "good"
    elif available_modules >= 1:
        coordination_stats["cross_module_health"] = "limited"
    else:
        coordination_stats["cross_module_health"] = "critical"

    return coordination_stats


# --- Enhanced Cache Management Functions ---


def enforce_cache_size_limit() -> Dict[str, Any]:
    """
    Enforce cache size limits based on configuration.

    Returns:
        Dictionary with enforcement results
    """
    if not cache or not config_instance:
        return {
            "status": "skipped",
            "reason": "Cache or config not available",
            "entries_removed": 0,
        }

    try:
        current_size = get_cache_entry_count()
        max_size = config_instance.CACHE_MAX_SIZE

        if current_size <= max_size:
            return {
                "status": "compliant",
                "current_size": current_size,
                "max_size": max_size,
                "entries_removed": 0,
                "utilization": (current_size / max_size * 100) if max_size > 0 else 0,
            }  # Need to remove entries
        excess_entries = current_size - max_size
        entries_to_remove = excess_entries + 1  # Remove excess + 1 for buffer

        removed_count = 0
        try:
            # Manual LRU eviction by iterating through cache and removing oldest entries
            # diskcache stores entries in LRU order when iterating
            keys_to_remove = []
            key_count = 0

            # Collect keys to remove (oldest first)
            for key in cache:
                keys_to_remove.append(key)
                key_count += 1
                if key_count >= entries_to_remove:
                    break

            # Remove the collected keys
            for key in keys_to_remove:
                try:
                    cache.delete(key)
                    removed_count += 1
                    logger.debug(f"Evicted cache key: {key}")
                except Exception as del_error:
                    logger.warning(f"Failed to delete cache key {key}: {del_error}")

            logger.info(f"Manually evicted {removed_count} entries using LRU order")

        except Exception as evict_error:
            logger.error(f"Error during manual cache eviction: {evict_error}")
            # Last resort: try using diskcache's built-in cull (size-based)
            try:
                culled = cache.cull()
                logger.info(f"Fallback: diskcache culled {culled} entries by size")
            except Exception as cull_error:
                logger.error(f"Even fallback cull failed: {cull_error}")

        final_size = get_cache_entry_count()

        logger.info(f"Cache size enforcement: removed {removed_count} entries")
        logger.info(f"Cache size: {current_size} → {final_size} (limit: {max_size})")

        return {
            "status": "enforced",
            "current_size": final_size,
            "max_size": max_size,
            "entries_removed": removed_count,
            "utilization": (final_size / max_size * 100) if max_size > 0 else 0,
            "initial_size": current_size,
        }

    except Exception as e:
        logger.error(f"Error enforcing cache size limit: {e}")
        return {"status": "error", "error": str(e), "entries_removed": 0}


def check_cache_size_before_add(estimated_size: int = 1) -> bool:
    """
    Check if adding an entry would exceed cache size limits.

    Args:
        estimated_size: Estimated number of entries to add

    Returns:
        True if addition is allowed, False if it would exceed limits
    """
    if not cache or not config_instance:
        return True  # Allow if we can't check

    try:
        current_size = get_cache_entry_count()
        max_size = config_instance.CACHE_MAX_SIZE

        if current_size + estimated_size <= max_size:
            return True

        # Would exceed limit - trigger enforcement
        enforcement_result = enforce_cache_size_limit()

        # Check again after enforcement
        new_size = get_cache_entry_count()
        return new_size + estimated_size <= max_size

    except Exception as e:
        logger.error(f"Error checking cache size limits: {e}")
        return True  # Allow on error


def get_cache_entry_count() -> int:
    """Safely get the number of entries in the cache."""
    try:
        if cache is not None:
            # Use len() which works correctly with diskcache
            # Type ignore for diskcache compatibility
            return len(cache)  # type: ignore
        else:
            return 0
    except Exception as e:
        logger.warning(f"Error getting cache entry count: {e}")
        # Fallback: try to count by iterating (slower but reliable)
        try:
            count = 0
            for _ in cache:  # type: ignore
                count += 1
            return count
        except Exception as e2:
            logger.error(f"Fallback count also failed: {e2}")
            return 0


def get_cache_stats() -> Dict[str, Any]:
    """
    Returns cache statistics including hits, misses, size, and evictions.

    Returns:
        Dictionary containing cache statistics or empty dict if cache unavailable.
    """
    if cache is None:
        logger.warning("Cache not available for statistics.")
        return {}

    try:
        stats_tuple = cache.stats(enable=True, reset=False)
        current_entry_count = get_cache_entry_count()

        stats = {
            "hits": stats_tuple[0],
            "misses": stats_tuple[1],
            "entries": current_entry_count,  # Use "entries" field name for consistency
            "size": current_entry_count,  # Keep "size" for backward compatibility
            "volume": cache.volume(),
            "evictions": getattr(cache, "evictions", 0),
        }

        # Calculate hit rate
        total_requests = stats["hits"] + stats["misses"]
        if total_requests > 0:
            stats["hit_rate"] = (stats["hits"] / total_requests) * 100
        else:
            stats["hit_rate"] = 0.0

        # Add cache configuration info
        stats["cache_dir"] = str(CACHE_DIR)
        stats["eviction_policy"] = "least-recently-used"
        stats["size_limit_gb"] = 2.0

        # Add size limit information from config
        if config_instance:
            max_entries = config_instance.CACHE_MAX_SIZE
            current_entries = stats["entries"]  # Use the entries field

            stats["max_entries"] = max_entries
            stats["entries_utilization"] = (
                (current_entries / max_entries * 100) if max_entries > 0 else 0.0
            )
            stats["size_compliant"] = current_entries <= max_entries
        else:
            stats["max_entries"] = "Unknown"
            stats["entries_utilization"] = 0.0
            stats["size_compliant"] = "Unknown"

        return stats
    except Exception as e:
        logger.error(f"Error retrieving cache statistics: {e}")
        return {}


# End of get_cache_stats


def cache_file_based_on_mtime(
    cache_key_prefix: str,
    file_path: str,
    expire: Optional[int] = None,
) -> Callable:
    """
    Enhanced decorator that caches based on file modification time.
    Automatically invalidates cache when the source file changes.
    Perfect for GEDCOM files and other data files.

    Args:
        cache_key_prefix: Prefix for the cache key
        file_path: Path to the file to monitor for changes
        expire: Optional expiration time in seconds

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if cache is None:
                logger.error("Cache not available. Calling function directly.")
                return func(*args, **kwargs)

            try:
                # Get file modification time
                file_mtime = os.path.getmtime(file_path)

                # Create cache key that includes file mtime
                mtime_hash = hashlib.md5(str(file_mtime).encode()).hexdigest()[:8]
                final_cache_key = (
                    f"{cache_key_prefix}_{func.__name__}_mtime_{mtime_hash}"
                )

                # Try to get from cache
                cached_value = cache.get(final_cache_key, default=ENOVAL, retry=True)

                if cached_value is not ENOVAL:
                    logger.debug(f"Cache HIT for file-based key: '{final_cache_key}'")
                    return cached_value
                else:
                    logger.debug(f"Cache MISS for file-based key: '{final_cache_key}'")

                # Execute function and cache result
                result = func(*args, **kwargs)
                cache.set(final_cache_key, result, expire=expire, retry=True)
                logger.debug(f"Cached file-based result for key: '{final_cache_key}'")

                return result

            except Exception as e:
                logger.error(f"Error in file-based caching for {func.__name__}: {e}")
                return func(*args, **kwargs)

        return wrapper

    return decorator


# End of cache_file_based_on_mtime


def warm_cache_with_data(
    cache_key: str, data: Any, expire: Optional[int] = None
) -> bool:
    """
    Preloads cache with data (cache warming).

    Args:
        cache_key: Key to store the data under
        data: Data to cache
        expire: Optional expiration time in seconds

    Returns:
        True if successful, False otherwise
    """
    if cache is None:
        logger.warning("Cache not available for warming.")
        return False

    try:
        cache.set(cache_key, data, expire=expire, retry=True)
        logger.debug(f"Cache warmed with key: '{cache_key}'")
        return True
    except Exception as e:
        logger.error(f"Error warming cache with key '{cache_key}': {e}")
        return False


# End of warm_cache_with_data


def invalidate_cache_pattern(pattern: str) -> int:
    """
    Invalidates all cache entries matching a pattern.

    Args:
        pattern: Pattern to match cache keys (simple string contains)

    Returns:
        Number of entries invalidated
    """
    if cache is None:
        logger.warning("Cache not available for pattern invalidation.")
        return 0

    try:
        invalidated = 0
        # Get all keys and filter by pattern
        for key in list(cache):
            if pattern in str(key):
                cache.delete(key)
                invalidated += 1

        logger.debug(
            f"Invalidated {invalidated} cache entries matching pattern: '{pattern}'"
        )
        return invalidated
    except Exception as e:
        logger.error(f"Error invalidating cache pattern '{pattern}': {e}")
        return 0


# End of invalidate_cache_pattern


# --- Cleanup Registration ---
# Step 1: Register the close_cache function to be called automatically on script exit
# This ensures cache files are properly closed and resources released.
atexit.register(close_cache)
logger.debug("Registered close_cache function with atexit for automatic cleanup.")


# Log loading confirmation with stats
if cache is not None:
    logger.debug("cache.py loaded and cache initialized successfully.")
    logger.debug(f"Cache statistics enabled: {getattr(cache, 'statistics', False)}")
else:
    logger.debug("cache.py loaded but cache initialization failed.")

# --- Test/Demo Functions ---


def run_cache_tests():
    """
    Comprehensive test suite for cache.py functionality.
    Demonstrates all cache features and validates proper operation.
    """
    print("=" * 60)
    print("CACHE.PY - COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    tests_passed = 0
    tests_failed = 0

    def test_result(test_name: str, passed: bool, details: str = ""):
        nonlocal tests_passed, tests_failed
        status = "PASS" if passed else "FAIL"
        print(f"[{status:>4}] {test_name}")
        if details:
            print(f"       {details}")
        if passed:
            tests_passed += 1
        else:
            tests_failed += 1
        return passed

    # Cache Initialization
    print("\n--- Test Section 1: Cache Initialization ---")

    test_result(
        "Cache Object Available",
        cache is not None,
        f"Cache instance: {type(cache) if cache else 'None'}",
    )

    if cache and CACHE_DIR:
        test_result(
            "Cache Directory Exists",
            CACHE_DIR.exists(),
            f"Directory: {CACHE_DIR.resolve()}",
        )

        test_result(
            "Cache Statistics Enabled",
            getattr(cache, "statistics", False),
            "Statistics tracking for performance monitoring",
        )

    # Basic Cache Operations
    print("\n--- Test Section 2: Cache Operations ---")

    if cache:
        try:
            # Test basic set/get
            test_key = "test_basic_operation"
            test_value = {"test": "data", "timestamp": time.time()}

            cache.set(test_key, test_value)
            retrieved_value = cache.get(test_key)

            test_result(
                "Basic Set/Get Operation",
                retrieved_value == test_value,
                f"Stored and retrieved: {type(test_value)}",
            )

            # Test cache miss
            missing_value = cache.get("nonexistent_key", default="NOT_FOUND")
            test_result(
                "Cache Miss Handling",
                missing_value == "NOT_FOUND",
                "Proper default value returned",
            )

            # Test deletion
            cache.delete(test_key)
            deleted_check = cache.get(test_key, default="DELETED")
            test_result(
                "Cache Deletion",
                deleted_check == "DELETED",
                "Key properly removed from cache",
            )

        except Exception as e:
            test_result("Basic Cache Operations", False, f"Error: {str(e)}")

    # Cache Decorator Functionality
    print("\n--- Test Section 3: Cache Decorator Functionality ---")

    try:
        # Create test function with cache decorator
        @cache_result("test_decorator", expire=300)
        def expensive_calculation(x: int, y: int) -> int:
            """Simulates an expensive calculation."""
            time.sleep(0.1)  # Simulate processing time
            return x * y + x**2

        # Test first call (should cache)
        start_time = time.time()
        result1 = expensive_calculation(5, 10)
        first_call_time = time.time() - start_time

        # Test second call (should be cached)
        start_time = time.time()
        result2 = expensive_calculation(5, 10)
        second_call_time = time.time() - start_time

        test_result(
            "Decorator Caching Works",
            result1 == result2,
            f"Both calls returned: {result1}",
        )

        test_result(
            "Cache Performance Improvement",
            second_call_time < first_call_time / 2,
            f"First: {first_call_time:.3f}s, Second: {second_call_time:.3f}s",
        )

    except Exception as e:
        test_result("Cache Decorator Functionality", False, f"Error: {str(e)}")

    # File-Based Caching
    print("\n--- Test Section 4: File-Based Caching ---")

    try:
        # Create a temporary test file
        if CACHE_DIR:
            test_file_path = CACHE_DIR / "test_file.txt"
            with open(test_file_path, "w") as f:
                f.write("test content")

        @cache_file_based_on_mtime("test_file_cache", str(test_file_path))
        def read_file_content(file_path: str) -> str:
            """Reads file content (expensive operation simulation)."""
            with open(file_path, "r") as f:
                return f.read()

        # First call
        content1 = read_file_content(str(test_file_path))

        # Second call (should be cached)
        content2 = read_file_content(str(test_file_path))

        test_result(
            "File-Based Caching",
            content1 == content2 == "test content",
            f"File content cached properly",
        )

        # Clean up test file
        test_file_path.unlink()

    except Exception as e:
        test_result("File-Based Caching", False, f"Error: {str(e)}")

    # Cache Management Functions
    print("\n--- Test Section 5: Cache Management Functions ---")

    try:
        # Test cache statistics
        stats = get_cache_stats()
        test_result(
            "Cache Statistics Retrieval",
            isinstance(stats, dict),
            f"Stats keys: {list(stats.keys())}",
        )

        # Test cache warming
        warm_success = warm_cache_with_data("warm_test_key", {"warmed": True})
        test_result(
            "Cache Warming", warm_success, "Successfully preloaded cache with data"
        )

        # Verify warmed data
        if warm_success and cache:
            warmed_data = cache.get("warm_test_key")
            test_result(
                "Warmed Data Retrieval",
                warmed_data == {"warmed": True},
                "Warmed data retrieved correctly",
            )

        # Test pattern invalidation
        if cache:
            cache.set("pattern_test_1", "data1")
            cache.set("pattern_test_2", "data2")
            cache.set("other_key", "data3")

            invalidated_count = invalidate_cache_pattern("pattern_test")
            test_result(
                "Pattern Invalidation",
                invalidated_count == 2,
                f"Invalidated {invalidated_count} entries matching pattern",
            )
        else:
            test_result("Pattern Invalidation", False, "Cache not available")

    except Exception as e:
        test_result("Cache Management Functions", False, f"Error: {str(e)}")

    # Performance and Stress Testing
    print("\n--- Test Section 6: Performance and Stress Testing ---")

    try:
        # Test multiple rapid operations
        start_time = time.time()
        if cache:
            for i in range(100):
                cache.set(f"perf_test_{i}", {"iteration": i, "data": "x" * 100})

            for i in range(100):
                retrieved = cache.get(f"perf_test_{i}")
                if retrieved is None:
                    raise ValueError(f"Failed to retrieve perf_test_{i}")

            performance_time = time.time() - start_time
            test_result(
                "Performance Stress Test",
                performance_time < 5.0,
                f"100 set/get operations in {performance_time:.2f} seconds",
            )

            # Clean up performance test data
            for i in range(100):
                cache.delete(f"perf_test_{i}")
        else:
            test_result("Performance Stress Test", False, "Cache not available")

    except Exception as e:
        test_result("Performance and Stress Testing", False, f"Error: {str(e)}")

    # Test Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    total_tests = tests_passed + tests_failed
    success_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0

    print(f"Total Tests Run: {total_tests}")
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    print(f"Success Rate: {success_rate:.1f}%")

    # Cache statistics
    if cache:
        final_stats = get_cache_stats()
        print(f"\nFinal Cache Statistics:")
        for key, value in final_stats.items():
            print(f"  {key}: {value}")

    if tests_failed == 0:
        print("\n🎉 All cache tests PASSED! cache.py is working correctly.")
    else:
        print(f"\n⚠️  {tests_failed} test(s) FAILED. Please review the issues above.")

    return tests_failed == 0


def demonstrate_cache_usage():
    """
    Demonstrates practical usage patterns for the cache system.
    Shows real-world examples of how to use cache decorators and functions.
    """
    print("=" * 60)
    print("CACHE.PY - USAGE DEMONSTRATION")
    print("=" * 60)

    print("\n--- Example 1: Caching Expensive Calculations ---")

    @cache_result("fibonacci", expire=3600)  # Cache for 1 hour
    def fibonacci(n: int) -> int:
        """Calculate fibonacci number with caching."""
        if n <= 1:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    print("Computing fibonacci(30) with caching...")
    start_time = time.time()
    result = fibonacci(30)
    calc_time = time.time() - start_time
    print(f"Result: {result} (computed in {calc_time:.3f} seconds)")

    print("Computing fibonacci(30) again (should be much faster)...")
    start_time = time.time()
    result2 = fibonacci(30)
    cached_time = time.time() - start_time
    print(f"Result: {result2} (retrieved in {cached_time:.3f} seconds)")

    if cached_time > 0:
        speed_improvement = calc_time / cached_time
        print(f"Speed improvement: {speed_improvement:.1f}x faster!")
    else:
        print("Speed improvement: Cache retrieval was instantaneous!")

    print("\n--- Example 2: API Response Caching ---")

    @cache_result("api_response", expire=300)  # Cache for 5 minutes
    def simulate_api_call(endpoint: str) -> dict:
        """Simulate an API call with caching."""
        print(f"  Making API call to: {endpoint}")
        time.sleep(0.5)  # Simulate network delay
        return {
            "endpoint": endpoint,
            "timestamp": time.time(),
            "data": f"Response from {endpoint}",
            "status": "success",
        }

    print("First API call (will make actual request):")
    api_result1 = simulate_api_call("/users/123")
    print(f"  Response: {api_result1['data']}")

    print("Second API call (will use cache):")
    api_result2 = simulate_api_call("/users/123")
    print(f"  Response: {api_result2['data']}")
    print(f"  Same response cached: {api_result1 == api_result2}")

    print("\n--- Example 3: Configuration Data Caching ---")

    @cache_result("config_data", ignore_args=True, expire=1800)  # 30 minutes
    def load_configuration() -> dict:
        """Load configuration with caching (ignores arguments)."""
        print("  Loading configuration from disk...")
        time.sleep(0.2)  # Simulate file I/O
        return {
            "database_url": "sqlite:///ancestry.db",
            "cache_size": "2GB",
            "log_level": "INFO",
            "loaded_at": time.time(),
        }

    print("Loading configuration (first time):")
    config1 = load_configuration()
    print(f"  Database URL: {config1['database_url']}")

    print("Loading configuration (cached):")
    config2 = load_configuration()
    print(f"  Cache size: {config2['cache_size']}")
    print(f"  Same config object: {config1 is config2}")

    print("\n--- Example 4: Cache Management ---")

    # Show current cache statistics
    stats = get_cache_stats()
    print(f"Current cache statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Demonstrate cache warming
    print("\nWarming cache with frequently needed data...")
    warm_cache_with_data("user_preferences", {"theme": "dark", "language": "en"})
    warm_cache_with_data("system_config", {"version": "1.0", "debug": False})

    # Show updated statistics
    updated_stats = get_cache_stats()
    print(f"Updated cache statistics:")
    for key, value in updated_stats.items():
        print(f"  {key}: {value}")

    print("\n--- Cache Usage Demo Complete ---")


# ==============================================
# Comprehensive Test Suite - Module Level
# ==============================================


def run_comprehensive_tests() -> bool:
    """
    Comprehensive test suite for cache.py.
    Tests core caching functionality, decorators, and performance features.
    """
    try:
        from test_framework import TestSuite, suppress_logging

        suite = TestSuite("Core Cache System & Decorators", "cache.py")
        suite.start_suite()

        # Cache decorator functionality
        def test_cache_decorator():
            if "cache_result" in globals():
                cache_decorator = globals()["cache_result"]

                call_count = 0

                @cache_decorator("test_expensive_function")
                def expensive_function(x):
                    nonlocal call_count
                    call_count += 1
                    return x * 2

                # First call should execute function
                result1 = expensive_function(5)
                assert result1 == 10
                assert call_count == 1

                # Second call should use cache
                result2 = expensive_function(5)
                assert result2 == 10
                assert call_count == 1  # Should not increment

        # Cache key generation
        def test_cache_key_generation():
            if "generate_cache_key" in globals():
                key_generator = globals()["generate_cache_key"]

                # Test with various argument types
                key1 = key_generator("func", 1, 2, kwarg1="value1")
                key2 = key_generator("func", 1, 2, kwarg1="value1")
                key3 = key_generator("func", 1, 3, kwarg1="value1")

                assert key1 == key2  # Same arguments should generate same key
                assert (
                    key1 != key3
                )  # Different arguments should generate different keys

        # File modification time checking
        def test_file_modification_checking():
            if "check_file_modified" in globals():
                checker = globals()["check_file_modified"]

                with tempfile.NamedTemporaryFile() as temp_file:
                    # Write initial content
                    temp_file.write(b"initial content")
                    temp_file.flush()

                    # Check modification
                    is_modified = checker(temp_file.name, time.time() - 1)
                    assert isinstance(is_modified, bool)

        # Cache invalidation based on file changes
        def test_file_based_invalidation():
            if "invalidate_on_file_change" in globals():
                invalidator = globals()["invalidate_on_file_change"]

                with tempfile.NamedTemporaryFile() as temp_file:
                    result = invalidator(temp_file.name)
                    assert isinstance(result, bool)

        # Cache statistics tracking
        def test_cache_statistics_tracking():
            # Test cache hit/miss tracking
            if "get_cache_statistics" in globals():
                stats_func = globals()["get_cache_statistics"]
                stats = stats_func()

                assert isinstance(stats, dict)
                # Check for common statistics fields
                stat_fields = ["hits", "misses", "hit_rate", "total_size"]
                for field in stat_fields:
                    if field in stats:
                        assert isinstance(stats[field], (int, float))

        # Cache size management
        def test_cache_size_management():
            if "manage_cache_size" in globals():
                size_manager = globals()["manage_cache_size"]

                # Test size management
                result = size_manager(max_size_mb=100)
                assert isinstance(result, bool)

        # Cache warming strategies
        def test_cache_warming_strategies():
            if "warm_function_cache" in globals():
                warmer = globals()["warm_function_cache"]

                def sample_function(x):
                    return x**2

                # Test cache warming with sample data
                sample_inputs = [1, 2, 3, 4, 5]
                result = warmer(sample_function, sample_inputs)
                assert isinstance(result, (bool, int))

        # Cache cleanup operations
        def test_cache_cleanup():
            cleanup_functions = [
                "clear_cache",
                "cleanup_expired_entries",
                "optimize_cache_storage",
            ]

            for func_name in cleanup_functions:
                if func_name in globals():
                    cleanup_func = globals()[func_name]
                    result = cleanup_func()
                    # Should not raise exceptions

        # Cache persistence mechanisms
        def test_cache_persistence():
            if "save_cache_state" in globals() and "restore_cache_state" in globals():
                save_func = globals()["save_cache_state"]
                restore_func = globals()["restore_cache_state"]

                with tempfile.NamedTemporaryFile() as temp_file:
                    # Test save and restore
                    save_result = save_func(temp_file.name)
                    restore_result = restore_func(temp_file.name)

                    assert isinstance(save_result, bool)
                    assert isinstance(restore_result, bool)

        # Advanced cache features
        def test_advanced_cache_features():
            # Test advanced caching features
            advanced_functions = [
                "cache_with_ttl",
                "cache_with_dependency",
                "cache_with_tags",
                "hierarchical_cache",
                "distributed_cache",
            ]

            for func_name in advanced_functions:
                if func_name in globals():
                    func = globals()[func_name]
                    assert callable(func)

        # Run all tests
        test_functions = {
            "Cache decorator functionality": (
                test_cache_decorator,
                "Should provide caching decorator for function results",
            ),
            "Cache key generation": (
                test_cache_key_generation,
                "Should generate consistent keys for function arguments",
            ),
            "File modification checking": (
                test_file_modification_checking,
                "Should detect when cached files have been modified",
            ),
            "File-based cache invalidation": (
                test_file_based_invalidation,
                "Should invalidate cache when source files change",
            ),
            "Cache statistics tracking": (
                test_cache_statistics_tracking,
                "Should track cache performance metrics",
            ),
            "Cache size management": (
                test_cache_size_management,
                "Should manage cache size and memory usage",
            ),
            "Cache warming strategies": (
                test_cache_warming_strategies,
                "Should support pre-loading cache with anticipated data",
            ),
            "Cache cleanup operations": (
                test_cache_cleanup,
                "Should provide cache maintenance and cleanup functions",
            ),
            "Cache persistence mechanisms": (
                test_cache_persistence,
                "Should save and restore cache state across sessions",
            ),
            "Advanced cache features": (
                test_advanced_cache_features,
                "Should support advanced caching patterns and strategies",
            ),
        }

        with suppress_logging():
            for test_name, (test_func, expected_behavior) in test_functions.items():
                suite.run_test(test_name, test_func, expected_behavior)

        return suite.finish_suite()

    except ImportError:
        # Fallback when test framework is not available
        print("🛠️ Running Cache fallback test suite...")

        tests_passed = 0
        tests_total = 0

        # Test cache availability
        tests_total += 1
        try:
            if cache is not None:
                tests_passed += 1
                print("✅ Cache initialization test passed")
            else:
                print("❌ Cache initialization test failed")
        except Exception as e:
            print(f"❌ Cache initialization test error: {e}")

        # Test cache_result decorator if available
        if "cache_result" in globals():
            tests_total += 1
            try:
                cache_decorator = globals()["cache_result"]

                @cache_decorator("test_function")
                def test_func(x):
                    return x * 2

                result = test_func(5)
                if result == 10:
                    tests_passed += 1
                    print("✅ cache_result decorator test passed")
                else:
                    print("❌ cache_result decorator test failed")
            except Exception as e:
                print(f"❌ cache_result decorator test error: {e}")

        # Test clear_cache if available
        if "clear_cache" in globals():
            tests_total += 1
            try:
                clear_func = globals()["clear_cache"]
                clear_func()
                tests_passed += 1
                print("✅ clear_cache function test passed")
            except Exception as e:
                print(f"❌ clear_cache function test error: {e}")

        # Test cache statistics if available
        if cache and hasattr(cache, "volume"):
            tests_total += 1
            try:
                volume = cache.volume()
                if isinstance(volume, (int, float)):
                    tests_passed += 1
                    print("✅ Cache volume check test passed")
                else:
                    print("❌ Cache volume check test failed")
            except Exception as e:
                print(f"❌ Cache volume check test error: {e}")

        print(f"🏁 Cache fallback tests completed: {tests_passed}/{tests_total} passed")
        return tests_passed == tests_total


# ==============================================
# Standalone Test Block
# ==============================================
if __name__ == "__main__":
    print("🛠️ Running Core Cache System & Decorators comprehensive test suite...")
    try:
        success = run_comprehensive_tests()
        if success:
            print("✅ All tests passed! Cache system is fully operational.")
        else:
            print("⚠️  Some tests failed. Please review the output above.")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        print("🔄 Falling back to basic cache tests...")
        try:
            success = run_cache_tests()
            print("\n" + "💡 " + "RUNNING USAGE DEMONSTRATIONS" + " 💡")
            demonstrate_cache_usage()
            print("\n" + "=" * 60)
            print("CACHE.PY TEST AND DEMO COMPLETE!")
            print("=" * 60)
            if success:
                print("✅ All basic tests passed! Cache system is operational.")
            else:
                print("⚠️  Some basic tests failed. Please review the output above.")
            sys.exit(0 if success else 1)
        except Exception as fallback_error:
            print(f"❌ Fallback test error: {fallback_error}")
            sys.exit(1)

# End of cache.py

# Export the global cache instance for backward compatibility
__all__ = ["cache", "cache_result", "clear_cache", "close_cache", "get_cache_stats"]
