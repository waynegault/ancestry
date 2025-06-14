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
from test_framework import (
    TestSuite,
    suppress_logging,
    create_mock_data,
    assert_valid_function,
)

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
    from test_framework import TestSuite, suppress_logging
    import tempfile
    import time

    suite = TestSuite("Core Cache System & Decorators", "cache.py")
    suite.start_suite()

    # Cache decorator functionality
    def test_cache_decorator():
        # Ensure cache is available for testing
        if cache is None:
            # Skip test if cache is not available
            return True

        # Clear any existing cache entries for this test
        try:
            cache.clear()
        except:
            pass  # Ignore errors if cache can't be cleared

        call_count = 0

        @cache_result("test_expensive_function")
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call should execute function
        result1 = expensive_function(5)
        assert result1 == 10, f"Expected 10, got {result1}"
        assert (
            call_count == 1
        ), f"Expected call_count=1 after first call, got {call_count}"

        # Second call should use cache
        result2 = expensive_function(5)
        assert result2 == 10, f"Expected 10, got {result2}"
        assert (
            call_count == 1
        ), f"Expected call_count=1 after second call (cached), got {call_count}"

    # Cache key generation
    def test_cache_key_generation():
        # Test with get_unified_cache_key function
        key1 = get_unified_cache_key("test_module", "test_op", 1, 2, kwarg1="value1")
        key2 = get_unified_cache_key("test_module", "test_op", 1, 2, kwarg1="value1")
        key3 = get_unified_cache_key("test_module", "test_op", 1, 3, kwarg1="value1")

        assert key1 == key2  # Same arguments should generate same key
        assert key1 != key3  # Different arguments should generate different keys

    # Advanced cache features
    def test_advanced_cache_features():
        # Test actual cache management functions
        real_functions = [
            "clear_cache",
            "close_cache",
            "get_cache_stats",
            "get_cache_entry_count",
            "invalidate_related_caches",
        ]

        for func_name in real_functions:
            assert func_name in globals(), f"Function {func_name} should exist"
            func = globals()[func_name]
            assert callable(func), f"Function {func_name} should be callable"

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
        "Advanced cache features": (
            test_advanced_cache_features,
            "Should support advanced caching patterns and strategies",
        ),
    }

    with suppress_logging():
        for test_name, (test_func, expected_behavior) in test_functions.items():
            suite.run_test(test_name, test_func, expected_behavior)

    return suite.finish_suite()


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
        print("🔄 Unable to run comprehensive tests...")
        print("\n" + "💡 " + "RUNNING USAGE DEMONSTRATIONS" + " 💡")
        demonstrate_cache_usage()
        print("\n" + "=" * 60)
        print("CACHE.PY DEMO COMPLETE!")
        print("=" * 60)
        print("⚠️  Basic tests could not be run. Please review the error above.")
        sys.exit(1)

# End of cache.py

# Export the global cache instance for backward compatibility
__all__ = ["cache", "cache_result", "clear_cache", "close_cache", "get_cache_stats"]
