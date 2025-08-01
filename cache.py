#!/usr/bin/env python3

"""
cache.py - Disk-Based Caching Utility

Provides a persistent caching mechanism using the `diskcache` library.
Includes a decorator (`@cache_result`) for easily caching function outputs
and utility functions for managing the cache lifecycle (clearing, closing).
Cache directory and default settings are configurable via `config.py`.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import (
    setup_module,
    register_function,
    get_function,
    is_function_available,
)

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

# --- Third-party imports ---
from diskcache import Cache

# Import constants used for checking cache misses vs. stored None values
from diskcache.core import ENOVAL, UNKNOWN

# --- Local application imports ---
from config import config_schema  # Use configured instance

# --- Test framework imports ---
from test_framework import (
    TestSuite,
    suppress_logging,
    create_mock_data,
    assert_valid_function,
    MagicMock,
)

# --- Global Cache Initialization ---

# Step 1: Define cache directory from configuration
if config_schema and getattr(config_schema.cache, "cache_dir", None):
    CACHE_DIR = Path(str(config_schema.cache.cache_dir))
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
    if not cache or not config_schema:
        return {
            "status": "skipped",
            "reason": "Cache or config not available",
            "entries_removed": 0,
        }

    try:
        current_size = get_cache_entry_count()
        max_size = getattr(config_schema.cache, "max_entries", 1000)

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
    if not cache or not config_schema:
        return True  # Allow if we can't check

    try:
        current_size = get_cache_entry_count()
        max_size = getattr(config_schema.cache, "max_entries", 1000)

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
            "entries": current_entry_count,
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
        if config_schema:
            max_entries = getattr(config_schema.cache, "max_entries", 1000)
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


def cache_module_tests() -> bool:
    """
    Run all cache tests and return True if successful.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    from test_framework import TestSuite

    def test_cache_initialization():
        """Test cache system initialization and configuration."""
        # Check if cache is initialized (may be None in some environments)
        cache_available = cache is not None
        if cache_available:
            assert hasattr(cache, "set"), "Cache should have set method"
            assert hasattr(cache, "get"), "Cache should have get method"
            assert hasattr(cache, "delete"), "Cache should have delete method"
            assert hasattr(cache, "clear"), "Cache should have clear method"
        return True

    def test_cache_interfaces():
        """Test cache interface classes."""
        base_module = BaseCacheModule()
        assert base_module.get_module_name() == "base_cache"

        # Test health status method
        health = base_module.get_health_status()
        assert isinstance(health, dict), "Health status should be dict"
        assert "health" in health, "Health status should contain 'health' key"
        assert "message" in health, "Health status should contain 'message' key"
        return True

    def test_basic_cache_operations():
        """Test fundamental cache set/get/delete operations."""
        if cache is None:
            return True  # Skip if cache not available

        # Test basic set/get
        test_key = "test_basic_ops"
        test_value = "test_value_123"

        cache.set(test_key, test_value)
        retrieved = cache.get(test_key)
        assert (
            retrieved == test_value
        ), f"Retrieved value {retrieved} doesn't match set value {test_value}"

        # Test delete
        cache.delete(test_key)
        retrieved_after_delete = cache.get(test_key)
        assert retrieved_after_delete is None, "Value should be None after deletion"
        return True

    def test_cache_decorator():
        """Test @cache_result decorator functionality."""
        call_count = 0

        @cache_result("test_decorator")
        def test_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        # First call should execute function
        result1 = test_function(1, 2)
        assert result1 == 3
        assert call_count == 1

        # Second call should use cache
        result2 = test_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Should not increment
        return True

    def test_cache_expiration():
        """Test cache TTL and expiration."""
        if cache is None:
            return True

        test_key = "test_expiration"
        test_value = "expires_soon"

        # Set with very short TTL
        cache.set(test_key, test_value, expire=0.1)  # 100ms

        # Should be available immediately
        assert cache.get(test_key) == test_value

        # Wait for expiration
        import time

        time.sleep(0.2)

        # Should be expired
        expired_value = cache.get(test_key)
        assert expired_value is None
        return True

    def test_cache_size_management():
        """Test cache size limits and eviction."""
        # This is a basic test - actual size management depends on diskcache config
        if cache is None:
            return True

        # Set multiple values
        for i in range(5):
            cache.set(f"size_test_{i}", f"value_{i}")

        # Verify they're stored
        for i in range(5):
            value = cache.get(f"size_test_{i}")
            if value is not None:  # May be evicted, that's okay
                assert value == f"value_{i}"
        return True

    def test_cache_clearing():
        """Test cache clearing functionality."""
        if cache is None:
            return True

        # Set some test data
        cache.set("clear_test_1", "data1")
        cache.set("clear_test_2", "data2")

        # Verify data exists
        assert cache.get("clear_test_1") == "data1"
        assert cache.get("clear_test_2") == "data2"

        # Clear cache
        cache.clear()

        # Verify data is gone
        assert cache.get("clear_test_1") is None
        assert cache.get("clear_test_2") is None
        return True

    def test_complex_data_types():
        """Test caching of complex data structures."""
        if cache is None:
            return True

        # Test dictionary
        test_dict = {"key1": "value1", "nested": {"key2": "value2"}}
        cache.set("test_dict", test_dict)
        retrieved_dict = cache.get("test_dict")
        assert retrieved_dict == test_dict

        # Test list
        test_list = [1, 2, {"nested": "value"}, [3, 4]]
        cache.set("test_list", test_list)
        retrieved_list = cache.get("test_list")
        assert retrieved_list == test_list
        return True

    def test_cache_performance():
        """Test cache performance and statistics."""
        if cache is None:
            return True

        # Performance test - basic operations should be fast
        import time

        start_time = time.time()

        for i in range(100):
            cache.set(f"perf_test_{i}", f"value_{i}")
            cache.get(f"perf_test_{i}")

        duration = time.time() - start_time
        assert (
            duration < 5.0
        ), f"100 cache operations took {duration}s, should be under 5s"
        return True

    def test_error_handling():
        """Test cache error handling and edge cases."""
        if cache is None:
            return True

        # Test with None values
        cache.set("test_none", None)
        retrieved_none = cache.get("test_none")
        # Note: this might be None due to the value OR due to key not found
        # The actual behavior depends on diskcache implementation

        # Test with empty string
        cache.set("test_empty", "")
        retrieved_empty = cache.get("test_empty")
        assert retrieved_empty == ""
        return True

    def test_health_monitoring():
        """Test cache health monitoring."""
        # Test module health functions
        if is_function_available("get_cache_stats"):
            stats_func = get_function("get_cache_stats")
            stats = stats_func()
            assert isinstance(stats, dict)

        # Test cache directory health
        if cache is not None:
            # Basic health check - cache should be operational
            cache.set("health_test", "healthy")
            assert cache.get("health_test") == "healthy"
        return True

    # Create test suite and run tests
    suite = TestSuite("Disk-Based Caching System", "cache.py")
    suite.start_suite()

    # Run all tests
    suite.run_test(
        "Cache System Initialization",
        test_cache_initialization,
        "Cache directory exists and cache object has required methods when available",
    )
    suite.run_test(
        "Cache Interface Classes",
        test_cache_interfaces,
        "BaseCacheModule interface works correctly with proper methods",
    )
    suite.run_test(
        "Basic Cache Operations",
        test_basic_cache_operations,
        "Fundamental set/get/delete operations work correctly",
    )
    suite.run_test(
        "Cache Decorator",
        test_cache_decorator,
        "@cache_result decorator caches function outputs correctly",
    )
    suite.run_test(
        "Cache Expiration", test_cache_expiration, "TTL and expiration work correctly"
    )
    suite.run_test(
        "Cache Size Management",
        test_cache_size_management,
        "Cache handles size limits and eviction properly",
    )
    suite.run_test(
        "Cache Clearing", test_cache_clearing, "Cache clearing removes all stored data"
    )
    suite.run_test(
        "Complex Data Types",
        test_complex_data_types,
        "Cache handles dictionaries, lists, and nested structures",
    )
    suite.run_test(
        "Cache Performance",
        test_cache_performance,
        "Cache operations perform within acceptable timeframes",
    )
    suite.run_test(
        "Error Handling",
        test_error_handling,
        "Cache handles edge cases and special values correctly",
    )
    suite.run_test(
        "Health Monitoring",
        test_health_monitoring,
        "Cache health monitoring functions work correctly",
    )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive cache tests using standardized TestSuite format."""
    return cache_module_tests()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
