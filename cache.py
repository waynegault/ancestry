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
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, Dict, Union

# --- Third-party imports ---
from diskcache import Cache

# Import constants used for checking cache misses vs. stored None values
from diskcache.core import ENOVAL, UNKNOWN

# --- Local application imports ---
from config import config_instance  # Use configured instance
from logging_config import logger  # Use configured logger

# --- Global Cache Initialization ---

# Step 1: Define cache directory from configuration
if config_instance:
    CACHE_DIR = config_instance.CACHE_DIR
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

                # Step 5: Store the result in the cache
                try:
                    cache.set(final_cache_key, result, expire=expire, retry=True)
                    expire_msg = (
                        f"with expiry {expire}s"
                        if expire is not None
                        else "with default expiry"
                    )
                    logger.debug(
                        f"Cached result for key: '{final_cache_key}' {expire_msg}."
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


# --- Enhanced Cache Management Functions ---


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
        stats = {
            "hits": stats_tuple[0],
            "misses": stats_tuple[1],
            "size": cache.count,  # Use count instead of len for Cache objects
            "volume": cache.volume(),
            "evictions": getattr(cache, "evictions", 0),
        }
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

# --- End of cache.py ---
