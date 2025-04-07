#!/usr/bin/env python3

# cache.py

"""
This module provides a caching mechanism using the diskcache library.

It defines a `cache_result` decorator and utility functions for managing
the application cache.
"""

import atexit
import logging
import os
import shutil  # Added for clear_cache
from functools import wraps
from typing import Any, Callable, Optional

# Third-party imports
from diskcache import Cache
from diskcache.core import ENOVAL, UNKNOWN  # Added for type checking cache misses

# Local application imports
from config import config_instance

# Initialize logging
logger = logging.getLogger("logger")

# Define the cache directory from configuration
CACHE_DIR = config_instance.CACHE_DIR
# Ensure the directory exists
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    logger.info(f"Cache directory set to: {CACHE_DIR}")
except OSError as e:
    logger.error(f"Failed to create cache directory {CACHE_DIR}: {e}", exc_info=True)
    # Fallback or raise? For now, let Cache handle potential errors later.
except Exception as e:
    logger.error(
        f"Unexpected error setting up cache directory {CACHE_DIR}: {e}", exc_info=True
    )


# Initialize the diskcache instance globally
# Consider adding size limits or eviction policies if needed
# e.g., cache = Cache(CACHE_DIR, size_limit=int(1e9)) # 1 GB limit
try:
    # Use expire parameter globally if desired (e.g., expire=config_instance.CACHE_TIMEOUT)
    # Use timeout parameter for connection timeout if cache is remote (not applicable here)
    cache: Optional[Cache] = Cache(
        CACHE_DIR,
        # expire=config_instance.CACHE_TIMEOUT # Example: Set default expiry from config
    )
    logger.info(f"DiskCache initialized at {CACHE_DIR}.")
except Exception as e:
    logger.critical(
        f"CRITICAL: Failed to initialize DiskCache at {CACHE_DIR}: {e}", exc_info=True
    )
    cache = None  # Ensure cache is None if initialization fails


def cache_result(
    cache_key_prefix: str,  # Use a prefix for clarity
    expire: Optional[int] = None,  # Allow overriding default expiry per decorator
    ignore_args: bool = False,  # Option to ignore function arguments for key generation
) -> Callable:
    """
    Decorator to cache function results using diskcache.

    Args:
        cache_key_prefix: A prefix for the cache key. The final key will often
                          include function name and arguments unless ignore_args=True.
        expire: Specific expiry time in seconds for this cache entry. Overrides
                any global default set on the Cache instance.
        ignore_args: If True, use only the prefix as the key, ignoring function
                     arguments. Useful for caching global state fetches.

    Returns:
        The decorated function.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if cache is None:
                logger.error("Cache is not initialized. Bypassing cache.")
                return func(*args, **kwargs)

            # --- Generate Cache Key ---
            if ignore_args:
                final_cache_key = cache_key_prefix
            else:
                # Create a stable representation of args and kwargs for the key
                # Note: This relies on args/kwargs having a stable str() representation.
                # For complex objects, a more robust serialization might be needed.
                arg_key_part = f"_args{str(args)}_kwargs{str(sorted(kwargs.items()))}"
                # Combine prefix, function name, and args/kwargs representation
                final_cache_key = f"{cache_key_prefix}_{func.__name__}{arg_key_part}"
            # --- End Key Generation ---

            # Check if the result is already cached
            try:
                # Use get with default=ENOVAL to distinguish miss from None value
                cached_value = cache.get(final_cache_key, default=ENOVAL, retry=True)
                if cached_value is not ENOVAL:
                    logger.debug(f"Cache hit for key: {final_cache_key}")
                    return cached_value
                else:
                    # Only log miss if the value was truly not found
                    logger.debug(f"Cache miss for key: {final_cache_key}")

            except Exception as e:
                # Log errors during cache read but proceed as if it's a miss
                logger.error(
                    f"Cache read error for key '{final_cache_key}': {e}", exc_info=True
                )
                # Fall through to execute the function

            # If not cached or read error, execute the function
            try:
                result = func(*args, **kwargs)
                # Cache the result with optional specific expiry
                cache.set(final_cache_key, result, expire=expire, retry=True)
                logger.debug(
                    f"Cached result for key: {final_cache_key} (Expire: {expire}s)"
                )
                return result
            except Exception as e:
                logger.error(
                    f"Error during function execution or caching for key '{final_cache_key}': {e}",
                    exc_info=True,
                )
                # Do NOT cache if the function failed.
                # Re-raise the original error so the caller knows something went wrong.
                # Avoids returning potentially incorrect data or masking errors.
                raise  # Reraise the exception from func or cache.set

        return wrapper

    return decorator


# End of cache_result


def clear_cache():
    """Removes all items from the cache."""
    if cache:
        try:
            count = cache.clear()
            logger.info(f"Cache cleared. {count} items removed.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}", exc_info=True)
            return False
    else:
        logger.warning("Cache not initialized, cannot clear.")
        # Attempt to manually remove directory if cache object failed init
        if os.path.exists(CACHE_DIR):
            try:
                shutil.rmtree(CACHE_DIR)
                logger.info(f"Manually removed cache directory: {CACHE_DIR}")
                # Recreate the directory after clearing
                os.makedirs(CACHE_DIR, exist_ok=True)
                return True
            except Exception as e:
                logger.error(
                    f"Failed to manually remove cache directory {CACHE_DIR}: {e}",
                    exc_info=True,
                )
                return False
        return False


def close_cache():
    """Closes the cache connection."""
    if cache:
        try:
            cache.close()
            logger.info("DiskCache connection closed.")
        except Exception as e:
            logger.error(f"Error closing cache: {e}", exc_info=True)


# Register close_cache to be called on script exit
atexit.register(close_cache)

logger.debug("cache.py loaded.")

# --- End of cache.py ---
