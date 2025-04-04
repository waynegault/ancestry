#!/usr/bin/env python3

# cache.py

"""
This module provides a simple caching mechanism using the diskcache library.

It defines a `cache_result` decorator that can be used to cache the return
values of functions to disk.
"""

import os
import logging
from functools import wraps
from typing import Callable, Any
from diskcache import Cache
from config import config_instance

# Initialize logging
logger = logging.getLogger("logger")

# Define the cache directory
CACHE_DIR = config_instance.CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize the diskcache
cache = Cache(CACHE_DIR)


def cache_result(cache_key: str) -> Callable:
    """Decorator to cache function results using diskcache.

    Args:
        cache_key: The key to use for storing and retrieving the cached data.

    Returns:
        The decorated function.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if the result is already cached
            try:
                if cache_key in cache:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return cache[cache_key]
            except Exception as e:
                logger.error(f"Cache read error: {e}", exc_info=True)
                # Bypass cache if there's an error reading
            logger.debug(f"Cache miss for key: {cache_key}")  # Log cache miss
            try:
                # If not cached, execute the function and cache the result
                result = func(*args, **kwargs)
                cache[cache_key] = result
                logger.debug(f"Cached result for key: {cache_key}")
                return result
            except Exception as e:
                logger.error(
                    f"Error during function execution or caching: {e}", exc_info=True
                )
                # Important: Don't cache if there's an error
                return func(*args, **kwargs)  # Execute and return without caching

        return wrapper

    return decorator


# End of cache_result


# End
