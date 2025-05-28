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
)
from config import config_instance
from logging_config import logger

# --- Global Variables ---
_MEMORY_CACHE: Dict[str, Tuple[Any, float]] = {}  # In-memory cache with timestamps
_CACHE_MAX_AGE = 3600  # 1 hour default for memory cache
_GEDCOM_CACHE_PREFIX = "gedcom_data"


# --- Memory Cache Management ---


def _get_memory_cache_key(file_path: str, operation: str) -> str:
    """Generate a consistent cache key for memory cache."""
    file_mtime = os.path.getmtime(file_path)
    path_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
    mtime_hash = hashlib.md5(str(file_mtime).encode()).hexdigest()[:8]
    return f"{operation}_{path_hash}_{mtime_hash}"


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


# End of gedcom_cache.py
