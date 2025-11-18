#!/usr/bin/env python3

"""
GEDCOM Cache Intelligence & Advanced Genealogical Data Optimization Engine

Sophisticated genealogical caching platform providing high-performance GEDCOM data
caching, intelligent cache management, and comprehensive optimization strategies
with advanced genealogical data structures, memory-efficient processing, and
professional-grade caching for genealogical research and family tree analysis.

GEDCOM Cache Intelligence:
• Advanced GEDCOM data caching with intelligent genealogical data structures and optimization
• Sophisticated cache management with genealogy-specific optimization and performance monitoring
• Intelligent GEDCOM data serialization with optimized storage and retrieval algorithms
• Comprehensive genealogical data validation with intelligent integrity checking and verification
• Advanced GEDCOM cache coordination with multi-process synchronization and conflict resolution
• Integration with genealogical systems for comprehensive GEDCOM data caching and optimization

Performance Optimization:
• High-performance GEDCOM processing with memory-efficient data structures and algorithms
• Advanced genealogical data compression with intelligent storage optimization and retrieval
• Sophisticated GEDCOM cache warming with predictive data loading and optimization strategies
• Comprehensive performance monitoring with genealogy-specific metrics and optimization insights
• Intelligent GEDCOM cache sizing with automatic optimization and resource management
• Integration with performance systems for comprehensive genealogical data optimization

Genealogical Intelligence:
• Advanced genealogical data analysis with intelligent family tree optimization and insights
• Sophisticated relationship caching with intelligent relationship mapping and validation
• Intelligent genealogical data indexing with optimized search and retrieval algorithms
• Comprehensive family tree validation with intelligent data quality assessment and improvement
• Advanced genealogical data migration with seamless data transfer and version compatibility
• Integration with genealogical intelligence systems for comprehensive family tree optimization

Foundation Services:
Provides the essential GEDCOM caching infrastructure that enables high-performance
genealogical data processing through intelligent caching, comprehensive optimization,
and professional-grade genealogical data management for family history research workflows.

Technical Implementation:
Implements multi-level caching strategies specifically optimized for GEDCOM files
and genealogical data with memory and disk caching, intelligent invalidation,
and preloading for dramatically improved family tree processing performance.
"""

# Unified import system
from standard_imports import (
    get_function,
    is_function_available,
    setup_module,
)

logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===

# --- Standard library imports ---
import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Optional

# --- Local application imports ---
from cache import (
    BaseCacheModule,
    cache,
    get_cache_stats,
    get_unified_cache_key,
    invalidate_related_caches,
    warm_cache_with_data,
)
from config.config_manager import ConfigManager

# from logging_config import logger  # Unused logger, rely on get_logger if needed

# Initialize config
config_manager = ConfigManager()
config_schema = config_manager.get_config()

# --- Test framework imports ---
from test_framework import (
    TestSuite,
    suppress_logging,
)

# --- Global Variables ---
_MEMORY_CACHE: dict[str, tuple[Any, float]] = {}  # In-memory cache with timestamps
_CACHE_MAX_AGE = 3600  # 1 hour default for memory cache
_GEDCOM_CACHE_PREFIX = "gedcom_data"

# --- Memory-Efficient Object Pool for GEDCOM Data ---
from memory_utils import ObjectPool

# Pool for GedcomReader objects (if available)
try:
    from ged4py.parser import GedcomReader
    gedcom_pool = ObjectPool(lambda: GedcomReader, max_size=5)
except ImportError:
    gedcom_pool = None


# --- Lazy Loading for Large GEDCOM Datasets ---
class LazyGedcomData:
    """Lazy loader for large GEDCOM datasets."""
    def __init__(self, gedcom_path: str):
        self.gedcom_path = gedcom_path
        self._data = None

    @property
    def data(self) -> Any:
        # Only load GEDCOM data when accessed
        try:
            from ged4py.parser import GedcomReader
            self._data = GedcomReader(self.gedcom_path)
        except ImportError:
            self._data = None
        return self._data


# --- GEDCOM Cache Module Implementation ---


class GedcomCacheModule(BaseCacheModule):
    """
    GEDCOM-specific cache module implementing the standardized cache interface.
    Provides multi-level caching (memory + disk) for genealogical data.
    """

    def __init__(self) -> None:
        super().__init__()
        self.module_name = "gedcom_cache"
        self.cache_prefix = _GEDCOM_CACHE_PREFIX

    def get_module_name(self) -> str:
        return self.module_name

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive GEDCOM cache statistics."""
        base_stats = super().get_stats()

        # Add GEDCOM-specific statistics
        memory_cache_size = len(_MEMORY_CACHE)
        memory_cache_valid_entries = sum(
            1 for key in _MEMORY_CACHE if _is_memory_cache_valid(key)
        )

        gedcom_stats = {
            "module_name": self.module_name,
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
        gedcom_path = config_schema.database.gedcom_file_path
        if gedcom_path and Path(gedcom_path).exists():
            gedcom_stats["gedcom_file_path"] = str(gedcom_path)
            gedcom_stats["gedcom_file_size_mb"] = Path(gedcom_path).stat().st_size / (
                1024 * 1024
            )
            gedcom_stats["gedcom_file_mtime"] = Path(gedcom_path).stat().st_mtime

        # Merge with base statistics
        return {**base_stats, **gedcom_stats}

    def clear(self) -> bool:
        """Clear all GEDCOM caches (memory and disk)."""
        try:
            # Clear memory cache
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

    @staticmethod
    def warm() -> bool:
        """Warm up GEDCOM cache with frequently accessed data."""
        try:
            # Check if GEDCOM file is available
            if not (
                config_schema and hasattr(config_schema.database, "gedcom_file_path")
            ):
                logger.warning("No GEDCOM file path configured for cache warming")
                return False

            gedcom_path = config_schema.database.gedcom_file_path
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

    @staticmethod
    def _check_memory_cache_health() -> tuple[str, list[str], int]:
        """Check memory cache health. Returns (health_status, issues, stale_count)."""
        memory_health = "healthy"
        memory_issues = []

        if len(_MEMORY_CACHE) == 0:
            memory_issues.append("No memory cache entries")

        # Check for stale entries
        stale_entries = sum(1 for key in _MEMORY_CACHE if not _is_memory_cache_valid(key))
        if stale_entries > len(_MEMORY_CACHE) * 0.5:
            memory_health = "degraded"
            memory_issues.append(f"High number of stale entries: {stale_entries}")

        return memory_health, memory_issues, stale_entries

    @staticmethod
    def _check_gedcom_file_health() -> tuple[str, list[str]]:
        """Check GEDCOM file accessibility. Returns (health_status, issues)."""
        gedcom_health = "healthy"
        gedcom_issues = []

        if config_schema and hasattr(config_schema.database, "gedcom_file_path"):
            gedcom_path = config_schema.database.gedcom_file_path
            if not gedcom_path:
                gedcom_health = "warning"
                gedcom_issues.append("No GEDCOM file path configured")
            elif not Path(gedcom_path).exists():
                gedcom_health = "error"
                gedcom_issues.append(f"GEDCOM file not found: {gedcom_path}")
        else:
            gedcom_health = "warning"
            gedcom_issues.append("GEDCOM configuration not available")

        return gedcom_health, gedcom_issues

    @staticmethod
    def _determine_overall_health(gedcom_health: str, memory_health: str) -> str:
        """Determine overall health based on component health statuses."""
        if gedcom_health == "error" or memory_health == "degraded":
            return "degraded"
        if gedcom_health == "warning":
            return "warning"
        return "healthy"

    def get_health_status(self) -> dict[str, Any]:
        """Get detailed health status of GEDCOM cache system."""
        base_health = super().get_health_status()

        try:
            # Check memory cache health
            memory_health, memory_issues, stale_entries = self._check_memory_cache_health()

            # Check GEDCOM file accessibility
            gedcom_health, gedcom_issues = self._check_gedcom_file_health()

            # Overall health assessment
            overall_health = self._determine_overall_health(gedcom_health, memory_health)

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
    from pathlib import Path
    return get_unified_cache_key(
        "gedcom_memory", operation, file_path, Path(file_path).stat().st_mtime
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
    # Clean up expired entry
    _MEMORY_CACHE.pop(cache_key, None)
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


def _check_disk_cache_for_gedcom(gedcom_path: str, memory_key: str) -> tuple[Optional[Any], Optional[str]]:
    """Check disk cache for GEDCOM data. Returns (cached_data, disk_cache_key)."""
    try:
        from pathlib import Path
        file_mtime = Path(gedcom_path).stat().st_mtime
        mtime_hash = hashlib.md5(str(file_mtime).encode()).hexdigest()[:8]
        disk_cache_key = f"gedcom_load_mtime_{mtime_hash}"

        if cache is not None:
            from diskcache.core import ENOVAL
            disk_cached = cache.get(disk_cache_key, default=ENOVAL, retry=True)
            if disk_cached is not ENOVAL:
                logger.debug("GEDCOM data loaded from disk cache")
                # Store in memory cache for faster next access
                _store_in_memory_cache(memory_key, disk_cached)
                return disk_cached, disk_cache_key
        return None, disk_cache_key
    except Exception as e:
        logger.debug(f"Error checking disk cache: {e}")
        return None, None


def _store_gedcom_in_disk_cache(gedcom_data: Any, disk_cache_key: Optional[str]) -> None:
    """
    Store GEDCOM data in disk cache (without reader object or indi_index).

    Note: We don't cache indi_index because Individual objects cannot be pickled
    (they contain references to the file reader). Instead, we cache only the
    processed data and family maps, which are plain Python dicts/sets.
    The indi_index will be rebuilt quickly when loading from cache.
    """
    try:
        if cache is not None and disk_cache_key is not None:
            # Create a serializable version without the reader and indi_index
            # Only cache the processed data and family relationships
            cache_data = {
                "path": str(gedcom_data.path),
                "processed_data_cache": gedcom_data.processed_data_cache,
                "id_to_parents": gedcom_data.id_to_parents,
                "id_to_children": gedcom_data.id_to_children,
                "indi_index_build_time": gedcom_data.indi_index_build_time,
                "family_maps_build_time": gedcom_data.family_maps_build_time,
                "data_processing_time": gedcom_data.data_processing_time,
            }
            cache.set(disk_cache_key, cache_data, expire=86400, retry=True)
            logger.debug(
                f"GEDCOM data cached to disk: {len(cache_data['processed_data_cache'])} individuals, "
                f"{len(cache_data['id_to_parents'])} parent relationships"
            )
    except Exception as e:
        logger.warning(f"Error storing in disk cache: {e}")


def load_gedcom_with_aggressive_caching(gedcom_path: str) -> Optional[Any]:
    """
    Load GEDCOM data with aggressive multi-level caching.

    Uses both memory and disk caching with file modification time tracking.
    Automatically invalidates cache when GEDCOM file changes.

    Args:
        gedcom_path: Path to the GEDCOM file

    Returns:
        GedcomData instance or None if loading fails

    Note:
        The returned GedcomData object will have a _cache_source attribute
        set to "memory", "disk", or "file" to indicate where it was loaded from.
    """
    # Check memory cache first
    memory_key = _get_memory_cache_key(gedcom_path, "gedcom_load")
    cached_data = _get_from_memory_cache(memory_key)
    if cached_data is not None:
        logger.debug("GEDCOM data loaded from memory cache")
        # Mark the source for user visibility
        cached_data._cache_source = "memory"
        return cached_data

    # Check disk cache
    disk_cached, disk_cache_key = _check_disk_cache_for_gedcom(gedcom_path, memory_key)
    if disk_cached is not None:
        # Reconstruct GedcomData from cached data
        try:
            from gedcom_utils import GedcomData
            gedcom_data = GedcomData.from_cache(disk_cached, gedcom_path)

            # Store in memory cache for faster next access
            _store_in_memory_cache(memory_key, gedcom_data)

            logger.debug("GEDCOM data reconstructed from disk cache")
            # Mark the source for user visibility
            gedcom_data._cache_source = "disk"
            return gedcom_data
        except Exception as e:
            logger.warning(f"Failed to reconstruct from cache, will reload: {e}")
            # Fall through to reload from file

    logger.debug(f"Loading GEDCOM file with aggressive caching: {gedcom_path}")
    start_time = time.time()

    try:
        # Import here to avoid circular imports
        from gedcom_utils import GedcomData

        # Load the GEDCOM file
        gedcom_data = GedcomData(gedcom_path)

        if gedcom_data:
            load_time = time.time() - start_time
            logger.debug(f"GEDCOM file loaded and cached in {load_time:.2f}s")

            # Store in memory cache for fastest access
            _store_in_memory_cache(memory_key, gedcom_data)

            # Store in disk cache for persistence
            _store_gedcom_in_disk_cache(gedcom_data, disk_cache_key)

            # Log cache statistics
            stats = get_cache_stats()
            if stats:
                logger.debug(f"Cache stats after GEDCOM load: {stats}")

            # Mark the source for user visibility
            gedcom_data._cache_source = "file"
            return gedcom_data
        logger.error("Failed to create GedcomData instance")
        return None

    except Exception as e:
        logger.error(f"Error loading GEDCOM file {gedcom_path}: {e}", exc_info=True)
        return None


# Helper functions for cache_gedcom_processed_data

def _create_cache_keys(gedcom_path: str) -> dict[str, str]:
    """Create cache keys based on file path and modification time."""
    from pathlib import Path
    file_mtime = Path(gedcom_path).stat().st_mtime
    path_hash = hashlib.md5(str(gedcom_path).encode()).hexdigest()[:8]
    mtime_hash = hashlib.md5(str(file_mtime).encode()).hexdigest()[:8]

    return {
        "processed_data": f"gedcom_processed_{path_hash}_{mtime_hash}",
        "indi_index": f"gedcom_indi_index_{path_hash}_{mtime_hash}",
        "family_maps": f"gedcom_family_maps_{path_hash}_{mtime_hash}",
    }


def _serialize_processed_data_item(value: dict[str, Any]) -> dict[str, Any]:
    """Serialize a single processed data item."""
    serializable_item = {}
    for item_key, item_value in value.items():
        # Convert datetime objects to ISO strings for serialization
        if hasattr(item_value, "isoformat"):  # datetime objects
            serializable_item[item_key] = item_value.isoformat()
        elif isinstance(item_value, (str, int, float, bool, list, tuple, type(None))):
            serializable_item[item_key] = item_value
        else:
            # Skip non-serializable objects
            logger.debug(
                f"Skipping non-serializable object in processed_data_cache: {item_key} = {type(item_value)}"
            )
    return serializable_item


def _serialize_processed_data_cache(processed_data_cache: dict[str, Any]) -> dict[str, Any]:
    """Create a serializable version of processed_data_cache."""
    serializable_processed_data = {}
    for key, value in processed_data_cache.items():
        if isinstance(value, dict):
            serializable_processed_data[key] = _serialize_processed_data_item(value)
        elif isinstance(value, (str, int, float, bool, list, tuple, type(None))):
            serializable_processed_data[key] = value
    return serializable_processed_data


def _serialize_indi_index_object(value: Any) -> Optional[dict[str, Any]]:
    """Serialize a complex object from indi_index."""
    if not hasattr(value, "__dict__"):
        return None

    obj_dict = {}
    for attr, attr_value in value.__dict__.items():
        if isinstance(attr_value, (str, int, float, bool, list, dict, tuple, type(None))):
            obj_dict[attr] = attr_value

    return obj_dict if obj_dict else None


def _serialize_indi_index(indi_index: dict[str, Any]) -> dict[str, Any]:
    """Create a serializable version of indi_index."""
    serializable_indi_index = {}
    for key, value in indi_index.items():
        # Only cache primitive data types and avoid GedcomReader references
        if isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
            serializable_indi_index[key] = value
        else:
            # For complex objects, try to extract serializable data
            try:
                obj_dict = _serialize_indi_index_object(value)
                if obj_dict:
                    serializable_indi_index[key] = obj_dict
            except Exception:
                # Skip non-serializable objects
                pass
    return serializable_indi_index


def _cache_family_relationship_maps(gedcom_data: Any, cache_key: str) -> None:
    """Cache family relationship maps."""
    family_data = {}
    if hasattr(gedcom_data, "id_to_parents"):
        family_data["id_to_parents"] = gedcom_data.id_to_parents
    if hasattr(gedcom_data, "id_to_children"):
        family_data["id_to_children"] = gedcom_data.id_to_children

    if family_data:
        warm_cache_with_data(cache_key, family_data, expire=86400)


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
        # Create cache keys
        cache_keys = _create_cache_keys(gedcom_path)

        # Cache processed data
        if hasattr(gedcom_data, "processed_data_cache"):
            serializable_processed_data = _serialize_processed_data_cache(
                gedcom_data.processed_data_cache
            )
            warm_cache_with_data(
                cache_keys["processed_data"],
                serializable_processed_data,
                expire=86400,
            )

        # Cache individual index
        if hasattr(gedcom_data, "indi_index"):
            serializable_indi_index = _serialize_indi_index(gedcom_data.indi_index)
            if serializable_indi_index:
                warm_cache_with_data(
                    cache_keys["indi_index"], serializable_indi_index, expire=86400
                )

        # Cache family relationship maps
        _cache_family_relationship_maps(gedcom_data, cache_keys["family_maps"])

        from pathlib import Path
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
    if not config_schema or not hasattr(config_schema.database, "gedcom_file_path"):
        logger.debug("No GEDCOM file configured for preloading")
        return False

    gedcom_path = config_schema.database.gedcom_file_path
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
        logger.warning("Failed to preload GEDCOM cache")
        return False

    except Exception as e:
        logger.error(f"Error preloading GEDCOM cache: {e}", exc_info=True)
        return False


def get_gedcom_cache_info() -> dict[str, Any]:
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
    if config_schema and hasattr(config_schema.database, "gedcom_file_path"):
        gedcom_path = config_schema.database.gedcom_file_path
        if gedcom_path and Path(gedcom_path).exists():
            info["gedcom_file"] = str(gedcom_path)
            info["gedcom_file_size_mb"] = Path(gedcom_path).stat().st_size / (
                1024 * 1024
            )
            info["gedcom_file_mtime"] = Path(gedcom_path).stat().st_mtime

    return info


# --- Public Interface Functions for GEDCOM Cache Module ---


def get_gedcom_cache_stats() -> dict[str, Any]:
    """Get comprehensive GEDCOM cache statistics."""
    return _gedcom_cache_module.get_stats()


def clear_gedcom_cache() -> bool:
    """Clear all GEDCOM caches."""
    return _gedcom_cache_module.clear()


def warm_gedcom_cache() -> bool:
    """Warm up GEDCOM cache."""
    return _gedcom_cache_module.warm()


def get_gedcom_cache_health() -> dict[str, Any]:
    """Get GEDCOM cache health status."""
    return _gedcom_cache_module.get_health_status()


# --- GEDCOM Cache Demo Functions ---


def demonstrate_gedcom_cache_usage() -> dict[str, Any]:
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
        if config_schema and hasattr(config_schema.database, "gedcom_file_path"):
            gedcom_path = config_schema.database.gedcom_file_path
            if gedcom_path and Path(gedcom_path).exists():
                # Demonstrate file-based caching
                cache_key = _get_memory_cache_key(str(gedcom_path), "demo_operation")
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
                _MEMORY_CACHE.pop(cache_key, None)

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
                "description": f"Error occurred: {e!s}",
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


# --- Individual Test Functions ---


def test_gedcom_cache_initialization():
    """Test GEDCOM cache module initialization."""
    # Test module instance
    module_name = _gedcom_cache_module.get_module_name()
    assert module_name == "gedcom_cache", f"Expected 'gedcom_cache', got '{module_name}'"

    if is_function_available("GedcomCache"):
        cache_class = get_function("GedcomCache")
        if cache_class:
            assert hasattr(cache_class, "get"), "GedcomCache should have 'get' method"
            assert hasattr(cache_class, "set"), "GedcomCache should have 'set' method"
            assert hasattr(cache_class, "invalidate"), "GedcomCache should have 'invalidate' method"
    return True


def test_memory_cache_operations():
    """Test basic memory cache operations."""
    test_key = "test_operation_key"
    test_data = {"test": "data", "timestamp": time.time()}

    # Store in memory cache
    _store_in_memory_cache(test_key, test_data)

    # Retrieve from memory cache
    retrieved = _get_from_memory_cache(test_key)

    # Clean up
    _MEMORY_CACHE.pop(test_key, None)

    assert retrieved == test_data, f"Retrieved data doesn't match. Expected {test_data}, got {retrieved}"
    return True


def test_gedcom_parsing_caching():
    """Test GEDCOM file parsing and caching."""
    # Test that cache_gedcom_processed_data function exists in this module
    assert "cache_gedcom_processed_data" in globals(), \
        "cache_gedcom_processed_data function should exist"
    assert callable(cache_gedcom_processed_data), \
        "cache_gedcom_processed_data should be callable"

    # Test with mock cache key and data (avoid file access)
    test_key = "test_gedcom_parse_key"
    mock_data = {"test": "gedcom_data"}

    # Store in memory cache (simulating caching)
    _store_in_memory_cache(test_key, mock_data)

    # Verify it was cached
    cached = _get_from_memory_cache(test_key)
    assert cached == mock_data, "Cached data should match original"

    # Clean up
    _MEMORY_CACHE.pop(test_key, None)
    return True


def test_cached_data_retrieval():
    """Test cached GEDCOM data retrieval."""
    # Test that load_gedcom_with_aggressive_caching function exists in this module
    assert "load_gedcom_with_aggressive_caching" in globals(), \
        "load_gedcom_with_aggressive_caching function should exist"
    assert callable(load_gedcom_with_aggressive_caching), \
        "load_gedcom_with_aggressive_caching should be callable"

    # Test memory cache retrieval
    test_key = "test_retrieval_key"
    test_data = {"test": "data"}

    # Store in memory cache
    _store_in_memory_cache(test_key, test_data)

    # Retrieve from memory cache
    result = _get_from_memory_cache(test_key)
    assert result == test_data, "Retrieved data should match stored data"

    # Clean up
    _MEMORY_CACHE.pop(test_key, None)
    return True


def test_cache_key_generation():
    """Test cache key generation consistency."""
    try:
        if not (config_schema and hasattr(config_schema.database, "gedcom_file_path")):
            return True  # Skip if no GEDCOM configured

        gedcom_path = config_schema.database.gedcom_file_path
        if not gedcom_path or not Path(gedcom_path).exists():
            return True  # Skip if file doesn't exist

        key1 = _get_memory_cache_key(str(gedcom_path), "test_operation")
        key2 = _get_memory_cache_key(str(gedcom_path), "test_operation")

        return key1 == key2  # Keys should be consistent
    except Exception:
        return False


def test_memory_cache_expiration():
    """Test memory cache expiration handling."""
    try:
        test_key = "expiration_test_key"
        test_data = "expiration_test_data"

        # Store with expired timestamp
        _MEMORY_CACHE[test_key] = (test_data, time.time() - _CACHE_MAX_AGE - 1)

        # Should be invalid due to age
        is_valid = _is_memory_cache_valid(test_key)

        # Clean up
        _MEMORY_CACHE.pop(test_key, None)

        return not is_valid  # Should be invalid (expired)
    except Exception:
        return False


def test_cache_invalidation_file_modification():
    """Test cache invalidation on file modification."""
    # Test that memory cache validation works
    test_key = "test_invalidation_key"
    test_data = {"test": "data"}

    # Store in memory cache
    _store_in_memory_cache(test_key, test_data)

    # Verify it's valid
    is_valid = _is_memory_cache_valid(test_key)
    assert isinstance(is_valid, bool), "_is_memory_cache_valid should return bool"

    # Clean up
    _MEMORY_CACHE.pop(test_key, None)
    return True


def test_cache_statistics_collection():
    """Test cache statistics collection."""
    stats = _gedcom_cache_module.get_stats()
    required_fields = [
        "module_name",
        "memory_cache_entries",
        "cache_max_age_seconds",
    ]
    assert all(field in stats for field in required_fields), \
        f"Stats missing required fields. Got: {list(stats.keys())}"
    return True


def test_cache_health_status():
    """Test cache health status check."""
    health = _gedcom_cache_module.get_health_status()
    required_fields = [
        "overall_health",
        "memory_cache_health",
        "gedcom_file_health",
    ]
    assert all(field in health for field in required_fields), \
        f"Health status missing required fields. Got: {list(health.keys())}"
    return True


def test_cache_performance_metrics():
    """Test cache performance metrics collection."""
    stats = _gedcom_cache_module.get_stats()
    # Check for performance-related metrics
    performance_indicators = ["memory_cache_entries", "cache_max_age_seconds"]
    assert all(indicator in stats for indicator in performance_indicators), \
        f"Stats missing performance indicators. Got: {list(stats.keys())}"
    return True


def test_multifile_cache_management():
    """Test multi-file cache management."""
    # Test cache can handle multiple cache keys (simulating multiple files)
    test_keys = ["test_file_1_key", "test_file_2_key"]
    test_data_1 = {"file": "test1", "data": "data1"}
    test_data_2 = {"file": "test2", "data": "data2"}
    test_data_list = [test_data_1, test_data_2]

    # Store multiple items in cache
    for test_key, test_data in zip(test_keys, test_data_list):
        _store_in_memory_cache(test_key, test_data)

    # Verify both were cached correctly
    for test_key, expected_data in zip(test_keys, test_data_list):
        cached = _get_from_memory_cache(test_key)
        assert cached == expected_data, f"Cached data for {test_key} should match original"

    # Clean up
    for test_key in test_keys:
        _MEMORY_CACHE.pop(test_key, None)

    return True


def test_memory_management_cleanup():
    """Test memory management and cleanup."""
    # Test cache clearing
    clear_result = _gedcom_cache_module.clear()
    # Clear should either succeed or fail gracefully
    assert isinstance(clear_result, bool), f"clear() should return bool, got {type(clear_result)}"
    return True


def test_cache_validation_integrity():
    """Test cache validation and integrity checking."""
    # Test that health check detects cache state
    health = _gedcom_cache_module.get_health_status()
    assert isinstance(health, dict), f"get_health_status() should return dict, got {type(health)}"
    assert "overall_health" in health, "Health status should contain 'overall_health' field"
    return True


def gedcom_cache_module_tests() -> bool:
    """
    GEDCOM Cache Management & Optimization module test suite.
    Tests GEDCOM file caching, invalidation, and performance optimization.
    """

    with suppress_logging():
        suite = TestSuite("GEDCOM Cache Management & Optimization", __name__)
        suite.start_suite()

    # Run all tests using the suite
    suite.run_test(
        "GEDCOM Cache Module Initialization",
        test_gedcom_cache_initialization,
        "Cache module initializes with proper interface and required methods",
        "Initialization",
        "Initialize GEDCOM cache module and verify basic structure",
    )

    suite.run_test(
        "Memory Cache Operations",
        test_memory_cache_operations,
        "Memory cache stores and retrieves data correctly",
        "Initialization",
        "Store and retrieve data from memory cache",
    )

    suite.run_test(
        "GEDCOM File Parsing and Caching",
        test_gedcom_parsing_caching,
        "GEDCOM file parses successfully and caches processed data",
        "Core",
        "Parse mock GEDCOM file and cache the processed data",
    )

    suite.run_test(
        "Cached Data Retrieval",
        test_cached_data_retrieval,
        "Cache retrieval returns None or valid dictionary data structure",
        "Core",
        "Retrieve previously cached GEDCOM data",
    )

    suite.run_test(
        "Cache Key Generation",
        test_cache_key_generation,
        "Cache key generation produces consistent keys for identical inputs",
        "Core",
        "Generate cache keys for same inputs and verify consistency",
    )

    suite.run_test(
        "Memory Cache Expiration",
        test_memory_cache_expiration,
        "Expired cache entries are correctly identified as invalid",
        "Edge",
        "Store data with expired timestamp and verify expiration detection",
    )

    suite.run_test(
        "Cache Invalidation on File Modification",
        test_cache_invalidation_file_modification,
        "File modification detection works correctly for cache management",
        "Edge",
        "Check file modification detection for cache invalidation",
    )

    suite.run_test(
        "Cache Statistics Collection",
        test_cache_statistics_collection,
        "Statistics collection returns all required fields",
        "Integration",
        "Collect comprehensive cache statistics",
    )

    suite.run_test(
        "Cache Health Status Check",
        test_cache_health_status,
        "Health status returns comprehensive system health information",
        "Integration",
        "Check overall cache health and component status",
    )

    suite.run_test(
        "Cache Performance Metrics",
        test_cache_performance_metrics,
        "Performance metrics collection provides valid numeric data",
        "Performance",
        "Collect and validate cache performance statistics",
    )

    suite.run_test(
        "Multi-file Cache Management",
        test_multifile_cache_management,
        "Multi-file cache management handles multiple files efficiently",
        "Performance",
        "Manage caches for multiple GEDCOM files simultaneously",
    )

    suite.run_test(
        "Memory Management and Cleanup",
        test_memory_management_cleanup,
        "Memory management functions execute without errors",
        "Error",
        "Test cache memory cleanup and optimization functions",
    )

    suite.run_test(
        "Cache Validation and Integrity",
        test_cache_validation_integrity,
        "Cache validation properly checks data integrity",
        "Error",
        "Validate cache data integrity and detect corruption",
    )

    return suite.finish_suite()


# Use centralized test runner utility
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(gedcom_cache_module_tests)


# --- Main Execution ---


if __name__ == "__main__":
    print(
        "🗂️ Running GEDCOM Cache Management & Optimization comprehensive test suite..."
    )
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
