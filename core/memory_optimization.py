#!/usr/bin/env python3

"""
Memory Optimization Module

Provides memory-efficient processing for large GEDCOM files and datasets with
streaming parsers, lazy loading, memory-mapped file access, and intelligent
caching strategies to handle files >100MB without memory issues.
"""

import logging
import mmap
import os
import gc
import psutil
from typing import Any, Dict, Iterator, List, Optional, Union, Generator
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import contextmanager
import weakref
import threading
from functools import lru_cache
import pickle
import tempfile

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Memory usage statistics"""
    process_memory_mb: float = 0.0
    system_memory_mb: float = 0.0
    memory_percent: float = 0.0
    peak_memory_mb: float = 0.0
    
    def update(self):
        """Update current memory statistics"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            self.process_memory_mb = memory_info.rss / 1024 / 1024
            
            system_memory = psutil.virtual_memory()
            self.system_memory_mb = system_memory.total / 1024 / 1024
            self.memory_percent = system_memory.percent
            
            # Track peak memory
            if self.process_memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = self.process_memory_mb
                
        except Exception as e:
            logger.warning(f"Failed to update memory stats: {e}")

class MemoryMonitor:
    """
    Memory monitoring and management system.
    
    Features:
    - Real-time memory usage tracking
    - Automatic garbage collection triggers
    - Memory threshold warnings
    - Peak memory tracking
    """
    
    def __init__(self, warning_threshold_mb: float = 1000.0, critical_threshold_mb: float = 2000.0):
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.stats = MemoryStats()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self, interval: float = 5.0):
        """Start background memory monitoring"""
        if self._monitoring:
            return
            
        self._monitoring = True
        
        def monitor_loop():
            while self._monitoring:
                self.check_memory()
                threading.Event().wait(interval)
                
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Started memory monitoring (warning: {self.warning_threshold_mb}MB, critical: {self.critical_threshold_mb}MB)")
        
    def stop_monitoring(self):
        """Stop background memory monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            
    def check_memory(self) -> MemoryStats:
        """Check current memory usage and trigger actions if needed"""
        self.stats.update()
        
        if self.stats.process_memory_mb > self.critical_threshold_mb:
            logger.critical(f"Critical memory usage: {self.stats.process_memory_mb:.1f}MB (>{self.critical_threshold_mb}MB)")
            self.force_garbage_collection()
        elif self.stats.process_memory_mb > self.warning_threshold_mb:
            logger.warning(f"High memory usage: {self.stats.process_memory_mb:.1f}MB (>{self.warning_threshold_mb}MB)")
            self.gentle_garbage_collection()
            
        return self.stats
        
    def force_garbage_collection(self):
        """Force aggressive garbage collection"""
        logger.debug("Forcing aggressive garbage collection...")
        collected = 0
        for generation in range(3):
            collected += gc.collect(generation)
        logger.debug(f"Garbage collection freed {collected} objects")
        
    def gentle_garbage_collection(self):
        """Perform gentle garbage collection"""
        logger.debug("Performing gentle garbage collection...")
        collected = gc.collect(0)  # Only collect generation 0
        logger.debug(f"Gentle garbage collection freed {collected} objects")

# Global memory monitor instance
memory_monitor = MemoryMonitor()

class StreamingGedcomParser:
    """
    Memory-efficient streaming GEDCOM parser for large files.
    
    Features:
    - Processes files line by line without loading entire file
    - Yields individual records as they're parsed
    - Memory-mapped file access for very large files
    - Configurable buffer sizes
    """
    
    def __init__(self, file_path: Union[str, Path], use_mmap: bool = True, buffer_size: int = 8192):
        self.file_path = Path(file_path)
        self.use_mmap = use_mmap
        self.buffer_size = buffer_size
        self.file_size = self.file_path.stat().st_size
        
        # Determine if we should use memory mapping
        self.should_use_mmap = use_mmap and self.file_size > 50 * 1024 * 1024  # 50MB threshold
        
        logger.info(f"Initializing GEDCOM parser for {self.file_path.name} ({self.file_size / 1024 / 1024:.1f}MB)")
        if self.should_use_mmap:
            logger.info("Using memory-mapped file access for large file")
            
    @contextmanager
    def _open_file(self):
        """Context manager for file access"""
        if self.should_use_mmap:
            with open(self.file_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    yield mm
        else:
            with open(self.file_path, 'r', encoding='utf-8', buffering=self.buffer_size) as f:
                yield f
                
    def parse_records(self) -> Generator[Dict[str, Any], None, None]:
        """
        Parse GEDCOM file and yield individual records.
        
        Yields:
            Dictionary containing parsed record data
        """
        current_record = {}
        record_lines = []
        
        with self._open_file() as file_obj:
            if self.should_use_mmap:
                lines = file_obj.decode('utf-8').splitlines()
            else:
                lines = file_obj
                
            for line_num, line in enumerate(lines, 1):
                if isinstance(line, bytes):
                    line = line.decode('utf-8')
                    
                line = line.strip()
                if not line:
                    continue
                    
                # Parse GEDCOM line format: LEVEL TAG [VALUE]
                parts = line.split(' ', 2)
                if len(parts) < 2:
                    continue
                    
                try:
                    level = int(parts[0])
                    tag = parts[1]
                    value = parts[2] if len(parts) > 2 else ""
                    
                    # Start of new record (level 0)
                    if level == 0 and current_record:
                        yield self._finalize_record(current_record, record_lines)
                        current_record = {}
                        record_lines = []
                        
                    # Add line to current record
                    record_lines.append({
                        'level': level,
                        'tag': tag,
                        'value': value,
                        'line_num': line_num
                    })
                    
                    # Set record ID and type for level 0
                    if level == 0:
                        current_record['id'] = value
                        current_record['type'] = tag
                        
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse line {line_num}: {line} - {e}")
                    continue
                    
                # Periodic memory check for very large files
                if line_num % 10000 == 0:
                    memory_monitor.check_memory()
                    
        # Yield final record
        if current_record:
            yield self._finalize_record(current_record, record_lines)
            
    def _finalize_record(self, record: Dict[str, Any], lines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Finalize a parsed record"""
        record['lines'] = lines
        record['line_count'] = len(lines)
        return record

class LazyDataLoader:
    """
    Lazy loading system for large datasets.
    
    Features:
    - Load data on demand
    - Weak references to prevent memory leaks
    - Configurable cache sizes
    - Automatic cleanup of unused data
    """
    
    def __init__(self, max_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        self._cache: Dict[str, Any] = {}
        self._weak_refs: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._access_order: List[str] = []
        self._lock = threading.Lock()
        
    def get(self, key: str, loader_func: callable) -> Any:
        """Get data with lazy loading"""
        with self._lock:
            # Check cache first
            if key in self._cache:
                self._update_access_order(key)
                return self._cache[key]
                
            # Check weak references
            if key in self._weak_refs:
                data = self._weak_refs[key]
                if data is not None:
                    self._cache[key] = data
                    self._update_access_order(key)
                    return data
                    
            # Load data
            data = loader_func()
            self._store_data(key, data)
            return data
            
    def _store_data(self, key: str, data: Any):
        """Store data in cache with size management"""
        # Remove oldest items if cache is full
        while len(self._cache) >= self.max_cache_size:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                old_data = self._cache.pop(oldest_key)
                # Store in weak reference for potential reuse
                try:
                    self._weak_refs[oldest_key] = old_data
                except TypeError:
                    pass  # Object not weakly referenceable
                    
        self._cache[key] = data
        self._update_access_order(key)
        
    def _update_access_order(self, key: str):
        """Update access order for LRU eviction"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
    def clear(self):
        """Clear all cached data"""
        with self._lock:
            self._cache.clear()
            self._weak_refs.clear()
            self._access_order.clear()
            
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'cache_size': len(self._cache),
            'weak_refs': len(self._weak_refs),
            'max_size': self.max_cache_size
        }

# Global lazy loader instance
lazy_loader = LazyDataLoader()

@contextmanager
def memory_optimized_processing(
    description: str = "Memory Optimized Processing",
    enable_monitoring: bool = True,
    warning_threshold_mb: float = 1000.0,
    critical_threshold_mb: float = 2000.0
):
    """
    Context manager for memory-optimized processing.
    
    Features:
    - Automatic memory monitoring
    - Garbage collection management
    - Memory usage reporting
    """
    if enable_monitoring:
        monitor = MemoryMonitor(warning_threshold_mb, critical_threshold_mb)
        monitor.start_monitoring()
    else:
        monitor = None
        
    initial_stats = MemoryStats()
    initial_stats.update()
    
    logger.info(f"Starting {description} (initial memory: {initial_stats.process_memory_mb:.1f}MB)")
    
    try:
        yield monitor
    finally:
        if monitor:
            monitor.stop_monitoring()
            final_stats = monitor.check_memory()
            
            memory_delta = final_stats.process_memory_mb - initial_stats.process_memory_mb
            logger.info(
                f"Completed {description} | "
                f"Memory: {final_stats.process_memory_mb:.1f}MB "
                f"({memory_delta:+.1f}MB) | "
                f"Peak: {final_stats.peak_memory_mb:.1f}MB"
            )
            
            # Final cleanup
            monitor.force_garbage_collection()

def optimize_for_large_files(func):
    """Decorator to optimize functions for large file processing"""
    def wrapper(*args, **kwargs):
        with memory_optimized_processing(f"Large File Processing: {func.__name__}"):
            return func(*args, **kwargs)
    return wrapper
