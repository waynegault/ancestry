#!/usr/bin/env python3

"""
Memory Optimization Utilities for Ancestry Project

Provides memory usage monitoring, optimization patterns, and lazy loading utilities
to improve memory efficiency across the application.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# === STANDARD LIBRARY IMPORTS ===
import gc
import os
import psutil
import sys
import time
import weakref
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

# === THIRD-PARTY IMPORTS ===
from dataclasses import dataclass

# === TYPE DEFINITIONS ===
T = TypeVar('T')


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    process_memory_mb: float
    system_memory_mb: float
    memory_percent: float
    gc_objects: int
    timestamp: float


class MemoryMonitor:
    """Monitor and track memory usage patterns."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_memory: Optional[float] = None
        self.peak_memory: float = 0
        self.memory_history: List[MemoryStats] = []
        self.gc_threshold_original = gc.get_threshold()
        
    def get_current_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        process_memory_mb = memory_info.rss / 1024 / 1024
        system_memory_mb = system_memory.total / 1024 / 1024
        memory_percent = (memory_info.rss / system_memory.total) * 100
        
        # Update peak memory
        if process_memory_mb > self.peak_memory:
            self.peak_memory = process_memory_mb
            
        return MemoryStats(
            process_memory_mb=process_memory_mb,
            system_memory_mb=system_memory_mb,
            memory_percent=memory_percent,
            gc_objects=len(gc.get_objects()),
            timestamp=time.time()
        )
    
    def set_baseline(self):
        """Set current memory usage as baseline."""
        stats = self.get_current_memory_stats()
        self.baseline_memory = stats.process_memory_mb
        logger.info(f"Memory baseline set: {self.baseline_memory:.2f} MB")
    
    def get_memory_delta(self) -> float:
        """Get memory usage change from baseline."""
        if self.baseline_memory is None:
            self.set_baseline()
            return 0.0
        
        current_stats = self.get_current_memory_stats()
        return current_stats.process_memory_mb - self.baseline_memory
    
    def log_memory_usage(self, context: str = ""):
        """Log current memory usage with context."""
        stats = self.get_current_memory_stats()
        delta = self.get_memory_delta()
        
        context_str = f" ({context})" if context else ""
        logger.info(
            f"Memory usage{context_str}: {stats.process_memory_mb:.2f} MB "
            f"(Δ{delta:+.2f} MB, {stats.memory_percent:.1f}% system, "
            f"{stats.gc_objects} objects)"
        )
        
        # Store in history
        self.memory_history.append(stats)
        
        # Keep only last 100 entries
        if len(self.memory_history) > 100:
            self.memory_history = self.memory_history[-100:]
    
    def optimize_gc(self):
        """Optimize garbage collection settings for memory efficiency."""
        # More aggressive garbage collection for memory-constrained environments
        gc.set_threshold(700, 10, 10)  # More frequent collection
        collected = gc.collect()
        logger.debug(f"Garbage collection optimized, collected {collected} objects")
    
    def restore_gc(self):
        """Restore original garbage collection settings."""
        gc.set_threshold(*self.gc_threshold_original)
        logger.debug("Garbage collection settings restored")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report."""
        current_stats = self.get_current_memory_stats()
        
        return {
            "current_memory_mb": current_stats.process_memory_mb,
            "baseline_memory_mb": self.baseline_memory,
            "peak_memory_mb": self.peak_memory,
            "memory_delta_mb": self.get_memory_delta(),
            "system_memory_percent": current_stats.memory_percent,
            "gc_objects": current_stats.gc_objects,
            "history_entries": len(self.memory_history),
            "recommendations": self._generate_memory_recommendations(current_stats)
        }
    
    def _generate_memory_recommendations(self, stats: MemoryStats) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        if stats.process_memory_mb > 1000:  # > 1GB
            recommendations.append("Consider implementing lazy loading for large datasets")
        
        if stats.memory_percent > 80:
            recommendations.append("High system memory usage - consider memory cleanup")
        
        if stats.gc_objects > 100000:
            recommendations.append("High object count - consider object pooling or cleanup")
        
        if len(self.memory_history) > 10:
            # Check for memory growth trend
            recent_memory = [h.process_memory_mb for h in self.memory_history[-10:]]
            if len(recent_memory) > 5 and recent_memory[-1] > recent_memory[0] * 1.5:
                recommendations.append("Memory usage growing - potential memory leak")
        
        return recommendations


# Global memory monitor instance
memory_monitor = MemoryMonitor()


def memory_profile(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to profile memory usage of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        function_name = func.__name__
        
        # Log memory before
        memory_monitor.log_memory_usage(f"before {function_name}")
        start_memory = memory_monitor.get_current_memory_stats().process_memory_mb
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Log memory after
            memory_monitor.log_memory_usage(f"after {function_name}")
            end_memory = memory_monitor.get_current_memory_stats().process_memory_mb
            memory_delta = end_memory - start_memory
            
            if abs(memory_delta) > 10:  # Log significant memory changes
                logger.info(
                    f"Function {function_name} memory impact: {memory_delta:+.2f} MB"
                )
    
    return wrapper


def lazy_property(func: Callable[..., T]) -> property:
    """Decorator to create a lazy-loaded property."""
    attr_name = f'_lazy_{func.__name__}'
    
    @property
    @wraps(func)
    def wrapper(self) -> T:
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    
    return wrapper


class LazyList:
    """Memory-efficient lazy list implementation."""
    
    def __init__(self, generator_func: Callable[[], List[T]]):
        self._generator_func = generator_func
        self._data: Optional[List[T]] = None
        self._loaded = False
    
    def _ensure_loaded(self):
        """Ensure data is loaded."""
        if not self._loaded:
            self._data = self._generator_func()
            self._loaded = True
    
    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._data or [])
    
    def __getitem__(self, index: Union[int, slice]) -> Union[T, List[T]]:
        self._ensure_loaded()
        return (self._data or [])[index]
    
    def __iter__(self):
        self._ensure_loaded()
        return iter(self._data or [])
    
    def is_loaded(self) -> bool:
        """Check if data has been loaded."""
        return self._loaded


def memory_efficient_batch_processor(items: List[T], batch_size: int = 1000) -> List[T]:
    """Process items in memory-efficient batches."""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Process batch
        batch_results = []
        for item in batch:
            # Process individual item
            batch_results.append(item)
        
        results.extend(batch_results)
        
        # Force garbage collection after each batch
        if i % (batch_size * 10) == 0:  # Every 10 batches
            gc.collect()
    
    return results


def run_comprehensive_tests() -> bool:
    """Run comprehensive memory optimization tests."""
    from test_framework import TestSuite
    
    suite = TestSuite("Memory Optimization", "memory_optimizer.py")
    suite.start_suite()
    
    def test_memory_monitor():
        """Test memory monitoring functionality."""
        monitor = MemoryMonitor()
        monitor.set_baseline()
        
        # Test memory stats
        stats = monitor.get_current_memory_stats()
        assert stats.process_memory_mb > 0
        assert stats.system_memory_mb > 0
        assert 0 <= stats.memory_percent <= 100
        
        # Test memory delta
        delta = monitor.get_memory_delta()
        assert isinstance(delta, float)
        
        return True
    
    def test_lazy_property():
        """Test lazy property decorator."""
        class TestClass:
            def __init__(self):
                self.call_count = 0
            
            @lazy_property
            def expensive_property(self):
                self.call_count += 1
                return "expensive_result"
        
        obj = TestClass()
        assert obj.call_count == 0
        
        # First access should call the function
        result1 = obj.expensive_property
        assert result1 == "expensive_result"
        assert obj.call_count == 1
        
        # Second access should use cached value
        result2 = obj.expensive_property
        assert result2 == "expensive_result"
        assert obj.call_count == 1  # Should not increment
        
        return True
    
    def test_lazy_list():
        """Test lazy list implementation."""
        call_count = 0
        
        def generator():
            nonlocal call_count
            call_count += 1
            return [1, 2, 3, 4, 5]
        
        lazy_list = LazyList(generator)
        assert not lazy_list.is_loaded()
        assert call_count == 0
        
        # First access should load data
        length = len(lazy_list)
        assert length == 5
        assert lazy_list.is_loaded()
        assert call_count == 1
        
        # Subsequent access should not reload
        item = lazy_list[0]
        assert item == 1
        assert call_count == 1
        
        return True
    
    suite.run_test(
        "Memory Monitor",
        test_memory_monitor,
        "Test memory monitoring and statistics collection"
    )
    
    suite.run_test(
        "Lazy Property",
        test_lazy_property,
        "Test lazy property decorator functionality"
    )
    
    suite.run_test(
        "Lazy List",
        test_lazy_list,
        "Test lazy list implementation"
    )
    
    return suite.finish_suite()


if __name__ == "__main__":
    print("🧠 Running Memory Optimization comprehensive test suite...")
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
