#!/usr/bin/env python3

# test_cache_system.py

"""
test_cache_system.py - Test Script for Aggressive Caching System

Tests the new aggressive caching implementation including GEDCOM caching,
API response caching, and cache management functionality.
"""

import time
from pathlib import Path

# Import the caching modules
from cache import get_cache_stats, clear_cache
from cache_manager import initialize_aggressive_caching, get_cache_performance_report, log_cache_status
from logging_config import logger, setup_logging


def test_basic_cache():
    """Test basic cache functionality."""
    print("\n=== Testing Basic Cache Functionality ===")
    
    try:
        # Get initial cache stats
        stats = get_cache_stats()
        print(f"Initial cache stats: {stats}")
        
        # Test cache decorator
        from cache import cache_result
        
        @cache_result("test_function", expire=60)
        def test_function(x, y):
            """Test function that returns sum of x and y."""
            time.sleep(0.1)  # Simulate some work
            return x + y
        
        # First call (should be slow)
        start_time = time.time()
        result1 = test_function(5, 3)
        time1 = time.time() - start_time
        print(f"First call result: {result1}, time: {time1:.3f}s")
        
        # Second call (should be fast - cached)
        start_time = time.time()
        result2 = test_function(5, 3)
        time2 = time.time() - start_time
        print(f"Second call result: {result2}, time: {time2:.3f}s")
        
        # Verify results are the same and second call was faster
        assert result1 == result2, "Results should be identical"
        assert time2 < time1, "Second call should be faster (cached)"
        
        print("✓ Basic cache functionality test passed")
        
    except Exception as e:
        print(f"✗ Basic cache test failed: {e}")
        return False
    
    return True


def test_gedcom_cache():
    """Test GEDCOM caching functionality."""
    print("\n=== Testing GEDCOM Cache Functionality ===")
    
    try:
        from config import config_instance
        
        # Check if GEDCOM file is configured
        if not config_instance or not hasattr(config_instance, 'GEDCOM_FILE_PATH'):
            print("⚠ GEDCOM file not configured, skipping GEDCOM cache test")
            return True
        
        gedcom_path = config_instance.GEDCOM_FILE_PATH
        if not gedcom_path or not Path(gedcom_path).exists():
            print(f"⚠ GEDCOM file not found at {gedcom_path}, skipping GEDCOM cache test")
            return True
        
        from gedcom_cache import load_gedcom_with_aggressive_caching, get_gedcom_cache_info
        
        print(f"Testing GEDCOM cache with file: {Path(gedcom_path).name}")
        
        # First load (should be slow)
        start_time = time.time()
        gedcom_data1 = load_gedcom_with_aggressive_caching(str(gedcom_path))
        time1 = time.time() - start_time
        print(f"First GEDCOM load time: {time1:.3f}s")
        
        # Second load (should be fast - cached)
        start_time = time.time()
        gedcom_data2 = load_gedcom_with_aggressive_caching(str(gedcom_path))
        time2 = time.time() - start_time
        print(f"Second GEDCOM load time: {time2:.3f}s")
        
        # Verify both loads succeeded
        assert gedcom_data1 is not None, "First GEDCOM load should succeed"
        assert gedcom_data2 is not None, "Second GEDCOM load should succeed"
        assert time2 < time1, "Second load should be faster (cached)"
        
        # Get cache info
        cache_info = get_gedcom_cache_info()
        print(f"GEDCOM cache info: {cache_info}")
        
        print("✓ GEDCOM cache functionality test passed")
        
    except Exception as e:
        print(f"✗ GEDCOM cache test failed: {e}")
        return False
    
    return True


def test_cache_manager():
    """Test cache manager functionality."""
    print("\n=== Testing Cache Manager Functionality ===")
    
    try:
        # Test cache initialization
        init_result = initialize_aggressive_caching()
        print(f"Cache initialization result: {init_result}")
        
        # Get performance report
        report = get_cache_performance_report()
        print(f"Cache performance report keys: {list(report.keys())}")
        
        # Test cache statistics logging
        log_cache_status()
        print("✓ Cache statistics logged successfully")
        
        print("✓ Cache manager functionality test passed")
        
    except Exception as e:
        print(f"✗ Cache manager test failed: {e}")
        return False
    
    return True


def test_api_cache():
    """Test API caching functionality."""
    print("\n=== Testing API Cache Functionality ===")
    
    try:
        from api_cache import get_api_cache_stats, create_api_cache_key
        
        # Test cache key generation
        params1 = {"name": "John", "age": 30}
        params2 = {"age": 30, "name": "John"}  # Same params, different order
        
        key1 = create_api_cache_key("test_endpoint", params1)
        key2 = create_api_cache_key("test_endpoint", params2)
        
        assert key1 == key2, "Cache keys should be identical for same params"
        print(f"✓ API cache key generation works correctly")
        
        # Get API cache stats
        api_stats = get_api_cache_stats()
        print(f"API cache stats keys: {list(api_stats.keys())}")
        
        print("✓ API cache functionality test passed")
        
    except Exception as e:
        print(f"✗ API cache test failed: {e}")
        return False
    
    return True


def main():
    """Run all cache system tests."""
    print("Starting Aggressive Caching System Tests")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    
    # Track test results
    tests = [
        ("Basic Cache", test_basic_cache),
        ("GEDCOM Cache", test_gedcom_cache),
        ("Cache Manager", test_cache_manager),
        ("API Cache", test_api_cache),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_name} test failed")
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All cache system tests passed!")
        
        # Show final cache statistics
        print("\n=== Final Cache Statistics ===")
        try:
            log_cache_status()
        except Exception as e:
            print(f"Error showing final stats: {e}")
    else:
        print("❌ Some cache system tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
