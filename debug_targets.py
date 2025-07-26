#!/usr/bin/env python3

"""
Debug the performance test to see which targets are failing
"""

import sys
import time
from performance_cache import get_cache_stats
from action10_optimized import action10_module_tests_optimized, run_action10_integration_test_optimized

def debug_performance_targets():
    print("🔍 Debugging Performance Targets")
    print("=" * 50)
    
    try:
        # Test 1: Run optimized tests
        print("\n1️⃣ Testing optimized action10...")
        start_time = time.time()
        result1 = action10_module_tests_optimized()
        first_run_time = time.time() - start_time
        
        # Test 2: Run again for cache test
        print("\n2️⃣ Testing cache performance...")
        start_time = time.time()
        result2 = action10_module_tests_optimized()
        second_run_time = time.time() - start_time
        
        # Test 3: Integration test
        print("\n3️⃣ Testing integration...")
        integration_success = run_action10_integration_test_optimized()
        
        # Calculate results
        baseline_time = 98.64
        best_time = min(first_run_time, second_run_time)
        speedup = baseline_time / best_time if best_time > 0 else 0
        cache_speedup = first_run_time / second_run_time if second_run_time > 0 else 1
        
        print("\n📊 TARGET ANALYSIS:")
        print(f"  Target 1 (Under 20s): {best_time:.3f}s - {'✅ PASS' if best_time <= 20 else '❌ FAIL'}")
        print(f"  Target 2 (4x speedup): {speedup:.1f}x - {'✅ PASS' if speedup >= 4 else '❌ FAIL'}")
        print(f"  Target 3 (Cache 2x): {cache_speedup:.1f}x - {'✅ PASS' if cache_speedup >= 2 else '❌ FAIL'}")
        print(f"  Target 4 (Integration): {'✅ PASS' if integration_success else '❌ FAIL'}")
        
        # Cache stats
        cache_stats = get_cache_stats()
        print(f"\n📈 Cache Stats: {cache_stats}")
        
        return {
            'target_20s': best_time <= 20,
            'target_4x': speedup >= 4, 
            'cache_effective': cache_speedup >= 2,
            'integration_success': integration_success
        }
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    debug_performance_targets()
