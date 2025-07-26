#!/usr/bin/env python3

"""
Quick performance test for action10 optimization
"""

import sys
import time

def test_performance():
    try:
        print("🔄 Testing action10 optimized performance...")
        
        # Import the optimized module
        import action10_optimized
        
        # Run the optimized test
        start_time = time.time()
        result = action10_optimized.action10_module_tests_optimized()
        execution_time = time.time() - start_time
        
        print(f"📊 Results:")
        print(f"  Execution time: {execution_time:.2f}s")
        print(f"  Baseline: 98.64s")
        print(f"  Speedup: {98.64/execution_time:.1f}x")
        print(f"  Target (20s): {'✅ MET' if execution_time <= 20 else '❌ MISSED'}")
        
        return execution_time <= 20
        
    except Exception as e:
        print(f"❌ Error testing performance: {e}")
        return False

if __name__ == "__main__":
    success = test_performance()
    sys.exit(0 if success else 1)
