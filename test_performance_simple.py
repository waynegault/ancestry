#!/usr/bin/env python3
"""
Simple test of performance monitor without background threads
"""

import time
import psutil
from datetime import datetime

print("Testing basic performance monitoring...")

try:
    # Test basic psutil functionality
    print(f"CPU percent: {psutil.cpu_percent(interval=0.1)}%")
    print(f"Memory percent: {psutil.virtual_memory().percent}%")

    # Test disk usage with correct Windows path
    import os

    disk_path = "C:\\" if os.name == "nt" else "/"
    disk = psutil.disk_usage(disk_path)
    print(f"Disk usage: {disk.percent}%")

    # Test process info
    process = psutil.Process()
    print(f"Process memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")

    print("✅ Basic performance monitoring works!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
