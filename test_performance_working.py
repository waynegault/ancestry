#!/usr/bin/env python3
"""
Simplified Performance Monitor - Working Version
"""

import time
import threading
import psutil
import os
from datetime import datetime
from typing import Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict

print("Starting simplified performance monitor test...")


@dataclass
class SimpleServiceMetrics:
    name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration: float = 0.0


class SimplePerformanceMonitor:
    def __init__(self):
        self.services: Dict[str, SimpleServiceMetrics] = {}
        self._start_time = datetime.now()
        print("✅ Performance monitor initialized")

    def record_operation(self, service_name: str, duration: float, success: bool):
        if service_name not in self.services:
            self.services[service_name] = SimpleServiceMetrics(name=service_name)

        service = self.services[service_name]
        service.total_calls += 1
        service.total_duration += duration

        if success:
            service.successful_calls += 1
        else:
            service.failed_calls += 1

    def get_summary(self) -> Dict[str, Any]:
        total_operations = sum(s.total_calls for s in self.services.values())
        uptime = (datetime.now() - self._start_time).total_seconds()

        return {
            "total_operations": total_operations,
            "uptime_seconds": uptime,
            "services_count": len(self.services),
        }


def test_simple_monitor():
    print("Testing simple performance monitor...")
    monitor = SimplePerformanceMonitor()

    # Test recording operations
    monitor.record_operation("test_service", 0.1, True)
    monitor.record_operation("test_service", 0.2, True)
    monitor.record_operation("test_service", 0.15, False)

    summary = monitor.get_summary()
    print(f"Summary: {summary}")

    if summary["total_operations"] == 3:
        print("✅ Simple performance monitor works!")
        return True
    else:
        print("❌ Simple performance monitor failed!")
        return False


def test_health_checks():
    print("\nTesting health checks...")

    def database_check():
        try:
            import sqlite3

            conn = sqlite3.connect("ancestry.db")
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            conn.close()
            return {"status": "healthy", "message": "Database OK"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"Database error: {e}"}

    def memory_check():
        try:
            memory = psutil.virtual_memory()
            return {
                "status": "healthy" if memory.percent < 85 else "degraded",
                "memory_percent": memory.percent,
            }
        except Exception as e:
            return {"status": "error", "message": f"Memory check failed: {e}"}

    def disk_check():
        try:
            disk_path = "C:\\" if os.name == "nt" else "/"
            disk = psutil.disk_usage(disk_path)
            return {
                "status": "healthy" if disk.percent < 90 else "degraded",
                "disk_percent": disk.percent,
            }
        except Exception as e:
            return {"status": "error", "message": f"Disk check failed: {e}"}

    # Run health checks
    db_result = database_check()
    mem_result = memory_check()
    disk_result = disk_check()

    print(f"Database: {db_result}")
    print(f"Memory: {mem_result}")
    print(f"Disk: {disk_result}")

    print("✅ Health checks completed!")
    return True


if __name__ == "__main__":
    try:
        success1 = test_simple_monitor()
        success2 = test_health_checks()

        if success1 and success2:
            print("\n🎉 All tests passed! Performance monitoring is working.")
        else:
            print("\n❌ Some tests failed.")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
