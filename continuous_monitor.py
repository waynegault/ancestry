#!/usr/bin/env python3
"""
Continuous Action 6 Health Monitor
Provides real-time monitoring of Action 6 execution with health status updates.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import psutil


def _print_monitor_header(pid: int, interval: int) -> None:
    """Print monitoring header information."""
    print(f"CONTINUOUS ACTION 6 HEALTH MONITOR - PID: {pid}")
    print("=" * 70)
    print(f"Monitoring interval: {interval} seconds")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 70)


def _get_process_metrics(process) -> tuple[float, float, float]:
    """Get process metrics: memory_mb, cpu_percent, runtime_minutes."""
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    cpu_percent = process.cpu_percent()

    start_time = process.create_time()
    runtime_minutes = (time.time() - start_time) / 60

    return memory_mb, cpu_percent, runtime_minutes


def _get_status_indicators(memory_mb: float, cpu_percent: float, runtime_minutes: float) -> tuple[str, str, str]:
    """Get status indicators for metrics."""
    memory_status = "✅" if memory_mb < 500 else "⚠️" if memory_mb < 1000 else "🔴"
    cpu_status = "✅" if cpu_percent < 50 else "⚠️" if cpu_percent < 80 else "🔴"
    runtime_status = "✅" if runtime_minutes < 20 else "⚠️" if runtime_minutes < 30 else "🔴"
    return memory_status, cpu_status, runtime_status


def _print_alerts(memory_mb: float, cpu_percent: float, runtime_minutes: float, children_count: int) -> None:
    """Print special alerts for concerning metrics."""
    if memory_mb > 1000:
        print(f"           🚨 HIGH MEMORY USAGE: {memory_mb:.1f} MB")

    if cpu_percent > 80:
        print(f"           🚨 HIGH CPU USAGE: {cpu_percent:.1f}%")

    if runtime_minutes > 25:
        print(f"           ⏰ LONG RUNTIME: {runtime_minutes:.1f} minutes")
        print("           💡 This is normal for Action 6 (typically 12-20 minutes)")

    if children_count > 5:
        print(f"           📊 Multiple child processes: {children_count}")


def monitor_action6_continuous(pid: int, interval: int = 30) -> None:
    """Continuously monitor Action 6 process health."""
    _print_monitor_header(pid, interval)

    start_monitor_time = time.time()
    check_count = 0

    try:
        while True:
            check_count += 1
            current_time = datetime.now().strftime('%H:%M:%S')

            try:
                process = psutil.Process(pid)

                if not process.is_running():
                    print(f"\n[{current_time}] ❌ PROCESS STOPPED")
                    print("Action 6 has completed or terminated")
                    break

                memory_mb, cpu_percent, runtime_minutes = _get_process_metrics(process)
                memory_status, cpu_status, runtime_status = _get_status_indicators(memory_mb, cpu_percent, runtime_minutes)

                print(f"[{current_time}] Check #{check_count:2d} | "
                      f"Runtime: {runtime_minutes:5.1f}m {runtime_status} | "
                      f"Memory: {memory_mb:6.1f}MB {memory_status} | "
                      f"CPU: {cpu_percent:4.1f}% {cpu_status}")

                children_count = len(process.children())
                _print_alerts(memory_mb, cpu_percent, runtime_minutes, children_count)

            except psutil.NoSuchProcess:
                print(f"\n[{current_time}] ❌ PROCESS NOT FOUND")
                print("Action 6 process has terminated")
                break
            except Exception as e:
                print(f"\n[{current_time}] ❌ MONITORING ERROR: {e}")

            time.sleep(interval)

    except KeyboardInterrupt:
        monitor_duration = (time.time() - start_monitor_time) / 60
        print("\n\n📊 MONITORING SUMMARY:")
        print(f"Monitoring duration: {monitor_duration:.1f} minutes")
        print(f"Total checks performed: {check_count}")
        print("Monitoring stopped by user")

    except Exception as e:
        print(f"\n❌ MONITORING FAILED: {e}")

def check_action6_logs() -> None:
    """Check for Action 6 log output."""
    print("\n📋 CHECKING FOR ACTION 6 LOGS:")
    print("-" * 40)

    # Look for recent log files or output
    log_files = []
    for path in Path().iterdir():
        try:
            if path.is_file():
                name = path.name
                if (name.endswith(".log") or "log" in name.lower()) and "action" in name.lower():
                    stat = path.stat()
                    mod_time = stat.st_mtime
                    if time.time() - mod_time < 7200:  # Modified in last 2 hours
                        log_files.append((name, mod_time, stat.st_size))
        except Exception:
            continue

    if log_files:
        print("Recent Action 6 log files:")
        for log_file, mod_time, size in log_files:
            size_mb = size / 1024 / 1024
            print(f"  📄 {log_file}")
            print(f"     Modified: {time.ctime(mod_time)}")
            print(f"     Size: {size_mb:.2f} MB")
    else:
        print("No recent Action 6 log files found")

def main() -> None:
    """Main monitoring function."""
    if len(sys.argv) < 2:
        print("Usage: python continuous_monitor.py <PID> [interval_seconds]")
        print("Example: python continuous_monitor.py 43412 30")
        sys.exit(1)

    try:
        pid = int(sys.argv[1])
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    except ValueError:
        print("Error: PID and interval must be numbers")
        sys.exit(1)

    # Initial health check
    try:
        process = psutil.Process(pid)
        print("🎯 INITIAL STATUS CHECK:")
        print(f"Process: {process.name()} (PID: {pid})")
        print(f"Status: {process.status()}")
        print(f"Command: {' '.join(process.cmdline())}")
        print()
    except psutil.NoSuchProcess:
        print(f"❌ Process {pid} not found")
        sys.exit(1)

    # Check for logs
    check_action6_logs()

    # Start continuous monitoring
    print("\n🔍 STARTING CONTINUOUS MONITORING...")
    monitor_action6_continuous(pid, interval)

if __name__ == "__main__":
    main()
