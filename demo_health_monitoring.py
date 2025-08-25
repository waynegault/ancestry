#!/usr/bin/env python3

"""
demo_health_monitoring.py - Demonstration of Health Monitoring System

Shows how the health monitoring system works with real-time metrics,
alerts, and performance recommendations.
"""

import time

from health_monitor import get_health_monitor, get_performance_recommendations


def demo_health_monitoring() -> None:
    """Demonstrate the health monitoring system capabilities."""
    print("🚀 HEALTH MONITORING SYSTEM DEMONSTRATION")
    print("=" * 60)

    # Initialize health monitor
    monitor = get_health_monitor()
    print("✅ Health monitor initialized")

    # Simulate normal operation
    print("\n📊 PHASE 1: Normal Operation")
    print("-" * 40)

    # Update metrics with good values
    monitor.update_metric("api_response_time", 2.5)
    monitor.update_metric("memory_usage_mb", 150.0)
    monitor.update_metric("error_rate", 0.02)
    monitor.update_metric("session_age_minutes", 15.0)
    monitor.update_metric("browser_age_minutes", 10.0)
    monitor.update_metric("pages_since_refresh", 8.0)
    monitor.update_metric("cpu_usage_percent", 45.0)
    monitor.update_metric("disk_usage_percent", 60.0)

    # Record some API calls
    for response_time in [2.1, 2.3, 2.0, 2.4, 2.2]:
        monitor.record_api_response_time(response_time)

    # Record page processing times
    for processing_time in [25.0, 28.0, 22.0, 30.0]:
        monitor.record_page_processing_time(processing_time)

    # Get dashboard
    dashboard = monitor.get_health_dashboard()
    print(f"Health Score: {dashboard['health_score']:.1f}/100 ({dashboard['health_status'].upper()})")
    print(f"Risk Level: {dashboard['risk_level']} (Score: {dashboard['risk_score']:.2f})")
    print(f"Avg API Response: {dashboard['performance_summary']['avg_api_response_time']:.1f}s")
    print(f"Memory Usage: {dashboard['performance_summary']['current_memory_mb']:.1f}MB")

    # Get performance recommendations
    recommendations = get_performance_recommendations(dashboard['health_score'], dashboard['risk_score'])
    print(f"Recommended Settings: {recommendations['max_concurrency']} workers, batch {recommendations['batch_size']}")
    print(f"Action Required: {recommendations['action_required']}")

    time.sleep(2)

    # Simulate degrading performance
    print("\n⚠️ PHASE 2: Performance Degradation")
    print("-" * 40)

    # Update metrics with warning values
    monitor.update_metric("api_response_time", 6.5)
    monitor.update_metric("memory_usage_mb", 250.0)
    monitor.update_metric("error_rate", 0.08)
    monitor.update_metric("session_age_minutes", 35.0)
    monitor.update_metric("browser_age_minutes", 28.0)
    monitor.update_metric("pages_since_refresh", 22.0)

    # Record slower API calls
    for response_time in [6.1, 7.3, 8.0, 6.4, 7.2]:
        monitor.record_api_response_time(response_time)

    # Record some errors
    monitor.record_error("ConnectionTimeout")
    monitor.record_error("RateLimitError")

    dashboard = monitor.get_health_dashboard()
    print(f"Health Score: {dashboard['health_score']:.1f}/100 ({dashboard['health_status'].upper()})")
    print(f"Risk Level: {dashboard['risk_level']} (Score: {dashboard['risk_score']:.2f})")
    print(f"Avg API Response: {dashboard['performance_summary']['avg_api_response_time']:.1f}s")
    print(f"Total Errors: {dashboard['performance_summary']['total_errors']}")

    recommendations = get_performance_recommendations(dashboard['health_score'], dashboard['risk_score'])
    print(f"Recommended Settings: {recommendations['max_concurrency']} workers, batch {recommendations['batch_size']}")
    print(f"Action Required: {recommendations['action_required']}")

    # Show recent alerts
    if dashboard['recent_alerts']:
        print("\n🚨 Recent Alerts:")
        for alert in dashboard['recent_alerts'][-3:]:
            print(f"   {alert['level'].upper()}: {alert['message']}")

    time.sleep(2)

    # Simulate critical conditions
    print("\n🚨 PHASE 3: Critical Conditions")
    print("-" * 40)

    # Update metrics with critical values
    monitor.update_metric("api_response_time", 12.0)
    monitor.update_metric("memory_usage_mb", 450.0)
    monitor.update_metric("error_rate", 0.25)
    monitor.update_metric("session_age_minutes", 65.0)
    monitor.update_metric("browser_age_minutes", 40.0)
    monitor.update_metric("pages_since_refresh", 38.0)

    # Record very slow API calls
    for response_time in [12.1, 15.3, 18.0, 14.4, 16.2]:
        monitor.record_api_response_time(response_time)

    # Record many errors
    for _i in range(10):
        monitor.record_error("SessionDeathError")

    dashboard = monitor.get_health_dashboard()
    print(f"Health Score: {dashboard['health_score']:.1f}/100 ({dashboard['health_status'].upper()})")
    print(f"Risk Level: {dashboard['risk_level']} (Score: {dashboard['risk_score']:.2f})")
    print(f"Avg API Response: {dashboard['performance_summary']['avg_api_response_time']:.1f}s")
    print(f"Total Errors: {dashboard['performance_summary']['total_errors']}")

    recommendations = get_performance_recommendations(dashboard['health_score'], dashboard['risk_score'])
    print(f"Recommended Settings: {recommendations['max_concurrency']} workers, batch {recommendations['batch_size']}")
    print(f"Action Required: {recommendations['action_required']}")

    # Show recommended actions
    print("\n💡 Recommended Actions:")
    for action in dashboard['recommended_actions'][:3]:
        print(f"   • {action}")

    # Show recent alerts
    if dashboard['recent_alerts']:
        print("\n🚨 Recent Critical Alerts:")
        for alert in dashboard['recent_alerts'][-3:]:
            print(f"   {alert['level'].upper()}: {alert['message']}")

    print("\n" + "=" * 60)
    print("🎯 HEALTH MONITORING DEMONSTRATION COMPLETE")
    print("The system successfully detected performance degradation")
    print("and provided actionable recommendations for optimization!")

    return True

if __name__ == "__main__":
    demo_health_monitoring()
