"""
Performance Monitoring Dashboard for Ancestry Project

This module provides comprehensive performance monitoring and reporting
for the adaptive rate limiting and configuration optimization systems.

Author: Ancestry Automation System
Created: August 6, 2025
Phase: 11.1 - Configuration Optimization & Adaptive Processing
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import standard modules
from standard_imports import *

# Set up logging
logger = get_logger(__name__)


class PerformanceDashboard:
    """
    Comprehensive performance monitoring dashboard for adaptive systems.
    """

    def __init__(self, data_file: Optional[str] = None):
        """
        Initialize performance dashboard.

        Args:
            data_file: Optional file path to store performance data
        """
        self.data_file = data_file or "performance_data.json"
        self.performance_data: Dict[str, Any] = self._load_performance_data()
        self.session_start_time = datetime.now()

    def _load_performance_data(self) -> Dict[str, Any]:
        """Load existing performance data from file."""
        try:
            data_path = Path(self.data_file)
            if data_path.exists():
                with data_path.open('r') as f:
                    data = json.load(f)
                logger.debug(f"Loaded performance data from {self.data_file}")
                return data
        except Exception as e:
            logger.warning(f"Could not load performance data: {e}")

        # Return default structure
        return {
            "sessions": [],
            "rate_limiting_history": [],
            "batch_processing_history": [],
            "optimization_history": [],
            "system_metrics": []
        }

    def _save_performance_data(self):
        """Save performance data to file."""
        try:
            data_path = Path(self.data_file)
            with data_path.open('w') as f:
                json.dump(self.performance_data, f, indent=2, default=str)
            logger.debug(f"Saved performance data to {self.data_file}")
        except Exception as e:
            logger.error(f"Could not save performance data: {e}")

    def record_session_start(self, session_info: Dict[str, Any]):
        """Record the start of a new session."""
        session_data = {
            "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "start_time": datetime.now().isoformat(),
            "session_info": session_info,
            "metrics": []
        }

        self.performance_data["sessions"].append(session_data)
        self.current_session = session_data
        logger.info(f"Started performance monitoring session: {session_data['session_id']}")

    def record_rate_limiting_metrics(self, metrics: Dict[str, Any]):
        """Record rate limiting performance metrics."""
        metric_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "rate_limiting",
            "metrics": metrics
        }

        self.performance_data["rate_limiting_history"].append(metric_entry)

        # Also add to current session if available
        if hasattr(self, 'current_session'):
            self.current_session["metrics"].append(metric_entry)

    def record_batch_processing_metrics(self, metrics: Dict[str, Any]):
        """Record batch processing performance metrics."""
        metric_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "batch_processing",
            "metrics": metrics
        }

        self.performance_data["batch_processing_history"].append(metric_entry)

        # Also add to current session if available
        if hasattr(self, 'current_session'):
            self.current_session["metrics"].append(metric_entry)

    def record_optimization_event(self, optimization_data: Dict[str, Any]):
        """Record configuration optimization events."""
        optimization_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "optimization",
            "data": optimization_data
        }

        self.performance_data["optimization_history"].append(optimization_entry)

        # Also add to current session if available
        if hasattr(self, 'current_session'):
            self.current_session["metrics"].append(optimization_entry)

    def record_system_metrics(self, system_data: Dict[str, Any]):
        """Record general system performance metrics."""
        system_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "system",
            "metrics": system_data
        }

        self.performance_data["system_metrics"].append(system_entry)

        # Also add to current session if available
        if hasattr(self, 'current_session'):
            self.current_session["metrics"].append(system_entry)

    def generate_performance_report(self, hours_back: int = 24) -> str:
        """
        Generate a comprehensive performance report.

        Args:
            hours_back: Number of hours of data to include in report

        Returns:
            Formatted performance report
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        # Filter recent data
        recent_rate_limiting = [
            entry for entry in self.performance_data["rate_limiting_history"]
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]

        recent_batch_processing = [
            entry for entry in self.performance_data["batch_processing_history"]
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]

        recent_optimizations = [
            entry for entry in self.performance_data["optimization_history"]
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]

        # Generate report
        report = f"""
ANCESTRY PROJECT PERFORMANCE REPORT
===================================
Report Period: Last {hours_back} hours
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RATE LIMITING PERFORMANCE:
{self._generate_rate_limiting_summary(recent_rate_limiting)}

BATCH PROCESSING PERFORMANCE:
{self._generate_batch_processing_summary(recent_batch_processing)}

OPTIMIZATION EVENTS:
{self._generate_optimization_summary(recent_optimizations)}

RECOMMENDATIONS:
{self._generate_recommendations(recent_rate_limiting, recent_batch_processing)}
"""

        return report.strip()

    def _generate_rate_limiting_summary(self, rate_limiting_data: List[Dict]) -> str:
        """Generate rate limiting performance summary."""
        if not rate_limiting_data:
            return "No rate limiting data available"

        latest = rate_limiting_data[-1]["metrics"]

        summary = f"""
  Current RPS: {latest.get('current_rps', 'N/A')}
  Success Rate: {latest.get('success_rate', 0):.2%}
  Average Response Time: {latest.get('average_response_time', 0):.2f}s
  Rate Limit Errors: {latest.get('rate_limit_errors', 0)}
  Adaptive Adjustments: {latest.get('adaptive_adjustments', 0)}
  Status: {latest.get('status', 'Unknown')}
"""

        return summary.strip()

    def _generate_batch_processing_summary(self, batch_data: List[Dict]) -> str:
        """Generate batch processing performance summary."""
        if not batch_data:
            return "No batch processing data available"

        latest = batch_data[-1]["metrics"]

        summary = f"""
  Current Batch Size: {latest.get('current_batch_size', 'N/A')}
  Average Processing Time: {latest.get('average_processing_time', 0):.1f}s
  Average Success Rate: {latest.get('average_success_rate', 0):.2%}
  Batches Processed: {latest.get('batches_processed', 0)}
  Efficiency: {latest.get('efficiency', 0):.2f}
  Status: {latest.get('status', 'Unknown')}
"""

        return summary.strip()

    def _generate_optimization_summary(self, optimization_data: List[Dict]) -> str:
        """Generate optimization events summary."""
        if not optimization_data:
            return "No optimization events recorded"

        high_priority = sum(1 for opt in optimization_data
                           if any(rec.get('priority') == 'high'
                                 for rec in opt.get('data', {}).get('recommendations', [])))

        medium_priority = sum(1 for opt in optimization_data
                             if any(rec.get('priority') == 'medium'
                                   for rec in opt.get('data', {}).get('recommendations', [])))

        summary = f"""
  Total Optimization Events: {len(optimization_data)}
  High Priority Recommendations: {high_priority}
  Medium Priority Recommendations: {medium_priority}
  Latest Event: {optimization_data[-1]['timestamp'] if optimization_data else 'None'}
"""

        return summary.strip()

    def _generate_recommendations(self, rate_data: List[Dict], batch_data: List[Dict]) -> str:
        """Generate performance recommendations."""
        recommendations = []

        # Analyze rate limiting performance
        if rate_data:
            latest_rate = rate_data[-1]["metrics"]
            success_rate = latest_rate.get('success_rate', 0)
            error_rate = latest_rate.get('error_rate', 0)

            if success_rate < 0.9:
                recommendations.append("🔴 Low success rate detected - consider reducing RPS")
            elif success_rate > 0.98 and error_rate < 0.01:
                recommendations.append("🟢 Excellent performance - consider increasing RPS")

        # Analyze batch processing performance
        if batch_data:
            latest_batch = batch_data[-1]["metrics"]
            avg_time = latest_batch.get('average_processing_time', 0)

            if avg_time > 60:
                recommendations.append("🟡 Batch processing time high - consider reducing batch size")
            elif avg_time < 15:
                recommendations.append("🟢 Fast batch processing - consider increasing batch size")

        if not recommendations:
            recommendations.append("✅ System performing within normal parameters")

        return "\n  ".join(["", *recommendations])

    def get_current_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session performance."""
        if not hasattr(self, 'current_session'):
            return {"status": "No active session"}

        session_duration = datetime.now() - self.session_start_time
        metrics_count = len(self.current_session.get("metrics", []))

        return {
            "session_id": self.current_session.get("session_id"),
            "duration": str(session_duration),
            "metrics_recorded": metrics_count,
            "start_time": self.session_start_time.isoformat()
        }

    def export_data(self, export_file: str) -> bool:
        """Export performance data to a file."""
        try:
            export_path = Path(export_file)
            with export_path.open('w') as f:
                json.dump(self.performance_data, f, indent=2, default=str)
            logger.info(f"Exported performance data to {export_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to export performance data: {e}")
            return False

    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old performance data."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        # Clean up each data type
        for data_type in ["rate_limiting_history", "batch_processing_history",
                         "optimization_history", "system_metrics"]:
            if data_type in self.performance_data:
                original_count = len(self.performance_data[data_type])
                self.performance_data[data_type] = [
                    entry for entry in self.performance_data[data_type]
                    if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
                ]
                cleaned_count = original_count - len(self.performance_data[data_type])
                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} old {data_type} entries")

        # Clean up old sessions
        if "sessions" in self.performance_data:
            original_count = len(self.performance_data["sessions"])
            self.performance_data["sessions"] = [
                session for session in self.performance_data["sessions"]
                if datetime.fromisoformat(session["start_time"]) > cutoff_date
            ]
            cleaned_count = original_count - len(self.performance_data["sessions"])
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old session entries")

        self._save_performance_data()

    def finalize_session(self):
        """Finalize the current session and save data."""
        if hasattr(self, 'current_session'):
            self.current_session["end_time"] = datetime.now().isoformat()
            self.current_session["duration"] = str(datetime.now() - self.session_start_time)
            logger.info(f"Finalized session: {self.current_session['session_id']}")

        self._save_performance_data()


# Test functions
def test_performance_dashboard():
    """Test the performance dashboard system."""
    try:
        # Use a simple test that doesn't rely on file I/O
        dashboard = PerformanceDashboard(":memory:")  # Use in-memory to avoid file issues

        # Test basic instantiation
        assert dashboard is not None, "Dashboard should instantiate"
        assert hasattr(dashboard, 'performance_data'), "Dashboard should have performance_data"

        # Test basic functionality without file operations
        dashboard.performance_data = {
            "sessions": [],
            "rate_limiting_history": [],
            "batch_processing_history": [],
            "optimization_history": [],
            "system_metrics": []
        }

        # Test report generation with minimal data
        report = dashboard.generate_performance_report(hours_back=1)
        assert isinstance(report, str), "Report should be a string"

        return True
    except Exception as e:
        print(f"Performance dashboard test failed: {e}")
        return False


def test_session_and_metric_recording():
    """Verify session start, metric recording, and session summary structure."""
    dash = PerformanceDashboard(":memory:")
    dash.record_session_start({"purpose": "unit_test"})
    dash.record_rate_limiting_metrics({"current_rps": 5, "success_rate": 0.95, "error_rate": 0.02})
    dash.record_batch_processing_metrics({"current_batch_size": 10, "average_processing_time": 12, "average_success_rate": 0.99})
    summary = dash.get_current_session_summary()
    assert summary.get("metrics_recorded", 0) >= 2


def test_report_recommendations_variants():
    """Trigger different recommendation branches (low success vs good)."""
    dash = PerformanceDashboard(":memory:")
    dash.record_rate_limiting_metrics({"success_rate": 0.85, "error_rate": 0.05})
    dash.record_batch_processing_metrics({"average_processing_time": 70})
    report1 = dash.generate_performance_report(hours_back=1)
    assert "Low success rate" in report1 or "low success rate" in report1.lower()
    dash.record_rate_limiting_metrics({"success_rate": 0.99, "error_rate": 0.0})
    dash.record_batch_processing_metrics({"average_processing_time": 5})
    report2 = dash.generate_performance_report(hours_back=1)
    assert "Excellent performance" in report2 or "Fast batch processing" in report2


def test_export_and_cleanup():
    """Test data export and cleanup_old_data does not raise and trims entries."""
    dash = PerformanceDashboard("temp_perf_data.json")
    dash.record_rate_limiting_metrics({"success_rate": 0.95})
    assert dash.export_data("temp_perf_export.json")
    # Inject an old entry
    old_timestamp = (datetime.now() - timedelta(days=40)).isoformat()
    dash.performance_data["rate_limiting_history"].append({"timestamp": old_timestamp, "type": "rate_limiting", "metrics": {}})
    before = len(dash.performance_data["rate_limiting_history"])
    dash.cleanup_old_data(days_to_keep=30)
    after = len(dash.performance_data["rate_limiting_history"])
    assert after < before
    # Cleanup files
    for fn in ["temp_perf_data.json", "temp_perf_export.json"]:
        try:
            Path(fn).unlink()
        except Exception:
            pass


def performance_dashboard_module_tests() -> bool:
    """
    Comprehensive test suite for performance_dashboard.py with real functionality testing.
    Tests performance monitoring, metrics collection, and dashboard visualization systems.
    """
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Performance Monitoring & Dashboard", "performance_dashboard.py")
    suite.start_suite()

    with suppress_logging():
        suite.run_test(
            "Dashboard basic smoke",
            test_performance_dashboard,
            "Instantiate and generate basic report",
            "Create PerformanceDashboard and call generate_performance_report",
            "Basic instantiation + report",
        )
        suite.run_test(
            "Session + metric recording",
            test_session_and_metric_recording,
            "Session summary reflects recorded metrics",
            "Start session and record metrics then inspect summary",
            "Session summary correctness",
        )
        suite.run_test(
            "Recommendation variants",
            test_report_recommendations_variants,
            "Report contains variant recommendations for different performance states",
            "Record contrasting metrics then generate reports",
            "Recommendation branch coverage",
        )
        suite.run_test(
            "Export + cleanup",
            test_export_and_cleanup,
            "Export succeeds and cleanup removes old entries",
            "Inject old data then cleanup",
            "Export and retention maintenance",
        )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive performance dashboard tests using standardized TestSuite format."""
    return performance_dashboard_module_tests()


if __name__ == "__main__":
    """
    Execute comprehensive performance dashboard tests when run directly.
    Tests performance monitoring, metrics collection, and dashboard visualization systems.
    """
    success = run_comprehensive_tests()
    import sys
    sys.exit(0 if success else 1)
