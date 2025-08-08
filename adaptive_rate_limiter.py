"""
Adaptive Rate Limiting System for Ancestry Project

This module provides intelligent rate limiting that adapts based on API response
patterns, success rates, and system performance. Optimizes throughput while
maintaining API stability and avoiding 429 rate limit errors.

Author: Ancestry Automation System
Created: August 6, 2025
Phase: 11.1 - Configuration Optimization & Adaptive Processing
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import statistics

# Import standard modules
from standard_imports import *

# Set up logging
logger = get_logger(__name__)


@dataclass
class APIResponseMetrics:
    """Tracks API response metrics for adaptive rate limiting."""
    
    timestamp: datetime
    success: bool
    response_time: float
    status_code: Optional[int] = None
    error_type: Optional[str] = None


@dataclass
class RateLimitingStats:
    """Statistics for rate limiting performance."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limit_errors: int = 0
    average_response_time: float = 0.0
    current_rps: float = 0.0
    adaptive_adjustments: int = 0


class AdaptiveRateLimiter:
    """
    Intelligent rate limiter that adapts based on API response patterns.
    Optimizes throughput while maintaining stability and avoiding rate limits.
    """

    def __init__(
        self,
        initial_rps: float = 0.5,
        min_rps: float = 0.1,
        max_rps: float = 2.0,
        initial_delay: float = 2.0,
        min_delay: float = 0.5,
        max_delay: float = 10.0,
        adaptation_window: int = 50,
        success_threshold: float = 0.95,
        rate_limit_threshold: float = 0.02
    ):
        """
        Initialize adaptive rate limiter.
        
        Args:
            initial_rps: Starting requests per second
            min_rps: Minimum allowed RPS
            max_rps: Maximum allowed RPS
            initial_delay: Starting delay between requests
            min_delay: Minimum delay between requests
            max_delay: Maximum delay between requests
            adaptation_window: Number of recent requests to consider for adaptation
            success_threshold: Success rate threshold for increasing RPS
            rate_limit_threshold: Rate limit error threshold for decreasing RPS
        """
        self.initial_rps = initial_rps
        self.min_rps = min_rps
        self.max_rps = max_rps
        self.current_rps = initial_rps
        
        self.initial_delay = initial_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.current_delay = initial_delay
        
        self.adaptation_window = adaptation_window
        self.success_threshold = success_threshold
        self.rate_limit_threshold = rate_limit_threshold
        
        # Metrics tracking
        self.response_history: deque = deque(maxlen=adaptation_window)
        self.last_request_time: Optional[float] = None
        self.stats = RateLimitingStats()
        
        # Adaptive behavior settings
        self.adaptation_enabled = True
        self.last_adaptation_time = time.time()
        self.adaptation_cooldown = 30.0  # Seconds between adaptations
        
        logger.debug(f"Initialized adaptive rate limiter: {initial_rps} RPS, {initial_delay}s delay")

    def wait(self) -> float:
        """
        Wait for the appropriate amount of time based on current rate limiting.
        Returns the actual wait time.
        """
        current_time = time.time()
        
        if self.last_request_time is not None:
            time_since_last = current_time - self.last_request_time
            required_delay = 1.0 / self.current_rps
            
            if time_since_last < required_delay:
                wait_time = required_delay - time_since_last
                time.sleep(wait_time)
                self.last_request_time = time.time()
                return wait_time
        
        self.last_request_time = current_time
        return 0.0

    def record_response(
        self,
        success: bool,
        response_time: float,
        status_code: Optional[int] = None,
        error_type: Optional[str] = None
    ):
        """
        Record an API response for adaptive rate limiting.
        
        Args:
            success: Whether the request was successful
            response_time: Time taken for the request
            status_code: HTTP status code
            error_type: Type of error if request failed
        """
        # Record metrics
        metrics = APIResponseMetrics(
            timestamp=datetime.now(),
            success=success,
            response_time=response_time,
            status_code=status_code,
            error_type=error_type
        )
        
        self.response_history.append(metrics)
        
        # Update statistics
        self.stats.total_requests += 1
        if success:
            self.stats.successful_requests += 1
        else:
            self.stats.failed_requests += 1
            if status_code == 429 or "rate limit" in str(error_type).lower():
                self.stats.rate_limit_errors += 1
        
        # Calculate current stats
        if self.response_history:
            response_times = [m.response_time for m in self.response_history]
            self.stats.average_response_time = statistics.mean(response_times)
            self.stats.current_rps = self.current_rps
        
        # Trigger adaptation if enabled and enough data
        if self.adaptation_enabled and len(self.response_history) >= 10:
            self._adapt_rate_limiting()

    def _adapt_rate_limiting(self):
        """Adapt rate limiting based on recent response patterns."""
        current_time = time.time()
        
        # Check adaptation cooldown
        if current_time - self.last_adaptation_time < self.adaptation_cooldown:
            return
        
        # Calculate recent metrics
        recent_responses = list(self.response_history)[-20:]  # Last 20 responses
        if len(recent_responses) < 10:
            return
        
        success_rate = sum(1 for r in recent_responses if r.success) / len(recent_responses)
        rate_limit_rate = sum(1 for r in recent_responses 
                             if r.status_code == 429 or "rate limit" in str(r.error_type).lower()) / len(recent_responses)
        avg_response_time = statistics.mean([r.response_time for r in recent_responses])
        
        # Determine adaptation strategy
        adaptation_made = False
        
        # Simple rate limiting response
        if rate_limit_rate > 0.02:  # More than 2% rate limiting
            new_rps = max(self.min_rps, self.current_rps * 0.5)
            if new_rps != self.current_rps:
                logger.info(f"Decreasing RPS due to rate limiting: {self.current_rps:.2f} → {new_rps:.2f}")
                self.current_rps = new_rps
                self.current_delay = min(self.max_delay, self.current_delay * 2.0)
                adaptation_made = True
        
        # Increase RPS more aggressively if performance is excellent
        elif success_rate > 0.98 and avg_response_time < 1.0 and rate_limit_rate == 0:
            # Aggressive increase for excellent performance
            new_rps = min(self.max_rps, self.current_rps * 1.4)
            if new_rps != self.current_rps:
                logger.info(f"Aggressively increasing RPS due to excellent performance: {self.current_rps:.2f} → {new_rps:.2f}")
                self.current_rps = new_rps
                self.current_delay = max(self.min_delay, self.current_delay * 0.7)
                adaptation_made = True

        # Moderate increase for good performance
        elif success_rate > self.success_threshold and avg_response_time < 2.0:
            new_rps = min(self.max_rps, self.current_rps * 1.2)  # Increased from 1.1 to 1.2
            if new_rps != self.current_rps:
                logger.info(f"Increasing RPS due to good performance: {self.current_rps:.2f} → {new_rps:.2f}")
                self.current_rps = new_rps
                self.current_delay = max(self.min_delay, self.current_delay * 0.85)  # More aggressive decrease
                adaptation_made = True
        
        # Decrease RPS if success rate is low (more conservative threshold)
        elif success_rate < 0.9:  # Increased threshold from 0.8 to 0.9 for earlier intervention
            new_rps = max(self.min_rps, self.current_rps * 0.7)  # More aggressive reduction from 0.8 to 0.7
            if new_rps != self.current_rps:
                logger.info(f"Decreasing RPS due to low success rate: {self.current_rps:.2f} → {new_rps:.2f}")
                self.current_rps = new_rps
                self.current_delay = min(self.max_delay, self.current_delay * 1.5)  # Increased from 1.2 to 1.5
                adaptation_made = True
        
        if adaptation_made:
            self.stats.adaptive_adjustments += 1
            self.last_adaptation_time = current_time

            # Simple cooldown adjustment
            if rate_limit_rate > 0:
                self.adaptation_cooldown = min(30.0, self.adaptation_cooldown * 1.2)
            elif success_rate > 0.95:
                self.adaptation_cooldown = max(10.0, self.adaptation_cooldown * 0.9)

            logger.debug(f"⚡ Adaptive rate limiting: RPS={self.current_rps:.2f}, Delay={self.current_delay:.2f}s, "
                        f"Success={success_rate:.2%}, RateLimit={rate_limit_rate:.2%}")

    def get_current_settings(self) -> Dict[str, float]:
        """Get current rate limiting settings."""
        return {
            "rps": self.current_rps,
            "delay": self.current_delay,
            "requests_per_second": self.current_rps,
            "delay_between_requests": 1.0 / self.current_rps
        }

    def get_statistics(self) -> RateLimitingStats:
        """Get current rate limiting statistics."""
        return self.stats

    def reset_to_defaults(self):
        """Reset rate limiting to initial settings."""
        self.current_rps = self.initial_rps
        self.current_delay = self.initial_delay
        self.response_history.clear()
        self.stats = RateLimitingStats()
        logger.info(f"Reset adaptive rate limiter to defaults: {self.initial_rps} RPS, {self.initial_delay}s delay")

    def enable_adaptation(self):
        """Enable adaptive rate limiting."""
        self.adaptation_enabled = True
        logger.info("Enabled adaptive rate limiting")

    def disable_adaptation(self):
        """Disable adaptive rate limiting."""
        self.adaptation_enabled = False
        logger.info("Disabled adaptive rate limiting")

    def record_rate_limit(self):
        """Simple rate limit recording - just increase delay."""
        self.current_delay = min(self.max_delay, self.current_delay * 2.0)
        logger.info(f"Rate limit detected, increasing delay to {self.current_delay:.1f}s")

    def is_throttled(self) -> bool:
        """Check if currently throttled due to rate limiting."""
        return self.current_rps < self.initial_rps * 0.8

    def get_performance_report(self) -> str:
        """Generate a performance report for the adaptive rate limiter."""
        if not self.response_history:
            return "No performance data available"
        
        recent_responses = list(self.response_history)[-50:]
        success_rate = sum(1 for r in recent_responses if r.success) / len(recent_responses)
        avg_response_time = statistics.mean([r.response_time for r in recent_responses])
        rate_limit_errors = sum(1 for r in recent_responses 
                               if r.status_code == 429 or "rate limit" in str(r.error_type).lower())
        
        report = f"""
Adaptive Rate Limiter Performance Report:
==========================================
Current Settings:
  - RPS: {self.current_rps:.2f} (Range: {self.min_rps:.2f} - {self.max_rps:.2f})
  - Delay: {self.current_delay:.2f}s (Range: {self.min_delay:.2f} - {self.max_delay:.2f})
  - Adaptation: {'Enabled' if self.adaptation_enabled else 'Disabled'}

Recent Performance (Last {len(recent_responses)} requests):
  - Success Rate: {success_rate:.2%}
  - Average Response Time: {avg_response_time:.2f}s
  - Rate Limit Errors: {rate_limit_errors}
  - Adaptive Adjustments: {self.stats.adaptive_adjustments}

Overall Statistics:
  - Total Requests: {self.stats.total_requests}
  - Successful: {self.stats.successful_requests}
  - Failed: {self.stats.failed_requests}
  - Rate Limit Errors: {self.stats.rate_limit_errors}
"""
        return report.strip()


class SmartBatchProcessor:
    """
    Intelligent batch processing that adapts batch sizes based on performance.
    """

    def __init__(
        self,
        initial_batch_size: int = 5,
        min_batch_size: int = 1,
        max_batch_size: int = 20,
        target_processing_time: float = 30.0
    ):
        """
        Initialize smart batch processor.
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            target_processing_time: Target time per batch in seconds
        """
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        self.target_processing_time = target_processing_time
        
        self.processing_history: deque = deque(maxlen=20)
        self.last_adaptation_time = time.time()
        self.adaptation_cooldown = 60.0  # Seconds between batch size adaptations
        
        logger.debug(f"Initialized smart batch processor: {initial_batch_size} batch size")

    def get_next_batch_size(self) -> int:
        """Get the recommended batch size for the next batch."""
        return self.current_batch_size

    def record_batch_performance(self, batch_size: int, processing_time: float, success_rate: float):
        """
        Record batch processing performance.
        
        Args:
            batch_size: Size of the processed batch
            processing_time: Time taken to process the batch
            success_rate: Success rate of the batch (0.0 to 1.0)
        """
        self.processing_history.append({
            "batch_size": batch_size,
            "processing_time": processing_time,
            "success_rate": success_rate,
            "timestamp": time.time()
        })
        
        # Adapt batch size if enough data and cooldown passed
        current_time = time.time()
        if (len(self.processing_history) >= 3 and 
            current_time - self.last_adaptation_time > self.adaptation_cooldown):
            self._adapt_batch_size()

    def _adapt_batch_size(self):
        """Adapt batch size based on recent performance."""
        if len(self.processing_history) < 3:
            return
        
        recent_batches = list(self.processing_history)[-5:]
        avg_processing_time = statistics.mean([b["processing_time"] for b in recent_batches])
        avg_success_rate = statistics.mean([b["success_rate"] for b in recent_batches])
        
        adaptation_made = False
        
        # Increase batch size if processing is fast and successful
        if (avg_processing_time < self.target_processing_time * 0.7 and 
            avg_success_rate > 0.95 and 
            self.current_batch_size < self.max_batch_size):
            
            new_batch_size = min(self.max_batch_size, self.current_batch_size + 2)
            logger.info(f"Increasing batch size: {self.current_batch_size} → {new_batch_size}")
            self.current_batch_size = new_batch_size
            adaptation_made = True
        
        # Decrease batch size if processing is slow or unsuccessful
        elif (avg_processing_time > self.target_processing_time * 1.3 or 
              avg_success_rate < 0.8) and self.current_batch_size > self.min_batch_size:
            
            new_batch_size = max(self.min_batch_size, self.current_batch_size - 1)
            logger.info(f"Decreasing batch size: {self.current_batch_size} → {new_batch_size}")
            self.current_batch_size = new_batch_size
            adaptation_made = True
        
        if adaptation_made:
            self.last_adaptation_time = time.time()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for batch processing."""
        if not self.processing_history:
            return {"status": "No data available"}
        
        recent_batches = list(self.processing_history)
        return {
            "current_batch_size": self.current_batch_size,
            "batches_processed": len(recent_batches),
            "average_processing_time": statistics.mean([b["processing_time"] for b in recent_batches]),
            "average_success_rate": statistics.mean([b["success_rate"] for b in recent_batches]),
            "target_processing_time": self.target_processing_time
        }


# Test functions
def test_adaptive_rate_limiter():
    """Test the adaptive rate limiter system."""
    logger.info("Testing adaptive rate limiter system...")
    
    limiter = AdaptiveRateLimiter(initial_rps=1.0, min_rps=0.5, max_rps=2.0)
    
    # Test basic functionality
    wait_time = limiter.wait()
    assert wait_time >= 0, "Wait time should be non-negative"
    
    # Test response recording
    limiter.record_response(success=True, response_time=0.5, status_code=200)
    limiter.record_response(success=False, response_time=2.0, status_code=429, error_type="rate limit")
    
    stats = limiter.get_statistics()
    assert stats.total_requests == 2, "Should have recorded 2 requests"
    assert stats.rate_limit_errors == 1, "Should have recorded 1 rate limit error"
    
    # Test settings retrieval
    settings = limiter.get_current_settings()
    assert "rps" in settings, "Settings should include RPS"
    assert "delay" in settings, "Settings should include delay"
    
    logger.info("✅ Adaptive rate limiter test passed")
    return True


def test_smart_batch_processor():
    """Test the smart batch processor system."""
    logger.info("Testing smart batch processor system...")
    
    processor = SmartBatchProcessor(initial_batch_size=5, min_batch_size=1, max_batch_size=10)
    
    # Test basic functionality
    batch_size = processor.get_next_batch_size()
    assert batch_size == 5, "Should return initial batch size"
    
    # Test performance recording
    processor.record_batch_performance(batch_size=5, processing_time=10.0, success_rate=0.95)
    
    summary = processor.get_performance_summary()
    assert "current_batch_size" in summary, "Summary should include current batch size"
    assert summary["batches_processed"] == 1, "Should have recorded 1 batch"
    
    logger.info("✅ Smart batch processor test passed")
    return True


class ConfigurationOptimizer:
    """
    Analyzes system performance and recommends configuration optimizations.
    """

    def __init__(self):
        """Initialize configuration optimizer."""
        self.performance_history: deque = deque(maxlen=100)
        self.optimization_recommendations: List[Dict[str, Any]] = []

    def analyze_performance(
        self,
        rate_limiter_stats: RateLimitingStats,
        batch_processor_summary: Dict[str, Any],
        system_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze system performance and generate optimization recommendations.

        Args:
            rate_limiter_stats: Statistics from adaptive rate limiter
            batch_processor_summary: Summary from smart batch processor
            system_metrics: Optional system performance metrics

        Returns:
            Analysis results with recommendations
        """
        analysis = {
            "timestamp": datetime.now(),
            "rate_limiting_analysis": self._analyze_rate_limiting(rate_limiter_stats),
            "batch_processing_analysis": self._analyze_batch_processing(batch_processor_summary),
            "recommendations": []
        }

        # Generate recommendations based on analysis
        recommendations = []

        # Rate limiting recommendations
        if rate_limiter_stats.rate_limit_errors > rate_limiter_stats.total_requests * 0.05:
            recommendations.append({
                "type": "rate_limiting",
                "priority": "high",
                "description": "High rate limit error rate detected",
                "recommendation": "Consider reducing initial RPS or increasing delays",
                "current_value": rate_limiter_stats.current_rps,
                "suggested_value": rate_limiter_stats.current_rps * 0.8
            })

        # Batch processing recommendations
        if batch_processor_summary.get("average_processing_time", 0) > 60:
            recommendations.append({
                "type": "batch_processing",
                "priority": "medium",
                "description": "Batch processing time is high",
                "recommendation": "Consider reducing batch size for better responsiveness",
                "current_value": batch_processor_summary.get("current_batch_size", 5),
                "suggested_value": max(1, batch_processor_summary.get("current_batch_size", 5) - 2)
            })

        # Performance recommendations
        success_rate = (rate_limiter_stats.successful_requests /
                       max(1, rate_limiter_stats.total_requests))
        if success_rate > 0.98 and rate_limiter_stats.average_response_time < 1.0:
            recommendations.append({
                "type": "performance",
                "priority": "low",
                "description": "System performing well, could increase throughput",
                "recommendation": "Consider increasing RPS or batch sizes for better throughput",
                "current_rps": rate_limiter_stats.current_rps,
                "suggested_rps": min(2.0, rate_limiter_stats.current_rps * 1.2)
            })

        analysis["recommendations"] = recommendations
        self.optimization_recommendations.extend(recommendations)

        return analysis

    def _analyze_rate_limiting(self, stats: RateLimitingStats) -> Dict[str, Any]:
        """Analyze rate limiting performance."""
        if stats.total_requests == 0:
            return {"status": "insufficient_data"}

        success_rate = stats.successful_requests / stats.total_requests
        error_rate = stats.rate_limit_errors / stats.total_requests

        return {
            "success_rate": success_rate,
            "error_rate": error_rate,
            "average_response_time": stats.average_response_time,
            "current_rps": stats.current_rps,
            "adaptive_adjustments": stats.adaptive_adjustments,
            "status": "healthy" if success_rate > 0.95 and error_rate < 0.02 else "needs_attention"
        }

    def _analyze_batch_processing(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze batch processing performance."""
        if not summary or summary.get("batches_processed", 0) == 0:
            return {"status": "insufficient_data"}

        avg_time = summary.get("average_processing_time", 0)
        avg_success = summary.get("average_success_rate", 0)
        target_time = summary.get("target_processing_time", 30)

        return {
            "average_processing_time": avg_time,
            "average_success_rate": avg_success,
            "current_batch_size": summary.get("current_batch_size", 5),
            "efficiency": target_time / max(1, avg_time),
            "status": "optimal" if avg_time < target_time and avg_success > 0.95 else "suboptimal"
        }

    def get_optimization_report(self) -> str:
        """Generate a comprehensive optimization report."""
        if not self.optimization_recommendations:
            return "No optimization recommendations available yet."

        high_priority = [r for r in self.optimization_recommendations if r.get("priority") == "high"]
        medium_priority = [r for r in self.optimization_recommendations if r.get("priority") == "medium"]
        low_priority = [r for r in self.optimization_recommendations if r.get("priority") == "low"]

        report = "Configuration Optimization Report:\n"
        report += "=" * 40 + "\n\n"

        if high_priority:
            report += "🔴 HIGH PRIORITY RECOMMENDATIONS:\n"
            for rec in high_priority[-3:]:  # Last 3 high priority
                report += f"  • {rec['description']}\n"
                report += f"    {rec['recommendation']}\n\n"

        if medium_priority:
            report += "🟡 MEDIUM PRIORITY RECOMMENDATIONS:\n"
            for rec in medium_priority[-3:]:  # Last 3 medium priority
                report += f"  • {rec['description']}\n"
                report += f"    {rec['recommendation']}\n\n"

        if low_priority:
            report += "🟢 LOW PRIORITY RECOMMENDATIONS:\n"
            for rec in low_priority[-2:]:  # Last 2 low priority
                report += f"  • {rec['description']}\n"
                report += f"    {rec['recommendation']}\n\n"

        return report.strip()


def test_configuration_optimizer():
    """Test the configuration optimizer system."""
    logger.info("Testing configuration optimizer system...")

    optimizer = ConfigurationOptimizer()

    # Test data
    test_stats = RateLimitingStats(
        total_requests=100,
        successful_requests=95,
        failed_requests=5,
        rate_limit_errors=2,
        average_response_time=1.5,
        current_rps=1.0
    )

    test_batch_summary = {
        "current_batch_size": 5,
        "batches_processed": 10,
        "average_processing_time": 25.0,
        "average_success_rate": 0.96,
        "target_processing_time": 30.0
    }

    # Test analysis
    analysis = optimizer.analyze_performance(test_stats, test_batch_summary)

    assert "rate_limiting_analysis" in analysis, "Analysis should include rate limiting"
    assert "batch_processing_analysis" in analysis, "Analysis should include batch processing"
    assert "recommendations" in analysis, "Analysis should include recommendations"

    logger.info("✅ Configuration optimizer test passed")
    return True


def adaptive_rate_limiter_module_tests() -> bool:
    """
    Comprehensive test suite for adaptive_rate_limiter.py with real functionality testing.
    Tests adaptive rate limiting, smart batch processing, and configuration optimization.
    """
    from test_framework import TestSuite, suppress_logging

    suite = TestSuite("Adaptive Rate Limiting & Optimization", "adaptive_rate_limiter.py")
    suite.start_suite()

    with suppress_logging():
        suite.run_test(
            "Adaptive rate limiter system",
            test_adaptive_rate_limiter,
            "Dynamic rate limiting with automatic adjustment based on response times",
            "Test adaptive rate limiter with response time monitoring and adjustment",
            "Test AdaptiveRateLimiter with initial_rps=1.0, min_rps=0.5, max_rps=2.0",
        )

        suite.run_test(
            "Smart batch processor system",
            test_smart_batch_processor,
            "Intelligent batch processing with dynamic size optimization",
            "Test smart batch processor with adaptive batch sizing",
            "Test SmartBatchProcessor with initial_batch_size=5, min=1, max=10",
        )

        suite.run_test(
            "Configuration optimizer system",
            test_configuration_optimizer,
            "Configuration optimization with performance analysis and recommendations",
            "Test configuration optimizer with performance metrics and suggestions",
            "Test ConfigurationOptimizer with performance analysis and optimization recommendations",
        )

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive adaptive rate limiter tests using standardized TestSuite format."""
    return adaptive_rate_limiter_module_tests()


if __name__ == "__main__":
    """
    Execute comprehensive adaptive rate limiter tests when run directly.
    Tests adaptive rate limiting, smart batch processing, and configuration optimization.
    """
    success = run_comprehensive_tests()
    import sys
    sys.exit(0 if success else 1)
