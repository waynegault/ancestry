#!/usr/bin/env python3
"""
Enhanced logging configuration for Action 6.
Provides structured logging with health metrics and cascade detection.
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class Action6Formatter(logging.Formatter):
    """Custom formatter for Action 6 with structured data."""
    
    def format(self, record):
        # Add timestamp and structured data
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add cascade information if available
        if hasattr(record, 'cascade_count'):
            log_data['cascade_count'] = record.cascade_count
            
        # Add health metrics if available
        if hasattr(record, 'health_metrics'):
            log_data['health_metrics'] = record.health_metrics
            
        # Add API call information if available
        if hasattr(record, 'api_call_info'):
            log_data['api_call_info'] = record.api_call_info
            
        return json.dumps(log_data, indent=2 if record.levelno >= logging.WARNING else None)

class CascadeDetectionFilter(logging.Filter):
    """Filter to detect and highlight cascade-related log messages."""
    
    def filter(self, record):
        # Mark cascade-related messages
        cascade_keywords = ['cascade', 'session death', 'halt signal', 'emergency shutdown']
        message = record.getMessage().lower()
        
        if any(keyword in message for keyword in cascade_keywords):
            record.is_cascade_related = True
            
        return True

def setup_enhanced_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up enhanced logging for Action 6."""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("Logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger("action6")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(log_dir / "action6_enhanced.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(Action6Formatter())
    file_handler.addFilter(CascadeDetectionFilter())
    
    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    # Critical handler for cascade alerts
    critical_handler = logging.FileHandler(log_dir / "action6_critical.log")
    critical_handler.setLevel(logging.CRITICAL)
    critical_handler.setFormatter(Action6Formatter())
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(critical_handler)
    
    return logger

def log_cascade_event(logger: logging.Logger, cascade_count: int, details: Dict[str, Any]):
    """Log a cascade event with structured data."""
    logger.critical(
        f"🚨 SESSION DEATH CASCADE #{cascade_count}",
        extra={
            'cascade_count': cascade_count,
            'cascade_details': details
        }
    )

def log_health_metrics(logger: logging.Logger, metrics: Dict[str, Any]):
    """Log health metrics."""
    logger.info(
        f"📊 Health Check: Success Rate {metrics.get('success_rate', 0):.1f}%",
        extra={'health_metrics': metrics}
    )

def log_api_call(logger: logging.Logger, endpoint: str, success: bool, response_time: float, details: Optional[Dict] = None):
    """Log API call with performance metrics."""
    status = "✅ SUCCESS" if success else "❌ FAILED"
    logger.debug(
        f"{status} API Call: {endpoint} ({response_time:.2f}s)",
        extra={
            'api_call_info': {
                'endpoint': endpoint,
                'success': success,
                'response_time': response_time,
                'details': details or {}
            }
        }
    )

def log_emergency_shutdown(logger: logging.Logger, reason: str, context: Dict[str, Any]):
    """Log emergency shutdown event."""
    logger.critical(
        f"🚨 EMERGENCY SHUTDOWN: {reason}",
        extra={
            'shutdown_reason': reason,
            'shutdown_context': context
        }
    )
