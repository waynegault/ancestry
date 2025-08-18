#!/usr/bin/env python3

"""
Reliable Session Manager - Simplified Architecture for Action 6

This module implements a simplified, reliable session management system designed to
address the critical race conditions and reliability issues identified in the 
browser refresh system.

Key Principles:
- Single browser instance per session (no concurrency)
- Proactive restart strategy every N pages
- Immediate error detection and halt on critical errors
- Resource-aware operations with health monitoring
- Simple state management for easy backup/restore

Design Goals:
- Eliminate "WebDriver became None" race conditions
- Prevent cascade failures through early detection
- Ensure reliable processing of 724-page workloads
- Maintain simplicity while improving reliability
"""

import time
import logging
from typing import Optional, Dict, Any, List
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class CriticalError(Exception):
    """Exception raised for critical errors that require immediate halt."""
    pass


class ResourceNotReadyError(Exception):
    """Exception raised when system resources are not ready for operation."""
    pass


class BrowserStartupError(Exception):
    """Exception raised when browser fails to start properly."""
    pass


class BrowserValidationError(Exception):
    """Exception raised when browser fails validation checks."""
    pass


class BrowserRestartError(Exception):
    """Exception raised when browser restart fails."""
    pass


class SystemHealthError(Exception):
    """Exception raised when system health checks fail."""
    pass


@dataclass
class SessionState:
    """
    Lightweight session state management for reliable backup/restore.
    
    Focuses on essential state only to minimize complexity.
    """
    current_page: int = 0
    pages_processed: int = 0
    session_start_time: float = 0.0
    last_successful_page: int = 0
    error_count: int = 0
    restart_count: int = 0
    
    def __post_init__(self):
        if self.session_start_time == 0.0:
            self.session_start_time = time.time()
    
    def create_backup(self) -> Dict[str, Any]:
        """Create lightweight backup of current state."""
        return {
            'current_page': self.current_page,
            'pages_processed': self.pages_processed,
            'last_successful_page': self.last_successful_page,
            'error_count': self.error_count,
            'restart_count': self.restart_count,
            'backup_timestamp': time.time()
        }
        
    def restore_backup(self, backup: Dict[str, Any]) -> None:
        """Restore state from backup."""
        if backup:
            self.current_page = backup.get('current_page', 0)
            self.pages_processed = backup.get('pages_processed', 0)
            self.last_successful_page = backup.get('last_successful_page', 0)
            self.error_count = backup.get('error_count', 0)
            self.restart_count = backup.get('restart_count', 0)
            
    def update_progress(self, page_num: int, success: bool = True) -> None:
        """Update progress tracking."""
        self.current_page = page_num
        if success:
            self.last_successful_page = page_num
            self.pages_processed += 1
        else:
            self.error_count += 1


class CriticalErrorDetector:
    """
    Sophisticated error detection system for early intervention.
    
    Focuses on specific error patterns that lead to cascade failures.
    """
    
    def __init__(self):
        self.error_patterns = {
            'webdriver_death': {
                'patterns': [
                    'WebDriver became None',
                    'invalid session id',
                    'session deleted',
                    'chrome not reachable',
                    'browser process died'
                ],
                'severity': 'critical',
                'action': 'immediate_halt'
            },
            'memory_pressure': {
                'patterns': [
                    'OutOfMemoryError',
                    'MemoryError',
                    'cannot allocate memory',
                    'virtual memory exhausted'
                ],
                'severity': 'critical',
                'action': 'immediate_restart'
            },
            'network_failure': {
                'patterns': [
                    'ConnectionError',
                    'TimeoutError',
                    'DNS resolution failed',
                    'network unreachable'
                ],
                'severity': 'warning',
                'action': 'retry_with_backoff'
            },
            'auth_loss': {
                'patterns': [
                    'login',
                    'signin',
                    'authenticate',
                    'unauthorized',
                    '401',
                    '403'
                ],
                'severity': 'critical',
                'action': 'immediate_halt'
            },
            'rate_limiting': {
                'patterns': [
                    '429',
                    'rate limit',
                    'too many requests',
                    'throttled'
                ],
                'severity': 'warning',
                'action': 'exponential_backoff'
            }
        }
        
        self.error_history = deque(maxlen=1000)
        self.cascade_threshold = 5  # Errors in 60 seconds
        
    def analyze_error(self, error: Exception) -> tuple[str, str]:
        """Analyze error and return category with recommended action."""
        error_msg = str(error).lower()
        timestamp = time.time()
        
        # Pattern matching
        for category, config in self.error_patterns.items():
            if any(pattern.lower() in error_msg for pattern in config['patterns']):
                self.error_history.append({
                    'timestamp': timestamp,
                    'category': category,
                    'severity': config['severity'],
                    'message': str(error)
                })
                
                # Check for cascade pattern
                if self._detect_cascade_pattern(category):
                    return category, 'emergency_halt'
                    
                return category, config['action']
                
        # Unknown error
        self.error_history.append({
            'timestamp': timestamp,
            'category': 'unknown',
            'severity': 'info',
            'message': str(error)
        })
        
        return 'unknown', 'continue'
        
    def _detect_cascade_pattern(self, category: str) -> bool:
        """Detect if errors are occurring in cascade pattern."""
        recent_errors = [
            e for e in self.error_history 
            if e['timestamp'] > time.time() - 60 and e['category'] == category
        ]
        
        return len(recent_errors) >= self.cascade_threshold
        
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors for monitoring."""
        recent_errors = [
            e for e in self.error_history 
            if e['timestamp'] > time.time() - 300  # Last 5 minutes
        ]
        
        summary = {}
        for error in recent_errors:
            category = error['category']
            if category not in summary:
                summary[category] = {'count': 0, 'latest': None}
            summary[category]['count'] += 1
            summary[category]['latest'] = error['timestamp']
            
        return summary


class ResourceMonitor:
    """
    Real-time system resource monitoring for proactive management.
    
    Monitors memory, processes, network, and browser health.
    """
    
    def __init__(self):
        self.memory_threshold_mb = 1000
        self.process_threshold = 10
        self.network_timeout = 5
        self.browser_memory_limit_mb = 500
        
    def check_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health check."""
        health_status = {
            'memory': self._check_memory_health(),
            'processes': self._check_process_health(),
            'network': self._check_network_health(),
            'timestamp': time.time()
        }
        
        health_status['overall'] = all([
            health_status['memory']['status'] in ['healthy', 'warning'],
            health_status['processes']['status'] in ['healthy', 'warning'],
            health_status['network']['status'] in ['healthy', 'warning']
        ])
        
        return health_status
        
    def _check_memory_health(self) -> Dict[str, Any]:
        """Check system memory availability."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)
            
            if available_mb < self.memory_threshold_mb:
                return {
                    'status': 'critical',
                    'available_mb': available_mb,
                    'threshold_mb': self.memory_threshold_mb,
                    'message': f'Low memory: {available_mb:.1f}MB available'
                }
            elif available_mb < self.memory_threshold_mb * 1.5:
                return {
                    'status': 'warning',
                    'available_mb': available_mb,
                    'threshold_mb': self.memory_threshold_mb,
                    'message': f'Memory pressure: {available_mb:.1f}MB available'
                }
            else:
                return {
                    'status': 'healthy',
                    'available_mb': available_mb,
                    'threshold_mb': self.memory_threshold_mb,
                    'message': f'Memory OK: {available_mb:.1f}MB available'
                }
                
        except Exception as e:
            return {
                'status': 'unknown',
                'error': str(e),
                'message': 'Unable to check memory status'
            }

    def _check_process_health(self) -> Dict[str, Any]:
        """Check for zombie browser processes."""
        try:
            import psutil
            browser_processes = []

            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                try:
                    if any(browser in proc.info['name'].lower()
                          for browser in ['chrome', 'firefox', 'edge', 'safari']):
                        browser_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'memory_mb': proc.info['memory_info'].rss / (1024 * 1024)
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            if len(browser_processes) > self.process_threshold:
                return {
                    'status': 'critical',
                    'process_count': len(browser_processes),
                    'threshold': self.process_threshold,
                    'processes': browser_processes,
                    'message': f'Too many browser processes: {len(browser_processes)}'
                }
            else:
                return {
                    'status': 'healthy',
                    'process_count': len(browser_processes),
                    'threshold': self.process_threshold,
                    'processes': browser_processes,
                    'message': f'Process count OK: {len(browser_processes)}'
                }

        except Exception as e:
            return {
                'status': 'unknown',
                'error': str(e),
                'message': 'Unable to check process status'
            }

    def _check_network_health(self) -> Dict[str, Any]:
        """Check network connectivity."""
        try:
            import requests
            from config_schema import config_schema

            # Test connectivity to Ancestry.com
            test_url = config_schema.api.base_url
            response = requests.get(test_url, timeout=self.network_timeout)

            if response.status_code == 200:
                return {
                    'status': 'healthy',
                    'response_time': response.elapsed.total_seconds(),
                    'status_code': response.status_code,
                    'message': f'Network OK: {response.elapsed.total_seconds():.2f}s response'
                }
            else:
                return {
                    'status': 'warning',
                    'response_time': response.elapsed.total_seconds(),
                    'status_code': response.status_code,
                    'message': f'Network warning: HTTP {response.status_code}'
                }

        except Exception as e:
            return {
                'status': 'critical',
                'error': str(e),
                'message': f'Network failure: {e}'
            }

    def ready_for_restart(self) -> bool:
        """Check if system is ready for browser restart."""
        health = self.check_system_health()

        # Require healthy memory and process status
        memory_ready = health['memory']['status'] in ['healthy', 'warning']
        process_ready = health['processes']['status'] == 'healthy'
        network_ready = health['network']['status'] in ['healthy', 'warning']

        return memory_ready and process_ready and network_ready

    def memory_pressure_detected(self) -> bool:
        """Quick check for memory pressure."""
        memory_health = self._check_memory_health()
        return memory_health['status'] in ['critical', 'warning']


class ReliableSessionManager:
    """
    Simplified session manager focused on reliability over complexity.

    Key Principles:
    - Single browser instance per session
    - Proactive restart strategy
    - Immediate error detection and halt
    - Resource-aware operations
    """

    def __init__(self):
        self.browser_manager = None
        self.session_state = SessionState()
        self.restart_interval = 50  # Pages between restarts
        self.error_detector = CriticalErrorDetector()
        self.resource_monitor = ResourceMonitor()
        self.max_session_hours = 2  # Maximum session duration

        logger.info("🚀 ReliableSessionManager initialized with simplified architecture")

    def process_pages(self, start_page: int, end_page: int) -> bool:
        """
        Main processing loop with built-in reliability checks.

        Returns True if all pages processed successfully, False otherwise.
        """
        logger.info(f"📊 Starting page processing: {start_page} to {end_page}")

        try:
            for page_num in range(start_page, end_page + 1):
                # Pre-processing checks
                if self._should_restart_browser():
                    logger.info(f"🔄 Browser restart needed at page {page_num}")
                    self._safe_browser_restart()

                if not self._system_health_check():
                    raise SystemHealthError("System not ready for processing")

                # Process page with error detection
                try:
                    result = self._process_single_page(page_num)
                    self.session_state.update_progress(page_num, success=True)
                    logger.debug(f"✅ Page {page_num} processed successfully")

                except Exception as e:
                    logger.error(f"❌ Error processing page {page_num}: {e}")

                    # Analyze error for criticality
                    error_category, action = self.error_detector.analyze_error(e)

                    if action in ['immediate_halt', 'emergency_halt']:
                        logger.critical(f"🚨 Critical error detected: {error_category}")
                        self.session_state.update_progress(page_num, success=False)
                        raise CriticalError(f"Critical error on page {page_num}: {error_category}")
                    else:
                        # Handle non-critical errors with retry
                        logger.warning(f"⚠️ Recoverable error: {error_category}, action: {action}")
                        retry_success = self._handle_recoverable_error(e, page_num, action)
                        self.session_state.update_progress(page_num, success=retry_success)

                        if not retry_success:
                            logger.error(f"❌ Failed to recover from error on page {page_num}")
                            return False

            logger.info(f"🎉 Successfully processed pages {start_page} to {end_page}")
            return True

        except (CriticalError, SystemHealthError) as e:
            logger.critical(f"🚨 Processing halted due to critical error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error in page processing: {e}")
            return False

    def _should_restart_browser(self) -> bool:
        """Determine if browser restart is needed."""
        if not self.browser_manager:
            return True

        # Check restart interval
        if self.session_state.pages_processed >= self.restart_interval:
            logger.debug(f"📊 Restart needed: {self.session_state.pages_processed} pages processed")
            return True

        # Check memory pressure
        if self.resource_monitor.memory_pressure_detected():
            logger.debug("📊 Restart needed: Memory pressure detected")
            return True

        # Check session age
        session_age_hours = (time.time() - self.session_state.session_start_time) / 3600
        if session_age_hours > self.max_session_hours:
            logger.debug(f"📊 Restart needed: Session age {session_age_hours:.1f} hours")
            return True

        # Check browser health
        if not self._quick_browser_health_check():
            logger.debug("📊 Restart needed: Browser health check failed")
            return True

        return False

    def _safe_browser_restart(self) -> None:
        """Restart browser with full validation and rollback capability."""
        logger.info(f"🔄 Initiating safe browser restart at page {self.session_state.current_page}")

        # Capture current state for rollback
        state_backup = self.session_state.create_backup()

        try:
            # Verify system readiness
            if not self.resource_monitor.ready_for_restart():
                raise ResourceNotReadyError("System not ready for browser restart")

            # Close old browser cleanly
            if self.browser_manager:
                logger.debug("🔄 Closing old browser instance")
                self.browser_manager.close_browser()
                self.browser_manager = None

            # Create new browser with validation
            logger.debug("🔄 Creating new browser instance")
            from core.browser_manager import BrowserManager
            new_browser = BrowserManager()

            if not new_browser.start_browser("ReliableSessionManager"):
                raise BrowserStartupError("Failed to start new browser")

            # Verify new browser functionality
            if not self._validate_browser_functionality(new_browser):
                new_browser.close_browser()
                raise BrowserValidationError("New browser failed validation")

            # Atomic assignment
            self.browser_manager = new_browser
            self.session_state.pages_processed = 0  # Reset counter
            self.session_state.restart_count += 1

            logger.info("✅ Browser restart completed successfully")

        except Exception as e:
            logger.error(f"❌ Browser restart failed: {e}")
            # Attempt to restore previous state
            self.session_state.restore_backup(state_backup)
            raise BrowserRestartError(f"Failed to restart browser: {e}")

    def _validate_browser_functionality(self, browser_manager) -> bool:
        """Comprehensive validation of browser functionality."""
        try:
            # Test 1: Basic session validity
            if not browser_manager.is_session_valid():
                logger.warning("❌ Browser session invalid")
                return False

            # Test 2: Navigation capability
            from utils import nav_to_page
            from config_schema import config_schema
            base_url = config_schema.api.base_url

            if base_url:
                nav_success = nav_to_page(browser_manager.driver, base_url)
                if not nav_success:
                    logger.warning("❌ Browser failed navigation test")
                    return False

            # Test 3: Cookie access (the originally failing operation)
            cookies = browser_manager.driver.get_cookies()
            if not isinstance(cookies, list):
                logger.warning("❌ Browser cookie access failed")
                return False

            # Test 4: JavaScript execution
            js_result = browser_manager.driver.execute_script("return document.readyState;")
            if js_result != "complete":
                logger.warning(f"❌ JavaScript execution test failed: {js_result}")
                return False

            # Test 5: Authentication state verification
            current_url = browser_manager.driver.current_url
            if current_url and "login" in current_url.lower():
                logger.warning("❌ Browser appears to be on login page - authentication lost")
                return False

            logger.debug("✅ Browser functionality validation passed")
            return True

        except Exception as e:
            logger.error(f"❌ Browser validation failed: {e}")
            return False

    def _quick_browser_health_check(self) -> bool:
        """Quick health check for current browser."""
        try:
            if not self.browser_manager:
                return False

            if not self.browser_manager.is_session_valid():
                return False

            # Quick cookie test
            cookies = self.browser_manager.driver.get_cookies()
            return isinstance(cookies, list)

        except Exception:
            return False

    def _system_health_check(self) -> bool:
        """Check if system is healthy enough for processing."""
        health = self.resource_monitor.check_system_health()

        if not health['overall']:
            logger.warning(f"⚠️ System health check failed: {health}")
            return False

        return True

    def _process_single_page(self, page_num: int) -> Dict[str, Any]:
        """
        Process a single page with the current browser.

        This is a placeholder that should be replaced with actual page processing logic.
        """
        if not self.browser_manager:
            raise RuntimeError("No browser manager available")

        # Placeholder for actual page processing
        # This would be replaced with the actual Action 6 page processing logic
        logger.debug(f"🔄 Processing page {page_num}")

        # Simulate some browser operations
        current_url = self.browser_manager.driver.current_url
        cookies = self.browser_manager.driver.get_cookies()

        return {
            'page_num': page_num,
            'url': current_url,
            'cookie_count': len(cookies),
            'timestamp': time.time()
        }

    def _handle_recoverable_error(self, error: Exception, page_num: int, action: str) -> bool:
        """Handle recoverable errors with appropriate retry strategies."""
        logger.info(f"🔄 Handling recoverable error on page {page_num}: {action}")

        if action == 'retry_with_backoff':
            return self._retry_with_backoff(page_num, max_attempts=3)
        elif action == 'exponential_backoff':
            return self._retry_with_exponential_backoff(page_num, max_attempts=3)
        elif action == 'immediate_restart':
            try:
                self._safe_browser_restart()
                return True
            except Exception as e:
                logger.error(f"❌ Failed to restart browser for recovery: {e}")
                return False
        else:
            logger.warning(f"⚠️ Unknown recovery action: {action}")
            return False

    def _retry_with_backoff(self, page_num: int, max_attempts: int = 3) -> bool:
        """Retry page processing with linear backoff."""
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"🔄 Retry attempt {attempt}/{max_attempts} for page {page_num}")
                time.sleep(attempt * 2)  # Linear backoff: 2s, 4s, 6s

                result = self._process_single_page(page_num)
                logger.info(f"✅ Retry successful on attempt {attempt}")
                return True

            except Exception as e:
                logger.warning(f"⚠️ Retry attempt {attempt} failed: {e}")
                if attempt == max_attempts:
                    logger.error(f"❌ All retry attempts failed for page {page_num}")

        return False

    def _retry_with_exponential_backoff(self, page_num: int, max_attempts: int = 3) -> bool:
        """Retry page processing with exponential backoff."""
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"🔄 Exponential retry attempt {attempt}/{max_attempts} for page {page_num}")
                time.sleep(2 ** attempt)  # Exponential backoff: 2s, 4s, 8s

                result = self._process_single_page(page_num)
                logger.info(f"✅ Exponential retry successful on attempt {attempt}")
                return True

            except Exception as e:
                logger.warning(f"⚠️ Exponential retry attempt {attempt} failed: {e}")
                if attempt == max_attempts:
                    logger.error(f"❌ All exponential retry attempts failed for page {page_num}")

        return False

    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary for monitoring."""
        return {
            'session_state': {
                'current_page': self.session_state.current_page,
                'pages_processed': self.session_state.pages_processed,
                'last_successful_page': self.session_state.last_successful_page,
                'error_count': self.session_state.error_count,
                'restart_count': self.session_state.restart_count,
                'session_duration_hours': (time.time() - self.session_state.session_start_time) / 3600
            },
            'system_health': self.resource_monitor.check_system_health(),
            'error_summary': self.error_detector.get_error_summary(),
            'browser_status': {
                'available': self.browser_manager is not None,
                'valid': self._quick_browser_health_check() if self.browser_manager else False
            }
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("🧹 Cleaning up ReliableSessionManager resources")

        if self.browser_manager:
            try:
                self.browser_manager.close_browser()
            except Exception as e:
                logger.warning(f"⚠️ Error closing browser during cleanup: {e}")
            finally:
                self.browser_manager = None
