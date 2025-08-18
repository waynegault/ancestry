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
    Enhanced error detection system for early intervention.

    Phase 2: Expanded patterns, early warning system, and sophisticated intervention.
    """

    def __init__(self):
        self.error_patterns = {
            'webdriver_death': {
                'patterns': [
                    'WebDriver became None',
                    'invalid session id',
                    'session deleted',
                    'chrome not reachable',
                    'browser process died',
                    'session not created',
                    'chrome crashed',
                    'browser disconnected'
                ],
                'severity': 'critical',
                'action': 'immediate_halt'
            },
            'memory_pressure': {
                'patterns': [
                    'OutOfMemoryError',
                    'MemoryError',
                    'cannot allocate memory',
                    'virtual memory exhausted',
                    'memory allocation failed',
                    'insufficient memory',
                    'heap space'
                ],
                'severity': 'critical',
                'action': 'immediate_restart'
            },
            'network_failure': {
                'patterns': [
                    'ConnectionError',
                    'TimeoutError',
                    'DNS resolution failed',
                    'network unreachable',
                    'connection refused',
                    'connection reset',
                    'socket timeout',
                    'name resolution failed',
                    'no route to host'
                ],
                'severity': 'warning',
                'action': 'network_resilience_retry'
            },
            'auth_loss': {
                'patterns': [
                    'login',
                    'signin',
                    'authenticate',
                    'unauthorized',
                    '401',
                    '403',
                    'session expired',
                    'access denied',
                    'authentication required',
                    'please sign in'
                ],
                'severity': 'critical',
                'action': 'auth_recovery'
            },
            'rate_limiting': {
                'patterns': [
                    '429',
                    'rate limit',
                    'too many requests',
                    'throttled',
                    'quota exceeded',
                    'api limit',
                    'request limit',
                    'slow down'
                ],
                'severity': 'warning',
                'action': 'adaptive_backoff'
            },
            'ancestry_specific': {
                'patterns': [
                    'ancestry.com error',
                    'service unavailable',
                    'maintenance mode',
                    'temporarily unavailable',
                    'server error',
                    'internal server error',
                    '500',
                    '502',
                    '503',
                    '504'
                ],
                'severity': 'warning',
                'action': 'ancestry_service_retry'
            },
            'selenium_specific': {
                'patterns': [
                    'element not found',
                    'stale element',
                    'element not clickable',
                    'element not visible',
                    'no such element',
                    'timeout waiting for',
                    'element click intercepted'
                ],
                'severity': 'warning',
                'action': 'selenium_recovery'
            },
            'javascript_errors': {
                'patterns': [
                    'javascript error',
                    'script timeout',
                    'execution timeout',
                    'script error',
                    'js error',
                    'uncaught exception'
                ],
                'severity': 'warning',
                'action': 'page_refresh'
            }
        }

        self.error_history = deque(maxlen=1000)
        self.cascade_threshold = 5  # Errors in 60 seconds

        # Phase 2: Enhanced monitoring
        self.early_warning_thresholds = {
            'error_rate_1min': 3,      # 3 errors in 1 minute
            'error_rate_5min': 10,     # 10 errors in 5 minutes
            'error_rate_15min': 25,    # 25 errors in 15 minutes
            'critical_errors_1min': 1, # 1 critical error in 1 minute
            'network_errors_5min': 5   # 5 network errors in 5 minutes
        }

        self.intervention_history = deque(maxlen=100)
        self.last_early_warning = 0
        
    def analyze_error(self, error: Exception) -> tuple[str, str]:
        """
        Enhanced error analysis with early warning detection.

        Phase 2: Includes pattern matching, cascade detection, and early warning system.
        """
        error_msg = str(error).lower()
        timestamp = time.time()

        # Pattern matching
        for category, config in self.error_patterns.items():
            if any(pattern.lower() in error_msg for pattern in config['patterns']):
                error_record = {
                    'timestamp': timestamp,
                    'category': category,
                    'severity': config['severity'],
                    'message': str(error),
                    'action': config['action']
                }
                self.error_history.append(error_record)

                # Check for cascade pattern
                if self._detect_cascade_pattern(category):
                    return category, 'emergency_halt'

                # Check for early warning conditions
                early_warning = self._check_early_warning_conditions(timestamp)
                if early_warning:
                    return category, early_warning

                return category, config['action']

        # Unknown error
        self.error_history.append({
            'timestamp': timestamp,
            'category': 'unknown',
            'severity': 'info',
            'message': str(error),
            'action': 'continue'
        })

        return 'unknown', 'continue'
        
    def _detect_cascade_pattern(self, category: str) -> bool:
        """Detect if errors are occurring in cascade pattern."""
        recent_errors = [
            e for e in self.error_history 
            if e['timestamp'] > time.time() - 60 and e['category'] == category
        ]
        
        return len(recent_errors) >= self.cascade_threshold

    def _check_early_warning_conditions(self, current_time: float) -> Optional[str]:
        """
        Check for early warning conditions that require intervention.

        Phase 2: Proactive detection before cascade failures occur.
        """
        # Avoid spam - only check every 30 seconds
        if current_time - self.last_early_warning < 30:
            return None

        # Check error rate thresholds
        time_windows = [
            (60, self.early_warning_thresholds['error_rate_1min'], 'enhanced_monitoring'),
            (300, self.early_warning_thresholds['error_rate_5min'], 'proactive_intervention'),
            (900, self.early_warning_thresholds['error_rate_15min'], 'immediate_intervention')
        ]

        for window_seconds, threshold, action in time_windows:
            window_start = current_time - window_seconds
            errors_in_window = [
                e for e in self.error_history
                if e['timestamp'] >= window_start
            ]

            if len(errors_in_window) >= threshold:
                self.last_early_warning = current_time
                self._record_intervention(action, f"{len(errors_in_window)} errors in {window_seconds}s")
                return action

        # Check critical error patterns
        critical_window = current_time - 60  # 1 minute
        critical_errors = [
            e for e in self.error_history
            if e['timestamp'] >= critical_window and e['severity'] == 'critical'
        ]

        if len(critical_errors) >= self.early_warning_thresholds['critical_errors_1min']:
            self.last_early_warning = current_time
            self._record_intervention('immediate_halt', f"{len(critical_errors)} critical errors in 1min")
            return 'immediate_halt'

        # Check network error patterns
        network_window = current_time - 300  # 5 minutes
        network_errors = [
            e for e in self.error_history
            if e['timestamp'] >= network_window and e['category'] == 'network_failure'
        ]

        if len(network_errors) >= self.early_warning_thresholds['network_errors_5min']:
            self.last_early_warning = current_time
            self._record_intervention('network_recovery', f"{len(network_errors)} network errors in 5min")
            return 'network_recovery'

        return None

    def _record_intervention(self, intervention_type: str, reason: str) -> None:
        """Record intervention for monitoring and analysis."""
        self.intervention_history.append({
            'timestamp': time.time(),
            'type': intervention_type,
            'reason': reason
        })

    def get_early_warning_status(self) -> Dict[str, Any]:
        """Get current early warning system status."""
        current_time = time.time()

        # Calculate current error rates
        error_rates = {}
        for window_name, window_seconds in [('1min', 60), ('5min', 300), ('15min', 900)]:
            window_start = current_time - window_seconds
            errors_in_window = [
                e for e in self.error_history
                if e['timestamp'] >= window_start
            ]
            error_rates[window_name] = len(errors_in_window)

        # Get recent interventions
        recent_interventions = [
            i for i in self.intervention_history
            if i['timestamp'] > current_time - 3600  # Last hour
        ]

        return {
            'error_rates': error_rates,
            'thresholds': self.early_warning_thresholds,
            'recent_interventions': recent_interventions,
            'last_warning': self.last_early_warning,
            'status': 'active' if error_rates['1min'] > 0 else 'monitoring'
        }
        
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
    Enhanced real-time system resource monitoring for proactive management.

    Phase 2: Monitors memory, processes, network resilience, and authentication state.
    """

    def __init__(self):
        self.memory_threshold_mb = 1000
        self.process_threshold = 10
        self.network_timeout = 5
        self.browser_memory_limit_mb = 500

        # Phase 2: Enhanced monitoring
        self.network_retry_attempts = 3
        self.network_backoff_factor = 2.0
        self.auth_check_interval = 300  # 5 minutes
        self.last_auth_check = 0
        self.network_failure_count = 0
        self.max_network_failures = 5
        
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
        """
        Enhanced network connectivity check with resilience.

        Phase 2: Includes retry logic, multiple endpoints, and failure tracking.
        """
        # Test multiple endpoints for better reliability
        test_endpoints = [
            'https://www.ancestry.com',
            'https://www.google.com',
            'https://www.cloudflare.com'
        ]

        best_result = None
        all_failed = True

        for endpoint in test_endpoints:
            result = self._test_single_endpoint(endpoint)

            if result['status'] in ['healthy', 'warning']:
                all_failed = False
                if best_result is None or result['status'] == 'healthy':
                    best_result = result

        if all_failed:
            self.network_failure_count += 1
            return {
                'status': 'critical',
                'failure_count': self.network_failure_count,
                'max_failures': self.max_network_failures,
                'message': f'All network endpoints failed (failure #{self.network_failure_count})'
            }
        else:
            # Reset failure count on success
            self.network_failure_count = 0
            return best_result

    def _test_single_endpoint(self, url: str) -> Dict[str, Any]:
        """Test connectivity to a single endpoint with retry logic."""
        for attempt in range(self.network_retry_attempts):
            try:
                import requests

                timeout = self.network_timeout * (attempt + 1)  # Progressive timeout
                response = requests.get(url, timeout=timeout)

                if response.status_code == 200:
                    return {
                        'status': 'healthy',
                        'endpoint': url,
                        'response_time': response.elapsed.total_seconds(),
                        'status_code': response.status_code,
                        'attempt': attempt + 1,
                        'message': f'Network OK: {response.elapsed.total_seconds():.2f}s response'
                    }
                elif response.status_code in [429, 503, 504]:
                    # Temporary issues - continue retrying
                    if attempt < self.network_retry_attempts - 1:
                        time.sleep(self.network_backoff_factor ** attempt)
                        continue
                    else:
                        return {
                            'status': 'warning',
                            'endpoint': url,
                            'response_time': response.elapsed.total_seconds(),
                            'status_code': response.status_code,
                            'attempt': attempt + 1,
                            'message': f'Network warning: HTTP {response.status_code} after {attempt + 1} attempts'
                        }
                else:
                    return {
                        'status': 'warning',
                        'endpoint': url,
                        'response_time': response.elapsed.total_seconds(),
                        'status_code': response.status_code,
                        'attempt': attempt + 1,
                        'message': f'Network warning: HTTP {response.status_code}'
                    }

            except Exception as e:
                if attempt < self.network_retry_attempts - 1:
                    time.sleep(self.network_backoff_factor ** attempt)
                    continue
                else:
                    return {
                        'status': 'critical',
                        'endpoint': url,
                        'error': str(e),
                        'attempt': attempt + 1,
                        'message': f'Network failure: {e} after {attempt + 1} attempts'
                    }

        return {
            'status': 'critical',
            'endpoint': url,
            'message': 'All retry attempts failed'
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
        """
        Enhanced error handling with sophisticated recovery strategies.

        Phase 2: Includes network resilience, auth recovery, and adaptive strategies.
        """
        logger.info(f"🔄 Handling recoverable error on page {page_num}: {action}")

        if action == 'retry_with_backoff':
            return self._retry_with_backoff(page_num, max_attempts=3)
        elif action == 'exponential_backoff':
            return self._retry_with_exponential_backoff(page_num, max_attempts=3)
        elif action == 'adaptive_backoff':
            return self._adaptive_backoff_retry(page_num, max_attempts=5)
        elif action == 'network_resilience_retry':
            return self._network_resilience_retry(page_num, max_attempts=3)
        elif action == 'ancestry_service_retry':
            return self._ancestry_service_retry(page_num, max_attempts=3)
        elif action == 'selenium_recovery':
            return self._selenium_recovery(page_num, max_attempts=2)
        elif action == 'page_refresh':
            return self._page_refresh_recovery(page_num)
        elif action == 'auth_recovery':
            return self._authentication_recovery(page_num)
        elif action == 'network_recovery':
            return self._network_recovery(page_num)
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

    def _adaptive_backoff_retry(self, page_num: int, max_attempts: int = 5) -> bool:
        """
        Adaptive backoff retry with intelligent delay calculation.

        Phase 2: Adjusts delay based on error history and system load.
        """
        for attempt in range(1, max_attempts + 1):
            try:
                # Calculate adaptive delay based on recent error rate
                recent_errors = len([
                    e for e in self.error_detector.error_history
                    if e['timestamp'] > time.time() - 300  # Last 5 minutes
                ])

                # Adaptive delay: more errors = longer delay
                base_delay = 2 ** attempt
                error_multiplier = 1 + (recent_errors * 0.5)
                adaptive_delay = min(base_delay * error_multiplier, 60)  # Cap at 60 seconds

                logger.info(f"🔄 Adaptive retry attempt {attempt}/{max_attempts} for page {page_num} (delay: {adaptive_delay:.1f}s)")
                time.sleep(adaptive_delay)

                result = self._process_single_page(page_num)
                logger.info(f"✅ Adaptive retry successful on attempt {attempt}")
                return True

            except Exception as e:
                logger.warning(f"⚠️ Adaptive retry attempt {attempt} failed: {e}")
                if attempt == max_attempts:
                    logger.error(f"❌ All adaptive retry attempts failed for page {page_num}")

        return False

    def _network_resilience_retry(self, page_num: int, max_attempts: int = 3) -> bool:
        """
        Network-aware retry with connectivity validation.

        Phase 2: Validates network health before retry attempts.
        """
        for attempt in range(1, max_attempts + 1):
            try:
                # Check network health before retry
                network_health = self.resource_monitor._check_network_health()
                if network_health['status'] == 'critical':
                    logger.warning(f"⚠️ Network critical, waiting before retry attempt {attempt}")
                    time.sleep(10 * attempt)  # Wait longer for network recovery

                logger.info(f"🌐 Network resilience retry attempt {attempt}/{max_attempts} for page {page_num}")
                time.sleep(3 * attempt)  # Progressive delay

                result = self._process_single_page(page_num)
                logger.info(f"✅ Network resilience retry successful on attempt {attempt}")
                return True

            except Exception as e:
                logger.warning(f"⚠️ Network resilience retry attempt {attempt} failed: {e}")
                if attempt == max_attempts:
                    logger.error(f"❌ All network resilience retry attempts failed for page {page_num}")

        return False

    def _ancestry_service_retry(self, page_num: int, max_attempts: int = 3) -> bool:
        """
        Ancestry.com service-specific retry with service status awareness.

        Phase 2: Handles Ancestry.com specific service issues.
        """
        for attempt in range(1, max_attempts + 1):
            try:
                # Longer delays for service issues
                service_delay = 30 * attempt  # 30s, 60s, 90s
                logger.info(f"🏛️ Ancestry service retry attempt {attempt}/{max_attempts} for page {page_num} (delay: {service_delay}s)")
                time.sleep(service_delay)

                # Check if we can reach Ancestry.com
                network_health = self.resource_monitor._test_single_endpoint('https://www.ancestry.com')
                if network_health['status'] == 'critical':
                    logger.warning(f"⚠️ Ancestry.com still unreachable on attempt {attempt}")
                    continue

                result = self._process_single_page(page_num)
                logger.info(f"✅ Ancestry service retry successful on attempt {attempt}")
                return True

            except Exception as e:
                logger.warning(f"⚠️ Ancestry service retry attempt {attempt} failed: {e}")
                if attempt == max_attempts:
                    logger.error(f"❌ All Ancestry service retry attempts failed for page {page_num}")

        return False

    def _selenium_recovery(self, page_num: int, max_attempts: int = 2) -> bool:
        """
        Selenium-specific recovery for element and interaction issues.

        Phase 2: Handles stale elements, timeouts, and interaction failures.
        """
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"🔧 Selenium recovery attempt {attempt}/{max_attempts} for page {page_num}")

                # Refresh page to clear stale elements
                if self.browser_manager and self.browser_manager.driver:
                    self.browser_manager.driver.refresh()
                    time.sleep(5)  # Wait for page load

                    # Verify page is ready
                    ready_state = self.browser_manager.driver.execute_script("return document.readyState;")
                    if ready_state != "complete":
                        logger.warning(f"⚠️ Page not ready after refresh: {ready_state}")
                        time.sleep(5)

                result = self._process_single_page(page_num)
                logger.info(f"✅ Selenium recovery successful on attempt {attempt}")
                return True

            except Exception as e:
                logger.warning(f"⚠️ Selenium recovery attempt {attempt} failed: {e}")
                if attempt == max_attempts:
                    logger.error(f"❌ All Selenium recovery attempts failed for page {page_num}")

        return False

    def _page_refresh_recovery(self, page_num: int) -> bool:
        """
        Simple page refresh recovery for JavaScript errors.

        Phase 2: Handles JavaScript execution issues.
        """
        try:
            logger.info(f"🔄 Page refresh recovery for page {page_num}")

            if self.browser_manager and self.browser_manager.driver:
                # Clear any JavaScript errors
                self.browser_manager.driver.execute_script("console.clear();")

                # Refresh page
                self.browser_manager.driver.refresh()
                time.sleep(3)

                # Verify JavaScript is working
                js_test = self.browser_manager.driver.execute_script("return typeof jQuery !== 'undefined';")
                logger.debug(f"JavaScript test result: {js_test}")

            result = self._process_single_page(page_num)
            logger.info(f"✅ Page refresh recovery successful")
            return True

        except Exception as e:
            logger.error(f"❌ Page refresh recovery failed: {e}")
            return False

    def _authentication_recovery(self, page_num: int) -> bool:
        """
        Authentication recovery for session expiration.

        Phase 2: Handles authentication loss and session expiration.
        """
        try:
            logger.info(f"🔐 Authentication recovery for page {page_num}")

            # Check current URL for login indicators
            if self.browser_manager and self.browser_manager.driver:
                current_url = self.browser_manager.driver.current_url
                if any(indicator in current_url.lower() for indicator in ['login', 'signin', 'authenticate']):
                    logger.critical(f"🚨 Authentication lost - on login page: {current_url}")
                    return False  # Cannot recover automatically

            # Try to access a protected resource to verify auth
            try:
                cookies = self.browser_manager.driver.get_cookies()
                auth_cookies = [c for c in cookies if 'auth' in c['name'].lower() or 'session' in c['name'].lower()]

                if not auth_cookies:
                    logger.warning(f"⚠️ No authentication cookies found")
                    return False

                logger.info(f"✅ Authentication appears valid ({len(auth_cookies)} auth cookies)")

            except Exception as cookie_error:
                logger.error(f"❌ Cannot access cookies for auth check: {cookie_error}")
                return False

            # Try processing the page
            result = self._process_single_page(page_num)
            logger.info(f"✅ Authentication recovery successful")
            return True

        except Exception as e:
            logger.error(f"❌ Authentication recovery failed: {e}")
            return False

    def _network_recovery(self, page_num: int) -> bool:
        """
        Network recovery for persistent connectivity issues.

        Phase 2: Handles network instability and connectivity problems.
        """
        try:
            logger.info(f"🌐 Network recovery for page {page_num}")

            # Wait for network to stabilize
            max_wait_time = 60  # Maximum 60 seconds
            wait_interval = 10   # Check every 10 seconds

            for wait_time in range(0, max_wait_time, wait_interval):
                network_health = self.resource_monitor._check_network_health()

                if network_health['status'] in ['healthy', 'warning']:
                    logger.info(f"✅ Network recovered after {wait_time}s")
                    break

                logger.info(f"⏳ Waiting for network recovery... ({wait_time}s/{max_wait_time}s)")
                time.sleep(wait_interval)
            else:
                logger.error(f"❌ Network did not recover within {max_wait_time}s")
                return False

            # Try processing the page
            result = self._process_single_page(page_num)
            logger.info(f"✅ Network recovery successful")
            return True

        except Exception as e:
            logger.error(f"❌ Network recovery failed: {e}")
            return False

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Enhanced comprehensive session summary for monitoring.

        Phase 2: Includes early warning status, intervention history, and network resilience data.
        """
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
            'early_warning': self.error_detector.get_early_warning_status(),
            'network_resilience': {
                'failure_count': self.resource_monitor.network_failure_count,
                'max_failures': self.resource_monitor.max_network_failures,
                'retry_attempts': self.resource_monitor.network_retry_attempts
            },
            'browser_status': {
                'available': self.browser_manager is not None,
                'valid': self._quick_browser_health_check() if self.browser_manager else False
            },
            'phase2_features': {
                'enhanced_error_patterns': len(self.error_detector.error_patterns),
                'intervention_history_count': len(self.error_detector.intervention_history),
                'network_endpoints_tested': 3,
                'recovery_strategies_available': 8
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


# ============================================================================
# EMBEDDED TESTS - Following user preference for tests in same file
# ============================================================================

def test_session_state_management():
    """Test SessionState backup and restore functionality."""
    print("🧪 Testing SessionState management...")

    # Create session state
    state = SessionState()
    state.current_page = 10
    state.pages_processed = 5
    state.error_count = 2

    # Create backup
    backup = state.create_backup()

    # Modify state
    state.current_page = 20
    state.pages_processed = 15
    state.error_count = 5

    # Restore from backup
    state.restore_backup(backup)

    # Verify restoration
    assert state.current_page == 10, f"Expected current_page=10, got {state.current_page}"
    assert state.pages_processed == 5, f"Expected pages_processed=5, got {state.pages_processed}"
    assert state.error_count == 2, f"Expected error_count=2, got {state.error_count}"

    print("   ✅ SessionState backup/restore working correctly")
    return True


def test_critical_error_detection():
    """Test CriticalErrorDetector pattern matching and cascade detection."""
    print("🧪 Testing CriticalErrorDetector...")

    detector = CriticalErrorDetector()

    # Test webdriver death detection
    webdriver_error = Exception("WebDriver became None during operation")
    category, action = detector.analyze_error(webdriver_error)
    assert category == 'webdriver_death', f"Expected webdriver_death, got {category}"
    assert action == 'immediate_halt', f"Expected immediate_halt, got {action}"

    # Test memory pressure detection
    memory_error = Exception("OutOfMemoryError: cannot allocate memory")
    category, action = detector.analyze_error(memory_error)
    assert category == 'memory_pressure', f"Expected memory_pressure, got {category}"
    assert action == 'immediate_restart', f"Expected immediate_restart, got {action}"

    # Test cascade detection
    for i in range(6):  # Trigger cascade threshold
        detector.analyze_error(Exception("WebDriver became None"))

    # Next error should trigger emergency halt
    category, action = detector.analyze_error(Exception("WebDriver became None"))
    assert action == 'emergency_halt', f"Expected emergency_halt for cascade, got {action}"

    print("   ✅ CriticalErrorDetector pattern matching and cascade detection working")
    return True


def test_enhanced_error_patterns():
    """Test Phase 2 enhanced error patterns and detection."""
    print("🧪 Testing Phase 2 enhanced error patterns...")

    # Create fresh detector to avoid early warning interference
    detector = CriticalErrorDetector()

    # Test new ancestry-specific error detection
    ancestry_error = Exception("ancestry.com error: service unavailable")
    category, action = detector.analyze_error(ancestry_error)
    assert category == 'ancestry_specific', f"Expected ancestry_specific, got {category}"
    assert action == 'ancestry_service_retry', f"Expected ancestry_service_retry, got {action}"

    # Create fresh detector for selenium test
    detector = CriticalErrorDetector()
    selenium_error = Exception("element not found: stale element reference")
    category, action = detector.analyze_error(selenium_error)
    assert category == 'selenium_specific', f"Expected selenium_specific, got {category}"
    assert action == 'selenium_recovery', f"Expected selenium_recovery, got {action}"

    # Create fresh detector for javascript test
    detector = CriticalErrorDetector()
    js_error = Exception("javascript error: script timeout")
    category, action = detector.analyze_error(js_error)
    assert category == 'javascript_errors', f"Expected javascript_errors, got {category}"
    assert action == 'page_refresh', f"Expected page_refresh, got {action}"

    print("   ✅ Enhanced error patterns working correctly")
    return True


def test_early_warning_system():
    """Test Phase 2 early warning system."""
    print("🧪 Testing Phase 2 early warning system...")

    detector = CriticalErrorDetector()

    # Test early warning thresholds
    assert 'error_rate_1min' in detector.early_warning_thresholds
    assert 'critical_errors_1min' in detector.early_warning_thresholds
    assert 'network_errors_5min' in detector.early_warning_thresholds

    # Test early warning status
    warning_status = detector.get_early_warning_status()
    assert 'error_rates' in warning_status
    assert 'thresholds' in warning_status
    assert 'recent_interventions' in warning_status
    assert 'status' in warning_status

    # Simulate multiple errors to trigger early warning
    current_time = time.time()
    for i in range(4):  # Trigger 1-minute threshold (3 errors)
        detector.error_history.append({
            'timestamp': current_time - (i * 10),  # Spread over 30 seconds
            'category': 'network_failure',
            'severity': 'warning',
            'message': f'Test error {i}'
        })

    # Check if early warning would trigger
    warning_action = detector._check_early_warning_conditions(current_time)
    assert warning_action is not None, "Early warning should trigger with 4 recent errors"

    print("   ✅ Early warning system working correctly")
    return True


def test_resource_monitor():
    """Test ResourceMonitor system health checks."""
    print("🧪 Testing ResourceMonitor...")

    monitor = ResourceMonitor()

    # Test system health check
    health = monitor.check_system_health()
    assert 'memory' in health, "Health check should include memory status"
    assert 'processes' in health, "Health check should include process status"
    assert 'network' in health, "Health check should include network status"
    assert 'overall' in health, "Health check should include overall status"

    # Test memory pressure detection
    memory_pressure = monitor.memory_pressure_detected()
    assert isinstance(memory_pressure, bool), "Memory pressure should return boolean"

    # Test restart readiness
    ready = monitor.ready_for_restart()
    assert isinstance(ready, bool), "Restart readiness should return boolean"

    print("   ✅ ResourceMonitor health checks working correctly")
    return True


def test_network_resilience():
    """Test Phase 2 network resilience features."""
    print("🧪 Testing Phase 2 network resilience...")

    monitor = ResourceMonitor()

    # Test enhanced network monitoring attributes
    assert hasattr(monitor, 'network_retry_attempts')
    assert hasattr(monitor, 'network_backoff_factor')
    assert hasattr(monitor, 'network_failure_count')
    assert hasattr(monitor, 'max_network_failures')

    # Test single endpoint testing
    result = monitor._test_single_endpoint('https://www.google.com')
    assert 'status' in result
    assert 'endpoint' in result
    assert 'attempt' in result

    print("   ✅ Network resilience features working correctly")
    return True


def test_reliable_session_manager_basic():
    """Test basic ReliableSessionManager functionality."""
    print("🧪 Testing ReliableSessionManager basic functionality...")

    # Test initialization
    session_manager = ReliableSessionManager()
    assert session_manager.session_state is not None, "Session state should be initialized"
    assert session_manager.error_detector is not None, "Error detector should be initialized"
    assert session_manager.resource_monitor is not None, "Resource monitor should be initialized"

    # Test session summary
    summary = session_manager.get_session_summary()
    assert 'session_state' in summary, "Summary should include session state"
    assert 'system_health' in summary, "Summary should include system health"
    assert 'error_summary' in summary, "Summary should include error summary"
    assert 'browser_status' in summary, "Summary should include browser status"

    # Test Phase 2 additions
    assert 'early_warning' in summary, "Summary should include early warning status"
    assert 'network_resilience' in summary, "Summary should include network resilience data"
    assert 'phase2_features' in summary, "Summary should include Phase 2 feature info"

    # Test cleanup
    session_manager.cleanup()

    print("   ✅ ReliableSessionManager basic functionality working")
    return True


def run_embedded_tests():
    """Run all embedded tests for reliable session manager."""
    print("🚀 Running Embedded Tests for Reliable Session Manager...")
    print("=" * 60)

    tests = [
        ("SessionState Management", test_session_state_management),
        ("Critical Error Detection", test_critical_error_detection),
        ("Enhanced Error Patterns", test_enhanced_error_patterns),
        ("Early Warning System", test_early_warning_system),
        ("Resource Monitoring", test_resource_monitor),
        ("Network Resilience", test_network_resilience),
        ("ReliableSessionManager Basic", test_reliable_session_manager_basic),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            result = test_func()
            if result:
                passed += 1
                print(f"✅ PASSED: {test_name}")
            else:
                failed += 1
                print(f"❌ FAILED: {test_name}")
        except Exception as e:
            failed += 1
            print(f"❌ FAILED: {test_name} - {e}")

    print(f"\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All embedded tests passed!")
        return True
    else:
        print(f"❌ {failed} tests failed!")
        return False


if __name__ == "__main__":
    # Run embedded tests when file is executed directly
    success = run_embedded_tests()
    import sys
    sys.exit(0 if success else 1)
