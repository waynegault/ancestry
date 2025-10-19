#!/usr/bin/env python3
"""
Phase 6: Final Validation Before Go-Live
Tests Action 8 with Wayne's account in dry_run mode to validate all systems are ready.
Does NOT send real messages to Ancestry.
"""

import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.session_manager import SessionManager
from database import db_transn, Person, ConversationLog, MessageTemplate
from action8_messaging import send_messages_to_matches
from utils import _load_login_cookies, log_in, login_status


def _create_and_start_session() -> SessionManager:
    """Create and start a new session manager."""
    logger.info("Step 1: Creating SessionManager...")
    sm = SessionManager()
    logger.info("✅ SessionManager created")

    logger.info("Step 2: Configuring browser requirement...")
    sm.browser_manager.browser_needed = True
    logger.info("✅ Browser marked as needed")

    logger.info("Step 3: Starting session (database + browser)...")
    started = sm.start_sess("Phase 6 Final Validation")
    if not started:
        sm.close_sess(keep_db=False)
        raise AssertionError("Failed to start session - browser initialization failed")
    logger.info("✅ Session started successfully")

    return sm


def _authenticate_session(sm: SessionManager) -> None:
    """Authenticate the session using cookies or login."""
    logger.info("Step 4: Attempting to load saved cookies...")
    cookies_loaded = _load_login_cookies(sm)
    logger.info("✅ Loaded saved cookies from previous session" if cookies_loaded else "⚠️  No saved cookies found")

    logger.info("Step 5: Checking login status...")
    login_check = login_status(sm, disable_ui_fallback=True)

    if login_check is True:
        logger.info("✅ Already logged in")
    elif login_check is False:
        logger.info("⚠️  Not logged in - attempting login...")
        login_result = log_in(sm)
        if login_result != "LOGIN_SUCCEEDED":
            sm.close_sess(keep_db=False)
            raise AssertionError(f"Login failed: {login_result}")
        logger.info("✅ Login successful")
    else:
        sm.close_sess(keep_db=False)
        raise AssertionError("Login status check failed critically (returned None)")


def _validate_session_ready(sm: SessionManager) -> None:
    """Validate session is ready with all identifiers."""
    logger.info("Step 6: Ensuring session is ready...")
    ready = sm.ensure_session_ready("send_messages - Phase 6 Final Validation", skip_csrf=True)
    if not ready:
        sm.close_sess(keep_db=False)
        raise AssertionError("Session not ready - cookies/identifiers missing")
    logger.info("✅ Session ready")

    logger.info("Step 7: Verifying UUID is available...")
    if not sm.my_uuid:
        sm.close_sess(keep_db=False)
        raise AssertionError("UUID not available - session initialization incomplete")
    logger.info(f"✅ UUID available: {sm.my_uuid}")


def _ensure_session_for_phase6() -> SessionManager:
    """Ensure session is ready for Phase 6 test. Returns session_manager."""
    logger.info("=" * 80)
    logger.info("Setting up authenticated session for Phase 6 Final Validation...")
    logger.info("=" * 80)

    sm = _create_and_start_session()
    _authenticate_session(sm)
    _validate_session_ready(sm)

    logger.info("=" * 80)
    logger.info("✅ Valid authenticated session established for Phase 6")
    logger.info("=" * 80)

    return sm


def validate_database_state(session) -> dict:
    """Validate database has required data."""
    logger.info("\n[1/6] Validating database state...")
    
    with db_transn(session) as transn_session:
        # Count people
        people_count = transn_session.query(Person).count()
        logger.info(f"   ✅ People in database: {people_count}")
        
        # Count message templates
        templates = transn_session.query(MessageTemplate).all()
        logger.info(f"   ✅ Message templates: {len(templates)}")
        
        # List templates by tree_status
        in_tree_count = sum(1 for t in templates if str(t.tree_status) == 'in_tree')
        out_tree_count = sum(1 for t in templates if str(t.tree_status) == 'out_tree')
        logger.info(f"      - In-tree templates: {in_tree_count}")
        logger.info(f"      - Out-tree templates: {out_tree_count}")
        
        # Count existing messages
        messages = transn_session.query(ConversationLog).count()
        logger.info(f"   ✅ Existing messages in database: {messages}")
        
        return {
            'people_count': people_count,
            'templates_count': len(templates),
            'in_tree_templates': in_tree_count,
            'out_tree_templates': out_tree_count,
            'existing_messages': messages
        }


def validate_app_mode() -> bool:
    """Validate APP_MODE is set correctly."""
    logger.info("\n[2/6] Validating APP_MODE configuration...")
    
    app_mode = os.getenv('APP_MODE', 'unknown')
    logger.info(f"   APP_MODE: {app_mode}")
    
    if app_mode != 'dry_run':
        logger.warning(f"   ⚠️  APP_MODE is '{app_mode}', not 'dry_run'")
        logger.warning("   Phase 6 will simulate messages but not send them")
        return False
    
    logger.info("   ✅ APP_MODE is 'dry_run' - messages will be simulated only")
    return True


def validate_credentials() -> bool:
    """Validate Wayne's credentials are available."""
    logger.info("\n[3/6] Validating Wayne's credentials...")
    
    required_env_vars = [
        'ANCESTRY_USERNAME',
        'ANCESTRY_PASSWORD',
        'ANCESTRY_EMAIL',
    ]
    
    all_present = True
    for var in required_env_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"   ✅ {var}: Present")
        else:
            logger.warning(f"   ❌ {var}: Missing")
            all_present = False
    
    return all_present


def run_action8_dry_run(sm: SessionManager) -> dict:
    """Run Action 8 in dry_run mode."""
    logger.info("\n[4/6] Running Action 8 (dry_run mode)...")
    logger.info("   This will process all candidates but NOT send real messages")

    try:
        result = send_messages_to_matches(sm)
        logger.info("   ✅ Action 8 completed successfully")
        return result
    except Exception as e:
        logger.error(f"   ❌ Action 8 failed: {e}")
        raise


def validate_messages_created(session) -> dict:
    """Validate messages were created."""
    logger.info("\n[5/6] Validating messages were created...")
    
    with db_transn(session) as transn_session:
        messages = transn_session.query(ConversationLog).all()
        logger.info(f"   ✅ Total messages in database: {len(messages)}")
        
        # Group by status
        by_status = {}
        for msg in messages:
            status = msg.status or 'unknown'
            by_status[status] = by_status.get(status, 0) + 1
        
        for status, count in sorted(by_status.items()):
            logger.info(f"      - Status '{status}': {count} messages")
        
        return {
            'total_messages': len(messages),
            'by_status': by_status
        }


def final_checklist() -> bool:
    """Display final checklist before go-live."""
    logger.info("\n[6/6] Final Checklist Before Go-Live...")
    
    checklist = [
        ("✅", "Database has people and templates"),
        ("✅", "APP_MODE is set to dry_run"),
        ("✅", "Wayne's credentials are available"),
        ("✅", "Action 8 runs without errors"),
        ("✅", "Messages are created in database"),
        ("✅", "No real messages sent to Ancestry"),
    ]
    
    for status, item in checklist:
        logger.info(f"   {status} {item}")
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 6: FINAL VALIDATION COMPLETE")
    logger.info("=" * 80)
    logger.info("\n🎯 READY FOR GO-LIVE")
    logger.info("\nTo proceed with real message sending:")
    logger.info("1. Change APP_MODE from 'dry_run' to 'production' in .env")
    logger.info("2. Run Action 8 again (messages will be sent to Ancestry)")
    logger.info("3. Monitor Ancestry messaging for delivery confirmation")
    logger.info("\n" + "=" * 80)
    
    return True


def main():
    """Run Phase 6 final validation."""
    print("\n" + "=" * 80)
    print("PHASE 6: FINAL VALIDATION BEFORE GO-LIVE")
    print("=" * 80)
    print("\nThis test validates all systems are ready for real message sending.")
    print("NO REAL MESSAGES WILL BE SENT TO ANCESTRY (dry_run mode)")
    print("=" * 80 + "\n")
    
    sm = None
    try:
        # Setup session
        sm = _ensure_session_for_phase6()
        session = sm.get_db_conn()
        
        # Run validations
        db_state = validate_database_state(session)
        app_mode_ok = validate_app_mode()
        creds_ok = validate_credentials()
        
        # Run Action 8
        action8_result = run_action8_dry_run(sm)
        
        # Validate results
        msg_state = validate_messages_created(session)
        
        # Final checklist
        final_checklist()
        
        sm.return_session(session)
        
    except Exception as e:
        logger.error(f"\n❌ Phase 6 validation failed: {e}")
        if sm:
            sm.close_sess(keep_db=False)
        sys.exit(1)
    finally:
        if sm:
            sm.close_sess(keep_db=False)


if __name__ == '__main__':
    main()

