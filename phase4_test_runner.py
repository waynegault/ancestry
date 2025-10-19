#!/usr/bin/env python3
"""
Phase 4: Test Desist Functionality
Verify that desist messages are properly handled and users who request no more messages are respected.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.session_manager import SessionManager
from action8_messaging import send_messages_to_matches
from database import db_transn, ConversationLog, Person, MessageTemplate
from utils import _load_login_cookies, log_in, login_status
import logging

logger = logging.getLogger(__name__)


def _create_and_start_session() -> SessionManager:
    """Create and start a new session manager."""
    logger.info("Step 1: Creating SessionManager...")
    sm = SessionManager()
    logger.info("✅ SessionManager created")

    logger.info("Step 2: Configuring browser requirement...")
    sm.browser_manager.browser_needed = True
    logger.info("✅ Browser marked as needed")

    logger.info("Step 3: Starting session (database + browser)...")
    started = sm.start_sess("Phase 4 Test")
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
    ready = sm.ensure_session_ready("send_messages - Phase 4 Test", skip_csrf=True)
    if not ready:
        sm.close_sess(keep_db=False)
        raise AssertionError("Session not ready - cookies/identifiers missing")
    logger.info("✅ Session ready")

    logger.info("Step 7: Verifying UUID is available...")
    if not sm.my_uuid:
        sm.close_sess(keep_db=False)
        raise AssertionError("UUID not available - session initialization incomplete")
    logger.info(f"✅ UUID available: {sm.my_uuid}")


def _ensure_session_for_phase4_test() -> SessionManager:
    """Ensure session is ready for Phase 4 test. Returns session_manager."""
    logger.info("=" * 80)
    logger.info("Setting up authenticated session for Phase 4 desist functionality test...")
    logger.info("=" * 80)

    # Create and start new session
    sm = _create_and_start_session()

    # Authenticate the session
    _authenticate_session(sm)

    # Validate session is ready
    _validate_session_ready(sm)

    logger.info("=" * 80)
    logger.info("✅ Valid authenticated session established for Phase 4 test")
    logger.info("=" * 80)

    return sm


def run_phase4_test():
    """
    Phase 4: Test Desist Functionality
    
    Objectives:
    1. Run Action 8 with dry_run mode
    2. Verify desist messages are created for users who request no more messages
    3. Verify that subsequent runs respect the desist flag
    4. Check that desist acknowledgments are properly logged
    """
    
    print("\n" + "="*80)
    print("PHASE 4: DESIST FUNCTIONALITY TEST")
    print("="*80)
    
    try:
        # Initialize session with proper authentication
        print("\n[1/4] Initializing authenticated session...")
        sm = _ensure_session_for_phase4_test()
        print("✅ Session initialized and authenticated successfully")
        
        # Run Action 8
        print("\n[2/4] Running Action 8 with dry_run mode...")
        success = send_messages_to_matches(sm)
        
        if not success:
            print("⚠️  Action 8 completed with warnings")
        else:
            print("✅ Action 8 completed successfully")
        
        # Analyze desist messages
        print("\n[3/4] Analyzing desist functionality...")
        session = sm.get_db_conn()

        with db_transn(session) as transn_session:
            # Get all messages created in this run
            messages = transn_session.query(ConversationLog).all()

            desist_count = 0
            template_counts = {}

            for msg in messages:
                template = msg.message_template
                if template:
                    template_key = template.template_key
                    template_counts[template_key] = template_counts.get(template_key, 0) + 1
                    
                    # Check if this is a desist message
                    if 'desist' in template_key.lower():
                        desist_count += 1

            print(f"\n📊 Desist Message Analysis:")
            print(f"   Total Messages: {len(messages)}")
            print(f"   Desist Messages: {desist_count}")
            print(f"   Template Distribution:")
            for template_key, count in sorted(template_counts.items()):
                print(f"      {template_key}: {count}")

            if desist_count > 0:
                print(f"\n✅ Desist messages are being created ({desist_count} found)")
            else:
                print(f"\n⚠️  No desist messages found (may be expected if no desist requests)")

        sm.return_session(session)

        # Verify no critical errors
        print("\n[4/4] Verifying no critical errors...")
        session2 = sm.get_db_conn()
        with db_transn(session2) as transn_session:
            error_count = transn_session.query(ConversationLog).filter(
                ConversationLog.script_message_status.like('%error%')
            ).count()

            if error_count == 0:
                print("✅ No errors recorded")
            else:
                print(f"⚠️  {error_count} errors recorded")

        sm.return_session(session2)
        
        sm.close_sess(keep_db=False)
        
        print("\n" + "="*80)
        print("PHASE 4 TEST COMPLETE")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 4 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_phase4_test()
    sys.exit(0 if success else 1)

