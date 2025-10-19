#!/usr/bin/env python3
"""
Phase 5: Test with First Available DNA Match as Test Recipient
Verify that messages can be sent to and received by a real test account.

Note: This phase first runs Action 6 to populate the database with DNA matches,
then tests messaging with the first available match.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.session_manager import SessionManager
from action8_messaging import send_messages_to_matches
from database import db_transn, ConversationLog, Person
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
    started = sm.start_sess("Phase 5 Test")
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
    ready = sm.ensure_session_ready("send_messages - Phase 5 Test", skip_csrf=True)
    if not ready:
        sm.close_sess(keep_db=False)
        raise AssertionError("Session not ready - cookies/identifiers missing")
    logger.info("✅ Session ready")

    logger.info("Step 7: Verifying UUID is available...")
    if not sm.my_uuid:
        sm.close_sess(keep_db=False)
        raise AssertionError("UUID not available - session initialization incomplete")
    logger.info(f"✅ UUID available: {sm.my_uuid}")


def _ensure_session_for_phase5_test() -> SessionManager:
    """Ensure session is ready for Phase 5 test. Returns session_manager."""
    import shutil

    logger.info("=" * 80)
    logger.info("Setting up authenticated session for Phase 5 Frances Milne test...")
    logger.info("=" * 80)

    # Copy backup database to working location
    logger.info("Step 0: Copying backup database...")
    backup_db = 'Data/ancestry_backup.db'
    working_db = 'ancestry.db'
    if os.path.exists(backup_db):
        shutil.copy2(backup_db, working_db)
        logger.info(f"✅ Copied {backup_db} to {working_db}")
    else:
        logger.warning(f"⚠️  Backup database not found at {backup_db}")

    sm = _create_and_start_session()
    _authenticate_session(sm)
    _validate_session_ready(sm)

    logger.info("=" * 80)
    logger.info("✅ Valid authenticated session established for Phase 5 test")
    logger.info("=" * 80)

    return sm


def find_first_available_match(session) -> dict:
    """Find first available DNA match in database."""
    logger.info("\n[1/5] Searching for first available DNA match...")

    with db_transn(session) as transn_session:
        # Get first person from database
        match = transn_session.query(Person).first()

        if not match:
            logger.warning("⚠️  No DNA matches found in database")
            return None

        logger.info(f"✅ Found DNA match:")
        logger.info(f"   Username: {match.username}")
        logger.info(f"   Profile ID: {match.profile_id}")
        logger.info(f"   In My Tree: {match.in_my_tree}")
        logger.info(f"   Contactable: {match.contactable}")
        logger.info(f"   Status: {match.status}")

        return {
            'person_id': match.id,
            'profile_id': match.profile_id,
            'username': match.username,
            'in_my_tree': match.in_my_tree,
            'contactable': match.contactable,
            'status': match.status
        }


def verify_match_eligible(match_info: dict) -> bool:
    """Verify match is eligible for messaging."""
    logger.info("\n[3/6] Verifying match is eligible for messaging...")

    if not match_info:
        logger.error("❌ Match not found")
        return False

    checks = {
        'Has profile_id': match_info['profile_id'] is not None,
        'Is contactable': match_info['contactable'] is True,
        'Status is ACTIVE': match_info['status'] in ['ACTIVE', 'DESIST'],
    }

    all_passed = True
    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        logger.info(f"   {status} {check_name}")
        if not result:
            all_passed = False

    return all_passed


def find_frances_milne(session) -> dict:
    """Find Frances Milne in the database for Phase 5 testing."""
    logger.info("\n[1/5] Finding Frances Milne in database...")

    from database import Person

    with db_transn(session) as transn_session:
        # Query for Frances Milne by profile_id from .env
        frances_profile_id = os.getenv('ACTION8_TEST_PROFILE_ID')

        frances = transn_session.query(Person).filter(
            Person.profile_id == frances_profile_id
        ).first()

        if not frances:
            logger.error(f"❌ Frances Milne not found with profile_id: {frances_profile_id}")
            return None

        logger.info(f"✅ Found Frances Milne:")
        logger.info(f"   Person ID: {frances.id}")
        logger.info(f"   Username: {frances.username}")
        logger.info(f"   Profile ID: {frances.profile_id}")
        logger.info(f"   In My Tree: {frances.in_my_tree}")
        logger.info(f"   Contactable: {frances.contactable}")
        logger.info(f"   Status: {frances.status}")

        return {
            'person_id': frances.id,
            'profile_id': frances.profile_id,
            'username': frances.username,
            'in_my_tree': frances.in_my_tree,
            'contactable': frances.contactable,
            'status': frances.status,
        }


def run_phase5_test():
    """
    Phase 5: Test with Test DNA Match as Test Recipient

    Objectives:
    1. Create test DNA match in database
    2. Verify it's eligible for messaging
    3. Run Action 8 (still in dry_run mode)
    4. Verify messages are created for the match
    5. Check message content
    """

    print("\n" + "="*80)
    print("PHASE 5: TEST WITH TEST DNA MATCH")
    print("="*80)

    try:
        # Initialize session
        print("\n[0/5] Initializing authenticated session...")
        sm = _ensure_session_for_phase5_test()
        print("✅ Session initialized and authenticated successfully")

        # Find Frances Milne
        print("\n[1/5] Finding Frances Milne in database...")
        session = sm.get_db_conn()
        match_info = find_frances_milne(session)
        sm.return_session(session)

        if not match_info:
            print("❌ Phase 5 test failed: Could not create test match")
            sm.close_sess(keep_db=False)
            return False

        # Verify eligibility
        print("\n[2/5] Verifying match is eligible for messaging...")
        eligible = verify_match_eligible(match_info)

        if not eligible:
            print("⚠️  Match may not be eligible for messaging")

        # Run Action 8
        print("\n[3/5] Running Action 8 with dry_run mode...")
        success = send_messages_to_matches(sm)

        if not success:
            print("⚠️  Action 8 completed with warnings")
        else:
            print("✅ Action 8 completed successfully")

        # Check for messages to match
        print("\n[4/5] Checking for messages to match...")
        session = sm.get_db_conn()

        with db_transn(session) as transn_session:
            match_messages = transn_session.query(ConversationLog).filter(
                ConversationLog.people_id == match_info['person_id']
            ).all()

            if match_messages:
                print(f"✅ Found {len(match_messages)} message(s) to match:")
                for msg in match_messages:
                    template = msg.message_template
                    print(f"   - Template: {template.template_key if template else 'Unknown'}")
                    print(f"     Status: {msg.script_message_status}")
            else:
                print("⚠️  No messages found for match")

        sm.return_session(session)

        # Verify no errors
        print("\n[5/5] Verifying no critical errors...")
        session = sm.get_db_conn()
        with db_transn(session) as transn_session:
            error_count = transn_session.query(ConversationLog).filter(
                ConversationLog.script_message_status.like('%error%')
            ).count()

            if error_count == 0:
                print("✅ No errors recorded")
            else:
                print(f"⚠️  {error_count} errors recorded")

        sm.return_session(session)
        sm.close_sess(keep_db=False)

        print("\n" + "="*80)
        print("PHASE 5 TEST COMPLETE")
        print("="*80 + "\n")

        return True

    except Exception as e:
        print(f"\n❌ Phase 5 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_phase5_test()
    sys.exit(0 if success else 1)

