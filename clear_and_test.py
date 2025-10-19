#!/usr/bin/env python3
"""
Clear conversation_log and run Phase 2 test with full error traceback capture.
"""

import sys
import os
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def clear_conversation_log():
    """Clear the conversation_log table."""
    try:
        from core.session_manager import SessionManager
        
        sm = SessionManager()
        db_session = sm.get_db_conn()
        
        if not db_session:
            print("❌ Failed to get database session")
            return False
        
        try:
            from database import ConversationLog, db_transn
            
            with db_transn(db_session) as sess:
                count = sess.query(ConversationLog).delete()
            
            print(f"✅ Cleared {count} records from conversation_log table")
            return True
        finally:
            sm.return_session(db_session)
    
    except Exception as e:
        print(f"❌ Error clearing conversation_log: {e}")
        traceback.print_exc()
        return False


def run_phase2_test():
    """Run Phase 2 test with full error capture."""
    try:
        from core.session_manager import SessionManager
        from action8_messaging import send_messages_to_matches
        
        print("\n" + "="*80)
        print("PHASE 2 TEST: Running with full error capture")
        print("="*80 + "\n")
        
        sm = SessionManager()
        sm.browser_needed = True
        sm.start_sess("Phase 2 Test")
        
        try:
            result = send_messages_to_matches(sm)
            print(f"\n✅ Test completed. Result: {result}")
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            print("\n" + "="*80)
            print("FULL TRACEBACK:")
            print("="*80)
            traceback.print_exc()
            print("="*80)
        finally:
            try:
                sm.close_sess(keep_db=False)
            except Exception as close_err:
                print(f"⚠️  Error closing session: {close_err}")
    
    except Exception as e:
        print(f"❌ Failed to initialize test: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("Step 1: Clearing conversation_log table...")
    if clear_conversation_log():
        print("\nStep 2: Running Phase 2 test...")
        run_phase2_test()
    else:
        print("Skipping test due to database clear failure")

