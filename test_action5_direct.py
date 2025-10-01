"""Direct test of Action 5 (check_login_actn)."""

import sys
sys.path.insert(0, '.')

from core.session_manager import SessionManager
from main import check_login_actn

print("=" * 70)
print("DIRECT ACTION 5 TEST")
print("=" * 70)

# Create session manager
print("\n[1/2] Creating SessionManager...")
session_manager = SessionManager()
print("✅ SessionManager created")

# Start browser first (Action 5 requires it)
print("\n[2/3] Starting browser...")
browser_started = session_manager.start_browser("Action 5 Test")
if browser_started:
    print("✅ Browser started successfully")

    # Test Action 5
    print("\n[3/3] Running Action 5 (check_login_actn)...")
    print("This will:")
    print("- Check login status")
    print("- Attempt login if needed")
    print("- Handle 2FA if required")

    print("\n" + "=" * 70)
    print("PLEASE WATCH THE BROWSER WINDOW!")
    print("If 2FA appears, you have 120 seconds to enter the code.")
    print("=" * 70)

    try:
        result = check_login_actn(session_manager)
        print(f"\n✅ Action 5 completed with result: {result}")
    except Exception as e:
        print(f"\n❌ Action 5 failed with error: {e}")
else:
    print("❌ Failed to start browser")

# Cleanup
print("\n[Cleanup] Closing session...")
session_manager.close_sess()
print("✅ Session closed")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
