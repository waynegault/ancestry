# session_helpers.py is deprecated. The initialize_session function is now in utils.py and this file is safe to remove.

    """Initialize the session with proper authentication for standalone usage"""
    global session_manager
    if not session_manager.driver_live:
        print("Initializing browser session...")
        session_manager.ensure_driver_live()
    if not session_manager.session_ready:
        print("Authenticating with Ancestry...")
        success = session_manager.ensure_session_ready()
        if not success:
            print(
                "Failed to authenticate with Ancestry. Please login manually when the browser opens."
            )
            input("Press Enter after you've logged in manually...")
        else:
            print("Authentication successful.")
    if not session_manager.my_tree_id:
        print("Loading tree information...")
        session_manager._retrieve_identifiers()
        if not session_manager.my_tree_id:
            print("WARNING: Could not load tree ID. Some functionality may be limited.")
        else:
            print(f"Tree ID loaded successfully: {session_manager.my_tree_id}")
    return session_manager.session_ready
