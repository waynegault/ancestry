#!/usr/bin/env python3

"""
delete_conversation_log.py - Script to delete all entries from the conversation_log table

This script connects to the database and deletes all entries from the conversation_log table.
It can be used to reset the conversation log without affecting other data.
"""

import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# Import the database models and config
from database import ConversationLog, db_transn
from config import config_instance
from logging_config import logger, setup_logging

def delete_conversation_log():
    """
    Deletes all entries from the conversation_log table.
    
    Returns:
        bool: True if deletion was successful, False otherwise.
    """
    # Get the database file path from config
    db_path = config_instance.DATABASE_FILE
    if db_path is None:
        logger.error("DATABASE_FILE is not configured. Deletion aborted.")
        return False
    
    # Create engine and session
    engine = None
    session = None
    try:
        # Create engine
        db_path_str = str(db_path.resolve())
        logger.info(f"Connecting to database: {db_path_str}")
        engine = create_engine(f"sqlite:///{db_path_str}", echo=False)
        
        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Count records before deletion
        count_before = session.query(ConversationLog).count()
        logger.info(f"Found {count_before} records in conversation_log table.")
        
        # Delete all records within a transaction
        with db_transn(session) as sess:
            # Delete all records from conversation_log
            deleted_count = sess.query(ConversationLog).delete(synchronize_session=False)
            logger.info(f"Deleted {deleted_count} records from conversation_log table.")
        
        # Verify deletion
        count_after = session.query(ConversationLog).count()
        logger.info(f"Remaining records in conversation_log table: {count_after}")
        
        if count_after == 0:
            logger.info("All conversation log entries successfully deleted.")
            return True
        else:
            logger.warning(f"Some records ({count_after}) remain in the conversation_log table.")
            return False
            
    except SQLAlchemyError as e:
        logger.error(f"Database error during conversation log deletion: {e}", exc_info=True)
        if session:
            try:
                session.rollback()
            except Exception:
                pass
        return False
    except Exception as e:
        logger.error(f"Unexpected error during conversation log deletion: {e}", exc_info=True)
        if session:
            try:
                session.rollback()
            except Exception:
                pass
        return False
    finally:
        # Close session and dispose engine
        if session:
            session.close()
        if engine:
            engine.dispose()
            logger.debug("Database engine disposed.")

if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()
    
    print("=== Conversation Log Deletion Tool ===")
    print("This will delete ALL entries from the conversation_log table.")
    print("This action cannot be undone.")
    
    # Ask for confirmation
    confirm = input("Are you sure you want to proceed? (yes/no): ").strip().lower()
    if confirm in ["yes", "y"]:
        print("\nDeleting conversation log entries...")
        success = delete_conversation_log()
        if success:
            print("\n✓ All conversation log entries successfully deleted.")
        else:
            print("\n✗ Error occurred during deletion. Check the log file for details.")
    else:
        print("\nOperation cancelled.")
    
    print("\nExecution finished.")
