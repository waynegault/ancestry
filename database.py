#!/usr/bin/env python3

# database.py

# Imports
import os
import shutil
from typing import List, Dict, Optional, Tuple, Any, Type, Literal
from pathlib import Path
from datetime import datetime, timezone 
import time
import gc
from config import config_instance
import contextlib
from contextlib import contextmanager
import enum
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    event,
    Boolean,
    UniqueConstraint,
    Enum,
    Index,
    func,
    Float
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session, joinedload
from sqlalchemy.exc import SQLAlchemyError, IntegrityError # Added IntegrityError
import logging
import re
import inspect


# Initialize logging
logger = logging.getLogger("logger")

# ----------------------------------------------------------------------
# SQLAlchemy Models
# ----------------------------------------------------------------------

Base = declarative_base()


class RoleType(enum.Enum):
    AUTHOR = 'AUTHOR'
    RECIPIENT = 'RECIPIENT'

class InboxStatus(Base):
    __tablename__ = "inbox_status"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String, nullable=True)
    people_id = Column(Integer, ForeignKey("people.id"), nullable=False, index=True)
    my_role = Column(Enum(RoleType), nullable=False, name='my_role')
    last_message = Column(String, nullable=True)
    last_message_timestamp = Column(DateTime, nullable=True, index=True) # Add index=True
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    last_updated = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)
    person = relationship("Person", back_populates="inbox_status")
    __table_args__ = (Index('ix_inbox_status_people_id_timestamp', 'people_id', 'last_message_timestamp'),)
# End of class InboxStatus

class MessageType(Base):  # Add MessageType
    __tablename__ = "message_types"
    id = Column(Integer, primary_key=True)
    type_name = Column(String, unique=True, nullable=False)
    messages = relationship(
        "MessageHistory",
        back_populates="message_type",
        cascade="all, delete, delete-orphan",
    )
# End of class MessageType

class MessageHistory(Base):
    __tablename__ = "message_history"
    id = Column(Integer, primary_key=True)
    people_id = Column(Integer, ForeignKey("people.id"), nullable=False)
    message_type_id = Column(
        Integer, ForeignKey("message_types.id"), nullable=False
    )
    message_text = Column(String, nullable=False)
    status = Column(String, default="none", nullable=False)
    sent_at = Column(DateTime, default=datetime.now, nullable=False)
    person = relationship("Person", back_populates="message_history")
    message_type = relationship("MessageType", back_populates="messages")
# End of class messagehistory

class DnaMatch(Base):
    __tablename__ = "dna_match"
    id = Column(Integer, primary_key=True)
    compare_link = Column(String, nullable=False)
    cM_DNA = Column(Integer, nullable=False)
    predicted_relationship = Column(String, nullable=False)
    people_id = Column(Integer, ForeignKey("people.id"), unique=True, nullable=False, index=True)
    shared_segments = Column(Integer, nullable=True)
    longest_shared_segment = Column(Float, nullable=True)
    meiosis = Column(Integer, nullable=True)
    from_my_fathers_side = Column(Boolean, default=False, nullable=False)
    from_my_mothers_side = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )
    person = relationship("Person", back_populates="dna_match")
# End of class DnaMatch

class FamilyTree(Base): # Point 8 Refinement
    __tablename__ = "family_tree"
    id = Column(Integer, primary_key=True)
    people_id = Column(Integer, ForeignKey("people.id"), unique=True, nullable=False, index=True)
    cfpid = Column(String,  unique=True, nullable=True)
    person_name_in_tree = Column(String, nullable=True)
    facts_link = Column(String, nullable=True)
    view_in_tree_link = Column(String, nullable=True)
    actual_relationship = Column(String, nullable=True)
    relationship_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )
    person = relationship("Person", back_populates="family_tree")
# End of class family tree

class Person(Base):
    __tablename__ = "people"
    id = Column(Integer, primary_key=True)
    uuid = Column(String, nullable=True, unique=True, index=True) # Keep UUID unique
    # --- RE-ADD unique=True to profile_id ---
    profile_id = Column(String,  unique=True, nullable=True, index=True)
    # --- END RE-ADD ---
    username = Column(String,  unique=False, nullable=False)
    first_name = Column(String, nullable=True)
    gender = Column(String(1), nullable=True) # Store 'f', 'm', or None
    birth_year = Column(Integer, nullable=True)
    message_link = Column(String, unique=False, nullable=True)
    in_my_tree = Column(Boolean, default=False)
    contactable =  Column(Boolean, default=False)
    last_logged_in = Column(DateTime, nullable=True, index=True) # Store as naive UTC
    administrator_profile_id = Column(String, nullable=True, index=True)
    administrator_username = Column(String, nullable=True)
    status = Column(String, default="active", nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )

    # Relationships (remain the same)
    family_tree = relationship("FamilyTree", back_populates="person", uselist=False, cascade="all, delete, delete-orphan")
    dna_match = relationship("DnaMatch", back_populates="person", uselist=False, cascade="all, delete, delete-orphan")
    inbox_status = relationship("InboxStatus", back_populates="person", uselist=False, cascade="all, delete, delete-orphan", foreign_keys="InboxStatus.people_id") # Corrected foreign_keys usage
    message_history = relationship("MessageHistory", back_populates="person", cascade="all, delete, delete-orphan")

   
    # If you prefer SQLAlchemy checks, uncomment this, otherwise DB unique handles it
    # __table_args__ = (
    #     UniqueConstraint('profile_id', name='uq_people_profile_id'),
    #     UniqueConstraint('uuid', name='uq_people_uuid') # Ensure UUID constraint name if needed
    # )
    
# End of class Person

# ----------------------------------------------------------------------
# COnnection Pooling
# ----------------------------------------------------------------------

class ConnectionPool:
    """
    Manages a pool of SQLAlchemy Session objects for SQLite database connections.
    Includes configuration for foreign key support.
    """

    def __init__(self, db_path: str, pool_size: int): # Expects string path from config
        """
        Initializes the connection pool.

        Args:
            db_path (str): The string path to the SQLite database file.
            pool_size (int): The maximum number of sessions to keep in the pool.
        """
        self.db_path_str = db_path # Store the string path used

        # --- Log Absolute Path ---
        try:
            # Convert to Path object and resolve for absolute path logging
            abs_path = Path(db_path).resolve()
            logger.debug(f"ConnectionPool initializing for DB:\n{abs_path}")
        except Exception as path_e:
            logger.error(f"Error resolving absolute path for DB '{db_path}': {path_e}")
            # Continue using the provided db_path string even if resolving fails
        # --- End Path Logging ---

        # Create SQLAlchemy engine using the string path for the URI
        try:
            # echo=False is standard, set to True for verbose SQL logging if needed
            self.engine = create_engine(f"sqlite:///{self.db_path_str}", echo=False)
            logger.debug(f"SQLAlchemy engine created for:\n{self.db_path_str}")
        except Exception as engine_e:
            logger.critical(f"Failed to create SQLAlchemy engine: {engine_e}", exc_info=True)
            raise # Re-raise critical error
         # --- SQLite Specific Event Listener for Foreign Keys ---
        @event.listens_for(self.engine, "connect")
        def enable_foreign_keys(dbapi_connection, connection_record):
            """Enable foreign key support for SQLite on each connection."""
            cursor = None # Initialize cursor to None
            try:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
            except Exception as pragma_e:
                 logger.error(f"Failed to execute PRAGMA foreign_keys=ON: {pragma_e}")
            finally:
                 if cursor:
                      cursor.close() # Ensure cursor is closed
        # --- End Foreign Key Listener ---

        # Create a configured "Session" class
        self.Session = sessionmaker(
            bind=self.engine,
            expire_on_commit=False # Recommended for managing sessions manually
        )

        # Initialize the pool list and size
        self._pool: List[Session] = []
        self.pool_size = pool_size

        # Optionally pre-fill the pool, or fill on demand
        # self._fill_pool()
        logger.debug(f"ConnectionPool initialized with target pool size {self.pool_size}.")
    # end __init__

    def _create_session(self) -> Optional[Session]:
        """Creates a new SQLAlchemy session instance."""
        try:
            new_session = self.Session()
            return new_session
        except SQLAlchemyError as e:
            logger.error(f"Error creating new SQLAlchemy session: {e}", exc_info=True)
            return None
    # end _create_session

    def _fill_pool(self):
        """Fills the connection pool up to the desired size."""
        logger.debug(f"Attempting to fill pool (current size: {len(self._pool)}, target: {self.pool_size}).")
        while len(self._pool) < self.pool_size:
            session = self._create_session()
            if session:
                self._pool.append(session)
            else:
                logger.warning("Failed to create session during pool filling.")
                break # Stop filling if session creation fails
        logger.debug(f"Pool filling finished. Pool size: {len(self._pool)}.")
    # end _fill_pool

    def get_session(self) -> Optional[Session]:
        """
        Gets a Session from the pool or creates a new one if pool is empty.
        Returns None if session creation fails.
        """
        if not self._pool:
            session = self._create_session()
            if not session:
                 logger.error("Failed to get db session: Pool empty and creation failed.")
                 return None
            # Optionally fill pool if it was empty
            # self._fill_pool() # Or just return the single created session
            return session
        else:
             session = self._pool.pop()
             logger.debug(f"Retrieved session {id(session)} from pool. Pool size now: {len(self._pool)}")
             # Check if session is still active (optional, adds overhead)
             # if not session.is_active:
             #    logger.warning(f"Session {id(session)} retrieved from pool was inactive. Creating new one.")
             #    session.close() # Close inactive session
             #    return self.get_session() # Recursively get another one
             return session
    #  end get_session

    def return_session(self, session: Session):
        """Returns a Session to the pool if space is available, otherwise closes it."""
        if not session:
            logger.warning("Attempted to return a None session to the pool.")
            return

        session_id = id(session) # Get ID for logging before potential close

        # Rollback any pending changes before returning/closing
        # This prevents returning a session with an open transaction.
        try:
            if session.is_active and session.dirty:
                 logger.warning(f"Session {session_id} returned dirty. Rolling back...")
                 session.rollback()
        except Exception as rb_err:
             logger.error(f"Error rolling back dirty session {session_id} on return: {rb_err}")
             # Proceed to close/discard the session if rollback fails
             try: session.close()
             except: pass
             return

        # Check if pool has space
        if len(self._pool) < self.pool_size:
            self._pool.append(session)
            logger.debug(f"Returned session {session_id} to pool. Pool size now: {len(self._pool)}")
        else:
            # Pool is full, close the session instead of adding it back
            logger.debug(f"Pool full ({len(self._pool)}/{self.pool_size}). Closing returned session {session_id}.")
            try:
                session.close()
                logger.debug(f"Session {session_id} closed.")
            except SQLAlchemyError as e:
                logger.error(f"Error closing session {session_id} when returning to full pool: {e}")
    # end return_session

    def clse_all_sess(self):
        """Closes all sessions currently held in the pool and clears the pool."""
        if not self._pool and not self.engine:
             logger.debug("clse_all_sess called, but no pool or engine exists.")
             return

        closed_count = 0
        pool_size_before = len(self._pool)
        logger.debug(f"Closing all {pool_size_before} sessions in the pool...")
        # Iterate over a copy of the list to allow removal
        for session in list(self._pool):
            session_id = id(session)
            try:
                if session.is_active: session.rollback() # Rollback before closing
                session.close()
                closed_count += 1
                logger.debug(f"Closed pooled session {session_id}.")
            except SQLAlchemyError as e:
                logger.error(f"Error closing pooled session {session_id}: {e}")
            # Ensure removal from pool even if closing fails
            if session in self._pool:
                 self._pool.remove(session)

        self._pool = [] # Explicitly clear the pool list
        logger.debug(f"Closed {closed_count}/{pool_size_before} sessions. Pool cleared.")

        # Also dispose the engine associated with this pool
        if self.engine:
             try:
                  logger.debug("Disposing SQLAlchemy engine...")
                  self.engine.dispose()
                  self.engine = None # Mark as disposed
                  logger.debug("SQLAlchemy engine disposed.")
             except Exception as e:
                  logger.error(f"Error disposing SQLAlchemy engine: {e}")
    # end clse_all_sess
# end connectionpool class

# ----------------------------------------------------------------------
# Context Manager
# ----------------------------------------------------------------------

@contextlib.contextmanager
def db_transn(session: Session):
    """
    Context manager for database transactions. Handles committing and rolling back.
    Includes basic check for session validity.
    """
    if not session or not session.is_active:
        raise SQLAlchemyError("Invalid or inactive session provided to db_transn.")

    try:
        yield session
        # Attempt commit only if session is still active after the 'yield' block
        if session.is_active:
             session.commit()
    except Exception as e:
        logger.error(f"Exception within db_transn block: {e}. Rolling back.", exc_info=True)
        # Attempt rollback only if session is still active
        if session and session.is_active:
             session.rollback()
        # Re-raise the original exception after rollback attempt
        raise
    finally:
        # Optional: Session closure/return logic could be added here if needed,
        # but typically managed by ConnectionPool.return_session outside the transaction.
        pass
# end db_transn

# ==================== CRUD OPERATIONS ====================


# ----------------------------------------------------------------------
# Create/ Insert
# ----------------------------------------------------------------------

def create_person(session: Session, person_data: Dict[str, Any]) -> int:
    """
    Creates a new person record in the database.
    Checks for existing profile_id before insertion to prevent UNIQUE constraint errors.
    Assumes UUID uniqueness is checked by the caller or DB constraint.

    Args:
        session: The SQLAlchemy Session.
        person_data: A dictionary containing the person's data. Must include 'uuid', 'username'.
                    Should also include 'profile_id' (can be None).

    Returns:
        The ID of the newly created person, or 0 if creation failed.
    """
    required_keys = ("username", "uuid")
    if not all(key in person_data and person_data[key] is not None for key in required_keys):
         logger.warning(f"Missing required non-null data for person creation: Needs username and uuid. Data: {person_data}")
         return 0

    profile_id = person_data.get("profile_id") # Can be None
    username = person_data["username"]
    uuid = person_data["uuid"]
    log_ref = f"UUID={uuid} / ProfileID={profile_id or 'NULL'} / User='{username}'"

    try:
        # --- Explicit Check for Profile ID Conflict BEFORE insert ---
        if profile_id:
            existing_by_profile = session.query(Person).filter(Person.profile_id == profile_id.upper()).first()
            if existing_by_profile:
                # Log the conflict and prevent insertion attempt
                logger.error(f"create_person FAILED for {log_ref}: Profile ID '{profile_id}' already exists for Person ID {existing_by_profile.id} (UUID: {existing_by_profile.uuid}). Cannot create duplicate profile_id.")
                return 0 # Fail creation due to profile_id uniqueness requirement
        # --- End Profile ID Check ---

        # Proceed with creation if profile_id is NULL or not conflicting
        logger.debug(f"Proceeding with Person creation for {log_ref}.")
        new_person = Person(
            uuid=uuid.upper(),
            profile_id=profile_id.upper() if profile_id else None,
            username=username,
            administrator_profile_id=person_data.get("administrator_profile_id"),
            administrator_username=person_data.get("administrator_username"),
            message_link=person_data.get("message_link"),
            in_my_tree=bool(person_data.get("in_my_tree", False)),
            status="active",
            first_name=person_data.get("first_name"),
            gender=person_data.get("gender"),
            birth_year=person_data.get("birth_year"),
            contactable=person_data.get("contactable", True),
            last_logged_in=person_data.get("last_logged_in"), # Store as naive UTC from details_fetched
        )

        session.add(new_person)
        session.flush() # Flush to get the ID and check constraints (like UUID UNIQUE)

        if new_person.id is None:
            logger.error(f"ID not assigned after flush for person {log_ref}! Data: {person_data}")
            session.rollback()
            return 0

        logger.debug(f"Created Person record ID {new_person.id} for {log_ref}.")
        return int(new_person.id)

    except IntegrityError as ie:
         # Handles potential unique constraint violations (primarily UUID now)
         session.rollback()
         logger.error(f"IntegrityError creating person {log_ref}: {ie}. UUID likely exists.", exc_info=False)
         return 0
    except SQLAlchemyError as e:
        logger.error(f"Database error creating person {log_ref}: {e}", exc_info=True)
        if session.is_active: session.rollback() # Ensure rollback on general DB error
        return 0
    except Exception as e: # Catch any other unexpected error
         logger.critical(f"Unexpected error in create_person for {log_ref}: {e}", exc_info=True)
         if session.is_active: session.rollback()
         return 0
# End of create_person


def create_dna_match(session: Session, match_data: Dict[str, Any]) -> Literal["created", "skipped", "error"]:
    """
    Creates a new DNA Match record associated with a Person ID.
    Handles validation and inclusion of ALL relevant fields.
    Skips creation if a record for the people_id already exists.

    Args:
        session: The SQLAlchemy Session.
        match_data: Dictionary containing DNA match details. Must include 'people_id'.
                    Should include 'compare_link', 'cM_DNA', 'predicted_relationship'.
                    Can include 'shared_segments', 'longest_shared_segment', 'meiosis',
                    'from_my_fathers_side', 'from_my_mothers_side'.

    Returns:
        "created" if a new record was added.
        "skipped" if a record already existed.
        "error" if any validation or database error occurred.
    """
    people_id = match_data.get("people_id")
    log_ref = f"PersonID={people_id}, KitUUID={match_data.get('uuid', 'N/A')}" # Add UUID for context if available

    # --- 1. Basic Validation ---
    if not people_id or not isinstance(people_id, int) or people_id <= 0:
        logger.error(f"create_dna_match: Invalid or missing people_id for {log_ref}.")
        return "error"

    required_keys = ("compare_link", "cM_DNA", "predicted_relationship")
    if not all(key in match_data and match_data[key] is not None for key in required_keys):
        logger.error(
            f"create_dna_match: Missing required non-null core DNA data for {log_ref}. Match data: {match_data}"
        )
        return "error"

    # --- 2. Detailed Field Validation (Copy logic from old _do_DNA) ---
    try:
        cm_dna_val = int(match_data["cM_DNA"])
        if cm_dna_val < 0:
            raise ValueError("cM_DNA cannot be negative")
    except (TypeError, ValueError, KeyError):
        logger.error(
            f"create_dna_match: Invalid cM_DNA value '{match_data.get('cM_DNA')}' for {log_ref}."
        )
        return "error"

    # Validate optional numeric fields (helper function)
    def validate_optional_numeric(key, value, allow_float=False):
        if value is None:
            return None # None is valid
        try:
            if isinstance(value, str) and not value.replace('.', '', 1).isdigit():
                logger.warning(
                    f"create_dna_match: Non-numeric value '{value}' for {key} in {log_ref}. Setting to None."
                )
                return None
            return float(value) if allow_float else int(value)
        except (TypeError, ValueError):
            logger.warning(
                f"create_dna_match: Invalid {key} '{value}' for {log_ref}. Setting to None."
            )
            return None

    shared_segments_val = validate_optional_numeric(
        "shared_segments", match_data.get("shared_segments")
    )
    longest_segment_val = validate_optional_numeric(
        "longest_shared_segment", match_data.get("longest_shared_segment"), allow_float=True
    )
    meiosis_val = validate_optional_numeric("meiosis", match_data.get("meiosis"))

    # --- 3. Check Existence & Create ---
    try:
        # Check if a record already exists for this person_id
        dna_match = session.query(DnaMatch).filter_by(people_id=people_id).first()

        if dna_match:
            logger.debug(
                f"Existing 'dna_match' record found for {log_ref}. Skipping creation."
            )
            return "skipped"
        else:
            # Create new DNA match record including ALL fields
            logger.debug(f"Creating new 'dna_match' record for {log_ref}")
            new_dna_match = DnaMatch(
                people_id=people_id,
                compare_link=match_data["compare_link"],
                cM_DNA=cm_dna_val, # Use validated int value
                predicted_relationship=match_data["predicted_relationship"],
                # --- INCLUDE ALL OTHER FIELDS ---
                shared_segments=shared_segments_val,
                longest_shared_segment=longest_segment_val,
                meiosis=meiosis_val,
                from_my_fathers_side=bool(match_data.get("from_my_fathers_side", False)), # Ensure boolean
                from_my_mothers_side=bool(match_data.get("from_my_mothers_side", False)), # Ensure boolean
                # created_at/updated_at have defaults
            )
            session.add(new_dna_match)
            # No flush here, commit happens in the calling function (_do_match)
            logger.debug(f"Staged new 'dna_match' record for creation: {log_ref}")
            return "created"

    except IntegrityError as ie: # Should be rare now due to explicit check
        session.rollback() # Rollback immediately on integrity error
        logger.error(
            f"IntegrityError in create_dna_match for {log_ref}: {ie}. Likely concurrent creation.",
            exc_info=False, # Less verbose for integrity error
        )
        # Double-check existence after rollback, in case of race condition
        existing = session.query(DnaMatch).filter_by(people_id=people_id).first()
        if existing:
            logger.warning(f"Found existing DnaMatch for {log_ref} after IntegrityError, returning 'skipped'.")
            return "skipped"
        return "error" # Return error if still not found after rollback
    except SQLAlchemyError as e:
        # Don't rollback here; let the calling transaction handle it
        logger.error(
            f"Database error in create_dna_match for {log_ref}: {e}", exc_info=True
        )
        return "error"
    except Exception as e: # Catch any other unexpected error
        logger.error(
            f"Unexpected error in create_dna_match for {log_ref}: {e}", exc_info=True
        )
        return "error"
# End of create_dna_match


def create_family_tree(session: Session, tree_data: Dict[str, Any]) -> Literal["created", "skipped", "error"]:
    """
    Creates a new FamilyTree record if one doesn't exist for the people_id.
    Corrected: Ensures ONLY valid keys from the input dictionary are used
               when instantiating the FamilyTree object.
    """
    people_id = tree_data.get("people_id")
    if not people_id:
        logger.error("Cannot create FamilyTree: 'people_id' missing from data.")
        return "error"

    log_ref = f"PersonID={people_id}, CFPID={tree_data.get('cfpid', 'N/A')}"

    try:
        # Check if a FamilyTree record already exists
        existing_tree = session.query(FamilyTree).filter_by(people_id=people_id).first()
        if existing_tree:
            logger.debug(f"FamilyTree already exists for {log_ref}. Skipping creation.")
            return "skipped"

        logger.debug(f"Creating new FamilyTree record for {log_ref}")

        # --- FINAL CORRECTION: Explicitly map known valid keys ---
        # Create the FamilyTree object by explicitly referencing the keys
        # expected in the tree_data dictionary passed from _do_match.
        new_tree = FamilyTree(
            people_id=people_id, # Must be present
            cfpid=tree_data.get("cfpid"),
            person_name_in_tree=tree_data.get("person_name_in_tree"), # Correct model field name
            facts_link=tree_data.get("facts_link"),
            view_in_tree_link=tree_data.get("view_in_tree_link"),
            actual_relationship=tree_data.get("actual_relationship"),
            relationship_path=tree_data.get("relationship_path")
            # DO NOT pass tree_data directly or use **tree_data here.
        )
        # --- END FINAL CORRECTION ---

        session.add(new_tree)
        logger.debug(f"Staged new FamilyTree record for {log_ref}")
        return "created"

    except TypeError as te:
         # Rollback handled by caller (_do_match)
         # This log now clearly indicates an issue with the constructor call itself
         logger.critical(f"TypeError creating FamilyTree object for {log_ref}: {te}. Check arguments passed to FamilyTree().", exc_info=True)
         logger.error(f"Data keys attempted: {list(tree_data.keys())}") # Log keys that were available
         return "error"
    except IntegrityError as ie:
        # Rollback handled by caller (_do_match)
        logger.error(f"IntegrityError creating FamilyTree for {log_ref}: {ie}", exc_info=False)
        return "error"
    except SQLAlchemyError as e:
        # Rollback handled by caller (_do_match)
        logger.error(f"SQLAlchemyError creating FamilyTree for {log_ref}: {e}", exc_info=True)
        return "error"
    except Exception as e:
        # Rollback handled by caller (_do_match)
        logger.critical(f"Unexpected error in create_family_tree for {log_ref}: {e}", exc_info=True)
        return "error"
# End create_family_tree

# ----------------------------------------------------------------------
# Retrieve
# ----------------------------------------------------------------------

def get_person_by_profile_id_and_username(session: Session, profile_id: str, username: str) -> Optional[Person]:
    """Retrieves a Person by profile_id AND username."""
    if not profile_id or not username:
        logger.warning("get_person_by_profile_id_and_username: profile_id and username are required.")
        return None
    try:
        return session.query(Person).filter_by(profile_id=profile_id, username=username).first()
    except Exception as e:
        logger.error(f"Error retrieving person by profile_id/username: {e}", exc_info=True)
        return None
# end get_person_by_profile_id_and_username

def get_person_by_profile_id(session: Session, profile_id: str) -> Optional[Person]:
    """
    Retrieves a Person record based on profile_id.

    Args:
        session: The SQLAlchemy database session.
        profile_id: The profile ID to search for.

    Returns:
        The Person object if found, otherwise None.
    """
    if not profile_id:
        logger.warning("get_person_by_profile_id: profile_id is required.")
        return None

    try:
        person = session.query(Person).filter_by(profile_id=profile_id).first()
        return person
    except Exception as e:
        logger.error(f"Error retrieving person by profile_id '{profile_id}': {e}", exc_info=True)
        return None
# end of get_person_by_profile_id

def get_person_and_dna_match(
    session: Session, match_data: Dict[str, Any]
) -> Tuple[Optional[Person], Optional[DnaMatch]]:
    """
    Retrieves a Person and their associated DnaMatch record using the profile_id and username.
    Handles potential errors gracefully.

    Args:
        session: The SQLAlchemy database session.
        match_data: A dictionary containing match data, must include 'profile_id' and 'username'.

    Returns:
        A tuple containing the Person and DnaMatch objects, or (None, None) if not found or error occurs.
    """
    profile_id = match_data.get("profile_id")
    username = match_data.get("username")

    if not profile_id or not username:
        logger.warning("get_person_and_dna_match: profile_id and username are required.")
        return None, None # Return None tuple if input is invalid

    try:
        # Eager load dna_match to avoid separate query
        person = (
             session.query(Person)
             .options(relationship(Person.dna_match)) # Use relationship() for eager load option
             .filter_by(profile_id=profile_id, username=username)
             .first()
        )
        if person:
            # dna_match is already loaded due to options() if it exists
            dna_match = person.dna_match
            return person, dna_match
        else:
            return None, None # Person not found
    except Exception as e:
        logger.error(f"Error retrieving person/DNA match for profile {profile_id}/{username}: {e}", exc_info=True)
        return None, None # Return None tuple on error
# end of get_person_and_dna_match

def find_existing_person(session: Session, identifier_data: Dict[str, Any]) -> Optional[Person]:
    """
    Finds an existing person based on available unique identifiers (uuid or profile_id).
    Prioritizes UUID. If multiple matches on profile_id, uses username for disambiguation.
    Returns the Person object or None if not found or cannot be reliably identified.
    """
    person_uuid = identifier_data.get("uuid")
    person_profile_id = identifier_data.get("profile_id")
    person_username = identifier_data.get("username") # Get username for disambiguation

    person = None
    # Create a more informative log reference
    log_parts = []
    if person_uuid: log_parts.append(f"UUID='{person_uuid}'")
    if person_profile_id: log_parts.append(f"ProfileID='{person_profile_id}'")
    if person_username: log_parts.append(f"User='{person_username}'")
    log_ref = " / ".join(log_parts)

    try:
        # 1. Prioritize UUID lookup
        if person_uuid:
            uuid_upper = str(person_uuid).upper()
            # Query using the Person model and filter
            person = session.query(Person).filter(Person.uuid == uuid_upper).first()
            if person:
                logger.debug(f"Found existing person by UUID: {uuid_upper} (ID: {person.id})")
                return person

        # 2. Fallback to Profile ID
        # Check 'person is None' explicitly here
        if person is None and person_profile_id:
            profile_id_upper = str(person_profile_id).upper()
            # Query using the Person model and filter, get all potential matches
            potential_matches = session.query(Person).filter(Person.profile_id == profile_id_upper).all()

            if not potential_matches:
                logger.debug(f"No person found by Profile ID: {profile_id_upper}.")
                # Continue to final return None below

            elif len(potential_matches) == 1:
                person = potential_matches[0]
                logger.debug(f"Found unique person by Profile ID: {profile_id_upper} (ID: {person.id})")
                return person # Return the single match

            else: # Multiple matches found for the same profile_id
                logger.warning(f"Multiple people found ({len(potential_matches)}) for Profile ID: {profile_id_upper}. Attempting disambiguation by username.")
                if person_username:
                    username_lower = person_username.lower()
                    found_by_username = None
                    for p in potential_matches:
                        # Case-insensitive username comparison
                        if p.username and p.username.lower() == username_lower:
                            # Check if we already found one (shouldn't happen with unique constraint, but safety check)
                            if found_by_username is not None:
                                 logger.error(f"CRITICAL: Found multiple people matching BOTH Profile ID {profile_id_upper} AND Username '{person_username}' (IDs: {found_by_username.id}, {p.id}). Cannot reliably identify.")
                                 return None # Cannot reliably identify
                            found_by_username = p

                    if found_by_username:
                        logger.debug(f"Disambiguated by username: Found Person ID {found_by_username.id} ('{found_by_username.username}') matching Profile ID {profile_id_upper}.")
                        return found_by_username
                    else:
                        # If username provided but none matched among the profile_id matches
                        logger.warning(f"Multiple matches for Profile ID {profile_id_upper}, but none matched username '{person_username}'. Cannot reliably identify.")
                        return None # Return None to prevent updating the wrong record
                else:
                    # Multiple matches, but no username provided for disambiguation
                    logger.warning(f"Multiple matches for Profile ID {profile_id_upper}, but no username provided for disambiguation. Cannot reliably identify.")
                    return None # Return None to prevent updating the wrong record

        # If not found by UUID and either not found by Profile ID or disambiguation failed
        # This check ensures the final log message is accurate
        if person is None:
            logger.debug(f"No existing person reliably identified for {log_ref}.")

    except SQLAlchemyError as e:
         logger.error(f"Database error in find_existing_person for {log_ref}: {e}", exc_info=True)
         return None # Return None on DB error
    except Exception as e:
         logger.error(f"Unexpected error in find_existing_person for {log_ref}: {e}", exc_info=True)
         return None

    # Return the found person (which might be None if not found/identified)
    return person
# end of find_existing_person

def get_person_by_uuid(session: Session, uuid: str) -> Optional[Person]:
    """Retrieves a Person record based on their UUID (DNA sampleId)."""
    if not uuid:
        logger.warning("get_person_by_uuid: UUID is required.")
        return None
    try:
        uuid_upper = str(uuid).upper()
        # Optionally eager load related data if often needed after getting by uuid
        person = (
             session.query(Person)
             # --- MODIFICATION: Eager load family_tree as well ---
             .options(relationship(Person.dna_match), relationship(Person.family_tree)) # Use relationship() for options
             # --- END MODIFICATION ---
             .filter(Person.uuid == uuid_upper)
             .first()
        )
        return person
    except Exception as e:
        logger.error(f"Error retrieving person by UUID {uuid}: {e}", exc_info=True)
        return None
# end of get_person_by_uuid

# ----------------------------------------------------------------------
# Update
# ----------------------------------------------------------------------

# --- Function Modified ---
def need_update(
    session: Session, match: Dict[str, Any]
) -> Tuple[Optional[Person], bool, bool]:
    """
    REVISED: Checks if related data needs updating based *only* on UUID lookup.
    Determines if DNA record needs creation or Tree data needs fetching.
    Tree data is needed if flag changes False->True OR if flag is True but record missing.
    Does NOT determine if Person record itself needs update (handled later).
    Expires object state before comparisons.

    Args:
        session: The SQLAlchemy session.
        match: Dictionary containing initial match data including 'uuid', 'username', 'in_my_tree'.

    Returns:
        Tuple: (existing_person_hint, create_dna_needed, fetch_tree_data)
               existing_person_hint is the Person object found by UUID (or None).
    """
    create_dna_needed = False
    fetch_tree_data = False
    existing_person_hint: Optional[Person] = None

    match_uuid = match.get("uuid")
    match_username = match.get("username", f"Unknown Match UUID {match_uuid or 'N/A'}")
    match_in_my_tree = match.get("in_my_tree", False)

    log_ref = f"UUID='{match_uuid or 'N/A'}' / User='{match_username}'"

    if not match_uuid:
        logger.error(f"Cannot process need_update for {match_username}: 'uuid' is missing.")
        return None, False, False # Cannot proceed without UUID

    try:
        # --- Lookup primarily by UUID ---
        # Eager load relationships needed for checks
        existing_person_hint = (
            session.query(Person)
            .options(joinedload(Person.dna_match), joinedload(Person.family_tree)) # Eager load
            .filter(Person.uuid == match_uuid.upper())
            .first()
        )

        # --- Expire state after fetch, before comparison (still useful) ---
        if existing_person_hint:
            try:
                logger.debug(
                    f"{log_ref}: Expiring state for existing Person ID {existing_person_hint.id} before checks."
                )
                session.expire(existing_person_hint, ['dna_match', 'family_tree']) # Expire specific relationships too if needed
            except Exception as expire_e:
                logger.warning(
                    f"Could not expire session state for Person ID {existing_person_hint.id}: {expire_e}"
                )
        # --- End Expire ---

        # --- Determine need based on existing record ---
        if existing_person_hint:
            # Person found by UUID
            logger.debug(f"{log_ref}: Found existing Person ID {existing_person_hint.id} by UUID.")

            # Check DnaMatch Need: Needs creation if Person exists but DnaMatch doesn't
            # Accessing existing_person_hint.dna_match will trigger load if expired/not loaded
            existing_dna_record = existing_person_hint.dna_match
            if existing_dna_record is None:
                logger.debug(f"{log_ref}: Existing Person found, but no DnaMatch record. Needs DNA creation.")
                create_dna_needed = True
            else:
                logger.debug(f"{log_ref}: Existing DnaMatch record found. No DNA creation needed.")
                create_dna_needed = False

            # --- MODIFIED FamilyTree Need Check ---
            db_in_my_tree = existing_person_hint.in_my_tree
            existing_tree_record = existing_person_hint.family_tree # Access relationship

            if match_in_my_tree and not db_in_my_tree:
                logger.debug(
                    f"{log_ref}: Status changed to 'in_my_tree'. Needs tree data fetch."
                )
                fetch_tree_data = True
            elif db_in_my_tree and not match_in_my_tree:
                 logger.warning(f"{log_ref}: Status changed FROM 'in_my_tree' to False. Skipping tree fetch, existing tree data might become stale.")
                 fetch_tree_data = False
            elif match_in_my_tree and db_in_my_tree and existing_tree_record is None:
                 # NEW Condition: Flag is True, but record is missing (inconsistency)
                 logger.warning(f"{log_ref}: Status 'in_my_tree' is True, but FamilyTree record is MISSING. Needs tree data fetch to fix.")
                 fetch_tree_data = True
            else:
                # Covers: Both False, or Both True and record exists.
                logger.debug(f"{log_ref}: 'in_my_tree' status unchanged ({match_in_my_tree}) or no tree creation needed. No tree data fetch triggered.")
                fetch_tree_data = False
            # --- END MODIFIED ---

        # --- Person Not Found by UUID ---
        else:
            logger.debug(f"{log_ref}: No existing person found by UUID.")
            existing_person_hint = None # Ensure it's None
            # If person doesn't exist, DNA match also doesn't exist
            create_dna_needed = True
            # If person doesn't exist, fetch tree data only if the flag is set
            fetch_tree_data = match_in_my_tree

    except SQLAlchemyError as e:
         logger.error(f"Database error during need_update check for {log_ref}: {e}", exc_info=True)
         return None, False, False # Return defaults on error
    except Exception as e:
         logger.error(f"Unexpected error during need_update check for {log_ref}: {e}", exc_info=True)
         return None, False, False

    # --- Final Decision Logging ---
    actions_needed = []
    if create_dna_needed: actions_needed.append("Create DNA Match")
    if fetch_tree_data: actions_needed.append("Fetch/Process Tree Data")

    if not actions_needed: final_decision = "No DNA/Tree actions needed."
    else: final_decision = " AND ".join(actions_needed)
    logger.debug(f"{log_ref}: Need Update Check Complete. Actions needed: {final_decision}")

    return existing_person_hint, create_dna_needed, fetch_tree_data
# --- End Modified Function ---


def create_or_update_person(
    session: Session, person_data: Dict[str, Any]
) -> Tuple[Optional[Person], Literal["created", "updated", "skipped", "error"]]:
    """
    Creates a new Person or updates an existing one based *primarily* on UUID.
    If UUID not found, attempts creation (will fail safely if profile_id UNIQUE constraint hit).
    Handles updates for existing records found by UUID, including profile_id changes.
    Fixes timezone comparison issue for last_logged_in (ignoring microseconds).
    Prevents overwriting existing birth_year with None.
    RELIES ON `from datetime import timezone` at top of file.
    """
    uuid_val = person_data.get("uuid")
    profile_id_val = person_data.get("profile_id") # Can be None
    username_val = person_data.get("username", "Unknown")

    if not uuid_val:
        logger.error(f"Cannot create/update person for ProfileID='{profile_id_val or 'N/A'}', User='{username_val}': UUID is missing.")
        return None, "error"

    log_ref = f"UUID={uuid_val} / ProfileID={profile_id_val or 'NULL'} / User='{username_val}'"
    existing_person = None
    updated = False
    person_id_for_logging = None

    try:
        # --- Revised Lookup Strategy: PRIORITIZE UUID ---
        existing_person = session.query(Person).filter(Person.uuid == uuid_val.upper()).first()

        if existing_person:
             person_id_for_logging = existing_person.id
             logger.debug(f"{log_ref}: Found existing Person ID {existing_person.id} by UUID.")

             # --- Update Logic for existing person found by UUID ---
             logger.debug(f"{log_ref}: Checking for updates on existing Person ID {existing_person.id}.")
             fields_to_update = {
                  "username": username_val,
                  "profile_id": profile_id_val.upper() if profile_id_val else None,
                  "in_my_tree": person_data.get("in_my_tree"),
                  "first_name": person_data.get("first_name"),
                  "last_logged_in": person_data.get("last_logged_in"),
                  "contactable": person_data.get("contactable"),
                  "birth_year": person_data.get("birth_year"), # Keep birth_year here
                  "administrator_profile_id": person_data.get("administrator_profile_id"),
                  "administrator_username": person_data.get("administrator_username"),
                  "gender": person_data.get("gender"),
                  "message_link": person_data.get("message_link"),
                  "status": "active"
             }

             for field, new_value in fields_to_update.items():
                  # Skip check if new value is None and field doesn't allow None (except booleans and specific nullable fields)
                  allow_none = field in ["profile_id", "administrator_profile_id", "administrator_username", "message_link", "birth_year", "last_logged_in", "gender"]
                  is_boolean_field = field in ["in_my_tree", "contactable"]

                  # *** MODIFICATION: Prevent overwriting birth_year with None ***
                  # If the field is birth_year and the new value is None, skip the update
                  # *unless* the current value is also None (or missing).
                  if field == "birth_year" and new_value is None:
                       current_value = getattr(existing_person, field, None)
                       if current_value is not None: # Don't overwrite an existing birth year with None
                            logger.debug(f"  Skipping update for {field}: New value is None, but existing value is '{current_value}'.")
                            continue
                       # If current_value is also None, proceed to standard comparison below (will result in no change)
                  # *** END MODIFICATION ***

                  # Standard skip check for other non-nullable fields
                  elif new_value is None and not allow_none and not is_boolean_field:
                      continue # Skip if new value is None for a field that must have a value

                  current_value = getattr(existing_person, field, None)
                  should_update = False

                  # --- Special handling for last_logged_in comparison ---
                  if field == "last_logged_in":
                        current_naive = None
                        new_naive = None
                        # Convert current DB value to naive UTC, truncated to seconds
                        if isinstance(current_value, datetime):
                             db_aware = current_value.tzinfo is not None and current_value.tzinfo.utcoffset(current_value) is not None
                             if db_aware:
                                 current_naive = current_value.astimezone(timezone.utc).replace(tzinfo=None, microsecond=0)
                             else: # Assume naive stored is UTC
                                 current_naive = current_value.replace(microsecond=0)
                        # Convert new incoming value to naive UTC, truncated to seconds
                        if isinstance(new_value, datetime):
                             # Assume new_value is always aware from API parsing
                             new_naive = new_value.astimezone(timezone.utc).replace(tzinfo=None, microsecond=0)

                        # Compare naive, truncated values
                        if current_naive != new_naive:
                             logger.debug(f"  Comparing {field}: Current Naive (DB): '{current_naive}', New Naive (API): '{new_naive}' -> Difference detected.")
                             should_update = True
                        # else: # Add else for clarity if needed
                        #      logger.debug(f"  Comparing {field}: Current Naive (DB): '{current_naive}', New Naive (API): '{new_naive}' -> No difference detected.")

                  # --- Special handling for username (case-insensitive) ---
                  elif field == "username":
                      current_str = str(current_value) if current_value is not None else ""
                      new_str = str(new_value) if new_value is not None else ""
                      if current_str.lower() != new_str.lower():
                           should_update = True

                  # --- Standard comparison for other fields ---
                  elif current_value != new_value:
                       should_update = True

                  # --- Perform update if needed ---
                  if should_update:
                      logger.debug(f"  Flagging update for {field}: '{current_value}' -> '{new_value}'")
                      setattr(existing_person, field, new_value)
                      updated = True # Set flag if *any* field needs update

             # --- Determine final status based on update flag ---
             if updated:
                  logger.debug(f"{log_ref}: Person ID {existing_person.id} requires update based on field checks.")
                  existing_person.updated_at = datetime.now() # Update timestamp if changed
                  return existing_person, "updated"
             else:
                  logger.debug(f"{log_ref}: Person ID {existing_person.id} requires no update based on field checks.")
                  return existing_person, "skipped"
        else:
            # --- UUID not found, proceed to CREATE ---
            logger.debug(f"{log_ref}: UUID not found. Calling create_person.")
            # Pass all relevant data collected so far
            new_person_id = create_person(session, person_data)
            if new_person_id > 0:
                 new_person_obj = session.get(Person, new_person_id)
                 if new_person_obj:
                      return new_person_obj, "created"
                 else:
                      logger.error(f"Failed to fetch newly created person with ID {new_person_id} for {log_ref}.")
                      return None, "error"
            else:
                 logger.error(f"create_person failed for {log_ref}.")
                 return None, "error"

    except IntegrityError as ie:
        session.rollback()
        logger.error(f"IntegrityError processing person {log_ref}: {ie}. Rolling back.", exc_info=False)
        return None, "error"
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"SQLAlchemyError processing person {log_ref}: {e}", exc_info=True)
        return None, "error"
    except NameError as ne: # Catch missing 'timezone'
         session.rollback()
         logger.critical(f"NameError processing person {log_ref}: {ne}. Ensure 'timezone' is imported from datetime.", exc_info=True)
         return None, "error"
    except TypeError as te:
         session.rollback()
         logger.critical(f"TypeError during person update comparison for {log_ref}: {te}", exc_info=True)
         return None, "error"
    except Exception as e:
        if session.is_active:
            session.rollback()
        logger.critical(f"Unexpected critical error processing person {log_ref}: {e}", exc_info=True)
        return None, "error"
# End create_or_update_person

def update_person(
    session: Session, profile_id: str, username: str, update_data: Dict[str, Any]
) -> bool:
    """Updates an existing person record. Uses profile_id and username for lookup.

    Args:
        session: The SQLAlchemy Session.
        profile_id: The profile ID of the person to update.
        username: The username of the person to update.
        update_data: A dictionary containing the fields to update.

    Returns:
        True if the update was successful, False otherwise.
    """
    try:
        person = session.query(Person).filter_by(profile_id=profile_id, username=username).first()
        if not person:
            logger.warning(f"update_person: Person with profile_id {profile_id} and username {username} not found.")
            return False

        # Update fields (whitelist approach for safety)
        updated = False # Track if any changes are made
        allowed_fields = [
             "uuid", "profile_id", "username", "administrator_profile_id",
             "administrator_username", "message_link", "in_my_tree", "status"
        ] # Define mutable fields
        for key, value in update_data.items():
            if key in allowed_fields and hasattr(person, key):
                current_value = getattr(person, key)
                if current_value != value: # Only update if value changed
                     setattr(person, key, value)
                     logger.debug(f"Updating {key} for Person ID {person.id} to {value}")
                     updated = True
            elif key not in allowed_fields:
                logger.warning(
                    f"Attempted to update non-allowed attribute '{key}' on Person ID {person.id}."
                )

        if updated:
             person.updated_at = datetime.now() # Update timestamp if changed
             session.flush() # Stage the update
             logger.info(f"Updated person with profile_id {profile_id} and username {username} (ID: {person.id}).")
             return True
        else:
             logger.debug(f"No update needed for person profile_id {profile_id} / username {username}.")
             return True # Return True even if no changes needed, as the record exists

    except IntegrityError as ie:
         session.rollback()
         logger.error(f"IntegrityError updating person {profile_id}/{username}: {ie}.", exc_info=True)
         return False
    except SQLAlchemyError as e:
        logger.error(f"Database error updating person {profile_id}/{username}: {e}", exc_info=True)
        session.rollback()
        return False
    except Exception as e: # Catch any other unexpected error
         logger.critical(f"Unexpected error in update_person for {profile_id}/{username}: {e}", exc_info=True)
         if session.is_active: session.rollback()
         return False
# End of update_person

# ----------------------------------------------------------------------
# Delete
# ----------------------------------------------------------------------

def delete_person(session: Session, profile_id: str, username: str) -> bool:
    """Deletes a person and associated records (using cascading deletes). Uses profile_id and username.

    Args:
        session: The SQLAlchemy Session.
        profile_id: The profile ID of the person to delete.
        username: The username of the person to delete.


    Returns:
        True if the deletion was successful, False otherwise.
    """
    try:
        person = session.query(Person).filter_by(profile_id=profile_id, username=username).first()
        if not person:
            logger.warning(f"delete_person: Person with profile_id {profile_id} and username {username} not found.")
            return False

        person_id = person.id # Get ID for logging before delete
        session.delete(person)  # Cascading delete will handle related records
        session.flush() # Execute delete in DB
        logger.info(f"Deleted person ID {person_id} (profile_id {profile_id}, username {username}) and related records.")
        return True

    except SQLAlchemyError as e:
        logger.error(f"Database error deleting person {profile_id}/{username}: {e}", exc_info=True)
        session.rollback()
        return False
    except Exception as e: # Catch any other unexpected error
         logger.critical(f"Unexpected error in delete_person for {profile_id}/{username}: {e}", exc_info=True)
         if session.is_active: session.rollback()
         return False
# End of delete_person

def delete_database(session_manager, db_path: Path, max_attempts=5): # Increased max_attempts to 5
    """
    Delete the database file (Path object) with retry and exponential backoff.
    More aggressive cleanup and error handling for file locks.
    """
    if not isinstance(db_path, Path):
        try:
            db_path = Path(db_path)
            logger.warning("Converted db_path string to Path object in delete_database.")
        except TypeError:
             logger.error(f"Cannot convert db_path {db_path} to Path object.")
             return # Cannot proceed

    logger.debug(f"Attempting to delete database file: {db_path}")
    last_error = None # Store last error

    for attempt in range(max_attempts):
        logger.debug(f"Delete attempt {attempt + 1}/{max_attempts}...")
        try:
            # 1. Aggressive Cleanup: Close connections, dispose engine, garbage collect
            logger.debug("Closing DB connections and disposing engine...")
            if session_manager:
                 session_manager.cls_db_conn() # Should dispose engine
            else:
                 logger.warning("No session_manager provided to delete_database.")

            # Multiple GC calls and longer sleep
            gc.collect()
            time.sleep(0.5) # Short sleep
            gc.collect()
            time.sleep(1.0 + attempt) # Increase sleep slightly with attempts

            # 2. Check existence and attempt deletion
            if db_path.exists():
                logger.debug(f"Attempting os.remove on {db_path}...")
                os.remove(db_path)
                # Verify deletion
                time.sleep(0.1) # Tiny pause before checking again
                if not db_path.exists():
                    logger.info(f"'{db_path}' deleted successfully.") # Use INFO for success
                    return  # SUCCESS
                else:
                     logger.warning(f"os.remove called, but file '{db_path}' still exists. Retrying.")
                     last_error = OSError(f"File still exists after os.remove attempt {attempt + 1}")
                     # Continue to retry logic below
            else:
                logger.info(f"Database '{db_path}' does not exist (or already deleted).")
                return  # SUCCESS (already gone)

        # Catch specific permission errors likely related to file locks
        except PermissionError as e:
             logger.warning(f"PermissionError deleting '{db_path}' (Attempt {attempt + 1}): {e}. File likely locked.")
             last_error = e
             # Continue to retry logic below
        except OSError as e:
            # Catch other OS errors, check for WinError 32 specifically
            if hasattr(e, 'winerror') and e.winerror == 32:
                 logger.warning(f"OSError (WinError 32) deleting '{db_path}' (Attempt {attempt + 1}): {e}. File likely locked.")
            else:
                 logger.error(f"OSError deleting '{db_path}' (Attempt {attempt + 1}): {e}", exc_info=True)
            last_error = e
            # Continue to retry logic below
        except Exception as e:
             # Catch any other unexpected errors
             logger.critical(f"Unexpected error during delete attempt {attempt + 1} for '{db_path}': {e}", exc_info=True)
             last_error = e
             # Continue to retry logic below

        # --- Retry Logic ---
        if attempt < max_attempts - 1:
            wait_time = 2 ** attempt # Exponential backoff
            logger.info(f"Waiting {wait_time} seconds before next delete attempt...")
            time.sleep(wait_time)
        else:
            # Last attempt failed
            logger.critical(f"Failed to delete database '{db_path}' after {max_attempts} attempts. Last error: {last_error or 'Unknown Error'}")
            # Re-raise the last caught exception to signal failure to caller
            if last_error:
                 raise last_error
            else:
                 raise OSError(f"Failed to delete {db_path} after max attempts.")

    # Should not be reached if max_attempts > 0
    logger.error(f"Exited delete_database loop unexpectedly for {db_path}.")
# End of delete_database

# ----------------------------------------------------------------------
# Backup and Recovery
#################################################################################

def backup_database(
    session_manager=None, # Keep session_manager arg for potential future use or remove if definitely unused
):
    """Backs up the database file specified in config_instance to the 'Data' folder."""
    db_path = config_instance.DATABASE_FILE  # This is now a Path object
    backup_dir = config_instance.DATA_DIR # Use DATA_DIR from config, also a Path object
    # backup_dir = Path("Data") # Or define directly if DATA_DIR isn't in config_instance

    # Ensure backup directory exists (makedirs works with Path)
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Construct backup path using Path's / operator
    backup_path = backup_dir / "ancestry_backup.db"

    try:
        # os.path.exists works with Path objects
        if db_path.exists():
            # shutil.copy2 works with Path objects
            shutil.copy2(db_path, backup_path)
            logger.info(f"Database backed up successfully to '{backup_path}'.") # Use INFO for success
        else:
            logger.warning(f"Database file '{db_path}' not found. No backup created.")
    except Exception as e:
        logger.error(f"Error backing up database from '{db_path}' to '{backup_path}': {e}", exc_info=True)
        # Re-raise the exception if backup failure should halt execution
        # raise
# End of backup_database

# ----------------------------------------------------------------------
# Standalone execution (for testing database setup)
#################################################################################

def main():
    """
    Main function for standalone database setup and testing.
    Uses Path object from config_instance for database file.
    """
    # Local import for standalone use
    # from logging_config import setup_logging # Commented out - already handled globally if imported
    from config import config_instance

    # --- Setup Logging (Assume logger is already configured if imported) ---
    # setup_logging might be called elsewhere, ensure logger exists
    global logger
    if 'logger' not in globals():
         # Fallback basic config if run truly standalone without main.py setup
         logging.basicConfig(level=logging.DEBUG)
         logger = logging.getLogger(__name__) # Use module name logger

    print("")
    logger.info("Starting database.py in standalone DEBUG mode.")

    db_path_obj = config_instance.DATABASE_FILE # Get Path object
    db_path_str = str(db_path_obj) # Convert to string for SQLAlchemy create_engine URI

    logger.info(f"Using database file: {db_path_str}")
    engine = None # Initialize engine to None
    conn_pool = None # Initialize conn_pool to None

    try:
        # Use string path for the connection URI
        engine = create_engine(f'sqlite:///{db_path_str}', echo=False)

        # --- Add PRAGMA foreign_keys=ON listener ---
        @event.listens_for(engine, "connect")
        def connect(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
            finally:
                 cursor.close()
        # --- End Listener ---

        Base.metadata.create_all(engine) # Create tables if they don't exist
        logger.info(f"Database tables created/verified in: {db_path_str}")

        # --- Test Connection Pool ---
        # Pass string path to ConnectionPool
        pool_size = config_instance.DB_POOL_SIZE
        conn_pool = ConnectionPool(db_path_str, pool_size=pool_size)
        session = conn_pool.get_session()
        if session:
            logger.info("Connection pool session obtained successfully.")
            # Perform a simple query to test
            try:
                 result = session.query(func.count(Person.id)).scalar()
                 logger.info(f"Test query successful (found {result} people).")
            except Exception as test_e:
                 logger.error(f"Test query failed: {test_e}", exc_info=True)
            finally:
                 conn_pool.return_session(session)
        else:
            logger.error("Failed to obtain connection pool session.")

    except SQLAlchemyError as db_e:
         logger.critical(f"Database setup/connection error: {db_e}", exc_info=True)
    except Exception as e:
         logger.critical(f"Unexpected error in database.py main: {e}", exc_info=True)
    finally:
        # Cleanly close pool and dispose engine if they exist
        if conn_pool: # Check if conn_pool was successfully created
             conn_pool.clse_all_sess()
        # Check if engine was successfully created before disposing
        elif engine: # If pool failed but engine exists
             engine.dispose()
             logger.debug("SQLAlchemy engine disposed (pool cleanup skipped/failed).")

        logger.info("Database setup and test completed.")
# End of main

if __name__ == "__main__":
    main()