#!/usr/bin/env python3

# database.py

# Imports
import os
import shutil
from typing import List, Dict, Optional, Tuple, Any, Type, Literal
from pathlib import Path
from datetime import datetime
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
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
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
    uuid = Column(String, nullable=True, unique=True, index=True)
    profile_id = Column(String,  unique=True, nullable=True, index=True)
    username = Column(String,  unique=False, nullable=False)
    first_name = Column(String, nullable=True)
    gender = Column(String(1), nullable=True) # Store 'f', 'm', or None
    birth_year = Column(Integer, nullable=True)
    message_link = Column(String, unique=False, nullable=True)
    in_my_tree = Column(Boolean, default=False)
    contactable =  Column(Boolean, default=False)
    last_logged_in = Column(DateTime, nullable=True, index=True)
    administrator_profile_id = Column(String, nullable=True, index=True)
    administrator_username = Column(String, nullable=True)
    status = Column(String, default="active", nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )

    # Relationships
    family_tree = relationship("FamilyTree", back_populates="person", uselist=False, cascade="all, delete, delete-orphan")
    dna_match = relationship("DnaMatch", back_populates="person", uselist=False, cascade="all, delete, delete-orphan")
    inbox_status = relationship("InboxStatus", back_populates="person", uselist=False, cascade="all, delete, delete-orphan", foreign_keys="InboxStatus.people_id") # Corrected foreign_keys usage
    message_history = relationship("MessageHistory", back_populates="person", cascade="all, delete, delete-orphan")

    __table_args__ = (
        UniqueConstraint('profile_id', 'username', name='uq_people_profile_id_username'),
    )
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

def create_person(session: Session, match_data: Dict[str, Any]) -> int:
    """Creates a new person record in the database.

    Args:
        session: The SQLAlchemy Session.
        match_data: A dictionary containing the person's data. Must include:
                    'profile_id', 'username'.
        Can optionally include: 'message_link', 'administrator_profile_id', 'administrator_username', 'uuid', 'in_my_tree'.

    Returns:
        The ID of the newly created person, or 0 if creation failed.
    """
    try:
        # --- Enhanced Validation ---
        required_keys = ("profile_id", "username")
        if not all(key in match_data and match_data[key] is not None for key in required_keys):
            logger.warning(f"Missing required non-null data for person creation: Needs profile_id and username. Data: {match_data}")
            return 0

        profile_id = match_data["profile_id"]
        username = match_data["username"]

        # Check if person already exists based on profile_id AND username
        existing_person = (
            session.query(Person)
            .filter_by(profile_id=profile_id, username=username)
            .first()
        )
        if existing_person:
            logger.info(f"Person with profile_id {profile_id} and username {username} already exists (ID: {existing_person.id}). Skipping creation.")
            return int(existing_person.id) if existing_person.id is not None else 0

        # Create a new Person object using the provided data.
        new_person = Person(
            uuid=match_data.get("uuid"), # Add uuid if available
            profile_id=profile_id,
            username=username,
            administrator_profile_id=match_data.get("administrator_profile_id"), # Add admin id
            administrator_username=match_data.get("administrator_username"), # Add admin name
            message_link=match_data.get("message_link"),
            in_my_tree=bool(match_data.get("in_my_tree", False)), # Add in_my_tree
            status="active",
        )

        session.add(new_person)
        session.flush() # Flush to get the ID and check constraints

        if new_person.id is None:
            logger.error( # Use ERROR for critical failure
                f"ID not assigned after flush for person {username} (profile_id: {profile_id})! Data: {match_data}"
            )
            session.rollback() # Rollback if ID assignment failed
            return 0

        logger.debug(f"Created Person record ID {new_person.id} for {username}.")
        return int(new_person.id)

    except IntegrityError as ie:
         # Handles potential unique constraint violations if flush() detects them
         session.rollback()
         logger.error(f"IntegrityError creating person {match_data.get('username')}: {ie}. ProfileID/Username likely exists.", exc_info=True)
         return 0
    except SQLAlchemyError as e:
        logger.error(f"Database error creating person {match_data.get('username')}: {e}", exc_info=True)
        session.rollback()
        return 0
    except Exception as e: # Catch any other unexpected error
         logger.critical(f"Unexpected error in create_person for {match_data.get('username')}: {e}", exc_info=True)
         if session.is_active: session.rollback()
         return 0
# End of create_person


def create_dna_match(session: Session, match_data: Dict[str, Any]) -> int:
    """Creates a new DNA match record, handling existing entries and validation.

    Args:
        session: The SQLAlchemy Session.
        match_data: Dictionary with 'compare_link', 'cM_DNA',
                    'predicted_relationship', and 'people_id'.

    Returns:
        The ID of the created/existing DNA match, or 0 on failure.
    """
    # 1. Data Validation (BEFORE any database interaction)
    required_keys = ("compare_link", "cM_DNA", "predicted_relationship", "people_id")
    if not all(key in match_data for key in required_keys):
        logger.warning(f"create_dna_match: Missing required data: {match_data}")
        return 0
    people_id = match_data.get("people_id")
    if not isinstance(people_id, int) or people_id <= 0:
         logger.warning(f"create_dna_match: Invalid people_id: {people_id}")
         return 0
    try:
        cm_dna_val = int(match_data["cM_DNA"])
        if cm_dna_val < 0: raise ValueError("cM_DNA cannot be negative")
    except (TypeError, ValueError):
        logger.warning(f"create_dna_match: Invalid cM_DNA value: {match_data.get('cM_DNA')}")
        return 0

    # 2. Check for Existing Entry (using people_id which should be unique now)
    try:
        existing_dna_match = (
            session.query(DnaMatch)
            .filter_by(people_id=people_id) # Filter by the unique people_id
            .first()
        )
        if existing_dna_match:
            logger.info(
                f"DNA match exists for people_id {people_id} (ID: {existing_dna_match.id}). Skipping creation."
            )
            # Optionally update existing record here if needed, or just return ID
            return int(existing_dna_match.id) if existing_dna_match.id is not None and isinstance(existing_dna_match.id, int) else 0

        # 3. Create and Add New Entry
        new_dna_match = DnaMatch(
            people_id=people_id, # Use validated people_id
            compare_link=match_data["compare_link"],
            cM_DNA=cm_dna_val, # Use validated cM value
            predicted_relationship=match_data["predicted_relationship"],
        )

        session.add(new_dna_match)
        session.flush()  # Get the ID and check constraints

        # --- CRITICAL CHECK: Ensure ID is assigned after flush ---
        if new_dna_match.id is None:
            logger.error( # Use ERROR
                f"ID not assigned after flush for DNA Match (people_id: {people_id})! Data: {match_data}"
            )
            session.rollback()
            return 0

        logger.debug(f"Created new DNA match (ID: {new_dna_match.id}) for people_id {people_id}.")
        return int(new_dna_match.id) if new_dna_match.id is not None and isinstance(new_dna_match.id, int) else 0

    except IntegrityError as ie:
         # This could happen if somehow a duplicate people_id is attempted despite the check
         session.rollback()
         logger.error(f"IntegrityError creating DNA match for people_id {people_id}: {ie}.", exc_info=True)
         return 0
    except SQLAlchemyError as e:
        logger.error(f"Database error creating DNA match for people_id {people_id}: {e}", exc_info=True)
        session.rollback()
        return 0
    except Exception as e: # Catch any other unexpected error
         logger.critical(f"Unexpected error in create_dna_match for people_id {people_id}: {e}", exc_info=True)
         if session.is_active: session.rollback()
         return 0
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
         session.rollback()
         # This log now clearly indicates an issue with the constructor call itself
         logger.critical(f"TypeError creating FamilyTree object for {log_ref}: {te}. Check arguments passed to FamilyTree().", exc_info=True)
         logger.error(f"Data keys attempted: {list(tree_data.keys())}") # Log keys that were available
         return "error"
    except IntegrityError as ie:
        session.rollback()
        logger.error(f"IntegrityError creating FamilyTree for {log_ref}: {ie}", exc_info=False)
        return "error"
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"SQLAlchemyError creating FamilyTree for {log_ref}: {e}", exc_info=True)
        return "error"
    except Exception as e:
        session.rollback()
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
             # .options(joinedload(Person.dna_match), joinedload(Person.family_tree)) # Example - Use joinedload from sqlalchemy.orm
             .options(relationship(Person.dna_match), relationship(Person.family_tree)) # Use relationship() for options
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

def create_or_update_person(
    session: Session, person_data: Dict[str, Any]
) -> Tuple[Optional[Person], Literal["created", "updated", "skipped", "error"]]:
    """
    Creates a new Person or updates an existing one based on UUID and Profile ID.
    Handles potential conflicts where a profile ID might be associated with multiple UUIDs.
    Prioritizes updating based on profile_id if found. If profile_id is None in input,
    attempts lookup via UUID. Handles UUID/ProfileID mismatches during update.
    """
    uuid_val = person_data.get("uuid")
    profile_id_val = person_data.get("profile_id") # Can be None for managed kits
    username_val = person_data.get("username", "Unknown")

    # UUID is essential for linking DNA matches, Profile ID less so if managed
    if not uuid_val:
        logger.error(f"Cannot create/update person for ProfileID='{profile_id_val or 'N/A'}', User='{username_val}': UUID is missing.")
        return None, "error"

    log_ref = f"UUID={uuid_val} / ProfileID={profile_id_val or 'NULL'} / User='{username_val}'"
    existing_person = None
    updated = False
    person_id_for_logging = None

    try:
        # --- Revised Lookup Strategy ---
        # 1. Prioritize lookup by UUID, as it's guaranteed from the match list.
        existing_person = session.query(Person).filter(Person.uuid == uuid_val.upper()).first()

        if existing_person:
            person_id_for_logging = existing_person.id
            logger.debug(f"{log_ref}: Found existing Person ID {existing_person.id} by UUID.")

            # Check for Profile ID consistency
            # Case A: Input has profile_id, DB also has one, but they differ.
            if profile_id_val and existing_person.profile_id and existing_person.profile_id.upper() != profile_id_val.upper():
                 logger.warning(f"{log_ref}: UUID matches existing Person ID {existing_person.id}, but Profile IDs differ (Existing='{existing_person.profile_id}', New='{profile_id_val}'). Updating Profile ID.")
                 existing_person.profile_id = profile_id_val.upper()
                 updated = True
            # Case B: Input has profile_id, DB does NOT have one. Add it.
            elif profile_id_val and not existing_person.profile_id:
                 logger.debug(f"{log_ref}: UUID matches existing Person ID {existing_person.id}. Adding missing Profile ID '{profile_id_val}'.")
                 existing_person.profile_id = profile_id_val.upper()
                 updated = True
            # Case C: Input has NO profile_id (managed kit), DB *does* have one. Set DB to NULL.
            elif not profile_id_val and existing_person.profile_id:
                 logger.warning(f"{log_ref}: UUID matches existing Person ID {existing_person.id}. Input indicates managed kit (ProfileID=NULL), but DB has ProfileID '{existing_person.profile_id}'. Setting DB ProfileID to NULL.")
                 existing_person.profile_id = None
                 updated = True
            # Case D: Both input and DB have profile_id and they match, or both are None. No change needed for profile_id itself.

        # 2. If not found by UUID, check if the input profile_id exists (potential conflict/reattach)
        elif profile_id_val:
             person_by_profile = session.query(Person).filter(Person.profile_id == profile_id_val.upper()).first()
             if person_by_profile:
                  # Found by profile_id, but not by the current UUID. This implies the profile_id
                  # might be associated with a *different* UUID (or the UUID was missing previously).
                  person_id_for_logging = person_by_profile.id
                  logger.warning(f"{log_ref}: Found Person ID {person_by_profile.id} by Profile ID, but UUID mismatch (Existing UUID='{person_by_profile.uuid or 'NULL'}'). Updating existing record with new UUID '{uuid_val}'.")
                  existing_person = person_by_profile # Target this existing record for update
                  existing_person.uuid = uuid_val.upper() # Update the UUID
                  updated = True
             # else: No existing record by UUID or Profile ID, proceed to create.

        # --- END Revised Lookup Strategy ---


        # 3. Decide: Update Existing or Create New
        if existing_person:
             # --- Update Logic ---
             logger.debug(f"{log_ref}: Updating existing Person ID {existing_person.id}.")
             # Define fields available for update
             fields_to_update = {
                  "username": username_val,
                  "profile_id": profile_id_val.upper() if profile_id_val else None, # Reflects input accurately now
                  "uuid": uuid_val.upper(), # Already updated if necessary above
                  "in_my_tree": person_data.get("in_my_tree"),
                  "first_name": person_data.get("first_name"),
                  "last_logged_in": person_data.get("last_logged_in"),
                  "contactable": person_data.get("contactable"),
                  "birth_year": person_data.get("birth_year"),
                  "administrator_profile_id": person_data.get("administrator_profile_id"),
                  "administrator_username": person_data.get("administrator_username"),
                  "gender": person_data.get("gender"),
                  "message_link": person_data.get("message_link"),
                  "status": "active" # Ensure status is active on update
             }

             for field, new_value in fields_to_update.items():
                  current_value = getattr(existing_person, field, None)
                  # Check if update is needed (value differs and new value is not None)
                  # For boolean fields (in_my_tree, contactable), allow update even if new_value is False
                  is_boolean_field = field in ["in_my_tree", "contactable"]
                  should_update = False

                  if field == "username": # Case-insensitive compare for username
                      if isinstance(current_value, str) and isinstance(new_value, str):
                          if current_value.lower() != new_value.lower():
                               should_update = True
                      elif current_value != new_value: # Handle None comparison
                           should_update = True
                  elif current_value != new_value:
                       should_update = True

                  # Apply the update if needed and allowed
                  if should_update and (new_value is not None or is_boolean_field):
                      # logger.debug(f"  Updating {field}: '{current_value}' -> '{new_value}'")
                      setattr(existing_person, field, new_value)
                      updated = True # Mark that an update occurred

             if updated:
                  logger.debug(f"{log_ref}: Person ID {existing_person.id} flagged for update.")
                  return existing_person, "updated"
             else:
                  logger.debug(f"{log_ref}: Person ID {existing_person.id} requires no update.")
                  return existing_person, "skipped"
        else:
             # --- Create New Logic ---
             logger.debug(f"{log_ref}: Creating new person record.")
             new_person = Person(
                  uuid=uuid_val.upper(), # UUID is guaranteed present here
                  profile_id=profile_id_val.upper() if profile_id_val else None, # Can be None
                  username=username_val,
                  first_name=person_data.get("first_name"),
                  gender=person_data.get("gender"),
                  birth_year=person_data.get("birth_year"),
                  message_link=person_data.get("message_link"),
                  in_my_tree=person_data.get("in_my_tree", False),
                  contactable=person_data.get("contactable", True),
                  last_logged_in=person_data.get("last_logged_in"),
                  administrator_profile_id=person_data.get("administrator_profile_id"),
                  administrator_username=person_data.get("administrator_username"),
                  status="active",
             )
             session.add(new_person)
             session.flush() # Flush to get the new ID for logging
             person_id_for_logging = new_person.id
             logger.debug(f"{log_ref}: Created new Person record ID {person_id_for_logging}.")
             return new_person, "created"

    except IntegrityError as ie:
        session.rollback()
        logger.error(f"IntegrityError processing person {log_ref}: {ie}. Rolling back.", exc_info=False)
        # Try to find which constraint failed and log the existing record
        existing_by_profile = None
        existing_by_uuid = None
        if profile_id_val: existing_by_profile = session.query(Person).filter(Person.profile_id == profile_id_val.upper()).first()
        if uuid_val: existing_by_uuid = session.query(Person).filter(Person.uuid == uuid_val.upper()).first()

        if "people.profile_id" in str(ie) and existing_by_profile:
             logger.warning(f"  UNIQUE constraint failed on profile_id. Existing record: ID={existing_by_profile.id}, UUID={existing_by_profile.uuid}, User={existing_by_profile.username}")
        elif "people.uuid" in str(ie) and existing_by_uuid:
             logger.warning(f"  UNIQUE constraint failed on uuid. Existing record: ID={existing_by_uuid.id}, ProfileID={existing_by_uuid.profile_id}, User={existing_by_uuid.username}")
        else:
             logger.error(f"  IntegrityError details: {ie.orig}")
        logger.error(f"  Attempted Data: {person_data}")
        return None, "error"
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"SQLAlchemyError processing person {log_ref}: {e}", exc_info=True)
        return None, "error"
    except Exception as e:
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