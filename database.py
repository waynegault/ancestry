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
    Enum as SQLEnum, 
    Index,
    func,
    Float,
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    relationship,
    Session,
    joinedload,
)
from sqlalchemy.exc import SQLAlchemyError, IntegrityError  # Added IntegrityError
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
    AUTHOR = "AUTHOR"
    RECIPIENT = "RECIPIENT"
# end of class RoleType

class InboxStatus(Base):
    __tablename__ = "inbox_status"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String, nullable=True)
    people_id = Column(Integer, ForeignKey("people.id"), nullable=False, index=True)
    my_role = Column(SQLEnum(RoleType), nullable=False, name="my_role") # Use SQLEnum here
    last_message = Column(String, nullable=True)
    last_message_timestamp = Column(
        DateTime, nullable=True, index=True
    ) # Add index=True
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    last_updated = Column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )
    person = relationship("Person", back_populates="inbox_status")
    __table_args__ = (
        Index(
            "ix_inbox_status_people_id_timestamp", "people_id", "last_message_timestamp"
        ),
    )
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
    message_type_id = Column(Integer, ForeignKey("message_types.id"), nullable=False)
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
    people_id = Column(
        Integer, ForeignKey("people.id"), unique=True, nullable=False, index=True
    )
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


class FamilyTree(Base):  # Point 8 Refinement
    __tablename__ = "family_tree"
    id = Column(Integer, primary_key=True)
    people_id = Column(
        Integer, ForeignKey("people.id"), unique=True, nullable=False, index=True
    )
    cfpid = Column(String, unique=True, nullable=True)
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
    uuid = Column(String, nullable=True, unique=True, index=True)  # Keep UUID unique
    # --- RE-ADD unique=True to profile_id ---
    profile_id = Column(String, unique=True, nullable=True, index=True)
    # --- END RE-ADD ---
    username = Column(String, unique=False, nullable=False)
    first_name = Column(String, nullable=True)
    gender = Column(String(1), nullable=True)  # Store 'f', 'm', or None
    birth_year = Column(Integer, nullable=True)
    message_link = Column(String, unique=False, nullable=True)
    in_my_tree = Column(Boolean, default=False)
    contactable = Column(Boolean, default=False)
    last_logged_in = Column(DateTime, nullable=True, index=True)  # Store as naive UTC
    administrator_profile_id = Column(String, nullable=True, index=True)
    administrator_username = Column(String, nullable=True)
    status = Column(String, default="active", nullable=False)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )

    # Relationships (remain the same)
    family_tree = relationship(
        "FamilyTree",
        back_populates="person",
        uselist=False,
        cascade="all, delete, delete-orphan",
    )
    dna_match = relationship(
        "DnaMatch",
        back_populates="person",
        uselist=False,
        cascade="all, delete, delete-orphan",
    )
    inbox_status = relationship(
        "InboxStatus",
        back_populates="person",
        uselist=False,
        cascade="all, delete, delete-orphan",
        foreign_keys="InboxStatus.people_id",
    )  # Corrected foreign_keys usage
    message_history = relationship(
        "MessageHistory", back_populates="person", cascade="all, delete, delete-orphan"
    )

    # If you prefer SQLAlchemy checks, uncomment this, otherwise DB unique handles it
    # __table_args__ = (
    #     UniqueConstraint('profile_id', name='uq_people_profile_id'),
    #     UniqueConstraint('uuid', name='uq_people_uuid') # Ensure UUID constraint name if needed
    # )
# End of class Person


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
        logger.error(
            f"Exception within db_transn block: {e}. Rolling back.", exc_info=True
        )
        # Attempt rollback only if session is still active
        if session and session.is_active:
            session.rollback()
        # Re-raise the original exception after rollback attempt
        raise
    finally:
        pass


# end db_transn

# ==================== CRUD OPERATIONS ====================


# ----------------------------------------------------------------------
# Create/ Insert
# ----------------------------------------------------------------------


def create_person(session: Session, person_data: Dict[str, Any]) -> int:
    """
    V1.1 REVISED: Creates a new person record in the database.
    - Checks for existing profile_id (case-insensitive) before insertion.
    - Ensures profile_id and uuid are stored uppercase.
    """
    required_keys = ("username", "uuid")
    if not all(
        key in person_data and person_data[key] is not None for key in required_keys
    ):
        logger.warning(
            f"Missing required non-null data for person creation: Needs username and uuid. Data: {person_data}"
        )
        return 0

    profile_id_raw = person_data.get("profile_id") # Can be None
    # --- Convert to uppercase EARLY for check and insert ---
    profile_id_upper = profile_id_raw.upper() if profile_id_raw else None
    uuid_upper = str(person_data["uuid"]).upper() if person_data.get("uuid") else None
    # --- End Uppercase Conversion ---

    username = person_data["username"]
    log_ref = f"UUID={uuid_upper or 'NULL'} / ProfileID={profile_id_upper or 'NULL'} / User='{username}'"

    try:
        # --- Explicit Check for Profile ID Conflict BEFORE insert ---
        if profile_id_upper: # Check using the uppercased version
            existing_by_profile = (
                session.query(Person)
                .filter(Person.profile_id == profile_id_upper)
                .first()
            )
            if existing_by_profile:
                logger.error(
                    f"create_person FAILED for {log_ref}: Profile ID '{profile_id_upper}' already exists for Person ID {existing_by_profile.id} (UUID: {existing_by_profile.uuid}). Cannot create duplicate profile_id."
                )
                return 0 # Fail creation due to profile_id uniqueness requirement
        # --- End Profile ID Check ---

        # Proceed with creation if profile_id is NULL or not conflicting
        logger.debug(f"Proceeding with Person creation for {log_ref}.")
        new_person = Person(
            uuid=uuid_upper, # Store uppercase UUID
            profile_id=profile_id_upper, # Store uppercase Profile ID or None
            username=username,
            administrator_profile_id=(person_data.get("administrator_profile_id").upper() if person_data.get("administrator_profile_id") else None), # Uppercase Admin ID
            administrator_username=person_data.get("administrator_username"),
            message_link=person_data.get("message_link"),
            in_my_tree=bool(person_data.get("in_my_tree", False)),
            status="active",
            first_name=person_data.get("first_name"),
            gender=person_data.get("gender"),
            birth_year=person_data.get("birth_year"),
            contactable=person_data.get("contactable", True),
            last_logged_in=person_data.get("last_logged_in"),
        )

        session.add(new_person)
        session.flush() # Flush to get the ID and check constraints (like UUID UNIQUE)

        if new_person.id is None:
            logger.error(
                f"ID not assigned after flush for person {log_ref}! Data: {person_data}"
            )
            session.rollback()
            return 0

        logger.debug(f"Created Person record ID {new_person.id} for {log_ref}.")
        return int(new_person.id)

    except IntegrityError as ie:
        session.rollback()
        logger.error(
            f"IntegrityError creating person {log_ref}: {ie}. UUID likely exists.",
            exc_info=False,
        )
        return 0
    except SQLAlchemyError as e:
        logger.error(f"Database error creating person {log_ref}: {e}", exc_info=True)
        if session.is_active:
            session.rollback()
        return 0
    except Exception as e:
        logger.critical(
            f"Unexpected error in create_person for {log_ref}: {e}", exc_info=True
        )
        if session.is_active:
            session.rollback()
        return 0
# End of create_person


def create_dna_match(
    session: Session, match_data: Dict[str, Any]
) -> Literal["created", "skipped", "error"]:
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
    log_ref = f"PersonID={people_id}, KitUUID={match_data.get('uuid', 'N/A')}"  # Add UUID for context if available

    # --- 1. Basic Validation ---
    if not people_id or not isinstance(people_id, int) or people_id <= 0:
        logger.error(f"create_dna_match: Invalid or missing people_id for {log_ref}.")
        return "error"

    required_keys = ("compare_link", "cM_DNA", "predicted_relationship")
    if not all(
        key in match_data and match_data[key] is not None for key in required_keys
    ):
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
            return None  # None is valid
        try:
            if isinstance(value, str) and not value.replace(".", "", 1).isdigit():
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
        "longest_shared_segment",
        match_data.get("longest_shared_segment"),
        allow_float=True,
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
                cM_DNA=cm_dna_val,  # Use validated int value
                predicted_relationship=match_data["predicted_relationship"],
                # --- INCLUDE ALL OTHER FIELDS ---
                shared_segments=shared_segments_val,
                longest_shared_segment=longest_segment_val,
                meiosis=meiosis_val,
                from_my_fathers_side=bool(
                    match_data.get("from_my_fathers_side", False)
                ),  # Ensure boolean
                from_my_mothers_side=bool(
                    match_data.get("from_my_mothers_side", False)
                ),  # Ensure boolean
                # created_at/updated_at have defaults
            )
            session.add(new_dna_match)
            # No flush here, commit happens in the calling function (_do_match)
            logger.debug(f"Staged new 'dna_match' record for creation: {log_ref}")
            return "created"

    except IntegrityError as ie:  # Should be rare now due to explicit check
        session.rollback()  # Rollback immediately on integrity error
        logger.error(
            f"IntegrityError in create_dna_match for {log_ref}: {ie}. Likely concurrent creation.",
            exc_info=False,  # Less verbose for integrity error
        )
        # Double-check existence after rollback, in case of race condition
        existing = session.query(DnaMatch).filter_by(people_id=people_id).first()
        if existing:
            logger.warning(
                f"Found existing DnaMatch for {log_ref} after IntegrityError, returning 'skipped'."
            )
            return "skipped"
        return "error"  # Return error if still not found after rollback
    except SQLAlchemyError as e:
        # Don't rollback here; let the calling transaction handle it
        logger.error(
            f"Database error in create_dna_match for {log_ref}: {e}", exc_info=True
        )
        return "error"
    except Exception as e:  # Catch any other unexpected error
        logger.error(
            f"Unexpected error in create_dna_match for {log_ref}: {e}", exc_info=True
        )
        return "error"
# End of create_dna_match


def create_family_tree(
    session: Session, tree_data: Dict[str, Any]
) -> Literal["created", "updated", "skipped", "error"]:
    """
    V13 REVISED: Creates or updates a FamilyTree record.
    - If record exists, checks if key fields have changed before updating.
    - Uses only known valid keys from tree_data.
    """
    people_id = tree_data.get("people_id")
    if not people_id:
        logger.error("Cannot create/update FamilyTree: 'people_id' missing.")
        return "error"

    cfpid_val = tree_data.get("cfpid")  # Can be None
    log_ref = f"PersonID={people_id}, CFPID={cfpid_val or 'N/A'}"
    updated = False  # Flag for updates

    try:
        existing_tree = session.query(FamilyTree).filter_by(people_id=people_id).first()

        # Prepare data for creation or update, using only valid keys
        valid_tree_args = {
            "people_id": people_id,
            "cfpid": cfpid_val,
            "person_name_in_tree": tree_data.get("person_name_in_tree"),
            "facts_link": tree_data.get("facts_link"),
            "view_in_tree_link": tree_data.get("view_in_tree_link"),
            "actual_relationship": tree_data.get("actual_relationship"),
            "relationship_path": tree_data.get("relationship_path"),
        }
        # Remove keys with None values if desired, although SQLAlchemy handles them
        # valid_tree_args = {k: v for k, v in valid_tree_args.items() if v is not None}

        if existing_tree:
            logger.debug(f"Checking for updates to existing FamilyTree for {log_ref}")
            # Compare key fields to see if an update is needed
            fields_to_check = [
                "cfpid",
                "person_name_in_tree",
                "facts_link",
                "view_in_tree_link",
                "actual_relationship",
                "relationship_path",
            ]
            for field in fields_to_check:
                new_value = valid_tree_args.get(field)
                old_value = getattr(existing_tree, field, None)
                if new_value != old_value:
                    logger.debug(
                        f"  Updating FamilyTree field '{field}' for {log_ref}: '{old_value}' -> '{new_value}'"
                    )
                    setattr(existing_tree, field, new_value)
                    updated = True

            if updated:
                existing_tree.updated_at = datetime.now()
                logger.debug(f"Staged update for FamilyTree record {log_ref}")
                return "updated"
            else:
                logger.debug(
                    f"No update needed for existing FamilyTree record {log_ref}"
                )
                return "skipped"
        else:
            # Create new record
            logger.debug(f"Creating new FamilyTree record for {log_ref}")
            new_tree = FamilyTree(**valid_tree_args)
            session.add(new_tree)
            logger.debug(f"Staged new FamilyTree record for {log_ref}")
            return "created"

    except TypeError as te:
        logger.critical(
            f"TypeError creating/updating FamilyTree for {log_ref}: {te}.",
            exc_info=True,
        )
        logger.error(f"Data keys provided: {list(tree_data.keys())}")
        return "error"
    except IntegrityError as ie:
        logger.error(
            f"IntegrityError creating/updating FamilyTree for {log_ref}: {ie}",
            exc_info=False,
        )
        return "error"
    except SQLAlchemyError as e:
        logger.error(
            f"SQLAlchemyError creating/updating FamilyTree for {log_ref}: {e}",
            exc_info=True,
        )
        return "error"
    except Exception as e:
        logger.critical(
            f"Unexpected error in create_family_tree for {log_ref}: {e}", exc_info=True
        )
        return "error"
# End create_family_tree

# ----------------------------------------------------------------------
# Retrieve
# ----------------------------------------------------------------------


def get_person_by_profile_id_and_username(
    session: Session, profile_id: str, username: str
) -> Optional[Person]:
    """Retrieves a Person by profile_id AND username."""
    if not profile_id or not username:
        logger.warning(
            "get_person_by_profile_id_and_username: profile_id and username are required."
        )
        return None
    try:
        return (
            session.query(Person)
            .filter_by(profile_id=profile_id.upper(), username=username) # Ensure uppercase comparison
            .first()
        )
    except Exception as e:
        logger.error(
            f"Error retrieving person by profile_id/username: {e}", exc_info=True
        )
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
        person = session.query(Person).filter_by(profile_id=profile_id.upper()).first() # Ensure uppercase comparison
        return person
    except Exception as e:
        logger.error(
            f"Error retrieving person by profile_id '{profile_id}': {e}", exc_info=True
        )
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
        logger.warning(
            "get_person_and_dna_match: profile_id and username are required."
        )
        return None, None  # Return None tuple if input is invalid

    try:
        # Eager load dna_match to avoid separate query
        person = (
            session.query(Person)
            .options(
                joinedload(Person.dna_match) # Use joinedload directly
            )
            .filter_by(profile_id=profile_id.upper(), username=username) # Ensure uppercase comparison
            .first()
        )
        if person:
            # dna_match is already loaded due to options() if it exists
            dna_match = person.dna_match
            return person, dna_match
        else:
            return None, None  # Person not found
    except Exception as e:
        logger.error(
            f"Error retrieving person/DNA match for profile {profile_id}/{username}: {e}",
            exc_info=True,
        )
        return None, None  # Return None tuple on error
# end of get_person_and_dna_match


def find_existing_person(
    session: Session, identifier_data: Dict[str, Any]
) -> Optional[Person]:
    """
    Finds an existing person based on available unique identifiers (uuid or profile_id).
    Prioritizes UUID. If multiple matches on profile_id, uses username for disambiguation.
    Returns the Person object or None if not found or cannot be reliably identified.
    """
    person_uuid = identifier_data.get("uuid")
    person_profile_id = identifier_data.get("profile_id")
    person_username = identifier_data.get("username")  # Get username for disambiguation

    person = None
    # Create a more informative log reference
    log_parts = []
    if person_uuid:
        log_parts.append(f"UUID='{person_uuid}'")
    if person_profile_id:
        log_parts.append(f"ProfileID='{person_profile_id}'")
    if person_username:
        log_parts.append(f"User='{person_username}'")
    log_ref = " / ".join(log_parts)

    try:
        # 1. Prioritize UUID lookup
        if person_uuid:
            uuid_upper = str(person_uuid).upper()
            # Query using the Person model and filter
            person = session.query(Person).filter(Person.uuid == uuid_upper).first()
            if person:
                logger.debug(
                    f"Found existing person by UUID: {uuid_upper} (ID: {person.id})"
                )
                return person

        # 2. Fallback to Profile ID
        # Check 'person is None' explicitly here
        if person is None and person_profile_id:
            profile_id_upper = str(person_profile_id).upper()
            # Query using the Person model and filter, get all potential matches
            potential_matches = (
                session.query(Person)
                .filter(Person.profile_id == profile_id_upper)
                .all()
            )

            if not potential_matches:
                logger.debug(f"No person found by Profile ID: {profile_id_upper}.")
                # Continue to final return None below

            elif len(potential_matches) == 1:
                person = potential_matches[0]
                logger.debug(
                    f"Found unique person by Profile ID: {profile_id_upper} (ID: {person.id})"
                )
                return person  # Return the single match

            else:  # Multiple matches found for the same profile_id
                logger.warning(
                    f"Multiple people found ({len(potential_matches)}) for Profile ID: {profile_id_upper}. Attempting disambiguation by username."
                )
                if person_username:
                    username_lower = person_username.lower()
                    found_by_username = None
                    for p in potential_matches:
                        # Case-insensitive username comparison
                        if p.username and p.username.lower() == username_lower:
                            # Check if we already found one (shouldn't happen with unique constraint, but safety check)
                            if found_by_username is not None:
                                logger.error(
                                    f"CRITICAL: Found multiple people matching BOTH Profile ID {profile_id_upper} AND Username '{person_username}' (IDs: {found_by_username.id}, {p.id}). Cannot reliably identify."
                                )
                                return None  # Cannot reliably identify
                            found_by_username = p

                    if found_by_username:
                        logger.debug(
                            f"Disambiguated by username: Found Person ID {found_by_username.id} ('{found_by_username.username}') matching Profile ID {profile_id_upper}."
                        )
                        return found_by_username
                    else:
                        # If username provided but none matched among the profile_id matches
                        logger.warning(
                            f"Multiple matches for Profile ID {profile_id_upper}, but none matched username '{person_username}'. Cannot reliably identify."
                        )
                        return None  # Return None to prevent updating the wrong record
                else:
                    # Multiple matches, but no username provided for disambiguation
                    logger.warning(
                        f"Multiple matches for Profile ID {profile_id_upper}, but no username provided for disambiguation. Cannot reliably identify."
                    )
                    return None  # Return None to prevent updating the wrong record

        # If not found by UUID and either not found by Profile ID or disambiguation failed
        # This check ensures the final log message is accurate
        if person is None:
            logger.debug(f"No existing person reliably identified for {log_ref}.")

    except SQLAlchemyError as e:
        logger.error(
            f"Database error in find_existing_person for {log_ref}: {e}", exc_info=True
        )
        return None  # Return None on DB error
    except Exception as e:
        logger.error(
            f"Unexpected error in find_existing_person for {log_ref}: {e}",
            exc_info=True,
        )
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
            .options(
                joinedload(Person.dna_match), joinedload(Person.family_tree) # Use joinedload
            )
            # --- END MODIFICATION ---
            .filter(Person.uuid == uuid_upper).first()
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
    session: Session,
    person_data: Dict[str, Any],
    existing_person: Optional[Person] = None,
) -> Tuple[Optional[Person], Literal["created", "updated", "skipped", "error"]]:
    """
    V9 REVISED: Creates a new Person or updates an existing one based on input data.
    Assumes caller has already performed the lookup and determined if create/update needed.
    Handles the DB write/update logic ONLY.

    Args:
        session: The SQLAlchemy Session.
        person_data: Dictionary containing person data (must include 'uuid', 'username').
        existing_person: The pre-fetched Person object if it exists, otherwise None.

    Returns:
        Tuple: (
            Person object (new or updated) or None on error,
            status ["created", "updated", "skipped", "error"]
        )
    """
    uuid_val = person_data.get("uuid")
    profile_id_val = person_data.get("profile_id")  # For logging/creation
    username_val = person_data.get("username")  # Should always be present per checks

    if not uuid_val or not username_val:
        logger.error(
            f"Cannot create/update person for ProfileID='{profile_id_val or 'N/A'}', User='{username_val or 'N/A'}': UUID or Username missing in input data."
        )
        return None, "error"

    log_ref = f"UUID={uuid_val} / ProfileID={profile_id_val or 'NULL'} / User='{username_val}'"
    updated = False  # Flag to track if any allowed *Person* field update occurred

    try:
        if existing_person:
            # --- PERSON EXISTS --- Update logic ---
            person_id_for_logging = existing_person.id
            logger.debug(
                f"{log_ref}: Updating existing Person ID {person_id_for_logging}."
            )

            # Expire state to ensure fresh data comparison if needed (optional but good practice)
            try:
                # logger.debug(f"{log_ref}: Expiring state for existing Person ID {person_id_for_logging} before updates.") # Less verbose
                session.expire(existing_person)
            except Exception as expire_e:
                logger.warning(
                    f"Could not expire session state for Person ID {person_id_for_logging}: {expire_e}"
                )

            # --- Apply Updates to Restricted Fields ---
            # 1. last_logged_in
            new_last_logged_in = person_data.get("last_logged_in")
            current_last_logged_in = existing_person.last_logged_in
            current_naive = None
            new_naive = None
            # Convert current DB value to naive UTC, truncated to seconds
            if isinstance(current_last_logged_in, datetime):
                db_aware = (
                    current_last_logged_in.tzinfo is not None
                    and current_last_logged_in.tzinfo.utcoffset(current_last_logged_in)
                    is not None
                )
                if db_aware:
                    current_naive = current_last_logged_in.astimezone(
                        timezone.utc
                    ).replace(tzinfo=None, microsecond=0)
                else:  # Assume naive stored is UTC
                    current_naive = current_last_logged_in.replace(microsecond=0)
            # Convert new incoming value to naive UTC, truncated to seconds
            if isinstance(new_last_logged_in, datetime):
                new_naive = new_last_logged_in.astimezone(timezone.utc).replace(
                    tzinfo=None, microsecond=0
                )

            if current_naive != new_naive:
                logger.debug(
                    f"  Updating last_logged_in for {log_ref}: '{current_naive}' -> '{new_naive}'"
                )
                existing_person.last_logged_in = (
                    new_last_logged_in  # Store the original aware datetime
                )
                updated = True

            # 2. contactable
            new_contactable = person_data.get(
                "contactable", False
            )  # Default to False if missing
            current_contactable = existing_person.contactable
            if bool(current_contactable) != bool(new_contactable):
                logger.debug(
                    f"  Updating contactable for {log_ref}: '{current_contactable}' -> '{new_contactable}'"
                )
                existing_person.contactable = bool(new_contactable)
                updated = True

            # 3. birth_year (only update if current is None and new is not None)
            new_birth_year = person_data.get("birth_year")
            current_birth_year = existing_person.birth_year
            if new_birth_year is not None and current_birth_year is None:
                try:
                    birth_year_int = int(new_birth_year)
                    logger.debug(
                        f"  Updating birth_year for {log_ref}: '{current_birth_year}' -> '{birth_year_int}'"
                    )
                    existing_person.birth_year = birth_year_int
                    updated = True
                except (ValueError, TypeError):
                    logger.warning(
                        f"  Skipping birth_year update for {log_ref}: New value '{new_birth_year}' is not a valid integer."
                    )

            # 4. in_my_tree (Update if different, allow False->True and True->False)
            new_in_my_tree = bool(person_data.get("in_my_tree", False))
            current_in_my_tree = existing_person.in_my_tree
            if bool(current_in_my_tree) != new_in_my_tree:
                logger.debug(
                    f"  Updating in_my_tree for {log_ref}: '{current_in_my_tree}' -> '{new_in_my_tree}'"
                )
                existing_person.in_my_tree = new_in_my_tree
                updated = True

            # 5. Gender (Update if current is None and new is not None)
            new_gender = person_data.get("gender")
            current_gender = existing_person.gender
            if new_gender is not None and current_gender is None:
                if isinstance(new_gender, str) and new_gender.lower() in ("f", "m"):
                    logger.debug(
                        f"  Updating gender for {log_ref}: '{current_gender}' -> '{new_gender.lower()}'"
                    )
                    existing_person.gender = new_gender.lower()
                    updated = True
                else:
                    logger.warning(
                        f"  Skipping gender update for {log_ref}: New value '{new_gender}' is not 'f' or 'm'."
                    )

            # 6. Admin Info (Update if changed)
            new_admin_id = person_data.get("administrator_profile_id")
            new_admin_user = person_data.get("administrator_username")
            current_admin_id = existing_person.administrator_profile_id
            current_admin_user = existing_person.administrator_username
            new_admin_id_upper = new_admin_id.upper() if new_admin_id else None

            if current_admin_id != new_admin_id_upper:
                logger.debug(f"  Updating admin ID for {log_ref}: '{current_admin_id}' -> '{new_admin_id_upper}'")
                existing_person.administrator_profile_id = new_admin_id_upper
                updated = True
            if current_admin_user != new_admin_user:
                logger.debug(f"  Updating admin username for {log_ref}: '{current_admin_user}' -> '{new_admin_user}'")
                existing_person.administrator_username = new_admin_user
                updated = True

            # 7. Message Link (Update if changed)
            new_message_link = person_data.get("message_link")
            current_message_link = existing_person.message_link
            if current_message_link != new_message_link:
                logger.debug(f"  Updating message link for {log_ref}: '{current_message_link}' -> '{new_message_link}'")
                existing_person.message_link = new_message_link
                updated = True

            # --- Determine Final Status ---
            if updated:
                existing_person.updated_at = datetime.now()  # Update timestamp
                logger.debug(f"{log_ref}: Person ID {person_id_for_logging} updated.")
                return existing_person, "updated"
            else:
                logger.debug(
                    f"{log_ref}: Person ID {person_id_for_logging} requires no update."
                )
                return existing_person, "skipped"

        else:
            # --- PERSON DOES NOT EXIST --- Create new logic ---
            logger.debug(f"{log_ref}: Creating new Person.")
            # Call create_person helper (which handles the actual insert)
            new_person_id = create_person(
                session, person_data
            )  # Pass the full data dict

            if new_person_id > 0:
                # Fetch the newly created object to return it
                # Use session.get which is efficient for primary key lookup
                new_person_obj = session.get(Person, new_person_id)
                if new_person_obj:
                    logger.debug(
                        f"{log_ref}: New Person created (ID: {new_person_id})."
                    )
                    return new_person_obj, "created"
                else:
                    # This should ideally not happen if create_person succeeded
                    logger.error(
                        f"Failed to fetch newly created person with ID {new_person_id} for {log_ref} after successful creation."
                    )
                    # Rollback might be needed if create_person didn't handle it on failure
                    if session.is_active:
                        session.rollback()
                    return None, "error"
            else:
                logger.error(f"create_person failed for {log_ref}.")
                # Rollback should have been handled by create_person on failure
                return None, "error"

    except IntegrityError as ie:
        # Handle potential constraint violations not caught by create_person's pre-check
        # (e.g., UUID collision if pre-check is removed/fails)
        session.rollback()
        logger.error(
            f"IntegrityError processing person {log_ref}: {ie}. Rolling back.",
            exc_info=False,
        )
        return None, "error"
    except SQLAlchemyError as e:
        if session.is_active:
            session.rollback()
        logger.error(f"SQLAlchemyError processing person {log_ref}: {e}", exc_info=True)
        return None, "error"
    except (
        NameError
    ) as ne:  # Catch missing 'timezone' import if datetime handling fails
        if session.is_active:
            session.rollback()
        logger.critical(
            f"NameError processing person {log_ref}: {ne}. Ensure 'timezone' is imported from datetime.",
            exc_info=True,
        )
        return None, "error"
    except TypeError as te:
        if session.is_active:
            session.rollback()
        logger.critical(
            f"TypeError during person update comparison for {log_ref}: {te}",
            exc_info=True,
        )
        return None, "error"
    except Exception as e:
        # Catch any other unexpected error
        if session.is_active:
            session.rollback()
        logger.critical(
            f"Unexpected critical error processing person {log_ref}: {e}", exc_info=True
        )
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
    if not profile_id or not username:
         logger.warning("update_person: profile_id and username required.")
         return False
    try:
        person = (
            session.query(Person)
            .filter_by(profile_id=profile_id.upper(), username=username) # Ensure uppercase lookup
            .first()
        )
        if not person:
            logger.warning(
                f"update_person: Person with profile_id {profile_id} and username {username} not found."
            )
            return False

        # Update fields (whitelist approach for safety)
        updated = False  # Track if any changes are made
        allowed_fields = [
            "uuid",
            "profile_id", # Allow updating profile_id if needed (e.g., placeholder)
            "username",
            "administrator_profile_id",
            "administrator_username",
            "message_link",
            "in_my_tree",
            "status",
            "first_name",
            "gender",
            "birth_year",
            "contactable",
            "last_logged_in",
        ]  # Expanded allowed fields based on create_or_update logic
        for key, value in update_data.items():
            if key in allowed_fields and hasattr(person, key):
                current_value = getattr(person, key)
                # Handle potential case differences for IDs
                value_to_compare = value
                if key in ('profile_id', 'administrator_profile_id', 'uuid') and value:
                     value_to_compare = value.upper()

                if current_value != value_to_compare:  # Only update if value changed
                    setattr(person, key, value_to_compare) # Store corrected case if ID
                    logger.debug(f"Updating {key} for Person ID {person.id} to {value_to_compare}")
                    updated = True
            elif key not in allowed_fields:
                logger.warning(
                    f"Attempted to update non-allowed attribute '{key}' on Person ID {person.id}."
                )

        if updated:
            person.updated_at = datetime.now()  # Update timestamp if changed
            session.flush()  # Stage the update
            logger.info(
                f"Updated person with profile_id {profile_id} and username {username} (ID: {person.id})."
            )
            return True
        else:
            logger.debug(
                f"No update needed for person profile_id {profile_id} / username {username}."
            )
            return True  # Return True even if no changes needed, as the record exists

    except IntegrityError as ie:
        session.rollback()
        logger.error(
            f"IntegrityError updating person {profile_id}/{username}: {ie}.",
            exc_info=False, # Less verbose for integrity
        )
        return False
    except SQLAlchemyError as e:
        logger.error(
            f"Database error updating person {profile_id}/{username}: {e}",
            exc_info=True,
        )
        session.rollback()
        return False
    except Exception as e:  # Catch any other unexpected error
        logger.critical(
            f"Unexpected error in update_person for {profile_id}/{username}: {e}",
            exc_info=True,
        )
        if session.is_active:
            session.rollback()
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
    if not profile_id or not username:
         logger.warning("delete_person: profile_id and username required.")
         return False
    try:
        person = (
            session.query(Person)
            .filter_by(profile_id=profile_id.upper(), username=username) # Ensure uppercase lookup
            .first()
        )
        if not person:
            logger.warning(
                f"delete_person: Person with profile_id {profile_id} and username {username} not found."
            )
            return False

        person_id = person.id  # Get ID for logging before delete
        session.delete(person)  # Cascading delete will handle related records
        session.flush()  # Execute delete in DB
        logger.info(
            f"Deleted person ID {person_id} (profile_id {profile_id}, username {username}) and related records."
        )
        return True

    except SQLAlchemyError as e:
        logger.error(
            f"Database error deleting person {profile_id}/{username}: {e}",
            exc_info=True,
        )
        session.rollback()
        return False
    except Exception as e:  # Catch any other unexpected error
        logger.critical(
            f"Unexpected error in delete_person for {profile_id}/{username}: {e}",
            exc_info=True,
        )
        if session.is_active:
            session.rollback()
        return False
# End of delete_person


def delete_database(
    session_manager, db_path: Path, max_attempts=5
):  # Increased max_attempts to 5
    """
    Delete the database file (Path object) with retry and exponential backoff.
    More aggressive cleanup and error handling for file locks.
    """
    if not isinstance(db_path, Path):
        try:
            db_path = Path(db_path)
            logger.warning(
                "Converted db_path string to Path object in delete_database."
            )
        except TypeError:
            logger.error(f"Cannot convert db_path {db_path} to Path object.")
            return  # Cannot proceed

    logger.debug(f"Attempting to delete database file: {db_path}")
    last_error = None  # Store last error

    for attempt in range(max_attempts):
        logger.debug(f"Delete attempt {attempt + 1}/{max_attempts}...")
        try:
            # 1. Aggressive Cleanup: Close connections, dispose engine, garbage collect
            logger.debug("Closing DB connections and disposing engine...")
            if session_manager:
                session_manager.cls_db_conn()  # Should dispose engine
            else:
                logger.warning("No session_manager provided to delete_database.")

            # Multiple GC calls and longer sleep
            gc.collect()
            time.sleep(0.5)  # Short sleep
            gc.collect()
            time.sleep(1.0 + attempt)  # Increase sleep slightly with attempts

            # 2. Check existence and attempt deletion
            if db_path.exists():
                logger.debug(f"Attempting os.remove on {db_path}...")
                os.remove(db_path)
                # Verify deletion
                time.sleep(0.1)  # Tiny pause before checking again
                if not db_path.exists():
                    logger.debug(
                        f"'{db_path}' deleted successfully."
                    )  # Use INFO for success
                    return  # SUCCESS
                else:
                    logger.warning(
                        f"os.remove called, but file '{db_path}' still exists. Retrying."
                    )
                    last_error = OSError(
                        f"File still exists after os.remove attempt {attempt + 1}"
                    )
                    # Continue to retry logic below
            else:
                logger.info(
                    f"Database '{db_path}' does not exist (or already deleted)."
                )
                return  # SUCCESS (already gone)

        # Catch specific permission errors likely related to file locks
        except PermissionError as e:
            logger.warning(
                f"PermissionError deleting '{db_path}' (Attempt {attempt + 1}): {e}. File likely locked."
            )
            last_error = e
            # Continue to retry logic below
        except OSError as e:
            # Catch other OS errors, check for WinError 32 specifically
            if hasattr(e, "winerror") and e.winerror == 32:
                logger.warning(
                    f"OSError (WinError 32) deleting '{db_path}' (Attempt {attempt + 1}): {e}. File likely locked."
                )
            else:
                logger.error(
                    f"OSError deleting '{db_path}' (Attempt {attempt + 1}): {e}",
                    exc_info=True,
                )
            last_error = e
            # Continue to retry logic below
        except Exception as e:
            # Catch any other unexpected errors
            logger.critical(
                f"Unexpected error during delete attempt {attempt + 1} for '{db_path}': {e}",
                exc_info=True,
            )
            last_error = e
            # Continue to retry logic below

        # --- Retry Logic ---
        if attempt < max_attempts - 1:
            wait_time = 2**attempt  # Exponential backoff
            logger.info(f"Waiting {wait_time} seconds before next delete attempt...")
            time.sleep(wait_time)
        else:
            # Last attempt failed
            logger.critical(
                f"Failed to delete database '{db_path}' after {max_attempts} attempts. Last error: {last_error or 'Unknown Error'}"
            )
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
    session_manager=None,  # Keep session_manager arg for potential future use or remove if definitely unused
):
    """Backs up the database file specified in config_instance to the 'Data' folder."""
    db_path = config_instance.DATABASE_FILE  # This is now a Path object
    backup_dir = (
        config_instance.DATA_DIR
    )  # Use DATA_DIR from config, also a Path object
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
            logger.info(f"Backed up to '{backup_path}' OK.\n")  # Use INFO for success
        else:
            logger.warning(f"Database file '{db_path}' not found. No backup created.\n")
    except Exception as e:
        logger.error(
            f"Error backing up database from '{db_path}' to '{backup_path}': {e}\n",
            exc_info=True,
        )
        # Re-raise the exception if backup failure should halt execution
        # raise
# End of backup_database

# ----------------------------------------------------------------------
# Standalone execution (for testing database setup)
#################################################################################


if __name__ == "__main__":
    # Import sys here for the forced logging stream
    import sys
    import json # Needed for seeding

    # --- Force basic logging config for standalone execution ---
    logging.basicConfig(
        level=logging.DEBUG, # Set to DEBUG for detailed output during test
        format='%(asctime)s %(levelname).3s [%(name)-12s %(lineno)-4d] %(message)s', # Simplified format
        datefmt='%H:%M:%S',
        stream=sys.stderr # Force output to stderr for visibility
    )
    # Get the logger again *after* basicConfig is set
    standalone_logger = logging.getLogger("db_standalone") # Use a specific name
    standalone_logger.info("--- Starting database.py standalone test ---")

    # --- Get DB Path ---
    try:
        db_path_obj = config_instance.DATABASE_FILE
        db_path_str = str(db_path_obj.resolve()) # Resolve for absolute path
        standalone_logger.info(f"Using database file: {db_path_str}")
    except Exception as config_err:
        standalone_logger.critical(f"CRITICAL: Error getting database path from config: {config_err}. Cannot proceed.")
        sys.exit(1) # Exit if config is broken

    engine = None
    conn_pool = None

    try:
        # --- Create Engine ---
        standalone_logger.debug("Creating SQLAlchemy engine...")
        engine = create_engine(f"sqlite:///{db_path_str}", echo=False)

        # --- Add PRAGMA foreign_keys=ON listener ---
        @event.listens_for(engine, "connect")
        def enable_foreign_keys(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            try: cursor.execute("PRAGMA foreign_keys=ON")
            except Exception as pragma_e: standalone_logger.error(f"Failed PRAGMA: {pragma_e}")
            finally: cursor.close()
        standalone_logger.debug("Foreign key listener attached.")

        # --- Create Tables ---
        standalone_logger.debug("Creating/Verifying database tables...")
        Base.metadata.create_all(engine)
        standalone_logger.info(f"Database tables OK: {db_path_str}")

        # --- Seed Message Types (Essential for FK constraints) ---
        standalone_logger.info("Seeding MessageType table...")
        SessionSeed = sessionmaker(bind=engine)
        seed_session = None # Initialize
        try:
            seed_session = SessionSeed()
            script_dir = Path(__file__).resolve().parent
            messages_file = script_dir / "messages.json"
            if messages_file.exists():
                with messages_file.open("r", encoding="utf-8") as f: messages_data = json.load(f)
                if isinstance(messages_data, dict):
                    with db_transn(seed_session) as sess: # Use context manager
                        types_to_add = []
                        for name in messages_data:
                            exists = sess.query(MessageType).filter_by(type_name=name).first()
                            if not exists: types_to_add.append(MessageType(type_name=name))
                        if types_to_add:
                            standalone_logger.debug(f"Adding {len(types_to_add)} message types...")
                            sess.add_all(types_to_add)
                        else: standalone_logger.debug("Message types already exist.")
                    count = seed_session.query(func.count(MessageType.id)).scalar() or 0
                    standalone_logger.info(f"MessageType seeding OK. Total types: {count}")
                else: standalone_logger.error("'messages.json' has incorrect format.")
            else: standalone_logger.warning(f"'messages.json' not found at '{messages_file}', skipping seeding.")
        except Exception as seed_err:
            standalone_logger.error(f"Error seeding MessageType table: {seed_err}", exc_info=True)
        finally:
            if seed_session: seed_session.close()

    except SQLAlchemyError as db_e:
        standalone_logger.critical(f"Database setup/connection error: {db_e}", exc_info=True)
    except Exception as e:
        standalone_logger.critical(f"Unexpected error during standalone test: {e}", exc_info=True)
    finally:
        # --- Final Cleanup ---
        standalone_logger.debug("Performing final cleanup...")
        if conn_pool:
            conn_pool.clse_all_sess() # Closes pool and disposes engine
        elif engine:
            engine.dispose()
            standalone_logger.debug("SQLAlchemy engine disposed (pool cleanup skipped/failed).")

        standalone_logger.info("--- Database.py standalone test finished ---")