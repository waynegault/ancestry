#!/usr/bin/env python3

# database.py
# V1.29 - Merging latest schema with functions from user-provided older version, adapting functions minimally.

# Imports
import os
import shutil
from typing import List, Dict, Optional, Tuple, Any, Type, Literal
from pathlib import Path
from datetime import datetime, timezone  # Ensure timezone is imported
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
    Text,  # Added Text
    PrimaryKeyConstraint,  # Added PrimaryKeyConstraint (though not used explicitly below if composite)
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    relationship,
    Session,
    joinedload,
)
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import logging
import re
import inspect
import sys  # Added sys for standalone block
import json  # Added json for standalone block


# Initialize logging
logger = logging.getLogger("logger")

# ----------------------------------------------------------------------
# SQLAlchemy Models (Using LATEST Schema Definitions)
# ----------------------------------------------------------------------

Base = declarative_base()


# --- Define Enums Globally ---
class MessageDirectionEnum(enum.Enum):
    IN = "IN"
    OUT = "OUT"


class RoleType(enum.Enum):  # Kept enum from user file
    AUTHOR = "AUTHOR"
    RECIPIENT = "RECIPIENT"


class PersonStatusEnum(enum.Enum):  # Added Enum for status
    ACTIVE = "active"
    DESIST = "desist"
    ARCHIVE = "archive"
    BLOCKED = "blocked"


# --- Model Definitions (Current Schema) ---


class ConversationLog(Base):
    __tablename__ = "conversation_log"
    # Composite Primary Key
    conversation_id = Column(String, primary_key=True, index=True)
    direction = Column(
        SQLEnum(MessageDirectionEnum, name="message_direction_enum_v5"),
        primary_key=True,  # Uses correct Enum
    )

    people_id = Column(Integer, ForeignKey("people.id"), nullable=False, index=True)
    latest_message_content = Column(Text, nullable=True)
    latest_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    ai_sentiment = Column(String, nullable=True, index=True)
    message_type_id = Column(Integer, ForeignKey("message_types.id"), nullable=True)
    script_message_status = Column(String, nullable=True)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationship defined with backref (creates 'conversation_log_entries' on Person)
    person = relationship("Person", backref="conversation_log_entries")
    message_type = relationship("MessageType")  # Assumes MessageType defined below

    __table_args__ = (
        Index(
            "ix_conversation_log_people_id_direction_ts",
            "people_id",
            "direction",
            "latest_timestamp",
        ),
        Index("ix_conversation_log_timestamp", "latest_timestamp"),
    )


# End of class ConversationLog


class MessageType(Base):  # Kept from user file (was correct)
    __tablename__ = "message_types"
    id = Column(Integer, primary_key=True)
    type_name = Column(String, unique=True, nullable=False)
    # Removed messages relationship as MessageHistory is removed
    # messages = relationship("MessageHistory", back_populates="message_type", cascade="all, delete, delete-orphan")


# End of class MessageType


class DnaMatch(Base):  # Kept from user file, added timezone=True
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
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )  # Set timezone
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )  # Set timezone
    person = relationship("Person", back_populates="dna_match")


# End of class DnaMatch


class FamilyTree(Base):  # Kept from user file, added timezone=True
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
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )  # Set timezone
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )  # Set timezone
    person = relationship("Person", back_populates="family_tree")


# End of class family tree


class Person(Base):  # Updated Person Model
    __tablename__ = "people"
    id = Column(Integer, primary_key=True)
    uuid = Column(String, nullable=True, unique=True, index=True)
    profile_id = Column(
        String, unique=True, nullable=True, index=True
    )  # Kept unique=True
    username = Column(String, unique=False, nullable=False)
    first_name = Column(String, nullable=True)
    gender = Column(String(1), nullable=True)
    birth_year = Column(Integer, nullable=True)
    message_link = Column(String, unique=False, nullable=True)
    in_my_tree = Column(Boolean, default=False)
    contactable = Column(Boolean, default=False)
    last_logged_in = Column(
        DateTime(timezone=True), nullable=True, index=True
    )  # Use timezone=True
    administrator_profile_id = Column(String, nullable=True, index=True)
    administrator_username = Column(String, nullable=True)
    status = Column(  # Use Enum for status
        SQLEnum(
            PersonStatusEnum, name="person_status_enum_v3"
        ),  # Kept v3 name but uses correct Enum
        default=PersonStatusEnum.ACTIVE,
        nullable=False,
        index=True,
    )
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships using back_populates where appropriate
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
    # Removed relationships to InboxStatus and MessageHistory
    # conversation_log_entries is created by backref in ConversationLog model


# End of class Person

# ----------------------------------------------------------------------
# Context Manager (Kept from user version - appears correct)
# ----------------------------------------------------------------------


@contextlib.contextmanager
def db_transn(session: Session):
    """Context manager for database transactions."""
    if not session or not session.is_active:
        raise SQLAlchemyError("Invalid or inactive session provided to db_transn.")
    try:
        yield session
        if session.is_active:
            session.commit()
    except Exception as e:
        logger.error(
            f"Exception within db_transn block: {e}. Rolling back.", exc_info=True
        )
        if session and session.is_active:
            session.rollback()
        raise
    finally:
        pass  # Session closing handled by SessionManager


# end db_transn

# ==================== CRUD OPERATIONS ====================
# Keeping functions from user-provided version, adapted where necessary for schema

# ----------------------------------------------------------------------
# Create/ Insert
# ----------------------------------------------------------------------


def create_person(session: Session, person_data: Dict[str, Any]) -> int:
    """
    V1.1 (from user file) REVISED: Creates a new person record.
    - Checks existing profile_id/uuid.
    - Uses PersonStatusEnum.
    - Adapts status assignment.
    - Corrects return value logic.
    """
    required_keys = ("username", "uuid")
    if not all(
        key in person_data and person_data[key] is not None for key in required_keys
    ):
        logger.warning(
            f"Missing required non-null data: Needs username and uuid. Data: {person_data}"
        )
        return 0
    profile_id_raw = person_data.get("profile_id")
    profile_id_upper = profile_id_raw.upper() if profile_id_raw else None
    uuid_upper = str(person_data["uuid"]).upper() if person_data.get("uuid") else None
    username = person_data["username"]
    log_ref = f"UUID={uuid_upper or 'NULL'} / ProfileID={profile_id_upper or 'NULL'} / User='{username}'"

    try:
        if profile_id_upper:
            existing_by_profile = (
                session.query(Person.id)
                .filter(Person.profile_id == profile_id_upper)
                .first()
            )
            if existing_by_profile:
                logger.error(
                    f"Create FAILED {log_ref}: Profile ID exists (ID {existing_by_profile.id})."
                )
                return 0
        if uuid_upper:
            existing_by_uuid = (
                session.query(Person.id).filter(Person.uuid == uuid_upper).first()
            )
            if existing_by_uuid:
                logger.error(
                    f"Create FAILED {log_ref}: UUID exists (ID {existing_by_uuid.id})."
                )
                return 0

        logger.debug(f"Proceeding with Person creation for {log_ref}.")
        last_logged_in_dt = person_data.get("last_logged_in")
        if isinstance(last_logged_in_dt, datetime) and last_logged_in_dt.tzinfo is None:
            last_logged_in_dt = last_logged_in_dt.replace(tzinfo=timezone.utc)

        # Adapt status: Use PersonStatusEnum, default to ACTIVE if not provided or invalid
        status_value = person_data.get("status", PersonStatusEnum.ACTIVE)
        if not isinstance(status_value, PersonStatusEnum):
            try:
                status_enum = PersonStatusEnum(
                    str(status_value).lower()
                )  # Try converting string
            except ValueError:
                logger.warning(
                    f"Invalid status '{status_value}' provided for {log_ref}, defaulting to ACTIVE."
                )
                status_enum = PersonStatusEnum.ACTIVE
        else:
            status_enum = status_value  # Already an Enum

        new_person = Person(
            uuid=uuid_upper,
            profile_id=profile_id_upper,
            username=username,
            administrator_profile_id=(
                person_data.get("administrator_profile_id").upper()
                if person_data.get("administrator_profile_id")
                else None
            ),
            administrator_username=person_data.get("administrator_username"),
            message_link=person_data.get("message_link"),
            in_my_tree=bool(person_data.get("in_my_tree", False)),
            status=status_enum,  # Use the validated Enum value
            first_name=person_data.get("first_name"),
            gender=person_data.get("gender"),
            birth_year=person_data.get("birth_year"),
            contactable=person_data.get("contactable", True),
            last_logged_in=last_logged_in_dt,
            # created_at/updated_at use timezone=True defaults now
        )
        session.add(new_person)
        session.flush()  # Flush to get ID and check constraints

        if new_person.id is None:
            logger.error(
                f"ID not assigned after flush for person {log_ref}! Rolling back."
            )
            session.rollback()
            return 0

        logger.debug(f"Created Person record ID {new_person.id} for {log_ref}.")
        return int(new_person.id)  # Return the actual ID

    except IntegrityError as ie:
        session.rollback()
        logger.error(f"IntegrityError create_person {log_ref}: {ie}.", exc_info=False)
        return 0
    except SQLAlchemyError as e:
        logger.error(f"DB error create_person {log_ref}: {e}", exc_info=True)
        session.rollback() if session.is_active else None
        return 0
    except Exception as e:
        logger.critical(f"Unexpected error create_person {log_ref}: {e}", exc_info=True)
        session.rollback() if session.is_active else None
        return 0


# End of create_person


def create_dna_match(
    session: Session, match_data: Dict[str, Any]
) -> Literal["created", "skipped", "error"]:
    """(Kept from user version - check logic carefully) Creates DNA Match record."""
    people_id = match_data.get("people_id")
    log_ref = f"PersonID={people_id}, KitUUID={match_data.get('uuid', 'N/A')}"
    if not people_id or not isinstance(people_id, int) or people_id <= 0:
        logger.error(f"create_dna_match: Invalid people_id {log_ref}.")
        return "error"
    required_keys = ("compare_link", "cM_DNA", "predicted_relationship")
    if not all(
        key in match_data and match_data[key] is not None for key in required_keys
    ):
        logger.error(
            f"create_dna_match: Missing required DNA data {log_ref}. Data: {match_data}"
        )
        return "error"
    try:
        cm_dna_val = int(match_data["cM_DNA"])
        assert cm_dna_val >= 0
    except:
        logger.error(
            f"create_dna_match: Invalid cM_DNA {match_data.get('cM_DNA')} for {log_ref}."
        )
        return "error"

    def validate_optional_numeric(key, value, allow_float=False):
        if value is None:
            return None
        try:
            if isinstance(value, str) and not value.replace(".", "", 1).isdigit():
                logger.warning(
                    f"Non-numeric '{value}' for {key} in {log_ref}. Setting None."
                )
                return None
            return float(value) if allow_float else int(value)
        except (TypeError, ValueError):
            logger.warning(f"Invalid {key} '{value}' for {log_ref}. Setting None.")
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
    try:
        dna_match = session.query(DnaMatch).filter_by(people_id=people_id).first()
        if dna_match:
            logger.debug(f"Existing 'dna_match' found {log_ref}. Skipping.")
            return "skipped"
        else:
            new_dna_match = DnaMatch(
                people_id=people_id,
                compare_link=match_data["compare_link"],
                cM_DNA=cm_dna_val,
                predicted_relationship=match_data["predicted_relationship"],
                shared_segments=shared_segments_val,
                longest_shared_segment=longest_segment_val,
                meiosis=meiosis_val,
                from_my_fathers_side=bool(
                    match_data.get("from_my_fathers_side", False)
                ),
                from_my_mothers_side=bool(
                    match_data.get("from_my_mothers_side", False)
                ),
            )
            session.add(new_dna_match)
            logger.debug(f"Staged new 'dna_match' {log_ref}")
            return "created"
    except IntegrityError as ie:
        session.rollback()
        logger.error(
            f"IntegrityError create_dna_match {log_ref}: {ie}.", exc_info=False
        )
        return "error"
    except SQLAlchemyError as e:
        logger.error(f"DB error create_dna_match {log_ref}: {e}", exc_info=True)
        return "error"
    except Exception as e:
        logger.error(f"Unexpected error create_dna_match {log_ref}: {e}", exc_info=True)
        return "error"


# End of create_dna_match


def create_family_tree(
    session: Session, tree_data: Dict[str, Any]
) -> Literal["created", "updated", "skipped", "error"]:
    """(Kept from user version - check logic carefully) Creates or updates FamilyTree record."""
    people_id = tree_data.get("people_id")
    if not people_id:
        logger.error("Cannot create/update FamilyTree: 'people_id' missing.")
        return "error"
    cfpid_val = tree_data.get("cfpid")
    log_ref = f"PersonID={people_id}, CFPID={cfpid_val or 'N/A'}"
    updated = False
    try:
        existing_tree = session.query(FamilyTree).filter_by(people_id=people_id).first()
        valid_tree_args = {
            "people_id": people_id,
            "cfpid": cfpid_val,
            "person_name_in_tree": tree_data.get("person_name_in_tree"),
            "facts_link": tree_data.get("facts_link"),
            "view_in_tree_link": tree_data.get("view_in_tree_link"),
            "actual_relationship": tree_data.get("actual_relationship"),
            "relationship_path": tree_data.get("relationship_path"),
        }
        if existing_tree:
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
                    setattr(existing_tree, field, new_value)
                    updated = True
            if updated:
                existing_tree.updated_at = datetime.now(timezone.utc)
                logger.debug(f"Staged update FT {log_ref}")
                return "updated"
            else:
                logger.debug(f"No update needed FT {log_ref}")
                return "skipped"
        else:
            new_tree = FamilyTree(**valid_tree_args)
            session.add(new_tree)
            logger.debug(f"Staged new FT {log_ref}")
            return "created"
    except TypeError as te:
        logger.critical(
            f"TypeError create/update FT {log_ref}: {te}. Data keys: {list(tree_data.keys())}",
            exc_info=True,
        )
        return "error"
    except IntegrityError as ie:
        logger.error(f"IntegrityError create/update FT {log_ref}: {ie}", exc_info=False)
        return "error"
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemyError create/update FT {log_ref}: {e}", exc_info=True)
        return "error"
    except Exception as e:
        logger.critical(
            f"Unexpected error create_family_tree {log_ref}: {e}", exc_info=True
        )
        return "error"


# End create_family_tree


# ----------------------------------------------------------------------
# Retrieve (Kept functions from user version)
# ----------------------------------------------------------------------


def get_person_by_profile_id_and_username(
    session: Session, profile_id: str, username: str
) -> Optional[Person]:
    """Retrieves a Person by profile_id AND username."""
    if not profile_id or not username:
        logger.warning(
            "get_person_by_profile_id_and_username: profile_id and username required."
        )
        return None
    try:
        return (
            session.query(Person)
            .filter_by(profile_id=profile_id.upper(), username=username)
            .first()
        )
    except Exception as e:
        logger.error(
            f"Error retrieving person by profile_id/username: {e}", exc_info=True
        )
        return None


# end get_person_by_profile_id_and_username


def get_person_by_profile_id(session: Session, profile_id: str) -> Optional[Person]:
    """Retrieves a Person record based on profile_id."""
    if not profile_id:
        logger.warning("get_person_by_profile_id: profile_id required.")
        return None
    try:
        return session.query(Person).filter_by(profile_id=profile_id.upper()).first()
    except Exception as e:
        logger.error(
            f"Error retrieving person by profile_id '{profile_id}': {e}", exc_info=True
        )
        return None


# end of get_person_by_profile_id


def get_person_and_dna_match(
    session: Session, match_data: Dict[str, Any]
) -> Tuple[Optional[Person], Optional[DnaMatch]]:
    """Retrieves a Person and their associated DnaMatch record using profile_id/username."""
    profile_id = match_data.get("profile_id")
    username = match_data.get("username")
    if not profile_id or not username:
        logger.warning("get_person_and_dna_match: profile_id and username required.")
        return None, None
    try:
        person = (
            session.query(Person)
            .options(joinedload(Person.dna_match))
            .filter_by(profile_id=profile_id.upper(), username=username)
            .first()
        )
        if person:
            return person, person.dna_match
        else:
            return None, None
    except Exception as e:
        logger.error(
            f"Error retrieving person/DNA match for profile {profile_id}/{username}: {e}",
            exc_info=True,
        )
        return None, None


# end of get_person_and_dna_match


def find_existing_person(
    session: Session, identifier_data: Dict[str, Any]
) -> Optional[Person]:
    """Finds an existing person based on uuid or profile_id, handling disambiguation."""
    # ...(Logic from user version kept, appears compatible)...
    person_uuid = identifier_data.get("uuid")
    person_profile_id = identifier_data.get("profile_id")
    person_username = identifier_data.get("username")
    person = None
    log_parts = []
    if person_uuid:
        log_parts.append(f"UUID='{person_uuid}'")
    if person_profile_id:
        log_parts.append(f"ProfileID='{person_profile_id}'")
    if person_username:
        log_parts.append(f"User='{person_username}'")
    log_ref = " / ".join(log_parts) or "No Identifiers"
    try:
        if person_uuid:
            person = (
                session.query(Person)
                .filter(Person.uuid == str(person_uuid).upper())
                .first()
            )
            if person:
                logger.debug(
                    f"Found existing person by UUID: {person_uuid} (ID: {person.id})"
                )
                return person
        if person is None and person_profile_id:
            profile_id_upper = str(person_profile_id).upper()
            potential_matches = (
                session.query(Person)
                .filter(Person.profile_id == profile_id_upper)
                .all()
            )
            if not potential_matches:
                logger.debug(f"No person found by Profile ID: {profile_id_upper}.")
            elif len(potential_matches) == 1:
                person = potential_matches[0]
                logger.debug(
                    f"Found unique person by Profile ID: {profile_id_upper} (ID: {person.id})"
                )
                return person
            else:  # Multiple matches
                logger.warning(
                    f"Multiple ({len(potential_matches)}) people found for Profile ID: {profile_id_upper}. Disambiguating..."
                )
                if person_username:
                    username_lower = person_username.lower()
                    found_by_username = None
                    for p in potential_matches:
                        if p.username and p.username.lower() == username_lower:
                            if found_by_username is not None:
                                logger.error(
                                    f"CRITICAL: Found multiple people matching BOTH Profile ID {profile_id_upper} AND Username '{person_username}' (IDs: {found_by_username.id}, {p.id}). Cannot identify."
                                )
                                return None
                            found_by_username = p
                    if found_by_username:
                        logger.debug(
                            f"Disambiguated: Found Person ID {found_by_username.id} ('{found_by_username.username}') matching Profile ID {profile_id_upper}."
                        )
                        return found_by_username
                    else:
                        logger.warning(
                            f"Multiple matches for Profile ID {profile_id_upper}, but none matched username '{person_username}'. Cannot identify."
                        )
                        return None
                else:
                    logger.warning(
                        f"Multiple matches for Profile ID {profile_id_upper}, but no username provided. Cannot identify."
                    )
                    return None
        if person is None:
            logger.debug(f"No existing person reliably identified for {log_ref}.")
    except SQLAlchemyError as e:
        logger.error(f"DB error find_existing_person for {log_ref}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error find_existing_person for {log_ref}: {e}", exc_info=True
        )
        return None
    return person


# end of find_existing_person


def get_person_by_uuid(session: Session, uuid: str) -> Optional[Person]:
    """(Kept from user version) Retrieves a Person record based on their UUID."""
    if not uuid:
        logger.warning("get_person_by_uuid: UUID required.")
        return None
    try:
        return (
            session.query(Person)
            .options(joinedload(Person.dna_match), joinedload(Person.family_tree))
            .filter(Person.uuid == str(uuid).upper())
            .first()
        )
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
    existing_person: Optional[Person] = None,  # existing_person arg kept for signature
) -> Tuple[Optional[Person], Literal["created", "updated", "skipped", "error"]]:
    """
    (Kept structure from user file V9, adapted for new schema, fixed syntax)
    Creates a new Person or updates an existing one based on input data.
    - Relies on caller passing pre-fetched existing_person.
    - Uses create_person helper for insertion.
    - Performs inline update logic if existing_person provided.
    - Uses PersonStatusEnum.
    """
    uuid_val = person_data.get("uuid")
    profile_id_val = person_data.get("profile_id")
    username_val = person_data.get("username")

    if not uuid_val or not username_val:
        logger.error(
            f"Cannot create/update person for ProfileID='{profile_id_val or 'N/A'}', User='{username_val or 'N/A'}': UUID or Username missing."
        )
        return None, "error"

    log_ref = f"UUID={uuid_val} / ProfileID={profile_id_val or 'NULL'} / User='{username_val}'"
    updated = False  # Flag to track if any allowed Person field update occurred

    # Query for existing person based on UUID (as primary identifier now)
    # This overrides the passed-in existing_person to ensure we use the definitive lookup
    try:
        actual_existing_person = (
            session.query(Person).filter(Person.uuid == uuid_val.upper()).first()
        )
    except SQLAlchemyError as e:
        logger.error(
            f"DB error looking up person by UUID {uuid_val} in create_or_update_person: {e}"
        )
        return None, "error"

    try:
        if actual_existing_person:
            # --- PERSON EXISTS --- Update logic ---
            existing_person = actual_existing_person  # Use the one we just looked up
            person_id_for_logging = existing_person.id
            logger.debug(
                f"{log_ref}: Updating existing Person ID {person_id_for_logging}."
            )
            try:
                session.expire(existing_person)
            except Exception as expire_e:
                logger.warning(
                    f"Could not expire session state for Person ID {person_id_for_logging}: {expire_e}"
                )

            # Apply Updates to Restricted Fields
            person_update_needed = False
            # (Comparison logic largely kept from user version's logic)
            new_last_logged_in = person_data.get("last_logged_in")
            current_last_logged_in = existing_person.last_logged_in
            current_naive = None
            new_naive = None
            if isinstance(current_last_logged_in, datetime):
                db_aware = (
                    current_last_logged_in.tzinfo is not None
                    and current_last_logged_in.tzinfo.utcoffset(current_last_logged_in)
                    is not None
                )
                current_naive = (
                    current_last_logged_in.astimezone(timezone.utc).replace(
                        tzinfo=None, microsecond=0
                    )
                    if db_aware
                    else current_last_logged_in.replace(microsecond=0)
                )
            if isinstance(new_last_logged_in, datetime):
                new_naive = new_last_logged_in.astimezone(timezone.utc).replace(
                    tzinfo=None, microsecond=0
                )
            if current_naive != new_naive:
                existing_person.last_logged_in = new_last_logged_in
                updated = True
                person_update_needed = True

            new_contactable = person_data.get("contactable", False)
            current_contactable = existing_person.contactable
            if bool(current_contactable) != bool(new_contactable):
                existing_person.contactable = bool(new_contactable)
                updated = True
                person_update_needed = True

            new_birth_year = person_data.get("birth_year")
            current_birth_year = existing_person.birth_year
            if new_birth_year is not None and current_birth_year is None:
                try:
                    birth_year_int = int(new_birth_year)
                    existing_person.birth_year = birth_year_int
                    updated = True
                    person_update_needed = True
                except (ValueError, TypeError):
                    logger.warning(
                        f"Skipping birth_year update {log_ref}: Invalid integer '{new_birth_year}'."
                    )

            new_in_my_tree = bool(person_data.get("in_my_tree", False))
            current_in_my_tree = existing_person.in_my_tree
            if bool(current_in_my_tree) != new_in_my_tree:
                existing_person.in_my_tree = new_in_my_tree
                updated = True
                person_update_needed = True

            new_gender = person_data.get("gender")
            current_gender = existing_person.gender
            if new_gender is not None and current_gender is None:
                if isinstance(new_gender, str) and new_gender.lower() in ("f", "m"):
                    existing_person.gender = new_gender.lower()
                    updated = True
                    person_update_needed = True
                else:
                    logger.warning(
                        f"Skipping gender update {log_ref}: Invalid value '{new_gender}'."
                    )

            new_admin_id = person_data.get("administrator_profile_id")
            new_admin_user = person_data.get("administrator_username")
            current_admin_id = existing_person.administrator_profile_id
            current_admin_user = existing_person.administrator_username
            new_admin_id_upper = new_admin_id.upper() if new_admin_id else None
            if current_admin_id != new_admin_id_upper:
                existing_person.administrator_profile_id = new_admin_id_upper
                updated = True
                person_update_needed = True
            if current_admin_user != new_admin_user:
                existing_person.administrator_username = new_admin_user
                updated = True
                person_update_needed = True

            new_message_link = person_data.get("message_link")
            current_message_link = existing_person.message_link
            if not current_message_link and new_message_link:
                existing_person.message_link = new_message_link
                updated = True
                person_update_needed = True
            elif (
                current_message_link
                and new_message_link
                and current_message_link != new_message_link
            ):
                logger.debug(
                    f"Skipping message_link update for {log_ref} (existing present)"
                )

            new_username = person_data.get("username")
            current_username = existing_person.username
            if not current_username and new_username:
                existing_person.username = new_username
                updated = True
                person_update_needed = True
            elif current_username and new_username and current_username != new_username:
                logger.debug(
                    f"Skipping username update for {log_ref} (existing present and different)"
                )

            if updated:
                existing_person.updated_at = datetime.now(timezone.utc)
                return existing_person, "updated"
            else:
                return existing_person, "skipped"
        else:
            # --- PERSON DOES NOT EXIST --- Create new logic ---
            logger.debug(f"{log_ref}: Creating new Person.")
            create_status_code = create_person(session, person_data)
            if create_status_code > 0:  # create_person returns ID on success
                new_person_obj = session.get(
                    Person, create_status_code
                )  # Get the created object
                if new_person_obj:
                    return new_person_obj, "created"
                else:
                    logger.error(
                        f"Failed fetch created person {log_ref} ID {create_status_code}."
                    )
                    session.rollback() if session.is_active else None
                    return None, "error"
            else:
                logger.error(f"create_person failed for {log_ref}.")
                return None, "error"

    # --- Corrected Syntax for Exception Blocks ---
    except IntegrityError as ie:
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
    except NameError as ne:
        if session.is_active:
            session.rollback()
        logger.critical(
            f"NameError processing person {log_ref}: {ne}. Imports?", exc_info=True
        )
        return None, "error"
    except TypeError as te:
        if session.is_active:
            session.rollback()
        logger.critical(
            f"TypeError during person update {log_ref}: {te}", exc_info=True
        )
        return None, "error"
    except Exception as e:
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
    """(Kept from user version, adapted status update) Updates existing Person."""
    if not profile_id or not username:
        logger.warning("update_person: profile_id and username required.")
        return False
    try:
        person = (
            session.query(Person)
            .filter_by(profile_id=profile_id.upper(), username=username)
            .first()
        )
        if not person:
            logger.warning(f"update_person: Person {profile_id}/{username} not found.")
            return False
        updated = False
        allowed_fields = [
            "uuid",
            "profile_id",
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
        ]
        for key, value in update_data.items():
            if key in allowed_fields and hasattr(person, key):
                current_value = getattr(person, key)
                value_to_compare = value
                if key in ("profile_id", "administrator_profile_id", "uuid") and value:
                    value_to_compare = value.upper()

                # Correctly handle status enum comparison and assignment
                if key == "status":
                    enum_value = None
                    try:
                        if isinstance(value_to_compare, PersonStatusEnum):
                            enum_value = value_to_compare
                        elif isinstance(value_to_compare, str):
                            enum_value = PersonStatusEnum(value_to_compare.lower())
                        else:
                            logger.warning(
                                f"Invalid type for status update: {type(value_to_compare)}"
                            )
                            continue

                        if current_value != enum_value:
                            setattr(person, key, enum_value)
                            updated = True
                    except ValueError:
                        logger.warning(
                            f"Invalid status value '{value_to_compare}' for update on Person ID {person.id}."
                        )
                # Datetime comparison (already handled timezone conversion in create_or_update)
                elif isinstance(current_value, datetime) and isinstance(
                    value_to_compare, datetime
                ):
                    current_naive = (
                        current_value.astimezone(timezone.utc)
                        if current_value.tzinfo
                        else current_value
                    ).replace(tzinfo=None, microsecond=0)
                    new_naive = (
                        value_to_compare.astimezone(timezone.utc)
                        if value_to_compare.tzinfo
                        else value_to_compare
                    ).replace(tzinfo=None, microsecond=0)
                    if current_naive != new_naive:
                        setattr(person, key, value_to_compare)
                        updated = True
                # Standard comparison for other types
                elif current_value != value_to_compare:
                    setattr(person, key, value_to_compare)
                    updated = True
            elif key not in allowed_fields:
                logger.warning(
                    f"Attempted update non-allowed attr '{key}' on Person ID {person.id}."
                )

        if updated:
            person.updated_at = datetime.now(timezone.utc)
            session.flush()
            logger.info(f"Updated person {profile_id}/{username} (ID: {person.id}).")
        else:
            logger.debug(f"No update needed for person {profile_id}/{username}.")
        return True
    except IntegrityError as ie:
        session.rollback()
        logger.error(
            f"IntegrityError updating person {profile_id}/{username}: {ie}.",
            exc_info=False,
        )
        return False
    except SQLAlchemyError as e:
        logger.error(
            f"DB error updating person {profile_id}/{username}: {e}", exc_info=True
        )
        session.rollback()
        return False
    except Exception as e:
        logger.critical(
            f"Unexpected error update_person {profile_id}/{username}: {e}",
            exc_info=True,
        )
        session.rollback() if session.is_active else None
        return False


# End of update_person

# ----------------------------------------------------------------------
# Delete (Kept functions from user version)
# ----------------------------------------------------------------------


def delete_person(session: Session, profile_id: str, username: str) -> bool:
    """Deletes a person and associated records using profile_id and username."""
    if not profile_id or not username:
        logger.warning("delete_person: profile_id and username required.")
        return False
    try:
        person = (
            session.query(Person)
            .filter_by(profile_id=profile_id.upper(), username=username)
            .first()
        )
        if not person:
            logger.warning(f"delete_person: Person {profile_id}/{username} not found.")
            return False
        person_id = person.id
        session.delete(person)
        session.flush()
        logger.info(
            f"Deleted person ID {person_id} ({profile_id}/{username}) and related records."
        )
        return True
    except SQLAlchemyError as e:
        logger.error(
            f"DB error deleting person {profile_id}/{username}: {e}", exc_info=True
        )
        session.rollback()
        return False
    except Exception as e:
        logger.critical(
            f"Unexpected error delete_person {profile_id}/{username}: {e}",
            exc_info=True,
        )
        session.rollback() if session.is_active else None
        return False


# End of delete_person


def delete_database(
    session_manager: Optional[Any], db_path: Path, max_attempts: int = 5
):
    """Deletes the database file with retry and cleanup."""
    # ...(Logic kept from user version - appears correct)...
    if not isinstance(db_path, Path):
        try:
            db_path = Path(db_path)
            logger.warning("Converted db_path string to Path object.")
        except TypeError:
            logger.error(f"Cannot convert db_path {db_path} to Path object.")
            return
    logger.debug(f"Attempting to delete database file: {db_path}")
    last_error = None
    for attempt in range(max_attempts):
        logger.debug(f"Delete attempt {attempt + 1}/{max_attempts}...")
        try:
            logger.debug("Running GC before delete attempt...")
            gc.collect()
            time.sleep(0.5)
            gc.collect()
            time.sleep(1.0 + attempt)
            if db_path.exists():
                logger.debug(f"Attempting os.remove on {db_path}...")
                os.remove(db_path)
                time.sleep(0.1)
                if not db_path.exists():
                    logger.info(f"'{db_path}' deleted successfully.")
                    return
                else:
                    logger.warning(
                        f"os.remove called, but file '{db_path}' still exists."
                    )
                    last_error = OSError(
                        f"File exists after os.remove attempt {attempt + 1}"
                    )
            else:
                logger.info(f"Database '{db_path}' does not exist.")
                return
        except PermissionError as e:
            logger.warning(
                f"PermissionError deleting '{db_path}' (Attempt {attempt + 1}): {e}. File locked?"
            )
            last_error = e
        except OSError as e:
            if hasattr(e, "winerror") and e.winerror == 32:
                logger.warning(
                    f"OSError (WinError 32) deleting '{db_path}' (Attempt {attempt + 1}): {e}. File locked?"
                )
            else:
                logger.error(
                    f"OSError deleting '{db_path}' (Attempt {attempt + 1}): {e}",
                    exc_info=True,
                )
            last_error = e
        except Exception as e:
            logger.critical(
                f"Unexpected error during delete attempt {attempt + 1} for '{db_path}': {e}",
                exc_info=True,
            )
            last_error = e
        if attempt < max_attempts - 1:
            wait_time = 2**attempt
            logger.info(f"Waiting {wait_time} seconds before next delete attempt...")
            time.sleep(wait_time)
        else:
            logger.critical(
                f"Failed to delete database '{db_path}' after {max_attempts} attempts. Last error: {last_error or 'Unknown'}"
            )
            raise last_error or OSError(f"Failed to delete {db_path}")
    logger.error(f"Exited delete_database loop unexpectedly for {db_path}.")


# End of delete_database

# ----------------------------------------------------------------------
# Backup and Recovery (Kept functions from user version)
#################################################################################


def backup_database(session_manager=None):
    """Backs up the database file specified in config_instance."""
    db_path = config_instance.DATABASE_FILE
    backup_dir = config_instance.DATA_DIR
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / "ancestry_backup.db"
    try:
        if db_path.exists():
            shutil.copy2(db_path, backup_path)
            logger.info(f"Backed up to '{backup_path}' OK.")
        else:
            logger.warning(f"Database file '{db_path}' not found. No backup.")
    except Exception as e:
        logger.error(
            f"Error backing up DB from '{db_path}' to '{backup_path}': {e}",
            exc_info=True,
        )


# End of backup_database

# ----------------------------------------------------------------------
# Standalone execution (Updated for new schema)
#################################################################################
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname).3s [%(name)-12s %(lineno)-4d] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    standalone_logger = logging.getLogger("db_standalone")
    standalone_logger.info(
        "--- Starting database.py standalone test (Merged: New Schema + Old Functions) ---"
    )
    try:
        db_path_obj = config_instance.DATABASE_FILE
        db_path_str = str(db_path_obj.resolve())
    except Exception as config_err:
        standalone_logger.critical(f"CRITICAL: Error getting DB path: {config_err}.")
        sys.exit(1)
    standalone_logger.info(f"Using database file: {db_path_str}")
    engine = None
    try:
        engine = create_engine(f"sqlite:///{db_path_str}", echo=False)

        @event.listens_for(engine, "connect")
        def enable_foreign_keys(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
            finally:
                cursor.close()

        # Explicitly Drop ConversationLog Table Before Creation
        try:
            with engine.connect() as connection:
                with connection.begin():
                    standalone_logger.warning(
                        "Attempting to drop 'conversation_log' table if it exists..."
                    )
                    connection.execute(text("DROP TABLE IF EXISTS conversation_log"))
                    standalone_logger.info(
                        "'conversation_log' table dropped (if existed)."
                    )
                    # NOTE: If other tables related to old schema (InboxStatus, MessageHistory) existed,
                    # they would ideally be dropped here too for a clean slate.
        except Exception as drop_err:
            standalone_logger.error(
                f"Error during explicit table drop: {drop_err}", exc_info=True
            )

        standalone_logger.info(
            "Creating/Verifying database tables using current schema..."
        )
        Base.metadata.create_all(
            engine
        )  # Create tables based on current models (ConversationLog, etc.)
        standalone_logger.info(f"Database tables OK: {db_path_str}")

        standalone_logger.info("Seeding MessageType table...")
        SessionSeed = sessionmaker(bind=engine)
        seed_session = None
        try:
            seed_session = SessionSeed()
            script_dir = Path(__file__).resolve().parent
            messages_file = script_dir / "messages.json"
            if messages_file.exists():
                with messages_file.open("r", encoding="utf-8") as f:
                    messages_data = json.load(f)
                if isinstance(messages_data, dict):
                    with db_transn(seed_session) as sess:
                        types_to_add = [
                            MessageType(type_name=name)
                            for name in messages_data
                            if not sess.query(MessageType.id)
                            .filter_by(type_name=name)
                            .first()
                        ]
                        if types_to_add:
                            sess.add_all(types_to_add)
                            standalone_logger.debug(
                                f"Added {len(types_to_add)} msg types."
                            )
                        else:
                            standalone_logger.debug("Message types already exist.")
                    count = seed_session.query(func.count(MessageType.id)).scalar() or 0
                    standalone_logger.info(
                        f"MessageType seeding OK. Total types: {count}"
                    )
                else:
                    standalone_logger.error("'messages.json' incorrect format.")
            else:
                standalone_logger.warning(f"'messages.json' not found.")
        except Exception as seed_err:
            standalone_logger.error(
                f"Error seeding MessageType: {seed_err}", exc_info=True
            )
        finally:
            if seed_session:
                seed_session.close()
    except Exception as e:
        standalone_logger.critical(f"Unexpected error during setup: {e}", exc_info=True)
    finally:
        if engine:
            engine.dispose()
            standalone_logger.debug("SQLAlchemy engine disposed.")
        standalone_logger.info("--- Database.py standalone test finished ---")
        sys.exit(0)
# End of database.py
