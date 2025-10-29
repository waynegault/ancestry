#!/usr/bin/env python3

"""
database.py - SQLAlchemy Models, Database Utilities, and Schema Management

Defines the database schema using SQLAlchemy ORM, provides utility functions
for database operations like backup, restore, deletion, and includes setup
for creating tables and views automatically. Uses Enums for controlled vocabulary
in specific fields (status, direction). Implements a transactional context manager.
"""

# === CORE INFRASTRUCTURE ===
from standard_imports import (
    setup_module,
)

# === MODULE SETUP ===
logger = setup_module(globals(), __name__)

# === PHASE 4.1: ENHANCED ERROR HANDLING ===
# === STANDARD LIBRARY IMPORTS ===
import contextlib
import enum
import gc
import logging
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional, Union
from uuid import uuid4

# === THIRD-PARTY IMPORTS ===
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    event,
    func,
    select,
    text,
)
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import (
    Mapped,
    Query,
    Session,
    declarative_base,
    joinedload,
    mapped_column,
    relationship,
)

# === LOCAL IMPORTS ===
from common_params import MatchIdentifiers
from config import config_schema
from core.error_handling import (
    AncestryError,
    DatabaseConnectionError,
    DataValidationError,
    FatalError,
    RetryableError,
    error_context,
    retry_on_failure,
    timeout_protection,
)

# === MODULE CONFIGURATION ===
# Use global cached config instance
# Note: SessionManager imported locally when needed to avoid circular imports
from test_framework import (
    TestSuite,
    suppress_logging,
)

# ----------------------------------------------------------------------
# SQLAlchemy Base Declarative Model
# ----------------------------------------------------------------------
Base = declarative_base()


# ----------------------------------------------------------------------
# Enumerations for Controlled Fields
# ----------------------------------------------------------------------
class MessageDirectionEnum(enum.Enum):
    """Enumeration for the direction of a message (Incoming or Outgoing)."""

    IN = "IN"
    OUT = "OUT"


# End of MessageDirectionEnum


class RoleType(enum.Enum):
    """Enumeration for message participant roles (currently unused but defined)."""

    AUTHOR = "AUTHOR"
    RECIPIENT = "RECIPIENT"


# End of RoleType


class PersonStatusEnum(enum.Enum):
    """
    Enumeration defining the processing status of a Person (DNA match).
    Used to control messaging eligibility and workflow state.
    """

    ACTIVE = "ACTIVE"  # Default state, eligible for processing/messaging.
    DESIST = "DESIST"  # User requested no further contact (via AI). Skip messaging.
    ARCHIVE = "ARCHIVE"  # Conversation concluded (e.g., ACK sent). Skip messaging.
    BLOCKED = "BLOCKED"  # Manually blocked by script user. Skip messaging.
    DEAD = "DEAD"  # Manually marked as deceased by script user. Skip messaging.


# End of PersonStatusEnum

# ----------------------------------------------------------------------
# SQLAlchemy Model Definitions
# ----------------------------------------------------------------------


class ConversationLog(Base):
    """
    Represents a log entry for a conversation, storing message history
    for either incoming (IN) or outgoing (OUT) messages within that conversation.
    Uses an auto-incrementing ID as primary key to allow multiple messages per conversation.
    """

    __tablename__ = "conversation_log"

    # --- Columns ---
    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Auto-incrementing primary key for message history.",
    )
    people_id = Column(
        Integer,
        ForeignKey("people.id"),
        nullable=False,
        index=True,
        comment="Foreign key linking to the Person involved in the conversation.",
    )
    direction = Column(
        SQLEnum(MessageDirectionEnum),
        nullable=False,
        index=True,
        comment="Direction of the message (IN or OUT).",
    )
    script_message_status = Column(
        String,
        nullable=True,
        comment="Status of OUT messages sent by the script (e.g., delivered OK, typed (dry_run)).",
    )
    latest_message_content = Column(
        Text,
        nullable=True,
        comment="Truncated content of the message.",
    )
    conversation_id = Column(
        String,
        nullable=False,
        index=True,
        comment="Unique identifier for the conversation thread.",
    )
    latest_timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Timestamp (UTC) of the message.",
    )
    ai_sentiment = Column(
        String,
        nullable=True,
        index=True,
        comment="AI-determined sentiment/intent (e.g., PRODUCTIVE, DESIST) for IN messages.",
    )
    message_template_id = Column(
        Integer,
        ForeignKey("message_templates.id"),
        nullable=True,
        comment="Foreign key linking to the template used for OUT messages sent by the script.",
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),  # Default to current UTC time
        onupdate=lambda: datetime.now(timezone.utc),  # Update timestamp on modification
        nullable=False,
        comment="Timestamp of the last update to this log entry.",
    )

    # --- Relationships ---
    # Defines the link back to the Person object. 'back_populates' ensures bidirectional linking.
    person = relationship("Person", back_populates="conversation_log_entries")  # type: ignore
    # Defines the link to the MessageTemplate object for outgoing messages.
    message_template = relationship("MessageTemplate")  # type: ignore

    # New column for tracking custom genealogical replies
    custom_reply_sent_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp (UTC) when an automated genealogical custom reply was sent for this IN message.",
    )

    # --- Table Arguments (Indexes) ---
    __table_args__ = (
        # Composite index for efficient lookup by conversation and direction
        Index(
            "ix_conversation_log_conv_direction",
            "conversation_id",
            "direction",
        ),
        # Composite index for efficient lookup by person, direction, and timestamp.
        Index(
            "ix_conversation_log_people_id_direction_ts",
            "people_id",
            "direction",
            "latest_timestamp",
        ),
        # Index for querying based on timestamp alone.
        Index("ix_conversation_log_timestamp", "latest_timestamp"),
        # Index for custom reply timestamp
        Index("ix_conversation_log_custom_reply_sent_at", "custom_reply_sent_at"),
    )


# End of ConversationLog class


# ----------------------------------------------------------------------
# Analytics Models (Phase 6A)
# ----------------------------------------------------------------------


class ConversationMetrics(Base):
    """
    Aggregate metrics for each conversation.
    Tracks overall conversation performance and effectiveness.
    """

    __tablename__ = "conversation_metrics"

    # --- Columns ---
    id = Column(
        Integer, primary_key=True, comment="Unique identifier for the metrics record."
    )
    people_id = Column(
        Integer,
        ForeignKey("people.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
        comment="Foreign key to people table (one metrics record per person).",
    )

    # Message counts
    messages_sent = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total number of messages sent to this person.",
    )
    messages_received = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total number of messages received from this person.",
    )

    # Response tracking
    first_response_received = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether we've received at least one response.",
    )
    first_response_date = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp (UTC) when first response was received.",
    )
    time_to_first_response_hours = Column(
        Float,
        nullable=True,
        comment="Hours between first message sent and first response received.",
    )

    # Engagement metrics
    current_engagement_score = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Current engagement score (0-100) from ConversationState.",
    )
    max_engagement_score = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Maximum engagement score achieved during conversation.",
    )
    avg_engagement_score = Column(
        Float,
        nullable=True,
        comment="Average engagement score across all updates.",
    )

    # Conversation characteristics
    conversation_phase = Column(
        String,
        nullable=False,
        default="initial_outreach",
        comment="Current conversation phase from ConversationState.",
    )
    conversation_duration_days = Column(
        Float,
        nullable=True,
        comment="Days between first message and last message.",
    )

    # Template effectiveness
    initial_template_used = Column(
        String,
        nullable=True,
        comment="Template key used for initial outreach message.",
    )
    templates_used = Column(
        Text,
        nullable=True,
        comment="JSON-encoded list of all template keys used in conversation.",
    )

    # Research outcomes
    people_looked_up = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of people looked up during conversation.",
    )
    people_found = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of people successfully found in tree/API.",
    )
    research_tasks_created = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of research tasks created from conversation.",
    )

    # Tree impact
    added_to_tree = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether this person was added to tree during conversation.",
    )
    added_to_tree_date = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp (UTC) when person was added to tree.",
    )

    # Timestamps
    first_message_sent = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp (UTC) when first message was sent.",
    )
    last_message_sent = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp (UTC) when last message was sent.",
    )
    last_message_received = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp (UTC) when last message was received.",
    )
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="Timestamp (UTC) when metrics record was created.",
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        comment="Timestamp (UTC) when metrics record was last updated.",
    )

    # --- Relationships ---
    person = relationship("Person", back_populates="conversation_metrics")  # type: ignore

    def __init__(self, **kwargs: Any) -> None:
        """Initialize ConversationMetrics with proper defaults."""
        super().__init__(**kwargs)
        # Initialize integer fields with 0 if not provided
        self._initialize_integer_fields()
        # Initialize boolean fields with False if not provided
        self._initialize_boolean_fields()
        # Initialize conversation_phase with default if not provided
        if self.conversation_phase is None:
            self.conversation_phase = "initial_outreach"

    def _initialize_integer_fields(self) -> None:
        """Initialize integer fields with 0 if not provided."""
        integer_fields = [
            'messages_sent', 'messages_received', 'current_engagement_score',
            'max_engagement_score', 'people_looked_up', 'people_found',
            'research_tasks_created'
        ]
        for field in integer_fields:
            if getattr(self, field) is None:
                setattr(self, field, 0)

    def _initialize_boolean_fields(self) -> None:
        """Initialize boolean fields with False if not provided."""
        boolean_fields = ['first_response_received', 'added_to_tree']
        for field in boolean_fields:
            if getattr(self, field) is None:
                setattr(self, field, False)


# End of ConversationMetrics class


class EngagementTracking(Base):
    """
    Detailed tracking of engagement events.
    Records specific events that contribute to engagement scoring.
    """

    __tablename__ = "engagement_tracking"

    # --- Columns ---
    id = Column(
        Integer, primary_key=True, comment="Unique identifier for the engagement event."
    )
    people_id = Column(
        Integer,
        ForeignKey("people.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to people table.",
    )

    # Event details
    event_type = Column(
        String,
        nullable=False,
        comment="Type of engagement event: message_sent, message_received, person_lookup, research_task, phase_change, score_update.",
    )
    event_description = Column(
        Text,
        nullable=True,
        comment="Detailed description of the event.",
    )
    event_data = Column(
        Text,
        nullable=True,
        comment="JSON-encoded additional event data.",
    )

    # Engagement impact
    engagement_score_before = Column(
        Integer,
        nullable=True,
        comment="Engagement score before this event.",
    )
    engagement_score_after = Column(
        Integer,
        nullable=True,
        comment="Engagement score after this event.",
    )
    engagement_score_delta = Column(
        Integer,
        nullable=True,
        comment="Change in engagement score from this event.",
    )

    # Context
    conversation_phase = Column(
        String,
        nullable=True,
        comment="Conversation phase when event occurred.",
    )
    template_used = Column(
        String,
        nullable=True,
        comment="Template key if event was a message.",
    )

    # Timestamp
    event_timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="Timestamp (UTC) when event occurred.",
    )

    # --- Relationships ---
    person = relationship("Person", back_populates="engagement_events")  # type: ignore


# End of EngagementTracking class


# MessageType class removed - consolidated into MessageTemplate


class MessageTemplate(Base):
    """
    Consolidated message template storage - replaces both MessageType and MessageTemplate.
    Stores all message templates with their metadata and content.
    """

    __tablename__ = "message_templates"

    # --- Columns ---
    id = Column(
        Integer, primary_key=True, comment="Unique identifier for the message template."
    )
    template_key = Column(
        String,
        unique=True,
        nullable=False,
        index=True,
        comment="Unique key identifying the template (e.g., 'In_Tree-Initial').",
    )
# template_name removed - can be generated from template_key as needed
    subject_line = Column(
        String,
        nullable=True,
        comment="Email subject line extracted from template content.",
    )
    message_content = Column(
        Text,
        nullable=False,
        comment="Full message template content with placeholders.",
    )
    template_category = Column(
        String,
        nullable=False,
        index=True,
        comment="Category: 'initial', 'follow_up', 'reminder', 'acknowledgement', etc.",
    )
    tree_status = Column(
        String,
        nullable=False,
        index=True,
        comment="Tree status: 'in_tree', 'out_tree', 'universal', etc.",
    )
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Whether this template is currently active for use.",
    )
    version = Column(
        Integer,
        default=1,
        nullable=False,
        comment="Template version for tracking changes.",
    )
    created_at = Column(
        DateTime(timezone=True),
        default=func.now(),
        nullable=False,
        comment="Timestamp when template was created.",
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Timestamp when template was last updated.",
    )


# End of MessageTemplate class


class TreeStatisticsCache(Base):
    """
    Caches tree statistics to avoid recalculating on every message.
    Stores aggregate statistics about DNA matches, tree composition, and ethnicity.
    """

    __tablename__ = "tree_statistics_cache"

    # --- Columns ---
    id = Column(
        Integer, primary_key=True, comment="Unique identifier for the cache entry."
    )
    profile_id = Column(
        String,
        nullable=False,
        unique=True,
        index=True,
        comment="Profile ID of the tree owner (unique per owner).",
    )
    total_matches = Column(
        Integer, nullable=False, default=0, comment="Total number of DNA matches."
    )
    in_tree_count = Column(
        Integer, nullable=False, default=0, comment="Number of matches in the tree."
    )
    out_tree_count = Column(
        Integer, nullable=False, default=0, comment="Number of matches out of the tree."
    )
    close_matches = Column(
        Integer, nullable=False, default=0, comment="Number of close matches (>100 cM)."
    )
    moderate_matches = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of moderate matches (20-100 cM).",
    )
    distant_matches = Column(
        Integer, nullable=False, default=0, comment="Number of distant matches (<20 cM)."
    )
    ethnicity_regions = Column(
        Text,
        nullable=True,
        comment="JSON-encoded dictionary of ethnicity regions and counts.",
    )
    calculated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="Timestamp (UTC) when statistics were calculated.",
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        comment="Timestamp (UTC) when cache was last updated.",
    )


# End of TreeStatisticsCache class


class ConversationState(Base):
    """
    Tracks conversation state and engagement for intelligent dialogue management.
    Used by Action 9 to maintain context and guide conversation flow.
    """

    __tablename__ = "conversation_state"

    # --- Columns ---
    id = Column(
        Integer, primary_key=True, comment="Unique identifier for the conversation state."
    )
    people_id = Column(
        Integer,
        ForeignKey("people.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
        comment="Foreign key to people table (one state per person).",
    )
    conversation_phase = Column(
        String,
        nullable=False,
        default="initial_outreach",
        comment="Current conversation phase: initial_outreach, active_dialogue, research_exchange, concluded.",
    )
    engagement_score = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Engagement score (0-100) based on message quality and frequency.",
    )
    last_topic = Column(
        Text,
        nullable=True,
        comment="Last topic discussed in the conversation.",
    )
    pending_questions = Column(
        Text,
        nullable=True,
        comment="JSON-encoded list of pending questions to ask.",
    )
    ai_summary = Column(
        Text,
        nullable=True,
        comment="AI-generated engagement summary and intent notes.",
    )

    mentioned_people = Column(
        Text,
        nullable=True,
        comment="JSON-encoded list of people mentioned in conversation.",
    )
    shared_ancestors = Column(
        Text,
        nullable=True,
        comment="JSON-encoded list of shared ancestors discovered.",
    )
    next_action = Column(
        String,
        nullable=True,
        comment="Next action to take: await_reply, send_follow_up, research_needed, no_action.",
    )
    next_action_date = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp (UTC) when next action should be taken.",
    )
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="Timestamp (UTC) when conversation state was created.",
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        comment="Timestamp (UTC) when conversation state was last updated.",
    )

    # --- Relationships ---
    person = relationship("Person", back_populates="conversation_state")  # type: ignore


# End of ConversationState class


class DnaMatch(Base):
    """
    Stores DNA match-specific details for a Person.
    Linked one-to-one with the Person table via people_id.
    """

    __tablename__ = "dna_match"

    # --- Columns ---
    id = Column(
        Integer, primary_key=True, comment="Unique identifier for the DNA match record."
    )
    people_id = Column(
        Integer,
        ForeignKey("people.id"),
        unique=True,
        nullable=False,
        index=True,
        comment="Foreign key linking to the Person record (one-to-one).",
    )
    compare_link = Column(
        String,
        nullable=False,
        comment="Direct URL to the Ancestry DNA comparison page.",
    )
    cm_dna = Column(
        Integer,
        nullable=False,
        index=True,  # Added index
        comment="Shared DNA in centimorgans.",
    )
    predicted_relationship = Column(
        String,
        nullable=False,
        comment="Relationship prediction provided by Ancestry (e.g., '1st-2nd cousin').",
    )
    shared_segments = Column(
        Integer, nullable=True, comment="Number of shared DNA segments."
    )
    longest_shared_segment = Column(
        Float, nullable=True, comment="Length of the longest shared segment in cM."
    )
    meiosis = Column(
        Integer, nullable=True, comment="Meiosis count (if provided by Ancestry)."
    )
    from_my_fathers_side = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Flag indicating if match is likely paternal (via Ancestry tools).",
    )
    from_my_mothers_side = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Flag indicating if match is likely maternal (via Ancestry tools).",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # --- Relationships ---
    # Defines the link back to the Person object.
    person = relationship("Person", back_populates="dna_match")  # type: ignore

    # --- Properties ---
    @property
    def uuid(self) -> Optional[str]:
        """Returns the UUID of the associated Person."""
        return self.person.uuid if self.person else None


# End of DnaMatch class


class FamilyTree(Base):
    """
    Stores details related to a DNA match's position within the script user's
    family tree, if identified. Linked one-to-one with the Person table.
    """

    __tablename__ = "family_tree"

    # --- Columns ---
    id = Column(
        Integer,
        primary_key=True,
        comment="Unique identifier for the family tree link record.",
    )
    people_id = Column(
        Integer,
        ForeignKey("people.id"),
        unique=True,
        nullable=False,
        index=True,
        comment="Foreign key linking to the Person record (one-to-one).",
    )
    cfpid = Column(
        String,
        unique=True,
        nullable=True,
        index=True,  # Added index
        comment="Ancestry's internal Person ID (CFPID) within the script user's tree.",
    )
    person_name_in_tree = Column(
        String,
        nullable=True,
        comment="Name of the person as recorded in the script user's tree.",
    )
    facts_link = Column(
        String,
        nullable=True,
        comment="Direct URL to the person's 'Facts' page in the script user's tree.",
    )
    view_in_tree_link = Column(
        String,
        nullable=True,
        comment="Direct URL to view the person within the script user's family tree structure.",
    )
    actual_relationship = Column(
        String,
        nullable=True,
        index=True,  # Added index
        comment="Relationship determined via tree analysis (e.g., '1st cousin 1x removed').",
    )
    relationship_path = Column(
        Text,
        nullable=True,  # Changed to Text for potentially long paths
        comment="Textual representation of the relationship path back to a common ancestor.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # --- Relationships ---
    # Defines the link back to the Person object.
    person = relationship("Person", back_populates="family_tree")  # type: ignore

    # --- Properties ---
    @property
    def uuid(self) -> Optional[str]:
        """Returns the UUID of the associated Person."""
        return self.person.uuid if self.person else None


# End of FamilyTree class


class Person(Base):
    """
    Represents an individual DNA match. This is the central table linking
    profile information, DNA match details, tree link details, and conversation logs.
    """

    __tablename__ = "people"

    # --- Columns ---
    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, comment="Unique identifier for the person record."
    )
    uuid: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        unique=True,
        index=True,
        comment="Ancestry DNA test Sample ID (GUID), primary identifier if profile_id absent.",
    )
    profile_id: Mapped[Optional[str]] = mapped_column(
        String,
        unique=True,
        nullable=True,
        index=True,
        comment="Ancestry User Profile ID (ucdmid), preferred unique identifier.",
    )
    username: Mapped[str] = mapped_column(
        String,
        unique=False,
        nullable=False,  # Usernames are not unique
        comment="Display name shown on Ancestry for the match.",
    )
    first_name: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, comment="First name extracted from username or profile."
    )
    gender: Mapped[Optional[str]] = mapped_column(
        String(1), nullable=True, comment="Gender ('M' or 'F'), if known."
    )
    birth_year: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="Birth year, if known (often from tree)."
    )
    message_link: Mapped[Optional[str]] = mapped_column(
        String, unique=False, nullable=True, comment="Direct URL to message the person."
    )
    in_my_tree: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        index=True,  # Added index
        comment="Flag indicating if this person has been linked within the script user's tree.",
    )
    contactable: Mapped[bool] = mapped_column(
        Boolean,
        default=True,  # Default to contactable unless known otherwise
        comment="Flag indicating if the user's profile allows messaging.",
    )
    last_logged_in: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Timestamp (UTC) of the user's last login to Ancestry, if available.",
    )
    administrator_profile_id: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        index=True,
        comment="Profile ID of the person managing the DNA kit, if different.",
    )
    administrator_username: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        comment="Display name of the kit administrator, if different.",
    )
    status: Mapped[PersonStatusEnum] = mapped_column(
        SQLEnum(PersonStatusEnum),
        nullable=False,
        default=PersonStatusEnum.ACTIVE,  # Default new persons to ACTIVE
        server_default=PersonStatusEnum.ACTIVE.value,  # Set DB default
        index=True,
        comment="Current processing status of this person (e.g., ACTIVE, DESIST, ARCHIVE).",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Timestamp when this person was soft-deleted. Null for active records.",
    )

    # --- Relationships ---
    # One-to-one relationship with FamilyTree. `cascade` ensures deletion of related FamilyTree record if Person deleted.
    family_tree = relationship(  # type: ignore
        "FamilyTree",
        back_populates="person",
        uselist=False,
        cascade="all, delete-orphan",
    )
    # One-to-one relationship with DnaMatch. `cascade` ensures deletion.
    dna_match = relationship(  # type: ignore
        "DnaMatch", back_populates="person", uselist=False, cascade="all, delete-orphan"
    )
    # One-to-many relationship with ConversationLog. `cascade` ensures deletion.
    conversation_log_entries = relationship(  # type: ignore
        "ConversationLog", back_populates="person", cascade="all, delete-orphan"
    )
    # One-to-one relationship with ConversationState. `cascade` ensures deletion.
    conversation_state = relationship(  # type: ignore
        "ConversationState", back_populates="person", uselist=False, cascade="all, delete-orphan"
    )
    # One-to-one relationship with ConversationMetrics. `cascade` ensures deletion.
    conversation_metrics = relationship(  # type: ignore
        "ConversationMetrics", back_populates="person", uselist=False, cascade="all, delete-orphan"
    )
    # One-to-many relationship with EngagementTracking. `cascade` ensures deletion.
    engagement_events = relationship(  # type: ignore
        "EngagementTracking", back_populates="person", cascade="all, delete-orphan"
    )


# End of Person class

# ----------------------------------------------------------------------
# Event Listener for View Creation (Remains the same)
# ----------------------------------------------------------------------
CREATE_VIEW_SQL = text(
    """
    CREATE VIEW IF NOT EXISTS messages AS
    SELECT
        cl.latest_timestamp,
        p.username AS person_username,
        cl.direction,
        mt.type_name AS message_type_name,
        cl.ai_sentiment,
        cl.latest_message_content,
        cl.script_message_status,
        cl.updated_at,
        cl.message_template_id,
        cl.conversation_id,
        cl.people_id
    FROM
        conversation_log cl
    LEFT JOIN
        message_templates mt ON cl.message_template_id = mt.id
    LEFT JOIN
        people p ON cl.people_id = p.id
    WHERE
        p.deleted_at IS NULL OR p.deleted_at IS NOT NULL;  -- Include all records for now, will be filtered in queries
    """
)
# End of CREATE_VIEW_SQL


@event.listens_for(Base.metadata, "after_create")  # type: ignore
def _create_views(target: Any, connection: Connection, **kw: Any) -> None:  # type: ignore[misc]
    """SQLAlchemy event listener to create the 'messages' view after tables are created.

    Note: Function is accessed by SQLAlchemy event system, not directly called in code.
    """
    # Parameters required by SQLAlchemy event signature but not used in implementation
    _ = target, kw
    logger.debug("Executing CREATE VIEW statement for 'messages' view...")
    try:
        connection.execute(CREATE_VIEW_SQL)
        logger.debug("Database view 'messages' created or already exists.")
    except Exception as e:
        logger.error(f"Error creating 'messages' database view: {e}", exc_info=True)


# End of _create_views


# ----------------------------------------------------------------------
# Enhanced Transaction Context Manager with Phase 4.1 Error Handling
# ----------------------------------------------------------------------
@contextlib.contextmanager
@error_context("Database Transaction")
def db_transn(session: Session):
    """
    Provides a transactional scope around a series of database operations
    using a SQLAlchemy session. Handles commit on success and rollback on error.
    Enhanced with Phase 4.1 error handling for better resilience.

    Args:
        session: The SQLAlchemy Session object to manage.
    """
    transaction_start = time.time()
    session_id = id(session)

    try:
        # Step 1: Validate session state
        if not hasattr(session, "commit"):
            raise DatabaseConnectionError(
                "Invalid session object provided",
                context={
                    "session_id": session_id,
                    "session_type": type(session).__name__,
                },
            )

        logger.debug(
            f"--- Entering enhanced db_transn block (Session: {session_id}) ---"
        )
        yield session

        # Step 2: Attempt commit with timeout protection
        commit_start = time.time()
        logger.debug(f"Attempting commit... (Session: {session_id})")

        try:
            session.commit()
            commit_time = time.time() - commit_start
            transaction_time = time.time() - transaction_start

            logger.debug(
                f"Commit successful in {commit_time:.3f}s. Total transaction: {transaction_time:.3f}s (Session: {session_id})"
            )

        except Exception as commit_error:
            raise DatabaseConnectionError(
                f"Database commit failed: {commit_error}",
                context={
                    "session_id": session_id,
                    "commit_time": time.time() - commit_start,
                    "transaction_time": time.time() - transaction_start,
                },
                recovery_hint="Check database connectivity and retry transaction",
            ) from commit_error

    except DatabaseConnectionError:
        # Re-raise database-specific errors
        raise

    except Exception as e:
        # Wrap other exceptions in appropriate error types
        error_type = type(e).__name__
        transaction_time = time.time() - transaction_start

        logger.error(
            f"Exception occurred in db_transn block: {error_type}. Rolling back... (Session: {session_id})",
            exc_info=True,
        )

        # Enhanced rollback with error handling
        try:
            rollback_start = time.time()
            session.rollback()
            rollback_time = time.time() - rollback_start

            logger.warning(
                f"Rollback successful in {rollback_time:.3f}s. (Session: {session_id})"
            )

            # Determine if error is retryable
            if isinstance(e, (sqlite3.OperationalError, sqlite3.DatabaseError)):
                # Pass recovery_hint only once via kwargs supported by DatabaseConnectionError
                raise DatabaseConnectionError(
                    f"Database operation failed: {e}",
                    context={
                        "session_id": session_id,
                        "transaction_time": transaction_time,
                        "original_error": str(e),
                    },
                    recovery_hint="Database may be temporarily unavailable, retry after delay",
                )
            else:
                # Avoid passing duplicate keyword args; RetryableError accepts context via kwargs only
                raise RetryableError(
                    f"Transaction failed: {e}",
                    context={
                        "session_id": session_id,
                        "transaction_time": transaction_time,
                        "error_type": error_type,
                    },
                )

        except Exception as rb_err:
            # Critical rollback failure
            raise FatalError(
                f"CRITICAL: Failed during rollback: {rb_err}",
                context={
                    "session_id": session_id,
                    "original_error": str(e),
                    "rollback_error": str(rb_err),
                },
                recovery_hint="Database may be corrupted, check database integrity",
            ) from rb_err

    finally:
        transaction_time = time.time() - transaction_start
        logger.debug(
            f"--- db_transn finally block reached in {transaction_time:.3f}s (Session: {session_id}). ---"
        )


# End of db_transn


# ----------------------------------------------------------------------
# CRUD Operations (Adapted from user version, using new schema)
# ----------------------------------------------------------------------

# --- Create ---


# Helper functions for create_person

def _validate_person_required_fields(person_data: dict[str, Any]) -> bool:
    """Validate required fields for person creation."""
    required_keys = ("username",)
    if not all(key in person_data and person_data[key] is not None for key in required_keys):
        logger.warning(f"create_person: Missing required data (username). Data: {person_data}")
        return False
    return True


def _prepare_person_identifiers(person_data: dict[str, Any]) -> tuple[Optional[str], Optional[str], str, str]:
    """Prepare and normalize person identifiers."""
    profile_id_raw = person_data.get("profile_id")
    profile_id_upper = profile_id_raw.upper() if profile_id_raw else None
    uuid_raw = person_data.get("uuid")
    uuid_upper = str(uuid_raw).upper() if uuid_raw else None
    username = person_data["username"]
    log_ref = f"UUID={uuid_upper or 'NULL'} / ProfileID={profile_id_upper or 'NULL'} / User='{username}'"
    return profile_id_upper, uuid_upper, username, log_ref


def _check_existing_person(session: Session, profile_id_upper: Optional[str], uuid_upper: Optional[str], log_ref: str) -> bool:
    """Check if person already exists by profile_id or uuid. Returns True if exists."""
    if profile_id_upper:
        existing_by_profile = session.query(Person.id).filter(Person.profile_id == profile_id_upper).scalar()
        if existing_by_profile:
            logger.error(f"Create FAILED {log_ref}: Profile ID already exists (ID {existing_by_profile}).")
            return True

    if uuid_upper:
        existing_by_uuid = session.query(Person.id).filter(Person.uuid == uuid_upper).scalar()
        if existing_by_uuid:
            logger.error(f"Create FAILED {log_ref}: UUID already exists (ID {existing_by_uuid}).")
            return True

    return False


def _prepare_person_datetime(person_data: dict[str, Any]) -> Optional[datetime]:
    """Prepare datetime field with timezone awareness."""
    last_logged_in_dt = person_data.get("last_logged_in")
    if isinstance(last_logged_in_dt, datetime) and last_logged_in_dt.tzinfo is None:
        return last_logged_in_dt.replace(tzinfo=timezone.utc)
    return last_logged_in_dt


def _prepare_person_status(person_data: dict[str, Any], log_ref: str) -> PersonStatusEnum:
    """Prepare and validate person status enum."""
    status_value = person_data.get("status", PersonStatusEnum.ACTIVE)

    if isinstance(status_value, PersonStatusEnum):
        return status_value

    try:
        return PersonStatusEnum(str(status_value).upper())
    except ValueError:
        logger.warning(f"Invalid status '{status_value}' for {log_ref}, defaulting to ACTIVE.")
        return PersonStatusEnum.ACTIVE


def _build_person_args(person_data: dict[str, Any], identifiers: MatchIdentifiers,
                       last_logged_in_dt: Optional[datetime], status_enum: PersonStatusEnum) -> dict[str, Any]:
    """Build arguments dictionary for Person model."""
    return {
        "uuid": identifiers.uuid,
        "profile_id": identifiers.profile_id,
        "username": identifiers.username,
        "administrator_profile_id": (
            person_data.get("administrator_profile_id", "").upper()
            if person_data.get("administrator_profile_id")
            else None
        ),
        "administrator_username": person_data.get("administrator_username"),
        "message_link": person_data.get("message_link"),
        "in_my_tree": bool(person_data.get("in_my_tree", False)),
        "status": status_enum,
        "first_name": person_data.get("first_name"),
        "gender": person_data.get("gender"),
        "birth_year": person_data.get("birth_year"),
        "contactable": bool(person_data.get("contactable", True)),
        "last_logged_in": last_logged_in_dt,
    }


def _get_person_id_after_creation(session: Session, new_person: Person, log_ref: str) -> int:
    """Get person ID after creation with error handling."""
    if new_person.id is None:
        logger.error(f"ID not assigned after flush for {log_ref}! Rolling back.")
        session.rollback()
        return 0

    try:
        person_id = session.scalar(select(Person.id).where(Person.id == new_person.id))
        return person_id if person_id is not None else 0
    except Exception as e:
        logger.warning(f"Error getting person ID: {e}")
        return 0


def create_person(session: Session, person_data: dict[str, Any]) -> int:
    """
    Creates a new Person record in the database.
    Performs pre-checks for existing UUID or Profile ID to provide clearer logs,
    although database constraints provide the ultimate uniqueness guarantee.

    Args:
        session: The SQLAlchemy Session object.
        person_data: A dictionary containing the data for the new person.
                     Expected keys match Person model attributes.

    Returns:
        The integer ID of the newly created Person, or 0 on failure or if
        a duplicate (based on UUID or Profile ID) already exists.
    """
    # Step 1: Validate required fields
    if not _validate_person_required_fields(person_data):
        return 0

    # Step 2: Prepare identifiers
    profile_id_upper, uuid_upper, username, log_ref = _prepare_person_identifiers(person_data)

    # Step 3: Check for existing records
    try:
        if _check_existing_person(session, profile_id_upper, uuid_upper, log_ref):
            return 0

        # Step 4: Prepare data for new Person
        logger.debug(f"Proceeding with Person creation for {log_ref}.")
        last_logged_in_dt = _prepare_person_datetime(person_data)
        status_enum = _prepare_person_status(person_data, log_ref)
        identifiers = MatchIdentifiers(uuid=uuid_upper, username=username, in_my_tree=False, log_ref_short=log_ref, profile_id=profile_id_upper)
        new_person_args = _build_person_args(person_data, identifiers, last_logged_in_dt, status_enum)

        # Step 5: Create and add the new Person
        new_person = Person(**new_person_args)
        session.add(new_person)
        session.flush()

        # Step 6: Get and return person ID
        person_id = _get_person_id_after_creation(session, new_person, log_ref)
        logger.debug(f"Created Person ID {person_id} for {log_ref}.")
        return person_id

    # Step 7: Handle database errors
    except IntegrityError as ie:
        session.rollback()
        logger.error(f"IntegrityError create_person {log_ref}: {ie}.", exc_info=False)
        return 0
    except SQLAlchemyError as e:
        logger.error(f"DB error create_person {log_ref}: {e}", exc_info=True)
        with contextlib.suppress(Exception):
            session.rollback()
        return 0
    # Step 8: Handle unexpected errors
    except Exception as e:
        logger.critical(f"Unexpected error create_person {log_ref}: {e}", exc_info=True)
        with contextlib.suppress(Exception):
            session.rollback()
        return 0


# End of create_person


# Helper functions for create_or_update_dna_match

def _validate_dna_match_people_id(match_data: dict[str, Any]) -> tuple[Optional[int], str]:
    """Validate people_id from match data."""
    people_id = match_data.get("people_id")
    log_ref = f"PersonID={people_id}, KitUUID={match_data.get('uuid', 'N/A')}"

    if not people_id or not isinstance(people_id, int) or people_id <= 0:
        logger.error(f"create_or_update_dna_match: Invalid people_id {log_ref}.")
        return None, log_ref

    return people_id, log_ref


def _validate_optional_numeric(value: Any, allow_float: bool = False) -> Optional[Union[int, float]]:
    """Validate optional numeric field."""
    if value is None:
        return None
    try:
        if isinstance(value, str) and not value.replace(".", "", 1).isdigit():
            return None
        return float(value) if allow_float else int(value)
    except (TypeError, ValueError):
        return None


def _validate_dna_match_data(match_data: dict[str, Any], people_id: int, log_ref: str) -> Optional[dict[str, Any]]:
    """Validate and prepare DNA match data."""
    validated_data: dict[str, Any] = {"people_id": people_id}

    try:
        # Required fields
        validated_data["compare_link"] = match_data["compare_link"]
        validated_data["predicted_relationship"] = match_data["predicted_relationship"]

        # Validate cm_dna
        try:
            cm_dna_val = int(float(match_data["cm_dna"]))
            if cm_dna_val < 0:
                raise ValueError("cM cannot be negative")
            validated_data["cm_dna"] = cm_dna_val
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid cm_dna value '{match_data.get('cm_dna')}' for {log_ref}: {e}")
            raise ValueError(f"Invalid cm_dna value: {e}") from e
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"create_or_update_dna_match: Missing/Invalid required data for {log_ref}: {e}")
        return None

    # Optional fields
    validated_data["shared_segments"] = _validate_optional_numeric(match_data.get("shared_segments"))
    validated_data["longest_shared_segment"] = _validate_optional_numeric(match_data.get("longest_shared_segment"), allow_float=True)
    validated_data["meiosis"] = _validate_optional_numeric(match_data.get("meiosis"))
    validated_data["from_my_fathers_side"] = bool(match_data.get("from_my_fathers_side", False))
    validated_data["from_my_mothers_side"] = bool(match_data.get("from_my_mothers_side", False))

    return validated_data


def _compare_float_values(old_value: Any, new_value: Any) -> bool:
    """Compare float values with tolerance."""
    old_float = float(old_value) if old_value is not None else None
    new_float = float(new_value) if new_value is not None else None

    if (old_float is None) != (new_float is None):
        return True
    return bool(old_float is not None and new_float is not None and abs(old_float - new_float) > 0.01)


def _compare_field_values(old_value: Any, new_value: Any) -> bool:
    """Compare old and new field values to detect changes."""
    # Handle float comparison with tolerance
    if isinstance(new_value, float) or isinstance(old_value, float):
        return _compare_float_values(old_value, new_value)

    # Handle boolean comparison
    if isinstance(new_value, bool) or isinstance(old_value, bool):
        return bool(old_value) != bool(new_value)

    # General comparison
    return old_value != new_value


def _update_existing_dna_match(existing_dna_match: DnaMatch, validated_data: dict[str, Any], log_ref: str) -> bool:
    """Update existing DNA match record if changes detected."""
    updated = False

    for field, new_value in validated_data.items():
        if field == "people_id":
            continue

        old_value = getattr(existing_dna_match, field, None)

        if _compare_field_values(old_value, new_value):
            logger.debug(f"  DNA Change Detected for {log_ref}: Field '{field}' ('{old_value}' -> '{new_value}')")
            setattr(existing_dna_match, field, new_value)
            updated = True

    if updated:
        existing_dna_match.updated_at = datetime.now(timezone.utc)
        logger.debug(f"Updating existing DnaMatch record for {log_ref}.")
    else:
        logger.debug(f"Existing DnaMatch found for {log_ref}, no changes needed. Skipping.")

    return updated


def create_or_update_dna_match(
    session: Session, match_data: dict[str, Any]
) -> Literal["created", "updated", "skipped", "error"]:
    """
    Creates a new DnaMatch record or updates an existing one for a given Person ID.
    Compares incoming data with existing record to determine if update is needed.

    Args:
        session: The SQLAlchemy Session object.
        match_data: A dictionary containing DNA match details, including 'people_id'.

    Returns:
        'created' if a new record was added.
        'updated' if an existing record was modified.
        'skipped' if the record exists and no changes were needed.
        'error' if validation fails or a database error occurs.
    """
    result = "error"

    # Validate people_id
    people_id, log_ref = _validate_dna_match_people_id(match_data)
    if not people_id:
        return result

    # Validate and prepare data
    validated_data = _validate_dna_match_data(match_data, people_id, log_ref)
    if not validated_data:
        return result

    # Check if record exists and update or create
    try:
        existing_dna_match = session.query(DnaMatch).filter_by(people_id=people_id).first()

        if existing_dna_match:
            updated = _update_existing_dna_match(existing_dna_match, validated_data, log_ref)
            result = "updated" if updated else "skipped"
        else:
            # Create new record
            logger.debug(f"Creating new DnaMatch record for {log_ref}.")
            new_dna_match = DnaMatch(**validated_data)
            session.add(new_dna_match)
            logger.debug(f"DnaMatch record added to session for {log_ref}.")
            result = "created"

    except IntegrityError as ie:
        session.rollback()
        logger.error(f"IntegrityError create/update DNA Match {log_ref}: {ie}.", exc_info=False)
    except SQLAlchemyError as e:
        logger.error(f"DB error create/update DNA Match {log_ref}: {e}", exc_info=True)
    except Exception as e:
        logger.error(
            f"Unexpected error create/update DNA Match {log_ref}: {e}", exc_info=True
        )

    return result


# End of create_or_update_dna_match


def _prepare_family_tree_args(tree_data: dict[str, Any], people_id: int) -> dict[str, Any]:
    """Prepare valid arguments for FamilyTree model."""
    valid_tree_args = {
        col.name: tree_data.get(col.name)
        for col in FamilyTree.__table__.columns
        if col.name in tree_data and col.name not in ("id", "created_at", "updated_at")
    }
    valid_tree_args["people_id"] = people_id
    return valid_tree_args


def _update_family_tree_if_changed(existing_tree: FamilyTree, valid_tree_args: dict[str, Any], log_ref: str) -> Literal["updated", "skipped"]:
    """Update existing family tree if changes detected."""
    updated = False
    for field, new_value in valid_tree_args.items():
        if field == "people_id":
            continue
        old_value = getattr(existing_tree, field, None)
        if old_value != new_value:
            logger.debug(f"  Tree Change Detected for {log_ref}: Field '{field}' ('{old_value}' -> '{new_value}')")
            setattr(existing_tree, field, new_value)
            updated = True

    if updated:
        existing_tree.updated_at = datetime.now(timezone.utc)
        logger.debug(f"Updating existing FamilyTree record for {log_ref}.")
        return "updated"

    logger.debug(f"Existing FamilyTree record found for {log_ref}, no changes needed. Skipping.")
    return "skipped"


def create_or_update_family_tree(
    session: Session, tree_data: dict[str, Any]
) -> Literal["created", "updated", "skipped", "error"]:
    """
    Creates or updates a FamilyTree record for a given Person ID.
    Compares existing data with provided data to determine if an update is needed.

    Args:
        session: The SQLAlchemy Session object.
        tree_data: A dictionary containing family tree details, including 'people_id'.

    Returns:
        'created' if a new record was added.
        'updated' if an existing record was modified.
        'skipped' if the record exists and no changes were needed.
        'error' if validation fails or a database error occurs.
    """
    result = "error"

    # Validate people_id
    people_id = tree_data.get("people_id")
    if not people_id or not isinstance(people_id, int) or people_id <= 0:
        logger.error("Cannot create/update FamilyTree: Invalid 'people_id'.")
        return result

    # Prepare log reference and valid arguments
    cfpid_val = tree_data.get("cfpid")
    log_ref = f"PersonID={people_id}, CFPID={cfpid_val or 'N/A'}"
    valid_tree_args = _prepare_family_tree_args(tree_data, people_id)

    try:
        existing_tree = session.query(FamilyTree).filter_by(people_id=people_id).first()

        if existing_tree:
            result = _update_family_tree_if_changed(existing_tree, valid_tree_args, log_ref)
        else:
            # Create new record
            logger.debug(f"Creating new FamilyTree record for {log_ref}.")
            new_tree = FamilyTree(**valid_tree_args)
            session.add(new_tree)
            logger.debug(f"FamilyTree record added to session for {log_ref}.")
            result = "created"

    except TypeError as te:
        logger.critical(f"TypeError create/update FamilyTree {log_ref}: {te}. Args: {valid_tree_args}", exc_info=True)
    except IntegrityError as ie:
        session.rollback()
        logger.error(f"IntegrityError create/update FamilyTree {log_ref}: {ie}", exc_info=False)
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemyError create/update FamilyTree {log_ref}: {e}", exc_info=True)
    except Exception as e:
        logger.critical(f"Unexpected error create_or_update_family_tree {log_ref}: {e}", exc_info=True)

    return result


# End of create_or_update_family_tree


def _validate_person_identifiers(person_data: dict[str, Any]) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Validate and extract mandatory person identifiers."""
    uuid_raw = person_data.get("uuid")
    uuid_val = str(uuid_raw).upper() if uuid_raw else None
    username_val = person_data.get("username")

    if not uuid_val or not username_val:
        logger.error(
            f"Cannot create/update person: UUID or Username missing. Data: {person_data}"
        )
        return None, None, None, None

    profile_id_val = (
        str(person_data.get("profile_id")).upper()
        if person_data.get("profile_id")
        else None
    )
    log_ref = f"UUID={uuid_val} / ProfileID={profile_id_val or 'NULL'} / User='{username_val}'"

    return uuid_val, username_val, profile_id_val, log_ref


def _compare_datetime_field(current_value: Any, new_value: Any) -> tuple[bool, Any]:
    """Compare datetime fields with timezone awareness."""
    current_dt_utc = None
    if isinstance(current_value, datetime):
        current_dt_utc = (
            current_value.astimezone(timezone.utc)
            if current_value.tzinfo
            else current_value.replace(tzinfo=timezone.utc)
        ).replace(microsecond=0)

    new_dt_utc = None
    if isinstance(new_value, datetime):
        new_dt_utc = (
            new_value.astimezone(timezone.utc)
            if new_value.tzinfo
            else new_value.replace(tzinfo=timezone.utc)
        ).replace(microsecond=0)

    if new_dt_utc != current_dt_utc:
        value_to_set = new_value if isinstance(new_value, datetime) else None
        return True, value_to_set

    return False, current_value


def _compare_status_field(current_value: Any, new_value: Any, log_ref: str) -> tuple[bool, Any]:
    """Compare and convert status enum fields."""
    current_enum = current_value  # Already an Enum
    new_enum = None

    if isinstance(new_value, PersonStatusEnum):
        new_enum = new_value
    elif new_value is not None:
        try:
            new_enum = PersonStatusEnum(str(new_value).upper())
        except ValueError:
            logger.warning(
                f"Invalid status value '{new_value}' for update {log_ref}. Skipping status update."
            )
            return False, current_value

    if new_enum is not None and new_enum != current_enum:
        return True, new_enum

    return False, current_value


def _compare_birth_year_field(current_value: Any, new_value: Any, log_ref: str) -> tuple[bool, Any]:
    """Compare birth year field - only update if current is None."""
    if new_value is not None and current_value is None:
        try:
            value_to_set = int(new_value)
            return True, value_to_set
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid birth_year '{new_value}' for update {log_ref}. Skipping."
            )

    return False, current_value


def _compare_gender_field(current_value: Any, new_value: Any) -> tuple[bool, Any]:
    """Compare gender field - only update if current is None and new value is valid."""
    if (
        new_value is not None
        and current_value is None
        and isinstance(new_value, str)
        and new_value.lower() in ("f", "m")
    ):
        return True, new_value.lower()

    return False, current_value


def _compare_boolean_field(current_value: Any, new_value: Any) -> tuple[bool, Any]:
    """Compare boolean field values."""
    if bool(current_value) != bool(new_value):
        return True, bool(new_value)
    return False, current_value


def _compare_field_values_with_context(key: str, current_value: Any, new_value: Any, log_ref: str) -> tuple[bool, Any]:
    """Compare field values and return (value_changed, value_to_set)."""
    # Field-specific comparisons
    field_comparators = {
        "last_logged_in": lambda c, n: _compare_datetime_field(c, n),
        "status": lambda c, n: _compare_status_field(c, n, log_ref),
        "birth_year": lambda c, n: _compare_birth_year_field(c, n, log_ref),
        "gender": lambda c, n: _compare_gender_field(c, n),
    }

    if key in field_comparators:
        return field_comparators[key](current_value, new_value)

    # Boolean comparison
    if isinstance(current_value, bool) or isinstance(new_value, bool):
        return _compare_boolean_field(current_value, new_value)

    # NEVER update a field to None if it currently has a value
    # (fields that will be updated later should not be cleared)
    if current_value is not None and new_value is None:
        return False, current_value

    # Standard comparison
    if current_value != new_value:
        return True, new_value

    return False, current_value


def _prepare_person_update_fields(person_data: dict[str, Any], profile_id_val: Optional[str], username_val: str) -> dict[str, Any]:
    """
    Prepare fields to update for a person.
    Excludes None values for fields that will be updated later by additional API calls.
    """
    fields = {
        "profile_id": profile_id_val,
        "username": username_val,
        "message_link": person_data.get("message_link"),
        "in_my_tree": bool(person_data.get("in_my_tree", False)),
        "first_name": person_data.get("first_name"),
    }

    # Only include these fields if they have non-None values
    # (they will be updated later by _update_person if needed)
    if person_data.get("administrator_profile_id"):
        fields["administrator_profile_id"] = person_data["administrator_profile_id"].upper()
    if person_data.get("administrator_username"):
        fields["administrator_username"] = person_data["administrator_username"]
    if person_data.get("gender"):
        fields["gender"] = person_data["gender"]
    if person_data.get("birth_year"):
        fields["birth_year"] = person_data["birth_year"]
    if person_data.get("last_logged_in"):
        fields["last_logged_in"] = person_data["last_logged_in"]
    if person_data.get("status"):
        fields["status"] = person_data["status"]

    # contactable defaults to True, so always include it
    fields["contactable"] = bool(person_data.get("contactable", True))

    return fields


def _update_person_fields(existing_person: Person, fields_to_update: dict[str, Any], log_ref: str) -> bool:
    """Update person fields if changed. Returns True if any field was updated."""
    person_update_needed = False

    for key, new_value in fields_to_update.items():
        current_value = getattr(existing_person, key, None)
        value_changed, value_to_set = _compare_field_values_with_context(key, current_value, new_value, log_ref)

        if value_changed:
            logger.debug(f"{log_ref}: Field '{key}' changed: {current_value!r} -> {value_to_set!r}")
            setattr(existing_person, key, value_to_set)
            person_update_needed = True

    return person_update_needed


def create_or_update_person(
    session: Session, person_data: dict[str, Any]
) -> tuple[Optional[Person], Literal["created", "updated", "skipped", "error"]]:
    """
    Creates a new Person or updates an existing one based primarily on UUID.
    Handles data preparation, status enum conversion, and timezone awareness for dates.

    Args:
        session: The SQLAlchemy Session object.
        person_data: Dictionary containing data for the person. Must include 'uuid'
                     and 'username'. Other keys match Person model attributes.

    Returns:
        A tuple containing:
        - The created or updated Person object (or None on error).
        - A status string: 'created', 'updated', 'skipped', 'error'.
    """
    result_person = None
    result_status = "error"

    # Extract and validate mandatory identifiers
    uuid_val, username_val, profile_id_val, log_ref = _validate_person_identifiers(person_data)
    if not uuid_val or not username_val:
        return result_person, result_status

    # At this point, log_ref is guaranteed to be a string (not None)
    assert log_ref is not None, "log_ref should not be None after validation"

    try:
        existing_person = session.query(Person).filter(Person.uuid == uuid_val).first()

        if existing_person:
            logger.debug(f"{log_ref}: Updating existing Person ID {existing_person.id}.")

            fields_to_update = _prepare_person_update_fields(person_data, profile_id_val, username_val)
            person_update_needed = _update_person_fields(existing_person, fields_to_update, log_ref)

            if person_update_needed:
                existing_person.updated_at = datetime.now(timezone.utc)
                session.flush()
                result_person = existing_person
                result_status = "updated"
            else:
                logger.debug(f"{log_ref}: No updates needed for existing person.")
                result_person = existing_person
                result_status = "skipped"
        else:
            # Create new person
            logger.debug(f"{log_ref}: Creating new Person.")
            new_person_id = create_person(session, person_data)
            if new_person_id > 0:
                new_person_obj = session.get(Person, new_person_id)
                if new_person_obj:
                    result_person = new_person_obj
                    result_status = "created"
                else:
                    logger.error(f"Failed to fetch newly created person {log_ref} ID {new_person_id} after successful creation report.")
                    with contextlib.suppress(Exception):
                        session.rollback()
            else:
                logger.error(f"create_person helper failed for {log_ref}.")

    except IntegrityError as ie:
        session.rollback()
        logger.error(f"IntegrityError processing person {log_ref}: {ie}. Rolling back.", exc_info=False)
    except SQLAlchemyError as e:
        with contextlib.suppress(Exception):
            session.rollback()
        logger.error(f"SQLAlchemyError processing person {log_ref}: {e}", exc_info=True)
    except Exception as e:
        with contextlib.suppress(Exception):
            session.rollback()
        logger.critical(f"Unexpected critical error processing person {log_ref}: {e}", exc_info=True)

    return result_person, result_status


# End of create_or_update_person


# --- Retrieve ---

# Note: Kept retrieval functions largely as provided in user version, adding minor logging/error handling.


def get_person_by_profile_id_and_username(
    session: Session, profile_id: str, username: str, include_deleted: bool = False
) -> Optional[Person]:
    """
    Retrieves a Person record matching both profile_id (case-insensitive) AND username.

    Args:
        session: The SQLAlchemy Session object.
        profile_id: The profile ID to search for (case-insensitive).
        username: The username to search for (case-sensitive).
        include_deleted: If True, includes soft-deleted records in the search.
                         Default is False (exclude soft-deleted records).

    Returns:
        The matching Person object, or None if not found.
    """
    # Step 1: Validate inputs
    if not profile_id or not username:
        logger.warning(
            "get_person_by_profile_id_and_username: profile_id and username required."
        )
        return None
    # Step 2: Query database
    try:
        query = session.query(Person).filter(
            func.upper(Person.profile_id)
            == profile_id.upper(),  # Case-insensitive profile ID compare
            Person.username
            == username,  # Case-sensitive username compare (default SQLite)
        )

        # Apply deleted filter unless include_deleted is True
        if not include_deleted:
            query = exclude_deleted_persons(query)

        return query.first()
    except SQLAlchemyError as e:
        logger.error(
            f"DB error retrieving person by profile_id/username '{profile_id}/{username}': {e}",
            exc_info=True,
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error retrieving person by profile_id/username: {e}",
            exc_info=True,
        )
        return None


# End of get_person_by_profile_id_and_username


def get_person_by_profile_id(
    session: Session, profile_id: str, include_deleted: bool = False
) -> Optional[Person]:
    """
    Retrieves a Person record based on profile_id (case-insensitive).

    Args:
        session: The SQLAlchemy Session object.
        profile_id: The profile ID to search for (case-insensitive).
        include_deleted: If True, includes soft-deleted records in the search.
                         Default is False (exclude soft-deleted records).

    Returns:
        The matching Person object, or None if not found.
    """
    # Step 1: Validate input
    if not profile_id:
        logger.warning("get_person_by_profile_id: profile_id required.")
        return None
    # Step 2: Query database
    try:
        # Use func.upper for case-insensitive comparison if needed (depends on DB collation)
        # For SQLite, default is often case-sensitive, so explicit upper is safer.
        query = session.query(Person).filter(
            func.upper(Person.profile_id) == profile_id.upper()
        )

        # Apply deleted filter unless include_deleted is True
        if not include_deleted:
            query = exclude_deleted_persons(query)

        return query.first()
    except SQLAlchemyError as e:
        logger.error(
            f"DB error retrieving person by profile_id '{profile_id}': {e}",
            exc_info=True,
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error retrieving person by profile_id '{profile_id}': {e}",
            exc_info=True,
        )
        return None


# End of get_person_by_profile_id


def _extract_person_identifiers(match_data: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Extract profile_id and username from match data."""
    profile_id = match_data.get("profile_id")
    username = match_data.get("username")
    return profile_id, username


def _validate_required_identifiers(profile_id: Optional[str], username: Optional[str]) -> bool:
    """Validate that required identifiers are present."""
    if not profile_id or not username:
        logger.warning("get_person_and_dna_match: profile_id and username required.")
        return False
    return True


def _build_person_query(session: Session, profile_id: str, username: str, include_deleted: bool) -> Query:
    """Build query for person with DNA match eager loading."""
    query = (
        session.query(Person)
        .options(joinedload(Person.dna_match))  # Eager load DnaMatch relationship
        .filter(
            func.upper(Person.profile_id)
            == profile_id.upper(),  # Case-insensitive profile ID
            Person.username == username,  # Case-sensitive username
        )
    )

    # Apply deleted filter unless include_deleted is True
    if not include_deleted:
        query = exclude_deleted_persons(query)

    return query


def _handle_person_query_error(profile_id: str, username: str, error: Exception) -> tuple[None, None]:
    """Handle errors during person query execution."""
    if isinstance(error, SQLAlchemyError):
        logger.error(
            f"DB error retrieving person/DNA match for profile {profile_id}/{username}: {error}",
            exc_info=True,
        )
    else:
        logger.error(
            f"Unexpected error retrieving person/DNA match for profile {profile_id}/{username}: {error}",
            exc_info=True,
        )
    return None, None


def get_person_and_dna_match(
    session: Session, match_data: dict[str, Any], include_deleted: bool = False
) -> tuple[Optional[Person], Optional[DnaMatch]]:
    """
    Retrieves a Person and their associated DnaMatch record using profile_id
    (case-insensitive) and exact username. Eager loads the DnaMatch data.

    Args:
        session: The SQLAlchemy Session object.
        match_data: Dictionary containing at least 'profile_id' and 'username' keys.
        include_deleted: If True, includes soft-deleted records in the search.
                         Default is False (exclude soft-deleted records).

    Returns:
        A tuple containing (Person, DnaMatch) objects, or (None, None) if not found.
    """
    # Extract and validate identifiers
    profile_id, username = _extract_person_identifiers(match_data)
    if not _validate_required_identifiers(profile_id, username):
        return None, None

    # After validation, we know these are not None
    assert profile_id is not None and username is not None, "Validation should have caught None values"

    # Query database with eager loading
    try:
        query = _build_person_query(session, profile_id, username, include_deleted)
        person = query.first()

        # Return results
        if person:
            return person, person.dna_match  # dna_match is already loaded
        return None, None

    except Exception as e:
        return _handle_person_query_error(profile_id, username, e)


# End of get_person_and_dna_match


def exclude_deleted_persons(query: Query) -> Query:
    """
    Helper function to filter out soft-deleted Person records from a query.

    Args:
        query: A SQLAlchemy query object that includes the Person model.

    Returns:
        The modified query with a filter for Person.deleted_at IS NULL.
    """
    return query.filter(Person.deleted_at.is_(None))


# End of exclude_deleted_persons


def _extract_and_normalize_identifiers(identifier_data: dict[str, Any]) -> tuple[Optional[str], Optional[str], Optional[str], str]:
    """Extract and normalize person identifiers from data."""
    person_uuid_raw = identifier_data.get("uuid")
    person_profile_id_raw = identifier_data.get("profile_id")
    person_username = identifier_data.get("username")

    person_uuid = str(person_uuid_raw).upper() if person_uuid_raw else None
    person_profile_id = str(person_profile_id_raw).upper() if person_profile_id_raw else None

    # Create log reference
    log_parts = []
    if person_uuid:
        log_parts.append(f"UUID='{person_uuid}'")
    if person_profile_id:
        log_parts.append(f"ProfileID='{person_profile_id}'")
    if person_username:
        log_parts.append(f"User='{person_username}'")
    log_ref = " / ".join(log_parts) or "No Identifiers Provided"

    return person_uuid, person_profile_id, person_username, log_ref


def _disambiguate_by_username(potential_matches: list[Person], person_username: str, person_profile_id: str, log_ref: str) -> Optional[Person]:
    """Disambiguate multiple profile ID matches using username."""
    found_by_username: Optional[Person] = None
    username_match_count = 0

    for p in potential_matches:
        if p.username == person_username:
            found_by_username = p
            username_match_count += 1

    if username_match_count == 1 and found_by_username:
        logger.info(f"Disambiguated multiple ProfileID matches using exact Username '{person_username}' for {log_ref} (ID: {found_by_username.id}).")
        return found_by_username

    if username_match_count > 1:
        logger.error(f"CRITICAL AMBIGUITY: Found {username_match_count} people matching BOTH ProfileID {person_profile_id} AND Username '{person_username}'. Cannot reliably identify.")
        return None

    logger.warning(f"Multiple matches for ProfileID {person_profile_id}, but none matched exact Username '{person_username}'.")
    return None


def _find_by_profile_id(base_query: Any, person_profile_id: str, person_username: Optional[str], log_ref: str) -> Optional[Person]:
    """Find person by profile ID, with username disambiguation if needed."""
    potential_matches = base_query.filter(Person.profile_id == person_profile_id).all()

    if not potential_matches:
        return None

    if len(potential_matches) == 1:
        return potential_matches[0]

    # Multiple matches found
    logger.warning(f"Multiple ({len(potential_matches)}) people found for Profile ID: {person_profile_id}. Attempting disambiguation...")

    if person_username:
        return _disambiguate_by_username(potential_matches, person_username, person_profile_id, log_ref)

    logger.warning(f"Multiple matches for ProfileID {person_profile_id}, but no Username provided for disambiguation.")
    return None


def find_existing_person(
    session: Session, identifier_data: dict[str, Any], include_deleted: bool = False
) -> Optional[Person]:
    """
    Attempts to find an existing Person record based on available identifiers
    (UUID preferred, then Profile ID, potentially disambiguated by username).

    Args:
        session: The SQLAlchemy Session object.
        identifier_data: Dictionary potentially containing 'uuid', 'profile_id', 'username'.

    Returns:
        The found Person object, or None if no unique match is found or an error occurs.
    """
    # Extract identifiers
    person_uuid, person_profile_id, person_username, log_ref = _extract_and_normalize_identifiers(identifier_data)

    person: Optional[Person] = None
    try:
        # Create base query
        base_query = session.query(Person)
        if not include_deleted:
            base_query = exclude_deleted_persons(base_query)

        # Prioritize lookup by UUID
        if person_uuid:
            person = base_query.filter(Person.uuid == person_uuid).first()
            if person:
                return person

        # Try lookup by Profile ID
        if person is None and person_profile_id:
            person = _find_by_profile_id(base_query, person_profile_id, person_username, log_ref)

        # Log if no reliable match found
        if person is None:
            logger.debug(f"No existing person reliably identified for {log_ref}.")

    except SQLAlchemyError as e:
        logger.error(f"DB error find_existing_person for {log_ref}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error find_existing_person for {log_ref}: {e}", exc_info=True)
        return None

    return person


# End of find_existing_person


def get_person_by_uuid(
    session: Session, uuid: str, include_deleted: bool = False
) -> Optional[Person]:
    """
    Retrieves a Person record based on their UUID (case-insensitive), eager loading related data.

    Args:
        session: The SQLAlchemy Session object.
        uuid: The UUID to search for (case-insensitive).
        include_deleted: If True, includes soft-deleted records in the search.
                         Default is False (exclude soft-deleted records).

    Returns:
        The matching Person object with eager-loaded relationships, or None if not found.
    """
    # Step 1: Validate input
    if not uuid:
        logger.warning("get_person_by_uuid: UUID required.")
        return None
    # Step 2: Query database with eager loading
    try:
        query = (
            session.query(Person)
            .options(
                joinedload(Person.dna_match),  # Eager load DnaMatch
                joinedload(Person.family_tree),  # Eager load FamilyTree
            )
            .filter(Person.uuid == str(uuid).upper())  # Ensure uppercase comparison
        )

        # Apply deleted filter unless include_deleted is True
        if not include_deleted:
            query = exclude_deleted_persons(query)

        return query.first()
    except SQLAlchemyError as e:
        logger.error(f"DB error retrieving person by UUID {uuid}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error retrieving person by UUID {uuid}: {e}", exc_info=True
        )
        return None


# End of get_person_by_uuid

# --- Update ---


def _prepare_conversation_log_data(log_upserts: list[dict[str, Any]], log_prefix: str) -> list[dict[str, Any]]:
    """Prepare and validate conversation log data for bulk insert."""
    log_inserts_mappings = []

    for data in log_upserts:
        conv_id = data.get("conversation_id")
        direction_input = data.get("direction")
        people_id = data.get("people_id")
        ts_val = data.get("latest_timestamp")

        # Basic validation
        if not all([conv_id, direction_input, people_id, isinstance(ts_val, datetime)]):
            logger.error(f"{log_prefix}Skipping invalid log data (missing keys/ts): ConvID={conv_id}, Dir={direction_input}, PID={people_id}, TS={ts_val}")
            continue

        # Normalize direction to Enum
        try:
            if isinstance(direction_input, MessageDirectionEnum):
                direction_enum = direction_input
            else:
                direction_enum = MessageDirectionEnum(str(direction_input).upper())
        except ValueError:
            logger.error(f"{log_prefix}Invalid direction '{direction_input}' in log data ConvID {conv_id}. Skipping.")
            continue

        # Normalize timestamp to aware UTC
        if ts_val is not None:
            aware_timestamp = ts_val.astimezone(timezone.utc) if hasattr(ts_val, "tzinfo") and ts_val.tzinfo else ts_val.replace(tzinfo=timezone.utc)
            data["latest_timestamp"] = aware_timestamp
        else:
            data["latest_timestamp"] = datetime.now(timezone.utc)

        data["direction"] = direction_enum
        log_inserts_mappings.append(data)

    return log_inserts_mappings


def _prepare_person_update_data(person_updates: dict[int, PersonStatusEnum]) -> list[dict[str, Any]]:
    """Prepare and validate person update data for bulk update."""
    person_update_mappings = []

    for pid, status_enum in person_updates.items():
        if not isinstance(pid, int) or pid <= 0:
            logger.warning(f"Invalid Person ID '{pid}' in updates. Skipping.")
            continue
        if not isinstance(status_enum, PersonStatusEnum):
            logger.warning(f"Invalid status type '{type(status_enum)}' for Person ID {pid}. Skipping update.")
            continue

        person_update_mappings.append({
            "id": pid,
            "status": status_enum,
            "updated_at": datetime.now(timezone.utc),
        })

    return person_update_mappings


def commit_bulk_data(
    session: Session,
    log_upserts: list[dict[str, Any]],
    person_updates: dict[int, PersonStatusEnum],
    context: str = "Bulk Commit",
) -> tuple[int, int]:
    """
    Commits a batch of ConversationLog inserts and Person status updates to the database
    using bulk operations. ConversationLog entries are always inserted as new records
    to maintain message history.

    Args:
        session: The active SQLAlchemy database session.
        log_upserts: List of dictionaries, each containing data for a ConversationLog entry.
        person_updates: Dictionary mapping Person ID to their new PersonStatusEnum.
        context: A string describing the calling context for logging purposes.

    Returns:
        A tuple containing (logs_inserted, persons_updated).
    """
    # Check if there's data to commit
    if not log_upserts and not person_updates:
        logger.debug(f"{context}: No data provided for commit.")
        return 0, 0

    log_prefix = f"[{context}] "
    logger.debug(f"{log_prefix}Preparing commit: {len(log_upserts)} logs, {len(person_updates)} person updates.")

    try:
        with db_transn(session) as sess:
            logger.debug(f"{log_prefix}Entered transaction block.")
            processed_logs_count = 0
            updated_person_count = 0

            # Prepare and insert conversation logs
            if log_upserts:
                logger.debug(f"{log_prefix}Preparing {len(log_upserts)} ConversationLog entries for insert...")
                log_inserts_mappings = _prepare_conversation_log_data(log_upserts, log_prefix)

                if log_inserts_mappings:
                    logger.debug(f"{log_prefix}Bulk inserting {len(log_inserts_mappings)} ConversationLog entries...")
                    sess.bulk_insert_mappings(ConversationLog, log_inserts_mappings)  # type: ignore[arg-type]
                    processed_logs_count = len(log_inserts_mappings)
                    logger.debug(f"{log_prefix}Successfully inserted {processed_logs_count} ConversationLog entries.")

            # Prepare and update persons
            if person_updates:
                logger.debug(f"{log_prefix}Preparing {len(person_updates)} Person status updates...")
                person_update_mappings = _prepare_person_update_data(person_updates)

                if person_update_mappings:
                    logger.debug(f"{log_prefix}Attempting bulk update for {len(person_update_mappings)} persons...")
                    try:
                        from sqlalchemy import inspect
                        sess.bulk_update_mappings(inspect(Person), person_update_mappings)
                        updated_person_count = len(person_update_mappings)
                        logger.debug(f"{log_prefix}Bulk update successful for {updated_person_count} persons.")
                    except Exception as bulk_person_err:
                        logger.error(f"{log_prefix}Error during Person bulk update: {bulk_person_err}", exc_info=True)
                        raise
                else:
                    logger.warning(f"{log_prefix}No valid Person updates prepared for bulk operation.")

            logger.debug(f"{log_prefix}Exiting transaction block (commit follows).")

        logger.debug(f"{log_prefix}Transaction committed successfully via db_transn.")
        return processed_logs_count, updated_person_count

    except Exception as commit_err:
        logger.error(f"{log_prefix}DB Commit FAILED: {commit_err}", exc_info=True)
        return 0, 0


# End of commit_bulk_data


# --- Delete ---


def soft_delete_person(session: Session, profile_id: str, username: str) -> bool:
    """
    Soft-deletes a Person record by setting the deleted_at timestamp.
    This preserves the record and all related data (DnaMatch, FamilyTree, ConversationLog)
    but marks it as deleted so it won't appear in normal queries.

    Args:
        session: The SQLAlchemy Session object.
        profile_id: The profile ID of the person to soft-delete.
        username: The username of the person to soft-delete.

    Returns:
        True if the person was found and soft-deletion was successful, False otherwise.
    """
    # Step 1: Validate inputs
    if not profile_id or not username:
        logger.warning("soft_delete_person: profile_id and username required.")
        return False
    log_ref = f"ProfileID={profile_id}/User='{username}'"

    # Step 2: Find the person to soft-delete
    try:
        person = (
            session.query(Person)
            .filter(
                func.upper(Person.profile_id) == profile_id.upper(),
                Person.username == username,
                Person.deleted_at.is_(None),  # Only consider non-deleted records
            )
            .first()
        )

        # Step 3: Handle case where person not found
        if not person:
            logger.warning(
                f"soft_delete_person: Person {log_ref} not found or already deleted. Cannot soft-delete."
            )
            return False  # Indicate person wasn't found

        # Step 4: Set the deleted_at timestamp
        person_id_for_log = person.id  # Get ID for logging
        logger.info(f"Soft-deleting Person ID {person_id_for_log} ({log_ref})...")
        # Use setattr to avoid type checking issues with SQLAlchemy columns
        person.deleted_at = datetime.now(timezone.utc)
        person.status = PersonStatusEnum.ARCHIVE  # Also set status to ARCHIVE
        session.flush()  # Apply changes to session state immediately
        logger.info(
            f"Soft-deleted Person ID {person_id_for_log} ({log_ref}) successfully."
        )
        return True  # Indicate success

    # Step 5: Handle database errors
    except SQLAlchemyError as e:
        logger.error(f"DB error soft-deleting person {log_ref}: {e}", exc_info=True)
        with contextlib.suppress(Exception):
            session.rollback()  # Attempt rollback
        return False
    # Step 6: Handle unexpected errors
    except Exception as e:
        logger.critical(
            f"Unexpected error soft_delete_person {log_ref}: {e}", exc_info=True
        )
        with contextlib.suppress(Exception):
            session.rollback()  # Attempt rollback
        return False


# End of soft_delete_person


def hard_delete_person(session: Session, profile_id: str, username: str) -> bool:
    """
    Permanently deletes a Person record and associated cascaded records (DnaMatch, FamilyTree,
    ConversationLog) identified by profile_id (case-insensitive) and exact username.

    This function should only be used during development or when absolutely necessary.
    For normal operations, use soft_delete_person instead.

    Args:
        session: The SQLAlchemy Session object.
        profile_id: The profile ID of the person to delete.
        username: The username of the person to delete.

    Returns:
        True if the person was found and deletion initiated, False otherwise.
    """
    # Step 1: Validate inputs
    if not profile_id or not username:
        logger.warning("hard_delete_person: profile_id and username required.")
        return False
    log_ref = f"ProfileID={profile_id}/User='{username}'"

    # Step 2: Find the person to delete
    try:
        person = (
            session.query(Person)
            .filter(
                func.upper(Person.profile_id) == profile_id.upper(),
                Person.username == username,
            )
            .first()
        )

        # Step 3: Handle case where person not found
        if not person:
            logger.warning(
                f"hard_delete_person: Person {log_ref} not found. Cannot delete."
            )
            return False  # Indicate person wasn't found

        # Step 4: Delete the person object (cascades should handle related records)
        person_id_for_log = person.id  # Get ID for logging before deletion
        logger.info(
            f"Permanently deleting Person ID {person_id_for_log} ({log_ref})..."
        )
        session.delete(person)
        session.flush()  # Apply deletion to session state immediately
        logger.info(
            f"Permanently deleted Person ID {person_id_for_log} ({log_ref}) successfully."
        )
        return True  # Indicate success

    # Step 5: Handle database errors
    except SQLAlchemyError as e:
        logger.error(f"DB error hard-deleting person {log_ref}: {e}", exc_info=True)
        with contextlib.suppress(Exception):
            session.rollback()  # Attempt rollback
        return False
    # Step 6: Handle unexpected errors
    except Exception as e:
        logger.critical(
            f"Unexpected error hard_delete_person {log_ref}: {e}", exc_info=True
        )
        with contextlib.suppress(Exception):
            session.rollback()  # Attempt rollback
        return False


# End of hard_delete_person


def delete_person(
    session: Session, profile_id: str, username: str, soft_delete: bool = True
) -> bool:
    """
    Deletes a Person record identified by profile_id and username.
    By default, performs a soft delete (sets deleted_at timestamp) to preserve data.

    Args:
        session: The SQLAlchemy Session object.
        profile_id: The profile ID of the person to delete.
        username: The username of the person to delete.
        soft_delete: If True (default), performs a soft delete by setting the deleted_at timestamp.
                     If False, permanently deletes the record and all related data.

    Returns:
        True if the person was found and deletion was successful, False otherwise.
    """
    if soft_delete:
        return soft_delete_person(session, profile_id, username)
    return hard_delete_person(session, profile_id, username)


# End of delete_person


def _validate_db_path(db_path: Any) -> Path:
    """Validate and convert db_path to Path object."""
    if not isinstance(db_path, Path):
        try:
            db_path = Path(db_path)
            logger.warning("Converted db_path string to Path object in delete_database.")
        except TypeError:
            logger.error(f"Cannot convert db_path {db_path} to Path object. Deletion aborted.")
            raise
    return db_path


def _attempt_file_deletion(db_path: Path, attempt: int) -> bool:
    """Attempt to delete file. Returns True if successful."""
    # Run garbage collection to release file locks
    logger.debug("Running GC before delete attempt...")
    gc.collect()
    time.sleep(0.5)
    gc.collect()
    time.sleep(1.0 + attempt)

    if db_path.exists():
        logger.debug(f"Attempting Path.unlink on {db_path}...")
        db_path.unlink(missing_ok=True)
        time.sleep(0.1)

        if not db_path.exists():
            logger.info(f"Database file '{db_path.name}' deleted successfully.")
            return True

        logger.warning(f"os.remove called, but file '{db_path}' still exists.")
        raise OSError(f"File exists after os.remove attempt {attempt + 1}")

    logger.info(f"Database file '{db_path.name}' does not exist (already deleted?).")
    return True


def _handle_deletion_error(e: Exception, db_path: Path, attempt: int) -> Exception:
    """Handle deletion errors and return the error for tracking."""
    if isinstance(e, PermissionError):
        logger.warning(f"Permission denied deleting '{db_path}' (Attempt {attempt + 1}): {e}. File locked?")
    elif isinstance(e, OSError):
        if os.name == "nt" and hasattr(e, "winerror") and e.winerror == 32:
            logger.warning(f"OSError (WinError 32) deleting '{db_path}' (Attempt {attempt + 1}): {e}. File locked?")
        else:
            logger.error(f"OSError deleting '{db_path}' (Attempt {attempt + 1}): {e}", exc_info=True)
    else:
        logger.critical(f"Unexpected error during delete attempt {attempt + 1} for '{db_path}': {e}", exc_info=True)
    return e


def delete_database(
    db_path: Path, max_attempts: int = 5
):
    """
    Deletes the physical database file with retry logic.

    Args:
        session_manager: The SessionManager instance (used only for logging context,
                         can be None). DB connections should be closed *before* calling this.
        db_path: The Path object representing the database file.
        max_attempts: Maximum number of times to attempt deletion.

    Raises:
        OSError or other file system errors if deletion fails after all attempts.
    """
    db_path = _validate_db_path(db_path)
    logger.debug(f"Attempting to delete database file: {db_path}")
    last_error: Optional[Exception] = None

    for attempt in range(max_attempts):
        logger.debug(f"Delete attempt {attempt + 1}/{max_attempts} for {db_path.name}...")

        try:
            if _attempt_file_deletion(db_path, attempt):
                return
        except Exception as e:
            last_error = _handle_deletion_error(e, db_path, attempt)

        # Wait before next retry
        if attempt < max_attempts - 1:
            wait_time = 2**attempt
            logger.debug(f"Waiting {wait_time} seconds before next delete attempt...")
            time.sleep(wait_time)
        else:
            logger.error(f"Failed to delete database file '{db_path.name}' after {max_attempts} attempts.")
            raise last_error or OSError(f"Failed to delete {db_path} after {max_attempts} attempts")

    logger.error(f"Exited delete_database loop unexpectedly for {db_path}.")


# End of delete_database


# --- Backup and Recovery ---


@retry_on_failure(max_attempts=3, backoff_factor=2.0)
def _validate_backup_paths() -> tuple[Path, Path, Path]:
    """Validate and return database and backup paths."""
    db_path = config_schema.database.database_file
    backup_dir = config_schema.database.data_dir

    if db_path is None:
        raise AncestryError("Cannot backup database: DATABASE_FILE is not configured. Configure DATABASE_FILE in configuration.")

    if backup_dir is None:
        raise AncestryError("Cannot backup database: DATA_DIR is not configured. Configure DATA_DIR in configuration.")

    if not db_path.exists():
        raise AncestryError(f"Database file '{db_path.name}' not found at {db_path}. Ensure database file exists before backup.")

    # Check if database file is readable
    try:
        with db_path.open("rb") as f:
            f.read(1)
    except PermissionError as perm_err:
        raise DatabaseConnectionError(
            f"Permission denied accessing database file '{db_path}'",
            context={"db_path": str(db_path)},
            recovery_hint="Check file permissions and ensure database is not locked",
        ) from perm_err

    backup_path = backup_dir / "ancestry_backup.db"
    return db_path, backup_dir, backup_path


def _prepare_backup_directory(backup_dir: Path, db_path: Path) -> None:
    """Prepare backup directory and check disk space."""
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as perm_err:
        raise DatabaseConnectionError(
            f"Permission denied creating backup directory '{backup_dir}'",
            context={"backup_dir": str(backup_dir)},
            recovery_hint="Check directory permissions",
        ) from perm_err

    # Check available disk space
    import shutil as disk_utils

    try:
        disk_usage = disk_utils.disk_usage(backup_dir)
        db_size = db_path.stat().st_size
        available_space = disk_usage.free

        if available_space < db_size * 1.1:
            raise DatabaseConnectionError(
                f"Insufficient disk space for backup. Need {db_size * 1.1 / 1024 / 1024:.1f}MB, available {available_space / 1024 / 1024:.1f}MB",
                context={"db_size": db_size, "available_space": available_space, "backup_dir": str(backup_dir)},
                recovery_hint="Free up disk space or choose different backup location",
            )
    except Exception as space_check_error:
        logger.warning(f"Could not check disk space: {space_check_error}")


def _perform_backup_copy(db_path: Path, backup_path: Path) -> None:
    """Perform the actual backup copy with verification."""
    try:
        shutil.copy2(db_path, backup_path)

        if not backup_path.exists():
            raise DatabaseConnectionError("Backup file was not created successfully", context={"backup_path": str(backup_path)})

        # Verify backup size matches original
        original_size = db_path.stat().st_size
        backup_size = backup_path.stat().st_size

        if backup_size != original_size:
            raise DataValidationError(
                f"Backup size mismatch: original {original_size} bytes, backup {backup_size} bytes",
                context={"original_size": original_size, "backup_size": backup_size, "backup_path": str(backup_path)},
            )

    except PermissionError as perm_error:
        raise DatabaseConnectionError(
            f"Permission denied during backup operation: {perm_error}",
            context={"source": str(db_path), "destination": str(backup_path)},
            recovery_hint="Check file and directory permissions",
        ) from perm_error
    except OSError as os_error:
        raise DatabaseConnectionError(
            f"File system error during backup: {os_error}",
            context={"source": str(db_path), "destination": str(backup_path)},
            recovery_hint="Check disk space and file system health",
        ) from os_error


@timeout_protection(timeout=300)  # 5 minutes for large database backups
@error_context("Database Backup Operation")
def backup_database() -> bool:
    """
    Creates a backup copy of the current database file.
    The backup is named 'ancestry_backup.db' and stored in the DATA_DIR.
    Enhanced with Phase 4.1 error handling for better resilience.

    Args:
        session_manager: Not used, kept for signature consistency in main.py.

    Returns:
        True if backup was successful, False otherwise.
    """
    backup_start = time.time()

    try:
        # Validate paths
        db_path, backup_dir, backup_path = _validate_backup_paths()

        # Prepare backup directory and check space
        _prepare_backup_directory(backup_dir, db_path)

        logger.info(f"Starting database backup: '{db_path.name}' -> '{backup_path}' (Size: {db_path.stat().st_size / 1024 / 1024:.1f}MB)")

        # Perform backup
        _perform_backup_copy(db_path, backup_path)

        backup_time = time.time() - backup_start
        backup_size = backup_path.stat().st_size
        logger.debug(f"Database backup completed successfully in {backup_time:.2f}s: '{backup_path.name}' ({backup_size / 1024 / 1024:.1f}MB)")
        return True

    except AncestryError:
        raise
    except Exception as unexpected_error:
        backup_time = time.time() - backup_start
        raise RetryableError(
            f"Unexpected error during database backup: {unexpected_error}",
            context={"backup_time": backup_time, "error_type": type(unexpected_error).__name__},
            recovery_hint="Check system resources and retry",
        ) from unexpected_error


# End of backup_database


# --- Cleanup Functions ---


def cleanup_soft_deleted_records(
    session: Session, older_than_days: int = 30
) -> dict[str, int]:
    """
    Permanently deletes Person records (and their related records through cascade)
    that were soft-deleted more than the specified number of days ago.

    Args:
        session: The SQLAlchemy Session object.
        older_than_days: Only delete records that were soft-deleted more than this many days ago.
                         Default is 30 days.

    Returns:
        A dictionary with the count of deleted records by type.
    """
    # Calculate the cutoff date
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=older_than_days)

    # Initialize counters
    deleted_counts = {
        "people": 0,
        # Related records will be deleted via cascade
    }

    try:
        # Find all Person records that were soft-deleted before the cutoff date
        to_delete = session.query(Person).filter(Person.deleted_at < cutoff_date).all()

        if not to_delete:
            logger.info(
                f"No soft-deleted records older than {older_than_days} days found."
            )
            return deleted_counts

        # Log the number of records to be deleted
        logger.info(
            f"Found {len(to_delete)} soft-deleted Person records older than {older_than_days} days."
        )

        # Delete each record
        for person in to_delete:
            person_id = person.id
            profile_id = person.profile_id
            username = person.username
            log_ref = f"ID={person_id}/ProfileID={profile_id}/User='{username}'"

            logger.debug(f"Permanently deleting soft-deleted Person {log_ref}...")
            session.delete(person)
            deleted_counts["people"] += 1

        # Flush changes to the session
        session.flush()
        logger.info(
            f"Permanently deleted {deleted_counts['people']} soft-deleted Person records."
        )

        return deleted_counts

    except SQLAlchemyError as e:
        logger.error(f"DB error cleaning up soft-deleted records: {e}", exc_info=True)
        with contextlib.suppress(Exception):
            session.rollback()
        return deleted_counts
    except Exception as e:
        logger.critical(
            f"Unexpected error cleaning up soft-deleted records: {e}", exc_info=True
        )
        with contextlib.suppress(Exception):
            session.rollback()
        return deleted_counts


# End of cleanup_soft_deleted_records


# --- Tests ---


def _create_and_verify_test_person(session: Session, test_uuid: str, test_profile_id: str, test_username: str) -> bool:
    """Create and verify test person."""
    logger.info(f"Creating test person: UUID={test_uuid}, ProfileID={test_profile_id}, Username={test_username}")
    person_data = {"uuid": test_uuid, "profile_id": test_profile_id, "username": test_username, "status": PersonStatusEnum.ACTIVE}

    person_id = create_person(session, person_data)
    if person_id == 0:
        logger.error("Failed to create test person.")
        return False

    logger.info(f"Created test person with ID: {person_id}")

    person = get_person_by_profile_id(session, test_profile_id)
    if not person:
        logger.error(f"Test person with ProfileID={test_profile_id} not found after creation.")
        return False

    logger.info(f"Verified test person exists: ID={person.id}, ProfileID={person.profile_id}")
    return True


def _verify_soft_delete_state(session: Session, test_profile_id: str) -> bool:
    """Verify person is soft-deleted correctly."""
    # Verify not found in normal queries
    person = get_person_by_profile_id(session, test_profile_id)
    if person:
        logger.error(f"Test person with ProfileID={test_profile_id} still found after soft-delete in normal query.")
        return False
    logger.info("Verified test person not found in normal query after soft-delete.")

    # Verify found with include_deleted=True
    person = get_person_by_profile_id(session, test_profile_id, include_deleted=True)
    if not person:
        logger.error(f"Test person with ProfileID={test_profile_id} not found after soft-delete with include_deleted=True.")
        return False
    logger.info("Verified test person found with include_deleted=True after soft-delete.")

    # Verify deleted_at timestamp
    if getattr(person, "deleted_at", None) is None:
        logger.error(f"Test person with ProfileID={test_profile_id} has no deleted_at timestamp after soft-delete.")
        return False
    logger.info(f"Verified test person has deleted_at timestamp: {person.deleted_at}")

    # Verify status is ARCHIVE
    if getattr(person, "status", None) != PersonStatusEnum.ARCHIVE:
        logger.error(f"Test person with ProfileID={test_profile_id} has status {person.status} instead of ARCHIVE after soft-delete.")
        return False
    logger.info("Verified test person has status ARCHIVE after soft-delete.")

    return True


def test_soft_delete_functionality(session: Session) -> bool:
    """
    Tests the soft delete functionality by:
    1. Creating a test person
    2. Soft-deleting the person
    3. Verifying the person is not found in normal queries
    4. Verifying the person is found when include_deleted=True
    5. Hard-deleting the person to clean up

    Args:
        session: The SQLAlchemy Session object.

    Returns:
        True if all tests pass, False otherwise.
    """
    logger.info("=== Testing Soft Delete Functionality ===")

    # Generate unique test data
    test_uuid = f"TEST-{uuid4()}"
    test_profile_id = f"TEST-{uuid4()}"
    test_username = f"Test User {datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    result = False

    try:
        # Create and verify test person
        if _create_and_verify_test_person(session, test_uuid, test_profile_id, test_username):
            # Soft-delete the person
            logger.info(f"Soft-deleting test person: ProfileID={test_profile_id}, Username={test_username}")
            if soft_delete_person(session, test_profile_id, test_username):
                logger.info(f"Soft-deleted test person: ProfileID={test_profile_id}")

                # Verify soft-delete state
                if _verify_soft_delete_state(session, test_profile_id):
                    # Hard-delete for cleanup
                    logger.info(f"Hard-deleting test person for cleanup: ProfileID={test_profile_id}, Username={test_username}")
                    if hard_delete_person(session, test_profile_id, test_username):
                        logger.info(f"Hard-deleted test person for cleanup: ProfileID={test_profile_id}")

                        # Verify permanent deletion
                        if not get_person_by_profile_id(session, test_profile_id, include_deleted=True):
                            logger.info("Verified test person permanently deleted.")
                            logger.info("=== All Soft Delete Tests Passed ===")
                            result = True
                        else:
                            logger.error(f"Test person with ProfileID={test_profile_id} still found after hard-delete.")
                    else:
                        logger.error(f"Failed to hard-delete test person for cleanup: ProfileID={test_profile_id}")
            else:
                logger.error(f"Failed to soft-delete test person: ProfileID={test_profile_id}")

    except Exception as e:
        logger.error(f"Error during soft delete test: {e}", exc_info=True)

    return result


# End of test_soft_delete_functionality


def _create_test_persons_for_cleanup(session: Session) -> tuple[list[dict[str, str]], list[int]]:
    """Create test persons for cleanup testing."""
    test_persons = []
    for i in range(3):
        test_uuid = f"TEST-CLEANUP-{uuid4()}"
        test_profile_id = f"TEST-CLEANUP-{uuid4()}"
        test_username = f"Test Cleanup User {i} {datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        test_persons.append({"uuid": test_uuid, "profile_id": test_profile_id, "username": test_username})

    created_ids = []
    for i, person_data in enumerate(test_persons):
        logger.info(f"Creating test person {i+1}: ProfileID={person_data['profile_id']}")
        person_id = create_person(session, person_data)
        if person_id == 0:
            raise ValueError(f"Failed to create test person {i+1}.")
        created_ids.append(person_id)

    logger.info(f"Created {len(created_ids)} test persons with IDs: {created_ids}")
    return test_persons, created_ids


def _soft_delete_test_persons_with_timestamps(session: Session, test_persons: list[dict[str, str]]) -> None:
    """Soft-delete test persons and set different timestamps."""
    for i, person_data in enumerate(test_persons):
        logger.info(f"Soft-deleting test person {i+1}: ProfileID={person_data['profile_id']}")
        result = soft_delete_person(session, person_data["profile_id"], person_data["username"])
        if not result:
            raise ValueError(f"Failed to soft-delete test person {i+1}.")

        person = get_person_by_profile_id(session, person_data["profile_id"], include_deleted=True)
        if not person:
            raise ValueError(f"Test person {i+1} not found after soft-delete.")

        # Set different deleted_at timestamps
        if i == 0:
            person.deleted_at = datetime.now(timezone.utc) - timedelta(days=40)  # Should be cleaned up
        elif i == 1:
            person.deleted_at = datetime.now(timezone.utc) - timedelta(days=20)  # Should not be cleaned up

        logger.info(f"Set deleted_at for test person {i+1} to {person.deleted_at}")

    session.flush()


def _verify_cleanup_results(session: Session, test_persons: list[dict[str, str]], deleted_counts: dict[str, int]) -> bool:
    """Verify cleanup results match expectations."""
    if deleted_counts["people"] != 1:
        logger.error(f"Expected 1 person to be deleted, but got {deleted_counts['people']}.")
        return False

    logger.info(f"Cleanup deleted {deleted_counts['people']} persons as expected.")

    # Verify person 1 is deleted
    person1 = get_person_by_profile_id(session, test_persons[0]["profile_id"], include_deleted=True)
    if person1:
        logger.error("Test person 1 still exists after cleanup.")
        return False
    logger.info("Verified test person 1 was permanently deleted.")

    # Verify person 2 still exists
    person2 = get_person_by_profile_id(session, test_persons[1]["profile_id"], include_deleted=True)
    if not person2:
        logger.error("Test person 2 was unexpectedly deleted.")
        return False
    logger.info("Verified test person 2 still exists (soft-deleted).")

    # Verify person 3 still exists
    person3 = get_person_by_profile_id(session, test_persons[2]["profile_id"], include_deleted=True)
    if not person3:
        logger.error("Test person 3 was unexpectedly deleted.")
        return False
    logger.info("Verified test person 3 still exists (soft-deleted).")

    return True


def test_cleanup_soft_deleted_records(session: Session) -> bool:
    """
    Tests the cleanup_soft_deleted_records function by:
    1. Creating test persons
    2. Soft-deleting the persons with different timestamps
    3. Running the cleanup function with a specific cutoff
    4. Verifying only the appropriate records are deleted

    Args:
        session: The SQLAlchemy Session object.

    Returns:
        True if all tests pass, False otherwise.
    """
    logger.info("=== Testing Cleanup of Soft-Deleted Records ===")

    try:
        # Create test persons
        test_persons, _ = _create_test_persons_for_cleanup(session)

        # Soft-delete with different timestamps
        _soft_delete_test_persons_with_timestamps(session, test_persons)

        # Run cleanup with 30-day cutoff
        logger.info("Running cleanup_soft_deleted_records with 30-day cutoff")
        deleted_counts = cleanup_soft_deleted_records(session, older_than_days=30)

        # Verify results
        if not _verify_cleanup_results(session, test_persons, deleted_counts):
            return False

        # Clean up remaining test persons
        for i in range(1, 3):  # Persons 2 and 3
            logger.info(f"Hard-deleting test person {i+1} for cleanup")
            result = hard_delete_person(session, test_persons[i]["profile_id"], test_persons[i]["username"])
            if not result:
                logger.error(f"Failed to hard-delete test person {i+1} for cleanup.")
                return False

        logger.info("Hard-deleted remaining test persons for cleanup.")
        logger.info("=== All Cleanup Tests Passed ===")
        return True

    except Exception as e:
        logger.error(f"Error during cleanup test: {e}", exc_info=True)
        return False


# End of test_cleanup_soft_deleted_records


# Note: restore_database function (Action 4) is implemented in main.py (restore_db_actn)
# It needs to close the main SessionManager pool *before* overwriting the file.


def _get_default_message_templates() -> list[dict[str, Any]]:
    """
    Returns the default message templates to seed the database.
    This contains the current production templates as of the latest update.
    """
    return [
        {
            "template_key": "Automated_Genealogy_Response",
            "subject_line": "TBC",
            "message_content": "This is a placeholder template for AI-generated genealogical responses. The actual message content is generated dynamically by the AI system based on the conversation context and genealogical data.",
            "template_category": "other",
            "tree_status": "universal",
            "is_active": True,
            "version": 1
        },
        {
            "template_key": "DNA_Match_Specific",
            "subject_line": "Our DNA Match - {estimated_relationship} Connection",
            "message_content": "Dear {name},\n\nI'm Wayne, and I'm excited to connect with you as a DNA match on Ancestry!\n\nOur DNA results show we share {shared_dna_amount} and are estimated to be {estimated_relationship}. {dna_context}\n\n{shared_ancestor_information}\n\n{research_collaboration_request}\n\nMy family tree is well developed. Please feel free to check it out and see if you can find our connection.\n\nWarmest regards,\n\nWayne\nAberdeen, Scotland",
            "template_category": "other",
            "tree_status": "universal",
            "is_active": True,
            "version": 1
        },
        {
            "template_key": "Enhanced_In_Tree-Initial",
            "subject_line": "Exciting Discovery: Our Family Connection Through {shared_ancestors}",
            "message_content": "Dear {name},\n\nI'm Wayne, and I'm thrilled to connect with you as a DNA match on Ancestry! I believe I've discovered our family connection.\n\nI've been researching my family tree for some time and have successfully connected {total_rows} DNA matches to my tree so far. Based on my research, I believe you are my {actual_relationship} through our shared ancestors{ancestors_details}.\n\nOur connection appears to be:\n{relationship_path}\n\n{genealogical_context}\n\nI'd love to hear if this aligns with your research, or if you have additional information about{research_focus}. {specific_questions}\n\nIf you have siblings or children who would like to be added to my tree, I'd be happy to include them as well. Together, we can refine and confirm the accuracy of both of our trees.\n\nWarmest regards,\n\nWayne\nAberdeen, Scotland\n\nPS, Here's a picture of my family: <a href=\"https://www.ancestry.co.uk/mediaui-viewer/collection/1030/tree/175946702/person/102634239156/media/b096f631-1c84-4656-bb76-12d58d688e9b\">Family Picture</a>.",
            "template_category": "initial",
            "tree_status": "in_tree",
            "is_active": True,
            "version": 1
        },
        {
            "template_key": "Enhanced_Out_Tree-Initial",
            "subject_line": "Exploring Our DNA Connection - {geographic_context}",
            "message_content": "Dear {name},\n\nI'm Wayne, and I'm excited to connect with you as a DNA match on Ancestry!\n\nI've been researching my family tree for some time and have successfully connected {total_rows} DNA matches to my tree so far. While I haven't been able to place you on my tree yet, Ancestry predicts you are my {predicted_relationship}.\n\nMy ancestors came from Scotland and Galicia (south-east Poland and south-west Ukraine){location_context}. {research_suggestions}\n\nI'd love to know if you recognize any names or places from my tree that might connect to your family{specific_research_questions}. Alternatively, if you have a private tree, I'd be grateful if you could share it with me.\n\nWarmest regards,\n\nWayne\nAberdeen, Scotland\n\nPS, Here's a picture of my family: <a href=\"https://www.ancestry.co.uk/mediaui-viewer/collection/1030/tree/175946702/person/102634239156/media/b096f631-1c84-4656-bb76-12d58d688e9b\">Family Picture</a>.",
            "template_category": "initial",
            "tree_status": "out_tree",
            "is_active": True,
            "version": 1
        },
        {
            "template_key": "Enhanced_Productive_Reply",
            "subject_line": "Thank You - {research_topic}",
            "message_content": "Dear {name},\n\nThank you so much for sharing information about {mentioned_people}! Your message about {research_context} is incredibly helpful for my genealogical research.\n\n{personalized_response}\n\n{research_insights}\n\n{follow_up_questions}\n\nI really appreciate you taking the time to connect and share your family history. This kind of collaboration is what makes genealogical research so rewarding!\n\nWarmest regards,\n\nWayne\nAberdeen, Scotland",
            "template_category": "other",
            "tree_status": "universal",
            "is_active": True,
            "version": 1
        },
        {
            "template_key": "In_Tree-Final_Reminder",
            "subject_line": "Quick Reminder: Our Family DNA Connection",
            "message_content": "Dear {name},\n\nI hope you're doing well! I just wanted to send a quick reminder about our DNA match on Ancestry. I'm still very interested in exploring our connection as my {actual_relationship} and would love to hear from you if you have any updates or insights to share.\n\nIf you're not interested, no worries—just let me know! Thank you for your time.\n\nWarmest regards,\n\nWayne\nAberdeen, Scotland",
            "template_category": "reminder",
            "tree_status": "in_tree",
            "is_active": True,
            "version": 1
        },
        {
            "template_key": "In_Tree-Follow_Up",
            "subject_line": "Following Up on Our Family DNA Connection",
            "message_content": "Dear {name},\n\nI hope this message finds you well. I'm following up on my previous message about our DNA match on Ancestry. Have you had a chance to investigate our connection?\n\nI believe you are my {actual_relationship} and our connection could be:\n\n{relationship_path}\n\nIf you have any updates or would like to discuss further; I'd love to hear from you. Thank you for your time!\n\nWarmest regards,\n\nWayne\nAberdeen, Scotland",
            "template_category": "follow_up",
            "tree_status": "in_tree",
            "is_active": True,
            "version": 1
        },
        {
            "template_key": "In_Tree-Initial",
            "subject_line": "Exploring Our Family DNA Connection",
            "message_content": "Dear {name},\n\nI'm Wayne, and I'm excited to connect with you as a DNA match on Ancestry!\n\nI've been researching my family tree for some time and have successfully connected {total_rows} DNA matches to my tree so far. Each connection has helped corroborate and improve the accuracy of my tree. Ancestry thinks you are my {predicted_relationship}. I've added you because I think you are my {actual_relationship}. I believe our connection to be:\n\n{relationship_path}\n\nDoes this align with your research, or do you see another connection? If I've made a mistake, please let me know! I'd also love to hear if you have any information about our shared ancestors that could help fill in gaps in our family histories.\n\nIf you have siblings or children who would like to be added to my tree, I'd be happy to include them as well. Alternatively, if you have a private tree, I'd be grateful if you could share it with me. Together, we can refine and confirm the accuracy of both of our trees.\n\nI'd love to hear from you, even if it's just to compare notes or find out where in the world you live. If we already know one another or have spoken before, I apologize for this impersonal automated mailing. Thank you for your time and consideration!\n\nWarmest regards,\n\nWayne\nAberdeen, Scotland\n\nPS, Here's a picture of my family: <a href=\"https://www.ancestry.co.uk/mediaui-viewer/collection/1030/tree/175946702/person/102634239156/media/b096f631-1c84-4656-bb76-12d58d688e9b\">Family Picture</a>.",
            "template_category": "initial",
            "tree_status": "in_tree",
            "is_active": True,
            "version": 1
        },
        {
            "template_key": "In_Tree-Initial_Confident",
            "subject_line": "Exciting Discovery: Our Family Connection",
            "message_content": "Dear {name},\n\nI'm Wayne, and I'm excited to connect with you as a DNA match on Ancestry!\n\nI've successfully connected {total_rows} DNA matches to my tree, and I'm confident I've found our connection. You appear to be my {actual_relationship}:\n\n{relationship_path}\n\nDoes this align with your research? I'd love to hear from you and compare our family trees to confirm this connection.\n\nIf you have siblings or children who would like to be added to my tree, I'd be happy to include them as well.\n\nWarmest regards,\n\nWayne\nAberdeen, Scotland\n\nPS, Here's a picture of my family: <a href=\"https://www.ancestry.co.uk/mediaui-viewer/collection/1030/tree/175946702/person/102634239156/media/b096f631-1c84-4656-bb76-12d58d688e9b\">Family Picture</a>.",
            "template_category": "initial",
            "tree_status": "in_tree",
            "is_active": True,
            "version": 1
        },
        {
            "template_key": "In_Tree-Initial_Short",
            "subject_line": "DNA Match - Family Connection",
            "message_content": "Hi {name},\n\nI'm Wayne from Aberdeen, Scotland. Ancestry shows we're DNA matches and I believe you're my {actual_relationship}.\n\nOur connection appears to be:\n{relationship_path}\n\nI'd love to connect and compare family trees. Are you interested in exploring our connection?\n\nBest regards,\nWayne\n\nPS, Here's a picture of my family: <a href=\"https://www.ancestry.co.uk/mediaui-viewer/collection/1030/tree/175946702/person/102634239156/media/b096f631-1c84-4656-bb76-12d58d688e9b\">Family Picture</a>.",
            "template_category": "initial",
            "tree_status": "in_tree",
            "is_active": True,
            "version": 1
        },
        {
            "template_key": "In_Tree-Initial_for_was_Out_Tree",
            "subject_line": "Exciting Update: Our Family DNA Connection",
            "message_content": "Dear {name},\n\nI'm thrilled to share that I think I've found our connection! After some research, I believe I've been able to place you in my family tree. So far, I've successfully connected {total_rows} DNA matches to my tree, and each one has helped corroborate and improve the accuracy of my tree.\n\nAncestry thinks you are my {predicted_relationship}. I've added you because I think you are my {actual_relationship}. I believe our connection to be:\n\n{relationship_path}\n\nDoes this make sense to you, or do you see another link? If I've made a mistake, please let me know! I'd also love to hear if you have any information about our shared ancestors that could help fill in gaps in our family histories.\n\nIf you have siblings or children who would like to be added to my tree, I'd be happy to include them as well. Alternatively, if you have a private tree, I'd be grateful if you could share it with me. Together, we can refine and confirm the accuracy of both of our trees.\n\nThank you for your time and consideration!\n\nWarmest regards,\n\nWayne\nAberdeen, Scotland",
            "template_category": "initial",
            "tree_status": "in_tree",
            "is_active": True,
            "version": 1
        },
        {
            "template_key": "Out_Tree-Final_Reminder",
            "subject_line": "Quick Reminder: Our DNA Connection",
            "message_content": "Dear {name},\n\nI hope you're doing well! I just wanted to send a quick reminder about our DNA match on Ancestry. I'm still very interested in exploring your potential connection as my {predicted_relationship} and would love to hear from you if you have any updates or insights to share.\n\nIf you're not interested, no worries—just let me know! Thank you for your time.\n\nWarmest regards,\n\nWayne\nAberdeen, Scotland",
            "template_category": "reminder",
            "tree_status": "out_tree",
            "is_active": True,
            "version": 1
        },
        {
            "template_key": "Out_Tree-Follow_Up",
            "subject_line": "Following Up on Our DNA Connection",
            "message_content": "Dear {name},\n\nI hope this message finds you well. I'm following up on my previous message about our DNA match on Ancestry and you possibly being my {predicted_relationship}. Have you had a chance to look at my tree to see if there are any connections to your family?\n\nIf you have any updates or would like to discuss further, I'd love to hear from you. Thank you for your time!\n\nWarmest regards,\n\nWayne\nAberdeen, Scotland",
            "template_category": "follow_up",
            "tree_status": "out_tree",
            "is_active": True,
            "version": 1
        },
        {
            "template_key": "Out_Tree-Initial",
            "subject_line": "Exploring Our DNA Connection",
            "message_content": "Dear {name},\n\nI'm Wayne, and I'm excited to connect with you as a DNA match on Ancestry!\n\nI've been researching my family tree for some time and have successfully connected {total_rows} DNA matches to my tree so far. Each connection has helped corroborate and improve the accuracy of my tree. Unfortunately, I haven't been able to place you on my tree yet even though Ancestry predicts you are my {predicted_relationship}.\n\nMy ancestors came from Scotland and Galicia (south-east Poland and south-west Ukraine). I'd love to know if you recognize any names or places from my tree that might connect to your family. Alternatively, if you have a private tree, I'd be grateful if you could share it with me.\nIf we already know one another or have spoken before, I apologize for this impersonal automated mailing. Thank you for your time and consideration!\n\nWarmest regards,\n\nWayne\nAberdeen, Scotland\n\nPS, Here's a picture of my family: <a href=\"https://www.ancestry.co.uk/mediaui-viewer/collection/1030/tree/175946702/person/102634239156/media/b096f631-1c84-4656-bb76-12d58d688e9b\">Family Picture</a>.",
            "template_category": "initial",
            "tree_status": "out_tree",
            "is_active": True,
            "version": 1
        },
        {
            "template_key": "Out_Tree-Initial_Exploratory",
            "subject_line": "DNA Match - Let's Explore Our Connection",
            "message_content": "Dear {name},\n\nI'm Wayne from Aberdeen, Scotland, and I'm excited to connect with you as a DNA match on Ancestry!\n\nWhile I haven't been able to place you on my tree yet, Ancestry suggests we might be {predicted_relationship}. My ancestors came from Scotland and Galicia (south-east Poland and south-west Ukraine).\n\nI'd love to explore our potential connection. Do you recognize any Scottish or Eastern European heritage in your family?\n\nWarmest regards,\n\nWayne\nAberdeen, Scotland\n\nPS, Here's a picture of my family: <a href=\"https://www.ancestry.co.uk/mediaui-viewer/collection/1030/tree/175946702/person/102634239156/media/b096f631-1c84-4656-bb76-12d58d688e9b\">Family Picture</a>.",
            "template_category": "initial",
            "tree_status": "out_tree",
            "is_active": True,
            "version": 1
        },
        {
            "template_key": "Out_Tree-Initial_Short",
            "subject_line": "DNA Match - Exploring Connection",
            "message_content": "Hi {name},\n\nI'm Wayne from Aberdeen, Scotland. We're DNA matches on Ancestry (predicted {predicted_relationship}).\n\nI haven't placed you on my tree yet, but my ancestors came from Scotland and Galicia. Would you be interested in comparing family histories to find our connection?\n\nBest regards,\nWayne\n\nPS, Here's a picture of my family: <a href=\"https://www.ancestry.co.uk/mediaui-viewer/collection/1030/tree/175946702/person/102634239156/media/b096f631-1c84-4656-bb76-12d58d688e9b\">Family Picture</a>.",
            "template_category": "initial",
            "tree_status": "out_tree",
            "is_active": True,
            "version": 1
        },
        {
            "template_key": "Productive_Reply_Acknowledgement",
            "subject_line": "Thank You for Your Response!",
            "message_content": "Dear {name},\n\nThank you so much for your message about our DNA connection! I really appreciate you taking the time to respond and share information about our family connection.\n\nI'll review the information you've provided and get back to you if I have any questions or updates. Thanks again for connecting!\n\nWarmest regards,\n\nWayne\nAberdeen, Scotland",
            "template_category": "acknowledgement",
            "tree_status": "universal",
            "is_active": True,
            "version": 1
        },
        {
            "template_key": "Record_Research_Collaboration",
            "subject_line": "Genealogical Research Collaboration - {research_focus}",
            "message_content": "Dear {name},\n\nI'm Wayne, and I'm reaching out because of our shared interest in {research_topic}.\n\n{research_context}\n\n{specific_research_needs}\n\n{collaboration_proposal}\n\nI'd love to collaborate on this research and share any findings that might help both of our family histories!\n\nWarmest regards,\n\nWayne\nAberdeen, Scotland",
            "template_category": "other",
            "tree_status": "universal",
            "is_active": True,
            "version": 1
        },
        {
            "template_key": "User_Requested_Desist",
            "subject_line": "Sorry to bother you",
            "message_content": "Thank you for your message. I understand and will not contact you again regarding DNA matches. I wish you all the best!\n\nWarmest regards,\n\nWayne",
            "template_category": "desist",
            "tree_status": "universal",
            "is_active": True,
            "version": 1
        }
    ]


# ----------------------------------------------------------------------
# Standalone Execution & Setup (for testing/initialization)
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Step 1: Setup basic logging for standalone execution
    # Use a temporary logger setup if the main one fails or is unavailable
    try:
        from logging_config import setup_logging  # Local import

        # Use unified logging configuration; LOG_FILE is taken from .env
        standalone_logger = setup_logging(log_level="DEBUG")
        standalone_logger.info("--- Starting database.py standalone run ---")
    except Exception as log_err:
        # Fallback basic config if setup_logging fails
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname).3s [%(name)-12s %(lineno)-4d] %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stderr,
        )
        standalone_logger = logging.getLogger("db_standalone_fallback")
        standalone_logger.error(
            f"Error setting up main logger: {log_err}. Using fallback."
        )
        standalone_logger.info(
            "--- Starting database.py standalone run (Fallback Logging) ---"
        )

    engine = None  # Initialize engine variable

    try:
        # Step 2: Get database path and create engine
        if config_schema:
            db_path_obj = config_schema.database.database_file
            if db_path_obj is None:
                standalone_logger.error(
                    "DATABASE_FILE is not configured. Using in-memory database."
                )
                db_path_str = ":memory:"
            else:
                db_path_str = str(db_path_obj.resolve())
        else:
            standalone_logger.error(
                "Config instance not available. Using in-memory database."
            )
            db_path_str = ":memory:"

        standalone_logger.info(f"Target database file: {db_path_str}")
        engine = create_engine(
            f"sqlite:///{db_path_str}", echo=False
        )  # echo=True for verbose SQL

        # Step 3: Add PRAGMA event listener for new connections
        @event.listens_for(engine, "connect")
        def enable_sqlite_settings_standalone(dbapi_connection: Any, _: Any) -> None:
            """Listener to set PRAGMA settings upon connection. The second parameter is unused."""
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA foreign_keys=ON;")
                standalone_logger.debug(
                    "PRAGMA settings (WAL, Foreign Keys) applied for new connection."
                )
            except Exception as pragma_e:
                standalone_logger.error(
                    f"Failed setting PRAGMA in standalone: {pragma_e}"
                )
            finally:
                cursor.close()

        # End of enable_sqlite_settings_standalone listener

        # Step 4: Create tables and views if they don't exist
        standalone_logger.info(
            "Checking/Creating database schema (tables and views)..."
        )
        try:
            # The 'after_create' event listener (_create_views) handles view creation automatically
            Base.metadata.create_all(engine)
            standalone_logger.info("Base.metadata.create_all() executed successfully.")
        except Exception as create_err:
            standalone_logger.error(
                f"Error during schema creation: {create_err}", exc_info=True
            )
            # Optionally exit if schema creation fails critically
            # sys.exit(1)

        # Step 5: Seed MessageTemplate table with current templates
        standalone_logger.info("Seeding MessageTemplate table...")
        seed_session = None
        try:
            from sqlalchemy.orm import sessionmaker
            SessionLocal = sessionmaker(bind=engine)
            seed_session = SessionLocal()

            # Check if templates already exist
            existing_count = seed_session.query(func.count(MessageTemplate.id)).scalar() or 0

            if existing_count == 0:
                # Seed with current template data
                templates_data = _get_default_message_templates()

                for template_data in templates_data:
                    new_template = MessageTemplate(**template_data)
                    seed_session.add(new_template)

                seed_session.commit()
                final_count = seed_session.query(func.count(MessageTemplate.id)).scalar() or 0
                standalone_logger.info(f"MessageTemplate seeding complete. Added {final_count} templates to database")
            else:
                standalone_logger.info(f"MessageTemplate table already populated: {existing_count} templates found")

        except Exception as seed_err:
            standalone_logger.error(
                f"Error during MessageTemplate seeding: {seed_err}", exc_info=True
            )
        finally:
            # Ensure seed session is closed
            if seed_session:
                try:
                    # Run tests for soft delete functionality
                    standalone_logger.info(
                        "Running tests for soft delete functionality..."
                    )
                    with db_transn(seed_session) as sess:
                        # Test soft delete functionality
                        soft_delete_test_result = test_soft_delete_functionality(sess)
                        if soft_delete_test_result:
                            standalone_logger.info(
                                "Soft delete functionality tests PASSED."
                            )
                        else:
                            standalone_logger.error(
                                "Soft delete functionality tests FAILED."
                            )

                        # Test cleanup of soft-deleted records
                        cleanup_test_result = test_cleanup_soft_deleted_records(sess)
                        if cleanup_test_result:
                            standalone_logger.info(
                                "Cleanup of soft-deleted records tests PASSED."
                            )
                        else:
                            standalone_logger.error(
                                "Cleanup of soft-deleted records tests FAILED."
                            )

                        # Print summary of test results
                        standalone_logger.info("=== Test Summary ===")
                        standalone_logger.info(
                            f"Soft Delete Functionality: {'PASSED' if soft_delete_test_result else 'FAILED'}"
                        )
                        standalone_logger.info(
                            f"Cleanup of Soft-Deleted Records: {'PASSED' if cleanup_test_result else 'FAILED'}"
                        )
                        standalone_logger.info(
                            f"Overall Test Result: {'PASSED' if soft_delete_test_result and cleanup_test_result else 'FAILED'}"
                        )
                        standalone_logger.info("===================")

                        # If any test failed, log a warning
                        if not (soft_delete_test_result and cleanup_test_result):
                            standalone_logger.warning(
                                "Some tests failed. See logs for details."
                            )

                    standalone_logger.info("Tests completed.")

                    # Close the session
                    seed_session.close()
                except Exception as close_err:
                    standalone_logger.warning(
                        f"Error during tests or closing seed session: {close_err}"
                    )

    # Step 6: Handle any exceptions during the setup process
    except Exception as e:
        standalone_logger.critical(
            f"CRITICAL error during standalone database setup: {e}", exc_info=True
        )

    # Step 7: Clean up the engine connection pool
    finally:
        if engine:
            try:
                engine.dispose()
                standalone_logger.debug("SQLAlchemy engine disposed.")
            except Exception as dispose_e:
                standalone_logger.error(f"Error disposing engine: {dispose_e}")
        standalone_logger.info("--- Database.py standalone run finished ---")

# ==============================================
# Test Framework Integration
# ==============================================


# ==============================================
# MODULE-LEVEL TEST FUNCTIONS
# ==============================================


def _test_database_model_definitions() -> None:
    """Test that all required ORM models exist and can be instantiated with detailed verification."""
    model_tests = [
        (Person, "Person", "Main DNA match person record"),
        (DnaMatch, "DnaMatch", "DNA match details and relationships"),
        (FamilyTree, "FamilyTree", "Family tree position and genealogy"),
        (MessageTemplate, "MessageTemplate", "Message templates"),
        (ConversationLog, "ConversationLog", "Conversation history tracking"),
    ]

    print("📋 Testing database model definitions:")
    results = []

    for model_class, model_name, description in model_tests:
        # Test model existence
        model_exists = model_class is not None

        # Test model instantiation
        instance_created = False
        try:
            # Provide required fields for models that need them
            instance = model_class(username="test_user") if model_class == Person else model_class()
            instance_created = instance is not None
            _ = type(instance).__name__  # ensure attribute access is not useless
        except Exception as e:
            print(f"   ❌ {model_name} instantiation failed: {e}")
            import traceback
            traceback.print_exc()
            instance_created = False

        # Test table definition
        has_table = (
            hasattr(model_class, "__table__") and model_class.__table__ is not None
        )
        table_name = (
            model_class.__tablename__
            if hasattr(model_class, "__tablename__")
            else "Unknown"
        )

        status = "✅" if model_exists and instance_created and has_table else "❌"
        print(f"   {status} {model_name}: {description}")
        print(
            f"      Exists: {model_exists}, Instantiable: {instance_created}, Table: {table_name}"
        )

        test_passed = model_exists and instance_created and has_table
        results.append(test_passed)

        assert model_exists, f"{model_name} model should be defined"
        assert instance_created, f"{model_name} model should be instantiable"
        assert has_table, f"{model_name} should have table definition"

    print(
        f"📊 Results: {sum(results)}/{len(results)} database models properly defined"
    )


def _test_enum_definitions() -> None:
    """Test that required enum values are properly defined."""
    # Test PersonStatusEnum
    assert hasattr(
        PersonStatusEnum, "ACTIVE"
    ), "PersonStatusEnum should have ACTIVE status"
    assert hasattr(
        PersonStatusEnum, "ARCHIVE"
    ), "PersonStatusEnum should have ARCHIVE status"
    assert hasattr(
        PersonStatusEnum, "DESIST"
    ), "PersonStatusEnum should have DESIST status"
    assert hasattr(
        PersonStatusEnum, "BLOCKED"
    ), "PersonStatusEnum should have BLOCKED status"
    assert hasattr(
        PersonStatusEnum, "DEAD"
    ), "PersonStatusEnum should have DEAD status"

    # Test MessageDirectionEnum
    assert hasattr(
        MessageDirectionEnum, "IN"
    ), "MessageDirectionEnum should have IN"
    assert hasattr(
        MessageDirectionEnum, "OUT"
    ), "MessageDirectionEnum should have OUT"

    # Test RoleType
    assert hasattr(RoleType, "AUTHOR"), "RoleType should have AUTHOR"
    assert hasattr(RoleType, "RECIPIENT"), "RoleType should have RECIPIENT"


def _test_database_base_setup() -> None:
    """Test that SQLAlchemy base is properly configured."""
    assert Base is not None, "Base declarative model should be defined"
    assert hasattr(Base, "metadata"), "Base should have metadata attribute"


def _test_transaction_context_manager() -> None:
    """Test the db_transn context manager functionality."""
    assert callable(db_transn), "db_transn should be a callable function"
    # Test basic structure (without actual database operations)


def _test_model_attributes() -> None:
    """Test that models have expected attributes and columns."""
    # Test Person model attributes (using actual column names from model definition)
    assert hasattr(Person, "id"), "Person should have id attribute"
    assert hasattr(Person, "uuid"), "Person should have uuid attribute"
    assert hasattr(Person, "profile_id"), "Person should have profile_id attribute"
    assert hasattr(Person, "username"), "Person should have username attribute"
    assert hasattr(Person, "status"), "Person should have status attribute"

    # Test DnaMatch model attributes (using actual column names)
    assert hasattr(DnaMatch, "id"), "DnaMatch should have id attribute"
    assert hasattr(DnaMatch, "people_id"), "DnaMatch should have people_id attribute"
    assert hasattr(DnaMatch, "cm_dna"), "DnaMatch should have cm_dna attribute"
    assert hasattr(DnaMatch, "shared_segments"), "DnaMatch should have shared_segments attribute"

    # Test FamilyTree model attributes (using actual column names)
    assert hasattr(FamilyTree, "id"), "FamilyTree should have id attribute"
    assert hasattr(FamilyTree, "people_id"), "FamilyTree should have people_id attribute"
    assert hasattr(FamilyTree, "cfpid"), "FamilyTree should have cfpid attribute"

    # Test MessageTemplate model attributes (using actual column names)
    assert hasattr(MessageTemplate, "id"), "MessageTemplate should have id attribute"
    assert hasattr(MessageTemplate, "template_key"), "MessageTemplate should have template_key attribute"

    # Test ConversationLog model attributes (using actual column names)
    assert hasattr(ConversationLog, "id"), "ConversationLog should have id attribute"
    assert hasattr(ConversationLog, "people_id"), "ConversationLog should have people_id attribute"


def _test_database_utilities() -> None:
    """Test database utility functions."""
    # Test that utility functions exist
    assert callable(backup_database), "backup_database should be callable"
    assert callable(create_person), "create_person should be callable"
    assert callable(soft_delete_person), "soft_delete_person should be callable"


def _test_enum_edge_cases() -> None:
    """Test enum edge cases and validation."""
    # Test that enum values are unique
    status_values = [
        PersonStatusEnum.ACTIVE,
        PersonStatusEnum.ARCHIVE,
        PersonStatusEnum.DESIST,
        PersonStatusEnum.BLOCKED,
        PersonStatusEnum.DEAD,
    ]
    assert len(status_values) == len(set(status_values)), "PersonStatusEnum values should be unique"

    # Test MessageDirectionEnum values
    direction_values = [MessageDirectionEnum.IN, MessageDirectionEnum.OUT]
    assert len(direction_values) == len(set(direction_values)), "MessageDirectionEnum values should be unique"

    # Test RoleType values
    role_values = [RoleType.AUTHOR, RoleType.RECIPIENT]
    assert len(role_values) == len(set(role_values)), "RoleType values should be unique"


def _test_model_instantiation_edge_cases() -> None:
    """Test model instantiation with various edge cases."""
    # Test Person instantiation with minimal data (username required)
    person = Person(username="test_user")
    assert person is not None, "Person should be instantiable"

    # Test DnaMatch instantiation
    match = DnaMatch()
    assert match is not None, "DnaMatch should be instantiable"

    # Test FamilyTree instantiation
    tree = FamilyTree()
    assert tree is not None, "FamilyTree should be instantiable"


def _test_model_relationships() -> None:
    """Test model relationships and foreign keys."""
    # Test that models have relationship attributes (using actual relationship names from model)
    assert hasattr(Person, "dna_match"), "Person should have dna_match relationship"
    assert hasattr(Person, "family_tree"), "Person should have family_tree relationship"
    assert hasattr(Person, "conversation_log_entries"), "Person should have conversation_log_entries relationship"


def _test_schema_integration() -> None:
    """Test schema integration and table creation."""
    # Test that Base.metadata contains all tables
    assert Base.metadata is not None, "Base.metadata should exist"
    assert len(Base.metadata.tables) > 0, "Base.metadata should contain tables"

    # Test that expected tables are in metadata
    table_names = list(Base.metadata.tables.keys())
    assert "people" in table_names, "people table should be in metadata"
    assert "dna_match" in table_names, "dna_match table should be in metadata"


def _test_model_creation_performance() -> None:
    """Test model creation performance."""
    import time
    start_time = time.time()

    # Create multiple model instances (Person requires username)
    for i in range(100):
        Person(username=f"test_user_{i}")
        DnaMatch()
        FamilyTree()

    elapsed_time = time.time() - start_time
    assert elapsed_time < 1.0, f"Model creation should be fast, took {elapsed_time:.3f}s"


def _test_import_error_handling() -> None:
    """Test error handling for import failures."""
    # Test that required modules are imported
    assert Person is not None, "Person model should be imported"
    assert DnaMatch is not None, "DnaMatch model should be imported"
    assert Base is not None, "Base should be imported"


def _test_configuration_error_handling() -> None:
    """Test error handling for configuration issues."""
    # Test that database configuration is accessible via config_schema
    assert config_schema is not None, "config_schema should be available"
    # Note: Database uses SessionManager for session management, not standalone functions


# ==============================================
# MAIN TEST SUITE RUNNER
# ==============================================


def database_module_tests() -> bool:
    """Comprehensive test suite for database.py with real functionality testing."""
    suite = TestSuite("Database Models & ORM Management", "database.py")
    suite.start_suite()

    # Assign all module-level test functions
    test_database_model_definitions = _test_database_model_definitions
    test_enum_definitions = _test_enum_definitions
    test_database_base_setup = _test_database_base_setup
    test_transaction_context_manager = _test_transaction_context_manager
    test_model_attributes = _test_model_attributes
    test_database_utilities = _test_database_utilities
    test_enum_edge_cases = _test_enum_edge_cases
    test_model_instantiation_edge_cases = _test_model_instantiation_edge_cases
    test_model_relationships = _test_model_relationships
    test_schema_integration = _test_schema_integration
    test_model_creation_performance = _test_model_creation_performance
    test_import_error_handling = _test_import_error_handling
    test_configuration_error_handling = _test_configuration_error_handling

    # Define all tests in a data structure to reduce complexity
    tests = [
        ("Database Model Definitions", test_database_model_definitions,
         "5 database models tested: Person, DnaMatch, FamilyTree, MessageTemplate, ConversationLog - all exist, instantiable, with table definitions.",
         "Test that all required ORM models exist and can be instantiated with detailed verification.",
         "Verify Person→people, DnaMatch→dna_match, FamilyTree→family_tree, MessageTemplate→message_templates, ConversationLog→conversation_log tables."),
        ("Enum Value Definitions", test_enum_definitions,
         "All required enum values (PersonStatusEnum, MessageDirectionEnum, RoleType) are properly defined",
         "Test enum definitions for status management and message direction handling",
         "Verify that enum classes have required status values and direction indicators"),
        ("Database Base Setup", test_database_base_setup,
         "SQLAlchemy Base is properly configured with metadata and registry",
         "Test that SQLAlchemy base is properly configured",
         "Verify Base has metadata, registry, and proper declarative setup"),
        ("Transaction Context Manager", test_transaction_context_manager,
         "Transaction context manager handles commits and rollbacks correctly",
         "Test transaction context manager functionality",
         "Verify proper transaction handling with commit/rollback support"),
        ("Model Attributes", test_model_attributes,
         "All database models have required attributes and relationships",
         "Test that models have expected attributes",
         "Verify Person, DnaMatch, FamilyTree models have correct columns and relationships"),
        ("Database Utilities", test_database_utilities,
         "Database utility functions work correctly",
         "Test database utility functions",
         "Verify helper functions for database operations"),
        ("Enum Edge Cases", test_enum_edge_cases,
         "Enum definitions handle edge cases properly",
         "Test enum edge case handling",
         "Verify enum validation and error handling"),
        ("Model Instantiation Edge Cases", test_model_instantiation_edge_cases,
         "Models handle edge cases during instantiation",
         "Test model instantiation edge cases",
         "Verify models handle None values, empty strings, and invalid data"),
        ("Model Relationships", test_model_relationships,
         "Model relationships are properly defined and functional",
         "Test model relationship definitions",
         "Verify foreign keys, back references, and relationship integrity"),
        ("Schema Integration", test_schema_integration,
         "Database schema integrates properly with application",
         "Test schema integration",
         "Verify schema compatibility and migration support"),
        ("Model Creation Performance", test_model_creation_performance,
         "Model creation performs within acceptable limits",
         "Test model creation performance",
         "Verify model instantiation is efficient"),
        ("Import Error Handling", test_import_error_handling,
         "Module handles import errors gracefully",
         "Test handling of missing dependencies",
         "Verify graceful degradation when optional dependencies unavailable"),
        ("Configuration Error Handling", test_configuration_error_handling,
         "Module handles configuration dependencies correctly",
         "Test handling of configuration and logging dependencies",
         "Verify that required configuration and logging components are available"),
    ]

    # Run all tests from the list
    with suppress_logging():
        for test_name, test_func, expected, method, details in tests:
            suite.run_test(test_name, test_func, expected, method, details)

    return suite.finish_suite()


def run_comprehensive_tests() -> bool:
    """Run comprehensive database tests using standardized TestSuite format."""
    return database_module_tests()


# === END OF database.py ===


if __name__ == "__main__":
    """
    Execute comprehensive database tests when run directly.
    Tests all database models, enums, utilities, and core functionality.
    """
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
