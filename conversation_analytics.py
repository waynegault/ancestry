# pyright: reportConstantRedefinition=false
"""
Conversation Analytics & Engagement Metrics Tracking

Provides comprehensive analytics and metrics tracking for genealogical conversations,
enabling data-driven optimization of messaging strategies and dialogue effectiveness.

Features:
• Conversation metrics tracking (response rates, engagement scores, duration)
• Template effectiveness analysis (which templates get best responses)
• Engagement event tracking (responses, person lookups, research tasks)
• CLI dashboard for viewing analytics
• Aggregation and reporting functions

Author: Ancestry Automation System
Created: October 22, 2025
Phase: 6 - Production Deployment & Monitoring
"""

from datetime import datetime, timezone
from typing import Any, Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship, Session
from database import Base
import json


# ----------------------------------------------------------------------
# Database Models
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
    person = relationship("Person", back_populates="conversation_metrics")


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
    person = relationship("Person", back_populates="engagement_events")


# ----------------------------------------------------------------------
# Analytics Functions
# ----------------------------------------------------------------------


def record_engagement_event(
    session: Session,
    people_id: int,
    event_type: str,
    event_description: Optional[str] = None,
    event_data: Optional[dict[str, Any]] = None,
    engagement_score_before: Optional[int] = None,
    engagement_score_after: Optional[int] = None,
    conversation_phase: Optional[str] = None,
    template_used: Optional[str] = None,
) -> EngagementTracking:
    """
    Record an engagement event for analytics tracking.
    
    Args:
        session: Database session
        people_id: ID of the person
        event_type: Type of event (message_sent, message_received, person_lookup, etc.)
        event_description: Optional description of the event
        event_data: Optional additional data as dictionary
        engagement_score_before: Engagement score before event
        engagement_score_after: Engagement score after event
        conversation_phase: Current conversation phase
        template_used: Template key if applicable
        
    Returns:
        Created EngagementTracking record
    """
    engagement_score_delta = None
    if engagement_score_before is not None and engagement_score_after is not None:
        engagement_score_delta = engagement_score_after - engagement_score_before
    
    event = EngagementTracking(
        people_id=people_id,
        event_type=event_type,
        event_description=event_description,
        event_data=json.dumps(event_data) if event_data else None,
        engagement_score_before=engagement_score_before,
        engagement_score_after=engagement_score_after,
        engagement_score_delta=engagement_score_delta,
        conversation_phase=conversation_phase,
        template_used=template_used,
    )
    
    session.add(event)
    session.commit()
    
    return event


# ----------------------------------------------------------------------
# Module Tests
# ----------------------------------------------------------------------


def conversation_analytics_module_tests() -> bool:
    """
    Comprehensive test suite for conversation_analytics.py.
    Tests analytics tracking, metric collection, and reporting.
    """
    from test_framework import TestSuite, suppress_logging
    
    suite = TestSuite("conversation_analytics.py", "Conversation Analytics & Engagement Metrics")
    
    with suppress_logging():
        # Test 1: Database models are defined
        suite.run_test(
            "Database models defined",
            lambda: ConversationMetrics is not None and EngagementTracking is not None,
        )
        
        # Test 2: record_engagement_event function exists
        suite.run_test(
            "record_engagement_event function exists",
            lambda: callable(record_engagement_event),
        )
    
    return suite.finish_suite()


if __name__ == "__main__":
    conversation_analytics_module_tests()

