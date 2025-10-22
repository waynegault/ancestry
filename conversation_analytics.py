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

import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Session, relationship

from database import Base

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


def update_conversation_metrics(
    session: Session,
    people_id: int,
    message_sent: bool = False,
    message_received: bool = False,
    template_used: Optional[str] = None,
    engagement_score: Optional[int] = None,
    conversation_phase: Optional[str] = None,
    person_looked_up: bool = False,
    person_found: bool = False,
    research_task_created: bool = False,
    added_to_tree: bool = False,
) -> ConversationMetrics:
    """
    Update conversation metrics for a person.
    Creates metrics record if it doesn't exist.

    Args:
        session: Database session
        people_id: ID of the person
        message_sent: Whether a message was sent
        message_received: Whether a message was received
        template_used: Template key if message was sent
        engagement_score: Current engagement score
        conversation_phase: Current conversation phase
        person_looked_up: Whether a person was looked up
        person_found: Whether a person was found
        research_task_created: Whether a research task was created
        added_to_tree: Whether person was added to tree

    Returns:
        Updated ConversationMetrics record
    """
    # Get or create metrics record
    metrics = session.query(ConversationMetrics).filter_by(people_id=people_id).first()

    if not metrics:
        metrics = ConversationMetrics(people_id=people_id)
        session.add(metrics)

    # Update message counts
    if message_sent:
        setattr(metrics, 'messages_sent', getattr(metrics, 'messages_sent') + 1)
        now = datetime.now(timezone.utc)
        setattr(metrics, 'last_message_sent', now)
        if getattr(metrics, 'first_message_sent') is None:
            setattr(metrics, 'first_message_sent', now)

        # Track initial template
        if getattr(metrics, 'messages_sent') == 1 and template_used:
            setattr(metrics, 'initial_template_used', template_used)

        # Track all templates used
        if template_used:
            templates_json = getattr(metrics, 'templates_used')
            templates = json.loads(templates_json) if templates_json else []
            if template_used not in templates:
                templates.append(template_used)
            setattr(metrics, 'templates_used', json.dumps(templates))

    if message_received:
        setattr(metrics, 'messages_received', getattr(metrics, 'messages_received') + 1)
        now = datetime.now(timezone.utc)
        setattr(metrics, 'last_message_received', now)

        # Track first response
        if getattr(metrics, 'first_response_received') is not True:
            setattr(metrics, 'first_response_received', True)
            setattr(metrics, 'first_response_date', now)

            # Calculate time to first response
            first_sent = getattr(metrics, 'first_message_sent')
            if first_sent is not None:
                time_diff = now - first_sent
                setattr(metrics, 'time_to_first_response_hours', time_diff.total_seconds() / 3600)

    # Update engagement metrics
    if engagement_score is not None:
        setattr(metrics, 'current_engagement_score', engagement_score)
        current_max = getattr(metrics, 'max_engagement_score')
        setattr(metrics, 'max_engagement_score', max(engagement_score, current_max))

        # Update average engagement score
        # Simple moving average based on number of updates
        current_avg = getattr(metrics, 'avg_engagement_score')
        if current_avg is None:
            setattr(metrics, 'avg_engagement_score', float(engagement_score))
        else:
            # Weighted average (give more weight to recent scores)
            new_avg = (current_avg * 0.8) + (engagement_score * 0.2)
            setattr(metrics, 'avg_engagement_score', new_avg)

    # Update conversation phase
    if conversation_phase:
        setattr(metrics, 'conversation_phase', conversation_phase)

    # Update conversation duration
    first_sent = getattr(metrics, 'first_message_sent')
    last_received = getattr(metrics, 'last_message_received')
    if first_sent is not None and last_received is not None:
        time_diff = last_received - first_sent
        setattr(metrics, 'conversation_duration_days', time_diff.total_seconds() / 86400)

    # Update research outcomes
    if person_looked_up:
        setattr(metrics, 'people_looked_up', getattr(metrics, 'people_looked_up') + 1)
    if person_found:
        setattr(metrics, 'people_found', getattr(metrics, 'people_found') + 1)
    if research_task_created:
        setattr(metrics, 'research_tasks_created', getattr(metrics, 'research_tasks_created') + 1)

    # Update tree impact
    if added_to_tree and getattr(metrics, 'added_to_tree') is not True:
        setattr(metrics, 'added_to_tree', True)
        setattr(metrics, 'added_to_tree_date', datetime.now(timezone.utc))

    session.commit()

    return metrics


def get_overall_analytics(session: Session) -> dict[str, Any]:
    """
    Get overall analytics across all conversations.

    Args:
        session: Database session

    Returns:
        Dictionary with overall analytics
    """
    from sqlalchemy import func

    # Get total conversations with metrics
    total_conversations = session.query(ConversationMetrics).count()

    # Get response rate
    conversations_with_responses = session.query(ConversationMetrics).filter(
        ConversationMetrics.first_response_received.is_(True)
    ).count()
    response_rate = (conversations_with_responses / total_conversations * 100) if total_conversations > 0 else 0

    # Get average engagement score
    avg_engagement = session.query(func.avg(ConversationMetrics.current_engagement_score)).scalar() or 0

    # Get average time to first response (in hours)
    avg_time_to_response = session.query(func.avg(ConversationMetrics.time_to_first_response_hours)).filter(
        ConversationMetrics.time_to_first_response_hours.isnot(None)
    ).scalar() or 0

    # Get conversation phase distribution
    phase_distribution = {}
    phases = session.query(
        ConversationMetrics.conversation_phase,
        func.count(ConversationMetrics.id)
    ).group_by(ConversationMetrics.conversation_phase).all()

    for phase, count in phases:
        phase_distribution[phase] = count

    # Get template effectiveness (response rate by template)
    template_effectiveness = {}
    templates = session.query(ConversationMetrics.initial_template_used).distinct().all()

    for (template,) in templates:
        if template:
            total_with_template = session.query(ConversationMetrics).filter(
                ConversationMetrics.initial_template_used == template
            ).count()
            responses_with_template = session.query(ConversationMetrics).filter(
                ConversationMetrics.initial_template_used == template,
                ConversationMetrics.first_response_received.is_(True)
            ).count()

            template_effectiveness[template] = {
                "total": total_with_template,
                "responses": responses_with_template,
                "response_rate": (responses_with_template / total_with_template * 100) if total_with_template > 0 else 0
            }

    # Get research outcomes
    total_people_looked_up = session.query(func.sum(ConversationMetrics.people_looked_up)).scalar() or 0
    total_people_found = session.query(func.sum(ConversationMetrics.people_found)).scalar() or 0
    total_research_tasks = session.query(func.sum(ConversationMetrics.research_tasks_created)).scalar() or 0

    # Get tree impact
    people_added_to_tree = session.query(ConversationMetrics).filter(
        ConversationMetrics.added_to_tree.is_(True)
    ).count()

    return {
        "total_conversations": total_conversations,
        "conversations_with_responses": conversations_with_responses,
        "response_rate_percent": round(response_rate, 2),
        "avg_engagement_score": round(avg_engagement, 2),
        "avg_time_to_first_response_hours": round(avg_time_to_response, 2),
        "phase_distribution": phase_distribution,
        "template_effectiveness": template_effectiveness,
        "research_outcomes": {
            "people_looked_up": total_people_looked_up,
            "people_found": total_people_found,
            "success_rate_percent": round((total_people_found / total_people_looked_up * 100) if total_people_looked_up > 0 else 0, 2),
            "research_tasks_created": total_research_tasks,
        },
        "tree_impact": {
            "people_added_to_tree": people_added_to_tree,
            "conversion_rate_percent": round((people_added_to_tree / total_conversations * 100) if total_conversations > 0 else 0, 2),
        }
    }


def print_analytics_dashboard(session: Session) -> None:
    """
    Print a CLI dashboard with conversation analytics.

    Args:
        session: Database session
    """
    analytics = get_overall_analytics(session)

    print("\n" + "=" * 80)
    print("CONVERSATION ANALYTICS DASHBOARD")
    print("=" * 80)

    print("\n📊 OVERALL METRICS")
    print(f"   Total Conversations: {analytics['total_conversations']}")
    print(f"   Conversations with Responses: {analytics['conversations_with_responses']}")
    print(f"   Response Rate: {analytics['response_rate_percent']}%")
    print(f"   Average Engagement Score: {analytics['avg_engagement_score']}/100")
    print(f"   Average Time to First Response: {analytics['avg_time_to_first_response_hours']:.1f} hours")

    print("\n📈 CONVERSATION PHASES")
    for phase, count in analytics['phase_distribution'].items():
        print(f"   {phase}: {count}")

    print("\n📧 TEMPLATE EFFECTIVENESS")
    for template, data in analytics['template_effectiveness'].items():
        print(f"   {template}:")
        print(f"      Total: {data['total']}, Responses: {data['responses']}, Rate: {data['response_rate']:.1f}%")

    print("\n🔍 RESEARCH OUTCOMES")
    print(f"   People Looked Up: {analytics['research_outcomes']['people_looked_up']}")
    print(f"   People Found: {analytics['research_outcomes']['people_found']}")
    print(f"   Success Rate: {analytics['research_outcomes']['success_rate_percent']}%")
    print(f"   Research Tasks Created: {analytics['research_outcomes']['research_tasks_created']}")

    print("\n🌳 TREE IMPACT")
    print(f"   People Added to Tree: {analytics['tree_impact']['people_added_to_tree']}")
    print(f"   Conversion Rate: {analytics['tree_impact']['conversion_rate_percent']}%")

    print("\n" + "=" * 80 + "\n")


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

        # Test 3: update_conversation_metrics function exists
        suite.run_test(
            "update_conversation_metrics function exists",
            lambda: callable(update_conversation_metrics),
        )

        # Test 4: get_overall_analytics function exists
        suite.run_test(
            "get_overall_analytics function exists",
            lambda: callable(get_overall_analytics),
        )

        # Test 5: print_analytics_dashboard function exists
        suite.run_test(
            "print_analytics_dashboard function exists",
            lambda: callable(print_analytics_dashboard),
        )

        # Test 6: ConversationMetrics has required fields
        suite.run_test(
            "ConversationMetrics has required fields",
            lambda: hasattr(ConversationMetrics, "messages_sent") and
                    hasattr(ConversationMetrics, "messages_received") and
                    hasattr(ConversationMetrics, "current_engagement_score") and
                    hasattr(ConversationMetrics, "first_response_received"),
        )

        # Test 7: EngagementTracking has required fields
        suite.run_test(
            "EngagementTracking has required fields",
            lambda: hasattr(EngagementTracking, "event_type") and
                    hasattr(EngagementTracking, "engagement_score_delta") and
                    hasattr(EngagementTracking, "event_timestamp"),
        )

    return suite.finish_suite()


if __name__ == "__main__":
    conversation_analytics_module_tests()

