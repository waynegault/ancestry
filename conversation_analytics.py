#!/usr/bin/env python3
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
from importlib import import_module
from typing import Any, Optional, Protocol, cast as typing_cast

from sqlalchemy import String as SQLString, cast
from sqlalchemy.orm import Session

# Import analytics models from database.py (they're defined there to avoid circular imports)
from database import ConversationMetrics, EngagementTracking


class _DatabaseManagerProtocol(Protocol):
    """Interface subset required from DatabaseManager for tests."""

    def get_session(self) -> Optional[Session]:  # pragma: no cover - protocol definition
        """Return a SQLAlchemy session bound to the project database."""


_database_manager_class: type[_DatabaseManagerProtocol] | None = None
_database_manager_error: Exception | None = None

try:  # pragma: no cover - optional dependency
    _database_module = import_module("core.database_manager")
except Exception as exc:
    _database_manager_error = exc
else:
    _database_candidate = getattr(_database_module, "DatabaseManager", None)
    if isinstance(_database_candidate, type):
        _database_manager_class = typing_cast(type[_DatabaseManagerProtocol], _database_candidate)
    else:
        _database_manager_error = RuntimeError(
            "DatabaseManager class missing from core.database_manager"
        )


def _resolve_database_manager_class() -> Optional[type[_DatabaseManagerProtocol]]:
    """Return the cached DatabaseManager class if available."""

    return _database_manager_class


def _create_database_manager() -> _DatabaseManagerProtocol:
    """Instantiate DatabaseManager or raise a helpful error."""

    db_class = _resolve_database_manager_class()
    if db_class is None:
        error_detail = f": {_database_manager_error!r}" if _database_manager_error else ""
        raise RuntimeError(f"DatabaseManager unavailable{error_detail}")

    try:
        return db_class()
    except Exception as exc:  # pragma: no cover - defensive instantiation guard
        raise RuntimeError("Failed to instantiate DatabaseManager") from exc


def _create_database_session() -> Session:
    """Convenience helper to obtain a SQLAlchemy session for analytics tests."""

    session = _create_database_manager().get_session()
    if session is None:  # pragma: no cover - guard for unexpected behavior
        raise RuntimeError("DatabaseManager.get_session() returned None")
    return session


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


def _update_sent_message_metrics(metrics: ConversationMetrics, template_used: Optional[str]) -> None:
    """Update metrics when a message is sent."""
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
        templates: list[str] = json.loads(templates_json) if templates_json else []
        if template_used not in templates:
            templates.append(template_used)
        setattr(metrics, 'templates_used', json.dumps(templates))


def _update_received_message_metrics(metrics: ConversationMetrics) -> None:
    """Update metrics when a message is received."""
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
            # Ensure first_sent is timezone-aware
            if first_sent.tzinfo is None:
                first_sent = first_sent.replace(tzinfo=timezone.utc)
            time_diff = now - first_sent
            setattr(metrics, 'time_to_first_response_hours', time_diff.total_seconds() / 3600)


def _update_engagement_metrics(metrics: ConversationMetrics, engagement_score: int) -> None:
    """Update engagement score metrics."""
    setattr(metrics, 'current_engagement_score', engagement_score)
    current_max = getattr(metrics, 'max_engagement_score')
    setattr(metrics, 'max_engagement_score', max(engagement_score, current_max))

    # Update average engagement score (weighted average)
    current_avg = getattr(metrics, 'avg_engagement_score')
    if current_avg is None:
        setattr(metrics, 'avg_engagement_score', float(engagement_score))
    else:
        new_avg = (current_avg * 0.8) + (engagement_score * 0.2)
        setattr(metrics, 'avg_engagement_score', new_avg)


def _update_research_outcomes(
    metrics: ConversationMetrics,
    person_looked_up: bool,
    person_found: bool,
    research_task_created: bool
) -> None:
    """Update research outcome metrics."""
    if person_looked_up:
        setattr(metrics, 'people_looked_up', getattr(metrics, 'people_looked_up') + 1)
    if person_found:
        setattr(metrics, 'people_found', getattr(metrics, 'people_found') + 1)
    if research_task_created:
        setattr(metrics, 'research_tasks_created', getattr(metrics, 'research_tasks_created') + 1)


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
    metrics = _get_or_create_metrics(session, people_id)

    # Update message counts
    _update_message_metrics(metrics, message_sent, message_received, template_used)

    # Update engagement and phase
    _update_engagement_and_phase(metrics, engagement_score, conversation_phase)

    # Update conversation duration
    _update_conversation_duration(metrics)

    # Update research outcomes
    _update_research_outcomes(metrics, person_looked_up, person_found, research_task_created)

    # Update tree impact
    _update_tree_impact(metrics, added_to_tree)

    session.commit()
    return metrics


def _get_or_create_metrics(session: Session, people_id: int) -> ConversationMetrics:
    """Get or create metrics record for a person."""
    metrics = session.query(ConversationMetrics).filter_by(people_id=people_id).first()
    if not metrics:
        metrics = ConversationMetrics(people_id=people_id)
        session.add(metrics)
    return metrics


def _update_message_metrics(
    metrics: ConversationMetrics,
    message_sent: bool,
    message_received: bool,
    template_used: Optional[str]
) -> None:
    """Update message-related metrics."""
    if message_sent:
        _update_sent_message_metrics(metrics, template_used)
    if message_received:
        _update_received_message_metrics(metrics)


def _update_engagement_and_phase(
    metrics: ConversationMetrics,
    engagement_score: Optional[int],
    conversation_phase: Optional[str]
) -> None:
    """Update engagement score and conversation phase."""
    if engagement_score is not None:
        _update_engagement_metrics(metrics, engagement_score)
    if conversation_phase:
        setattr(metrics, 'conversation_phase', conversation_phase)


def _update_conversation_duration(metrics: ConversationMetrics) -> None:
    """Update conversation duration based on first and last message timestamps."""
    first_sent = getattr(metrics, 'first_message_sent')
    last_received = getattr(metrics, 'last_message_received')
    if first_sent is not None and last_received is not None:
        # Ensure both are timezone-aware
        if first_sent.tzinfo is None:
            first_sent = first_sent.replace(tzinfo=timezone.utc)
        if last_received.tzinfo is None:
            last_received = last_received.replace(tzinfo=timezone.utc)
        time_diff = last_received - first_sent
        setattr(metrics, 'conversation_duration_days', time_diff.total_seconds() / 86400)


def _update_tree_impact(metrics: ConversationMetrics, added_to_tree: bool) -> None:
    """Update tree impact metrics."""
    if added_to_tree and getattr(metrics, 'added_to_tree') is not True:
        setattr(metrics, 'added_to_tree', True)
        setattr(metrics, 'added_to_tree_date', datetime.now(timezone.utc))


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
        cast(ConversationMetrics.conversation_phase, SQLString),
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
def _test_database_models_available() -> None:
    """Test that database models are available."""
    assert ConversationMetrics is not None, "ConversationMetrics should be available"
    assert EngagementTracking is not None, "EngagementTracking should be available"


def _test_record_engagement_event() -> None:
    """Test recording an engagement event."""
    session = _create_database_session()

    try:
        # Ensure a valid person exists
        from database import Person
        person = session.query(Person).first()
        if not person:
            person = Person(username="Test Analytics User")
            session.add(person)
            session.commit()

        # Create a test event
        event = record_engagement_event(
            session=session,
            people_id=person.id,
            event_type="test_event",
            event_description="Test event description",
            event_data={"key": "value"},
            engagement_score_before=50,
            engagement_score_after=60,
            conversation_phase="initial",
            template_used="test_template"
        )

        # Verify event was created
        assert event is not None, "Event should be created"
        assert getattr(event, 'people_id') == person.id, "Event should have correct people_id"
        assert getattr(event, 'event_type') == "test_event", "Event should have correct event_type"
        assert getattr(event, 'engagement_score_delta') == 10, "Event should calculate delta correctly"

        # Clean up
        session.delete(event)
        session.commit()
    finally:
        session.close()


def _test_update_conversation_metrics_new() -> None:
    """Test updating conversation metrics for new conversation."""
    from database import Person
    session = _create_database_session()

    try:
        # Get a valid person from the database
        person = session.query(Person).first()
        if not person:
            # Skip test if no people in database
            return

        # Delete any existing metrics for this person
        existing_metrics = session.query(ConversationMetrics).filter_by(people_id=person.id).first()
        if existing_metrics:
            session.delete(existing_metrics)
            session.commit()

        # Update metrics for a new conversation
        metrics = update_conversation_metrics(
            session=session,
            people_id=person.id,
            message_sent=True,
            template_used="initial_contact",
            engagement_score=50,
            conversation_phase="initial"
        )

        # Verify metrics were created
        assert metrics is not None, "Metrics should be created"
        assert getattr(metrics, 'people_id') == person.id, "Metrics should have correct people_id"
        assert getattr(metrics, 'messages_sent') == 1, "Metrics should have 1 message sent"
        assert getattr(metrics, 'current_engagement_score') == 50, "Metrics should have correct engagement score"
        assert getattr(metrics, 'initial_template_used') == "initial_contact", "Metrics should track initial template"

        # Clean up
        session.delete(metrics)
        session.commit()
    finally:
        session.close()


def _test_update_conversation_metrics_existing() -> None:
    """Test updating existing conversation metrics."""
    from database import Person
    session = _create_database_session()

    try:
        # Get a valid person from the database
        person = session.query(Person).first()
        if not person:
            # Skip test if no people in database
            return

        # Delete any existing metrics for this person
        existing_metrics = session.query(ConversationMetrics).filter_by(people_id=person.id).first()
        if existing_metrics:
            session.delete(existing_metrics)
            session.commit()

        # Create initial metrics
        update_conversation_metrics(
            session=session,
            people_id=person.id,
            message_sent=True,
            template_used="initial_contact",
            engagement_score=50
        )

        # Update with a received message
        metrics2 = update_conversation_metrics(
            session=session,
            people_id=person.id,
            message_received=True,
            engagement_score=60
        )

        # Verify metrics were updated
        assert getattr(metrics2, 'messages_sent') == 1, "Should still have 1 message sent"
        assert getattr(metrics2, 'messages_received') == 1, "Should have 1 message received"
        assert getattr(metrics2, 'current_engagement_score') == 60, "Should have updated engagement score"
        assert getattr(metrics2, 'first_response_received') is True, "Should mark first response received"

        # Clean up
        session.delete(metrics2)
        session.commit()
    finally:
        session.close()


def _test_get_overall_analytics_empty() -> None:
    """Test getting overall analytics with no data."""
    session = _create_database_session()

    try:
        # Get analytics (may have existing data, so just verify structure)
        analytics = get_overall_analytics(session)

        # Verify structure
        assert isinstance(analytics, dict), "Analytics should be a dictionary"
        assert 'total_conversations' in analytics, "Should have total_conversations"
        assert 'response_rate_percent' in analytics, "Should have response_rate_percent"
        assert 'avg_engagement_score' in analytics, "Should have avg_engagement_score"
        assert 'template_effectiveness' in analytics, "Should have template_effectiveness"
        assert 'research_outcomes' in analytics, "Should have research_outcomes"
        assert 'tree_impact' in analytics, "Should have tree_impact"
    finally:
        session.close()


def _test_print_analytics_dashboard() -> None:
    """Test printing analytics dashboard."""
    session = _create_database_session()

    try:
        # Should not raise exception
        print_analytics_dashboard(session)
    finally:
        session.close()


def _test_engagement_score_delta_calculation() -> None:
    """Test engagement score delta calculation."""
    session = _create_database_session()

    try:
        # Ensure a valid person exists
        from database import Person
        person = session.query(Person).first()
        if not person:
            person = Person(username="Test Analytics User 2")
            session.add(person)
            session.commit()

        # Test with score increase
        event1 = record_engagement_event(
            session=session,
            people_id=person.id,
            event_type="test",
            engagement_score_before=40,
            engagement_score_after=60
        )
        assert getattr(event1, 'engagement_score_delta') == 20, "Should calculate positive delta"

        # Test with score decrease
        event2 = record_engagement_event(
            session=session,
            people_id=person.id,
            event_type="test",
            engagement_score_before=60,
            engagement_score_after=40
        )
        assert getattr(event2, 'engagement_score_delta') == -20, "Should calculate negative delta"

        # Test with no scores
        event3 = record_engagement_event(
            session=session,
            people_id=person.id,
            event_type="test"
        )
        assert getattr(event3, 'engagement_score_delta') is None, "Should have None delta when no scores"

        # Clean up
        session.delete(event1)
        session.delete(event2)
        session.delete(event3)
        session.commit()
    finally:
        session.close()


def _test_template_tracking() -> None:
    """Test template usage tracking."""
    from database import Person
    session = _create_database_session()

    try:
        # Get a valid person from the database
        person = session.query(Person).first()
        if not person:
            # Skip test if no people in database
            return

        # Delete any existing metrics for this person
        existing_metrics = session.query(ConversationMetrics).filter_by(people_id=person.id).first()
        if existing_metrics:
            session.delete(existing_metrics)
            session.commit()

        # Send multiple messages with different templates
        update_conversation_metrics(
            session=session,
            people_id=person.id,
            message_sent=True,
            template_used="template1"
        )

        metrics = update_conversation_metrics(
            session=session,
            people_id=person.id,
            message_sent=True,
            template_used="template2"
        )

        # Verify templates are tracked
        templates_json = getattr(metrics, 'templates_used')
        templates = json.loads(templates_json) if templates_json else []
        assert "template1" in templates, "Should track template1"
        assert "template2" in templates, "Should track template2"
        assert getattr(metrics, 'initial_template_used') == "template1", "Should track initial template"

        # Clean up
        session.delete(metrics)
        session.commit()
    finally:
        session.close()


def _test_research_outcomes_tracking() -> None:
    """Test research outcomes tracking."""
    from database import Person
    session = _create_database_session()

    try:
        # Get a valid person from the database
        person = session.query(Person).first()
        if not person:
            # Skip test if no people in database
            return

        # Delete any existing metrics for this person
        existing_metrics = session.query(ConversationMetrics).filter_by(people_id=person.id).first()
        if existing_metrics:
            session.delete(existing_metrics)
            session.commit()

        # Track research outcomes
        metrics = update_conversation_metrics(
            session=session,
            people_id=person.id,
            person_looked_up=True,
            person_found=True,
            research_task_created=True
        )

        # Verify tracking
        assert getattr(metrics, 'people_looked_up') == 1, "Should track people looked up"
        assert getattr(metrics, 'people_found') == 1, "Should track people found"
        assert getattr(metrics, 'research_tasks_created') == 1, "Should track research tasks"

        # Clean up
        session.delete(metrics)
        session.commit()
    finally:
        session.close()


def _test_tree_impact_tracking() -> None:
    """Test tree impact tracking."""
    from database import Person
    session = _create_database_session()

    try:
        # Get a valid person from the database
        person = session.query(Person).first()
        if not person:
            # Skip test if no people in database
            return

        # Delete any existing metrics for this person
        existing_metrics = session.query(ConversationMetrics).filter_by(people_id=person.id).first()
        if existing_metrics:
            session.delete(existing_metrics)
            session.commit()

        # Track tree addition
        metrics = update_conversation_metrics(
            session=session,
            people_id=person.id,
            added_to_tree=True
        )

        # Verify tracking
        assert getattr(metrics, 'added_to_tree') is True, "Should mark added to tree"
        assert getattr(metrics, 'added_to_tree_date') is not None, "Should record date added"

        # Clean up
        session.delete(metrics)
        session.commit()
    finally:
        session.close()


def conversation_analytics_module_tests() -> bool:
    """Comprehensive test suite for conversation_analytics.py"""
    from test_framework import TestSuite

    suite = TestSuite("Conversation Analytics", "conversation_analytics.py")
    suite.start_suite()

    # Category 1: Module Availability Tests
    suite.run_test(
        "Database models available",
        _test_database_models_available,
        "ConversationMetrics and EngagementTracking models are available",
        "ConversationMetrics, EngagementTracking",
        "Verifies database models can be imported"
    )

    # Category 2: Engagement Event Tests
    suite.run_test(
        "Record engagement event",
        _test_record_engagement_event,
        "Engagement events are recorded correctly",
        "record_engagement_event()",
        "Tests event creation with all parameters"
    )

    suite.run_test(
        "Engagement score delta calculation",
        _test_engagement_score_delta_calculation,
        "Engagement score deltas calculated correctly",
        "record_engagement_event()",
        "Tests positive, negative, and None delta calculations"
    )

    # Category 3: Conversation Metrics Tests
    suite.run_test(
        "Update conversation metrics - new",
        _test_update_conversation_metrics_new,
        "New conversation metrics created correctly",
        "update_conversation_metrics()",
        "Tests creating metrics for new conversation"
    )

    suite.run_test(
        "Update conversation metrics - existing",
        _test_update_conversation_metrics_existing,
        "Existing conversation metrics updated correctly",
        "update_conversation_metrics()",
        "Tests updating existing metrics with new data"
    )

    suite.run_test(
        "Template tracking",
        _test_template_tracking,
        "Template usage tracked correctly",
        "update_conversation_metrics()",
        "Tests initial template and templates list tracking"
    )

    # Category 4: Research Outcomes Tests
    suite.run_test(
        "Research outcomes tracking",
        _test_research_outcomes_tracking,
        "Research outcomes tracked correctly",
        "update_conversation_metrics()",
        "Tests people looked up, found, and research tasks"
    )

    # Category 5: Tree Impact Tests
    suite.run_test(
        "Tree impact tracking",
        _test_tree_impact_tracking,
        "Tree additions tracked correctly",
        "update_conversation_metrics()",
        "Tests added_to_tree flag and date"
    )

    # Category 6: Analytics Tests
    suite.run_test(
        "Get overall analytics",
        _test_get_overall_analytics_empty,
        "Overall analytics structure is correct",
        "get_overall_analytics()",
        "Verifies analytics dictionary structure"
    )

    suite.run_test(
        "Print analytics dashboard",
        _test_print_analytics_dashboard,
        "Analytics dashboard prints without error",
        "print_analytics_dashboard()",
        "Tests dashboard printing function"
    )

    return suite.finish_suite()


# Create standard test runner
from test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(conversation_analytics_module_tests)


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_comprehensive_tests() else 1)
