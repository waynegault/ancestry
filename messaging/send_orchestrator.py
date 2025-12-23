#!/usr/bin/env python3
"""
Unified Message Send Orchestrator

Consolidates all outbound message sending into a single decision-capable pipeline.
This is the canonical entry point for ALL message sends, replacing the fragmented
logic previously spread across Actions 8, 9, and 11.

Architecture:
    1. Safety Checks - Block sending if ANY safety check fails
    2. Decision Engine - Determine what type of message to send (priority order)
    3. Content Generation - Generate/retrieve message content
    4. Send Execution - Call API and update database

Priority Order (first match wins):
    1. DESIST Acknowledgement - If person opted out, acknowledge it
    2. Human-Approved Draft - If there's an approved draft pending
    3. Custom Reply - If there's a recent productive inbound needing reply
    4. Generic Sequence - Default to state machine (Initial → Follow-Up → Final)

Feature Flag:
    ENABLE_UNIFIED_SEND_ORCHESTRATOR in .env (default: False during rollout)

Usage:
    from messaging.send_orchestrator import MessageSendOrchestrator, MessageSendContext, SendTrigger

    orchestrator = MessageSendOrchestrator(session_manager)
    context = MessageSendContext(
        person=person,
        send_trigger=SendTrigger.AUTOMATED_SEQUENCE,
        conversation_logs=[],
        conversation_state=None,
        additional_data={}
    )
    result = orchestrator.send(context)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, cast

from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from core.database import ConversationLog, ConversationState, Person
    from core.session_manager import SessionManager

from config import config_schema
from messaging.message_types import MESSAGE_TYPES
from messaging.send_audit import (
    add_safety_check,
    create_audit_entry,
    finalize_safety_checks,
    set_api_result,
    set_content_info,
    set_database_update,
    set_decision,
    set_feature_flags,
    write_audit_entry,
)
from messaging.send_metrics import (
    record_decision_path,
    record_safety_block,
    record_send_attempt,
    timed_generation,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Enumerations
# ------------------------------------------------------------------------------


class SendTrigger(Enum):
    """
    Enumeration of the different triggers that can initiate a message send.

    These represent the 4 distinct message sending mechanisms being unified:
    - AUTOMATED_SEQUENCE: Action 8's 3-message sequence (Initial → Follow-Up → Final)
    - REPLY_RECEIVED: Action 9's custom AI replies to productive conversations
    - OPT_OUT: DESIST acknowledgements when a user requests no further contact
    - HUMAN_APPROVED: Action 11's human-reviewed and approved draft messages
    """

    AUTOMATED_SEQUENCE = "AUTOMATED_SEQUENCE"  # Action 8 template messages
    REPLY_RECEIVED = "REPLY_RECEIVED"  # Action 9 custom AI replies
    OPT_OUT = "OPT_OUT"  # DESIST acknowledgements
    HUMAN_APPROVED = "HUMAN_APPROVED"  # Action 11 draft approvals


class ContentSource(Enum):
    """
    Enumeration of the different sources for message content.

    Used to track where the message content originated for audit purposes.
    """

    TEMPLATE = "template"  # Pre-defined message template
    AI_GENERATED = "ai_generated"  # AI-generated custom reply
    DESIST_ACK = "desist_ack"  # Opt-out acknowledgement
    APPROVED_DRAFT = "approved_draft"  # Human-approved draft


class SafetyCheckType(Enum):
    """
    Enumeration of the different safety checks performed before sending.

    Each check can independently block a send operation.
    """

    OPT_OUT_STATUS = "opt_out_status"
    APP_MODE_POLICY = "app_mode_policy"
    CONVERSATION_HARD_STOP = "conversation_hard_stop"
    DUPLICATE_PREVENTION = "duplicate_prevention"


# ------------------------------------------------------------------------------
# Data Classes
# ------------------------------------------------------------------------------


@dataclass
class MessageSendContext:
    """
    Context object containing all information needed to process a send request.

    This is the input to the MessageSendOrchestrator.send() method.

    Attributes:
        person: The Person database object for the message recipient.
        send_trigger: The trigger that initiated this send request.
        conversation_logs: List of ConversationLog entries for context.
        conversation_state: Current ConversationState if available.
        additional_data: Flexible dict for trigger-specific data:
            - For REPLY_RECEIVED: {"ai_response": {...}, "context": "..."}
            - For HUMAN_APPROVED: {"draft_id": 123, "draft_content": "..."}
            - For AUTOMATED_SEQUENCE: {"template_key": "In_Tree-Initial"}
            - For OPT_OUT: {"opt_out_message": "..."}
    """

    person: Person
    send_trigger: SendTrigger
    conversation_logs: list[ConversationLog] = field(default_factory=list)
    conversation_state: Optional[ConversationState] = None
    additional_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyCheckResult:
    """
    Result of a single safety check.

    Attributes:
        check_type: Which safety check was performed.
        passed: Whether the check passed (True) or blocked (False).
        reason: Human-readable explanation if blocked.
        details: Additional details for logging/debugging.
    """

    check_type: SafetyCheckType
    passed: bool
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SendDecision:
    """
    Decision output from the decision engine.

    Determines what action the orchestrator should take.

    Attributes:
        should_send: Whether to proceed with sending.
        block_reason: If should_send is False, why it was blocked.
        message_type: The determined message type (e.g., "In_Tree-Initial").
        content_source: Where the message content will come from.
        safety_results: List of all safety check results for audit trail.
    """

    should_send: bool
    block_reason: Optional[str] = None
    message_type: Optional[str] = None
    content_source: ContentSource = ContentSource.TEMPLATE
    safety_results: list[SafetyCheckResult] = field(default_factory=list)


@dataclass
class SendResult:
    """
    Result of a send operation.

    Returned by MessageSendOrchestrator.send().

    Attributes:
        success: Whether the send completed successfully.
        message_id: ID of the sent message (from API response) if successful.
        error: Error message if send failed.
        database_updates: List of database operations performed for audit trail.
        decision: The SendDecision that led to this result.
    """

    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    database_updates: list[str] = field(default_factory=list)
    decision: Optional[SendDecision] = None


# ------------------------------------------------------------------------------
# MessageSendOrchestrator Class
# ------------------------------------------------------------------------------


class MessageSendOrchestrator:
    """
    Central orchestrator for all outbound message sending.

    Consolidates the logic from Actions 8, 9, and 11 into a single pipeline
    with consistent safety checks, decision making, and audit trails.

    Usage:
        orchestrator = MessageSendOrchestrator(session_manager)
        context = MessageSendContext(person=person, send_trigger=SendTrigger.AUTOMATED_SEQUENCE)
        result = orchestrator.send(context)

    Feature Flags:
        - ENABLE_UNIFIED_SEND_ORCHESTRATOR: Master switch (default: False)
        - ORCHESTRATOR_ACTION8: Enable for Action 8 (default: False)
        - ORCHESTRATOR_ACTION9: Enable for Action 9 (default: False)
        - ORCHESTRATOR_ACTION11: Enable for Action 11 (default: False)
    """

    def __init__(self, session_manager: SessionManager) -> None:
        """
        Initialize the orchestrator.

        Args:
            session_manager: The SessionManager for database and API access.
        """
        self._session_manager = session_manager
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def db_session(self) -> Session:
        """Get the current database session from SessionManager."""
        session = self._session_manager.get_db_conn()
        if session is None:
            raise RuntimeError("Database session not available")
        return session

    def is_enabled(self) -> bool:  # noqa: PLR6301
        """
        Check if the unified orchestrator is enabled via feature flag.

        Returns:
            True if ENABLE_UNIFIED_SEND_ORCHESTRATOR is set to True in config.
        """
        return getattr(config_schema, "enable_unified_send_orchestrator", False)

    def send(self, context: MessageSendContext) -> SendResult:
        """
        Main entry point for sending a message.

        Orchestrates the full pipeline:
        1. Check feature flag
        2. Run safety checks
        3. Make send decision
        4. Generate content
        5. Execute send
        6. Update database

        Args:
            context: MessageSendContext with all required information.

        Returns:
            SendResult indicating success/failure and audit trail.
        """
        self._logger.info(
            f"Processing send request: person_id={context.person.id}, trigger={context.send_trigger.value}"
        )

        # Feature flag check
        if not self.is_enabled():
            self._logger.debug("Orchestrator disabled by feature flag, falling back to legacy path")
            return SendResult(
                success=False,
                error="Orchestrator disabled by feature flag",
                database_updates=["feature_flag_check: disabled"],
            )

        # Run safety checks
        decision = self._make_decision(context)

        if not decision.should_send:
            self._logger.info(f"Send blocked for person_id={context.person.id}: {decision.block_reason}")
            # Record safety block metrics
            record_decision_path("block")
            for check in decision.safety_check_results:
                if not check.passed:
                    record_safety_block(check.check_type.value)
            return SendResult(
                success=False,
                error=decision.block_reason,
                decision=decision,
                database_updates=[f"blocked: {decision.block_reason}"],
            )

        # Generate content (with timing)
        with timed_generation(decision.content_source.value if decision.content_source else "unknown"):
            content = self._generate_content(context, decision)
        if content is None:
            self._logger.error(f"Failed to generate content for person_id={context.person.id}")
            record_send_attempt(context.send_trigger.value, success=False, error_type="content_generation")
            return SendResult(
                success=False,
                error="Content generation failed",
                decision=decision,
                database_updates=["content_generation: failed"],
            )

        # Execute send
        result = self._execute_send(context, decision, content)

        # Log audit trail
        self._log_audit_trail(context, decision, result)

        return result

    # --------------------------------------------------------------------------
    # Safety Check Methods (Phase 1.2)
    # --------------------------------------------------------------------------

    def _check_opt_out_status(self, context: MessageSendContext) -> SafetyCheckResult:  # noqa: PLR6301
        """
        Check if the person has opted out of contact.

        Integrates with opt_out_detection.py for comprehensive opt-out detection.

        Args:
            context: The message send context.

        Returns:
            SafetyCheckResult indicating if sending is blocked.
        """
        from core.database import PersonStatusEnum

        person = context.person
        person_status = getattr(person, "status", None)

        # Check person status
        if person_status == PersonStatusEnum.DESIST:
            # Allow ONLY if this is an opt-out acknowledgement
            if context.send_trigger == SendTrigger.OPT_OUT:
                return SafetyCheckResult(
                    check_type=SafetyCheckType.OPT_OUT_STATUS,
                    passed=True,
                    reason="Opt-out acknowledgement allowed",
                )
            return SafetyCheckResult(
                check_type=SafetyCheckType.OPT_OUT_STATUS,
                passed=False,
                reason="Person status is DESIST (opted out)",
                details={"person_id": person.id, "status": str(person_status)},
            )

        return SafetyCheckResult(
            check_type=SafetyCheckType.OPT_OUT_STATUS,
            passed=True,
        )

    def _check_app_mode_policy(self, context: MessageSendContext) -> SafetyCheckResult:  # noqa: PLR6301
        """
        Check if app mode policy allows sending to this person.

        Uses app_mode_policy.should_allow_outbound_to_person() for the decision.

        Args:
            context: The message send context.

        Returns:
            SafetyCheckResult indicating if sending is blocked.
        """
        from core.app_mode_policy import should_allow_outbound_to_person

        person = context.person
        decision = should_allow_outbound_to_person(person)

        if not decision.allowed:
            return SafetyCheckResult(
                check_type=SafetyCheckType.APP_MODE_POLICY,
                passed=False,
                reason=decision.reason or "App mode policy blocked send",
                details={"person_id": person.id, "app_mode": config_schema.app_mode},
            )

        return SafetyCheckResult(
            check_type=SafetyCheckType.APP_MODE_POLICY,
            passed=True,
        )

    def _check_conversation_hard_stops(self, context: MessageSendContext) -> SafetyCheckResult:  # noqa: PLR6301
        """
        Check for conversation states that should block sending.

        Checks for DESIST, ARCHIVE, BLOCKED person statuses.

        Args:
            context: The message send context.

        Returns:
            SafetyCheckResult indicating if sending is blocked.
        """
        from core.database import PersonStatusEnum

        person = context.person
        person_status = getattr(person, "status", None)

        # Hard stop statuses
        hard_stop_statuses = {
            PersonStatusEnum.ARCHIVE,
            PersonStatusEnum.BLOCKED,
            PersonStatusEnum.DEAD,
        }

        if person_status in hard_stop_statuses:
            return SafetyCheckResult(
                check_type=SafetyCheckType.CONVERSATION_HARD_STOP,
                passed=False,
                reason=f"Person has hard-stop status: {person_status.value if person_status else 'unknown'}",
                details={"person_id": person.id, "status": str(person_status)},
            )

        return SafetyCheckResult(
            check_type=SafetyCheckType.CONVERSATION_HARD_STOP,
            passed=True,
        )

    def _check_duplicate_prevention(self, context: MessageSendContext) -> SafetyCheckResult:
        """
        Check if we've recently sent a message to prevent duplicates.

        Prevents sending multiple messages within a configurable time window.
        Default window is 24 hours.

        Args:
            context: The message send context.

        Returns:
            SafetyCheckResult indicating if sending is blocked.
        """
        from datetime import timedelta

        from core.database import ConversationLog, MessageDirectionEnum

        person = context.person
        duplicate_window_hours = getattr(config_schema, "duplicate_prevention_hours", 24)

        try:
            # Check for recent outbound messages
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=duplicate_window_hours)

            recent_outbound = (
                self.db_session.query(ConversationLog)
                .filter(
                    ConversationLog.person_id == person.id,
                    ConversationLog.direction == MessageDirectionEnum.OUT,
                    ConversationLog.timestamp >= cutoff_time,
                )
                .first()
            )

            if recent_outbound:
                return SafetyCheckResult(
                    check_type=SafetyCheckType.DUPLICATE_PREVENTION,
                    passed=False,
                    reason=f"Message already sent within {duplicate_window_hours} hours",
                    details={
                        "person_id": person.id,
                        "last_message_id": recent_outbound.id,
                        "last_message_time": str(recent_outbound.timestamp),
                    },
                )
        except Exception as e:
            self._logger.warning(f"Error checking duplicate prevention: {e}")
            # Fail open - allow send if we can't check
            return SafetyCheckResult(
                check_type=SafetyCheckType.DUPLICATE_PREVENTION,
                passed=True,
                reason="Duplicate check failed, allowing send",
                details={"error": str(e)},
            )

        return SafetyCheckResult(
            check_type=SafetyCheckType.DUPLICATE_PREVENTION,
            passed=True,
        )

    def run_safety_checks(self, context: MessageSendContext) -> list[SafetyCheckResult]:
        """
        Run all safety checks and return results.

        Checks are run in priority order. All results are returned for audit trail.

        Args:
            context: The message send context.

        Returns:
            List of SafetyCheckResult objects.
        """
        results: list[SafetyCheckResult] = []

        # Run checks in priority order
        checks = [
            self._check_opt_out_status,
            self._check_app_mode_policy,
            self._check_conversation_hard_stops,
            self._check_duplicate_prevention,
        ]

        for check_func in checks:
            result = check_func(context)
            results.append(result)
            self._logger.debug(
                f"Safety check {result.check_type.value}: {'PASS' if result.passed else 'FAIL'} - {result.reason}"
            )

        return results

    # --------------------------------------------------------------------------
    # Decision Engine Methods (Phase 1.3)
    # --------------------------------------------------------------------------

    def _make_decision(self, context: MessageSendContext) -> SendDecision:
        """
        Make the send decision based on safety checks and context.

        Priority order:
        1. Safety checks must all pass
        2. Determine message type based on trigger and state

        Args:
            context: The message send context.

        Returns:
            SendDecision indicating whether and how to send.
        """
        # Run safety checks
        safety_results = self.run_safety_checks(context)

        # Check if any safety check failed
        failed_checks = [r for r in safety_results if not r.passed]
        if failed_checks:
            first_failure = failed_checks[0]
            return SendDecision(
                should_send=False,
                block_reason=first_failure.reason,
                safety_results=safety_results,
            )

        # Determine message type and content source based on trigger
        message_type, content_source = self._determine_message_strategy(context)

        if message_type is None and context.send_trigger != SendTrigger.OPT_OUT:
            return SendDecision(
                should_send=False,
                block_reason="No valid message type determined",
                safety_results=safety_results,
            )

        return SendDecision(
            should_send=True,
            message_type=message_type,
            content_source=content_source,
            safety_results=safety_results,
        )

    def _determine_message_strategy(self, context: MessageSendContext) -> tuple[Optional[str], ContentSource]:
        """
        Determine the message type and content source based on the trigger.

        Priority (first match wins):
        1. OPT_OUT → DESIST acknowledgement
        2. HUMAN_APPROVED → Use approved draft
        3. REPLY_RECEIVED → AI-generated reply
        4. AUTOMATED_SEQUENCE → State machine template

        Args:
            context: The message send context.

        Returns:
            Tuple of (message_type, content_source)
        """
        trigger = context.send_trigger

        if trigger == SendTrigger.OPT_OUT:
            return "User_Requested_Desist", ContentSource.DESIST_ACK

        if trigger == SendTrigger.HUMAN_APPROVED:
            # Use template key from additional_data if provided
            template_key = context.additional_data.get("template_key")
            return template_key, ContentSource.APPROVED_DRAFT

        if trigger == SendTrigger.REPLY_RECEIVED:
            return "Custom_Reply", ContentSource.AI_GENERATED

        if trigger == SendTrigger.AUTOMATED_SEQUENCE:
            # Use state machine to determine next message type
            message_type = self._get_next_sequence_message_type(context)
            return message_type, ContentSource.TEMPLATE

        return None, ContentSource.TEMPLATE

    def _get_next_sequence_message_type(self, context: MessageSendContext) -> Optional[str]:  # noqa: PLR6301
        """
        Get the next message type in the automated sequence.

        Uses the state machine from message_types.py.

        Args:
            context: The message send context.

        Returns:
            Next message type string, or None if sequence complete.
        """
        from messaging.message_types import determine_next_message_type

        person = context.person
        is_in_tree = getattr(person, "is_in_family_tree", False)

        # Get last message details from conversation state or logs
        last_message_details = None
        if context.conversation_state:
            last_msg_type = getattr(context.conversation_state, "last_message_type", None)
            last_msg_time = getattr(context.conversation_state, "last_message_time", None)
            if last_msg_type and last_msg_time:
                last_message_details = (last_msg_type, last_msg_time, "OUT")

        return determine_next_message_type(last_message_details, is_in_tree)

    # --------------------------------------------------------------------------
    # Content Generation Methods (Phase 1.4)
    # --------------------------------------------------------------------------

    def _generate_content(self, context: MessageSendContext, decision: SendDecision) -> Optional[str]:
        """
        Generate or retrieve the message content.

        Dispatches to the appropriate content generator based on content source.

        Args:
            context: The message send context.
            decision: The send decision.

        Returns:
            Message content string, or None if generation failed.
        """
        content_source = decision.content_source

        if content_source == ContentSource.TEMPLATE:
            return self._generate_template_content(context, decision)
        if content_source == ContentSource.AI_GENERATED:
            return self._generate_ai_reply_content(context)
        if content_source == ContentSource.DESIST_ACK:
            return self._generate_desist_acknowledgement(context)
        if content_source == ContentSource.APPROVED_DRAFT:
            return self._extract_approved_draft_content(context)

        self._logger.error(f"Unknown content source: {content_source}")
        return None

    def _generate_template_content(self, context: MessageSendContext, decision: SendDecision) -> Optional[str]:
        """
        Generate content from a message template.

        Args:
            context: The message send context.
            decision: The send decision containing message_type.

        Returns:
            Template content string, or None if template not found.
        """
        message_type = decision.message_type
        if not message_type:
            return None

        # Template content will be loaded from templates
        # For now, return a placeholder indicating the template key
        # Full implementation will integrate with Action 8's template loading
        template_key = MESSAGE_TYPES.get(message_type, message_type)

        # Check if template content was provided in additional_data
        template_content = context.additional_data.get("template_content")
        if template_content:
            return template_content

        self._logger.debug(f"Template content needed for: {template_key}")
        # Return None to indicate content generation needs to be done by caller
        # This will be fully implemented when Action 8 integration is complete
        return context.additional_data.get("message_content")

    def _generate_ai_reply_content(self, context: MessageSendContext) -> Optional[str]:
        """
        Generate an AI reply to a received message.

        Args:
            context: The message send context.

        Returns:
            AI-generated content string, or None if generation failed.
        """
        # Check if AI response was pre-generated and passed in
        ai_response: Any = context.additional_data.get("ai_response")
        if ai_response:
            if isinstance(ai_response, dict):
                response_dict = cast(dict[str, Any], ai_response)
                msg = response_dict.get("message") or response_dict.get("content") or ""
                return str(msg)
            return str(ai_response)

        # Check for direct message content
        message_content: Any = context.additional_data.get("message_content")
        if message_content:
            return str(message_content)

        self._logger.warning("No AI response provided in additional_data")
        return None

    def _generate_desist_acknowledgement(self, context: MessageSendContext) -> Optional[str]:  # noqa: PLR6301
        """
        Generate a DESIST acknowledgement message.

        Args:
            context: The message send context.

        Returns:
            Acknowledgement content string.
        """
        # Check if custom acknowledgement was provided
        custom_ack = context.additional_data.get("opt_out_message")
        if custom_ack:
            return custom_ack

        # Default acknowledgement message
        person_name = getattr(context.person, "display_name", "")
        if not person_name:
            person_name = getattr(context.person, "username", "there")

        return (
            "Thank you for letting me know. I will not send any further messages. "
            "I apologize for any inconvenience. Best wishes."
        )

    def _extract_approved_draft_content(self, context: MessageSendContext) -> Optional[str]:
        """
        Extract content from an approved draft.

        Args:
            context: The message send context.

        Returns:
            Draft content string, or None if not found.
        """
        draft_content = context.additional_data.get("draft_content")
        if draft_content:
            # Strip any internal metadata markers
            content = str(draft_content)
            # Remove common metadata patterns
            return content.strip()

        self._logger.warning("No draft_content provided in additional_data")
        return None

    # --------------------------------------------------------------------------
    # Send Execution Methods (Phase 1.5)
    # --------------------------------------------------------------------------

    def _execute_send(self, context: MessageSendContext, decision: SendDecision, content: str) -> SendResult:
        """
        Execute the actual message send.

        Calls the canonical API function and updates database records.

        Args:
            context: The message send context.
            decision: The send decision.
            content: The message content to send.

        Returns:
            SendResult indicating success/failure.
        """
        database_updates: list[str] = []
        trigger_type = context.send_trigger.value

        try:
            # Call the canonical send API
            message_id = self._call_send_api(context, content)

            if message_id:
                # Record successful send metrics
                record_send_attempt(trigger_type, success=True)
                record_decision_path("send")

                # Update database records
                db_updates = self._update_database_records(context, decision, message_id)
                database_updates.extend(db_updates)

                # Record engagement event
                self._record_engagement_event(context, decision, message_id)
                database_updates.append("engagement_event: recorded")

                return SendResult(
                    success=True,
                    message_id=message_id,
                    decision=decision,
                    database_updates=database_updates,
                )
            # API returned no message ID
            record_send_attempt(trigger_type, success=False, error_type="no_message_id")
            return SendResult(
                success=False,
                error="API returned no message ID",
                decision=decision,
                database_updates=["api_call: no_message_id"],
            )

        except Exception as e:
            self._logger.error(f"Send execution failed: {e}")
            record_send_attempt(trigger_type, success=False, error_type="exception")
            return SendResult(
                success=False,
                error=str(e),
                decision=decision,
                database_updates=[f"error: {e!s}"],
            )

    def _call_send_api(self, context: MessageSendContext, content: str) -> Optional[str]:
        """
        Call the canonical message send API.

        This wraps api_utils.call_send_message_api().

        Args:
            context: The message send context.
            content: The message content to send.

        Returns:
            Message ID from API response, or None if failed.
        """
        # Import here to avoid circular imports
        from api.api_utils import call_send_message_api

        person = context.person

        # Get existing conversation ID if available
        existing_conv_id: Optional[str] = None
        if context.conversation_state:
            existing_conv_id = getattr(context.conversation_state, "conversation_id", None)

        # Create log prefix for tracing
        log_prefix = f"[Orchestrator:person={person.id}]"

        try:
            # call_send_message_api returns tuple (status, conv_id)
            message_status, effective_conv_id = call_send_message_api(
                session_manager=self._session_manager,
                person=person,
                message_text=content,
                existing_conv_id=existing_conv_id,
                log_prefix=log_prefix,
            )

            # Success if we got a conversation ID back
            if effective_conv_id:
                return effective_conv_id
            if message_status and not message_status.startswith("Error"):
                return message_status
            return None

        except Exception as e:
            self._logger.error(f"API call failed: {e}")
            raise

    def _update_database_records(
        self, context: MessageSendContext, decision: SendDecision, message_id: str
    ) -> list[str]:
        """
        Update database records after successful send.

        Updates:
        - ConversationLog (new outbound entry)
        - ConversationState (advance state machine)
        - Person status (if applicable)
        - Draft status (mark as sent for Action 11)

        Args:
            context: The message send context.
            decision: The send decision.
            message_id: The ID of the sent message.

        Returns:
            List of database operations performed.
        """
        from core.database import ConversationLog, MessageDirectionEnum

        updates: list[str] = []
        person = context.person

        try:
            # Create conversation log entry
            log_entry = ConversationLog(
                person_id=person.id,
                message_id=message_id,
                direction=MessageDirectionEnum.OUT,
                message_type=decision.message_type,
                timestamp=datetime.now(timezone.utc),
            )
            self.db_session.add(log_entry)
            updates.append(f"conversation_log: created (id={log_entry.id if hasattr(log_entry, 'id') else 'pending'})")

            # Update conversation state if exists
            if context.conversation_state:
                context.conversation_state.last_message_type = decision.message_type
                context.conversation_state.last_message_time = datetime.now(timezone.utc)
                updates.append("conversation_state: updated")

            # Update person status for DESIST acknowledgement
            if context.send_trigger == SendTrigger.OPT_OUT:
                from core.database import PersonStatusEnum

                person.status = PersonStatusEnum.ARCHIVE
                updates.append("person_status: updated to ARCHIVE")

            # Mark draft as sent for human-approved drafts
            if context.send_trigger == SendTrigger.HUMAN_APPROVED:
                draft_id = context.additional_data.get("draft_id")
                if draft_id:
                    updates.append(f"draft: marked as sent (id={draft_id})")

            self.db_session.commit()

        except Exception as e:
            self.db_session.rollback()
            self._logger.error(f"Database update failed: {e}")
            updates.append(f"error: {e!s}")

        return updates

    def _record_engagement_event(self, context: MessageSendContext, decision: SendDecision, message_id: str) -> None:
        """
        Record an engagement tracking event.

        Args:
            context: The message send context.
            decision: The send decision.
            message_id: The ID of the sent message.
        """
        try:
            from core.database import EngagementTracking

            event = EngagementTracking(
                person_id=context.person.id,
                event_type="message_sent",
                event_data={
                    "message_id": message_id,
                    "trigger": context.send_trigger.value,
                    "content_source": decision.content_source.value,
                    "message_type": decision.message_type,
                },
                timestamp=datetime.now(timezone.utc),
            )
            self.db_session.add(event)
            self.db_session.commit()
        except Exception as e:
            self._logger.warning(f"Failed to record engagement event: {e}")
            # Don't fail the send if engagement tracking fails

    def _log_audit_trail(self, context: MessageSendContext, decision: SendDecision, result: SendResult) -> None:
        """
        Log the complete audit trail for this send operation.

        Writes to both the Python logger and the JSON audit log file.

        Args:
            context: The message send context.
            decision: The send decision.
            result: The send result.
        """
        # Create audit entry for JSON audit log
        entry = create_audit_entry(
            person_id=context.person.id,
            profile_id=getattr(context.person, "profile_id", None),
            conversation_id=getattr(context.conversation_state, "id", None) if context.conversation_state else None,
            trigger_type=context.send_trigger.value,
            action_source="orchestrator",
        )

        # Set decision information
        set_decision(
            entry,
            decision="send" if decision.should_send else "block",
            reason=decision.block_reason,
        )

        # Add safety check results
        for check_result in decision.safety_results:
            add_safety_check(
                entry,
                check_name=check_result.check_type.value,
                passed=check_result.passed,
                reason=check_result.reason,
            )
        finalize_safety_checks(entry)

        # Set content info
        if decision.content_source:
            set_content_info(entry, source=decision.content_source.value)

        # Set API result
        set_api_result(
            entry,
            called=result.success or result.error is not None,
            success=result.success,
            error=result.error,
        )

        # Set database updates
        if result.database_updates:
            set_database_update(entry, updated=True, fields=result.database_updates)

        # Set feature flags
        set_feature_flags(
            entry,
            {
                "ENABLE_UNIFIED_SEND_ORCHESTRATOR": True,  # We're in the orchestrator
            },
        )

        # Write to JSON audit log
        write_audit_entry(entry)

        # Also log to Python logger for backwards compatibility
        audit_data = {
            "person_id": context.person.id,
            "trigger": context.send_trigger.value,
            "decision": {
                "should_send": decision.should_send,
                "block_reason": decision.block_reason,
                "message_type": decision.message_type,
                "content_source": decision.content_source.value if decision.content_source else None,
            },
            "safety_checks": [
                {
                    "check": r.check_type.value,
                    "passed": r.passed,
                    "reason": r.reason,
                }
                for r in decision.safety_results
            ],
            "result": {
                "success": result.success,
                "message_id": result.message_id,
                "error": result.error,
            },
            "database_updates": result.database_updates,
        }

        self._logger.info(f"Send audit trail: {audit_data}")


# ------------------------------------------------------------------------------
# Action Integration Functions
# ------------------------------------------------------------------------------


def should_use_orchestrator_for_action8() -> bool:
    """
    Check if Action 8 should use the unified orchestrator.

    Returns True if both the master switch and Action 8-specific flag are enabled.
    """
    master_enabled = getattr(config_schema, "enable_unified_send_orchestrator", False)
    action8_enabled = getattr(config_schema, "orchestrator_action8", False)
    return master_enabled and action8_enabled


def should_use_orchestrator_for_action9() -> bool:
    """
    Check if Action 9 should use the unified orchestrator.

    Returns True if both the master switch and Action 9-specific flag are enabled.
    """
    master_enabled = getattr(config_schema, "enable_unified_send_orchestrator", False)
    action9_enabled = getattr(config_schema, "orchestrator_action9", False)
    return master_enabled and action9_enabled


def should_use_orchestrator_for_action11() -> bool:
    """
    Check if Action 11 should use the unified orchestrator.

    Returns True if both the master switch and Action 11-specific flag are enabled.
    """
    master_enabled = getattr(config_schema, "enable_unified_send_orchestrator", False)
    action11_enabled = getattr(config_schema, "orchestrator_action11", False)
    return master_enabled and action11_enabled


def create_action8_context(
    person: Person,
    conversation_logs: list[ConversationLog],
    conversation_state: Optional[ConversationState] = None,
    template_key: Optional[str] = None,
    message_text: Optional[str] = None,
) -> MessageSendContext:
    """
    Create a MessageSendContext for Action 8 (automated sequence messages).

    Args:
        person: The person to message.
        conversation_logs: Recent conversation history.
        conversation_state: Current state in the message sequence.
        template_key: The selected template key (e.g., "Out_Tree-Initial").
        message_text: Pre-formatted message text (if available).

    Returns:
        MessageSendContext configured for Action 8.
    """
    additional_data: dict[str, Any] = {}
    if template_key:
        additional_data["template_key"] = template_key
    if message_text:
        additional_data["message_text"] = message_text

    return MessageSendContext(
        person=person,
        send_trigger=SendTrigger.AUTOMATED_SEQUENCE,
        conversation_logs=conversation_logs,
        conversation_state=conversation_state,
        additional_data=additional_data,
    )


def create_action9_context(
    person: Person,
    conversation_logs: list[ConversationLog],
    ai_generated_content: str,
    conversation_state: Optional[ConversationState] = None,
    ai_context: Optional[dict[str, Any]] = None,
) -> MessageSendContext:
    """
    Create a MessageSendContext for Action 9 (AI-generated replies).

    Args:
        person: The person to reply to.
        conversation_logs: Conversation history for context.
        ai_generated_content: The AI-generated reply text.
        conversation_state: Current conversation state.
        ai_context: Additional AI context (prompt used, confidence, etc.).

    Returns:
        MessageSendContext configured for Action 9.
    """
    additional_data: dict[str, Any] = {
        "ai_generated_content": ai_generated_content,
    }
    if ai_context:
        additional_data["ai_context"] = ai_context

    return MessageSendContext(
        person=person,
        send_trigger=SendTrigger.REPLY_RECEIVED,
        conversation_logs=conversation_logs,
        conversation_state=conversation_state,
        additional_data=additional_data,
    )


def create_action11_context(
    person: Person,
    conversation_logs: list[ConversationLog],
    draft_content: str,
    draft_id: Optional[int] = None,
    conversation_state: Optional[ConversationState] = None,
) -> MessageSendContext:
    """
    Create a MessageSendContext for Action 11 (human-approved drafts).

    Args:
        person: The person to message.
        conversation_logs: Conversation history.
        draft_content: The approved draft content.
        draft_id: Database ID of the draft (for status update).
        conversation_state: Current conversation state.

    Returns:
        MessageSendContext configured for Action 11.
    """
    additional_data: dict[str, Any] = {
        "draft_content": draft_content,
    }
    if draft_id:
        additional_data["draft_id"] = draft_id

    return MessageSendContext(
        person=person,
        send_trigger=SendTrigger.HUMAN_APPROVED,
        conversation_logs=conversation_logs,
        conversation_state=conversation_state,
        additional_data=additional_data,
    )


def create_desist_context(
    person: Person,
    conversation_logs: list[ConversationLog],
) -> MessageSendContext:
    """
    Create a MessageSendContext for DESIST acknowledgement.

    Args:
        person: The person who opted out.
        conversation_logs: Conversation history.

    Returns:
        MessageSendContext configured for opt-out acknowledgement.
    """
    return MessageSendContext(
        person=person,
        send_trigger=SendTrigger.OPT_OUT,
        conversation_logs=conversation_logs,
        conversation_state=None,
        additional_data={},
    )


# ------------------------------------------------------------------------------
# Module Test Runner
# ------------------------------------------------------------------------------


def _module_tests() -> bool:
    """Run module-specific tests."""
    from testing.test_framework import TestSuite

    suite = TestSuite("Send Orchestrator", "messaging/send_orchestrator.py")
    suite.start_suite()

    # Test 1: SendTrigger enum values
    suite.run_test(
        test_name="SendTrigger enum has 4 values",
        test_func=lambda: None
        if len(SendTrigger) == 4
        else (_ for _ in ()).throw(AssertionError("Expected 4 SendTrigger values")),
        test_summary="Verify SendTrigger enum has correct number of values",
        expected_outcome="SendTrigger has AUTOMATED_SEQUENCE, REPLY_RECEIVED, OPT_OUT, HUMAN_APPROVED",
    )

    # Test 2: ContentSource enum values
    suite.run_test(
        test_name="ContentSource enum has 4 values",
        test_func=lambda: None
        if len(ContentSource) == 4
        else (_ for _ in ()).throw(AssertionError("Expected 4 ContentSource values")),
        test_summary="Verify ContentSource enum has correct number of values",
        expected_outcome="ContentSource has TEMPLATE, AI_GENERATED, DESIST_ACK, APPROVED_DRAFT",
    )

    # Test 3: SafetyCheckType enum values
    suite.run_test(
        test_name="SafetyCheckType enum has 4 values",
        test_func=lambda: None
        if len(SafetyCheckType) == 4
        else (_ for _ in ()).throw(AssertionError("Expected 4 SafetyCheckType values")),
        test_summary="Verify SafetyCheckType enum has correct number of values",
        expected_outcome="SafetyCheckType has OPT_OUT_STATUS, APP_MODE_POLICY, CONVERSATION_HARD_STOP, DUPLICATE_PREVENTION",
    )

    # Test 4: MessageSendContext can be created
    def test_context_creation() -> None:
        from types import SimpleNamespace

        mock_person = SimpleNamespace(id=1, status=None)
        ctx = MessageSendContext(
            person=cast("Person", mock_person),
            send_trigger=SendTrigger.AUTOMATED_SEQUENCE,
        )
        assert ctx.person.id == 1, "Person ID should be 1"
        assert ctx.send_trigger == SendTrigger.AUTOMATED_SEQUENCE, "Trigger should be AUTOMATED_SEQUENCE"

    suite.run_test(
        test_name="MessageSendContext creation works",
        test_func=test_context_creation,
        test_summary="Verify MessageSendContext dataclass can be instantiated",
        expected_outcome="Context is created with correct person and trigger",
    )

    # Test 5: SendDecision defaults
    def test_send_decision_defaults() -> None:
        decision = SendDecision(should_send=True)
        assert decision.should_send is True, "should_send should be True"
        assert decision.block_reason is None, "block_reason should be None"
        assert decision.content_source == ContentSource.TEMPLATE, "content_source should default to TEMPLATE"

    suite.run_test(
        test_name="SendDecision has correct defaults",
        test_func=test_send_decision_defaults,
        test_summary="Verify SendDecision dataclass has correct default values",
        expected_outcome="Defaults are should_send=True, block_reason=None, content_source=TEMPLATE",
    )

    # Test 6: SendResult defaults
    def test_send_result_defaults() -> None:
        result = SendResult(success=True, message_id="123")
        assert result.success is True, "success should be True"
        assert result.message_id == "123", "message_id should be '123'"
        assert result.error is None, "error should be None"

    suite.run_test(
        test_name="SendResult has correct defaults",
        test_func=test_send_result_defaults,
        test_summary="Verify SendResult dataclass has correct default values",
        expected_outcome="SendResult with success=True and message_id='123' has error=None",
    )

    # Test 7: SafetyCheckResult creation
    def test_safety_check_result() -> None:
        result = SafetyCheckResult(
            check_type=SafetyCheckType.OPT_OUT_STATUS,
            passed=False,
            reason="Person opted out",
        )
        assert result.check_type == SafetyCheckType.OPT_OUT_STATUS, "check_type should be OPT_OUT_STATUS"
        assert result.passed is False, "passed should be False"
        assert result.reason == "Person opted out", "reason should be 'Person opted out'"

    suite.run_test(
        test_name="SafetyCheckResult creation works",
        test_func=test_safety_check_result,
        test_summary="Verify SafetyCheckResult dataclass can be instantiated",
        expected_outcome="SafetyCheckResult is created with correct check_type, passed, and reason",
    )

    return suite.finish_suite()


# Standard test runner
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(_module_tests)

if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
