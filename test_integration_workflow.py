import contextlib
import pathlib
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(str(pathlib.Path.cwd()))

from core.action_runner import exec_actn
from core.database_manager import DatabaseManager
from core.session_manager import SessionManager
from core.workflow_actions import gather_dna_matches, process_productive_messages_action, srch_inbox_actn
from database import Base, ConversationLog, MessageDirectionEnum, MessageTemplate, Person
from session_utils import set_global_session


class MockMetrics:
    """Mock metrics object that handles float formatting in logs."""

    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.rate = 1.0
        self.avg_wait = 0.1
        self.adjustments = {"down": 0, "up": 0}

    def __getattr__(self, name: str):
        return 0


class TestIntegrationWorkflow(unittest.TestCase):
    def setUp(self):
        """Set up test environment with file-based DB and mocked session."""
        # Use a file-based DB for integration tests to ensure persistence across connections
        self.db_path = "test_integration.db"
        if pathlib.Path(self.db_path).exists():
            with contextlib.suppress(PermissionError):
                pathlib.Path(self.db_path).unlink()

        # Initialize DatabaseManager with file path
        self.db_manager = DatabaseManager(db_path=self.db_path)
        self.db_manager.ensure_ready()  # Ensure engine is initialized

        # Create tables
        if self.db_manager.engine:
            Base.metadata.create_all(self.db_manager.engine)

        # Initialize SessionManager with this DB manager
        self.session_manager = MagicMock(spec=SessionManager)
        self.session_manager.db_manager = self.db_manager
        self.session_manager.get_db_conn_context.side_effect = self.db_manager.get_session_context
        self.session_manager.is_sess_valid.return_value = True
        self.session_manager.ensure_session_ready.return_value = True
        self.session_manager.my_profile_id = "TEST_PROFILE_ID"
        self.session_manager.my_uuid = "TEST_UUID"

        # Mock RateLimiter
        self.session_manager.rate_limiter = MagicMock()
        self.session_manager.rate_limiter.get_metrics.return_value = MockMetrics()
        self.session_manager.rate_limiter.current_delay = 0.0  # Fix for Action 7 logging
        self.session_manager.session_ready = True  # Fix for Action 6 attribute access
        self.session_manager.session_start_time = datetime.now(
            timezone.utc
        ).timestamp()  # Fix for Action 6 session age check

        # Register as global session for modules that use get_global_session()
        set_global_session(self.session_manager)

        # Seed database with required data
        self.session = self.db_manager.get_session()

        # Ensure get_db_conn returns the real session so queries work (fixing Action 9 comparison error)
        self.session_manager.get_db_conn.return_value = self.session

        # Seed Message Templates (Required for Action 9)
        if self.session:
            template = MessageTemplate(
                template_key="Productive_Reply_Acknowledgement",
                subject_line="Re: Family History",
                message_content="Thank you for your reply.",
                template_category="acknowledgement",
                tree_status="universal",
                updated_at=datetime.now(timezone.utc),
            )
            self.session.add(template)
            self.session.commit()

    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, "session") and self.session:
            self.session.close()

        # Close DB connections
        if self.db_manager.engine:
            self.db_manager.engine.dispose()

        # Remove test DB file
        if pathlib.Path(self.db_path).exists():
            with contextlib.suppress(PermissionError):
                pathlib.Path(self.db_path).unlink()

    @patch("action6_gather.get_matches")
    def test_action6_gather_integration(self, mock_fetch: MagicMock):
        """Test Action 6 (Gather) integration."""
        print("\n[Integration] Testing Action 6: Gather DNA Matches...")

        # Mock API response
        mock_match_uuid = "MATCH_UUID_1"
        mock_fetch.return_value = (
            [
                {
                    "uuid": mock_match_uuid,
                    "username": "Test Match",
                    "first_name": "Test",
                    "initials": "TM",
                    "gender": "unknown",
                    "profile_id": "PROFILE_1",
                    "administrator_profile_id_hint": None,
                    "administrator_username_hint": None,
                    "photoUrl": "http://test.com/photo",
                    "cm_dna": 50,
                    "numSharedSegments": 2,
                    "predicted_relationship": "Distant Cousin",
                    "compare_link": "http://test.com/compare",
                    "message_link": None,
                    "in_my_tree": False,
                    "createdDate": "2023-01-01",
                }
            ],
            1,
        )  # (matches, total_pages)

        # Execute Action 6
        success = exec_actn(gather_dna_matches, self.session_manager, "6 1")

        # Verify success
        self.assertTrue(success, "Action 6 should complete successfully")

        # Verify DB side effects
        if self.session:
            person = self.session.query(Person).filter_by(uuid=mock_match_uuid).first()
            self.assertIsNotNone(person, "Person should be created in DB")
            if person:
                self.assertEqual(person.username, "Test Match")

    @patch("action7_inbox.InboxProcessor._get_all_conversations_api")
    @patch("action7_inbox.InboxProcessor._fetch_conversation_context")
    @patch("action7_inbox.InboxProcessor._classify_message_with_ai")
    def test_action7_inbox_integration(
        self, mock_classify: MagicMock, mock_fetch_context: MagicMock, mock_get_convs: MagicMock
    ):
        """Test Action 7 (Inbox) integration."""
        print("\n[Integration] Testing Action 7: Inbox Processing...")

        # Mock inbox processing
        mock_get_convs.return_value = (
            [
                {
                    "conversation_id": "CONV_001",
                    "profile_id": "TEST_PROFILE_001",
                    "username": "Test User",
                    "last_message_timestamp": datetime.now(timezone.utc),
                }
            ],
            None,
        )  # (conversations, has_next_page)

        mock_fetch_context.return_value = [
            {
                "content": "Hello",
                "author": "TEST_PROFILE_001",
                "timestamp": datetime.now(timezone.utc),
                "conversation_id": "CONV_001",
            }
        ]

        mock_classify.return_value = "PRODUCTIVE"

        # Execute Action 7
        success = exec_actn(srch_inbox_actn, self.session_manager, "7")

        # Verify success
        self.assertTrue(success, "Action 7 should complete successfully")

    @patch("action9_process_productive.PersonProcessor.process_person")
    @patch("action9_process_productive._load_templates_for_action9")
    def test_action9_productive_integration(self, mock_load_templates: MagicMock, mock_process_person: MagicMock):
        """Test Action 9 (Productive) integration."""
        print("\n[Integration] Testing Action 9: Productive Conversation Management...")

        # Mock template loading
        mock_load_templates.return_value = {
            "Productive_Reply_Acknowledgement": "Subject: Re: Family History\n\nThank you for your reply."
        }

        if self.session:
            # Seed MessageTemplate
            existing_template = (
                self.session.query(MessageTemplate).filter_by(template_key="Productive_Reply_Acknowledgement").first()
            )
            if not existing_template:
                template = MessageTemplate(
                    template_key="Productive_Reply_Acknowledgement",
                    subject_line="Re: Family History",
                    message_content="Thank you.",
                    template_category="acknowledgement",
                    tree_status="universal",
                )
                self.session.add(template)
                self.session.commit()

            # Seed a productive conversation
            person = Person(
                uuid="PROD_MATCH_UUID", username="Productive Match", profile_id="PROD_PROFILE", status="PRODUCTIVE"
            )
            self.session.add(person)
            self.session.commit()

            # Seed ConversationLog (Incoming message)
            log = ConversationLog(
                people_id=person.id,
                direction=MessageDirectionEnum.IN,
                latest_message_content="Hello, I think we are related.",
                latest_timestamp=datetime.now(timezone.utc),
                conversation_id="CONV_PROD_1",
            )
            self.session.add(log)
            self.session.commit()

        # Mock processing result
        mock_process_person.return_value = (True, "success")

        # Execute Action 9
        success = exec_actn(process_productive_messages_action, self.session_manager, "9")

        # Verify success
        self.assertTrue(success, "Action 9 should complete successfully")


if __name__ == "__main__":
    unittest.main()
