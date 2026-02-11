#!/usr/bin/env python3
"""
Action 12: Shared Match Scraper

Fetches shared matches for specific DNA matches (e.g., > 20cM) from Ancestry.
Stores the relationships in the SharedMatch table.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, cast
from urllib.parse import urlencode, urljoin

from sqlalchemy import and_, select, update
from sqlalchemy.orm import Session

from api.api_constants import API_PATH_SHARED_MATCHES
from config import config_schema
from core.api_manager import RequestConfig
from core.database import DnaMatch, Person, SharedMatch, db_transn
from core.logging_utils import log_action_banner
from core.session_manager import SessionManager

logger = logging.getLogger(__name__)

# Constants
MIN_SHARED_CM_THRESHOLD = 9  # Only fetch shared matches for matches > 9cM
BATCH_SIZE = config_schema.batch_size  # Process in batches defined in config


def fetch_shared_matches(session_manager: SessionManager) -> bool:
    """
    Main entry point for Action 12.
    """
    log_action_banner("ACTION 12: SHARED MATCH SCRAPER", "start")

    # Ensure session is ready (browser + API)
    if not session_manager.ensure_session_ready():
        logger.error("Session not ready. Aborting.")
        log_action_banner("ACTION 12: SHARED MATCH SCRAPER", "failure")
        return False

    try:
        _process_shared_matches(session_manager)
        log_action_banner("ACTION 12: SHARED MATCH SCRAPER", "success")
        return True
    except Exception as e:
        logger.exception(f"Action 12 failed: {e}")
        log_action_banner("ACTION 12: SHARED MATCH SCRAPER", "failure")
        return False


def _process_shared_matches(session_manager: SessionManager) -> None:
    """
    Process matches that need shared match fetching.
    """
    # Use db_manager.get_session()
    db_session = session_manager.db_manager.get_session()
    if not db_session:
        logger.error("Could not get DB session")
        return

    with db_transn(db_session) as session:
        # Find candidates
        candidates = _get_candidates(session)
        total = len(candidates)
        logger.info(f"Found {total} matches needing shared match fetching (> {MIN_SHARED_CM_THRESHOLD}cM)")

    if total == 0:
        logger.info("No matches found needing shared match fetching.")
        return

    for i, (match_id, match_uuid, person_name) in enumerate(candidates, 1):
        logger.info(f"Processing {i}/{total}: {person_name} ({match_uuid})")

        success = _fetch_and_store_shared_matches(session_manager, match_id, match_uuid)

        if success:
            # Update status
            db_session = session_manager.db_manager.get_session()
            if db_session:
                with db_transn(db_session) as session:
                    stmt = (
                        update(DnaMatch)
                        .where(DnaMatch.id == match_id)
                        .values(shared_matches_fetched=True, shared_matches_fetched_date=datetime.now(UTC))
                    )
                    session.execute(stmt)
                    session.commit()


def _get_candidates(db_session: Session) -> list[tuple[int, str, str]]:
    """
    Get list of matches that need shared matches fetched.
    Returns list of (dna_match_id, match_uuid, person_name).
    """
    stmt = (
        select(DnaMatch.id, Person.uuid, Person.username)
        .join(Person)
        .where(DnaMatch.cm_dna >= MIN_SHARED_CM_THRESHOLD)
        .where(DnaMatch.shared_matches_fetched.is_(False))
        .order_by(DnaMatch.cm_dna.desc())
        .limit(BATCH_SIZE)
    )
    results = db_session.execute(stmt).all()
    return [(r[0], r[1], r[2]) for r in results]


def _fetch_and_store_shared_matches(session_manager: SessionManager, match_id: int, match_uuid: str) -> bool:
    """
    Fetch shared matches from API and store in DB.
    """
    relative_url = API_PATH_SHARED_MATCHES.format(my_uuid=session_manager.my_uuid)
    url = urljoin(config_schema.api.base_url, relative_url)

    # Fetch page 1 (usually sufficient for top shared matches)
    # We must pass relationguid to get shared matches
    bookmark_data = {"moreMatchesAvailable": True, "lastMatchesServicePageIdx": 0}
    params = {
        "page": 1,
        "count": 100,
        "relationguid": match_uuid,
        "bookmarkdata": json.dumps(bookmark_data),
    }
    full_url = f"{url}?{urlencode(params)}"

    try:
        config = RequestConfig(url=full_url, method="GET", api_description=f"Shared Matches {match_uuid}")
        result = session_manager.api_manager.request(config, session_manager=session_manager)

        if not result.success:
            logger.warning(f"Failed to fetch shared matches for {match_uuid}. Status: {result.status_code}")
            return False

        data = result.json

        matches_data: list[dict[str, Any]] = []
        if data and "matchGroups" in data:
            for group in data["matchGroups"]:
                matches_data.extend(group.get("matches", []))
        elif data and "matches" in data:
            matches_data = cast(list[dict[str, Any]], data["matches"])
        elif isinstance(data, list):
            matches_data = cast(list[dict[str, Any]], data)
        else:
            logger.warning(
                f"Unexpected JSON structure for shared matches: {data.keys() if isinstance(data, dict) else type(data)}"
            )
            return False

        _store_shared_matches(session_manager, match_id, matches_data)
        logger.info(f"Successfully fetched {len(matches_data)} shared matches for {match_uuid}")
        return True

    except Exception as e:
        logger.error(f"Error fetching shared matches for {match_uuid}: {e}")
        return False


def _store_shared_matches(
    session_manager: SessionManager, primary_match_id: int, matches_data: list[dict[str, Any]]
) -> None:
    """
    Store shared matches in DB.
    """
    if not matches_data:
        return

    db_session = session_manager.db_manager.get_session()
    if not db_session:
        return

    with db_transn(db_session) as session:
        # Get all UUIDs from the response
        shared_uuids = [str(m.get("testGuid", "")).upper() for m in matches_data if m.get("testGuid")]

        if not shared_uuids:
            return

        # Find which of these exist in our Person table
        stmt = select(Person.uuid, Person.id).where(Person.uuid.in_(shared_uuids))
        existing_people = session.execute(stmt).all()
        uuid_to_id = {r[0]: r[1] for r in existing_people}

        # Resolve primary match Person ID
        primary_person_id_stmt = select(DnaMatch.people_id).where(DnaMatch.id == primary_match_id)
        primary_person_id = session.execute(primary_person_id_stmt).scalar_one_or_none()

        if not primary_person_id:
            logger.error(f"Could not find Person ID for DnaMatch ID {primary_match_id}")
            return

        new_shared_matches: list[SharedMatch] = []
        for match in matches_data:
            uuid = str(match.get("testGuid", "")).upper()
            if uuid in uuid_to_id:
                shared_match_id = uuid_to_id[uuid]

                # Check existence
                exists = session.execute(
                    select(SharedMatch).where(
                        and_(SharedMatch.person_id == primary_person_id, SharedMatch.shared_match_id == shared_match_id)
                    )
                ).first()

                if not exists:
                    # Extract shared cM if available
                    shared_cm = None
                    match_relationship = cast(dict[str, Any], match.get("relationship", {}))
                    if match_relationship:
                        shared_cm = match_relationship.get("sharedCentimorgans")

                    new_shared_matches.append(
                        SharedMatch(person_id=primary_person_id, shared_match_id=shared_match_id, shared_cm=shared_cm)
                    )

        if new_shared_matches:
            session.add_all(new_shared_matches)
            logger.info(f"Added {len(new_shared_matches)} new shared matches for Person ID {primary_person_id}")


# --- Tests ---


def action12_module_tests() -> bool:
    from unittest.mock import MagicMock

    from testing.test_framework import TestSuite

    suite = TestSuite("Action 12 Shared Matches", __name__)
    suite.start_suite()

    def test_get_candidates() -> bool:
        # Mock DB session
        mock_session = MagicMock()
        mock_session.execute.return_value.all.return_value = [(1, "UUID1", "Name1"), (2, "UUID2", "Name2")]

        candidates = _get_candidates(mock_session)
        return len(candidates) == 2 and candidates[0][1] == "UUID1"

    suite.run_test(
        "Get candidates query",
        test_get_candidates,
        "Should return list of candidates",
        "candidates",
        "Mock DB session and verify candidate list",
    )

    def test_url_construction() -> bool:
        from urllib.parse import urljoin

        from api.api_constants import API_PATH_SHARED_MATCHES
        from config import config_schema

        # Mock SessionManager
        mock_sm = MagicMock()
        mock_sm.my_uuid = "MY_UUID"
        match_uuid = "MATCH_UUID"

        # Setup mock response to avoid actual API call and stop execution
        mock_result = MagicMock()
        mock_result.success = False
        mock_sm.api_manager.request.return_value = mock_result

        # Execute
        _fetch_and_store_shared_matches(mock_sm, 1, match_uuid)

        # Verify URL in RequestConfig
        if not mock_sm.api_manager.request.called:
            return False

        call_args = mock_sm.api_manager.request.call_args
        request_config = call_args[0][0]

        relative_url = API_PATH_SHARED_MATCHES.format(my_uuid="MY_UUID")
        expected_base = urljoin(config_schema.api.base_url, relative_url)

        return request_config.url.startswith(expected_base)

    suite.run_test(
        "URL Construction",
        test_url_construction,
        "Should construct full URL with base_url",
        "url_construction",
        "Mock SessionManager and verify RequestConfig URL",
    )

    return suite.finish_suite()


# Standard test runner for test discovery
from testing.test_utilities import create_standard_test_runner

run_comprehensive_tests = create_standard_test_runner(action12_module_tests)


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
