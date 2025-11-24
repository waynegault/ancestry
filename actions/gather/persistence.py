from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Optional

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session as SqlAlchemySession, joinedload

from core.database_manager import db_transn
from core.session_manager import SessionManager
from database import Person
from standard_imports import setup_module
from test_framework import TestSuite, create_standard_test_runner

logger = setup_module(globals(), __name__)


@dataclass(frozen=True)
class GatherBatchSummary:
    """Lightweight record describing a persistence batch outcome."""

    created: int
    updated: int
    skipped: int

    @property
    def processed(self) -> int:
        """Return the total number of matches processed in the batch."""

        return self.created + self.updated + self.skipped


class GatherPersistenceService:
    """Summarizes batch activity until full persistence code migrates here."""

    def __init__(self) -> None:
        self._batches_recorded = 0

    @property
    def batches_recorded(self) -> int:
        return self._batches_recorded

    def summarize_batch(self, created: int, updated: int, skipped: int) -> GatherBatchSummary:
        if min(created, updated, skipped) < 0:
            raise ValueError("Batch counts cannot be negative")
        summary = GatherBatchSummary(created=created, updated=updated, skipped=skipped)
        self._batches_recorded += 1
        logger.debug(
            "Summarized batch",
            extra={
                "matches_created": created,
                "matches_updated": updated,
                "matches_skipped": skipped,
                "matches_processed": summary.processed,
                "persistence_batches_recorded": self._batches_recorded,
            },
        )
        return summary


@dataclass(slots=True)
class BatchLookupArtifacts:
    """Artifacts produced by the batch lookup pipeline."""

    existing_persons_map: dict[str, Person]
    fetch_candidates_uuid: set[str]
    matches_to_process_later: list[dict[str, Any]]
    skipped_count: int


@dataclass(frozen=True)
class PersistenceHooks:
    """Callables required for persistence orchestration."""

    process_single_match: Callable[
        [
            SqlAlchemySession,
            SessionManager,
            dict[str, Any],
            dict[str, Person],
            dict[str, dict[str, Any]],
        ],
        tuple[Optional[dict[str, Any]], Literal["new", "updated", "skipped", "error"], Optional[str]],
    ]
    execute_bulk_db_operations: Callable[[SqlAlchemySession, list[dict[str, Any]], dict[str, Person]], bool]


def process_batch_lookups(
    batch_session: SqlAlchemySession,
    matches_on_page: list[dict[str, Any]],
    current_page: int,
    page_statuses: dict[str, int],
) -> BatchLookupArtifacts:
    """Look up existing people and classify matches for subsequent processing."""

    logger.debug("Page %s: Looking up existing persons...", current_page)
    uuids_on_page = [m["uuid"].upper() for m in matches_on_page if m.get("uuid")]
    logger.debug("Page %s: DB lookup for %d matches...", current_page, len(uuids_on_page))
    existing_persons_map = _lookup_existing_persons(batch_session, uuids_on_page)
    logger.debug(
        "Page %s: Found %d in database (will fetch %d new)",
        current_page,
        len(existing_persons_map),
        len(uuids_on_page) - len(existing_persons_map),
    )

    logger.debug("Batch %s: Identifying candidates...", current_page)
    fetch_candidates_uuid, matches_to_process_later, skipped_count = _identify_fetch_candidates(
        matches_on_page,
        existing_persons_map,
    )
    page_statuses["skipped"] = skipped_count

    return BatchLookupArtifacts(
        existing_persons_map=existing_persons_map,
        fetch_candidates_uuid=fetch_candidates_uuid,
        matches_to_process_later=matches_to_process_later,
        skipped_count=skipped_count,
    )


def prepare_and_commit_batch_data(
    batch_session: SqlAlchemySession,
    session_manager: SessionManager,
    matches_to_process_later: list[dict[str, Any]],
    existing_persons_map: dict[str, Person],
    prefetched_data: dict[str, Any],
    current_page: int,
    page_statuses: dict[str, int],
    hooks: PersistenceHooks,
) -> None:
    """Prepare batch persistence payloads and commit them in a transaction."""

    logger.debug("Batch %s: Preparing DB data...", current_page)
    prepared_bulk_data, prep_statuses = _prepare_bulk_db_data(
        batch_session,
        session_manager,
        matches_to_process_later,
        existing_persons_map,
        prefetched_data,
        hooks,
    )
    page_statuses["new"] = prep_statuses.get("new", 0)
    page_statuses["updated"] = prep_statuses.get("updated", 0)
    page_statuses["error"] = prep_statuses.get("error", 0)

    logger.debug("Batch %s: Executing DB Commit...", current_page)
    if prepared_bulk_data:
        _execute_batch_db_commit(
            batch_session,
            prepared_bulk_data,
            existing_persons_map,
            current_page,
            page_statuses,
            hooks,
        )
    else:
        logger.debug("No data prepared for bulk DB operations on page %s.", current_page)


def _lookup_existing_persons(session: SqlAlchemySession, uuids_on_page: list[str]) -> dict[str, Person]:
    """Query the database for existing Person rows keyed by UUID."""

    existing_persons_map: dict[str, Person] = {}
    if not uuids_on_page:
        return existing_persons_map

    try:
        uuids_norm = {str(uuid_val).upper() for uuid_val in uuids_on_page}
        if not uuids_norm:
            return existing_persons_map

        stmt = (
            select(Person)
            .options(joinedload(Person.dna_match), joinedload(Person.family_tree))
            .where(Person.deleted_at.is_(None))
            .where(Person.uuid.in_(list(uuids_norm)))
        )
        existing_persons = session.scalars(stmt).all()
        existing_persons_map = {
            str(person.uuid).upper(): person for person in existing_persons if person.uuid is not None
        }
    except SQLAlchemyError as db_lookup_err:
        if "is not among the defined enum values" in str(db_lookup_err):
            raise ValueError("Database enum mismatch detected during person lookup.") from db_lookup_err
        logger.error("Database lookup failed during prefetch: %s", db_lookup_err, exc_info=True)
        raise
    except Exception as exc:
        logger.error("Unexpected error during Person lookup: %s", exc, exc_info=True)
        raise

    return existing_persons_map


def _identify_fetch_candidates(
    matches_on_page: list[dict[str, Any]],
    existing_persons_map: dict[str, Any],
) -> tuple[set[str], list[dict[str, Any]], int]:
    """Return UUIDs that require API fetches plus bookkeeping info."""

    fetch_candidates_uuid: set[str] = set()
    skipped_count_this_batch = 0
    matches_to_process_later: list[dict[str, Any]] = []
    invalid_uuid_count = 0

    for match_api_data in matches_on_page:
        uuid_val = match_api_data.get("uuid")
        if not uuid_val:
            logger.warning("Skipping match missing UUID: %s", match_api_data)
            invalid_uuid_count += 1
            continue

        existing_person = existing_persons_map.get(uuid_val.upper())

        if not existing_person:
            fetch_candidates_uuid.add(uuid_val)
            match_api_data["_needs_ethnicity_refresh"] = True
            matches_to_process_later.append(match_api_data)
            continue

        needs_fetch = _check_if_fetch_needed(existing_person, match_api_data, uuid_val)
        if needs_fetch:
            fetch_candidates_uuid.add(uuid_val)
            existing_dna = getattr(existing_person, "dna_match", None)
            if existing_dna is None:
                match_api_data["_needs_ethnicity_refresh"] = True
            else:
                match_api_data["_needs_ethnicity_refresh"] = needs_ethnicity_refresh(existing_dna)
            matches_to_process_later.append(match_api_data)
        else:
            skipped_count_this_batch += 1

    if invalid_uuid_count > 0:
        logger.error("%s matches skipped during identification due to missing UUID.", invalid_uuid_count)
    logger.debug(
        "Identified %d candidates for API detail fetch, %d skipped (no change detected from list view).",
        len(fetch_candidates_uuid),
        skipped_count_this_batch,
    )

    return fetch_candidates_uuid, matches_to_process_later, skipped_count_this_batch


def _check_if_fetch_needed(existing_person: Any, match_api_data: dict[str, Any], uuid_val: str) -> bool:
    """Compare list data vs DB records to determine if we need detail fetches."""

    needs_fetch = False
    existing_dna = existing_person.dna_match
    existing_tree = existing_person.family_tree
    db_in_tree = existing_person.in_my_tree
    api_in_tree = match_api_data.get("in_my_tree", False)

    if existing_dna:
        try:
            api_cm = int(match_api_data.get("cm_dna", 0))
            db_cm = existing_dna.cm_dna
            if api_cm != db_cm:
                needs_fetch = True
                logger.debug("  Fetch needed (UUID %s): cM changed (%s -> %s)", uuid_val, db_cm, api_cm)

            api_segments = int(match_api_data.get("numSharedSegments", 0))
            db_segments = existing_dna.shared_segments
            if api_segments != db_segments:
                needs_fetch = True
                logger.debug(
                    "  Fetch needed (UUID %s): Segments changed (%s -> %s)",
                    uuid_val,
                    db_segments,
                    api_segments,
                )
        except (ValueError, TypeError, AttributeError) as comp_err:
            logger.warning(
                "Error comparing list DNA data for UUID %s: %s. Assuming fetch needed.",
                uuid_val,
                comp_err,
            )
            needs_fetch = True
    else:
        needs_fetch = True
        logger.debug("  Fetch needed (UUID %s): No existing DNA record.", uuid_val)

    if bool(api_in_tree) != bool(db_in_tree):
        needs_fetch = True
        logger.debug("  Fetch needed (UUID %s): Tree status changed (%s -> %s)", uuid_val, db_in_tree, api_in_tree)
    elif api_in_tree and not existing_tree:
        needs_fetch = True
        logger.debug("  Fetch needed (UUID %s): Marked in tree but no DB record.", uuid_val)

    return needs_fetch


def needs_ethnicity_refresh(existing_dna_match: Optional[Any]) -> bool:
    """Return True when ethnicity data should be refreshed for the match."""

    if existing_dna_match is None:
        return True

    refresh_interval_days = 7
    last_updated = getattr(existing_dna_match, "ethnicity_updated_at", None)
    if not last_updated:
        return True

    try:
        age_seconds = (datetime.now(timezone.utc) - last_updated).total_seconds()
        return age_seconds >= refresh_interval_days * 86400
    except Exception:
        return True


def _prepare_bulk_db_data(
    session: SqlAlchemySession,
    session_manager: SessionManager,
    matches_to_process: list[dict[str, Any]],
    existing_persons_map: dict[str, Person],
    prefetched_data: dict[str, dict[str, Any]],
    hooks: PersistenceHooks,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Generate bulk persistence payloads for the provided matches."""

    prepared_bulk_data: list[dict[str, Any]] = []
    page_statuses: dict[str, int] = {"new": 0, "updated": 0, "error": 0}
    num_to_process = len(matches_to_process)

    if not num_to_process:
        return [], page_statuses

    logger.debug("--- Preparing DB data structures for %d candidates ---", num_to_process)
    process_start_time = time.time()

    for match_list_data in matches_to_process:
        uuid_val = match_list_data.get("uuid")
        log_ref_short = f"UUID={uuid_val or 'MISSING'} User='{match_list_data.get('username', 'Unknown')}'"

        try:
            prepared_data, status, error_msg = hooks.process_single_match(
                session,
                session_manager,
                match_list_data,
                existing_persons_map,
                prefetched_data,
            )
            _handle_match_processing_result(
                prepared_data,
                status,
                error_msg,
                log_ref_short,
                prepared_bulk_data,
                page_statuses,
            )
        except Exception as inner_exc:
            logger.error(
                "Critical unexpected error processing %s in _prepare_bulk_db_data: %s",
                log_ref_short,
                inner_exc,
                exc_info=True,
            )
            page_statuses["error"] += 1

    process_duration = time.time() - process_start_time
    logger.debug("--- Finished preparing DB data structures. Duration: %.2fs ---", process_duration)
    return prepared_bulk_data, page_statuses


def _handle_match_processing_result(
    prepared_data: Optional[dict[str, Any]],
    status: Literal["new", "updated", "skipped", "error"],
    error_msg: Optional[str],
    log_ref_short: str,
    prepared_bulk_data: list[dict[str, Any]],
    page_statuses: dict[str, int],
) -> None:
    """Update counters and stash prepared data when processing completes."""

    _update_page_statuses(status, page_statuses, log_ref_short)

    if status not in {"error", "skipped"} and prepared_data:
        prepared_bulk_data.append(prepared_data)
    elif status == "error":
        logger.error("Error preparing DB data for %s: %s", log_ref_short, error_msg or "Unknown error in _do_match")


def _update_page_statuses(
    status: Literal["new", "updated", "skipped", "error"],
    page_statuses: dict[str, int],
    log_ref_short: str,
) -> None:
    """Increment counters for the current page based on a match outcome."""

    if status not in page_statuses:
        logger.warning("Unknown status '%s' from _do_match for %s. Counting as error.", status, log_ref_short)
        page_statuses["error"] += 1
        return

    page_statuses[status] += 1


def _execute_batch_db_commit(
    batch_session: SqlAlchemySession,
    prepared_bulk_data: list[dict[str, Any]],
    existing_persons_map: dict[str, Person],
    current_page: int,
    page_statuses: dict[str, int],
    hooks: PersistenceHooks,
) -> None:
    """Run the transactional bulk DB operations for the prepared payloads."""

    logger.debug("Attempting bulk DB operations for page %s...", current_page)
    try:
        with db_transn(batch_session) as sess:
            bulk_success = hooks.execute_bulk_db_operations(sess, prepared_bulk_data, existing_persons_map)
            if not bulk_success:
                logger.error("Bulk DB ops FAILED page %s. Adjusting counts.", current_page)
                failed_items = len(prepared_bulk_data)
                page_statuses["error"] += failed_items
                page_statuses["new"] = 0
                page_statuses["updated"] = 0
        logger.debug("Transaction block finished page %s.", current_page)
    except (IntegrityError, SQLAlchemyError, ValueError) as bulk_db_err:
        logger.error("Bulk DB transaction FAILED page %s: %s", current_page, bulk_db_err, exc_info=True)
        failed_items = len(prepared_bulk_data)
        page_statuses["error"] += failed_items
        page_statuses["new"] = 0
        page_statuses["updated"] = 0
    except Exception as exc:
        logger.error("Unexpected error during bulk DB transaction page %s: %s", current_page, exc, exc_info=True)
        failed_items = len(prepared_bulk_data)
        page_statuses["error"] += failed_items
        page_statuses["new"] = 0
        page_statuses["updated"] = 0


# ---------------------------------------------------------------------------
# Module tests
# ---------------------------------------------------------------------------


def _test_persistence_summary_totals() -> bool:
    service = GatherPersistenceService()
    summary = service.summarize_batch(created=3, updated=4, skipped=1)
    assert summary.processed == 8
    assert service.batches_recorded == 1
    return True


def _test_negative_values_raise() -> bool:
    service = GatherPersistenceService()
    try:
        service.summarize_batch(created=-1, updated=0, skipped=0)
    except ValueError:
        return True
    raise AssertionError("Expected ValueError for negative counts")


def _test_needs_ethnicity_refresh_without_timestamp() -> bool:
    class _Dummy:
        ethnicity_updated_at = None

    assert needs_ethnicity_refresh(None) is True
    assert needs_ethnicity_refresh(_Dummy()) is True
    return True


def _test_needs_ethnicity_refresh_with_recent_timestamp() -> bool:
    class _Dummy:
        ethnicity_updated_at = datetime.now(timezone.utc)

    assert needs_ethnicity_refresh(_Dummy()) is False
    return True


def module_tests() -> bool:
    suite = TestSuite("actions.gather.persistence", "actions/gather/persistence.py")
    suite.run_test(
        "Summarize totals",
        _test_persistence_summary_totals,
        "Ensures the summary aggregates processed counts.",
    )
    suite.run_test(
        "Reject negative counts",
        _test_negative_values_raise,
        "Ensures negative counts are rejected to avoid misleading telemetry.",
    )
    suite.run_test(
        "Needs ethnicity refresh defaults",
        _test_needs_ethnicity_refresh_without_timestamp,
        "Validates the helper refreshes when data is missing.",
    )
    suite.run_test(
        "Needs ethnicity refresh respects recency",
        _test_needs_ethnicity_refresh_with_recent_timestamp,
        "Ensures recent updates skip refresh operations.",
    )
    return suite.finish_suite()


run_comprehensive_tests = create_standard_test_runner(module_tests)


if __name__ == "__main__":
    success = module_tests()
    raise SystemExit(0 if success else 1)
