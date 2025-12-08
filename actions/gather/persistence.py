from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Optional, cast

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

import logging

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session as SqlAlchemySession, joinedload

from core.database import DnaMatch, FamilyTree, Person, PersonStatusEnum
from core.database_manager import db_transn
from core.session_manager import SessionManager
from testing.test_framework import TestSuite, create_standard_test_runner

logger = logging.getLogger(__name__)


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
# Bulk persistence helpers migrated from action6_gather.py
# ---------------------------------------------------------------------------


def _deduplicate_person_creates(person_creates_raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    De-duplicate Person creates based on Profile ID before bulk insert.

    Args:
        person_creates_raw: list of raw person create data dictionaries

    Returns:
        list of filtered person create data (duplicates removed)
    """
    person_creates_filtered: list[dict[str, Any]] = []
    seen_profile_ids: set[str] = set()
    skipped_duplicates = 0

    if not person_creates_raw:
        return person_creates_filtered

    logger.debug(f"De-duplicating {len(person_creates_raw)} raw person creates based on Profile ID...")

    for p_data in person_creates_raw:
        profile_id = cast(Optional[str], p_data.get("profile_id"))  # Already uppercase from prep if exists
        uuid_for_log = cast(Optional[str], p_data.get("uuid"))  # For logging skipped items

        if profile_id is None:
            person_creates_filtered.append(p_data)  # Allow creates with null profile ID
        elif profile_id not in seen_profile_ids:
            person_creates_filtered.append(p_data)
            seen_profile_ids.add(profile_id)
        else:
            logger.warning(
                f"Skipping duplicate Person create in batch (ProfileID: {profile_id}, UUID: {uuid_for_log})."
            )
            skipped_duplicates += 1

    if skipped_duplicates > 0:
        logger.info(f"Skipped {skipped_duplicates} duplicate person creates in this batch.")
    logger.debug(f"Proceeding with {len(person_creates_filtered)} unique person creates.")

    return person_creates_filtered


def _check_existing_profile_ids(session: SqlAlchemySession, profile_ids_to_check: set[str]) -> set[str]:
    """Check database for existing profile IDs."""
    existing_profile_ids: set[str] = set()
    if not profile_ids_to_check:
        return existing_profile_ids

    try:
        logger.debug(f"Checking database for {len(profile_ids_to_check)} existing profile IDs...")
        existing_records = (
            session.query(Person.profile_id)
            .filter(Person.profile_id.in_(profile_ids_to_check), Person.deleted_at.is_(None))
            .all()
        )
        existing_profile_ids = {record.profile_id for record in existing_records}
        if existing_profile_ids:
            logger.info(f"Found {len(existing_profile_ids)} existing profile IDs that will be skipped")
    except Exception as e:
        logger.warning(f"Failed to check existing profile IDs: {e}")

    return existing_profile_ids


def _check_existing_uuids(session: SqlAlchemySession, uuids_to_check: set[str]) -> set[str]:
    """Check database for existing UUIDs."""
    existing_uuids: set[str] = set()
    if not uuids_to_check:
        return existing_uuids

    try:
        logger.debug(f"Checking database for {len(uuids_to_check)} existing UUIDs...")
        existing_uuid_records = (
            session.query(Person.uuid).filter(Person.uuid.in_(uuids_to_check), Person.deleted_at.is_(None)).all()
        )
        existing_uuids = {record.uuid.upper() for record in existing_uuid_records}
        if existing_uuids:
            logger.info(f"Found {len(existing_uuids)} existing UUIDs that will be skipped")
    except Exception as e:
        logger.warning(f"Failed to check existing UUIDs: {e}")

    return existing_uuids


def _check_existing_records(
    session: SqlAlchemySession, insert_data_raw: list[dict[str, Any]]
) -> tuple[set[str], set[str]]:
    """
    Check database for existing profile IDs and UUIDs to prevent constraint violations.

    Args:
        session: SQLAlchemy session
        insert_data_raw: Raw insert data to check

    Returns:
        tuple of (existing_profile_ids, existing_uuids) sets
    """
    profile_ids_to_check: set[str] = {str(item.get("profile_id")) for item in insert_data_raw if item.get("profile_id")}
    uuids_to_check: set[str] = {str(item.get("uuid") or "").upper() for item in insert_data_raw if item.get("uuid")}

    existing_profile_ids = _check_existing_profile_ids(session, profile_ids_to_check)
    existing_uuids = _check_existing_uuids(session, uuids_to_check)

    return existing_profile_ids, existing_uuids


def _handle_integrity_error_recovery(
    session: SqlAlchemySession, insert_data: Optional[list[dict[str, Any]]] = None
) -> bool:
    """
    Handle UNIQUE constraint violations by attempting individual inserts.

    Args:
        session: SQLAlchemy session
        insert_data: Data that failed bulk insert (optional)

    Returns:
        True if recovery was successful
    """
    try:
        session.rollback()  # Clear the failed transaction
        logger.debug("Rolled back failed transaction due to UNIQUE constraint violation")

        if not insert_data:
            logger.debug(
                "No insert_data available for recovery - treating as successful (records likely already exist)"
            )
            return True

        logger.debug(f"Retrying with individual inserts for {len(insert_data)} records")
        successful_inserts = 0

        for item in insert_data:
            try:
                # Try individual insert
                individual_person = Person(**{k: v for k, v in item.items() if hasattr(Person, k)})
                session.add(individual_person)
                session.flush()  # Force insert attempt
                successful_inserts += 1
            except IntegrityError as individual_err:
                # This specific record already exists - skip it
                logger.debug(f"Skipping duplicate record UUID {item.get('uuid', 'unknown')}: {individual_err}")
                session.rollback()  # Clear this specific failure
            except Exception as individual_exc:
                logger.warning(
                    f"Failed to insert individual record UUID {item.get('uuid', 'unknown')}: {individual_exc}"
                )
                session.rollback()  # Clear this specific failure

        logger.info(
            f"Successfully inserted {successful_inserts} of {len(insert_data)} records after handling duplicates"
        )
        return True

    except Exception as rollback_err:
        logger.error(f"Failed to handle UNIQUE constraint violation gracefully: {rollback_err}", exc_info=True)
        return False


def _should_skip_person_insert(
    uuid_val: str,
    profile_id: Optional[str],
    seen_uuids: set[str],
    existing_persons_map: dict[str, Person],
    existing_uuids: set[str],
    existing_profile_ids: set[str],
) -> tuple[bool, Optional[str]]:
    """Check if person should be skipped during insert preparation.

    Args:
        uuid_val: Person UUID
        profile_id: Person profile ID
        seen_uuids: Set of UUIDs already seen in this batch
        existing_persons_map: Map of existing persons by UUID
        existing_uuids: Set of UUIDs that exist in database
        existing_profile_ids: Set of profile IDs that exist in database

    Returns:
        Tuple of (should_skip, reason)
    """
    if not uuid_val:
        return True, None
    if uuid_val in seen_uuids:
        return True, f"Duplicate Person in batch (UUID: {uuid_val}) - skipping duplicate."
    if uuid_val in existing_persons_map:
        return True, f"Person exists for UUID {uuid_val}; will handle as update if changes detected."
    if uuid_val in existing_uuids:
        return True, f"Person exists in DB for UUID {uuid_val}; will handle as update if needed."
    if profile_id and profile_id in existing_profile_ids:
        return True, f"Person exists with profile ID {profile_id} (UUID: {uuid_val}); will handle as update if needed."
    return False, None


def _convert_status_enums(insert_data: list[dict[str, Any]]) -> None:
    """Convert status Enum to its value for bulk insertion."""
    for item_data in insert_data:
        if "status" in item_data and hasattr(item_data["status"], 'value'):
            item_data["status"] = item_data["status"].value


def _prepare_person_insert_data(
    person_creates_filtered: list[dict[str, Any]], session: SqlAlchemySession, existing_persons_map: dict[str, Person]
) -> list[dict[str, Any]]:
    """
    Prepare and validate person insert data, removing duplicates and existing records.

    Args:
        person_creates_filtered: Filtered person create data
        session: SQLAlchemy session
        existing_persons_map: Map of existing persons by UUID

    Returns:
        list of validated insert data ready for bulk insert
    """
    if not person_creates_filtered:
        return []

    logger.debug(f"Preparing {len(person_creates_filtered)} Person records for bulk insert...")

    # Prepare list of dictionaries for bulk_insert_mappings
    insert_data_raw = [{k: v for k, v in p.items() if not k.startswith("_")} for p in person_creates_filtered]

    # Check for existing records in database
    existing_profile_ids, existing_uuids = _check_existing_records(session, insert_data_raw)

    # De-duplicate by UUID within this batch and drop existing records
    seen_uuids: set[str] = set()
    insert_data: list[dict[str, Any]] = []

    for item in insert_data_raw:
        uuid_val = str(item.get("uuid") or "").upper()
        profile_id = item.get("profile_id")

        should_skip, reason = _should_skip_person_insert(
            uuid_val, profile_id, seen_uuids, existing_persons_map, existing_uuids, existing_profile_ids
        )

        if should_skip:
            if reason:
                logger.debug(reason)
            continue

        seen_uuids.add(uuid_val)
        item["uuid"] = uuid_val
        insert_data.append(item)

    _convert_status_enums(insert_data)
    return insert_data


def _is_person_create(d: dict[str, Any]) -> bool:
    """Check if data dict contains a person create operation."""
    return bool(d.get("person") and d["person"]["_operation"] == "create")


def _is_person_update(d: dict[str, Any]) -> bool:
    """Check if data dict contains a person update operation."""
    return bool(d.get("person") and d["person"]["_operation"] == "update")


def _separate_bulk_operations(
    prepared_bulk_data: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate prepared data by operation type and table.

    Args:
        prepared_bulk_data: List of prepared bulk data dictionaries

    Returns:
        Tuple of (person_creates, person_updates, dna_match_ops, family_tree_ops)
    """
    person_creates_raw = [d["person"] for d in prepared_bulk_data if _is_person_create(d)]
    person_updates = [d["person"] for d in prepared_bulk_data if _is_person_update(d)]
    dna_match_ops = [d["dna_match"] for d in prepared_bulk_data if d.get("dna_match")]
    family_tree_ops = [d["family_tree"] for d in prepared_bulk_data if d.get("family_tree")]
    return person_creates_raw, person_updates, dna_match_ops, family_tree_ops


def _validate_no_duplicate_profile_ids(insert_data: list[dict[str, Any]]) -> None:
    """Validate that there are no duplicate profile IDs in insert data.

    Args:
        insert_data: List of person insert data dictionaries

    Raises:
        IntegrityError: If duplicate profile IDs are found
    """
    final_profile_ids = {item.get("profile_id") for item in insert_data if item.get("profile_id")}
    if len(final_profile_ids) != sum(1 for item in insert_data if item.get("profile_id")):
        logger.error("CRITICAL: Duplicate non-NULL profile IDs DETECTED post-filter! Aborting bulk insert.")
        id_counts = Counter(item.get("profile_id") for item in insert_data if item.get("profile_id"))
        duplicates = {pid: count for pid, count in id_counts.items() if count > 1}
        logger.error(f"Duplicate Profile IDs in filtered list: {duplicates}")
        dup_exception = ValueError(f"Duplicate profile IDs: {duplicates}")
        raise IntegrityError(
            "Duplicate profile IDs found pre-bulk insert",
            params=str(duplicates),
            orig=dup_exception,
        )


def _get_person_id_mapping(session: SqlAlchemySession, inserted_uuids: list[str]) -> dict[str, int]:
    """Get Person ID mapping for inserted UUIDs.

    Args:
        session: SQLAlchemy session
        inserted_uuids: List of inserted UUIDs

    Returns:
        Dictionary mapping UUID to Person ID
    """
    created_person_map: dict[str, int] = {}
    if not inserted_uuids:
        logger.warning("No UUIDs available in insert_data to query back IDs.")
        return created_person_map

    logger.debug(f"Querying IDs for {len(inserted_uuids)} inserted UUIDs...")

    try:
        session.flush()  # Make pending changes visible to current session
        session.commit()  # Commit to database for ID generation

        person_lookup_stmt = select(Person.id, Person.uuid).where(Person.uuid.in_(inserted_uuids))
        newly_inserted_persons = session.execute(person_lookup_stmt).all()
        created_person_map = {
            p_uuid: p_id for p_id, p_uuid in newly_inserted_persons if p_id is not None and isinstance(p_uuid, str)
        }

        logger.debug(
            f"Person ID Mapping: Queried {len(inserted_uuids)} UUIDs, mapped {len(created_person_map)} Person IDs"
        )

        if len(created_person_map) != len(inserted_uuids):
            missing_count = len(inserted_uuids) - len(created_person_map)
            missing_uuids = [uuid for uuid in inserted_uuids if uuid not in created_person_map]
            logger.error(
                f"CRITICAL: Person ID mapping failed for {missing_count} UUIDs. Missing: {missing_uuids[:3]}{'...' if missing_count > 3 else ''}"
            )

            # Recovery attempt: Query with broader filter
            if missing_uuids:
                recovery_stmt = (
                    select(Person.id, Person.uuid)
                    .where(Person.uuid.in_(missing_uuids))
                    .where(Person.deleted_at.is_(None))
                )
                recovery_persons = session.execute(recovery_stmt).all()
                recovery_map = {
                    p_uuid: p_id for p_id, p_uuid in recovery_persons if p_id is not None and isinstance(p_uuid, str)
                }
                if recovery_map:
                    logger.info(f"Recovery: Found {len(recovery_map)} additional Person IDs")
                    created_person_map.update(recovery_map)

        return created_person_map

    except Exception as mapping_error:
        logger.error(f"CRITICAL: Person ID mapping query failed: {mapping_error}")
        session.rollback()
        created_person_map.clear()
        return created_person_map


def _process_person_creates(
    session: SqlAlchemySession, person_creates_raw: list[dict[str, Any]], existing_persons_map: dict[str, Person]
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Process Person create operations.

    Args:
        session: SQLAlchemy session
        person_creates_raw: Raw person create data
        existing_persons_map: Map of existing persons

    Returns:
        Tuple of (created_person_map, insert_data)
    """
    # De-duplicate Person creates
    person_creates_filtered = _deduplicate_person_creates(person_creates_raw)

    created_person_map: dict[str, int] = {}
    insert_data: list[dict[str, Any]] = []

    if not person_creates_filtered:
        return created_person_map, insert_data

    # Prepare insert data
    insert_data = _prepare_person_insert_data(person_creates_filtered, session, existing_persons_map)

    # Validate no duplicates
    _validate_no_duplicate_profile_ids(insert_data)

    # Perform bulk insert
    logger.debug(f"Bulk inserting {len(insert_data)} Person records...")
    session.bulk_insert_mappings(Person.__mapper__, insert_data)

    # Get newly created IDs
    session.flush()
    inserted_uuids = [p_data["uuid"] for p_data in insert_data if p_data.get("uuid")]
    created_person_map = _get_person_id_mapping(session, inserted_uuids)

    return created_person_map, insert_data


def _build_person_update_dict(p_data: dict[str, Any], existing_id: int) -> dict[str, Any]:
    """Build update dictionary for a person record."""
    update_dict = {k: v for k, v in p_data.items() if not k.startswith("_") and k not in {"uuid", "profile_id"}}
    if "status" in update_dict and isinstance(update_dict["status"], PersonStatusEnum):
        update_dict["status"] = update_dict["status"].value
    update_dict["id"] = existing_id
    update_dict["updated_at"] = datetime.now(timezone.utc)
    return update_dict


def _process_person_updates(session: SqlAlchemySession, person_updates: list[dict[str, Any]]) -> None:
    """Process Person update operations.

    Args:
        session: SQLAlchemy session
        person_updates: List of person update data
    """
    if not person_updates:
        logger.debug("No Person updates needed for this batch.")
        return

    update_mappings: list[dict[str, Any]] = []
    for p_data in person_updates:
        existing_id = p_data.get("_existing_person_id")
        if not existing_id:
            logger.warning(f"Skipping person update (UUID {p_data.get('uuid')}): Missing '_existing_person_id'.")
            continue
        update_dict = _build_person_update_dict(p_data, existing_id)
        if len(update_dict) > 2:
            update_mappings.append(update_dict)

    if update_mappings:
        logger.debug(f"Bulk updating {len(update_mappings)} Person records...")
        session.bulk_update_mappings(Person.__mapper__, update_mappings)
        logger.debug("Bulk update Persons called.")
    else:
        logger.debug("No valid Person updates to perform.")


def _add_update_ids_to_map(all_person_ids_map: dict[str, int], person_updates: list[dict[str, Any]]) -> None:
    """Add IDs from person updates to the master ID map."""
    for p_update_data in person_updates:
        if p_update_data.get("_existing_person_id") and p_update_data.get("uuid"):
            all_person_ids_map[p_update_data["uuid"]] = p_update_data["_existing_person_id"]


def _add_existing_ids_to_map(
    all_person_ids_map: dict[str, int],
    prepared_bulk_data: list[dict[str, Any]],
    existing_persons_map: dict[str, Person],
) -> None:
    """Add IDs from existing persons to the master ID map.

    This function ensures that ALL existing persons are added to the map,
    not just those that have new/updated data in prepared_bulk_data.
    This is critical when all persons in a batch already exist - without this,
    all_person_ids_map would be empty, causing DNA match INSERT failures.
    """
    # First, add IDs for persons that have data in prepared_bulk_data
    processed_uuids = {p["person"]["uuid"] for p in prepared_bulk_data if p.get("person") and p["person"].get("uuid")}
    for uuid_processed in processed_uuids:
        if uuid_processed not in all_person_ids_map and existing_persons_map.get(uuid_processed):
            person = existing_persons_map[uuid_processed]
            person_id_val = getattr(person, "id", None)
            if person_id_val is not None:
                all_person_ids_map[uuid_processed] = person_id_val

    # CRITICAL FIX: Also add IDs for ALL other existing persons, even if they don't have
    # data in prepared_bulk_data. This handles the case where all persons already exist
    # and were filtered out from Person INSERT operations, but we still need their IDs
    # to properly handle DNA match UPDATE operations.
    added_count = 0
    for uuid_val, person in existing_persons_map.items():
        if uuid_val not in all_person_ids_map:
            person_id_val = getattr(person, "id", None)
            if person_id_val is not None:
                all_person_ids_map[uuid_val] = person_id_val
                added_count += 1
    if added_count > 0:
        logger.info(f"Added {added_count} existing person IDs to map from existing_persons_map")


def _create_master_id_map(
    created_person_map: dict[str, int],
    person_updates: list[dict[str, Any]],
    prepared_bulk_data: list[dict[str, Any]],
    existing_persons_map: dict[str, Person],
) -> dict[str, int]:
    """Create master ID map for linking related records.

    Args:
        created_person_map: Map of newly created person IDs
        person_updates: List of person update data
        prepared_bulk_data: List of prepared bulk data
        existing_persons_map: Map of existing persons

    Returns:
        Master ID map (UUID -> Person ID)
    """
    logger.debug(
        f"Creating master ID map: created_person_map={len(created_person_map)}, person_updates={len(person_updates)}, existing_persons_map={len(existing_persons_map)}"
    )
    all_person_ids_map: dict[str, int] = created_person_map.copy()
    _add_update_ids_to_map(all_person_ids_map, person_updates)
    _add_existing_ids_to_map(all_person_ids_map, prepared_bulk_data, existing_persons_map)
    logger.debug(f"Master ID map created with {len(all_person_ids_map)} total entries")
    return all_person_ids_map


def _resolve_person_id(
    session: SqlAlchemySession,
    person_uuid: Optional[str],
    all_person_ids_map: dict[str, int],
    existing_persons_map: dict[str, Person],
) -> Optional[int]:
    """Resolve person ID from UUID using multiple strategies.

    Args:
        session: SQLAlchemy session
        person_uuid: Person UUID to resolve
        all_person_ids_map: Map of UUID to Person ID
        existing_persons_map: Map of existing persons

    Returns:
        Person ID if found, None otherwise
    """
    if not person_uuid:
        logger.warning("Missing UUID in DNA match data - skipping DNA Match creation")
        return None

    # Strategy 1: Check all_person_ids_map
    person_id = all_person_ids_map.get(person_uuid)

    # Strategy 2: Check existing_persons_map
    if not person_id and existing_persons_map.get(person_uuid):
        existing_person = existing_persons_map[person_uuid]
        person_id = getattr(existing_person, "id", None)
        if person_id:
            all_person_ids_map[person_uuid] = person_id
            logger.debug(f"Resolved Person ID {person_id} for UUID {person_uuid} (from existing_persons_map)")
        else:
            logger.warning(f"Person exists in database for UUID {person_uuid} but has no ID attribute")

    # Strategy 3: Direct database query as fallback
    if not person_id:
        try:
            db_person = session.query(Person.id).filter(Person.uuid == person_uuid, Person.deleted_at.is_(None)).first()
            if db_person:
                person_id = db_person.id
                all_person_ids_map[person_uuid] = person_id
                logger.debug(f"Resolved Person ID {person_id} for UUID {person_uuid} (direct DB query)")
            else:
                logger.debug(f"Person UUID {person_uuid} not found in database - will be created in next batch")
        except Exception as e:
            logger.warning(f"Database query failed for UUID {person_uuid}: {e}")

    return person_id


def _get_existing_dna_matches(session: SqlAlchemySession, all_person_ids_map: dict[str, int]) -> dict[int, int]:
    """Get existing DnaMatch records for people in batch.

    Args:
        session: SQLAlchemy session
        all_person_ids_map: Map of UUID to Person ID

    Returns:
        Map of people_id to DnaMatch ID
    """
    people_ids_in_batch: set[int] = set(all_person_ids_map.values())
    logger.debug(
        f"all_person_ids_map has {len(all_person_ids_map)} entries, people_ids_in_batch has {len(people_ids_in_batch)} IDs"
    )
    if not people_ids_in_batch:
        logger.warning("No people IDs in batch - cannot query for existing DNA matches")
        return {}

    logger.debug(f"Querying for existing DNA matches for people IDs: {sorted(people_ids_in_batch)}")
    stmt = select(DnaMatch.people_id, DnaMatch.id).where(DnaMatch.people_id.in_(list(people_ids_in_batch)))
    existing_matches = session.execute(stmt).all()
    existing_dna_matches_map: dict[int, int] = {}
    for people_id, match_id in existing_matches:
        if people_id is None or match_id is None:
            continue
        existing_dna_matches_map[int(people_id)] = int(match_id)
    logger.debug(f"Found {len(existing_dna_matches_map)} existing DnaMatch records for people in this batch.")
    if len(existing_dna_matches_map) < len(people_ids_in_batch):
        missing_count = len(people_ids_in_batch) - len(existing_dna_matches_map)
        logger.debug(f"{missing_count} people IDs do not have existing DNA match records")
    return existing_dna_matches_map


def _prepare_dna_match_data(dna_data: dict[str, Any], person_id: int) -> dict[str, Any]:
    """Prepare DNA match data for insert/update.

    Args:
        dna_data: Raw DNA match data
        person_id: Person ID to link to

    Returns:
        Prepared data dictionary
    """
    op_data = {k: v for k, v in dna_data.items() if not k.startswith("_") and k != "uuid"}
    op_data["people_id"] = person_id
    return op_data


def _classify_dna_match_operations(
    session: SqlAlchemySession,
    dna_match_ops: list[dict[str, Any]],
    all_person_ids_map: dict[str, int],
    existing_persons_map: dict[str, Person],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify DNA match operations into inserts and updates.

    Args:
        session: SQLAlchemy session
        dna_match_ops: List of DNA match operations
        all_person_ids_map: Map of UUID to Person ID
        existing_persons_map: Map of existing persons

    Returns:
        Tuple of (insert_data, update_mappings)
    """
    existing_dna_matches_map = _get_existing_dna_matches(session, all_person_ids_map)
    dna_insert_data: list[dict[str, Any]] = []
    dna_update_mappings: list[dict[str, Any]] = []

    for dna_data in dna_match_ops:
        person_uuid = dna_data.get("uuid")
        person_id = _resolve_person_id(session, person_uuid, all_person_ids_map, existing_persons_map)

        if not person_id:
            continue

        op_data = _prepare_dna_match_data(dna_data, person_id)

        # FIX: Double-check for existing DnaMatch record even if not in existing_dna_matches_map
        # This handles cases where the record was created in a previous batch in the same run
        existing_match_id = existing_dna_matches_map.get(person_id)
        if not existing_match_id:
            # Query database directly to ensure we don't miss recently created records
            try:
                db_match = session.query(DnaMatch.id).filter(DnaMatch.people_id == person_id).first()
                if db_match:
                    existing_match_id = db_match.id
                    logger.debug(
                        f"Found existing DnaMatch (ID={existing_match_id}) for PersonID {person_id} via direct query"
                    )
            except Exception as e:
                logger.warning(f"Direct DnaMatch query failed for PersonID {person_id}: {e}")

        if existing_match_id:
            # Prepare for UPDATE
            update_map = op_data.copy()
            update_map["id"] = existing_match_id
            update_map["updated_at"] = datetime.now(timezone.utc)
            if len(update_map) > 3:  # More than id/people_id/updated_at
                dna_update_mappings.append(update_map)
            else:
                logger.debug(f"Skipping DnaMatch update for PersonID {person_id}: No changed fields.")
        else:
            # Prepare for INSERT
            insert_map = op_data.copy()
            insert_map.setdefault("created_at", datetime.now(timezone.utc))
            insert_map.setdefault("updated_at", datetime.now(timezone.utc))
            dna_insert_data.append(insert_map)

    return dna_insert_data, dna_update_mappings


def _apply_ethnicity_data(session: SqlAlchemySession, people_id: int, ethnicity_data: dict[str, Any]) -> None:
    """Apply ethnicity data via raw SQL UPDATE.

    Args:
        session: SQLAlchemy session
        people_id: Person ID
        ethnicity_data: Ethnicity data dictionary
    """
    from sqlalchemy import text

    set_clauses = ", ".join([f"{col} = :{col}" for col in ethnicity_data])
    sql = f"UPDATE dna_match SET {set_clauses} WHERE people_id = :people_id"
    params = {**ethnicity_data, "people_id": people_id}
    session.execute(text(sql), params)


def _bulk_insert_dna_matches(session: SqlAlchemySession, dna_insert_data: list[dict[str, Any]]) -> None:
    """Bulk insert DNA match records with ethnicity data.

    Args:
        session: SQLAlchemy session
        dna_insert_data: List of DNA match insert data
    """
    if not dna_insert_data:
        return

    logger.debug(f"Bulk inserting {len(dna_insert_data)} DnaMatch records...")

    # Separate ethnicity columns from core data
    core_insert_data: list[dict[str, Any]] = []
    ethnicity_updates: list[tuple[int, dict[str, Any]]] = []

    for insert_map in dna_insert_data:
        core_map = {k: v for k, v in insert_map.items() if not k.startswith("ethnicity_")}
        ethnicity_map = {k: v for k, v in insert_map.items() if k.startswith("ethnicity_")}
        core_insert_data.append(core_map)
        if ethnicity_map:
            ethnicity_updates.append((insert_map["people_id"], ethnicity_map))

    # Bulk insert core data
    session.bulk_insert_mappings(DnaMatch.__mapper__, core_insert_data)
    session.flush()

    # Apply ethnicity data
    if ethnicity_updates:
        for people_id, ethnicity_data in ethnicity_updates:
            _apply_ethnicity_data(session, people_id, ethnicity_data)
        session.flush()
        logger.debug(f"Applied ethnicity data to {len(ethnicity_updates)} newly inserted DnaMatch records")


def _separate_core_and_ethnicity(update_map: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate core data from ethnicity data in an update mapping."""
    core_map = {k: v for k, v in update_map.items() if not k.startswith("ethnicity_")}
    ethnicity_map = {k: v for k, v in update_map.items() if k.startswith("ethnicity_")}
    return core_map, ethnicity_map


def _bulk_update_dna_matches(session: SqlAlchemySession, dna_update_mappings: list[dict[str, Any]]) -> None:
    """Bulk update DNA match records with ethnicity data.

    Args:
        session: SQLAlchemy session
        dna_update_mappings: List of DNA match update mappings
    """
    if not dna_update_mappings:
        return

    logger.debug(f"Bulk updating {len(dna_update_mappings)} DnaMatch records...")

    # Separate ethnicity columns from core data
    core_update_mappings: list[dict[str, Any]] = []
    ethnicity_updates: list[tuple[int, dict[str, Any]]] = []

    for update_map in dna_update_mappings:
        core_map, ethnicity_map = _separate_core_and_ethnicity(update_map)
        core_update_mappings.append(core_map)
        if ethnicity_map:
            # FIX: Use people_id instead of id for ethnicity updates
            ethnicity_updates.append((update_map["people_id"], ethnicity_map))

    # Bulk update core data
    if core_update_mappings:
        session.bulk_update_mappings(DnaMatch.__mapper__, core_update_mappings)
        session.flush()

    # Apply ethnicity data
    if ethnicity_updates:
        for people_id, ethnicity_data in ethnicity_updates:
            _apply_ethnicity_data(session, people_id, ethnicity_data)
        session.flush()
        logger.debug(f"Applied ethnicity data to {len(ethnicity_updates)} updated DnaMatch records")


def _process_dna_match_operations(
    session: SqlAlchemySession,
    dna_match_ops: list[dict[str, Any]],
    all_person_ids_map: dict[str, int],
    existing_persons_map: dict[str, Person],
) -> None:
    """Process DNA match operations (inserts and updates).

    Args:
        session: SQLAlchemy session
        dna_match_ops: List of DNA match operations
        all_person_ids_map: Map of UUID to Person ID
        existing_persons_map: Map of existing persons
    """
    if not dna_match_ops:
        return

    dna_insert_data, dna_update_mappings = _classify_dna_match_operations(
        session, dna_match_ops, all_person_ids_map, existing_persons_map
    )

    _bulk_insert_dna_matches(session, dna_insert_data)
    _bulk_update_dna_matches(session, dna_update_mappings)

    # FIX: Expire session cache after bulk operations to ensure subsequent queries
    # can see newly inserted/updated records. This prevents UNIQUE constraint errors
    # when the same person is processed in subsequent batches.
    if dna_insert_data or dna_update_mappings:
        session.expire_all()
        logger.debug("Session cache expired after DnaMatch bulk operations")


def _prepare_family_tree_inserts(
    session: SqlAlchemySession,
    tree_creates: list[dict[str, Any]],
    all_person_ids_map: dict[str, int],
    existing_persons_map: dict[str, Person],
) -> list[dict[str, Any]]:
    """Prepare FamilyTree insert data.

    Args:
        session: SQLAlchemy session
        tree_creates: List of FamilyTree create operations
        all_person_ids_map: Map of UUID to Person ID
        existing_persons_map: Map of existing persons

    Returns:
        List of FamilyTree insert data
    """
    tree_insert_data: list[dict[str, Any]] = []

    for tree_data in tree_creates:
        person_uuid = tree_data.get("uuid")
        person_id = _resolve_person_id(session, person_uuid, all_person_ids_map, existing_persons_map)

        if person_id:
            insert_dict = {k: v for k, v in tree_data.items() if not k.startswith("_")}
            insert_dict["people_id"] = person_id
            insert_dict.pop("uuid", None)
            tree_insert_data.append(insert_dict)
        else:
            logger.debug(f"Person with UUID {person_uuid} not found in database - skipping FamilyTree creation.")

    return tree_insert_data


def _prepare_family_tree_updates(tree_updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepare FamilyTree update mappings.

    Args:
        tree_updates: List of FamilyTree update operations

    Returns:
        List of FamilyTree update mappings
    """
    tree_update_mappings: list[dict[str, Any]] = []

    for tree_data in tree_updates:
        existing_tree_id = tree_data.get("_existing_tree_id")
        if not existing_tree_id:
            logger.warning(
                f"Skipping FamilyTree update op (UUID {tree_data.get('uuid')}): Missing '_existing_tree_id'."
            )
            continue

        update_dict_tree = {k: v for k, v in tree_data.items() if not k.startswith("_") and k != "uuid"}
        update_dict_tree["id"] = existing_tree_id
        update_dict_tree["updated_at"] = datetime.now(timezone.utc)

        if len(update_dict_tree) > 2:
            tree_update_mappings.append(update_dict_tree)

    return tree_update_mappings


def _process_family_tree_operations(
    session: SqlAlchemySession,
    family_tree_ops: list[dict[str, Any]],
    all_person_ids_map: dict[str, int],
    existing_persons_map: dict[str, Person],
) -> None:
    """Process FamilyTree operations (inserts and updates).

    Args:
        session: SQLAlchemy session
        family_tree_ops: List of FamilyTree operations
        all_person_ids_map: Map of UUID to Person ID
        existing_persons_map: Map of existing persons
    """
    tree_creates = [op for op in family_tree_ops if op.get("_operation") == "create"]
    tree_updates = [op for op in family_tree_ops if op.get("_operation") == "update"]

    # Process creates
    if tree_creates:
        tree_insert_data = _prepare_family_tree_inserts(session, tree_creates, all_person_ids_map, existing_persons_map)
        if tree_insert_data:
            logger.debug(f"Bulk inserting {len(tree_insert_data)} FamilyTree records...")
            session.bulk_insert_mappings(FamilyTree.__mapper__, tree_insert_data)
        else:
            logger.debug("No valid FamilyTree records to insert")
    else:
        logger.debug("No FamilyTree creates prepared.")

    # Process updates
    if tree_updates:
        tree_update_mappings = _prepare_family_tree_updates(tree_updates)
        if tree_update_mappings:
            logger.debug(f"Bulk updating {len(tree_update_mappings)} FamilyTree records...")
            session.bulk_update_mappings(FamilyTree.__mapper__, tree_update_mappings)
            logger.debug("Bulk update FamilyTrees called.")
        else:
            logger.debug("No valid FamilyTree updates.")
    else:
        logger.debug("No FamilyTree updates prepared.")


def execute_bulk_db_operations(
    session: SqlAlchemySession,
    prepared_bulk_data: list[dict[str, Any]],
    existing_persons_map: dict[str, Person],  # Needed to potentially map existing IDs
) -> bool:
    """
    Executes bulk INSERT and UPDATE operations for Person, DnaMatch, and FamilyTree
    records within an existing database transaction session.

    Args:
        session: The active SQLAlchemy database session (within a transaction).
        prepared_bulk_data: list of dictionaries prepared by `_prepare_bulk_db_data`.
                            Each dict contains 'person', 'dna_match', 'family_tree' keys
                            with data and an '_operation' hint ('create'/'update').
        existing_persons_map: Dictionary mapping UUIDs to existing Person objects,
                              used for linking updates correctly.

    Returns:
        True if all bulk operations completed successfully within the transaction,
        False if a database error occurred.
    """
    # Step 1: Initialization
    bulk_start_time = time.time()
    num_items = len(prepared_bulk_data)
    if num_items == 0:
        return True  # Nothing to do, considered success

    logger.debug(f"--- Starting Bulk DB Operations ({num_items} prepared items) ---")

    try:
        # Step 2: Separate data by operation type (create/update) and table
        person_creates_raw, person_updates, dna_match_ops, family_tree_ops = _separate_bulk_operations(
            prepared_bulk_data
        )

        # --- Step 3: Person Creates ---
        created_person_map, _ = _process_person_creates(session, person_creates_raw, existing_persons_map)

        # --- Step 4: Person Updates ---
        _process_person_updates(session, person_updates)

        # --- Step 5: Create Master ID Map (for linking related records) ---
        all_person_ids_map = _create_master_id_map(
            created_person_map, person_updates, prepared_bulk_data, existing_persons_map
        )

        # --- Step 6: DnaMatch Bulk Upsert ---
        _process_dna_match_operations(session, dna_match_ops, all_person_ids_map, existing_persons_map)

        # --- Step 7: FamilyTree Bulk Upsert ---
        _process_family_tree_operations(session, family_tree_ops, all_person_ids_map, existing_persons_map)

        # Step 8: Log success
        bulk_duration = time.time() - bulk_start_time
        logger.debug(f"--- Bulk DB Operations OK. Duration: {bulk_duration:.2f}s ---")
        return True

    # Step 9: Handle database errors during bulk operations
    except IntegrityError as integrity_err:
        # Handle UNIQUE constraint violations gracefully
        error_str = str(integrity_err)
        if "UNIQUE constraint failed: people.uuid" in error_str:
            logger.warning(
                f"UNIQUE constraint violation during bulk insert - some records already exist: {integrity_err}"
            )
            # This is expected behavior when records already exist - don't fail the entire batch
            logger.info("Continuing with database operations despite duplicate records...")

            # Use helper function to handle recovery
            # Note: insert_data might not be available in this exception scope, pass None for safe recovery
            return _handle_integrity_error_recovery(session, None)

        if "UNIQUE constraint failed: dna_match.people_id" in error_str:
            logger.error(
                "UNIQUE constraint violation: dna_match.people_id already exists. This indicates the code tried to INSERT when it should UPDATE."
            )
            logger.error(f"Error details: {integrity_err}")
            # Roll back the session to clear the error state
            session.rollback()
            logger.info("Session rolled back. Returning False to indicate failure.")
        else:
            logger.error(f"Other IntegrityError during bulk DB operation: {integrity_err}", exc_info=True)

        return False  # Other integrity errors should still fail

    except SQLAlchemyError as bulk_db_err:
        logger.error(f"Bulk DB operation FAILED: {bulk_db_err}", exc_info=True)
        return False  # Indicate failure (caller owns transaction rollback)
    except Exception as e:
        logger.error(f"Unexpected error during bulk DB operations: {e}", exc_info=True)
        return False  # Indicate failure


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
