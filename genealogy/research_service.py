import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from config import config_schema
from genealogy.gedcom.gedcom_cache import load_gedcom_with_aggressive_caching
from genealogy.gedcom.gedcom_utils import GedcomData, calculate_match_score, get_full_name
from research.relationship_utils import convert_gedcom_path_to_unified_format, fast_bidirectional_bfs

logger = logging.getLogger(__name__)

MatchScoreResult = tuple[float, dict[str, float], list[str]]
CacheKey = tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]


class ResearchService:
    """
    Service for genealogical research operations including searching, scoring,
    and relationship pathfinding.
    """

    def __init__(self, gedcom_path: Optional[str] = None):
        self.gedcom_data: Optional[GedcomData] = None
        self._cached_root_id: Optional[str] = None
        if gedcom_path:
            self.load_gedcom(gedcom_path)

    def load_gedcom(self, gedcom_path: str) -> None:
        """Load GEDCOM data from the specified path."""
        path = Path(gedcom_path)
        if path.exists():
            logger.info(f"Loading GEDCOM from {path}")
            # Use aggressive caching for performance
            self.gedcom_data = load_gedcom_with_aggressive_caching(str(path))
            if self.gedcom_data:
                logger.info(f"Loaded {len(self.gedcom_data.indi_index)} individuals")
                # Pre-resolve root ID if possible
                self._resolve_root_id()
        else:
            logger.error(f"GEDCOM file not found at {path}")

    def _resolve_root_id(self) -> Optional[str]:
        """Resolve 'ROOT' to the user's GEDCOM ID based on configured user name."""
        if self._cached_root_id:
            return self._cached_root_id

        if not self.gedcom_data:
            return None

        user_name = getattr(config_schema, "user_name", "").strip().lower()
        if not user_name or user_name == "tree owner":
            return None

        # Search for user in GEDCOM
        # 1. Exact match
        for indi_id, indi in self.gedcom_data.indi_index.items():
            name = get_full_name(indi).lower()
            if name == user_name:
                logger.info(f"Resolved ROOT to {indi_id} ({get_full_name(indi)})")
                self._cached_root_id = indi_id
                return indi_id

        # 2. Containment match
        for indi_id, indi in self.gedcom_data.indi_index.items():
            name = get_full_name(indi).lower()
            if user_name in name:
                logger.info(f"Resolved ROOT to {indi_id} ({get_full_name(indi)}) (Partial Match)")
                self._cached_root_id = indi_id
                return indi_id

        logger.warning(f"Could not resolve ROOT ID for user '{user_name}'")
        return None

    def search_people(
        self,
        filter_criteria: dict[str, Any],
        scoring_criteria: dict[str, Any],
        scoring_weights: dict[str, Any],
        date_flex: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Filter and score individuals based on criteria using universal scoring.
        Returns a list of matches sorted by total score (descending).
        """
        if not self.gedcom_data:
            logger.warning("GEDCOM data not loaded. Cannot perform search.")
            return []

        logger.debug("--- Filtering and Scoring Individuals (using universal scoring) ---")
        processing_start_time = time.time()

        # Get the year range for matching from configuration
        year_range = date_flex.get("year_match_range", 10)

        # For caching match scores
        score_cache: dict[CacheKey, MatchScoreResult] = {}
        scored_matches: list[dict[str, Any]] = []

        # For progress tracking
        total_records = len(self.gedcom_data.processed_data_cache)
        progress_interval = max(1, total_records // 10)  # Update every 10%

        logger.debug(f"Processing {total_records} individuals from cache...")

        for processed, (indi_id_norm, indi_data) in enumerate(self.gedcom_data.processed_data_cache.items(), start=1):
            # Show progress updates
            if processed % progress_interval == 0:
                percent_done = (processed / total_records) * 100
                logger.debug(f"Processing: {percent_done:.1f}% complete ({processed}/{total_records})")

            match_data = self._process_individual(
                indi_id_norm,
                indi_data,
                filter_criteria,
                scoring_criteria,
                scoring_weights,
                date_flex,
                year_range,
                score_cache,
            )

            if match_data:
                scored_matches.append(match_data)

        processing_duration = time.time() - processing_start_time
        logger.debug(f"Filtering & Scoring completed in {processing_duration:.2f}s.")
        logger.debug(f"Found {len(scored_matches)} individual(s) matching OR criteria and scored.")

        return sorted(scored_matches, key=lambda x: x["total_score"], reverse=True)

    def get_relationship_path(self, start_id: str, end_id: str) -> Optional[list[dict[str, Any]]]:
        """
        Calculate the relationship path between two individuals.
        Handles 'ROOT' as start_id by resolving it to the user's GEDCOM ID.
        """
        if not self.gedcom_data:
            logger.warning("GEDCOM data not loaded. Cannot calculate relationship.")
            return None

        real_start_id = start_id
        if start_id == "ROOT":
            resolved = self._resolve_root_id()
            if resolved:
                real_start_id = resolved
            else:
                # If we can't resolve ROOT, we can't calculate the path
                # Log only once per session/run to avoid spamming?
                # For now, just return None to avoid the FastBiBFS warning spam
                return None

        path_ids = fast_bidirectional_bfs(
            real_start_id,
            end_id,
            self.gedcom_data.id_to_parents,
            self.gedcom_data.id_to_children,
        )
        if not path_ids:
            return None

        return convert_gedcom_path_to_unified_format(
            path_ids,
            self.gedcom_data.reader,
            self.gedcom_data.id_to_parents,
            self.gedcom_data.id_to_children,
            self.gedcom_data.indi_index,
        )

    # === Private Helper Methods ===

    @staticmethod
    def _process_individual(
        indi_id_norm: str,
        indi_data: dict[str, Any],
        filter_criteria: dict[str, Any],
        scoring_criteria: dict[str, Any],
        scoring_weights: dict[str, Any],
        date_flex: dict[str, Any],
        year_range: int,
        score_cache: dict[CacheKey, MatchScoreResult],
    ) -> Optional[dict[str, Any]]:
        """Process a single individual for filtering and scoring."""
        try:
            extracted_data = ResearchService._extract_individual_data(indi_data)

            if ResearchService._evaluate_filter_criteria(extracted_data, filter_criteria, year_range):
                # Calculate match score with caching for performance
                total_score, field_scores, reasons = ResearchService._calculate_match_score_cached(
                    search_criteria=scoring_criteria,
                    candidate_data=indi_data,
                    scoring_weights=scoring_weights,
                    date_flex=date_flex,
                    cache=score_cache,
                )

                return ResearchService._create_match_data(indi_id_norm, indi_data, total_score, field_scores, reasons)
        except ValueError as ve:
            logger.error(f"Value error processing individual {indi_id_norm}: {ve}")
        except KeyError as ke:
            logger.error(f"Missing key for individual {indi_id_norm}: {ke}")
        except Exception as ex:
            logger.error(f"Error processing individual {indi_id_norm}: {ex}", exc_info=True)

        return None

    @staticmethod
    def _extract_individual_data(indi_data: dict[str, Any]) -> dict[str, Any]:
        """Extract needed values for filtering from individual data."""
        return {
            "givn_lower": indi_data.get("first_name", "").lower(),
            "surn_lower": indi_data.get("surname", "").lower(),
            "sex_lower": indi_data.get("gender_norm"),
            "birth_year": indi_data.get("birth_year"),
            "birth_place_lower": (
                indi_data.get("birth_place_disp", "").lower() if indi_data.get("birth_place_disp") else None
            ),
            "death_place_lower": (
                indi_data.get("death_place_disp", "").lower() if indi_data.get("death_place_disp") else None
            ),
            "death_date_obj": indi_data.get("death_date_obj"),
        }

    @staticmethod
    def _evaluate_filter_criteria(
        extracted_data: dict[str, Any], filter_criteria: dict[str, Any], year_range: int
    ) -> bool:
        """Evaluate if individual passes filter criteria."""
        # Precompute simple matches
        fn_match_filter = ResearchService._matches_criterion(
            "first_name", filter_criteria, extracted_data["givn_lower"]
        )
        sn_match_filter = ResearchService._matches_criterion("surname", filter_criteria, extracted_data["surn_lower"])
        bp_match_filter = ResearchService._matches_criterion(
            "birth_place", filter_criteria, extracted_data.get("birth_place_lower")
        )
        dp_match_filter = ResearchService._matches_criterion(
            "death_place", filter_criteria, extracted_data.get("death_place_lower")
        )
        by_match_filter = ResearchService._matches_year_criterion(
            "birth_year", filter_criteria, extracted_data["birth_year"], year_range
        )
        alive_match = extracted_data["death_date_obj"] is None

        # Enforce mandatory place presence/match only when a non-empty criterion value is provided
        place_checks: list[bool] = []
        bp_crit = filter_criteria.get("birth_place")
        dp_crit = filter_criteria.get("death_place")
        if bp_crit:
            place_checks.append(bp_match_filter)
        if dp_crit:
            place_checks.append(dp_match_filter)
        if place_checks and not all(place_checks):
            return False

        # Enforce mandatory names when provided (non-empty)
        has_fn = bool(filter_criteria.get("first_name"))
        has_sn = bool(filter_criteria.get("surname"))
        if has_fn or has_sn:
            checks: list[bool] = []
            if has_fn:
                checks.append(fn_match_filter)
            if has_sn:
                checks.append(sn_match_filter)
            return all(checks) if checks else True

        # No names provided: broader OR filter (birth/death place, birth year, or alive)
        return any((bp_match_filter, dp_match_filter, by_match_filter, alive_match))

    @staticmethod
    def _matches_criterion(criterion_name: str, filter_criteria: dict[str, Any], candidate_value: Any) -> bool:
        """Check if a candidate value matches a criterion (case-insensitive for strings)."""
        criterion = filter_criteria.get(criterion_name)
        if isinstance(criterion, str):
            criterion = criterion.lower()
        return bool(criterion and candidate_value and criterion in candidate_value)

    @staticmethod
    def _matches_year_criterion(
        criterion_name: str,
        filter_criteria: dict[str, Any],
        candidate_value: Optional[int],
        year_range: int,
    ) -> bool:
        """Check if a candidate year matches a year criterion within range."""
        criterion = filter_criteria.get(criterion_name)
        return bool(criterion and candidate_value and abs(candidate_value - criterion) <= year_range)

    @staticmethod
    def _calculate_match_score_cached(
        search_criteria: dict[str, Any],
        candidate_data: dict[str, Any],
        scoring_weights: Mapping[str, int | float],
        date_flex: dict[str, Any],
        cache: Optional[dict[CacheKey, MatchScoreResult]] = None,
    ) -> MatchScoreResult:
        """Calculate match score with caching for performance."""
        if cache is None:
            cache = {}
        # Create a hash key from the relevant parts of the inputs
        # We use a tuple of immutable representations of the data
        criterion_hash = tuple(sorted((k, str(v)) for k, v in search_criteria.items() if v is not None))
        candidate_hash = tuple(sorted((k, str(v)) for k, v in candidate_data.items() if k in search_criteria))
        cache_key = (criterion_hash, candidate_hash)

        if cache_key not in cache:
            result = calculate_match_score(
                search_criteria=search_criteria,
                candidate_processed_data=candidate_data,
                scoring_weights=scoring_weights,
                date_flexibility=date_flex,
            )

            cache[cache_key] = result

        return cache[cache_key]

    @staticmethod
    def _create_match_data(
        indi_id_norm: str,
        indi_data: dict[str, Any],
        total_score: float,
        field_scores: dict[str, Any],
        reasons: list[str],
    ) -> dict[str, Any]:
        """Create match data dictionary for display and analysis."""
        return {
            "id": indi_id_norm,
            "display_id": indi_data.get("display_id", indi_id_norm),
            "full_name_disp": indi_data.get("full_name_disp", "N/A"),
            "total_score": total_score,
            "field_scores": field_scores,
            "reasons": reasons,
            "gender": indi_data.get("gender_raw", "N/A"),
            "birth_date": indi_data.get("birth_date_disp", "N/A"),
            "birth_place": indi_data.get("birth_place_disp"),
            "death_date": indi_data.get("death_date_disp"),
            "death_place": indi_data.get("death_place_disp"),
            "raw_data": indi_data,  # Store the raw data for detailed analysis
        }


# -----------------------------------------------------------------------------
# Standard Test Runner
# -----------------------------------------------------------------------------
from testing.test_utilities import create_standard_test_runner


def _test_module_integrity() -> bool:
    "Test that module can be imported and definitions are valid."
    return True


run_comprehensive_tests = create_standard_test_runner(_test_module_integrity)

if __name__ == "__main__":
    import sys

    sys.exit(0 if run_comprehensive_tests() else 1)
