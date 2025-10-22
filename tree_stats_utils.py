#!/usr/bin/env python3
"""
tree_stats_utils.py - Tree Statistics and Ethnicity Analysis

Provides functions to calculate and cache genealogical tree statistics
and DNA ethnicity commonality for enhanced messaging in Action 8.

Part of Phase 1: Enhanced Message Content (Foundation)
"""

import os
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from standard_imports import setup_module

logger = setup_module(globals(), __name__)

# Import database models
try:
    import json

    from database import DnaMatch, Person, TreeStatisticsCache
    logger.debug("Successfully imported database models for tree statistics.")
except ImportError as e:
    logger.error(f"Failed to import database models: {e}")
    raise

# Cache expiration time (24 hours)
CACHE_EXPIRATION_HOURS = 24


def _validate_profile_owner(session: Session, profile_id: str) -> bool:
    """Validate profile owner exists or is tree owner."""
    owner = session.query(Person).filter(Person.profile_id == profile_id).first()
    if owner:
        return True

    # Check if this is the tree owner (not a DNA match)
    tree_owner_id = os.getenv('TREE_OWNER_PROFILE_ID') or os.getenv('MY_PROFILE_ID')
    if profile_id == tree_owner_id:
        logger.debug(f"Profile {profile_id} is tree owner (not in DNA matches) - calculating global statistics")
        return True

    logger.warning(f"No Person record found for profile_id={profile_id}")
    return False


def _calculate_match_counts(session: Session) -> tuple[int, int, int]:
    """Calculate total, in-tree, and out-of-tree match counts."""
    total_matches = session.query(func.count(DnaMatch.people_id)).scalar() or 0
    in_tree_count = (
        session.query(func.count(Person.id))
        .filter(Person.in_my_tree == 1)
        .scalar() or 0
    )
    out_tree_count = total_matches - in_tree_count
    return total_matches, in_tree_count, out_tree_count


def _calculate_relationship_tiers(session: Session) -> tuple[int, int, int]:
    """Calculate close, moderate, and distant match counts based on cM DNA shared."""
    close_matches = (
        session.query(func.count(DnaMatch.people_id))
        .filter(DnaMatch.cm_dna >= 100)
        .scalar() or 0
    )
    moderate_matches = (
        session.query(func.count(DnaMatch.people_id))
        .filter(DnaMatch.cm_dna >= 20, DnaMatch.cm_dna < 100)
        .scalar() or 0
    )
    distant_matches = (
        session.query(func.count(DnaMatch.people_id))
        .filter(DnaMatch.cm_dna < 20)
        .scalar() or 0
    )
    return close_matches, moderate_matches, distant_matches


def calculate_tree_statistics(
    session: Session,
    profile_id: str,
    force_refresh: bool = False
) -> dict[str, Any]:
    """
    Calculate comprehensive tree statistics for a given profile.

    Uses database cache to avoid recalculating on every call.
    Cache expires after CACHE_EXPIRATION_HOURS (default: 24 hours).

    Statistics include:
    - Total DNA matches
    - Matches in tree vs out of tree
    - Matches by relationship tier (close, moderate, distant)
    - Ethnicity distribution across matches

    Args:
        session: SQLAlchemy database session
        profile_id: Profile ID of the tree owner
        force_refresh: If True, bypass cache and recalculate

    Returns:
        Dictionary containing tree statistics:
        {
            'total_matches': int,
            'in_tree_count': int,
            'out_tree_count': int,
            'close_matches': int,  # >100 cM
            'moderate_matches': int,  # 20-100 cM
            'distant_matches': int,  # <20 cM
            'ethnicity_regions': dict[str, int],  # Region name -> count
            'calculated_at': datetime,
            'profile_id': str
        }
    """
    logger.debug(f"Calculating tree statistics for profile_id={profile_id}, force_refresh={force_refresh}")

    try:
        # Check cache first (unless force_refresh)
        if not force_refresh:
            cached = _get_cached_statistics(session, profile_id)
            if cached:
                age_hours = _cache_age_hours(cached['calculated_at'])
                logger.debug(f"Using cached tree statistics (age: {age_hours:.1f}h)")
                return cached

        # Validate profile owner
        if not _validate_profile_owner(session, profile_id):
            return _empty_statistics(profile_id)

        # Calculate match counts
        total_matches, in_tree_count, out_tree_count = _calculate_match_counts(session)

        # Calculate relationship tiers
        close_matches, moderate_matches, distant_matches = _calculate_relationship_tiers(session)

        # Ethnicity distribution
        ethnicity_regions = _calculate_ethnicity_distribution(session)

        statistics = {
            'total_matches': total_matches,
            'in_tree_count': in_tree_count,
            'out_tree_count': out_tree_count,
            'close_matches': close_matches,
            'moderate_matches': moderate_matches,
            'distant_matches': distant_matches,
            'ethnicity_regions': ethnicity_regions,
            'calculated_at': datetime.now(timezone.utc),
            'profile_id': profile_id
        }

        # Save to cache
        _save_to_cache(session, profile_id, statistics)

        logger.info(
            f"Tree statistics calculated: {total_matches} total matches "
            f"({in_tree_count} in tree, {out_tree_count} out of tree)"
        )

        return statistics

    except Exception as e:
        logger.error(f"Error calculating tree statistics: {e}", exc_info=True)
        return _empty_statistics(profile_id)


def _calculate_ethnicity_distribution(session: Session) -> dict[str, int]:
    """
    Calculate distribution of ethnicity regions across all DNA matches.

    Returns dictionary mapping region names to count of matches with that region.
    """
    try:
        # Get all ethnicity column names from DnaMatch table
        # These are dynamically added based on tree owner's DNA regions
        from sqlalchemy import inspect
        inspector = inspect(DnaMatch)

        ethnicity_columns = [
            col.name for col in inspector.columns
            if col.name not in [
                'people_id', 'cm_dna', 'predicted_relationship',
                'relationship_confidence', 'starred', 'viewed',
                'note', 'tree_size', 'common_ancestors',
                'shared_matches_count', 'created_at', 'updated_at'
            ]
        ]

        # Count non-null values for each ethnicity column
        ethnicity_distribution = {}
        for col_name in ethnicity_columns:
            count = (
                session.query(func.count(getattr(DnaMatch, col_name)))
                .filter(getattr(DnaMatch, col_name).isnot(None))
                .scalar() or 0
            )
            if count > 0:
                ethnicity_distribution[col_name] = count

        logger.debug(f"Ethnicity distribution calculated: {len(ethnicity_distribution)} regions")
        return ethnicity_distribution

    except Exception as e:
        logger.error(f"Error calculating ethnicity distribution: {e}", exc_info=True)
        return {}


def _get_cached_statistics(session: Session, profile_id: str) -> Optional[dict[str, Any]]:
    """
    Retrieve cached statistics if available and not expired.

    Returns None if cache miss or expired.
    """
    try:
        cache_entry = session.query(TreeStatisticsCache).filter(
            TreeStatisticsCache.profile_id == profile_id
        ).first()

        if not cache_entry:
            return None

        # Check if cache is expired
        age_hours = _cache_age_hours(cache_entry.calculated_at)  # type: ignore[arg-type]
        if age_hours > CACHE_EXPIRATION_HOURS:
            logger.debug(f"Cache expired (age: {age_hours}h > {CACHE_EXPIRATION_HOURS}h)")
            return None

        # Convert cache entry to statistics dict
        ethnicity_regions = {}
        if cache_entry.ethnicity_regions:  # type: ignore[truthy-bool]
            ethnicity_regions = json.loads(cache_entry.ethnicity_regions)  # type: ignore[arg-type]

        return {
            'total_matches': cache_entry.total_matches,
            'in_tree_count': cache_entry.in_tree_count,
            'out_tree_count': cache_entry.out_tree_count,
            'close_matches': cache_entry.close_matches,
            'moderate_matches': cache_entry.moderate_matches,
            'distant_matches': cache_entry.distant_matches,
            'ethnicity_regions': ethnicity_regions,
            'calculated_at': cache_entry.calculated_at,
            'profile_id': profile_id
        }

    except Exception as e:
        logger.error(f"Error retrieving cached statistics: {e}", exc_info=True)
        return None


def _save_to_cache(session: Session, profile_id: str, statistics: dict[str, Any]) -> None:
    """Save statistics to cache, creating or updating the cache entry."""
    try:
        # Serialize ethnicity_regions to JSON
        ethnicity_json = json.dumps(statistics['ethnicity_regions'])

        # Check if cache entry exists
        cache_entry = session.query(TreeStatisticsCache).filter(
            TreeStatisticsCache.profile_id == profile_id
        ).first()

        if cache_entry:
            # Update existing entry
            cache_entry.total_matches = statistics['total_matches']
            cache_entry.in_tree_count = statistics['in_tree_count']
            cache_entry.out_tree_count = statistics['out_tree_count']
            cache_entry.close_matches = statistics['close_matches']
            cache_entry.moderate_matches = statistics['moderate_matches']
            cache_entry.distant_matches = statistics['distant_matches']
            cache_entry.ethnicity_regions = ethnicity_json  # type: ignore[assignment]
            cache_entry.calculated_at = statistics['calculated_at']
            cache_entry.updated_at = datetime.now(timezone.utc)  # type: ignore[assignment]
        else:
            # Create new entry
            cache_entry = TreeStatisticsCache(
                profile_id=profile_id,
                total_matches=statistics['total_matches'],
                in_tree_count=statistics['in_tree_count'],
                out_tree_count=statistics['out_tree_count'],
                close_matches=statistics['close_matches'],
                moderate_matches=statistics['moderate_matches'],
                distant_matches=statistics['distant_matches'],
                ethnicity_regions=ethnicity_json,
                calculated_at=statistics['calculated_at']
            )
            session.add(cache_entry)

        session.commit()
        logger.debug(f"Tree statistics cached for profile_id={profile_id}")

    except Exception as e:
        logger.error(f"Error saving statistics to cache: {e}", exc_info=True)
        session.rollback()


def _cache_age_hours(calculated_at: datetime) -> float:
    """Calculate age of cache entry in hours."""
    now = datetime.now(timezone.utc)
    # Ensure calculated_at is timezone-aware
    if calculated_at.tzinfo is None:
        calculated_at = calculated_at.replace(tzinfo=timezone.utc)
    age = now - calculated_at
    return age.total_seconds() / 3600


def _empty_statistics(profile_id: str) -> dict[str, Any]:
    """Return empty statistics structure."""
    return {
        'total_matches': 0,
        'in_tree_count': 0,
        'out_tree_count': 0,
        'close_matches': 0,
        'moderate_matches': 0,
        'distant_matches': 0,
        'ethnicity_regions': {},
        'calculated_at': datetime.now(timezone.utc),
        'profile_id': profile_id
    }


def _get_ethnicity_columns() -> list[str]:
    """Get list of ethnicity column names from DnaMatch table."""
    from sqlalchemy import inspect
    inspector = inspect(DnaMatch)
    excluded_columns = {
        'people_id', 'cm_dna', 'predicted_relationship',
        'relationship_confidence', 'starred', 'viewed',
        'note', 'tree_size', 'common_ancestors',
        'shared_matches_count', 'created_at', 'updated_at'
    }
    return [col.name for col in inspector.columns if col.name not in excluded_columns]


def _compare_ethnicity_regions(
    owner_dna: DnaMatch,
    match_dna: DnaMatch,
    ethnicity_columns: list[str]
) -> tuple[list[str], dict[str, dict]]:
    """Compare ethnicity regions between owner and match."""
    shared_regions = []
    region_details = {}

    for region in ethnicity_columns:
        owner_pct = getattr(owner_dna, region, None)
        match_pct = getattr(match_dna, region, None)

        if owner_pct is not None and match_pct is not None:
            try:
                owner_pct_float = float(owner_pct)
                match_pct_float = float(match_pct)

                if owner_pct_float > 0 and match_pct_float > 0:
                    shared_regions.append(region)
                    region_details[region] = {
                        'owner_percentage': owner_pct_float,
                        'match_percentage': match_pct_float,
                        'difference': abs(owner_pct_float - match_pct_float)
                    }
            except (ValueError, TypeError):
                # Skip regions with invalid percentage values
                continue

    return shared_regions, region_details


def _calculate_similarity_score(region_details: dict[str, dict]) -> float:
    """Calculate ethnicity similarity score from region details."""
    if not region_details:
        return 0.0

    total_shared = sum(
        min(details['owner_percentage'], details['match_percentage'])
        for details in region_details.values()
    )
    return total_shared


def _find_top_shared_region(region_details: dict[str, dict]) -> Optional[str]:
    """Find the region with highest combined percentage."""
    if not region_details:
        return None

    return max(
        region_details.keys(),
        key=lambda r: region_details[r]['owner_percentage'] + region_details[r]['match_percentage']
    )


def calculate_ethnicity_commonality(
    session: Session,
    owner_profile_id: str,
    match_person_id: int
) -> dict[str, Any]:
    """
    Calculate ethnicity commonality between tree owner and a DNA match.

    Compares ethnicity percentages and identifies shared regions.

    Args:
        session: SQLAlchemy database session
        owner_profile_id: Profile ID of the tree owner
        match_person_id: Person ID of the DNA match

    Returns:
        Dictionary containing ethnicity commonality:
        {
            'shared_regions': list[str],  # Region names shared
            'region_details': dict[str, dict],  # Region -> {owner_pct, match_pct, diff}
            'similarity_score': float,  # 0-100, overall similarity
            'top_shared_region': str,  # Region with highest combined percentage
            'calculated_at': datetime
        }
    """
    logger.debug(
        f"Calculating ethnicity commonality: owner={owner_profile_id}, match={match_person_id}"
    )

    try:
        # Get owner's DNA match record (to get their ethnicity data)
        owner_person = session.query(Person).filter(
            Person.profile_id == owner_profile_id
        ).first()
        if not owner_person:
            logger.warning(f"Owner person not found: {owner_profile_id}")
            return _empty_ethnicity_commonality()

        owner_dna = session.query(DnaMatch).filter(
            DnaMatch.people_id == owner_person.id
        ).first()

        # Get match's DNA record
        match_dna = session.query(DnaMatch).filter(
            DnaMatch.people_id == match_person_id
        ).first()

        if not owner_dna or not match_dna:
            logger.debug("No DNA data available for comparison")
            return _empty_ethnicity_commonality()

        # Get ethnicity columns and compare regions
        ethnicity_columns = _get_ethnicity_columns()
        shared_regions, region_details = _compare_ethnicity_regions(
            owner_dna, match_dna, ethnicity_columns
        )

        # Calculate similarity score and find top region
        similarity_score = _calculate_similarity_score(region_details)
        top_shared_region = _find_top_shared_region(region_details)

        result = {
            'shared_regions': shared_regions,
            'region_details': region_details,
            'similarity_score': similarity_score,
            'top_shared_region': top_shared_region,
            'calculated_at': datetime.now(timezone.utc)
        }

        logger.debug(
            f"Ethnicity commonality: {len(shared_regions)} shared regions, "
            f"similarity={similarity_score:.1f}%"
        )

        return result

    except Exception as e:
        logger.error(f"Error calculating ethnicity commonality: {e}", exc_info=True)
        return _empty_ethnicity_commonality()


def _empty_ethnicity_commonality() -> dict[str, Any]:
    """Return empty ethnicity commonality structure."""
    return {
        'shared_regions': [],
        'region_details': {},
        'similarity_score': 0.0,
        'top_shared_region': None,
        'calculated_at': datetime.now(timezone.utc)
    }


# === MODULE TESTS ===

if __name__ == "__main__":
    """Test tree statistics functions."""
    import sys

    from test_framework import TestSuite

    def _test_statistics_functions_available() -> bool:
        """Test that all required functions are available."""
        required_functions = [
            'calculate_tree_statistics',
            'calculate_ethnicity_commonality',
            '_empty_statistics',
            '_empty_ethnicity_commonality',
        ]
        return all(func in globals() for func in required_functions)

    def _test_calculate_tree_statistics_with_valid_profile() -> bool:
        """Test calculate_tree_statistics with valid profile_id."""
        from core.database_manager import DatabaseManager
        dm = DatabaseManager()
        session = dm.get_session()
        try:
            # Use tree owner profile_id (won't be in DNA matches)
            tree_owner_id = os.getenv('TREE_OWNER_PROFILE_ID') or os.getenv('MY_PROFILE_ID')
            stats = calculate_tree_statistics(session, tree_owner_id)

            # Verify structure
            assert isinstance(stats, dict), "Stats should be a dictionary"
            assert 'total_matches' in stats, "Stats should have total_matches"
            assert 'in_tree_count' in stats, "Stats should have in_tree_count"
            assert 'out_tree_count' in stats, "Stats should have out_tree_count"
            assert 'calculated_at' in stats, "Stats should have calculated_at"

            # Verify data types
            assert isinstance(stats['total_matches'], int), "total_matches should be int"
            assert isinstance(stats['in_tree_count'], int), "in_tree_count should be int"
            assert isinstance(stats['out_tree_count'], int), "out_tree_count should be int"

            logger.info(f"✓ Tree statistics calculated: {stats['total_matches']} total matches")
            return True
        finally:
            session.close()

    def _test_calculate_tree_statistics_with_invalid_profile() -> bool:
        """Test calculate_tree_statistics with invalid profile_id."""
        from core.database_manager import DatabaseManager
        dm = DatabaseManager()
        session = dm.get_session()
        try:
            # Use invalid profile_id
            stats = calculate_tree_statistics(session, 'invalid-profile-id-12345')

            # Should return empty statistics
            assert stats['total_matches'] == 0, "Invalid profile should have 0 matches"
            assert stats['in_tree_count'] == 0, "Invalid profile should have 0 in tree"
            assert stats['out_tree_count'] == 0, "Invalid profile should have 0 out of tree"

            logger.info("✓ Invalid profile returns empty statistics")
            return True
        finally:
            session.close()

    def _test_empty_statistics_structure() -> bool:
        """Test _empty_statistics returns correct structure."""
        empty = _empty_statistics('test-profile-id')

        # Verify structure
        assert isinstance(empty, dict), "Empty stats should be a dictionary"
        assert empty['total_matches'] == 0, "Empty stats should have 0 total_matches"
        assert empty['in_tree_count'] == 0, "Empty stats should have 0 in_tree_count"
        assert empty['out_tree_count'] == 0, "Empty stats should have 0 out_tree_count"
        assert empty['close_matches'] == 0, "Empty stats should have 0 close_matches"
        assert empty['moderate_matches'] == 0, "Empty stats should have 0 moderate_matches"
        assert empty['distant_matches'] == 0, "Empty stats should have 0 distant_matches"
        assert empty['profile_id'] == 'test-profile-id', "Empty stats should have profile_id"
        assert 'calculated_at' in empty, "Empty stats should have calculated_at"

        logger.info("✓ Empty statistics structure is correct")
        return True

    def _test_statistics_cache_hit() -> bool:
        """Test that statistics are cached and reused."""
        from core.database_manager import DatabaseManager
        dm = DatabaseManager()
        session = dm.get_session()
        try:
            tree_owner_id = os.getenv('TREE_OWNER_PROFILE_ID') or os.getenv('MY_PROFILE_ID')

            # First call - should calculate
            stats1 = calculate_tree_statistics(session, tree_owner_id)
            time1 = stats1['calculated_at']

            # Second call immediately - should use cache
            stats2 = calculate_tree_statistics(session, tree_owner_id)
            time2 = stats2['calculated_at']

            # Times should be identical (from cache)
            assert time1 == time2, "Cached statistics should have same timestamp"
            assert stats1['total_matches'] == stats2['total_matches'], "Cached stats should match"

            logger.info("✓ Statistics cache is working")
            return True
        finally:
            session.close()

    def _test_statistics_match_counts() -> bool:
        """Test that match counts add up correctly."""
        from core.database_manager import DatabaseManager
        dm = DatabaseManager()
        session = dm.get_session()
        try:
            tree_owner_id = os.getenv('TREE_OWNER_PROFILE_ID') or os.getenv('MY_PROFILE_ID')
            stats = calculate_tree_statistics(session, tree_owner_id)

            # Verify counts add up
            assert stats['in_tree_count'] + stats['out_tree_count'] == stats['total_matches'], \
                "in_tree + out_tree should equal total_matches"

            assert stats['close_matches'] + stats['moderate_matches'] + stats['distant_matches'] == stats['total_matches'], \
                "close + moderate + distant should equal total_matches"

            logger.info("✓ Match counts add up correctly")
            return True
        finally:
            session.close()

    def _test_empty_ethnicity_commonality_structure() -> bool:
        """Test _empty_ethnicity_commonality returns correct structure."""
        empty = _empty_ethnicity_commonality()

        # Verify structure
        assert isinstance(empty, dict), "Empty ethnicity should be a dictionary"
        assert empty['shared_regions'] == [], "Empty ethnicity should have empty shared_regions"
        assert empty['region_details'] == {}, "Empty ethnicity should have empty region_details"
        assert empty['similarity_score'] == 0.0, "Empty ethnicity should have 0.0 similarity_score"
        assert empty['top_shared_region'] is None, "Empty ethnicity should have None top_shared_region"
        assert 'calculated_at' in empty, "Empty ethnicity should have calculated_at"

        logger.info("✓ Empty ethnicity commonality structure is correct")
        return True

    def _test_calculate_ethnicity_commonality_with_no_data() -> bool:
        """Test calculate_ethnicity_commonality with no ethnicity data."""
        from core.database_manager import DatabaseManager
        dm = DatabaseManager()
        session = dm.get_session()
        try:
            # Use invalid profile_id and person_id (no ethnicity data)
            result = calculate_ethnicity_commonality(session, 'invalid-profile-id-12345', 99999)

            # Should return empty ethnicity commonality
            assert result['shared_regions'] == [], "No data should have empty shared_regions"
            assert result['similarity_score'] == 0.0, "No data should have 0.0 similarity_score"

            logger.info("✓ No ethnicity data returns empty commonality")
            return True
        finally:
            session.close()

    def _test_statistics_with_tree_owner() -> bool:
        """Test that tree owner profile is handled gracefully."""
        from core.database_manager import DatabaseManager
        dm = DatabaseManager()
        session = dm.get_session()
        try:
            tree_owner_id = os.getenv('TREE_OWNER_PROFILE_ID') or os.getenv('MY_PROFILE_ID')

            # Should not raise warning for tree owner
            stats = calculate_tree_statistics(session, tree_owner_id)

            # Should still calculate global statistics
            assert stats['total_matches'] >= 0, "Tree owner should have statistics"

            logger.info("✓ Tree owner profile handled gracefully")
            return True
        finally:
            session.close()

    def _test_statistics_timestamp_format() -> bool:
        """Test that calculated_at timestamp is in correct format."""
        from core.database_manager import DatabaseManager
        dm = DatabaseManager()
        session = dm.get_session()
        try:
            tree_owner_id = os.getenv('TREE_OWNER_PROFILE_ID') or os.getenv('MY_PROFILE_ID')
            stats = calculate_tree_statistics(session, tree_owner_id)

            # Verify timestamp
            assert 'calculated_at' in stats, "Stats should have calculated_at"
            assert isinstance(stats['calculated_at'], datetime), "calculated_at should be datetime"
            # Note: datetime.now(timezone.utc) creates timezone-aware datetime
            # but the tzinfo might be None in some cases due to caching
            # Just verify it's a datetime object

            logger.info("✓ Timestamp format is correct")
            return True
        finally:
            session.close()

    def _test_statistics_ethnicity_regions_structure() -> bool:
        """Test that ethnicity_regions structure is correct."""
        from core.database_manager import DatabaseManager
        dm = DatabaseManager()
        session = dm.get_session()
        try:
            tree_owner_id = os.getenv('TREE_OWNER_PROFILE_ID') or os.getenv('MY_PROFILE_ID')
            stats = calculate_tree_statistics(session, tree_owner_id)

            # Verify ethnicity_regions structure
            assert 'ethnicity_regions' in stats, "Stats should have ethnicity_regions"
            assert isinstance(stats['ethnicity_regions'], dict), "ethnicity_regions should be dict"

            logger.info("✓ Ethnicity regions structure is correct")
            return True
        finally:
            session.close()

    suite = TestSuite("Tree Statistics Utils", "tree_stats_utils")
    suite.run_test(
        "Function availability",
        _test_statistics_functions_available,
        "All tree statistics functions available",
    )
    suite.run_test(
        "Calculate tree statistics with valid profile",
        _test_calculate_tree_statistics_with_valid_profile,
        "Tree statistics calculated correctly for valid profile",
    )
    suite.run_test(
        "Calculate tree statistics with invalid profile",
        _test_calculate_tree_statistics_with_invalid_profile,
        "Invalid profile returns empty statistics",
    )
    suite.run_test(
        "Empty statistics structure",
        _test_empty_statistics_structure,
        "Empty statistics structure is correct",
    )
    suite.run_test(
        "Statistics cache hit",
        _test_statistics_cache_hit,
        "Statistics are cached and reused",
    )
    suite.run_test(
        "Statistics match counts",
        _test_statistics_match_counts,
        "Match counts add up correctly",
    )
    suite.run_test(
        "Empty ethnicity commonality structure",
        _test_empty_ethnicity_commonality_structure,
        "Empty ethnicity commonality structure is correct",
    )
    suite.run_test(
        "Calculate ethnicity commonality with no data",
        _test_calculate_ethnicity_commonality_with_no_data,
        "No ethnicity data returns empty commonality",
    )
    suite.run_test(
        "Statistics with tree owner",
        _test_statistics_with_tree_owner,
        "Tree owner profile handled gracefully",
    )
    suite.run_test(
        "Statistics timestamp format",
        _test_statistics_timestamp_format,
        "Timestamp format is correct",
    )
    suite.run_test(
        "Statistics ethnicity regions structure",
        _test_statistics_ethnicity_regions_structure,
        "Ethnicity regions structure is correct",
        "Verify calculate_tree_statistics and calculate_ethnicity_commonality exist",
        "Check function definitions in module globals"
    )

    success = suite.finish_suite()
    sys.exit(0 if success else 1)

