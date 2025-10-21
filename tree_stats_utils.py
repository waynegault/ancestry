#!/usr/bin/env python3
"""
tree_stats_utils.py - Tree Statistics and Ethnicity Analysis

Provides functions to calculate and cache genealogical tree statistics
and DNA ethnicity commonality for enhanced messaging in Action 8.

Part of Phase 1: Enhanced Message Content (Foundation)
"""

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

        # Get the tree owner's Person record
        owner = session.query(Person).filter(Person.profile_id == profile_id).first()
        if not owner:
            logger.warning(f"No Person record found for profile_id={profile_id}")
            return _empty_statistics(profile_id)

        # Total DNA matches
        total_matches = session.query(func.count(DnaMatch.people_id)).scalar() or 0

        # In-tree vs out-of-tree counts
        in_tree_count = (
            session.query(func.count(Person.id))
            .filter(Person.in_my_tree == 1)
            .scalar() or 0
        )
        out_tree_count = total_matches - in_tree_count

        # Relationship tier counts (based on cM DNA shared)
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
        age_hours = _cache_age_hours(cache_entry.calculated_at)
        if age_hours > CACHE_EXPIRATION_HOURS:
            logger.debug(f"Cache expired (age: {age_hours}h > {CACHE_EXPIRATION_HOURS}h)")
            return None

        # Convert cache entry to statistics dict
        ethnicity_regions = {}
        if cache_entry.ethnicity_regions:
            ethnicity_regions = json.loads(cache_entry.ethnicity_regions)

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
            cache_entry.ethnicity_regions = ethnicity_json
            cache_entry.calculated_at = statistics['calculated_at']
            cache_entry.updated_at = datetime.now(timezone.utc)
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

        # Compare ethnicity regions
        shared_regions = []
        region_details = {}

        # Get ethnicity columns
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

        # Compare each region
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

        # Calculate similarity score (simple average of shared percentages)
        similarity_score = 0.0
        if shared_regions:
            total_shared = sum(
                min(details['owner_percentage'], details['match_percentage'])
                for details in region_details.values()
            )
            similarity_score = total_shared

        # Find top shared region
        top_shared_region = None
        if region_details:
            top_shared_region = max(
                region_details.keys(),
                key=lambda r: region_details[r]['owner_percentage'] + region_details[r]['match_percentage']
            )

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

    def test_statistics_functions_available() -> bool:
        """Test that all required functions are available."""
        required_functions = [
            'calculate_tree_statistics',
            'calculate_ethnicity_commonality',
        ]
        return all(func in globals() for func in required_functions)

    suite = TestSuite("Tree Statistics Utils", "tree_stats_utils")
    suite.run_test(
        "Function availability",
        test_statistics_functions_available,
        "All tree statistics functions available",
        "Verify calculate_tree_statistics and calculate_ethnicity_commonality exist",
        "Check function definitions in module globals"
    )

    success = suite.finish_suite()
    sys.exit(0 if success else 1)

