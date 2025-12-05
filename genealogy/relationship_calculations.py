#!/usr/bin/env python3
"""
Relationship Calculations Module.

Pure logic functions for calculating genealogical relationships based on
parent/child ID mappings. Decoupled from GEDCOM parsing and other dependencies.
"""


def is_ancestor_at_generation(
    descendant_id: str, ancestor_id: str, generations: int, id_to_parents: dict[str, set[str]]
) -> bool:
    """
    Check if ancestor_id is an ancestor of descendant_id at a specific generation level.

    Args:
        descendant_id: ID of the descendant
        ancestor_id: ID of the potential ancestor
        generations: Number of generations up (1=parent, 2=grandparent, 3=great-grandparent, etc.)
        id_to_parents: Dictionary mapping individual IDs to their parent IDs

    Returns:
        True if ancestor_id is an ancestor at the specified generation level
    """
    if generations < 1:
        return False

    # Start with the descendant
    current_generation = {descendant_id}

    # Walk up the specified number of generations
    for _ in range(generations):
        next_generation: set[str] = set()
        for person_id in current_generation:
            parents = id_to_parents.get(person_id, set())
            next_generation.update(parents)

        if not next_generation:
            return False  # No more ancestors at this level

        current_generation = next_generation

    # Check if ancestor_id is in the final generation
    return ancestor_id in current_generation


def is_descendant_at_generation(
    ancestor_id: str, descendant_id: str, generations: int, id_to_children: dict[str, set[str]]
) -> bool:
    """
    Check if descendant_id is a descendant of ancestor_id at a specific generation level.

    Args:
        ancestor_id: ID of the ancestor
        descendant_id: ID of the potential descendant
        generations: Number of generations down (1=child, 2=grandchild, 3=great-grandchild, etc.)
        id_to_children: Dictionary mapping individual IDs to their child IDs

    Returns:
        True if descendant_id is a descendant at the specified generation level
    """
    if generations < 1:
        return False

    # Start with the ancestor
    current_generation = {ancestor_id}

    # Walk down the specified number of generations
    for _ in range(generations):
        next_generation: set[str] = set()
        for person_id in current_generation:
            children = id_to_children.get(person_id, set())
            next_generation.update(children)

        if not next_generation:
            return False  # No more descendants at this level

        current_generation = next_generation

    # Check if descendant_id is in the final generation
    return descendant_id in current_generation


def is_grandparent(id1: str, id2: str, id_to_parents: dict[str, set[str]]) -> bool:
    """Check if id2 is a grandparent of id1."""
    return is_ancestor_at_generation(id1, id2, 2, id_to_parents)


def is_grandchild(id1: str, id2: str, id_to_children: dict[str, set[str]]) -> bool:
    """Check if id2 is a grandchild of id1."""
    return is_descendant_at_generation(id1, id2, 2, id_to_children)


def is_great_grandparent(id1: str, id2: str, id_to_parents: dict[str, set[str]]) -> bool:
    """Check if id2 is a great-grandparent of id1."""
    return is_ancestor_at_generation(id1, id2, 3, id_to_parents)


def is_great_grandchild(id1: str, id2: str, id_to_children: dict[str, set[str]]) -> bool:
    """Check if id2 is a great-grandchild of id1."""
    return is_descendant_at_generation(id1, id2, 3, id_to_children)


def are_siblings(id1: str, id2: str, id_to_parents: dict[str, set[str]]) -> bool:
    """Check if two individuals are siblings (share at least one parent)."""
    parents_1 = id_to_parents.get(id1, set())
    parents_2 = id_to_parents.get(id2, set())
    return bool(parents_1 and parents_2 and not parents_1.isdisjoint(parents_2))


def is_aunt_or_uncle(
    id1: str,
    id2: str,
    id_to_parents: dict[str, set[str]],
    id_to_children: dict[str, set[str]],
) -> bool:
    """Check if id2 is an aunt or uncle of id1."""
    # Get parents of id1
    parents = id_to_parents.get(id1, set())

    # For each parent, check if id2 is their sibling
    for parent_id in parents:
        # Get grandparents (parents of parent)
        grandparents = id_to_parents.get(parent_id, set())

        # For each grandparent, get their children
        for grandparent_id in grandparents:
            aunts_uncles = id_to_children.get(grandparent_id, set())

            # If id2 is a child of a grandparent and not a parent, it's an aunt/uncle
            if id2 in aunts_uncles and id2 != parent_id:
                return True

    return False


def is_niece_or_nephew(
    id1: str,
    id2: str,
    id_to_parents: dict[str, set[str]],
    id_to_children: dict[str, set[str]],
) -> bool:
    """Check if id2 is a niece or nephew of id1."""
    # This is the reverse of aunt/uncle relationship
    return is_aunt_or_uncle(id2, id1, id_to_parents, id_to_children)


def are_cousins(
    id1: str,
    id2: str,
    id_to_parents: dict[str, set[str]],
) -> bool:
    """Check if id1 and id2 are cousins (children of siblings)."""
    # Get parents of id1
    parents1 = id_to_parents.get(id1, set())
    parents2 = id_to_parents.get(id2, set())

    # For each parent of id1, check if they have a sibling who is a parent of id2
    for parent1 in parents1:
        # Get grandparents of id1
        grandparents1 = id_to_parents.get(parent1, set())

        for parent2 in parents2:
            # Get grandparents of id2
            grandparents2 = id_to_parents.get(parent2, set())

            # If they share a grandparent but have different parents, they're cousins
            if (grandparents1 and grandparents2 and not grandparents1.isdisjoint(grandparents2)) and (
                parent1 != parent2
            ):  # Make sure they don't have the same parent (which would make them siblings)
                return True

    return False


def find_direct_relationship(
    id1: str,
    id2: str,
    id_to_parents: dict[str, set[str]],
    id_to_children: dict[str, set[str]],
) -> list[str]:
    """
    Find a direct relationship between two individuals.

    Args:
        id1: ID of the first individual
        id2: ID of the second individual
        id_to_parents: Dictionary mapping individual IDs to their parent IDs
        id_to_children: Dictionary mapping individual IDs to their child IDs

    Returns:
        A list of IDs representing the path from id1 to id2, or an empty list if no direct relationship
    """
    # Check if id2 is a parent of id1
    if id2 in id_to_parents.get(id1, set()):
        return [id1, id2]

    # Check if id2 is a child of id1
    if id2 in id_to_children.get(id1, set()):
        return [id1, id2]

    # Check if id1 and id2 are siblings (share at least one parent)
    parents_1 = id_to_parents.get(id1, set())
    parents_2 = id_to_parents.get(id2, set())
    common_parents = parents_1.intersection(parents_2)
    if common_parents:
        # Use the first common parent
        common_parent = next(iter(common_parents))
        return [id1, common_parent, id2]

    # No direct relationship found
    return []


def has_direct_relationship(
    id1: str,
    id2: str,
    id_to_parents: dict[str, set[str]],
    id_to_children: dict[str, set[str]],
) -> bool:
    """
    Check if two individuals have a direct relationship (parent-child, siblings).
    Note: Spouses are not checked here as that requires family record access.

    Args:
        id1: ID of the first individual
        id2: ID of the second individual
        id_to_parents: Dictionary mapping individual IDs to their parent IDs
        id_to_children: Dictionary mapping individual IDs to their child IDs

    Returns:
        True if a direct relationship exists
    """
    # Check parent/child
    if id2 in id_to_parents.get(id1, set()) or id2 in id_to_children.get(id1, set()):
        return True

    # Check siblings
    parents_1 = id_to_parents.get(id1, set())
    parents_2 = id_to_parents.get(id2, set())
    return bool(parents_1 and parents_2 and not parents_1.isdisjoint(parents_2))


# =============================================================================
# TESTS
# =============================================================================


def _test_relationship_calculations() -> bool:
    """Test relationship calculation functions."""
    # Setup a simple 3-generation family tree
    # G1: GP1, GP2
    # G2: P1 (child of GP1, GP2), P2 (child of GP1, GP2), P3 (spouse of P2), P4 (unrelated)
    # G3: C1 (child of P1), C2 (child of P2, P3), C3 (child of P2, P3)

    id_to_parents = {
        "P1": {"GP1", "GP2"},
        "P2": {"GP1", "GP2"},
        "C1": {"P1"},
        "C2": {"P2", "P3"},
        "C3": {"P2", "P3"},
    }

    id_to_children = {
        "GP1": {"P1", "P2"},
        "GP2": {"P1", "P2"},
        "P1": {"C1"},
        "P2": {"C2", "C3"},
        "P3": {"C2", "C3"},
    }

    # Test Ancestors
    assert is_ancestor_at_generation("C1", "P1", 1, id_to_parents) is True, "P1 should be parent of C1"
    assert is_ancestor_at_generation("C1", "GP1", 2, id_to_parents) is True, "GP1 should be grandparent of C1"
    assert is_ancestor_at_generation("C1", "P2", 1, id_to_parents) is False, "P2 should NOT be parent of C1"

    # Test Descendants
    assert is_descendant_at_generation("P1", "C1", 1, id_to_children) is True, "C1 should be child of P1"
    assert is_descendant_at_generation("GP1", "C1", 2, id_to_children) is True, "C1 should be grandchild of GP1"

    # Test Grandparents/Grandchildren
    assert is_grandparent("C2", "GP1", id_to_parents) is True, "GP1 is grandparent of C2"
    assert is_grandchild("GP1", "C2", id_to_children) is True, "C2 is grandchild of GP1"

    # Test Siblings
    assert are_siblings("P1", "P2", id_to_parents) is True, "P1 and P2 should be siblings"
    assert are_siblings("C2", "C3", id_to_parents) is True, "C2 and C3 should be siblings"
    assert are_siblings("C1", "C2", id_to_parents) is False, "C1 and C2 should NOT be siblings"

    # Test Cousins (C1 and C2 are children of siblings P1 and P2)
    assert are_cousins("C1", "C2", id_to_parents) is True, "C1 and C2 should be cousins"
    assert are_cousins("C2", "C3", id_to_parents) is False, "C2 and C3 are siblings, not cousins"

    # Test Aunt/Uncle
    assert is_aunt_or_uncle("C1", "P2", id_to_parents, id_to_children) is True, "P2 is aunt/uncle of C1"
    assert is_aunt_or_uncle("C2", "P1", id_to_parents, id_to_children) is True, "P1 is aunt/uncle of C2"
    assert is_aunt_or_uncle("C1", "P4", id_to_parents, id_to_children) is False, "P4 is unrelated"

    # Test Direct Relationship
    assert has_direct_relationship("P1", "C1", id_to_parents, id_to_children) is True, "Parent/Child is direct"
    assert has_direct_relationship("P1", "P2", id_to_parents, id_to_children) is True, "Siblings is direct"
    assert has_direct_relationship("GP1", "C1", id_to_parents, id_to_children) is False, (
        "Grandparent is NOT direct (by this def)"
    )

    return True


def module_tests() -> bool:
    """Run module tests for relationship_calculations."""
    from testing.test_framework import TestSuite

    suite = TestSuite("relationship_calculations", "genealogy/relationship_calculations.py")

    suite.run_test(
        "Relationship Logic",
        _test_relationship_calculations,
        "Validates core relationship logic (ancestor, descendant, sibling, cousin, etc.)",
    )

    return suite.finish_suite()


if __name__ == "__main__":
    import sys

    from testing.test_framework import create_standard_test_runner

    run_comprehensive_tests = create_standard_test_runner(module_tests)
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
