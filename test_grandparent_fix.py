#!/usr/bin/env python3
"""
Test the _is_grandparent and _is_grandchild fixes
"""

print("Testing grandparent/grandchild function fixes...")

# Test data
# Family structure:
# - grandparent1 -> parent1 -> child1
# - grandparent1 -> parent2 -> child2
id_to_parents = {
    "child1": {"parent1"},
    "child2": {"parent2"},
    "parent1": {"grandparent1"},
    "parent2": {"grandparent1"},
}

id_to_children = {
    "grandparent1": {"parent1", "parent2"},
    "parent1": {"child1"},
    "parent2": {"child2"},
}


# Define the fixed functions locally to test them
def _is_grandparent_fixed(id1: str, id2: str, id_to_parents) -> bool:
    """Check if id2 is a grandparent of id1."""
    if not id1 or not id2:
        return False
    # Get parents of id1
    parents = id_to_parents.get(id1, set())
    # For each parent, check if id2 is their parent
    for parent_id in parents:
        grandparents = id_to_parents.get(parent_id, set())
        if id2 in grandparents:
            return True
    return False


def _is_grandchild_fixed(id1: str, id2: str, id_to_children) -> bool:
    """Check if id2 is a grandchild of id1."""
    if not id1 or not id2:
        return False
    # Get children of id1
    children = id_to_children.get(id1, set())
    # For each child, check if id2 is their child
    for child_id in children:
        grandchildren = id_to_children.get(child_id, set())
        if id2 in grandchildren:
            return True
    return False


# Test cases
tests = [
    # (function, args, expected_result, description)
    (
        _is_grandparent_fixed,
        ("child1", "grandparent1", id_to_parents),
        True,
        "child1 has grandparent1 as grandparent",
    ),
    (
        _is_grandparent_fixed,
        ("parent1", "grandparent1", id_to_parents),
        False,
        "parent1 has grandparent1 as parent, not grandparent",
    ),
    (
        _is_grandparent_fixed,
        ("grandparent1", "child1", id_to_parents),
        False,
        "grandparent1 is not the grandchild",
    ),
    (
        _is_grandchild_fixed,
        ("grandparent1", "child1", id_to_children),
        True,
        "grandparent1 has child1 as grandchild",
    ),
    (
        _is_grandchild_fixed,
        ("parent1", "child1", id_to_children),
        False,
        "parent1 has child1 as child, not grandchild",
    ),
    (
        _is_grandchild_fixed,
        ("child1", "grandparent1", id_to_children),
        False,
        "child1 is not the grandparent",
    ),
]

all_passed = True

for func, args, expected, description in tests:
    result = func(*args)
    status = "✓ PASS" if result == expected else "❌ FAIL"
    if result != expected:
        all_passed = False
    print(f"{status} - {description}: {result} (expected {expected})")

if all_passed:
    print("\n✅ All grandparent/grandchild tests PASSED!")
else:
    print("\n❌ Some tests FAILED!")
