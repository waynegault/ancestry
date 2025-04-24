# action10.py: GEDCOM Report (local fuzzy search, family, relationship to WGG)
import sys
import os
from pathlib import Path
from logging_config import setup_logging
from config import config_instance
from utils import SessionManager

# Import all relevant helpers from utils
from utils import (
    GEDCOM_LIB_AVAILABLE,
    GedcomReader,
    build_indi_index,
    _is_name,
    _normalize_id,
    _get_full_name,
    _is_individual,
    format_relative_info,
    get_related_individuals,
    _find_family_records_where_individual_is_child,
    _find_family_records_where_individual_is_parent,
    format_life_dates,
    format_full_life_details,
    find_individual_by_id,
    extract_and_fix_id,
)

from collections import deque
from typing import Dict, Optional, Union
import logging

# --- GEDCOM File Helpers (migrated from gedcom.py/utils.py) ---

def _get_ancestors_map(reader, start_id_norm):
    """Builds a map of ancestor IDs to their depth relative to the start ID."""
    ancestors = {}
    queue = deque([(start_id_norm, 0)])  # Queue stores (ID, depth)
    visited = {start_id_norm}
    if not reader or not start_id_norm:
        logging.error("_get_ancestors_map: Invalid input.")
        return ancestors
    logging.debug(f"_get_ancestors_map: Starting for {start_id_norm}")
    processed_count = 0
    while queue:
        current_id, depth = queue.popleft()
        ancestors[current_id] = depth
        processed_count += 1
        current_indi = find_individual_by_id(reader, current_id)
        if not current_indi:
            logging.warning(
                f"Ancestor map: Could not find individual for ID {current_id}"
            )
            continue
        father = getattr(current_indi, "father", None)
        mother = getattr(current_indi, "mother", None)
        parents = [father, mother]
        for parent_indi in parents:
            if (
                parent_indi is not None
                and _is_name(getattr(parent_indi, "name", None))
                and hasattr(parent_indi, "xref_id")
                and parent_indi.xref_id
            ):
                parent_id = _normalize_id(parent_indi.xref_id)
                if parent_id and parent_id not in visited:
                    visited.add(parent_id)
                    queue.append((parent_id, depth + 1))
    logging.debug(
        f"_get_ancestors_map for {start_id_norm} finished. Processed {processed_count} individuals. Found {len(ancestors)} ancestors (incl. self)."
    )
    return ancestors


def _build_relationship_path_str(reader, start_id_norm, end_id_norm):
    """Builds shortest ancestral path string from start up to end using BFS."""
    if not reader or not start_id_norm or not end_id_norm:
        logging.error("_build_relationship_path_str: Invalid input.")
        return []
    logging.debug(
        f"_build_relationship_path_str: Building path {start_id_norm} -> {end_id_norm}"
    )
    queue = deque(
        [(start_id_norm, [])]
    )  # Queue stores (current_id, path_list_of_names)
    visited = {start_id_norm}
    processed_count = 0
    while queue:
        current_id, current_path_names = queue.popleft()
        processed_count += 1
        current_indi = find_individual_by_id(reader, current_id)
        current_name = (
            _get_full_name(current_indi)
            if current_indi
            else f"Unknown/Error ({current_id})"
        )
        current_full_path = current_path_names + [current_name]
        if current_id == end_id_norm:
            logging.debug(
                f"Path build: Reached target {end_id_norm} after checking {processed_count} nodes. Path found."
            )
            return current_full_path
        if current_indi:
            father = getattr(current_indi, "father", None)
            mother = getattr(current_indi, "mother", None)
            parents = [father, mother]
            for parent_indi in parents:
                if (
                    parent_indi is not None
                    and _is_name(getattr(parent_indi, "name", None))
                    and hasattr(parent_indi, "xref_id")
                    and parent_indi.xref_id
                ):
                    parent_id = _normalize_id(parent_indi.xref_id)
                    if parent_id and parent_id not in visited:
                        visited.add(parent_id)
                        queue.append((parent_id, current_full_path))
    logging.warning(
        f"_build_relationship_path_str: Could not find ANCESTRAL path from {start_id_norm} to {end_id_norm} after checking {processed_count} nodes."
    )
    return []  # Return empty list if no path found


def _find_lca_from_maps(
    ancestors1: Dict[str, int], ancestors2: Dict[str, int]
) -> Optional[str]:
    """Finds Lowest Common Ancestor (LCA) ID from two ancestor maps."""
    if not ancestors1 or not ancestors2:
        return None
    common_ancestor_ids = set(ancestors1.keys()) & set(ancestors2.keys())
    if not common_ancestor_ids:
        logging.debug("_find_lca_from_maps: No common ancestors found.")
        return None
    lca_candidates: Dict[str, Union[int, float]] = {
        cid: ancestors1.get(cid, float("inf")) + ancestors2.get(cid, float("inf"))
        for cid in common_ancestor_ids
    }
    if not lca_candidates:
        return None
    lca_id = min(lca_candidates.keys(), key=lambda k: lca_candidates[k])
    logging.debug(
        f"_find_lca_from_maps: LCA ID: {lca_id} (Depth Sum: {lca_candidates[lca_id]})"
    )
    return lca_id


def _reconstruct_path(start_id, end_id, meeting_id, visited_fwd, visited_bwd):
    """Reconstructs the path from start to end via the meeting point using predecessor maps."""
    path_fwd = []
    curr = meeting_id
    while curr is not None:
        path_fwd.append(curr)
        curr = visited_fwd.get(curr)
    path_fwd.reverse()  # Reverse to get path from start to meeting point

    path_bwd = []
    curr = visited_bwd.get(
        meeting_id
    )  # Start from the predecessor of the meeting point in the backward search
    while curr is not None:
        path_bwd.append(curr)
        curr = visited_bwd.get(curr)

    path = path_fwd + path_bwd

    if not path:
        logging.error("_reconstruct_path: Failed to reconstruct any path.")
        return []
    if path[0] != start_id:
        logging.warning(
            f"_reconstruct_path: Path doesn't start with start_id ({path[0]} != {start_id}). Prepending."
        )
        path.insert(0, start_id)  # Attempt fix
    if path[-1] != end_id:
        logging.warning(
            f"_reconstruct_path: Path doesn't end with end_id ({path[-1]} != {end_id}). Appending."
        )
        path.append(end_id)  # Attempt fix

    logging.debug(f"_reconstruct_path: Final reconstructed path IDs: {path}")
    return path


def explain_relationship_path(path_ids, reader, id_to_parents, id_to_children):
    """Return a human-readable explanation of the relationship path with relationship labels."""
    if not path_ids or len(path_ids) < 2:
        return "(No relationship path explanation available)"
    steps = []
    for i in range(len(path_ids) - 1):
        id_a, id_b = path_ids[i], path_ids[i + 1]
        indi_a = find_individual_by_id(reader, id_a)
        indi_b = find_individual_by_id(reader, id_b)
        name_a = _get_full_name(indi_a) if indi_a else f"Unknown ({id_a})"
        name_b = _get_full_name(indi_b) if indi_b else f"Unknown ({id_b})"
        rel = "related"
        label = rel
        if id_b in id_to_parents.get(id_a, set()):
            rel = "child"
            sex_a = getattr(indi_a, "sex", None) if indi_a else None
            label = "daughter" if sex_a == "F" else "son" if sex_a == "M" else "child"
        elif id_a in id_to_parents.get(id_b, set()):
            rel = "parent"
            sex_b = getattr(indi_b, "sex", None) if indi_b else None
            label = "mother" if sex_b == "F" else "father" if sex_b == "M" else "parent"
        elif id_b in id_to_children.get(id_a, set()):
            rel = "parent"
            label = "parent"
        elif id_a in id_to_children.get(id_b, set()):
            rel = "child"
            label = "child"
        steps.append(f"{name_a} is the {label} of {name_b}")
    return " -> ".join(steps)


def fast_bidirectional_bfs(
    start_id,
    end_id,
    id_to_parents,
    id_to_children,
    max_depth=20,
    node_limit=100000,
    timeout_sec=30,
    log_progress=False,
):
    """Performs bidirectional BFS using maps & predecessors. Returns path as list of IDs."""
    from time import time as timer
    start_time = timer()
    if start_id == end_id:
        return [start_id]
    visited_fwd = {start_id: None}
    visited_bwd = {end_id: None}
    queue_fwd = deque([start_id])
    queue_bwd = deque([end_id])
    found = False
    meeting_id = None
    processed = 0
    while queue_fwd and queue_bwd and not found and processed < node_limit:
        processed += 1
        if len(queue_fwd) <= len(queue_bwd):
            curr = queue_fwd.popleft()
            neighbors = id_to_parents.get(curr, set()) | id_to_children.get(curr, set())
            for neighbor in neighbors:
                if neighbor not in visited_fwd:
                    visited_fwd[neighbor] = curr
                    queue_fwd.append(neighbor)
                if neighbor in visited_bwd:
                    found = True
                    meeting_id = neighbor
                    break
        else:
            curr = queue_bwd.popleft()
            neighbors = id_to_parents.get(curr, set()) | id_to_children.get(curr, set())
            for neighbor in neighbors:
                if neighbor not in visited_bwd:
                    visited_bwd[neighbor] = curr
                    queue_bwd.append(neighbor)
                if neighbor in visited_fwd:
                    found = True
                    meeting_id = neighbor
                    break
        if timer() - start_time > timeout_sec:
            logging.warning("Bidirectional BFS timed out.")
            break
    if found and meeting_id:
        return _reconstruct_path(start_id, end_id, meeting_id, visited_fwd, visited_bwd)
    else:
        logging.warning(
            f"Bidirectional BFS: No path found between {start_id} and {end_id} after processing {processed} nodes."
        )
        return []


def get_relationship_path(reader, id1: str, id2: str):
    """Calculates and formats relationship path using fast bidirectional BFS with pre-built maps."""
    # Build parent and child maps using low-level GEDCOM record API
    id_to_parents = {}
    id_to_children = {}
    # Build a map from family xref_id to family record for quick lookup
    fam_records = {fam.xref_id: fam for fam in reader.records0("FAM") if hasattr(fam, "xref_id")}
    for indi in reader.records0("INDI"):
        indi_id = _normalize_id(getattr(indi, "xref_id", None))
        if not indi_id:
            continue
        parents = set()
        children = set()
        # Find parent families (FAMC tags)
        indi_sub_tags = indi.sub_tags() if callable(getattr(indi, "sub_tags", None)) else getattr(indi, "sub_tags", [])
        famc_tags = [famc for famc in indi_sub_tags if getattr(famc, "tag", "") == "FAMC"]
        for famc in famc_tags:
            fam_id = _normalize_id(famc.value)
            fam = fam_records.get(fam_id)
            if fam:
                # Add parents from this family
                fam_sub_tags = fam.sub_tags() if callable(getattr(fam, "sub_tags", None)) else getattr(fam, "sub_tags", [])
                for sub in fam_sub_tags:
                    if getattr(sub, "tag", "").upper() == "HUSB" and getattr(sub, "value", None):
                        father = _normalize_id(getattr(sub, "value", None))
                        if father:
                            parents.add(father)
                    elif getattr(sub, "tag", "").upper() == "WIFE" and getattr(sub, "value", None):
                        mother = _normalize_id(getattr(sub, "value", None))
                        if mother:
                            parents.add(mother)
        # Find spouse/parent families (FAMS tags)
        fams_tags = [fams for fams in indi_sub_tags if getattr(fams, "tag", "") == "FAMS"]
        for fams in fams_tags:
            fam_id = _normalize_id(fams.value)
            fam = fam_records.get(fam_id)
            if fam:
                # Add children from this family
                fam_sub_tags = fam.sub_tags() if callable(getattr(fam, "sub_tags", None)) else getattr(fam, "sub_tags", [])
                for sub in fam_sub_tags:
                    if getattr(sub, "tag", "").upper() == "CHIL" and getattr(sub, "value", None):
                        children.add(_normalize_id(getattr(sub, "value", None)))
        id_to_parents[indi_id] = {pid for pid in parents if pid}
        id_to_children[indi_id] = {cid for cid in children if cid}
    path_ids = fast_bidirectional_bfs(id1, id2, id_to_parents, id_to_children)
    explanation = explain_relationship_path(path_ids, reader, id_to_parents, id_to_children)
    return path_ids, explanation


def find_potential_matches(
    reader,
    first_name: Optional[str],
    surname: Optional[str],
    dob_str: Optional[str],  # Birth date string
    pob: Optional[str],  # Birth place
    dod_str: Optional[str],  # Death date string (NEW)
    pod: Optional[str],  # Death place (NEW)
    gender: Optional[str] = None,
):
    """
    Finds potential matches in GEDCOM based on various criteria including death info.
    Prioritizes name matches.
    """
    matches = []
    for indi in reader.records0("INDI"):
        indi_full_name = _get_full_name(indi)
        birth_date_str_ged = None
        birth_place_str_ged = None
        death_date_str_ged = None
        death_place_str_ged = None
        # Extract birth/death info from events (if present)
        for event in getattr(indi, "events", []):
            etype = getattr(event, "tag", "").lower()
            if etype == "birth":
                birth_date_str_ged = getattr(event, "date", None)
                birth_place_str_ged = getattr(event, "place", None)
            elif etype == "death":
                death_date_str_ged = getattr(event, "date", None)
                death_place_str_ged = getattr(event, "place", None)
        # Robust extraction: also check sub_tags for BIRT
        indi_sub_tags = indi.sub_tags() if callable(getattr(indi, "sub_tags", None)) else getattr(indi, "sub_tags", [])
        for sub in indi_sub_tags:
            if getattr(sub, "tag", "").upper() == "BIRT":
                bsub_tags = sub.sub_tags() if callable(getattr(sub, "sub_tags", None)) else getattr(sub, "sub_tags", [])
                for bsub in bsub_tags:
                    if getattr(bsub, "tag", "").upper() == "DATE" and not birth_date_str_ged:
                        birth_date_str_ged = getattr(bsub, "value", None)
                    if getattr(bsub, "tag", "").upper() == "PLAC" and not birth_place_str_ged:
                        birth_place_str_ged = getattr(bsub, "value", None)
        score = 0
        reasons = []
        if first_name and first_name.lower() in indi_full_name.lower():
            score += 10
            reasons.append("First name match")
        if surname and surname.lower() in indi_full_name.lower():
            score += 10
            reasons.append("Surname match")
        if dob_str and birth_date_str_ged and dob_str in str(birth_date_str_ged):
            score += 8
            reasons.append(f"Birth date match ({birth_date_str_ged})")
        if pob and birth_place_str_ged and pob.lower() in str(birth_place_str_ged).lower():
            score += 4
            reasons.append(f"Place of birth match ({birth_place_str_ged})")
        if dod_str and death_date_str_ged and dod_str in str(death_date_str_ged):
            score += 4
            reasons.append(f"Death date match ({death_date_str_ged})")
        if pod and death_place_str_ged and pod.lower() in str(death_place_str_ged).lower():
            score += 2
            reasons.append(f"Place of death match ({death_place_str_ged})")
        matches.append(
            {
                "id": getattr(indi, "xref_id", None),
                "name": indi_full_name,
                "birth_date": birth_date_str_ged,
                "birth_place": birth_place_str_ged,
                "death_date": death_date_str_ged,
                "death_place": death_place_str_ged,
                "score": score,
                "reasons": ", ".join(reasons),
            }
        )
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches


def display_gedcom_family_details(reader, individual):
    """Display full family details using robust logic from temp.py (parents, siblings, spouses, children)."""
    if not reader or not _is_individual(individual):
        print("  Error: Cannot display GEDCOM details for invalid input.")
        return

    indi_name = _get_full_name(individual)
    print(f"Name: {indi_name}")
    birth_info, death_info = format_full_life_details(individual)
    print(birth_info)
    if death_info:
        print(death_info)

    # Parents
    print("\nParents:")
    parents = get_related_individuals(reader, individual, "parents")
    if parents:
        for p in parents:
            print(f"  - {format_relative_info(p)}")
    else:
        print("  (None found)")

    # Siblings
    print("\nSiblings:")
    siblings = get_related_individuals(reader, individual, "siblings")
    if siblings:
        for s in siblings:
            print(f"  - {format_relative_info(s)}")
    else:
        print("  (None found)")

    # Spouses
    print("\nSpouse(s):")
    spouses = get_related_individuals(reader, individual, "spouses")
    if spouses:
        for s in spouses:
            print(f"  - {format_relative_info(s)}")
    else:
        print("  (None found)")

    # Children
    print("\nChildren:")
    children = get_related_individuals(reader, individual, "children")
    if children:
        for c in children:
            print(f"  - {format_relative_info(c)}")
    else:
        print("  (None found)")


import utils


def run_action10():
    if not GEDCOM_LIB_AVAILABLE or GedcomReader is None:
        print("ged4py library unavailable. Cannot run GEDCOM report.")
        return
    gedcom_path_str = getattr(config_instance, "GEDCOM_FILE_PATH", None)
    if not gedcom_path_str:
        print("GEDCOM_FILE_PATH not set in config.")
        return
    gedcom_path = Path(gedcom_path_str)
    print(f"[DEBUG] GEDCOM file path being loaded: {gedcom_path}")
    if not gedcom_path.is_file():
        print(f"GEDCOM not found: {gedcom_path}")
        return
    reader = GedcomReader(str(gedcom_path))
    build_indi_index(reader)
    print(f"[DEBUG] Number of individuals indexed: {len(utils.INDI_INDEX)}")
    wgg_search_name_lower = "wayne gordon gault"
    wayne_gault_indi = None
    wayne_gault_id_gedcom = None
    for indi in utils.INDI_INDEX.values():
        name_rec = indi.name
        name_str = name_rec.format() if _is_name(name_rec) else ""
        if name_str.lower() == wgg_search_name_lower:
            wayne_gault_indi = indi
            wayne_gault_id_gedcom = _normalize_id(indi.xref_id)
            break
    print(f"[DEBUG] Wayne Gordon Gault ID: {wayne_gault_id_gedcom}")
    print(f"[DEBUG] Wayne Gordon Gault Name: {wgg_search_name_lower}")
    if not wayne_gault_indi or not wayne_gault_id_gedcom:
        print(
            "Wayne Gordon Gault (reference person) not found in local GEDCOM. Cannot calculate relationships."
        )
        return
    print("\nEnter as many details as you know. Leave blank to skip a field.")
    first_name = input("First name: ").strip() or None
    surname = input("Surname (or maiden name): ").strip() or None
    dob_str = input("Date of birth (YYYY-MM-DD or year): ").strip() or None
    pob = input("Place of birth: ").strip() or None
    gender = input("Gender (M/F, optional): ").strip().lower() or None
    if gender and gender not in ("m", "f"):
        gender = None
    dod_str = input("Date of death (YYYY-MM-DD or year, optional): ").strip() or None
    pod = input("Place of death (optional): ").strip() or None
    matches = find_potential_matches(
        reader,
        first_name,
        surname,
        dob_str,
        pob,
        dod_str,
        pod,
        gender,
    )
    filtered_matches = [m for m in matches if m["birth_date"] or m["birth_place"]]
    if not filtered_matches:
        print("No matches with birth info found")
        return
    print(f"\nUsing top match: {filtered_matches[0]['name']}")
    selected_id = extract_and_fix_id(filtered_matches[0]["id"])
    if not selected_id:
        print("ERROR: Invalid ID in selected match.")
        return
    selected_indi = find_individual_by_id(reader, selected_id)
    if not selected_indi:
        print("ERROR: Could not retrieve individual record from GEDCOM.")
        return
    print("\n=== PERSON DETAILS ===")
    print(f"Name: {filtered_matches[0]['name']}")
    print(f"Gender: {gender.upper() if gender else 'N/A'}")
    print(f"Birth : {filtered_matches[0]['birth_date']} in {filtered_matches[0]['birth_place']}")
    if filtered_matches[0]["death_date"] or filtered_matches[0]["death_place"]:
        print(
            f"Death : {filtered_matches[0]['death_date']} in {filtered_matches[0]['death_place']}"
        )
    tree_id = os.getenv("MY_TREE_ID")
    ancestry_id = (
        selected_id[1:]
        if selected_id
        and selected_id[0] in ("I", "F", "S", "T", "N", "M", "C", "X", "O")
        else selected_id
    )
    if tree_id and ancestry_id:
        print(
            f"Link in Tree: https://www.ancestry.co.uk/family-tree/person/tree/{tree_id}/person/{ancestry_id}/facts"
        )
    else:
        print(f"Link in Tree: (unavailable)")
    display_gedcom_family_details(reader, selected_indi)
    path_ids, explanation = get_relationship_path(reader, selected_id, wayne_gault_id_gedcom)
    print("\nRelationship Path:\n")
    print(explanation.strip())


if __name__ == "__main__":
    run_action10()
