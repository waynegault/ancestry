# action11.py: API Report (Ancestry Online, family, relationship to WGG)
import os
from logging_config import setup_logging
from utils import SessionManager, AncestryAPISearch, initialize_session


def run_action11(session_manager, config_instance, *args):
    logger = setup_logging(log_file="gedcom_processor.log", log_level="INFO")
    api_search = AncestryAPISearch(session_manager)
    print("\n--- Person Details & Relationship to WGG (API) ---")
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
    query = " ".join([x for x in [first_name, surname] if x]).strip()
    if not query:
        print("Search cancelled.")
        return
    persons = api_search.search_by_name(query)
    if not persons:
        print("\nNo matches found in Ancestry API.")
        return

    # Score and filter candidates using additional fields
    from utils import score_api_candidate

    scored = []
    for person in persons:
        score, reasons = score_api_candidate(person, first_name, surname, dob_str, pob, gender, dod_str, pod, api_search)
        scored.append((score, reasons, person))
    scored.sort(reverse=True, key=lambda x: x[0])
    shortlist = scored[:5]
    print(
        f"\nFound {len(shortlist)} potential match{'es' if len(shortlist) != 1 else ''}:"
    )
    for i, (score, reasons, person) in enumerate(shortlist):
        display_name = api_search._extract_display_name(person)
        events = person.get("Events") or person.get("events") or []
        birth_date = birth_place = death_date = death_place = ""
        for event in events:
            if (
                event.get("t", "").lower() == "birth"
                or event.get("type", "").lower() == "birth"
            ):
                birth_date = (
                    event.get("d") or event.get("date") or event.get("nd") or "?"
                )
                birth_place = (
                    event.get("Place") or event.get("place") or event.get("p") or "?"
                )
            if (
                event.get("t", "").lower() == "death"
                or event.get("type", "").lower() == "death"
            ):
                death_date = (
                    event.get("d") or event.get("date") or event.get("nd") or "?"
                )
                death_place = event.get("p") or event.get("place") or "?"
        line = f"  {i+1}. {display_name}\n     Born : {birth_date} in {birth_place}"
        if (death_date and death_date != "?" and death_date.strip()) or (
            death_place and death_place != "?" and death_place.strip()
        ):
            line += f"\n     Died : {death_date} in {death_place}"
        if reasons:
            line += f"\n     Reasons: {reasons}"
        print(line)
    if len(shortlist) == 1 or (
        len(shortlist) > 1 and shortlist[0][0] != shortlist[1][0]
    ):
        selected_person = shortlist[0][2]
        print(f"\nAuto-selected: {api_search._extract_display_name(selected_person)}")
    else:
        try:
            choice = int(input("\nSelect person (or 0 to cancel): "))
            if choice < 1 or choice > len(shortlist):
                print("Selection cancelled or invalid.")
                return
            selected_person = shortlist[choice - 1][2]
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
            return
    print("\n=== PERSON DETAILS ===")
    print(f"Name: {api_search._extract_display_name(selected_person)}")
    api_gender = None
    if "Genders" in selected_person and selected_person["Genders"]:
        api_gender = selected_person["Genders"][0].get("g")
    elif "gender" in selected_person:
        api_gender = selected_person["gender"]
    print(f"Gender: {api_gender.upper() if api_gender else 'N/A'}")
    events = selected_person.get("Events") or selected_person.get("events") or []
    birth_date = birth_place = death_date = death_place = None
    for event in events:
        if (
            event.get("t", "").lower() == "birth"
            or event.get("type", "").lower() == "birth"
        ):
            birth_date = event.get("d") or event.get("date") or event.get("nd")
            birth_place = event.get("p") or event.get("place")
            break
    print(
        f"Birth : {birth_date if birth_date else '?'} in {birth_place if birth_place else '?'}"
    )
    for event in events:
        if (
            event.get("t", "").lower() == "death"
            or event.get("type", "").lower() == "death"
        ):
            death_date = event.get("d") or event.get("date") or event.get("nd")
            death_place = event.get("p") or event.get("place")
            break
    if (death_date and str(death_date).strip() and death_date != "?") or (
        death_place and str(death_place).strip() and death_place != "?"
    ):
        print(
            f"Death : {death_date if death_date else '?'} in {death_place if death_place else '?'}"
        )
    tree_id = api_search._get_tree_id()
    selected_id = (
        selected_person.get("pid")
        or selected_person.get("id")
        or (
            selected_person.get("gid", {}).get("v")
            if isinstance(selected_person.get("gid"), dict)
            else None
        )
    )
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

    # --- Fetch and show family details (parents, spouses, children) if available ---
    from utils import fetch_facts_json

    profile_id = (
        getattr(api_search.session_manager, "my_profile_id", None)
        or "07bdd45e-0006-0000-0000-000000000000"
    )
    tree_id = api_search._get_tree_id()
    person_id = ancestry_id
    facts_json = None
    if profile_id and tree_id and person_id:
        facts_json = fetch_facts_json(profile_id, tree_id, person_id, session_manager, session_manager.driver)
    if facts_json and isinstance(facts_json, dict):
        family = facts_json.get("family") or facts_json.get("Family")

        from utils import print_family_section

        if family:
            print_family_section("Parents", family.get("parents", []))
            print_family_section("Spouse(s)", family.get("spouses", []))
            print_family_section("Children", family.get("children", []))
        else:
            print("  (Family details not available in facts JSON response.)")
    else:
        facts_url = f"https://www.ancestry.co.uk/family-tree/person/tree/{tree_id}/person/{person_id}/facts"
        print(
            "\n(Family details could not be retrieved. You can view full family details in your browser:"
        )
        print(f"  {facts_url}\n")


