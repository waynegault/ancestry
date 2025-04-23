# action11.py: API Report (Ancestry Online, family, relationship to WGG)
import os
from logging_config import setup_logging
from utils import SessionManager, AncestryAPISearch, initialize_session


def run_action11():
    logger = setup_logging(log_file="gedcom_processor.log", log_level="INFO")
    if not initialize_session():
        print("Failed to initialize session. Cannot proceed with API operations.")
        return
    api_search = AncestryAPISearch(SessionManager())
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
    def score_api_candidate(person):
        score = 0
        reasons = []
        display_name = api_search._extract_display_name(person).lower()
        if first_name and first_name.lower() in display_name:
            score += 10
            reasons.append("First name match")
        if surname and surname.lower() in display_name:
            score += 10
            reasons.append("Surname match")
        api_gender = None
        if "Genders" in person and person["Genders"]:
            api_gender = person["Genders"][0].get("g", "").lower()
        elif "gender" in person:
            api_gender = str(person["gender"]).lower()
        if gender and api_gender and gender == api_gender[0]:
            score += 5
            reasons.append(f"Gender match ({api_gender})")
        elif gender and api_gender and gender != api_gender[0]:
            score -= 5
            reasons.append(f"Gender mismatch ({api_gender})")
        events = person.get("Events") or person.get("events") or []
        birth_date = None
        birth_place = None
        for event in events:
            if (
                event.get("t", "").lower() == "birth"
                or event.get("type", "").lower() == "birth"
            ):
                birth_date = event.get("d") or event.get("date") or event.get("nd")
                birth_place = (
                    event.get("p")
                    or event.get("place")
                    or (event.get("pl") if "pl" in event else None)
                )
                if isinstance(birth_place, dict):
                    birth_place = birth_place.get("v") or birth_place.get("name")
                if not birth_place:
                    birth_place = "?"
                break
        if dob_str and birth_date and dob_str in str(birth_date):
            score += 8
            reasons.append(f"Birth date match ({birth_date})")
        if pob and birth_place and pob.lower() in birth_place.lower():
            score += 4
            reasons.append(f"Place of birth match ({birth_place})")
        death_date = None
        death_place = None
        for event in events:
            if (
                event.get("t", "").lower() == "death"
                or event.get("type", "").lower() == "death"
            ):
                death_date = event.get("d") or event.get("date") or event.get("nd")
                death_place = event.get("p") or event.get("place")
                break
        if dod_str and death_date and dod_str in str(death_date):
            score += 4
            reasons.append(f"Death date match ({death_date})")
        if pod and death_place and pod.lower() in death_place.lower():
            score += 2
            reasons.append(f"Place of death match ({death_place})")
        return score, ", ".join(reasons)

    scored = []
    for person in persons:
        score, reasons = score_api_candidate(person)
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
    def fetch_facts_json(profile_id, tree_id, person_id):
        from utils import _api_req

        url = f"https://www.ancestry.co.uk/family-tree/person/facts/user/{profile_id}/tree/{tree_id}/person/{person_id}"
        try:
            return _api_req(
                url=url,
                driver=api_search.session_manager.driver,
                session_manager=api_search.session_manager,
                method="GET",
                headers=None,
                use_csrf_token=False,
                api_description="Ancestry Facts JSON Endpoint",
                referer_url=f"https://www.ancestry.co.uk/family-tree/tree/{tree_id}/family",
                timeout=20,
            )
        except Exception:
            return None

    profile_id = (
        getattr(api_search.session_manager, "my_profile_id", None)
        or "07bdd45e-0006-0000-0000-000000000000"
    )
    tree_id = api_search._get_tree_id()
    person_id = ancestry_id
    facts_json = None
    if profile_id and tree_id and person_id:
        facts_json = fetch_facts_json(profile_id, tree_id, person_id)
    if facts_json and isinstance(facts_json, dict):
        family = facts_json.get("family") or facts_json.get("Family")

        def print_family_section(label, people):
            print(f"\n{label}:")
            if people:
                for rel in people:
                    name = (
                        rel.get("displayName")
                        or rel.get("fullName")
                        or rel.get("name")
                        or rel.get("gname")
                        or rel.get("sname")
                        or rel.get("id")
                        or "(Unknown)"
                    )
                    life = rel.get("birthDate") or rel.get("birth") or ""
                    if rel.get("deathDate") or rel.get("death"):
                        life += f" - {rel.get('deathDate') or rel.get('death')}"
                    print(f"  - {name}{f' ({life})' if life else ''}")
            else:
                print("  (None found)")

        if family:
            print_family_section("Parents", family.get("parents", []))
            print_family_section("Spouse(s)", family.get("spouses", []))
            print_family_section("Children", family.get("children", []))
        else:
            print("  (Family details not available in facts JSON response.)")
    else:
        facts_url = f"https://www.ancestry.co.uk/family-tree/person/facts/user/{profile_id}/tree/{tree_id}/person/{person_id}"
        print(
            "\n(Family details could not be retrieved. You can view full family details in your browser:"
        )
        print(f"  {facts_url}\n")


if __name__ == "__main__":
    run_action11()
