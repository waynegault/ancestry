# action11.py: API Report (Ancestry Online, family, relationship to WGG)
import os
from logging_config import setup_logging
from utils import SessionManager, initialize_session


import urllib.parse
from utils import _api_req

def _extract_display_name(person: dict) -> str:
    names = person.get("Names") or person.get("names")
    if names and isinstance(names, list) and names:
        given = names[0].get("g") or ""
        surname = names[0].get("s") or ""
        full_name = f"{given} {surname}".strip()
        if full_name:
            return full_name
    given = person.get("gname") or ""
    surname = person.get("sname") or ""
    if given or surname:
        return f"{given} {surname}".strip()
    return str(person.get("pid") or person.get("id") or "(Unknown)")

def search_by_name(session_manager, first_name: str = None, last_name: str = None, limit: int = 10) -> list[dict]:
    """
    Standalone Ancestry API person search by first and last name.
    """
    # Get tree_id from session_manager
    tree_id = getattr(session_manager, 'my_tree_id', None)
    if not tree_id:
        # Try to retrieve identifiers if not set
        if hasattr(session_manager, '_retrieve_identifiers'):
            session_manager._retrieve_identifiers()
        tree_id = getattr(session_manager, 'my_tree_id', None)
    if not tree_id:
        return []
    base_url = "https://www.ancestry.co.uk"
    params = []
    if first_name:
        params.append(f"fn={urllib.parse.quote(first_name)}")
    if last_name:
        params.append(f"ln={urllib.parse.quote(last_name)}")
    params.append(f"limit={limit}")
    params.append("fields=GENDERS,KINSHIP,NAMES,EVENTS")
    params.append("isGetFullPersonObject=true")
    query_string = "&".join(params)
    persons_url = f"{base_url}/api/treesui-list/trees/{tree_id}/persons?{query_string}"
    try:
        persons_response = _api_req(
            url=persons_url,
            driver=getattr(session_manager, 'driver', None),
            session_manager=session_manager,
            method="GET",
            headers=None,
            use_csrf_token=False,
            api_description="Ancestry Search by Name (fn/ln)",
            referer_url=f"{base_url}/family-tree/tree/{tree_id}/family",
            timeout=20,
        )
        if not persons_response or not isinstance(persons_response, list):
            return []
        return persons_response
    except Exception:
        return []

def format_person_details(person: dict) -> str:
    name = _extract_display_name(person)
    gender = None
    if "Genders" in person and person["Genders"]:
        gender = person["Genders"][0].get("g")
    elif "gender" in person:
        gender = person["gender"]
    gender_str = f"Gender: {gender}" if gender else "Gender: Unknown"
    birth_info = ""
    events = person.get("Events") or person.get("events") or []
    for event in events:
        if (
            event.get("t", "").lower() == "birth"
            or event.get("type", "").lower() == "birth"
        ):
            date = event.get("d") or event.get("date") or event.get("nd")
            place = event.get("p") or event.get("place")
            birth_info = f"Birth: {date or '?'} in {place or '?'}"
            break
    pid = person.get("pid") or person.get("id") or person.get("gid", {}).get("v")
    pid_str = f"Person ID: {pid}" if pid else ""
    lines = [f"Name: {name}", gender_str]
    if birth_info:
        lines.append(birth_info)
    if pid_str:
        lines.append(pid_str)
    return "\n".join(lines)

def get_relationship_ladder(session_manager, person_id: str):
    import re, json, time
    base_url = "https://www.ancestry.co.uk"
    tree_id = getattr(session_manager, 'my_tree_id', None)
    if not tree_id or not person_id:
        return {"error": "Missing tree_id or person_id."}
    url = f"{base_url}/family-tree/person/tree/{tree_id}/person/{person_id}/getladder?callback=jQuery1234567890_1234567890&_={int(time.time()*1000)}"
    try:
        response = _api_req(
            url=url,
            driver=getattr(session_manager, 'driver', None),
            session_manager=session_manager,
            method="GET",
            headers=None,
            use_csrf_token=False,
            api_description="Ancestry Relationship Ladder",
            referer_url=f"{base_url}/family-tree/tree/{tree_id}/family",
            timeout=20,
        )
        if isinstance(response, str):
            match = re.search(r"\\((\{.*\})\\)", response, re.DOTALL)
            if match:
                json_str = match.group(1)
                first_brace = json_str.find("{")
                last_brace = json_str.rfind("}")
                if (
                    first_brace != -1
                    and last_brace != -1
                    and last_brace > first_brace
                ):
                    json_str = json_str[first_brace : last_brace + 1]
                    json_str = bytes(json_str, "utf-8").decode("unicode_escape")
                    try:
                        ladder_json = json.loads(json_str)
                    except Exception:
                        return {"error": "JSON decode failed."}
                    if isinstance(ladder_json, dict) and "html" in ladder_json:
                        return parse_ancestry_ladder_html(ladder_json["html"])
                    else:
                        return {"error": "No 'html' key in ladder JSON."}
                else:
                    return {
                        "error": "Could not extract JSON object from JSONP response."
                    }
            else:
                return {"error": "Could not parse JSONP response."}
        elif isinstance(response, dict):
            return response
        else:
            return {"error": "Unexpected response type."}
    except Exception as e:
        return {"error": str(e)}

def format_ladder_details(ladder_data) -> str:
    if not ladder_data:
        return "No relationship ladder data available."
    if isinstance(ladder_data, dict):
        if "error" in ladder_data:
            return f"Error: {ladder_data['error']}"
        rel = ladder_data.get("actual_relationship")
        path = ladder_data.get("relationship_path")
        out = []
        if rel:
            out.append(f"Relationship: {rel}")
        if path:
            out.append(f"Path:\n{path}")
        return "\n".join(out) if out else str(ladder_data)
    return str(ladder_data)

def parse_ancestry_ladder_html(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    ladder_data = {}
    rel_elem = soup.select_one(
        "ul.textCenter > li:first-child > i > b"
    ) or soup.select_one("ul.textCenter > li > i > b")
    if rel_elem:
        ladder_data["actual_relationship"] = rel_elem.get_text(strip=True).title()
    path_items = soup.select('ul.textCenter > li:not([class*="iconArrowDown"])')
    path_list = []
    for i, item in enumerate(path_items):
        name_text = ""
        desc_text = ""
        name_container = item.find("a") or item.find("b")
        if name_container:
            name_text = name_container.get_text(strip=True)
        if i > 0:
            desc_element = item.find("i")
            if desc_element:
                desc_text = desc_element.get_text(strip=True)
        if name_text:
            path_list.append(
                f"{name_text} ({desc_text})" if desc_text else name_text
            )
    if path_list:
        ladder_data["relationship_path"] = "\n↓\n".join(path_list)
    return ladder_data

def run_action11(session_manager, config_instance, *args):
    logger = setup_logging(log_file="gedcom_processor.log", log_level="INFO")
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
    persons = search_by_name(session_manager, first_name, surname)
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
