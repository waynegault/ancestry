"""
Test script: Inspect Edit Relationships API shape and validate extraction location.

Usage:
  python scripts/test_editrelationships_shape.py

Notes:
- This script uses the single global session registered by main.py (no local SessionManager creation).
- It then calls the Edit Relationships API and prints the structure of the payload,
  focusing on where fathers/mothers/spouses/children live.
- Historically, the arrays are under parsed_json['person'] (NOT top-level and NOT 'res').
- If no global session is registered, run main.py first to register and authenticate it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

# Ensure repository root is on sys.path (pathlib-based)
REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api_utils import call_edit_relationships_api
from session_utils import get_authenticated_session, get_global_session

# ---- Test subject (Fraser Gault) ----
PERSON_ID = "102281560744"
TREE_ID = "175946702"
OWNER_PROFILE_ID = "07bdd45e-0006-0000-0000-000000000000"


def _short(s: str, n: int = 120) -> str:
    return s if len(s) <= n else s[:n] + "..."


def main() -> int:
    exit_code = 0
    print("[TEST] Using global session (registered by main.py)...")

    if get_global_session() is None:
        print("[ERROR] No global session registered. Please run main.py to register the global session, then re-run this script.")
        exit_code = 2
    else:
        try:
            sm, uuid = get_authenticated_session(action_name="Test EditRelationships Shape")
            print(f"[TEST] Session authenticated. UUID={uuid}")
        except Exception as e:
            print(f"[ERROR] Could not authenticate session: {e}")
            exit_code = 2

    if exit_code != 0:
        print("[TEST] DONE (errors occurred before API call)")
        return exit_code

    print("[TEST] Calling Edit Relationships API...")
    resp = call_edit_relationships_api(
        session_manager=sm,
        user_id=OWNER_PROFILE_ID,
        tree_id=TREE_ID,
        person_id=PERSON_ID,
    )
    if not resp or not isinstance(resp, dict):
        print("[ERROR] No/invalid response from Edit Relationships API")
        exit_code = 3
    else:
        print("[TEST] Top-level keys:", list(resp.keys()))

        # Newer responses deliver: { cssBundleUrl, jsBundleUrl, data: JSON_STRING }
        payload: dict[str, Any] | None = None
        if isinstance(resp.get("data"), str):
            try:
                payload = cast(dict[str, Any], json.loads(resp["data"]))
                print("[TEST] Parsed 'data' field (JSON string) to dict. Keys:", list(payload.keys()))
            except json.JSONDecodeError as je:
                print("[ERROR] Failed to parse 'data' JSON string:", je)
                print("[DEBUG] First 200 of data:", _short(resp.get("data", "")[:200]))
                exit_code = 4
        elif isinstance(resp.get("res"), dict):
            payload = cast(dict[str, Any], resp["res"])  # clarify type for pylance
            print("[TEST] Using 'res' dict. Keys:", list(payload.keys()))
        else:
            print("[ERROR] Neither 'data' (string) nor 'res' (dict) present in usable form.")
            print("[DEBUG] Response keys:", list(resp.keys()))
            exit_code = 5

        # Where are the relationship arrays?
        if exit_code == 0 and not isinstance(payload, dict):
            print("[ERROR] Parsed payload is not a dict.")
            exit_code = 6

        if exit_code == 0:
            assert isinstance(payload, dict)
            # Common current shape: payload['person'] contains the arrays
            person_section = payload.get("person") if isinstance(payload, dict) else None
            if isinstance(person_section, dict):
                print("[TEST] person.keys():", list(person_section.keys()))
                for k in ("fathers", "mothers", "spouses", "children", "targetPerson"):
                    v = person_section.get(k)
                    if isinstance(v, list):
                        print(f"  - {k}: list with {len(v)} items")
                    elif isinstance(v, dict):
                        print(f"  - {k}: dict with {len(v.keys())} keys")
                    else:
                        print(f"  - {k}: {type(v).__name__}")
            else:
                print("[WARN] payload['person'] missing or not a dict. Checking top-level...")
                for k in ("fathers", "mothers", "spouses", "children"):
                    v = payload.get(k)
                    if isinstance(v, list):
                        print(f"  - {k}: list with {len(v)} items (TOP-LEVEL)")
                    else:
                        print(f"  - {k}: not present at top-level ({type(v).__name__})")

            # Quick sample dump of names
            def _name_from_rel(p: dict[str, Any]) -> str:
                name = p.get("name")
                if isinstance(name, dict):
                    given = name.get("given", "")
                    surname = name.get("surname", "")
                    return (given + " " + surname).strip()
                return str(name or "Unknown")

            def _dump_people(label: str, items: Any, limit: int = 3) -> None:
                if not isinstance(items, list):
                    print(f"  [DEBUG] {label}: not a list")
                    return
                print(f"[TEST] {label} (first {limit}) →", [ _name_from_rel(x) for x in items[:limit] if isinstance(x, dict) ])

            section = person_section if isinstance(person_section, dict) else payload
            fathers = section.get("fathers", []) if isinstance(section, dict) else []
            mothers = section.get("mothers", []) if isinstance(section, dict) else []
            spouses = section.get("spouses", []) if isinstance(section, dict) else []
            children = section.get("children", []) if isinstance(section, dict) else []

            # Flatten children if nested lists
            flat_children: list[dict[str, Any]] = []
            if isinstance(children, list):
                for it in children:
                    if isinstance(it, list):
                        flat_children.extend([c for c in it if isinstance(c, dict)])
                    elif isinstance(it, dict):
                        flat_children.append(it)

            print("[TEST] Counts → fathers:", len(fathers), "mothers:", len(mothers), "spouses:", len(spouses), "children:", len(flat_children))
            _dump_people("Fathers", fathers)
            _dump_people("Mothers", mothers)
            _dump_people("Spouses", spouses)
            _dump_people("Children", flat_children)

    print("[TEST] DONE")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

