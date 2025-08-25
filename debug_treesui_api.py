#!/usr/bin/env python3

"""
Debug script to examine the TreesUI API response structure.
"""

import sys

# Add current directory to path for imports
from pathlib import Path

current_dir = str(Path(__file__).resolve().parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config import config_schema
from core.session_manager import SessionManager
from core_imports import *
from standard_imports import setup_module

logger = setup_module(globals(), __name__)

def _build_treesui_url(session_manager: SessionManager) -> str | None:
    tree_id = session_manager.my_tree_id or getattr(config_schema.test, "test_tree_id", "")
    if not tree_id:
        print("❌ No tree ID available")
        return None
    base_url = config_schema.api.base_url.rstrip('/')
    from urllib.parse import quote
    encoded_name = quote("fraser gault")
    endpoint = f"/api/treesui-list/trees/{tree_id}/persons"
    params = "name={}&fields=EVENTS,GENDERS,KINSHIP,NAMES,RELATIONS&isGetFullPersonObject=true".format(encoded_name)
    return f"{base_url}{endpoint}?{params}"


def _request_treesui(url: str, session_manager: SessionManager):
    from utils import _api_req
    return _api_req(
        url=url,
        driver=session_manager.driver,
        session_manager=session_manager,
        method="GET",
        api_description="Debug TreesUI List API",
        headers={"_use_enhanced_headers": "true", "_tree_id": session_manager.my_tree_id or "", "_person_id": "search"},
        timeout=30,
    )


def _analyze_response(response) -> None:
    print("\n📊 API Response Analysis:")
    print(f"Response type: {type(response)}")
    if not response:
        print("❌ No response received")
        return
    if isinstance(response, dict):
        print(f"Response keys: {list(response.keys())}")
        persons = response.get("persons")
        if persons is not None:
            print(f"Persons type: {type(persons)}")
            print(f"Persons count: {len(persons) if isinstance(persons, list) else 'Not a list'}")
            if isinstance(persons, list) and persons:
                first_person = persons[0]
                print("\n🧑 First Person Analysis:")
                print(f"Person type: {type(first_person)}")
                if isinstance(first_person, dict):
                    print(f"Person keys: {list(first_person.keys())}")
                    if "names" in first_person:
                        print(f"Names: {first_person['names']}")
                    if "events" in first_person:
                        events = first_person["events"]
                        print(f"Events count: {len(events) if isinstance(events, list) else 'Not a list'}")
                        if isinstance(events, list) and events:
                            print(f"First event: {events[0]}")
                    if "gender" in first_person:
                        print(f"Gender: {first_person['gender']}")
                    print(f"Person ID: {first_person.get('personId') or first_person.get('id')}")
        data = response.get("data")
        if data is not None:
            print(f"Data type: {type(data)}")
            if isinstance(data, list):
                print(f"Data count: {len(data)}")
    elif isinstance(response, list):
        print(f"Response is a list with {len(response)} items")
        if response:
            print(f"First item type: {type(response[0])}")
            if isinstance(response[0], dict):
                print(f"First item keys: {list(response[0].keys())}")
    else:
        print(f"Response content: {str(response)[:500]}")


def debug_treesui_api() -> None:
    """Debug the TreesUI API response structure."""
    try:
        session_manager = SessionManager()
        if not session_manager.is_sess_valid():
            print("❌ Session not valid - cannot test API")
            return
        url = _build_treesui_url(session_manager)
        if not url:
            return
        print(f"🔍 Testing TreesUI API: {url}")
        response = _request_treesui(url, session_manager)
        _analyze_response(response)
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Debug error: {e}", exc_info=True)

if __name__ == "__main__":
    debug_treesui_api()
