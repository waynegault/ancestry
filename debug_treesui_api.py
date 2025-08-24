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

def debug_treesui_api():
    """Debug the TreesUI API response structure."""
    try:
        # Create session manager
        session_manager = SessionManager()

        # Check if session is valid
        if not session_manager.is_sess_valid():
            print("❌ Session not valid - cannot test API")
            return

        # Get configuration
        tree_id = session_manager.my_tree_id
        if not tree_id:
            tree_id = getattr(config_schema.test, "test_tree_id", "")
            if not tree_id:
                print("❌ No tree ID available")
                return

        base_url = config_schema.api.base_url.rstrip('/')

        # Build the TreesUI endpoint URL
        from urllib.parse import quote
        full_name = "fraser gault"
        encoded_name = quote(full_name)

        # Construct the new TreesUI endpoint
        endpoint = f"/api/treesui-list/trees/{tree_id}/persons"
        params = f"name={encoded_name}&fields=EVENTS,GENDERS,KINSHIP,NAMES,RELATIONS&isGetFullPersonObject=true"
        url = f"{base_url}{endpoint}?{params}"

        print(f"🔍 Testing TreesUI API: {url}")

        # Make the API request
        from utils import _api_req
        response = _api_req(
            url=url,
            driver=session_manager.driver,
            session_manager=session_manager,
            method="GET",
            api_description="Debug TreesUI List API",
            headers={
                "_use_enhanced_headers": "true",
                "_tree_id": tree_id,
                "_person_id": "search"
            },
            timeout=30
        )

        print("\n📊 API Response Analysis:")
        print(f"Response type: {type(response)}")

        if response:
            if isinstance(response, dict):
                print(f"Response keys: {list(response.keys())}")

                # Look for persons data
                if "persons" in response:
                    persons = response["persons"]
                    print(f"Persons type: {type(persons)}")
                    print(f"Persons count: {len(persons) if isinstance(persons, list) else 'Not a list'}")

                    if isinstance(persons, list) and len(persons) > 0:
                        first_person = persons[0]
                        print("\n🧑 First Person Analysis:")
                        print(f"Person type: {type(first_person)}")
                        if isinstance(first_person, dict):
                            print(f"Person keys: {list(first_person.keys())}")

                            # Check for names
                            if "names" in first_person:
                                names = first_person["names"]
                                print(f"Names: {names}")

                            # Check for events
                            if "events" in first_person:
                                events = first_person["events"]
                                print(f"Events count: {len(events) if isinstance(events, list) else 'Not a list'}")
                                if isinstance(events, list) and len(events) > 0:
                                    print(f"First event: {events[0]}")

                            # Check for gender
                            if "gender" in first_person:
                                gender = first_person["gender"]
                                print(f"Gender: {gender}")

                            # Check for person ID
                            person_id = first_person.get("personId") or first_person.get("id")
                            print(f"Person ID: {person_id}")
                        else:
                            print(f"First person is not a dict: {first_person}")
                else:
                    print("No 'persons' key in response")

                # Check for other possible data structures
                if "data" in response:
                    data = response["data"]
                    print(f"Data type: {type(data)}")
                    if isinstance(data, list):
                        print(f"Data count: {len(data)}")

            elif isinstance(response, list):
                print(f"Response is a list with {len(response)} items")
                if len(response) > 0:
                    print(f"First item type: {type(response[0])}")
                    if isinstance(response[0], dict):
                        print(f"First item keys: {list(response[0].keys())}")
            else:
                print(f"Response content: {str(response)[:500]}")
        else:
            print("❌ No response received")

    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Debug error: {e}", exc_info=True)

if __name__ == "__main__":
    debug_treesui_api()
