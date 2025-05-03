import sys
import os
import json
import requests
from urllib.parse import urljoin, urlencode

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the SessionManager from utils
try:
    from utils import SessionManager
except ImportError:
    print("Could not import SessionManager from utils. Make sure the utils module is available.")
    sys.exit(1)

def test_api_params():
    # Initialize session manager
    print("Initializing Ancestry session...")
    session_manager = SessionManager()
    session_init_ok = session_manager.ensure_session_ready(action_name="API Parameter Test")
    if not session_init_ok:
        print("Failed to initialize session. Check logs.")
        return False

    # Get tree ID and base URL
    tree_id = getattr(session_manager, "my_tree_id", None)
    if not tree_id:
        print("Could not determine tree ID. Check logs.")
        return False
    
    base_url = "https://www.ancestry.co.uk"
    
    # Test different parameter formats
    test_params = [
        # Test 1: Basic search for Gault
        {"partialLastName": "gault", "isHideVeiledRecords": "false"},
        
        # Test 2: Search for Gault with birth year as a parameter
        {"partialLastName": "gault", "birthYear": "1941", "isHideVeiledRecords": "false"},
        
        # Test 3: Search for Gault with birth year range
        {"partialLastName": "gault", "birthYearFrom": "1940", "birthYearTo": "1942", "isHideVeiledRecords": "false"},
        
        # Test 4: Search for Fraser Gault
        {"partialFirstName": "fraser", "partialLastName": "gault", "isHideVeiledRecords": "false"},
        
        # Test 5: Search for Fraser Gault with birth year
        {"partialFirstName": "fraser", "partialLastName": "gault", "birthYear": "1941", "isHideVeiledRecords": "false"},
        
        # Test 6: Try with different parameter names
        {"partialLastName": "gault", "byear": "1941", "isHideVeiledRecords": "false"},
        
        # Test 7: Try with different parameter names
        {"partialLastName": "gault", "birth_year": "1941", "isHideVeiledRecords": "false"},
        
        # Test 8: Try with different parameter names
        {"partialLastName": "gault", "year": "1941", "isHideVeiledRecords": "false"},
    ]
    
    for i, params in enumerate(test_params):
        print(f"\n=== Test {i+1}: {params} ===")
        
        # Construct URL
        query_string = urlencode(params)
        url = f"{base_url}/api/person-picker/suggest/{tree_id}?{query_string}"
        print(f"URL: {url}")
        
        # Set up headers
        headers = {
            "accept": "application/json",
            "accept-language": "en-GB,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "referer": f"{base_url}/family-tree/tree/{tree_id}",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        }
        
        # Make the request
        try:
            # Sync cookies
            session_manager._sync_cookies()
            scraper = session_manager.scraper
            
            # Make request
            response = scraper.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Check if Fraser Gault is in the results
            fraser_found = False
            for j, person in enumerate(data):
                name = person.get("FullName", person.get("Name", ""))
                if not name and "GivenName" in person and "Surname" in person:
                    name = f"{person.get('GivenName', '')} {person.get('Surname', '')}".strip()
                
                birth_year = person.get("BirthYear", "N/A")
                
                if "fraser" in name.lower() and "gault" in name.lower():
                    fraser_found = True
                    print(f"Found Fraser Gault at position {j+1} with birth year {birth_year}")
            
            if not fraser_found:
                print("Fraser Gault not found in results")
            
            # Print total results
            print(f"Total results: {len(data)}")
            
            # Print first few results
            print("First few results:")
            for j, person in enumerate(data[:3]):
                name = person.get("FullName", person.get("Name", ""))
                if not name and "GivenName" in person and "Surname" in person:
                    name = f"{person.get('GivenName', '')} {person.get('Surname', '')}".strip()
                
                birth_year = person.get("BirthYear", "N/A")
                print(f"  {j+1}. {name} (Birth Year: {birth_year})")
            
        except Exception as e:
            print(f"Error: {e}")
    
    return True

if __name__ == "__main__":
    test_api_params()
