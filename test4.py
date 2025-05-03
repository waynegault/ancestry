import sys
import os
import time
import json
from urllib.parse import urljoin, urlencode
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the necessary functions from utils
try:
    from utils import SessionManager
except ImportError:
    logger.error("Could not import SessionManager from utils. Make sure the utils module is available.")
    sys.exit(1)

def test_api_endpoints():
    """Test different API endpoint formats for incorporating birth year."""
    
    # Initialize session manager
    logger.info("Initializing Ancestry session...")
    session_manager = SessionManager()
    session_init_ok = session_manager.ensure_session_ready(action_name="API Endpoint Test")
    if not session_init_ok:
        logger.error("Failed to initialize session.")
        return False
    
    # Get base URL and tree ID
    base_url = getattr(session_manager, "base_url", "https://www.ancestry.co.uk")
    tree_id = getattr(session_manager, "my_tree_id", None)
    if not tree_id:
        logger.error("Could not determine tree ID.")
        return False
    
    # Get owner profile ID for referer
    owner_profile_id = getattr(session_manager, "my_profile_id", None)
    owner_tree_id = getattr(session_manager, "my_tree_id", None)
    
    # Construct referer URL
    owner_facts_referer = None
    if owner_tree_id and owner_profile_id:
        owner_facts_referer = urljoin(
            base_url,
            f"/family-tree/tree/{owner_tree_id}/person/{owner_profile_id}/facts",
        )
    else:
        logger.warning("Cannot construct owner facts referer: Tree ID or Profile ID missing.")
        owner_facts_referer = base_url  # Fallback
    
    # Test different API endpoints and parameter formats
    test_cases = [
        # Test 1: Original URL from the question
        {
            "description": "Original URL from the question",
            "url": f"{base_url}/api/treesui-list/trees/{tree_id}/persons",
            "params": {"fn": "", "ln": "gault"},
        },
        # Test 2: Original URL with birth year parameter
        {
            "description": "Original URL with birth year parameter",
            "url": f"{base_url}/api/treesui-list/trees/{tree_id}/persons",
            "params": {"fn": "", "ln": "gault", "by": "1941"},
        },
        # Test 3: Original URL with birth year parameter (alternate format)
        {
            "description": "Original URL with birth year parameter (alternate format)",
            "url": f"{base_url}/api/treesui-list/trees/{tree_id}/persons",
            "params": {"fn": "", "ln": "gault", "birthyear": "1941"},
        },
        # Test 4: Original URL with birth year range
        {
            "description": "Original URL with birth year range",
            "url": f"{base_url}/api/treesui-list/trees/{tree_id}/persons",
            "params": {"fn": "", "ln": "gault", "bymin": "1940", "bymax": "1942"},
        },
        # Test 5: Original URL with birth year range (alternate format)
        {
            "description": "Original URL with birth year range (alternate format)",
            "url": f"{base_url}/api/treesui-list/trees/{tree_id}/persons",
            "params": {"fn": "", "ln": "gault", "birthyearmin": "1940", "birthyearmax": "1942"},
        },
        # Test 6: Original URL with first name and birth year
        {
            "description": "Original URL with first name and birth year",
            "url": f"{base_url}/api/treesui-list/trees/{tree_id}/persons",
            "params": {"fn": "fraser", "ln": "gault", "by": "1941"},
        },
    ]
    
    # Run the tests
    for i, test_case in enumerate(test_cases):
        logger.info(f"\n=== Test {i+1}: {test_case['description']} ===")
        
        # Construct URL with parameters
        query_string = urlencode(test_case['params'])
        url = f"{test_case['url']}?{query_string}"
        logger.info(f"URL: {url}")
        
        # Set up headers
        headers = {
            "accept": "application/json",
            "accept-language": "en-GB,en;q=0.9",
            "cache-control": "no-cache",
            "content-type": "application/json",
            "referer": owner_facts_referer,
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        }
        
        # Make the request using direct scraper access
        try:
            # Sync cookies
            session_manager._sync_cookies()
            scraper = session_manager.scraper
            
            # Add a delay to avoid rate limiting
            time.sleep(1)
            
            # Make request
            logger.info(f"Making request to {url}")
            response = scraper.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse response
            try:
                data = response.json()
                
                # Save the response to a file for inspection
                filename = f"test4_response_{i+1}.json"
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Response saved to {filename}")
                
                # Check if it's a list or dictionary
                if isinstance(data, list):
                    logger.info(f"Response is a list with {len(data)} items")
                    
                    # Check if Fraser Gault is in the results
                    fraser_found = False
                    for j, person in enumerate(data):
                        name = person.get("FullName", person.get("Name", ""))
                        if not name and "GivenName" in person and "Surname" in person:
                            name = f"{person.get('GivenName', '')} {person.get('Surname', '')}".strip()
                        
                        birth_year = person.get("BirthYear", "N/A")
                        
                        if "fraser" in name.lower() and "gault" in name.lower():
                            fraser_found = True
                            logger.info(f"Found Fraser Gault at position {j+1} with birth year {birth_year}")
                            print(f"Found Fraser Gault at position {j+1} with birth year {birth_year}")
                    
                    if not fraser_found:
                        logger.info("Fraser Gault not found in results")
                        print("Fraser Gault not found in results")
                    
                    # Print first few results
                    logger.info("First few results:")
                    print(f"First few results from {test_case['description']}:")
                    for j, person in enumerate(data[:3]):
                        name = person.get("FullName", person.get("Name", ""))
                        if not name and "GivenName" in person and "Surname" in person:
                            name = f"{person.get('GivenName', '')} {person.get('Surname', '')}".strip()
                        
                        birth_year = person.get("BirthYear", "N/A")
                        logger.info(f"  {j+1}. {name} (Birth Year: {birth_year})")
                        print(f"  {j+1}. {name} (Birth Year: {birth_year})")
                else:
                    # It's a dictionary or other format
                    logger.info(f"Response is not a list. Type: {type(data)}")
                    print(f"Response is not a list. Type: {type(data)}")
                    
                    if isinstance(data, dict):
                        logger.info(f"Response keys: {data.keys()}")
                        print(f"Response keys: {data.keys()}")
                        
                        # Try to extract useful information
                        if "count" in data:
                            logger.info(f"Count: {data['count']}")
                            print(f"Count: {data['count']}")
                        
                        if "results" in data and isinstance(data["results"], list):
                            results = data["results"]
                            logger.info(f"Results list has {len(results)} items")
                            print(f"Results list has {len(results)} items")
                            
                            # Check if Fraser Gault is in the results
                            fraser_found = False
                            for j, person in enumerate(results):
                                name = person.get("fullName", person.get("name", ""))
                                birth_year = person.get("birthYear", "N/A")
                                
                                if "fraser" in name.lower() and "gault" in name.lower():
                                    fraser_found = True
                                    logger.info(f"Found Fraser Gault at position {j+1} with birth year {birth_year}")
                                    print(f"Found Fraser Gault at position {j+1} with birth year {birth_year}")
                            
                            if not fraser_found:
                                logger.info("Fraser Gault not found in results")
                                print("Fraser Gault not found in results")
                            
                            # Print first few results
                            logger.info("First few results:")
                            print("First few results:")
                            for j, person in enumerate(results[:3]):
                                name = person.get("fullName", person.get("name", ""))
                                birth_year = person.get("birthYear", "N/A")
                                logger.info(f"  {j+1}. {name} (Birth Year: {birth_year})")
                                print(f"  {j+1}. {name} (Birth Year: {birth_year})")
            except ValueError as json_err:
                logger.error(f"JSON parsing error: {json_err}")
                logger.debug(f"Response text: {response.text[:500]}")
                print(f"Error parsing JSON: {json_err}")
            
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}")
    
    return True

if __name__ == "__main__":
    test_api_endpoints()
