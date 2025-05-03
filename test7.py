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
    from utils import SessionManager, _api_req
except ImportError:
    logger.error("Could not import SessionManager from utils. Make sure the utils module is available.")
    sys.exit(1)

def test_api_endpoints():
    """Test different API endpoint formats for incorporating birth year using the specified URL format."""
    
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
    
    # Test different API endpoints and parameter formats based on the provided URL
    test_cases = [
        # Test 1: Base URL format as provided but with a smaller limit
        {
            "description": "Base URL format with smaller limit",
            "url": f"{base_url}/api/treesui-list/trees/{tree_id}/persons",
            "params": {"ln": "gault", "limit": "20", "fields": "NAMES"},
        },
        # Test 2: Add birth year parameter (by) with smaller limit
        {
            "description": "With birth year parameter (by) and smaller limit",
            "url": f"{base_url}/api/treesui-list/trees/{tree_id}/persons",
            "params": {"ln": "gault", "limit": "20", "fields": "NAMES", "by": "1941"},
        },
        # Test 3: Add first name 'fraser' and birth year with smaller limit
        {
            "description": "With first name 'fraser', birth year, and smaller limit",
            "url": f"{base_url}/api/treesui-list/trees/{tree_id}/persons",
            "params": {"fn": "fraser", "ln": "gault", "limit": "20", "fields": "NAMES", "by": "1941"},
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
            "Accept": "application/json",
            "Referer": owner_facts_referer,
        }
        
        # Make the request using _api_req function from utils
        try:
            # Add a delay to avoid rate limiting
            time.sleep(1)
            
            # Make request using _api_req which handles authentication
            logger.info(f"Making request to {url}")
            data = _api_req(
                url=url,
                driver=session_manager.driver,
                session_manager=session_manager,
                method="GET",
                api_description=f"Test {i+1}: {test_case['description']}",
                headers=headers,
                timeout=15,  # Shorter timeout
            )
            
            if data is None:
                logger.error(f"API request returned None for {url}")
                print(f"API request returned None for {url}")
                continue
            
            # Save the response to a file for inspection
            filename = f"test7_response_{i+1}.json"
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
        
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}")
    
    return True

if __name__ == "__main__":
    test_api_endpoints()
