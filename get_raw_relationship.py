import sys
import os
import logging
from api_utils import call_getladder_api
import utils

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Initialize session manager
    session_manager = utils.SessionManager()
    if not session_manager.start_sess():
        print("Failed to start session")
        return

    if not session_manager.ensure_session_ready():
        print("Failed to ensure session is ready")
        return

    # Get the tree ID and person ID from command line arguments or use defaults
    tree_id = "175946702"  # Default tree ID
    person_id = "102281560544"  # Default person ID (Frances Margaret Milne)

    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        person_id = sys.argv[1]

    base_url = "https://www.ancestry.co.uk"

    # Call the getladder API
    print(
        f"Fetching relationship data for tree ID {tree_id} and person ID {person_id}..."
    )
    relationship_data = call_getladder_api(
        session_manager, tree_id, person_id, base_url
    )

    # Print the raw API response
    print("\nRaw API Response:")
    print("=" * 80)
    print(relationship_data)
    print("=" * 80)

    # Close the session
    session_manager.close_sess()


if __name__ == "__main__":
    main()
