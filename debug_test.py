#!/usr/bin/env python3
"""Debug test for get_api_family_details"""

import api_search_utils
from unittest.mock import MagicMock, patch


def test_family_details_no_tree_id():
    mock_session = MagicMock()
    mock_session.is_sess_valid.return_value = True
    mock_session.my_tree_id = None

    print("About to call get_api_family_details with mocked config...")
    with patch("api_search_utils.get_config_value", return_value=""):
        result = api_search_utils.get_api_family_details(mock_session, "person123")
        print(f"Function returned: {result}")
        print(f"Type: {type(result)}")
        print(f"result == {{}}: {result == {}}")
        return result == {}


if __name__ == "__main__":
    test_result = test_family_details_no_tree_id()
    print(f"Test function returned: {test_result}")
