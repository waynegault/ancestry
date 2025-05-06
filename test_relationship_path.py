import sys
import os
import logging
from api_utils import format_api_relationship_path

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Sample HTML response from getladder API
sample_html = r"""
no({"status":"success","html":"<div class=\"relationshipPath\"><ul class=\"textCenter\"><li><i>Gordon Milne is <b>Wayne Gault's<\/b> maternal grandfather<\/i><\/li><li class=\"iconArrowDown\"><\/li><li><a href=\"/family-tree/person/tree/175946702/person/102281560544/facts\">Frances Margaret Milne<\/a> 1947<br><i>is daughter's<\/i><\/li><li class=\"iconArrowDown\"><\/li><li><a href=\"/family-tree/person/tree/175946702/person/102281560836/facts\">Wayne Gordon Gault<\/a><\/li><\/ul><\/div>"})
"""


# Test the function
def test_format_relationship_path():
    print("Testing relationship path formatting...")
    formatted_path = format_api_relationship_path(
        sample_html, "Wayne Gault", "Gordon Milne"
    )
    print("\nFormatted Relationship Path:")
    print("=" * 50)
    print(formatted_path)
    print("=" * 50)


if __name__ == "__main__":
    test_format_relationship_path()
