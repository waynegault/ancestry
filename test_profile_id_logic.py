#!/usr/bin/env python3
"""
Test script to fetch and analyze profile_id assignment logic for EC's test case.
EC's test UUID: 0dc89b73-fc37-49a3-b1d4-7a659fdd2328
EC's test is managed by TANEJ
"""

import json
import requests
from typing import Optional

# Test case details
MY_UUID = "FB609BA5-5A0D-46EE-BF18-C300D8DE5AB7"
EC_UUID = "0dc89b73-fc37-49a3-b1d4-7a659fdd2328"
BASE_URL = "https://www.ancestry.co.uk"

def fetch_match_details(match_uuid: str) -> Optional[dict]:
    """Fetch match details from Ancestry API."""
    url = f"{BASE_URL}/discoveryui-matchesservice/api/samples/{MY_UUID}/matches/{match_uuid}/details?pmparentaldata=true"
    
    print(f"\n{'='*80}")
    print(f"Fetching details for match: {match_uuid}")
    print(f"URL: {url}")
    print(f"{'='*80}\n")
    
    try:
        # Note: This will fail without proper authentication cookies
        # User should run this in their browser or with their session
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to fetch: HTTP {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def analyze_profile_assignment(data: dict) -> None:
    """Analyze the profile_id assignment logic based on API data."""
    
    # Extract key fields
    tester_profile_id = data.get("userId")
    admin_profile_id = data.get("adminUcdmId")
    tester_username = data.get("displayName")
    admin_username = data.get("adminDisplayName")
    
    print("\n" + "="*80)
    print("API DATA ANALYSIS")
    print("="*80)
    print(f"Tester Profile ID (userId):        {tester_profile_id}")
    print(f"Admin Profile ID (adminUcdmId):     {admin_profile_id}")
    print(f"Tester Username (displayName):      {tester_username}")
    print(f"Admin Username (adminDisplayName):  {admin_username}")
    print("="*80)
    
    # Apply the logic from _resolve_profile_assignment
    print("\n" + "="*80)
    print("PROFILE ASSIGNMENT LOGIC")
    print("="*80)
    
    profile_id_to_save = None
    administrator_profile_id_to_save = None
    administrator_username_to_save = None
    
    # Both tester and admin IDs present
    if tester_profile_id and admin_profile_id:
        print(f"✓ Both tester_profile_id and admin_profile_id present")
        
        if tester_profile_id == admin_profile_id:
            print(f"  → Same ID: {tester_profile_id}")
            
            # Check if usernames match
            if tester_username and admin_username and tester_username.lower() == admin_username.lower():
                print(f"  → Usernames match: '{tester_username}'")
                print(f"  → SCENARIO A: Member with own test")
                profile_id_to_save = tester_profile_id
            else:
                print(f"  → Usernames differ: '{tester_username}' vs '{admin_username}'")
                print(f"  → SCENARIO: Same profile_id but different usernames (unusual)")
                administrator_profile_id_to_save = admin_profile_id
                administrator_username_to_save = admin_username
        else:
            print(f"  → Different IDs: tester={tester_profile_id}, admin={admin_profile_id}")
            print(f"  → SCENARIO C: Member administering another member's test")
            profile_id_to_save = tester_profile_id
            administrator_profile_id_to_save = admin_profile_id
            administrator_username_to_save = admin_username
    
    # Only tester ID present
    elif tester_profile_id:
        print(f"✓ Only tester_profile_id present: {tester_profile_id}")
        print(f"  → SCENARIO: Member with own test (no admin)")
        profile_id_to_save = tester_profile_id
    
    # Only admin ID present
    elif admin_profile_id:
        print(f"✓ Only admin_profile_id present: {admin_profile_id}")
        print(f"  → SCENARIO B: Non-member test administered by member")
        administrator_profile_id_to_save = admin_profile_id
        administrator_username_to_save = admin_username
    
    else:
        print(f"❌ Neither tester_profile_id nor admin_profile_id present!")
    
    print("\n" + "="*80)
    print("FINAL DATABASE RECORD")
    print("="*80)
    print(f"uuid:                      {EC_UUID}")
    print(f"profile_id:                {profile_id_to_save or 'NULL'}")
    print(f"administrator_profile_id:  {administrator_profile_id_to_save or 'NULL'}")
    print(f"administrator_username:    {administrator_username_to_save or 'NULL'}")
    print(f"username:                  {tester_username}")
    print("="*80)
    
    # Determine which scenario this is
    print("\n" + "="*80)
    print("SCENARIO CLASSIFICATION")
    print("="*80)
    
    if profile_id_to_save and not administrator_profile_id_to_save:
        print("✓ SCENARIO A: Member with their own test")
        print(f"  - EC is an Ancestry member")
        print(f"  - EC owns their own DNA test")
        print(f"  - profile_id = {profile_id_to_save}")
    elif not profile_id_to_save and administrator_profile_id_to_save:
        print("✓ SCENARIO B: Non-member test administered by member")
        print(f"  - EC is NOT an Ancestry member")
        print(f"  - EC's test is managed by {admin_username}")
        print(f"  - administrator_profile_id = {administrator_profile_id_to_save}")
    elif profile_id_to_save and administrator_profile_id_to_save:
        print("✓ SCENARIO C: Member administering another member's test")
        print(f"  - EC is an Ancestry member (profile_id = {profile_id_to_save})")
        print(f"  - EC's test is managed by {admin_username}")
        print(f"  - administrator_profile_id = {administrator_profile_id_to_save}")
    
    print("="*80)


def main():
    """Main function."""
    print("\n" + "="*80)
    print("PROFILE_ID ASSIGNMENT LOGIC TEST")
    print("="*80)
    print(f"Test Case: EC (managed by TANEJ)")
    print(f"EC's UUID: {EC_UUID}")
    print(f"Your UUID: {MY_UUID}")
    print("="*80)
    
    print("\n⚠️  NOTE: This script requires authentication cookies to work.")
    print("Please run the curl command manually and paste the JSON response below,")
    print("or run this script with proper session cookies.\n")
    
    # Try to fetch (will likely fail without auth)
    data = fetch_match_details(EC_UUID)
    
    if data:
        analyze_profile_assignment(data)
    else:
        print("\n" + "="*80)
        print("MANUAL TESTING INSTRUCTIONS")
        print("="*80)
        print("\n1. Run the curl command you provided")
        print("2. Copy the JSON response")
        print("3. Look for these fields:")
        print("   - userId (tester_profile_id)")
        print("   - adminUcdmId (admin_profile_id)")
        print("   - displayName (tester_username)")
        print("   - adminDisplayName (admin_username)")
        print("\n4. Based on the values, determine:")
        print("   - If userId == adminUcdmId AND displayName == adminDisplayName:")
        print("     → SCENARIO A: profile_id = userId, administrator_profile_id = NULL")
        print("   - If userId is NULL/missing:")
        print("     → SCENARIO B: profile_id = NULL, administrator_profile_id = adminUcdmId")
        print("   - If userId != adminUcdmId:")
        print("     → SCENARIO C: profile_id = userId, administrator_profile_id = adminUcdmId")
        print("="*80)


if __name__ == "__main__":
    main()

