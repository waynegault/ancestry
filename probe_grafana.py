import base64
import socket

import requests


def check_port(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0


def probe():
    print("--- Probing Grafana ---")

    # 1. Check Port
    if check_port("localhost", 3000):
        print("✓ Port 3000 is open")
    else:
        print("✗ Port 3000 is CLOSED. Grafana is not running.")
        return

    # 2. Check Health
    try:
        resp = requests.get("http://localhost:3000/api/health", timeout=5)
        print(f"✓ Health Check: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"✗ Health Check Failed: {e}")
        return

    # 3. Check Auth
    creds = [("admin", "admin"), ("admin", "ancestry")]
    valid_creds = None

    for user, pwd in creds:
        print(f"Testing credentials: {user}:{pwd} ...")
        auth_str = f"{user}:{pwd}"
        b64 = base64.b64encode(auth_str.encode()).decode()
        headers = {"Authorization": f"Basic {b64}"}

        try:
            resp = requests.get("http://localhost:3000/api/org", headers=headers, timeout=5)
            if resp.status_code == 200:
                print(f"  ✓ Success! Org: {resp.json().get('name')}")
                valid_creds = (user, pwd)
                break
            print(f"  ✗ Failed: {resp.status_code}")
        except Exception as e:
            print(f"  ✗ Exception: {e}")

    if not valid_creds:
        print("✗ Could not authenticate with standard credentials.")
        return

    # 4. Check Datasources with valid creds
    user, pwd = valid_creds
    auth_str = f"{user}:{pwd}"
    b64 = base64.b64encode(auth_str.encode()).decode()
    headers = {"Authorization": f"Basic {b64}"}

    try:
        resp = requests.get("http://localhost:3000/api/datasources", headers=headers, timeout=5)
        print(f"Datasources Status: {resp.status_code}")
        if resp.status_code == 200:
            ds_list = resp.json()
            print(f"Found {len(ds_list)} datasources:")
            for ds in ds_list:
                print(f"  - {ds['name']} (UID: {ds.get('uid')}, Type: {ds['type']})")
    except Exception as e:
        print(f"✗ Failed to list datasources: {e}")


if __name__ == "__main__":
    probe()
