import base64
import subprocess
import sys

import requests


def reset_and_rebuild():
    base_url = "http://localhost:3000"
    creds_list = [("admin", "admin"), ("admin", "ancestry")]

    session = requests.Session()
    authenticated = False

    for user, password in creds_list:
        auth_str = f"{user}:{password}"
        b64_auth = base64.b64encode(auth_str.encode()).decode()
        headers = {"Authorization": f"Basic {b64_auth}", "Content-Type": "application/json"}

        try:
            resp = session.get(f"{base_url}/api/org", headers=headers)
            if resp.status_code == 200:
                print(f"Authenticated as {user}")
                session.headers.update(headers)
                authenticated = True
                break
        except Exception:
            pass

    if authenticated:
        # Delete all datasources
        resp = session.get(f"{base_url}/api/datasources")
        if resp.status_code == 200:
            datasources = resp.json()

            for ds in datasources:
                print(f"Deleting datasource: {ds['name']} (UID: {ds.get('uid')})")
                # Try deleting by UID first
                del_resp = session.delete(f"{base_url}/api/datasources/uid/{ds.get('uid')}")

                if del_resp.status_code != 200:
                    # Try by ID
                    print(f"Delete by UID failed ({del_resp.status_code}), trying by ID {ds['id']}...")
                    del_resp = session.delete(f"{base_url}/api/datasources/{ds['id']}")

                if del_resp.status_code == 200:
                    print("Deleted.")
                else:
                    print(f"Failed to delete: {del_resp.status_code} {del_resp.text}")
        else:
            print(f"Failed to list datasources: {resp.status_code}")
    else:
        print(
            "Could not authenticate to delete datasources. Proceeding to setup anyway (might fail if conflicts persist)."
        )

    print("Running fix_grafana.py to recreate datasources and dashboards...")
    subprocess.run([sys.executable, "fix_grafana.py"], check=False)


if __name__ == "__main__":
    reset_and_rebuild()
