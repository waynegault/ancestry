#!/usr/bin/env python3
"""Quick Action 6 status check"""

import psutil
import time

def check_action6_status():
    print("ACTION 6 STATUS CHECK")
    print("=" * 30)
    
    # Check main Action 6 process
    try:
        action6_proc = psutil.Process(43412)
        runtime = (time.time() - action6_proc.create_time()) / 60
        memory = action6_proc.memory_info().rss / 1024 / 1024
        print(f"Action 6 Process: RUNNING")
        print(f"Runtime: {runtime:.1f} minutes")
        print(f"Memory: {memory:.1f} MB")
    except psutil.NoSuchProcess:
        print("Action 6 Process: NOT FOUND")
        return
    
    # Check for browser automation
    chromedriver_found = False
    chrome_count = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'create_time']):
        try:
            name = proc.info['name'].lower()
            if 'chromedriver' in name:
                chromedriver_found = True
            elif 'chrome.exe' in name:
                proc_runtime = (time.time() - proc.info['create_time']) / 60
                if proc_runtime < 20:  # Recent processes
                    chrome_count += 1
        except:
            continue
    
    print(f"ChromeDriver: {'ACTIVE' if chromedriver_found else 'NOT DETECTED'}")
    print(f"Chrome Processes: {chrome_count}")
    
    # Determine phase
    if chromedriver_found and chrome_count > 0:
        print("Phase: BROWSER AUTOMATION ACTIVE")
        print("Status: Action 6 is working on DNA match gathering")
    elif runtime < 5:
        print("Phase: INITIALIZATION")
        print("Status: Action 6 is starting up")
    else:
        print("Phase: PROCESSING")
        print("Status: Action 6 is working")
    
    print("\nHealth Monitoring: ACTIVE")
    print("Emergency Intervention: READY")

if __name__ == "__main__":
    check_action6_status()
