#!/usr/bin/env python3

"""
Test script for action11.py
Runs action11.py with predefined inputs for Fraser Gault
"""

import os
import sys
import subprocess
import time

def main():
    """Run action11.py with predefined inputs"""
    print("Running action11.py with predefined inputs for Fraser Gault...")
    
    # Start the process
    process = subprocess.Popen(
        ["python", "action11.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Define the inputs
    inputs = [
        "Fraser\n",  # First name
        "Gault\n",   # Surname
        "m\n",       # Gender
        "1941\n",    # Birth year
        "\n",        # Birth place (skip)
        "\n",        # Death year (skip)
        "\n"         # Death place (skip)
    ]
    
    # Send inputs with a delay
    for input_text in inputs:
        print(f"Sending input: {input_text.strip()}")
        process.stdin.write(input_text)
        process.stdin.flush()
        time.sleep(1)  # Give time for processing
    
    # Wait for the process to complete
    print("Waiting for process to complete...")
    stdout, stderr = process.communicate(timeout=300)
    
    # Print output
    print("\n--- STDOUT ---")
    print(stdout)
    
    print("\n--- STDERR ---")
    print(stderr)
    
    print(f"\nProcess exited with code: {process.returncode}")
    return process.returncode

if __name__ == "__main__":
    sys.exit(main())
