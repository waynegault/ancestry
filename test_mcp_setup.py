"""
Test script to verify Blackbox AI and MCP setup
"""
import json
import os
import subprocess
import sys
from pathlib import Path


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def check_file_exists(filepath: str) -> bool:
    """Check if a file exists"""
    exists = Path(filepath).exists()
    status = "✅" if exists else "❌"
    print(f"{status} {filepath}: {'Found' if exists else 'NOT FOUND'}")
    return exists


def check_json_valid(filepath: str) -> bool:
    """Check if JSON file is valid"""
    try:
        with open(filepath) as f:
            data = json.load(f)
        print(f"✅ {filepath}: Valid JSON")
        return True
    except json.JSONDecodeError as e:
        print(f"❌ {filepath}: Invalid JSON - {e}")
        return False
    except Exception as e:
        print(f"❌ {filepath}: Error reading - {e}")
        return False


def check_api_key(filepath: str) -> bool:
    """Check if API key is configured"""
    try:
        with open(filepath) as f:
            data = json.load(f)

        api_key = data.get('mcpServers', {}).get('supermemory', {}).get('env', {}).get('SUPERMEMORY_API_KEY', '')

        if not api_key:
            print("❌ API Key: Not found in configuration")
            return False
        if api_key == "your_api_key_here":
            print("❌ API Key: Still using placeholder value")
            return False
        if api_key.startswith("sm_"):
            print("✅ API Key: Configured (starts with 'sm_')")
            return True
        print("⚠️  API Key: Configured but format unexpected")
        return True
    except Exception as e:
        print(f"❌ API Key: Error checking - {e}")
        return False


def check_node_installed() -> bool:
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(['node', '--version'],
                              capture_output=True,
                              text=True,
                              timeout=5, check=False)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Node.js: Installed ({version})")
            return True
        print("❌ Node.js: Not found")
        return False
    except FileNotFoundError:
        print("❌ Node.js: Not installed")
        return False
    except Exception as e:
        print(f"❌ Node.js: Error checking - {e}")
        return False


def check_npm_installed() -> bool:
    """Check if npm is installed"""
    try:
        result = subprocess.run(['npm', '--version'],
                              capture_output=True,
                              text=True,
                              timeout=5, check=False)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ npm: Installed ({version})")
            return True
        print("❌ npm: Not found")
        return False
    except FileNotFoundError:
        print("❌ npm: Not installed")
        return False
    except Exception as e:
        print(f"❌ npm: Error checking - {e}")
        return False


def check_dependencies_installed() -> bool:
    """Check if npm dependencies are installed"""
    node_modules = Path("supermemory-mcp/node_modules")
    if node_modules.exists():
        print("✅ Dependencies: node_modules directory exists")
        return True
    print("❌ Dependencies: node_modules directory not found")
    return False


def check_wrangler_available() -> bool:
    """Check if wrangler is available"""
    try:
        result = subprocess.run(['npx', 'wrangler', '--version'],
                              capture_output=True,
                              text=True,
                              timeout=10,
                              cwd='supermemory-mcp', check=False)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Wrangler: Available ({version})")
            return True
        print("❌ Wrangler: Not available")
        return False
    except Exception as e:
        print(f"❌ Wrangler: Error checking - {e}")
        return False


def main():
    """Run all setup checks"""
    print_section("Blackbox AI & MCP Setup Verification")

    all_checks_passed = True

    # Check 1: Configuration file
    print_section("1. Configuration Files")
    config_exists = check_file_exists("blackbox_mcp_settings.json")
    all_checks_passed &= config_exists

    if config_exists:
        config_valid = check_json_valid("blackbox_mcp_settings.json")
        all_checks_passed &= config_valid

        if config_valid:
            api_key_ok = check_api_key("blackbox_mcp_settings.json")
            all_checks_passed &= api_key_ok

    # Check 2: MCP Server directory
    print_section("2. MCP Server Directory")
    mcp_dir_exists = check_file_exists("supermemory-mcp")
    all_checks_passed &= mcp_dir_exists

    if mcp_dir_exists:
        package_json_exists = check_file_exists("supermemory-mcp/package.json")
        all_checks_passed &= package_json_exists

    # Check 3: Node.js and npm
    print_section("3. Node.js Environment")
    node_ok = check_node_installed()
    npm_ok = check_npm_installed()
    all_checks_passed &= node_ok and npm_ok

    # Check 4: Dependencies
    print_section("4. Dependencies")
    deps_ok = check_dependencies_installed()
    all_checks_passed &= deps_ok

    if deps_ok:
        wrangler_ok = check_wrangler_available()
        all_checks_passed &= wrangler_ok

    # Check 5: Setup guide
    print_section("5. Documentation")
    guide_exists = check_file_exists("BLACKBOX_SETUP_GUIDE.md")
    # Don't fail if guide is missing, it's just documentation

    # Final summary
    print_section("Setup Verification Summary")

    if all_checks_passed:
        print("✅ ALL CHECKS PASSED!")
        print("\n🎉 Your Blackbox AI and MCP setup is ready to use!")
        print("\nNext steps:")
        print("1. Read BLACKBOX_SETUP_GUIDE.md for usage instructions")
        print("2. Start the MCP server: cd supermemory-mcp && npx wrangler dev --port 3000")
        print("3. Connect your AI client (Blackbox, Claude, or Cline)")
        return 0
    print("❌ SOME CHECKS FAILED")
    print("\n⚠️  Please fix the issues above before proceeding.")
    print("\nCommon fixes:")
    print("- Install Node.js: https://nodejs.org/")
    print("- Install dependencies: cd supermemory-mcp && npm install")
    print("- Configure API key in blackbox_mcp_settings.json")
    return 1


if __name__ == "__main__":
    sys.exit(main())
