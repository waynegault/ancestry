import pathlib
import subprocess


def reset_password():
    grafana_cli = r"C:\Program Files\GrafanaLabs\grafana\bin\grafana-cli.exe"
    if not pathlib.Path(grafana_cli).exists():
        print(f"Error: {grafana_cli} not found.")
        return

    print("Resetting admin password to 'ancestry'...")
    try:
        # Set CWD to Grafana install dir so it finds defaults.ini
        cwd = r"C:\Program Files\GrafanaLabs\grafana"
        result = subprocess.run(
            [grafana_cli, "admin", "reset-admin-password", "ancestry"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        print("Password reset successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to reset password: {e}")
        print(e.stdout)
        print(e.stderr)


if __name__ == "__main__":
    reset_password()
