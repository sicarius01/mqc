"""Start the local workbench using the active Python environment."""
from pathlib import Path
import subprocess
import sys


def main():
    app = Path(__file__).with_name("app.py")
    command = [sys.executable, "-m", "streamlit", "run", str(app),
               "--server.address", "127.0.0.1",
               "--browser.gatherUsageStats", "false",
               "--client.showErrorLinks", "false", *sys.argv[1:]]
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
