"""Run the Streamlit dashboard."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])

