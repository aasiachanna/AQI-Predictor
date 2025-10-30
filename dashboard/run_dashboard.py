# dashboard/run_dashboard.py
import streamlit.web.cli as stcli
import sys
import os

def main():
    # Set the working directory to the dashboard folder
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the Streamlit app
    sys.argv = [
        "streamlit", 
        "run", 
        "app.py", 
        "--server.port=8501", 
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()