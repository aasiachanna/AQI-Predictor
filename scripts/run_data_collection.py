import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion import fetch_historical_data

def main():
    # Get data for the past year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"Starting historical data collection from {start_date.date()} to {end_date.date()}")
    
    fetch_historical_data(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )

if __name__ == "__main__":
    main()