import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ Get API key from .env
API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not API_KEY:
    raise ValueError("❌ OPENWEATHER_API_KEY not found in .env file")

# ✅ Coordinates for your city (e.g., Sukkur, Sindh)
LAT = 27.72489
LON = 68.79245

# ✅ Build API URL (use free-tier endpoint)
url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"

print(f"Fetching weather data from: {url}")

# ✅ Fetch data from API
response = requests.get(url)
print("Status Code:", response.status_code)

if response.status_code != 200:
    print("Response:", response.text)
    response.raise_for_status()

data = response.json()

# ✅ Create folder if not exists
os.makedirs("data/raw", exist_ok=True)

# ✅ Generate timestamped filename
ts = datetime.now().strftime("%Y%m%dT%H%M%S")
fname = f"data/raw/openweather_{ts}.json"

# ✅ Save the data
with open(fname, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"✅ Weather data saved successfully to {fname}")
