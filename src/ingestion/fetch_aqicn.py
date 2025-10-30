
import os, requests, json, time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
OUT_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw")
os.makedirs(OUT_DIR, exist_ok=True)
def fetch_aqicn_by_geo(lat, lon):
    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={AQICN_TOKEN}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

def save_raw(json_obj, prefix="aqicn"):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = f"{OUT_DIR}/{prefix}_{ts}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2)
    print("Saved", fname)
    return fname

if __name__ == "__main__":
   
    lat, lon = 27.72489, 68.79245   
    data = fetch_aqicn_by_geo(lat, lon)
    save_raw(data, prefix="aqicn")

