import os
import requests
import time
from datetime import datetime

# SILO API Documentation: https://www.longpaddock.qld.gov.au/silo/point-data/
# Base URL for Patched Point Dataset
SILO_URL = "https://www.longpaddock.qld.gov.au/cgi-bin/silo/PatchedPointDataset.php"

# Configuration
OUTPUT_DIR = os.path.join("data", "raw", "weather")
# Valid email is technically required by SILO.
USER_EMAIL = os.environ.get("SILO_EMAIL", "student_research@university.edu.au") 

START_DATE = "20150101"
# Current date formatted as YYYYMMDD
END_DATE = datetime.now().strftime("%Y%m%d")

# Capital City Stations (Approximations for State-level weather)
STATIONS = {
    "Sydney": "066062",    # NSW
    "Melbourne": "086071", # VIC
    "Brisbane": "040913",  # QLD
    "Adelaide": "023090",  # SA
    "Perth": "009021",     # WA
    "Hobart": "094029",    # TAS
    "Darwin": "014015",    # NT
    "Canberra": "070351"   # ACT
}

def setup_directories():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def download_station_data(station_name, station_id):
    print(f"Downloading data for {station_name} (ID: {station_id})...")
    
    # SILO Parameters:
    # start, stop: YYYYMMDD
    # station: Station Number
    # format: csv, also, flu
    # username: email address
    # comment: string
    
    params = {
        "start": START_DATE,
        "stop": END_DATE,
        "station": station_id,
        "format": "csv",
        "username": USER_EMAIL,
        "comment": "AUS-WxHealthBench"
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        # Notes: SILO sometimes requires SSL verification to be handled carefully or standard certificates
        response = requests.get(SILO_URL, params=params, headers=headers)
        response.raise_for_status()
        
        # Check if response is an error message (SILO often returns 200 OK even for errors, but with text)
        content = response.text
        if "Error" in content and len(content) < 500:
            print(f"  Error received from SILO: {content.strip()}")
            return False
            
        output_path = os.path.join(OUTPUT_DIR, f"{station_name}_{station_id}.csv")
        with open(output_path, "w", newline="") as f:
            f.write(content)
            
        print(f"  Saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"  Failed to download {station_name}: {e}")
        return False

def main():
    setup_directories()
    print(f"Using email for SILO API: {USER_EMAIL}")
    print("Note: If downloads fail, please set a valid email via 'SILO_EMAIL' env var or edit this script.")
    
    success_count = 0
    for name, site_id in STATIONS.items():
        if download_station_data(name, site_id):
            success_count += 1
        # Be nice to the server
        time.sleep(1) 
        
    print(f"Finished. Downloaded {success_count}/{len(STATIONS)} stations.")

if __name__ == "__main__":
    main()
