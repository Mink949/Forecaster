import os
import re
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# Constants
BASE_URL = "https://www.abs.gov.au"
LATEST_RELEASE_URL = "https://www.abs.gov.au/statistics/health/causes-death/provisional-mortality-statistics/latest-release"
OUTPUT_DIR = os.path.join("data", "raw", "abs")
OUTPUT_FILENAME = "deaths_weekly_2015_24.xlsx"

def setup_directories():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

def find_download_link(url):
    print(f"Scraping {url} for data links...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for links containing "Deaths by week" and ".xlsx"
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text()
            
            # Check either text OR href for the key terms
            # The link text might just be "Download", so checking href is safer
            if ("Deaths" in href and "week" in href and ".xlsx" in href) or \
               ("Deaths by week" in text and ".xlsx" in href):
                full_url = urljoin(BASE_URL, href)
                print(f"Found dataset link: {full_url}")
                return full_url
            
            # Fallback: sometimes the link text is just "Download" but it's inside a section
            # This is harder to robustly detect without more structure, 
            # but the ABS usually puts the title in the link text or nearby.
            
    except Exception as e:
        print(f"Error scraping ABS website: {e}")
    
    return None

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def main():
    setup_directories()
    
    print("Step 1: finding dataset URL...")
    download_url = find_download_link(LATEST_RELEASE_URL)
    
    if not download_url:
        print("Could not find the 'Deaths by week' dataset link on the latest release page.")
        print("Please check the ABS website manually: " + LATEST_RELEASE_URL)
        return

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    print("Step 2: Downloading file...")
    success = download_file(download_url, output_path)
    
    if success:
        print(f"SUCCESS: Data saved to {output_path}")
    else:
        print("FAILURE: Could not download data.")

if __name__ == "__main__":
    main()
