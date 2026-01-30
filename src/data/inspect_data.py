import pandas as pd
import os

ABS_FILE = os.path.join("data", "raw", "abs", "deaths_weekly_2015_24.xlsx")
WEATHER_DIR = os.path.join("data", "raw", "weather")

def inspect_abs():
    print(f"--- Inspecting {ABS_FILE} ---")
    if not os.path.exists(ABS_FILE):
        print("File not found.")
        return

    try:
        xl = pd.ExcelFile(ABS_FILE)
        print("Sheet names:", xl.sheet_names)
        
        # Read the first relevant looking sheet
        sheet_name = xl.sheet_names[1] if len(xl.sheet_names) > 1 else xl.sheet_names[0]
        print(f"Reading sheet: {sheet_name}")
        # Skip first 5 rows to see the header
        df = pd.read_excel(ABS_FILE, sheet_name=sheet_name, skiprows=5, nrows=10)
        print(df.head())
        print("\nColumns:", df.columns.tolist())
    except Exception as e:
        print(f"Error reading ABS file: {e}")

def inspect_weather():
    print(f"\n--- Inspecting First Weather File in {WEATHER_DIR} ---")
    files = [f for f in os.listdir(WEATHER_DIR) if f.endswith(".csv")]
    if not files:
        print("No CSV files found.")
        return
    
    target_file = os.path.join(WEATHER_DIR, files[0])
    print(f"Reading {target_file}")
    try:
        # SILO data often has a header block, need to skip rows?
        # Let's peek first
        with open(target_file, 'r') as f:
            head = [next(f) for _ in range(5)]
        print("First 5 lines:")
        for line in head:
            print(line.strip())
            
    except Exception as e:
        print(f"Error reading weather file: {e}")

if __name__ == "__main__":
    inspect_abs()
    inspect_weather()
