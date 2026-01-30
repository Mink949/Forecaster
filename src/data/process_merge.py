import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Config
ABS_FILE = os.path.join("data", "raw", "abs", "deaths_weekly_2015_24.xlsx")
WEATHER_DIR = os.path.join("data", "raw", "weather")
OUTPUT_FILE = os.path.join("data", "processed", "benchmark_dataset.csv")

def load_abs_mortality():
    print("Loading ABS Mortality data...")
    if not os.path.exists(ABS_FILE):
        raise FileNotFoundError(f"ABS file not found: {ABS_FILE}")
        
    xl = pd.ExcelFile(ABS_FILE)
    # Heuristic: Find 'Table 3.1' which usually contains total deaths
    sheet_name = next((s for s in xl.sheet_names if "Table 3.1" in s), None)
    if not sheet_name:
        sheet_name = xl.sheet_names[0]
        print(f"  'Table 3.1' not found, defaulting to {sheet_name}")
        
    # Skip metadata rows. Based on inspection, data starts around row 6 (index 5)
    # The header row with '2015', '2016' etc in column 0 is what we want?
    # Actually, inspecting showed: 
    # Col 0: "Persons, all ages", row 0="2015", row 1="2016"
    # Cols 2-53: Week numbers.
    
    # Let's read with header=5 (row 6) where 'Persons, all ages' is likely the header
    df_raw = pd.read_excel(ABS_FILE, sheet_name=sheet_name, skiprows=5)
    
    # Rename first column to Year
    df_raw.rename(columns={df_raw.columns[0]: 'Year'}, inplace=True)
    
    # Filter rows where Year is a number (2015..2024)
    df_clean = df_raw[pd.to_numeric(df_raw['Year'], errors='coerce').notna()].copy()
    df_clean['Year'] = df_clean['Year'].astype(int)
    
    # Drop "Unnamed: 1" (Unit 'no.')
    if "Unnamed: 1" in df_clean.columns:
        df_clean.drop(columns=["Unnamed: 1"], inplace=True)
        
    # Melt: Year | Week | Deaths
    # Columns now should be Year, Unnamed: 2 (Week 1), ... Unnamed: 53 (Week 52/53)
    # The headers in the Excel were merged or messy.
    # We assume columns 1..53 map to Week 1..53
    
    # Create valid column map
    col_map = {'Year': 'Year'}
    # df_clean.columns[1] is Week 1
    for i in range(1, len(df_clean.columns)):
        col_name = df_clean.columns[i]
        week_num = i # 1-based index roughly corresponds to week number
        if week_num > 53: break # Safety
        col_map[col_name] = week_num
        
    df_melted = df_clean.melt(id_vars=['Year'], var_name='WeekCol', value_name='Deaths')
    
    # Map WeekCol to Week Number
    df_melted['Week'] = df_melted['WeekCol'].map(col_map)
    df_melted = df_melted.dropna(subset=['Deaths', 'Week'])
    
    # Convert to numeric
    df_melted['Deaths'] = pd.to_numeric(df_melted['Deaths'], errors='coerce')
    df_melted['Week'] = pd.to_numeric(df_melted['Week'], errors='coerce')
    
    # Create Date from Year-Week
    # ABS usually uses ISO weeks or Week Ending.
    # We'll approximate: Date = Year-01-01 + (Week-1)*7 days
    def get_date_from_year_week(row):
        try:
            return datetime.strptime(f"{int(row['Year'])}-W{int(row['Week'])}-1", "%Y-W%W-%w")
        except:
            return None

    df_melted['Date'] = df_melted.apply(get_date_from_year_week, axis=1)
    df_melted = df_melted.dropna(subset=['Date'])
    
    # Sort
    df_melted = df_melted.sort_values('Date')
    df_melted['Region'] = 'Australia' 
    
    # HANDLE DUPLICATES: 
    # The excel likely contains rows for Australia (Total), plus States, plus Age groups.
    # We want the National Total. 
    # Heuristic: The Total row will have the largest number of deaths for a given date.
    df_unique = df_melted.groupby(['Date', 'Region'], as_index=False)['Deaths'].max()
    
    return df_unique[['Date', 'Region', 'Deaths']]

def load_weather():
    print("Loading Weather data...")
    all_weather = []
    
    # Station to Region mapping (Capital City -> State equivalent for now)
    # For now, if we only have National mortality, we should aggregate weather or pick one?
    # Picking SYDNEY + MELBOURNE average as "Australia" proxy is common, or Population Weighted.
    # Let's verify files exist
    files = [f for f in os.listdir(WEATHER_DIR) if f.endswith(".csv")]
    
    for f in files:
        path = os.path.join(WEATHER_DIR, f)
        df = pd.read_csv(path)
        
        # Ensure Date parsing
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Resample to Weekly (starting Mondays to match ABS?)
        # ABS dates we generated are Mondays (day 1 of week).
        df.set_index('Date', inplace=True)
        weekly = df.resample('W-MON').agg({
            'max_t': 'mean',
            'min_t': 'mean',
            'rain': 'sum'
        }).reset_index()
        
        # Station Name from filename
        station_name = f.split('_')[0]
        weekly['Station'] = station_name
        
        all_weather.append(weekly)
        
    if not all_weather:
        print("No weather data found!")
        return pd.DataFrame()
        
    full_weather = pd.concat(all_weather)
    
    # Create a National Proxy: Average of all capital cities
    national = full_weather.groupby('Date')[['max_t', 'min_t', 'rain']].mean().reset_index()
    national['Region'] = 'Australia'
    
    return national

def construct_features(df):
    print("Constructing features...")
    # Sort
    df = df.sort_values(['Region', 'Date'])
    
    # Lags for Weather (Covariates) and Deaths (Autoregressive)
    # We want to forecast Deaths.
    # Lagged features: t-1, t-2, t-3, t-4, t-8
    lags = [1, 2, 3, 4, 8]
    
    for l in lags:
        df[f'Deaths_lag_{l}'] = df.groupby('Region')['Deaths'].shift(l)
        df[f'max_t_lag_{l}'] = df.groupby('Region')['max_t'].shift(l)
        df[f'min_t_lag_{l}'] = df.groupby('Region')['min_t'].shift(l)
    
    # Rolling features (Mean, Std)
    df['Deaths_roll_mean_4'] = df.groupby('Region')['Deaths'].transform(lambda x: x.shift(1).rolling(4).mean())
    df['max_t_roll_mean_4'] = df.groupby('Region')['max_t'].transform(lambda x: x.shift(1).rolling(4).mean())
    
    # Calendar features
    df['Month'] = df['Date'].dt.month
    df['Week of Year'] = df['Date'].dt.isocalendar().week
    
    # Surge Label
    # Definition: Deaths > 95th percentile of the previous year? Or global 90th?
    # Simple: Global 90th percentile for the region
    p90 = df.groupby('Region')['Deaths'].transform(lambda x: x.quantile(0.90))
    df['is_surge'] = (df['Deaths'] > p90).astype(int)
    
    # Drop NaNs (first 8 weeks lost)
    df_clean = df.dropna()
    
    return df_clean

def main():
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")
        
    # 1. Load
    try:
        health_df = load_abs_mortality()
        print(f"Loaded {len(health_df)} health records.")
    except Exception as e:
        print(f"Failed to load health data: {e}")
        return

    weather_df = load_weather()
    print(f"Loaded {len(weather_df)} weather records (aggregated).")
    
    # 2. Merge
    # Merge on Date and Region
    # Ensure Dates align (exact match might fail if resampling differs slightly)
    # We used W-MON for weather. ABS logic was W%W-%w (Moday).
    combined = pd.merge(health_df, weather_df, on=['Date', 'Region'], how='inner')
    print(f"Merged: {len(combined)} records.")
    
    if len(combined) == 0:
        print("Merge resulted in 0 records. Check date alignment.")
        print("Health Dates example:", health_df['Date'].head())
        print("Weather Dates example:", weather_df['Date'].head())
        return

    # 3. Features
    final_df = construct_features(combined)
    
    # 4. Save
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved benchmark dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
