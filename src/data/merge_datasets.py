"""
Merge Air Quality and Weather Death datasets for combined forecasting.

This script:
1. Loads both datasets
2. Merges on Year and Week
3. Handles missing values
4. Creates lag features for air quality indicators
5. Saves combined dataset
"""

import os
import pandas as pd
import numpy as np

# Paths
AIR_QUALITY_FILE = os.path.join("data", "processed", "Air_Quality_Weekly_Evaluation.csv")
WEATHER_DEATH_FILE = os.path.join("data", "processed", "Weather_Death_Weekly_Merged_MinMax.csv")
OUTPUT_FILE = os.path.join("data", "processed", "combined_benchmark.csv")


def load_air_quality():
    """Load and preprocess Air Quality dataset."""
    print("Loading Air Quality data...")
    df = pd.read_csv(AIR_QUALITY_FILE)
    
    # Select useful numeric columns
    air_cols = ['Year', 'Week', 'Bad_days_count', 'AQI_weekly_mean', 'AQI_weekly_max',
                'PM25_weekly_mean', 'PM10_weekly_mean', 'O3_weekly_mean', 
                'NO2_weekly_mean', 'CO_weekly_mean', 'is_bad_air_week']
    
    # Keep only columns that exist
    available_cols = [c for c in air_cols if c in df.columns]
    df = df[available_cols].copy()
    
    print(f"  Loaded {len(df)} rows with columns: {list(df.columns)}")
    return df


def load_weather_death():
    """Load Weather Death dataset."""
    print("Loading Weather Death data...")
    df = pd.read_csv(WEATHER_DEATH_FILE)
    
    print(f"  Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def create_lag_features(df, cols, lags=[1, 2, 4]):
    """Create lagged features for specified columns."""
    print(f"Creating lag features for: {cols}")
    
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df


def merge_datasets():
    """Merge Air Quality and Weather Death datasets."""
    
    # Load both datasets
    air_df = load_air_quality()
    weather_df = load_weather_death()
    
    # Merge on Year and Week
    print("\nMerging datasets on Year and Week...")
    combined = pd.merge(weather_df, air_df, on=['Year', 'Week'], how='inner')
    print(f"  Combined dataset: {len(combined)} rows, {len(combined.columns)} columns")
    
    # Create lagged features for air quality indicators
    air_quality_cols = ['AQI_weekly_mean', 'PM25_weekly_mean', 'PM10_weekly_mean', 
                        'O3_weekly_mean', 'is_bad_air_week']
    air_quality_cols = [c for c in air_quality_cols if c in combined.columns]
    combined = create_lag_features(combined, air_quality_cols, lags=[1, 2, 4])
    
    # Handle missing values
    print("\nHandling missing values...")
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    
    # Count NaNs before
    nan_before = combined[numeric_cols].isna().sum().sum()
    
    # Fill: forward fill, then backward fill, then 0 for remaining
    combined[numeric_cols] = combined[numeric_cols].ffill().bfill().fillna(0)
    
    nan_after = combined[numeric_cols].isna().sum().sum()
    print(f"  NaN values: {nan_before} -> {nan_after}")
    
    # Create Date column from Year-Week for compatibility
    def create_date(row):
        try:
            year = int(row['Year'])
            week = int(row['Week'])
            # Use Monday of the week
            return pd.Timestamp.fromisocalendar(year, week, 1)
        except:
            return None
    
    combined['Date'] = combined.apply(create_date, axis=1)
    combined['Region'] = 'Australia'
    
    # Sort by date
    combined = combined.sort_values(['Year', 'Week']).reset_index(drop=True)
    
    # Ensure is_surge exists (recalculate if needed)
    if 'is_surge' not in combined.columns:
        print("Creating is_surge column (90th percentile of Deaths)...")
        p90 = combined['Deaths'].quantile(0.90)
        combined['is_surge'] = (combined['Deaths'] > p90).astype(int)
    
    return combined


def main():
    """Main function to merge and save datasets."""
    print("=" * 60)
    print("MERGING AIR QUALITY + WEATHER DEATH DATASETS")
    print("=" * 60)
    
    # Merge
    combined = merge_datasets()
    
    # Summary
    print("\n" + "=" * 60)
    print("COMBINED DATASET SUMMARY")
    print("=" * 60)
    print(f"Shape: {combined.shape}")
    print(f"Date range: {combined['Year'].min()} Week {combined['Week'].min()} to "
          f"{combined['Year'].max()} Week {combined['Week'].max()}")
    print(f"Deaths range: {combined['Deaths'].min():.0f} - {combined['Deaths'].max():.0f}")
    print(f"Surge events: {combined['is_surge'].sum()} ({100*combined['is_surge'].mean():.1f}%)")
    
    # Save
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")
    
    # Print feature groups
    print("\nFeature groups:")
    weather_cols = [c for c in combined.columns if any(x in c.lower() for x in ['temp', 'rain', 'evap', 'rad', 'vapour', 'rh_'])]
    air_cols = [c for c in combined.columns if any(x in c.lower() for x in ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'co', 'bad'])]
    print(f"  Weather features: {len(weather_cols)}")
    print(f"  Air Quality features: {len(air_cols)}")
    
    return combined


if __name__ == "__main__":
    main()
