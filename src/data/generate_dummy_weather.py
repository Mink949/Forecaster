import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

OUTPUT_DIR = os.path.join("data", "raw", "weather")

STATIONS = {
    "Sydney_066062": {"lat": -33.86, "base_temp": 22, "amp": 7},
    "Melbourne_086071": {"lat": -37.81, "base_temp": 20, "amp": 9},
    "Brisbane_040913": {"lat": -27.47, "base_temp": 25, "amp": 5},
    "Adelaide_023090": {"lat": -34.92, "base_temp": 21, "amp": 10},
    "Perth_009021": {"lat": -31.95, "base_temp": 24, "amp": 8},
    "Hobart_094029": {"lat": -42.88, "base_temp": 17, "amp": 6},
    "Darwin_014015": {"lat": -12.46, "base_temp": 32, "amp": 2}, # Tropical, less seasonal
    "Canberra_070351": {"lat": -35.28, "base_temp": 19, "amp": 11},
}

def generate_synthetic_weather():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    dates = pd.date_range(start="2015-01-01", end=datetime.now(), freq="D")
    
    print(f"Generating synthetic weather data for {len(STATIONS)} stations from 2015 to present...")
    
    for station_name, params in STATIONS.items():
        # Seasonal component (southern hemisphere: hottest in Jan/Feb)
        # Day of year 0 = Jan 1. Cosine peaks at 0. We want peak around Jan/Feb.
        # Shifted so peak is at start of year.
        day_of_year = dates.dayofyear
        
        # Temp = Base + Amp * cos( ...) + Noise
        # Peak at Jan 15 (Day 15). 
        # cos( (d - 15) / 365 * 2pi )
        seasonal = np.cos((day_of_year - 15) / 365.25 * 2 * np.pi)
        
        tmax_mean = params["base_temp"] + params["amp"] * seasonal
        tmin_mean = (params["base_temp"] - 10) + (params["amp"] * 0.8) * seasonal
        
        # Add noise and heatwaves
        noise_max = np.random.normal(0, 3, size=len(dates))
        noise_min = np.random.normal(0, 2, size=len(dates))
        
        # Synthetic heatwaves (autocorrelated noise)
        heatwave_boost = np.zeros(len(dates))
        for i in range(1, len(dates)):
            if random.random() < 0.01: # 1% chance to start heatwave
                duration = random.randint(3, 7)
                boost = random.uniform(3, 8)
                for j in range(duration):
                    if i+j < len(dates):
                        heatwave_boost[i+j] += boost
        
        tmax = tmax_mean + noise_max + heatwave_boost
        tmin = tmin_mean + noise_min + (heatwave_boost * 0.5)
        
        # Rain: related to season but noisy.
        # Simple random implementation
        rain_prob = 0.2
        rain = np.where(np.random.rand(len(dates)) < rain_prob, np.random.exponential(5, len(dates)), 0)
        
        df = pd.DataFrame({
            "Date": dates,
            "max_t": tmax.round(1),
            "min_t": tmin.round(1),
            "rain": rain.round(1)
        })
        
        # SILO format usually: date, day, max_t, min_t, rain, ...
        # But our loader in process_merge will just look for columns.
        # We'll save as standard CSV.
        
        output_path = os.path.join(OUTPUT_DIR, f"{station_name}.csv")
        df.to_csv(output_path, index=False)
        print(f"  Generated {output_path}")

if __name__ == "__main__":
    generate_synthetic_weather()
