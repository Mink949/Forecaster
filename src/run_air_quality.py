import os
import pandas as pd
from src.evaluation.run_baselines import run_baselines
from src.models.run_dl_baselines import run_dl_baselines
from src.models.run_transformer import run_transformer_exp

# Configuration for Air Quality Dataset
AIR_QUALITY_DATA_PATH = "Air_Quality_Weekly_Evaluation.csv"

def run_air_quality_benchmark():
    """
    Run the full benchmark suite on the Air Quality dataset.
    
    Target: AQI_weekly_mean
    Surge: is_bad_air_week
    """
    print("==================================================")
    print("BENCHMARKING AIR QUALITY DATASET")
    print(f"Dataset: {AIR_QUALITY_DATA_PATH}")
    print("==================================================")
    
    # Set Env Var override
    os.environ["BENCHMARK_DATA_PATH"] = AIR_QUALITY_DATA_PATH
    
    # 1. Baselines
    print("\n[1/3] Running Baselines (Ridge, XGBoost)...")
    try:
        run_baselines(target_col='AQI_weekly_mean', surge_col='is_bad_air_week')
    except Exception as e:
        print(f"Error in Baselines: {e}")
        import traceback
        traceback.print_exc()

    # 2. Deep Learning
    print("\n[2/3] Running DL Models (LSTM, DLinear)...")
    try:
        run_dl_baselines(target_col='AQI_weekly_mean', surge_col='is_bad_air_week')
    except Exception as e:
        print(f"Error in DL: {e}")
        import traceback
        traceback.print_exc()

    # 3. Transformer
    print("\n[3/3] Running Transformer...")
    try:
        run_transformer_exp(target_col='AQI_weekly_mean', surge_col='is_bad_air_week')
    except Exception as e:
        print(f"Error in Transformer: {e}")
        import traceback
        traceback.print_exc()

    print("\n==================================================")
    print("Air Quality Benchmark Complete.")
    print("==================================================")

if __name__ == "__main__":
    run_air_quality_benchmark()
