import os
import pandas as pd
from src.evaluation.run_baselines import run_baselines
from src.models.run_dl_baselines import run_dl_baselines
from src.models.run_transformer import run_transformer_exp

# Point to User Data
USER_DATA_PATH = "Weather_Death_Weekly_Merged_MinMax.csv"

def run_user_benchmark():
    print(f"==================================================")
    print(f"BENCHMARKING USER DATASET: {USER_DATA_PATH}")
    print(f"==================================================")
    
    # Set Env Var override
    os.environ["BENCHMARK_DATA_PATH"] = USER_DATA_PATH
    
    # 1. Baselines
    print("\n[1/3] Running Baselines...")
    try:
        run_baselines()
    except Exception as e:
        print(f"Error in Baselines: {e}")
        import traceback
        traceback.print_exc()

    # 2. Deep Learning
    print("\n[2/3] Running DL Models...")
    try:
        run_dl_baselines()
    except Exception as e:
        print(f"Error in DL: {e}")
        import traceback
        traceback.print_exc()

    # 3. Transformer
    print("\n[3/3] Running Transformer...")
    try:
        run_transformer_exp()
    except Exception as e:
        print(f"Error in Transformer: {e}")
        import traceback
        traceback.print_exc()

    print("\nBenchmark Complete.")

if __name__ == "__main__":
    run_user_benchmark()
