import os
import pandas as pd
import sys
from src.evaluation.run_baselines import run_baselines
from src.models.run_dl_baselines import run_dl_baselines
from src.models.run_transformer import run_transformer_exp
from src.analysis.explainability import analyze_ridge_weights

def run_all():
    print("==================================================")
    print("STARTING FULL BENCHMARK SUITE")
    print("==================================================")

    # 0. Merge Datasets (Air Quality + Weather Death)
    print("\n[0/5] Merging Air Quality + Weather Death Datasets...")
    combined_path = "data/processed/combined_benchmark.csv"
    try:
        from src.data.merge_datasets import merge_datasets
        # Only merge if combined dataset doesn't exist or is outdated
        if not os.path.exists(combined_path):
            combined_df = merge_datasets()
            combined_df.to_csv(combined_path, index=False)
            print(f"Created combined dataset: {combined_path}")
        else:
            print(f"Using existing combined dataset: {combined_path}")
    except Exception as e:
        print(f"Warning: Could not merge datasets: {e}")
        print("Attempting to use existing dataset...")

    # 1. Traditional Baselines
    print("\n[1/5] Running Traditional Baselines (Ridge, XGBoost)...")
    try:
        run_baselines()
    except Exception as e:
        print(f"Error in Baselines: {e}")

    # 2. Deep Learning Baselines
    print("\n[2/5] Running Deep Learning Baselines (LSTM, DLinear)...")
    try:
        run_dl_baselines()
    except Exception as e:
        print(f"Error in DL Baselines: {e}")

    # 3. Proposed Transformer
    print("\n[3/5] Running Proposed Lag-Aware Transformer...")
    try:
        run_transformer_exp()
    except Exception as e:
        print(f"Error in Transformer: {e}")

    # 4. Explainability
    print("\n[4/5] Running Explainability Analysis...")
    try:
        analyze_ridge_weights()
    except Exception as e:
        print(f"Error in Analysis: {e}")

    # Consolidate Report
    print("\n==================================================")
    print("CONSOLIDATING RESULTS")
    print("==================================================")
    
    csv_files = [
        "src/evaluation/baseline_results.csv",
        "src/evaluation/dl_results.csv",
        "src/evaluation/transformer_results.csv"
    ]
    
    dfs = []
    for f in csv_files:
        if os.path.exists(f):
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                print(f"Could not read {f}: {e}")
        else:
            print(f"Warning: {f} not found.")
            
    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        # Group by Model and calculate mean metrics across folds
        summary = full_df.groupby('Model')[['MAE', 'RMSE', 'MAPE', 'Surge_F1', 'Surge_Precision', 'Surge_Recall']].mean()
        summary = summary.sort_values('RMSE')
        
        print("\nFinal Benchmark Leaderboard (Sorted by RMSE - Lower is Better):")
        print(summary)
        
        output_path = "final_benchmark_report.csv"
        summary.to_csv(output_path)
        print(f"\nSummary saved to {output_path}")
    else:
        print("No results found.")

if __name__ == "__main__":
    # Ensure raw data/dirs exist? Assumed yes from previous steps.
    run_all()
