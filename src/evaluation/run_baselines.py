import pandas as pd
import numpy as np
import os
from src.evaluation.data_loader import load_data, TimeSeriesSplitter, create_sequences
from src.evaluation.metrics import get_metrics, get_surge_metrics
from src.models.baselines import RidgeLagModel, XGBoostBaseline

def run_baselines(target_col='Deaths', surge_col='is_surge'):
    """
    Run baseline models (Ridge, XGBoost) with dynamic target/surge columns.
    
    Args:
        target_col: Name of the target column (default: 'Deaths')
        surge_col: Name of the surge label column (default: 'is_surge')
    """
    default_path = "data/processed/combined_benchmark.csv"
    data_path = os.getenv("BENCHMARK_DATA_PATH", default_path)
    
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return

    print(f"Loading data from: {data_path}")
    df = load_data(data_path)
    
    # Handle dynamic target column
    if target_col not in df.columns:
        if 'AQI_weekly_mean' in df.columns:
            target_col = 'AQI_weekly_mean'
            surge_col = 'is_bad_air_week'
        else:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
    
    print(f"Target Column: {target_col}, Surge Column: {surge_col}")
    
    # Build input columns
    exclude_cols = ['Date', 'Region', target_col, surge_col]
    input_cols = [c for c in df.columns if c not in exclude_cols]
    input_cols_with_target = [target_col] + input_cols
    
    input_width = 8
    horizon = 4
    
    # Prepare Data
    X, y = create_sequences(df, target_col=target_col, input_cols=input_cols_with_target, 
                            input_width=input_width, horizon=horizon)
    
    # Handle surge column
    if surge_col in df.columns:
        y_surge = create_sequences(df, target_col=surge_col, input_cols=input_cols_with_target, 
                                   input_width=input_width, horizon=horizon)[1]
    else:
        y_surge = (y > np.percentile(y, 90)).astype(float)
    
    print(f"Data shapes: X={X.shape}, y={y.shape}")
    
    splitter = TimeSeriesSplitter(n_splits=3, test_size=52, input_width=input_width, horizon=horizon)
    
    models = {}
    try:
        from src.models.baselines import RidgeLagModel, SKLEARN_AVAILABLE
        if SKLEARN_AVAILABLE:
            models["Ridge"] = RidgeLagModel()
    except Exception as e:
        print(f"Skipping Ridge: {e}")

    try:
        from src.models.baselines import XGBoostBaseline, XGBOOST_AVAILABLE
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBoostBaseline()
    except Exception as e:
        print(f"Skipping XGBoost: {e}")

    if not models:
        print("No baseline models available due to missing dependencies.")
        return
    
    results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(df)):
        print(f"\nEvaluating Fold {fold_idx}...")
        
        # Split on arrays directly
        n_samples = len(X)
        test_size = 52
        test_end = n_samples - (fold_idx * test_size)
        test_start = test_end - test_size
        
        if test_start < 0: 
            break
        
        X_test_fold = X[test_start:test_end]
        y_test_fold = y[test_start:test_end]
        y_surge_fold = y_surge[test_start:test_end]
        
        # Train on everything before
        train_end = test_start
        X_train_fold = X[:train_end]
        y_train_fold = y[:train_end]
        
        print(f"  Train size: {len(X_train_fold)}, Test size: {len(X_test_fold)}")
        
        for name, model in models.items():
            # Re-initialize model each fold for fair comparison
            if name == "Ridge":
                model = RidgeLagModel()
            elif name == "XGBoost":
                model = XGBoostBaseline()
                
            print(f"  Training {name}...")
            model.fit(X_train_fold, y_train_fold)
            preds = model.predict(X_test_fold)
            
            # Metrics
            y_true_flat = y_test_fold.flatten()
            y_pred_flat = preds.flatten()
            
            scores = get_metrics(y_true_flat, y_pred_flat)
            
            # Surge: Value-based threshold (90th percentile of training data)
            threshold = np.percentile(y_train_fold, 90)
            pred_binary = (y_pred_flat > threshold).astype(int)
            surge_scores = get_surge_metrics(y_surge_fold.flatten(), pred_binary)
            
            scores.update(surge_scores)
            scores['Model'] = name
            scores['Fold'] = fold_idx
            results.append(scores)
            
            print(f"    {name} RMSE: {scores['RMSE']:.2f}, Surge F1: {scores['Surge_F1']:.2f}")

    # Summary
    res_df = pd.DataFrame(results)
    print("\n--- Final Results ---")
    print(res_df.groupby('Model').mean(numeric_only=True))
    res_df.to_csv("src/evaluation/baseline_results.csv", index=False)

if __name__ == "__main__":
    run_baselines()
