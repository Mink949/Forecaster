import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import math
from src.evaluation.data_loader import load_data, TimeSeriesSplitter, create_sequences
from src.models.transformer import TimeSeriesTransformer
from src.evaluation.metrics import get_metrics, get_surge_metrics

def run_transformer_exp(target_col='Deaths', surge_col='is_surge'):
    """
    Run Transformer experiment with CORRECTED scaling (no data leakage).
    
    FIX C3: Surge evaluation now uses value-based threshold like baselines,
            not the probability output (which is still used as auxiliary loss).
    """
    print("Running Proposed Model Experiment...")
    default_path = "data/processed/combined_benchmark.csv"
    data_path = os.getenv("BENCHMARK_DATA_PATH", default_path)
    print(f"Loading data from: {data_path}")
    df = load_data(data_path)
    
    # Config - dynamically determine target
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
    
    input_width = 12
    horizon = 4
    
    # Data Prep
    X, y = create_sequences(df, target_col=target_col, input_cols=input_cols_with_target, 
                            input_width=input_width, horizon=horizon)
    
    # Handle surge column
    if surge_col in df.columns:
        y_surge = create_sequences(df, target_col=surge_col, input_cols=input_cols_with_target, 
                                   input_width=input_width, horizon=horizon)[1]
    else:
        y_surge = (y > np.percentile(y, 90)).astype(float)
    
    splitter = TimeSeriesSplitter(n_splits=3, test_size=52, input_width=input_width, horizon=horizon)
    results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(df)):
        print(f"\nFold {fold_idx}:")
        
        # Test Start/End
        n_samples = len(X)
        test_size = 52
        test_end = n_samples - (fold_idx * test_size)
        test_start = test_end - test_size
        if test_start < 0: 
            break
        
        train_end = test_start
        
        # Split BEFORE scaling (raw numpy arrays)
        X_train_raw = X[:train_end]
        y_train_raw = y[:train_end]
        y_surge_train_raw = y_surge[:train_end]
        
        X_test_raw = X[test_start:test_end]
        y_test_raw = y[test_start:test_end]
        y_surge_test_raw = y_surge[test_start:test_end]
        
        # =============================================================
        # FIX C1: Fold-wise scaling (compute stats ONLY on training data)
        # =============================================================
        X_train_mean = X_train_raw.mean(axis=(0, 1))
        X_train_std = X_train_raw.std(axis=(0, 1)) + 1e-6
        
        y_train_mean = y_train_raw.mean()
        y_train_std = y_train_raw.std() + 1e-6
        
        # Scale using TRAINING statistics only
        X_train_scaled = (X_train_raw - X_train_mean) / X_train_std
        X_test_scaled = (X_test_raw - X_train_mean) / X_train_std
        
        y_train_scaled = (y_train_raw - y_train_mean) / y_train_std
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train_scaled)
        y_train = torch.FloatTensor(y_train_scaled)
        y_surge_train = torch.FloatTensor(y_surge_train_raw)
        
        X_test = torch.FloatTensor(X_test_scaled)
        
        # Model (re-initialize each fold)
        model = TimeSeriesTransformer(
            input_dim=X.shape[2],
            d_model=64,
            nhead=4,
            num_layers=2,
            output_len=horizon
        )
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        mse_crit = nn.MSELoss()
        bce_crit = nn.BCELoss()
        
        alpha = 0.5  # Trade-off between forecast and surge loss
        
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            pred_forecast, pred_surge = model(X_train)
            
            # Forecast Loss
            loss_f = mse_crit(pred_forecast, y_train)
            
            # Surge Loss (auxiliary - helps training but not used for final eval)
            loss_s = bce_crit(pred_surge, y_surge_train)
            
            loss = loss_f + alpha * loss_s
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        with torch.no_grad():
            p_cast, p_surge_prob = model(X_test)
            
        # Inverse Scale Forecast using TRAINING statistics
        preds_val = p_cast.numpy() * y_train_std + y_train_mean
        
        # Metrics
        y_true_flat = y_test_raw.flatten()
        y_pred_flat = preds_val.flatten()
        
        scores = get_metrics(y_true_flat, y_pred_flat)
        
        # =============================================================
        # FIX C3: Value-based surge threshold (consistent with baselines)
        # Use forecast predictions, not probability head, for fair comparison
        # =============================================================
        threshold = np.percentile(y_train_raw, 90)
        pred_binary = (y_pred_flat > threshold).astype(int)
        scores_surge = get_surge_metrics(y_surge_test_raw.flatten(), pred_binary)
        
        scores.update(scores_surge)
        scores['Model'] = "TimeSeriesTransformer"
        scores['Fold'] = fold_idx
        results.append(scores)
        print(f"  RMSE: {scores['RMSE']:.2f}, Surge F1: {scores['Surge_F1']:.2f}")

    res_df = pd.DataFrame(results)
    print("\n--- Final Proposed Results ---")
    print(res_df.groupby('Model').mean(numeric_only=True))
    res_df.to_csv("src/evaluation/transformer_results.csv", index=False)

if __name__ == "__main__":
    run_transformer_exp()
