import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from src.evaluation.data_loader import load_data, TimeSeriesSplitter, create_sequences
from src.models.dl_models import DLinear, LSTMModel
from src.evaluation.metrics import get_metrics, get_surge_metrics

def run_dl_baselines(target_col='Deaths', surge_col='is_surge'):
    """
    Run DL baselines with CORRECTED scaling (no data leakage).
    
    Args:
        target_col: Name of the target column (default: 'Deaths')
        surge_col: Name of the surge label column (default: 'is_surge')
    """
    default_path = "data/processed/combined_benchmark.csv"
    data_path = os.getenv("BENCHMARK_DATA_PATH", default_path)
    print(f"Loading data from: {data_path}")
    df = load_data(data_path)
    
    # Config - dynamically determine target
    # Handle case where target column might have different names
    if target_col not in df.columns:
        # Try common alternatives
        if 'AQI_weekly_mean' in df.columns:
            target_col = 'AQI_weekly_mean'
            surge_col = 'is_bad_air_week'
        else:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")
    
    print(f"Target Column: {target_col}, Surge Column: {surge_col}")
    
    # Build input columns (exclude Date, Region, target, surge)
    exclude_cols = ['Date', 'Region', target_col, surge_col]
    input_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Add target to input_cols for sequence creation (we need it in the features for lagged models)
    input_cols_with_target = [target_col] + input_cols  # Target FIRST for known index
    
    input_width = 12  # Longer window for DL
    horizon = 4
    
    # Create sequences (target is now at index 0)
    X, y = create_sequences(df, target_col=target_col, input_cols=input_cols_with_target, 
                            input_width=input_width, horizon=horizon)
    
    # Handle surge column
    if surge_col in df.columns:
        y_surge = create_sequences(df, target_col=surge_col, input_cols=input_cols_with_target, 
                                   input_width=input_width, horizon=horizon)[1]
    else:
        # Create synthetic surge labels (90th percentile)
        y_surge = (y > np.percentile(y, 90)).astype(float)
    
    # FIX C2: Target index is now guaranteed to be 0 (we put it first)
    TARGET_IDX = 0
    print(f"Target index in features: {TARGET_IDX}")
    
    splitter = TimeSeriesSplitter(n_splits=3, test_size=52, input_width=input_width, horizon=horizon)
    
    results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(df)):
        print(f"\nEvaluating Fold {fold_idx} (DL)...")
        
        # Align indices on arrays
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
        X_test_raw = X[test_start:test_end]
        y_test_raw = y[test_start:test_end]
        y_surge_fold = y_surge[test_start:test_end]
        
        # =============================================================
        # FIX C1: Fold-wise scaling (compute stats ONLY on training data)
        # =============================================================
        X_train_mean = X_train_raw.mean(axis=(0, 1))
        X_train_std = X_train_raw.std(axis=(0, 1)) + 1e-6
        
        y_train_mean = y_train_raw.mean()
        y_train_std = y_train_raw.std() + 1e-6
        
        # Scale using TRAINING statistics only
        X_train_scaled = (X_train_raw - X_train_mean) / X_train_std
        X_test_scaled = (X_test_raw - X_train_mean) / X_train_std  # Use train stats!
        
        y_train_scaled = (y_train_raw - y_train_mean) / y_train_std
        # y_test is not scaled (we compare predictions to original values)
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train_scaled)
        y_train = torch.FloatTensor(y_train_scaled)
        X_test = torch.FloatTensor(X_test_scaled)
        
        # Models (re-initialize each fold for fair comparison)
        models = {
            "LSTM": LSTMModel(input_size=X.shape[2], hidden_size=64, num_layers=2, output_len=horizon),
            "DLinear": DLinear(input_len=input_width, pred_len=horizon, enc_in=X.shape[2])
        }
        
        for name, model in models.items():
            print(f"  Training {name}...")
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.005)
            
            # Training Loop
            model.train()
            n_epochs = 50
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                out = model(X_train)
                
                if name == "DLinear":
                    # FIX C2: Use dynamic target index (now 0)
                    loss = criterion(out[:, :, TARGET_IDX], y_train)
                else:
                    # LSTM outputs [Batch, Horizon, 1]
                    loss = criterion(out.squeeze(-1), y_train)
                    
                loss.backward()
                optimizer.step()
                
            # Predict
            model.eval()
            with torch.no_grad():
                preds_scaled = model(X_test)
                if name == "DLinear":
                    preds_scaled = preds_scaled[:, :, TARGET_IDX]
                else:
                    preds_scaled = preds_scaled.squeeze(-1)
            
            # Inverse Scale using TRAINING statistics
            preds = preds_scaled.numpy() * y_train_std + y_train_mean
            
            # Metrics
            y_true_flat = y_test_raw.flatten()
            y_pred_flat = preds.flatten()
            
            scores = get_metrics(y_true_flat, y_pred_flat)
            
            # =============================================================
            # FIX C3: Value-based surge threshold (consistent with baselines)
            # =============================================================
            threshold = np.percentile(y_train_raw, 90)  # 90th percentile of TRAINING data
            pred_binary = (y_pred_flat > threshold).astype(int)
            surge_scores = get_surge_metrics(y_surge_fold.flatten(), pred_binary)
            
            scores.update(surge_scores)
            scores['Model'] = name
            scores['Fold'] = fold_idx
            results.append(scores)
            print(f"    {name} RMSE: {scores['RMSE']:.2f}")

    res_df = pd.DataFrame(results)
    print("\n--- Final DL Results ---")
    print(res_df.groupby('Model').mean(numeric_only=True))
    res_df.to_csv("src/evaluation/dl_results.csv", index=False)

if __name__ == "__main__":
    run_dl_baselines()
