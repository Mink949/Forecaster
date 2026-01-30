import pandas as pd
import numpy as np
from src.evaluation.data_loader import load_data, create_sequences
from src.models.baselines import RidgeLagModel, SKLEARN_AVAILABLE

def analyze_ridge_weights():
    if not SKLEARN_AVAILABLE:
        print("sklearn not available. Cannot analyze Ridge weights.")
        return

    df = load_data("data/processed/combined_benchmark.csv")
    target_col = 'Deaths'
    input_cols = [c for c in df.columns if c not in ['Date', 'Region', 'is_surge', 'Deaths']]
    input_width = 8
    horizon = 4
    
    X, y = create_sequences(df, target_col=target_col, input_cols=input_cols, input_width=input_width, horizon=horizon)
    
    # Fit on full data to interpret global patterns
    model = RidgeLagModel(alpha=1.0)
    model.fit(X, y)
    
    # Extract coefficients
    # Coeff shape: [Targets, Features_Flat] or [Features_Flat] depending on implementation.
    # Ridge with multi-output y results in coef_ of shape (n_targets, n_features).
    coef = model.model.coef_
    feature_names = []
    
    # Flattened feature names: [t-W, t-W+1...][Feat1, Feat2...]
    # Actually create_sequences: X is [Samples, Window, Features]
    # Flattened order: Window 0 (t-8), Window 1 (t-7)...
    
    for w in range(input_width):
        for col in input_cols:
            # FIX: Use window position (w0=oldest, w7=most recent for input_width=8)
            # Do NOT append _lag_ to columns that may already have lag suffixes
            # This avoids creating invalid names like 'max_t_lag_8_lag_5'
            window_pos = input_width - 1 - w  # w0 -> oldest (furthest back in time)
            feature_names.append(f"{col}@w{window_pos}")
            
    if coef.ndim == 2:
        # Average importance across horizon steps?
        # Or look at Horizon step 1.
        # Let's average absolute importance
        avg_coef = np.mean(np.abs(coef), axis=0) # Magnitude
        real_coef = np.mean(coef, axis=0) # Direction (but summing might cancel)
    else:
        avg_coef = np.abs(coef)
        real_coef = coef
        
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': avg_coef,
        'Coefficient': real_coef
    })
    
    importance_df = importance_df.sort_values('Importance', ascending=False)
    print("\nTop 20 Important Features:")
    print(importance_df.head(20))
    
    importance_df.to_csv("src/analysis/ridge_importance.csv", index=False)
    
    # Simplified Lag Analysis for 'max_t' (Temperature)
    temp_feats = [f for f in feature_names if 'max_t' in f and 'lag' in f]
    if temp_feats:
        print("\nTemperature Lag Coefficients for 'max_t':")
        # Extract indices preserving valid integer lags if possible
        # My feature naming above was generic "_lag_{lag_idx}".
        # Let's filter row by string match
        temp_df = importance_df[importance_df['Feature'].str.contains('max_t')]
        # Sort by lag index?
        print(temp_df)
        
if __name__ == "__main__":
    analyze_ridge_weights()
