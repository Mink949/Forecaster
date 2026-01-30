from src.evaluation.data_loader import load_data, TimeSeriesSplitter

def test_protocol():
    df = load_data()
    print(f"Loaded data: {len(df)} rows.")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    splitter = TimeSeriesSplitter(n_splits=3, test_size=52, input_width=8, horizon=4)
    
    print("\n--- Testing Splitter ---")
    fold = 0
    for train_idx, test_idx in splitter.split(df):
        print(f"Fold {fold}:")
        print(f"  Train: {len(train_idx)} samples. Range: {train_idx[0]} - {train_idx[-1]}")
        print(f"  Test:  {len(test_idx)} samples. Range: {test_idx[0]} - {test_idx[-1]}")
        
        # Verify timestamps
        train_dates = df.iloc[train_idx]['Date']
        test_dates = df.iloc[test_idx]['Date']
        
        print(f"  Train end: {train_dates.iloc[-1]}")
        print(f"  Test start: {test_dates.iloc[0]}")
        print(f"  Gap: {(test_dates.iloc[0] - train_dates.iloc[-1]).days} days")
        
        if test_dates.iloc[0] <= train_dates.iloc[-1]:
            print("  WARNING: Leakage detected!")
            
        fold += 1

if __name__ == "__main__":
    test_protocol()
