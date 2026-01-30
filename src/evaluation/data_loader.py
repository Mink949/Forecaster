import pandas as pd
import numpy as np

class TimeSeriesSplitter:
    """
    Rolling Origin Splitter for Time Series.
    """
    def __init__(self, n_splits=5, test_size=52, input_width=8, horizon=4, gap=0):
        """
        Args:
            n_splits: Number of rolling windows
            test_size: Number of time steps in each test/validation set (should match horizon usually, or be larger)
            input_width: Lookback window size
            horizon: Forecast horizon
            gap: Gap between train and test (to avoid leakage if any)
        """
        self.n_splits = n_splits
        self.test_size = test_size # usually equal to horizon for strict evaluation, or a block
        self.input_width = input_width
        self.horizon = horizon
        self.gap = gap

    def split(self, df):
        """
        Yields (train_indices, test_indices)
        df must be sorted by time.
        """
        n_samples = len(df)
        # We start from the end and move back
        # Fold 0: Test is [N-test_size : N]
        # Fold 1: Test is [N-2*test_size : N-test_size] ...
        # But for 'Rolling Origin' usually we test on specific horizons.
        
        # Implementation:
        # We want to forecast 'horizon' steps ahead.
        # We define cutoffs.
        
        indices = np.arange(n_samples)
        
        # Minimal training size
        min_train = self.input_width + 52 # at least 1 year of data
        
        step = self.test_size # Non-overlapping test sets for coverage
        
        for i in range(self.n_splits):
            # Test range: [end - test_size, end]
            # Valid range: [end - test_size*2, end - test_size] ???
            # Here we just return Train and Test/Val index.
            
            # Going backwards
            split_idx = i
            test_end = n_samples - (split_idx * step)
            test_start = test_end - self.test_size
            
            train_end = test_start - self.gap
            
            if train_end < min_train:
                break
                
            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]
            
            yield train_idx, test_idx

def load_data(path="data/processed/combined_benchmark.csv"):
    """
    Load dataset for benchmarking.
    
    Supports multiple dataset formats:
    - combined_benchmark.csv (Air Quality + Weather + Deaths)
    - Weather_Death_Weekly_Merged_MinMax.csv
    - Air_Quality_Weekly_Evaluation.csv
    
    Args:
        path: Path to the dataset CSV file
        
    Returns:
        DataFrame with Date, Region, and all numeric features
    """
    df = pd.read_csv(path)
    
    # Handle Date Construction
    if 'Date' not in df.columns:
        if 'Year' in df.columns and 'Week' in df.columns:
            # Create Date from Year-Week (ISO week)
            def fn(row):
                try:
                    year = int(row['Year'])
                    week = int(row['Week'])
                    # Use ISO calendar for consistency
                    return pd.Timestamp.fromisocalendar(year, week, 1)
                except:
                    try:
                        # Fallback: simple approximation
                        start = pd.Timestamp(year=int(row['Year']), month=1, day=1)
                        delta = pd.Timedelta(weeks=int(row['Week'])-1)
                        return start + delta
                    except:
                        return None
            df['Date'] = df.apply(fn, axis=1)
        else:
            raise ValueError("Dataset must have 'Date' column or 'Year' and 'Week' columns.")
            
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Handle missing Region
    if 'Region' not in df.columns:
        df['Region'] = 'Australia'
    
    # Handle NaN values (forward fill, then backward fill for any remaining)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0)
    
    # Ensure is_surge column exists if Deaths exists
    if 'Deaths' in df.columns and 'is_surge' not in df.columns:
        p90 = df['Deaths'].quantile(0.90)
        df['is_surge'] = (df['Deaths'] > p90).astype(int)
        
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def create_sequences(df, target_col='Deaths', input_cols=None, input_width=8, horizon=4):
    """
    Create (X, y) sequences for ML/DL models.
    X: (Samples, Input_Width, Features)
    y: (Samples, Horizon) -> Multi-step forecast
    """
    if input_cols is None:
        input_cols = [c for c in df.columns if c not in ['Date', 'Region']]
    
    # Filter to numeric columns only (exclude strings like 'Main_pollutant')
    numeric_cols = df[input_cols].select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < len(input_cols):
        excluded = set(input_cols) - set(numeric_cols)
        print(f"Note: Excluding non-numeric columns from features: {excluded}")
        input_cols = numeric_cols
        
    data = df[input_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)
    
    X, y = [], []
    
    # Iterate through valid start points
    # We need sequence: t-W...t-1 -> Predict t...t+H-1
    # DataFrame rows 0..W-1 is first input. Target is W..W+H-1
    
    for i in range(len(df) - input_width - horizon + 1):
        X.append(data[i : i+input_width])
        y.append(target[i+input_width : i+input_width+horizon])
        
    return np.array(X), np.array(y)
