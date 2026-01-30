try:
    from sklearn.linear_model import Ridge
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not found. RidgeLagModel will be disabled.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not found. XGBoostBaseline will be disabled.")

import numpy as np
import pandas as pd

class RidgeLagModel:
    """
    Validation of concept: Linear Lag Model (DLNM proxy).
    Uses Ridge regression on lagged features.
    """
    def __init__(self, alpha=1.0):
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for RidgeLagModel")
        self.model = Ridge(alpha=alpha)
        
    def fit(self, X, y):
        X_flat = X.reshape(X.shape[0], -1)
        self.model.fit(X_flat, y)
        return self
        
    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)

class XGBoostBaseline:
    """
    XGBoost Forecasting Model.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5):
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is required for XGBoostBaseline")
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            max_depth=max_depth, 
            objective='reg:squarederror'
        )
        
    def fit(self, X, y):
        X_flat = X.reshape(X.shape[0], -1)
        self.model.fit(X_flat, y)
        return self
        
    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)
