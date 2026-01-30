import numpy as np
import pandas as pd
# Removed sklearn dependency to avoid install blocks
# from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support

def get_metrics(y_true, y_pred, y_pred_lower=None, y_pred_upper=None, quantization_level=None):
    """
    Calculate deterministic and probabilistic metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Deterministic (NumPy implementation)
    residuals = y_true - y_pred
    mae = np.mean(np.abs(residuals))
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    
    # Avoid div by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs(residuals / (y_true + 1e-6))) * 100
    
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }
    
    # Probabilistic (Interval Score - MIS)
    if y_pred_lower is not None and y_pred_upper is not None and quantization_level is not None:
        alpha = quantization_level
        lower = np.array(y_pred_lower)
        upper = np.array(y_pred_upper)
        
        # Mean Interval Score (MIS)
        width = upper - lower
        penalty_lower = (2/alpha) * (lower - y_true) * (y_true < lower)
        penalty_upper = (2/alpha) * (y_true - upper) * (y_true > upper)
        
        mis = np.mean(width + penalty_lower + penalty_upper)
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        
        metrics["MIS"] = mis
        metrics["Coverage"] = coverage
        metrics["IntervalWidth"] = np.mean(width)

    return metrics

def get_surge_metrics(y_true_binary, y_pred_prob, threshold=0.5):
    """
    Calculate classification metrics for surge detection using NumPy.
    """
    y_true = np.array(y_true_binary)
    # y_pred_prob might be probability or binary prediction already if threshold passed before.
    # The caller passed pred_binary as y_pred_prob.
    y_pred = np.array(y_pred_prob)
    
    # TP, FP, FN
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "Surge_Precision": precision,
        "Surge_Recall": recall,
        "Surge_F1": f1
    }
