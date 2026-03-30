from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate(y_true, y_pred, n_features):
    n = len(y_true)
    p = n_features

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Adjusted R²
    if n - p - 1 == 0:
        adj_r2 = None  # avoid division by zero
    else:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "adjusted_r2": adj_r2
    }