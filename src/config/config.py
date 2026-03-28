CONFIG = {
    "models": {
        "lasso": {
            "param_grid": {
                "model__alpha": [0.001, 0.01, 0.1, 1.0]
            }
        },
        "ridge": {
            "param_grid": {
                "model__alpha": [0.01, 0.1, 1.0, 10.0]
            }
        },
        "elasticnet": {
            "param_grid": {
                "model__alpha": [0.01, 0.1, 1.0],
                "model__l1_ratio": [0.2, 0.5, 0.8]
            }
        }
    },
    "scoring": {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2"
    },
    "refit_metric": "mae"  # primary metric for selecting best model
}