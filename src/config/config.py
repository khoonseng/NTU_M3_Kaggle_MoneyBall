CONFIG = {
    "models_to_run": [        
        {"name": "linear", "gridsearch": False},
        # {"name": "lasso", "gridsearch": True},
        # {"name": "ridge", "gridsearch": True},
        # {"name": "elasticnet", "gridsearch": True},
    ],
    "models": {
        "linear": {
            "params": {}
        },
        "lasso": {
            "params": {
                "max_iter": 10000,
                "random_state": 30
            },
            "param_grid": {
                "model__alpha": [0.01, 0.1, 1.0],
                "model__max_iter": [5000, 10000, 15000, 20000],
                "model__tol": [1e-4, 1e-3, 1e-2],
                "model__selection": ["cyclic", "random"]
            }
        },
        "ridge": {
            "params": {
                "max_iter": None,
                "random_state": 30
            },
            "param_grid": {
                "model__alpha": [1e-3, 1e-2, 0.1, 1.0, 10.0],
                "model__solver": ["auto", "lsqr", "saga"],
                "model__max_iter": [None, 1000, 5000, 10000],
                "model__tol": [1e-3, 1e-2]
            }
        },
        "elasticnet": {
            "params": {
                "random_state": 30
            },
            "param_grid": {
                "model__alpha": [1e-2, 0.1, 1.0, 10.0],
                "model__max_iter": [5000, 10000],
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                "model__tol": [1e-4, 1e-3, 1e-2],
                "model__selection": ["cyclic", "random"]
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