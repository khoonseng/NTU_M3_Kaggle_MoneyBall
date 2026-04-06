CONFIG = {
    "models_to_run": [        
        {"name": "linear", "gridsearch": False},
        {"name": "lasso", "gridsearch": True},
        {"name": "ridge", "gridsearch": True},
        {"name": "elasticnet", "gridsearch": True},
        {"name": "lightgbm", "gridsearch": False},
        {"name": "lightgbm_pca", "gridsearch": False}
    ],
    "model_type": {
        "linear": "linear",
        "lasso": "linear",
        "ridge": "linear",
        "elasticnet": "linear",
        "lightgbm": "tree",
        "lightgbm_pca": "pca"
    },
    "models": {
        "linear": {
            "params": {}
        },
        "lasso": {
            "params": {
                "max_iter": 10000,
                "random_state": 42
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
                "random_state": 42
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
                "random_state": 42
            },
            "param_grid": {
                "model__alpha": [1e-2, 0.1, 1.0, 10.0],
                "model__max_iter": [5000, 10000],
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                "model__tol": [1e-4, 1e-3, 1e-2],
                "model__selection": ["cyclic", "random"]
            }
        },
        "lightgbm": {
            "params": {
                "objective": "regression_l1",
                "metric": "l1",

                "n_estimators": 200,
                "learning_rate": 0.05,

                "num_leaves": 31,
                "max_depth": -1,

                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,

                "reg_alpha": 0.1,
                "reg_lambda": 0.1,

                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1
            },
            "param_grid": { 
                # "model__n_estimators": [50, 100],
                "model__num_leaves": [31, 63],
                # "model__min_child_samples": [20, 50],
                # "model__max_depth": [-1, 5, 10],
                "model__learning_rate": [0.05, 0.1]
            },
            "param_dist": {
                # "model__n_estimators": [50, 100],
                # "model__num_leaves": [15, 31, 63],
                "model__min_child_samples": [10, 20],
                # # "model__max_depth": [-1, 5, 10],
                # "model__learning_rate": [0.03, 0.05, 0.1],
                # "model__subsample": [0.7, 0.8, 1.0],
                # "model__colsample_bytree": [0.7, 0.8, 1.0],
                # "model__reg_alpha": [0, 0.1, 0.5],
                # "model__reg_lambda": [0, 0.1, 0.5]
            }
        },
        "lightgbm_pca": {
            "params": {
                "objective": "regression_l1",
                "metric": "l1",

                "n_estimators": 200,
                "learning_rate": 0.05,

                "num_leaves": 31,
                "max_depth": -1,

                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,

                "reg_alpha": 0.1,
                "reg_lambda": 0.1,

                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1
            },
            "param_dist": {
                "preprocessor__num__pca__n_components": [0.8, 0.9, 0.95],
                "model__num_leaves": [31, 63],
                "model__learning_rate": [0.05, 0.1]
            }
        }
    },
    "linear_search": {
        "method": "grid",  # "grid" or "random"
        "scoring": {
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2"
        },
        "refit_metric": "mae",  # primary metric for selecting best model        
        "n_iter": 30,
        "cv": 5,
        "n_jobs": -1
    },
    "tree_search": {
        "method": "random",  # "grid" or "random"     
        "scoring": "neg_mean_absolute_error",
        "refit_metric": "neg_mean_absolute_error",
        "n_iter": 10,
        "cv": 3,
        "n_jobs": -1
    },
    "ensemble": {
        "enabled": True,
        "base_models": ["linear", "lasso", "ridge", "elasticnet", "lightgbm_pca"],
        # "base_models": ["linear", "ridge", "elasticnet", "lasso"],
        "final_model": "elasticnet",  # "ridge" or "elasticnet"
        "use_saved_models": True,
        "model_dir": "models"
    }
}