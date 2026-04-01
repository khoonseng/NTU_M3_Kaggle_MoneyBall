from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from lightgbm import LGBMRegressor

MODEL_REGISTRY = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
    "lightgbm": LGBMRegressor
}

def get_model(model_name, params=None):
    params = params or {}

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not supported")

    return MODEL_REGISTRY[model_name](**params)